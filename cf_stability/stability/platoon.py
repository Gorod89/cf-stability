"""Platoon growth test (docs/m3_contract.md, section 6; docs/m4_contract.md, section 2.2).

Identical vehicles behind a leader with a prescribed speed profile; vehicle ``n`` follows the
simulated vehicle ``n - 1``. All vehicles of all profiles advance together, one model call per
time step, with the arithmetic of ``rollout_model`` (float64, no gradient): the accelerations of
step ``k`` come from the histories up to ``k``, then every speed and position is updated, and
vehicle ``n`` reads the updated position and speed of vehicle ``n - 1`` exactly as
``rollout_model`` reads a recorded leader. The standard deviation of the speed per vehicle index is
compared with the empirical curve of an OpenACC run (speed standard deviation per platoon
position) or with hand-entered values (``configs/empirical_growth.yaml``); a platoon that collides has
no growth error, since the speeds after a collision are no car following. The hysteresis loop of the
first follower in the plane (gap, speed) gives its enclosed area (D83).

A platoon has to start in steady state: a vehicle that starts away from the equilibrium of the law
closes its gap first, vehicle ``n`` makes up ``n`` gaps, and the speed deviation then grows along the
platoon because of the start and not because of the law. The start ``anchored`` therefore uses the
equilibrium of the law inside the spacing band of its training data (D78, D79); the time gap is the
fallback where there is none (D73).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import parametrize

from cf_stability.data.openacc import parse_openacc_csv
from cf_stability.data.processing import resample_uniform
from cf_stability.data.schema import DT
from cf_stability.models.base import CFModel
from cf_stability.stability.certificate import to_json
from cf_stability.stability.equilibrium import NOMINAL_S0, NOMINAL_T, Band, find_equilibria
from cf_stability.train.calibration import resolve_device
from cf_stability.train.closed_loop import A_MAX, A_MIN
from cf_stability.utils import resolve_path

Curve = Sequence[float] | np.ndarray | Tensor
START_GAPS = ("anchored", "equilibrium", "time_gap")


def braking_pulse(
    v0: float, dv_pulse: float, b_pulse: float, hold_s: float, duration_s: float, dt: float = DT
) -> Tensor:
    """Leader speed ``[round(duration_s / dt) + 1]`` (float64): from ``v0`` down by ``dv_pulse`` at the
    deceleration ``b_pulse``, ``hold_s`` at the low speed, back up at ``b_pulse``, then ``v0``."""
    t = dt * torch.arange(round(duration_s / dt) + 1, dtype=torch.float64)
    down = torch.clamp(b_pulse * t, max=dv_pulse)
    up = torch.clamp(b_pulse * (t - dv_pulse / b_pulse - hold_s), min=0.0)
    return v0 - torch.clamp(down - up, min=0.0)


def openacc_profile(path: str | Path, dt: float = DT) -> dict[str, Any]:
    """Leader speed series and empirical speed standard deviation per platoon position of an OpenACC run.

    Speeds are resampled to ``dt`` and clipped at 0. Only the longest stretch of samples where
    every vehicle with a speed column has a speed is used, for both outputs. The leader is the
    first vehicle of ``Vehicle_order`` with a speed column (in ZalaZone the target car has none);
    position ``p`` is the vehicle ``p`` places behind it in ``Vehicle_order``, NaN without speed.
    """
    run = parse_openacc_csv(path)
    index = sorted(i for i, columns in run.vehicles.items() if "speed" in columns)
    if not index:
        raise ValueError(f"{path}: no speed columns")
    grid, series = resample_uniform(run.time, {f"speed{i}": run.vehicles[i]["speed"] for i in index}, dt=dt)
    speeds = np.clip(np.stack([series[f"speed{i}"] for i in index], axis=1), 0.0, None)
    complete = np.isfinite(speeds).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], complete.astype(np.int8), [0]))))
    if not len(edges):
        raise ValueError(f"{path}: no sample where every vehicle has a speed")
    starts, stops = edges[0::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    block = speeds[starts[k] : stops[k]]
    leader = index[0]
    std = np.full(max(len(run.vehicle_order), index[-1]) - leader + 1, np.nan)
    std[np.array(index) - leader] = block.std(axis=0)
    return {
        "file": run.path.name,
        "campaign": run.campaign,
        "acc_flag": run.acc_flag,
        "vehicle_order": run.vehicle_order[leader - 1 :],
        "vehicles_with_speed": [i - leader for i in index],
        "t_start": float(grid[starts[k]]),
        "duration_s": len(block) * dt,
        "v_lead": block[:, 0],
        "empirical_std": std,
    }


def _start_gaps(
    model: CFModel, v_start: Tensor, start_gap: str, gap_s0: float, gap_T: float, band: Band | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Start gap per profile, whether the model has an equilibrium at the start speed, and whether the
    start gap is an equilibrium of the model (else a time gap)."""
    if start_gap not in START_GAPS:
        raise ValueError(f"start_gap must be one of {START_GAPS}, got {start_gap!r}")
    eq = find_equilibria(model, v_start, band=band if start_gap == "anchored" else None)
    # a law without tensors (persistence, calibrated laws) is searched on the CPU: back to the platoon's device
    s, usable = eq.s.to(v_start), eq.usable.to(v_start.device)
    if start_gap == "equilibrium":  # an indifferent law is in equilibrium at the nominal spacing it gets
        return torch.where(usable, s, NOMINAL_S0 + NOMINAL_T * v_start), usable, usable
    time_gap = gap_s0 + gap_T * v_start
    if start_gap == "time_gap":
        return time_gap, usable, torch.zeros_like(usable)
    # anchored: the equilibrium inside the band of the start speed; a law that is indifferent has no
    # equilibrium of its own, and without band (at that speed, or at all) the time gap is used
    anchored = (eq.in_band & ~eq.indifferent).to(v_start.device)
    return torch.where(anchored, s, time_gap), usable, anchored


@torch.no_grad()
def simulate_platoons(
    model: CFModel,
    profiles: Sequence[Curve],
    n_vehicles: int = 50,
    dt: float = DT,
    *,
    start_gap: str = "equilibrium",
    gap_s0: float = NOMINAL_S0,
    gap_T: float = NOMINAL_T,
    device: str | torch.device = "cpu",
    band: Band | None = None,
) -> list[dict[str, Any]]:
    """Platoons of ``n_vehicles`` (the leader is vehicle 0) behind every leader speed profile ``[T_p]``.

    Every vehicle starts at the leader's first speed and at the start gap:

    * ``equilibrium``: the model's first equilibrium on ``[1, 200]`` m at that speed (the nominal
      spacing of ``equilibrium.py`` when there is none);
    * ``time_gap``: ``gap_s0 + gap_T * v``;
    * ``anchored``: the model's equilibrium inside ``band`` (the spacing band of its training data)
      at that speed; ``gap_s0 + gap_T * v`` where that speed has no band, where the model has no
      equilibrium inside the band or is indifferent, and everywhere without ``band``.

    ``start`` (``"equilibrium"`` or ``"time_gap"``) says which one a profile got, ``start_gap`` and
    ``start_gap_m`` hold the spacing, ``start_equilibrium`` whether the model has an equilibrium at the
    start speed at all. For a model with a window, ``window - 1`` samples of cruising precede the
    profile, so that the first step already sees the profile. Profiles are padded with their last
    speed and the padding is left out of every result. Series ``[n_vehicles, T_p]`` on the CPU are
    aligned with the profile; the gap of the leader is NaN. ``collision_step`` / ``collision_vehicle``:
    first sample with a gap ``<= 0`` and the lowest vehicle index with it (``None`` without collision).
    """
    if n_vehicles < 2:
        raise ValueError("a platoon needs at least 2 vehicles")
    device = resolve_device(device)
    m = copy.deepcopy(model).to(device).double().eval()
    lead = [torch.as_tensor(p, dtype=torch.float64).reshape(-1).cpu() for p in profiles]
    n_prof, w, n_f, lengths = len(lead), m.window, n_vehicles - 1, [len(p) for p in lead]
    t_max = max(lengths)
    v_lead = torch.stack([torch.cat((p[0].expand(w - 1), p, p[-1].expand(t_max - len(p)))) for p in lead]).to(device)
    x_lead = dt * (torch.cumsum(v_lead, dim=1) - v_lead[:, :1])  # x[k] = x[k-1] + dt v[k]
    x_lead = x_lead - x_lead[:, w - 1 : w]  # x = 0 at the start of the profile
    v_start = v_lead[:, w - 1]
    s_start, has_equilibrium, at_equilibrium = _start_gaps(m, v_start, start_gap, gap_s0, gap_T, band)
    positions = [x_lead[:, w - 1]]
    for _ in range(n_f):  # the order of the subtractions of consecutive two-vehicle rollouts
        positions.append(positions[-1] - s_start)
    x, v = torch.stack(positions, dim=1), v_start[:, None].expand(-1, n_vehicles)
    s = s_start[:, None].expand(-1, n_f)
    state = torch.stack((s, v[:, 1:] - v[:, :-1], v[:, 1:]), dim=-1).reshape(-1, 1, 3)
    history = state.expand(-1, w, -1).contiguous()
    series: dict[str, list[Tensor]] = {"v": [v], "x": [x], "s": [s], "a": []}
    with parametrize.cached():  # spectral norms of ResidualIDM once, not at every step
        for k in range(w - 1, w + t_max - 2):
            a = torch.clamp(m(history), A_MIN, A_MAX).reshape(n_prof, n_f)
            v_f = torch.clamp(v[:, 1:] + a * dt, min=0.0)
            x_f = x[:, 1:] + v_f * dt
            v = torch.cat((v_lead[:, k + 1 : k + 2], v_f), dim=1)
            x = torch.cat((x_lead[:, k + 1 : k + 2], x_f), dim=1)
            s = x[:, :-1] - x[:, 1:]
            state = torch.stack((s, v[:, 1:] - v[:, :-1], v[:, 1:]), dim=-1).reshape(-1, 1, 3)
            history = torch.cat((history[:, 1:], state), dim=1) if w > 1 else state
            for name, value in (("v", v), ("x", x), ("s", s), ("a", a)):
                series[name].append(value)
    v_all, x_all, s_all = (torch.stack(series[name], dim=-1).cpu() for name in ("v", "x", "s"))
    a_all = torch.stack(series["a"], dim=-1).cpu() if series["a"] else torch.zeros(n_prof, n_f, 0, dtype=torch.float64)
    s_all = torch.cat((torch.full_like(s_all[:, :1], torch.nan), s_all), dim=1)
    out = []
    for p, length in enumerate(lengths):
        v_p, x_p, s_p = v_all[p, :, :length], x_all[p, :, :length], s_all[p, :, :length]
        hit = s_p[1:] <= 0.0
        step = int(hit.any(dim=0).to(torch.int8).argmax()) if hit.any() else None
        out.append(
            {
                "speed_std": v_p.std(dim=1, unbiased=False),
                "min_gap": s_p.amin(dim=1),
                "collided": step is not None,
                "collision_step": step,
                "collision_vehicle": None if step is None else 1 + int(hit[:, step].to(torch.int8).argmax()),
                "max_abs_acc": float(a_all[p, :, : length - 1].abs().max()) if length > 1 else float("nan"),
                "start": "equilibrium" if at_equilibrium[p] else "time_gap",
                "start_gap": float(s_start[p]),
                "start_gap_m": float(s_start[p]),
                "start_equilibrium": bool(has_equilibrium[p]),
                "v": v_p,
                "x": x_p,
                "s": s_p,
            }
        )
    return out


def simulate_platoon(
    model: CFModel, v_lead: Curve, n_vehicles: int = 50, dt: float = DT, **options: Any
) -> dict[str, Any]:
    """:func:`simulate_platoons` of one profile."""
    return simulate_platoons(model, [v_lead], n_vehicles, dt, **options)[0]


def growth_error(model_curve: Curve, empirical_curve: Curve) -> float:
    """RMSE of the absolute speed standard deviations over the positions where the empirical curve
    is defined, divided by the maximum of the empirical curve."""
    model_curve = np.asarray(torch.as_tensor(model_curve, dtype=torch.float64).cpu(), dtype=np.float64)
    empirical = np.asarray(torch.as_tensor(empirical_curve, dtype=torch.float64).cpu(), dtype=np.float64)
    positions = np.flatnonzero(np.isfinite(empirical))
    if positions.size == 0:
        raise ValueError("the empirical curve has no finite value")
    if positions[-1] >= len(model_curve):
        raise ValueError(f"the model curve has {len(model_curve)} positions, the empirical one {positions[-1] + 1}")
    rmse = np.sqrt(np.mean((model_curve[positions] - empirical[positions]) ** 2))
    return float(rmse / empirical[positions].max())


def hysteresis_area(s: Curve, v: Curve) -> float:
    """Area enclosed by the trajectory ``(s_k, v_k)`` in the plane (gap, speed), closed by the straight
    line from the last point back to the first (D83), in m^2/s: ``0.5 * |sum_k (s_k v_{k+1} - s_{k+1} v_k)|``
    with the index ``n`` read as ``0`` (shoelace formula). Parts of the loop that are run through in
    opposite senses cancel; a trajectory back and forth along a line encloses nothing. NaN when a value
    is not finite.
    """
    s = np.asarray(torch.as_tensor(s, dtype=torch.float64).cpu(), dtype=np.float64).reshape(-1)
    v = np.asarray(torch.as_tensor(v, dtype=torch.float64).cpu(), dtype=np.float64).reshape(-1)
    if s.shape != v.shape:
        raise ValueError(f"gap and speed differ in length: {s.shape[0]} and {v.shape[0]}")
    if not (np.isfinite(s).all() and np.isfinite(v).all()):
        return float("nan")
    if s.size == 0:
        return 0.0
    s, v = s - s[0], v - v[0]  # the area does not depend on the origin; small coordinates cancel less
    return float(0.5 * abs(np.sum(s * np.roll(v, -1) - np.roll(s, -1) * v)))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    """Mean, or None without values or when one of them is None (a mean over part of them would hide it)."""
    if not values or any(x is None for x in values):
        return None
    return float(np.mean(values))


def _std_ratio(speed_std: Sequence[float | None]) -> float | None:
    """Speed std of the last vehicle over that of vehicle 1, None when either is missing or vehicle 1 is steady."""
    first, last = speed_std[1], speed_std[-1]
    if first is None or last is None or not first > 0.0:
        return None
    return float(last / first)


def platoon_summary(profiles: Mapping[str, Mapping[str, Any]], pulse: str = "pulse") -> dict[str, Any]:
    """Summary of the JSON-ready result of :func:`platoon_test` (docs/m4_contract.md, 2.2).

    A profile whose platoon collided has no growth error and no std ratio (None, whatever its record
    holds): the speeds after a collision are no car following. ``growth_error``: per OpenACC profile (the
    profiles with an empirical curve); its mean over the profiles without collision among all of them,
    among the human profiles (header flag ACC 0) and among the ACC profiles (every other flag: 1 ACC,
    2 mixed, unknown); a mean is None when no such profile is left or when one of their errors is None.
    ``growth_profiles`` and ``growth_collided`` (keys ``mean``, ``human``, ``acc``): the numbers of these
    profiles and of those among them that collided. ``n_collided`` of ``n_profiles`` profiles have a
    collision anywhere in the platoon; ``n_start_equilibrium`` of them started at an equilibrium of the
    model. ``std_ratio``: speed std of the last vehicle (vehicle 50 of the default 51) over that of
    vehicle 1, per profile. ``hysteresis_area_pulse``: hysteresis loop area of follower 1 behind the
    braking pulse (None when that follower collided).
    """
    collided = {name: bool(p.get("collided")) for name, p in profiles.items()}
    errors = {name: None if collided[name] else p["growth_error"]
              for name, p in profiles.items() if "growth_error" in p}  # fmt: skip
    kinds = {
        "mean": list(errors),
        "human": [name for name in errors if profiles[name].get("acc_flag") == 0],
        "acc": [name for name in errors if profiles[name].get("acc_flag") != 0],
    }  # fmt: skip
    means = {kind: _mean_or_none([errors[n] for n in names if not collided[n]]) for kind, names in kinds.items()}
    return {
        "growth_error": errors,
        "growth_error_mean": means["mean"],
        "growth_error_human": means["human"],
        "growth_error_acc": means["acc"],
        "growth_profiles": {kind: len(names) for kind, names in kinds.items()},
        "growth_collided": {kind: sum(collided[n] for n in names) for kind, names in kinds.items()},
        "n_profiles": len(profiles),
        "n_collided": sum(collided.values()),
        "n_start_equilibrium": sum(p.get("start") == "equilibrium" for p in profiles.values()),
        "std_ratio": {name: None if collided[name] else _std_ratio(p["speed_std"]) for name, p in profiles.items()},
        "hysteresis_area_pulse": profiles[pulse].get("hysteresis_area") if pulse in profiles else None,
    }


SUMMARY_KEYS = (
    "speed_std", "min_gap", "collided", "collision_step", "collision_vehicle", "max_abs_acc", "start", "start_gap",
    "start_gap_m", "start_equilibrium",
)  # fmt: skip


def platoon_test(model: CFModel, cfg: Mapping[str, Any], band: Band | None = None) -> dict[str, Any]:
    """Platoon growth test on the profiles of ``cfg`` (``configs/stability/platoon.yaml``), JSON-ready.

    All profiles are simulated together. Per OpenACC profile: model curve, collisions, empirical
    curve and growth error (None when the platoon collided: the speeds after a collision are no car
    following); the braking pulse has no empirical curve. Every profile gets its start (``start``,
    ``start_gap_m``; ``band`` is the spacing band of the model's training data for the start
    ``anchored``) and the hysteresis loop area of follower 1 (:func:`hysteresis_area`; None once that
    follower collided). Entries ``{profile, std}`` of the ``empirical_growth`` file add a growth error
    against hand-entered values for that profile (None as well after a collision).
    """
    n_vehicles, dt = int(cfg.get("n_vehicles", 50)), float(cfg.get("dt", DT))
    raw_dir = resolve_path(cfg.get("raw_dir", "data/raw/openacc"))
    empirical = {name: openacc_profile(raw_dir / name, dt) for name in cfg.get("profiles") or []}
    profiles = {name: profile["v_lead"] for name, profile in empirical.items()}
    if cfg.get("pulse"):
        profiles["pulse"] = braking_pulse(**cfg["pulse"], dt=dt)
    options = {key: cfg[key] for key in ("start_gap", "gap_s0", "gap_T", "device") if cfg.get(key) is not None}
    sims = simulate_platoons(model, list(profiles.values()), n_vehicles, dt, band=band, **options)
    results: dict[str, dict[str, Any]] = {}
    for name, sim in zip(profiles, sims):
        results[name] = {k: sim[k] for k in SUMMARY_KEYS}
        gap, speed = sim["s"][1], sim["v"][1]  # follower 1
        results[name]["hysteresis_area"] = None if bool((gap <= 0.0).any()) else hysteresis_area(gap, speed)
        if name in empirical:
            run = empirical[name]
            results[name].update({k: run[k] for k in ("acc_flag", "vehicle_order", "duration_s", "empirical_std")})
            results[name]["growth_error"] = (None if sim["collided"]
                                             else growth_error(sim["speed_std"], run["empirical_std"]))  # fmt: skip
    if cfg.get("empirical_growth"):
        references = yaml.safe_load(resolve_path(cfg["empirical_growth"]).read_text(encoding="utf-8")) or {}
        for ref_name, entry in references.items():
            if entry is not None:
                target = results[entry["profile"]]
                errors = target.setdefault("growth_error_reference", {})
                errors[ref_name] = None if target["collided"] else growth_error(target["speed_std"], entry["std"])
    return to_json(results)
