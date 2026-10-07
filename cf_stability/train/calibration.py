"""IDM calibration (docs/data_contract.md, section 6): closed-loop objective, batched DE.

Per event: ``J = NRMSE(s) + NRMSE(v)`` of a closed-loop rollout from the observed initial
state, ``J = collision_penalty + share of samples with s <= 0`` after a collision.
Global: NRMSE pooled over all events plus ``collision_penalty * collision rate``; the global
fit is generic over memoryless models (:func:`calibrate_global_model`, used for the OVM too).
Both fits can carry the term of D72 that demands a string-stability margin on a speed grid
(global: ``CalibrationConfig.stability_margin``; per event: the argument ``margin`` of
:func:`calibrate_per_event`, D118).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from cf_stability.data.schema import DT, Event
from cf_stability.models.idm import IDM_BOUNDS, IDM_PARAM_NAMES, idm_acc, idm_margin
from cf_stability.train.closed_loop import AccFn, iterate_memoryless, pad_events
from cf_stability.train.de import differential_evolution_batched

AT_BOUND_FRACTION = 0.01  # an estimate within 1 % of the range from a bound is "at the bound"
RESTART_SEED_STEP = 100_003  # seed offset between the restarts of the optimiser
ID_COLUMNS = ("event_id", "dataset", "site", "follower_id")


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class CalibrationConfig:
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(IDM_BOUNDS))
    popsize: int = 15
    maxiter: int = 300
    tol: float = 1e-3  # SciPy's default 0.01 leaves the parameters noisy (docs/decisions.md, D34)
    atol: float = 1e-4  # lets events with J close to 0 converge (SciPy's default is 0)
    global_tol: float = 1e-4  # the pooled objective is flat relative to its size: tol would stop too early
    restarts: int = 3  # independent runs per event (and for the global fit), the best one is kept
    mutation: float | tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    collision_penalty: float = 10.0
    batch_events: int = 4096
    device: str = "auto"
    seed: int = 0
    # parameters kept at a given value in the per-event fit, e.g. {"v0": 27.4, "b": 0.5};
    # the global fit always estimates all five
    fixed: dict[str, float] = field(default_factory=dict)
    # global IDM fit only: the analytic string-stability margin must be at least this value at
    # every speed of ``stability_speeds`` (the core of a certified hybrid); None = unconstrained.
    # The per-event fit takes its margin as an argument (calibrate_per_event, D118) with the weight
    # and the speeds below
    stability_margin: float | None = None
    stability_weight: float = 10.0  # weight of the shortfall, summed over the speeds
    stability_speeds: tuple[float, ...] = tuple(float(v) for v in range(5, 31))

    def __post_init__(self) -> None:
        unknown = set(self.fixed) - set(IDM_PARAM_NAMES)
        if unknown:
            raise ValueError(f"unknown fixed parameters: {sorted(unknown)}")
        if len(self.fixed) == len(IDM_PARAM_NAMES):
            raise ValueError("at least one parameter must be free")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CalibrationConfig":
        values = dict(mapping)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown calibration keys: {sorted(unknown)}")
        if "bounds" in values:
            bounds = {**IDM_BOUNDS, **values["bounds"]}
            values["bounds"] = {name: (float(bounds[name][0]), float(bounds[name][1])) for name in IDM_PARAM_NAMES}
        if "mutation" in values and not isinstance(values["mutation"], (int, float)):
            values["mutation"] = tuple(float(m) for m in values["mutation"])
        values["fixed"] = {name: float(value) for name, value in (values.get("fixed") or {}).items()}
        if "stability_speeds" in values:
            values["stability_speeds"] = tuple(float(v) for v in values["stability_speeds"])
        return cls(**values)

    def bound_tensors(self, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        lo, hi = zip(*(self.bounds[name] for name in IDM_PARAM_NAMES))
        return (
            torch.tensor(lo, dtype=torch.float64, device=device),
            torch.tensor(hi, dtype=torch.float64, device=device),
        )

    @property
    def free_index(self) -> list[int]:
        """Positions of the estimated parameters in ``IDM_PARAM_NAMES``."""
        return [k for k, name in enumerate(IDM_PARAM_NAMES) if name not in self.fixed]

    def expand(self, free: Tensor) -> Tensor:
        """Full parameter vectors ``[..., 5]`` from the estimated ones ``[..., n_free]``."""
        full = free.new_tensor([self.fixed.get(name, 0.0) for name in IDM_PARAM_NAMES])
        full = full.expand(*free.shape[:-1], len(IDM_PARAM_NAMES)).clone()
        full[..., self.free_index] = free
        return full


# ---------------------------------------------------------------------------- objective
def idm_acc_factory(params: Tensor) -> AccFn:
    """Acceleration law of IDM parameters ``[..., 5]`` (broadcast against the states)."""
    v0, T, s0, a, b = params.unbind(-1)
    return lambda s, dv, v: idm_acc(s, dv, v, v0, T, s0, a, b)


def idm_objective(
    params: Tensor, batch: Mapping[str, Tensor], *, collision_penalty: float = 10.0, dt: float = DT
) -> tuple[Tensor, dict[str, Tensor]]:
    """Objective ``J [N, P]`` of IDM parameters ``[N or 1, P, 5]`` on a padded batch of N events."""
    return closed_loop_objective(idm_acc_factory(params), batch, collision_penalty=collision_penalty, dt=dt)


def closed_loop_objective(
    acc_fn: AccFn, batch: Mapping[str, Tensor], *, collision_penalty: float = 10.0, dt: float = DT
) -> tuple[Tensor, dict[str, Tensor]]:
    """Objective ``J [N, P]`` of a memoryless law on a padded batch of N events.

    ``acc_fn`` receives states ``[N, 1]`` and broadcasts them over P parameter vectors.
    Squared errors are accumulated inside the time loop (memory O(N P), not O(N P T)).
    ``parts`` holds the sums needed for pooling: ``sse_s, sse_v, n_coll`` ``[N, P]`` and
    ``ss_s, ss_v, n`` ``[N, 1]`` (sums over the valid samples of the observed squares).
    """
    s_obs, v_obs = batch["s"], batch["v"]
    w = batch["mask"].to(s_obs.dtype)
    sse_s = sse_v = n_coll = s_obs.new_zeros(())
    states = iterate_memoryless(
        acc_fn, batch["x_lead"][:, None, :], batch["v_lead"][:, None, :], s_obs[:, :1], v_obs[:, :1], dt=dt
    )
    for k, (s, v, _, _) in enumerate(states):
        w_k = w[:, k : k + 1]
        err_s = s - s_obs[:, k : k + 1]
        err_v = v - v_obs[:, k : k + 1]
        sse_s = sse_s + err_s * err_s * w_k
        sse_v = sse_v + err_v * err_v * w_k
        n_coll = n_coll + (s <= 0.0) * w_k
    parts = {
        "sse_s": sse_s,
        "sse_v": sse_v,
        "n_coll": n_coll,
        "ss_s": (s_obs**2 * w).sum(-1, keepdim=True),
        "ss_v": (v_obs**2 * w).sum(-1, keepdim=True),
        "n": w.sum(-1, keepdim=True),
    }
    return objective_from_parts(parts, collision_penalty), parts


def objective_from_parts(parts: Mapping[str, Tensor], collision_penalty: float = 10.0) -> Tensor:
    fit = torch.sqrt(parts["sse_s"] / parts["ss_s"]) + torch.sqrt(parts["sse_v"] / parts["ss_v"])
    return torch.where(parts["n_coll"] > 0, collision_penalty + parts["n_coll"] / parts["n"], fit)


def metrics_from_parts(parts: Mapping[str, Tensor]) -> dict[str, Tensor]:
    n = parts["n"].expand_as(parts["sse_s"])
    return {
        "nrmse_s": torch.sqrt(parts["sse_s"] / parts["ss_s"]),
        "nrmse_v": torch.sqrt(parts["sse_v"] / parts["ss_v"]),
        "rmse_s": torch.sqrt(parts["sse_s"] / n),
        "rmse_v": torch.sqrt(parts["sse_v"] / n),
        "collided": parts["n_coll"] > 0,
        "collision_fraction": parts["n_coll"] / n,
        "n_samples": n,
    }


def _energy(pop: Tensor, batch: Mapping[str, Tensor], collision_penalty: float) -> Tensor:
    return idm_objective(pop, batch, collision_penalty=collision_penalty)[0]


def _with_margin(pop: Tensor, energy: Callable[[Tensor], Tensor], margin: float, cfg: CalibrationConfig) -> Tensor:
    """``energy(pop)`` plus ``cfg.stability_weight`` x the shortfall of the margin over ``cfg.stability_speeds``
    (D118)."""
    return energy(pop) + cfg.stability_weight * idm_stability_shortfall(pop, margin, cfg.stability_speeds)


def _length_batches(events: Sequence[Event], batch_size: int) -> Iterator[np.ndarray]:
    order = np.argsort([len(ev) for ev in events], kind="stable")
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _evaluate_batch(x: Tensor, batch: Mapping[str, Tensor], collision_penalty: float) -> dict[str, np.ndarray]:
    """Metrics of one parameter vector per event (``x [N, 5]``) as numpy columns."""
    J, parts = idm_objective(x[:, None, :], batch, collision_penalty=collision_penalty)
    columns = {"objective": J, **metrics_from_parts(parts)}
    out = {name: values[:, 0].cpu().numpy() for name, values in columns.items()}
    out["n_samples"] = out["n_samples"].astype(np.int64)
    return out


def _id_frame(events: Sequence[Event]) -> pd.DataFrame:
    return pd.DataFrame({name: [getattr(ev, name) for ev in events] for name in ID_COLUMNS})


# ---------------------------------------------------------------------------- per event
MARGIN_COLUMNS = ("fit_objective", "margin_min", "margin_holds")  # per-event fit with a margin (D118)
# the margin holds when margin_min >= margin - MARGIN_TOLERANCE: the optimum sits on the constraint and the
# differential evolution ends a few 1e-6 on either side of it (smoke run of D118: 0.1999988 for a margin of 0.2)
MARGIN_TOLERANCE = 1e-4


def calibrate_per_event(
    events: Sequence[Event], cfg: CalibrationConfig | None = None, *, margin: float | None = None
) -> pd.DataFrame:
    """Calibrate one IDM parameter vector per event; rows follow the input order.

    Events are sorted by length and optimised in batches of ``cfg.batch_events``
    (batch ``i`` uses the seed ``cfg.seed + i``).

    ``margin`` (D118): the objective of every event also carries ``cfg.stability_weight`` x the
    shortfall of the analytic string-stability margin below ``margin`` summed over
    ``cfg.stability_speeds`` (the term of D72, :func:`idm_stability_shortfall`). The frame then
    has ``objective`` = that total (the minimised value), ``fit_objective`` = ``J`` alone,
    ``margin_min`` (smallest margin over the speeds, -10 at a speed without equilibrium) and
    ``margin_holds`` (``margin_min >= margin - MARGIN_TOLERANCE``). Without ``margin`` the frame is that of M1.
    """
    cfg = cfg or CalibrationConfig()
    device = resolve_device(cfg.device)
    lower, upper = (bound[cfg.free_index] for bound in cfg.bound_tensors(device))
    frames = []
    for i, idx in enumerate(_length_batches(events, cfg.batch_events)):
        batch = pad_events([events[j] for j in idx], device=device)
        energy = partial(_energy, batch=batch, collision_penalty=cfg.collision_penalty)
        if margin is not None:
            energy = partial(_with_margin, energy=energy, margin=float(margin), cfg=cfg)
        x = fun = converged = n_generations = None
        for r in range(max(1, cfg.restarts)):
            res = differential_evolution_batched(
                lambda pop: energy(cfg.expand(pop)),
                lower,
                upper,
                len(idx),
                popsize=cfg.popsize,
                maxiter=cfg.maxiter,
                tol=cfg.tol,
                atol=cfg.atol,
                mutation=cfg.mutation,
                recombination=cfg.recombination,
                seed=cfg.seed + i + RESTART_SEED_STEP * r,
                device=device,
            )
            if x is None:
                x, fun, converged, n_generations = res.x, res.fun, res.converged, res.n_generations
            else:  # keep, per event, the run with the lowest objective
                better = res.fun < fun
                x = torch.where(better[:, None], res.x, x)
                fun = torch.where(better, res.fun, fun)
                converged = torch.where(better, res.converged, converged)
                n_generations = torch.where(better, res.n_generations, n_generations)
        x = cfg.expand(x)
        frame = pd.DataFrame(x.cpu().numpy(), columns=list(IDM_PARAM_NAMES), index=idx)
        frame = frame.join(pd.DataFrame(_evaluate_batch(x, batch, cfg.collision_penalty), index=idx))
        if margin is not None:
            frame["fit_objective"] = frame["objective"]
            shortfall = idm_stability_shortfall(x, float(margin), cfg.stability_speeds)
            frame["objective"] = (frame["fit_objective"] + cfg.stability_weight * shortfall.cpu().numpy()).to_numpy()
            frame["margin_min"] = idm_margin_min(x, cfg.stability_speeds).cpu().numpy()
            frame["margin_holds"] = frame["margin_min"] >= float(margin) - MARGIN_TOLERANCE
        frame["converged"] = converged.cpu().numpy()
        frame["n_generations"] = n_generations.cpu().numpy()
        frames.append(frame)
    df = pd.concat([_id_frame(events), pd.concat(frames).sort_index()], axis=1)
    for name in IDM_PARAM_NAMES:
        lo, hi = cfg.bounds[name]
        near = AT_BOUND_FRACTION * (hi - lo)
        on_bound = (df[name] - lo <= near) | (hi - df[name] <= near)
        df[f"at_bound_{name}"] = on_bound & (name not in cfg.fixed)
    columns = [
        *ID_COLUMNS,
        *IDM_PARAM_NAMES,
        "objective",
        *(MARGIN_COLUMNS if margin is not None else ()),
        "nrmse_s",
        "nrmse_v",
        "rmse_s",
        "rmse_v",
        "collided",
        "converged",
        "n_generations",
        *(f"at_bound_{name}" for name in IDM_PARAM_NAMES),
        "n_samples",
    ]
    return df[columns]


def evaluate_idm(
    events: Sequence[Event],
    params: Mapping[str, float] | pd.DataFrame | np.ndarray | Sequence[float],
    *,
    collision_penalty: float = 10.0,
    batch_events: int = 4096,
    device: str | torch.device = "auto",
) -> pd.DataFrame:
    """Per-event closed-loop metrics for one parameter vector or one per event.

    ``params``: mapping ``name -> value``, array ``[5]`` or ``[N, 5]``, or a frame with
    the parameter columns (one row per event, in event order).
    """
    if isinstance(params, pd.DataFrame):
        values = params[list(IDM_PARAM_NAMES)].to_numpy(dtype=np.float64)
    elif isinstance(params, Mapping):
        values = np.array([params[name] for name in IDM_PARAM_NAMES], dtype=np.float64)
    else:
        values = np.asarray(params, dtype=np.float64)
    values = np.broadcast_to(values, (len(events), len(IDM_PARAM_NAMES)))
    device = resolve_device(device)
    frames = []
    for idx in _length_batches(events, batch_events):
        batch = pad_events([events[j] for j in idx], device=device)
        x = torch.as_tensor(values[idx], dtype=torch.float64, device=device)
        frames.append(pd.DataFrame(_evaluate_batch(x, batch, collision_penalty), index=idx))
    return pd.concat([_id_frame(events), pd.concat(frames).sort_index()], axis=1)


# ---------------------------------------------------------------------------- global
def idm_stability_shortfall(params: Tensor, margin: float, speeds: Sequence[float]) -> Tensor:
    """Sum over the speeds of ``relu(margin - M)`` for IDM parameters ``[..., 5]``.

    ``M`` is the analytic string-stability margin at the equilibrium of the speed; a speed without
    equilibrium (``v >= v0``) counts as a margin of -10.
    """
    v = torch.as_tensor(speeds, dtype=params.dtype, device=params.device)
    m = idm_margin(v, *(params[..., k, None] for k in range(len(IDM_PARAM_NAMES))))
    m = torch.nan_to_num(m, nan=-10.0).clamp(min=-10.0)
    return torch.clamp(margin - m, min=0.0).sum(-1)


def idm_margin_min(params: Tensor, speeds: Sequence[float]) -> Tensor:
    """Smallest analytic string-stability margin over ``speeds`` for IDM parameters ``[..., 5]``; a speed
    without equilibrium (``v >= v0``) counts as -10, as in :func:`calibrate_global`'s ``margin_min``."""
    v = torch.as_tensor(speeds, dtype=params.dtype, device=params.device)
    m = idm_margin(v, *(params[..., k, None] for k in range(len(IDM_PARAM_NAMES))))
    return torch.nan_to_num(m, nan=-10.0).amin(-1)


def calibrate_global(events: Sequence[Event], cfg: CalibrationConfig | None = None) -> dict[str, Any]:
    """One IDM parameter vector for all events (pooled NRMSE + penalty x collision rate).

    With ``cfg.stability_margin`` the objective also carries ``cfg.stability_weight`` x the
    shortfall of the string-stability margin over ``cfg.stability_speeds``; the result then has
    ``margin_min`` and ``fit_objective`` (the objective without that term).
    """
    cfg = cfg or CalibrationConfig()
    if cfg.stability_margin is None:
        return calibrate_global_model(idm_acc_factory, IDM_PARAM_NAMES, cfg.bounds, events, cfg)
    margin, speeds = float(cfg.stability_margin), cfg.stability_speeds
    out = calibrate_global_model(
        idm_acc_factory, IDM_PARAM_NAMES, cfg.bounds, events, cfg,
        penalty=lambda pop: cfg.stability_weight * idm_stability_shortfall(pop, margin, speeds),
    )  # fmt: skip
    theta = torch.tensor([out["params"][name] for name in IDM_PARAM_NAMES], dtype=torch.float64)
    m = idm_margin(torch.as_tensor(speeds, dtype=torch.float64), *theta)
    out["stability_margin"] = margin
    out["margin_min"] = float(torch.nan_to_num(m, nan=-10.0).min())
    out["fit_objective"] = out["nrmse_s"] + out["nrmse_v"] + cfg.collision_penalty * out["collision_rate"]
    return out


def calibrate_global_model(
    acc_factory: Callable[[Tensor], AccFn],
    param_names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    events: Sequence[Event],
    cfg: CalibrationConfig | None = None,
    *,
    penalty: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, Any]:
    """One parameter vector of a memoryless model for all events (pooled NRMSE + penalty x collision rate).

    ``acc_factory`` maps parameters ``[..., len(param_names)]`` to ``acc_fn(s, dv, v)``. The
    optimiser settings come from ``cfg``; its IDM-specific ``bounds`` and ``fixed`` are not used.
    ``penalty`` maps a population ``[P, len(param_names)]`` to a term ``[P]`` added to the objective.
    """
    cfg = cfg or CalibrationConfig()
    device = resolve_device(cfg.device)
    batches = [pad_events([events[j] for j in idx], device=device) for idx in _length_batches(events, cfg.batch_events)]
    n_events = len(events)

    def pooled(params: Tensor) -> dict[str, Tensor]:
        acc_fn = acc_factory(params)
        totals: dict[str, Tensor] = {}
        for batch in batches:
            _, parts = closed_loop_objective(acc_fn, batch, collision_penalty=cfg.collision_penalty)
            sums = {name: parts[name].sum(0) for name in ("sse_s", "sse_v", "ss_s", "ss_v", "n")}
            sums["collided"] = (parts["n_coll"] > 0).sum(0)
            totals = {name: totals.get(name, 0) + value for name, value in sums.items()}
        return totals

    def energy(totals: Mapping[str, Tensor]) -> Tensor:
        fit = torch.sqrt(totals["sse_s"] / totals["ss_s"]) + torch.sqrt(totals["sse_v"] / totals["ss_v"])
        return fit + cfg.collision_penalty * totals["collided"] / n_events

    lower, upper = (
        torch.tensor([bounds[name][k] for name in param_names], dtype=torch.float64, device=device) for k in (0, 1)
    )
    res = None
    for r in range(max(1, cfg.restarts)):
        run = differential_evolution_batched(
            lambda pop: (energy(pooled(pop)) + (0.0 if penalty is None else penalty(pop[0])))[None, :],
            lower,
            upper,
            1,
            popsize=cfg.popsize,
            maxiter=cfg.maxiter,
            tol=cfg.global_tol,
            atol=0.0,
            mutation=cfg.mutation,
            recombination=cfg.recombination,
            seed=cfg.seed + RESTART_SEED_STEP * r,
            device=device,
        )
        if res is None or float(run.fun[0]) < float(res.fun[0]):
            res = run
    totals = {name: float(value.reshape(-1)[0]) for name, value in pooled(res.x[:, None, :]).items()}
    return {
        "params": dict(zip(param_names, res.x[0].tolist())),
        "objective": float(res.fun[0]),
        "rmse_s": (totals["sse_s"] / totals["n"]) ** 0.5,
        "rmse_v": (totals["sse_v"] / totals["n"]) ** 0.5,
        "nrmse_s": (totals["sse_s"] / totals["ss_s"]) ** 0.5,
        "nrmse_v": (totals["sse_v"] / totals["ss_v"]) ** 0.5,
        "collision_rate": totals["collided"] / n_events,
        "n_events": n_events,
        "n_generations": int(res.n_generations[0]),
        "converged": bool(res.converged[0]),
    }


# ---------------------------------------------------------------------------- spread
def parameter_spread(
    df: pd.DataFrame,
    bounds: Mapping[str, tuple[float, float]] = IDM_BOUNDS,
    fixed: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Distribution of per-event estimates: summary statistics, bound shares, correlations.

    Parameters in ``fixed`` were not estimated: they are reported under ``"fixed"`` only.
    """
    fixed = dict(fixed or {})
    names = [name for name in IDM_PARAM_NAMES if name not in fixed]
    stats = {}
    for name in names:
        x = df[name].astype(np.float64)
        lo, hi = bounds[name]
        margin = AT_BOUND_FRACTION * (hi - lo)
        mean, std = float(x.mean()), float(x.std())
        stats[name] = {
            "mean": mean,
            "std": std,
            "median": float(x.median()),
            **{f"q{round(100 * q):02d}": float(x.quantile(q)) for q in (0.05, 0.25, 0.75, 0.95)},
            "cv": std / mean,
            "share_at_lower": float((x - lo <= margin).mean()),
            "share_at_upper": float((hi - x <= margin).mean()),
        }
    corr = df[names].astype(np.float64).corr(method="pearson")
    return {
        "n_events": int(len(df)),
        "fixed": fixed,
        "parameters": stats,
        "correlation": {r: {c: float(corr.loc[r, c]) for c in names} for r in names},
    }
