"""Predictive validity of single-vehicle metrics for the macro error of the corridor (M11, D127, hypothesis H-N1).

    python scripts/analysis/predictive_validity.py            # runs/_tables/m11/predictive_validity{,_pairs}.{csv,md}
    python scripts/analysis/predictive_validity.py --states 20000 --resamples 5000 --permutations 10000

Units: the (law, corridor) pairs of the pooled correlation of H12.3 (``runs/_tables/m8/correlation_pooled_laws.csv``,
37 pairs: 19 laws on I-80, 18 on US-101) and, as a separate check, these plus the six pairs of the three laws of the
factorial ablation (``idm_core_margin``, ``idm_margin_i80``, ``residual_idm_margin_free_r0.3`` on both corridors).
Response: the macro error of the pair, the mean over its runs (``runs/_tables/m5/laws.csv``). Predictors (D127):

* (a) the pre-specified share of unstable equilibria (the H1.1 statistic of the members, ``unstable_eq`` of
  correlation_pooled and of the table instability; the exact gain of the parameter sets for the IDM laws without
  member runs and for the cores of ``residual_idm_certified_het``);
* (b) the share of the grid speeds in support with a band that are not stable in the band (``1 - band_numerical
  stable`` of the member audits; for a closed-form law the same categories from its exact gain and its equilibrium
  spacing against the band of the members' training contexts);
* (c) the largest numerical gain over the speeds in support and the 25 audit frequencies, (d) the same over the
  frequencies <= 0.1 rad/s (member audits; closed form: the exact discrete-time gain);
* (e) the transient growth of the platoon test behind the braking pulse: ``max_n sigma_n / sigma_1`` over the 50
  followers (speed standard deviation per position, ``platoon.json``; the platoon step was run for the members without
  it); a platoon with a collision has no bounded growth and counts as ``+inf``; mean over the members (``+inf`` once a
  member collided), for the laws without member runs over their parameter sets (all of them, or 100 drawn uniformly
  for the per-event laws), simulated as the platoon step simulates the pulse;
* (f) the frozen gain along the corridor trajectories: in the seed-0 runs of the pair (its three scenarios) every
  follower state ``(s, dv, v)`` with a leader in its lane within the 50 m of the loop (gap ``0 < s <= 50`` m, the
  leader the next vehicle ahead at the same whole second, ``s = x_lead - length_lead - x``, ``dv = v - v_lead``), 20 000
  of them drawn uniformly; at every state the memoryless partial derivatives of the law of that vehicle (its member,
  its parameter set or core, as the run drew them: ``law.assign`` with the generator of seed 0, checked against
  ``vehicles.npz``; autograd in float64, the memoryless view of the audit for the recurrent laws, central differences
  of the audit's step for k-NN) and the frozen gain ``max_w |G(e^{i w dt})|`` of the discrete-time linearisation of
  the audit (``transfer_discrete``, dt 0.1 s) over the 25 audit frequencies; (f1) the share of states with a gain
  above 1.02, (f2) the mean log gain;
* (g) the contact rate per 1000 vehicle-km (``laws.csv``); (h) the asymmetry index of the pair (mean over its runs,
  ``runs/_tables/m8/asymmetry.csv``), and (h') its distance to the drivers, the mean absolute relative error of the
  index against the ground truth of the run's scenario (same table; shown beside the index, whose sign has no
  "worse" direction).

Statistics: Spearman per corridor and pooled with the corridor as a stratum (``stratified_spearman``), the percentile
interval of a cluster bootstrap over the laws (``cluster_bootstrap_spearman``: the rows of a law on both corridors move
together), the permutation p-value (whole laws permuted, ``linked_permutation_p``, for the predictors that are one
value per law; within every corridor, ``stratified_permutation_p``, for the predictors of a pair: (f), (g), (h)), and
for every predictor P the paired cluster-bootstrap interval of ``rho_P - rho_(a)`` (the same resamples of the laws for
both). D-vs-E rank check: P ranks E (``residual_idm_certified``) better than D (``residual_idm_margin_free_r0.3``) on
both corridors when ``P(E) < P(D)`` strictly on each. Rule of D127: P "predicts better" when the interval of
``rho_P - rho_(a)`` lies above zero and P ranks E better than D on both corridors.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cf_stability.eval.stats import (  # noqa: E402
    cluster_bootstrap_spearman,
    linked_permutation_p,
    multiplicities,
    stratified_permutation_p,
    stratified_spearman,
    weighted_pearson,
    weighted_stratified_ranks,
)
from cf_stability.eval.tables import Column, Table, markdown  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

CORRIDORS = {"I-80": ("i80_p0", "i80_p1", "i80_p2"), "US-101": ("us101_p0", "us101_p1", "us101_p2")}
ABLATION_LAWS = ("idm_core_margin", "idm_margin_i80", "residual_idm_margin_free_r0.3")
LAW_E, LAW_D = "residual_idm_certified", "residual_idm_margin_free_r0.3"
V_GRID = tuple(float(v) for v in range(5, 31))
OMEGA = np.geomspace(0.02, 2.0, 25)  # the audit frequencies (FrequencyConfig)
LOW_OMEGA = 0.1  # rad/s: (d) the frequencies <= 0.1 rad/s
DT = 0.1
THRESHOLD = 1.02
LEADER_RANGE = 50.0
FOUND = ("ok", "multiple", "outside")
INF_SENTINEL = 1e12  # +inf (a collided platoon) as the largest value: the statistics are rank statistics
N_PLATOON_SETS = 100  # parameter sets of a per-event law simulated in the platoon test
SEED = 20261019
PREDICTORS = {  # key: (letter, name, unit of the value: law or pair)
    "unstable_eq": ("a", "share of unstable equilibria (H1.1, pre-specified)", "law"),
    "not_stable": ("b", "share of grid speeds not stable in the band", "law"),
    "max_gain": ("c", "largest numerical gain (speeds in support, 25 frequencies)", "law"),
    "max_gain_low": ("d", "largest numerical gain at <= 0.1 rad/s", "law"),
    "platoon_growth": ("e", "platoon transient growth behind the braking pulse", "law"),
    "frozen_share": ("f1", "frozen gain along the trajectories: share of states > 1.02", "pair"),
    "frozen_log": ("f2", "frozen gain along the trajectories: mean log gain", "pair"),
    "contacts": ("g", "contact rate per 1000 vehicle-km", "pair"),
    "asymmetry": ("h", "asymmetry index", "pair"),
    "asymmetry_error": ("h'", "|relative error| of the asymmetry index (distance to the drivers)", "pair"),
}
REFERENCE = "unstable_eq"


# ------------------------------------------------------------------------------------------ the frozen gain


def frozen_gain(f_s: np.ndarray, f_dv: np.ndarray, f_v: np.ndarray, omega: np.ndarray = OMEGA, dt: float = DT
                ) -> np.ndarray:  # fmt: skip
    """``max_w |G(e^{i w dt})|`` ``[n]`` of the discrete-time linearisation of the audit (``transfer_discrete``) at the
    partial derivatives of every state."""
    from cf_stability.stability.analytic import transfer_discrete

    f = [torch.as_tensor(np.asarray(x, dtype=np.float64)) for x in (f_s, f_dv, f_v)]
    gain = transfer_discrete(*f, torch.as_tensor(np.asarray(omega, dtype=np.float64)), dt).abs()
    return gain.max(dim=1).values.numpy()


def frozen_statistics(gain: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    """(f1) share of the states with a frozen gain above ``threshold``; (f2) mean log gain (a gain of 0 counts as the
    smallest positive float)."""
    g = np.asarray(gain, dtype=np.float64)
    g = g[np.isfinite(g)]
    if not len(g):
        return {"frozen_share": math.nan, "frozen_log": math.nan, "n_states": 0}
    return {"frozen_share": float(np.mean(g > threshold)),
            "frozen_log": float(np.mean(np.log(np.maximum(g, np.finfo(np.float64).tiny)))), "n_states": int(len(g))}  # fmt: skip


def follower_states(trajectories: Mapping[str, np.ndarray], lengths: np.ndarray, leader_range: float = LEADER_RANGE
                    ) -> dict[str, np.ndarray]:  # fmt: skip
    """Follower states of a run from ``trajectories.npz`` (the samples of every vehicle at whole seconds on the main
    line of the section) and the vehicle lengths of ``vehicles.npz``: at every second and lane the leader of a vehicle
    is the next vehicle ahead; ``s = x_lead - length_lead - x`` (the positions are those of the fronts, SUMO's lane
    positions with ``minGap = 0``), ``dv = v - v_lead``. Kept: ``0 < s <= leader_range`` (a leader inside the range of
    the loop, no contact). ``{vehicle, t, s, dv, v}``."""
    t = np.asarray(trajectories["t"], dtype=np.float64)
    veh = np.asarray(trajectories["vehicle"], dtype=np.int64)
    x = np.asarray(trajectories["x"], dtype=np.float64)
    lane = np.asarray(trajectories["lane"], dtype=np.int64)
    v = np.asarray(trajectories["v"], dtype=np.float64)
    order = np.lexsort((x, lane, t))
    t, veh, x, lane, v = t[order], veh[order], x[order], lane[order], v[order]
    same = (t[1:] == t[:-1]) & (lane[1:] == lane[:-1])  # the next row is the vehicle ahead in the same lane and second
    follower, leader = np.flatnonzero(same), np.flatnonzero(same) + 1
    s = x[leader] - np.asarray(lengths, dtype=np.float64)[veh[leader]] - x[follower]
    keep = (s > 0.0) & (s <= leader_range)
    f, ld = follower[keep], leader[keep]
    return {"vehicle": veh[f], "t": t[f], "s": s[keep], "dv": v[f] - v[ld], "v": v[f]}


def idm_partials_rows(params: np.ndarray, s: np.ndarray, dv: np.ndarray, v: np.ndarray, delta: float = 4.0,
                      s_eps: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """``(f_s, f_dv, f_v)`` of the IDM with one parameter set per state (``params [n, 5]``: v0, T, s0, a, b) at the
    states ``(s, dv, v)``: float64 autograd of ``idm_acc``."""
    from cf_stability.models.idm import idm_acc

    p = torch.as_tensor(np.asarray(params, dtype=np.float64))
    x = [torch.as_tensor(np.asarray(a, dtype=np.float64)).clone().requires_grad_(True) for a in (s, dv, v)]
    with torch.enable_grad():
        acc = idm_acc(x[0], x[1], x[2], *p.unbind(-1), delta=delta, s_eps=s_eps)
        grads = torch.autograd.grad(acc.sum(), x, allow_unused=True)
    zero = np.zeros(len(x[0]))
    return tuple(zero if g is None else g.numpy() for g in grads)  # type: ignore[return-value]


def residual_partials(model: Any, s: np.ndarray, dv: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(f_s, f_dv, f_v)`` of the residual of a ResidualIDM alone at the states (float64 autograd)."""
    from cf_stability.stability.equilibrium import constant_history

    m = copy.deepcopy(model).cpu().double().eval()
    x = [torch.as_tensor(np.asarray(a, dtype=np.float64)).clone().requires_grad_(True) for a in (s, dv, v)]
    with torch.enable_grad():
        history = constant_history(m, x[0], x[2], x[1])
        grads = torch.autograd.grad(m.residual(history).sum(), x, allow_unused=True)
    zero = np.zeros(len(x[0]))
    return tuple(zero if g is None else g.numpy() for g in grads)


def law_partials(law: Any, rows: np.ndarray, s: np.ndarray, dv: np.ndarray, v: np.ndarray,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """The memoryless partial derivatives of the law that drives the vehicles ``rows`` (after ``law.assign``) at their
    states: per-vehicle IDM parameters (``IDMLaw``), the vehicle's member (``ModelLaw``), its core plus the residual of
    its member (``ResidualHetLaw``)."""
    from cf_stability.corridor.laws import IDM_DEFAULTS, IDMLaw, ModelLaw, ResidualHetLaw
    from cf_stability.stability.robust import model_partials

    rows = np.asarray(rows, dtype=np.int64)
    out = [np.full(len(rows), np.nan) for _ in range(3)]
    if isinstance(law, IDMLaw):
        f = idm_partials_rows(law.params[rows], s, dv, v, **IDM_DEFAULTS)
        return f
    if isinstance(law, (ModelLaw, ResidualHetLaw)):
        member = law.member[rows]
        for m, model in enumerate(law.models):
            idx = np.flatnonzero(member == m)
            if not len(idx):
                continue
            if isinstance(law, ResidualHetLaw):
                core = idm_partials_rows(law.params[rows[idx]], s[idx], dv[idx], v[idx], delta=model.idm.delta,
                                         s_eps=model.idm.s_eps)  # fmt: skip
                res = residual_partials(model, s[idx], dv[idx], v[idx])
                f = tuple(a + b for a, b in zip(core, res))
            else:
                f = model_partials(model, s[idx], v[idx], dv[idx], device=device)
            for k in range(3):
                out[k][idx] = f[k]
        return tuple(out)  # type: ignore[return-value]
    raise TypeError(f"unknown law type {type(law).__name__}")


def pair_frozen_gain(
    law_name: str, corridor: str, corridor_root: Path, n_states: int, seed: int, device: str = "cpu",
) -> dict[str, Any]:  # fmt: skip
    """(f) of one pair: the follower states of the seed-0 runs of the corridor's scenarios, ``n_states`` drawn
    uniformly without replacement from all of them, the frozen gain at each."""
    from cf_stability.corridor.laws import load_law

    law, _ = load_law(corridor_root / "laws" / f"{law_name}.json", "cpu")
    parts, problems = [], []
    for k, scenario in enumerate(CORRIDORS[corridor]):
        run = corridor_root / scenario / law_name / "seed0"
        if not (run / "trajectories.npz").is_file() or not (run / "vehicles.npz").is_file():
            problems.append(f"{scenario}/{law_name}/seed0: trajectories.npz or vehicles.npz missing")
            continue
        with np.load(run / "trajectories.npz") as data:
            traj = {key: data[key] for key in data.files}
        with np.load(run / "vehicles.npz") as data:
            vehicles = {key: data[key] for key in data.files}
        states = follower_states(traj, vehicles["length"])
        states["scenario"] = np.full(len(states["s"]), k)
        parts.append((scenario, states, vehicles))
    if not parts:
        return {"problems": problems, "frozen_share": math.nan, "frozen_log": math.nan, "n_states": 0}
    sizes = [len(p[1]["s"]) for p in parts]
    total = int(sum(sizes))
    rng = np.random.default_rng([seed, zlib.crc32(law_name.encode()), zlib.crc32(corridor.encode())])
    pick = np.sort(rng.choice(total, size=min(n_states, total), replace=False))
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    gains = np.full(len(pick), np.nan)
    per_scenario = {}
    for k, (scenario, states, vehicles) in enumerate(parts):
        mine = pick[(pick >= offsets[k]) & (pick < offsets[k + 1])] - offsets[k]
        per_scenario[scenario] = int(len(mine))
        n_vehicles = len(vehicles["vehicle_id"])
        members = law.assign(n_vehicles, np.random.default_rng(0))  # the draws of the run (seed 0)
        if not np.array_equal(np.asarray(members, dtype=np.int64), vehicles["member"].astype(np.int64)):
            raise RuntimeError(f"{scenario}/{law_name}/seed0: the members drawn with seed 0 differ from vehicles.npz")
        if not len(mine):
            continue
        rows = states["vehicle"][mine]
        f = law_partials(law, rows, states["s"][mine], states["dv"][mine], states["v"][mine], device=device)
        where = np.flatnonzero((pick >= offsets[k]) & (pick < offsets[k + 1]))
        gains[where] = frozen_gain(*f)
    out = frozen_statistics(gains)
    out.update(problems=problems, states_available=total, states_per_scenario=per_scenario,
               nonfinite=int((~np.isfinite(gains)).sum()), gain_median=float(np.nanmedian(gains)),
               gain_p90=float(np.nanquantile(gains, 0.9)))  # fmt: skip
    return out


# ------------------------------------------------------------------------- member audits and closed forms


def audit_values(audit: Mapping[str, Any]) -> dict[str, float]:
    """(b), (c), (d) of one member audit: ``1 - band_numerical["support"]["stable"]``; the largest numerical gain over
    the equilibria in support (inside or outside the band: the speeds of the H1.1 statistic) and every frequency, and
    over the frequencies <= 0.1 rad/s."""
    omega = np.asarray(audit["omega"], dtype=np.float64)
    low = omega <= LOW_OMEGA + 1e-12
    band = ((audit.get("summary") or {}).get("band_numerical") or {}).get("support")
    gains = [np.asarray([np.nan if g is None else g for g in r["gain"]], dtype=np.float64)
             for r in audit["equilibria"] if r.get("in_support") and r.get("status") in FOUND and r.get("gain")]  # fmt: skip
    if gains:
        stack = np.stack(gains)
        finite = np.isfinite(stack)
        max_gain = float(np.nanmax(stack)) if finite.any() else math.nan
        max_low = float(np.nanmax(stack[:, low])) if finite[:, low].any() else math.nan
    else:
        max_gain = max_low = math.nan
    return {"not_stable": math.nan if band is None else 1.0 - float(band["stable"]), "max_gain": max_gain,
            "max_gain_low": max_low}  # fmt: skip


def closed_form_values(params: np.ndarray, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """(a)-(d) of IDM parameter sets ``[n, 5]`` (v0, T, s0, a, b) in the training context of a member: the grid speeds
    in support (``box_low[2] .. box_high[2]``), the exact discrete-time gain at the equilibrium (as ``idm_unstable_share``
    of the corridor tables), the band categories of the audit from the equilibrium spacing and the band of the context
    (stable: equilibrium inside the band and largest gain <= 1.02). Per set: ``unstable_eq`` (NaN without an
    equilibrium in support), ``not_stable``, ``max_gain``, ``max_gain_low``."""
    from cf_stability.models.idm import idm_equilibrium_partials
    from cf_stability.stability.analytic import transfer_discrete
    from cf_stability.stability.equilibrium import Band

    support = (float(context["box_low"][2]), float(context["box_high"][2]))
    speeds = np.array([v for v in V_GRID if support[0] <= v <= support[1]])
    p = torch.as_tensor(np.asarray(params, dtype=np.float64))
    v = torch.as_tensor(speeds)[None, :]
    s_e, f_s, f_dv, f_v = idm_equilibrium_partials(v, *(p[:, k : k + 1] for k in range(5)))
    n, m = s_e.shape
    gain = transfer_discrete(f_s.reshape(-1), f_dv.reshape(-1), f_v.reshape(-1), torch.as_tensor(OMEGA), DT).abs()
    gain = gain.reshape(n, m, len(OMEGA)).numpy()
    exists = np.isfinite(s_e.numpy()) & np.isfinite(gain).all(axis=-1)
    top = np.where(exists, gain.max(axis=-1), np.nan)
    top_low = np.where(exists, gain[..., OMEGA <= LOW_OMEGA + 1e-12].max(axis=-1), np.nan)
    unstable = exists & (top > THRESHOLD)
    n_eq = exists.sum(axis=1)
    out = {
        "unstable_eq": np.where(n_eq > 0, unstable.sum(axis=1) / np.maximum(n_eq, 1), np.nan),
        "max_gain": np.where(n_eq > 0, np.nanmax(np.where(exists, top, -np.inf), axis=1), np.nan),
        "max_gain_low": np.where(n_eq > 0, np.nanmax(np.where(exists, top_low, -np.inf), axis=1), np.nan),
    }
    band = Band.from_mapping(context["band"])
    low, high, has = (x.numpy() for x in band.at(torch.as_tensor(speeds)))
    if has.any():
        s_np = s_e.numpy()
        inside = exists & (s_np >= low[None, :]) & (s_np <= high[None, :])
        stable = inside & ~unstable
        out["not_stable"] = 1.0 - stable[:, has].sum(axis=1) / has.sum()
    else:
        out["not_stable"] = np.full(n, np.nan)
    return out


def transient_growth(profile: Mapping[str, Any]) -> float:
    """``max_n sigma_n / sigma_1`` over the followers (``speed_std``, index 0 the leader) of a platoon profile; ``+inf``
    when a follower collided (``min_gap <= 0`` or ``collided``); NaN when follower 1 does not move."""
    std = np.asarray([np.nan if x is None else x for x in profile["speed_std"]], dtype=np.float64)
    gaps = profile.get("min_gap")
    collided = bool(profile.get("collided"))
    if gaps is not None:
        g = np.asarray([np.nan if x is None else x for x in gaps], dtype=np.float64)
        collided = collided or bool(np.any(g[1:] <= 0.0))
    if collided:
        return math.inf
    if len(std) < 3 or not np.isfinite(std[1]) or std[1] <= 0.0:
        return math.nan
    return float(np.nanmax(std[1:]) / std[1])


def law_mean(values: Sequence[float]) -> float:
    """Mean over the members; ``+inf`` when one is ``+inf``; NaN without a finite or infinite value."""
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return math.nan
    return math.inf if np.isinf(x).any() else float(x.mean())


def pulse_growth_of_models(models: Sequence[Any], contexts: Sequence[Mapping[str, Any] | None],
                           platoon_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """The braking pulse of the platoon test (``configs/stability/platoon.yaml``: its pulse, vehicles, start and
    options) simulated for every model with the band of its context: ``[{growth, collided}]``."""
    from cf_stability.stability.equilibrium import Band
    from cf_stability.stability.platoon import braking_pulse, simulate_platoons

    dt = float(platoon_cfg.get("dt", DT))
    pulse = braking_pulse(**platoon_cfg["pulse"], dt=dt)
    options = {key: platoon_cfg[key] for key in ("start_gap", "gap_s0", "gap_T", "device")
               if platoon_cfg.get(key) is not None}  # fmt: skip
    out = []
    for model, context in zip(models, contexts):
        band = Band.from_context(context, model) if context is not None else None
        sim = simulate_platoons(model, [pulse], int(platoon_cfg.get("n_vehicles", 51)), dt, band=band, **options)[0]
        record = {"speed_std": sim["speed_std"].tolist(), "min_gap": sim["min_gap"].tolist(), "collided": sim["collided"]}
        out.append({"growth": transient_growth(record), "collided": bool(sim["collided"])})
    return out


# --------------------------------------------------------------------------------------------- statistics


def paired_difference(
    x: np.ndarray, x_ref: np.ndarray, y: np.ndarray, strata: np.ndarray, clusters: np.ndarray,
    draws: np.ndarray, names: Sequence[Any], level: float = 0.95,
) -> dict[str, float]:  # fmt: skip
    """``rho(x, y) - rho(x_ref, y)`` (stratified Spearman) over the rows where both are finite, with the percentile
    interval of the cluster bootstrap whose resamples are ``draws [R, k]`` (indices into ``names``, the clusters of
    all rows): every resample weights the rows of a cluster by its multiplicity, the same for both predictors."""
    x, x_ref, y = (np.asarray(a, dtype=np.float64) for a in (x, x_ref, y))
    keep = np.isfinite(x) & np.isfinite(x_ref) & np.isfinite(y)
    out = {"difference": math.nan, "low": math.nan, "high": math.nan, "n": int(keep.sum()), "valid": 0}
    if keep.sum() < 3:
        return out
    x, x_ref, y, s = x[keep], x_ref[keep], y[keep], np.asarray(strata)[keep]
    index = {name: k for k, name in enumerate(names)}
    codes = np.array([index[c] for c in np.asarray(clusters)[keep]], dtype=np.int64)
    out["difference"] = stratified_spearman(x, y, s) - stratified_spearman(x_ref, y, s)
    weights = multiplicities(draws, len(names))[:, codes]
    ranks_y = weighted_stratified_ranks(y, s, weights)
    r = weighted_pearson(weighted_stratified_ranks(x, s, weights), ranks_y, weights)
    r_ref = weighted_pearson(weighted_stratified_ranks(x_ref, s, weights), ranks_y, weights)
    diff = r - r_ref
    valid = diff[np.isfinite(diff)]
    out["valid"] = int(len(valid))
    if len(valid):
        alpha = 1.0 - level
        out["low"], out["high"] = (float(q) for q in np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0]))
    return out


def rank_check(values: Mapping[tuple[str, str], float], corridors: Sequence[str], better: str = LAW_E,
               worse: str = LAW_D) -> bool | None:  # fmt: skip
    """``P(better) < P(worse)`` strictly on every corridor (a larger value of P means a larger predicted error);
    None when a value is missing."""
    out = True
    for corridor in corridors:
        a, b = values.get((better, corridor), math.nan), values.get((worse, corridor), math.nan)
        if not (np.isfinite(a) or np.isinf(a)) or not (np.isfinite(b) or np.isinf(b)):
            return None
        out = out and bool(a < b)
    return out


def correlation_rows(
    pairs: pd.DataFrame, unit: str, n_resamples: int, n_permutations: int, seed: int, level: float = 0.95,
    all_pairs: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """One row per predictor: Spearman per corridor and pooled, the law-cluster interval, the permutation p-value, the
    paired difference to (a) and the D-vs-E check (on the pairs of D and E in ``all_pairs``, whatever the unit set:
    D is an ablation law, not one of the 37 pairs of correlation_pooled)."""
    corridors = list(dict.fromkeys(pairs["corridor"]))
    y_all = pairs["macro_error"].to_numpy(np.float64)
    names = list(dict.fromkeys(pairs["law"]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(n_resamples, len(names)))  # one set of law resamples for every predictor
    source = pairs if all_pairs is None else all_pairs
    values_all = {key: {(r["law"], r["corridor"]): _value(r[key]) for r in source.to_dict("records")}
                  for key in PREDICTORS}  # fmt: skip
    rows = []
    for key, (letter, name, per) in PREDICTORS.items():
        x = np.array([_value(v) for v in pairs[key]], dtype=np.float64)
        row: dict[str, Any] = {"unit_set": unit, "letter": letter, "predictor": key, "name": name, "per": per}
        for corridor in corridors:
            mine = (pairs["corridor"] == corridor).to_numpy()
            row[f"rho_{corridor}"] = stratified_spearman(x[mine], y_all[mine], np.zeros(mine.sum()))
            row[f"n_{corridor}"] = int((np.isfinite(x[mine]) & np.isfinite(y_all[mine])).sum())
        strata, laws = pairs["corridor"].to_numpy(), pairs["law"].to_numpy()
        boot = cluster_bootstrap_spearman(x, y_all, strata, laws, np.random.default_rng(seed + 1), n_resamples, level)
        row.update(n=int((np.isfinite(x) & np.isfinite(y_all)).sum()), rho=boot["estimate"], rho_low=boot["low"],
                   rho_high=boot["high"], rho_valid=boot["n_valid"])  # fmt: skip
        if per == "law":
            row["p"] = linked_permutation_p(x, y_all, strata, laws, n_permutations, seed)
            row["p_method"] = "laws permuted as wholes"
        else:
            row["p"] = stratified_permutation_p(x, y_all, strata, n_permutations, seed)
            row["p_method"] = "within every corridor"
        ref = np.array([_value(v) for v in pairs[REFERENCE]], dtype=np.float64)
        diff = paired_difference(x, ref, y_all, strata, laws, draws, names, level)
        row.update(difference=diff["difference"], difference_low=diff["low"], difference_high=diff["high"],
                   difference_n=diff["n"], difference_valid=diff["valid"])  # fmt: skip
        check = rank_check(values_all[key], corridors)
        row["rank_e_better_than_d"] = check
        for law, tag in ((LAW_E, "E"), (LAW_D, "D")):
            for corridor in corridors:
                row[f"{tag}_{corridor}"] = values_all[key].get((law, corridor), math.nan)
        above = np.isfinite(row["difference_low"]) and row["difference_low"] > 0.0
        row["difference_above_zero"] = bool(above) if key != REFERENCE else None
        row["predicts_better"] = (bool(above and check)) if key != REFERENCE else None
        rows.append(row)
    return rows


def _value(value: Any) -> float:
    """A predictor value for the rank statistics: ``+inf`` as :data:`INF_SENTINEL`, None as NaN."""
    if value is None:
        return math.nan
    value = float(value)
    return INF_SENTINEL if value == math.inf else value


# --------------------------------------------------------------------------------------------------- inputs


def read_inputs(tables_root: Path) -> pd.DataFrame:
    """The pairs (37 of correlation_pooled plus the six of the ablation laws) with the macro error, (a), the contact
    rate and the asymmetry values of the tables."""
    pooled = pd.read_csv(tables_root / "m8" / "correlation_pooled_laws.csv")
    laws = pd.read_csv(tables_root / "m5" / "laws.csv")
    laws = laws[laws["law"] != "ground truth"]
    instability = pd.read_csv(tables_root / "m5" / "instability.csv")
    asymmetry = pd.read_csv(tables_root / "m8" / "asymmetry.csv")
    rows = [{"corridor": r["corridor"], "law": r["law"], "in_pooled": True} for r in pooled.to_dict("records")]
    for law in ABLATION_LAWS:
        for corridor in CORRIDORS:
            rows.append({"corridor": corridor, "law": law, "in_pooled": False})
    frame = pd.DataFrame(rows)
    key = ["corridor", "law"]
    frame = frame.merge(laws[key + ["macro_error", "collisions_per_1000_vkm", "runs"]], on=key, how="left")
    unstable = pd.concat([pooled[key + ["unstable_eq"]], instability[key + ["unstable_eq"]]]).drop_duplicates(key)
    frame = frame.merge(unstable, on=key, how="left")
    frame = frame.merge(instability[key + ["not_stable"]].rename(columns={"not_stable": "not_stable_table"}), on=key,
                        how="left")  # fmt: skip
    frame = frame.merge(asymmetry[key + ["asymmetry_index", "abs_error_asymmetry_index"]], on=key, how="left")
    return frame.rename(columns={"collisions_per_1000_vkm": "contacts", "asymmetry_index": "asymmetry",
                                 "abs_error_asymmetry_index": "asymmetry_error"})  # fmt: skip


def context_of(run: str | Path) -> dict[str, Any]:
    return read_json(resolve_path(run) / "metrics.json")["context"]


def law_values(law: str, laws_root: Path, platoon_cfg: Mapping[str, Any], seed: int, notes: list[str]) -> dict[str, Any]:
    """(b), (c), (d), (e) of a law (and (a) for checking): member audits and platoon tests, or the closed forms of its
    parameter sets."""
    payload = read_json(laws_root / f"{law}.json")
    kind = payload.get("kind")
    members = [m["run"] for m in payload.get("members") or []]
    out: dict[str, Any] = {"kind": kind, "members": len(members)}
    if kind == "models" or (kind == "idm" and members):
        audits, growths, collided, missing = [], [], 0, []
        for run in members:
            stability = resolve_path(run) / "stability.json"
            if stability.is_file():
                audits.append(audit_values(read_json(stability)["audit"]))
            else:
                missing.append(f"{run}/stability.json")
            platoon = resolve_path(run) / "platoon.json"
            payload_p = read_json(platoon) if platoon.is_file() else None
            if payload_p is None or payload_p.get("error") or "pulse" not in (payload_p.get("profiles") or {}):
                missing.append(f"{run}/platoon.json")
                continue
            profile = payload_p["profiles"]["pulse"]
            growths.append(transient_growth(profile))
            collided += int(math.isinf(growths[-1]))
        for name in ("not_stable", "max_gain", "max_gain_low"):
            values = [a[name] for a in audits if np.isfinite(a[name])]
            out[name] = float(np.mean(values)) if values else math.nan
        out.update(platoon_growth=law_mean(growths), platoon_members=len(growths), platoon_collided=collided,
                   source="member audits and platoon.json")  # fmt: skip
        for item in missing:
            notes.append(f"{law}: {item} missing")
        return out
    # closed forms: parameter sets with the contexts of their runs
    from cf_stability.corridor.laws import IDM_KEYS
    from cf_stability.models import load_model
    from cf_stability.models.idm import IDM

    rng = np.random.default_rng([seed, zlib.crc32(law.encode())])
    source = payload.get("source") or {}
    if kind == "idm":  # idm_core_margin (cores_of runs) and idm_margin_i80 (calibrations per fold)
        params = np.array([[float(p[k]) for k in IDM_KEYS] for p in payload["params"]])
        runs = source.get("core_runs") or source.get("support_runs") or []
        if len(runs) != len(params):
            raise ValueError(f"{law}: {len(params)} parameter sets but {len(runs)} runs of their contexts")
        contexts = [context_of(r) for r in runs]
        per_set = [closed_form_values(params[k : k + 1], contexts[k]) for k in range(len(params))]
        platoon_sets, platoon_contexts = list(range(len(params))), contexts
        models = [IDM({k: float(v) for k, v in zip(IDM_KEYS, params[i])}, dtype=torch.float64) for i in platoon_sets]
        out["source"] = f"closed form of its {len(params)} parameter sets with the contexts of {len(runs)} runs"
    else:  # idm_heterogeneous(_all): a table; residual_heterogeneous: certified cores (+ the residual of a member)
        with np.load(laws_root / payload["table"]) as data:
            columns = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
        n_sets = len(columns["T"])  # per-event arrays; v0 and b of idm_heterogeneous are scalars of the table
        params = np.stack([np.broadcast_to(np.asarray(columns[k] if k in columns else payload[k], dtype=np.float64),
                                           (n_sets,)) for k in IDM_KEYS], axis=1)  # fmt: skip
        runs = members if kind == "residual_heterogeneous" else source.get("support_runs") or []
        contexts = [context_of(r) for r in runs]
        per_set = [closed_form_values(params, context) for context in contexts]  # every set in every context
        platoon_sets = sorted(rng.choice(n_sets, size=min(N_PLATOON_SETS, n_sets), replace=False).tolist())
        if kind == "residual_heterogeneous":  # the vehicle's core and the residual of a member drawn as the law draws
            member_models = [load_model(resolve_path(m["model"])) for m in payload["members"]]
            which = rng.integers(0, len(member_models), size=len(platoon_sets))
            models, platoon_contexts = [], []
            for i, m in zip(platoon_sets, which):
                model = copy.deepcopy(member_models[m])
                model.idm.set_params({k: float(v) for k, v in zip(IDM_KEYS, params[i])})
                models.append(model)
                platoon_contexts.append(contexts[m])
        else:
            models = [IDM({k: float(v) for k, v in zip(IDM_KEYS, params[i])}, dtype=torch.float64) for i in platoon_sets]
            platoon_contexts = [contexts[k % len(contexts)] for k in range(len(platoon_sets))]
        out["source"] = (f"closed form of its {n_sets} parameter sets in the contexts of {len(runs)} runs; platoon: "
                         f"{len(platoon_sets)} sets drawn uniformly")  # fmt: skip
    for name in ("unstable_eq", "not_stable", "max_gain", "max_gain_low"):
        values = np.concatenate([np.asarray(v[name], dtype=np.float64).reshape(-1) for v in per_set])
        values = values[np.isfinite(values)]
        out[f"{name}_closed" if name == "unstable_eq" else name] = float(values.mean()) if len(values) else math.nan
    growth = pulse_growth_of_models(models, platoon_contexts, platoon_cfg)
    out.update(platoon_growth=law_mean([g["growth"] for g in growth]), platoon_members=len(growth),
               platoon_collided=int(sum(g["collided"] for g in growth)))  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------------- tables


def pairs_table(pairs: pd.DataFrame) -> Table:
    columns = [
        Column("corridor", "corridor", "text"), Column("law", "law", "text"),
        Column("in_pooled", "in correlation_pooled", "flag"), Column("macro_error", "macro error", digits=3),
        *(Column(key, f"({letter})", digits=3) for key, (letter, _, _) in PREDICTORS.items()),
        Column("platoon_collided", "collided pulse platoons", "int"), Column("platoon_members", "platoons", "int"),
        Column("n_states", "states (f)", "int"), Column("states_available", "states available", "int"),
    ]  # fmt: skip
    notes = [
        "The pairs of D127 and their predictor values (scripts/analysis/predictive_validity.py; the letters of the "
        "predictors are those of docs/m11_contract.md, D127): " + "; ".join(
            f"({letter}) {name}" for letter, name, _ in PREDICTORS.values()) + ".",
        "(e): inf = a pulse platoon of a member (or parameter set) collided (the statistics rank it above every finite "
        "value); collided pulse platoons and platoons: their numbers. (f): states drawn from the follower states of "
        "the seed-0 runs of the pair's three scenarios (states available).",
    ]
    frame = pairs.copy()
    return Table("predictive_validity_pairs", "Predictive validity (D127): the predictor values of the pairs", notes,
                 frame, columns)  # fmt: skip


def results_table(rows: pd.DataFrame, n_resamples: int, n_permutations: int, seed: int, notes_extra: Sequence[str]
                  ) -> Table:  # fmt: skip
    frame = rows.copy()
    for column in ("E_I-80", "E_US-101", "D_I-80", "D_US-101"):
        frame[column] = frame[column].map(lambda x: math.inf if x == INF_SENTINEL else x)
    frame["d_vs_e"] = frame["rank_e_better_than_d"].map({True: "E ranked better", False: "not", None: "n/a"})
    frame["rule"] = [("reference" if r["predictor"] == REFERENCE else
                      "predicts better" if r["predicts_better"] else "not better") for r in frame.to_dict("records")]  # fmt: skip
    columns = [
        Column("unit_set", "pairs", "text"), Column("letter", "", "text"), Column("name", "predictor", "text"),
        Column("rho_I-80", "rho I-80", digits=3), Column("rho_US-101", "rho US-101", digits=3), Column("n", "n", "int"),
        Column("rho", "rho pooled", digits=3, ci=True), Column("p", "p (permutation)", "p"),
        Column("difference", "rho - rho(a)", digits=3, ci=True), Column("E_I-80", "E I-80", digits=4),
        Column("D_I-80", "D I-80", digits=4), Column("E_US-101", "E US-101", digits=4),
        Column("D_US-101", "D US-101", digits=4), Column("d_vs_e", "D vs E", "text"), Column("rule", "rule", "text"),
    ]  # fmt: skip
    notes = [
        "H-N1 (D127): Spearman correlation of the macro error of a (law, corridor) pair (mean over its runs, "
        "runs/_tables/m5/laws.csv) with single-vehicle predictors of the law; per corridor over its laws, pooled with "
        "the corridor as a stratum (ranks within the corridor scaled to (0, 1), Pearson of the pooled ranks). pairs: "
        "the 37 pairs of correlation_pooled (19 laws on I-80, 18 on US-101), and the separate check with the six pairs "
        "of the ablation laws (idm_core_margin, idm_margin_i80, residual_idm_margin_free_r0.3) added (43).",
        f"rho pooled with the 95 % percentile interval of a cluster bootstrap over the laws ({n_resamples} resamples; "
        "the rows of a law on both corridors move together; cf_stability.eval.stats.cluster_bootstrap_spearman); p: "
        f"two-sided permutation test ({n_permutations} permutations): the instability of whole laws permuted "
        "(linked_permutation_p) for the predictors (a)-(e), one value per law; within every corridor "
        "(stratified_permutation_p) for (f)-(h), one value per pair.",
        "rho - rho(a): the difference of the pooled Spearman correlations of P and of the pre-specified predictor (a) "
        "over the pairs where both exist, with the percentile interval of the same law resamples for both (paired "
        f"cluster bootstrap, numpy default_rng({seed}): {n_resamples} draws of the laws, the same for every "
        "predictor).",
        "E: residual_idm_certified; D: residual_idm_margin_free_r0.3; their values of P on each corridor (the "
        "per-law predictors have one value on both). D vs E: P ranks E better than D when P(E) < P(D) strictly on "
        "both corridors (every predictor is oriented so that a larger value means a larger expected error, except "
        "(h), the raw index, shown as the contract names it, and (h') its distance to the drivers).",
        "rule (D127): P predicts better when the interval of rho - rho(a) lies above 0 and P ranks E better than D on "
        "both corridors; H-N1 is confirmed when the frozen-gain statistic (f) predicts better, partially confirmed "
        "when another predictor does, refuted when none does (the verdict is the main session's).",
        "(e): the transient growth of the braking-pulse platoon, max over the 50 followers of the speed standard "
        "deviation over that of follower 1; a collided platoon counts as +inf (inf in the table; ranked above every "
        "finite value); mean over the members (inf once a member collided).",
        *notes_extra,
    ]
    return Table("predictive_validity", "Predictive validity of single-vehicle metrics for the macro error (D127, H-N1)",
                 notes, frame, columns)  # fmt: skip


def write(table: Table, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.frame.to_csv(out / f"{table.name}.csv", index=False)
    (out / f"{table.name}.md").write_text(markdown(table), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------- main


def statistics_only(args: argparse.Namespace) -> int:
    """The tables of the statistics from the predictor values of an earlier run (``predictive_validity_pairs.csv``)."""
    pairs = pd.read_csv(resolve_path(args.from_pairs))
    pairs["in_pooled"] = pairs["in_pooled"].astype(bool)
    rows = []
    for unit, part in (("correlation_pooled (37)", pairs[pairs["in_pooled"]]), ("with the ablation laws (43)", pairs)):
        rows += correlation_rows(part.reset_index(drop=True), unit, args.resamples, args.permutations, args.seed,
                                 all_pairs=pairs)  # fmt: skip
    out = resolve_path(args.out)
    notes = [f"statistics recomputed from {args.from_pairs} (the predictor values of the run that wrote it)."]
    notes_file = out / "predictive_validity_notes.txt"
    if notes_file.is_file():
        notes = [line for line in notes_file.read_text(encoding="utf-8").splitlines() if line] + notes
    write(results_table(pd.DataFrame(rows), args.resamples, args.permutations, args.seed, notes), out)
    _print(rows)
    return 0


def _print(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        print(f"{r['unit_set']:<28} ({r['letter']:<2}) rho {r['rho']:+.3f} [{r['rho_low']:+.3f}, {r['rho_high']:+.3f}] "
              f"p {r['p']:.4f}  diff {r['difference']:+.3f} [{r['difference_low']:+.3f}, {r['difference_high']:+.3f}]  "
              f"E<D {r['rank_e_better_than_d']}  better {r['predicts_better']}")  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", default="runs/_tables")
    parser.add_argument("--corridor-root", default="runs/corridor")
    parser.add_argument("--platoon-config", default="configs/stability/platoon.yaml")
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--states", type=int, default=20000)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu", help="device of the partial derivatives of the networks in (f)")
    parser.add_argument("--laws", nargs="*", default=None, help="only these laws (a check; the statistics need all)")
    parser.add_argument("--from-pairs", default=None, help="the statistics only, from a predictive_validity_pairs.csv "
                                                           "of an earlier run (its notes are not repeated)")
    args = parser.parse_args(argv)
    if args.from_pairs:
        return statistics_only(args)
    import yaml

    t_start = time.perf_counter()
    tables, corridor_root = resolve_path(args.tables), resolve_path(args.corridor_root)
    laws_root = corridor_root / "laws"
    platoon_cfg = yaml.safe_load(resolve_path(args.platoon_config).read_text(encoding="utf-8"))
    pairs = read_inputs(tables)
    if args.laws:
        pairs = pairs[pairs["law"].isin(args.laws)].reset_index(drop=True)
    notes: list[str] = []
    per_law: dict[str, dict[str, Any]] = {}
    for law in dict.fromkeys(pairs["law"]):
        t0 = time.perf_counter()
        per_law[law] = law_values(law, laws_root, platoon_cfg, args.seed, notes)
        v = per_law[law]
        print(f"LAW {law:<32} (b) {v.get('not_stable', math.nan):.3f} (c) {v.get('max_gain', math.nan):.3g} "
              f"(d) {v.get('max_gain_low', math.nan):.3g} (e) {v.get('platoon_growth', math.nan):.3g} "
              f"[{v.get('platoon_collided', 0)}/{v.get('platoon_members', 0)} collided]  {time.perf_counter() - t0:.0f} s",
              flush=True)  # fmt: skip
    for key in ("not_stable", "max_gain", "max_gain_low", "platoon_growth", "platoon_members", "platoon_collided",
                "unstable_eq_closed", "source", "kind"):  # fmt: skip
        pairs[key] = [per_law[law].get(key, math.nan) for law in pairs["law"]]
    # (b) of the laws with member audits equals the column of the table instability (a check)
    check = pairs[pairs["not_stable_table"].notna()]
    worst = float((check["not_stable"] - check["not_stable_table"]).abs().max()) if len(check) else math.nan
    notes.append(f"check: (b) of the member audits equals not_stable of runs/_tables/m5/instability.csv for "
                 f"{len(check)} pairs (largest difference {worst:.1e}).")  # fmt: skip
    closed = pairs[pairs["unstable_eq_closed"].notna()]
    worst_a = float((closed["unstable_eq_closed"] - closed["unstable_eq"]).abs().max()) if len(closed) else math.nan
    notes.append(f"check: (a) recomputed in closed form for the laws without member runs equals the tables for "
                 f"{len(closed)} pairs (largest difference {worst_a:.1e}).")  # fmt: skip
    frozen = []
    for row in pairs.to_dict("records"):
        t0 = time.perf_counter()
        result = pair_frozen_gain(row["law"], row["corridor"], corridor_root, args.states, args.seed, args.device)
        notes.extend(result.pop("problems", []))
        frozen.append(result)
        print(f"FROZEN {row['corridor']:<7} {row['law']:<32} share>1.02 {result['frozen_share']:.3f}  mean log "
              f"{result['frozen_log']:+.4f}  states {result['n_states']}/{result['states_available']}  "
              f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    for key in ("frozen_share", "frozen_log", "n_states", "states_available", "nonfinite", "gain_median", "gain_p90"):
        pairs[key] = [f.get(key, math.nan) for f in frozen]
    out = resolve_path(args.out)
    write(pairs_table(pairs), out)
    rows = []
    for unit, part in (("correlation_pooled (37)", pairs[pairs["in_pooled"]]), ("with the ablation laws (43)", pairs)):
        rows += correlation_rows(part.reset_index(drop=True), unit, args.resamples, args.permutations, args.seed,
                                 all_pairs=pairs)  # fmt: skip
    frame = pd.DataFrame(rows)
    write(results_table(frame, args.resamples, args.permutations, args.seed, notes), out)
    (out / "predictive_validity_notes.txt").write_text("".join(f"{n}\n" for n in notes), encoding="utf-8")
    _print(rows)
    print(f"predictive_validity: {len(pairs)} pairs, {len(per_law)} laws, {time.perf_counter() - t_start:.0f} s -> "
          f"{out / 'predictive_validity.md'}, {out / 'predictive_validity_pairs.md'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
