"""Physics-informed reconstruction of longitudinal trajectories (constrained smoothing).

One quadratic programme per track, solved with OSQP:

    min_x  sum_k w_k (x_k - x_obs_k)^2 + lam * sum_k (D3 x / dt^3)_k^2 + rho * sum_j e_j^2
    s.t.   D1 x / dt >= 0,   a_min <= D2 x / dt^2 <= a_max,   |D3 x / dt^3| <= jerk_max,
           x_k - e_j <= x_lead_rear_k - s_min     (frames with a leader, soft through e_j),

where ``D1, D2, D3`` are forward differences, ``lam = (2 sin(pi f_c dt) / dt)^-6`` puts the
gain 1/2 of the unconstrained smoother at ``f_c`` and ``w`` are Huber weights obtained by
iteratively reweighted least squares (without leader constraint; constrained re-solves reuse them).
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from typing import Any, Mapping, Protocol

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import solveh_banded

from cf_stability.data.schema import DT

_DIFF_COEFFS = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0), 3: (-1.0, 3.0, -3.0, 1.0)}
_QUANTILES = {"q50": 0.5, "q90": 0.9, "q99": 0.99, "max": 1.0}


@dataclass(frozen=True)
class ReconstructionConfig:
    f_c: float = 0.5  # Hz, cut-off frequency of the smoother
    huber: float = 1.0  # m, Huber threshold of the robust weights
    irls_iter: int = 2  # re-weighting iterations after the first solve
    a_min: float = -8.0
    a_max: float = 4.0
    jerk_max: float = 15.0
    # m, minimum net gap to the leader (soft); above extraction.min_spacing = 0.5 m, so that the
    # frames on the constraint are not cut out of the events
    s_min: float = 0.75
    # 1e3 left violations up to 6 cm for the 5-7 m corrections of real I-80 tracks; 1e4: < 1 cm
    slack_weight: float = 1e4
    gap_tol: float = 0.01  # m, a gap below s_min - gap_tol triggers a constrained re-solve
    # m; frames whose leader constraint would push the follower back by more than this (relative to
    # its unconstrained reconstruction) are left unconstrained: on NGSIM I-80 such overlaps (up to
    # 19 m, from vehicle lengths or positions) made the corrections grow along platoons pass by pass
    max_push: float = 5.0
    max_passes: int = 5
    min_samples: int = 5  # shorter tracks are returned unchanged
    eps: float = 1e-5  # OSQP absolute and relative tolerance
    max_iter: int = 20_000

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ReconstructionConfig":
        unknown = set(mapping) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"unknown reconstruction options: {sorted(unknown)}")
        return cls(**dict(mapping))

    def lam(self, dt: float) -> float:
        """Weight of the squared jerk (in (m/s^3)^2) in the objective."""
        return (2.0 * math.sin(math.pi * self.f_c * dt) / dt) ** -6


@dataclass
class ReconstructedTrack:
    x: np.ndarray  # m
    v: np.ndarray  # m/s
    a: np.ndarray  # m/s^2
    jerk: np.ndarray  # m/s^3
    weights: np.ndarray  # Huber weights of the last solve
    status: str  # OSQP status of the last solve, or "short"
    leader_constrained: bool
    rms_correction: float  # m, RMS of x - x_obs
    max_correction: float  # m
    n_downweighted: int  # samples with Huber weight < 1 in the last solve
    max_violation: float  # m, largest violation of the soft leader constraint
    n_iter: int  # ADMM iterations summed over the solves


class TrackLike(Protocol):
    t: np.ndarray  # s, contiguous 10 Hz grid
    x: np.ndarray  # front bumper, m
    length: float  # m
    preceding: np.ndarray  # object array: leader track id or None per sample


def derivatives(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Speed, acceleration and jerk of ``x`` at the sample times (centred differences).

    ``v[k] = (x[k+1] - x[k-1]) / (2 dt)`` (the mean of the two adjacent interval speeds),
    ``a[k] = (x[k+1] - 2 x[k] + x[k-1]) / dt^2``, ``jerk[k] = (a[k+1] - a[k-1]) / (2 dt)``;
    at the ends ``v`` is the speed of the first/last interval and ``a``, ``jerk`` repeat their
    nearest value. The series are means of the forward differences constrained by the quadratic
    programme, so they obey the same bounds. State and acceleration refer to the same instant,
    as in the other datasets (docs/decisions.md, D35).
    """
    n = len(x)
    if n < 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    step = np.diff(x) / dt  # speed of the interval [k, k + 1]
    v = np.concatenate([step[:1], 0.5 * (step[1:] + step[:-1]), step[-1:]])
    if n < 3:
        return v, np.zeros(n), np.zeros(n)
    second = np.diff(step) / dt  # centred at the samples 1 .. n - 2
    a = np.concatenate([second[:1], second, second[-1:]])
    jerk = np.zeros(n)
    jerk[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    jerk[0], jerk[-1] = jerk[1], jerk[-2]
    return v, a, jerk


def _diff_matrix(n: int, order: int) -> sp.csc_matrix:
    return sp.diags(_DIFF_COEFFS[order], range(order + 1), shape=(n - order, n), format="csc")


def _huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    return delta / np.maximum(np.abs(r), delta)


def reconstruct_track(
    x_obs: np.ndarray,
    dt: float,
    cfg: ReconstructionConfig,
    upper: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> ReconstructedTrack:
    """Reconstruct one track; ``upper`` (leader rear bumper minus ``s_min``, NaN/inf where there
    is no leader) adds the soft platoon-consistency constraint ``x <= upper``; given ``weights``
    replace the IRLS iterations by a single solve."""
    x_obs = np.asarray(x_obs, dtype=np.float64)
    n = len(x_obs)
    if n < cfg.min_samples:
        return ReconstructedTrack(x_obs.copy(), *derivatives(x_obs, dt), np.ones(n), "short", False, 0.0, 0.0, 0, 0.0, 0)

    mu = cfg.lam(dt) / dt**6  # weight of the squared third difference
    d1, d2, d3 = (_diff_matrix(n, k) for k in (1, 2, 3))
    m3 = (d3.T @ d3).tocsc()
    band = np.zeros((4, n))
    for k in range(4):
        band[3 - k, k:] = mu * m3.diagonal(k)
    band[3] += 1.0
    # Unconstrained smoother with unit weights. The QP is posed in the deviation z = x - x_ref,
    # which keeps the variables and the constraint values small (OSQP tolerances are relative).
    x_ref = solveh_banded(band, x_obs)

    idx = np.flatnonzero(np.isfinite(upper)) if upper is not None else np.empty(0, dtype=np.int64)
    m = len(idx)
    j1, j2, j3 = d1 @ x_ref / dt, d2 @ x_ref / dt**2, d3 @ x_ref / dt**3
    A = sp.vstack([d1 / dt, d2 / dt**2, d3 / dt**3], format="csc")
    lo = [-j1, cfg.a_min - j2, -cfg.jerk_max - j3]
    hi = [np.full(n - 1, np.inf), cfg.a_max - j2, cfg.jerk_max - j3]
    hess = mu * m3
    if m:
        sel = sp.csc_matrix((np.ones(m), (np.arange(m), idx)), shape=(m, n))
        A = sp.bmat([[A, None], [sel, -sp.eye(m)]], format="csc")
        lo.append(np.full(m, -np.inf))
        hi.append(upper[idx] - x_ref[idx])
        hess = sp.block_diag([hess, cfg.slack_weight * sp.eye(m)])
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    P = sp.triu(2.0 * (hess + sp.diags(np.r_[w, np.zeros(m)])), format="csc")
    P.sort_indices()
    diag_pos = P.indptr[1 : n + 1] - 1  # the diagonal entry closes every column of triu(P)
    base_diag = P.data[diag_pos] - 2.0 * w

    def gradient(w: np.ndarray) -> np.ndarray:
        return np.r_[2.0 * (w * (x_ref - x_obs) + mu * (m3 @ x_ref)), np.zeros(m)]

    solver = osqp.OSQP()
    # no polishing: at eps = 1e-5 the ADMM solution is accurate enough, and OSQP's polishing step
    # prints to stdout from C even with verbose=False
    solver.setup(
        P, gradient(w), A, np.concatenate(lo), np.concatenate(hi),
        verbose=False, eps_abs=cfg.eps, eps_rel=cfg.eps, max_iter=cfg.max_iter, polishing=False,
    )
    x, status, n_iter = x_ref, "unsolved", 0
    for it in range(1 if weights is not None else cfg.irls_iter + 1):
        if it:
            w = _huber_weights(x_obs - x, cfg.huber)
            P.data[diag_pos] = base_diag + 2.0 * w
            solver.update(Px=P.data, q=gradient(w))
        res = solver.solve(raise_error=False)
        status, n_iter = res.info.status, n_iter + int(res.info.iter)
        if res.x is None or not np.all(np.isfinite(res.x)):
            break
        x = x_ref + res.x[:n]
    # removes solver round-off so that v >= 0 holds exactly (downstream checks use v >= 0)
    x = np.maximum.accumulate(x)
    corr = x - x_obs
    violation = float(np.max(x[idx] - upper[idx], initial=0.0)) if m else 0.0
    return ReconstructedTrack(
        x, *derivatives(x, dt), w, status, bool(m), float(np.sqrt(np.mean(corr**2))),
        float(np.max(np.abs(corr))), int(np.sum(w < 1.0)), violation, n_iter,
    )


def _reconstruct_free(args: tuple[np.ndarray, float, ReconstructionConfig]) -> ReconstructedTrack:
    return reconstruct_track(*args)


def leader_links(track: TrackLike, tracks: Mapping[str, TrackLike], dt: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(leader id, follower sample indices, leader sample indices)] aligned in time."""
    codes, leaders = pd.factorize(np.asarray(track.preceding, dtype=object))
    links = []
    for c, lid in enumerate(leaders):
        if lid not in tracks:
            continue
        k = np.flatnonzero(codes == c)
        lead = tracks[lid]
        j = np.rint((np.asarray(track.t)[k] - lead.t[0]) / dt).astype(np.int64)
        ok = (j >= 0) & (j < len(lead.t))
        links.append((lid, k[ok], j[ok]))
    return links


def _lead_rear(links: list, n: int, xs: Mapping[str, np.ndarray], tracks: Mapping[str, TrackLike]) -> np.ndarray:
    out = np.full(n, np.nan)
    for lid, k, j in links:
        out[k] = xs[lid][j] - tracks[lid].length
    return out


def _upper(links: list, x_free: np.ndarray, xs: Mapping[str, np.ndarray], tracks: Mapping[str, TrackLike], cfg: ReconstructionConfig) -> np.ndarray:
    """Bound ``x <= leader rear - s_min``; NaN without leader and where it needs a push > max_push."""
    upper = _lead_rear(links, len(x_free), xs, tracks) - cfg.s_min
    upper[x_free - upper > cfg.max_push] = np.nan
    return upper


def reconstruct_all(tracks: Mapping[str, TrackLike], cfg: ReconstructionConfig, n_jobs: int = 1) -> dict:
    """Reconstruct all tracks of a site; returns ``{"tracks": {id: ReconstructedTrack}, "report": {...}}``.

    Pass 1 solves every track without leader constraint. Then, in passes over the tracks ordered
    by entry time, every track whose net gap to its reconstructed leader falls below
    ``s_min - gap_tol`` is re-solved with the leader constraint (with its pass-1 Huber weights, on
    the frames where the constraint needs a push-back of at most ``max_push``, and unless it was
    already solved against the current trajectories of its leaders), until no track is re-solved
    or ``max_passes`` is reached.
    """
    dt = DT
    ids = list(tracks)
    raw = {tid: np.asarray(tracks[tid].x, dtype=np.float64) for tid in ids}
    if n_jobs > 1:
        chunk = max(1, len(ids) // (8 * n_jobs))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            recs = dict(zip(ids, pool.map(_reconstruct_free, [(raw[t], dt, cfg) for t in ids], chunksize=chunk)))
    else:
        recs = {tid: reconstruct_track(raw[tid], dt, cfg) for tid in ids}

    links = {tid: leader_links(tracks[tid], tracks, dt) for tid in ids}
    free = dict(recs)
    x_rec = {tid: rec.x for tid, rec in recs.items()}
    order = sorted((t for t in ids if links[t] and len(raw[t]) >= cfg.min_samples), key=lambda t: (float(tracks[t].t[0]), t))
    version = dict.fromkeys(ids, 0)  # number of re-solves of each track
    solved_against: dict[str, dict[str, int]] = {}  # leader versions used by the last constrained solve
    resolved = []
    for _ in range(cfg.max_passes):
        count = 0
        for tid in order:
            leaders = {lid: version[lid] for lid, _, _ in links[tid]}
            if solved_against.get(tid) == leaders:
                continue  # a re-solve would reproduce the residual violation of the soft constraint
            upper = _upper(links[tid], free[tid].x, x_rec, tracks, cfg)
            if np.any(x_rec[tid] - upper > cfg.gap_tol):
                recs[tid] = reconstruct_track(raw[tid], dt, cfg, upper=upper, weights=free[tid].weights)
                x_rec[tid] = recs[tid].x
                version[tid] += 1
                solved_against[tid] = leaders
                count += 1
        resolved.append(count)
        if not count:
            break
    return {"tracks": recs, "report": _report(tracks, raw, free, recs, links, resolved, cfg, dt)}


def _quantiles(values: list[np.ndarray]) -> dict[str, float]:
    v = np.abs(np.concatenate(values)) if values else np.empty(0)
    return {k: float(np.quantile(v, q)) if len(v) else float("nan") for k, q in _QUANTILES.items()}


def _report(tracks, raw, free, recs, links, resolved, cfg: ReconstructionConfig, dt: float) -> dict:
    x_rec = {tid: rec.x for tid, rec in recs.items()}
    n_lead, n_beyond = 0, 0
    for tid, lk in links.items():
        if lk:
            lead = _lead_rear(lk, len(x_rec[tid]), x_rec, tracks)
            n_lead += int(np.sum(np.isfinite(lead)))
            n_beyond += int(np.sum(free[tid].x - (lead - cfg.s_min) > cfg.max_push))

    def gap_shares(xs: Mapping[str, np.ndarray]) -> dict[str, float]:
        gaps = [(_lead_rear(lk, len(xs[t]), xs, tracks) - xs[t]) for t, lk in links.items() if lk]
        g = np.concatenate(gaps) if gaps else np.empty(0)
        g = g[np.isfinite(g)]
        return {
            "n_frames_with_leader": int(len(g)),
            # same criterion as the re-solve trigger: the soft constraint leaves gaps a few mm below s_min
            "share_gap_below_s_min": float(np.mean(g < cfg.s_min - cfg.gap_tol)) if len(g) else 0.0,
            "share_gap_nonpositive": float(np.mean(g <= 0.0)) if len(g) else 0.0,
        }

    def kinematics(xs: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
        return {
            "abs_acc": _quantiles([np.diff(x, 2) / dt**2 for x in xs.values()]),
            "abs_jerk": _quantiles([np.diff(x, 3) / dt**3 for x in xs.values()]),
        }

    status = Counter(rec.status for rec in recs.values())
    corr = np.concatenate([x_rec[t] - raw[t] for t in raw]) if raw else np.empty(0)
    return {
        "n_tracks": len(recs),
        "n_short": status.get("short", 0),
        "resolved_per_pass": resolved,
        "n_leader_constrained": sum(rec.leader_constrained for rec in recs.values()),
        "share_leader_frames_beyond_max_push": n_beyond / n_lead if n_lead else 0.0,
        "max_leader_violation": max((rec.max_violation for rec in recs.values()), default=0.0),
        "gaps_before": gap_shares(raw),
        "gaps_after": gap_shares(x_rec),
        "kinematics_before": kinematics(raw),
        "kinematics_after": kinematics(x_rec),
        "rms_correction": float(np.sqrt(np.mean(corr**2))) if len(corr) else 0.0,
        "status_counts": dict(status),
        "n_solver_failures": sum(n for s, n in status.items() if s not in ("solved", "short")),
    }
