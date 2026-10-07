"""Constrained trajectory reconstruction on IDM platoons with synthetic NGSIM-like noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cf_stability.data.reconstruction import ReconstructionConfig, derivatives, reconstruct_all, reconstruct_track

DT = 0.1
CFG = ReconstructionConfig()
IDM = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a": 1.5, "b": 2.0}


def leader_speed(t: np.ndarray, v_hi: float = 12.0) -> np.ndarray:
    """Stop-and-go cycle of 31 s with cosine ramps: cruise 8 s, brake 8 s, stand 3 s, accelerate 12 s."""
    tc = t % 31.0
    brake = 0.5 * v_hi * (1 + np.cos(np.pi * (tc - 8) / 8))
    accel = 0.5 * v_hi * (1 - np.cos(np.pi * (tc - 19) / 12))
    return np.select([tc < 8, tc < 16, tc < 19], [v_hi, brake, 0.0], accel)


def simulate_platoon(lengths: np.ndarray, duration: float = 60.0, phase: float = 0.0, x0: float = 300.0):
    """Leader with the stop-and-go profile followed by IDM vehicles (semi-implicit Euler, 0.1 s).

    Returns ``t [n]`` and front-bumper positions ``x [n_vehicles, n]``; ``phase`` in [0, 8) s.
    """
    lengths = np.asarray(lengths, dtype=float)
    n = int(round(duration / DT)) + 1
    t = np.arange(n) * DT
    v0, T, s0, a, b = (IDM[k] for k in ("v0", "T", "s0", "a", "b"))
    v_eq = 12.0
    s_eq = (s0 + v_eq * T) / np.sqrt(1 - (v_eq / v0) ** 4)
    x, v = np.empty((len(lengths), n)), np.empty((len(lengths), n))
    x[:, 0] = x0 - np.arange(len(lengths)) * s_eq - np.r_[0.0, np.cumsum(lengths[:-1])]
    v[:, 0] = v_eq
    v[0] = leader_speed(t + phase, v_eq)
    for k in range(n - 1):
        x[0, k + 1] = x[0, k] + v[0, k + 1] * DT
        s = x[:-1, k] - lengths[:-1] - x[1:, k]
        vf, dv = v[1:, k], v[1:, k] - v[:-1, k]
        s_star = s0 + np.maximum(0.0, vf * T + vf * dv / (2 * np.sqrt(a * b)))
        acc = np.clip(a * (1 - (vf / v0) ** 4 - (s_star / s) ** 2), -8.0, 4.0)
        v[1:, k + 1] = np.maximum(vf + acc * DT, 0.0)
        x[1:, k + 1] = x[1:, k] + v[1:, k + 1] * DT
    return t, x


def add_noise(x: np.ndarray, rng: np.random.Generator, sigma: float = 0.3, n_jumps: int = 3) -> np.ndarray:
    """White noise plus a few isolated jumps of 2-3 m per vehicle (away from the track ends)."""
    noisy = x + rng.normal(0.0, sigma, x.shape)
    for row in noisy:
        idx = rng.choice(np.arange(20, x.shape[-1] - 20), n_jumps, replace=False)
        row[idx] += rng.choice([-1.0, 1.0], n_jumps) * rng.uniform(2.0, 3.0, n_jumps)
    return noisy


def assert_feasible(x: np.ndarray, cfg: ReconstructionConfig = CFG, rtol: float = 1e-3) -> None:
    assert np.all(np.diff(x) >= 0.0)
    acc, jerk = np.diff(x, 2) / DT**2, np.diff(x, 3) / DT**3
    assert acc.min() >= cfg.a_min * (1 + rtol) and acc.max() <= cfg.a_max * (1 + rtol)
    assert np.abs(jerk).max() <= cfg.jerk_max * (1 + rtol)


@dataclass
class SimpleTrack:
    t: np.ndarray
    x: np.ndarray
    length: float
    preceding: np.ndarray


@pytest.fixture(scope="module")
def platoon():
    lengths = np.array([4.5, 4.5, 12.0, 4.5, 4.5, 4.5])
    t, x = simulate_platoon(lengths)
    return t, x, lengths, add_noise(x, np.random.default_rng(0))


def test_noise_free_feasible_trajectory_is_unchanged(platoon):
    _, x, _, _ = platoon
    for truth in x:
        assert_feasible(truth)
        rec = reconstruct_track(truth, DT, CFG)
        assert rec.status == "solved"
        assert np.abs(rec.x - truth).max() < 0.05


def test_noisy_tracks_are_improved_and_feasible(platoon):
    _, x, _, noisy = platoon
    acc_err = []
    for truth, obs in zip(x, noisy):
        rec = reconstruct_track(obs, DT, CFG)
        assert rec.status == "solved"
        assert_feasible(rec.x)
        assert rec.v.min() >= 0.0
        rmse_obs, rmse_rec = np.sqrt(np.mean((obs - truth) ** 2)), np.sqrt(np.mean((rec.x - truth) ** 2))
        assert rmse_rec < 0.5 * rmse_obs
        acc_err.append(rec.a - derivatives(truth, DT)[1])
        assert rec.n_downweighted >= 3  # the jumps
        v, a, _ = derivatives(rec.x, DT)
        np.testing.assert_array_equal(v, rec.v)
        np.testing.assert_array_equal(a, rec.a)
    assert np.sqrt(np.mean(np.concatenate(acc_err) ** 2)) < 0.5


def test_platoon_consistency_with_overstated_lengths(platoon):
    """Overstated leader lengths make the raw net gaps negative at standstill; the leader
    constraint must restore gap >= s_min on every frame (chain of three followers)."""
    t, _, lengths, noisy = platoon
    stated = lengths + np.array([2.5, 2.5, 0.0, 0.0, 0.0, 0.0])
    ids = [f"v{i}" for i in range(len(lengths))]
    tracks = {
        tid: SimpleTrack(t, noisy[i], float(stated[i]), np.full(len(t), ids[i - 1] if i else None, dtype=object))
        for i, tid in enumerate(ids)
    }
    tracks["short"] = SimpleTrack(t[:3], noisy[0, :3] + 500.0, 4.5, np.full(3, None, dtype=object))
    result = reconstruct_all(tracks, CFG)
    recs, report = result["tracks"], result["report"]

    for i in range(1, len(ids)):
        gap = recs[ids[i - 1]].x - stated[i - 1] - recs[ids[i]].x
        assert gap.min() >= CFG.s_min - 0.05
    for tid in ids:
        assert_feasible(recs[tid].x)
    np.testing.assert_array_equal(recs["short"].x, tracks["short"].x)
    assert recs["short"].status == "short" and report["n_short"] == 1
    assert report["n_tracks"] == len(tracks)
    assert report["resolved_per_pass"][0] >= 2 and report["resolved_per_pass"][-1] == 0
    assert report["gaps_before"]["share_gap_nonpositive"] > 0.0
    assert report["gaps_after"]["share_gap_nonpositive"] == 0.0
    assert report["gaps_after"]["share_gap_below_s_min"] == 0.0
    assert report["kinematics_after"]["abs_jerk"]["max"] <= CFG.jerk_max * (1 + 1e-3)
    assert report["kinematics_before"]["abs_acc"]["max"] > CFG.a_max
    assert report["n_solver_failures"] == 0


def test_conflicts_beyond_max_push_stay_unconstrained(platoon):
    t, _, lengths, noisy = platoon
    stated = [lengths[0] + 2.5, lengths[1]]  # needs a push-back of about 1 m at standstill
    tracks = {
        "v0": SimpleTrack(t, noisy[0], stated[0], np.full(len(t), None, dtype=object)),
        "v1": SimpleTrack(t, noisy[1], stated[1], np.full(len(t), "v0", dtype=object)),
    }
    report = reconstruct_all(tracks, ReconstructionConfig(max_push=0.5))["report"]
    assert report["share_leader_frames_beyond_max_push"] > 0.0
    assert report["gaps_after"]["share_gap_nonpositive"] > 0.0


def test_parallel_pass_matches_serial(platoon):
    t, _, lengths, noisy = platoon
    tracks = {f"v{i}": SimpleTrack(t, noisy[i], 4.5, np.full(len(t), None, dtype=object)) for i in range(3)}
    serial = reconstruct_all(tracks, CFG)["tracks"]
    parallel = reconstruct_all(tracks, CFG, n_jobs=2)["tracks"]
    for tid in tracks:
        np.testing.assert_array_equal(serial[tid].x, parallel[tid].x)
