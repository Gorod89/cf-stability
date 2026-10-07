"""Acceleration asymmetry and oscillation spectrum of the corridor (cf_stability/corridor/asymmetry.py,
scripts/corridor_asymmetry.py; docs/m8_contract.md, section 9; D121), on hand-made trajectories."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cf_stability.corridor.asymmetry import (
    FILE,
    AsymmetryConfig,
    acceleration_samples,
    asymmetry_metrics,
    asymmetry_of,
    band_summary,
    detector_speed_series,
    update_asymmetry,
    welch_psd,
)
from cf_stability.corridor.macro import Geometry, MacroConfig, prepare_trajectories
from cf_stability.utils import REPO_ROOT, read_json, write_json

GEOMETRY = Geometry(20.0, 500.0, (0.0, 600.0))
F0 = 1.0 / 64.0  # Hz: a line of the Welch grid of 128 s segments (k / 128 Hz, k = 2)


def stream(speed, n_vehicles=700, duration=700, lane_of=lambda i: 1 + i % 6):
    """Vehicles entering at x = 20 m one per second, every vehicle driving the common speed profile ``speed(t)``
    (m/s, piecewise linear between whole seconds); rows at whole seconds with 20 <= x <= 500."""
    t = np.arange(duration + 1, dtype=np.float64)
    v = np.array([speed(s) for s in t])
    x_of_t = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))))  # exact for piecewise linear speeds
    rows = {"t": [], "vehicle": [], "x": [], "lane": [], "v": []}
    for i in range(n_vehicles):
        mine = t >= i
        x = 20.0 + x_of_t[mine] - x_of_t[i]
        keep = x <= 500.0
        n = int(keep.sum())
        rows["t"].append(t[mine][keep])
        rows["vehicle"].append(np.full(n, i))
        rows["x"].append(x[keep])
        rows["lane"].append(np.full(n, lane_of(i)))
        rows["v"].append(v[mine][keep])
    out = {key: np.concatenate(value) for key, value in rows.items()}
    return {"t": out["t"].astype(np.float32), "vehicle": out["vehicle"].astype(np.int32),
            "x": out["x"].astype(np.float32), "lane": out["lane"].astype(np.int8), "v": out["v"].astype(np.float32)}  # fmt: skip


def sawtooth(t: float) -> float:
    """Rises by 0.2 m/s per second for 40 s, then falls by 2 m/s per second for 4 s (period 44 s)."""
    phase = t % 44.0
    return 8.0 + 0.2 * phase if phase <= 40.0 else 16.0 - 2.0 * (phase - 40.0)


def test_acceleration_and_index_by_hand():
    traj = stream(sawtooth, n_vehicles=50, duration=400)
    a = acceleration_samples(traj, (0.0, 400.0))
    assert set(np.round(a, 4)) <= {0.2, -2.0}
    out = asymmetry_of(a, 0.1)
    assert out["asymmetry_index"] == pytest.approx(0.1, rel=1e-5)
    assert out["mean_acceleration"] == pytest.approx(0.2, rel=1e-5)
    assert out["mean_deceleration"] == pytest.approx(2.0, rel=1e-5)
    up, down = out["accelerating_share"], out["decelerating_share"]
    assert up == pytest.approx((a > 0.1).mean()) and up + down + out["cruising_share"] == pytest.approx(1.0)
    assert up == pytest.approx(40.0 / 44.0, abs=0.02) and out["n_samples"] == len(a)
    # window: both samples of a pair inside; lanes: the lane of the first sample of the pair
    inside = acceleration_samples(traj, (100.0, 200.0))
    assert len(inside) == len(acceleration_samples({k: v for k, v in traj.items()}, (100.0, 200.0), lanes=range(1, 7)))
    t = traj["t"]
    pairs = sum(int(((t[traj["vehicle"] == i] >= 100) & (t[traj["vehicle"] == i] <= 199)).sum()) for i in range(50))
    assert len(inside) <= pairs
    assert len(acceleration_samples(traj, (0.0, 400.0), lanes=[1])) < len(a)
    # a gap in the samples of a vehicle (a teleport) gives no pair; a constant speed has no asymmetry
    flat = asymmetry_of(np.zeros(10))
    assert flat["asymmetry_index"] is None and flat["cruising_share"] == 1.0
    assert asymmetry_of(np.zeros(0))["accelerating_share"] is None


def test_welch_peak_and_centroid_of_a_sinusoid():
    t = 2.0 * np.arange(300)
    series = 10.0 + 2.0 * np.sin(2.0 * np.pi * F0 * t)
    f, p, segments = welch_psd(series, 2.0, 128.0)
    assert np.allclose(np.diff(f), 1.0 / 128.0) and segments == 1 + (300 - 64) // 32
    out = band_summary(f, p, 0.002, 0.05)
    assert out["peak_frequency"] == pytest.approx(F0) and out["peak_period_s"] == pytest.approx(64.0)
    assert out["n_lines"] == 6 and abs(out["centroid"] - F0) < 0.004
    # the band power is the variance of the sinusoid (2^2 / 2) up to the leakage of the Hann window
    assert out["band_power"] == pytest.approx(2.0, rel=0.1) and out["band_rms"] == pytest.approx(np.sqrt(2.0), rel=0.05)
    assert welch_psd(series[:50], 2.0, 128.0) is None  # shorter than one segment
    assert band_summary(f, p, 0.3, 0.4)["peak_frequency"] is None


def test_detector_series_fills_gaps():
    prepared = {"t": np.array([0.0, 1.0, 10.0, 11.0]), "x": np.array([95.0, 105.0, 95.0, 105.0]),
                "v": np.array([10.0, 10.0, 14.0, 14.0]), "piece": np.array([0, 0, 1, 1]),
                "vehicle": np.array([0, 0, 1, 1]), "lane": np.array([1, 1, 2, 2])}  # fmt: skip
    edges = np.arange(0.0, 14.0, 2.0)
    series, coverage, passages = detector_speed_series(prepared, 100.0, edges)
    assert passages == 2 and coverage == pytest.approx(2 / 6)
    assert series[0] == 10.0 and series[-1] == 14.0 and series[2] == pytest.approx(10.0 + 4.0 * 4.0 / 10.0)
    only_two, coverage_two, _ = detector_speed_series(prepared, 100.0, edges, lanes=[2])
    assert np.allclose(only_two, 14.0) and coverage_two == pytest.approx(1 / 6)
    empty, coverage_none, n = detector_speed_series(prepared, 100.0, edges, lanes=[5])
    assert np.isnan(empty).all() and coverage_none == 0.0 and n == 0


def test_asymmetry_metrics_end_to_end():
    """A stream whose common speed oscillates with the period 64 s: every detector sees it, the spectrum peaks
    there, the index of a sinusoid is about 1; lane 7 is outside the field of I-80."""
    speed = lambda s: 10.0 + 2.0 * np.sin(2.0 * np.pi * F0 * s)  # noqa: E731
    traj = stream(speed, lane_of=lambda i: 7 if i % 10 == 0 else 1 + i % 6)
    vehicles = {"depart": np.arange(700, dtype=float), "entry_x": np.full(700, 20.0), "entry_lane": np.ones(700),
                "exit_t": np.full(700, np.nan)}  # fmt: skip
    cfg = MacroConfig()
    out = asymmetry_metrics(traj, vehicles, GEOMETRY, cfg, AsymmetryConfig())
    assert out["lanes"] == [1, 2, 3, 4, 5, 6] and out["window"] == [0.0, 600.0]
    accel, spectrum = out["acceleration"], out["spectrum"]
    assert accel["asymmetry_index"] == pytest.approx(1.0, abs=0.05)
    assert spectrum["n_detectors"] == 5 and set(spectrum["detectors"]) == {"100", "200", "300", "400", "450"}
    assert spectrum["peak_frequency"] == pytest.approx(F0) and spectrum["n_samples"] == 300
    assert all(d["used"] and d["coverage"] > 0.85 for d in spectrum["detectors"].values())  # the far ones fill late
    assert len(spectrum["psd"]) == len(spectrum["frequency"]) == 33
    lanes_all = asymmetry_metrics(traj, vehicles, GEOMETRY, MacroConfig.from_mapping({"waves": {"lanes": [1, 2, 3, 4, 5, 6, 7]}}),
                                  AsymmetryConfig())  # fmt: skip
    assert lanes_all["acceleration"]["n_samples"] > accel["n_samples"]
    # a detector with too few passages is left out of the mean spectrum
    sparse = asymmetry_metrics(traj, vehicles, GEOMETRY, cfg, AsymmetryConfig(min_coverage=1.01))
    assert sparse["spectrum"]["n_detectors"] == 0 and sparse["spectrum"]["peak_frequency"] is None
    prepared = prepare_trajectories(traj, vehicles, GEOMETRY, cfg)
    assert prepared["piece"].max() == 699


def test_config_rejects_unknown_keys():
    assert AsymmetryConfig.from_mapping({"threshold": 0.2}).threshold == 0.2
    with pytest.raises(ValueError, match="unknown asymmetry keys"):
        AsymmetryConfig.from_mapping({"thresh": 0.2})


@pytest.fixture()
def tree(tmp_path):
    """A corridor tree: one scenario with its ground truth and two runs (one without run.json)."""
    root = tmp_path / "corridor"
    scenario = root / "scenarios" / "syn_p0"
    truth = stream(lambda s: 10.0 + 2.0 * np.sin(2.0 * np.pi * F0 * s), n_vehicles=300, duration=400)
    vehicles = {"depart": np.arange(300, dtype=float), "depart_planned": np.arange(300, dtype=float),
                "entry_x": np.full(300, 20.0), "entry_lane": np.ones(300), "exit_t": np.full(300, np.nan)}  # fmt: skip
    scenario.mkdir(parents=True)
    np.savez(scenario / "ground_truth.npz", **truth)
    np.savez(scenario / "vehicles_truth.npz", **vehicles)
    write_json(scenario / "scenario.json", {"scenario": "syn_p0", "analysis_window": [0.0, 300.0],
                                             "geometry": {"x_in": 20.0, "x_out": 500.0}})  # fmt: skip
    for law, run_json in (("law_a", True), ("law_b", False)):
        run = root / "syn_p0" / law / "seed0"
        run.mkdir(parents=True)
        np.savez(run / "trajectories.npz", **stream(sawtooth, n_vehicles=300, duration=400))
        np.savez(run / "vehicles.npz", **vehicles)
        if run_json:
            write_json(run / "run.json", {"scenario": "syn_p0", "law": law, "seed": 0})
    return root


def test_update_asymmetry_files(tree):
    lines = list(update_asymmetry(tree, MacroConfig(), AsymmetryConfig()))
    assert sorted(line.split(" ", 1)[0] for line in lines) == ["OK", "OK", "SKIPPED"], lines
    assert any("SKIPPED syn_p0/law_b/seed0: run.json missing" in line for line in lines)
    truth = read_json(tree / "scenarios" / "syn_p0" / FILE)
    run = read_json(tree / "syn_p0" / "law_a" / "seed0" / FILE)
    assert truth["kind"] == "truth" and run["kind"] == "run" and run["law"] == "law_a" and run["seed"] == 0
    assert truth["window"] == [0.0, 300.0] and truth["config"]["lanes"] == [1, 2, 3, 4, 5, 6]
    assert run["acceleration"]["asymmetry_index"] == pytest.approx(0.1, rel=1e-4)
    assert truth["spectrum"]["peak_frequency"] == pytest.approx(F0)
    assert truth["config_hash"] == run["config_hash"]
    again = list(update_asymmetry(tree, MacroConfig(), AsymmetryConfig()))
    assert sum(line.startswith("KEPT") for line in again) == 2
    changed = list(update_asymmetry(tree, MacroConfig(), AsymmetryConfig(threshold=0.2)))  # another hash
    assert sum(line.startswith("OK") for line in changed) == 2
    pooled = list(update_asymmetry(tree, MacroConfig(), AsymmetryConfig(), force=True, workers=2))
    assert sum(line.startswith("OK") for line in pooled) == 2
    assert read_json(tree / "syn_p0" / "law_a" / "seed0" / FILE)["acceleration"] == run["acceleration"]
    only = list(update_asymmetry(tree, MacroConfig(), AsymmetryConfig(), law="law_a", force=True))
    assert len(only) == 1 and only[0].startswith("OK syn_p0/law_a/seed0")


def test_script(tree, tmp_path):
    quoted = lambda p: f"'{p.as_posix()}'"  # noqa: E731
    args = [sys.executable, str(REPO_ROOT / "scripts" / "corridor_asymmetry.py"),
            f"paths.corridor_root={quoted(tree)}", f"paths.runs_root={quoted(tmp_path)}",
            f"hydra.run.dir={quoted(tmp_path / 'outputs')}"]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DONE OK 2, SKIPPED 1" in proc.stdout, proc.stdout
    assert (Path(tree) / "scenarios" / "syn_p0" / FILE).exists()
