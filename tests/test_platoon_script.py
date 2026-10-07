"""scripts/platoon_test.py end to end on tiny models: platoon.json, start at the equilibrium inside the band
of the run's context, summary line, skipping, failures."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cf_stability.models import IDM, load_model, save_model
from cf_stability.stability.equilibrium import Band
from cf_stability.stability.platoon import openacc_profile, platoon_summary, platoon_test
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json
from test_platoon import write_run

CALIBRATED = {"v0": 29.57, "T": 0.676, "s0": 2.015, "a": 0.351, "b": 0.5}  # global calibration on highD
BAND = {  # spacing band of a training context (D78): at 9.7 m/s 3.9 .. 20.9 m, at 20 m/s 8 .. 33.8 m
    "v": [5.0, 25.0], "s_low": [2.0, 10.0], "s_median": [8.0, 20.0], "s_high": [15.0, 40.0], "n": [500, 500],
    "settings": {"dv_max": 0.5, "a_max": 0.3, "half_width": 0.5, "quantiles": [0.05, 0.5, 0.95], "min_samples": 200},
}  # fmt: skip
PROFILES = ["Zala/human.csv", "Zala/acc.csv"]
TOP_KEYS = {"config", "config_hash", "git_revision", "run", "model", "band", "profiles", "summary", "wall_time_s"}
SUMMARY_KEYS = {
    "growth_error", "growth_error_mean", "growth_error_human", "growth_error_acc", "growth_profiles",
    "growth_collided", "n_profiles", "n_collided", "n_start_equilibrium", "std_ratio", "hysteresis_area_pulse",
}  # fmt: skip


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def same(a, b) -> bool:
    """Equal JSON-like values, floats up to rounding."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b and type(a) is type(b)


def idm_equilibrium(v: float) -> float:
    """Equilibrium spacing of the IDM: ``(s0 + v T) / sqrt(1 - (v / v0)^4)``."""
    p = CALIBRATED
    return (p["s0"] + v * p["T"]) / math.sqrt(1.0 - (v / p["v0"]) ** 4)


def small_overrides(tmp_path: Path) -> list[str]:
    return [
        "stability.n_vehicles=4", f"stability.raw_dir={quoted(tmp_path / 'openacc')}",
        f"stability.profiles=[{','.join(PROFILES)}]", "stability.pulse.duration_s=20.0",
        "stability.empirical_growth=null", "stability.device=cpu",
    ]  # fmt: skip


def run_script(tmp_path: Path, *overrides: str) -> list[str]:
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "platoon_test.py"), *overrides, *small_overrides(tmp_path),
        f"paths.runs_root={quoted(tmp_path / 'runs')}", f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.strip().splitlines()


def test_platoon_script(tmp_path):
    write_run(tmp_path / "openacc" / PROFILES[0], acc_flag=0)
    write_run(tmp_path / "openacc" / PROFILES[1], acc_flag=1)
    runs = tmp_path / "runs" / "exp" / "synthetic"
    good, broken, unbanded = (runs / name / "driver_fold0_seed0" for name in ("idm", "broken", "no_band"))
    for run in (good, broken, unbanded):
        run.mkdir(parents=True)
    for run in (good, unbanded):
        save_model(IDM(CALIBRATED), run / "model.pt")
    write_json(good / "metrics.json", {"context": {"band": BAND}})  # the unbanded run has no metrics.json
    (broken / "model.pt").write_bytes(b"not a checkpoint")

    lines = run_script(tmp_path, "experiment=exp", "data=synthetic")
    assert len(lines) == 3 and "FAILED" in lines[0] and lines[1].startswith("idm/driver_fold0_seed0")
    assert "growth error" in lines[1] and "collisions 0/3" in lines[1] and "hysteresis" in lines[1]
    assert "start at equilibrium 3/3" in lines[1] and "start at equilibrium 0/3" in lines[2]
    result = read_json(good / "platoon.json")
    assert set(result) == TOP_KEYS and result["model"] == "idm" and result["run"] == str(good) and result["band"]
    assert result["config_hash"] == config_hash(result["config"]["stability"])
    assert result["config"]["stability"]["start_gap"] == "anchored"  # the default
    assert set(result["profiles"]) == {*PROFILES, "pulse"} and set(result["summary"]) == SUMMARY_KEYS
    assert all("hysteresis_area" in profile for profile in result["profiles"].values())

    # every leader starts inside the band: the platoon starts at the equilibrium of the IDM
    first_speed = openacc_profile(tmp_path / "openacc" / PROFILES[0])["v_lead"][0]
    for name, v_start in ((PROFILES[0], first_speed), (PROFILES[1], first_speed), ("pulse", 20.0)):
        profile = result["profiles"][name]
        assert profile["start"] == "equilibrium" and profile["start_equilibrium"]
        assert profile["start_gap_m"] == pytest.approx(idm_equilibrium(v_start), rel=1e-9)
    unbanded_result = read_json(unbanded / "platoon.json")
    assert not unbanded_result["band"] and unbanded_result["summary"]["n_start_equilibrium"] == 0
    assert unbanded_result["profiles"]["pulse"]["start_gap_m"] == 2.0 + 1.5 * 20.0  # the time gap

    # the same numbers as the functions of the package
    cfg = {**result["config"]["stability"], "raw_dir": str(tmp_path / "openacc")}
    model = load_model(good / "model.pt")
    expected = platoon_test(model, cfg, band=Band.from_context({"band": BAND}, model))
    assert same(result["profiles"], expected)
    summary = result["summary"]
    assert same(summary, platoon_summary(expected)) and summary["n_start_equilibrium"] == 3
    assert summary["growth_error_human"] == pytest.approx(expected[PROFILES[0]]["growth_error"])
    assert summary["growth_error_acc"] == pytest.approx(expected[PROFILES[1]]["growth_error"])
    assert summary["hysteresis_area_pulse"] == pytest.approx(expected["pulse"]["hysteresis_area"])
    assert summary["hysteresis_area_pulse"] > 0.0
    assert same(unbanded_result["profiles"], platoon_test(model, {**cfg, "start_gap": "time_gap"}))
    failed = read_json(broken / "platoon.json")
    assert failed["model"] is None and "error" in failed and "profiles" not in failed

    assert run_script(tmp_path, "experiment=exp", "data=synthetic") == []  # results newer than model.pt are kept
    lines = run_script(tmp_path, f"run={quoted(good)}")
    assert len(lines) == 1 and lines[0].startswith("idm/driver_fold0_seed0")
