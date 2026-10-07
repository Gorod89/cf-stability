"""scripts/corridor_metrics.py and cf_stability.corridor.macro.update_macro on a hand-made tree of runs.

The tree follows docs/m5_contract.md, section 1: a ground truth (uniform flow), a run identical to it
(macro error 0), a slower run, a run with a broken file and a run without run.json. Checked: one line
per file, the keys of macro.json, files that are up to date are kept, a changed input or configuration
recomputes, filters, and the script as a child process (metrics and tables).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from cf_stability.corridor.macro import Geometry, MacroConfig, find_items, update_macro
from cf_stability.utils import REPO_ROOT, read_json
from test_corridor_macro import constant_paths, sample, uniform_flow

MACRO_KEYS = {
    "window", "detectors", "throughput_vph", "throughput_vph_per_lane", "mean_speed", "fd", "fd_scatter",
    "queue_discharge_flow", "flow_peak_2min", "capacity_drop", "congested_share", "waves", "travel_time",
    "collisions_per_1000_vkm", "collision_episodes", "vehicles_in_contact_share", "inserted_share",
    "mean_depart_delay_s", "config", "config_hash",
}  # fmt: skip


def save(directory: Path, names: tuple[str, str], trajectories: dict, vehicles: dict, run_json: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / names[0], **trajectories)
    np.savez_compressed(directory / names[1], **vehicles)
    if run_json:
        (directory / "run.json").write_text(json.dumps({"n_planned": len(vehicles["depart"])}), encoding="utf-8")


def make_tree(root: Path) -> Path:
    truth = uniform_flow()
    scenario = root / "scenarios" / "sc"
    save(scenario, ("ground_truth.npz", "vehicles_truth.npz"), *truth, run_json=False)
    (scenario / "scenario.json").write_text(json.dumps({"scenario": "sc", "geometry": {"x_in": 20, "x_out": 500},
                                                        "window": [120, 840]}), encoding="utf-8")  # fmt: skip
    save(root / "sc" / "law_a" / "seed0", ("trajectories.npz", "vehicles.npz"), *truth)
    slower = sample(constant_paths(8.0, 2.0, 1, 0.37) + constant_paths(12.0, 3.0, 2, 1.13))
    save(root / "sc" / "law_a" / "seed1", ("trajectories.npz", "vehicles.npz"), *slower)
    np.savez(root / "sc" / "law_a" / "seed1" / "collisions.npz", t=np.array([300.0, 500.0, 900.0]),
             vehicle=np.array([5, 6, 7]), x=np.array([30.0, 470.0, 250.0]), lane=np.array([1, 2, 1]))  # fmt: skip
    broken = root / "sc" / "law_b" / "seed0"
    save(broken, ("trajectories.npz", "vehicles.npz"), *truth)
    (broken / "trajectories.npz").write_bytes(b"not a zip archive")
    save(root / "sc" / "law_b" / "seed1", ("trajectories.npz", "vehicles.npz"), *truth, run_json=False)
    (root / "laws").mkdir()
    (root / "laws" / "law_a.json").write_text("{}", encoding="utf-8")
    (root / "_logs" / "x" / "seed0").mkdir(parents=True)
    return root


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path / "corridor")


def kinds(lines: list[str]) -> list[tuple[str, str]]:
    """(status, label) of every printed line."""
    return [(line.split(" ", 1)[0], line.split(" ", 1)[1].split(":", 1)[0]) for line in lines]


def test_find_items(tree):
    (tree / "sc" / "law_a" / "seed10").mkdir()
    labels = [item.label for item in find_items(tree)]
    assert labels == ["truth sc", "sc/law_a/seed0", "sc/law_a/seed1", "sc/law_a/seed10", "sc/law_b/seed0", "sc/law_b/seed1"]
    assert [i.label for i in find_items(tree, law="law_b")] == ["sc/law_b/seed0", "sc/law_b/seed1"]
    assert [i.label for i in find_items(tree, scenario="sc", seed=1)] == ["sc/law_a/seed1", "sc/law_b/seed1"]
    assert find_items(tree, scenario="other") == [] and find_items(tree / "nothing") == []


def test_update_macro_lines_and_files(tree):
    lines = list(update_macro(tree, MacroConfig()))
    assert kinds(lines) == [("OK", "truth sc"), ("OK", "sc/law_a/seed0"), ("OK", "sc/law_a/seed1"),
                            ("FAILED", "sc/law_b/seed0"), ("SKIPPED", "sc/law_b/seed1")]  # fmt: skip
    assert "run.json missing" in lines[4] and "macro error 0.000" in lines[1]
    truth = read_json(tree / "scenarios" / "sc" / "macro.json")
    same = read_json(tree / "sc" / "law_a" / "seed0" / "macro.json")
    slower = read_json(tree / "sc" / "law_a" / "seed1" / "macro.json")
    assert MACRO_KEYS <= set(truth) and "macro_error" not in truth and truth["kind"] == "truth"
    assert MACRO_KEYS | {"macro_error"} <= set(same) and same["law"] == "law_a" and same["seed"] == 0
    assert set(truth["detectors"]) == {"100", "200", "300", "400", "450"}
    assert {"flow_vph", "speed"} <= set(truth["detectors"]["450"])
    assert {"density_bins", "flow", "count"} <= set(truth["fd"])
    assert {"n_waves", "wave_speed", "wave_speed_xcorr", "wave_amplitude", "per_wave", "xcorr"} <= set(truth["waves"])
    assert truth["window"] == same["window"] == [120.0, 840.0] and truth["geometry_from_configuration"] == []
    assert {"n", "mean", "median", "p10", "p90", "values"} <= set(truth["travel_time"])
    assert same["macro_error"]["value"] == 0.0 and same["macro_error"]["n_components"] >= 4
    assert set(same["macro_error_dynamic"]["components"]) == {"fd", "wave_speed", "waves", "wave_amplitude"}
    assert same["macro_error_dynamic"]["value"] == 0.0 and "macro_error_dynamic" not in truth
    assert slower["macro_error_dynamic"]["value"] is not None and "dynamic" in lines[2]
    assert set(same["macro_error"]["components"]) == {"throughput", "mean_speed", "queue_discharge_flow", "fd",
                                                      "wave_speed", "n_waves", "wave_amplitude", "travel_time"}  # fmt: skip
    assert slower["macro_error"]["value"] > 0.05 and slower["macro_error"]["components"]["mean_speed"] < -0.1
    assert slower["collision_source"] == "collisions.npz" and slower["collision_episodes"] == 2.0  # 900 s: after
    assert slower["collisions_by_zone"] == {"entry": 1, "exit": 1, "elsewhere": 0}
    assert same["collision_source"] == "collided" and same["collision_episodes"] == 0.0  # vehicles.npz of the loop
    assert same["config_hash"] == truth["config_hash"] == slower["config_hash"]
    assert not (tree / "sc" / "law_b" / "seed0" / "macro.json").exists()
    assert not list(tree.rglob("*.tmp"))

    again = list(update_macro(tree, MacroConfig()))
    assert [k[0] for k in kinds(again)] == ["KEPT", "KEPT", "KEPT", "FAILED", "SKIPPED"]

    future = time.time() + 10  # a changed input makes its macro.json stale
    os.utime(tree / "sc" / "law_a" / "seed1" / "trajectories.npz", (future, future))
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig())))] == ["KEPT", "KEPT", "OK", "FAILED", "SKIPPED"]
    os.utime(tree / "sc" / "law_a" / "seed1" / "collisions.npz", (future + 2, future + 2))
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig())))] == ["KEPT", "KEPT", "OK", "FAILED", "SKIPPED"]
    os.utime(tree / "scenarios" / "sc" / "scenario.json", (future + 5, future + 5))  # the truth, then every run
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig())))] == ["OK", "OK", "OK", "FAILED", "SKIPPED"]
    other = MacroConfig.from_mapping({"fd_bin": 20})  # another configuration recomputes everything
    assert [k[0] for k in kinds(list(update_macro(tree, other)))] == ["OK", "OK", "OK", "FAILED", "SKIPPED"]
    assert [k[0] for k in kinds(list(update_macro(tree, other, force=True)))][:3] == ["OK", "OK", "OK"]


def test_another_window_makes_every_file_stale(tree):
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig())))][:3] == ["OK", "OK", "OK"]
    scenario = tree / "scenarios" / "sc" / "scenario.json"
    stamp = scenario.stat()
    text = json.loads(scenario.read_text(encoding="utf-8"))
    scenario.write_text(json.dumps({**text, "window": [180, 840]}), encoding="utf-8")
    os.utime(scenario, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))  # the old time: the window alone decides
    lines = list(update_macro(tree, MacroConfig()))
    assert [k[0] for k in kinds(lines)][:3] == ["OK", "OK", "OK"]
    for path in (tree / "scenarios" / "sc", tree / "sc" / "law_a" / "seed0", tree / "sc" / "law_a" / "seed1"):
        assert read_json(path / "macro.json")["window"] == [180.0, 840.0]
    assert read_json(tree / "sc" / "law_a" / "seed0" / "macro.json")["macro_error"]["value"] == 0.0
    scenario.write_text(json.dumps({key: value for key, value in text.items() if key != "window"}), encoding="utf-8")
    os.utime(scenario, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    lines = list(update_macro(tree, MacroConfig()))  # no window in scenario.json: the default (180-840 s), unchanged
    assert [k[0] for k in kinds(lines)][:3] == ["KEPT", "KEPT", "KEPT"]
    moved = Geometry(20.0, 500.0, (240.0, 840.0))
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig(), moved)))][:3] == ["OK", "OK", "OK"]
    truth = read_json(tree / "scenarios" / "sc" / "macro.json")
    assert truth["window"] == [240.0, 840.0] and truth["geometry_from_configuration"] == ["window"]


def test_filters_bring_the_ground_truth_up_to_date(tree):
    lines = list(update_macro(tree, MacroConfig(), scenario="sc", law="law_a", seed=1))
    assert kinds(lines) == [("OK", "truth sc"), ("OK", "sc/law_a/seed1")]  # the truth is an input of the run
    assert kinds(list(update_macro(tree, MacroConfig(), law="law_a"))) == [
        ("KEPT", "truth sc"), ("OK", "sc/law_a/seed0"), ("KEPT", "sc/law_a/seed1")]
    assert list(update_macro(tree, MacroConfig(), scenario="other")) == []


def test_run_without_ground_truth_gets_no_macro_error(tree):
    for name in ("ground_truth.npz", "vehicles_truth.npz"):
        (tree / "scenarios" / "sc" / name).unlink()
    lines = list(update_macro(tree, MacroConfig(), law="law_a", seed=0))
    assert kinds(lines) == [("SKIPPED", "truth sc"), ("OK", "sc/law_a/seed0")]
    macro = read_json(tree / "sc" / "law_a" / "seed0" / "macro.json")
    error, dynamic = macro["macro_error"], macro["macro_error_dynamic"]
    assert error["value"] is None and error["n_components"] == 0 and "no ground truth" in error["error"]
    assert dynamic["value"] is None and dynamic["n_components"] == 0 and "no ground truth" in dynamic["error"]


def test_script_writes_metrics_and_tables(tree, tmp_path):
    out = tmp_path / "tables"
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "corridor_metrics.py"), "tables=true",
        f"paths.corridor_root='{tree.as_posix()}'", f"paths.runs_root='{tmp_path.as_posix()}'",
        f"paths.out_dir='{out.as_posix()}'", "design.scenarios=[sc]", "design.laws=[law_a,law_b]",
        "design.seeds=[0,1]", "design.reference=law_a", "design.candidate=law_b", "design.n_resamples=50",
        f"hydra.run.dir='{(tmp_path / 'outputs').as_posix()}'",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    tables = ("laws", "components", "instability", "tost", "h12_1", "h12_2", "e4_rmax", "sensitivity", "verdicts")
    tables_m8 = ("asymmetry", "asymmetry_contrasts", "correlation_pooled", "power", "temporal",  # runs_root/_tables/m8
                 "correlation_clustered", "contacts_absolute", "h12_2_error")  # the last three: review of M8
    assert [line.split(" ", 1)[0] for line in lines] == ["OK", "OK", "OK", "FAILED", "SKIPPED",
                                                         *["TABLE"] * (len(tables) + len(tables_m8))]  # fmt: skip
    for name in tables:
        assert (out / f"{name}.md").exists() and (out / f"{name}.csv").exists()
    for name in tables_m8:
        assert (tmp_path / "_tables" / "m8" / f"{name}.md").exists(), name
    missing = (out / "missing.txt").read_text(encoding="utf-8")
    assert "sc/law_b/seed0/macro.json missing" in missing and "laws/law_b.json missing" in missing

    single = subprocess.run([*args[:2], "scenario=sc", "law=law_a", "seed=0", *args[3:]], cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=600)  # fmt: skip
    assert single.returncode == 0, single.stdout + single.stderr
    assert [line.split(" ", 2)[:2] for line in single.stdout.strip().splitlines()] == [["KEPT", "truth"], ["KEPT", "sc/law_a/seed0:"]]


def test_metrics_settings_of_a_scenario(tree):
    """scenario.json["metrics"] (US-101, D112) sets the detectors of its ground truth and runs over the configuration;
    the hash of their macro.json follows; a scenario without the block keeps the configuration's hash."""
    base = list(update_macro(tree, MacroConfig()))
    assert [k[0] for k in kinds(base)][:3] == ["OK", "OK", "OK"]
    base_hash = read_json(tree / "scenarios" / "sc" / "macro.json")["config_hash"]
    scenario = tree / "scenarios" / "sc" / "scenario.json"
    text = json.loads(scenario.read_text(encoding="utf-8"))
    scenario.write_text(json.dumps({**text, "metrics": {"detectors": [150, 350], "throughput_x": 350,
                                                         "waves": {"lanes": [1]}}}), encoding="utf-8")  # fmt: skip
    lines = list(update_macro(tree, MacroConfig()))
    assert [k[0] for k in kinds(lines)][:3] == ["OK", "OK", "OK"]
    truth = read_json(tree / "scenarios" / "sc" / "macro.json")
    run = read_json(tree / "sc" / "law_a" / "seed0" / "macro.json")
    assert set(truth["detectors"]) == {"100", "150", "350"}  # queue_x 100 of the configuration stays
    assert truth["config"]["throughput_x"] == 350
    assert truth["config"]["waves"]["lanes"] == [1] and truth["config_hash"] == run["config_hash"] != base_hash
    assert truth["throughput_vph"] == pytest.approx(3000.0) and run["macro_error"]["value"] == 0.0
    assert [k[0] for k in kinds(list(update_macro(tree, MacroConfig())))][:3] == ["KEPT", "KEPT", "KEPT"]
