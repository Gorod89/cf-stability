"""Control loop of the corridor and scripts/run_corridor.py (docs/m5_contract.md, sections 1, 3 and 4).

libsumo is never imported here: every simulation runs in a child process, on synthetic scenarios built in a
temporary directory (about 60 s with a few dozen vehicles; a dense queue of 360 vehicles for the exit-count
feedback of the downstream boundary) and toy laws (IDM parameter sets).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cf_stability.corridor.laws import export_laws
from cf_stability.corridor.scenario import build_scenario
from cf_stability.models import IDM, save_model
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

I80 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "i80.yaml").read_text(encoding="utf-8"))
US101 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "us101.yaml").read_text(encoding="utf-8"))
STABLE = {"v0": 15.0, "T": 1.0, "s0": 2.0, "a": 1.5, "b": 2.0}
OTHER = {"v0": 13.0, "T": 1.4, "s0": 2.5, "a": 1.0, "b": 2.0}
TIGHT = {"v0": 15.0, "T": 0.75, "s0": 1.5, "a": 1.0, "b": 2.0}  # gap 5.3 m at 5 m/s, the queue of the data 5.5 m
TAIL = "sim.tail_s=20.0"  # last departure at 34 s: 54 s simulated
FEEDBACK_STATS = {"g_mean", "g_min", "g_max", "dN_mean", "dN_min", "dN_max"}
RUN_KEYS = {
    "scenario", "law", "seed", "n_planned", "n_inserted", "n_exited", "n_offramp", "n_in_network", "n_collisions",
    "n_sumo_collisions", "n_teleports", "n_not_inserted", "mean_depart_delay_s", "sim_time_s", "steps",
    "vehicle_steps", "vehicle_steps_per_second", "wall_time_s", "sumo_version", "config", "config_hash", "git_revision",
    *FEEDBACK_STATS,
}  # fmt: skip
VEHICLE_KEYS = {
    "vehicle_id", "depart_planned", "depart", "entry_x", "entry_lane", "exit_t", "exit_lane", "length", "v_class",
    "member", "n_collisions", "teleported",
}  # fmt: skip
HASHED = ("scenario", "law", "seed", "sim", "device", "scenario_hash", "law_hash")
N_SYN = 34


def track(track_id, frame0, x0, v, lanes, length=4.5, v_class=2):
    n = int(np.ceil((520.0 - x0) / (v * 0.1))) + 1
    x = x0 + v * 0.1 * np.arange(n)
    lane = np.full(n, lanes[0][1])
    for x_from, value in lanes[1:]:
        lane[x >= x_from] = value
    return pd.DataFrame({"track_id": track_id, "period": 0, "vehicle_id": int(track_id.split("_")[1]),
                         "frame_id": frame0 + np.arange(n), "lane_id": lane, "x": x, "v": v, "length": length,
                         "v_class": v_class})  # fmt: skip


def synthetic_tracks() -> pd.DataFrame:
    """Four vehicles per main lane, 10 s apart at 12 m/s; four on the on-ramp merging into lane 6 (one enters
    34 m before the end of the auxiliary lane); a truck; a vehicle that appears inside the section. At 27 s
    (frame 278, the origin is frame 8): vehicle 41 enters lane 5 at 12 m/s 15.3 m behind vehicle 40 at 5 m/s
    (SUMO's insertion check lets it wait), and vehicle 43 enters lane 3 overlapping the back of vehicle 42 by
    0.7 m. Departures within 0-35 s."""
    parts, k = [], 0
    for lane in range(1, 7):
        for j in range(4):
            k += 1
            parts.append(track(f"p0_{k}", 1 + 100 * j + 7 * lane, 20.2, 12.0, [(0, lane)]))
    for j in range(3):
        k += 1
        parts.append(track(f"p0_{k}", 20 + 130 * j, 100.5, 7.0, [(0, 7), (150.0 + 10 * j, 6)]))
    parts.append(track(f"p0_{k + 1}", 150, 20.4, 11.0, [(0, 4)], length=13.0, v_class=3))
    parts.append(track(f"p0_{k + 2}", 350, 260.0, 12.0, [(0, 2)]))
    parts += [
        track("p0_40", 278, 40.0, 5.0, [(0, 5)]), track("p0_41", 278, 20.2, 12.0, [(0, 5)]),
        track("p0_42", 278, 24.0, 5.0, [(0, 3)]), track("p0_43", 278, 20.2, 5.0, [(0, 3)]),
        track("p0_44", 208, 170.0, 5.0, [(0, 7), (200.0, 6)]),  # on-ramp, 34 m before the end of the lane
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def queue_tracks() -> pd.DataFrame:
    """A dense queue: every lane fed every 2 s at 5 m/s (net gap 5.5 m) for 2 minutes, 360 vehicles; the data
    pass x_out from 96 s to 214 s."""
    parts = [
        track(f"p0_{1 + 60 * (lane - 1) + j}", 1 + 20 * j + 3 * lane, 20.2, 5.0, [(0, lane)])
        for lane in range(1, 7) for j in range(60)
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def toy_law(path: Path, name: str, params: list[dict]) -> None:
    write_json(path / f"{name}.json", {"law": name, "kind": "idm", "device": "cpu", "members": [], "params": params,
                                       "support_v": None, "window": 1})  # fmt: skip


@pytest.fixture(scope="module")
def corridor(tmp_path_factory):
    root = tmp_path_factory.mktemp("corridor")
    build_scenario(synthetic_tracks(), {**I80, "name": "syn"}, 0, root / "scenarios" / "syn_p0")
    laws = root / "laws"
    toy_law(laws, "toy", [STABLE, OTHER])
    toy_law(laws, "toy_b", [{**STABLE, "T": 2.0}, {**OTHER, "T": 2.0}])
    for k, params in enumerate((STABLE, OTHER)):
        member = root / "members" / f"fold{k}"
        member.mkdir(parents=True)
        save_model(IDM(params), member / "model.pt")
        write_json(member / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.0], "box_high": [50.0, 3.0, 18.0]}})
    table = {"toy_models": {"kind": "models", "device": "cpu", "members": (root / "members" / "fold{fold}").as_posix()}}
    assert "2/2 members" in export_laws({"folds": [0, 1], "table": table}, laws)[0]
    broken = root / "members" / "broken" / "model.pt"
    broken.parent.mkdir()
    broken.write_bytes(b"not a checkpoint")
    write_json(laws / "broken.json", {"law": "broken", "kind": "models", "window": 1, "support_v": None,
                                      "members": [{"run": "x", "model": broken.as_posix()}]})  # fmt: skip
    return root


def run_script(root: Path, runs: str, *overrides: str, tail: str = TAIL) -> list[str]:
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_corridor.py"), *overrides, tail,
        f"paths.scenarios_root={quoted(root / 'scenarios')}", f"paths.laws_root={quoted(root / 'laws')}",
        f"paths.runs_root={quoted(root / runs)}", f"hydra.run.dir={quoted(root / 'outputs' / runs)}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [line for line in proc.stdout.splitlines() if line.startswith(("RUN", "CORRIDOR"))]


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def outputs(run_dir: Path) -> tuple[dict, dict, dict]:
    assert set(load(run_dir / "collisions.npz")) == {"t", "vehicle", "x", "lane"}
    return read_json(run_dir / "run.json"), load(run_dir / "trajectories.npz"), load(run_dir / "vehicles.npz")


@pytest.fixture(scope="module")
def toy_run(corridor):
    """syn_p0 / toy / seed 3, run once for the tests that read it."""
    lines = run_script(corridor, "single", "scenario=syn_p0", "law=toy", "seed=3")
    assert len(lines) == 1 and lines[0].startswith(f"RUN syn_p0/toy/seed3: inserted {N_SYN}/{N_SYN}"), lines
    return outputs(corridor / "single" / "syn_p0" / "toy" / "seed3")


def test_run_files_and_conservation(toy_run):
    run, traj, veh = toy_run
    assert set(run) == RUN_KEYS and run["scenario"] == "syn_p0" and run["law"] == "toy" and run["seed"] == 3
    assert run["config_hash"] == config_hash({key: run["config"][key] for key in HASHED})
    assert run["config"]["device"] == "cpu" and run["config"]["sim"]["leader_range"] == 50.0
    assert run["sumo_version"].startswith("SUMO 1.27")
    # conservation (collisions do not remove vehicles); a stable law: no collision, no teleport, everybody inserted
    assert run["n_offramp"] == 0  # I-80 has no off-ramp
    assert run["n_inserted"] == run["n_exited"] + run["n_offramp"] + run["n_in_network"] + run["n_teleports"]
    assert run["n_planned"] == N_SYN and run["n_inserted"] == N_SYN and run["n_not_inserted"] == 0
    assert run["n_collisions"] == run["n_sumo_collisions"] == run["n_teleports"] == 0
    assert run["n_exited"] > 0 and run["n_in_network"] > 0  # the last departures are still on the way
    last = float(veh["depart_planned"].max())
    assert run["sim_time_s"] == pytest.approx(last + 20.0) and run["steps"] == round(10 * (last + 20.0)) + 1
    assert run["vehicle_steps"] > 0 and run["vehicle_steps_per_second"] > 0
    assert all(run[key] is None for key in FEEDBACK_STATS)  # the analysis window (180-840 s) lies after the run

    assert set(veh) == VEHICLE_KEYS and len(veh["vehicle_id"]) == N_SYN
    assert veh["vehicle_id"].dtype == np.int32 and veh["member"].dtype == np.int8
    assert veh["n_collisions"].dtype == np.int32 and veh["teleported"].dtype == np.bool_
    assert not veh["n_collisions"].any()
    inserted = np.isfinite(veh["depart"])
    assert (veh["depart"][inserted] >= veh["depart_planned"][inserted] - 1e-9).all()
    assert run["mean_depart_delay_s"] == pytest.approx(float(np.mean(veh["depart"] - veh["depart_planned"])))
    exited = np.isfinite(veh["exit_t"])
    assert exited.sum() == run["n_exited"] and (veh["exit_t"][exited] > veh["depart"][exited]).all()
    assert set(veh["exit_lane"][exited].tolist()) <= set(range(1, 7)) and (veh["exit_lane"][~exited] == -1).all()
    assert set(veh["member"].tolist()) <= {0, 1}

    assert {key: traj[key].dtype for key in traj} == {
        "t": np.float32, "vehicle": np.int32, "x": np.float32, "lane": np.int8, "v": np.float32
    }  # fmt: skip
    assert np.array_equal(traj["t"], np.round(traj["t"])) and (np.diff(traj["t"]) >= 0).all()
    assert traj["x"].min() >= 20.0 and traj["x"].max() <= 500.0 and set(traj["lane"].tolist()) <= set(range(1, 8))
    assert traj["x"][traj["lane"] == 7].max() <= 204.0  # the auxiliary lane ends at 204 m
    ramp = np.flatnonzero(veh["entry_lane"] == 7)
    assert ramp.size == 4 and all(6 in traj["lane"][traj["vehicle"] == r] for r in ramp)  # the merges happened
    # every vehicle at every whole second from its insertion on while inside the section
    for row in (0, 5):
        mine = traj["vehicle"] == row
        assert np.allclose(np.diff(traj["t"][mine]), 1.0) and traj["t"][mine][0] - veh["depart"][row] < 1.0
    # the HOV lane: only the vehicles that use lane 1 in the data (vehicle ids 1-4) are ever on it
    hov_rows = np.flatnonzero(np.isin(veh["vehicle_id"], [1, 2, 3, 4]))
    assert np.isin(traj["vehicle"][traj["lane"] == 1], hov_rows).all() and (traj["lane"] == 1).sum() > 50


def test_insertion_setting(toy_run):
    """SUMO's default insertion checks with tau 1 s, decel 8 m/s^2: the leaders enter on time; a vehicle at 12 m/s
    15.3 m behind a leader at 5 m/s waits until the gap reaches about 19 m (1 s of reaction plus the difference
    of the braking distances); one that overlaps its leader waits beyond the overlap."""
    _, _, veh = toy_run
    row = {int(v): i for i, v in enumerate(veh["vehicle_id"])}
    for vid in (40, 41, 42, 43):
        assert veh["depart_planned"][row[vid]] == pytest.approx(27.0)
    assert veh["depart"][row[40]] == pytest.approx(27.0) and veh["depart"][row[42]] == pytest.approx(27.0)
    assert 27.0 + 1e-9 < veh["depart"][row[41]] < 29.0
    # vehicle 42 moves off the overlap at 27.2 s; the gap then needs a few metres more
    assert veh["depart"][row[43]] >= 27.5 - 1e-9 and not veh["n_collisions"].any()


def test_member_does_not_depend_on_the_simulation(corridor, toy_run):
    run_script(corridor, "single", "scenario=syn_p0", "law=toy_b", "seed=3")
    other = outputs(corridor / "single" / "syn_p0" / "toy_b" / "seed3")
    assert not np.array_equal(toy_run[1]["x"], other[1]["x"])  # other laws, other trajectories
    members = np.random.default_rng(3).integers(2, size=N_SYN)
    assert np.array_equal(toy_run[2]["member"], members) and np.array_equal(other[2]["member"], members)


def test_multi_run_reproducible_and_restart_safe(corridor, toy_run):
    grid = ("scenarios=[syn_p0]", "laws=[toy,broken,missing]", "seeds=[3,5]", "workers=2")
    lines = run_script(corridor, "grid", *grid)
    runs = corridor / "grid" / "syn_p0"
    done = sorted(line.split()[1] for line in lines if line.startswith("RUN") and " done: " in line)
    failed = sorted(line.split()[1] for line in lines if line.startswith("RUN") and " failed" in line)
    assert done == ["syn_p0/toy/seed3", "syn_p0/toy/seed5"], lines
    assert failed == [f"syn_p0/{law}/seed{s}" for law in ("broken", "missing") for s in (3, 5)], lines
    assert lines[-1] == "CORRIDOR done 2, complete 0, failed 4"
    assert (corridor / "grid" / "_logs" / "syn_p0__broken__seed3.log").exists()
    # the same run in another process and directory gives the same arrays
    grid3 = outputs(runs / "toy" / "seed3")
    for key in toy_run[1]:
        assert np.array_equal(toy_run[1][key], grid3[1][key]), key
    for key in toy_run[2]:
        assert np.array_equal(toy_run[2][key], grid3[2][key], equal_nan=toy_run[2][key].dtype.kind == "f"), key
    assert toy_run[0]["config_hash"] == grid3[0]["config_hash"]

    stamp = {s: (runs / "toy" / f"seed{s}" / "run.json").stat().st_mtime_ns for s in (3, 5)}
    lines = run_script(corridor, "grid", "scenarios=[syn_p0]", "laws=[toy]", "seeds=[3,5]")
    assert [line.split(" (")[0] for line in lines[:2]] == [f"RUN syn_p0/toy/seed{s} complete" for s in (3, 5)]
    assert lines[2].startswith("CORRIDOR 0 runs to do, 2 complete")
    assert {s: (runs / "toy" / f"seed{s}" / "run.json").stat().st_mtime_ns for s in (3, 5)} == stamp

    # a run killed before its run.json is run again, with the same result
    before = load(runs / "toy" / "seed5" / "trajectories.npz")
    (runs / "toy" / "seed5" / "run.json").unlink()
    lines = run_script(corridor, "grid", "scenarios=[syn_p0]", "laws=[toy]", "seeds=[3,5]")
    assert any(line.startswith("RUN syn_p0/toy/seed3 complete") for line in lines)
    assert any(line.startswith("RUN syn_p0/toy/seed5 done: ") for line in lines)
    after = load(runs / "toy" / "seed5" / "trajectories.npz")
    assert all(np.array_equal(before[key], after[key]) for key in before)
    # another config (sim, device) is another run
    lines = run_script(corridor, "grid", "scenarios=[syn_p0]", "laws=[toy]", "seeds=[3]", "sim.leader_range=40.0")
    assert any(line.startswith("RUN syn_p0/toy/seed3 done: ") for line in lines)


DRIVER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cf_stability.corridor import loop  # libsumo before pandas and torch
    import numpy as np
    from cf_stability.corridor.laws import load_law

    libsumo = loop.libsumo
    scenario = loop.Scenario.load(sys.argv[2])
    inner, device = load_law(sys.argv[3])
    sim = loop.SimConfig(tail_s=20.0)
    reach = sim.leader_range
    geometry = scenario.meta["geometry"]
    x0 = {edge["id"]: edge["x0"] for edge in geometry["edges"]}
    report = {"device": device, "range": reach, "n": 0, "ds": 0.0, "ddv": 0.0, "dv": 0.0, "wall": 0, "free": 0,
              "beyond": 0, "history": True, "calls": 0, "window": 0, "hov_lane": 0, "not_hov_on_hov_lane": 0}
    hov_index = {e["id"]: e["lanes"] - 1 for e in geometry["edges"] if e["id"] in geometry["controlled_edges"]}
    seen, buffer = {}, {}

    class Recording:
        window = 3

        def __init__(self, law):
            self.law, self.n_members = law, law.n_members

        def assign(self, n, rng):
            return self.law.assign(n, rng)

        def accelerations(self, rows, history):
            report["calls"] += 1
            report["window"] = history.shape[1]
            for row, h in zip(rows.tolist(), history):
                past = seen.setdefault(row, [])
                past.append(h[-1].copy())
                want = [past[max(0, len(past) - 3 + j)] for j in range(3)]
                report["history"] &= bool(np.array_equal(np.stack(want), h))
            return self.law.accelerations(rows, history)

    def observer(t, ids, states, acc, histories, speeds):
        for vid in libsumo.vehicle.getIDList():  # the HOV lane (NGSIM 1) of the section, every step
            road = libsumo.vehicle.getRoadID(vid)
            if road in hov_index and libsumo.vehicle.getLaneIndex(vid) == hov_index[road]:
                report["hov_lane"] += 1
                report["not_hov_on_hov_lane"] += libsumo.vehicle.getVehicleClass(vid) != "hov"
        for vid, (s, dv, v) in zip(ids, states):
            speed = libsumo.vehicle.getSpeed(vid)
            lead = libsumo.vehicle.getLeader(vid, 300.0)
            if lead and lead[0] and lead[1] <= reach:
                s_ref, dv_ref = lead[1], speed - libsumo.vehicle.getSpeed(lead[0])
            else:
                s_ref, dv_ref = reach, 0.0
                report["beyond" if lead and lead[0] else "free"] += 1
            road, index = libsumo.vehicle.getRoadID(vid), libsumo.vehicle.getLaneIndex(vid)
            if road == geometry["aux_edge"] and index == geometry["aux_index"]:
                wall = geometry["aux_end"] - (x0[road] + libsumo.vehicle.getLanePosition(vid))
                if wall < s_ref:
                    s_ref, dv_ref = wall, speed
                    report["wall"] += 1
            report["ds"] = max(report["ds"], abs(s - s_ref))
            report["ddv"] = max(report["ddv"], abs(dv - dv_ref))
            report["dv"] = max(report["dv"], abs(v - speed))
            report["n"] += 1
        for vid in libsumo.vehicle.getIDList():  # the speeds on the buffer
            if libsumo.vehicle.getRoadID(vid) in geometry["buffer_edges"]:
                buffer.setdefault(vid, []).append((round(t, 1), libsumo.vehicle.getSpeed(vid)))

    result = loop.run_simulation(scenario, Recording(inner), 7, sim, observer=observer)
    steps = [b[1] - a[1] for series in buffer.values() for a, b in zip(series[:-1], series[1:])
             if abs(b[0] - a[0] - 0.1) < 1e-6]
    report.update(buffer_pairs=len(steps), buffer_min=min(steps), buffer_max=max(steps))
    report["collisions"] = result["stats"]["n_collisions"]
    report["members"] = sorted(set(result["vehicles"]["member"].tolist()))
    report["pyarrow"] = "pyarrow" in sys.modules  # nothing the simulation imports may load it
    print("REPORT " + json.dumps(report))
    """
)


def test_state_leader_range_hov_and_buffer_speeds(corridor):
    """The state handed to the law is the gap and speeds of SUMO, with the leader range of 50 m (farther
    leaders and the end of the auxiliary lane beyond it give s = 50 m, dv = 0); the histories hold the
    last states (the first one repeated); no vehicle but those of class hov is ever on the HOV lane; on the
    buffer the speed changes by at most -3 / +2 m/s^2."""
    driver = corridor / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(driver), str(REPO_ROOT), str(corridor / "scenarios" / "syn_p0"),
         str(corridor / "laws" / "toy_models.json")],  # fmt: skip
        cwd=corridor, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("REPORT "))[7:])
    assert report["range"] == 50.0 and report["n"] > 5000, report
    assert report["ds"] < 1e-9 and report["ddv"] < 1e-9 and report["dv"] < 1e-9, report
    assert report["free"] > 0 and report["beyond"] > 0 and report["wall"] > 0, report  # every substitute occurred
    assert report["history"] and report["window"] == 3 and report["calls"] > 500, report
    assert report["hov_lane"] > 500 and report["not_hov_on_hov_lane"] == 0, report
    assert report["buffer_pairs"] > 100 and report["buffer_min"] >= -0.3 - 1e-9 and report["buffer_max"] <= 0.2 + 1e-9
    assert report["buffer_min"] < -0.29, report  # the vehicles do brake on the buffer, at the limit
    assert report["collisions"] == 0 and report["members"] == [0, 1] and report["device"] == "cpu"
    assert not report["pyarrow"]


@pytest.fixture(scope="module")
def queue(tmp_path_factory):
    """The dense queue with and without the exit-count feedback, analysis window 160-200 s; law TIGHT."""
    root = tmp_path_factory.mktemp("queue")
    tracks = queue_tracks()
    for name, feedback in (("queue", True), ("queuenofb", False)):
        boundary = {**I80["boundary"], "feedback": feedback}
        cfg = {**I80, "name": name, "analysis_window": [160.0, 200.0], "boundary": boundary}
        build_scenario(tracks, cfg, 0, root / "scenarios" / f"{name}_p0")
    toy_law(root / "laws", "tight", [TIGHT])
    lines = run_script(root, "runs", "scenarios=[queue_p0,queuenofb_p0]", "laws=[tight]", "seeds=[0]", "workers=2",
                       tail="sim.tail_s=90.0")  # fmt: skip
    assert lines[-1] == "CORRIDOR done 2, complete 0, failed 0", lines
    return root


def test_exit_count_feedback(queue):
    """A law with a tighter gap than the data lets out more than the data's flow at the prescribed speed: without
    feedback the queue drains, with it the simulated passages of x_out follow those of the data."""
    counts = {}
    for name in ("queue", "queuenofb"):
        run, traj, veh = outputs(queue / "runs" / f"{name}_p0" / "tight" / "seed0")
        truth = load(queue / "scenarios" / f"{name}_p0" / "ground_truth.npz")
        window = slice(160, 200)
        counts[name] = (np.bincount(traj["t"].astype(int), minlength=300)[window].mean(),
                        np.bincount(truth["t"].astype(int), minlength=300)[window].mean(), run)  # fmt: skip
        assert run["n_inserted"] == 360 and run["n_collisions"] == run["n_sumo_collisions"] == run["n_teleports"] == 0
    sim, data, run = counts["queue"]
    assert -8.0 <= run["dN_min"] <= run["dN_mean"] <= run["dN_max"] <= 8.0, run
    assert run["g_mean"] < 1.0 and run["g_min"] >= 0.2 and run["g_max"] <= 1.5
    assert abs(sim / data - 1.0) < 0.1, (sim, data)  # the queue of the data is in the section
    sim, data, run = counts["queuenofb"]
    assert run["dN_mean"] > 50.0 and run["g_min"] == run["g_max"] == 1.0, run
    assert sim < 0.6 * data, (sim, data)  # the section drains




CONTACT_DRIVER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cf_stability.corridor import loop  # libsumo before pandas and torch
    import numpy as np

    libsumo = loop.libsumo
    scenario = loop.Scenario.load(sys.argv[2])
    sim = loop.SimConfig(tail_s=80.0)
    steps = {"k": 0}
    follower = []  # per step of the follower (row 1): t, s, dv, v, commanded speed

    class Scripted:
        # row 0 (the leader) keeps its speed; row 1 closes in on it at 1 m/s whatever the gap (it runs into it
        # and pushes on), brakes from 20 s to 26 s, then closes in again
        n_members, window = 1, 1

        def assign(self, n, rng):
            return np.zeros(n, dtype=np.int8)

        def accelerations(self, rows, history):
            t = 0.1 * steps["k"]
            steps["k"] += 1
            push = np.clip((1.0 - history[:, -1, 1]) / 0.5, -8.0, 4.0) if t < 20.0 or t >= 26.0 else -8.0
            return np.where(rows == 1, push, 0.0)

    locked = set()

    def observer(t, ids, states, acc, histories, speeds):
        for vid, state, speed in zip(ids, states, speeds):
            if vid not in locked:  # no overtaking: the follower has to hit the leader
                libsumo.vehicle.setLaneChangeMode(vid, 0)
                locked.add(vid)
            if vid == "1":
                follower.append([t, *state.tolist(), float(speed)])

    result = loop.run_simulation(scenario, Scripted(), 0, sim, observer=observer)
    print("REPORT " + json.dumps({
        "follower": follower, "stats": result["stats"], "n_collisions": result["vehicles"]["n_collisions"].tolist(),
        "episodes": result["collisions"]["t"].tolist(), "rows": result["collisions"]["vehicle"].tolist(),
    }))
    """
)


@pytest.fixture(scope="module")
def contact(tmp_path_factory):
    """Two vehicles in lane 3 at t = 0: a leader at 30 m and 5 m/s, a follower 5.3 m behind it at 6 m/s."""
    root = tmp_path_factory.mktemp("contact")
    tracks = pd.concat([track("p0_1", 1, 30.0, 5.0, [(0, 3)]), track("p0_2", 1, 20.2, 6.0, [(0, 3)])])
    build_scenario(tracks, {**I80, "name": "contact"}, 0, root / "contact_p0")
    driver = root / "driver.py"
    driver.write_text(CONTACT_DRIVER, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(driver), str(REPO_ROOT), str(root / "contact_p0")], cwd=root, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("REPORT "))[7:])


def test_contact_episodes(contact):
    """A penetration is one collision episode however long the contact lasts; the follower stays in the
    simulation, capped so that it cannot penetrate further; a new episode begins only after the gap has been
    above 1 m (the follower brakes from 20 s to 26 s): two episodes."""
    stats, rows = contact["stats"], np.array(contact["follower"])
    t, s, dv, v, command = rows.T
    assert stats["n_inserted"] == 2 == stats["n_exited"] + stats["n_in_network"] + stats["n_teleports"]  # no removal
    # the follower is driven at every step from its insertion (SUMO lets it in once the gap is safe) to the end
    assert stats["n_in_network"] == 2 and np.allclose(np.diff(t), 0.1) and t[0] < 1.0 and t[-1] == pytest.approx(80.0)
    assert stats["n_collisions"] == 2 and contact["n_collisions"] == [0, 2] and contact["rows"] == [1, 1]
    first, second = contact["episodes"]
    assert 3.0 < first < 10.0 and 26.0 < second < 80.0, contact["episodes"]
    starts, in_contact, previous = [], False, np.inf
    for k in range(len(t)):
        start = not in_contact and s[k] <= 0.0
        if in_contact and s[k] > 1.0:
            in_contact = False
        in_contact = in_contact or start
        if start:
            starts.append(t[k])
        if in_contact:
            assert command[k] <= max(0.0, v[k] - dv[k] + s[k] / 0.1) + 1e-9  # the cap
            if not start:
                assert s[k] >= min(previous, 0.0) - 1e-6  # no further penetration (the leader keeps its speed)
        previous = s[k]
    assert np.allclose(starts, contact["episodes"], atol=1e-4)
    during = (t >= first) & (t < 20.0)
    assert during.sum() > 100 and (s[during] <= 0.0).sum() > 10 and s[during].max() <= 1.0  # one long episode
    assert s[(t > 21.0) & (t < second)].max() > 1.0 and s[t >= second].max() <= 1.0  # apart in between, then one


# ------------------------------------------------------------------------------------------------ US-101 (D112)


def track_to(track_id, frame0, x0, v, lanes, x_end=660.0):
    """A track at constant speed from ``x0`` to ``x_end`` (US-101 geometry); ``lanes``: [(x from which, lane), ...]."""
    n = int(np.ceil((x_end - x0) / (v * 0.1))) + 1
    x = x0 + v * 0.1 * np.arange(n)
    lane = np.full(n, lanes[0][1])
    for x_from, value in lanes[1:]:
        lane[x >= x_from] = value
    return pd.DataFrame({"track_id": track_id, "period": 0, "vehicle_id": int(track_id.split("_")[1]),
                         "frame_id": frame0 + np.arange(n), "lane_id": lane, "x": x, "v": v, "length": 4.5,
                         "v_class": 2})  # fmt: skip


ONRAMP = [(0, 7), (190.0, 6), (260.0, 5)]
OFFRAMP = [(0, 5), (250.0, 6), (405.0, 8)]


def us101_tracks() -> pd.DataFrame:
    """Two vehicles per through lane at 12 m/s from 26 m; on-ramp vehicles 20, 21 (lane 7 from 150 m at 10 m/s,
    then 6, then 5); off-ramp vehicles 22, 23 (lane 5, 6 from 250 m, 8 from 405 m; the tracks end at 450 m); and,
    after all of them, an off-ramp vehicle 24 and an on-ramp vehicle 25 (held in their lanes by the driver test)."""
    parts = [track_to(f"p0_{1 + 2 * (lane - 1) + j}", 1 + 100 * j + 7 * lane, 26.0, 12.0, [(0, lane)])
             for lane in range(1, 6) for j in range(2)]  # fmt: skip
    parts += [
        track_to("p0_20", 50, 150.0, 10.0, ONRAMP), track_to("p0_21", 150, 150.0, 10.0, ONRAMP),
        track_to("p0_22", 30, 26.0, 12.0, OFFRAMP, 450.0), track_to("p0_23", 230, 26.0, 12.0, OFFRAMP, 450.0),
        track_to("p0_24", 400, 26.0, 12.0, OFFRAMP, 450.0), track_to("p0_25", 420, 150.0, 10.0, ONRAMP),
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def us101(tmp_path_factory):
    root = tmp_path_factory.mktemp("us101")
    meta = build_scenario(us101_tracks(), {**US101, "name": "us"}, 0, root / "scenarios" / "us_p0")
    assert meta["counts"]["offramp"] == 3 and meta["counts"]["aux_users"] == 6
    toy_law(root / "laws", "toy", [STABLE, OTHER])
    toy_law(root / "laws", "stable", [STABLE])
    return root


def test_us101_offramp_run(us101):
    """Every vehicle routed to the off-ramp leaves by it, the others pass x_out; nothing is recorded on the
    off-ramp; the on-ramp vehicles merge into the main lanes."""
    lines = run_script(us101, "runs", "scenario=us_p0", "law=toy", "seed=1", tail="sim.tail_s=60.0")
    assert len(lines) == 1 and "(+3 by the off-ramp)" in lines[0], lines
    run, traj, veh = outputs(us101 / "runs" / "us_p0" / "toy" / "seed1")
    assert run["n_planned"] == run["n_inserted"] == 16 and run["n_offramp"] == 3 and run["n_exited"] == 13
    assert run["n_inserted"] == run["n_exited"] + run["n_offramp"] + run["n_in_network"] + run["n_teleports"]
    assert run["n_teleports"] == run["n_collisions"] == run["n_sumo_collisions"] == 0
    row = {int(v): i for i, v in enumerate(veh["vehicle_id"])}
    off = [row[k] for k in (22, 23, 24)]
    assert np.isnan(veh["exit_t"][off]).all() and (veh["exit_lane"][off] == -1).all()
    assert np.isfinite(np.delete(veh["exit_t"], off)).all()
    assert traj["x"][np.isin(traj["vehicle"], off)].max() <= 412.0  # not recorded on the off-ramp edge
    assert set(traj["lane"].tolist()) <= set(range(1, 7)) and traj["x"].max() <= 640.0
    for k in (20, 21, 25):  # inserted on the auxiliary lane, merged into lane 5 before its end
        mine = traj["vehicle"] == row[k]
        assert traj["lane"][mine][0] == 6 and (traj["lane"][mine] <= 5).any() and np.isfinite(veh["exit_t"][row[k]])


US101_DRIVER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cf_stability.corridor import loop  # libsumo before pandas and torch
    import numpy as np
    from cf_stability.corridor.laws import load_law

    libsumo = loop.libsumo
    scenario = loop.Scenario.load(sys.argv[2])
    law, _ = load_law(sys.argv[3])
    rows = {int(v): str(i) for i, v in enumerate(scenario.demand["vehicle_id"])}
    held = {rows[24]: "offramp_on_lane5", rows[25]: "main_on_aux"}
    geometry = scenario.meta["geometry"]
    x0 = {e["id"]: e["x0"] for e in geometry["edges"]}
    log = {name: [] for name in held.values()}
    roads = {}

    def observer(t, ids, states, acc, histories, speeds):
        for vid, (s, dv, v) in zip(ids, states):
            road = libsumo.vehicle.getRoadID(vid)
            roads.setdefault(vid, set()).add(road)
            if vid in held:
                if not log[held[vid]]:
                    libsumo.vehicle.setLaneChangeMode(vid, 0)  # no lane change at all: it stays where it was inserted
                x = x0[road] + libsumo.vehicle.getLanePosition(vid)
                lead = libsumo.vehicle.getLeader(vid, 100.0)
                log[held[vid]].append([t, road, libsumo.vehicle.getLaneIndex(vid), x, s, v, bool(lead and lead[0])])

    result = loop.run_simulation(scenario, law, 0, loop.SimConfig(tail_s=60.0), observer=observer)
    offramp = sorted(int(v) for v, r in roads.items() if "offramp" in r)
    print("REPORT " + json.dumps({"log": log, "stats": result["stats"], "offramp": offramp, "rows": rows}))
    """
)


def test_us101_end_of_the_auxiliary_lane(us101):
    """A vehicle whose lane does not lead on along its route sees a standing obstacle: an off-ramp vehicle held on
    lane 5 at the end 30 m before the end of the auxiliary lane (412 m), a main-line vehicle held on the auxiliary
    lane at its end; the off-ramp vehicles on the auxiliary lane drive on to the off-ramp edge."""
    driver = us101 / "driver.py"
    driver.write_text(US101_DRIVER, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(driver), str(REPO_ROOT), str(us101 / "scenarios" / "us_p0"),
         str(us101 / "laws" / "stable.json")],
        cwd=us101, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("REPORT "))[7:])
    for name, lane_index, end in (("offramp_on_lane5", 1, 382.0), ("main_on_aux", 0, 412.0)):
        log = np.array([row[2:] for row in report["log"][name] if row[1] == "main_b"], dtype=float)
        lane, x, s, v, has_leader = log.T
        assert (lane == lane_index).all(), name
        assert end - 2.5 < x[-1] < end - 1.5 and v[-1] < 0.01, (name, x[-1], v[-1])  # IDM s0 = 2 m before the end
        near = (x > end - 50.0) & (has_leader == 0)  # the end within the leader range and no vehicle ahead
        assert near.sum() > 50 and np.allclose(s[near], end - x[near], atol=1e-6), name
    rows = report["rows"]
    assert report["offramp"] == sorted(int(rows[str(k)]) for k in (22, 23))  # 24 is held on lane 5
    stats = report["stats"]
    assert stats["n_offramp"] == 2 and stats["n_teleports"] == 0 and stats["n_collisions"] == 0
