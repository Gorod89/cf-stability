"""Scenario of the corridor and law export (cf_stability/corridor/scenario.py, laws.py; docs/m5_contract.md,
sections 2-4; docs/m7_contract.md, sections 4-7: US-101, variants, the laws of D111 and D114). No libsumo in this
process: netconvert runs as a child process."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cf_stability.corridor.laws import export_laws, law_device, law_fingerprint
from cf_stability.corridor.scenario import (
    BoundaryConfig,
    Corridor,
    boundary_speeds,
    build_demand,
    build_network,
    build_scenario,
    exit_difference,
    expected_connections,
    read_tracks,
    scenario_config,
    scenario_name,
    truth_arrays,
)
from cf_stability.models import IDM, save_model
from cf_stability.utils import REPO_ROOT, config_hash, read_json, resolve_path, write_json

I80 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "i80.yaml").read_text(encoding="utf-8"))
CORRIDOR = Corridor.from_mapping(I80)
DATA = resolve_path(I80["source"])
US101 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "us101.yaml").read_text(encoding="utf-8"))
US = Corridor.from_mapping(US101)
DATA_US = resolve_path(US101["source"])
LAWS = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "laws.yaml").read_text(encoding="utf-8"))
I80_HASHES = {0: "2111a8388b2e", 1: "e6c408fbeff0", 2: "8e825de61fa3"}  # the scenarios of the 390 runs of M5


def track(track_id, vehicle_id, frame0, x0, v, lanes, period=0, length=4.5, v_class=2, x_end=520.0):
    """A synthetic track at constant speed from ``x0`` until ``x_end``; ``lanes``: [(x from which, lane), ...]."""
    n = int(np.ceil((x_end - x0) / (v * 0.1))) + 1
    x = x0 + v * 0.1 * np.arange(n)
    lane = np.full(n, lanes[0][1])
    for x_from, value in lanes[1:]:
        lane[x >= x_from] = value
    return pd.DataFrame(
        {
            "track_id": track_id, "period": period, "vehicle_id": vehicle_id, "frame_id": frame0 + np.arange(n),
            "lane_id": lane, "x": x, "v": v, "length": length, "v_class": v_class,
        }
    )  # fmt: skip


def synthetic_tracks() -> pd.DataFrame:
    """Lanes 1-6 with three vehicles each (entering upstream of x_in, at x_in, and downstream), two on the
    on-ramp (lane 7, merging into lane 6), one truck and one motorcycle; period 1 has one vehicle."""
    parts, vid = [], 0
    for lane in range(1, 7):
        for k, x0 in enumerate((15.0, 20.0, 35.0)):
            vid += 1
            parts.append(track(f"p0_{vid}", vid, 10 + 40 * k + 5 * lane, x0, 12.0, [(0, lane)]))
    for k in range(2):
        vid += 1
        parts.append(track(f"p0_{vid}", vid, 30 + 60 * k, 90.0, 8.0, [(0, 7), (150.0, 6)]))
    vid += 1
    parts.append(track(f"p0_{vid}", vid, 200, 21.0, 10.0, [(0, 3)], length=14.0, v_class=3))
    vid += 1
    parts.append(track(f"p0_{vid}", vid, 210, 21.0, 11.0, [(0, 4)], length=2.2, v_class=1))
    parts.append(track("p1_1", 1, 5, 20.5, 12.0, [(0, 2)], period=1))
    return pd.concat(parts, ignore_index=True)


def test_lane_mapping_and_network_rules():
    main_a, main_b, main_c, out = CORRIDOR.edges
    assert [main_a.index(lane) for lane in (1, 6)] == [5, 0]
    assert [main_b.index(lane) for lane in (1, 6, 7)] == [6, 1, 0]
    assert CORRIDOR.aux_edge.id == "main_b" and CORRIDOR.aux_edge.x1 == 204.0 and CORRIDOR.main_lanes == 6
    assert [e.id for e in CORRIDOR.controlled] == ["main_a", "main_b", "main_c"]
    assert [e.id for e in CORRIDOR.buffer] == ["out"]
    connections = expected_connections(CORRIDOR)
    assert ("main_a", 0, "main_b", 1) in connections and ("main_b", 1, "main_c", 0) in connections
    assert not [c for c in connections if c[0] == "main_b" and c[1] == 0]  # the auxiliary lane has no successor
    assert len(connections) == 18
    x = np.array([19.9, 20.0, 99.0, 100.0, 204.0, 205.0, 500.0, 500.1])
    assert CORRIDOR.on_network(x, np.full(8, 7)).tolist() == [False, False, False, True, True, False, False, False]
    assert CORRIDOR.on_network(x, np.full(8, 6)).tolist() == [False, True, True, True, True, True, True, False]
    assert CORRIDOR.locate(100.0, 3)[0].id == "main_b" and CORRIDOR.locate(204.0, 7) == (main_b, 104.0 - 1e-3)


def test_demand_and_truth():
    tracks = synthetic_tracks()
    period0 = tracks[tracks.period == 0]
    demand = build_demand(period0, CORRIDOR, int(period0.frame_id.min()))
    assert len(demand) == period0.track_id.nunique() == 22
    assert demand.entry_lane.value_counts().sort_index().to_dict() == {1: 3, 2: 3, 3: 4, 4: 4, 5: 3, 6: 3, 7: 2}
    assert (np.diff(demand.depart_planned) >= 0).all()
    # upstream of x_in the first sample at or after x_in, on the ramp the first sample on the auxiliary lane
    upstream = demand[demand.track_id == "p0_1"].iloc[0]
    assert 20.0 <= upstream.entry_x < 21.2 and upstream.edge == "main_a" and upstream["index"] == 5
    ramp = demand[demand.entry_lane == 7]
    assert (ramp.edge == "main_b").all() and (ramp["index"] == 0).all() and (ramp.entry_x >= 100.0).all()
    assert np.allclose(ramp.pos, ramp.entry_x - 100.0)
    inside = demand[demand.track_id == "p0_3"].iloc[0]  # lane 1, starts at 35 m
    assert inside.entry_x == 35.0 and inside.pos == 15.0 and inside.depart_planned == pytest.approx(8.0)
    assert sorted(demand.track_id[demand.hov]) == ["p0_1", "p0_2", "p0_3"]  # the tracks that use the HOV lane 1

    origin = int(period0.frame_id.min())
    trajectories, vehicles, extrapolated = truth_arrays(period0, demand, CORRIDOR, origin, 1.0)
    assert extrapolated == 0  # the synthetic tracks run on to 520 m
    assert set(trajectories) == {"t", "vehicle", "x", "lane", "v"} and trajectories["lane"].dtype == np.int8
    assert np.allclose(trajectories["t"], np.round(trajectories["t"]))
    assert trajectories["x"].min() >= 20.0 and trajectories["x"].max() <= 500.0
    assert trajectories["x"][trajectories["lane"] == 7].min() >= 100.0  # the ramp upstream of the network is left out
    row = int(np.flatnonzero(demand.track_id == "p0_2")[0])  # lane 1, from 20 m at 12 m/s, first frame 55
    assert vehicles["exit_t"][row] == pytest.approx((55 - origin) / 10 + 480.0 / 12.0, abs=1e-6)
    assert vehicles["exit_lane"][row] == 1 and vehicles["member"][row] == -1 and not vehicles["n_collisions"].any()
    assert np.array_equal(vehicles["depart"], vehicles["depart_planned"])
    assert vehicles["vehicle_id"].dtype == np.int32 and vehicles["member"].dtype == np.int8

    # a track that ends less than 1 m before x_out (the end of the camera's view) passes it at its last speed
    ramp_row = int(np.flatnonzero(demand.track_id == "p0_19")[0])  # 8 m/s: samples 0.8 m apart, lane 6 at the end
    cut = period0[(period0.track_id != "p0_19") | (period0.x <= 499.7)]
    last = cut[cut.track_id == "p0_19"].iloc[-1]
    assert 499.0 < last.x < 500.0
    _, extrapolated_vehicles, n = truth_arrays(cut, demand, CORRIDOR, origin, 1.0)
    assert n == 1 and extrapolated_vehicles["exit_lane"][ramp_row] == 6
    exit_t = (last.frame_id - origin) / 10 + (500.0 - last.x) / 8.0
    assert extrapolated_vehicles["exit_t"][ramp_row] == pytest.approx(exit_t)
    _, strict, n = truth_arrays(cut, demand, CORRIDOR, origin, 0.0)
    assert n == 0 and np.isnan(strict["exit_t"][ramp_row]) and strict["exit_lane"][ramp_row] == -1
    far = period0[(period0.track_id != "p0_19") | (period0.x <= 498.0)]
    assert truth_arrays(far, demand, CORRIDOR, origin, 1.0)[2] == 0
    standing = cut.copy()
    standing.loc[last.name, "v"] = 0.05  # a standing vehicle has no passage time
    _, stopped, n = truth_arrays(standing, demand, CORRIDOR, origin, 1.0)
    assert n == 0 and np.isnan(stopped["exit_t"][ramp_row])


def test_boundary_speeds_by_hand():
    frames = np.array([0, 10, 20, 26, 30, 100])  # seconds 0, 1, 2, 2.6, 3, 10 from the origin 0
    df = pd.DataFrame({"frame_id": frames, "lane_id": [2, 2, 2, 2, 3, 2], "x": [480, 490, 475, 469, 500, 471],
                       "v": [4.0, 6.0, 8.0, 100.0, 7.0, 1.0]})  # fmt: skip
    t, speed = boundary_speeds(df, 0, 14, [1, 2, 3], (470.0, 500.0), 2.5, fallback=29.0)
    assert t.tolist() == list(range(14)) and speed.shape == (3, 14)
    # lane 2 (x = 469 is outside): second 0 sees samples at 0, 1, 2 s; second 3 sees 1, 2 s; 4 sees 2 s;
    # 5..7 none (last value); 8..12 see the sample at 10 s; 13 none
    expected = [6.0, 6.0, 6.0, 7.0, 8.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert np.allclose(speed[1], expected)
    assert np.allclose(speed[2], 7.0)  # one sample at 3 s: seconds 1..5 and, by the fill rules, all others
    assert np.allclose(speed[0], 29.0)  # a lane without samples


def test_exit_difference_and_gain():
    t, exits = np.arange(5.0), np.array([0.0, 0.0, 2.0, 4.0, 4.0])
    assert exit_difference(1.5, 3, t, exits) == pytest.approx(2.0)  # data 1.0 at 1.5 s (linear)
    # a vehicle removed from the simulation leaves the data's count at its data passage (2.5 s)
    assert exit_difference(2.0, 1, t, exits, [2.5]) == pytest.approx(-1.0)
    assert exit_difference(3.0, 1, t, exits, [2.5]) == pytest.approx(-2.0)
    config = BoundaryConfig.from_mapping({**I80["boundary"]})
    assert config.feedback and (config.max_decel, config.max_accel, config.min_gap) == (3.0, 2.0, 1.0)
    assert config.gain_of(10.0) == pytest.approx(0.8) and config.gain_of(100.0) == 0.2 and config.gain_of(-100.0) == 1.5
    assert BoundaryConfig(feedback=False).gain_of(50.0) == 1.0


def test_network_and_routes_read_back(tmp_path):
    tracks = synthetic_tracks()
    meta = build_scenario(tracks, {**I80, "name": "syn"}, 0, tmp_path / "syn_p0")
    net = build_network(CORRIDOR, tmp_path / "net_only", I80["netconvert_options"])  # netconvert, child process
    assert net["internal"] == [] and net["connections"] == expected_connections(CORRIDOR)
    assert {edge: [round(length, 2) for _, length, _ in lanes] for edge, lanes in net["lanes"].items()} == {
        "main_a": [80.0] * 6, "main_b": [104.0] * 7, "main_c": [296.0] * 6, "out": [150.0] * 6,
    }  # fmt: skip
    # NGSIM lane 1 of the section (SUMO index 5, 6 on main_b) is reserved to hov; the buffer is open to all
    assert net["permissions"] == {("main_a", 5): "hov", ("main_b", 6): "hov", ("main_c", 5): "hov"}

    directory = tmp_path / "syn_p0"
    assert sorted(meta["files"]) == sorted(
        ["boundary.npz", "ground_truth.npz", "net.con.xml", "net.edg.xml", "net.net.xml", "net.nod.xml",
         "routes.rou.xml", "vehicles_truth.npz"]  # fmt: skip
    )
    assert meta["scenario"] == "syn_p0" and meta["counts"]["vehicles"] == 22 and meta["frame_origin"] == 15
    assert meta["analysis_window"] == [180.0, 840.0] and meta["geometry"]["x_in"] == 20.0
    assert meta["counts"]["exited"] == 22 and meta["counts"]["exits_extrapolated"] == 0 and meta["counts"]["hov"] == 3
    assert read_json(directory / "scenario.json")["config_hash"] == meta["config_hash"]
    routes = ET.parse(directory / "routes.rou.xml").getroot()
    types = {t.get("id"): t for t in routes.findall("vType")}
    for vtype in types.values():
        keys = ("minGap", "sigma", "lcKeepRight", "laneChangeModel", "tau", "decel", "emergencyDecel")
        assert tuple(vtype.get(key) for key in keys) == ("0", "0", "0", "LC2013", "1", "8", "8")
    assert {t.get("vClass") for t in types.values()} == {"passenger", "truck", "motorcycle", "hov"}
    vehicles = routes.findall("vehicle")
    with np.load(directory / "vehicles_truth.npz") as data:
        truth = {key: data[key] for key in data.files}
    assert [v.get("id") for v in vehicles] == [str(i) for i in range(22)]
    departs = np.array([float(v.get("depart")) for v in vehicles])
    assert np.allclose(departs, truth["depart_planned"]) and (np.diff(departs) >= 0).all()
    for v, lane, x, length in zip(vehicles, truth["entry_lane"], truth["entry_x"], truth["length"]):
        edge = CORRIDOR.locate(float(x), int(lane))[0]
        assert v.get("route") == f"from_{edge.id}" and int(v.get("departLane")) == edge.index(int(lane))
        assert v.get("insertionChecks") is None  # SUMO's default checks
        assert float(v.get("departPos")) == pytest.approx(x - edge.x0, abs=1e-4)
        assert float(types[v.get("type")].get("length")) == pytest.approx(length, abs=1e-3)
        assert (types[v.get("type")].get("vClass") == "hov") == (lane == 1)  # the users of lane 1 (never changed)
    route_edges = {r.get("id"): r.get("edges") for r in routes.findall("route")}
    assert route_edges["from_main_a"] == "main_a main_b main_c out"
    assert route_edges["from_main_b"] == "main_b main_c out"
    with np.load(directory / "boundary.npz") as data:
        boundary = {key: data[key] for key in data.files}
    assert set(boundary) == {"t", "speed", "exits"} and boundary["speed"].shape == (6, meta["boundary"]["n_seconds"])
    assert np.allclose(boundary["speed"][:, 30:40], 12.0)
    # cumulative passages of x_out of the data by every second
    exit_t = truth["exit_t"][np.isfinite(truth["exit_t"])]
    assert np.array_equal(boundary["exits"], [(exit_t <= t).sum() for t in boundary["t"]])
    assert boundary["exits"][-1] == 22 and boundary["t"][-1] >= exit_t.max()
    assert meta["boundary"]["feedback"] is True and meta["boundary"]["gain"] == 0.02


@pytest.mark.data
def test_counts_per_period_and_lane_against_the_data():
    if not DATA.exists():
        pytest.skip(f"{DATA} missing")
    tracks = read_tracks(DATA)
    for period, group in tracks.groupby("period"):
        demand = build_demand(group, CORRIDOR, int(group.frame_id.min()))
        # independent count: lane of the first sample of every track that lies on the network
        x, lane = group.x.to_numpy(), group.lane_id.to_numpy()
        ramp = (lane == 7) & (x >= 100.0) & (x <= 204.0)
        main = (lane <= 6) & (x >= 20.0) & (x <= 500.0)
        first = group[ramp | main].sort_values("frame_id").groupby("track_id").head(1)
        assert len(demand) == group.track_id.nunique() == len(first)
        assert demand.entry_lane.value_counts().to_dict() == first.lane_id.value_counts().to_dict(), period
    assert tracks.groupby("period").track_id.nunique().to_dict() == {0: 2052, 1: 1836, 2: 1790}


def write_member(run: Path, params: dict, box_v: tuple) -> None:
    run.mkdir(parents=True)
    save_model(IDM(params), run / "model.pt")
    write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, box_v[0]], "box_high": [50.0, 3.0, box_v[1]]}})


def test_law_export(tmp_path):
    params = {"v0": 26.0, "T": 1.5, "s0": 1.7, "a": 0.9, "b": 0.5}
    runs = tmp_path / "runs"
    write_member(runs / "idm" / "fold0", params, (0.0, 18.0))
    write_member(runs / "idm" / "fold1", {**params, "T": 1.2}, (0.5, 19.0))
    write_member(runs / "idm" / "fold2", {**params, "a": 1.1}, (0.2, 17.0))
    write_member(runs / "partial" / "fold0", params, (0.0, 18.0))
    write_member(runs / "partial" / "fold2", params, (0.0, 18.0))
    estimates = pd.DataFrame({
        "v0": 26.0, "T": [1.0, 0.3, 2.0], "s0": [2.0, 3.0, 4.0], "a": [1.0, 1.5, 0.5], "b": 0.5,
        "at_bound_v0": False, "at_bound_T": [False, True, False], "at_bound_s0": False, "at_bound_a": False,
        "at_bound_b": False,
    })  # fmt: skip
    estimates.to_parquet(tmp_path / "per_event.parquet")
    write_json(tmp_path / "spread.json", {"fixed": {"v0": 26.0, "b": 0.5}})
    members = (runs / "idm" / "fold{fold}").as_posix()
    het = {"kind": "idm_heterogeneous", "estimates": (tmp_path / "per_event.parquet").as_posix(),
           "spread": (tmp_path / "spread.json").as_posix(), "support_from": "idm_global"}  # fmt: skip
    table = {
        "folds": [0, 1, 2],
        "table": {
            "idm_global": {"kind": "idm", "device": "cpu", "members": members},
            "as_models": {"kind": "models", "device": "cuda", "members": members},
            "missing": {"kind": "models", "members": (runs / "none" / "fold{fold}").as_posix()},
            "partial": {"kind": "models", "members": (runs / "partial" / "fold{fold}").as_posix()},
            "het": het,
            "het_all": {**het, "selection": "all"},
        },
    }  # fmt: skip
    out = tmp_path / "laws"
    lines = export_laws(table, out)
    assert len(lines) == 6 and "3/3 members" in lines[0]
    # a law is written only with all its member runs (a law of fewer members would be another law)
    assert "not written: 3 of 3 member runs missing" in lines[2] and not (out / "missing.json").exists()
    assert "not written: 1 of 3 member runs missing" in lines[3] and "partial/fold1" in lines[3]
    assert "partial/fold0" not in lines[3] and not (out / "partial.json").exists()
    law = read_json(out / "idm_global.json")
    assert law["kind"] == "idm" and law["window"] == 1 and law["support_v"] == [0.0, 19.0] and law["device"] == "cpu"
    assert [m["run"] for m in law["members"]] == [members.format(fold=k) for k in range(3)]
    assert law["members"][0]["model"] == members.format(fold=0) + "/model.pt"
    assert law["params"][1]["T"] == pytest.approx(1.2) and law["params"][0]["v0"] == pytest.approx(26.0)
    as_models = read_json(out / "as_models.json")
    assert "params" not in as_models and as_models["device"] == "cuda"
    interior = read_json(out / "het.json")
    assert interior["kind"] == "idm_heterogeneous" and interior["members"] == [] and interior["table"] == "het.npz"
    assert interior["device"] == "cpu" and "selection" not in interior["source"]  # the M5 law file keeps its content
    assert law_device(law) == "cpu" and law_device(as_models) == "cuda" and law_device(as_models, "cpu") == "cpu"
    with pytest.raises(ValueError, match="CPU"):  # the IDM laws are computed on the CPU
        export_laws({"folds": [0], "table": {"idm_gpu": {"kind": "idm", "device": "cuda", "members": members}}}, out)
    with pytest.raises(ValueError, match="device"):
        law_device(as_models, "auto")
    with pytest.raises(ValueError, match="selection"):
        export_laws({"folds": [0], "table": {"het_x": {**het, "selection": "bounds", "support_from": None}}}, out)
    assert interior["support_v"] == [0.0, 19.0] and interior["source"]["n_interior"] == 2
    with np.load(out / "het.npz") as data:
        assert np.allclose(data["T"], [1.0, 2.0]) and float(data["v0"]) == 26.0 and float(data["b"]) == 0.5
    # D114: all estimates, the bounds-hitting one included, with the same fixed v0 and b and the same support
    every = read_json(out / "het_all.json")
    assert every["kind"] == "idm_heterogeneous" and every["table"] == "het_all.npz"
    assert every["support_v"] == [0.0, 19.0] and every["source"]["selection"] == "all"
    assert every["source"]["n_used"] == 3 and every["source"]["n_interior"] == 2
    assert "all 3 per-event estimates (1 on a bound, 2 interior)" in lines[5]
    with np.load(out / "het_all.npz") as data:
        assert np.allclose(data["T"], [1.0, 0.3, 2.0]) and np.allclose(data["a"], [1.0, 1.5, 0.5])
        assert float(data["v0"]) == 26.0 and float(data["b"]) == 0.5
    # the fingerprint follows the bytes of the member checkpoints
    before = law_fingerprint(out / "idm_global.json")
    assert law_fingerprint(out / "idm_global.json") == before
    save_model(IDM({**params, "a": 1.3}), runs / "idm" / "fold0" / "model.pt")
    assert law_fingerprint(out / "idm_global.json") != before


def test_laws_of_m7_in_the_table():
    """D111: the residual-amplitude sweep and its control are read from the fine-tuned runs of five folds, as
    residual_idm_certified; D114: the heterogeneous IDM of all estimates differs from the M5 law in the selection."""
    table = LAWS["table"]
    expected = {
        "residual_idm_certified_r0.1": "runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold{fold}_seed0",
        "residual_idm_certified_r0.2": "runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold{fold}_seed0",
        "residual_idm_certified_r0.5": "runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold{fold}_seed0",
        "residual_idm_free_r0.3": "runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold{fold}_seed0",
    }
    for name, members in expected.items():
        assert table[name] == {"kind": "models", "device": "cpu", "members": members}, name
    base = table["residual_idm_certified"]
    assert base["members"] == "runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold{fold}_seed0"
    assert LAWS["folds"] == [0, 1, 2, 3, 4]
    every, interior = table["idm_heterogeneous_all"], table["idm_heterogeneous"]
    assert every["selection"] == "all" and "selection" not in interior
    assert {k: v for k, v in every.items() if k != "selection"} == interior


def test_rmax_laws_from_a_run_tree(tmp_path):
    """The law files of D111 from a synthetic tree of member runs: all five folds give the law file, a missing
    fold leaves the law out with the missing run named (and an earlier file in place)."""
    names = ["residual_idm_certified_r0.1", "residual_idm_certified_r0.2", "residual_idm_certified_r0.5",
             "residual_idm_free_r0.3"]  # fmt: skip
    table = {name: {**LAWS["table"][name], "members": (tmp_path / LAWS["table"][name]["members"]).as_posix()}
             for name in names}  # fmt: skip
    for spec in table.values():
        for fold in LAWS["folds"]:
            write_member(Path(spec["members"].format(fold=fold)), {"v0": 26.0, "T": 1.5, "s0": 1.7, "a": 0.9, "b": 0.5},
                         (0.0, 18.0))  # fmt: skip
    out = tmp_path / "laws"
    lines = export_laws({"folds": LAWS["folds"], "table": table}, out)
    assert all("5/5 members" in line for line in lines), lines
    law = read_json(out / "residual_idm_free_r0.3.json")
    assert law["kind"] == "models" and law["device"] == "cpu" and len(law["members"]) == 5
    assert law["members"][4]["run"].endswith("e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0")
    stamp = (out / "residual_idm_certified_r0.2.json").stat().st_mtime_ns
    gone = Path(table["residual_idm_certified_r0.2"]["members"].format(fold=3))
    (gone / "model.pt").unlink()
    lines = export_laws({"folds": LAWS["folds"], "table": table}, out, only=["residual_idm_certified_r0.2"])
    assert len(lines) == 1 and "not written: 1 of 5 member runs missing" in lines[0]
    assert "driver_fold3_seed0" in lines[0]
    assert (out / "residual_idm_certified_r0.2.json").stat().st_mtime_ns == stamp


def test_scenario_hashes_and_variants():
    """The hashes of the I-80 scenarios of M5 (the scenario hash of the 390 runs) do not change; a variant adds its
    name to the hashed config; a site's metrics settings are not hashed."""
    for period, expected in I80_HASHES.items():
        assert config_hash(scenario_config(I80, period)) == expected
        assert "variant" not in scenario_config(I80, period)
    variant = scenario_config({**I80, "boundary": {**I80["boundary"], "gain": 0.04}}, 1, "gain0.04")
    assert variant["variant"] == "gain0.04" and config_hash(variant) not in I80_HASHES.values()
    assert scenario_name("i80", 1) == "i80_p1" and scenario_name("i80", 1, "lc_low") == "i80_p1_lc_low"
    assert "metrics" in US101 and "metrics" not in scenario_config(US101, 1)
    other = scenario_config({**US101, "metrics": {"detectors": [1]}}, 1)
    assert config_hash(other) == config_hash(scenario_config(US101, 1))


def run_build(tmp_path: Path, *overrides: str) -> subprocess.CompletedProcess:
    args = [sys.executable, str(REPO_ROOT / "scripts" / "build_corridor.py"), *overrides,
            f"hydra.run.dir='{(tmp_path / 'outputs').as_posix()}'"]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)


def test_build_script_stops_for_i24(tmp_path):
    proc = run_build(tmp_path, "corridor.i24.enabled=true", "laws=false",
                     f"paths.scenarios_root='{(tmp_path / 'scenarios').as_posix()}'")  # fmt: skip
    assert proc.returncode != 0 and "MOTION data are not on this machine" in proc.stdout + proc.stderr
    assert not (tmp_path / "scenarios").exists()


def test_build_script_scenario(tmp_path):
    source = tmp_path / "tracks.parquet"
    synthetic_tracks().to_parquet(source)
    common = (
        f"corridor.i80.source='{source.as_posix()}'", "corridor.i80.name=syn", "laws=false", "sites=[i80]",
        f"paths.scenarios_root='{(tmp_path / 'scenarios').as_posix()}'",
    )  # fmt: skip
    proc = run_build(tmp_path, *common, "periods=[0,1]")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("SCENARIO syn_")]
    assert len(lines) == 2 and "22 vehicles" in lines[0] and "1 vehicles" in lines[1], proc.stdout
    meta = read_json(tmp_path / "scenarios" / "syn_p0" / "scenario.json")
    assert meta["counts"]["entry_lane"] == {"1": 3, "2": 3, "3": 4, "4": 4, "5": 3, "6": 3, "7": 2}
    assert "variant" not in meta and "metrics" not in meta
    again = run_build(tmp_path, *common, "periods=[0]")
    assert again.returncode == 0 and "SCENARIO syn_p0: current" in again.stdout, again.stdout + again.stderr
    assert json.loads((tmp_path / "scenarios" / "syn_p0" / "scenario.json").read_text(encoding="utf-8")) == meta

    # variants (D113): the name and the overrides of the site on the command line, written to scenario.json
    proc = run_build(tmp_path, *common, "periods=[0]", "variant=gain0.04", "corridor.i80.boundary.gain=0.04")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    variant = read_json(tmp_path / "scenarios" / "syn_p0_gain0.04" / "scenario.json")
    assert variant["scenario"] == "syn_p0_gain0.04" and variant["variant"] == "gain0.04"
    assert variant["overrides"]["boundary.gain"] == 0.04 and variant["boundary"]["gain"] == 0.04
    assert variant["config"]["variant"] == "gain0.04" and variant["config_hash"] != meta["config_hash"]
    assert variant["counts"] == meta["counts"]  # the same period of the same data
    for name in ("ground_truth.npz", "vehicles_truth.npz"):
        scenarios = tmp_path / "scenarios"
        with np.load(scenarios / "syn_p0" / name) as a, np.load(scenarios / "syn_p0_gain0.04" / name) as b:
            assert all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files)
    lc = ("variant=lc_low", "+corridor.i80.vtype.lcSpeedGain=0.5", "+corridor.i80.vtype.lcAssertive=0.5")
    proc = run_build(tmp_path, *common, "periods=[0]", *lc)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    low = read_json(tmp_path / "scenarios" / "syn_p0_lc_low" / "scenario.json")
    assert low["overrides"]["vtype.lcSpeedGain"] == 0.5 and low["overrides"]["vtype.lcAssertive"] == 0.5
    routes = ET.parse(tmp_path / "scenarios" / "syn_p0_lc_low" / "routes.rou.xml").getroot()
    assert {(t.get("lcSpeedGain"), t.get("lcAssertive")) for t in routes.findall("vType")} == {("0.5", "0.5")}
    again = run_build(tmp_path, *common, "periods=[0]", *lc)
    assert again.returncode == 0 and "SCENARIO syn_p0_lc_low: current" in again.stdout, again.stdout + again.stderr
    bare = run_build(tmp_path, *common[2:], "periods=[0]", "variant=nothing")  # no override of a site: no variant
    assert bare.returncode != 0 and "a variant needs overrides" in bare.stdout + bare.stderr
    assert not (tmp_path / "scenarios" / "i80_p0_nothing").exists()


def us101_tracks() -> pd.DataFrame:
    """Synthetic tracks on the US-101 geometry (x_in 25, x_out 640, auxiliary lane 110-412 m, off-ramp from 412 m):
    two vehicles per through lane; an on-ramp vehicle (lane 7 from 150 m, 6 from 190 m, 5 from 260 m); two off-ramp
    vehicles (5 -> 6 -> 8, and 5 -> 7 -> 6 -> 8; their tracks end on lane 8 at 450-460 m); a through vehicle that
    uses the auxiliary lane (5 -> 6 -> 5); an on-ramp vehicle that returns from lane 8 to lane 5 at 430 m."""
    parts, vid = [], 0
    for lane in range(1, 6):
        for k in range(2):
            vid += 1
            parts.append(track(f"p0_{vid}", vid, 10 + 50 * k + 5 * lane, 10.0, 12.0, [(0, lane)], x_end=660.0))
    parts += [
        track("p0_20", 20, 30, 150.0, 10.0, [(0, 7), (190.0, 6), (260.0, 5)], x_end=660.0),
        track("p0_21", 21, 40, 10.0, 12.0, [(0, 5), (250.0, 6), (405.0, 8)], x_end=450.0),
        track("p0_22", 22, 60, 10.0, 12.0, [(0, 5), (130.0, 7), (190.0, 6), (405.0, 8)], x_end=460.0),
        track("p0_23", 23, 80, 10.0, 12.0, [(0, 5), (300.0, 6), (350.0, 5)], x_end=660.0),
        track("p0_24", 24, 90, 150.0, 10.0, [(0, 7), (190.0, 6), (405.0, 8), (430.0, 5)], x_end=660.0),
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def test_us101_geometry():
    """US-101 (D112): main line of 5 lanes, the auxiliary lane 6 on main_b (data lanes 7 and 8 are modelled as it
    there), an off-ramp edge from the end of the auxiliary lane, the auxiliary lane reserved to its users."""
    main_a, main_b, main_c, out = US.edges
    assert [e.id for e in US.edges] == ["main_a", "main_b", "main_c", "out"] and US.main_lanes == 5
    assert (main_a.x0, main_b.x0, main_b.x1, main_c.x1, out.x1) == (25.0, 110.0, 412.0, 640.0, 790.0)
    assert US.aux_edge is main_b and main_b.index(6) == 0 and main_b.index(1) == 5 and main_c.index(5) == 0
    assert US.offramp == type(main_a)("offramp", 412.0, 492.0, 1)
    assert US.offramp_lane == 8 and US.offramp_early_end == 30.0
    assert [e.id for e in US.section] == ["main_a", "main_b", "main_c"]
    assert [e.id for e in US.controlled] == ["main_a", "main_b", "main_c", "offramp"]
    assert [e.id for e in US.buffer] == ["out"]
    assert US.network_lane(7) == 6 and US.network_lane(8) == 6 and US.network_lane(5) == 5
    assert US.network_lane(np.array([5, 7, 8, 6])).tolist() == [5, 6, 6, 6]
    x = np.array([100.0, 120.0, 405.0, 420.0, 300.0, 30.0, 20.0, 650.0])
    lane = np.array([7, 7, 8, 8, 6, 5, 5, 5])
    assert US.on_network(x, lane).tolist() == [False, True, True, False, True, True, False, False]
    assert US.locate(150.0, 7) == (main_b, 40.0) and US.locate(412.0, 8) == (main_b, 302.0 - 1e-3)
    connections = expected_connections(US)
    assert ("main_b", 0, "offramp", 0) in connections and len(connections) == 5 * 3 + 1
    assert not [c for c in connections if c[0] == "main_b" and c[1] == 0 and c[2] == "main_c"]
    assert US.permissions() == {("main_b", 0): "custom1"} and CORRIDOR.offramp is None
    geometry = US.sim_geometry()
    assert geometry["offramp_edge"] == "offramp" and geometry["offramp_lane"] == 8 and geometry["aux_end"] == 412.0
    assert geometry["section_edges"] == ["main_a", "main_b", "main_c"] and geometry["edges"][-1]["id"] == "offramp"
    assert "offramp_edge" not in CORRIDOR.sim_geometry()  # I-80 as before


def test_us101_demand_truth_routes_and_network(tmp_path):
    tracks = us101_tracks()
    origin = int(tracks.frame_id.min())
    demand = build_demand(tracks, US, origin)
    row = {t: i for i, t in enumerate(demand.track_id)}
    assert len(demand) == 15 and sorted(demand.track_id[demand.offramp]) == ["p0_21", "p0_22"]
    assert sorted(demand.track_id[demand.aux_user]) == ["p0_20", "p0_21", "p0_22", "p0_23", "p0_24"]
    ramp = demand.iloc[row["p0_20"]]  # the on-ramp vehicle enters on the auxiliary lane where it appears
    assert ramp.edge == "main_b" and ramp["index"] == 0 and ramp.entry_lane == 7 and ramp.pos == pytest.approx(40.0)
    assert ramp.entry_x == 150.0 and ramp.entry_v == 10.0
    assert demand.iloc[row["p0_1"]].edge == "main_a" and demand.iloc[row["p0_1"]].entry_x >= 25.0

    trajectories, vehicles, _ = truth_arrays(tracks, demand, US, origin, 1.0)
    vehicle, x, lane = trajectories["vehicle"], trajectories["x"], trajectories["lane"]
    for name in ("p0_21", "p0_22"):  # off the section beyond the end of the auxiliary lane, and no passage of x_out
        assert x[vehicle == row[name]].max() <= 412.0 and np.isnan(vehicles["exit_t"][row[name]])
    back = row["p0_24"]  # returns from lane 8 to lane 5: in the section all the time, passes x_out
    assert (lane[(vehicle == back) & (x > 412.0)] == 8).any() and np.isfinite(vehicles["exit_t"][back])
    assert np.isfinite(vehicles["exit_t"][[row[f"p0_{k}"] for k in range(1, 11)]]).all()

    meta = build_scenario(tracks, {**US101, "name": "syn"}, 0, tmp_path / "syn_p0")
    assert meta["counts"]["offramp"] == 2 and meta["counts"]["aux_users"] == 5 and meta["counts"]["exited"] == 13
    assert meta["metrics"] == US101["metrics"] and meta["analysis_window"] == [90.0, 870.0]
    assert meta["geometry"]["offramp_early_end"] == 30.0 and meta["lane_permissions"] == {"main_b_0": "custom1"}
    net = read_json(tmp_path / "syn_p0" / "scenario.json")["netconvert"]
    assert "Success" in net
    network = build_network(US, tmp_path / "net_only", US101["netconvert_options"])
    assert network["lanes"]["offramp"] == [(0, 80.0, 29.0)] and network["connections"] == expected_connections(US)
    assert network["permissions"] == {("main_b", 0): "custom1"}
    routes = ET.parse(tmp_path / "syn_p0" / "routes.rou.xml").getroot()
    route_edges = {r.get("id"): r.get("edges") for r in routes.findall("route")}
    assert route_edges["from_main_a_offramp"] == "main_a main_b offramp"
    assert route_edges["from_main_a"] == "main_a main_b main_c out"
    assert route_edges["from_main_b"] == "main_b main_c out"
    types = {t.get("id"): t for t in routes.findall("vType")}
    for v in routes.findall("vehicle"):
        track_id = demand.track_id[int(v.get("id"))]
        assert v.get("route").endswith("_offramp") == (track_id in ("p0_21", "p0_22")), track_id
        vtype = types[v.get("type")]
        weaving = track_id in ("p0_20", "p0_21", "p0_22", "p0_23", "p0_24")
        assert (vtype.get("vClass") == "custom1") == weaving and (vtype.get("lcAssertive") == "2.5") == weaving
    with np.load(tmp_path / "syn_p0" / "boundary.npz") as data:
        assert data["speed"].shape == (5, meta["boundary"]["n_seconds"]) and data["exits"][-1] == 13


@pytest.mark.data
def test_us101_counts_against_the_data():
    if not DATA_US.exists():
        pytest.skip(f"{DATA_US} missing")
    tracks = read_tracks(DATA_US)
    assert tracks.groupby("period").track_id.nunique().to_dict() == {0: 2169, 1: 2017, 2: 1915}
    totals = {"onramp": 0, "offramp": 0, "aux": 0}
    for period, group in tracks.groupby("period"):
        demand = build_demand(group, US, int(group.frame_id.min()))
        first = group.groupby("track_id").lane_id.first()
        last = group.groupby("track_id").lane_id.last()
        assert len(demand) == group.track_id.nunique(), period
        ramp = demand.entry_lane == 7
        assert ramp.sum() == (first == 7).sum() and (demand.edge[ramp] == "main_b").all()
        assert demand.offramp.sum() == (last == 8).sum() and not demand.offramp_unreachable.any()
        totals["onramp"] += int((demand.entry_lane == 7).sum())
        totals["offramp"] += int(demand.offramp.sum())
        totals["aux"] += int(demand.aux_user.sum())
    # 402 on-ramp and 226 off-ramp tracks (3 both), and 18 through tracks that use the auxiliary lane
    assert totals == {"onramp": 402, "offramp": 226, "aux": 643}
