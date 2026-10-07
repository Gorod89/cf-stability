"""Corridor metrics of docs/m5_contract.md, section 5, on hand-made trajectories (cf_stability/corridor/macro.py).

Uniform flow gives flow, density, speed, throughput and travel time in closed form; detectors are
checked against crossings counted by hand; boundary points, gaps, empty inputs and the macro error.
The generators here are shared with the other corridor tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cf_stability.corridor.macro import (
    COMPONENTS,
    Geometry,
    MacroConfig,
    corridor_metrics,
    crossings,
    detector,
    fd_error,
    fundamental_diagram,
    geometry_from_scenario,
    macro_error,
    macro_error_dynamic,
    prepare_trajectories,
    scenario_macro_config,
    section_edges,
    travel_times,
)
from cf_stability.utils import config_hash

GEOMETRY = Geometry(20.0, 500.0, (120.0, 840.0))


def sample(paths: list[dict], x_in: float = 20.0, x_out: float = 500.0) -> tuple[dict, dict]:
    """``trajectories`` and ``vehicles`` arrays of section 1 from vehicle paths.

    Every path has ``depart``, ``lane``, ``t`` and ``x`` (a fine, increasing trajectory from the insertion
    on); rows at the whole seconds with ``x_in <= x <= x_out``, ``exit_t`` where the path passes ``x_out``.
    """
    rows = {"t": [], "vehicle": [], "x": [], "lane": [], "v": []}
    n = len(paths)
    vehicles = {
        "vehicle_id": np.arange(n, dtype=np.int32), "depart_planned": np.zeros(n), "depart": np.zeros(n),
        "entry_x": np.zeros(n), "entry_lane": np.zeros(n, np.int8), "exit_t": np.full(n, np.nan),
        "exit_lane": np.zeros(n, np.int8), "length": np.full(n, 4.5), "v_class": np.full(n, 2, np.int8),
        "member": np.zeros(n, np.int8), "collided": np.zeros(n, bool), "teleported": np.zeros(n, bool),
    }  # fmt: skip
    for i, p in enumerate(paths):
        t, x = np.asarray(p["t"], float), np.asarray(p["x"], float)
        whole = np.arange(math.ceil(t[0] - 1e-9), math.floor(t[-1] + 1e-9) + 1, dtype=float)
        xs = np.interp(whole, t, x)
        vs = np.interp(whole, t[:-1], np.diff(x) / np.diff(t)) if "v" not in p else np.interp(whole, t, p["v"])
        keep = (xs >= x_in) & (xs <= x_out)
        for key, value in (("t", whole[keep]), ("x", xs[keep]), ("v", vs[keep])):
            rows[key].append(value)
        rows["vehicle"].append(np.full(keep.sum(), i))
        rows["lane"].append(np.full(keep.sum(), p["lane"]))
        vehicles["depart"][i] = vehicles["depart_planned"][i] = t[0]
        vehicles["entry_x"][i], vehicles["entry_lane"][i] = x[0], p["lane"]
        if x[-1] >= x_out > x[0]:
            vehicles["exit_t"][i] = np.interp(x_out, x, t)
            vehicles["exit_lane"][i] = p["lane"]
    trajectories = {
        "t": np.concatenate(rows["t"]).astype(np.float32), "vehicle": np.concatenate(rows["vehicle"]).astype(np.int32),
        "x": np.concatenate(rows["x"]).astype(np.float32), "lane": np.concatenate(rows["lane"]).astype(np.int8),
        "v": np.concatenate(rows["v"]).astype(np.float32),
    }  # fmt: skip
    return trajectories, vehicles


def constant_paths(speed: float, headway: float, lane: int, offset: float, t_end: float = 900.0) -> list[dict]:
    """Vehicles inserted at x = 20 every ``headway`` seconds, driving at ``speed`` to beyond x = 500."""
    paths = []
    for depart in np.arange(offset, t_end, headway):
        t = np.array([depart, depart + 500.0 / speed])
        paths.append({"t": t, "x": 20.0 + speed * (t - depart), "lane": lane})
    return paths


def uniform_flow() -> tuple[dict, dict]:
    """Lane 1: 10 m/s every 2 s (1800 veh/h, 50 veh/km); lane 2: 15 m/s every 3 s (1200 veh/h, 22.2 veh/km)."""
    return sample(constant_paths(10.0, 2.0, 1, 0.37) + constant_paths(15.0, 3.0, 2, 1.13))


def band_speed(x: np.ndarray, t: np.ndarray, waves: list[tuple[float, float]], free: float, slow: float,
               c: float = -5.0, x0: float = 500.0) -> np.ndarray:  # fmt: skip
    """Speed field of stop waves: ``slow`` inside bands that pass ``x0`` at ``t0`` and travel at ``c`` m/s for
    ``duration`` seconds at every x (``waves``: [(t0, duration)]), ``free`` elsewhere."""
    v = np.full(np.shape(x), free)
    for t0, duration in waves:
        lead = t0 + (x - x0) / c  # time at which the wave reaches x
        v = np.where((t >= lead) & (t < lead + duration), slow, v)
    return v


def wave_paths(waves: list[tuple[float, float]], lanes: int = 3, headway: float = 2.0, free: float = 10.0,
               slow: float = 1.0, t_end: float = 900.0, dt: float = 0.05) -> list[dict]:  # fmt: skip
    """Vehicles inserted at x = 20 every ``headway`` seconds per lane, integrated (Euler, ``dt``) through
    :func:`band_speed`; the paths keep the positions at 10 Hz."""
    per_lane = [np.arange(0.3 + k * headway / lanes, t_end, headway) for k in range(lanes)]
    departs = np.concatenate(per_lane)
    lane = np.concatenate([np.full(len(d), k + 1) for k, d in enumerate(per_lane)])
    every, steps = int(round(0.1 / dt)), int(round((t_end + 150.0) / dt))
    x = np.full(len(departs), 20.0)
    track = np.full((steps // every + 1, len(departs)), np.nan)
    for k in range(steps + 1):
        t = k * dt
        active = (departs <= t + 1e-9) & (x <= 505.0)
        if k % every == 0:
            track[k // every, active] = x[active]
        x = np.where(active, x + band_speed(x, t, waves, free, slow) * dt, x)
    paths = []
    for i in range(len(departs)):
        rows = np.flatnonzero(np.isfinite(track[:, i]))
        if len(rows) > 1:
            paths.append({"t": rows * 0.1, "x": track[rows, i], "lane": int(lane[i])})
    return paths


@pytest.fixture(scope="module")
def uniform():
    trajectories, vehicles = uniform_flow()
    return trajectories, vehicles, corridor_metrics(trajectories, vehicles, GEOMETRY, MacroConfig())


def test_uniform_flow_is_exact(uniform):
    _, _, m = uniform
    assert m["throughput_vph"] == pytest.approx(3000.0)
    assert m["throughput_vph_per_lane"] == pytest.approx({"1": 1800.0, "2": 1200.0})
    q, k = 1800.0 + 1200.0, 1000.0 * (1 / 20.0 + 1 / 45.0)  # veh/h, veh/km over both lanes
    assert m["mean_speed"] == pytest.approx(q / k / 3.6, rel=1e-9)
    total = m["edie"]["total"]
    np.testing.assert_allclose(total["flow_vph"], q, rtol=1e-6)
    np.testing.assert_allclose(total["density_vpkm"], k, rtol=1e-6)
    np.testing.assert_allclose(m["edie"]["per_lane"]["1"]["speed"], 10.0, rtol=1e-6)
    np.testing.assert_allclose(m["edie"]["per_lane"]["2"]["speed"], 15.0, rtol=1e-6)
    assert m["edie"]["x_edges"] == [20.0, 100.0, 200.0, 300.0, 400.0, 500.0] and len(m["edie"]["t_edges"]) == 25
    for x, det in m["detectors"].items():
        assert det["count"] == [15 + 10] * 24, x  # 15 and 10 vehicles per 30 s
        np.testing.assert_allclose(det["flow_vph"], 3000.0)
        np.testing.assert_allclose(det["speed"], (15 * 10.0 + 10 * 15.0) / 25, rtol=1e-6)  # time-mean speed
    # every vehicle that drives from 30 m to 490 m inside the window takes 46 s (lane 1) or 30.67 s (lane 2)
    travel = m["travel_time"]
    values = np.array(travel["values"])
    assert set(np.round(values, 3)) == {46.0, round(460 / 15, 3)}  # float32 samples
    assert travel["n"] == len(values) and travel["median"] == pytest.approx(46.0)  # lane 1 has more vehicles
    fd = m["fd"]
    assert fd["density_bins"] == [70.0] and fd["count"] == [5 * 24] and fd["flow"] == [pytest.approx(q)]
    assert m["fd_scatter"] == pytest.approx(0.0, abs=1e-6)
    assert m["waves"]["n_waves"] == 0 and m["waves"]["wave_speed"] is None and m["waves"]["wave_amplitude"] is None
    # the speed at 100 m (12 m/s) is above 11 m/s: no congested interval
    assert m["queue_discharge_flow"] is None and m["capacity_drop"] is None and m["n_congested_intervals"] == 0
    assert m["congested_share"] == 0.0 and m["flow_peak_2min"] == pytest.approx(3000.0)
    assert m["collisions_per_1000_vkm"] == 0.0 and m["inserted_share"] == 1.0 and m["mean_depart_delay_s"] == 0.0
    assert m["vehicle_km"] == pytest.approx(q * 0.2 * 0.48, rel=1e-6)  # flow x hours x km


def test_congested_uniform_flow_has_no_capacity_drop():
    trajectories, vehicles = sample(constant_paths(8.0, 2.5, 1, 0.2) + constant_paths(8.0, 2.5, 2, 1.4))
    m = corridor_metrics(trajectories, vehicles, GEOMETRY)
    assert m["n_congested_intervals"] == 24 and m["congested_share"] == 1.0  # 8 m/s < 11 m/s at 100 m throughout
    assert m["queue_discharge_flow"] == pytest.approx(2880.0) and m["capacity_drop"] == pytest.approx(0.0, abs=1e-12)
    assert m["flow_peak_2min"] == pytest.approx(2880.0)
    later = corridor_metrics(trajectories, vehicles, Geometry(20.0, 500.0, (180.0, 840.0)))  # nothing assumes 120 s
    assert later["window"] == [180.0, 840.0] and len(later["detectors"]["450"]["count"]) == 22
    assert later["edie"]["t_edges"][0] == 180.0 and later["congested_share"] == 1.0


def test_capacity_drop_from_the_detector_series():
    from cf_stability.corridor.macro import capacity_drop

    speed = np.array([15.0, 15.0, 15.0, 15.0, 9.0, 8.0, np.nan, 12.0])
    flow = np.array([6000.0, 6400.0, 6200.0, 6600.0, 5000.0, 5200.0, 5100.0, 5900.0])
    out = capacity_drop(speed, flow, 11.0, 4)
    assert out["n_congested"] == 2 and out["queue_discharge_flow"] == pytest.approx(5100.0)
    assert out["largest_flow"] == pytest.approx(6300.0)  # (6000 + 6400 + 6200 + 6600) / 4
    assert out["capacity_drop"] == pytest.approx(1 - 5100 / 6300)
    assert capacity_drop(speed, flow, 5.0, 4)["capacity_drop"] is None  # nothing congested
    assert capacity_drop(speed[:3], flow[:3], 11.0, 4)["largest_flow"] is None  # fewer intervals than 2 minutes


def test_detectors_against_hand_counted_crossings():
    rows = [  # vehicle, t, x, lane, v
        (0, 120.0, 90.0, 3, 10.0), (0, 121.0, 101.0, 3, 12.0),  # passes 100 m at 120 + 10/11 s in lane 3
        (1, 130.0, 99.0, 2, 1.0), (1, 131.0, 100.0, 2, 0.0), (1, 132.0, 100.0, 2, 0.0), (1, 133.0, 104.0, 4, 4.0),
        (2, 149.0, 95.0, 1, 5.0), (2, 150.0, 100.0, 1, 5.0), (2, 151.0, 105.0, 1, 5.0),  # passes exactly at 150 s
        (3, 160.0, 101.0, 5, 9.0), (3, 161.0, 110.0, 5, 9.0),  # starts beyond the detector: no passage
        (4, 200.0, 60.0, 6, 20.0), (4, 201.0, 80.0, 6, 20.0), (4, 205.0, 160.0, 6, 20.0),  # gap of 4 s: not joined
    ]  # fmt: skip
    vehicle, t, x, lane, v = (np.array(col) for col in zip(*rows))
    trajectories = {"vehicle": vehicle, "t": t, "x": x, "lane": lane, "v": v}
    prepared = prepare_trajectories(trajectories, None, GEOMETRY, MacroConfig())
    c = crossings(prepared, 100.0)
    np.testing.assert_array_equal(c["vehicle"], [0, 1, 2])
    np.testing.assert_allclose(c["t"], [120.0 + 10 / 11, 131.0, 150.0])
    np.testing.assert_allclose(c["v"], [10.0 + 2.0 * 10 / 11, 0.0, 5.0])
    np.testing.assert_array_equal(c["lane"], [3, 2, 1])
    det = detector(prepared, 100.0, np.array([120.0, 150.0, 180.0, 210.0]))
    assert det["count"].tolist() == [2, 1, 0]  # the passage at exactly 150 s opens the second interval
    np.testing.assert_allclose(det["flow_vph"], [240.0, 120.0, 0.0])
    np.testing.assert_allclose(det["speed"][:2], [(10.0 + 20 / 11 + 0.0) / 2, 5.0])
    assert np.isnan(det["speed"][2])
    with_gap = MacroConfig(max_gap_s=5.0)
    assert crossings(prepare_trajectories(trajectories, None, GEOMETRY, with_gap), 100.0)["vehicle"].tolist() == [0, 1, 2, 4]


def test_boundary_points_close_the_trajectories():
    """A vehicle at 25 m/s: its first sample lies beyond 30 m and its last before 490 m; the insertion and
    the passage of x_out recover its travel time exactly."""
    t = np.array([200.4, 220.4])  # to 520 m
    trajectories, vehicles = sample([{"t": t, "x": 20.0 + 25.0 * (t - t[0]), "lane": 2}])
    assert trajectories["x"][0] > 30.0 and trajectories["x"][-1] < 490.0
    with_points = prepare_trajectories(trajectories, vehicles, GEOMETRY, MacroConfig())
    assert with_points["virtual"].sum() == 2
    assert travel_times(with_points, 30.0, 490.0, (120.0, 840.0)) == pytest.approx([460 / 25.0])
    without = prepare_trajectories(trajectories, vehicles, GEOMETRY, MacroConfig(boundary_points=False))
    assert len(travel_times(without, 30.0, 490.0, (120.0, 840.0))) == 0
    # insertion after the first sample or far before it is no boundary point
    late = {**vehicles, "depart": vehicles["depart"] + 5.0}
    assert prepare_trajectories(trajectories, late, GEOMETRY, MacroConfig())["virtual"].sum() == 1
    # a vehicle that never passes x_out (exit_t NaN, e.g. a track that ends at 499.8 m) gets no exit point
    no_exit = prepare_trajectories(trajectories, {**vehicles, "exit_t": np.array([np.nan])}, GEOMETRY, MacroConfig())
    assert no_exit["virtual"].sum() == 1 and no_exit["x"].max() < 490.0
    assert len(travel_times(no_exit, 30.0, 490.0, (120.0, 840.0))) == 0


def test_travel_time_needs_both_passages_inside_the_window():
    paths = [{"t": np.array([d, d + 48.0]), "x": np.array([20.0, 500.0]), "lane": 1} for d in (60.0, 100.0, 500.0, 800.0)]
    trajectories, vehicles = sample(paths)
    prepared = prepare_trajectories(trajectories, vehicles, GEOMETRY, MacroConfig())
    # passes 30 m at depart + 1 s and 490 m at depart + 47 s: only the departures at 500 s qualifies fully
    assert travel_times(prepared, 30.0, 490.0, (120.0, 840.0)) == pytest.approx([46.0])


def test_empty_and_collapsed_inputs_give_none():
    empty = {key: np.zeros(0, dtype) for key, dtype in
             (("t", np.float32), ("vehicle", np.int32), ("x", np.float32), ("lane", np.int8), ("v", np.float32))}  # fmt: skip
    n = 5
    vehicles = {"depart_planned": np.linspace(200, 300, n), "depart": np.linspace(200.5, 300.5, n),
                "entry_x": np.full(n, 20.0), "exit_t": np.full(n, np.nan), "collided": np.ones(n, bool)}  # fmt: skip
    m = corridor_metrics(empty, vehicles, GEOMETRY)
    assert m["throughput_vph"] == 0.0 and m["mean_speed"] is None and m["vehicle_km"] == 0.0
    assert m["collisions_per_1000_vkm"] is None and m["collision_episodes"] == n  # removed before their first sample
    assert m["collision_source"] == "collided" and m["vehicles_in_contact_share"] is None  # no vehicle in the window
    assert m["travel_time"]["n"] == 0 and m["travel_time"]["median"] is None
    assert m["waves"]["n_waves"] == 0 and m["queue_discharge_flow"] is None and m["capacity_drop"] is None
    assert m["waves"]["wave_speed_xcorr"] is None and m["congested_share"] == 0.0 and m["flow_peak_2min"] == 0.0
    assert m["inserted_share"] == 1.0 and m["mean_depart_delay_s"] == pytest.approx(0.5)
    for detector_ in m["detectors"].values():
        assert set(detector_["count"]) == {0} and all(s is None for s in detector_["speed"])
    nothing = corridor_metrics({}, None, GEOMETRY)
    assert nothing["inserted_share"] is None and nothing["n_vehicles"] == 0
    error = macro_error(m, corridor_metrics(*uniform_flow(), GEOMETRY))
    assert error["components"]["throughput"] == -1.0 and error["components"]["travel_time"] is None
    assert error["n_components"] == len([c for c in error["components"].values() if c is not None])


def test_collision_episodes_of_both_interfaces():
    """Vehicles at 10 m/s spend 48 s in the section: A inside the window (2 episodes), B from 100 s (28 of its
    48 s in the window, 3 episodes), C after the window (1 episode), D inside without contact."""
    departs = {"A": 200.0, "B": 100.0, "C": 900.0, "D": 300.0}
    trajectories, vehicles = sample([{"t": np.array([d, d + 50.0]), "x": np.array([20.0, 520.0]), "lane": 1}
                                     for d in departs.values()])  # fmt: skip
    episodes = {**vehicles, "n_collisions": np.array([2, 3, 1, 0], dtype=np.int32)}
    del episodes["collided"]
    m = corridor_metrics(trajectories, episodes, GEOMETRY)
    assert m["collision_source"] == "n_collisions" and m["n_vehicles_window"] == 3  # A, B and D
    assert m["collision_episodes"] == pytest.approx(2 + 3 * 28 / 48)
    assert m["vehicles_in_contact_share"] == pytest.approx(2 / 3)
    assert m["vehicle_km"] == pytest.approx((480 + 280 + 480) / 1000)
    assert m["collisions_per_1000_vkm"] == pytest.approx((2 + 3 * 28 / 48) / 1.24 * 1000)
    old = {**vehicles, "collided": np.array([True, True, True, False])}  # removed: the collision is at the end
    m = corridor_metrics(trajectories, old, GEOMETRY)
    assert m["collision_source"] == "collided" and m["collision_episodes"] == 2.0  # A at 247 s and B at 147 s
    assert m["vehicles_in_contact_share"] == pytest.approx(2 / 3)
    none = corridor_metrics(trajectories, {k: v for k, v in vehicles.items() if k != "collided"}, GEOMETRY)
    assert none["collision_episodes"] == 0.0 and none["vehicles_in_contact_share"] == 0.0
    assert none["collisions_by_zone"] is None  # the episodes of vehicles.npz have no position


def test_collision_episodes_of_collisions_npz():
    """collisions.npz: the start of every episode; those of the window count, by zone (x < 40 m, x > 450 m)."""
    departs = {"A": 200.0, "B": 100.0, "C": 900.0, "D": 300.0}
    trajectories, vehicles = sample([{"t": np.array([d, d + 50.0]), "x": np.array([20.0, 520.0]), "lane": 1}
                                     for d in departs.values()])  # fmt: skip
    rows = [  # t, vehicle, x: two of A in the window, one of B before it and one inside, C and D not in it
        (210.0, 0, 30.0), (230.0, 0, 300.0), (110.0, 1, 120.0), (140.0, 1, 460.0), (905.0, 2, 70.0), (840.0, 3, 200.0),
    ]  # fmt: skip
    t, vehicle, x = (np.array(col) for col in zip(*rows))
    episodes = {"t": t.astype(np.float32), "vehicle": vehicle.astype(np.int32), "x": x.astype(np.float32),
                "lane": np.ones(len(t), np.int8)}  # fmt: skip
    m = corridor_metrics(trajectories, {**vehicles, "n_collisions": np.array([9, 9, 9, 9])}, GEOMETRY, episodes=episodes)
    assert m["collision_source"] == "collisions.npz" and m["collision_episodes"] == 3.0  # the window is [120, 840)
    assert m["collisions_by_zone"] == {"entry": 1, "exit": 1, "elsewhere": 1} and m["collision_zones_m"] == [40.0, 450.0]
    assert m["vehicles_in_contact_share"] == pytest.approx(2 / 3)  # A and B of A, B, D
    assert m["collisions_per_1000_vkm"] == pytest.approx(3 / 1.24 * 1000)
    empty = {key: value[:0] for key, value in episodes.items()}
    m = corridor_metrics(trajectories, vehicles, GEOMETRY, MacroConfig(collision_exit_m=100.0), episodes=empty)
    assert m["collision_episodes"] == 0.0 and m["collisions_by_zone"] == {"entry": 0, "exit": 0, "elsewhere": 0}
    assert m["vehicles_in_contact_share"] == 0.0 and m["collision_zones_m"] == [40.0, 400.0]


def test_macro_error_of_identical_inputs_is_zero():
    trajectories, vehicles = sample(wave_paths([(400.0, 20.0)]))
    m = corridor_metrics(trajectories, vehicles, GEOMETRY)
    assert m["waves"]["wave_speed_xcorr"] is not None
    error = macro_error(m, m)
    assert error["n_components"] == len(COMPONENTS) and error["value"] == 0.0
    assert all(value == 0.0 for value in error["components"].values())
    dynamic = macro_error_dynamic(error)
    assert dynamic["n_components"] == 4 and dynamic["value"] == 0.0
    other = {**m, "window": [180.0, 840.0]}  # metrics of two windows are not compared
    error = macro_error(other, m)
    assert error["value"] is None and error["n_components"] == 0 and "window" in error["error"]


def test_macro_error_components():
    truth = {
        "throughput_vph": 6000.0, "mean_speed": 6.0, "queue_discharge_flow": 5000.0,
        "fd": {"density_bins": [100.0, 110.0, 120.0], "flow": [6000.0, 6200.0, 6400.0], "mean_flow": 6200.0},
        "waves": {"n_waves": 4, "wave_speed": -7.0, "wave_speed_xcorr": -5.0, "wave_amplitude": 4.0},
        "travel_time": {"values": [60.0, 70.0, 80.0], "mean": 70.0},
    }  # fmt: skip
    run = {
        "throughput_vph": 6600.0, "mean_speed": 4.5, "queue_discharge_flow": None,
        "fd": {"density_bins": [110.0, 120.0, 130.0], "flow": [6100.0, 6700.0, 7000.0], "mean_flow": 6000.0},
        "waves": {"n_waves": 0, "wave_speed": None, "wave_speed_xcorr": None, "wave_amplitude": None},
        "travel_time": {"values": [67.0, 77.0, 87.0], "mean": 77.0},
    }  # fmt: skip
    error = macro_error(run, truth)
    c = error["components"]
    assert c["throughput"] == pytest.approx(0.1) and c["mean_speed"] == pytest.approx(-0.25)
    assert c["queue_discharge_flow"] is None and c["wave_speed"] is None and c["wave_amplitude"] is None
    assert c["fd"] == pytest.approx(math.sqrt((100.0**2 + 300.0**2) / 2) / 6200.0)
    assert c["n_waves"] == pytest.approx(-1.0) and c["travel_time"] == pytest.approx(7.0 / 70.0)
    assert error["n_components"] == 5
    assert error["value"] == pytest.approx(np.mean([0.1, 0.25, c["fd"], 1.0, 0.1]))
    # the wave speed of the vector is the cross-correlation estimate; the leading edges do not enter it
    waves = {"n_waves": 2, "wave_speed": -100.0, "wave_speed_xcorr": -6.0}
    assert macro_error({**run, "waves": waves}, truth)["components"]["wave_speed"] == \
        pytest.approx(-0.2)  # (run - truth) / |truth|: 20 % faster upstream
    no_waves = {**truth, "waves": {"n_waves": 0}}
    assert macro_error({**run, "waves": {"n_waves": 3}}, no_waves)["components"]["n_waves"] == 3.0  # over max(0, 1)
    assert fd_error({"density_bins": [0.0], "flow": [1.0]}, truth["fd"]) is None  # no common bin


def test_dynamic_macro_error():
    """The mean of the absolute fundamental-diagram, wave-speed, wave-count and amplitude components (M6)."""
    error = {"components": {"throughput": 0.5, "mean_speed": -0.4, "queue_discharge_flow": None, "fd": 0.1,
                            "wave_speed": -0.2, "n_waves": -1.0, "wave_amplitude": None, "travel_time": 0.3},
             "value": 0.5, "n_components": 6}  # fmt: skip
    dynamic = macro_error_dynamic(error)
    assert dynamic["components"] == {"fd": 0.1, "wave_speed": -0.2, "waves": -1.0, "wave_amplitude": None}
    assert dynamic["n_components"] == 3 and dynamic["value"] == pytest.approx((0.1 + 0.2 + 1.0) / 3)
    assert "error" not in dynamic
    none = macro_error_dynamic({"components": dict.fromkeys(COMPONENTS), "value": None, "n_components": 0,
                                "error": "no ground truth macro.json"})  # fmt: skip
    assert none["value"] is None and none["n_components"] == 0 and none["error"] == "no ground truth macro.json"
    assert macro_error_dynamic({})["value"] is None


def test_fundamental_diagram_bins_and_scatter():
    fd = fundamental_diagram(np.array([100.0, 200.0, 300.0, 1000.0]), np.array([5.0, 9.99, 25.0, 12.0]), 10.0)
    assert fd["density_bins"] == [0.0, 10.0, 20.0] and fd["count"] == [2, 1, 1]
    assert fd["flow"] == [150.0, 1000.0, 300.0] and fd["flow_std"][1:] == [None, None]
    assert fd["scatter"] == pytest.approx(math.sqrt(2 * 50.0**2 / 1)) and fd["mean_flow"] == 400.0
    assert fundamental_diagram(np.zeros(0), np.zeros(0), 10.0)["scatter"] is None


def test_geometry_and_edges():
    scenario = {"scenario": "i80_p0", "geometry": {"x_in": 25.0, "x_out": 480.0}, "analysis": {"window": [100, 700]}}
    assert geometry_from_scenario(scenario) == Geometry(25.0, 480.0, (100.0, 700.0))
    assert geometry_from_scenario({}, GEOMETRY) == GEOMETRY
    assert geometry_from_scenario({"x_in": 20, "window_s": {"start": 60, "end": 600}}).window == (60.0, 600.0)
    # a scalar called window (the averaging window of the boundary speeds) is no analysis window
    boundary = {"config": {"boundary": {"window_s": 2.5}}, "geometry": {"x_in": 20.0}}
    assert geometry_from_scenario(boundary, GEOMETRY) == GEOMETRY
    assert geometry_from_scenario({**boundary, "config": {**boundary["config"], "window": [100, 800]}}).window == (100.0, 800.0)
    np.testing.assert_array_equal(section_edges(20.0, 500.0, 100.0), [20, 100, 200, 300, 400, 500])
    np.testing.assert_array_equal(section_edges(0.0, 250.0, 100.0), [0, 100, 200, 250])


def test_config_round_trip_and_unknown_keys():
    cfg = MacroConfig.from_mapping({"detectors": [100, 450], "waves": {"lanes": [1, 2], "threshold_share": 0.4}})
    assert cfg.detectors == (100.0, 450.0) and cfg.waves.lanes == (1, 2) and cfg.waves.threshold_share == 0.4
    assert MacroConfig.from_mapping(cfg.as_dict()) == cfg
    with pytest.raises(ValueError, match="unknown macro keys"):
        MacroConfig.from_mapping({"detector": [100]})
    with pytest.raises(ValueError, match="unknown wave keys"):
        MacroConfig.from_mapping({"waves": {"share": 0.4}})


def test_scenario_metrics_settings():
    """A scenario of another geometry (US-101, D112) brings its detectors and wave lanes in scenario.json["metrics"];
    nested keys are merged; without the block the configuration (and its hash) is that of the caller."""
    base = MacroConfig()
    assert scenario_macro_config(base, {}) is base and scenario_macro_config(base, {"metrics": None}) is base
    us = scenario_macro_config(base, {"metrics": {"detectors": [100, 200, 300, 400, 500, 590], "throughput_x": 590,
                                                  "waves": {"lanes": [1, 2, 3, 4, 5]}}})  # fmt: skip
    assert us.detectors == (100.0, 200.0, 300.0, 400.0, 500.0, 590.0) and us.throughput_x == 590 and us.queue_x == 100
    assert us.waves.lanes == (1, 2, 3, 4, 5) and us.waves.smooth == base.waves.smooth
    assert us.waves.xcorr == base.waves.xcorr
    assert config_hash(us.hashed()) != config_hash(base.hashed())
    with pytest.raises(ValueError, match="unknown macro keys"):
        scenario_macro_config(base, {"metrics": {"detector": [1]}})
