"""Edie's generalised definitions on analytic trajectories."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cf_stability.corridor.macro import compare_macro, edie_grid

X_EDGES = np.arange(0.0, 501.0, 50.0)
T_EDGES = np.arange(0.0, 101.0, 10.0)


def constant_platoon(speed: float, headway: float, lane: int, first_id: int) -> dict[str, np.ndarray]:
    """Vehicles at constant speed and time headway, sampled at 10 Hz off the grid lines."""
    t = np.arange(-40.0, 140.0, 0.1) + 0.05
    veh, ts, xs = [], [], []
    for j in range(-80, 80):
        x = speed * (t - j * headway - 0.03)
        on = (x > -100.0) & (x < 700.0)
        veh.append(np.full(on.sum(), first_id + j))
        ts.append(t[on])
        xs.append(x[on])
    veh, ts, xs = np.concatenate(veh), np.concatenate(ts), np.concatenate(xs)
    return {"vehicle": veh, "t": ts, "x": xs, "lane": np.full(len(ts), lane)}


@pytest.fixture(scope="module")
def two_lanes():
    a = constant_platoon(25.0, 2.0, lane=1, first_id=1000)
    b = constant_platoon(20.0, 2.5, lane=2, first_id=5000)
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def test_constant_platoon_gives_analytic_flow_density_speed(two_lanes):
    grid = edie_grid(two_lanes, X_EDGES, T_EDGES)
    lane = grid["per_lane"]
    np.testing.assert_array_equal(grid["lanes"], [1, 2])
    # dx / v and dt are multiples of the headway, so every cell holds the exact average
    np.testing.assert_allclose(lane["flow"][0], 1 / 2.0, rtol=1e-9)
    np.testing.assert_allclose(lane["flow"][1], 1 / 2.5, rtol=1e-9)
    np.testing.assert_allclose(lane["density"][0], 1 / (25.0 * 2.0), rtol=1e-9)
    np.testing.assert_allclose(lane["density"][1], 1 / (20.0 * 2.5), rtol=1e-9)
    np.testing.assert_allclose(lane["speed"][0], 25.0, rtol=1e-9)
    np.testing.assert_allclose(lane["speed"][1], 20.0, rtol=1e-9)
    total = grid["total"]
    np.testing.assert_allclose(total["flow"], 0.9, rtol=1e-9)
    np.testing.assert_allclose(total["density"], 0.04, rtol=1e-9)
    np.testing.assert_allclose(total["speed"], 22.5, rtol=1e-9)

    only_lane_2 = edie_grid(two_lanes, X_EDGES, T_EDGES, lanes=[2])
    np.testing.assert_allclose(only_lane_2["total"]["flow"], 0.4, rtol=1e-9)


def test_track_mapping_input_and_compare(two_lanes):
    arrays = edie_grid(two_lanes, X_EDGES, T_EDGES)
    veh = two_lanes["vehicle"]
    tracks = {
        v: SimpleNamespace(t=two_lanes["t"][veh == v], x=two_lanes["x"][veh == v], lane=two_lanes["lane"][veh == v])
        for v in np.unique(veh)
    }
    from_tracks = edie_grid(tracks, X_EDGES, T_EDGES)
    np.testing.assert_allclose(from_tracks["total"]["flow"], arrays["total"]["flow"], rtol=1e-12)

    same = compare_macro(arrays, from_tracks)["total"]
    assert same["n_cells"] == 100 and same["speed_corr"] == pytest.approx(1.0)
    assert same["rel_diff_distance"] == pytest.approx(0.0, abs=1e-12) and same["flow_rel_error"] == pytest.approx(0.0, abs=1e-12)

    # same headways at 90 % speed: flow unchanged, density up by 1/0.9
    slower = edie_grid({**two_lanes, "x": 0.9 * two_lanes["x"]}, X_EDGES, T_EDGES)
    diff = compare_macro(arrays, slower)["per_lane"]
    assert diff["speed_rmse"] > 1.0 and diff["flow_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert diff["density_rel_error"] == pytest.approx(1 / 0.9 - 1, rel=1e-6)


def test_empty_input_gives_empty_cells():
    """No vehicle (a law whose vehicles all collided): zero flow and density, no speed, no exception."""
    for empty in ({}, {"vehicle": np.zeros(0, int), "t": np.zeros(0), "x": np.zeros(0), "lane": np.zeros(0, int)}):
        grid = edie_grid(empty, X_EDGES, T_EDGES)
        assert grid["per_lane"]["flow"].shape == (0, 10, 10) and not grid["total"]["flow"].any()
        assert np.isnan(grid["total"]["speed"]).all()
    one = edie_grid({"vehicle": np.zeros(1, int), "t": np.ones(1), "x": np.ones(1)}, X_EDGES, T_EDGES, lanes=[0])
    assert one["total"]["time"].sum() == 0.0  # a single sample has no segment
