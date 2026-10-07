"""Stop waves (cf_stability/corridor/waves.py) on synthetic waves travelling upstream at -5 m/s.

Vehicles at 10 m/s drive through bands of 1 m/s that travel upstream at -5 m/s (tests/test_corridor_macro.py
generates them); the detection must find one wave per band with its speed within 0.5 m/s and an
amplitude of about 10 - 1 m/s, with the settings of the configuration and with those of the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from cf_stability.corridor.macro import MacroConfig, corridor_metrics
from cf_stability.corridor.waves import WaveConfig, XcorrConfig, find_waves, median_by_x, speed_field, xcorr_wave_speed
from test_corridor_macro import GEOMETRY, constant_paths, sample, uniform_flow, wave_paths

CONTRACT = WaveConfig(smooth=0, threshold_share=0.4, min_length=60.0, min_duration=6.0)  # docs/m5_contract.md, 5
DEFAULT = WaveConfig()  # configs/corridor_metrics.yaml


@pytest.fixture(scope="module")
def one_wave():
    return sample(wave_paths([(400.0, 20.0)]))  # passes 500 m at 400 s, 20 m at 496 s; 20 s long at every x


@pytest.mark.parametrize("waves", [DEFAULT, CONTRACT], ids=["default", "contract"])
def test_one_stop_wave_travelling_upstream(one_wave, waves):
    m = corridor_metrics(*one_wave, GEOMETRY, MacroConfig(waves=waves))
    w = m["waves"]
    assert w["n_waves"] == 1
    assert w["wave_speed"] == pytest.approx(-5.0, abs=0.5)
    wave = w["per_wave"][0]
    assert wave["speed"] == w["wave_speed"] and wave["fit_r2"] > 0.95
    assert wave["x_start"] <= 40.0 and wave["x_end"] >= 480.0  # the band crosses the whole section
    assert 390.0 <= wave["t_start"] <= 402.0 and 496.0 <= wave["t_end"] <= 540.0
    assert 0.99 <= wave["min_speed"] <= 1.5 and w["wave_amplitude"] == pytest.approx(9.0, abs=0.6)  # float32 samples
    assert np.allclose(w["median_speed"], 10.0, atol=0.2)  # the band is a small part of the window at every x


def test_two_waves_are_two(one_wave):
    trajectories, vehicles = sample(wave_paths([(300.0, 16.0), (600.0, 24.0)]))
    w = corridor_metrics(trajectories, vehicles, GEOMETRY)["waves"]
    assert w["n_waves"] == 2
    assert [wave["speed"] for wave in w["per_wave"]] == [pytest.approx(-5.0, abs=0.5)] * 2
    assert w["per_wave"][0]["t_start"] < w["per_wave"][1]["t_start"]


def test_uniform_flow_has_no_wave():
    w = corridor_metrics(*uniform_flow(), GEOMETRY)["waves"]
    assert w["n_waves"] == 0 and w["per_wave"] == [] and w["wave_speed"] is None


def band_field(c: float = -5.0, t0: float = 100.0, duration: float = 12.0, free: float = 10.0, slow: float = 0.0):
    """Speed field on 20 m x 2 s cells (x 0..400, t 0..300) with a band of ``slow`` that enters x = 400 at ``t0``."""
    x_edges, t_edges = np.arange(0.0, 401.0, 20.0), np.arange(0.0, 301.0, 2.0)
    xm, tm = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (t_edges[1:] + t_edges[:-1])
    lead = t0 + (xm[:, None] - 400.0) / c
    inside = (tm[None, :] >= lead) & (tm[None, :] < lead + duration)
    return np.where(inside, slow, free), x_edges, t_edges


def test_find_waves_on_a_drawn_band():
    speed, x_edges, t_edges = band_field()
    out = find_waves(speed, x_edges, t_edges, CONTRACT)
    assert out["n_waves"] == 1 and out["wave_speed"] == pytest.approx(-5.0, abs=0.05)
    assert out["wave_amplitude"] == 10.0 and out["per_wave"][0]["min_speed"] == 0.0
    np.testing.assert_allclose(out["median_speed"], 10.0)
    np.testing.assert_allclose(out["threshold"], 4.0)
    downstream, _, _ = band_field(c=+4.0, t0=100.0)  # a wave that travels downstream keeps its sign
    assert find_waves(downstream, x_edges, t_edges, CONTRACT)["wave_speed"] == pytest.approx(4.0, abs=0.1)


def test_minimum_size_and_missing_cells():
    speed = np.full((20, 100), 10.0)
    x_edges, t_edges = np.arange(0.0, 401.0, 20.0), np.arange(0.0, 201.0, 2.0)
    speed[5:10, 10:15] = 1.0  # 100 m x 10 s: kept
    speed[12:16, 40:45] = 1.0  # 80 m x 10 s: too short in x
    speed[2:8, 70:74] = 1.0  # 120 m x 8 s: too short in t
    speed[15:19, 80:95] = np.nan  # cells without vehicles are never slow
    out = find_waves(speed, x_edges, t_edges, DEFAULT)
    assert out["n_waves"] == 1 and out["per_wave"][0]["x_start"] == 100.0 and out["per_wave"][0]["x_end"] == 200.0
    assert out["per_wave"][0]["t_start"] == 20.0 and out["per_wave"][0]["t_end"] == 30.0
    assert out["per_wave"][0]["speed"] is None  # entered at the same time everywhere: no travelling front
    assert find_waves(speed, x_edges, t_edges, CONTRACT)["n_waves"] == 3  # the contract's 60 m x 6 s keeps all three


def test_connectivity_joins_cells_that_share_a_corner():
    speed = np.full((10, 50), 10.0)
    for i in range(6):  # a staircase of single cells, each touching the next at a corner
        speed[i, 10 + i] = 0.0
    x_edges, t_edges = np.arange(0.0, 201.0, 20.0), np.arange(0.0, 101.0, 2.0)
    small = {"min_length": 60.0, "min_duration": 6.0, "smooth": 0}
    assert find_waves(speed, x_edges, t_edges, WaveConfig(connectivity=8, **small))["n_waves"] == 1
    assert find_waves(speed, x_edges, t_edges, WaveConfig(connectivity=4, **small))["n_waves"] == 0


def test_speed_field_is_the_edie_speed_of_the_block():
    rng = np.random.default_rng(3)
    distance, time = rng.uniform(0.0, 50.0, (6, 8)), rng.uniform(1.0, 5.0, (6, 8))
    time[2, 3], distance[2, 3] = 0.0, 0.0  # an empty cell gets the speed of its neighbours
    raw = speed_field(distance, time)
    assert np.isnan(raw[2, 3]) and raw[0, 0] == pytest.approx(distance[0, 0] / time[0, 0])
    block = speed_field(distance, time, smooth=1)
    assert block[2, 3] == pytest.approx(distance[1:4, 2:5].sum() / time[1:4, 2:5].sum())
    assert block[0, 0] == pytest.approx(distance[:2, :2].sum() / time[:2, :2].sum())  # the edge uses what exists
    assert np.isnan(speed_field(np.zeros((2, 2)), np.zeros((2, 2)), smooth=1)).all()
    np.testing.assert_allclose(median_by_x(np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])), [2.0, np.nan])


def test_empty_field_has_no_wave():
    out = find_waves(np.zeros((0, 0)), np.array([0.0]), np.array([0.0]))
    assert out["n_waves"] == 0 and out["wave_speed"] is None and out["wave_amplitude"] is None
    nan_field = find_waves(np.full((3, 4), np.nan), np.arange(4.0) * 20, np.arange(5.0) * 2)
    assert nan_field["n_waves"] == 0


def test_config_from_mapping():
    cfg = WaveConfig.from_mapping({"lanes": [1, 2, 3], "connectivity": 4, "xcorr": {"distance": 100}})
    assert cfg.lanes == (1, 2, 3) and cfg.connectivity == 4 and cfg.xcorr == XcorrConfig(distance=100.0)
    with pytest.raises(ValueError, match="connectivity"):
        WaveConfig.from_mapping({"connectivity": 6})
    with pytest.raises(ValueError, match="direction"):
        XcorrConfig.from_mapping({"direction": "downstream"})
    with pytest.raises(ValueError, match="unknown xcorr keys"):
        XcorrConfig.from_mapping({"lag": 10})


# ------------------------------------------------------------------------------ cross-correlation estimate


@pytest.mark.parametrize("waves", [[(400.0, 20.0)], [(300.0, 16.0), (600.0, 24.0)]], ids=["one wave", "two waves"])
def test_xcorr_wave_speed_of_synthetic_waves(waves):
    w = corridor_metrics(*sample(wave_paths(waves)), GEOMETRY)["waves"]
    assert w["wave_speed_xcorr"] == pytest.approx(-5.0, abs=0.5)
    xcorr = w["xcorr"]
    assert xcorr["n_pairs"] == 14 and xcorr["n_used"] >= 10 and xcorr["median_correlation"] > 0.5  # 30-230 ... 290-490 m
    assert xcorr["distance"] == 200.0 and len(xcorr["pair_speeds"]) == xcorr["n_used"]


def test_xcorr_of_a_field_without_waves_is_none():
    one_lane = sample(constant_paths(10.0, 2.0, 1, 0.37))  # every cell at 10 m/s: constant series
    assert corridor_metrics(*one_lane, GEOMETRY)["waves"]["wave_speed_xcorr"] is None
    x_edges, t_edges = np.arange(0.0, 401.0, 20.0), np.arange(0.0, 601.0, 2.0)
    constant = xcorr_wave_speed(np.full((20, 300), 8.0), x_edges, t_edges)
    assert constant["wave_speed_xcorr"] is None and constant["n_pairs"] == 10 and constant["n_used"] == 0
    noise = np.random.default_rng(5).normal(8.0, 1.0, (20, 300))  # no propagation: no correlation reaches 0.3
    assert xcorr_wave_speed(noise, x_edges, t_edges)["wave_speed_xcorr"] is None
    assert xcorr_wave_speed(np.zeros((0, 0)), np.array([0.0]), np.array([0.0]))["wave_speed_xcorr"] is None


def test_xcorr_on_drawn_bands_and_directions():
    speed, x_edges, t_edges = band_field(c=-5.0, t0=100.0, duration=12.0)
    assert xcorr_wave_speed(speed, x_edges, t_edges)["wave_speed_xcorr"] == pytest.approx(-5.0, abs=0.1)
    speed[3, 50:60] = np.nan  # cells without vehicles are left out of the correlation
    assert xcorr_wave_speed(speed, x_edges, t_edges)["wave_speed_xcorr"] == pytest.approx(-5.0, abs=0.1)
    downstream, _, _ = band_field(c=4.0, t0=120.0, duration=12.0)  # passes x = 0 at 20 s and x = 400 at 120 s
    assert xcorr_wave_speed(downstream, x_edges, t_edges)["wave_speed_xcorr"] == pytest.approx(4.0, abs=0.2)
    upstream_only = XcorrConfig(direction="upstream")
    assert xcorr_wave_speed(downstream, x_edges, t_edges, upstream_only)["wave_speed_xcorr"] is None
    slow = XcorrConfig(max_lag_s=30.0)  # 200 m at 5 m/s takes 40 s: outside the lags, no peak
    assert xcorr_wave_speed(band_field(c=-5.0)[0], x_edges, t_edges, slow)["wave_speed_xcorr"] is None
