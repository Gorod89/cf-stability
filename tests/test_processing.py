import numpy as np
import pytest

from cf_stability.data.processing import (
    acceleration_from_speed,
    central_diff,
    contiguous_runs,
    resample_uniform,
    savgol_smooth,
    savgol_window_length,
)


def test_acceleration_of_quadratic_speed_is_exact():
    t = np.arange(200) * 0.1
    v = 12.0 + 0.5 * t - 0.03 * t**2
    a = acceleration_from_speed(v)
    np.testing.assert_allclose(a, 0.5 - 0.06 * t, rtol=0, atol=1e-9)  # interior and both ends


def test_central_diff_is_second_order_at_the_ends():
    t = np.arange(10) * 0.1
    np.testing.assert_allclose(central_diff(3.0 * t**2 - t), 6.0 * t - 1.0, rtol=0, atol=1e-9)


def test_window_is_11_samples_at_10_hz():
    assert savgol_window_length(0.1, 1.1) == 11
    assert savgol_window_length(0.04, 1.1) == 27  # 27.5 samples -> nearest odd number
    impulse = np.zeros(61)
    impulse[30] = 1.0
    assert np.count_nonzero(np.abs(savgol_smooth(impulse)) > 1e-12) == 11


def test_short_series():
    t = np.arange(7) * 0.1
    np.testing.assert_allclose(acceleration_from_speed(3.0 + 2.0 * t + t**2), 2.0 + 2.0 * t, atol=1e-9)
    np.testing.assert_allclose(savgol_smooth(np.array([1.0, 5.0, 2.0])), [1.0, 5.0, 2.0])  # window 3, exact fit
    np.testing.assert_array_equal(savgol_smooth(np.array([1.0, 4.0])), [1.0, 4.0])  # no window > order
    np.testing.assert_allclose(central_diff(np.array([1.0, 1.5])), [5.0, 5.0])
    np.testing.assert_array_equal(central_diff(np.array([2.0])), [0.0])
    assert acceleration_from_speed(np.array([])).shape == (0,)


def test_resample_25_to_10_hz():
    t = np.arange(500) * 0.04
    grid, out = resample_uniform(t, {"x": np.sin(0.5 * t) + 0.1 * t, "y": np.cos(t)})
    assert grid[0] == 0.0 and grid[-1] <= t[-1] and grid[-1] > t[-1] - 0.1
    np.testing.assert_allclose(np.diff(grid), 0.1, atol=1e-12)
    assert np.max(np.abs(out["x"] - (np.sin(0.5 * grid) + 0.1 * grid))) < 1e-3
    assert np.max(np.abs(out["y"] - np.cos(grid))) < 1e-3


def test_resample_gaps_become_nan():
    t = np.arange(100) * 0.1
    x = 2.0 * t
    y = 3.0 * t
    y[50:53] = np.nan  # 0.4 s gap in y only
    drop = (t > 2.05) & (t < 3.05)  # 1.1 s gap in both columns
    drop[70] = True  # single missing sample: 0.2 s gap, interpolated
    grid, out = resample_uniform(t[~drop], {"x": x[~drop], "y": y[~drop]})
    np.testing.assert_allclose(grid, t, atol=1e-12)
    inside = (grid > 2.05) & (grid < 3.05)
    assert np.isnan(out["x"][inside]).all() and np.isnan(out["y"][inside]).all()
    assert not np.isnan(out["x"][~inside]).any()
    assert np.isnan(out["y"][50:53]).all()
    assert not np.isnan(out["y"][~inside & ~np.isin(np.arange(100), [50, 51, 52])]).any()
    np.testing.assert_allclose(out["x"][~inside], 2.0 * grid[~inside], atol=1e-12)


def test_resample_sorts_drops_duplicates_and_respects_t0():
    t = np.array([0.2, 0.0, 0.1, 0.1, 0.3])
    x = np.array([2.0, 0.0, 1.0, 99.0, 3.0])
    grid, out = resample_uniform(t, {"x": x})
    np.testing.assert_allclose(out["x"], [0.0, 1.0, 2.0, 3.0], atol=1e-12)
    grid, out = resample_uniform(t, {"x": x}, t0=-0.05)
    np.testing.assert_allclose(grid, [-0.05, 0.05, 0.15, 0.25], atol=1e-12)
    assert np.isnan(out["x"][0])
    np.testing.assert_allclose(out["x"][1:], [0.5, 1.5, 2.5], atol=1e-12)


@pytest.mark.parametrize(
    "mask, runs",
    [
        ([], []),
        ([False, False], []),
        ([True, True, True], [(0, 3)]),
        ([False, True, True, False, True], [(1, 3), (4, 5)]),
    ],
)
def test_contiguous_runs(mask, runs):
    assert contiguous_runs(np.array(mask, dtype=bool)) == runs
