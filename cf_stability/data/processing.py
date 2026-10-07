"""Signal processing helpers: uniform resampling, Savitzky-Golay smoothing, derivatives."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.signal import savgol_filter

from cf_stability.data.schema import DT

# a grid point closer than this (s) to a source sample counts as lying on it
_TIME_TOL = 1e-6


def resample_uniform(
    t: np.ndarray,
    columns: Mapping[str, np.ndarray],
    dt: float = DT,
    t0: float | None = None,
    max_gap: float = 0.35,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Linearly interpolate every column onto the grid ``t0 + k * dt``.

    Source samples are put in time order (stable sort) and a repeated time stamp keeps its
    first sample. ``t0`` defaults to the first source time; the grid ends at or before the
    last source time. Per column, NaN samples are missing, and grid points outside the
    observed range or strictly inside a gap longer than ``max_gap`` seconds are NaN.
    """
    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    keep = np.isfinite(ts)
    keep[1:] &= ts[1:] != ts[:-1]
    ts = ts[keep]
    if ts.size == 0:
        grid = np.empty(0)
    else:
        start = float(ts[0]) if t0 is None else float(t0)
        n = max(int(math.floor((ts[-1] - start) / dt + 1e-9)) + 1, 0)
        grid = start + np.arange(n) * dt
    out = {}
    for name, column in columns.items():
        y = np.asarray(column, dtype=np.float64)
        if y.shape != t.shape:
            raise ValueError(f"column {name} has shape {y.shape}, expected {t.shape}")
        y = y[order][keep]
        ok = np.isfinite(y)
        out[name] = _interp_with_gaps(grid, ts[ok], y[ok], max_gap)
    return grid, out


def _interp_with_gaps(grid: np.ndarray, ts: np.ndarray, ys: np.ndarray, max_gap: float) -> np.ndarray:
    out = np.full(grid.shape, np.nan)
    if ts.size == 0:
        return out
    i = np.searchsorted(ts, grid + _TIME_TOL, side="right") - 1  # last sample at or before each grid point
    left = ts[np.maximum(i, 0)]
    right = ts[np.minimum(i + 1, ts.size - 1)]
    in_gap = (right - left > max_gap) & (grid - left > _TIME_TOL)
    use = (i >= 0) & (grid <= ts[-1] + _TIME_TOL) & ~in_gap
    out[use] = np.interp(grid[use], ts, ys)
    return out


def resample_nearest(t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value of the source sample nearest to each grid time (any dtype, e.g. strings or ids).

    Source samples are ordered and de-duplicated as in ``resample_uniform``.
    """
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values)
    order = np.argsort(t, kind="stable")
    ts, vs = t[order], values[order]
    keep = np.isfinite(ts)
    keep[1:] &= ts[1:] != ts[:-1]
    ts, vs = ts[keep], vs[keep]
    grid = np.asarray(grid, dtype=np.float64)
    if ts.size < 2:
        return np.repeat(vs[:1], grid.size) if ts.size else np.full(grid.size, None, dtype=object)
    right = np.clip(np.searchsorted(ts, grid), 1, ts.size - 1)
    left_nearer = grid - ts[right - 1] <= ts[right] - grid
    return vs[np.where(left_nearer, right - 1, right)]


def savgol_window_length(dt: float = DT, window_s: float = 1.1) -> int:
    """Odd number of samples nearest to ``window_s / dt`` (11 at 10 Hz); ties round up."""
    return 2 * int(math.floor((window_s / dt - 1.0) / 2.0 + 0.5)) + 1


def savgol_smooth(x: np.ndarray, dt: float = DT, window_s: float = 1.1, order: int = 2) -> np.ndarray:
    """Savitzky-Golay filter (``mode="interp"``) with a window of about ``window_s`` seconds.

    A series shorter than the window uses the largest odd window that fits and is longer
    than ``order``; if there is none, the series is returned unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    window = savgol_window_length(dt, window_s)
    if window <= order:
        raise ValueError(f"window of {window} samples is too short for order {order}")
    if x.shape[0] < window:
        window = x.shape[0] if x.shape[0] % 2 else x.shape[0] - 1
        if window <= order:
            return x.copy()
    return savgol_filter(x, window, order, mode="interp")


def central_diff(x: np.ndarray, dt: float = DT) -> np.ndarray:
    """Second-order central differences, second-order one-sided at the ends.

    Two samples give their forward difference, a single sample gives 0.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] >= 3:
        return np.gradient(x, dt, edge_order=2)
    if x.shape[0] == 2:
        return np.full(2, (x[1] - x[0]) / dt)
    return np.zeros_like(x)


def acceleration_from_speed(v: np.ndarray, dt: float = DT, window_s: float = 1.1, order: int = 2) -> np.ndarray:
    """Acceleration as central differences of the Savitzky-Golay filtered speed."""
    return central_diff(savgol_smooth(v, dt, window_s, order), dt)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges ``(start, stop)`` of the runs of True values."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m, [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
