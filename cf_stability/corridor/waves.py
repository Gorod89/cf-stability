"""Stop waves in a space-time speed field (docs/m5_contract.md, section 5; D98).

The field holds the speed of cells of ``dx`` metres by ``dt`` seconds (``[x cell, t cell]``, NaN where
no vehicle was): the time-mean speed of the vehicles in the cell, lanes together, i.e. Edie's distance
over time (:func:`speed_field`, optionally over the block of cells around each cell). A cell is in a
stop wave when its speed is below ``threshold_share`` times the median speed of its ``x`` over the
window; the connected regions of such cells that extend over at least ``min_length`` metres and
``min_duration`` seconds are the waves. Per wave a least-squares line through its leading edge (time
of entry into the wave, the start of its first cell, against ``x``) gives the wave speed ``dx/dt``
(negative: travelling upstream), and the amplitude is the median speed at the ``x`` of the slowest
cell minus the speed of that cell.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np
from scipy import ndimage


@dataclasses.dataclass(frozen=True)
class XcorrConfig:
    """Cross-correlation estimate of the wave speed (``configs/corridor_metrics.yaml``, ``waves.xcorr``).

    Every pair of cells of the speed field whose centres are ``distance`` apart; per pair the lag of the
    largest correlation of their speed series within ``min_lag_s <= |lag| <= max_lag_s`` (a local maximum,
    refined by a parabola), speed = -distance / lag; the median over the pairs.
    """

    distance: float = 200.0  # m, the fixed distance of the pairs of positions
    min_lag_s: float = 4.0  # s, smallest |lag|: lag 0 is a common trend of the whole section, not a wave
    max_lag_s: float = 120.0  # s, largest |lag| (200 m: speeds down to 1.7 m/s)
    direction: str = "both"  # both: a lag of either sign (positive speed: disturbances carried downstream); upstream
    min_correlation: float = 0.3  # a pair whose largest correlation is below this gives no speed
    min_overlap: float = 0.5  # share of the series that must have values in both cells at a lag
    smooth: int = 0  # block smoothing of the field for this estimate (0: the raw cells)
    refine: bool = True  # parabola through the largest correlation and its two neighbours: a lag between the cells

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> XcorrConfig:
        values = dict(raw or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown xcorr keys: {sorted(unknown)}")
        if values.get("direction", "both") not in ("both", "upstream"):
            raise ValueError(f"waves.xcorr.direction must be both or upstream, got {values['direction']!r}")
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class WaveConfig:
    """Cells, lanes and thresholds of the wave detection (``configs/corridor_metrics.yaml``, ``waves``).

    The contract names 40 %, 60 m and 6 s on the raw 20 m x 2 s field. On the NGSIM I-80 data that
    field (a few vehicle-seconds per cell) breaks the stop waves into fragments with leading edges of
    -50 to +17 m/s; the defaults smooth over 3 x 3 cells and take 50 %, 100 m and 10 s, which keeps the
    bands whole (check of M5 on the ground truth of the three periods).
    """

    dx: float = 20.0  # m, cell length of the speed field
    dt: float = 2.0  # s, cell duration
    lanes: tuple[int, ...] = (1, 2, 3, 4, 5, 6)  # NGSIM lanes of the field (the auxiliary lane 7 is left out)
    smooth: int = 1  # the speed of a cell is the Edie speed of the (2 smooth + 1)^2 cells around it (0: none)
    threshold_share: float = 0.5  # a cell is in a wave below this share of the median speed at its x
    min_length: float = 100.0  # m, smallest extent in x of a wave
    min_duration: float = 10.0  # s, smallest extent in t of a wave
    connectivity: int = 8  # 4: cells sharing a side; 8: also cells sharing a corner (a wave is a diagonal band)
    min_vehicle_seconds: float = 0.0  # cells with at most this occupancy time have no speed (0: any vehicle)
    xcorr: XcorrConfig = dataclasses.field(default_factory=XcorrConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> WaveConfig:
        values = dict(raw or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown wave keys: {sorted(unknown)}")
        if "lanes" in values:
            values["lanes"] = tuple(int(lane) for lane in values["lanes"])
        if values.get("connectivity", 8) not in (4, 8):
            raise ValueError(f"waves.connectivity must be 4 or 8, got {values['connectivity']}")
        values["xcorr"] = XcorrConfig.from_mapping(values.get("xcorr"))
        return cls(**values)


def speed_field(distance: np.ndarray, time: np.ndarray, smooth: int = 0, min_vehicle_seconds: float = 0.0) -> np.ndarray:
    """Speed of every cell from the Edie sums ``distance``, ``time [x, t]``: distance over time of the
    ``(2 smooth + 1)^2`` cells around it (the Edie speed of a moving block of cells; the grid edges use the
    cells that exist); NaN where that time is at most ``min_vehicle_seconds``."""
    distance, time = np.asarray(distance, dtype=np.float64), np.asarray(time, dtype=np.float64)
    if smooth > 0 and distance.size:
        kernel = np.ones((2 * smooth + 1, 2 * smooth + 1))
        distance = ndimage.convolve(distance, kernel, mode="constant", cval=0.0)
        time = ndimage.convolve(time, kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(time > max(min_vehicle_seconds, 0.0), distance / time, np.nan)


def median_by_x(speed: np.ndarray) -> np.ndarray:
    """Median over the time cells of every row of ``speed [x, t]``; NaN for a row without any speed."""
    out = np.full(speed.shape[0], np.nan)
    for i, row in enumerate(speed):
        finite = row[np.isfinite(row)]
        if len(finite):
            out[i] = np.median(finite)
    return out


def _leading_edge_speed(
    ix: np.ndarray, it: np.ndarray, x_mid: np.ndarray, t_edges: np.ndarray
) -> tuple[float | None, float | None]:
    """Wave speed (m/s) of the least-squares line ``t_entry = a + b x`` through the earliest cell of the
    wave at every ``x``, and the R^2 of that line; None when the line is undefined or vertical in ``t``."""
    columns = np.unique(ix)
    if len(columns) < 2:
        return None, None
    entry = np.array([t_edges[it[ix == c].min()] for c in columns])  # time of entry: start of the first cell
    x = x_mid[columns]
    slope, intercept = np.polyfit(x, entry, 1)
    residual = entry - (intercept + slope * x)
    spread = np.sum((entry - entry.mean()) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / spread) if spread > 0 else None
    # a line of zero slope is a region entered at the same time everywhere: no travelling front
    return (float(1.0 / slope) if abs(slope) > 1e-9 else None), r2


def find_waves(speed: np.ndarray, x_edges: np.ndarray, t_edges: np.ndarray, cfg: WaveConfig | None = None) -> dict:
    """Stop waves of the field ``speed [x cell, t cell]`` (m/s) on the grid ``x_edges``, ``t_edges``.

    Returns ``{n_waves, wave_speed, wave_amplitude, per_wave, median_speed, threshold}``: the medians
    over the waves of the wave speed (waves with a defined speed) and of the amplitude, None without
    waves; ``per_wave`` lists every wave with its extent, speed, amplitude and slowest cell.
    """
    cfg = cfg or WaveConfig()
    speed = np.asarray(speed, dtype=np.float64)
    x_edges, t_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(t_edges, dtype=np.float64)
    median = median_by_x(speed) if speed.size else np.full(len(x_edges) - 1, np.nan)
    threshold = cfg.threshold_share * median
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(speed) & (speed < threshold[:, None])
    structure = ndimage.generate_binary_structure(2, 2 if cfg.connectivity == 8 else 1)
    labels, n_regions = ndimage.label(mask, structure=structure) if mask.any() else (np.zeros_like(mask, int), 0)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx, dt = np.diff(x_edges), np.diff(t_edges)
    waves = []
    for label in range(1, n_regions + 1):
        ix, it = np.nonzero(labels == label)
        length = float(dx[ix.min() : ix.max() + 1].sum())
        duration = float(dt[it.min() : it.max() + 1].sum())
        if length < cfg.min_length - 1e-9 or duration < cfg.min_duration - 1e-9:
            continue
        inside = speed[ix, it]  # finite: the mask holds cells with a speed only
        k = int(np.argmin(inside))
        wave_speed, r2 = _leading_edge_speed(ix, it, x_mid, t_edges)
        waves.append({
            "x_start": float(x_edges[ix.min()]), "x_end": float(x_edges[ix.max() + 1]),
            "t_start": float(t_edges[it.min()]), "t_end": float(t_edges[it.max() + 1]),
            "n_cells": int(len(ix)), "speed": wave_speed, "fit_r2": r2,
            "min_speed": float(inside[k]), "x_min_speed": float(x_mid[ix[k]]),
            "t_min_speed": float(0.5 * (t_edges[it[k]] + t_edges[it[k] + 1])),
            "amplitude": float(median[ix[k]] - inside[k]),
        })  # fmt: skip
    waves.sort(key=lambda w: (w["t_start"], -w["x_end"]))
    speeds = [w["speed"] for w in waves if w["speed"] is not None]
    return {
        "n_waves": len(waves),
        "wave_speed": float(np.median(speeds)) if speeds else None,
        "wave_amplitude": float(np.median([w["amplitude"] for w in waves])) if waves else None,
        "per_wave": waves,
        "median_speed": median,
        "threshold": threshold,
    }


def _lagged_correlation(upstream: np.ndarray, downstream: np.ndarray, lag: int, min_count: int) -> float:
    """Pearson correlation of ``upstream[k + lag]`` with ``downstream[k]`` over the ``k`` where both have a
    value; NaN with fewer than ``min_count`` such cells or a constant series."""
    n = len(upstream)
    if abs(lag) >= n:
        return np.nan
    u, w = (upstream[lag:], downstream[: n - lag]) if lag >= 0 else (upstream[: n + lag], downstream[-lag:])
    both = np.isfinite(u) & np.isfinite(w)
    if both.sum() < max(min_count, 3):
        return np.nan
    u, w = u[both], w[both]
    su, sw = u.std(), w.std()
    if su == 0 or sw == 0:
        return np.nan
    return float(np.mean((u - u.mean()) * (w - w.mean())) / (su * sw))


def xcorr_wave_speed(speed: np.ndarray, x_edges: np.ndarray, t_edges: np.ndarray, cfg: XcorrConfig | None = None) -> dict:
    """Wave speed from the lag of the largest cross-correlation of the speed series of positions a fixed
    distance apart (:class:`XcorrConfig`), on the field ``speed [x cell, t cell]`` of equal time cells.

    A positive lag means the upstream cell sees what the downstream cell saw earlier: speed ``-distance / lag``
    (negative: travelling upstream). Per pair the largest local maximum of the correlation within the lag
    range counts when it reaches ``min_correlation``. Returns ``{wave_speed_xcorr, n_pairs, n_used,
    median_correlation, pair_speeds, distance}``; ``wave_speed_xcorr`` (median over the pairs) is None
    without any pair with a peak, e.g. in a field without waves (constant series have no correlation).
    """
    cfg = cfg or XcorrConfig()
    speed = np.asarray(speed, dtype=np.float64)
    x_edges, t_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(t_edges, dtype=np.float64)
    out: dict[str, Any] = {"wave_speed_xcorr": None, "n_pairs": 0, "n_used": 0, "median_correlation": None,
                           "pair_speeds": [], "distance": cfg.distance}  # fmt: skip
    if speed.ndim != 2 or speed.shape[0] < 2 or speed.shape[1] < 3:
        return out
    x_mid = 0.5 * (x_edges[1:] + x_edges[:-1])
    dt = float(np.median(np.diff(t_edges)))
    n = speed.shape[1]
    low = max(int(np.ceil(cfg.min_lag_s / dt - 1e-9)), 1)
    lags = list(range(low, int(np.floor(cfg.max_lag_s / dt + 1e-9)) + 1))
    if cfg.direction == "both":
        lags = [-lag for lag in reversed(lags)] + lags
    pairs = [(i, j) for i in range(len(x_mid)) for j in range(i + 1, len(x_mid))
             if abs(x_mid[j] - x_mid[i] - cfg.distance) < 1e-6]  # fmt: skip
    out["n_pairs"] = len(pairs)
    min_count = int(np.ceil(cfg.min_overlap * n))
    needed = sorted({lag + d for lag in lags for d in (-1, 0, 1)})  # the neighbours tell a peak from a slope
    speeds, peaks = [], []
    for i, j in pairs:
        c = {lag: _lagged_correlation(speed[i], speed[j], lag, min_count) for lag in needed}
        local = [lag for lag in lags if np.isfinite([c[lag - 1], c[lag], c[lag + 1]]).all()
                 and c[lag] >= c[lag - 1] and c[lag] >= c[lag + 1]]  # fmt: skip
        if not local:
            continue
        best = max(local, key=lambda lag: c[lag])
        if c[best] < cfg.min_correlation:
            continue
        shift = 0.0
        curvature = c[best - 1] - 2.0 * c[best] + c[best + 1]
        if cfg.refine and curvature < 0:
            shift = float(np.clip(0.5 * (c[best - 1] - c[best + 1]) / curvature, -0.5, 0.5))
        speeds.append(-cfg.distance / ((best + shift) * dt))
        peaks.append(c[best])
    if speeds:
        out.update(wave_speed_xcorr=float(np.median(speeds)), n_used=len(speeds),
                   median_correlation=float(np.median(peaks)), pair_speeds=[float(s) for s in speeds])  # fmt: skip
    return out
