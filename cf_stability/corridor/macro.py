"""Macroscopic traffic quantities from trajectories (Edie's generalised definitions) and the metrics of
the corridor (docs/m5_contract.md, section 5; D98, D99).

For a space-time cell of size ``dx * dt``: flow ``q = sum(distance) / (dx * dt)``, density
``k = sum(time) / (dx * dt)``, speed ``v = q / k``. SI units: veh/s, veh/m, m/s. The corridor metrics
(:func:`corridor_metrics`) report flows in veh/h and densities in veh/km where their names say so,
and every value that cannot be computed (no vehicle, no crossing, no wave) as None.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from cf_stability.corridor.waves import WaveConfig, find_waves, speed_field, xcorr_wave_speed
from cf_stability.utils import config_hash, read_json, write_json

_ARRAY_KEYS = ("vehicle", "t", "x")


def _arrays(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all(k in data for k in _ARRAY_KEYS):
        veh, t, x = (np.asarray(data[k]) for k in _ARRAY_KEYS)
        lane = np.asarray(data["lane"]) if "lane" in data else np.zeros(len(t), dtype=np.int64)
        return veh, np.asarray(t, dtype=np.float64), np.asarray(x, dtype=np.float64), lane
    tracks = list(data.values())
    if not tracks:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    return (
        np.concatenate([np.full(len(tr.t), i) for i, tr in enumerate(tracks)]),
        np.concatenate([tr.t for tr in tracks]).astype(np.float64),
        np.concatenate([tr.x for tr in tracks]).astype(np.float64),
        np.concatenate([tr.lane for tr in tracks]),
    )


def _pieces(ta, tb, xa, xb, x_edges, t_edges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every straight segment (ta, xa) -> (tb, xb) at the grid lines it crosses.

    Returns (segment index, lam0, lam1) of the pieces, ``lam`` being the segment parameter in [0, 1].
    """
    n = len(ta)
    seg = [np.arange(n), np.arange(n)]
    lam = [np.zeros(n), np.ones(n)]
    for a, b, edges in ((xa, xb, x_edges), (ta, tb, t_edges)):
        ia, ib = np.searchsorted(edges, a, side="right"), np.searchsorted(edges, b, side="right")
        count = np.abs(ib - ia)  # grid lines strictly between a and b (or at b)
        rep = np.repeat(np.arange(n), count)
        k = np.repeat(np.minimum(ia, ib), count) + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        seg.append(rep)
        lam.append((edges[k] - a[rep]) / (b[rep] - a[rep]))
    seg, lam = np.concatenate(seg), np.concatenate(lam)
    order = np.lexsort((lam, seg))
    seg, lam = seg[order], lam[order]
    same = seg[1:] == seg[:-1]
    return seg[:-1][same], lam[:-1][same], lam[1:][same]


def _quantities(distance: np.ndarray, time: np.ndarray, area: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(time > 0, distance / time, np.nan)
    return {"distance": distance, "time": time, "flow": distance / area, "density": time / area, "speed": speed}


def edie_grid(
    tracks_or_arrays: Mapping[str, Any],
    x_edges: np.ndarray,
    t_edges: np.ndarray,
    lanes: Sequence[int] | None = None,
) -> dict:
    """Edie flow, density and speed per lane and aggregated over the lanes.

    ``tracks_or_arrays``: a mapping with equal-length arrays ``vehicle``, ``t``, ``x`` (and
    optionally ``lane``), or a mapping of track-like objects with attributes ``t``, ``x``, ``lane``.
    The samples of one vehicle form one trajectory, linear between consecutive samples; a segment
    belongs to the lane of its first sample. Arrays in the result have shape
    ``[lane, x cell, t cell]`` (``per_lane``) and ``[x cell, t cell]`` (``total``).
    """
    x_edges, t_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(t_edges, dtype=np.float64)
    veh, t, x, lane = _arrays(tracks_or_arrays)
    order = np.lexsort((t, veh))
    veh, t, x, lane = veh[order], t[order], x[order], lane[order]
    lane_ids = np.unique(lane) if lanes is None else np.asarray(sorted(lanes))
    keep = (veh[1:] == veh[:-1]) & np.isin(lane[:-1], lane_ids)
    ta, tb, xa, xb = t[:-1][keep], t[1:][keep], x[:-1][keep], x[1:][keep]
    li = np.searchsorted(lane_ids, lane[:-1][keep])

    seg, lam0, lam1 = _pieces(ta, tb, xa, xb, x_edges, t_edges)
    mid = 0.5 * (lam0 + lam1)
    ix = np.searchsorted(x_edges, xa[seg] + mid * (xb[seg] - xa[seg]), side="right") - 1
    it = np.searchsorted(t_edges, ta[seg] + mid * (tb[seg] - ta[seg]), side="right") - 1
    nx, nt = len(x_edges) - 1, len(t_edges) - 1
    inside = (ix >= 0) & (ix < nx) & (it >= 0) & (it < nt)
    seg, frac, ix, it = seg[inside], (lam1 - lam0)[inside], ix[inside], it[inside]

    shape = (len(lane_ids), nx, nt)
    cell = np.ravel_multi_index((li[seg], ix, it), shape)
    distance = np.bincount(cell, frac * (xb - xa)[seg], minlength=np.prod(shape)).reshape(shape)
    time = np.bincount(cell, frac * (tb - ta)[seg], minlength=np.prod(shape)).reshape(shape)
    area = np.diff(x_edges)[:, None] * np.diff(t_edges)[None, :]
    return {
        "x_edges": x_edges,
        "t_edges": t_edges,
        "lanes": lane_ids,
        "per_lane": _quantities(distance, time, area[None]),
        "total": _quantities(distance.sum(axis=0), time.sum(axis=0), area),
    }


def _compare_level(raw: Mapping[str, np.ndarray], rec: Mapping[str, np.ndarray], min_vehicle_seconds: float) -> dict:
    mask = (raw["time"] >= min_vehicle_seconds) & (rec["time"] >= min_vehicle_seconds)
    va, vb = raw["speed"][mask], rec["speed"][mask]

    def weighted_rel_error(name: str) -> float:
        a, b = raw[name][mask], rec[name][mask]
        return float(np.sum(np.abs(b - a)) / np.sum(np.abs(a))) if mask.any() else float("nan")

    return {
        "rel_diff_distance": float(rec["distance"].sum() / raw["distance"].sum() - 1.0),
        "rel_diff_time": float(rec["time"].sum() / raw["time"].sum() - 1.0),
        "n_cells": int(mask.sum()),
        "speed_rmse": float(np.sqrt(np.mean((vb - va) ** 2))) if mask.any() else float("nan"),
        "speed_corr": float(np.corrcoef(va, vb)[0, 1]) if mask.sum() > 1 else float("nan"),
        "flow_rel_error": weighted_rel_error("flow"),
        "density_rel_error": weighted_rel_error("density"),
    }


def compare_macro(raw: Mapping[str, Any], reconstructed: Mapping[str, Any], min_vehicle_seconds: float = 5.0) -> dict:
    """Agreement of two :func:`edie_grid` results on the same grid (``total`` and ``per_lane``).

    Cell-level measures use the cells with at least ``min_vehicle_seconds`` in both inputs; the
    relative errors of flow and density are ``sum|rec - raw| / sum|raw|`` over those cells.
    """
    return {level: _compare_level(raw[level], reconstructed[level], min_vehicle_seconds) for level in ("total", "per_lane")}


# ============================================================================ corridor metrics (M5)

COMPONENTS = (
    "throughput", "mean_speed", "queue_discharge_flow", "fd", "wave_speed", "n_waves", "wave_amplitude", "travel_time",
)  # fmt: skip
# version of the definitions, hashed with the parameters: raise it when the code changes a metric, so that
# every macro.json written before is recomputed (2: collision episodes of vehicles that stay, wave_speed_xcorr;
# 3: episodes of collisions.npz with their time and zone; 4: macro_error_dynamic)
METRICS_VERSION = 4
# the components that follow the demand and the boundary of the data, and those the law decides (M6)
ANCHORED = ("throughput", "mean_speed", "queue_discharge_flow", "travel_time")
DYNAMIC = {"fd": "fd", "wave_speed": "wave_speed", "waves": "n_waves", "wave_amplitude": "wave_amplitude"}


@dataclasses.dataclass(frozen=True)
class Geometry:
    """Section and analysis window of a scenario: ``x_in <= x <= x_out`` (m), ``window`` (s from the period start).

    The values come from ``scenario.json``; these defaults are used only where it has none."""

    x_in: float = 20.0
    x_out: float = 500.0
    window: tuple[float, float] = (180.0, 840.0)

    def as_dict(self) -> dict[str, Any]:
        return {"x_in": self.x_in, "x_out": self.x_out, "window": list(self.window)}


@dataclasses.dataclass(frozen=True)
class MacroConfig:
    """Parameters of the corridor metrics (``configs/corridor_metrics.yaml``, ``macro``)."""

    detectors: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 450.0)  # m
    interval_s: float = 30.0  # counting interval of the detectors
    throughput_x: float = 450.0  # m, detector of the throughput and of the discharge flow
    queue_x: float = 100.0  # m, detector whose speed marks the congested intervals
    queue_speed: float = 11.0  # m/s, congested below this speed
    capacity_window_s: float = 120.0  # the largest flow over this many seconds is the pre-drop capacity
    cell_dx: float = 100.0  # m, Edie cells (grid lines at multiples of it inside the section)
    cell_dt: float = 30.0  # s
    fd_bin: float = 10.0  # veh/km, density bins of the fundamental diagram
    travel_margin: float = 10.0  # m, travel time from x_in + margin to x_out - margin
    collision_entry_m: float = 20.0  # m, collisions_by_zone: episodes begun at x < x_in + this (the insertion) ...
    collision_exit_m: float = 50.0  # ... and at x > x_out - this (the downstream boundary)
    max_gap_s: float = 1.5  # s, consecutive samples of a vehicle further apart are not joined (teleports)
    boundary_points: bool = True  # close the trajectories with (depart, entry_x) and (exit_t, x_out)
    boundary_max_gap_s: float = 10.0  # s, largest gap to such a boundary point
    waves: WaveConfig = dataclasses.field(default_factory=WaveConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MacroConfig:
        values = dict(raw or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown macro keys: {sorted(unknown)}")
        if "detectors" in values:
            values["detectors"] = tuple(float(x) for x in values["detectors"])
        values["waves"] = WaveConfig.from_mapping(values.get("waves"))
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["detectors"] = list(self.detectors)
        out["waves"]["lanes"] = list(self.waves.lanes)
        return out

    def hashed(self) -> dict[str, Any]:
        """``config`` of a macro.json: the parameters and the version of the definitions."""
        return {**self.as_dict(), "version": METRICS_VERSION}


def _merged(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        out[key] = _merged(out[key], value) if isinstance(value, Mapping) and isinstance(out.get(key), Mapping) else value
    return out


def scenario_macro_config(cfg: MacroConfig, scenario: Mapping[str, Any]) -> MacroConfig:
    """``cfg`` with the settings of ``scenario.json["metrics"]`` over it (nested keys merged): a site whose geometry
    differs from that of I-80 names its detectors, throughput and queue positions and the lanes of the wave field
    there (``configs/corridor/<site>.yaml``, ``metrics``; D112). Unchanged without that block, so the
    ``macro.json`` files of I-80 keep their hash."""
    overrides = scenario.get("metrics") if isinstance(scenario, Mapping) else None
    if not overrides:
        return cfg
    return MacroConfig.from_mapping(_merged(cfg.as_dict(), overrides))


# ----------------------------------------------------------------------------------------- inputs


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Every array of an ``.npz`` file, loaded into memory."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _find(mapping: Any, names: Sequence[str], convert: Any) -> Any:
    """First value of one of ``names`` in ``mapping`` or its nested mappings (breadth first) that ``convert``
    accepts (returns not None), converted; None if there is none."""
    queue = [mapping]
    while queue:
        item = queue.pop(0)
        if not isinstance(item, Mapping):
            continue
        for name in names:
            value = convert(item.get(name))
            if value is not None:
                return value
        queue.extend(value for value in item.values() if isinstance(value, Mapping))
    return None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _pair(value: Any) -> tuple[float, float] | None:
    """A window ``[start, end]`` or ``{start, end}``; anything else (e.g. a scalar averaging window) is no window."""
    if isinstance(value, Mapping):
        value = (value.get("start", value.get("t0")), value.get("end", value.get("t1")))
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value):
        return float(value[0]), float(value[1])
    return None


def scenario_geometry(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """The values of ``x_in``, ``x_out`` and the analysis window that a ``scenario.json`` holds (top level or
    nested, e.g. under ``geometry``); absent ones are missing from the result."""
    found = {
        "x_in": _find(scenario, ("x_in",), _number),
        "x_out": _find(scenario, ("x_out",), _number),
        "window": _find(scenario, ("analysis_window", "metrics_window", "window", "window_s"), _pair),
    }
    return {key: value for key, value in found.items() if value is not None}


def geometry_from_scenario(scenario: Mapping[str, Any], default: Geometry | None = None) -> Geometry:
    """Geometry of a ``scenario.json`` (:func:`scenario_geometry`); a missing value is taken from ``default``."""
    found = scenario_geometry(scenario)
    return dataclasses.replace(default or Geometry(), **found)


def prepare_trajectories(
    trajectories: Mapping[str, Any], vehicles: Mapping[str, Any] | None, geometry: Geometry, cfg: MacroConfig
) -> dict[str, np.ndarray]:
    """Rows of ``trajectories.npz`` sorted by vehicle and time, with pieces and boundary points.

    ``piece`` numbers the stretches of a vehicle without a gap longer than ``max_gap_s`` (a teleport
    is not a trajectory). With ``boundary_points`` the insertion ``(depart, entry_x)`` and the passage
    of ``x_out`` at ``exit_t`` from ``vehicles.npz`` are added as rows (``virtual``) when they lie
    before the first and after the last sample within ``boundary_max_gap_s``: the 1 Hz samples stop up
    to one second short of both ends, which would otherwise lose the travel time of every vehicle
    faster than ``travel_margin`` per second and bias the end cells of the Edie grid.
    """
    n_rows = len(np.asarray(trajectories.get("t", [])))
    t = np.asarray(trajectories.get("t", np.zeros(0)), dtype=np.float64).reshape(-1)
    x = np.asarray(trajectories.get("x", np.zeros(0)), dtype=np.float64).reshape(-1)
    veh = np.asarray(trajectories.get("vehicle", np.zeros(0)), dtype=np.int64).reshape(-1)
    lane = np.asarray(trajectories.get("lane", np.zeros(n_rows)), dtype=np.int64).reshape(-1)
    v = np.asarray(trajectories.get("v", np.full(n_rows, np.nan)), dtype=np.float64).reshape(-1)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x, veh, lane, v = t[ok], x[ok], veh[ok], lane[ok], v[ok]
    virtual = np.zeros(len(t), dtype=bool)
    order = np.lexsort((t, veh))
    t, x, veh, lane, v = t[order], x[order], veh[order], lane[order], v[order]

    if cfg.boundary_points and vehicles is not None and len(t):
        extra = _boundary_rows(t, x, veh, lane, v, vehicles, geometry, cfg)
        if extra is not None:
            t, x, veh, lane, v = (np.concatenate((a, b)) for a, b in zip((t, x, veh, lane, v), extra))
            virtual = np.concatenate((virtual, np.ones(len(extra[0]), dtype=bool)))
            order = np.lexsort((t, veh))
            t, x, veh, lane, v, virtual = t[order], x[order], veh[order], lane[order], v[order], virtual[order]

    new = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        gap = np.diff(t) > cfg.max_gap_s
        new[1:] = (veh[1:] != veh[:-1]) | (gap & ~virtual[1:] & ~virtual[:-1])
    piece = np.cumsum(new) - 1
    return {"t": t, "x": x, "v": v, "lane": lane, "vehicle": veh, "piece": piece, "virtual": virtual}


def _boundary_rows(
    t: np.ndarray, x: np.ndarray, veh: np.ndarray, lane: np.ndarray, v: np.ndarray, vehicles: Mapping[str, Any],
    geometry: Geometry, cfg: MacroConfig,
) -> tuple[np.ndarray, ...] | None:  # fmt: skip
    """Rows of the insertion and of the passage of ``x_out`` (see :func:`prepare_trajectories`); the samples
    are sorted by vehicle and time."""
    first = np.flatnonzero(np.r_[True, veh[1:] != veh[:-1]])
    last = np.r_[first[1:] - 1, len(t) - 1]
    ids = veh[first]
    n_vehicles = len(np.asarray(vehicles.get("depart", [])))
    known = (ids >= 0) & (ids < n_vehicles)
    first, last, ids = first[known], last[known], ids[known]
    if not len(ids):
        return None

    def column(name: str) -> np.ndarray:
        values = vehicles.get(name)
        return np.full(len(ids), np.nan) if values is None else np.asarray(values, dtype=np.float64)[ids]

    depart, entry_x, entry_lane, exit_t = (column(k) for k in ("depart", "entry_x", "entry_lane", "exit_t"))
    gap = cfg.boundary_max_gap_s
    with np.errstate(invalid="ignore"):
        enter = (np.isfinite(depart) & np.isfinite(entry_x) & (depart < t[first]) & (t[first] - depart <= gap)
                 & (entry_x <= x[first]))  # fmt: skip
        leave = (np.isfinite(exit_t) & (exit_t > t[last]) & (exit_t - t[last] <= gap)
                 & (x[last] < geometry.x_out))  # fmt: skip
    lane_in = np.where(np.isfinite(entry_lane) & (entry_lane > 0), entry_lane, lane[first]).astype(np.int64)
    rows = (
        np.r_[depart[enter], exit_t[leave]],
        np.r_[entry_x[enter], np.full(leave.sum(), geometry.x_out)],
        np.r_[ids[enter], ids[leave]],
        np.r_[lane_in[enter], lane[last][leave]],
        np.r_[v[first][enter], v[last][leave]],  # speed of the neighbouring sample: only positions matter here
    )
    return rows if len(rows[0]) else None


# ------------------------------------------------------------------------------------ crossings


def crossings(prepared: Mapping[str, np.ndarray], x: float) -> dict[str, np.ndarray]:
    """Passages of the position ``x``: every segment of a piece with ``x_a < x <= x_b``, linear in time.

    Returns ``vehicle``, ``t`` (s), ``v`` (m/s, the sampled speeds interpolated at the passage; the
    segment speed where a sample has no speed) and ``lane`` (of the first sample of the segment).
    """
    t, xs, v, piece = prepared["t"], prepared["x"], prepared["v"], prepared["piece"]
    if len(t) < 2:
        empty = np.zeros(0)
        return {"vehicle": empty.astype(np.int64), "t": empty, "v": empty, "lane": empty.astype(np.int64)}
    hit = np.flatnonzero((piece[1:] == piece[:-1]) & (xs[:-1] < x) & (xs[1:] >= x))
    lam = (x - xs[hit]) / (xs[hit + 1] - xs[hit])
    dt = t[hit + 1] - t[hit]
    tc = t[hit] + lam * dt
    vc = v[hit] + lam * (v[hit + 1] - v[hit])
    segment = (xs[hit + 1] - xs[hit]) / np.where(dt > 0, dt, np.nan)
    vc = np.where(np.isfinite(vc), vc, segment)
    return {"vehicle": prepared["vehicle"][hit], "t": tc, "v": vc, "lane": prepared["lane"][hit]}


def interval_edges(start: float, stop: float, step: float) -> np.ndarray:
    """Edges ``start, start + step, ...`` of the whole intervals within ``[start, stop]``."""
    n = int(math.floor((stop - start) / step + 1e-9))
    return start + step * np.arange(n + 1, dtype=np.float64)


def detector(prepared: Mapping[str, np.ndarray], x: float, t_edges: np.ndarray) -> dict[str, Any]:
    """Vehicles passing ``x`` per interval of ``t_edges``: counts, flow (veh/h) and mean speed of the
    passing vehicles (arithmetic mean of their speeds at the passage; NaN in an interval without any)."""
    c = crossings(prepared, x)
    n = len(t_edges) - 1
    idx = np.searchsorted(t_edges, c["t"], side="right") - 1
    ok = (idx >= 0) & (idx < n)
    count = np.bincount(idx[ok], minlength=n)[:n] if n > 0 else np.zeros(0, np.int64)
    speed_sum = np.bincount(idx[ok], weights=c["v"][ok], minlength=n)[:n] if n > 0 else np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(count > 0, speed_sum / np.maximum(count, 1), np.nan)
    return {"count": count, "flow_vph": count / np.diff(t_edges) * 3600.0, "speed": speed, "crossings": c}


def travel_times(prepared: Mapping[str, np.ndarray], x_a: float, x_b: float, window: Sequence[float]) -> np.ndarray:
    """Sorted travel times (s) from ``x_a`` to ``x_b`` of the vehicles that pass both inside ``window``."""
    a, b = crossings(prepared, x_a), crossings(prepared, x_b)
    ids_a, first_a = np.unique(a["vehicle"], return_index=True)  # a vehicle passes a position once; keep the first
    ids_b, first_b = np.unique(b["vehicle"], return_index=True)
    _, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    ta, tb = a["t"][first_a[ia]], b["t"][first_b[ib]]
    keep = (ta >= window[0]) & (tb <= window[1]) & (tb > ta)
    return np.sort(tb[keep] - ta[keep])


# ------------------------------------------------------------------------------------- the metrics


def _num(value: Any) -> float | None:
    """Finite float, else None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _list(values: Any) -> list:
    """Nested lists of finite floats (None for NaN)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return _num(array)  # type: ignore[return-value]
    return [_list(row) for row in array] if array.ndim > 1 else [_num(x) for x in array]


def fundamental_diagram(flow_vph: np.ndarray, density_vpkm: np.ndarray, bin_width: float) -> dict[str, Any]:
    """Mean flow per density bin of the cells (lanes together) with counts; pooled scatter within the bins.

    ``density_bins`` are the lower edges (veh/km) of the bins that hold a cell; ``flow_std`` the standard
    deviation of the flow in every bin (None with one cell); ``scatter`` the pooled within-bin standard
    deviation ``sqrt(sum (q - mean_bin)^2 / sum (n_bin - 1))`` over the bins with two cells or more.
    """
    q, k = np.asarray(flow_vph, dtype=np.float64).ravel(), np.asarray(density_vpkm, dtype=np.float64).ravel()
    ok = np.isfinite(q) & np.isfinite(k)
    q, k = q[ok], k[ok]
    out: dict[str, Any] = {"bin_width": bin_width, "density_bins": [], "flow": [], "count": [], "flow_std": [],
                           "n_points": int(len(q)), "mean_flow": _num(q.mean()) if len(q) else None}  # fmt: skip
    if not len(q):
        out["scatter"] = None
        return out
    index = np.floor(k / bin_width + 1e-9).astype(np.int64)
    squares, dof = 0.0, 0
    for b in np.unique(index):
        values = q[index == b]
        out["density_bins"].append(float(b * bin_width))
        out["flow"].append(float(values.mean()))
        out["count"].append(int(len(values)))
        out["flow_std"].append(float(values.std(ddof=1)) if len(values) > 1 else None)
        squares += float(np.sum((values - values.mean()) ** 2))
        dof += len(values) - 1
    out["scatter"] = math.sqrt(squares / dof) if dof > 0 else None
    return out


def capacity_drop(queue_speed: np.ndarray, discharge_flow: np.ndarray, threshold: float, window_intervals: int) -> dict:
    """Queue discharge flow (veh/h): mean flow downstream over the intervals whose upstream speed is below
    ``threshold``; capacity drop ``1 - discharge / largest flow over window_intervals consecutive intervals``."""
    speed, flow = np.asarray(queue_speed, dtype=np.float64), np.asarray(discharge_flow, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        congested = np.isfinite(speed) & (speed < threshold)
    discharge = float(flow[congested].mean()) if congested.any() else None
    largest = None
    if 0 < window_intervals <= len(flow):
        largest = float(np.convolve(flow, np.ones(window_intervals) / window_intervals, mode="valid").max())
    drop = None
    if discharge is not None and largest is not None and largest > 0:
        drop = 1.0 - discharge / largest
    return {"queue_discharge_flow": discharge, "largest_flow": largest, "capacity_drop": drop,
            "n_congested": int(congested.sum())}  # fmt: skip


def _describe(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "values": []}
    p10, p50, p90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(p50), "p10": float(p10),
            "p90": float(p90), "values": [float(x) for x in values]}  # fmt: skip


def _demand(vehicles: Mapping[str, Any] | None, window: Sequence[float]) -> dict[str, Any]:
    """Share of the vehicles planned inside the window that were inserted, and their mean insertion delay."""
    if vehicles is None or "depart_planned" not in vehicles:
        return {"n_planned": 0, "inserted_share": None, "mean_depart_delay_s": None}
    planned = np.asarray(vehicles["depart_planned"], dtype=np.float64)
    depart = np.asarray(vehicles.get("depart", np.full(len(planned), np.nan)), dtype=np.float64)
    inside = np.isfinite(planned) & (planned >= window[0]) & (planned < window[1])
    inserted = inside & np.isfinite(depart)
    return {
        "n_planned": int(inside.sum()),
        "inserted_share": float(inserted.sum() / inside.sum()) if inside.any() else None,
        "mean_depart_delay_s": float((depart[inserted] - planned[inserted]).mean()) if inserted.any() else None,
    }


def _presence(prepared: Mapping[str, np.ndarray], window: Sequence[float]) -> tuple[np.ndarray, ...]:
    """Per vehicle with samples (sorted ids): its time in the section inside the window, its whole time in
    the section, whether a sample lies inside the window, and its last sample time (boundary points count
    for the times, not as samples)."""
    veh, t, piece, virtual = prepared["vehicle"], prepared["t"], prepared["piece"], prepared["virtual"]
    ids, index = np.unique(veh, return_inverse=True)
    inside, total = np.zeros(len(ids)), np.zeros(len(ids))
    if len(t) > 1:
        same = piece[1:] == piece[:-1]
        ta, tb, owner = t[:-1][same], t[1:][same], index[:-1][same]
        total = np.bincount(owner, tb - ta, minlength=len(ids))
        inside = np.bincount(owner, np.clip(np.minimum(tb, window[1]) - np.maximum(ta, window[0]), 0.0, None),
                             minlength=len(ids))  # fmt: skip
    real = ~virtual
    sampled = np.bincount(index[real], (t[real] >= window[0]) & (t[real] <= window[1]), minlength=len(ids)) > 0
    last = np.full(len(ids), -np.inf)
    np.maximum.at(last, index[real], t[real])
    return ids, inside, total, sampled, np.where(np.isfinite(last), last, np.nan)


def collisions(
    prepared: Mapping[str, np.ndarray], vehicles: Mapping[str, Any] | None, window: Sequence[float],
    episodes: Mapping[str, Any] | None = None, zones: Sequence[float] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Collision episodes of the window and the share of the vehicles of the window with an episode.

    ``episodes`` (``collisions.npz``: ``t``, ``vehicle``, ``x``, ``lane`` of the start of every contact
    episode) counts the episodes that begin inside the window, and by zone (``zones = (x_a, x_b)``: begun
    at ``x < x_a``, at ``x > x_b``, elsewhere); a vehicle is in contact when one of its episodes begins in
    the window. Without it ``vehicles.npz`` gives the episodes per vehicle: ``n_collisions`` (the vehicle
    stays) counts in the window in proportion to the share of the vehicle's time in the section that lies
    inside the window (a vehicle without samples: all when it departed inside the window); ``collided``
    of the first loop (the follower was removed) counts when its last sample lies in the window; a vehicle
    is in contact with an episode at any time, and there are no zones. The vehicles of the window are
    those with time or a sample inside it.
    """
    out: dict[str, Any] = {"collision_episodes": 0.0, "vehicles_in_contact_share": None, "n_vehicles_window": 0,
                           "collision_source": None, "collisions_by_zone": None}  # fmt: skip
    ids, inside, total, sampled, last = _presence(prepared, window)
    present = (inside > 0) | sampled
    out["n_vehicles_window"] = int(present.sum())
    if episodes is not None and "t" in episodes:
        t = np.asarray(episodes["t"], dtype=np.float64).reshape(-1)
        x = np.asarray(episodes.get("x", np.full(len(t), np.nan)), dtype=np.float64).reshape(-1)
        who = np.asarray(episodes.get("vehicle", np.full(len(t), -1)), dtype=np.int64).reshape(-1)
        with np.errstate(invalid="ignore"):
            now = np.isfinite(t) & (t >= window[0]) & (t < window[1])
            entry = now & (x < zones[0]) if zones is not None else np.zeros(len(t), dtype=bool)
            exit_ = now & (x > zones[1]) if zones is not None else np.zeros(len(t), dtype=bool)
        out.update(collision_episodes=float(now.sum()), collision_source="collisions.npz")
        if zones is not None:
            out["collisions_by_zone"] = {"entry": int(entry.sum()), "exit": int(exit_.sum()),
                                         "elsewhere": int((now & ~entry & ~exit_).sum())}  # fmt: skip
        if present.any():
            out["vehicles_in_contact_share"] = float(np.isin(ids[present], who[now]).mean())
        return out
    if vehicles is None or not ("n_collisions" in vehicles or "collided" in vehicles):
        if present.any():
            out["vehicles_in_contact_share"] = 0.0 if vehicles is not None else None
        return out
    source = "n_collisions" if "n_collisions" in vehicles else "collided"
    episodes = np.nan_to_num(np.asarray(vehicles[source], dtype=np.float64).reshape(-1))
    depart = np.asarray(vehicles.get("depart", np.full(len(episodes), np.nan)), dtype=np.float64).reshape(-1)
    known = ids < len(episodes)
    mine = np.zeros(len(ids))
    mine[known] = episodes[ids[known]]
    if source == "collided":  # the collision began within a second after the last sample
        counted = float(np.sum(mine[(last >= window[0]) & (last < window[1])]))
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, inside / total, sampled.astype(float))
        counted = float(np.sum(mine * share))
    unsampled = np.setdiff1d(np.flatnonzero(episodes > 0), ids)  # removed or in contact before their first sample
    with np.errstate(invalid="ignore"):
        counted += float(np.sum(episodes[unsampled][(depart[unsampled] >= window[0]) & (depart[unsampled] < window[1])]))
    out.update(collision_episodes=counted, collision_source=source)
    if present.any():
        out["vehicles_in_contact_share"] = float(np.mean(mine[present] >= 1))
    return out


def edie_summary(grid: Mapping[str, Any]) -> dict[str, Any]:
    """The Edie grid in report units: flow veh/h, density veh/km, speed m/s, lanes together and per lane."""

    def level(q: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {"flow_vph": _list(q["flow"] * 3600.0), "density_vpkm": _list(q["density"] * 1000.0),
                "speed": _list(q["speed"])}  # fmt: skip

    per_lane = grid["per_lane"]
    return {
        "x_edges": _list(grid["x_edges"]),
        "t_edges": _list(grid["t_edges"]),
        "total": level(grid["total"]),
        "per_lane": {
            str(int(lane)): level({key: per_lane[key][i] for key in ("flow", "density", "speed")})
            for i, lane in enumerate(grid["lanes"])
        },
    }


def section_edges(x_in: float, x_out: float, step: float) -> np.ndarray:
    """``x_in``, the multiples of ``step`` strictly inside the section, ``x_out``: cells aligned with the detectors."""
    inner = step * np.arange(math.floor(x_in / step) + 1, math.ceil(x_out / step))
    return np.unique(np.r_[x_in, inner[(inner > x_in) & (inner < x_out)], x_out])


def corridor_metrics(
    trajectories: Mapping[str, Any], vehicles: Mapping[str, Any] | None, geometry: Geometry,
    cfg: MacroConfig | None = None, episodes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Metrics of section 5 from the arrays of ``trajectories.npz``, ``vehicles.npz`` and, where the run has
    it, ``collisions.npz`` (``episodes``) (the content of ``macro.json`` without ``macro_error``, ``config``
    and ``config_hash``)."""
    cfg = cfg or MacroConfig()
    w0, w1 = (float(w) for w in geometry.window)
    prepared = prepare_trajectories(trajectories, vehicles, geometry, cfg)
    arrays = {"vehicle": prepared["piece"], "t": prepared["t"], "x": prepared["x"], "lane": prepared["lane"]}

    # detectors: counts per interval and the throughput over the whole window
    t_det = interval_edges(w0, w1, cfg.interval_s)
    positions = sorted({*cfg.detectors, cfg.throughput_x, cfg.queue_x})
    series = {x: detector(prepared, x, t_det) for x in positions}
    detectors = {
        f"{x:g}": {"t_start": _list(t_det[:-1]), "count": [int(c) for c in s["count"]],
                   "flow_vph": _list(s["flow_vph"]), "speed": _list(s["speed"])}
        for x, s in series.items()
    }  # fmt: skip
    passing = series[cfg.throughput_x]["crossings"]
    in_window = (passing["t"] >= w0) & (passing["t"] < w1)
    hours = (w1 - w0) / 3600.0
    lanes, per_lane = np.unique(passing["lane"][in_window], return_counts=True)

    # Edie grid, mean speed and fundamental diagram
    grid = edie_grid(arrays, section_edges(geometry.x_in, geometry.x_out, cfg.cell_dx), interval_edges(w0, w1, cfg.cell_dt))
    total = grid["total"]
    distance, time = float(total["distance"].sum()), float(total["time"].sum())
    fd = fundamental_diagram(total["flow"] * 3600.0, total["density"] * 1000.0, cfg.fd_bin)
    drop = capacity_drop(
        series[cfg.queue_x]["speed"], series[cfg.throughput_x]["flow_vph"], cfg.queue_speed,
        int(round(cfg.capacity_window_s / cfg.interval_s)),
    )  # fmt: skip

    # stop waves on the fine speed field of the main lanes
    wc = cfg.waves
    wave_grid = edie_grid(arrays, interval_edges(geometry.x_in, geometry.x_out, wc.dx), interval_edges(w0, w1, wc.dt),
                          lanes=wc.lanes)  # fmt: skip
    distance_cells, time_cells = wave_grid["total"]["distance"], wave_grid["total"]["time"]
    field = speed_field(distance_cells, time_cells, wc.smooth, wc.min_vehicle_seconds)
    waves = find_waves(field, wave_grid["x_edges"], wave_grid["t_edges"], wc)
    xcorr_field = speed_field(distance_cells, time_cells, wc.xcorr.smooth, wc.min_vehicle_seconds)
    xcorr = xcorr_wave_speed(xcorr_field, wave_grid["x_edges"], wave_grid["t_edges"], wc.xcorr)

    x_a, x_b = geometry.x_in + cfg.travel_margin, geometry.x_out - cfg.travel_margin
    zones = (geometry.x_in + cfg.collision_entry_m, geometry.x_out - cfg.collision_exit_m)
    contact = collisions(prepared, vehicles, (w0, w1), episodes, zones)
    vehicle_km = distance / 1000.0
    return {
        "geometry": geometry.as_dict(),
        "window": [w0, w1],
        "n_vehicles": int(len(np.unique(prepared["vehicle"]))),
        "detectors": detectors,
        "throughput_vph": float(in_window.sum() / hours) if hours > 0 else None,
        "throughput_vph_per_lane": {str(int(k)): float(c / hours) for k, c in zip(lanes, per_lane)} if hours > 0 else {},
        "mean_speed": distance / time if time > 0 else None,
        "vehicle_km": vehicle_km,
        "vehicle_hours": time / 3600.0,
        "edie": edie_summary(grid),
        "fd": {key: value for key, value in fd.items() if key != "scatter"},
        "fd_scatter": fd["scatter"],
        "queue_discharge_flow": drop["queue_discharge_flow"],
        "flow_peak_2min": drop["largest_flow"],
        "capacity_drop": drop["capacity_drop"],
        "n_congested_intervals": drop["n_congested"],
        "congested_share": drop["n_congested"] / (len(t_det) - 1) if len(t_det) > 1 else None,
        "waves": {
            "n_waves": waves["n_waves"], "wave_speed": waves["wave_speed"], "wave_amplitude": waves["wave_amplitude"],
            "wave_speed_xcorr": xcorr["wave_speed_xcorr"],
            "xcorr": {key: value for key, value in xcorr.items() if key != "wave_speed_xcorr"},
            "per_wave": waves["per_wave"], "median_speed": _list(waves["median_speed"]),
        },  # fmt: skip
        "travel_time": {"from_x": x_a, "to_x": x_b, **_describe(travel_times(prepared, x_a, x_b, (w0, w1)))},
        **contact,
        "collision_zones_m": list(zones),  # entry: x below the first, exit: x above the second
        "collisions_per_1000_vkm": contact["collision_episodes"] / vehicle_km * 1000.0 if vehicle_km > 0 else None,
        **_demand(vehicles, (w0, w1)),
    }


# ------------------------------------------------------------------------------------ macro error


def _relative(run: Any, truth: Any) -> float | None:
    """``(run - truth) / |truth|``; None when a value is missing or the truth is 0."""
    run, truth = _num(run), _num(truth)
    if run is None or truth is None or truth == 0:
        return None
    return (run - truth) / abs(truth)


def fd_error(run_fd: Mapping[str, Any], truth_fd: Mapping[str, Any]) -> float | None:
    """RMSE of the binned flow over the bins both diagrams have, over the mean flow of the cells of the truth."""
    run_bins = dict(zip(run_fd.get("density_bins", []), run_fd.get("flow", [])))
    truth_bins = dict(zip(truth_fd.get("density_bins", []), truth_fd.get("flow", [])))
    common = sorted(set(run_bins) & set(truth_bins))
    scale = _num(truth_fd.get("mean_flow"))
    if not common or not scale:
        return None
    diff = np.array([run_bins[b] - truth_bins[b] for b in common], dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)) / scale)


def macro_error(run: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
    """Macro-error vector of a run against the ground truth of its scenario (D99).

    Relative errors ``(run - truth) / |truth|`` of throughput, mean speed, queue discharge flow, wave
    speed (the cross-correlation estimate ``wave_speed_xcorr``; the speed of the leading edges is
    reported beside it, not in the vector) and wave amplitude; number of waves over ``max(truth, 1)``;
    fundamental diagram by :func:`fd_error`; travel time: Wasserstein-1 distance of the travel times over
    their mean in the truth. ``value``: mean of the absolute components that exist (None without any).
    Metrics of two different analysis windows are not compared: no component, and ``error`` says why.
    """
    if run.get("window") is not None and truth.get("window") is not None and \
            [float(w) for w in run["window"]] != [float(w) for w in truth["window"]]:  # fmt: skip
        return {"components": dict.fromkeys(COMPONENTS), "value": None, "n_components": 0,
                "error": f"window {run['window']} of the run, {truth['window']} of the ground truth"}  # fmt: skip
    rw, tw = run.get("waves") or {}, truth.get("waves") or {}
    rt, tt = run.get("travel_time") or {}, truth.get("travel_time") or {}
    n_run, n_truth = rw.get("n_waves"), tw.get("n_waves")
    travel = None
    if rt.get("values") and tt.get("values") and _num(tt.get("mean")):
        travel = float(sps.wasserstein_distance(rt["values"], tt["values"]) / tt["mean"])
    components = {
        "throughput": _relative(run.get("throughput_vph"), truth.get("throughput_vph")),
        "mean_speed": _relative(run.get("mean_speed"), truth.get("mean_speed")),
        "queue_discharge_flow": _relative(run.get("queue_discharge_flow"), truth.get("queue_discharge_flow")),
        "fd": fd_error(run.get("fd") or {}, truth.get("fd") or {}),
        "wave_speed": _relative(rw.get("wave_speed_xcorr"), tw.get("wave_speed_xcorr")),
        "n_waves": None if n_run is None or n_truth is None else (n_run - n_truth) / max(n_truth, 1),
        "wave_amplitude": _relative(rw.get("wave_amplitude"), tw.get("wave_amplitude")),
        "travel_time": travel,
    }
    present = [abs(value) for value in components.values() if value is not None]
    return {
        "components": components,
        "value": float(np.mean(present)) if present else None,
        "n_components": len(present),
    }


def macro_error_dynamic(error: Mapping[str, Any]) -> dict[str, Any]:
    """The dynamic part of a macro-error vector (:func:`macro_error`): the components fundamental diagram,
    wave speed, number of waves (``waves``) and wave amplitude, with the same definitions; ``value`` is the
    mean of the absolute ones that exist. The anchored components (throughput, mean speed, queue discharge,
    travel time) follow largely the demand and the downstream boundary of the data, which every law shares."""
    given = error.get("components") or {}
    components = {name: given.get(source) for name, source in DYNAMIC.items()}
    present = [abs(value) for value in components.values() if value is not None]
    out: dict[str, Any] = {"components": components, "value": float(np.mean(present)) if present else None,
                           "n_components": len(present)}  # fmt: skip
    if error.get("error"):
        out["error"] = error["error"]
    return out


# ------------------------------------------------------------------------------------------- files

TRUTH_FILES = ("ground_truth.npz", "vehicles_truth.npz")
RUN_FILES = ("trajectories.npz", "vehicles.npz")
COLLISIONS_FILE = "collisions.npz"  # start of every contact episode of a run; without it vehicles.npz counts
RESERVED = ("scenarios", "laws")  # directories of runs/corridor that are no scenario of runs
SEED_DIR = re.compile(r"seed(\d+)")


@dataclasses.dataclass(frozen=True)
class MacroItem:
    """A ground truth (``law`` None: ``runs/corridor/scenarios/<scenario>/``) or a run
    (``runs/corridor/<scenario>/<law>/seed<s>/``) whose ``macro.json`` is written."""

    scenario: str
    directory: Path
    law: str | None = None
    seed: int | None = None

    @property
    def is_truth(self) -> bool:
        return self.law is None

    @property
    def macro(self) -> Path:
        return self.directory / "macro.json"

    @property
    def label(self) -> str:
        return f"truth {self.scenario}" if self.is_truth else f"{self.scenario}/{self.law}/seed{self.seed}"

    def data_files(self) -> tuple[Path, Path]:
        names = TRUTH_FILES if self.is_truth else RUN_FILES
        return self.directory / names[0], self.directory / names[1]


def scenario_directory(corridor_root: str | Path, scenario: str) -> Path:
    return Path(corridor_root) / "scenarios" / scenario


def find_items(
    corridor_root: str | Path, scenario: str | None = None, law: str | None = None, seed: int | None = None
) -> list[MacroItem]:
    """Ground truths and runs under ``corridor_root`` that match the filters, ground truths first.

    A ground truth is selected when neither ``law`` nor ``seed`` is given; directories starting with
    ``_`` are skipped. Every directory ``<scenario>/<law>/seed<s>`` counts as a run, complete or not.
    """
    root = Path(corridor_root)
    items: list[MacroItem] = []
    if law is None and seed is None and (root / "scenarios").is_dir():
        for directory in sorted((root / "scenarios").iterdir()):
            if directory.is_dir() and not directory.name.startswith("_") and scenario in (None, directory.name):
                items.append(MacroItem(directory.name, directory))
    for scenario_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not scenario_dir.is_dir() or scenario_dir.name in RESERVED or scenario_dir.name.startswith("_"):
            continue
        if scenario not in (None, scenario_dir.name):
            continue
        for law_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if law not in (None, law_dir.name):
                continue
            for run in sorted(law_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
                match = SEED_DIR.fullmatch(run.name)
                if run.is_dir() and match and seed in (None, int(match[1])):
                    items.append(MacroItem(scenario_dir.name, run, law_dir.name, int(match[1])))
    return items


def _inputs(item: MacroItem, corridor_root: Path) -> list[Path]:
    """Files whose change makes ``macro.json`` stale."""
    files = [*item.data_files(), scenario_directory(corridor_root, item.scenario) / "scenario.json"]
    if not item.is_truth:
        files += [item.directory / COLLISIONS_FILE, item.directory / "run.json",
                  scenario_directory(corridor_root, item.scenario) / "macro.json"]  # fmt: skip
    return files


def _scenario(corridor_root: str | Path, scenario: str) -> dict[str, Any]:
    path = scenario_directory(corridor_root, scenario) / "scenario.json"
    return read_json(path) if path.exists() else {}


def is_current(item: MacroItem, corridor_root: Path, hash_: str, geometry: Geometry) -> bool:
    """``macro.json`` exists, holds the configuration hash ``hash_`` and the ``geometry`` (section and analysis
    window) of the scenario now, and is not older than any of its inputs."""
    if not item.macro.exists():
        return False
    try:
        payload = read_json(item.macro)
        if payload.get("config_hash") != hash_ or payload.get("geometry") != geometry.as_dict():
            return False
    except (OSError, ValueError, UnicodeDecodeError, AttributeError):
        return False
    written = item.macro.stat().st_mtime
    return all(path.stat().st_mtime <= written for path in _inputs(item, corridor_root) if path.exists())


def item_metrics(
    item: MacroItem, corridor_root: str | Path, cfg: MacroConfig, default_geometry: Geometry | None = None,
    truth: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Content of ``macro.json`` of ``item``; a run gets its macro error against ``truth`` (the ground truth's
    ``macro.json``; without it, or with one of another analysis window, ``macro_error`` has no components
    and says why). The settings of the scenario's ``metrics`` block apply over ``cfg``
    (:func:`scenario_macro_config`)."""
    scenario = _scenario(corridor_root, item.scenario)
    cfg = scenario_macro_config(cfg, scenario)
    geometry = geometry_from_scenario(scenario, default_geometry)
    trajectories_file, vehicles_file = item.data_files()
    collisions_file = item.directory / COLLISIONS_FILE
    episodes = load_npz(collisions_file) if not item.is_truth and collisions_file.exists() else None
    metrics = corridor_metrics(load_npz(trajectories_file), load_npz(vehicles_file), geometry, cfg, episodes)
    config = cfg.hashed()
    from_config = sorted({"x_in", "x_out", "window"} - set(scenario_geometry(scenario)))
    header = {"kind": "truth" if item.is_truth else "run", "scenario": item.scenario, "law": item.law, "seed": item.seed,
              "geometry_from_configuration": from_config}  # fmt: skip
    out = {**header, **metrics}
    if not item.is_truth:
        if truth is None:
            out["macro_error"] = {"components": dict.fromkeys(COMPONENTS), "value": None, "n_components": 0,
                                  "error": "no ground truth macro.json"}  # fmt: skip
        else:
            out["macro_error"] = macro_error(metrics, truth)
        out["macro_error_dynamic"] = macro_error_dynamic(out["macro_error"])
    out["config"], out["config_hash"] = config, config_hash(config)
    return out


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a temporary file, so that an interrupted write never leaves a partial macro.json."""
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def summary_line(item: MacroItem, macro: Mapping[str, Any], seconds: float) -> str:
    """The printed line of a written ``macro.json``."""

    def number(value: Any, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    waves, travel = macro.get("waves") or {}, macro.get("travel_time") or {}
    text = (f"OK {item.label}: throughput {number(macro.get('throughput_vph'), 0)} veh/h, mean speed "
            f"{number(macro.get('mean_speed'), 2)} m/s, {waves.get('n_waves', 0)} waves "
            f"({number(waves.get('wave_speed'), 2)} m/s), travel time median {number(travel.get('median'), 1)} s")  # fmt: skip
    if not item.is_truth:
        error, dynamic = macro.get("macro_error") or {}, macro.get("macro_error_dynamic") or {}
        text += (f", macro error {number(error.get('value'), 3)} ({error.get('n_components', 0)} components), dynamic "
                 f"{number(dynamic.get('value'), 3)}, collisions {number(macro.get('collisions_per_1000_vkm'), 3)}"
                 "/1000 vkm")  # fmt: skip
    return f"{text} ({seconds:.1f} s) -> {item.macro}"


def update_macro(
    corridor_root: str | Path, cfg: MacroConfig, default_geometry: Geometry | None = None, *,
    scenario: str | None = None, law: str | None = None, seed: int | None = None, force: bool = False,
) -> Iterator[str]:  # fmt: skip
    """Write the ``macro.json`` of every selected ground truth and run; yields one line per file.

    A ``macro.json`` that holds the hash of ``cfg`` (with the ``metrics`` settings of its scenario over it)
    and the geometry and analysis window of its ``scenario.json`` now, and is not older than its inputs, is
    kept unless ``force``. The ground truth of the scenario of a selected run is brought up to date first
    (its ``macro.json`` is an input of the run, so a run and its ground truth always share the window). A run
    without ``run.json`` is incomplete and skipped; an error gives a ``FAILED`` line and the next file follows.
    """
    root = Path(corridor_root)
    selected = find_items(root, scenario, law, seed)
    truths = {item.scenario: item for item in selected if item.is_truth}
    for name in sorted({item.scenario for item in selected if not item.is_truth} - set(truths)):
        truths[name] = MacroItem(name, scenario_directory(root, name))  # a dependency of the selected runs
    loaded: dict[str, dict[str, Any] | None] = {}
    for item in [*sorted(truths.values(), key=lambda i: i.scenario), *(i for i in selected if not i.is_truth)]:
        missing = [path.name for path in item.data_files() if not path.exists()]
        if not item.is_truth and not missing and not (item.directory / "run.json").exists():
            missing.append("run.json")  # the loop writes run.json last: the run is not finished
        if missing:
            if item.is_truth:
                loaded[item.scenario] = None
            yield f"SKIPPED {item.label}: {', '.join(missing)} missing"
            continue
        start = time.perf_counter()
        try:
            scenario_json = _scenario(root, item.scenario)
            geometry = geometry_from_scenario(scenario_json, default_geometry)
            hash_ = config_hash(scenario_macro_config(cfg, scenario_json).hashed())
            if not force and is_current(item, root, hash_, geometry):
                if item.is_truth:
                    loaded[item.scenario] = read_json(item.macro)
                yield f"KEPT {item.label}: {item.macro} is up to date"
                continue
            payload = item_metrics(item, root, cfg, default_geometry, None if item.is_truth else loaded.get(item.scenario))
            _write_atomic(item.macro, payload)
        except Exception as exc:  # one broken file must not stop the others
            if item.is_truth:
                loaded[item.scenario] = None
            message = str(exc).splitlines()[0] if str(exc) else ""
            yield f"FAILED {item.label}: {type(exc).__name__}: {message}"
            continue
        if item.is_truth:
            loaded[item.scenario] = payload
        yield summary_line(item, payload, time.perf_counter() - start)
