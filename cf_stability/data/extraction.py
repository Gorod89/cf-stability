"""Event extraction from follower/leader series (criteria in docs/data_contract.md, section 4).

The dataclasses and signatures below are the public API that the dataset loaders rely on.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from cf_stability.data.processing import acceleration_from_speed, contiguous_runs, savgol_smooth
from cf_stability.data.schema import DT, Event

# measured speeds in (-NEGATIVE_SPEED_TOL, 0) are set to 0, lower ones make the sample invalid
NEGATIVE_SPEED_TOL = 0.1
# relative deviation of a time step from dt beyond which the grid has a gap
_GRID_TOL = 1e-3
STAT_KEYS = ("runs", "dropped_short", "dropped_not_moving", "chunks", "events", "samples")


@dataclass(frozen=True)
class ExtractionConfig:
    dt: float = DT
    min_duration: float = 15.0
    max_duration: float | None = None  # None: no chunking; otherwise >= 2 * min_duration
    max_spacing: float = 120.0
    min_moving_fraction: float = 0.8
    moving_speed_eps: float = 0.1
    sg_window_s: float = 1.1
    sg_order: int = 2
    smooth_speeds: bool = False
    min_spacing: float = 0.5  # a sample is valid when min_spacing < s < max_spacing (decision D41)

    def __post_init__(self) -> None:
        if self.max_duration is not None and self.max_duration < 2 * self.min_duration:
            raise ValueError("max_duration must be None or >= 2 * min_duration")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExtractionConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(mapping) - names
        if unknown:
            raise KeyError(f"unknown extraction options: {sorted(unknown)}")
        return cls(**{k: mapping[k] for k in mapping})

    @property
    def min_samples(self) -> int:
        # FollowNet rule: n * dt >= min_duration (150 samples at 10 Hz)
        return int(round(self.min_duration / self.dt))

    @property
    def max_samples(self) -> int | None:
        return None if self.max_duration is None else int(round(self.max_duration / self.dt))


@dataclass
class PairSeries:
    """Follower series on a uniform 10 Hz grid with its instantaneous leader."""

    dataset: str
    site: str
    follower_id: str  # globally unique, "<dataset>/<site>/<raw id>"
    t: np.ndarray  # [n] source time, s
    x_follower: np.ndarray  # [n] front bumper, NaN where unobserved
    v: np.ndarray  # [n]
    leader_id: np.ndarray  # [n] object array: globally unique leader id, None without leader
    x_lead: np.ndarray  # [n] leader rear bumper, NaN without leader
    v_lead: np.ndarray  # [n]
    a: np.ndarray | None = None  # dataset acceleration, if the source provides one
    lane_follower: np.ndarray | None = None  # [n] int
    lane_leader: np.ndarray | None = None  # [n] int
    segment_key: np.ndarray | None = None  # [n] any dtype (e.g. driver mode); a change ends a segment
    run: str | None = None  # recording tag, part of the event id when set
    meta: dict = field(default_factory=dict)


@dataclass
class _Series:
    """Cleaned PairSeries arrays with sample validity and segment boundaries."""

    t: np.ndarray
    x_follower: np.ndarray
    v: np.ndarray
    x_lead: np.ndarray
    v_lead: np.ndarray
    a: np.ndarray | None
    leader: np.ndarray  # str ids, "" without leader
    segment_key: np.ndarray | None  # object array
    valid: np.ndarray
    cut: np.ndarray  # cut[i]: segment boundary between samples i - 1 and i


def extract_events(pair: PairSeries, cfg: ExtractionConfig, stats: Counter | None = None) -> list[Event]:
    """Cut a PairSeries into events that satisfy the extraction criteria.

    ``stats`` (optional) counts what happened to the candidate runs: ``runs``,
    ``dropped_short``, ``dropped_not_moving``, ``chunks`` (extra events created by
    chunking), ``events`` and ``samples``.
    """
    stats = Counter() if stats is None else stats
    series = _prepare(pair, cfg)
    a_source = "savgol" if pair.a is None else "dataset"
    per_leader: Counter = Counter()
    events = []
    for start, stop, v, v_lead, a in _pieces(series, cfg, stats):
        leader_id = str(series.leader[start])
        raw_leader = leader_id.rsplit("/", 1)[-1]
        k = per_leader[raw_leader]
        per_leader[raw_leader] += 1
        meta = {**pair.meta, "t0": float(series.t[start]), "a_source": a_source}
        if series.segment_key is not None:
            meta["segment_key"] = _plain(series.segment_key[start])
        if pair.run is None:
            event_id = f"{pair.follower_id}|{raw_leader}|{k}"
        else:
            event_id = f"{pair.follower_id}|{raw_leader}|{pair.run}|{k}"
            meta["run"] = pair.run
        events.append(_make_event(event_id, pair, leader_id, series, start, stop, v, v_lead, a, meta, cfg.dt))
    return events


def filter_event(event: Event, cfg: ExtractionConfig, stats: Counter | None = None) -> list[Event]:
    """Apply the same criteria to an event of a pre-segmented source (may split or drop it).

    The event's acceleration is kept as it is. Resulting events get the id suffix ``_<j>``
    only when the event is split into more than one event.
    """
    stats = Counter() if stats is None else stats
    pair = PairSeries(
        dataset=event.dataset,
        site=event.site,
        follower_id=event.follower_id,
        t=float(event.meta.get("t0", 0.0)) + event.t,
        x_follower=event.x_follower,
        v=event.v,
        leader_id=np.full(len(event), event.leader_id, dtype=object),
        x_lead=event.x_lead,
        v_lead=event.v_lead,
        a=event.a,
    )
    series = _prepare(pair, cfg)
    pieces = list(_pieces(series, cfg, stats))
    out = []
    for j, (start, stop, v, v_lead, a) in enumerate(pieces):
        event_id = event.event_id if len(pieces) == 1 else f"{event.event_id}_{j}"
        meta = {**event.meta, "t0": float(series.t[start])}
        out.append(_make_event(event_id, pair, event.leader_id, series, start, stop, v, v_lead, a, meta, cfg.dt))
    return out


def _floats(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
    return arr


def _objects(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.shape != (n,):
        raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
    return arr


def _clip_speed(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set small negative speeds to 0; also return the mask of acceptable samples."""
    ok = ~(v <= -NEGATIVE_SPEED_TOL)
    return np.where(ok & (v < 0.0), 0.0, v), ok


def _plain(value: Any) -> str | int | float:
    """Plain Python scalar for JSON meta."""
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (str, int, float)) else str(value)


def _prepare(pair: PairSeries, cfg: ExtractionConfig) -> _Series:
    t = np.asarray(pair.t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t has shape {t.shape}, expected 1-D")
    n = t.shape[0]
    x_follower = _floats(pair.x_follower, n, "x_follower")
    x_lead = _floats(pair.x_lead, n, "x_lead")
    v, v_ok = _clip_speed(_floats(pair.v, n, "v"))
    v_lead, v_lead_ok = _clip_speed(_floats(pair.v_lead, n, "v_lead"))
    a = None if pair.a is None else _floats(pair.a, n, "a")
    leader_raw = _objects(pair.leader_id, n, "leader_id")
    present = ~pd.isna(leader_raw)
    leader = np.where(present, leader_raw, "").astype(str)

    cut = np.zeros(n, dtype=bool)
    cut[1:] = ~(np.abs(np.diff(t) - cfg.dt) <= _GRID_TOL * cfg.dt) | (leader[1:] != leader[:-1])
    finite = [t, x_follower, v, x_lead, v_lead] + ([] if a is None else [a])
    for lanes, name in ((pair.lane_follower, "lane_follower"), (pair.lane_leader, "lane_leader")):
        if lanes is not None:
            lanes = _floats(lanes, n, name)
            finite.append(lanes)
            cut[1:] |= lanes[1:] != lanes[:-1]
    segment_key = None
    if pair.segment_key is not None:
        segment_key = _objects(pair.segment_key, n, "segment_key")
        present &= ~pd.isna(segment_key)
        cut[1:] |= np.asarray(segment_key[1:] != segment_key[:-1], dtype=bool)

    s = x_lead - x_follower
    valid = present & np.isfinite(np.vstack(finite)).all(axis=0) & v_ok & v_lead_ok & (s > cfg.min_spacing) & (s < cfg.max_spacing)
    return _Series(t, x_follower, v, x_lead, v_lead, a, leader, segment_key, valid, cut)


def _runs(valid: np.ndarray, cut: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of valid samples without a segment boundary inside."""
    runs = []
    for start, stop in contiguous_runs(valid):
        edges = [start, *(start + 1 + np.flatnonzero(cut[start + 1 : stop])).tolist(), stop]
        runs.extend(zip(edges[:-1], edges[1:]))
    return runs


def _pieces(
    series: _Series, cfg: ExtractionConfig, stats: Counter
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    """Accepted ``(start, stop, v, v_lead, a)`` after the duration, chunking and moving filters."""
    stats.update(dict.fromkeys(STAT_KEYS, 0))  # every key present, also when zero
    for start, stop in _runs(series.valid, series.cut):
        stats["runs"] += 1
        n = stop - start
        if n < cfg.min_samples:
            stats["dropped_short"] += 1
            continue
        # filtered on the whole run, so that the chunks of a run share one filtered series
        v, v_lead = series.v[start:stop], series.v_lead[start:stop]
        if series.a is None:
            a = acceleration_from_speed(v, cfg.dt, cfg.sg_window_s, cfg.sg_order)
        else:
            a = series.a[start:stop]
        if cfg.smooth_speeds:
            v = np.maximum(savgol_smooth(v, cfg.dt, cfg.sg_window_s, cfg.sg_order), 0.0)
            v_lead = np.maximum(savgol_smooth(v_lead, cfg.dt, cfg.sg_window_s, cfg.sg_order), 0.0)
        n_chunks = 1 if cfg.max_samples is None else math.ceil(n / cfg.max_samples)
        stats["chunks"] += n_chunks - 1
        edges = [(i * n) // n_chunks for i in range(n_chunks + 1)]
        for i0, i1 in zip(edges[:-1], edges[1:]):
            if np.mean(v[i0:i1] > cfg.moving_speed_eps) < cfg.min_moving_fraction:
                stats["dropped_not_moving"] += 1
                continue
            stats["events"] += 1
            stats["samples"] += i1 - i0
            yield start + i0, start + i1, v[i0:i1], v_lead[i0:i1], a[i0:i1]


def _make_event(
    event_id: str,
    pair: PairSeries,
    leader_id: str,
    series: _Series,
    start: int,
    stop: int,
    v: np.ndarray,
    v_lead: np.ndarray,
    a: np.ndarray,
    meta: dict,
    dt: float,
) -> Event:
    x_follower = series.x_follower[start:stop].copy()
    x_lead = series.x_lead[start:stop].copy()
    event = Event(
        event_id=event_id,
        dataset=pair.dataset,
        site=pair.site,
        follower_id=pair.follower_id,
        leader_id=leader_id,
        t=np.arange(stop - start) * dt,
        s=x_lead - x_follower,
        dv=v - v_lead,
        v=v.copy(),
        a=a.copy(),
        v_lead=v_lead.copy(),
        x_lead=x_lead,
        x_follower=x_follower,
        meta=meta,
    )
    event.validate(dt=dt)
    return event
