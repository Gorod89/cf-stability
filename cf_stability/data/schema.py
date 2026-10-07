"""Common car-following Event schema and parquet storage.

Conventions (binding for the whole project, see docs/data_contract.md):

* SI units, 10 Hz (``DT = 0.1`` s), float64 arrays.
* ``dv = v - v_lead`` (approach rate, positive when closing in).
* ``x_lead`` is the longitudinal position of the leader's REAR bumper and
  ``x_follower`` the position of the follower's FRONT bumper, so the net gap is
  ``s = x_lead - x_follower`` exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from cf_stability.utils import json_default

DT = 0.1
ARRAY_FIELDS = ("t", "s", "dv", "v", "a", "v_lead", "x_lead", "x_follower")
ID_FIELDS = ("dataset", "site", "follower_id", "leader_id")

SAMPLES_FILE = "samples.parquet"
EVENTS_FILE = "events.parquet"
MANIFEST_FILE = "manifest.json"


class EventValidationError(ValueError):
    pass


@dataclass
class Event:
    """One car-following event on the common 10 Hz grid."""

    event_id: str
    dataset: str
    site: str
    follower_id: str
    leader_id: str
    t: np.ndarray
    s: np.ndarray
    dv: np.ndarray
    v: np.ndarray
    a: np.ndarray
    v_lead: np.ndarray
    x_lead: np.ndarray
    x_follower: np.ndarray
    # free-form, JSON-serialisable extras (leader_length, acc_on, lane, source_file, ...)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ARRAY_FIELDS:
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64))

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def validate(self, dt: float = DT, atol: float = 1e-6) -> None:
        """Raise EventValidationError if an invariant of the schema is violated."""
        n = len(self)
        if n < 2:
            raise EventValidationError(f"{self.event_id}: fewer than 2 samples")
        for name in ARRAY_FIELDS:
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise EventValidationError(f"{self.event_id}: {name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise EventValidationError(f"{self.event_id}: {name} contains non-finite values")
        if np.max(np.abs(np.diff(self.t) - dt)) > atol:
            raise EventValidationError(f"{self.event_id}: t is not uniform with step {dt}")
        if np.min(self.s) <= 0.0:
            raise EventValidationError(f"{self.event_id}: non-positive spacing (min {np.min(self.s):.3f})")
        if np.min(self.v) < -atol or np.min(self.v_lead) < -atol:
            raise EventValidationError(f"{self.event_id}: negative speed")
        if np.max(np.abs(self.dv - (self.v - self.v_lead))) > atol:
            raise EventValidationError(f"{self.event_id}: dv != v - v_lead")
        if np.max(np.abs(self.s - (self.x_lead - self.x_follower))) > atol:
            raise EventValidationError(f"{self.event_id}: s != x_lead - x_follower")


class EventSet:
    """Ordered collection of events with parquet round-trip.

    On disk an EventSet is a directory with ``samples.parquet`` (long format, one
    row per time sample, keyed by ``event_id``), ``events.parquet`` (one row per
    event: ids, n_samples, duration, meta_json) and ``manifest.json``.
    """

    def __init__(self, events: Iterable[Event] = ()) -> None:
        self.events: list[Event] = list(events)
        self._index: dict[str, int] = {}
        for i, ev in enumerate(self.events):
            if ev.event_id in self._index:
                raise ValueError(f"duplicate event_id {ev.event_id}")
            self._index[ev.event_id] = i

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, key: int | str) -> Event:
        if isinstance(key, str):
            return self.events[self._index[key]]
        return self.events[key]

    def subset(self, event_ids: Iterable[str]) -> "EventSet":
        return EventSet(self.events[self._index[e]] for e in event_ids)

    def validate(self, dt: float = DT) -> None:
        for ev in self.events:
            ev.validate(dt=dt)

    # ------------------------------------------------------------------ tables
    def events_frame(self) -> pd.DataFrame:
        rows = [
            {
                "event_id": ev.event_id,
                "dataset": ev.dataset,
                "site": ev.site,
                "follower_id": ev.follower_id,
                "leader_id": ev.leader_id,
                "n_samples": len(ev),
                "duration": ev.duration,
                "meta_json": json.dumps(ev.meta, sort_keys=True, default=json_default),
            }
            for ev in self.events
        ]
        columns = ["event_id", *ID_FIELDS, "n_samples", "duration", "meta_json"]
        return pd.DataFrame(rows, columns=columns)

    def samples_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=["event_id", *ARRAY_FIELDS])
        ids = np.concatenate([np.full(len(ev), i, dtype=np.int64) for i, ev in enumerate(self.events)])
        data = {"event_id": pd.Categorical.from_codes(ids, categories=[ev.event_id for ev in self.events])}
        for name in ARRAY_FIELDS:
            data[name] = np.concatenate([getattr(ev, name) for ev in self.events])
        return pd.DataFrame(data)

    def summary(self) -> dict:
        n = np.array([len(ev) for ev in self.events], dtype=np.int64)
        dur = np.array([ev.duration for ev in self.events], dtype=np.float64)
        return {
            "n_events": int(len(self.events)),
            "n_samples": int(n.sum()) if len(n) else 0,
            "n_followers": len({ev.follower_id for ev in self.events}),
            "n_sites": len({(ev.dataset, ev.site) for ev in self.events}),
            "duration_total_s": float(dur.sum()) if len(dur) else 0.0,
            "duration_median_s": float(np.median(dur)) if len(dur) else 0.0,
            "duration_min_s": float(dur.min()) if len(dur) else 0.0,
            "duration_max_s": float(dur.max()) if len(dur) else 0.0,
        }

    # ----------------------------------------------------------------- storage
    def to_parquet(self, directory: str | Path, manifest: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.samples_frame().to_parquet(directory / SAMPLES_FILE, index=False)
        self.events_frame().to_parquet(directory / EVENTS_FILE, index=False)
        payload = {"schema_version": 1, "dt": DT, "summary": self.summary(), **(manifest or {})}
        (directory / MANIFEST_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8"
        )
        return directory

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "EventSet":
        directory = Path(directory)
        meta = pd.read_parquet(directory / EVENTS_FILE)
        samples = pd.read_parquet(directory / SAMPLES_FILE)
        samples["event_id"] = samples["event_id"].astype(str)
        # rows of one event are contiguous and time-ordered by construction
        arrays = {name: samples[name].to_numpy(dtype=np.float64) for name in ARRAY_FIELDS}
        ids = samples["event_id"].to_numpy()
        bounds = np.flatnonzero(ids[1:] != ids[:-1]) + 1 if len(ids) else np.array([], dtype=np.int64)
        starts = np.concatenate([[0], bounds]).astype(np.int64) if len(ids) else np.array([], dtype=np.int64)
        stops = np.concatenate([bounds, [len(ids)]]).astype(np.int64) if len(ids) else np.array([], dtype=np.int64)
        span = {ids[a]: (a, b) for a, b in zip(starts, stops)}
        if len(span) != len(starts):
            raise ValueError(f"{directory}: rows of an event are not contiguous in {SAMPLES_FILE}")
        events = []
        for row in meta.itertuples(index=False):
            a, b = span[row.event_id]
            events.append(
                Event(
                    event_id=row.event_id,
                    dataset=row.dataset,
                    site=row.site,
                    follower_id=row.follower_id,
                    leader_id=row.leader_id,
                    meta=json.loads(row.meta_json),
                    **{name: arrays[name][a:b].copy() for name in ARRAY_FIELDS},
                )
            )
        return cls(events)
