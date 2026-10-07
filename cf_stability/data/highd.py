"""highD loader (format v1.0: ``XX_tracks.csv``, ``XX_tracksMeta.csv``, ``XX_recordingMeta.csv``).

``x, y`` is the upper-left corner of the bounding box, ``width`` the length along x; 25 frames
per second, frames start at 1, ``precedingId = 0`` means no leader. The longitudinal axis
points in the driving direction: direction 2 (increasing x) has front ``x + width``, rear
``x``, speed ``xVelocity``; direction 1 (decreasing x) has front ``-x``, rear
``-(x + width)``, speed ``-xVelocity`` (accelerations likewise). The leader state is taken
from the leader's own track at the same frame.

The highD licence forbids copies outside the project: ``assert_inside_project`` guards paths
that receive highD-derived data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events
from cf_stability.data.processing import resample_nearest, resample_uniform
from cf_stability.data.schema import DT, EventSet
from cf_stability.utils import REPO_ROOT, resolve_path

DATASET = "highd"
_TRACK_COLUMNS = ["frame", "id", "x", "width", "xVelocity", "xAcceleration", "dhw", "precedingId", "laneId"]
_META_COLUMNS = ["id", "width", "height", "class", "drivingDirection"]


def assert_inside_project(path: str | Path) -> Path:
    """Resolved ``path``; ``PermissionError`` if it lies outside the repository root."""
    resolved = resolve_path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise PermissionError(f"{resolved} is outside the project {REPO_ROOT}: highD-derived data must stay inside it")
    return resolved


@dataclass
class Recording:
    """One highD recording; ``tracks`` is sorted by (id, frame) and carries the longitudinal
    quantities ``front, rear, v, a`` and the leader's ``x_lead, v_lead, lane_lead`` (NaN without leader)."""

    recording: int
    frame_rate: float
    location_id: int
    vehicles: pd.DataFrame  # tracksMeta indexed by id
    tracks: pd.DataFrame

    @property
    def site(self) -> str:
        return f"loc{self.location_id}"

    @property
    def run(self) -> str:
        return f"r{self.recording:02d}"


def recording_ids(raw_dir: str | Path) -> list[int]:
    return sorted(int(path.name.split("_")[0]) for path in Path(raw_dir).glob("*_tracks.csv"))


def load_recording(raw_dir: str | Path, recording: int) -> Recording:
    prefix = Path(raw_dir) / f"{recording:02d}"
    info = pd.read_csv(f"{prefix}_recordingMeta.csv").iloc[0]
    vehicles = pd.read_csv(f"{prefix}_tracksMeta.csv", usecols=_META_COLUMNS).set_index("id")
    direction = vehicles["drivingDirection"]
    if not direction.isin([1, 2]).all():
        raise ValueError(f"recording {recording}: drivingDirection must be 1 or 2")
    tracks = pd.read_csv(f"{prefix}_tracks.csv", usecols=_TRACK_COLUMNS)
    tracks = tracks.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)

    forward = tracks["id"].map(direction).to_numpy() == 2
    x, width = tracks["x"].to_numpy(dtype=np.float64), tracks["width"].to_numpy(dtype=np.float64)
    sign = np.where(forward, 1.0, -1.0)
    tracks["front"] = np.where(forward, x + width, -x)
    tracks["rear"] = np.where(forward, x, -(x + width))
    tracks["v"] = sign * tracks["xVelocity"].to_numpy(dtype=np.float64)
    tracks["a"] = sign * tracks["xAcceleration"].to_numpy(dtype=np.float64)

    # leader row at the same frame, found through the sorted key id * K + frame
    frame, ids = tracks["frame"].to_numpy(np.int64), tracks["id"].to_numpy(np.int64)
    lead = tracks["precedingId"].to_numpy(np.int64)
    k = int(frame.max()) + 1
    keys = ids * k + frame
    row = np.minimum(np.searchsorted(keys, lead * k + frame), len(keys) - 1)
    found = (lead > 0) & (keys[row] == lead * k + frame)
    tracks["x_lead"] = np.where(found, tracks["rear"].to_numpy()[row], np.nan)
    tracks["v_lead"] = np.where(found, tracks["v"].to_numpy()[row], np.nan)
    tracks["lane_lead"] = np.where(found, tracks["laneId"].to_numpy(np.float64)[row], np.nan)
    return Recording(recording, float(info["frameRate"]), int(info["locationId"]), vehicles, tracks)


def recording_pairs(rec: Recording, dataset: str = DATASET, dt: float = DT) -> list[PairSeries]:
    """One PairSeries per track, resampled from the frame rate to ``dt`` (linear, D20).

    Ids and lanes are taken from the nearest frame; a grid point between two frames with a
    different leader or lane gets no leader state, so that interpolation never mixes two leaders.
    """
    tracks = rec.tracks
    ids = tracks["id"].to_numpy(np.int64)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    stops = np.r_[starts[1:], len(ids)]
    lead = tracks["precedingId"].to_numpy(np.int64)
    lane = tracks["laneId"].to_numpy(np.float64)
    lane_lead = tracks["lane_lead"].to_numpy(np.float64)
    changed = np.r_[True, (lead[1:] != lead[:-1]) | (lane[1:] != lane[:-1]) | ~_same(lane_lead[1:], lane_lead[:-1])]
    label = np.cumsum(changed).astype(np.float64)  # constant while leader and lanes are constant
    t_all = tracks["frame"].to_numpy(np.float64) / rec.frame_rate
    prefix = f"{dataset}/{rec.site}/{rec.run}_v"

    pairs = []
    for start, stop in zip(starts, stops):
        vid = int(ids[start])
        t = t_all[start:stop]
        columns = {name: tracks[name].to_numpy(np.float64)[start:stop] for name in ("front", "v", "a", "x_lead", "v_lead")}
        grid, series = resample_uniform(t, {**columns, "label": label[start:stop]}, dt=dt)
        if grid.size == 0:
            continue
        mixed = series["label"] != np.round(series["label"])
        leader = resample_nearest(t, lead[start:stop], grid)
        info = rec.vehicles.loc[vid]
        pairs.append(
            PairSeries(
                dataset=dataset,
                site=rec.site,
                follower_id=f"{prefix}{vid}",
                t=grid,
                x_follower=series["front"],
                v=series["v"],
                leader_id=np.array([f"{prefix}{p}" if p > 0 else None for p in leader], dtype=object),
                x_lead=np.where(mixed, np.nan, series["x_lead"]),
                v_lead=np.where(mixed, np.nan, series["v_lead"]),
                a=series["a"],
                lane_follower=resample_nearest(t, lane[start:stop], grid),
                lane_leader=resample_nearest(t, lane_lead[start:stop], grid),
                run=rec.run,
                meta={
                    "recording": rec.recording,
                    "follower_class": str(info["class"]),
                    "follower_length": float(info["width"]),
                    "driving_direction": int(info["drivingDirection"]),
                },
            )
        )
    return pairs


def build_events(data_cfg: Mapping, extraction: ExtractionConfig, stats: Counter | None = None) -> EventSet:
    """Events of the configured recordings (``recordings: null`` = all found in ``raw_dir``).

    Extra ``stats``: ``recordings``, ``dhw_frames`` and ``dhw_abs_diff_mean`` /
    ``dhw_abs_diff_max`` (m): computed net gap vs published ``dhw`` on frames with a leader.
    """
    stats = Counter() if stats is None else stats
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw/highd"))
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"highD directory not found: {raw_dir}")
    dataset = str(data_cfg.get("name", DATASET))
    recordings = data_cfg.get("recordings") or recording_ids(raw_dir)
    events = []
    diff_sum, diff_max, n_frames = 0.0, 0.0, 0
    for recording in recordings:
        rec = load_recording(raw_dir, int(recording))
        stats["recordings"] += 1
        gap = rec.tracks["x_lead"].to_numpy() - rec.tracks["front"].to_numpy()
        diff = np.abs(gap - rec.tracks["dhw"].to_numpy(np.float64))[np.isfinite(gap)]
        diff_sum, n_frames = diff_sum + float(diff.sum()), n_frames + diff.size
        diff_max = max(diff_max, float(diff.max(initial=0.0)))
        leaders = {
            f"{dataset}/{rec.site}/{rec.run}_v{vid}": (str(row["class"]), float(row["width"]))
            for vid, row in rec.vehicles.iterrows()
        }
        for pair in recording_pairs(rec, dataset, extraction.dt):
            for event in extract_events(pair, extraction, stats):
                k0 = int(round((event.meta["t0"] - pair.t[0]) / extraction.dt))
                leader_class, leader_length = leaders[event.leader_id]
                event.meta.update(lane=int(pair.lane_follower[k0]), leader_class=leader_class, leader_length=leader_length)
                events.append(event)
    stats["dhw_frames"] += n_frames
    stats["dhw_abs_diff_mean"] = diff_sum / n_frames if n_frames else 0.0
    stats["dhw_abs_diff_max"] = diff_max
    return EventSet(events)


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise equality with NaN == NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b))
