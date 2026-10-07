"""NGSIM I-80 / US-101 loader: raw table, tracks, leader-follower pairs and events.

Raw data: Socrata dataset ``8ect-6jqj`` (data.transportation.gov, downloaded by
``scripts/download_data.py``), feet and ft/s, 10 Hz, ``local_y`` = longitudinal position of the
vehicle's FRONT centre. Every location consists of three original files (periods) whose vehicle and
frame ids restart; the combined table has no period column, but ``global_time - 100 * frame_id``
(the epoch of frame 0 of the file) identifies the period exactly.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cf_stability.corridor.macro import compare_macro, edie_grid
from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events
from cf_stability.data.reconstruction import (
    ReconstructedTrack,
    ReconstructionConfig,
    derivatives,
    leader_links,
    reconstruct_all,
)
from cf_stability.data.schema import DT, EventSet
from cf_stability.utils import config_hash, resolve_path, to_plain

FT = 0.3048
# only these columns are loaded (memory: the I-80 table has 4.6 M rows)
LOAD_COLUMNS = (
    "vehicle_id", "frame_id", "global_time", "local_y", "v_length", "v_class", "v_vel", "v_acc", "lane_id",
    "preceding", "space_headway", "location",
)
FEET_COLUMNS = ("local_y", "v_length", "v_vel", "v_acc", "space_headway")
INT_COLUMNS = ("vehicle_id", "frame_id", "global_time", "v_class", "lane_id", "preceding")
N_PERIODS = 3  # original files per location
MAINLINE_LANES = {"i80": (1, 2, 3, 4, 5, 6), "us101": (1, 2, 3, 4, 5)}
MACRO_CELL = (50.0, 10.0)  # m, s
MIN_FREE_BYTES = 2 * 1024**3  # the reconstruction cache is not written on a fuller disk
_REC_STATS = [f.name for f in dataclasses.fields(ReconstructedTrack) if f.name not in ("x", "v", "a", "jerk", "weights")]


@dataclass
class Track:
    """One contiguous 10 Hz track of one vehicle in one period (SI units)."""

    track_id: str
    period: int
    vehicle_id: int
    t: np.ndarray  # s since frame 0 of the location's first period
    frame: np.ndarray
    global_time: np.ndarray  # epoch ms
    x: np.ndarray  # front bumper, m
    lane: np.ndarray
    length: float
    v_class: int
    preceding: np.ndarray  # object array: leader track id or None
    v: np.ndarray  # raw speed, m/s
    a: np.ndarray  # raw acceleration, m/s^2
    rec: ReconstructedTrack | None = None


# ----------------------------------------------------------------------------- raw table
def load_raw(path: str | Path, location: str | None = None) -> pd.DataFrame:
    """Raw NGSIM table of one location (columns ``LOAD_COLUMNS``) in SI units, with ``period``,
    ``t`` and ``track_id``.

    Accepts the parquet written by ``scripts/download_data.py`` or a CSV export with lower-case or
    original column names. Duplicate rows (identical in the loaded columns) are dropped, and rows
    repeating a vehicle/frame/time key. ``period`` is the rank of the frame clock ``global_time -
    100 * frame_id``; ``t`` is the time in s since frame 0 of the first period; ``track_id`` is
    ``p<period>_<vehicle_id>``, with a suffix ``_<k>`` for the k-th further piece of a track that has
    a frame gap, so that every track is a contiguous 10 Hz series. Counts go to
    ``df.attrs["load_stats"]``.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=[c for c in pq.read_schema(path).names if c.lower() in LOAD_COLUMNS])
    else:
        df = pd.read_csv(path, usecols=lambda c: c.lower() in LOAD_COLUMNS)
    df.columns = [c.lower() for c in df.columns]
    df["location"] = df["location"].astype(str).str.lower().astype("category")
    if location is not None:
        df = df[df["location"] == location]
    locations = df["location"].unique()
    if len(locations) != 1:
        raise ValueError(f"{path}: expected rows of exactly one location, found {sorted(locations)}")

    n_rows = len(df)
    df = df.drop_duplicates(ignore_index=True)
    n_duplicates = n_rows - len(df)
    df = df.drop_duplicates(["vehicle_id", "frame_id", "global_time"], ignore_index=True)
    n_key_duplicates = n_rows - n_duplicates - len(df)
    df = df.astype({c: np.int64 for c in INT_COLUMNS})
    df[list(FEET_COLUMNS)] = df[list(FEET_COLUMNS)] * FT

    clock = df["global_time"].to_numpy() - 100 * df["frame_id"].to_numpy()
    clocks, period = np.unique(clock, return_inverse=True)
    if len(clocks) > N_PERIODS:
        raise ValueError(f"{path}: {len(clocks)} distinct frame clocks (global_time - 100 * frame_id), expected <= {N_PERIODS}")
    df["period"] = period
    df["t"] = np.rint((df["global_time"].to_numpy() - clocks[0]) / 100.0) * DT
    df = df.sort_values(["period", "vehicle_id", "frame_id"], ignore_index=True)

    per, vid = df["period"].to_numpy(), df["vehicle_id"].to_numpy()
    new_vehicle = np.r_[True, (per[1:] != per[:-1]) | (vid[1:] != vid[:-1])]
    new_track = new_vehicle | np.r_[True, np.diff(df["frame_id"].to_numpy()) != 1]
    starts = np.flatnonzero(new_track)
    piece = np.arange(len(starts)) - np.maximum.accumulate(np.where(new_vehicle[starts], np.arange(len(starts)), 0))
    ids = [f"p{p}_{v}" + (f"_{k}" if k else "") for p, v, k in zip(per[starts], vid[starts], piece)]
    df["track_id"] = pd.Categorical.from_codes(np.cumsum(new_track) - 1, categories=ids)
    df.attrs["load_stats"] = {
        "raw_rows": int(n_rows),
        "raw_duplicates": int(n_duplicates),
        "raw_key_duplicates": int(n_key_duplicates),
        "tracks": len(ids),
        "tracks_split": int((piece > 0).sum()),
    }
    return df


# ----------------------------------------------------------------------------- tracks
def build_tracks(df: pd.DataFrame) -> dict[str, Track]:
    """One :class:`Track` per ``track_id``; ``preceding`` is resolved to the leader's track id
    within the same period (None if 0 or if the leader has no record in that frame)."""
    df = df.assign(_code=df["track_id"].cat.codes).sort_values(["_code", "frame_id"], ignore_index=True)
    codes = df["_code"].to_numpy()
    period, frame = df["period"].to_numpy(np.int64), df["frame_id"].to_numpy()
    if frame.max(initial=0) >= 10**7 or df["vehicle_id"].max() >= 10**5:
        raise ValueError("frame or vehicle id out of the range of the lookup key")

    def key(vehicle: np.ndarray) -> np.ndarray:
        return (period * 10**5 + vehicle) * 10**7 + frame

    own = pd.Index(key(df["vehicle_id"].to_numpy()))
    first = ~own.duplicated()
    pos = pd.Index(own[first]).get_indexer(key(df["preceding"].to_numpy()))
    lead_code = np.where(pos >= 0, codes[np.flatnonzero(first)[np.maximum(pos, 0)]], -1)
    lead_code[df["preceding"].to_numpy() == 0] = -1
    names = np.array([*df["track_id"].cat.categories, None], dtype=object)
    preceding = names[lead_code]  # code -1 picks the trailing None

    cols = {c: df[c].to_numpy() for c in ("t", "global_time", "local_y", "lane_id", "v_length", "v_class", "v_vel", "v_acc", "vehicle_id")}
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    tracks = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        tid = names[codes[a]]
        tracks[tid] = Track(
            track_id=tid,
            period=int(period[a]),
            vehicle_id=int(cols["vehicle_id"][a]),
            t=cols["t"][a:b],
            frame=frame[a:b],
            global_time=cols["global_time"][a:b],
            x=cols["local_y"][a:b],
            lane=cols["lane_id"][a:b],
            length=float(np.median(cols["v_length"][a:b])),
            v_class=int(cols["v_class"][a]),
            preceding=preceding[a:b],
            v=cols["v_vel"][a:b],
            a=cols["v_acc"][a:b],
        )
    return tracks


def _kinematics(track: Track, positions: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positions == "raw":
        return track.x, track.v, track.a
    if positions != "reconstructed":
        raise ValueError(f"positions must be 'reconstructed' or 'raw', got {positions!r}")
    if track.rec is None:
        raise ValueError(f"track {track.track_id} has no reconstruction")
    return track.rec.x, track.rec.v, track.rec.a


def build_pairs(
    tracks: Mapping[str, Track],
    dataset: str,
    site: str,
    positions: str = "reconstructed",
    lanes: Sequence[int] | None = None,
) -> Iterator[PairSeries]:
    """Follower series with the instantaneous leader (rear bumper) for every track that has one.

    Samples where the follower or the leader is outside ``lanes`` (default: the site's mainline
    lanes) get no leader.
    """
    lanes = np.asarray(MAINLINE_LANES[site] if lanes is None else lanes)
    prefix = f"{dataset}/{site}/"
    for tid in sorted(tracks):
        tr = tracks[tid]
        n = len(tr.t)
        x, v, a = _kinematics(tr, positions)
        leader_id = np.full(n, None, dtype=object)
        x_lead, v_lead = np.full(n, np.nan), np.full(n, np.nan)
        lane_lead = np.full(n, -1, dtype=np.int64)
        for lid, k, j in leader_links(tr, tracks, DT):
            lead = tracks[lid]
            lx, lv, _ = _kinematics(lead, positions)
            leader_id[k] = prefix + lid
            x_lead[k] = lx[j] - lead.length
            v_lead[k] = lv[j]
            lane_lead[k] = lead.lane[j]
        off = ~(np.isin(tr.lane, lanes) & np.isin(lane_lead, lanes))
        leader_id[off], x_lead[off], v_lead[off] = None, np.nan, np.nan
        if off.all():
            continue
        yield PairSeries(
            dataset=dataset, site=site, follower_id=prefix + tid, t=tr.t, x_follower=x, v=v,
            leader_id=leader_id, x_lead=x_lead, v_lead=v_lead, a=a, lane_follower=tr.lane, lane_leader=lane_lead,
            meta={"period": tr.period, "follower_length": tr.length, "v_class": tr.v_class},
        )


# ----------------------------------------------------------------------------- validation
def _macro_arrays(tracks: Mapping[str, Track], xs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = list(tracks)
    return {
        "vehicle": np.concatenate([np.full(len(tracks[t].t), i) for i, t in enumerate(ids)]),
        "t": np.concatenate([tracks[t].t for t in ids]),
        "x": np.concatenate([xs[t] for t in ids]),
        "lane": np.concatenate([tracks[t].lane for t in ids]),
    }


def _edges(values: list[np.ndarray], step: float) -> np.ndarray:
    lo = np.floor(min(v.min() for v in values) / step) * step
    hi = np.floor(max(v.max() for v in values) / step) * step + step
    return np.arange(lo, hi + step / 2, step)


def validate_reconstruction(tracks_raw: Mapping[str, Track], tracks_rec: Mapping[str, Any], site_cfg: Mapping[str, Any]) -> dict:
    """Reconstruction report plus the Edie comparison raw vs reconstructed on a 50 m x 10 s grid.

    ``tracks_rec`` is the result of :func:`reconstruct_all` (``{"tracks": ..., "report": ...}``).
    """
    lanes = site_cfg.get("lanes") or MAINLINE_LANES[site_cfg["site"]]
    raw = _macro_arrays(tracks_raw, {t: tr.x for t, tr in tracks_raw.items()})
    rec = _macro_arrays(tracks_raw, {t: r.x for t, r in tracks_rec["tracks"].items()})
    x_edges = _edges([raw["x"], rec["x"]], MACRO_CELL[0])
    t_edges = _edges([raw["t"]], MACRO_CELL[1])
    macro = compare_macro(edie_grid(raw, x_edges, t_edges, lanes), edie_grid(rec, x_edges, t_edges, lanes))
    return {**tracks_rec["report"], "macro": macro}


# ----------------------------------------------------------------------------- reconstruction cache
def _write_cache(path: Path, tracks: Mapping[str, Track], result: Mapping[str, Any], digest: str) -> None:
    """One parquet file with the reconstructed trajectories; hash, report and per-track solver
    statistics in the file metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < MIN_FREE_BYTES:
        # the cache only saves time; the events can still be built from the result in memory
        print(f"NGSIM: reconstruction cache not written, only {free / 1e9:.2f} GB free (needs >= 2 GB)", flush=True)
        return
    ids, recs = list(tracks), result["tracks"]
    n = [len(tracks[t].t) for t in ids]
    frame = pd.DataFrame({
        "track_id": pd.Categorical.from_codes(np.repeat(np.arange(len(ids)), n), categories=ids),
        "period": np.repeat([tracks[t].period for t in ids], n),
        "vehicle_id": np.repeat([tracks[t].vehicle_id for t in ids], n),
        "frame_id": np.concatenate([tracks[t].frame for t in ids]),
        "global_time": np.concatenate([tracks[t].global_time for t in ids]),
        "lane_id": np.concatenate([tracks[t].lane for t in ids]),
        "x": np.concatenate([recs[t].x for t in ids]),
        "v": np.concatenate([recs[t].v for t in ids]),
        "a": np.concatenate([recs[t].a for t in ids]),
        "huber_weight": np.concatenate([recs[t].weights for t in ids]),
        "length": np.repeat([tracks[t].length for t in ids], n),
        "v_class": np.repeat([tracks[t].v_class for t in ids], n),
    })
    table = pa.Table.from_pandas(frame, preserve_index=False)
    track_stats = {t: {k: getattr(recs[t], k) for k in _REC_STATS} for t in ids}
    meta = {
        **table.schema.metadata,
        b"reconstruction_hash": digest.encode(),
        b"reconstruction_report": json.dumps(result["report"]).encode(),
        b"track_stats": json.dumps(track_stats).encode(),
    }
    part = path.with_name(path.name + ".part")
    pq.write_table(table.replace_schema_metadata(meta), part, compression="zstd")
    part.replace(path)


def _read_cache(path: Path, digest: str, tracks: Mapping[str, Track]) -> dict | None:
    """Result of :func:`reconstruct_all` from the cache, or None if absent or not matching."""
    if not path.exists():
        return None
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if meta.get(b"reconstruction_hash") != digest.encode():
        return None
    df = table.to_pandas()
    if set(df["track_id"].cat.categories) != set(tracks):
        return None
    codes = df["track_id"].cat.codes.to_numpy()
    names = df["track_id"].cat.categories
    frame, x, v, a, w = (df[c].to_numpy() for c in ("frame_id", "x", "v", "a", "huber_weight"))
    track_stats = json.loads(meta[b"track_stats"])
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    recs = {}
    for i, j in zip(bounds[:-1], bounds[1:]):
        tid = names[codes[i]]
        if not np.array_equal(frame[i:j], tracks[tid].frame):
            return None
        recs[tid] = ReconstructedTrack(x[i:j], v[i:j], a[i:j], derivatives(x[i:j], DT)[2], w[i:j], **track_stats[tid])
    return {"tracks": recs, "report": json.loads(meta[b"reconstruction_report"])}


# ----------------------------------------------------------------------------- entry point
def build_events(data_cfg: Mapping[str, Any], extraction: ExtractionConfig, stats: MutableMapping | None = None) -> EventSet:
    """Loader entry point: raw file -> tracks -> (cached) reconstruction -> pairs -> events.

    The reconstruction is cached in ``<raw dir>/cache/ngsim_<location>_reconstructed.parquet`` and
    reused when its hash of the reconstruction options matches.
    """
    cfg = to_plain(data_cfg)
    if not isinstance(extraction, ExtractionConfig):
        extraction = ExtractionConfig.from_mapping(to_plain(extraction))
    name, site, location = cfg["name"], cfg["site"], cfg["location"]
    raw_path = resolve_path(cfg["raw_path"])
    df = load_raw(raw_path, location=location)
    tracks = build_tracks(df)
    positions, report = "raw", None
    if cfg.get("reconstruct", True):
        options = dict(cfg.get("reconstruction") or {})
        n_jobs = int(options.pop("n_jobs", 1))
        rcfg = ReconstructionConfig.from_mapping(options)
        digest = config_hash(dataclasses.asdict(rcfg))
        cache = raw_path.parent / "cache" / f"ngsim_{location}_reconstructed.parquet"
        result = _read_cache(cache, digest, tracks)
        if result is None:
            print(f"NGSIM {location}: reconstructing {len(tracks)} tracks ({n_jobs} processes)", flush=True)
            result = reconstruct_all(tracks, rcfg, n_jobs=n_jobs)
            _write_cache(cache, tracks, result, digest)
        else:
            print(f"NGSIM {location}: reconstruction read from {cache}", flush=True)
        for tid, rec in result["tracks"].items():
            tracks[tid].rec = rec
        report = validate_reconstruction(tracks, result, cfg)
        positions = "reconstructed"

    lengths = {f"{name}/{site}/{tid}": tr.length for tid, tr in tracks.items()}
    counter: Counter = Counter()
    events = []
    for pair in build_pairs(tracks, name, site, positions, cfg.get("lanes")):
        track = tracks[pair.follower_id.rsplit("/", 1)[1]]
        for ev in extract_events(pair, extraction, counter):
            k = int(np.rint((ev.meta["t0"] - track.t[0]) / DT))
            ev.meta = {**ev.meta, "lane": int(track.lane[k]), "leader_length": lengths[ev.leader_id]}
            if positions == "reconstructed":
                ev.meta["a_source"] = "reconstruction"
            events.append(ev)
    if stats is not None:
        stats.update(counter)
        stats.update(df.attrs["load_stats"])
        if report is not None:
            stats["reconstruction_report"] = report
    return EventSet(events)
