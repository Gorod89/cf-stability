"""FollowNet car-following event sets (figshare, CC0; format in docs/data_contract.md, 4a).

Each event has four rows: spacing, follower speed, relative speed (``v_lead - v``), leader
speed. The published relative speed is not used (``dv = v - v_lead``, D17). Events carry no
identifiers: every event is its own driver (D19). Gross spacings are reduced by
``spacing_offset`` (D18). Pickled files are loaded only after a sha256 check (D21).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid

from cf_stability.data.extraction import ExtractionConfig, filter_event
from cf_stability.data.processing import acceleration_from_speed, resample_uniform
from cf_stability.data.schema import Event, EventSet
from cf_stability.utils import read_json, resolve_path

SOURCES_FILE = "SOURCES.json"


def load_follownet_array(path: str | Path, sources_json: str | Path) -> list[np.ndarray]:
    """Events of a FollowNet ``.npy`` file as ``[4, n]`` float64 arrays.

    A dense array is read without pickle. An object array is unpickled only when the file's
    sha256 equals the entry ``files.<name>.sha256`` of ``sources_json``; its content is then
    cached pickle-free in ``<dir>/cache/<stem>.npz`` (values + offsets), which later calls read.
    """
    path = Path(path)
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError:  # object array, stored with pickle
        pass
    else:
        return _checked(path, [np.asarray(values, dtype=np.float64) for values in array])
    expected = _expected_sha256(path, Path(sources_json))
    cache = path.parent / "cache" / f"{path.stem}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["sha256"]) == expected:
                return _split(stored["values"], stored["offsets"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} does not match {expected} in {sources_json}; refusing to unpickle")
    # the published object arrays are [n_events, 4] with one 1-D array per row
    rows = np.load(path, allow_pickle=True)
    events = _checked(path, [np.vstack([np.asarray(row, dtype=np.float64) for row in event]) for event in rows])
    _write_cache(cache, events, expected)
    return events


def build_events(data_cfg: Mapping, extraction: ExtractionConfig, stats: Counter | None = None) -> EventSet:
    """Events of one FollowNet file, passed through the extraction filters.

    Config keys: ``name``, ``file``, ``site``, ``source_dt`` (s), ``spacing_offset`` (m).
    Extra ``stats``: ``source_events``, ``source_events_kept`` (at least one event left),
    ``source_events_split`` (more than one), ``rel_speed_mad`` (mean absolute difference
    between the published relative speed and ``v_lead - v``, m/s).
    """
    stats = Counter() if stats is None else stats
    path = resolve_path(data_cfg["file"])
    arrays = load_follownet_array(path, path.parent / SOURCES_FILE)
    dataset, site = str(data_cfg["name"]), str(data_cfg["site"])
    source_dt = float(data_cfg["source_dt"])
    offset = float(data_cfg.get("spacing_offset", 0.0))
    events = []
    abs_diff, n_values = 0.0, 0
    for index, values in enumerate(arrays):
        abs_diff += float(np.sum(np.abs(values[2] - (values[3] - values[1]))))
        n_values += values.shape[1]
        event = follownet_event(values, index, dataset, site, source_dt, offset, extraction)
        event.meta["source_file"] = path.name
        kept = filter_event(event, extraction, stats)
        stats["source_events_kept"] += int(len(kept) > 0)
        stats["source_events_split"] += int(len(kept) > 1)
        events.extend(kept)
    stats["source_events"] += len(arrays)
    stats["rel_speed_mad"] = abs_diff / max(n_values, 1)
    return EventSet(events)


def follownet_event(
    values: np.ndarray,
    index: int,
    dataset: str,
    site: str,
    source_dt: float,
    spacing_offset: float,
    cfg: ExtractionConfig,
) -> Event:
    """Event from one ``[4, n]`` FollowNet array, not yet filtered (it may violate the schema)."""
    spacing, v, v_lead = values[0], values[1], values[3]
    if not math.isclose(source_dt, cfg.dt):  # linear interpolation, no anti-aliasing (D20)
        t_source = np.arange(values.shape[1]) * source_dt
        _, columns = resample_uniform(t_source, {"s": spacing, "v": v, "v_lead": v_lead}, dt=cfg.dt)
        spacing, v, v_lead = columns["s"], columns["v"], columns["v_lead"]
    s = spacing - spacing_offset
    x_lead = cumulative_trapezoid(v_lead, dx=cfg.dt, initial=0.0)
    follower_id = f"{dataset}/{site}/ev{index}"
    leader = f"ev{index}_leader"
    return Event(
        event_id=f"{follower_id}|{leader}|0",
        dataset=dataset,
        site=site,
        follower_id=follower_id,
        leader_id=f"{dataset}/{site}/{leader}",
        t=np.arange(s.shape[0]) * cfg.dt,
        s=s,
        dv=v - v_lead,
        v=v,
        a=acceleration_from_speed(v, cfg.dt, cfg.sg_window_s, cfg.sg_order),
        v_lead=v_lead,
        x_lead=x_lead,
        x_follower=x_lead - s,
        meta={
            "t0": 0.0,
            "a_source": "savgol",
            "source_index": index,
            "source_dt": source_dt,
            "spacing_offset": spacing_offset,
        },
    )


def _expected_sha256(path: Path, sources_json: Path) -> str:
    entry = read_json(sources_json).get("files", {}).get(path.name, {}) if sources_json.exists() else {}
    if not entry.get("sha256"):
        raise ValueError(f"{path}: no sha256 in {sources_json}; refusing to unpickle")
    return str(entry["sha256"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(path: Path, events: list[np.ndarray]) -> list[np.ndarray]:
    for index, values in enumerate(events):
        if values.ndim != 2 or values.shape[0] != 4:
            raise ValueError(f"{path}: event {index} has shape {values.shape}, expected (4, n)")
    return events


def _split(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[:, a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def _write_cache(cache: Path, events: list[np.ndarray], sha256: str) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([e.shape[1] for e in events])]).astype(np.int64)
    values = np.concatenate(events, axis=1) if events else np.empty((4, 0))
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, values=values, offsets=offsets, sha256=np.array(sha256))
    os.replace(tmp, cache)
