"""Waymo car-following pairs of Hu et al. (2022) (``all_seg_paired_cf_trj_final_with_large_vehicle.csv``).

A pair is (segment_id, follower_id, leader_id); the file holds the rows of both vehicles,
``local_veh_id`` says which one a row describes. Vehicle 0 is the Waymo AV. ``filter_pos`` is
the position of the bounding-box centre along the path, so the follower's front bumper is
``filter_pos + length / 2`` and the leader's rear bumper ``filter_pos - length / 2``.
``filter_speed`` / ``filter_accer`` are filtered by the authors; the follower's
``filter_accer`` is the event acceleration.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events
from cf_stability.data.processing import resample_uniform
from cf_stability.data.schema import DT, EventSet
from cf_stability.utils import resolve_path

DATASET = "waymo"
SITE = "all"
AV_ID = 0
PAIR_TYPES = ("HV-HV", "AV-HV", "HV-AV")  # follower type - leader type
LARGE_VEHICLE_LENGTH = 8.0  # m
DEFAULT_FILE = "all_seg_paired_cf_trj_final_with_large_vehicle.csv"
_KEYS = ["segment_id", "follower_id", "leader_id"]
_COLUMNS = [*_KEYS, "local_veh_id", "length", "local_time", "filter_pos", "filter_speed", "filter_accer"]


def vehicle_type(vehicle: int) -> str:
    return "AV" if vehicle == AV_ID else "HV"


def raw_vehicle_id(segment: int, vehicle: int) -> str:
    """``AV`` for the automated vehicle (one driver across segments), else ``s<segment>_v<id>``."""
    return "AV" if vehicle == AV_ID else f"s{segment}_v{vehicle}"


def load_pairs(path: str | Path, dataset: str = DATASET, dt: float = DT) -> Iterator[PairSeries]:
    """One PairSeries per pair; both vehicles resampled onto one grid starting at the earlier first sample."""
    frame = pd.read_csv(path, usecols=_COLUMNS)
    for (segment, follower, leader), rows in frame.groupby(_KEYS, sort=True):
        segment, follower, leader = int(segment), int(follower), int(leader)
        f_rows = rows[rows["local_veh_id"] == follower]
        l_rows = rows[rows["local_veh_id"] == leader]
        if f_rows.empty or l_rows.empty:
            continue
        t0 = float(min(f_rows["local_time"].min(), l_rows["local_time"].min()))
        f_len, l_len = float(f_rows["length"].iloc[0]), float(l_rows["length"].iloc[0])
        _, f = _resample(f_rows, dt, t0)
        _, lead = _resample(l_rows, dt, t0)
        n = max(f["pos"].size, lead["pos"].size)
        f = {name: _pad(values, n) for name, values in f.items()}
        lead = {name: _pad(values, n) for name, values in lead.items()}
        f_type, l_type = vehicle_type(follower), vehicle_type(leader)
        yield PairSeries(
            dataset=dataset,
            site=SITE,
            follower_id=f"{dataset}/{SITE}/{raw_vehicle_id(segment, follower)}",
            t=t0 + np.arange(n) * dt,
            x_follower=f["pos"] + 0.5 * f_len,
            v=f["speed"],
            leader_id=np.full(n, f"{dataset}/{SITE}/{raw_vehicle_id(segment, leader)}", dtype=object),
            x_lead=lead["pos"] - 0.5 * l_len,
            v_lead=lead["speed"],
            a=f["accel"],
            run=f"s{segment}",
            meta={
                "segment_id": segment,
                "follower_type": f_type,
                "leader_type": l_type,
                "pair_type": f"{f_type}-{l_type}",
                "follower_length": f_len,
                "leader_length": l_len,
                "large_vehicle": bool(max(f_len, l_len) >= LARGE_VEHICLE_LENGTH),
            },
        )


def build_events(data_cfg: Mapping, extraction: ExtractionConfig, stats: Counter | None = None) -> EventSet:
    """Events of the configured pair types (``HV-HV``, ``AV-HV`` = AV follows HV, ``HV-AV``).

    Extra ``stats``: ``pairs``, ``pairs_<type>``, ``events_<type>``.
    """
    stats = Counter() if stats is None else stats
    pair_types = list(data_cfg.get("pair_types") or PAIR_TYPES)
    unknown = set(pair_types) - set(PAIR_TYPES)
    if unknown:
        raise ValueError(f"unknown pair types {sorted(unknown)}; expected a subset of {PAIR_TYPES}")
    path = resolve_path(data_cfg.get("raw_dir", "data/raw/waymo")) / data_cfg.get("file", DEFAULT_FILE)
    events = []
    for pair in load_pairs(path, str(data_cfg.get("name", DATASET)), extraction.dt):
        kind = pair.meta["pair_type"]
        stats["pairs"] += 1
        stats[f"pairs_{kind}"] += 1
        if kind in pair_types:
            new = extract_events(pair, extraction, stats)
            stats[f"events_{kind}"] += len(new)
            events.extend(new)
    return EventSet(events)


def _resample(rows: pd.DataFrame, dt: float, t0: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    columns = {"pos": rows["filter_pos"], "speed": rows["filter_speed"], "accel": rows["filter_accer"]}
    return resample_uniform(rows["local_time"].to_numpy(np.float64), columns, dt=dt, t0=t0)


def _pad(values: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([values, np.full(n - values.size, np.nan)])
