"""Waymo pairs of Hu et al. (2022): bumper convention, AV id, pair types, dataset acceleration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cf_stability.data.extraction import ExtractionConfig
from cf_stability.data.waymo import build_events
from cf_stability.utils import resolve_path

T = np.round(np.arange(0.0, 20.0, 0.1), 2)
CFG = ExtractionConfig()


def vehicle_rows(segment: int, vehicle: int, length: float, t: np.ndarray, pos: np.ndarray, speed: float,
                 accel: float, follower: int, leader: int) -> pd.DataFrame:  # fmt: skip
    n = t.size
    return pd.DataFrame({
        "segment_id": float(segment), "local_veh_id": float(vehicle), "length": length, "local_time": t,
        "follower_id": follower, "leader_id": leader, "filter_pos": pos, "filter_speed": np.full(n, speed),
        "filter_accer": np.full(n, accel),
    })  # fmt: skip


def write_pairs(folder: Path) -> Path:
    """Segment 3: AV (vehicle 0) follows HV 5, HV 7 (12 m truck) follows the AV.
    Segment 4: HV 2 follows HV 1, whose time stamps are shifted by 0.05 s."""
    av = 50.0 + 15.0 * T - 25.0
    rows = [
        vehicle_rows(3, 5, 4.31, T, 50.0 + 15.0 * T, 15.0, 0.0, follower=0, leader=5),
        vehicle_rows(3, 0, 5.18, T, av, 15.0, 0.123, follower=0, leader=5),
        vehicle_rows(3, 7, 12.0, T, av - 30.0, 15.0, -0.05, follower=7, leader=0),
        vehicle_rows(3, 0, 5.18, T, av, 15.0, 0.123, follower=7, leader=0),
        vehicle_rows(4, 1, 4.5, T + 0.05, 10.0 + 12.0 * (T + 0.05), 12.0, 0.0, follower=2, leader=1),
        vehicle_rows(4, 2, 4.5, T, 10.0 + 12.0 * T - 20.0, 12.0, 0.2, follower=2, leader=1),
    ]
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(folder / "pairs.csv", index=False)
    return folder


def build(folder: Path, stats: Counter, **extra):
    events = build_events({"name": "waymo", "raw_dir": str(folder), "file": "pairs.csv", **extra}, CFG, stats)
    events.validate()
    return {ev.event_id: ev for ev in events}


def test_pairs_bumpers_and_ids(tmp_path: Path) -> None:
    stats: Counter = Counter()
    events = build(write_pairs(tmp_path), stats)
    assert sorted(events) == ["waymo/all/AV|s3_v5|s3|0", "waymo/all/s3_v7|AV|s3|0", "waymo/all/s4_v2|s4_v1|s4|0"]

    av = events["waymo/all/AV|s3_v5|s3|0"]
    assert (av.follower_id, av.leader_id) == ("waymo/all/AV", "waymo/all/s3_v5")
    t = av.meta["t0"] + av.t
    np.testing.assert_allclose(av.x_follower, 50.0 + 15.0 * t - 25.0 + 0.5 * 5.18, atol=1e-9)  # centre + L/2
    np.testing.assert_allclose(av.x_lead, 50.0 + 15.0 * t - 0.5 * 4.31, atol=1e-9)  # centre - L/2
    np.testing.assert_allclose(av.s, 25.0 - 0.5 * (5.18 + 4.31), atol=1e-9)
    np.testing.assert_allclose(av.a, 0.123)  # dataset acceleration, not derived from the speed
    assert av.meta | {"t0": 0.0} == {
        "segment_id": 3, "follower_type": "AV", "leader_type": "HV", "pair_type": "AV-HV",
        "follower_length": 5.18, "leader_length": 4.31, "large_vehicle": False, "t0": 0.0,
        "a_source": "dataset", "run": "s3",
    }  # fmt: skip

    truck = events["waymo/all/s3_v7|AV|s3|0"]
    assert (truck.follower_id, truck.leader_id) == ("waymo/all/s3_v7", "waymo/all/AV")
    assert truck.meta["pair_type"] == "HV-AV" and truck.meta["large_vehicle"] is True
    np.testing.assert_allclose(truck.s, 30.0 - 0.5 * (12.0 + 5.18), atol=1e-9)

    # the leader's shifted time stamps are aligned on the follower's grid; its missing first sample cuts
    hv = events["waymo/all/s4_v2|s4_v1|s4|0"]
    assert hv.meta["pair_type"] == "HV-HV" and hv.meta["t0"] == pytest.approx(0.1) and len(hv) == 199
    np.testing.assert_allclose(hv.s, 20.0 - 4.5, atol=1e-9)
    assert stats["pairs"] == 3 and stats["events_AV-HV"] == stats["events_HV-AV"] == stats["events_HV-HV"] == 1


def test_pair_type_selection(tmp_path: Path) -> None:
    folder = write_pairs(tmp_path)
    stats: Counter = Counter()
    assert list(build(folder, stats, pair_types=["HV-HV"])) == ["waymo/all/s4_v2|s4_v1|s4|0"]
    assert (stats["pairs"], stats["pairs_AV-HV"], stats["pairs_HV-AV"], stats["events_AV-HV"]) == (3, 1, 1, 0)
    with pytest.raises(ValueError, match="unknown pair types"):
        build(folder, Counter(), pair_types=["AV-AV"])


CONFIG = resolve_path("configs/data/waymo.yaml")


@pytest.mark.data
@pytest.mark.skipif(
    not resolve_path("data/raw/waymo/all_seg_paired_cf_trj_final_with_large_vehicle.csv").exists(),
    reason="Waymo pairs not downloaded",
)
def test_real_file_build_events() -> None:
    stats: Counter = Counter()
    events = build_events(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), CFG, stats)
    events.validate()
    assert (stats["pairs"], stats["pairs_HV-HV"], stats["pairs_AV-HV"], stats["pairs_HV-AV"]) == (1613, 1117, 210, 286)
    assert {ev.meta["pair_type"] for ev in events} == {"HV-HV", "AV-HV", "HV-AV"}
