"""highD loader on a synthetic recording: bumper convention, driving direction 1, leader and lane changes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cf_stability.data.extraction import ExtractionConfig
from cf_stability.data.highd import assert_inside_project, build_events
from cf_stability.utils import REPO_ROOT

FRAMES = np.arange(1, 1001)  # 40 s at 25 Hz
T = FRAMES / 25.0
CHANGE = FRAMES > 450  # vehicle 1 changes into lane 6 in front of vehicle 3 at frame 451

TRACK_COLUMNS = [
    "frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration",
    "frontSightDistance", "backSightDistance", "dhw", "thw", "ttc", "precedingXVelocity", "precedingId",
    "followingId", "leftPrecedingId", "leftAlongsideId", "leftFollowingId", "rightPrecedingId",
    "rightAlongsideId", "rightFollowingId", "laneId",
]  # fmt: skip


def front_3(t: np.ndarray) -> np.ndarray:
    return 160.0 + 25.0 * t + 2.5 * (np.cos(0.2 * t) - 1.0)


def x_5(t: np.ndarray) -> np.ndarray:
    return 400.0 - 22.0 * t + 4.2 + 25.0 + 3.0 * np.sin(0.3 * t)


def vehicles() -> dict[int, dict]:
    """Direction 2: 2 (car) leads 3 (truck); 1 (car) cuts in between them at frame 451.
    Direction 1: 4 (car) leads 5 (truck)."""
    zero = np.zeros_like(T)
    return {
        1: dict(length=4.5, cls="Car", direction=2, x=175.0 + 25.0 * T, vx=zero + 25.0, ax=zero,
                lane=np.where(CHANGE, 6, 5), lead=np.where(CHANGE, 2, 0), dhw=np.where(CHANGE, 20.5, 0.0)),
        2: dict(length=4.5, cls="Car", direction=2, x=200.0 + 25.0 * T, vx=zero + 25.0, ax=zero,
                lane=zero + 6, lead=zero, dhw=zero),
        3: dict(length=16.0, cls="Truck", direction=2, x=front_3(T) - 16.0, vx=25.0 - 0.5 * np.sin(0.2 * T),
                ax=-0.1 * np.cos(0.2 * T), lane=zero + 6, lead=np.where(CHANGE, 1, 2),
                dhw=np.where(CHANGE, 17.5 - 2.5 * np.cos(0.2 * T), 42.5 - 2.5 * np.cos(0.2 * T))),
        4: dict(length=4.2, cls="Car", direction=1, x=400.0 - 22.0 * T, vx=zero - 22.0, ax=zero,
                lane=zero + 2, lead=zero, dhw=zero),
        5: dict(length=12.0, cls="Truck", direction=1, x=x_5(T), vx=-22.0 + 0.9 * np.cos(0.3 * T),
                ax=-0.27 * np.sin(0.3 * T), lane=zero + 2, lead=zero + 4, dhw=25.0 + 3.0 * np.sin(0.3 * T)),
    }  # fmt: skip


def write_recording(folder: Path, recording: int = 1) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    for vid, veh in vehicles().items():
        n = FRAMES.size
        frame = pd.DataFrame({name: np.zeros(n) for name in TRACK_COLUMNS})
        frame["frame"], frame["id"] = FRAMES, vid
        frame["x"], frame["y"] = veh["x"], 3.5 * veh["lane"]
        frame["width"], frame["height"] = veh["length"], 2.5 if veh["cls"] == "Truck" else 1.8
        frame["xVelocity"], frame["xAcceleration"] = veh["vx"], veh["ax"]
        frame["dhw"], frame["precedingId"], frame["laneId"] = veh["dhw"], veh["lead"].astype(int), veh["lane"].astype(int)
        rows.append(frame)
        meta.append({
            "id": vid, "width": veh["length"], "height": 2.5 if veh["cls"] == "Truck" else 1.8, "initialFrame": 1,
            "finalFrame": int(FRAMES[-1]), "numFrames": n, "class": veh["cls"], "drivingDirection": veh["direction"],
            "traveledDistance": 900.0, "minXVelocity": 0.0, "maxXVelocity": 0.0, "meanXVelocity": 0.0,
            "minDHW": 0.0, "minTHW": 0.0, "minTTC": 0.0, "numLaneChanges": int(vid == 1),
        })  # fmt: skip
    tracks = pd.concat(rows).sort_values(["frame", "id"], kind="stable")  # interleaved, the loader sorts
    prefix = folder / f"{recording:02d}"
    tracks.to_csv(f"{prefix}_tracks.csv", index=False)
    pd.DataFrame(meta).to_csv(f"{prefix}_tracksMeta.csv", index=False)
    pd.DataFrame([{
        "id": recording, "frameRate": 25, "locationId": 2, "speedLimit": -1, "month": "09.2017", "weekDay": "Tue",
        "startTime": "08:00", "duration": 40.0, "totalDrivenDistance": 4500.0, "totalDrivenTime": 200.0,
        "numVehicles": 5, "numCars": 3, "numTrucks": 2, "upperLaneMarkings": "8.5;12.0;15.5",
        "lowerLaneMarkings": "21.0;24.5;28.0",
    }]).to_csv(f"{prefix}_recordingMeta.csv", index=False)  # fmt: skip
    return folder


@pytest.fixture
def events_and_stats(tmp_path: Path):
    stats: Counter = Counter()
    events = build_events({"name": "highd", "raw_dir": str(write_recording(tmp_path)), "recordings": None}, ExtractionConfig(), stats)
    events.validate()
    return {ev.event_id: ev for ev in events}, stats


def source_time(ev) -> np.ndarray:
    return ev.meta["t0"] + ev.t


def test_events_ids_and_leader_change(events_and_stats) -> None:
    events, _ = events_and_stats
    assert sorted(events) == [
        "highd/loc2/r01_v1|r01_v2|r01|0",
        "highd/loc2/r01_v3|r01_v1|r01|0",
        "highd/loc2/r01_v3|r01_v2|r01|0",
        "highd/loc2/r01_v5|r01_v4|r01|0",
    ]
    before, after = events["highd/loc2/r01_v3|r01_v2|r01|0"], events["highd/loc2/r01_v3|r01_v1|r01|0"]
    assert before.meta["t0"] == pytest.approx(0.04) and source_time(before)[-1] < 18.0 + 1e-9
    assert source_time(after)[0] == pytest.approx(18.04, abs=0.11)
    assert abs(len(before) - 180) <= 1 and abs(len(after) - 220) <= 1
    assert before.follower_id == "highd/loc2/r01_v3" and before.leader_id == "highd/loc2/r01_v2"
    # the vehicle that changed lanes follows vehicle 2 only in its new lane
    cut_in = events["highd/loc2/r01_v1|r01_v2|r01|0"]
    assert cut_in.meta["lane"] == 6 and source_time(cut_in)[0] > 18.0
    assert (after.meta["leader_class"], after.meta["leader_length"]) == ("Car", 4.5)


def test_bumper_convention_direction_2(events_and_stats) -> None:
    events, _ = events_and_stats
    ev = events["highd/loc2/r01_v3|r01_v2|r01|0"]
    t = source_time(ev)
    np.testing.assert_allclose(ev.x_follower, front_3(t), atol=1e-3)  # x + width
    np.testing.assert_allclose(ev.x_lead, 200.0 + 25.0 * t, atol=1e-3)  # leader x
    np.testing.assert_allclose(ev.s, 42.5 - 2.5 * np.cos(0.2 * t), atol=1e-3)
    np.testing.assert_allclose(ev.a, -0.1 * np.cos(0.2 * t), atol=1e-4)
    assert ev.meta["a_source"] == "dataset" and ev.meta["follower_class"] == "Truck" and ev.meta["driving_direction"] == 2


def test_direction_1_signs(events_and_stats) -> None:
    events, _ = events_and_stats
    ev = events["highd/loc2/r01_v5|r01_v4|r01|0"]
    t = source_time(ev)
    assert len(ev) == 400
    np.testing.assert_allclose(ev.x_follower, -x_5(t), atol=1e-3)  # front = -x
    np.testing.assert_allclose(ev.x_lead, -(400.0 - 22.0 * t + 4.2), atol=1e-3)  # rear = -(x + width)
    np.testing.assert_allclose(ev.s, 25.0 + 3.0 * np.sin(0.3 * t), atol=1e-3)
    np.testing.assert_allclose(ev.v, 22.0 - 0.9 * np.cos(0.3 * t), atol=1e-3)
    np.testing.assert_allclose(ev.v_lead, 22.0, atol=1e-9)
    np.testing.assert_allclose(ev.a, 0.27 * np.sin(0.3 * t), atol=1e-4)
    assert ev.meta | {"t0": 0} == {
        "recording": 1, "follower_class": "Truck", "follower_length": 12.0, "driving_direction": 1, "lane": 2,
        "leader_class": "Car", "leader_length": 4.2, "t0": 0, "a_source": "dataset", "run": "r01",
    }  # fmt: skip


def test_dhw_statistic_is_zero_for_net_gap(events_and_stats) -> None:
    _, stats = events_and_stats
    assert stats["recordings"] == 1 and stats["dhw_frames"] == 1000 + 550 + 1000
    assert stats["dhw_abs_diff_max"] < 1e-9 and stats["dhw_abs_diff_mean"] < 1e-9


def test_assert_inside_project(tmp_path: Path) -> None:
    assert assert_inside_project("data/events/highd") == (REPO_ROOT / "data/events/highd").resolve()
    with pytest.raises(PermissionError):
        assert_inside_project(tmp_path / "highd")
