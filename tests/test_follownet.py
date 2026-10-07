"""FollowNet loader: both storage forms, sign of dv, spacing offset, resampling, ids, pickle guard."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from cf_stability.data.extraction import ExtractionConfig
from cf_stability.data.follownet import build_events, load_follownet_array
from cf_stability.utils import resolve_path

CFG = ExtractionConfig()


def profile(n: int, dt: float) -> np.ndarray:
    """[4, n]: spacing, follower speed, relative speed (v_lead - v), leader speed."""
    t = np.arange(n) * dt
    v = 10.0 + np.sin(0.3 * t)
    v_lead = 10.0 + np.sin(0.3 * t + 0.2)
    return np.vstack([20.0 + 2.0 * np.sin(0.5 * t), v, v_lead - v, v_lead])


def write_object_file(path: Path, events: list[np.ndarray]) -> Path:
    """Layout of the published NGSIM / Waymo files: [n_events, 4] object array of 1-D rows."""
    array = np.empty((len(events), 4), dtype=object)
    for i, values in enumerate(events):
        for r in range(4):
            array[i, r] = values[r].copy()
    np.save(path, array, allow_pickle=True)
    return path


def write_sources(path: Path, sha256: str | None = None) -> Path:
    digest = sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    sources = path.parent / "SOURCES.json"
    sources.write_text(json.dumps({"files": {path.name: {"sha256": digest}}}), encoding="utf-8")
    return sources


def test_dense_25hz_resampling_sign_and_ids(tmp_path: Path) -> None:
    second = profile(375, 0.04)
    second[0] += 5.0
    path = tmp_path / "HighD_test.npy"
    np.save(path, np.stack([profile(375, 0.04), second]))
    stats: Counter = Counter()
    data_cfg = {"name": "fn_highd", "file": str(path), "site": "highd", "source_dt": 0.04, "spacing_offset": 0.0}
    events = build_events(data_cfg, CFG, stats)
    events.validate()
    assert len(events) == 2 and not (tmp_path / "cache").exists()
    ev = events[0]
    t = np.arange(150) * 0.1  # 375 frames at 25 Hz -> 150 samples at 10 Hz
    assert len(ev) == 150
    np.testing.assert_allclose(ev.t, t, atol=1e-12)
    np.testing.assert_allclose(ev.s, 20.0 + 2.0 * np.sin(0.5 * t), atol=1e-3)
    np.testing.assert_allclose(ev.v, 10.0 + np.sin(0.3 * t), atol=1e-3)
    # dv = v - v_lead = -(published relative speed)
    np.testing.assert_allclose(ev.dv, np.sin(0.3 * t) - np.sin(0.3 * t + 0.2), atol=1e-3)
    np.testing.assert_allclose(events[1].s, ev.s + 5.0, atol=1e-9)
    np.testing.assert_allclose(ev.x_lead[0], 0.0)
    np.testing.assert_allclose(np.diff(ev.x_lead), 0.05 * (ev.v_lead[1:] + ev.v_lead[:-1]), atol=1e-9)
    assert (ev.follower_id, ev.leader_id) == ("fn_highd/highd/ev0", "fn_highd/highd/ev0_leader")
    assert ev.event_id == "fn_highd/highd/ev0|ev0_leader|0" and events[1].follower_id == "fn_highd/highd/ev1"
    assert ev.meta["a_source"] == "savgol" and ev.meta["spacing_offset"] == 0.0 and ev.meta["source_dt"] == 0.04
    assert stats["source_events"] == 2 and stats["source_events_kept"] == 2 and stats["source_events_split"] == 0
    assert stats["rel_speed_mad"] == pytest.approx(0.0, abs=1e-12)


def test_object_10hz_offset_filter_and_cache(tmp_path: Path) -> None:
    long = profile(400, 0.1)
    long[0, 180:190] = 2.0  # 2 - 4.5 < 0: non-positive net gap, the event is split
    standing = profile(160, 0.1)
    standing[1] = 0.0
    standing[2] = standing[3] - standing[1]
    path = write_object_file(tmp_path / "NGSIM_test.npy", [profile(200, 0.1), long, standing])
    write_sources(path)
    stats: Counter = Counter()
    data_cfg = {"name": "fn_ngsim", "file": str(path), "site": "ngsim_i80", "source_dt": 0.1, "spacing_offset": 4.5}
    events = build_events(data_cfg, CFG, stats)
    events.validate()
    assert [ev.event_id for ev in events] == [
        "fn_ngsim/ngsim_i80/ev0|ev0_leader|0",
        "fn_ngsim/ngsim_i80/ev1|ev1_leader|0_0",
        "fn_ngsim/ngsim_i80/ev1|ev1_leader|0_1",
    ]
    assert [len(ev) for ev in events] == [200, 180, 210] and events[2].meta["t0"] == pytest.approx(19.0)
    np.testing.assert_allclose(events[0].s, profile(200, 0.1)[0] - 4.5, atol=1e-9)
    np.testing.assert_allclose(events[0].dv, -profile(200, 0.1)[2], atol=1e-9)
    assert events[0].meta["spacing_offset"] == 4.5 and events[0].meta["source_file"] == path.name
    assert (stats["source_events"], stats["source_events_kept"], stats["source_events_split"]) == (3, 2, 1)
    assert stats["dropped_not_moving"] == 1

    cache = tmp_path / "cache" / "NGSIM_test.npz"
    with np.load(cache, allow_pickle=False) as stored:
        assert stored["values"].dtype == np.float64 and list(stored["offsets"]) == [0, 200, 600, 760]
    # later calls read the pickle-free cache; the raw file is neither hashed nor unpickled again
    path.write_bytes(path.read_bytes() + b"tampered")
    cached = load_follownet_array(path, tmp_path / "SOURCES.json")
    np.testing.assert_array_equal(cached[1], long)


def test_hash_check_refuses_tampered_file(tmp_path: Path) -> None:
    path = write_object_file(tmp_path / "Waymo_test.npy", [profile(160, 0.1)])
    sources = write_sources(path)
    payload = bytearray(path.read_bytes())
    payload[-16:] = bytes(16)
    path.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="does not match"):
        load_follownet_array(path, sources)
    sources.write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no sha256"):
        load_follownet_array(path, sources)
    assert not (tmp_path / "cache").exists()


WAYMO_CONFIG = resolve_path("configs/data/follownet_waymo.yaml")


@pytest.mark.data
@pytest.mark.skipif(
    not resolve_path("data/raw/follownet/Waymo_filter_car_fol_event_set.npy").exists(), reason="FollowNet not downloaded"
)
def test_real_waymo_build_events() -> None:
    data_cfg = yaml.safe_load(WAYMO_CONFIG.read_text(encoding="utf-8"))
    stats: Counter = Counter()
    events = build_events(data_cfg, CFG, stats)
    events.validate()
    assert stats["source_events"] == 1440 and 0 < len(events)
    assert stats["rel_speed_mad"] == pytest.approx(0.0, abs=1e-9)
