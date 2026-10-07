import dataclasses
from collections import Counter

import numpy as np
import pytest

from cf_stability.data import synthetic
from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events, filter_event
from cf_stability.data.processing import acceleration_from_speed, savgol_smooth
from cf_stability.data.schema import DT

FOLLOWER = "ds/s1/F7"


def make_pair(n: int = 600, **overrides) -> PairSeries:
    """Follower 20 m behind its leader; positions need not match the speeds here."""
    t = np.arange(n) * DT
    x_follower = 50.0 + 10.0 * t
    fields = dict(
        dataset="ds",
        site="s1",
        follower_id=FOLLOWER,
        t=t,
        x_follower=x_follower,
        v=10.0 + np.sin(0.3 * t),
        leader_id=np.full(n, "ds/s1/L1", dtype=object),
        x_lead=x_follower + 20.0 + 2.0 * np.sin(0.1 * t),
        v_lead=10.0 + np.cos(0.2 * t),
    )
    fields.update(overrides)
    return PairSeries(**fields)


def spans(pair: PairSeries, events) -> list[tuple[int, int]]:
    """(index of the first sample in the pair, number of samples) of every event."""
    return [(int(np.flatnonzero(np.isclose(pair.t, ev.meta["t0"]))[0]), len(ev)) for ev in events]


def test_single_run():
    pair = make_pair(400)
    stats = Counter()
    (ev,) = extract_events(pair, ExtractionConfig(), stats)
    assert ev.event_id == f"{FOLLOWER}|L1|0" and ev.leader_id == "ds/s1/L1"
    assert ev.meta == {"t0": 0.0, "a_source": "savgol"}
    np.testing.assert_allclose(ev.t, np.arange(400) * DT, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ev.x_lead, pair.x_lead)
    np.testing.assert_array_equal(ev.x_follower, pair.x_follower)
    np.testing.assert_array_equal(ev.s, pair.x_lead - pair.x_follower)
    np.testing.assert_array_equal(ev.dv, pair.v - pair.v_lead)
    np.testing.assert_array_equal(ev.a, acceleration_from_speed(pair.v))
    assert dict(stats) == {"runs": 1, "dropped_short": 0, "dropped_not_moving": 0, "chunks": 0, "events": 1, "samples": 400}


BOUNDARY_CASES = [
    ("leader change", [(0, 250), (250, 350)]),
    ("leader lane change", [(0, 320), (320, 280)]),
    ("follower lane change", [(0, 180), (180, 420)]),
    ("missing leader", [(0, 280), (290, 310)]),
    ("nan speed", [(0, 300), (303, 297)]),
    ("nan dataset acceleration", [(0, 222), (223, 377)]),
    ("time gap", [(0, 350), (350, 250)]),
    ("spacing >= 120", [(0, 200), (210, 390)]),
    ("spacing <= 0", [(0, 440), (441, 159)]),
    ("negative speed", [(0, 300), (301, 299)]),
    ("segment key", [(0, 260), (260, 340)]),
    ("missing segment key", [(0, 330), (331, 269)]),
]


def boundary_pair(case: str) -> PairSeries:
    n = 600
    pair = make_pair(n, lane_follower=np.ones(n, dtype=np.int64), lane_leader=np.full(n, 2, dtype=np.int64))
    if case == "leader change":
        pair.leader_id[250:] = "ds/s1/L2"
    elif case == "leader lane change":
        pair.lane_leader[320:] = 3
    elif case == "follower lane change":
        pair.lane_follower[180:] = 2
    elif case == "missing leader":
        pair.leader_id[280:290] = None
        pair.x_lead[280:290] = np.nan
        pair.v_lead[280:290] = np.nan
    elif case == "nan speed":
        pair.v[300:303] = np.nan
    elif case == "nan dataset acceleration":
        pair.a = np.zeros(n)
        pair.a[222] = np.nan
    elif case == "time gap":
        pair.t[350:] += 0.5
    elif case == "spacing >= 120":
        pair.x_lead[200:210] = pair.x_follower[200:210] + 120.0
    elif case == "spacing <= 0":
        pair.x_lead[440] = pair.x_follower[440]
    elif case == "negative speed":
        pair.v_lead[300] = -0.3
        pair.v[100] = -0.05  # small negative speed: clipped to 0, no boundary
    elif case == "segment key":
        pair.segment_key = np.array(["ACC"] * 260 + ["Human"] * 340)
    elif case == "missing segment key":
        pair.segment_key = np.full(n, "ACC", dtype=object)
        pair.segment_key[330] = None
    return pair


@pytest.mark.parametrize("case, expected", BOUNDARY_CASES, ids=[c for c, _ in BOUNDARY_CASES])
def test_segment_boundaries(case, expected):
    pair = boundary_pair(case)
    events = extract_events(pair, ExtractionConfig())
    assert spans(pair, events) == expected
    for ev in events:
        ev.validate()
        assert ev.s.min() > 0.0 and ev.s.max() < 120.0


def test_min_spacing_cuts_where_the_gap_is_too_small():
    pair = make_pair(600)
    pair.x_lead[300:310] = pair.x_follower[300:310] + 0.3
    assert spans(pair, extract_events(pair, ExtractionConfig(min_spacing=0.0))) == [(0, 600)]
    assert ExtractionConfig().min_spacing == 0.5
    events = extract_events(pair, ExtractionConfig())
    assert spans(pair, events) == [(0, 300), (310, 290)]
    assert min(ev.s.min() for ev in events) > 0.5


def test_small_negative_speed_is_clipped():
    (ev, _) = extract_events(boundary_pair("negative speed"), ExtractionConfig())
    assert ev.v[100] == 0.0 and ev.dv[100] == -ev.v_lead[100]


def test_ids_count_segments_per_leader():
    pair = make_pair(600)
    pair.leader_id[200:400] = "ds/s1/L2"
    events = extract_events(pair, ExtractionConfig())
    assert [ev.event_id for ev in events] == [f"{FOLLOWER}|L1|0", f"{FOLLOWER}|L2|0", f"{FOLLOWER}|L1|1"]
    assert [ev.leader_id for ev in events] == ["ds/s1/L1", "ds/s1/L2", "ds/s1/L1"]
    assert [ev.meta["t0"] for ev in events] == pytest.approx([0.0, 20.0, 40.0])


def test_run_tag_and_segment_key_in_ids_and_meta():
    pair = make_pair(600, run="rec_3", segment_key=np.array([0] * 300 + [1] * 300, dtype=np.int64))
    events = extract_events(pair, ExtractionConfig())
    assert [ev.event_id for ev in events] == [f"{FOLLOWER}|L1|rec_3|0", f"{FOLLOWER}|L1|rec_3|1"]
    assert [ev.meta["segment_key"] for ev in events] == [0, 1]
    assert all(type(ev.meta["segment_key"]) is int and ev.meta["run"] == "rec_3" for ev in events)
    (ev,) = extract_events(make_pair(300, segment_key=np.full(300, "ACC")), ExtractionConfig())
    assert type(ev.meta["segment_key"]) is str and "run" not in ev.meta


def test_duration_rule_is_150_samples():
    cfg = ExtractionConfig()
    assert cfg.min_samples == 150
    assert len(extract_events(make_pair(150), cfg)) == 1
    stats = Counter()
    assert extract_events(make_pair(149), cfg, stats) == [] and stats["dropped_short"] == 1


def test_stats_counters():
    pair = make_pair(700)
    pair.v[100] = np.nan
    pair.v[400] = np.nan
    pair.v[101:400] = 0.0  # standing follower
    stats = Counter()
    events = extract_events(pair, ExtractionConfig(), stats)
    assert spans(pair, events) == [(401, 299)]
    assert dict(stats) == {"runs": 3, "dropped_short": 1, "dropped_not_moving": 1, "chunks": 0, "events": 1, "samples": 299}


@pytest.mark.parametrize("n_standing, kept", [(40, True), (41, False)])
def test_moving_fraction(n_standing, kept):
    pair = make_pair(200)
    pair.v[:n_standing] = 0.05
    events = extract_events(pair, ExtractionConfig())
    assert len(events) == int(kept)
    for ev in events:
        assert np.mean(ev.v > 0.1) >= 0.8


@pytest.mark.parametrize("n, lengths", [(400, [400]), (401, [200, 201]), (1000, [333, 333, 334])])
def test_chunking(n, lengths):
    pair = make_pair(n)
    stats = Counter()
    events = extract_events(pair, ExtractionConfig(max_duration=40.0), stats)
    assert [len(ev) for ev in events] == lengths
    assert [ev.event_id for ev in events] == [f"{FOLLOWER}|L1|{k}" for k in range(len(lengths))]
    assert spans(pair, events) == list(zip(np.cumsum([0] + lengths[:-1]).tolist(), lengths))
    assert stats["runs"] == 1 and stats["chunks"] == len(lengths) - 1 and stats["events"] == len(lengths)
    # chunks share the acceleration filtered over the whole run
    np.testing.assert_array_equal(np.concatenate([ev.a for ev in events]), acceleration_from_speed(pair.v))


def test_dataset_acceleration_is_used():
    pair = make_pair(300)
    pair.a = 0.3 * np.sin(pair.t)
    (ev,) = extract_events(pair, ExtractionConfig())
    np.testing.assert_array_equal(ev.a, pair.a)
    assert ev.meta["a_source"] == "dataset"


def test_smooth_speeds():
    pair = make_pair(400)
    rng = np.random.default_rng(0)
    pair.v = pair.v + rng.normal(0.0, 0.3, 400)
    pair.v_lead = pair.v_lead + rng.normal(0.0, 0.3, 400)
    (raw,) = extract_events(pair, ExtractionConfig())
    (ev,) = extract_events(pair, ExtractionConfig(smooth_speeds=True))
    np.testing.assert_array_equal(raw.v, pair.v)
    np.testing.assert_array_equal(ev.v, np.maximum(savgol_smooth(pair.v), 0.0))
    np.testing.assert_array_equal(ev.v_lead, np.maximum(savgol_smooth(pair.v_lead), 0.0))
    np.testing.assert_array_equal(ev.dv, ev.v - ev.v_lead)
    np.testing.assert_array_equal(ev.s, raw.s)
    np.testing.assert_array_equal(ev.a, raw.a)


def test_synthetic_events_satisfy_the_criteria():
    stats = Counter()
    data_cfg = {"n_platoons": 2, "n_vehicles": 5, "duration": 150.0, "noise_speed": 0.3, "seed": 7}
    events = synthetic.build_events(data_cfg, ExtractionConfig(max_duration=60.0), stats)
    assert len(events) > 0 and len({ev.event_id for ev in events}) == len(events)
    for ev in events:
        ev.validate()
        assert len(ev) >= 150
        assert ev.s.min() > 0.0 and ev.s.max() < 120.0
        assert np.mean(ev.v > 0.1) >= 0.8
        assert np.array_equal(ev.dv > 0, ev.v > ev.v_lead)
        follower = int(ev.follower_id.rsplit("/", 1)[-1])
        assert ev.leader_id == f"synthetic/{ev.site}/{follower - 1}"
        assert ev.event_id.split("|")[1] == str(follower - 1)
        assert ev.meta["a_source"] == "savgol" and ev.meta["leader_length"] > 0
    assert stats["events"] == len(events) and stats["samples"] == sum(len(ev) for ev in events)
    assert stats["events"] == stats["runs"] - stats["dropped_short"] - stats["dropped_not_moving"] + stats["chunks"]


def test_synthetic_platoon_follows_the_contract():
    platoon = synthetic.simulate_platoon(5, 120.0, seed=3, noise_speed=0.2)
    gaps = platoon.x[:-1] - platoon.length[:-1, None] - platoon.x[1:]
    assert gaps.min() > 0.0
    assert platoon.v[0].min() < 0.1 < platoon.v[0].max()  # head vehicle stops and goes
    # semi-implicit Euler: v[k+1] = max(v[k] + a[k] dt, 0), x[k+1] = x[k] + v[k+1] dt
    np.testing.assert_allclose(platoon.v[1:, 1:], np.maximum(platoon.v[1:, :-1] + platoon.a[1:, :-1] * DT, 0.0))
    np.testing.assert_allclose(np.diff(platoon.x, axis=1), platoon.v[:, 1:] * DT)
    assert platoon.v_measured.min() >= 0.0 and not np.array_equal(platoon.v_measured, platoon.v)


def test_filter_event():
    cfg = ExtractionConfig()
    (ev,) = extract_events(make_pair(600), cfg)
    ev.meta["t0"] = 100.0
    (same,) = filter_event(ev, cfg)
    assert same.event_id == ev.event_id and same.meta == ev.meta
    np.testing.assert_array_equal(same.v, ev.v)

    v = ev.v.copy()
    v[300] = np.nan
    stats = Counter()
    parts = filter_event(dataclasses.replace(ev, v=v), cfg, stats)
    assert [p.event_id for p in parts] == [f"{ev.event_id}_0", f"{ev.event_id}_1"]
    assert [p.meta["t0"] for p in parts] == pytest.approx([100.0, 130.1])
    assert [len(p) for p in parts] == [300, 299] and stats["runs"] == 2
    np.testing.assert_array_equal(parts[1].a, ev.a[301:])  # acceleration kept as it is
    assert all(p.meta["a_source"] == "savgol" for p in parts)

    v = ev.v.copy()
    v[150:] = 0.0
    assert filter_event(dataclasses.replace(ev, v=v), cfg) == []  # not moving
    assert filter_event(ev, dataclasses.replace(cfg, min_duration=70.0)) == []  # too short
