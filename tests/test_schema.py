import numpy as np
import pytest

from cf_stability.data.schema import ARRAY_FIELDS, DT, Event, EventSet, EventValidationError


def make_event(event_id: str = "ds/site/1|2|0", n: int = 60, seed: int = 0, **meta) -> Event:
    rng = np.random.default_rng(seed)
    v = 10.0 + rng.normal(0.0, 0.5, n)
    v_lead = 10.0 + rng.normal(0.0, 0.5, n)
    x_lead = 100.0 + np.cumsum(v_lead) * DT
    s = 20.0 + rng.normal(0.0, 1.0, n)
    return Event(
        event_id=event_id,
        dataset="ds",
        site="site",
        follower_id=event_id.split("|")[0],
        leader_id="ds/site/2",
        t=np.arange(n) * DT,
        s=s,
        dv=v - v_lead,
        v=v,
        a=rng.normal(0.0, 1.0, n),
        v_lead=v_lead,
        x_lead=x_lead,
        x_follower=x_lead - s,
        meta=meta,
    )


def test_parquet_round_trip_is_exact(tmp_path):
    meta = {"t0": 0.1 + 0.2, "a_source": "savgol", "lane": 3, "acc_on": True, "note": None,
            "nested": {"list": [1.5, -2e-300, "Город"], "k": 1}}
    events = EventSet([make_event("ds/site/1|2|0", 60, 0, **meta), make_event("ds/site/7|8|3", 31, 1),
                       make_event("ds/site/1|2|1", 2, 2, t0=1e9 + 0.1)])
    events.to_parquet(tmp_path / "ev", manifest={"config_hash": "abc"})
    loaded = EventSet.from_parquet(tmp_path / "ev")
    assert [ev.event_id for ev in loaded] == [ev.event_id for ev in events]
    for a, b in zip(events, loaded):
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert (a.dataset, a.site, a.follower_id, a.leader_id) == (b.dataset, b.site, b.follower_id, b.leader_id)
        assert a.meta == b.meta
        b.validate()


def test_validate_accepts_consistent_event():
    make_event().validate()


@pytest.mark.parametrize(
    "corrupt, message",
    [
        ("negative_spacing", "non-positive spacing"),
        ("dv_sign", "dv != v - v_lead"),
        ("time", "not uniform"),
        ("positions", "s != x_lead - x_follower"),
    ],
)
def test_validate_rejects(corrupt, message):
    ev = make_event()
    if corrupt == "negative_spacing":
        ev.s[10] = -0.5
        ev.x_follower[10] = ev.x_lead[10] + 0.5  # positions stay consistent
    elif corrupt == "dv_sign":
        ev.dv = ev.v_lead - ev.v
    elif corrupt == "time":
        ev.t[20:] += 0.05
    elif corrupt == "positions":
        ev.x_follower = ev.x_follower + 0.5
    with pytest.raises(EventValidationError, match=message):
        ev.validate()
