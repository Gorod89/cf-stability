"""Padded event tensors: window ends, windows, targets, segments, training statistics, spacing band."""

import dataclasses
import json

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models.idm import idm_acc
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.tensors import BandConfig, EventTensors, training_context

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
LENGTHS = (150, 163, 181, 157)


def make_event(i: int, n: int) -> Event:
    t = DT * np.arange(n)
    v_lead = 14.0 + 3.0 * np.sin(2 * np.pi * t / (7.0 + i) + i)
    x_lead = 40.0 * i + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, **P), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(22.0 + i), torch.tensor(13.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"t/a/{i}|L|0", dataset="t", site=f"site{i % 2}", follower_id=f"t/a/{i}", leader_id=f"t/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


@pytest.fixture(scope="module")
def events() -> list[Event]:
    return [make_event(i, n) for i, n in enumerate(LENGTHS)]


def observed_states(ev: Event) -> np.ndarray:
    return np.stack((ev.s, ev.v - ev.v_lead, ev.v), axis=1)


@pytest.mark.parametrize("window", [1, 2, 30])
def test_window_ends_stay_inside_their_event(events, window):
    data = EventTensors(events, dtype=torch.float64)
    ends = data.window_ends(window)
    assert ends.dtype == torch.long and ends.shape == (sum(n - window + 1 for n in LENGTHS), 2)
    expected = {(e, j) for e, n in enumerate(LENGTHS) for j in range(window - 1, n)}
    assert set(map(tuple, ends.tolist())) == expected  # no padding, no sample without window - 1 predecessors
    assert data.event_ids == [ev.event_id for ev in events] and data.sites == [ev.site for ev in events]
    assert data.follower_ids == [ev.follower_id for ev in events]
    assert data.lengths.tolist() == list(LENGTHS) and data.mask.sum().item() == sum(LENGTHS)


@pytest.mark.parametrize("window", [1, 30])
def test_windows_and_targets_are_slices_of_the_events(events, window):
    data = EventTensors(events, dtype=torch.float64)
    ends = data.window_ends(window)
    index = ends[torch.randperm(len(ends), generator=torch.Generator().manual_seed(0))[:200]]
    windows, targets = data.state_windows(index, window), data.targets(index)
    assert windows.shape == (200, window, 3) and targets.shape == (200,)
    for (e, j), win, target in zip(index.tolist(), windows.numpy(), targets.numpy()):
        ev = events[e]
        np.testing.assert_array_equal(win, observed_states(ev)[j - window + 1 : j + 1])
        np.testing.assert_allclose(win[:, 1], ev.dv[j - window + 1 : j + 1], rtol=0, atol=1e-9)
        assert target == ev.a[j]


def test_segments_are_masked_past_the_event_end(events):
    data = EventTensors(events, dtype=torch.float64)
    window, horizon = 5, 20
    index = torch.tensor([[0, 4], [1, 100], [2, 170], [3, 156]])  # the last two pass the end of their event
    seg = data.segments(index, window, horizon)
    assert seg["s"].shape == (4, window + horizon)
    for row, (e, j) in enumerate(index.tolist()):
        ev, cols = events[e], np.arange(j - window + 1, j + horizon + 1)
        valid = cols < len(ev)
        np.testing.assert_array_equal(seg["mask"][row].numpy(), valid)
        np.testing.assert_array_equal(seg["s"][row].numpy()[valid], ev.s[cols[valid]])
        np.testing.assert_array_equal(seg["v_lead"][row].numpy()[valid], ev.v_lead[cols[valid]])
        np.testing.assert_allclose(seg["x_lead"][row].numpy()[valid], ev.x_lead[cols[valid]] - ev.x_lead[j], atol=1e-9)
    assert seg["x_lead"][:, window - 1].abs().max() == 0.0


def test_float32_tensors_and_relative_positions(events):
    data = EventTensors(events)
    assert data.state.dtype == data.x_lead.dtype == data.a.dtype == torch.float32
    for i, ev in enumerate(events):
        np.testing.assert_allclose(data.x_lead[i, : len(ev)].numpy(), ev.x_lead - ev.x_lead[0], rtol=1e-6, atol=1e-4)
        assert (data.s[i, len(ev) :] == np.float32(ev.s[-1])).all()  # padding repeats the last sample


def test_training_context_equals_numpy(events):
    ctx = training_context(events, seed=3)
    states = np.concatenate([observed_states(ev) for ev in events])
    np.testing.assert_allclose(ctx["center"], states.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(ctx["scale"], states.std(axis=0), rtol=1e-12)
    np.testing.assert_allclose(ctx["box_low"], np.quantile(states, 0.01, axis=0), rtol=1e-12)
    np.testing.assert_allclose(ctx["box_high"], np.quantile(states, 0.99, axis=0), rtol=1e-12)
    assert ctx["var_s"] == pytest.approx(np.var(states[:, 0]), rel=1e-12)
    assert ctx["var_v"] == pytest.approx(np.var(states[:, 2]), rel=1e-12)
    assert ctx["n_samples"] == sum(LENGTHS) and ctx["seed"] == 3


def samples_event(i: int, s, v, dv=0.0, a=0.0) -> Event:
    """Event made of the given samples (only ``s, v, v_lead, a`` enter the training context)."""
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    v, dv, a = (np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)) for x in (v, dv, a))
    x_lead = np.cumsum(np.full(n, DT)) * 10.0
    return Event(
        event_id=f"b/{i}|L|0", dataset="b", site="a", follower_id=f"b/{i}", leader_id=f"b/L{i}", t=DT * np.arange(n),
        s=s, dv=dv, v=v, a=a, v_lead=v - dv, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


@pytest.fixture(scope="module")
def band_events() -> tuple[list[Event], dict[float, np.ndarray]]:
    """Near-steady samples at 10, 15 and 20 m/s (250, 150, 250) and samples every filter must drop."""
    rng = np.random.default_rng(0)
    spacings = ((10.0, 15.0, 25.0, 250), (15.0, 20.0, 30.0, 150), (20.0, 25.0, 45.0, 250))  # speed, range, count
    steady = {v: rng.uniform(low, high, n) for v, low, high, n in spacings}
    events = [
        samples_event(0, steady[10.0], v=10.0 + rng.uniform(-0.45, 0.45, 250), dv=rng.uniform(-0.45, 0.45, 250)),
        samples_event(1, steady[15.0], v=15.2, a=rng.uniform(-0.29, 0.29, 150)),
        samples_event(2, steady[20.0], v=19.6, dv=-0.3, a=0.2),
        # each of these would move the quantiles at 10 or 20 m/s if it passed its filter
        samples_event(3, np.full(300, 500.0), v=10.0, dv=0.5),  # |dv| not < dv_max
        samples_event(4, np.full(300, 500.0), v=20.0, dv=-0.7),
        samples_event(5, np.full(300, 1.0), v=10.0, a=-0.3),  # |a| not < a_max
        samples_event(6, np.full(300, 1.0), v=10.5),  # |v - v_g| not < half_width for 10 and 11 m/s
        samples_event(7, np.full(300, 1.0), v=4.4),  # no grid speed within half_width
        samples_event(8, np.full(1, 2.0), v=26.0),  # a single sample: fewer than min_samples
    ]
    return events, steady


def test_training_context_band_of_hand_made_samples(band_events):
    events, steady = band_events
    band = training_context(events, seed=0)["band"]
    q = (0.05, 0.5, 0.95)
    expected_rows = [(v, *np.quantile(steady[v], q)) for v in (10.0, 20.0)]
    assert band["v"] == [10.0, 20.0] and band["n"] == [250, 250]  # 15 m/s has 150 < 200 samples; 25 m/s: none steady
    for key, column in zip(("s_low", "s_median", "s_high"), list(zip(*expected_rows))[1:]):
        np.testing.assert_allclose(band[key], column, rtol=1e-12)
    settings = {"dv_max": 0.5, "a_max": 0.3, "half_width": 0.5, "quantiles": [0.05, 0.5, 0.95], "min_samples": 200}
    assert band["settings"] == settings
    assert json.loads(json.dumps(band)) == band  # plain lists and numbers: metrics.json keeps it unchanged

    wide = training_context(events, seed=0, band=BandConfig(min_samples=100, quantiles=(0.1, 0.5, 0.9)))["band"]
    assert wide["v"] == [10.0, 15.0, 20.0] and wide["n"] == [250, 150, 250]
    np.testing.assert_allclose(wide["s_low"], [np.quantile(steady[v], 0.1) for v in (10.0, 15.0, 20.0)], rtol=1e-12)
    # 250 samples within 0.45 m/s of 10 m/s: a narrower window keeps fewer of them
    narrow = training_context(events, seed=0, band=BandConfig(half_width=0.2, min_samples=1))["band"]
    assert narrow["v"][0] == 10.0 and 0 < narrow["n"][0] < 250


def test_training_context_without_band(band_events):
    events, _ = band_events
    assert training_context(events, seed=0, band=BandConfig(min_samples=251))["band"] is None  # no speed listed
    assert training_context(events[:1] + events[3:], seed=0)["band"] is None  # one speed listed
    assert training_context(events[:1] + events[3:], seed=0, band=BandConfig(dv_max=0.8))["band"] is not None


def test_band_config():
    cfg = BandConfig.from_mapping({"min_samples": 50, "quantiles": [0.1, 0.5, 0.9]})
    assert cfg.quantiles == (0.1, 0.5, 0.9) and cfg.min_samples == 50 and BandConfig.from_mapping(None) == BandConfig()
    assert dataclasses.replace(cfg, dv_max=1.0).dv_max == 1.0
    with pytest.raises(ValueError, match="unknown band keys"):
        BandConfig.from_mapping({"quantile": [0.1, 0.5, 0.9]})
    for bad in ((0.5, 0.1, 0.9), (0.1, 0.5), (0.0, 0.5, 1.5)):
        with pytest.raises(ValueError, match="quantiles"):
            BandConfig(quantiles=bad)
