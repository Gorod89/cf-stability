"""Closed-loop evaluation: scored samples, exactness, batching invariance, collisions, summaries."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from cf_stability.data.schema import DT, Event
from cf_stability.models.base import CFModel, InputScaler
from cf_stability.models.idm import IDM, idm_acc
from cf_stability.models.persistence import Persistence
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.evaluate import COLUMNS, evaluate_closed_loop, open_loop_rmse, summarise

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
OTHER = {**P, "T": 1.1, "a": 1.4}  # not the generating parameters: non-zero errors
LENGTHS = (150, 172, 150, 201, 163, 188)
WARMUP = 30


def make_event(i: int, n: int) -> Event:
    t = DT * np.arange(n)
    v_lead = 15.0 + 4.0 * np.sin(2 * np.pi * t / (8.0 + i) + 0.7 * i)
    x_lead = 60.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, **P), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(20.0 + 2.0 * i), torch.tensor(14.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"e/a/{i}|L|0", dataset="e", site=f"site{i % 3}", follower_id=f"e/a/{i // 2}", leader_id=f"e/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


@pytest.fixture(scope="module")
def events() -> list[Event]:
    return [make_event(i, n) for i, n in enumerate(LENGTHS)]


class WindowedIDM(CFModel):
    """IDM of the last state plus a term of the speed history: uses 30 states, ignores past spacings."""

    window = 30

    def __init__(self) -> None:
        super().__init__()
        self.idm = IDM(OTHER)

    def forward(self, state_history: Tensor) -> Tensor:
        v = state_history[:, -self.window :, 2]
        return self.idm(state_history) - 0.3 * (v[:, -1] - v[:, 0]) / (self.window * DT)


class Accelerating(CFModel):
    def forward(self, state_history: Tensor) -> Tensor:
        return torch.full_like(state_history[:, -1, 0], 2.0)


class SmallMLP(CFModel):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.scaler = InputScaler((25.0, 0.0, 15.0), (8.0, 1.5, 3.0))
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1)).double()

    def forward(self, state_history: Tensor) -> Tensor:
        return 0.3 * self.net(self.scaler(state_history[:, -1])).squeeze(-1)


def perturbed(ev: Event, samples: slice, delta: float = 0.5) -> Event:
    ev = copy.deepcopy(ev)
    ev.s[samples] += delta
    return ev


def test_generating_idm_is_exact(events):
    df = evaluate_closed_loop(IDM(P), events, device="cpu")
    assert list(df.columns) == list(COLUMNS) and df.event_id.tolist() == [ev.event_id for ev in events]
    assert df.n_scored.tolist() == [n - WARMUP + 1 for n in LENGTHS]  # samples 29 .. n - 1
    assert (df.rmse_s < 1e-6).all() and (df.rmse_v < 1e-6).all() and not df.collided.any()
    np.testing.assert_allclose(df.min_s, [ev.s[WARMUP - 1 :].min() for ev in events], rtol=1e-9)


def test_memoryless_rollout_starts_at_sample_29(events):
    model = IDM(OTHER)
    df = evaluate_closed_loop(model, events, device="cpu")
    for ev, row in zip(events, df.itertuples()):
        k = WARMUP - 1
        ref = rollout_memoryless(
            model.acc, torch.tensor(ev.x_lead[k:]), torch.tensor(ev.v_lead[k:]), torch.tensor(ev.s[k]), torch.tensor(ev.v[k])
        )
        assert row.rmse_s == pytest.approx(np.sqrt(np.mean((ref.s.numpy() - ev.s[k:]) ** 2)), rel=1e-9)
        assert row.rmse_v == pytest.approx(np.sqrt(np.mean((ref.v.numpy() - ev.v[k:]) ** 2)), rel=1e-9)


@pytest.mark.parametrize("model", [IDM(OTHER), WindowedIDM()], ids=["memoryless", "windowed"])
def test_scoring_starts_at_sample_29(events, model):
    base = evaluate_closed_loop(model, events, device="cpu")
    assert (base.rmse_s > 0.05).all() and base.n_scored.tolist() == [n - WARMUP + 1 for n in LENGTHS]
    early = evaluate_closed_loop(model, [perturbed(ev, slice(0, WARMUP - 1)) for ev in events], device="cpu")
    pd.testing.assert_frame_equal(early, base)
    start = evaluate_closed_loop(model, [perturbed(ev, slice(WARMUP - 1, WARMUP)) for ev in events], device="cpu")
    assert (start.rmse_s != base.rmse_s).all()


@pytest.mark.parametrize("model", [IDM(OTHER), WindowedIDM(), SmallMLP()], ids=["idm", "windowed", "mlp"])
def test_batching_and_order_do_not_change_results(events, model):
    one = evaluate_closed_loop(model, events, batch_events=64, device="cpu")
    for batch_events, order in ((2, list(range(len(events)))), (4, list(reversed(range(len(events)))))):
        other = evaluate_closed_loop(model, [events[i] for i in order], batch_events=batch_events, device="cpu")
        other = other.set_index("event_id").loc[one.event_id].reset_index()
        pd.testing.assert_frame_equal(other, one, check_exact=False, rtol=1e-12, atol=0)


def test_never_braking_model_collides(events):
    df = evaluate_closed_loop(Accelerating(), events, device="cpu")
    assert df.collided.all() and (df.min_s <= 0).all()
    assert summarise(df)["collision_rate"] == 1.0


def test_summarise_equals_pandas(events):
    df = evaluate_closed_loop(WindowedIDM(), events, device="cpu")
    df.loc[1, "collided"] = True  # a mixed collision column
    out = summarise(df)
    assert out["n_events"] == len(events) and out["collision_rate"] == pytest.approx(1 / len(events))
    for name in ("rmse_s", "rmse_v"):
        assert out[f"{name}_mean"] == pytest.approx(df[name].mean(), rel=1e-12)
        assert out[f"{name}_median"] == pytest.approx(df[name].median(), rel=1e-12)
        pooled = np.sqrt((df[name] ** 2 * df.n_scored).sum() / df.n_scored.sum())
        assert out[f"{name}_pooled"] == pytest.approx(pooled, rel=1e-12)
    df.loc[2, "rmse_s"] = np.nan
    assert np.isnan(summarise(df)["rmse_s_mean"])  # no silent skipping


def test_eval_mode_is_temporary(events):
    model = SmallMLP().train()
    evaluate_closed_loop(model, events, device="cpu")
    assert model.training


def test_open_loop_rmse_on_the_scored_samples(events):
    model = Persistence()
    expected = np.concatenate(
        [(np.diff(ev.v) / DT)[WARMUP - 2 :] - ev.a[WARMUP - 1 :] for ev in events]
    )  # prediction at sample j uses v[j - 1], v[j]
    assert open_loop_rmse(model, events, WARMUP, device="cpu") == pytest.approx(np.sqrt(np.mean(expected**2)), rel=1e-9)
    # the generating IDM is exact except at the last sample, where the stored a repeats a[n - 2]
    last = [idm_acc(*(torch.tensor(x[-1]) for x in (ev.s, ev.dv, ev.v)), **P).item() - ev.a[-1] for ev in events]
    expected_idm = np.sqrt(np.sum(np.square(last)) / sum(n - WARMUP + 1 for n in LENGTHS))
    assert open_loop_rmse(IDM(P), events, WARMUP, device="cpu") == pytest.approx(expected_idm, rel=1e-6)
