"""The callers that carry the state of a stateful model (docs/m12_contract.md, D135): the audit's frequency response,
the rollout gain, the platoon test, the closed-loop evaluation, the trainer and the corridor law (``ModelLaw``; the loop
itself with SUMO: ``tests/test_corridor_laws_m12.py``).

A toy model reveals the state it is given: it counts its calls (``state + 1``), starts from 1000 at the steady state
and from 0 at its initial state, and adds ``EPS * (state - 1000)`` to the IDM, so that the state it saw also shows in
its output; every call records the state it received. A caller that carries the state hands it 1000, 1001, ... (or 0,
1, ...) from call to call; one that does not hands it None."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from cf_stability.data.schema import DT, Event
from cf_stability.models.base import CFModel, steady_view
from cf_stability.models.idm import idm_acc, idm_equilibrium_spacing
from cf_stability.stability.audit import AuditConfig, audit_model
from cf_stability.stability.equilibrium import find_equilibria
from cf_stability.stability.frequency import FrequencyConfig, excite
from cf_stability.stability.penalties import PenaltyConfig, rollout_gain
from cf_stability.stability.platoon import simulate_platoons
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.evaluate import evaluate_closed_loop
from cf_stability.train.trainer import RolloutConfig, TrainConfig, train_model

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
STEADY = 1000.0
EPS = 1e-6


class Counting(CFModel):
    """A stateful toy: the IDM of ``P`` plus ``EPS * (state - 1000)``, the state counts the calls. Records the state
    of every call (class attribute: the audit and the platoon test work on deep copies)."""

    name = "counting"
    window = 3
    stateful = True
    RECORD: list = []

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float64))  # something to train

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, dtype=torch.float64, device=device)

    def steady_state(self, state_history):
        return torch.full(state_history.shape[:1], STEADY, dtype=torch.float64, device=state_history.device)

    def forward(self, state_history, state=None):
        last = state_history[:, -1].double()
        acc = self.gain * idm_acc(last[:, 0], last[:, 1], last[:, 2], **P)
        Counting.RECORD.append(None if state is None else state.detach().double().clone())
        if state is None:
            return acc.to(state_history.dtype)
        return (acc + EPS * (state - STEADY)).to(state_history.dtype), state + 1.0

    def config(self):
        return {}


@pytest.fixture(autouse=True)
def fresh_record():
    Counting.RECORD.clear()
    yield
    Counting.RECORD.clear()


def counted(record: list, start: float) -> bool:
    """The states of consecutive calls are start, start + 1, ... for every sequence of the batch."""
    return all(s is not None and torch.equal(s, torch.full_like(s, start + k)) for k, s in enumerate(record))


def test_audit_starts_its_rollouts_at_the_steady_state_and_carries_it():
    s_e = idm_equilibrium_spacing(torch.tensor([10.0, 20.0], dtype=torch.float64), P["v0"], P["T"], P["s0"])
    cfg = FrequencyConfig(n_omega=3, min_discard_s=5.0, min_measure_s=5.0)
    out = excite(Counting(), s_e, torch.tensor([10.0, 20.0], dtype=torch.float64), torch.tensor([0.5, 1.0]), cfg)
    steps = len(Counting.RECORD)
    assert steps > 100 and counted(Counting.RECORD, STEADY)  # 1000, 1001, ...: carried from the steady state
    assert torch.isfinite(out["gain"]).all()
    # the whole audit: equilibria of the steady-state view (the IDM's, the toy adds nothing there), rollouts as above
    Counting.RECORD.clear()
    audit = audit_model(Counting(), None, AuditConfig(device="cpu", speeds=(10.0, 20.0), frequency=cfg))
    assert audit["state"] == "steady" and audit["analytic_gain"] == "discrete"
    s = [r["s"] for r in audit["equilibria"]]
    assert s == pytest.approx(s_e.tolist(), abs=1e-6)
    received = [r for r in Counting.RECORD if r is not None]
    assert len(received) == len(Counting.RECORD)  # the stateful model was never called without its state
    longest = max(sum(cfg.schedule(float(w), DT)) for w in cfg.omegas())  # calls of the longest rollout
    assert max(float(r.max()) for r in received) == STEADY + (longest - 1) and steps < longest


def test_rollout_gain_carries_the_state():
    cfg = PenaltyConfig(kind="gain", omegas=(0.4, 0.8), horizon_s=3.0, measure_s=1.0)
    v = torch.tensor([12.0, 18.0], dtype=torch.float64)
    s = idm_equilibrium_spacing(v, P["v0"], P["T"], P["s0"])
    gain = rollout_gain(Counting(), s, v, cfg)
    assert gain.shape == (2, 2) and counted(Counting.RECORD, STEADY) and len(Counting.RECORD) == 30
    assert Counting.RECORD[0].shape == (4,)
    Counting.RECORD.clear()
    rollout_gain(Counting(), s, v, cfg, state=torch.full((4,), 7.0, dtype=torch.float64))  # a given state: used
    assert counted(Counting.RECORD, 7.0)


def test_platoon_gives_every_follower_its_state():
    profiles = [torch.full((80,), 15.0, dtype=torch.float64), torch.full((60,), 10.0, dtype=torch.float64)]
    sims = simulate_platoons(Counting(), profiles, n_vehicles=5, start_gap="equilibrium")
    steps = [r for r in Counting.RECORD if r.shape == (2 * 4,)]  # every follower of every profile (the equilibrium
    assert counted(steps, STEADY) and len(steps) == 80 - 1  # search before: other shapes); one call per step
    expected = idm_equilibrium_spacing(torch.tensor(15.0, dtype=torch.float64), P["v0"], P["T"], P["s0"])
    assert sims[0]["start_gap"] == pytest.approx(float(expected), abs=1e-6) and not sims[0]["collided"]
    # the equilibria of the steady-state view are those of the start
    eq = find_equilibria(steady_view(Counting()), torch.tensor([15.0], dtype=torch.float64))
    assert float(eq.s[0]) == pytest.approx(sims[0]["start_gap"], abs=1e-9)


def make_event(i: int, rng: np.random.Generator, n: int = 80) -> Event:
    t = DT * np.arange(n)
    v_lead = 12.0 + 2.0 * np.sin(2 * np.pi * t / 8.0 + rng.uniform(0, 2 * np.pi))
    x_lead = np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    v0 = v_lead[0]
    s0 = idm_equilibrium_spacing(torch.tensor(v0), P["v0"], P["T"], P["s0"]).item() + 2.0
    res = rollout_memoryless(lambda s, dv, v: idm_acc(s, dv, v, **P), torch.tensor(x_lead), torch.tensor(v_lead),
                             torch.tensor(s0), torch.tensor(v0))  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(event_id=f"t/a/{i}|L|0", dataset="t", site="a", follower_id=f"t/a/{i}", leader_id=f"t/a/L{i}", t=t,
                 s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s)  # fmt: skip


def test_closed_loop_evaluation_starts_from_the_initial_state_at_the_first_window():
    rng = np.random.default_rng(0)
    events = [make_event(i, rng, n=60 + 10 * i) for i in range(3)]
    frame = evaluate_closed_loop(Counting(), events, warmup=30)
    assert len(frame) == 3 and counted(Counting.RECORD, 0.0) and len(Counting.RECORD) == 80 - 30  # one call per step


def test_trainer_rollouts_start_from_the_pass():
    """A stateful model without the batched pass: the trainer's sequential pass gives the state after the windows
    ``window - 1 .. j - 1`` of the event (here their number, j - 2), and the rollout carries it from there."""
    rng = np.random.default_rng(1)
    events = [make_event(i, rng) for i in range(6)]
    model = Counting()
    cfg = TrainConfig(batch_size=64, max_epochs=1, steps_per_epoch=1, acc_weight=0.0, warmup=30,
                      rollout=RolloutConfig(horizon=4, segments=5), device="cpu")  # fmt: skip
    seen = {}
    real = train_model.__globals__["rollout_loss"]

    def spy(model_, data, starts, **kwargs):
        seen["starts"], seen["state"] = starts.clone(), kwargs["state"].clone()
        return real(model_, data, starts, **kwargs)

    train_model.__globals__["rollout_loss"] = spy
    try:
        res = train_model(model, events[:4], events[4:], cfg, seed=0)
    finally:
        train_model.__globals__["rollout_loss"] = real
    assert res["state_pass"]["overflow_steps"] == 0
    starts, state = seen["starts"], seen["state"]
    assert torch.equal(state, (starts[:, 1] - model.window + 1).double())  # one call per window of the pass
    first_rollout_call = next(r for r in Counting.RECORD if r is not None and r.shape == (5,) and torch.equal(r, state))
    assert first_rollout_call is not None


def test_model_law_keeps_one_state_per_vehicle():
    from cf_stability.corridor.laws import ModelLaw

    a, b = Counting(), Counting()
    law = ModelLaw("toy", [a, b], "cpu", stateful=True)
    member = law.assign(6, np.random.default_rng(0))
    history = np.random.default_rng(1).uniform(5.0, 30.0, size=(6, 3, 3))
    rows = np.flatnonzero(member == 0)[:2]
    law.accelerations(rows, history[rows])  # first step of these vehicles: the initial state
    assert torch.equal(Counting.RECORD[-1], torch.zeros(len(rows), dtype=torch.float64))
    law.accelerations(rows[:1], history[rows[:1]])  # the first vehicle again: its own carried state
    assert torch.equal(Counting.RECORD[-1], torch.ones(1, dtype=torch.float64))
    other = np.flatnonzero(member == 1)[:1]
    if len(other):
        law.accelerations(other, history[other])  # a vehicle of the other member: its first step
        assert torch.equal(Counting.RECORD[-1], torch.zeros(1, dtype=torch.float64))
    everyone = np.arange(6)
    law.accelerations(everyone, history)
    states = torch.cat([law.states[0][everyone[member == 0]], law.states[1][everyone[member == 1]]])
    assert states.max() <= 3.0 and law.states[0][rows[0]] == 3.0  # three steps of the first vehicle
    Counting.RECORD.clear()
    plain = ModelLaw("toy", [a, b], "cpu", stateful=False)  # the law file says stateful: false (or nothing)
    plain.assign(6, np.random.default_rng(0))
    plain.accelerations(everyone, history)
    assert plain.states is None and all(r is None for r in Counting.RECORD)
    assert math.isfinite(float(np.sum(plain.accelerations(everyone, history))))
