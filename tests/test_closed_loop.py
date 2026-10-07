"""Closed-loop integration, clipping, collisions, padding and metrics."""

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models.idm import IDM, idm_acc, idm_equilibrium_spacing
from cf_stability.train.closed_loop import closed_loop_metrics, pad_events, rollout_memoryless, rollout_model

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}


def t64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def leader(n: int, phase: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    t = DT * torch.arange(n, dtype=torch.float64)
    v_lead = 15.0 + 4.0 * torch.sin(2 * torch.pi * t / 9.0 + phase)
    x_lead = 60.0 + torch.cat((t64([0.0]), torch.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT, 0)))
    return x_lead, v_lead


def make_event(i: int, n: int, params: dict) -> Event:
    x_lead, v_lead = leader(n, phase=0.7 * i)
    res = rollout_memoryless(lambda s, dv, v: idm_acc(s, dv, v, **params), x_lead, v_lead, t64(25.0), t64(15.0))
    ev = Event(
        event_id=f"t/{i}", dataset="t", site="s", follower_id=f"t/s/{i}", leader_id=f"t/s/L{i}",
        t=DT * np.arange(n), s=res.s.numpy(), dv=(res.v - v_lead).numpy(), v=res.v.numpy(), a=res.a.numpy(),
        v_lead=v_lead.numpy(), x_lead=x_lead.numpy(), x_follower=res.x.numpy(),
    )  # fmt: skip
    ev.validate()
    return ev


def test_three_steps_by_hand():
    acc_fn = lambda s, dv, v: 0.1 * (s - 25.0) - 0.5 * dv  # noqa: E731
    res = rollout_memoryless(acc_fn, t64([30.0, 31.0, 32.0, 33.0]), t64([10.0] * 4), t64(25.0), t64(12.0))
    # k=0: dv=2      a=-1                          v1=11.9       x1=5+1.19=6.19          s1=24.81
    # k=1: dv=1.9    a=0.1*(-0.19)-0.95=-0.969     v2=11.8031    x2=6.19+1.18031         s2=24.62969
    # k=2: dv=1.8031 a=-0.037031-0.90155=-0.938581 v3=11.7092419 x3=7.37031+1.17092419 s3=24.45876581
    assert torch.allclose(res.a, t64([-1.0, -0.969, -0.938581, -0.938581]), rtol=0, atol=1e-12)
    assert torch.allclose(res.v, t64([12.0, 11.9, 11.8031, 11.7092419]), rtol=0, atol=1e-12)
    assert torch.allclose(res.x, t64([5.0, 6.19, 7.37031, 8.54123419]), rtol=0, atol=1e-12)
    assert torch.allclose(res.s, t64([25.0, 24.81, 24.62969, 24.45876581]), rtol=0, atol=1e-12)


def test_equilibrium_is_preserved():
    n, v_eq = 400, 20.0
    s_eq = idm_equilibrium_spacing(t64(v_eq), P["v0"], P["T"], P["s0"])
    x_lead = 100.0 + v_eq * DT * torch.arange(n, dtype=torch.float64)
    res = rollout_memoryless(IDM(P).acc, x_lead, torch.full((n,), v_eq, dtype=torch.float64), s_eq, t64(v_eq))
    assert torch.allclose(res.s, s_eq.expand(n), rtol=0, atol=1e-8)
    assert torch.allclose(res.v, torch.full((n,), v_eq, dtype=torch.float64), rtol=0, atol=1e-8)
    assert res.a.abs().max() < 1e-8


def test_clipping_and_non_negative_speed():
    n = 30
    x_lead, v_lead = t64([1000.0] * n), t64([0.0] * n)
    up = rollout_memoryless(lambda s, dv, v: torch.full_like(s, 100.0), x_lead, v_lead, t64(500.0), t64(10.0))
    assert (up.a == 4.0).all()
    assert torch.allclose(up.v, 10.0 + 0.4 * torch.arange(n, dtype=torch.float64))
    down = rollout_memoryless(lambda s, dv, v: torch.full_like(s, -100.0), x_lead, v_lead, t64(500.0), t64(10.0))
    assert (down.a == -8.0).all()
    assert (down.v >= 0.0).all()
    # 10 - 0.8 k: positive up to k = 12, zero from k = 13 on; the follower then stands still
    assert (down.v[:13] > 0).all() and (down.v[13:] == 0.0).all()
    assert (down.x[13:] == down.x[13]).all()


def test_collision_flag_and_start():
    n = 20  # standing leader, follower at a constant 10 m/s from s = 5 m: s_k = 5 - k
    res = rollout_memoryless(lambda s, dv, v: torch.zeros_like(s), t64([50.0] * n), t64([0.0] * n), t64(5.0), t64(10.0))
    assert torch.allclose(res.s, 5.0 - torch.arange(n, dtype=torch.float64))
    s_obs, v_obs = torch.full((n,), 5.0, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    m = closed_loop_metrics(res, s_obs, v_obs, torch.ones(n, dtype=torch.bool))
    assert m["collided"].item() and m["collision_fraction"].item() == pytest.approx(15 / 20)
    assert m["rmse_v"].item() == pytest.approx(10.0)
    early = closed_loop_metrics(res, s_obs, v_obs, torch.arange(n) < 5)
    assert not early["collided"].item() and early["collision_fraction"].item() == 0.0
    late = closed_loop_metrics(res, s_obs, v_obs, torch.ones(n, dtype=torch.bool), start=6)
    assert late["collision_fraction"].item() == 1.0 and late["n_samples"].item() == n - 6


def test_rollout_model_matches_memoryless():
    model = IDM({**P, "T": 1.1})
    x_lead, v_lead = leader(150)
    x_lead = torch.stack((x_lead, x_lead + 5.0, x_lead - 3.0))
    v_lead = v_lead.expand(3, -1)
    s0, v0 = t64([25.0, 30.0, 18.0]), t64([15.0, 17.0, 12.0])
    ref = rollout_memoryless(model.acc, x_lead, v_lead, s0, v0)
    res = rollout_model(model, x_lead, v_lead, s0[:, None], v0[:, None])
    for name in ("s", "v", "x", "a"):
        assert getattr(res, name).shape == (3, 150)
        assert torch.allclose(getattr(res, name), getattr(ref, name), rtol=0, atol=1e-12)


def test_padding_does_not_change_metrics():
    events = [make_event(i, n, P) for i, n in enumerate((151, 187, 240))]
    batch = pad_events(events)
    assert batch["s"].shape == (3, 240) and batch["lengths"].tolist() == [151, 187, 240]
    assert batch["mask"].sum(1).tolist() == [151, 187, 240]
    assert (batch["x_lead"][0, 151:] == events[0].x_lead[-1]).all()
    model = IDM({**P, "T": 1.2, "a": 1.4})  # not the generating parameters: non-zero errors

    def metrics(b: dict) -> dict:
        res = rollout_memoryless(model.acc, b["x_lead"], b["v_lead"], b["s"][:, 0], b["v"][:, 0])
        return closed_loop_metrics(res, b["s"], b["v"], b["mask"])

    padded = metrics(batch)
    assert (padded["rmse_s"] > 0.1).all()
    for i, ev in enumerate(events):
        single = metrics(pad_events([ev]))
        for key in ("rmse_s", "rmse_v", "nrmse_s", "nrmse_v", "collision_fraction", "n_samples"):
            assert padded[key][i].item() == pytest.approx(single[key][0].item(), rel=1e-12, abs=1e-15)
        assert padded["collided"][i].item() == single["collided"][0].item()
