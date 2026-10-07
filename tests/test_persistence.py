"""Persistence baseline: window handling and closed-loop behaviour."""

import pytest
import torch

from cf_stability.models.persistence import Persistence
from cf_stability.train.closed_loop import rollout_model


def history(v: list[float]) -> torch.Tensor:
    v = torch.tensor(v, dtype=torch.float64)
    return torch.stack((torch.full_like(v, 30.0), torch.zeros_like(v), v), dim=-1)[None]


def test_window_behaviour():
    model = Persistence()
    assert model.window == 2 and len(list(model.parameters())) == 0
    assert model(history([10.0])).tolist() == [0.0]
    assert model(history([10.0, 10.3])).item() == pytest.approx(3.0)
    assert model(history([4.0, 9.0, 10.0, 10.3])).item() == pytest.approx(3.0)  # last two steps only
    s = torch.tensor([30.0, 20.0], dtype=torch.float64)
    assert model.acc(s, torch.zeros_like(s), torch.tensor([5.0, 6.0], dtype=torch.float64)).tolist() == [0.0, 0.0]


def _rollout(v_hist: list[float], n: int = 400):
    t = 0.1 * torch.arange(n, dtype=torch.float64)
    x_lead, v_lead = (1000.0 + 20.0 * t)[None], torch.full((1, n), 20.0, dtype=torch.float64)
    s_hist = torch.tensor([[50.0, 50.0]], dtype=torch.float64)
    return rollout_model(Persistence(), x_lead, v_lead, s_hist, torch.tensor([v_hist], dtype=torch.float64))


def test_closed_loop_constant_acceleration_until_standstill():
    res = _rollout([10.0, 9.95])
    a, v = res.a[0], res.v[0]
    assert v[:2].tolist() == [10.0, 9.95] and res.s[0, :2].tolist() == [50.0, 50.0]
    stop = int(torch.nonzero(v == 0.0)[0])
    assert stop in (200, 201)  # 9.95 m/s at -0.5 m/s^2
    assert torch.allclose(a[:stop], torch.full((stop,), -0.5, dtype=torch.float64), rtol=0, atol=1e-9)
    assert (v[stop:] == 0.0).all() and (a[stop + 1 :] == 0.0).all()


def test_closed_loop_constant_acceleration_while_moving():
    res = _rollout([10.0, 10.1], n=200)
    assert torch.allclose(res.a[0], torch.ones(200, dtype=torch.float64), rtol=0, atol=1e-9)
    assert torch.allclose(res.v[0], 10.0 + 0.1 * torch.arange(200, dtype=torch.float64), rtol=0, atol=1e-9)
