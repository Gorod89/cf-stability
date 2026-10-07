"""IDM formula, equilibrium, conventions and the model wrapper."""

import pytest
import torch

from cf_stability.models.idm import IDM, IDM_BOUNDS, IDM_PARAM_NAMES, idm_acc, idm_equilibrium_spacing

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}


def t64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def test_bounds_follow_contract():
    assert IDM_PARAM_NAMES == ("v0", "T", "s0", "a", "b")
    assert IDM_BOUNDS == {"v0": (10.0, 45.0), "T": (0.3, 3.0), "s0": (0.5, 10.0), "a": (0.2, 4.0), "b": (0.5, 6.0)}


def test_zero_acceleration_at_equilibrium():
    v = t64([0.0, 5.0, 12.0, 20.0, 28.0])
    s_e = idm_equilibrium_spacing(v, P["v0"], P["T"], P["s0"])
    assert torch.allclose(idm_acc(s_e, torch.zeros_like(v), v, **P), torch.zeros_like(v), atol=1e-12)
    assert torch.isinf(idm_equilibrium_spacing(t64([30.0, 35.0]), P["v0"], P["T"], P["s0"])).all()


def test_reference_values():
    # s* = 2 + 20*1.5 + 20*2 / (2*sqrt(1*1.5)) = 32 + 16.329932 = 48.329932
    # a  = 1 * (1 - (20/30)^4 - (48.329932/30)^2) = 1 - 0.197531 - 2.595314 = -1.792845
    assert idm_acc(t64(30.0), t64(2.0), t64(20.0), **P).item() == pytest.approx(-1.792845, abs=1e-6)
    # strongly opening gap: v*T + v*dv/(2 sqrt(ab)) < 0, so s* = s0
    # a = 1 - (10/30)^4 - (2/30)^2 = 1 - 0.012346 - 0.004444 = 0.983210
    assert idm_acc(t64(30.0), t64(-20.0), t64(10.0), **P).item() == pytest.approx(0.983210, abs=1e-6)


def test_sign_conventions():
    dv = torch.linspace(-2.0, 5.0, 50, dtype=torch.float64)  # dv = v - v_lead, closing in when > 0
    assert (idm_acc(t64(30.0), dv, t64(20.0), **P).diff() < 0).all()
    s = torch.linspace(5.0, 100.0, 50, dtype=torch.float64)
    assert (idm_acc(s, t64(0.0), t64(20.0), **P).diff() > 0).all()
    v = torch.linspace(0.0, 35.0, 50, dtype=torch.float64)
    assert (idm_acc(t64(30.0), t64(0.0), v, **P).diff() < 0).all()


def test_spacing_is_clamped():
    acc = idm_acc(t64([-5.0, 0.0, 0.01]), t64([1.0, 1.0, 1.0]), t64([10.0, 10.0, 10.0]), **P)
    assert torch.isfinite(acc).all()
    assert acc[0] == acc[1] == acc[2]


def test_broadcasting_over_parameter_grid():
    gen = torch.Generator().manual_seed(0)
    n, p = 4, 6
    lo = t64([IDM_BOUNDS[k][0] for k in IDM_PARAM_NAMES])
    hi = t64([IDM_BOUNDS[k][1] for k in IDM_PARAM_NAMES])
    theta = lo + (hi - lo) * torch.rand(n, p, 5, generator=gen, dtype=torch.float64)
    s = 5.0 + 40.0 * torch.rand(n, 1, generator=gen, dtype=torch.float64)
    dv = -3.0 + 6.0 * torch.rand(n, 1, generator=gen, dtype=torch.float64)
    v = 25.0 * torch.rand(n, 1, generator=gen, dtype=torch.float64)
    acc = idm_acc(s, dv, v, *theta.unbind(-1))
    assert acc.shape == (n, p)
    for i in range(n):
        for j in range(p):
            ref = idm_acc(s[i, 0], dv[i, 0], v[i, 0], *theta[i, j].unbind())
            assert acc[i, j].item() == pytest.approx(ref.item(), rel=1e-12, abs=1e-12)


def test_model_wrapper_and_gradients():
    model = IDM(P, learnable=True)
    assert model.window == 1 and model.dt == pytest.approx(0.1)
    assert model.params_dict() == pytest.approx(P)
    s, dv, v = t64([20.0, 35.0, 50.0]), t64([1.0, 2.0, 0.5]), t64([10.0, 18.0, 25.0])
    history = torch.stack((s, dv, v), dim=-1)[:, None, :]
    stale = torch.stack((s + 7.0, dv - 1.0, v * 0.5), dim=-1)[:, None, :]
    acc = model(torch.cat((stale, history), dim=1))  # only the last step matters
    assert torch.allclose(acc, idm_acc(s, dv, v, **P), rtol=1e-14, atol=0.0)
    assert torch.allclose(model.acc(s, dv, v), acc, rtol=1e-14, atol=0.0)
    acc.sum().backward()
    grad = model.theta.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad != 0).all()


def test_model_is_frozen_by_default():
    model = IDM([P[k] for k in IDM_PARAM_NAMES])
    assert not model.theta.requires_grad
    assert model.theta.dtype == torch.float64
    assert model.params_dict() == pytest.approx(P)
    with pytest.raises(ValueError):
        IDM([1.0, 2.0])
