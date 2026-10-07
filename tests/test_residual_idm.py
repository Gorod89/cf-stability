"""ResidualIDM: starts at the IDM, bounded residual, certified Jacobian bound."""

import math

import pytest
import torch

from cf_stability.models import ResidualIDM
from cf_stability.models.base import PHYSICAL_SCALE, ModelContext
from cf_stability.models.idm import IDM_PARAM_NAMES, idm_acc

IDM_P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
CONTEXT = ModelContext(center=(30.0, 0.0, 15.0), scale=(15.0, 1.5, 6.0), idm_params=IDM_P)
LOW = torch.tensor([2.0, -5.0, 0.0], dtype=torch.float64)
SPAN = torch.tensor([98.0, 10.0, 35.0], dtype=torch.float64)


def states(n: int, seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (LOW + SPAN * torch.rand(n, 1, 3, generator=gen, dtype=torch.float64)).to(dtype)


def test_untrained_model_is_the_idm():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT).eval()
    assert model.scaler.center.tolist() == [0.0, 0.0, 0.0] and model.scaler.scale.tolist() == list(PHYSICAL_SCALE)
    x = states(2000, 0)
    idm = idm_acc(x[:, 0, 0], x[:, 0, 1], x[:, 0, 2], *(IDM_P[k] for k in IDM_PARAM_NAMES))
    assert (model(x).double() - idm).abs().max() < 0.02
    assert model.residual(x).abs().max() < 1e-5  # antisymmetric start: zero up to rounding


def test_residual_is_bounded():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT, r_max=0.5, lipschitz=4.0).eval()
    with torch.no_grad():
        for p in model.g.parameters():
            p.add_(3.0 * torch.randn_like(p))
    r = model.residual(3.0 * states(2000, 1))
    assert r.shape == (2000,) and r.abs().max() <= 0.5 and r.abs().max() > 0.1


def test_jacobian_bound_holds_after_adversarial_training():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT, lipschitz=2.0)
    with torch.no_grad():  # leave the antisymmetric start, where the sensitivity has no gradient
        for p in model.g.parameters():
            p.add_(0.1 * torch.randn_like(p))
    scale = torch.tensor(PHYSICAL_SCALE)
    opt = torch.optim.Adam(model.g.parameters(), lr=1e-2)
    for step in range(300):  # maximise the sensitivity of the residual to its scaled inputs
        x = states(256, step, torch.float32).requires_grad_()
        (jac,) = torch.autograd.grad(model.residual(x).sum(), x, create_graph=True)
        loss = -((jac[:, 0] * scale) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    norms = model.layer_norms()
    assert model.training and len(norms) == 3 and max(norms) <= 1.05
    bound = model.jacobian_bound()
    assert bound.shape == (3,)
    assert torch.allclose(bound, 2.0 * math.prod(norms) / torch.tensor(PHYSICAL_SCALE, dtype=torch.float64))
    model.eval()
    x = states(2000, 12345, torch.float32).requires_grad_()
    (jac,) = torch.autograd.grad(model.residual(x).sum(), x)
    ratio = jac[:, 0].abs().double() / bound
    assert ratio.max() <= 1.0 + 1e-5
    assert ratio.max() > 0.2  # the training did increase the sensitivity


@pytest.mark.parametrize("learnable", [False, True])
def test_idm_parameters_learn_only_when_learnable(learnable):
    model = ResidualIDM(CONTEXT, idm_learnable=learnable)
    assert model.idm.bounded == learnable
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4481 + 5 * learnable
    assert model.idm.params_dict() == pytest.approx(IDM_P, rel=1e-5)
    model(states(64, 0)).sum().backward()
    assert (model.idm.raw.grad is not None) == learnable
    assert model.g[-1].parametrizations.weight.original.grad is not None
    if learnable:
        assert torch.isfinite(model.idm.raw.grad).all() and model.idm.raw.grad.abs().sum() > 0
