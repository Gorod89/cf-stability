"""Linearisation and analytic criterion (cf_stability/stability/analytic.py)."""

import math

import pytest
import torch

from cf_stability.models import GRU, IDM, MLP, OVM
from cf_stability.models.base import InputScaler, ModelContext
from cf_stability.stability.analytic import (
    criterion,
    history_jacobian,
    partials,
    transfer_continuous,
    transfer_discrete,
    transfer_windowed,
)
from cf_stability.stability.equilibrium import find_equilibria

IDM_PARAMS = {"v0": 30.0, "T": 1.2, "s0": 2.0, "a": 1.0, "b": 1.5}
CONTEXT = ModelContext(center=(30.0, 0.0, 17.0), scale=(15.0, 1.0, 7.0))
OMEGA = torch.logspace(-2, math.log10(2.0), 25, dtype=torch.float64)


def f64(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def idm_derivatives(s: torch.Tensor, v: torch.Tensor, v0: float, T: float, s0: float, a: float, b: float):
    """Closed-form derivatives of ``a (1 - (v/v0)^4 - (s*/s)^2)``, ``s* = s0 + v T + v dv / (2 sqrt(ab))``, at dv = 0."""
    s_star = s0 + v * T
    f_s = 2.0 * a * s_star**2 / s**3
    f_dv = -2.0 * a * s_star / s**2 * v / (2.0 * math.sqrt(a * b))
    f_v = -4.0 * a * v**3 / v0**4 - 2.0 * a * s_star / s**2 * T
    return f_s, f_dv, f_v


def test_idm_partials_match_closed_form():
    idm = IDM(IDM_PARAMS)
    eq = find_equilibria(idm, [5.0, 12.0, 20.0, 28.0])
    for got, expected in zip(partials(idm, eq.s, eq.v), idm_derivatives(eq.s, eq.v, **IDM_PARAMS)):
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=0.0)


def test_ovm_is_string_unstable_on_the_congested_branch():
    params = {"v0": 25.5, "tau": 1.2, "s0": 3.0}
    ovm = OVM(ModelContext(ovm_params=params))
    eq = find_equilibria(ovm)
    congested = eq.v < params["v0"]
    f_s, f_dv, f_v = partials(ovm, eq.s[congested], eq.v[congested])
    tau = params["tau"]
    torch.testing.assert_close(f_s, torch.full_like(f_s, 1.0 / tau**2))
    torch.testing.assert_close(f_dv, torch.zeros_like(f_dv))
    torch.testing.assert_close(f_v, torch.full_like(f_v, -1.0 / tau))
    crit = criterion(f_s, f_dv, f_v)
    torch.testing.assert_close(crit["margin"], torch.full_like(f_s, -1.0 / tau**2))
    assert not crit["string_stable"].any() and crit["local_stable"].all()


@pytest.mark.parametrize("name", ["mlp", "gru"])
def test_create_graph_differentiates_with_respect_to_the_parameters(name):
    torch.manual_seed(0)
    model = (MLP(CONTEXT) if name == "mlp" else GRU(CONTEXT, hidden_size=8, window=5)).double()
    first = model.net[0].weight if name == "mlp" else model.rnn.weight_ih_l0
    with torch.backends.cudnn.flags(enabled=False):
        f_s, f_dv, f_v = partials(model, f64(20.0, 35.0), f64(10.0, 20.0), create_graph=True)
        loss = (f_s**2 + f_dv**2 + f_v**2).sum()
        (grad,) = torch.autograd.grad(loss, first)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_finite_differences_match_autograd_on_a_smooth_law():
    idm = IDM(IDM_PARAMS)
    idm.scaler = InputScaler((0.0, 0.0, 0.0), (2.0, 0.5, 1.0))  # sets the steps 0.25 * scale
    eq = find_equilibria(idm, [8.0, 15.0, 25.0])
    autograd = partials(idm, eq.s, eq.v)
    idm.differentiable = False
    for fd, exact in zip(partials(idm, eq.s, eq.v), autograd):
        torch.testing.assert_close(fd, exact, rtol=0.01, atol=0.0)


def test_criterion_flags():
    f_s = f64(0.1, 0.1, -0.1, 0.1, 0.1)
    f_dv = f64(-0.5, -0.5, -0.5, 0.2, -0.1)
    f_v = f64(-0.1, -0.5, -0.1, -0.1, 0.2)
    crit = criterion(f_s, f_dv, f_v)
    torch.testing.assert_close(crit["margin"], f64(-0.09, 0.55, 0.31, -0.23, -0.2))
    assert crit["string_stable"].tolist() == [False, True, True, False, False]
    assert crit["local_stable"].tolist() == [True, True, False, False, False]
    assert crit["rational"].tolist() == [True, True, False, False, False]
    torch.testing.assert_close(crit["band_upper"], f64(0.3, 0.0, 0.0, math.sqrt(0.23), math.sqrt(0.2)))


@pytest.mark.parametrize(
    "f", [(0.1, -0.5, -0.1), (1 / 1.44, 0.0, -1 / 1.2), (0.05, -0.3, -0.05), (0.1, -0.5, -0.5), (0.2, -0.6, -0.4)]
)
def test_continuous_gain_exceeds_one_exactly_on_the_band(f):
    f_s, f_dv, f_v = (f64(x) for x in f)
    crit = criterion(f_s, f_dv, f_v)
    omega = torch.linspace(1e-3, 3.0, 3000, dtype=torch.float64)
    gain = transfer_continuous(f_s, f_dv, f_v, omega).abs()[0]
    if crit["margin"] < 0:
        band = crit["band_upper"].item()
        away = (omega - band).abs() > 1e-6
        assert torch.equal((gain > 1.0)[away], (omega < band)[away])
        assert (omega < band).any()
    else:
        assert (gain <= 1.0 + 1e-12).all()


def test_discrete_tends_to_continuous_and_equals_windowed_for_a_memoryless_jacobian():
    f = (f64(0.1, 0.05), f64(-0.5, -0.3), f64(-0.1, -0.05))
    continuous = transfer_continuous(*f, OMEGA)
    errors = [(transfer_discrete(*f, OMEGA, dt) - continuous).abs().max().item() for dt in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert errors[-1] < 1e-3 and all(later < earlier / 5 for earlier, later in zip(errors, errors[1:]))
    for window in (1, 4):
        jacobian = torch.zeros(2, window, 3, dtype=torch.float64)
        jacobian[:, -1] = torch.stack(f, dim=-1)  # the latest state only
        torch.testing.assert_close(transfer_windowed(jacobian, OMEGA, 0.1), transfer_discrete(*f, OMEGA, 0.1))


def test_history_jacobian_sums_to_partials():
    torch.manual_seed(0)
    gru = GRU(CONTEXT, hidden_size=8, window=5).double()
    s, v = f64(20.0, 35.0), f64(10.0, 20.0)
    jacobian = history_jacobian(gru, s, v)
    assert jacobian.shape == (2, 5, 3)
    torch.testing.assert_close(jacobian.sum(1), torch.stack(partials(gru, s, v), dim=-1))
