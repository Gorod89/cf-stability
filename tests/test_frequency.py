"""Numerical frequency response (cf_stability/stability/frequency.py) against the exact linear responses."""

import math

import pytest
import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.stability.analytic import criterion, history_jacobian, partials, transfer_discrete, transfer_windowed
from cf_stability.stability.equilibrium import find_equilibria
from cf_stability.stability.frequency import FrequencyConfig, fit_sinusoid, frequency_groups, frequency_response


class LinearLaw(CFModel):
    """``a = c_s (s - s0 - T v) - c_dv dv`` of the state ``lag`` steps ago."""

    def __init__(self, c_s: float, T: float, c_dv: float, s0: float = 2.0, lag: int = 0) -> None:
        super().__init__()
        self.c_s, self.T, self.c_dv, self.s0, self.window = c_s, T, c_dv, s0, lag + 1

    def forward(self, state_history: Tensor) -> Tensor:
        state = state_history[..., -self.window, :]
        return self.c_s * (state[..., 0] - self.s0 - self.T * state[..., 2]) - self.c_dv * state[..., 1]


def wrapped(angle: Tensor) -> Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def assert_matches(response: dict[str, Tensor], exact: Tensor) -> None:
    assert (response["gain"] - exact.abs()).abs().max() < 1e-3
    assert wrapped(response["phase"] - exact.angle()).abs().max() < 1e-2
    assert not any(response[flag].any() for flag in ("clipped", "stopped", "collided"))


@pytest.mark.parametrize("c_s, T, c_dv, unstable", [(0.2, 2.0, 0.6, False), (0.5, 1.2, 0.1, True)])
def test_linear_law_matches_the_discrete_transfer_function(c_s, T, c_dv, unstable):
    law = LinearLaw(c_s, T, c_dv)
    cfg = FrequencyConfig()
    eq = find_equilibria(law, [8.0, 20.0])
    response = frequency_response(law, eq, cfg)
    f = partials(law, eq.s, eq.v)
    assert (criterion(*f)["margin"] < 0).tolist() == [unstable] * 2
    assert response["gain"].shape == (2, cfg.n_omega)
    assert_matches(response, transfer_discrete(*f, response["omega"], law.dt))
    assert response["unstable"].tolist() == [unstable] * 2
    assert torch.equal(response["omega_at_max"], response["omega"][response["gain"].argmax(1)])


def test_linear_law_with_memory_matches_the_windowed_transfer_function():
    law = LinearLaw(0.2, 2.0, 0.6, lag=5)
    eq = find_equilibria(law, [8.0, 20.0])
    response = frequency_response(law, eq, FrequencyConfig())
    windowed = transfer_windowed(history_jacobian(law, eq.s, eq.v), response["omega"], law.dt)
    assert_matches(response, windowed)
    # the delay matters: the memoryless view misses it
    memoryless = transfer_discrete(*partials(law, eq.s, eq.v), response["omega"], law.dt)
    assert (windowed.abs() - memoryless.abs()).abs().max() > 0.05


def test_only_usable_equilibria_are_excited():
    law = LinearLaw(0.2, 2.0, 0.6)
    eq = find_equilibria(law, [8.0, 20.0])
    eq.found[1] = False  # as if there were no equilibrium at 20 m/s
    cfg = FrequencyConfig(n_omega=3, omega_min=0.5, min_discard_s=20.0, min_measure_s=20.0, max_batch=2)
    response = frequency_response(law, eq, cfg)
    assert response["gain"][0].isfinite().all() and response["gain"][1].isnan().all()
    assert response["max_gain"][1].isnan() and response["omega_at_max"][1].isnan()
    assert response["unstable"].tolist() == [False, False]
    eq.found[0] = False  # no equilibrium at all: nothing is rolled out
    response = frequency_response(law, eq, cfg)
    assert response["gain"].isnan().all() and not response["clipped"].any()


def test_frequency_groups_and_schedule():
    cfg = FrequencyConfig()
    omega = cfg.omegas()
    groups = frequency_groups(omega, 0.1, cfg)
    assert len(groups) == 3 and sorted(int(j) for g in groups for j in g) == list(range(25))
    lengths = [max(sum(cfg.schedule(float(omega[j]), 0.1)) for j in g) for g in groups]
    assert lengths == sorted(lengths, reverse=True)
    # 0.02 rad/s: one period discarded, two measured; 2 rad/s: 60 s discarded, 20 periods (62.8 s) measured
    assert cfg.schedule(0.02, 0.1) == (math.ceil(100 * math.pi / 0.1), round(200 * math.pi / 0.1))
    assert cfg.schedule(2.0, 0.1) == (600, round(20 * math.pi / 0.1))


def test_fit_sinusoid_recovers_amplitude_phase_and_offset():
    t = 0.1 * torch.arange(400, dtype=torch.float64)  # 40 s
    omega = torch.tensor([0.5, 0.05, 2.0], dtype=torch.float64)
    amplitude, phase, offset = (torch.tensor(x, dtype=torch.float64) for x in ([0.3, 1.2, 0.05], [0.4, -2.0, 3.0], [20.0, -3.0, 0.0]))
    y = amplitude[:, None] * torch.sin(omega[:, None] * t + phase[:, None]) + offset[:, None]
    mask = torch.stack((t >= 0.0, t < 30.0, (t >= 10.0) & (t < 20.0)))  # the second window: a quarter period
    y_masked = torch.where(mask, y, 1e3)  # samples outside the window are ignored
    a, p, o, r = fit_sinusoid(y_masked, t, omega, mask)
    torch.testing.assert_close(a, amplitude)
    torch.testing.assert_close(wrapped(p - phase), torch.zeros_like(p), atol=1e-8, rtol=0.0)
    torch.testing.assert_close(o, offset)
    assert (r < 1e-8).all()
    # a second harmonic is orthogonal over whole periods and shows up in the residual only
    t_full = 0.1 * torch.arange(1257, dtype=torch.float64)  # 10 periods of 0.5 rad/s
    y = torch.sin(0.5 * t_full) + 0.1 * torch.sin(1.0 * t_full)
    a, _, _, r = fit_sinusoid(y[None], t_full, omega[:1])
    torch.testing.assert_close(a, torch.ones_like(a), atol=1e-3, rtol=0.0)
    torch.testing.assert_close(r, torch.full_like(r, 0.1 / math.sqrt(1.01)), atol=1e-3, rtol=0.0)
