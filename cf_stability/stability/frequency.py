"""Numerical frequency response of the two-vehicle closed loop (docs/m3_contract.md, section 4).

The leader drives at ``v_e + A sin(w t)`` from ``t = 0``, the last sample of a warm-up of
``model.window`` samples at the equilibrium; the follower is rolled out with ``rollout_model``.
Per frequency the first ``max(60 s, 1 period)`` after ``t = 0`` are discarded and the follower
speed is projected on ``sin(w t), cos(w t), 1`` over the smallest whole number of periods
``>= max(60 s, 2 periods)`` that follows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.stability.equilibrium import Equilibria, model_like
from cf_stability.train.closed_loop import A_MAX, A_MIN, rollout_model
from cf_stability.train.evaluate import evaluating

FLAGS = ("clipped", "stopped", "collided")
MEASURES = ("gain", "phase", "residual")


@dataclass
class FrequencyConfig:
    """Excitation, measurement and batching of the frequency response."""

    amplitude: float = 0.2  # leader speed amplitude, m/s
    omega_min: float = 0.02  # rad/s
    omega_max: float = 2.0
    n_omega: int = 25  # log-spaced
    min_discard_s: float = 60.0  # discarded time = max(min_discard_s, discard_periods periods)
    discard_periods: float = 1.0
    min_measure_s: float = 60.0  # measured time: whole periods >= max(min_measure_s, measure_periods periods)
    measure_periods: int = 2
    threshold: float = 1.02  # numerical flag: largest gain > threshold
    n_groups: int = 3  # frequencies are rolled out in groups of similar length
    max_batch: int = 4096  # sequences per rollout

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FrequencyConfig":
        unknown = set(mapping) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown frequency-response keys: {sorted(unknown)}")
        return cls(**dict(mapping))

    def omegas(self) -> np.ndarray:
        return np.geomspace(self.omega_min, self.omega_max, self.n_omega)

    def schedule(self, omega: float, dt: float) -> tuple[int, int]:
        """Numbers of discarded and of measured samples at ``omega``."""
        period = 2.0 * math.pi / omega
        discard = max(self.min_discard_s, self.discard_periods * period)
        # the epsilon keeps an exact whole number of periods (or samples) from rounding up
        periods = math.ceil(max(self.min_measure_s, self.measure_periods * period) / period - 1e-9)
        return math.ceil(discard / dt - 1e-9), round(periods * period / dt)


def fit_sinusoid(
    y: Tensor, t: Tensor, omega: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Least-squares fit ``y ~ amplitude * sin(omega t + phase) + offset`` over the samples in ``mask``.

    ``y [n, T]``, ``t [T]`` or ``[n, T]``, ``omega [n]``, ``mask [n, T]`` (every sample when ``None``);
    each sequence has its own window. Returns ``amplitude``, ``phase`` (rad), ``offset`` and the
    relative residual ``|y - fit| / |y - mean(y)|`` over the window (0 for a constant signal),
    each ``[n]`` in float64.
    """
    y = y.double()
    mask = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
    weight = mask.double()
    mean = (y * weight).sum(-1) / weight.sum(-1)
    centred = (y - mean[:, None]) * weight  # zero outside the window
    phase_arg = omega.double().to(y.device)[:, None] * t.double().to(y.device)
    basis = torch.stack((torch.sin(phase_arg).expand_as(y), torch.cos(phase_arg).expand_as(y), weight), -1)
    basis = basis * weight[..., None]
    coef = torch.linalg.solve(basis.transpose(-1, -2) @ basis, (basis * centred[..., None]).sum(-2))
    residual = torch.linalg.vector_norm(centred - (basis @ coef[..., None]).squeeze(-1), dim=-1)
    scale = torch.linalg.vector_norm(centred, dim=-1).clamp(min=torch.finfo(torch.float64).tiny)
    amplitude = torch.hypot(coef[:, 0], coef[:, 1])
    return amplitude, torch.atan2(coef[:, 1], coef[:, 0]), mean + coef[:, 2], residual / scale


@torch.no_grad()
def excite(model: CFModel, s_e: Tensor, v_e: Tensor, omega: Tensor, cfg: FrequencyConfig) -> dict[str, Tensor]:
    """Two-vehicle rollouts, one sequence per pair ``(s_e[i], v_e[i], omega[i])``, reduced to the fit.

    Returns ``[n]`` tensors: ``gain`` (follower / leader amplitude), ``phase`` (of the follower
    speed relative to the leader speed), ``offset``, ``residual`` of :func:`fit_sinusoid`, and the
    flags ``clipped`` (acceleration at a clip bound), ``stopped`` (speed 0) and ``collided``
    (spacing <= 0) over the rollout up to the end of the measured window. The model is used as it
    is (mode, device, dtype); the rollout runs in the model's dtype.
    """
    device, dtype = model_like(model)
    dt, window = model.dt, model.window
    samples = torch.tensor([cfg.schedule(w, dt) for w in omega.tolist()], device=device)  # [n, 2]
    start = window + samples[:, :1]  # first measured sample
    end = start + samples[:, 1:]  # one past the last measured sample
    k = torch.arange(int(end.max()), device=device)
    t = (k - (window - 1)).double() * dt  # t = 0 at the last warm-up sample
    v_e64, s_e64 = v_e.double().to(device)[:, None], s_e.double().to(device)[:, None]
    v_lead = v_e64 + cfg.amplitude * torch.sin(omega.double().to(device)[:, None] * t.clamp(min=0.0))
    x_lead = s_e64 + dt * (torch.cumsum(v_lead, -1) - v_lead[:, :1])  # x[k+1] = x[k] + dt v[k+1]
    hist = (s_e64.expand(-1, window), v_e64.expand(-1, window))
    res = rollout_model(model, x_lead.to(dtype), v_lead.to(dtype), *(h.to(dtype) for h in hist), dt=dt)
    simulated = (k >= window) & (k < end)
    applied = (k >= window - 1) & (k < end - 1)  # accelerations that drive the simulated samples
    amplitude, phase, offset, residual = fit_sinusoid(res.v, t, omega, (k >= start) & (k < end))
    return {
        "gain": amplitude / cfg.amplitude,
        "phase": phase,
        "offset": offset,
        "residual": residual,
        "clipped": (((res.a <= A_MIN) | (res.a >= A_MAX)) & applied).any(-1),
        "stopped": ((res.v <= 0.0) & simulated).any(-1),
        "collided": ((res.s <= 0.0) & simulated).any(-1),
    }


def frequency_groups(omega: np.ndarray, dt: float, cfg: FrequencyConfig) -> list[np.ndarray]:
    """Indices of the frequencies in ``cfg.n_groups`` groups of similar rollout length (longest first)."""
    length = [sum(cfg.schedule(float(w), dt)) for w in omega]
    order = np.argsort(length, kind="stable")[::-1].copy()
    return [group for group in np.array_split(order, min(cfg.n_groups, len(order))) if len(group)]


def frequency_response(model: CFModel, equilibria: Equilibria, cfg: FrequencyConfig) -> dict[str, Tensor]:
    """Numerical gains at every usable equilibrium (``equilibria.usable``) and frequency.

    Returns ``omega [n_w]``; ``gain``, ``phase``, ``residual`` ``[n_eq, n_w]`` (NaN at the
    equilibria that are not excited); ``max_gain``, ``omega_at_max`` ``[n_eq]``; ``unstable [n_eq]``
    (``max_gain > cfg.threshold``) and the flags ``clipped``, ``stopped``, ``collided``
    ``[n_eq, n_w]`` (False where not excited). Rollouts in the model's dtype on its device, in
    eval mode and without gradients; the mode is restored.
    """
    device, _ = model_like(model)
    omega = torch.as_tensor(cfg.omegas(), dtype=torch.float64, device=device)
    n_eq, n_w = equilibria.v.shape[0], omega.shape[0]
    out = {name: torch.full((n_eq, n_w), math.nan, dtype=torch.float64, device=device) for name in MEASURES}
    out.update({name: torch.zeros((n_eq, n_w), dtype=torch.bool, device=device) for name in FLAGS})
    eq_index = equilibria.usable.to(device).nonzero().squeeze(-1)
    groups = frequency_groups(cfg.omegas(), model.dt, cfg) if len(eq_index) else []
    with evaluating(model, None):
        for group in groups:
            pairs = torch.cartesian_prod(eq_index, torch.as_tensor(group, device=device)).reshape(-1, 2)
            for chunk in pairs.split(cfg.max_batch):
                e, j = chunk.unbind(1)
                res = excite(model, equilibria.s[e], equilibria.v[e], omega[j], cfg)
                for name in (*MEASURES, *FLAGS):
                    out[name][e, j] = res[name].to(out[name].dtype)
    max_gain = out["gain"].amax(1)  # NaN where not excited or where a rollout broke down
    return {
        "omega": omega,
        **out,
        "max_gain": max_gain,
        "omega_at_max": torch.where(max_gain.isnan(), math.nan, omega[out["gain"].argmax(1)]),
        "unstable": max_gain > cfg.threshold,
    }
