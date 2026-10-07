"""Closed-loop rollouts of car-following models and their error metrics.

Conventions (docs/data_contract.md, section 1): semi-implicit Euler
``v[k+1] = max(v[k] + a[k] dt, 0)``, ``x[k+1] = x[k] + v[k+1] dt``,
``s[k+1] = x_lead[k+1] - x[k+1]``, accelerations clipped to ``[A_MIN, A_MAX]``,
collision = ``s <= 0``. ``a[..., k]`` is the (clipped) acceleration applied between
``k`` and ``k + 1``; its last entry repeats the previous one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.data.schema import DT, Event
from cf_stability.models.base import CFModel

A_MIN = -8.0
A_MAX = 4.0
PAD_FIELDS = ("x_lead", "v_lead", "s", "v", "a")

AccFn = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass
class RolloutResult:
    """Simulated follower series of shape ``[..., T]``."""

    s: Tensor
    v: Tensor
    x: Tensor
    a: Tensor


def _euler_step(x: Tensor, v: Tensor, a: Tensor, x_lead_next: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor]:
    v = torch.clamp(v + a * dt, min=0.0)
    x = x + v * dt
    return x_lead_next - x, v, x


def iterate_memoryless(
    acc_fn: AccFn,
    x_lead: Tensor,
    v_lead: Tensor,
    s0: Tensor,
    v0: Tensor,
    *,
    dt: float = DT,
    a_min: float = A_MIN,
    a_max: float = A_MAX,
) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Yield ``(s_k, v_k, x_k, a_k)`` for ``k = 0..T-1`` of a memoryless rollout.

    Shared by :func:`rollout_memoryless` and the error-accumulating calibration loop,
    so that both follow exactly the same arithmetic.
    """
    n_steps = x_lead.shape[-1]
    if n_steps < 2:
        raise ValueError("a rollout needs at least 2 samples")
    s, v, x = s0, v0, x_lead[..., 0] - s0
    for k in range(n_steps - 1):
        a = torch.clamp(acc_fn(s, v - v_lead[..., k], v), a_min, a_max)
        if k == 0:  # the batch shape may come from the model parameters
            s, v, x, a = torch.broadcast_tensors(s, v, x, a)
        yield s, v, x, a
        s, v, x = _euler_step(x, v, a, x_lead[..., k + 1], dt)
    yield s, v, x, a


def rollout_memoryless(
    acc_fn: AccFn,
    x_lead: Tensor,
    v_lead: Tensor,
    s0: Tensor,
    v0: Tensor,
    *,
    dt: float = DT,
    a_min: float = A_MIN,
    a_max: float = A_MAX,
) -> RolloutResult:
    """Closed-loop rollout of ``acc_fn(s, dv, v)`` behind the leader ``x_lead, v_lead`` ``[..., T]``.

    The follower starts at ``x[0] = x_lead[..., 0] - s0`` with speed ``v0``; the
    rollout continues after a collision.
    """
    s, v, x, a = zip(*iterate_memoryless(acc_fn, x_lead, v_lead, s0, v0, dt=dt, a_min=a_min, a_max=a_max))
    return RolloutResult(*(torch.stack(series, dim=-1) for series in (s, v, x, a)))


def rollout_model(
    model: CFModel,
    x_lead: Tensor,
    v_lead: Tensor,
    s_hist: Tensor,
    v_hist: Tensor,
    *,
    dt: float = DT,
    a_min: float = A_MIN,
    a_max: float = A_MAX,
) -> RolloutResult:
    """Closed-loop rollout of a model with a state window.

    ``s_hist, v_hist`` ``[B, W]`` are the observed samples ``0..W-1`` of the event; the
    simulation starts from index ``W - 1``. The returned series (length ``T``) hold the
    observed values in the warm-up part, where ``a`` is the finite difference of the
    observed speeds. The model sees the last ``min(k + 1, window)`` states, kept in a rolling
    ``[B, steps, 3]`` tensor; the rollout is differentiable.
    """
    n_steps, n_hist = x_lead.shape[-1], s_hist.shape[-1]
    if n_hist < 1 or n_steps < max(n_hist, 2):
        raise ValueError(f"need 1 <= history ({n_hist}) <= T ({n_steps}) and T >= 2")
    keep = model.window - 1  # states carried over to the next step besides the new one
    history = torch.stack((s_hist, v_hist - v_lead[..., :n_hist], v_hist), dim=-1)[..., -model.window :, :].contiguous()
    s, v = s_hist[..., -1], v_hist[..., -1]
    x = x_lead[..., n_hist - 1] - s
    sim: tuple[list[Tensor], ...] = ([], [], [], [])
    for k in range(n_hist - 1, n_steps - 1):
        a = torch.clamp(model(history), a_min, a_max)
        s, v, x = _euler_step(x, v, a, x_lead[..., k + 1], dt)
        state = torch.stack((s, v - v_lead[..., k + 1], v), dim=-1)[..., None, :]
        history = torch.cat((history[..., max(0, history.shape[-2] - keep) :, :], state), dim=-2) if keep else state
        for series, value in zip(sim, (s, v, x, a)):
            series.append(value)
    observed = (s_hist, v_hist, x_lead[..., :n_hist] - s_hist, (v_hist[..., 1:] - v_hist[..., :-1]) / dt)
    s_all, v_all, x_all, a_all = (
        torch.cat((obs, torch.stack(series, dim=-1)), dim=-1) if series else obs for obs, series in zip(observed, sim)
    )
    return RolloutResult(s_all, v_all, x_all, torch.cat((a_all, a_all[..., -1:]), dim=-1))


def pad_events(
    events: Sequence[Event], device: torch.device | str | None = None, dtype: torch.dtype = torch.float64
) -> dict[str, Tensor]:
    """Stack events into ``[N, T_max]`` tensors; padding repeats the last valid sample."""
    lengths = np.array([len(ev) for ev in events], dtype=np.int64)
    t_max = int(lengths.max())
    out: dict[str, Tensor] = {}
    for name in PAD_FIELDS:
        arr = np.empty((len(events), t_max), dtype=np.float64)
        for i, ev in enumerate(events):
            values = getattr(ev, name)
            arr[i, : len(values)] = values
            arr[i, len(values) :] = values[-1]
        out[name] = torch.as_tensor(arr, dtype=dtype, device=device)
    out["lengths"] = torch.as_tensor(lengths, device=device)
    out["mask"] = torch.arange(t_max, device=device) < out["lengths"][:, None]
    return out


def closed_loop_metrics(
    result: RolloutResult, s_obs: Tensor, v_obs: Tensor, mask: Tensor, start: int = 0
) -> dict[str, Tensor]:
    """Per-sequence errors of a rollout over the valid samples with index ``>= start``.

    ``NRMSE(y) = sqrt(mean((y_sim - y_obs)^2)) / sqrt(mean(y_obs^2))`` (contract, section 6).
    """
    valid = mask & (torch.arange(result.s.shape[-1], device=mask.device) >= start)
    n = valid.sum(-1)
    out: dict[str, Tensor] = {}
    for name, sim, obs in (("s", result.s, s_obs), ("v", result.v, v_obs)):
        sse = torch.where(valid, (sim - obs) ** 2, 0.0).sum(-1)
        ss_obs = torch.where(valid, obs**2, 0.0).sum(-1)
        out[f"rmse_{name}"] = torch.sqrt(sse / n)
        out[f"nrmse_{name}"] = torch.sqrt(sse / ss_obs)
    hit = valid & (result.s <= 0.0)
    out["collided"] = hit.any(-1)
    out["collision_fraction"] = hit.sum(-1).to(result.s.dtype) / n
    out["n_samples"] = n
    return out
