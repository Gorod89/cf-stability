"""Newell's car-following model in the adapted form of PERL (Long et al. 2023, arXiv 2309.15284, eq. 6).

The follower repeats the acceleration of its leader after the time a wave needs to travel the
distance between them, ``a_f(t) = a_lead(t - D(t) / w)`` with the wave speed ``w``. ``D`` is the
net gap of the last state; the leader acceleration comes from ``v_lead = v - dv`` of the history.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.data.schema import DT, Event
from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import params_tensor

if TYPE_CHECKING:
    from cf_stability.train.calibration import CalibrationConfig

NEWELL_PARAM_NAMES = ("w",)
NEWELL_BOUNDS: dict[str, tuple[float, float]] = {"w": (1.0, 60.0)}
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
BRACKET_TOL = 0.01  # the golden-section search stops when the bracket is below this share of w


def leader_acceleration(state_history: Tensor, dt: float = DT) -> Tensor:
    """Leader acceleration ``[..., W]`` from ``v_lead = v - dv`` of a history ``[..., W, 3]``:
    central differences inside the window, one-sided at both ends, zeros for a single step."""
    v_lead = state_history[..., 2] - state_history[..., 1]
    if v_lead.shape[-1] < 2:
        return torch.zeros_like(v_lead)
    ends = (v_lead[..., 1:] - v_lead[..., :-1]) / dt
    inner = (v_lead[..., 2:] - v_lead[..., :-2]) / (2.0 * dt)
    return torch.cat((ends[..., :1], inner, ends[..., -1:]), dim=-1)


def newell_acc(state_history: Tensor, w: Tensor | float, dt: float = DT) -> Tensor:
    """Leader acceleration ``D / w`` seconds ago for a history ``[..., W, 3]`` -> ``[...]``.

    ``D`` is the net gap of the last state (clamped at 0); the delay is clamped to the span of the
    history, ``(W - 1) dt``, and the leader acceleration is interpolated linearly between samples.
    ``w`` broadcasts (scalar or ``[B]``); a history of one step gives zeros.
    """
    n = state_history.shape[-2]
    if n < 2:
        return torch.zeros_like(state_history[..., -1, 0])
    s_last = torch.clamp(state_history[..., -1, 0], min=0.0)
    p = (n - 1) - torch.clamp(s_last / (w * dt), max=n - 1)  # fractional index of t - delay
    k = torch.clamp(torch.floor(p), max=n - 2)
    # a gap that is not finite (a diverged rollout) has no index: read sample 0 there, the NaN of
    # `p` still reaches the result; on the GPU an undefined index is an illegal memory access
    index = torch.nan_to_num(k, nan=0.0).clamp(min=0).long().unsqueeze(-1)
    acc = leader_acceleration(state_history, dt)
    below = torch.gather(acc, -1, index).squeeze(-1)
    above = torch.gather(acc, -1, index + 1).squeeze(-1)
    return below + (p - k) * (above - below)


class _Law:
    """Newell's law with one wave speed per sequence, in the form ``rollout_model`` calls."""

    def __init__(self, w: Tensor, window: int) -> None:
        self.w, self.window = w, window

    def __call__(self, state_history: Tensor) -> Tensor:
        return newell_acc(state_history, self.w)


def calibrate_newell(
    events: Sequence[Event], cfg: CalibrationConfig | None = None, *, window: int = 30, n_grid: int = 40
) -> dict[str, Any]:
    """Global wave speed ``w``: minimum of the closed-loop objective of the IDM calibration.

    ``J(w) = NRMSE(s) + NRMSE(v)`` pooled over the events plus ``cfg.collision_penalty`` x share
    of events with a collision. As in ``evaluate_closed_loop`` the first ``window`` samples are
    observed and the samples ``window - 1 .. n - 1`` are scored. ``J`` is evaluated on a log-spaced
    grid of ``n_grid`` values over ``NEWELL_BOUNDS``, then a golden-section search between the
    neighbours of the best grid value runs until the bracket is below 1 % of ``w``. Events are
    sorted by length and rolled out in batches of at most ``cfg.batch_events`` sequences (several
    wave speeds per rollout when a batch has fewer events) on ``cfg.device``.
    """
    from cf_stability.train.calibration import CalibrationConfig, resolve_device  # they import this package
    from cf_stability.train.closed_loop import rollout_model
    from cf_stability.train.tensors import EventTensors

    cfg = cfg or CalibrationConfig()
    if min(len(ev) for ev in events) <= window:
        raise ValueError(f"every event needs more than window = {window} samples")
    device = resolve_device(cfg.device)
    order = np.argsort([len(ev) for ev in events], kind="stable")
    batches = [
        EventTensors([events[j] for j in order[k : k + cfg.batch_events]], device, torch.float64)
        for k in range(0, len(order), cfg.batch_events)
    ]
    results: dict[float, dict[str, float]] = {}

    def evaluate(ws: list[float]) -> None:
        totals = torch.zeros(len(ws), 6, dtype=torch.float64, device=device)
        for batch in batches:
            per_rollout = max(1, cfg.batch_events // len(batch))
            for start in range(0, len(ws), per_rollout):
                chunk = ws[start : start + per_rollout]
                w = torch.tensor(chunk, dtype=torch.float64, device=device).repeat_interleave(len(batch))
                x_lead, v_lead, s_obs, v_obs, mask = (
                    t.repeat(len(chunk), 1) for t in (batch.x_lead, batch.v_lead, batch.s, batch.v, batch.mask)
                )
                res = rollout_model(_Law(w, window), x_lead, v_lead, s_obs[:, :window], v_obs[:, :window])
                valid = mask & (torch.arange(mask.shape[1], device=device) >= window - 1)
                sums = torch.stack(
                    (
                        torch.where(valid, (res.s - s_obs) ** 2, 0.0).sum(-1),
                        torch.where(valid, (res.v - v_obs) ** 2, 0.0).sum(-1),
                        torch.where(valid, s_obs**2, 0.0).sum(-1),
                        torch.where(valid, v_obs**2, 0.0).sum(-1),
                        valid.sum(-1).double(),
                        (valid & (res.s <= 0.0)).any(-1).double(),
                    ),
                    dim=-1,
                )
                totals[start : start + len(chunk)] += sums.reshape(len(chunk), len(batch), 6).sum(1)
        for w, (sse_s, sse_v, ss_s, ss_v, n, collided) in zip(ws, totals.tolist()):
            nrmse_s, nrmse_v = math.sqrt(sse_s / ss_s), math.sqrt(sse_v / ss_v)
            results[w] = {
                "objective": nrmse_s + nrmse_v + cfg.collision_penalty * collided / len(events),
                "rmse_s": math.sqrt(sse_s / n),
                "rmse_v": math.sqrt(sse_v / n),
                "nrmse_s": nrmse_s,
                "nrmse_v": nrmse_v,
                "collision_rate": collided / len(events),
            }

    def objective(w: float) -> float:
        if w not in results:
            evaluate([w])
        return results[w]["objective"]

    grid = np.geomspace(*NEWELL_BOUNDS["w"], n_grid).tolist()
    evaluate(grid)
    best = int(np.argmin([results[w]["objective"] for w in grid]))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, n_grid - 1)]
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    while b - a > BRACKET_TOL * 0.5 * (a + b):
        if objective(c) <= objective(d):
            b, d = d, c
            c = b - GOLDEN * (b - a)
        else:
            a, c = c, d
            d = a + GOLDEN * (b - a)
    w_best = min(results, key=lambda w: results[w]["objective"])
    return {
        "params": {"w": w_best},
        **results[w_best],
        "n_events": len(events),
        "grid": {"w": grid, "objective": [results[w]["objective"] for w in grid]},
    }


def newell_fit(events: Sequence[Event], context: ModelContext, window: int) -> dict[str, Any]:
    """Fit summary of the wave speed: ``context.newell_params`` when given, else :func:`calibrate_newell`."""
    if context.newell_params:
        return {"params": {n: float(context.newell_params[n]) for n in NEWELL_PARAM_NAMES}, "source": "context"}
    return calibrate_newell(events, window=window)


class Newell(CFModel):
    """Newell's model with the wave speed ``theta = (w,)`` as a buffer.

    The value comes from ``context.newell_params``; when that is empty it is NaN until
    :meth:`fit` or a checkpoint sets it.
    """

    name = "newell"
    trainable = False

    def __init__(self, context: ModelContext | None = None, *, window: int = 30) -> None:
        super().__init__()
        self.window = int(window)
        params = context.newell_params if context is not None else {}
        self.register_buffer("theta", params_tensor(params, NEWELL_PARAM_NAMES, torch.float64))

    def forward(self, state_history: Tensor) -> Tensor:
        return newell_acc(state_history[..., -self.window :, :], self.theta[0], self.dt)

    def params_dict(self) -> dict[str, float]:
        return dict(zip(NEWELL_PARAM_NAMES, self.theta.cpu().tolist()))

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Wave speed from ``context.newell_params`` when given, else the calibration on ``events``."""
        summary = newell_fit(events, context, self.window)
        self.theta.copy_(params_tensor(summary["params"], NEWELL_PARAM_NAMES, self.theta.dtype))
        return summary

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "window": self.window}
