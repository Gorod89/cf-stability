"""Closed-loop evaluation on full events (docs/m2_contract.md, section 4)."""

from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel
from cf_stability.train.calibration import resolve_device
from cf_stability.train.closed_loop import closed_loop_metrics, rollout_model
from cf_stability.train.tensors import EventTensors

COLUMNS = ("event_id", "follower_id", "site", "n_scored", "rmse_s", "rmse_v", "collided", "min_s")


def model_device(model: CFModel) -> torch.device:
    """Device of the first parameter or buffer (CPU for a model without any)."""
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        return tensor.device
    return torch.device("cpu")


def model_dtype(model: CFModel) -> torch.dtype:
    """dtype of the first floating-point parameter, else buffer; float64 for a model without any."""
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float64


@contextmanager
def evaluating(model: CFModel, device: str | torch.device | None) -> Iterator[torch.device]:
    """Eval mode on ``device`` (default: where the model is), ``no_grad``; the mode is restored."""
    device = model_device(model) if device is None else resolve_device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if model_device(model) != device:  # only then: .to() re-allocates the flat weights of RNNs
        model.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield device
    finally:
        model.train(was_training)


def _check_warmup(model: CFModel, events: Sequence[Event], warmup: int) -> None:
    if model.window > warmup:
        raise ValueError(f"model window {model.window} exceeds the warm-up of {warmup} samples")
    if min(len(ev) for ev in events) <= warmup:
        raise ValueError(f"every event needs more than {warmup} samples")


def evaluate_closed_loop(
    model: CFModel,
    events: Sequence[Event],
    *,
    warmup: int = 30,
    batch_events: int = 2048,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Per-event errors of full-event closed-loop rollouts, one row per event in input order.

    Samples ``0 .. warmup - 1`` are the observed history; the follower is rolled from sample
    ``warmup - 1`` to the end and scored on the samples ``warmup - 1 .. n - 1`` (``min_s`` and
    the collision flag too). Events are sorted by length and rolled out in batches, in the
    dtype of the model.
    """
    _check_warmup(model, events, warmup)
    order = np.argsort([len(ev) for ev in events], kind="stable")
    frames = []
    with evaluating(model, device) as device:
        dtype = model_dtype(model)
        for start in range(0, len(order), batch_events):
            idx = order[start : start + batch_events]
            batch = EventTensors([events[i] for i in idx], device, dtype)
            res = rollout_model(model, batch.x_lead, batch.v_lead, batch.s[:, :warmup], batch.v[:, :warmup])
            m = closed_loop_metrics(res, batch.s, batch.v, batch.mask, start=warmup - 1)
            scored = batch.mask & (torch.arange(batch.mask.shape[1], device=device) >= warmup - 1)
            columns = {
                "n_scored": m["n_samples"],
                "rmse_s": m["rmse_s"].double(),
                "rmse_v": m["rmse_v"].double(),
                "collided": m["collided"],
                "min_s": torch.where(scored, res.s, torch.inf).amin(-1).double(),
            }
            frames.append(pd.DataFrame({k: v.cpu().numpy() for k, v in columns.items()}, index=idx))
    df = pd.concat(frames).sort_index()
    for name in ("event_id", "follower_id", "site"):
        df[name] = [getattr(ev, name) for ev in events]
    return df.reset_index(drop=True)[list(COLUMNS)]


def summarise(df: pd.DataFrame) -> dict[str, float]:
    """Mean, median and pooled (sample-weighted) RMSE of spacing and speed, collision rate.

    A non-finite per-event error propagates to the summary (no silent skipping).
    """
    out: dict[str, float] = {"n_events": int(len(df)), "collision_rate": float(df["collided"].mean())}
    for name in ("rmse_s", "rmse_v"):
        values = df[name]
        out[f"{name}_mean"] = float(values.mean(skipna=False))
        out[f"{name}_median"] = float(values.median(skipna=False))
        out[f"{name}_pooled"] = float(np.sqrt((values**2 * df["n_scored"]).sum(skipna=False) / df["n_scored"].sum()))
    return out


def open_loop_rmse(
    model: CFModel,
    events: Sequence[Event],
    warmup: int = 30,
    *,
    batch_size: int = 65536,
    device: str | torch.device | None = None,
) -> float:
    """One-step acceleration RMSE with the observed history, pooled over the samples
    ``warmup - 1 .. n - 1`` (those of the closed-loop score). Reported, never used for selection."""
    _check_warmup(model, events, warmup)
    with evaluating(model, device) as device:
        data = EventTensors(events, device, model_dtype(model))
        ends = data.window_ends(warmup)
        sse = torch.zeros((), dtype=torch.float64, device=device)
        for index in ends.split(batch_size):
            err = model(data.state_windows(index, model.window)) - data.targets(index)
            sse += (err.double() ** 2).sum()
    return math.sqrt(sse.item() / len(ends))
