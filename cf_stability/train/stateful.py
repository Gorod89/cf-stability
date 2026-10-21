"""The state of a stateful model along the events of the training and the evaluation (docs/m12_contract.md, D135).

A stateful model (``CFModel.stateful``, e.g. ``residual_idm_tt2``) carries a state along the vehicle, advanced once per
call with the window of that call (``model(history, state) -> (acceleration, state)``). On the recorded events the state
of a vehicle at a sample is that of a teacher-forced pass: from the initial state before the first window of the event
(the window that ends at sample ``window - 1``), advanced once by every observed window up to that sample.

Training (D135 (a), :func:`states_at_starts`): before the closed-loop rollouts of a step, the state at every rollout
start ``(event, j)`` is the state after the observed windows that end at ``window - 1, ..., j - 1`` (the rollout's
first call, with the observed window that ends at ``j``, advances it once more); gradients are kept. The pass runs once
per event of the batch, up to the latest start of its rollouts, and is shared by the rollouts that start inside it: the
host plans it (:func:`pass_plan`: the distinct events, their windows, the slot and the number of windows of every
rollout), the device computes the targets of all windows in one batch (``model.window_drive``) and the state of every
rollout in closed form (``model.carry``). The plan is padded to a fixed capacity (:func:`pass_capacity`), so that the
training step keeps fixed shapes and can be replayed from a CUDA graph; a batch beyond the capacity runs eagerly with a
larger plan. A stateful model without ``window_drive`` / ``carry`` gets the sequential pass
(:func:`sequential_states`, the reference of the tests).

Evaluation (:func:`teacher_forced_path`): the state after every observed window of every event (the one-step
accelerations of the open-loop error, the latent on the test events); the closed-loop rollouts start from the initial
state at the first window of the event (``cf_stability/train/evaluate.py``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.train.tensors import EventTensors

PLAN_KEYS = ("flat_index", "flat_pos", "slot", "count")
CAPACITY_STEP = 4096  # the capacity of the plan is a multiple of this
CAPACITY_BATCHES = 256  # random batches whose plans set the capacity
CAPACITY_FACTOR = 1.25  # headroom over the largest of them


def is_stateful(model: Any) -> bool:
    return bool(getattr(model, "stateful", False))


def has_fast_pass(model: Any) -> bool:
    """The model computes the targets of many windows in one batch and the state after them in closed form."""
    return callable(getattr(model, "window_drive", None)) and callable(getattr(model, "carry", None))


def pass_size(starts: np.ndarray, window: int) -> int:
    """Windows of the pass of the rollout starts ``starts [S, 2]`` (event, j): the sum over the distinct events of
    ``max j - window + 1``."""
    events, inverse = np.unique(starts[:, 0], return_inverse=True)
    latest = np.full(len(events), -1, dtype=np.int64)
    np.maximum.at(latest, inverse.reshape(-1), starts[:, 1].astype(np.int64))
    return int(np.maximum(latest - window + 1, 0).sum())


def pass_plan(starts: np.ndarray, window: int, n_max: int, capacity: int | None = None) -> dict[str, np.ndarray]:
    """Host plan of the teacher-forced pass for the rollout starts ``starts [S, 2]`` (event, j; numpy).

    The distinct events get the slots ``0 .. E - 1`` (in increasing event order); event ``e`` needs the windows that
    end at ``window - 1 .. J_e - 1`` (``J_e``: the latest start of its rollouts), at the positions ``0 .. J_e - window``
    of its row of length ``n_max``. Returns ``flat_index [N, 2]`` (event, sample of the window end), ``flat_pos [N]``
    (``slot * n_max + position``), ``slot [S]`` and ``count [S]`` (windows before the start, ``j - window + 1``).
    With ``capacity`` (>= the windows needed) the window list is padded to it: the padding repeats the first window of
    the first event and writes behind the rows (``S * n_max + k``, distinct), where nothing is read.
    """
    starts = np.asarray(starts, dtype=np.int64).reshape(-1, 2)
    n_rollouts = len(starts)
    events, inverse = np.unique(starts[:, 0], return_inverse=True)
    inverse = inverse.reshape(-1)
    latest = np.full(len(events), -1, dtype=np.int64)
    np.maximum.at(latest, inverse, starts[:, 1])
    counts = np.maximum(latest - window + 1, 0)  # windows of the pass of every event
    if (counts > n_max).any():
        raise ValueError(f"a pass of {int(counts.max())} windows does not fit a row of {n_max}")
    needed = int(counts.sum())
    size = needed if capacity is None else int(capacity)
    if size < needed:
        raise ValueError(f"the pass needs {needed} windows, the capacity is {size}")
    slot_of_window = np.repeat(np.arange(len(events)), counts)
    position = np.arange(needed) - np.repeat(np.cumsum(counts) - counts, counts)
    flat_index = np.empty((size, 2), dtype=np.int64)
    flat_pos = np.empty(size, dtype=np.int64)
    flat_index[:needed, 0] = events[slot_of_window]
    flat_index[:needed, 1] = window - 1 + position
    flat_pos[:needed] = slot_of_window * n_max + position
    pad = size - needed
    if pad:
        flat_index[needed:, 0] = starts[0, 0]
        flat_index[needed:, 1] = window - 1
        flat_pos[needed:] = n_rollouts * n_max + np.arange(pad)
    count = np.maximum(starts[:, 1] - window + 1, 0)
    return {"flat_index": flat_index, "flat_pos": flat_pos, "slot": inverse.astype(np.int64), "count": count,
            "needed": np.array(needed)}  # fmt: skip


def round_capacity(n: int) -> int:
    return max(CAPACITY_STEP, int(math.ceil(n / CAPACITY_STEP)) * CAPACITY_STEP)


def pass_capacity(ends: np.ndarray, window: int, segments: int, n_max: int, seed: int) -> int:
    """Capacity of the plan: ``CAPACITY_FACTOR`` x the largest pass of ``CAPACITY_BATCHES`` random batches of
    ``segments`` starts drawn from the window ends ``ends [M, 2]`` (a generator of its own, derived from ``seed``: the
    batches of the training are not touched), rounded up to ``CAPACITY_STEP``; at most the size of a pass in which
    every rollout has its own event and starts at its end (``segments * n_max``)."""
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(2,)))
    largest = max(pass_size(ends[rng.integers(len(ends), size=segments)], window) for _ in range(CAPACITY_BATCHES))
    return int(min(segments * n_max, round_capacity(CAPACITY_FACTOR * largest)))


class PassBuffers:
    """The device tensors of a plan of fixed capacity, refilled in place before every step (read by a CUDA graph)."""

    def __init__(self, segments: int, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.tensors = {
            "flat_index": torch.zeros((self.capacity, 2), dtype=torch.long, device=device),
            "flat_pos": torch.zeros(self.capacity, dtype=torch.long, device=device),
            "slot": torch.zeros(segments, dtype=torch.long, device=device),
            "count": torch.zeros(segments, dtype=torch.long, device=device),
        }

    def fill(self, plan: dict[str, np.ndarray]) -> None:
        for key in PLAN_KEYS:
            self.tensors[key].copy_(torch.from_numpy(plan[key]))


def plan_tensors(plan: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {key: torch.from_numpy(plan[key]).to(device) for key in PLAN_KEYS}


def states_at_starts(model: CFModel, data: EventTensors, plan: dict[str, Tensor], n_max: int) -> Tensor:
    """The state ``[S]`` of every rollout at its start (the plan of :func:`pass_plan` on the device): the targets of all
    windows of the pass in one batch, scattered into one row per event, and per rollout the closed form of its first
    ``count`` targets from the initial state. Differentiable; fixed shapes."""
    w = model.window
    slot, count = plan["slot"], plan["count"]
    n_rollouts = slot.shape[0]
    drives = model.window_drive(data.state_windows(plan["flat_index"], w))  # [capacity]
    rows = drives.new_zeros(n_rollouts * n_max + drives.shape[0]).index_put((plan["flat_pos"],), drives)
    rows = rows[: n_rollouts * n_max].view(n_rollouts, n_max)[slot]  # [S, n_max]: the row of every rollout's event
    return model.carry(rows, count, model.initial_state(n_rollouts, drives.device))


def sequential_states(model: CFModel, data: EventTensors, starts: Tensor) -> list[Any]:
    """Reference pass: per rollout start ``(event, j)``, the initial state advanced by ``model(window, state)`` with the
    observed windows that end at ``window - 1 .. j - 1``, one call at a time (slow; tests and models without the batched
    pass)."""
    w, out = model.window, []
    for e, j in starts.tolist():
        state = model.initial_state(1, data.state.device)
        for t in range(w - 1, j):
            index = torch.tensor([[e, t]], device=data.state.device)
            _, state = model(data.state_windows(index, w), state)
        out.append(state)
    return out


def rollout_states(model: CFModel, data: EventTensors, starts: Tensor, plan: dict[str, Tensor] | None,
                   n_max: int) -> Any:  # fmt: skip
    """The states at the rollout starts: the batched pass with ``plan`` (models with ``window_drive`` and ``carry``),
    else the sequential one (concatenated along the batch)."""
    if plan is not None and has_fast_pass(model):
        return states_at_starts(model, data, plan, n_max)
    return torch.cat([s.reshape(1) for s in sequential_states(model, data, starts)])


@torch.no_grad()
def teacher_forced_path(model: CFModel, data: EventTensors, max_windows: int = 65536) -> tuple[Tensor, Tensor, Tensor]:
    """The state before and after every observed window of every event of ``data``: ``ends [M, 2]`` (event, sample of
    the window end, the order of ``data.window_ends(window)``) and ``before, after [M]`` (float64); the first window of
    an event starts from the initial state. Batched (``window_drive``, ``carry_path``) in groups of events of similar
    length with at most about ``max_windows`` windows (the sequential calls for a model without them). No gradients."""
    w = model.window
    ends = data.window_ends(w)
    device = ends.device
    before = torch.empty(len(ends), dtype=torch.float64, device=device)
    after = torch.empty_like(before)
    lengths = [int(x) for x in data.lengths.tolist()]
    counts = [max(n - w + 1, 0) for n in lengths]
    offsets = np.concatenate(([0], np.cumsum(counts)))  # window_ends: the events in order, the samples ascending
    groups, current = [], []
    for i in np.argsort(lengths, kind="stable").tolist():  # ascending: the last event of a group is its longest
        if current and (len(current) + 1) * max(counts[i], 1) > max_windows:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    fast = has_fast_pass(model) and callable(getattr(model, "carry_path", None))
    for group in groups:
        n = max(max(counts[i] for i in group), 1)
        rows = torch.tensor(group, device=device)
        t = torch.arange(w - 1, w - 1 + n, device=device)
        index = torch.stack(torch.broadcast_tensors(rows[:, None], t[None, :]), dim=-1).reshape(-1, 2)
        index[:, 1] = torch.minimum(index[:, 1], data.lengths[index[:, 0]] - 1)  # past the end: a valid window
        windows = data.state_windows(index, w)
        initial = model.initial_state(len(group), device)
        if fast:
            path = model.carry_path(model.window_drive(windows).reshape(len(group), n), initial)  # [G, n]
        else:
            state, steps = initial, []
            for m in range(n):
                _, state = model(windows.reshape(len(group), n, w, 3)[:, m], state)
                steps.append(state.reshape(len(group)))
            path = torch.stack(steps, dim=1)
        start = torch.cat((initial.reshape(-1, 1).to(path.dtype), path[:, :-1]), dim=1)
        for g, e in enumerate(group):
            if counts[e]:
                lo, hi = int(offsets[e]), int(offsets[e + 1])
                before[lo:hi] = start[g, : counts[e]].double()
                after[lo:hi] = path[g, : counts[e]].double()
    return ends, before, after
