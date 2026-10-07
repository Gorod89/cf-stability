"""Gradient training of car-following models (docs/m2_contract.md, section 3)."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.stability.penalties import (
    CAPTURABLE,
    INFO_KEYS,
    PenaltyConfig,
    combined_parts,
    sample_speeds,
    stability_penalty,
)
from cf_stability.train.calibration import resolve_device
from cf_stability.train.closed_loop import rollout_model
from cf_stability.train.evaluate import evaluate_closed_loop, evaluating, summarise
from cf_stability.train.tensors import BandConfig, EventTensors, training_context

DTYPES = {"float32": torch.float32, "float64": torch.float64}
TERMS = ("loss_acc", "loss_rollout", "loss_extra", "loss_penalty")
GRAPH_WARMUP = 3  # eager steps before the training step is captured as a CUDA graph

StepFn = Callable[[Tensor, Tensor, bool], tuple[Tensor, ...]]


@dataclass
class RolloutConfig:
    weight: float = 1.0  # 0 disables the rollout loss
    horizon: int = 50  # teacher-free steps per segment (5 s)
    segments: int = 256  # segments per training step


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 100
    steps_per_epoch: int = 300
    patience: int = 10
    acc_weight: float = 1.0  # weight of the one-step acceleration MSE
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dtype: str = "float32"
    device: str = "auto"
    warmup: int = 30  # observed samples of the closed-loop evaluation
    grad_clip: float = 10.0
    cuda_graph: bool = True  # replay full-batch steps from a CUDA graph (launch-bound rollouts); not with "gain"
    band: BandConfig = field(default_factory=BandConfig)  # spacing band of the data (docs/m4_contract.md, 1.1)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)  # stability penalty (docs/m3_contract.md, 7)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainConfig":
        values = dict(mapping)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown training keys: {sorted(unknown)}")
        if "rollout" in values:
            values["rollout"] = RolloutConfig(**values["rollout"])
        if "penalty" in values:
            values["penalty"] = PenaltyConfig.from_mapping(values["penalty"])
        if "band" in values:
            values["band"] = BandConfig.from_mapping(values["band"])
        if values.get("dtype", cls.dtype) not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        return cls(**values)


def seed_everything(seed: int) -> None:
    """Seed torch on every device and make cuDNN deterministic."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rollout_loss(
    model: CFModel, data: EventTensors, starts: Tensor, *, horizon: int, var_s: float, var_v: float
) -> Tensor:
    """``MSE(s) / var_s + MSE(v) / var_v`` of teacher-free rollouts behind the recorded leader.

    Each segment starts at a window end ``(event, j)`` of ``starts [S, 2]`` after the observed
    warm-up ``j - window + 1 .. j`` and runs ``horizon`` steps with the clipping of the contract;
    samples past the end of an event are masked. Gradients flow through the whole rollout.
    """
    w = model.window
    seg = data.segments(starts, w, horizon)
    res = rollout_model(model, seg["x_lead"], seg["v_lead"], seg["s"][:, :w], seg["v"][:, :w])
    mask = seg["mask"][:, w:]
    sse_s = torch.where(mask, (res.s[:, w:] - seg["s"][:, w:]) ** 2, 0.0).sum()
    sse_v = torch.where(mask, (res.v[:, w:] - seg["v"][:, w:]) ** 2, 0.0).sum()
    return (sse_s / var_s + sse_v / var_v) / mask.sum().clamp(min=1)


@dataclass
class EpochChoice:
    """Best epoch and early stopping, one epoch at a time (docs/m4_contract.md, 1.5, D89).

    Without penalty (``tolerance`` None) the rule of M2: an epoch is the new best when its
    validation RMSE is below the best one, the patience counts the epochs since. With a penalty an
    epoch is feasible when its validation penalty is at most ``tolerance``; a feasible epoch beats
    an infeasible one, then the smaller RMSE wins; an epoch with a non-finite RMSE is never best.
    The patience also restarts, while no epoch has been feasible, when the penalty falls below
    ``(1 - progress)`` times its smallest earlier value: a run that is still reaching feasibility
    must not stop on the RMSE alone.
    """

    patience: int
    tolerance: float | None = None  # None: no penalty
    progress: float = 0.0
    best_epoch: int = 0  # 0: the initial weights
    best_rmse: float = math.inf  # RMSE of the best epoch; the initial one when it is eligible
    best_feasible: bool = False
    n_feasible: int = 0
    min_penalty: float = math.inf  # smallest validation penalty so far
    stale: int = 0  # epochs since the patience restarted

    def update(self, epoch: int, rmse: float, penalty: float | None = None) -> tuple[bool, bool]:
        """Record ``epoch`` with its validation RMSE and penalty; returns (new best epoch, stop after it)."""
        feasible = progressed = False
        if self.tolerance is None:
            better = rmse < self.best_rmse
        else:
            feasible = penalty <= self.tolerance  # NaN: infeasible
            self.n_feasible += feasible
            if feasible != self.best_feasible:
                better = feasible and math.isfinite(rmse)  # feasible beats infeasible
            else:
                better = rmse < self.best_rmse
            # progress of the penalty restarts the patience only while no epoch has been feasible
            progressed = self.n_feasible == 0 and penalty < (1.0 - self.progress) * self.min_penalty
            if not math.isnan(penalty):
                self.min_penalty = min(self.min_penalty, penalty)
        if better:
            self.best_epoch, self.best_rmse, self.best_feasible, self.stale = epoch, rmse, feasible, 0
            return True, False
        self.stale = 0 if progressed else self.stale + 1
        return False, self.stale >= self.patience


def validation_penalty(model: CFModel, speeds: Tensor, penalty: PenaltyConfig) -> dict[str, float]:
    """``val_penalty`` and its parts ``val_penalty_<info key>``: the penalty of ``model`` at ``speeds``
    with the settings (and band) of the training, in eval mode as the validation rollouts (D89);
    ``combined``: the sum of its rollout part and its Jacobian part, both unweighted (D110).

    Runs outside any CUDA graph and leaves the training untouched: the derivatives inside the
    penalty are taken with respect to the states only (no gradient reaches the parameters), eval
    mode keeps buffers such as the power iteration of spectral normalisation unchanged, and the
    static buffers of a captured step are not read or written.
    """
    with evaluating(model, None):
        value, info = stability_penalty(model, speeds, penalty)
    return {"val_penalty": float(value), **{f"val_penalty_{key}": float(info[key]) for key in INFO_KEYS[penalty.kind]}}


def _clone_state(model: CFModel) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _storage(model: CFModel) -> list[int]:
    return [tensor.data_ptr() for tensor in itertools.chain(model.parameters(), model.buffers())]


def _cuda_generators(model: CFModel) -> list[torch.Generator]:
    """CUDA generators held by the modules of a model (e.g. the collocation sampler of PIDL)."""
    return [
        value
        for module in model.modules()
        for value in vars(module).values()
        if isinstance(value, torch.Generator) and value.device.type == "cuda"
    ]


class _GraphedStep:
    """Training step replayed from a CUDA graph.

    The step-by-step rollouts are bound by kernel launches; replaying the whole step (losses,
    backward, clipping, Adam) is several times faster. The first ``GRAPH_WARMUP`` calls run
    eagerly on a side stream (lazy initialisation, optimizer state), the next one is captured;
    later calls copy their inputs into the static buffers of the capture and replay it. CUDA
    generators of the model are registered, so that every replay draws new numbers. The graph
    is captured again if the model's tensors have moved (``.to()`` re-allocates RNN weights).
    A capture that fails (e.g. out of memory for the long rollouts of the gain penalty) leaves
    the step eager for the rest of the training, with a warning.
    """

    def __init__(self, step: StepFn, model: CFModel) -> None:
        self.step = step
        self.model = model
        self.eager_left = GRAPH_WARMUP
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: tuple[Tensor, ...] = ()  # static inputs and outputs of the capture
        self.out: tuple[Tensor, ...] = ()
        self.storage: list[int] = []
        self.captured = False  # a capture has succeeded
        self.failed = False  # a capture has failed: eager from then on

    def __call__(self, index: Tensor, starts: Tensor) -> tuple[Tensor, ...]:
        if self.failed:
            return self.step(index, starts, True)
        if self.graph is not None and _storage(self.model) != self.storage:
            self.graph = None
        if self.graph is None and self.eager_left > 0:
            self.eager_left -= 1
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out = self.step(index, starts, True)
            torch.cuda.current_stream().wait_stream(stream)
            return out
        if self.graph is None:
            self.inputs = (index.clone(), starts.clone())
            graph = torch.cuda.CUDAGraph()
            for generator in _cuda_generators(self.model):
                graph.register_generator_state(generator)
            try:
                with torch.cuda.graph(graph):
                    self.out = self.step(*self.inputs, False)
            except RuntimeError as exc:  # nothing of the step has run: the capture records, it does not execute
                self.failed, self.inputs, self.out = True, (), ()
                del graph
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                warnings.warn(f"CUDA graph capture of the training step failed, it runs eagerly: {exc}", stacklevel=2)
                return self.step(index, starts, True)
            self.graph, self.captured = graph, True
            self.storage = _storage(self.model)
        else:
            self.inputs[0].copy_(index)
            self.inputs[1].copy_(starts)
        self.graph.replay()
        return self.out


def train_model(
    model: CFModel, train_events: Sequence[Event], val_events: Sequence[Event], cfg: TrainConfig, seed: int
) -> dict[str, Any]:
    """Adam on acceleration MSE + rollout loss + ``model.extra_loss`` + ``lambda`` x stability
    penalty, early stopping on the validation closed-loop spacing RMSE (mean over events); the
    best weights are restored.

    The model is moved to ``cfg.device`` and cast to ``cfg.dtype``. A non-finite loss stops the
    run (``stop_reason = "non_finite_loss"``). Batches and rollout segments come from a
    generator seeded by ``seed``, so a run is reproducible on a given device. The speeds of the
    penalty are redrawn every epoch from a second generator derived from ``seed``, so that the
    batches are the same with and without penalty. With ``existence: band`` the penalty gets the
    spacing band of the training data (``training_context(..., cfg.band)["band"]``) unless its
    config holds one; ``existence: fixed`` reproduces M3 in full, without band.

    The penalty enters the steps whose running number (from 0, over the whole training) is a
    multiple of ``penalty.every``, multiplied by ``every`` (D90); the history reports its unscaled
    mean over those steps of the epoch (NaN in an epoch without one). On CUDA with ``cuda_graph``
    the steps without penalty are replayed from the graph of the plain step, the penalty steps
    from their own graph (``penalty_graph`` and ``plain_graph`` in the result; eager if a capture
    fails); every mode gives the weights of the fully eager training bit by bit.

    ``combined`` (D110): the rollout part is the penalty of the steps above (``loss_penalty`` in the
    history is its mean), the Jacobian part enters every step with ``jacobian_weight`` (inside the
    plain step and its graph); the info of the Jacobian part (``penalty_jacobian``, ...) is the mean
    over all steps of the epoch, that of the rollout part over the penalty steps.

    With a penalty the best epoch and the patience follow :class:`EpochChoice` (D89): after
    every epoch the penalty is evaluated at the speeds of ``V_GRID`` inside ``[v_min, v_max]``
    (:func:`validation_penalty`, ``val_penalty*`` in the history), and a feasible epoch
    (``val_penalty <= tolerance``) beats an infeasible one. Without penalty the rule of M2.
    """
    t_start = time.perf_counter()
    if not val_events:
        raise ValueError("early stopping needs validation events")
    seed_everything(seed)
    device, dtype = resolve_device(cfg.device), DTYPES[cfg.dtype]
    model.to(device=device, dtype=dtype)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"model {model.name!r} has no trainable parameters")
    # capturable Adam on every CUDA run: with and without the graph the arithmetic is the same
    optimizer = torch.optim.Adam(params, lr=cfg.lr, capturable=device.type == "cuda")
    stats = training_context(train_events, seed, cfg.band)
    data = EventTensors(train_events, device, dtype)
    ends = data.window_ends(model.window)
    steps = min(math.ceil(len(ends) / cfg.batch_size), cfg.steps_per_epoch)
    sampler = torch.Generator().manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
    weight, acc_weight = float(cfg.rollout.weight), float(cfg.acc_weight)
    penalty, penalty_weight = cfg.penalty, float(cfg.penalty.weight)
    if penalty.active and penalty.exist_v_min is None and penalty.exist_v_max is None:
        # an equilibrium is demanded at the speeds the training data cover (1 % - 99 % quantiles)
        penalty = dataclasses.replace(
            penalty,
            exist_v_min=max(penalty.v_min, float(stats["box_low"][2])),
            exist_v_max=min(penalty.v_max, float(stats["box_high"][2])),
        )
    grid = None
    if penalty.active:
        # existence "band": the equilibria of the penalty are anchored to the spacing band of the
        # training data (docs/m4_contract.md, 1.3); "fixed" and a data set without band: M3 in full
        band = (stats["band"] if penalty.band is None else penalty.band) if penalty.existence == "band" else None
        penalty = dataclasses.replace(penalty, band=band)
        # speeds of the validation penalty (D89), created once, outside any CUDA graph
        grid = torch.tensor([v for v in V_GRID if penalty.v_min <= v <= penalty.v_max], device=device, dtype=dtype)
        if not len(grid):
            raise ValueError(f"no speed of V_GRID in [{penalty.v_min}, {penalty.v_max}] for the validation penalty")
    info_keys = INFO_KEYS[penalty.kind]
    speed_seed = int(np.random.SeedSequence(seed, spawn_key=(1,)).generate_state(1)[0])
    speed_sampler = torch.Generator().manual_seed(speed_seed)
    # static buffer (read by the CUDA graphs), refilled in place every epoch
    speeds = torch.empty(penalty.n_equilibria, device=device, dtype=dtype)
    # a penalty step carries `every` times the penalty: its weight per step stays `weight` on average (D90)
    every = penalty.every if penalty.active else 1
    penalty_scale = penalty_weight * every
    # combined (D110): the Jacobian part of the memoryless view enters every step with its own weight
    combined = penalty.kind == "combined"
    jacobian_weight = float(penalty.jacobian_weight)
    every_step = combined and jacobian_weight > 0

    def train_step(index: Tensor, starts: Tensor, guard: bool, with_penalty: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Loss, loss terms and penalty info of one batch and the update, skipped when ``guard``
        finds the loss non-finite; the penalty only when ``with_penalty`` (combined: its rollout
        part; the Jacobian part on every step)."""
        optimizer.zero_grad(set_to_none=False)  # gradient buffers stay in place for the CUDA graphs
        state, target = data.state_windows(index, model.window), data.targets(index)
        loss_acc = torch.mean((model(state) - target) ** 2)
        if weight > 0:
            loss_rollout = rollout_loss(
                model, data, starts, horizon=cfg.rollout.horizon, var_s=stats["var_s"], var_v=stats["var_v"]
            )
        else:
            loss_rollout = torch.zeros_like(loss_acc)
        loss_extra = model.extra_loss({"state": state, "target": target})
        loss = acc_weight * loss_acc + weight * loss_rollout + loss_extra
        loss_penalty, info = torch.zeros_like(loss_acc), torch.zeros(0, dtype=torch.float64, device=device)
        if combined and (with_penalty or every_step):  # one search for the equilibria of both parts
            rollout_part, jacobian_part, values = combined_parts(model, speeds, penalty, rollout=with_penalty)
            if with_penalty:
                loss_penalty = rollout_part
                loss = loss + penalty_scale * loss_penalty
            if every_step:
                loss = loss + jacobian_weight * jacobian_part
            info = torch.stack([values[k] for k in info_keys]).detach()
        elif with_penalty:
            loss_penalty, values = stability_penalty(model, speeds, penalty)
            loss = loss + penalty_scale * loss_penalty
            info = torch.stack([values[k] for k in info_keys]).detach()
        if not guard or torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
        return loss.detach(), torch.stack((loss_acc, loss_rollout, loss_extra, loss_penalty)).detach().double(), info

    # CUDA graphs of the plain step and of the penalty step of the capturable kinds; both replay on
    # the same parameters, gradient buffers, optimizer state and static speeds. A graph is captured
    # at its first full batch after its eager warm-up, so with every = 1 only the penalty graph exists.
    use_graph = cfg.cuda_graph and device.type == "cuda"
    capture_penalty = use_graph and penalty.active and penalty.kind in CAPTURABLE  # every kind so far
    graphs = {
        False: _GraphedStep(functools.partial(train_step, with_penalty=False), model) if use_graph else None,
        True: _GraphedStep(functools.partial(train_step, with_penalty=True), model) if capture_penalty else None,
    }
    use_graph = use_graph and (penalty.kind in CAPTURABLE or every > 1)  # some step is replayed from a graph
    history: list[dict[str, float]] = []
    # epoch 0 = the initial weights: a hybrid starts as its physics model and must not end worse.
    # With a penalty the initial weights are not eligible: they were not trained under it, and the
    # unpenalised physics would win every selection in which the penalty costs accuracy.
    initial = summarise(evaluate_closed_loop(model, val_events, warmup=cfg.warmup, device=device))
    initial_rmse = initial["rmse_s_mean"]
    best_state = _clone_state(model)
    eligible = math.isfinite(initial_rmse) and not penalty.active
    choice = EpochChoice(
        cfg.patience, tolerance=penalty.tolerance if penalty.active else None, progress=penalty.progress,
        best_rmse=initial_rmse if eligible else math.inf,
    )  # fmt: skip
    stop_reason, non_finite, step_number, penalty_steps = "max_epochs", None, 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        # segment starts are drawn even without rollout loss: the batches stay the same
        order = torch.randperm(len(ends), generator=sampler)[: steps * cfg.batch_size].to(device)
        seg_starts = torch.randint(len(ends), (steps, cfg.rollout.segments), generator=sampler).to(device)
        if penalty.active:  # also in an epoch without penalty step: the speeds do not depend on `every`
            speeds.copy_(sample_speeds(penalty, speed_sampler, device, dtype))
        totals = torch.zeros(len(TERMS), dtype=torch.float64, device=device)
        infos = []  # penalty info of the steps of the epoch that evaluate a penalty (combined: every step)
        n_penalty = 0  # penalty steps of the epoch
        for step in range(steps):
            index = ends[order[step * cfg.batch_size : (step + 1) * cfg.batch_size]]
            with_penalty = penalty.active and step_number % every == 0
            step_number += 1
            graph = graphs[with_penalty] if len(index) == cfg.batch_size else None
            if graph is not None:
                loss, terms, info = graph(index, ends[seg_starts[step]])
            else:
                loss, terms, info = train_step(index, ends[seg_starts[step]], True, with_penalty)
            if not torch.isfinite(loss):  # the best weights are restored below
                non_finite = {"epoch": epoch, "step": step, **dict(zip(TERMS, terms.tolist()))}
                break
            totals += terms  # a step without penalty adds 0 to loss_penalty
            if with_penalty:
                penalty_steps += 1
                n_penalty += 1
            if with_penalty or every_step:
                infos.append(info.clone())  # a replayed graph overwrites its outputs
        if non_finite is not None:
            stop_reason = "non_finite_loss"
            break
        t_val = time.perf_counter()
        val = summarise(evaluate_closed_loop(model, val_events, warmup=cfg.warmup, device=device))
        checks = validation_penalty(model, grid, penalty) if penalty.active else {}
        means = dict(zip(TERMS, (totals / steps).tolist()))
        loss_mean = acc_weight * means["loss_acc"] + weight * means["loss_rollout"] + means["loss_extra"]
        if penalty.active:
            # unscaled mean over the penalty steps of the epoch (with every = 1 all steps: as before)
            if n_penalty != steps:
                penalty_sum = totals[TERMS.index("loss_penalty")].item()
                means["loss_penalty"] = penalty_sum / n_penalty if n_penalty else math.nan
            if n_penalty:  # the mean loss that was optimised: `every` x the penalty on n_penalty of the steps
                loss_mean += penalty_weight * (every * n_penalty / steps) * means["loss_penalty"]
            # NaN entries (combined: the rollout part on a step without it) are left out of the means
            info_means = torch.nanmean(torch.stack(infos), dim=0).tolist() if infos else [math.nan] * len(info_keys)
            means.update({f"penalty_{k}": value for k, value in zip(info_keys, info_means)})
            if every_step:  # combined: jacobian_weight x the Jacobian part on every step (D110)
                loss_mean += jacobian_weight * means["penalty_jacobian"]
        history.append(
            {
                "epoch": epoch,
                "loss": loss_mean,
                **means,
                "val_rmse_s": val["rmse_s_mean"],
                "val_rmse_v": val["rmse_v_mean"],
                "val_collision_rate": val["collision_rate"],
                **checks,
                "time_train_s": t_val - t_epoch,
                "time_val_s": time.perf_counter() - t_val,
            }
        )
        is_best, stop = choice.update(epoch, val["rmse_s_mean"], checks.get("val_penalty"))
        if is_best:
            best_state = _clone_state(model)
        elif stop:
            stop_reason = "early_stopping"
            break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "history": history,
        "best_epoch": choice.best_epoch,
        "best_val_rmse_s": choice.best_rmse,
        "best_feasible": choice.best_feasible if penalty.active else None,  # D89; None without penalty
        "n_feasible": choice.n_feasible if penalty.active else None,
        "penalty_steps": penalty_steps,  # training steps with the penalty term (D90)
        # the penalty steps were replayed from their CUDA graph (False: eager, also after a failed capture)
        "penalty_graph": graphs[True] is not None and graphs[True].captured and not graphs[True].failed,
        # the same for the steps without penalty (combined: they hold the Jacobian part, D110)
        "plain_graph": graphs[False] is not None and graphs[False].captured and not graphs[False].failed,
        "initial_val_rmse_s": initial_rmse,
        "epochs": len(history),
        "stop_reason": stop_reason,
        "non_finite": non_finite,
        "steps_per_epoch": steps,
        "n_window_ends": int(len(ends)),
        "var_s": stats["var_s"],
        "var_v": stats["var_v"],
        "cuda_graph": use_graph,
        "penalty_band": penalty.active and penalty.band is not None,  # False: M3 equilibria and existence term
        "device": str(device),
        "dtype": cfg.dtype,
        "wall_time_s": time.perf_counter() - t_start,
    }
