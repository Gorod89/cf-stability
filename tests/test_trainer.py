"""Gradient training: convergence, rollout loss, reproducibility, early stopping, loss terms, NaN guard,
stability penalty."""

import copy
import dataclasses
import json
import math
import os
import time
import warnings
from typing import Sequence

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cf_stability.data.schema import DT, Event
from cf_stability.models import ResidualIDM
from cf_stability.models.base import CFModel, InputScaler, ModelContext
from cf_stability.models.idm import idm_acc, idm_equilibrium_spacing
from cf_stability.stability.analytic import criterion, partials
from cf_stability.stability.equilibrium import V_GRID, find_equilibria
from cf_stability.stability.penalties import INFO_KEYS, PenaltyConfig, stability_penalty
from cf_stability.train import trainer as trainer_module
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.evaluate import evaluate_closed_loop, summarise
from cf_stability.train.tensors import BandConfig, EventTensors, training_context
from cf_stability.train.trainer import EpochChoice, RolloutConfig, TrainConfig, rollout_loss, train_model

# the synthetic data have too few near-steady samples for the default band (200 per speed): with 20
# the band lists 10 speeds, 10-19 m/s
SMALL_BAND = BandConfig(min_samples=20)
# opt-in: the GPU is shared, and hiding it with CUDA_VISIBLE_DEVICES crashes torch on this machine
GPU_ONLY = pytest.mark.skipif(
    not os.environ.get("CF_GPU_TESTS") or not torch.cuda.is_available(), reason="GPU test: set CF_GPU_TESTS=1"
)
# the history of an epoch without penalty (M2)
M2_HISTORY_KEYS = {
    "epoch", "loss", "loss_acc", "loss_rollout", "loss_extra", "loss_penalty", "val_rmse_s", "val_rmse_v",
    "val_collision_rate", "time_train_s", "time_val_s",
}  # fmt: skip

P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}


def make_event(i: int, rng: np.random.Generator, n: int = 150) -> Event:
    t = DT * np.arange(n)
    base, amp, period = rng.uniform(8.0, 20.0), rng.uniform(1.0, 4.0), rng.uniform(6.0, 14.0)
    v_lead = base + amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    x_lead = np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    v0 = v_lead[0] + rng.uniform(-2.0, 2.0)
    s0 = idm_equilibrium_spacing(torch.tensor(v0), P["v0"], P["T"], P["s0"]).item() + rng.uniform(-3.0, 5.0)
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, **P), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(s0), torch.tensor(v0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"tr/a/{i}|L|0", dataset="tr", site="a", follower_id=f"tr/a/{i}", leader_id=f"tr/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


@pytest.fixture(scope="module")
def data() -> tuple[list[Event], list[Event]]:
    rng = np.random.default_rng(0)
    events = [make_event(i, rng) for i in range(32)]
    return events[:24], events[24:]


class MLP(CFModel):
    name = "test_mlp"

    def __init__(self, events: list[Event], hidden: int = 32) -> None:
        super().__init__()
        ctx = training_context(events, 0)
        self.scaler = InputScaler(ctx["center"], ctx["scale"])
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state_history: Tensor) -> Tensor:
        return self.net(self.scaler(state_history[:, -1])).squeeze(-1)


class WindowGRU(CFModel):
    name = "test_gru"
    window = 10

    def __init__(self, events: list[Event]) -> None:
        super().__init__()
        ctx = training_context(events, 0)
        self.scaler = InputScaler(ctx["center"], ctx["scale"])
        self.gru = nn.GRU(3, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, state_history: Tensor) -> Tensor:
        out, _ = self.gru(self.scaler(state_history[:, -self.window :]))
        return self.head(out[:, -1]).squeeze(-1)


class Constant(CFModel):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, state_history: Tensor) -> Tensor:
        return self.c.expand(state_history.shape[0])


class Counting(MLP):
    """Counts the forward calls in training mode; returns NaN after ``nan_after`` of them."""

    def __init__(self, events: list[Event], nan_after: int | None = None) -> None:
        super().__init__(events)
        self.calls, self.nan_after = 0, nan_after

    def forward(self, state_history: Tensor) -> Tensor:
        if self.training:
            self.calls += 1
        out = super().forward(state_history)
        return out + torch.nan if self.nan_after is not None and self.calls > self.nan_after else out


class WithExtraLoss(MLP):
    def __init__(self, events: list[Event]) -> None:
        super().__init__(events)
        self.batch_shapes: set[tuple] = set()

    def extra_loss(self, batch: dict[str, Tensor]) -> Tensor:
        self.batch_shapes.add((tuple(batch["state"].shape), tuple(batch["target"].shape)))
        return 0.25 + 0.0 * self.net[0].weight.sum()


def config(**kwargs) -> TrainConfig:
    base = dict(
        lr=3e-3, batch_size=256, max_epochs=3, steps_per_epoch=5, patience=100,
        rollout=RolloutConfig(weight=1.0, horizon=30, segments=64), device="cpu",
    )  # fmt: skip
    return TrainConfig(**{**base, **kwargs})


def weights(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_mlp_learns_the_idm_in_closed_loop(data):
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    t0 = time.perf_counter()
    res = train_model(model, train, val, config(max_epochs=30, steps_per_epoch=15), seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0
    assert res["epochs"] == 30 and res["stop_reason"] == "max_epochs"
    assert res["best_val_rmse_s"] < 1.0, [round(h["val_rmse_s"], 3) for h in res["history"]]
    assert res["history"][-1]["loss_rollout"] < 0.1 * res["history"][0]["loss_rollout"]
    # the best weights are the ones in the model
    val_now = summarise(evaluate_closed_loop(model, val, device="cpu"))["rmse_s_mean"]
    assert val_now == res["best_val_rmse_s"] == res["history"][res["best_epoch"] - 1]["val_rmse_s"]


@pytest.mark.parametrize("model_cls", [MLP, WindowGRU])
def test_rollout_loss_has_gradients_and_decreases(data, model_cls):
    train, val = data
    torch.manual_seed(0)
    model = model_cls(train)
    tensors, stats = EventTensors(train), training_context(train, 0)
    ends = tensors.window_ends(model.window)
    starts = ends[torch.randint(len(ends), (32,), generator=torch.Generator().manual_seed(0))]
    rollout_loss(model, tensors, starts, horizon=20, var_s=stats["var_s"], var_v=stats["var_v"]).backward()
    for name, p in model.named_parameters():  # the loss depends on the parameters only through the rollout
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    res = train_model(model, train, val, config(max_epochs=4), seed=0)
    assert res["history"][-1]["loss_rollout"] < res["history"][0]["loss_rollout"]


@pytest.mark.parametrize("value", [100.0, -100.0])
def test_rollout_loss_clips_accelerations_and_speeds(data, value):
    train, _ = data
    tensors, model, horizon = EventTensors(train, dtype=torch.float64), Constant(value), 20
    starts = torch.tensor([[0, 0], [3, 40], [5, 140]])  # the last segment passes the end of its event
    loss = rollout_loss(model, tensors, starts, horizon=horizon, var_s=2.0, var_v=0.5)
    sse_s = sse_v = n = 0.0
    for e, j in starts.tolist():
        ev, stop = train[e], min(j + horizon, len(train[e]) - 1)
        ref = rollout_memoryless(
            lambda s, dv, v: torch.full_like(s, value), torch.tensor(ev.x_lead[j : stop + 1]),
            torch.tensor(ev.v_lead[j : stop + 1]), torch.tensor(ev.s[j]), torch.tensor(ev.v[j]),
        )  # fmt: skip
        sse_s += np.sum((ref.s.numpy()[1:] - ev.s[j + 1 : stop + 1]) ** 2)
        sse_v += np.sum((ref.v.numpy()[1:] - ev.v[j + 1 : stop + 1]) ** 2)
        n += stop - j
    assert loss.item() == pytest.approx((sse_s / 2.0 + sse_v / 0.5) / n, rel=1e-9)
    loss.backward()
    assert model.c.grad.item() == 0.0  # the clipped acceleration does not depend on the parameter


def test_same_seed_same_run(data):
    train, val = data
    runs = []
    for seed in (0, 0, 1):
        torch.manual_seed(0)  # same initial weights; the seed of the run drives the batches
        model = MLP(train)
        res = train_model(model, train, val, config(), seed=seed)
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((history, weights(model)))
    (h0, w0), (h1, w1), (h2, w2) = runs
    assert h0 == h1 and all(torch.equal(w0[k], w1[k]) for k in w0)
    assert h0 != h2 and not all(torch.equal(w0[k], w2[k]) for k in w0)


def test_early_stopping_restores_the_best_weights(data):
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    res = train_model(model, train, val, config(lr=0.05, max_epochs=40, patience=2), seed=0)
    val_rmse = [h["val_rmse_s"] for h in res["history"]]
    assert res["stop_reason"] == "early_stopping" and res["epochs"] == res["best_epoch"] + 2 < 40
    assert res["best_val_rmse_s"] == min(val_rmse) < val_rmse[-1]
    assert summarise(evaluate_closed_loop(model, val, device="cpu"))["rmse_s_mean"] == min(val_rmse)


def test_zero_rollout_weight_trains_on_the_acceleration_only(data):
    train, val = data
    for weight, calls_per_step in ((0.0, 1), (1.0, 1 + 30)):
        torch.manual_seed(0)
        model = Counting(train)
        res = train_model(model, train, val, config(rollout=RolloutConfig(weight, 30, 64)), seed=0)
        assert model.calls == res["epochs"] * res["steps_per_epoch"] * calls_per_step  # no rollout without weight
        for h in res["history"]:
            assert (h["loss_rollout"] == 0.0) == (weight == 0.0)
            assert h["loss"] == pytest.approx(h["loss_acc"] + weight * h["loss_rollout"], rel=1e-12)


def test_extra_loss_is_added(data):
    train, val = data
    torch.manual_seed(0)
    model = WithExtraLoss(train)
    res = train_model(model, train, val, config(), seed=0)
    assert model.batch_shapes == {((256, 1, 3), (256,))}
    for h in res["history"]:
        assert h["loss_extra"] == 0.25
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + 0.25, rel=1e-12)


def mean_margin(model: CFModel) -> float:
    """Mean string-stability margin over the equilibria of the speed grid."""
    m = copy.deepcopy(model).double().eval()
    eq = find_equilibria(m)
    return criterion(*partials(m, eq.s[eq.found], eq.v[eq.found]))["margin"].mean().item()


def test_penalty_config():
    assert TrainConfig().penalty.kind == "none"
    cfg = TrainConfig.from_mapping({"penalty": {"kind": "gain", "weight": 0.1}})
    assert cfg.penalty.kind == "gain" and cfg.penalty.weight == 0.1 and cfg.penalty.n_equilibria == 3
    with pytest.raises(ValueError, match="unknown penalty keys"):
        TrainConfig.from_mapping({"penalty": {"kind": "gain", "lamda": 0.1}})
    cfg = TrainConfig.from_mapping(
        {"band": {"min_samples": 50}, "penalty": {"kind": "linear_gain", "existence": "fixed", "aggregate": "mean"}}
    )
    assert cfg.band == BandConfig(min_samples=50)
    assert (cfg.penalty.existence, cfg.penalty.aggregate) == ("fixed", "mean")
    assert TrainConfig().band == BandConfig() and TrainConfig().penalty.band is None
    json.dumps(dataclasses.asdict(cfg))  # scripts/train.py writes it to metrics.json
    with pytest.raises(ValueError, match="unknown band keys"):
        TrainConfig.from_mapping({"band": {"min_sample": 50}})


def test_jacobian_penalty_raises_the_margin(data):
    train, val = data
    runs = []
    jacobian = PenaltyConfig(kind="jacobian", weight=1.0)
    for penalty in (PenaltyConfig(), jacobian, jacobian):
        torch.manual_seed(0)  # same initial weights and, with or without penalty, the same batches
        model = MLP(train)
        res = train_model(model, train, val, config(max_epochs=8, penalty=penalty), seed=0)
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((history, weights(model), mean_margin(model), res))
    (h_none, _, margin_none, res_none), (h0, w0, margin, res), (h1, w1, _, _) = runs
    # the initial network has no equilibrium: only the existence term acts at first; the margin of
    # the IDM data is about 0
    assert margin > margin_none + 0.01, (margin, margin_none)
    assert res["best_epoch"] == res_none["best_epoch"] == 8 and res["cuda_graph"] is False
    assert repr(h0) == repr(h1) and all(torch.equal(w0[k], w1[k]) for k in w0)  # reproducible (NaN != NaN)
    for h in h_none:
        assert h["loss_penalty"] == 0.0 and not any(k.startswith("penalty_") for k in h)
    for h in h0:
        assert 0.0 <= h["penalty_n_no_equilibrium"] <= 16.0
        terms = h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + h["loss_penalty"]
        assert h["loss"] == pytest.approx(terms, rel=1e-12)
    # the existence term gives the network equilibria within the first epoch and then vanishes
    assert 0.0 < h0[0]["penalty_n_no_equilibrium"] < 16.0 and h0[0]["penalty_existence"] > 0.0
    assert h0[-1]["penalty_n_no_equilibrium"] < h0[0]["penalty_n_no_equilibrium"]
    assert h0[-1]["penalty_existence"] < h0[0]["penalty_existence"] and h0[-1]["loss_penalty"] > 0.0
    assert res["penalty_band"] is False  # too few near-steady samples for the default band: the terms of M3


def test_trainer_passes_the_band_of_the_training_data(data, monkeypatch):
    train, val = data
    seen: list[PenaltyConfig] = []

    def recording(model, speeds, cfg, equilibria=None):
        seen.append(cfg)
        return stability_penalty(model, speeds, cfg, equilibria)

    monkeypatch.setattr(trainer_module, "stability_penalty", recording)
    own = {"v": [5.0, 30.0], "s_low": [5.0, 40.0], "s_median": [10.0, 50.0], "s_high": [15.0, 60.0]}
    cases = (
        (SMALL_BAND, PenaltyConfig(kind="jacobian"), training_context(train, 0, SMALL_BAND)["band"]),
        (BandConfig(), PenaltyConfig(kind="jacobian"), None),  # no band in the data
        (SMALL_BAND, PenaltyConfig(kind="linear_gain", band=own), own),  # a band of the config is kept
        # existence "fixed" reproduces M3 in full: no band, not even one of the config (D89)
        (SMALL_BAND, PenaltyConfig(kind="jacobian", existence="fixed"), None),
        (SMALL_BAND, PenaltyConfig(kind="linear_gain", existence="fixed", band=own), None),
    )
    for band_cfg, penalty, expected in cases:
        seen.clear()
        torch.manual_seed(0)
        cfg = config(max_epochs=1, steps_per_epoch=2, band=band_cfg, penalty=penalty)
        res = train_model(MLP(train), train, val, cfg, seed=0)
        # two training steps and the validation penalty of the epoch, all with the same settings
        assert len(seen) == 3 and all(p.band == expected for p in seen) and seen[0] is seen[2]
        assert res["penalty_band"] is (expected is not None)
        assert cfg.penalty is penalty and cfg.penalty.band == penalty.band  # the config of the run is not changed
    assert len(cases[0][2]["v"]) == 10


def choose(
    rmse: Sequence[float], penalty: Sequence[float] | None = None, *, patience: int = 3, initial: float = math.inf
) -> tuple[EpochChoice, int]:
    """The choice after prescribed validation sequences (epochs 1, 2, ...) and the last epoch run."""
    choice = EpochChoice(patience, tolerance=None if penalty is None else 0.01, progress=0.05, best_rmse=initial)
    for epoch, (value, pen) in enumerate(zip(rmse, penalty or [None] * len(rmse)), start=1):
        if choice.update(epoch, value, pen)[1]:
            return choice, epoch
    return choice, len(rmse)


def test_epoch_choice_without_penalty_is_the_rule_of_m2():
    choice, stop = choose([5.0, 4.0, 4.5, 4.2, 4.1, 3.9])
    assert (choice.best_epoch, choice.best_rmse, stop) == (2, 4.0, 5)  # three epochs without a smaller RMSE
    assert choose([5.0, 5.0, 4.0])[0].best_epoch == 3  # an equal RMSE is no improvement
    choice, stop = choose([5.0, 4.0, 3.5], initial=3.0)  # eligible initial weights that stay best
    assert (choice.best_epoch, stop) == (0, 3)
    assert choose([math.nan, math.inf, 4.0], patience=5)[0].best_epoch == 3


def test_epoch_choice_feasible_beats_a_smaller_rmse():
    # the smallest RMSE (epoch 2) is infeasible; epoch 3 is the first feasible one, epoch 5 is feasible but worse
    choice, stop = choose([3.0, 2.0, 2.5, 2.8, 2.6], [0.5, 0.2, 0.005, 0.02, 0.001], patience=10)
    assert (choice.best_epoch, choice.best_rmse, choice.best_feasible, choice.n_feasible, stop) == (3, 2.5, True, 2, 5)
    # a later feasible epoch with a smaller RMSE wins; an infeasible one never does again
    assert choose([3.0, 2.5, 2.4, 1.0], [0.5, 0.005, 0.01, 0.3], patience=10)[0].best_epoch == 3
    # without a feasible epoch: the smallest RMSE (the rule of the specification)
    choice, _ = choose([3.0, 2.0, 2.5], [0.5, 0.4, 0.35], patience=10)
    assert (choice.best_epoch, choice.best_feasible, choice.n_feasible) == (2, False, 0)
    # a feasible epoch with a non-finite RMSE is not chosen; a NaN penalty is infeasible
    choice, _ = choose([2.0, math.nan, 2.5], [0.5, 0.001, math.nan], patience=10)
    assert (choice.best_epoch, choice.n_feasible, choice.min_penalty) == (1, 1, 0.001)


def test_epoch_choice_patience_restarts_on_progress_before_the_first_feasible_epoch():
    rising = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
    # the penalty falls by 10 % per epoch: the run goes on although the RMSE rises (patience 2)
    assert choose(rising[:6], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], patience=2)[1] == 6
    # 3 % per epoch is less than the progress of 5 %: two epochs without improvement stop the run
    assert choose(rising[:6], [1.0, 0.97, 0.94, 0.91, 0.88, 0.85], patience=2)[1] == 3
    # progress is measured against the smallest earlier value: 0.77 is not 5 % below 0.8
    assert choose(rising[:5], [1.0, 0.8, 0.9, 0.77, 0.6], patience=2)[1] == 4
    # after the first feasible epoch (5) a falling penalty no longer restarts the patience
    choice, stop = choose(rising, [1.0, 0.9, 0.8, 0.7, 0.005, 0.004, 0.003, 0.002, 0.001], patience=2)
    assert (choice.best_epoch, choice.n_feasible, stop) == (5, 3, 7)
    choice, stop = choose(rising, [1.0, 0.9, 0.8, 0.7, 0.005, 0.5, 0.2, 0.1, 0.05], patience=2)  # infeasible again
    assert (choice.best_epoch, stop) == (5, 7)


def prescribe_validation(monkeypatch, rmse: Sequence[float], penalty: Sequence[float]) -> list[dict[str, Tensor]]:
    """``train_model`` with the validation RMSE (initial weights first) and penalty of every epoch
    replaced by prescribed values; returns the weights of every epoch as they were validated."""
    rmse_values, penalty_values, states = iter(rmse), iter(penalty), []

    def validation_penalty(model, speeds, penalty_cfg):
        states.append(weights(model))
        return {"val_penalty": next(penalty_values)}

    def summarise(_) -> dict[str, float]:
        return {"rmse_s_mean": next(rmse_values), "rmse_v_mean": 0.0, "collision_rate": 0.0}

    monkeypatch.setattr(trainer_module, "evaluate_closed_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "summarise", summarise)
    monkeypatch.setattr(trainer_module, "validation_penalty", validation_penalty)
    return states


@pytest.mark.parametrize(
    "rmse, penalty, best, epochs, feasible, stop",
    [
        # the RMSE-best epoch 2 is infeasible, the later epoch 3 is feasible (the initial RMSE 1.0 is not
        # eligible); epochs 4 and 5 are no better: patience 2 ends the run after epoch 5
        ([1.0, 3.0, 2.0, 2.5, 2.8, 2.6], [0.5, 0.2, 0.005, 0.02, 0.001], 3, 5, 2, "early_stopping"),
        # the RMSE rises from epoch 1, the penalty falls by 10 % per epoch until epoch 5 is feasible;
        # from then on only a better epoch restarts the patience: the run stops after epoch 7
        ([1.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], [1.0, 0.9, 0.8, 0.7, 0.005, 0.004, 0.003, 0.002], 5, 7, 3,
         "early_stopping"),  # fmt: skip
        # no feasible epoch: the smallest RMSE; the penalty falls by more than 5 % per epoch, no stop
        ([1.0, 3.0, 2.0, 2.5, 2.9], [0.5, 0.4, 0.35, 0.3], 2, 4, 0, "max_epochs"),
    ],
)
def test_best_epoch_under_a_penalty(data, monkeypatch, rmse, penalty, best, epochs, feasible, stop):
    train, val = data
    states = prescribe_validation(monkeypatch, rmse, penalty)
    torch.manual_seed(0)
    model = MLP(train)
    cfg = config(max_epochs=len(penalty), steps_per_epoch=2, patience=2, penalty=PenaltyConfig(kind="jacobian"))
    res = train_model(model, train, val, cfg, seed=0)
    assert (res["best_epoch"], res["epochs"], res["n_feasible"], res["stop_reason"]) == (best, epochs, feasible, stop)
    assert res["best_feasible"] is (feasible > 0) and res["best_val_rmse_s"] == rmse[best]
    assert [h["val_penalty"] for h in res["history"]] == penalty[:epochs]
    final = weights(model)
    assert all(torch.equal(final[k], states[best - 1][k]) for k in final)  # the weights of the chosen epoch


def test_validation_penalty_is_the_penalty_at_the_grid_speeds(data):
    """``val_penalty`` is the penalty of the validated weights at every speed of V_GRID inside
    [v_min, v_max], with the band and the settings of the training, in eval mode."""
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    cfg = config(max_epochs=2, band=SMALL_BAND, penalty=PenaltyConfig(kind="linear_gain", v_min=8.0, v_max=25.0))
    res = train_model(model, train, val, cfg, seed=0)
    stats = training_context(train, 0, SMALL_BAND)
    low, high = max(8.0, stats["box_low"][2]), min(25.0, stats["box_high"][2])  # the speeds of the training data
    settings = dataclasses.replace(cfg.penalty, band=stats["band"], exist_v_min=low, exist_v_max=high)
    grid = torch.tensor([v for v in V_GRID if 8.0 <= v <= 25.0])  # float32, the dtype of the training
    value, info = stability_penalty(model.eval(), grid, settings)
    best = res["history"][res["best_epoch"] - 1]
    assert best["val_penalty"] == value.item() and len(grid) == 18
    parts = {k: best[f"val_penalty_{k}"] for k in info}
    assert parts == pytest.approx({k: v.item() for k, v in info.items()}, nan_ok=True)
    assert res["n_feasible"] == sum(h["val_penalty"] <= 0.01 for h in res["history"])
    parts = {f"{prefix}_{k}" for k in info for prefix in ("penalty", "val_penalty")}
    assert set(best) == M2_HISTORY_KEYS | {"val_penalty"} | parts


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_ONLY)])
def test_validation_penalty_leaves_the_training_unchanged(data, monkeypatch, device):
    """The same epochs and weights as with a validation penalty that computes nothing: no gradient,
    no buffer (the power iteration of ResidualIDM runs in training mode only) and, on the GPU, no
    static buffer of the captured step is touched."""
    train, val = data
    runs = []
    for stub in (False, True):
        if stub:
            monkeypatch.setattr(trainer_module, "validation_penalty", lambda *args: {"val_penalty": 1.0})
        torch.manual_seed(0)
        model = residual_idm(train)
        penalty = PenaltyConfig(kind="linear_gain", n_equilibria=4, tolerance=-1.0)  # never feasible: the same choice
        cfg = config(device=device, batch_size=64, max_epochs=3, steps_per_epoch=10, band=SMALL_BAND, penalty=penalty)
        res = train_model(model, train, val, cfg, seed=0)
        history = [{k: v for k, v in h.items() if not k.startswith(("time", "val_penalty"))} for h in res["history"]]
        runs.append((repr(history), weights(model), res["cuda_graph"]))
    (h_real, w_real, graph), (h_stub, w_stub, _) = runs
    assert graph is (device == "cuda") and h_real == h_stub
    assert all(torch.equal(w_real[k], w_stub[k]) for k in w_real)


class PenaltyProbe:
    """Stub of ``stability_penalty`` that returns the prescribed ``values`` in turn and records, for
    every training step that evaluates it, the running number of the step (one forward call of a
    ``Counting`` model per step without rollout loss) and the gradient of the step's loss with
    respect to the penalty. The validation penalty (eval mode) gets 0.5."""

    def __init__(self, model: "Counting", values: Sequence[float]) -> None:
        self.model, self.values = model, iter(values)
        self.steps: list[int] = []
        self.grads: list[float] = []

    def __call__(self, model, speeds, cfg, equilibria=None):
        if not model.training:
            return torch.tensor(0.5), {k: torch.tensor(0.5, dtype=torch.float64) for k in INFO_KEYS[cfg.kind]}
        self.steps.append(self.model.calls - 1)
        value = torch.tensor(next(self.values), requires_grad=True)
        value.register_hook(lambda grad: self.grads.append(grad.item()))
        return value, {k: value.detach().double() for k in INFO_KEYS[cfg.kind]}  # parts equal to the value


@pytest.mark.parametrize(
    "every, steps, means",
    [
        # 3 epochs of 5 steps (running numbers 0-14); the penalty values are 0.1, 0.2, ... in turn
        (1, list(range(15)), [0.3, 0.8, 1.3]),
        (3, [0, 3, 6, 9, 12], [0.15, 0.35, 0.5]),  # epoch 1 holds the steps 0 and 3, epoch 2 6 and 9
        (12, [0, 12], [0.1, math.nan, 0.2]),  # the second epoch holds no penalty step
    ],
)
def test_penalty_on_every_nth_step(data, monkeypatch, every, steps, means):
    train, val = data
    torch.manual_seed(0)
    model = Counting(train)
    probe = PenaltyProbe(model, [0.1 * (k + 1) for k in range(15)])
    monkeypatch.setattr(trainer_module, "stability_penalty", probe)
    penalty = PenaltyConfig(kind="jacobian", weight=0.5, every=every)
    res = train_model(model, train, val, config(rollout=RolloutConfig(weight=0.0), penalty=penalty), seed=0)
    assert res["steps_per_epoch"] == 5 and res["epochs"] == 3 and model.calls == 15
    assert probe.steps == steps and res["penalty_steps"] == len(steps)
    # a penalty step carries every x the penalty: the gradient of its loss with respect to it is weight x every
    assert probe.grads == [0.5 * every] * len(steps)
    for h, mean in zip(res["history"], means):
        # unscaled means over the penalty steps of the epoch, NaN without one; the validation penalty every epoch
        assert h["loss_penalty"] == pytest.approx(mean, nan_ok=True) and h["val_penalty"] == 0.5
        assert all(h[f"penalty_{k}"] == pytest.approx(mean, nan_ok=True) for k in INFO_KEYS["jacobian"])
        n_penalty = sum(s // 5 == h["epoch"] - 1 for s in steps)
        penalty_part = 0.5 * every * n_penalty / 5 * h["loss_penalty"] if n_penalty else 0.0
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + penalty_part, rel=1e-12)


def test_penalty_on_every_nth_step_trains(data):
    """The real penalty on the steps 0, 3, 6, 9 of 10: other weights than on every step, finite parts."""
    train, val = data
    runs = {}
    for every in (1, 3):
        torch.manual_seed(0)
        model = MLP(train)
        cfg = config(max_epochs=2, band=SMALL_BAND, penalty=PenaltyConfig(kind="jacobian", every=every))
        runs[every] = (train_model(model, train, val, cfg, seed=0), weights(model))
    (res_1, w_1), (res_3, w_3) = runs[1], runs[3]
    assert (res_1["penalty_steps"], res_3["penalty_steps"]) == (10, 4)
    assert res_1["penalty_graph"] is res_3["penalty_graph"] is False  # CPU: every step eager
    assert not all(torch.equal(w_1[k], w_3[k]) for k in w_1)
    for h in res_3["history"]:
        assert math.isfinite(h["loss_penalty"]) and math.isfinite(h["penalty_existence"])


class CombinedProbe:
    """Stub of ``combined_parts`` (D110) that returns prescribed values of the two parts in turn and records,
    for every training step that calls it, the running number of the step (one forward call of a
    ``Counting`` model per step without rollout loss), whether the rollout part was asked for, and the
    gradients of the step's loss with respect to both parts."""

    def __init__(self, model: "Counting", rollout: Sequence[float], jacobian: Sequence[float]) -> None:
        self.model, self.rollout, self.jacobian = model, iter(rollout), iter(jacobian)
        self.calls: list[tuple[int, bool]] = []
        self.grads: list[tuple[str, float]] = []

    def part(self, name: str, value: float) -> Tensor:
        tensor = torch.tensor(value, requires_grad=True)
        tensor.register_hook(lambda grad: self.grads.append((name, grad.item())))
        return tensor

    def __call__(self, model, speeds, cfg, equilibria=None, *, rollout=True):
        self.calls.append((self.model.calls - 1, rollout))
        nan = torch.tensor(math.nan, dtype=torch.float64)
        jacobian = self.part("jacobian", next(self.jacobian))
        info = {key: nan for key in INFO_KEYS["combined"]}
        info.update(n_no_equilibrium=torch.tensor(1.0, dtype=torch.float64), jacobian=jacobian.detach().double())
        part = torch.tensor(0.0)
        if rollout:
            part = self.part("rollout", next(self.rollout))
            info.update(rollout=part.detach().double(), existence=part.detach().double())
        return part, jacobian, info


def test_combined_rollout_part_on_every_nth_step_jacobian_part_on_every_step(data, monkeypatch):
    """D110: the rollout part on the steps 0, 3, 6, ... with weight x every, the Jacobian part on every step
    with jacobian_weight; history: the rollout part as loss_penalty (mean over the penalty steps), the
    Jacobian part as penalty_jacobian (mean over all steps), both in the optimised loss."""
    train, val = data
    torch.manual_seed(0)
    model = Counting(train)
    rollout, jacobian = [0.1 * (k + 1) for k in range(5)], [0.01 * (k + 1) for k in range(15)]
    probe = CombinedProbe(model, rollout=rollout, jacobian=jacobian)
    monkeypatch.setattr(trainer_module, "combined_parts", probe)
    monkeypatch.setattr(trainer_module, "validation_penalty", lambda *args: {"val_penalty": 0.5})
    penalty = PenaltyConfig(kind="combined", weight=0.5, every=3, jacobian_weight=2.0, guard=3.0)
    res = train_model(model, train, val, config(rollout=RolloutConfig(weight=0.0), penalty=penalty), seed=0)
    assert res["steps_per_epoch"] == 5 and res["epochs"] == 3 and model.calls == 15
    assert probe.calls == [(k, k % 3 == 0) for k in range(15)] and res["penalty_steps"] == 5
    assert sorted(probe.grads) == sorted([("jacobian", 2.0)] * 15 + [("rollout", 0.5 * 3)] * 5)
    rollout_means, jacobian_means, n_penalty = [0.15, 0.35, 0.5], [0.03, 0.08, 0.13], [2, 2, 1]
    for h, rollout, jacobian, n in zip(res["history"], rollout_means, jacobian_means, n_penalty):
        assert h["loss_penalty"] == pytest.approx(rollout) and h["penalty_rollout"] == pytest.approx(rollout)
        assert h["penalty_existence"] == pytest.approx(rollout)  # a part of the rollout part: its steps only
        assert h["penalty_jacobian"] == pytest.approx(jacobian) and h["penalty_n_no_equilibrium"] == 1.0
        assert math.isnan(h["penalty_needle"]) and h["val_penalty"] == 0.5
        parts = 0.5 * 3 * n / 5 * h["loss_penalty"] + 2.0 * h["penalty_jacobian"]
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + parts, rel=1e-12)


def test_combined_kind_trains(data):
    """``kind: combined`` on a GRU with the real parts: their info in the history, the validation penalty of
    the epoch choice is the sum of the unweighted parts (D89, D110)."""
    train, val = data
    torch.manual_seed(0)
    model = WindowGRU(train)
    penalty = PenaltyConfig(
        kind="combined", weight=0.1, every=3, jacobian_weight=1.0, guard=0.05, n_equilibria=4, horizon_s=4.0,
        measure_s=2.0,
    )  # fmt: skip
    res = train_model(model, train, val, config(max_epochs=2, band=SMALL_BAND, penalty=penalty), seed=0)
    assert res["penalty_band"] is True and res["penalty_steps"] == 4  # steps 0, 3, 6, 9 of 10
    keys = {f"{prefix}_{key}" for prefix in ("penalty", "val_penalty") for key in INFO_KEYS["combined"]}
    for h in res["history"]:
        assert set(h) == M2_HISTORY_KEYS | {"val_penalty"} | keys
        assert all(math.isfinite(h[f"penalty_{k}"]) for k in ("rollout", "existence", "jacobian", "n_no_equilibrium"))
        assert h["penalty_rollout"] == pytest.approx(h["loss_penalty"], rel=1e-6)
        parts = 0.1 * 3 * 2 / 5 * h["loss_penalty"] + 1.0 * h["penalty_jacobian"]  # two penalty steps per epoch
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + parts, rel=1e-12)
        assert h["val_penalty"] == pytest.approx(h["val_penalty_rollout"] + h["val_penalty_jacobian"], rel=1e-6)
    assert res["n_feasible"] == sum(h["val_penalty"] <= 0.01 for h in res["history"])


def test_existence_kind_trains(data):
    """``kind: existence``: the existence term alone, with its keys in the history and the validation penalty."""
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    cfg = config(max_epochs=2, band=SMALL_BAND, penalty=PenaltyConfig(kind="existence", every=2))
    res = train_model(model, train, val, cfg, seed=0)
    assert res["penalty_band"] is True and res["penalty_steps"] == 5 and res["n_feasible"] is not None
    for h in res["history"]:
        parts = {f"{p}_{k}" for p in ("penalty", "val_penalty") for k in ("n_no_equilibrium", "existence")}
        assert set(h) == M2_HISTORY_KEYS | {"val_penalty"} | parts
        assert h["loss_penalty"] == pytest.approx(h["penalty_existence"], rel=1e-6)  # the penalty is the term
        assert h["val_penalty"] == h["val_penalty_existence"]
    assert res["history"][-1]["penalty_existence"] < res["history"][0]["penalty_existence"]


def test_monotone_kind_trains(data):
    """``kind: monotone`` (D117): the monotonicity terms of RACER with the existence term of the band, its keys
    in the history and in the validation penalty of the epoch choice; the string margin is reported only."""
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    cfg = config(max_epochs=2, band=SMALL_BAND, penalty=PenaltyConfig(kind="monotone"))
    res = train_model(model, train, val, cfg, seed=0)
    assert res["penalty_band"] is True and res["penalty_steps"] == 10 and res["n_feasible"] is not None
    keys = {f"{p}_{k}" for p in ("penalty", "val_penalty") for k in ("n_no_equilibrium", "mean_margin", "existence")}
    for h in res["history"]:
        assert set(h) == M2_HISTORY_KEYS | {"val_penalty"} | keys
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + h["loss_penalty"])
        assert h["loss_penalty"] >= h["penalty_existence"] - 1e-6  # the monotonicity terms add a non-negative part
    # the validation penalty of the best epoch is the penalty of the restored weights at the grid speeds
    grid = torch.tensor([v for v in V_GRID if 5.0 <= v <= 30.0], dtype=torch.float32)
    penalty = dataclasses.replace(cfg.penalty, band=training_context(train, 0, SMALL_BAND)["band"])
    value, info = stability_penalty(model, grid, penalty)
    best = res["history"][res["best_epoch"] - 1]
    assert best["val_penalty"] == pytest.approx(value.item(), rel=1e-5)
    assert best["val_penalty_mean_margin"] == pytest.approx(info["mean_margin"].item(), rel=1e-5)


def test_without_penalty_the_rule_of_m2(data, monkeypatch):
    """No validation penalty, the history of M2, the best epoch and the stop of the M2 rule."""
    train, val = data

    def fail(*args):
        raise AssertionError("the validation penalty runs without penalty")

    monkeypatch.setattr(trainer_module, "validation_penalty", fail)
    torch.manual_seed(0)
    res = train_model(MLP(train), train, val, config(lr=0.05, max_epochs=40, patience=2), seed=0)
    assert all(set(h) == M2_HISTORY_KEYS for h in res["history"])
    assert res["best_feasible"] is None and res["n_feasible"] is None
    choice, stop = choose([h["val_rmse_s"] for h in res["history"]], patience=2, initial=res["initial_val_rmse_s"])
    assert (res["best_epoch"], res["epochs"], res["best_val_rmse_s"]) == (choice.best_epoch, stop, choice.best_rmse)
    assert res["stop_reason"] == "early_stopping"


@pytest.mark.parametrize("kind", ["jacobian", "linear_gain"])
def test_band_existence_term_anchors_the_equilibria(data, kind):
    """The initial network has no equilibrium inside the band; the existence term of the band edges
    gives it one at every sampled speed within a few epochs and then almost vanishes."""
    train, val = data
    torch.manual_seed(0)
    model = MLP(train)
    cfg = config(max_epochs=8, band=SMALL_BAND, penalty=PenaltyConfig(kind=kind))
    res = train_model(model, train, val, cfg, seed=0)
    history = res["history"]
    assert res["penalty_band"] is True and history[0]["penalty_existence"] > 0.1
    assert history[-1]["penalty_existence"] < 0.2 * history[0]["penalty_existence"]
    assert history[0]["penalty_n_no_equilibrium"] > 5.0 and history[-1]["penalty_n_no_equilibrium"] == 0.0


def residual_idm(events: list[Event]) -> ResidualIDM:
    """ResidualIDM whose IDM differs from the one of the data: the residual has something to learn."""
    return ResidualIDM(ModelContext(idm_params={**P, "T": 1.1, "a": 1.4}))


@GPU_ONLY
@pytest.mark.parametrize("band", [BandConfig(), SMALL_BAND], ids=["no_band", "band"])
@pytest.mark.parametrize("kind", ["jacobian", "linear_gain"])
@pytest.mark.parametrize("make", [MLP, residual_idm, WindowGRU])
def test_penalty_inside_the_cuda_graph_gives_the_eager_weights(data, make, kind, band):
    """50 full-batch steps replayed from a CUDA graph give bitwise the weights of the eager steps,
    without and with the band of the data (band scan, anchored equilibria, existence term of the
    band edges); the validation penalty after the first epoch runs between the replays."""
    train, val = data
    runs = []
    for graph in (False, True):
        torch.manual_seed(0)
        model = make(train)
        cfg = config(device="cuda", batch_size=64, max_epochs=2, steps_per_epoch=25, cuda_graph=graph,
                     band=band, penalty=PenaltyConfig(kind=kind, n_equilibria=4))  # fmt: skip
        res = train_model(model, train, val, cfg, seed=0)
        assert res["cuda_graph"] is graph and res["steps_per_epoch"] == 25 and res["epochs"] == 2
        assert res["penalty_band"] is (band is SMALL_BAND)
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((repr(history), weights(model)))
    (h_eager, w_eager), (h_graph, w_graph) = runs
    assert h_eager == h_graph
    assert all(torch.equal(w_eager[k], w_graph[k]) for k in w_eager)


COMBINED = {"jacobian_weight": 1.0, "guard": 0.05}  # D110: the Jacobian part with the guard on every step


@GPU_ONLY
@pytest.mark.parametrize(
    "make, kind, every, extra",
    [
        (MLP, "jacobian", 3, {}), (WindowGRU, "gain", 3, {}), (WindowGRU, "gain", 1, {}),
        (residual_idm, "linear_gain", 4, {}), (MLP, "existence", 1, {}), (WindowGRU, "existence", 3, {}),
        (WindowGRU, "combined", 3, COMBINED), (WindowGRU, "combined", 1, COMBINED),
        (MLP, "monotone", 1, {}), (residual_idm, "monotone", 3, {}),  # D117
    ],
)  # fmt: skip
def test_penalty_every_nth_step_from_the_graphs_gives_the_eager_weights(data, make, kind, every, extra):
    """Plain steps from the graph of the plain step, penalty steps from their own graph (the rollout
    penalty too, D90; combined: the Jacobian part in both graphs, D110): bitwise the weights of the
    fully eager training with the same ``every``."""
    train, val = data
    runs = []
    for graph in (False, True):
        torch.manual_seed(0)
        model = make(train)
        penalty = PenaltyConfig(kind=kind, n_equilibria=4, every=every, **extra)
        cfg = config(device="cuda", batch_size=64, max_epochs=2, steps_per_epoch=10, cuda_graph=graph,
                     band=SMALL_BAND, penalty=penalty)  # fmt: skip
        res = train_model(model, train, val, cfg, seed=0)
        assert res["cuda_graph"] is graph and res["penalty_steps"] == len(range(0, 20, every))
        assert res["penalty_graph"] is graph  # with every = 3: penalty steps 0, 3, 6 warm up, 9 is captured
        assert res["plain_graph"] is (graph and every > 1)  # with every = 1 every step is a penalty step
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((repr(history), weights(model)))
    (h_eager, w_eager), (h_graph, w_graph) = runs
    assert h_eager == h_graph
    assert all(torch.equal(w_eager[k], w_graph[k]) for k in w_eager)


@GPU_ONLY
def test_failed_capture_of_the_penalty_step_falls_back_to_the_eager_step(data, monkeypatch):
    """A capture that fails (here: the penalty raises while it is captured) leaves the penalty
    steps eager with a warning; nothing of the failed capture has run: the eager weights."""
    train, val = data
    real = trainer_module.stability_penalty

    def failing(model, speeds, cfg, equilibria=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("capture refused by the test")
        return real(model, speeds, cfg, equilibria)

    runs = []
    for graph in (False, True):
        if graph:
            monkeypatch.setattr(trainer_module, "stability_penalty", failing)
        torch.manual_seed(0)
        model = WindowGRU(train)
        cfg = config(device="cuda", batch_size=64, max_epochs=2, steps_per_epoch=10, cuda_graph=graph,
                     band=SMALL_BAND, penalty=PenaltyConfig(kind="gain", n_equilibria=4, every=3))  # fmt: skip
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = train_model(model, train, val, cfg, seed=0)
        assert res["penalty_graph"] is False and res["cuda_graph"] is graph
        assert any("capture refused by the test" in str(w.message) for w in caught) is graph
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((repr(history), weights(model)))
    (h_eager, w_eager), (h_graph, w_graph) = runs
    assert h_eager == h_graph
    assert all(torch.equal(w_eager[k], w_graph[k]) for k in w_eager)


def test_non_finite_loss_stops_the_run(data):
    train, val = data
    torch.manual_seed(0)
    model = Counting(train, nan_after=7)  # one call per step without rollout: NaN from epoch 2, step 2
    res = train_model(model, train, val, config(rollout=RolloutConfig(weight=0.0)), seed=0)
    assert res["stop_reason"] == "non_finite_loss" and len(res["history"]) == 1 and res["best_epoch"] == 1
    assert (res["non_finite"]["epoch"], res["non_finite"]["step"]) == (2, 2)
    assert np.isnan(res["non_finite"]["loss_acc"])
    model.nan_after = None
    assert summarise(evaluate_closed_loop(model, val, device="cpu"))["rmse_s_mean"] == res["best_val_rmse_s"]
