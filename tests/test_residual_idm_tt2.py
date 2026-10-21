"""ResidualIDMTT2 (docs/m12_contract.md, D135): the two-timescale certified hybrid with the latent carried along the
vehicle, and the teacher-forced pass of the trainer.

The latent target lies in [-0.2, 1], the untrained model is the margin core, ``state=None`` is the model of D129 (the
window-restarted latent) bit for bit, the persistent latent of a constant input converges to its target with the time
constant tau_z (not to the window bound 0.182), the trainer's pass gives at every rollout start the latent of the
sequential calls from the first window of the event (gradients kept), and the model trains, saves and loads like any
other model of the zoo."""

import importlib.util
import math

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT
from cf_stability.models import FrozenLatentHybrid, ResidualIDMTT, ResidualIDMTT2, build_model, load_model, save_model
from cf_stability.models.base import SteadyStateView, steady_view
from cf_stability.models.idm import IDM_PARAM_NAMES, idm_acc
from cf_stability.models.residual_idm_tt2 import LATENT_SWEEP_TT2
from cf_stability.stability.certificate import enforce_budget, latent_interval, latent_sweep, reachable_latent
from cf_stability.stability.equilibrium import V_GRID, find_equilibria
from cf_stability.train.closed_loop import rollout_model
from cf_stability.train.evaluate import evaluate_closed_loop, open_loop_rmse
from cf_stability.train.stateful import (
    pass_capacity,
    pass_plan,
    pass_size,
    plan_tensors,
    sequential_states,
    states_at_starts,
    teacher_forced_path,
)
from cf_stability.train.tensors import EventTensors, training_context
from cf_stability.train.trainer import RolloutConfig, TrainConfig, rollout_loss, train_model
from cf_stability.utils import REPO_ROOT

_spec = importlib.util.spec_from_file_location("tt_tests", REPO_ROOT / "tests" / "test_residual_idm_tt.py")
tt_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt_tests)
CONTEXT, IDM_P, histories, perturbed, make_event = (tt_tests.CONTEXT, tt_tests.IDM_P, tt_tests.histories,
                                                    tt_tests.perturbed, tt_tests.make_event)  # fmt: skip
STORED_TT = REPO_ROOT / "runs" / "e4_tt" / "follownet_highd" / "residual_idm_tt" / "driver_fold0_seed0"


def moving(model: ResidualIDMTT2, seed: int = 1) -> ResidualIDMTT2:
    """Random weights of the latent's GRU and head: the target depends on the window."""
    perturbed(model.latent_rnn, 1.0, seed)
    perturbed(model.latent_head, 1.0, seed + 1)
    return model


class TanhTarget(ResidualIDMTT2):
    """The persistent model with the target of D129 (tanh): its ``state=None`` path must be the model of D129."""

    def latent_target(self, g):
        return torch.tanh(g)


# ----------------------------------------------------------------------------------------------- the model


def test_untrained_model_is_the_margin_core_and_its_range():
    torch.manual_seed(0)
    model = ResidualIDMTT2(CONTEXT, r_max=0.3, tau_z=15.0).eval()
    x = histories(500, 0, dtype=torch.float32)
    last = x[:, -1].double()
    core = idm_acc(last[:, 0], last[:, 1], last[:, 2], *(IDM_P[k] for k in IDM_PARAM_NAMES))
    assert model.window_drive(x).abs().max() < 1e-7  # sigmoid(ln 0.2) = 1/6: z_target = -0.2 + 1.2 / 6 = 0
    assert model.latent(x).abs().max() < 1e-7 and model.residual(x).abs().max() < 1e-6
    assert (model(x).double() - core).abs().max() < 1e-4
    acc, z = model(x, model.initial_state(500))
    assert (acc.double() - core).abs().max() < 1e-4 and z.abs().max() < 1e-9
    assert model.stateful and model.LATENT_SWEEP == LATENT_SWEEP_TT2 and len(LATENT_SWEEP_TT2) == 21
    assert LATENT_SWEEP_TT2[0] == -0.2 and LATENT_SWEEP_TT2[-1] == 1.0 and LATENT_SWEEP_TT2[1] == pytest.approx(-0.14)
    assert model.latent_range() == (-0.2, 1.0) and model.latent_reach() == 1.0
    assert model.config()["z_lo"] == -0.2 and model.config()["z_hi"] == 1.0 and model.name == "residual_idm_tt2"
    assert model.time_gap(torch.tensor(-0.2)).item() == pytest.approx(IDM_P["T"] * 1.5**-0.2, rel=1e-6)  # 0.922 T_c
    model.frozen_z = 0.3
    assert not model.stateful and model.steady_state(x) is None and model.window == 1
    with pytest.raises(ValueError):
        ResidualIDMTT2(CONTEXT, z_lo=0.1, z_hi=1.0)  # z = 0 must lie inside the range


def test_target_stays_in_the_range():
    model = moving(ResidualIDMTT2(CONTEXT, tau_z=15.0).double().eval(), 3)
    perturbed(model.latent_head, 30.0, 4)  # saturate the sigmoid
    u = model.drive(histories(3000, 5))
    assert u.min() >= -0.2 and u.max() <= 1.0 and u.min() < -0.15 and u.max() > 0.95


@pytest.mark.skipif(not (STORED_TT / "model.pt").exists(), reason="the stored e4_tt run of M11 is not here")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_none_is_the_m11_model_bit_for_bit_on_a_stored_e4_tt_run(dtype):
    """The stored e4_tt run of M11 (fold 0) in the persistent class with the target of D129: without a state the
    accelerations of single calls and of closed-loop rollouts are those of the M11 model, bit for bit."""
    m11 = load_model(STORED_TT / "model.pt").to(dtype)
    assert type(m11) is ResidualIDMTT and not m11.stateful
    checkpoint = torch.load(STORED_TT / "model.pt", map_location="cpu", weights_only=True)
    config = {k: v for k, v in checkpoint["config"].items() if k != "name"}
    model = TanhTarget(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(dtype)
    x = histories(4000, 6, dtype=dtype)
    assert torch.equal(model(x), m11(x)) and torch.equal(model.latent(x), m11.latent(x))
    gen = torch.Generator().manual_seed(7)
    v_lead = (15.0 + 3.0 * torch.sin(0.3 * DT * torch.arange(200) + torch.rand(64, 1, generator=gen))).to(dtype)
    x_lead = DT * torch.cumsum(v_lead, -1) + 40.0
    s_hist, v_hist = torch.full((64, 30), 30.0, dtype=dtype), v_lead[:, :30].clone()
    a = rollout_model(model, x_lead, v_lead, s_hist, v_hist)
    b = rollout_model(m11, x_lead, v_lead, s_hist, v_hist)
    assert a.state is None and all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("s", "v", "x", "a"))
    # the persistent model of D135 itself, without a state: the latent of the window from z = 0, with its target
    tt2 = ResidualIDMTT2(**config)
    tt2.load_state_dict(checkpoint["state_dict"])
    tt2 = tt2.eval().to(dtype)
    u = tt2.drive(x)
    c = tt2.rate
    weights = c * (1.0 - c) ** torch.arange(29, -1, -1, dtype=torch.float64)
    assert torch.equal(tt2.latent(x), u @ weights.to(u.dtype)) and not torch.equal(tt2(x), m11(x))


def test_persistent_latent_of_a_constant_input_converges_to_the_target_with_tau_z():
    tau = 15.0
    model = moving(ResidualIDMTT2(CONTEXT, r_max=0.3, lipschitz=0.3, tau_z=tau).double().eval(), 7)
    with torch.no_grad():
        model.latent_head.bias.add_(2.0)  # targets well above the window bound
    x = histories(1, 8)[:, -1:].expand(1, 30, 3).repeat(4, 1, 1)  # one constant window, four copies
    x[1:, :, 2] += torch.tensor([[2.0], [5.0], [9.0]])  # other speeds: other targets
    target = model.steady_state(x)
    assert torch.equal(target, model.window_drive(x)) and target.min() > 0.3
    c = DT / tau
    z, path = model.initial_state(4), []
    for _ in range(1500):  # 150 s = 10 tau_z
        _, z = model(x, z)
        path.append(z)
    path = torch.stack(path, dim=1)
    k = torch.arange(1, 1501, dtype=torch.float64)
    assert torch.allclose(path, target[:, None] * (1.0 - (1.0 - c) ** k), rtol=0, atol=1e-12)
    one_tau = round(tau / DT)  # 1 - (1 - c)^(tau / dt) = 1 - 1/e to 2e-3
    assert torch.allclose(path[:, one_tau - 1] / target, torch.full((4,), 1.0 - math.exp(-1.0), dtype=torch.float64), atol=2e-3)
    assert torch.allclose(path[:, -1], target, rtol=0, atol=1e-4)  # converged to the target ...
    window = model.latent(x)  # ... while the window-restarted latent of D129 stays below 1 - (1 - c)^30 = 0.182
    weights = c * (1.0 - c) ** torch.arange(29, -1, -1, dtype=torch.float64)
    assert torch.allclose(window, model.drive(x) @ weights, rtol=0, atol=1e-12)
    assert (window.abs() <= model.latent_reach(30)).all() and model.latent_reach(30) == pytest.approx(0.182, abs=1e-3)
    assert (path[:, -1] > 0.182).all() and (path[:, -1] > 4.0 * window).all()
    # the steady state is the fixed point: a call at it returns it unchanged and the acceleration of the steady view
    acc, z_next = model(x, target)
    assert torch.equal(z_next, target) and torch.equal(steady_view(model)(x), acc)
    # a random walk of inputs: the persistent latent stays in [z_lo, z_hi]
    z = model.initial_state(500)
    for step in range(300):
        _, z = model(histories(500, 100 + step), z)
        assert z.min() >= -0.2 and z.max() <= 1.0


def test_carry_is_the_recursion():
    model = ResidualIDMTT2(CONTEXT, tau_z=3.0).double()
    gen = torch.Generator().manual_seed(9)
    drives = -0.2 + 1.2 * torch.rand(6, 40, generator=gen, dtype=torch.float64)
    z0 = torch.rand(6, generator=gen, dtype=torch.float64)
    counts = torch.tensor([0, 1, 7, 39, 40, 13])
    z, path = z0.clone(), [z0.clone()]
    for k in range(40):
        z = model.advance(z, drives[:, k])
        path.append(z)
    path = torch.stack(path, dim=1)  # [6, 41]: the latent after 0 .. 40 steps
    assert torch.allclose(model.carry(drives, counts, z0), path[torch.arange(6), counts], rtol=0, atol=1e-14)
    assert torch.allclose(model.carry_path(drives, z0), path[:, 1:], rtol=0, atol=1e-14)


def test_gradients_flow_through_the_persistent_latent():
    model = moving(ResidualIDMTT2(CONTEXT, r_max=0.3, lipschitz=0.3, tau_z=1.0).double(), 10)
    perturbed(model.g, 0.3, 11)
    z = model.initial_state(64)
    for step in range(5):
        acc, z = model(histories(64, 20 + step), z)
    (acc.sum() + z.sum()).backward()
    for name, p in model.named_parameters():
        if name.startswith(("g.", "latent_")):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


STABLE_CORE = {"v0": 33.0, "T": 1.5, "s0": 2.0, "a": 2.0, "b": 1.0}  # string stable on the whole grid at T / 1.5^0.2


def test_frozen_latent_and_certificate_sweep():
    model = moving(ResidualIDMTT2.from_residual_idm(tt_tests.trained_residual_idm(1), CONTEXT, tau_z=15.0).eval(), 12)
    x = histories(300, 13, dtype=torch.float32)
    for z in (-0.2, 0.0, 0.64, 1.0):
        model.frozen_z = z
        view = model.at_z(z)
        assert isinstance(view, FrozenLatentHybrid) and torch.allclose(view(x), model(x), rtol=1e-5, atol=1e-5)
        acc, state = model(x, torch.zeros(300))  # a frozen latent ignores the state
        assert torch.equal(acc, model(x)) and torch.equal(state, torch.zeros(300))
    model.frozen_z = None
    assert latent_sweep(model) == LATENT_SWEEP_TT2 and latent_interval(model) == (-0.2, 1.0)
    assert reachable_latent(model) == LATENT_SWEEP_TT2  # the budget at the 21 values of the sweep
    m11 = ResidualIDMTT(CONTEXT, tau_z=15.0)
    assert latent_sweep(m11)[0] == -1.0 and len(latent_sweep(m11)) == 21
    assert latent_interval(m11) == (-m11.latent_reach(), m11.latent_reach())
    stable = ResidualIDMTT2(tt_tests.ModelContext(idm_params=STABLE_CORE), r_max=0.3, tau_z=15.0)
    budget = enforce_budget(stable, 0.9, v=(5.0, 15.0, 25.0), n_scan=100)
    assert budget["latent"]["z"] == list(LATENT_SWEEP_TT2) and budget["latent"]["range"] == [-0.2, 1.0]
    assert budget["admissible"] == pytest.approx(min(budget["latent"]["admissible_per_z"]))


def test_checkpoint_and_factory(tmp_path):
    model = build_model({"name": "residual_idm_tt2", "r_max": 0.3, "tau_z": 60.0, "latent_hidden": 8, "z_hi": 0.8},
                        CONTEXT)  # fmt: skip
    assert isinstance(model, ResidualIDMTT2) and model.LATENT_SWEEP[-1] == 0.8
    perturbed(model, 0.2, 15)
    save_model(model, tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")
    assert isinstance(loaded, ResidualIDMTT2) and loaded.config() == model.config() and loaded.stateful
    model.eval()
    x = histories(300, 16, dtype=torch.float32)
    z = torch.rand(300)
    assert torch.equal(loaded(x), model(x))
    assert all(torch.equal(a, b) for a, b in zip(loaded(x, z), model(x, z)))


def test_equilibria_of_the_steady_view():
    """The steady-state view (the latent at the target of the constant window) is the law whose zeros are the
    equilibria of the persistent model: at them the model with its steady latent has zero acceleration."""
    model = moving(ResidualIDMTT2.from_residual_idm(tt_tests.trained_residual_idm(2), CONTEXT, tau_z=15.0), 17)
    model = model.double().eval()
    view = steady_view(model)
    assert isinstance(view, SteadyStateView) and view.window == 30 and steady_view(ResidualIDMTT(CONTEXT)) is not view
    eq = find_equilibria(view, V_GRID)
    assert eq.found.all()
    hist = torch.stack((eq.s, torch.zeros_like(eq.s), eq.v), dim=-1)[:, None, :].expand(-1, 30, -1)
    acc, z = model(hist, model.steady_state(hist))
    assert acc.abs().max() < 1e-6 and torch.equal(z, model.steady_state(hist))
    window = find_equilibria(model, V_GRID)  # the window-restarted latent of D129 has other equilibria
    assert not torch.allclose(window.s, eq.s)


# --------------------------------------------------------------------------------------- the trainer's pass


@pytest.fixture(scope="module")
def events():
    rng = np.random.default_rng(0)
    return [make_event(i, rng, n=60 + 25 * i) for i in range(6)]


def test_pass_plan_shares_the_events_and_pads():
    starts = np.array([[4, 40], [1, 29], [4, 70], [2, 35], [4, 55]])
    plan = pass_plan(starts, 30, n_max=80, capacity=96)
    assert int(plan["needed"]) == pass_size(starts, 30) == 0 + 6 + 41  # event 1: none, 2: 29..34, 4: 29..69
    assert plan["slot"].tolist() == [2, 0, 2, 1, 2] and plan["count"].tolist() == [11, 0, 41, 6, 26]
    real = plan["flat_index"][:47]
    assert real[:6].tolist() == [[2, t] for t in range(29, 35)] and real[6:].tolist() == [[4, t] for t in range(29, 70)]
    assert plan["flat_pos"][:6].tolist() == [80 + p for p in range(6)]  # slot 1
    assert plan["flat_pos"][6:47].tolist() == [160 + p for p in range(41)]  # slot 2
    pad = plan["flat_pos"][47:]
    assert len(pad) == 49 and len(set(pad.tolist())) == 49 and pad.min() >= 5 * 80  # behind the rows, distinct
    with pytest.raises(ValueError):
        pass_plan(starts, 30, n_max=80, capacity=40)
    ends = np.array([[e, t] for e in range(3) for t in range(29, 100)])
    cap = pass_capacity(ends, 30, 8, 71, seed=0)  # at most every rollout in an event of its own, at its end
    assert cap == 8 * 71 and cap == pass_capacity(ends, 30, 8, 71, seed=0)
    ends = np.array([[e, t] for e in range(400) for t in range(29, 100)])
    cap = pass_capacity(ends, 30, 64, 71, seed=0)  # 1.25 x the largest of 256 random batches, rounded up to 4096
    assert cap % 4096 == 0 and cap < 64 * 71


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_pass_gives_the_sequential_latent_at_every_rollout_start(events, dtype):
    """The hand-made events: at every rollout start (several in one event, one at the first window, one at the end) the
    state of the batched pass equals the state of the sequential calls from the first window of the event."""
    model = moving(ResidualIDMTT2(CONTEXT, r_max=0.3, lipschitz=0.3, tau_z=1.0), 18).to(dtype).eval()
    data = EventTensors(events, dtype=dtype)
    starts = torch.tensor([[0, 29], [0, 45], [2, 70], [2, 99], [5, 184], [1, 30], [3, 100], [2, 29], [5, 60]])
    n_max = data.mask.shape[1] - 30 + 1
    plan = plan_tensors(pass_plan(starts.numpy(), 30, n_max, capacity=4096), torch.device("cpu"))
    batched = states_at_starts(model, data, plan, n_max)
    sequential = torch.cat([s.reshape(1) for s in sequential_states(model, data, starts)])
    tol = 1e-12 if dtype == torch.float64 else 2e-5  # float32: the GRU of one window and of a batch round apart
    assert torch.allclose(batched, sequential, rtol=0, atol=tol) and sequential.abs().max() > 0.05
    assert batched[0] == 0.0 and batched[7] == 0.0  # a start at the first window: the initial state
    # the rollout's first call (the observed window that ends at j) advances it once more: the latent of the pass at j
    window = data.state_windows(starts[[1]], 30)
    _, z_j = model(window, batched[[1]])
    plan_j = plan_tensors(pass_plan(np.array([[0, 46]]), 30, n_max, capacity=4096), torch.device("cpu"))
    assert torch.allclose(z_j, states_at_starts(model, data, plan_j, n_max), rtol=0, atol=tol)
    # the teacher-forced path of the evaluation: the same states
    ends, before, after = teacher_forced_path(model, data, max_windows=150)
    for (e, j), z in zip(starts.tolist(), sequential.tolist()):
        k = int(((ends[:, 0] == e) & (ends[:, 1] == j)).nonzero())
        assert before[k].item() == pytest.approx(z, abs=tol) and (j == 29 or after[k - 1].item() == pytest.approx(z, abs=tol))


def test_pass_keeps_the_gradients(events):
    model = moving(ResidualIDMTT2(CONTEXT, r_max=0.3, lipschitz=0.3, tau_z=0.5).double(), 19)
    data = EventTensors(events, dtype=torch.float64)
    starts = torch.tensor([[3, 80], [4, 120], [5, 150], [2, 60]])
    n_max = data.mask.shape[1] - 30 + 1
    plan = plan_tensors(pass_plan(starts.numpy(), 30, n_max, capacity=4096), torch.device("cpu"))
    grads = []
    for detach in (False, True):
        model.zero_grad()
        state = states_at_starts(model, data, plan, n_max)
        state = state.detach() if detach else state
        rollout_loss(model, data, starts, horizon=20, var_s=10.0, var_v=1.0, state=state).backward()
        grads.append(model.latent_head.weight.grad.clone())
    assert grads[0].abs().sum() > 0 and not torch.allclose(grads[0], grads[1])  # the pass contributes to the gradient


def test_trains_with_the_pass_and_evaluates_from_the_first_window(events):
    rng = np.random.default_rng(1)
    train_events = [make_event(i, rng, n=150 + 10 * (i % 5)) for i in range(18)]
    val = [make_event(100 + i, rng) for i in range(6)]
    stats = training_context(train_events, 0)
    context = tt_tests.ModelContext(center=stats["center"], scale=stats["scale"], idm_params=tt_tests.P_TOY)
    torch.manual_seed(0)
    model = moving(ResidualIDMTT2(context, r_max=0.3, lipschitz=0.2, tau_z=2.0), 20)
    cfg = TrainConfig(lr=3e-3, batch_size=256, max_epochs=2, steps_per_epoch=4, patience=100, acc_weight=0.0,
                      rollout=RolloutConfig(weight=1.0, horizon=20, segments=16), device="cpu")  # fmt: skip
    res = train_model(model, train_events, val, cfg, seed=0)
    assert res["epochs"] == 2 and res["state_pass"]["overflow_steps"] == 0
    assert res["state_pass"]["n_max"] == max(len(e) for e in train_events) - 30 + 1  # 190 - 29
    assert res["state_pass"]["capacity"] == 16 * res["state_pass"]["n_max"]  # tiny data: the bound
    assert all(math.isfinite(h["loss_rollout"]) for h in res["history"])
    # the closed-loop evaluation: rollouts from z = 0 at the first window of every event, the state carried
    frame = evaluate_closed_loop(model, val, warmup=30)
    data = EventTensors(val, dtype=torch.float32)
    with torch.no_grad():
        ref = rollout_model(model, data.x_lead, data.v_lead, data.s[:, :30], data.v[:, :30], state=model.initial_state(6))
        window = rollout_model(model, data.x_lead, data.v_lead, data.s[:, :30], data.v[:, :30])
    err = ((ref.s - data.s) ** 2)[:, 29:].mean(-1).sqrt().double().numpy()
    assert np.allclose(frame["rmse_s"].to_numpy(), err, rtol=1e-5)
    assert not torch.equal(ref.s, window.s)  # without the state: the window-restarted latent
    # the open-loop error with the teacher-forced latent
    assert math.isfinite(open_loop_rmse(model, val, 30))


def test_training_without_a_stateful_model_is_unchanged(events):
    """The trainer's new branch is off for a stateless model: no state_pass in its result."""
    rng = np.random.default_rng(2)
    train_events = [make_event(i, rng) for i in range(12)]
    model = ResidualIDMTT(CONTEXT, r_max=0.3, lipschitz=0.2, tau_z=15.0)
    cfg = TrainConfig(batch_size=256, max_epochs=1, steps_per_epoch=2, acc_weight=0.0,
                      rollout=RolloutConfig(horizon=10, segments=8), device="cpu")  # fmt: skip
    res = train_model(model, train_events[:9], train_events[9:], cfg, seed=0)
    assert "state_pass" not in res and not model.stateful


@pytest.mark.skipif(not torch.cuda.is_available() or not __import__("os").environ.get("CF_GPU_TESTS"),
                    reason="GPU test: set CF_GPU_TESTS=1")  # fmt: skip
def test_the_pass_inside_the_cuda_graph_gives_the_eager_weights():
    """The fixed-capacity pass replayed from the CUDA graph of the training step: the weights of the eager steps."""
    rng = np.random.default_rng(3)
    train_events = [make_event(i, rng, n=150 + 20 * (i % 4)) for i in range(20)]
    val = [make_event(100 + i, rng) for i in range(4)]
    runs = []
    for graph in (False, True):
        torch.manual_seed(0)
        model = moving(ResidualIDMTT2(CONTEXT, r_max=0.3, lipschitz=0.2, tau_z=2.0), 21)
        cfg = TrainConfig(batch_size=64, max_epochs=2, steps_per_epoch=8, acc_weight=0.0, cuda_graph=graph,
                          rollout=RolloutConfig(horizon=15, segments=16), device="cuda")  # fmt: skip
        res = train_model(model, train_events, val, cfg, seed=0)
        assert res["plain_graph"] is graph
        runs.append(({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]))  # fmt: skip
    (w_eager, h_eager), (w_graph, h_graph) = runs
    assert repr(h_eager) == repr(h_graph) and all(torch.equal(w_eager[k], w_graph[k]) for k in w_eager)
