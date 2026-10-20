"""ResidualIDMTT (docs/m11_contract.md, D129): the certified hybrid with a slow latent time gap.

The model reduces to the E4 hybrid at the frozen latent z = 0, the online latent stays in its range and moves with
its time constant, the residual bound in (s, dv, v) is the same for every z, gradients flow through z, and the model
trains, saves and loads like any other model of the zoo."""

import math

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models import FrozenLatentHybrid, ResidualIDM, ResidualIDMTT, build_model, load_model, save_model
from cf_stability.models.base import PHYSICAL_SCALE, ModelContext
from cf_stability.models.idm import IDM_PARAM_NAMES, idm_acc, idm_equilibrium_spacing
from cf_stability.stability.equilibrium import V_GRID, find_equilibria
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.tensors import EventTensors, training_context
from cf_stability.train.trainer import RolloutConfig, TrainConfig, rollout_loss, train_model

IDM_P = {"v0": 33.0, "T": 1.2, "s0": 2.0, "a": 1.5, "b": 1.5}
CONTEXT = ModelContext(center=(30.0, 0.0, 15.0), scale=(15.0, 1.5, 6.0), idm_params=IDM_P)
LOW = torch.tensor([2.0, -5.0, 0.0], dtype=torch.float64)
SPAN = torch.tensor([98.0, 10.0, 35.0], dtype=torch.float64)


def histories(n: int, seed: int, steps: int = 30, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (LOW + SPAN * torch.rand(n, steps, 3, generator=gen, dtype=torch.float64)).to(dtype)


def perturbed(model: torch.nn.Module, scale: float, seed: int) -> torch.nn.Module:
    """Leave the antisymmetric (zero) start of the residual and the zero head of the latent."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return model


def trained_residual_idm(seed: int = 0) -> ResidualIDM:
    torch.manual_seed(seed)
    model = ResidualIDM(CONTEXT, r_max=0.3, lipschitz=0.3)
    perturbed(model.g, 0.3, seed)
    return model.eval()


def test_untrained_model_is_the_margin_core():
    torch.manual_seed(0)
    model = ResidualIDMTT(CONTEXT, r_max=0.3, tau_z=15.0).eval()
    x = histories(500, 0, dtype=torch.float32)
    last = x[:, -1].double()
    core = idm_acc(last[:, 0], last[:, 1], last[:, 2], *(IDM_P[k] for k in IDM_PARAM_NAMES))
    assert model.latent(x).abs().max() == 0.0  # zero head: z = 0, T = T_c
    assert model.residual(x).abs().max() < 1e-6  # antisymmetric start of ResidualIDM
    assert (model(x).double() - core).abs().max() < 1e-4
    assert model.window == 30 and not model.memoryless and model.config()["window"] == 30


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reduces_to_the_e4_hybrid_at_frozen_zero(dtype):
    source = trained_residual_idm().to(dtype)
    model = ResidualIDMTT.from_residual_idm(source, CONTEXT, tau_z=30.0).eval()
    perturbed(model.latent_rnn, 1.0, 1)  # the online latent no longer zero
    perturbed(model.latent_head, 1.0, 2)
    x = histories(2000, 3, dtype=dtype)
    model.frozen_z = 0.0
    assert model.window == 1 and model.memoryless
    out, ref = model(x), source(x)
    # same outputs: to the last bits in float64 (the first layer sums four products instead of three, the fourth
    # 0 * 0, in an order that depends on the BLAS of the platform), to 1e-6 in float32
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    assert (out - ref).abs().max() <= tol and ref.abs().max() > 0.1
    assert (model.residual(x) - source.residual(x)).abs().max() <= tol and source.residual(x).abs().max() > 0.01
    assert model.layer_norms() == pytest.approx(source.layer_norms(), rel=1e-12)  # [W | 0] has the singular values of W
    rel = 1e-12 if dtype == torch.float64 else 1e-6  # a float32 source holds its scaler in float32
    assert torch.allclose(model.jacobian_bound().double(), source.jacobian_bound().double(), rtol=rel, atol=0)
    assert model.core_params(0.0) == source.idm.params_dict()
    model.frozen_z = None
    assert model.window == 30 and not torch.equal(model(x), ref)  # the online latent moves T away from T_c


def test_frozen_view_is_the_frozen_model():
    model = ResidualIDMTT.from_residual_idm(trained_residual_idm(1), CONTEXT, tau_z=15.0).eval()
    x = histories(1000, 4, dtype=torch.float32)
    for z in (-1.0, -0.37, 0.0, 0.5, 1.0):
        view = model.at_z(z)
        assert isinstance(view, FrozenLatentHybrid) and isinstance(view, ResidualIDM) and view.window == 1
        model.frozen_z = z
        assert torch.allclose(view(x), model(x), rtol=1e-5, atol=1e-5)  # T(z) rounded to float32 in two ways
        assert view.idm.params_dict()["T"] == pytest.approx(IDM_P["T"] * 1.5**z, rel=1e-6)
        assert torch.equal(view.jacobian_bound(), model.jacobian_bound())
    with pytest.raises(TypeError):
        model.at_z(0.0).config()


def test_latent_stays_in_its_range_and_moves_with_the_time_constant():
    tau = 15.0
    model = ResidualIDMTT(CONTEXT, tau_z=tau, window=300).double().eval()
    c = DT / tau
    with torch.no_grad():
        model.latent_head.bias.fill_(40.0)  # tanh(g) = 1 at every state
    x = histories(4, 5, steps=300)
    path = model.latent_path(x)
    t = torch.arange(1, 301, dtype=torch.float64)
    assert torch.allclose(path, (1.0 - (1.0 - c) ** t).expand_as(path), rtol=0, atol=1e-12)
    k = round(tau / DT)  # one time constant: 1 - (1 - c)^(tau / dt), about 1 - 1/e
    assert path[0, k - 1].item() == pytest.approx(1.0 - math.exp(-1.0), abs=2e-3)
    assert torch.allclose(model.latent(x), path[:, -1], rtol=0, atol=1e-12)  # closed form = recursion
    with torch.no_grad():
        model.latent_head.bias.fill_(-40.0)
    assert torch.allclose(model.latent_path(x), -(1.0 - (1.0 - c) ** t).expand_as(path), rtol=0, atol=1e-12)
    # random weights: |z| <= 1 - (1 - c)^window < 1 for every input, and the closed form equals the recursion
    model = ResidualIDMTT(CONTEXT, tau_z=tau).double().eval()
    perturbed(model.latent_rnn, 3.0, 6)
    perturbed(model.latent_head, 30.0, 7)
    x = histories(3000, 8)
    z, path = model.latent(x), model.latent_path(x)
    assert torch.allclose(z, path[:, -1], rtol=0, atol=1e-12)
    assert path.abs().max() <= model.latent_reach() < 1.0 and z.abs().max() > 0.5 * model.latent_reach()
    assert model.latent_reach() == pytest.approx(1.0 - (1.0 - c) ** 30, rel=1e-12)
    # a shorter history (fewer updates) stays inside the reach of its length
    assert model.latent(x[:, -5:]).abs().max() <= model.latent_reach(5)


def test_time_gap_follows_the_latent():
    model = ResidualIDMTT(CONTEXT, tau_z=30.0).double().eval()
    z = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    assert torch.allclose(model.time_gap(z), IDM_P["T"] * 1.5**z, rtol=1e-6, atol=0)  # T stored in float32
    assert model.core_params(-1.0)["T"] == pytest.approx(IDM_P["T"] / 1.5, rel=1e-6)
    last = histories(200, 9)[:, -1]
    for value in (-1.0, 0.3, 1.0):
        zz = torch.full((200,), value, dtype=torch.float64)
        p = {**IDM_P, "T": model.core_params(value)["T"]}
        ref = idm_acc(last[:, 0], last[:, 1], last[:, 2], *(p[k] for k in IDM_PARAM_NAMES))
        assert torch.allclose(model.core_acc(last, zz), ref, rtol=1e-6, atol=1e-6)


def test_residual_bound_is_the_same_for_every_latent():
    torch.manual_seed(0)
    model = ResidualIDMTT(CONTEXT, r_max=0.5, lipschitz=2.0).double()
    perturbed(model.g, 0.5, 10)
    opt = torch.optim.Adam(model.g.parameters(), lr=1e-2)
    scale = torch.tensor(PHYSICAL_SCALE, dtype=torch.float64)
    for step in range(100):  # maximise the sensitivity to the scaled inputs (power iteration in training mode)
        x = histories(256, 100 + step)[:, -1].requires_grad_()
        z = (2.0 * torch.rand(256, generator=torch.Generator().manual_seed(step), dtype=torch.float64) - 1.0)
        z.requires_grad_()
        jac = torch.autograd.grad(model.residual_at(x, z).sum(), (x, z), create_graph=True)
        loss = -((jac[0] * scale) ** 2).sum(-1).mean() - (jac[1] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    bound, latent_bound = model.jacobian_bound(), model.latent_bound()
    x = histories(4000, 11)[:, -1].requires_grad_()
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        z = torch.full((4000,), value, dtype=torch.float64, requires_grad=True)
        jx, jz = torch.autograd.grad(model.residual_at(x, z).sum(), (x, z))
        assert (jx.abs() <= bound * (1 + 1e-9)).all() and (jx.abs().amax(0) > 0).all(), value
        assert (jz.abs() <= latent_bound * (1 + 1e-9)).all()
        # the same bound for a frozen view (what the certificate reads)
        assert torch.equal(model.at_z(value).jacobian_bound(), bound)


def test_gradients_flow_through_the_latent():
    model = ResidualIDMTT(CONTEXT, r_max=0.3, tau_z=15.0).double()
    perturbed(model.g, 0.3, 12)
    perturbed(model.latent_head, 0.5, 13)
    x = histories(64, 14)
    model(x).sum().backward()
    for name, p in model.named_parameters():
        if name.startswith(("g.", "latent_")):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    model.zero_grad()
    model.frozen_z = 0.2  # a frozen latent: nothing reaches the GRU
    model(x).sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.latent_rnn.parameters())


def test_checkpoint_and_factory(tmp_path):
    model = build_model({"name": "residual_idm_tt", "r_max": 0.3, "tau_z": 60.0, "latent_hidden": 8, "window": 20},
                        CONTEXT)  # fmt: skip
    assert isinstance(model, ResidualIDMTT) and model.window == 20
    perturbed(model, 0.2, 15)
    model.frozen_z = 0.5
    save_model(model, tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")
    assert isinstance(loaded, ResidualIDMTT) and loaded.frozen_z is None and loaded.config() == model.config()
    model.frozen_z = None
    model.eval()
    x = histories(300, 16, dtype=torch.float32)
    assert torch.equal(loaded(x), model(x))
    with pytest.raises(ValueError):
        build_model({"name": "residual_idm_tt", "tau_z": 0.0}, CONTEXT)


def test_equilibria_of_the_online_and_the_frozen_model():
    model = ResidualIDMTT.from_residual_idm(trained_residual_idm(2), CONTEXT, tau_z=15.0).double().eval()
    perturbed(model.latent_head, 0.5, 17)
    eq = find_equilibria(model, V_GRID)  # online: the constant window of the audit drives the latent
    assert eq.found.all()
    for z in (-0.5, 0.0, 0.5):
        model.frozen_z = z
        frozen = find_equilibria(model, V_GRID)
        view = find_equilibria(model.at_z(z), V_GRID)
        assert torch.allclose(frozen.s, view.s, rtol=1e-9, atol=1e-9)
    model.frozen_z = 0.5  # larger T: larger equilibrium spacing (the residual is small)
    s_hi = find_equilibria(model, V_GRID).s
    model.frozen_z = -0.5
    assert (s_hi > find_equilibria(model, V_GRID).s).all()


# ------------------------------------------------------------------------------------------ training


P_TOY = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}


def make_event(i: int, rng: np.random.Generator, n: int = 150) -> Event:
    t = DT * np.arange(n)
    base, amp, period = rng.uniform(8.0, 20.0), rng.uniform(1.0, 4.0), rng.uniform(6.0, 14.0)
    v_lead = base + amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    x_lead = np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    v0 = v_lead[0] + rng.uniform(-2.0, 2.0)
    s0 = idm_equilibrium_spacing(torch.tensor(v0), P_TOY["v0"], P_TOY["T"], P_TOY["s0"]).item() + rng.uniform(-3, 5)
    T = P_TOY["T"] * rng.uniform(0.8, 1.25)  # drivers with their own time gap
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, **{**P_TOY, "T": T}), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(s0), torch.tensor(v0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"tr/a/{i}|L|0", dataset="tr", site="a", follower_id=f"tr/a/{i}", leader_id=f"tr/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


def test_trains_with_the_rollout_loss_and_keeps_its_budget():
    rng = np.random.default_rng(0)
    events = [make_event(i, rng) for i in range(24)]
    train, val = events[:18], events[18:]
    stats = training_context(train, 0)
    context = ModelContext(center=stats["center"], scale=stats["scale"], idm_params=P_TOY)
    torch.manual_seed(0)
    model = ResidualIDMTT(context, r_max=0.3, lipschitz=0.2, tau_z=15.0)
    perturbed(model.latent_head, 0.1, 18)
    tensors = EventTensors(train)
    ends = tensors.window_ends(model.window)
    starts = ends[torch.randint(len(ends), (32,), generator=torch.Generator().manual_seed(0))]
    rollout_loss(model, tensors, starts, horizon=20, var_s=stats["var_s"], var_v=stats["var_v"]).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    model.zero_grad()
    bound = model.jacobian_bound().clone()
    cfg = TrainConfig(lr=3e-3, batch_size=256, max_epochs=3, steps_per_epoch=5, patience=100, acc_weight=0.0,
                      rollout=RolloutConfig(weight=1.0, horizon=30, segments=64), device="cpu")  # fmt: skip
    res = train_model(model, train, val, cfg, seed=0)
    assert res["epochs"] == 3 and res["history"][-1]["loss_rollout"] <= res["history"][0]["loss_rollout"]
    assert not model.training and model.idm.params_dict() == pytest.approx(P_TOY, rel=1e-6)  # the core is fixed
    assert torch.allclose(model.jacobian_bound().double(), bound.double(), rtol=0.05)  # spectral norm: norms near 1
