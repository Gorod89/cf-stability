"""Model zoo: registry, common interface, checkpoints, bounded IDM, fits, PIDL and PERL.

Newell's and OVM's own tests are in test_newell.py and test_ovm.py."""

import copy
import dataclasses
from functools import partial

import numpy as np
import pytest
import torch
import yaml

import cf_stability.train.calibration as calibration
from cf_stability.data.schema import DT, Event
from cf_stability.models import MODELS, IDM, OVM, Newell, build_model, load_model, save_model
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import IDM_BOUNDS, IDM_PARAM_NAMES, idm_acc
from cf_stability.models.newell import newell_acc
from cf_stability.models.ovm import OVM_PARAM_NAMES
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.utils import REPO_ROOT

IDM_P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
NEWELL_P = {"w": 20.0}
OVM_P = {"v0": 28.0, "tau": 1.2, "s0": 2.5}
CONTEXT = ModelContext(
    center=(30.0, 0.0, 15.0), scale=(15.0, 1.5, 6.0), box_low=(2.0, -3.0, 0.0), box_high=(80.0, 3.0, 30.0),
    idm_params=IDM_P, newell_params=NEWELL_P, ovm_params=OVM_P, seed=0,
)  # fmt: skip
N_TRAINABLE = {
    "idm": 0, "newell": 0, "ovm": 0, "persistence": 0, "knn": 0, "mlp": 4481, "gru": 13313, "lstm": 17729,
    "pidl": 4481, "perl": 17729, "residual_idm": 4481,
}  # fmt: skip
DIFFERENTIABLE = ("mlp", "gru", "lstm", "pidl", "perl", "residual_idm", "idm", "newell", "ovm")
DATA_SCALED = ("mlp", "gru", "lstm", "pidl")
LOW = torch.tensor([3.0, -3.0, 0.0], dtype=torch.float64)
SPAN = torch.tensor([77.0, 6.0, 30.0], dtype=torch.float64)


def model_cfg(name: str) -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "model" / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg.pop("train_overrides", None)  # read by scripts/train.py, not a constructor argument
    return cfg


def history(steps: int, batch: int = 6, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (LOW + SPAN * torch.rand(batch, steps, 3, generator=gen, dtype=torch.float64)).to(dtype)


def idm_event(i: int, n: int = 160) -> Event:
    """IDM follower behind an oscillating leader."""
    t = DT * np.arange(n)
    v_lead = 14.0 + 4.0 * np.sin(2 * np.pi * t / 9.0 + i)
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    p = [IDM_P[k] for k in IDM_PARAM_NAMES]
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, *p), torch.tensor(x_lead), torch.tensor(v_lead), torch.tensor(25.0),
        torch.tensor(14.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"syn/a/{i}|L{i}|0", dataset="syn", site="a", follower_id=f"syn/a/{i}", leader_id=f"syn/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


EVENTS = [idm_event(i) for i in range(3)]


def make(name: str, context: ModelContext = CONTEXT):
    """Model in eval mode (in training mode every call runs a power iteration of the spectral norms)."""
    torch.manual_seed(0)
    model = build_model(model_cfg(name), context)
    if name == "knn":
        model.fit(EVENTS, context)
    return model.eval()


def n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.fixture()
def quick_calibration(monkeypatch):
    """The fits use the default CalibrationConfig: keep it short and on the CPU."""
    monkeypatch.setattr(
        calibration, "CalibrationConfig", partial(calibration.CalibrationConfig, device="cpu", maxiter=3, restarts=1)
    )


# ---------------------------------------------------------------------------- registry
def test_registry_and_configs():
    files = {p.stem for p in (REPO_ROOT / "configs" / "model").glob("*.yaml")}
    assert set(MODELS) == files == set(N_TRAINABLE)
    for name in MODELS:
        cfg = model_cfg(name)
        model = build_model(cfg, CONTEXT)
        assert model.name == cfg["name"] == name
        assert cfg.items() <= model.config().items()  # the yaml holds the defaults
    with pytest.raises(ValueError, match="unknown keys"):
        build_model({**model_cfg("mlp"), "hidden": 3}, CONTEXT)
    with pytest.raises(ValueError, match="unknown keys"):
        build_model({"name": "gru", "cell": "lstm"}, CONTEXT)
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "transformer"}, CONTEXT)


# ---------------------------------------------------------------------------- common interface
@pytest.mark.parametrize("name", sorted(MODELS))
def test_common_interface(name):
    model = make(name)
    w = model.window
    assert model.memoryless == (w == 1)
    assert model.trainable == (N_TRAINABLE[name] > 0) and n_trainable(model) == N_TRAINABLE[name]
    x = history(w + 5)
    out = model(x)
    assert out.shape == (6,) and torch.isfinite(out).all()
    assert torch.equal(model(x[:, -w:]), out)
    changed = x.clone()
    changed[:, :5] = history(5, seed=1)
    assert torch.equal(model(changed), out)  # only the last `window` steps count
    if w > 1:  # the earlier steps of the window count too
        changed = x.clone()
        changed[:, -w:-1] = history(w - 1, seed=2)
        assert not torch.equal(model(changed), out)
    out32, out64 = model(x.float()), model(x)
    assert torch.isfinite(out32).all() and torch.allclose(out32.double(), out64.double(), rtol=1e-4, atol=1e-4)
    double = copy.deepcopy(model).double()
    assert double(x).dtype == torch.float64 and torch.allclose(double(x), out64.double(), rtol=1e-4, atol=1e-4)
    short = model(x[:, -1:])  # a history shorter than the window is used as it is
    assert short.shape == (6,) and torch.isfinite(short).all()
    if model.memoryless:
        s, dv, v = history(1, batch=6, seed=3)[:, 0].reshape(2, 3, 3).unbind(-1)
        acc = model.acc(s, dv, v)
        assert acc.shape == (2, 3)
        assert torch.equal(acc, model(torch.stack((s, dv, v), -1).reshape(-1, 1, 3)).reshape(2, 3))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_checkpoint_round_trip(name, tmp_path):
    model = make(name)
    with torch.no_grad():  # move away from the initial values (zero heads, antisymmetric start)
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = history(model.window + 2)
    out = model(x)
    save_model(model, tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")
    assert type(loaded) is type(model) and loaded.config() == model.config() and not loaded.training
    assert torch.equal(loaded(x), out)
    rebuilt = build_model(model.config(), CONTEXT)
    rebuilt.load_state_dict(model.state_dict())
    assert torch.equal(rebuilt.eval()(x), out)


@pytest.mark.parametrize("name", DATA_SCALED)
def test_scaler_is_part_of_the_model(name):
    other = dataclasses.replace(CONTEXT, center=(10.0, 1.0, 5.0), scale=(5.0, 3.0, 2.0))
    model, shifted = make(name), make(name, other)
    shifted.load_state_dict({k: v for k, v in model.state_dict().items() if "scaler" not in k}, strict=False)
    x = history(model.window + 1)
    c, s = (torch.tensor(v, dtype=torch.float64) for v in (CONTEXT.center, CONTEXT.scale))
    c_o, s_o = (torch.tensor(v, dtype=torch.float64) for v in (other.center, other.scale))
    assert torch.allclose(shifted(c_o + (x - c) / s * s_o), model(x), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(shifted(x), model(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", DIFFERENTIABLE)
def test_gradients_reach_the_raw_inputs(name):
    model = make(name)
    x = history(model.window + 2, dtype=torch.float32).requires_grad_()
    model(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[:, -1].abs().sum() > 0
    assert (x.grad[:, :2] == 0).all()


# ---------------------------------------------------------------------------- IDM
def test_bounded_idm_stays_in_bounds():
    model = IDM(IDM_P, learnable=True, bounded=True)
    assert model.trainable and model.config()["bounded"]
    lo, hi = (torch.tensor([IDM_BOUNDS[k][j] for k in IDM_PARAM_NAMES], dtype=torch.float64) for j in (0, 1))
    direction = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=50.0)
    for _ in range(20):
        opt.zero_grad()
        (model.theta * direction).sum().backward()
        opt.step()
        theta = model.theta.detach()
        assert torch.isfinite(theta).all() and (theta >= lo).all() and (theta <= hi).all()
    assert torch.equal(theta, torch.where(direction > 0, lo, hi))
    assert torch.isfinite(model(history(1))).all()


def test_bounded_and_plain_idm_agree():
    plain, bounded = IDM(IDM_P), IDM(IDM_P, bounded=True)
    assert not plain.trainable and not bounded.raw.requires_grad
    assert bounded.params_dict() == pytest.approx(IDM_P, rel=1e-12)
    x = history(1, batch=50)
    assert torch.allclose(bounded(x), plain(x), rtol=1e-12, atol=1e-12)
    edge = IDM({**IDM_P, "v0": 45.0, "T": 0.3}, bounded=True)  # on the bounds: moved a hair inside
    assert torch.isfinite(edge.raw).all()
    assert edge.params_dict()["v0"] == pytest.approx(45.0, abs=1e-4) and edge.params_dict()["T"] > 0.3


def test_fits_set_the_calibrated_parameters(quick_calibration):
    empty = ModelContext()
    for model in (IDM(context=empty), IDM(context=empty, learnable=True, bounded=True)):
        assert torch.isnan(model.theta).all()  # not calibrated yet
        summary = model.fit(EVENTS, empty)
        assert summary["n_events"] == 3 and set(summary["params"]) == set(IDM_PARAM_NAMES)
        assert model.params_dict() == pytest.approx(summary["params"], rel=1e-9)
    ovm = OVM(empty)
    summary = ovm.fit(EVENTS, empty)
    assert set(summary["params"]) == set(OVM_PARAM_NAMES) and summary["n_events"] == 3
    assert ovm.params_dict() == pytest.approx(summary["params"], rel=1e-12)
    assert ovm.fit(EVENTS, CONTEXT) == {"params": OVM_P, "source": "context"} and ovm.params_dict() == OVM_P
    newell = Newell(empty)
    summary = newell.fit(EVENTS, empty)
    assert set(summary["params"]) == {"w"} and summary["n_events"] == 3 and len(summary["grid"]["w"]) == 40
    assert newell.params_dict() == pytest.approx(summary["params"], rel=1e-12)
    perl = build_model(model_cfg("perl"), empty)
    assert perl.fit(EVENTS, empty)["params"] == pytest.approx(summary["params"], rel=1e-12)  # same calibration
    assert perl.newell_theta.tolist() == pytest.approx([summary["params"]["w"]], rel=1e-6)
    assert perl.fit(EVENTS, CONTEXT) == {"params": NEWELL_P, "source": "context"}
    assert perl.newell_theta.tolist() == pytest.approx([NEWELL_P["w"]], rel=1e-6)


# ---------------------------------------------------------------------------- PIDL
class Recorder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return x.new_zeros(x.shape[0])


def test_pidl_collocation_states():
    pidl = make("pidl")
    pidl.mlp = Recorder()
    batch = {"state": torch.zeros(500, 1, 3)}
    assert pidl.extra_loss(batch) > 0
    x = pidl.mlp.x[:, 0]
    low, high = torch.tensor(CONTEXT.box_low), torch.tensor(CONTEXT.box_high)
    assert x.shape == (500, 3) and (x >= low).all() and (x <= high).all()
    assert ((x.amin(0) - low) < 0.02 * (high - low)).all() and ((high - x.amax(0)) < 0.02 * (high - low)).all()


def test_pidl_physics_loss():
    pidl = make("pidl")
    batch = {"state": torch.zeros(256, 1, 3)}
    first = pidl.extra_loss(batch)
    assert first.shape == () and first > 0
    assert pidl.extra_loss(batch) != first  # the generator moves on
    as_idm = make("pidl")
    clipped = copy.deepcopy(as_idm.idm)
    clipped.forward = lambda x, idm=as_idm.idm: torch.clamp(idm(x), -8.0, 4.0)
    as_idm.mlp = clipped  # network output = IDM clipped to the range of the rollouts
    assert as_idm.extra_loss(batch) == 0.0
    unclipped = make("pidl")
    unclipped.mlp = copy.deepcopy(unclipped.idm)
    assert unclipped.extra_loss(batch) > 0.0  # the physics target is the clipped IDM
    tripled = build_model({**model_cfg("pidl"), "alpha": 3.0}, CONTEXT)
    tripled.load_state_dict(pidl.state_dict())
    assert tripled.extra_loss(batch).item() == pytest.approx(3.0 * first.item(), rel=1e-6)  # same seed, same states
    for learnable in (False, True):
        model = build_model({**model_cfg("pidl"), "idm_learnable": learnable}, CONTEXT)
        assert model.idm.bounded == learnable and n_trainable(model) == 4481 + 5 * learnable
        assert model.idm.params_dict() == pytest.approx(IDM_P, rel=1e-5)
        model.extra_loss(batch).backward()
        assert (model.idm.raw.grad is not None) == learnable
        assert model.mlp.net[0].weight.grad.abs().sum() > 0
        if learnable:
            assert torch.isfinite(model.idm.raw.grad).all() and model.idm.raw.grad.abs().sum() > 0


# ---------------------------------------------------------------------------- PERL
def test_perl_starts_as_newell_and_uses_the_history():
    perl = make("perl")
    x = history(40)
    assert torch.allclose(perl(x).double(), newell_acc(x[:, -30:], NEWELL_P["w"]), rtol=1e-4, atol=1e-4)
    assert torch.equal(perl.residual(x), torch.zeros(6))
    with torch.no_grad():
        perl.residual.head.weight.normal_()
        perl.residual.head.bias.fill_(0.1)
    early = x.clone()
    early[:, -25] += torch.tensor([5.0, 1.0, 2.0], dtype=torch.float64)  # inside the window of 30
    assert not torch.equal(perl.residual(early), perl.residual(x))  # the residual reads the history
    outside = x.clone()
    outside[:, 0] += 5.0
    assert torch.equal(perl(outside), perl(x))
