"""Stability penalties: exact gradients at fixed equilibria, zero for a stable law, positive for the OVM,
anchored equilibria and the existence term of the spacing band; the parts of ``combined`` (D110) and the
memoryless view of a recurrent law; the monotonicity terms of ``monotone`` (D117)."""

import dataclasses
import json
import math
from typing import Callable, Sequence

import pytest
import torch
from torch import Tensor, nn

from cf_stability.models.base import CFModel, InputScaler, ModelContext
from cf_stability.models.idm import idm_acc
from cf_stability.models.ovm import OVM
from cf_stability.stability.analytic import history_jacobian, transfer_discrete
from cf_stability.stability.equilibrium import constant_history, find_equilibria
from cf_stability.stability.penalties import (
    INFO_KEYS,
    PenaltyConfig,
    combined_parts,
    combined_penalty,
    existence_only_penalty,
    existence_penalty,
    gain_penalty,
    jacobian_penalty,
    linear_gain_penalty,
    memoryless_jacobian_part,
    memoryless_partials,
    monotone_penalty,
    monotone_terms,
    penalty_band,
    sample_speeds,
    stability_penalty,
)

IDM_P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
SCALER = ((30.0, 0.0, 15.0), (15.0, 1.5, 6.0))
KINDS = ["jacobian", "gain", "linear_gain", "combined"]
WITH_MONOTONE = [*KINDS, "monotone"]  # D117: the existence term and the anchoring of every penalty, no string term
# a short rollout for the gradient checks (10 steps after the warm-up): the gain is then a mere fit
SHORT_ROLLOUT = {"omegas": (2.0, 4.0), "horizon_s": 1.0, "measure_s": 0.5, "gain_margin": 0.9}


def settings(kind: str, **more) -> PenaltyConfig:
    """The penalty ``kind``; ``combined`` with both parts and the needle guard of D110 switched on."""
    extra = {"jacobian_weight": 1.0, "guard": 3.0} if kind == "combined" else {}
    return PenaltyConfig(kind=kind, **{**extra, **more})


class LinearLaw(CFModel):
    """``a = f_s (s - s0) + f_dv dv + f_v v``: the same partial derivatives at every state."""

    def __init__(self, f_s: float, f_dv: float, f_v: float, s0: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("coef", torch.tensor([f_s, f_dv, f_v], dtype=torch.float64))
        self.s0 = s0

    def forward(self, state_history: Tensor) -> Tensor:
        last = state_history[..., -1, :]
        return (last[..., 0] - self.s0) * self.coef[0] + last[..., 1] * self.coef[1] + last[..., 2] * self.coef[2]


class TinyMLP(CFModel):
    """Linear law plus a small tanh MLP (the parameters of the gradient check), float64."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.scaler = InputScaler(*SCALER)
        self.net = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)).double()

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[..., -1, :]
        law = 0.1 * (x[..., 0] - 2.0 - 1.5 * x[..., 2]) + 0.05 * x[..., 1]  # f_dv > 0: two active terms
        return law + 0.3 * self.net(self.scaler(x)).squeeze(-1)


class TinyGRU(CFModel):
    """IDM plus a small GRU over a window of 10 states (the parameters of the gradient check), float64."""

    window = 10

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.scaler = InputScaler(*SCALER)
        self.gru = nn.GRU(3, 8, batch_first=True).double()
        self.head = nn.Linear(8, 1).double()

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[:, -self.window :]
        last = x[:, -1]
        out, _ = self.gru(self.scaler(x))
        return idm_acc(last[:, 0], last[:, 1], last[:, 2], **IDM_P) + 0.3 * self.head(out[:, -1]).squeeze(-1)


def check_value_gradient(model: CFModel, value_fn: Callable[[], Tensor], h: float = 1e-5) -> None:
    """Autograd gradient of ``value_fn()`` with respect to the parameters against central differences."""
    params = list(model.parameters())
    value = value_fn()
    assert value.item() > 0.0  # the relu terms are active
    grads = torch.autograd.grad(value, params, allow_unused=True)  # e.g. output bias: no effect on derivatives
    grad = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
    fd = []
    with torch.no_grad():
        for p in params:
            flat = p.view(-1)
            for j in range(flat.numel()):
                old = flat[j].item()
                flat[j] = old + h
                up = value_fn().item()
                flat[j] = old - h
                down = value_fn().item()
                flat[j] = old
                fd.append((up - down) / (2.0 * h))
    fd = torch.tensor(fd, dtype=torch.float64)
    relevant = grad.abs() > 1e-3 * grad.abs().max()
    assert relevant.sum() >= 0.25 * grad.numel()
    rel = (grad - fd).abs() / torch.maximum(grad.abs(), fd.abs())
    assert rel[relevant].max() < 1e-4, rel[relevant].max()
    assert (grad - fd).abs().max() < 1e-5 * grad.abs().max()


def check_gradient(model: CFModel, penalty, speeds: Tensor, cfg: PenaltyConfig, h: float = 1e-5) -> None:
    """Autograd gradient of the penalty against central differences, equilibria (with the band of
    ``cfg``) held fixed.

    ``h = 1e-5``: with 1e-6 the rounding of a rollout (about 1e-13) already reaches 1e-4 of the
    smallest relevant gradients.
    """
    eq = find_equilibria(model, speeds, band=penalty_band(model, cfg))
    assert eq.found.all()
    check_value_gradient(model, lambda: penalty(model, speeds, cfg, eq)[0], h)


def band_mapping(v: Sequence[float], s_low: Sequence[float], s_high: Sequence[float]) -> dict:
    """A band as the training context lists it."""
    median = [0.5 * (lo + hi) for lo, hi in zip(s_low, s_high)]
    return {"v": list(v), "s_low": list(s_low), "s_median": median, "s_high": list(s_high), "n": [200] * len(v)}


def test_config():
    assert PenaltyConfig().kind == "none" and PenaltyConfig().n_equilibria == 0
    assert PenaltyConfig(kind="jacobian").n_equilibria == 16 and PenaltyConfig(kind="linear_gain").n_equilibria == 16
    assert PenaltyConfig(kind="monotone").n_equilibria == 16
    assert set(INFO_KEYS["monotone"]) == set(INFO_KEYS["jacobian"])
    cfg = PenaltyConfig.from_mapping({"kind": "gain", "omegas": [0.1, 1]})
    assert cfg.n_equilibria == 3 and cfg.omegas == (0.1, 1.0) and cfg.active
    with pytest.raises(ValueError, match="unknown penalty keys"):
        PenaltyConfig.from_mapping({"kind": "gain", "lambda": 1.0})
    with pytest.raises(ValueError, match="kind"):
        PenaltyConfig(kind="spectral")
    speeds = sample_speeds(cfg, torch.Generator().manual_seed(0), dtype=torch.float64)
    again = sample_speeds(cfg, torch.Generator().manual_seed(0), dtype=torch.float64)
    assert speeds.shape == (3,) and torch.equal(speeds, again) and ((speeds >= 5.0) & (speeds <= 30.0)).all()
    # band, existence and aggregate (docs/m4_contract.md, 1.3)
    assert (cfg.existence, cfg.aggregate, cfg.band) == ("band", "max", None)
    band = band_mapping([10.0, 20.0], [15.0, 25.0], [20.0, 30.0])
    with_band = dataclasses.replace(PenaltyConfig(kind="jacobian", aggregate="mean"), band=band)
    assert with_band.band == band and with_band.aggregate == "mean" and with_band.n_equilibria == 16
    assert json.loads(json.dumps(dataclasses.asdict(with_band)))["band"] == band  # the settings stay JSON-serialisable
    with pytest.raises(ValueError, match="band lacks"):
        PenaltyConfig(kind="jacobian", band={"v": [10.0, 20.0], "s_low": [1.0, 2.0]})
    for bad in ({"existence": "always"}, {"aggregate": "sum"}):
        with pytest.raises(ValueError, match="existence must be"):
            PenaltyConfig(kind="jacobian", **bad)
    assert PenaltyConfig(kind="gain").every == 1 and PenaltyConfig.from_mapping({"kind": "gain", "every": 8}).every == 8
    with pytest.raises(ValueError, match="every"):
        PenaltyConfig(kind="gain", every=0)
    # the band tensors are created once per device and dtype (no host-to-device copy in a CUDA graph)
    model = TinyMLP()
    first = penalty_band(model, with_band)
    assert first is penalty_band(model, dataclasses.replace(with_band)) and first.v.dtype == torch.float64
    assert penalty_band(model, PenaltyConfig(kind="jacobian")) is None
    # combined (D110): 16 speeds as the rollout penalty of E2; its two settings are off by default and
    # belong to it alone; the guard is part of the Jacobian part
    combined = PenaltyConfig.from_mapping({"kind": "combined", "jacobian_weight": 1, "guard": 3})
    assert combined.n_equilibria == 16 and (combined.jacobian_weight, combined.guard) == (1.0, 3.0)
    assert (PenaltyConfig().jacobian_weight, PenaltyConfig().guard) == (0.0, 0.0)
    assert PenaltyConfig(kind="combined").jacobian_weight == 0.0  # the rollout part alone
    for bad, message in (
        ({"kind": "gain", "jacobian_weight": 1.0}, "belong to kind combined"),
        ({"kind": "jacobian", "guard": 3.0}, "belong to kind combined"),
        ({"kind": "combined", "guard": 3.0}, "needs jacobian_weight"),
        ({"kind": "combined", "jacobian_weight": -1.0}, ">= 0"),
        ({"kind": "combined", "jacobian_weight": 1.0, "guard": math.nan}, ">= 0"),
    ):
        with pytest.raises(ValueError, match=message):
            PenaltyConfig(**bad)


@pytest.mark.parametrize("with_band", [False, True])
def test_jacobian_penalty_gradient(with_band):
    model, speeds = TinyMLP(), torch.tensor([8.0, 14.0, 21.0, 27.0], dtype=torch.float64)
    cfg = PenaltyConfig(kind="jacobian")
    if with_band:  # 0.3 m around the equilibria: the existence term of the band edges is active as well
        s_e = find_equilibria(model, speeds).s.tolist()
        band = band_mapping(speeds.tolist(), [s - 0.3 for s in s_e], [s + 0.3 for s in s_e])
        cfg = PenaltyConfig(kind="jacobian", band=band)
        assert jacobian_penalty(model, speeds, cfg)[1]["existence"].item() > 0.0
    check_gradient(model, jacobian_penalty, speeds, cfg)


@pytest.mark.parametrize("aggregate", ["max", "mean"])
def test_gain_penalty_gradient(aggregate):
    # short rollout (10 steps after the warm-up) to keep 642 evaluations fast; the gain is then a mere
    # fit, which does not matter for the gradient
    cfg = PenaltyConfig(kind="gain", aggregate=aggregate, **SHORT_ROLLOUT)
    check_gradient(TinyGRU(), gain_penalty, torch.tensor([10.0, 22.0], dtype=torch.float64), cfg)


# ------------------------------------------------------------------------------------ combined (D110)
GRU_SPEEDS = torch.tensor([8.0, 10.0, 14.0, 21.0, 27.0], dtype=torch.float64)
# TinyGRU at its equilibria: f_s 0.02-0.13, f_dv -0.26 to -0.46, f_v -0.12 to -0.19 1/s, margins -0.05 to 0.05.
# With the threshold 0.15 1/s both guard terms act (|f_v| above it at 8 and 10 m/s only) beside the margin term
GRU_GUARD = 0.15


def common_shift_partials(model: CFModel, s: Tensor, v: Tensor, h: float = 1e-5) -> Tensor:
    """``[n, 3]`` central differences of the acceleration when every entry of the constant window moves by
    ``h`` in ``s``, ``dv`` or ``v`` together (``dv = 0`` before the shift)."""
    zero, out = torch.zeros_like(s), []
    with torch.no_grad():
        for ds, ddv, dvv in ((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h)):
            up = model(constant_history(model, s + ds, v + dvv, zero + ddv))
            down = model(constant_history(model, s - ds, v - dvv, zero - ddv))
            out.append((up - down) / (2.0 * h))
    return torch.stack(out, dim=-1)


def test_memoryless_view_of_a_gru_is_the_response_to_a_common_shift_of_the_window():
    """docs/m3_contract.md, section 1: the memoryless view of a windowed law is the derivative with respect
    to a common shift of the whole constant window, the sum over the window positions of the history
    Jacobian (the static gains). Checked on a GRU against finite differences of its output."""
    model = TinyGRU()
    s = torch.tensor([14.0, 25.0, 40.0, 62.0], dtype=torch.float64)
    v = torch.tensor([8.0, 15.0, 21.0, 27.0], dtype=torch.float64)
    view = torch.stack(memoryless_partials(model, s, v), dim=-1)
    assert view.requires_grad  # differentiable with respect to the parameters (create_graph)
    fd = common_shift_partials(model, s, v)
    assert torch.allclose(view.detach(), fd, rtol=1e-7, atol=1e-10), (view, fd)
    window = history_jacobian(model, s, v)  # [n, window, 3]
    assert torch.allclose(view.detach(), window.sum(dim=1), rtol=1e-12, atol=1e-14)
    # a recurrent law: the earlier window positions matter (4e-4 to 9e-3 here, far above the error of the
    # differences), the latest state alone is not the view
    assert (window[:, :-1].abs().sum(dim=1) > 1e-4).all() and (view.detach() - fd).abs().max() < 1e-8
    assert not torch.allclose(window[:, -1], window.sum(dim=1), rtol=1e-2)


def test_jacobian_part_of_combined_from_finite_differences():
    """The Jacobian part of ``combined`` of a GRU at its equilibria: the terms of the Jacobian penalty and the
    needle guard of the derivatives under a common shift of the window, mean over the equilibria."""
    model = TinyGRU()
    cfg = PenaltyConfig(kind="combined", jacobian_weight=1.0, guard=GRU_GUARD)
    eq = find_equilibria(model, GRU_SPEEDS)
    assert eq.found.all()
    f_s, f_dv, f_v = common_shift_partials(model, eq.s, eq.v).unbind(-1)
    margin = f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s
    local = torch.relu(-f_s) + torch.relu(f_dv) + torch.relu(f_v + f_dv)
    guard = torch.relu(f_dv.abs() - GRU_GUARD) + torch.relu(f_v.abs() - GRU_GUARD)
    assert (guard > 0).all() and (torch.relu(f_v.abs() - GRU_GUARD) > 0).sum() == 2  # both guard terms act
    value, info = memoryless_jacobian_part(model, GRU_SPEEDS, cfg)
    expected = (torch.relu(0.5 - margin) + local + guard).mean()
    assert value.item() == pytest.approx(expected.item(), rel=1e-7) and value.requires_grad
    assert info["jacobian"].item() == value.item()
    assert info["guard"].item() == pytest.approx(guard.mean().item(), rel=1e-7)
    assert info["mean_margin"].item() == pytest.approx(margin.mean().item(), rel=1e-6)
    assert info["needle"].item() == pytest.approx(torch.maximum(f_dv.abs(), f_v.abs()).max().item(), rel=1e-7)
    # without the guard: the terms of the Jacobian penalty of the specification alone, no guard info
    value, info = memoryless_jacobian_part(model, GRU_SPEEDS, dataclasses.replace(cfg, guard=0.0))
    assert value.item() == pytest.approx((torch.relu(0.5 - margin) + local).mean().item(), rel=1e-7)
    assert math.isnan(info["guard"].item()) and info["needle"].item() > 0.4


def test_jacobian_part_of_combined_gradient():
    """Gradient of the new terms (memoryless view of a GRU, needle guard) against central differences."""
    cfg = PenaltyConfig(kind="combined", jacobian_weight=1.0, guard=GRU_GUARD)
    check_gradient(TinyGRU(), memoryless_jacobian_part, GRU_SPEEDS, cfg)


def test_combined_penalty_gradient():
    """The whole ``combined`` value (short rollout part with its existence term + Jacobian part with the guard)."""
    cfg = PenaltyConfig(kind="combined", jacobian_weight=1.0, guard=GRU_GUARD, **SHORT_ROLLOUT)
    speeds = torch.tensor([10.0, 22.0], dtype=torch.float64)
    rollout, jacobian, _ = combined_parts(TinyGRU(), speeds, cfg)
    assert rollout.item() > 0.0 and jacobian.item() > 0.0  # both parts act
    check_gradient(TinyGRU(), combined_penalty, speeds, cfg)


def test_combined_parts_and_their_info():
    """The rollout part is the gain penalty with its existence term, the Jacobian part has no existence term;
    without ``rollout`` and with ``jacobian_weight = 0`` a part is zero and its info NaN; the value of
    ``combined`` (the penalty of the epoch choice, D89) is the sum of the unweighted parts."""
    model, speeds = TinyGRU(), GRU_SPEEDS
    s_e = find_equilibria(model, speeds).s.tolist()
    # 0.3 m around the equilibria: the existence term of the band edges acts, the equilibria stay anchored
    band = band_mapping(speeds.tolist(), [s - 0.3 for s in s_e], [s + 0.3 for s in s_e])
    cfg = PenaltyConfig(kind="combined", jacobian_weight=0.5, guard=GRU_GUARD, band=band, **SHORT_ROLLOUT)
    rollout, jacobian, info = combined_parts(model, speeds, cfg)
    assert set(info) == set(INFO_KEYS["combined"])
    gain, gain_info = gain_penalty(model, speeds, cfg)
    part, part_info = memoryless_jacobian_part(model, speeds, cfg)
    assert rollout.item() == gain.item() and jacobian.item() == part.item()  # unweighted: jacobian_weight 0.5
    assert {k: info[k].item() for k in gain_info} == {k: v.item() for k, v in gain_info.items()}
    assert {k: info[k].item() for k in part_info} == {k: v.item() for k, v in part_info.items()}
    assert info["rollout"].item() == rollout.item() and info["existence"].item() > 0.0
    value, value_info = stability_penalty(model, speeds, cfg)
    assert value.item() == pytest.approx(rollout.item() + jacobian.item(), rel=1e-12)
    assert value_info.keys() == info.keys()
    # a training step without the rollout part (D90): the Jacobian part alone, the rollout info NaN
    rollout, jacobian, info = combined_parts(model, speeds, cfg, rollout=False)
    assert rollout.item() == 0.0 and not rollout.requires_grad and jacobian.item() == part.item()
    assert all(math.isnan(info[k].item()) for k in ("rollout", "max_gain", "existence"))
    assert info["n_no_equilibrium"].item() == 0.0
    # jacobian_weight = 0: the rollout part alone, the gain penalty of E2
    off = dataclasses.replace(cfg, jacobian_weight=0.0, guard=0.0)
    value, info = stability_penalty(model, speeds, off)
    assert value.item() == gain.item() and all(math.isnan(info[k].item()) for k in ("jacobian", "guard", "needle"))


def test_combined_on_a_memoryless_model_reduces_to_the_jacobian_penalty():
    """Choice of D110: ``combined`` does not refuse a memoryless model (window 1). Its memoryless view is the
    law itself, so the Jacobian part is exactly the Jacobian penalty of the specification without its
    existence term, and ``combined`` is the gain penalty plus ``jacobian_weight`` times that part (the
    guard adds its terms)."""
    model, speeds = TinyMLP(), torch.tensor([8.0, 14.0, 21.0, 27.0], dtype=torch.float64)
    assert model.window == 1
    expected, expected_info = jacobian_penalty(model, speeds, PenaltyConfig(kind="jacobian", existence_weight=0.0))
    cfg = PenaltyConfig(kind="combined", jacobian_weight=1.0)
    rollout, part, info = combined_parts(model, speeds, cfg)
    assert part.item() == expected.item() > 0.0 and info["mean_margin"].item() == expected_info["mean_margin"].item()
    assert rollout.item() == gain_penalty(model, speeds, cfg)[0].item()
    assert combined_penalty(model, speeds, cfg)[0].item() == pytest.approx(rollout.item() + expected.item(), rel=1e-12)
    eq = find_equilibria(model, speeds)
    f_s, f_dv, f_v = memoryless_partials(model, eq.s, eq.v)
    guard = (torch.relu(f_dv.abs() - 0.155) + torch.relu(f_v.abs() - 0.155)).mean()  # |f_v| 0.153-0.160
    assert 0.0 < guard.item() < 0.01
    guarded = combined_parts(model, speeds, dataclasses.replace(cfg, guard=0.155))[1]
    assert guarded.item() == pytest.approx(expected.item() + guard.item(), rel=1e-12)


# ------------------------------------------------------------------------------------ monotone (D117)
MONOTONE_SPEEDS = torch.tensor([6.0, 12.0, 18.0, 24.0], dtype=torch.float64)


class TinyIrrational(TinyMLP):
    """TinyMLP whose linear law violates two of the rational constraints of RACER: ``f_dv = 0.05 > 0`` and
    ``f_v = 0.1 > 0`` (equilibrium spacing ``40 - v`` before the network), float64."""

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[..., -1, :]
        law = 0.1 * (x[..., 0] - 40.0 + x[..., 2]) + 0.05 * x[..., 1]
        return law + 0.3 * self.net(self.scaler(x)).squeeze(-1)


def test_monotone_penalty_of_linear_laws_and_of_the_ovm():
    """``relu(-f_s) + relu(f_dv) + relu(f_v)`` at the equilibria, no string term: zero for the OVM, which is
    rational and string unstable at every equilibrium (margin -1), where the Jacobian penalty is 1.5; the
    margin is reported in the info."""
    cfg = PenaltyConfig(kind="monotone")
    terms = monotone_terms(*(torch.tensor([x], dtype=torch.float64) for x in (-0.1, 0.3, 0.2)))
    assert terms.item() == pytest.approx(0.6)
    cases = (  # law, monotone value, margin f_v^2 + 2 f_v f_dv - 2 f_s
        (LinearLaw(0.2, 0.3, 0.2, s0=40.0), 0.5, 0.04 + 0.12 - 0.4),  # f_dv and f_v positive; s_e = 40 - v
        (LinearLaw(0.2, -1.0, 0.2, s0=40.0), 0.2, 0.04 - 0.4 - 0.4),  # f_v > 0 although f_v + f_dv < 0
        (LinearLaw(*STABLE), 0.0, 2.6),
    )
    for law, value, margin in cases:
        result, info = monotone_penalty(law, MONOTONE_SPEEDS, cfg)
        assert result.item() == pytest.approx(value, abs=1e-12) and info["n_no_equilibrium"].item() == 0.0
        assert info["mean_margin"].item() == pytest.approx(margin, rel=1e-9) and info["existence"].item() == 0.0
        assert stability_penalty(law, MONOTONE_SPEEDS, cfg)[0].item() == result.item()
    # the second law: the Jacobian penalty sees the margin, not f_v > 0 (its local term is relu(f_v + f_dv))
    jacobian = jacobian_penalty(cases[1][0], MONOTONE_SPEEDS, PenaltyConfig(kind="jacobian"))[0]
    assert jacobian.item() == pytest.approx(0.5 + 0.76, rel=1e-9)
    ovm = OVM(ModelContext(ovm_params={"v0": 40.0, "tau": 1.0, "s0": 2.0}))
    value, info = monotone_penalty(ovm, MONOTONE_SPEEDS, cfg)
    assert value.item() == 0.0 and info["mean_margin"].item() == pytest.approx(-1.0)
    assert jacobian_penalty(ovm, MONOTONE_SPEEDS, PenaltyConfig(kind="jacobian"))[0].item() == pytest.approx(1.5)


@pytest.mark.parametrize("with_band", [False, True])
def test_monotone_penalty_gradient(with_band):
    """Gradient of the monotonicity terms (and of the existence term of the band edges) against central
    differences, equilibria held fixed."""
    model = TinyIrrational()
    eq = find_equilibria(model, MONOTONE_SPEEDS)
    f_s, f_dv, f_v = memoryless_partials(model, eq.s, eq.v)
    assert (f_dv > 0).all() and (f_v > 0).all() and (f_s > 0).all()  # two terms act at every equilibrium
    cfg = PenaltyConfig(kind="monotone")
    if with_band:  # 0.3 m around the equilibria: the existence term of the band edges is active as well
        s_e = eq.s.tolist()
        cfg = PenaltyConfig(kind="monotone", band=band_mapping(MONOTONE_SPEEDS.tolist(), [s - 0.3 for s in s_e],
                                                               [s + 0.3 for s in s_e]))  # fmt: skip
        assert monotone_penalty(model, MONOTONE_SPEEDS, cfg)[1]["existence"].item() > 0.0
    value = monotone_penalty(model, MONOTONE_SPEEDS, cfg)[0]
    expected = monotone_terms(f_s, f_dv, f_v).mean() + monotone_penalty(model, MONOTONE_SPEEDS, cfg)[1]["existence"]
    assert value.item() == pytest.approx(expected.item(), rel=1e-12)
    check_gradient(model, monotone_penalty, MONOTONE_SPEEDS, cfg)


@pytest.mark.parametrize("aggregate", ["max", "mean"])
def test_linear_gain_penalty_gradient(aggregate):
    cfg = PenaltyConfig(kind="linear_gain", omegas=(0.05, 0.4, 1.5), gain_margin=0.5, aggregate=aggregate)
    check_gradient(TinyGRU(), linear_gain_penalty, torch.tensor([10.0, 22.0], dtype=torch.float64), cfg)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_for_a_stable_law_positive_for_the_ovm(kind):
    speeds = torch.tensor([6.0, 12.0, 18.0, 24.0, 29.0], dtype=torch.float64)
    cfg = settings(kind)
    stable = LinearLaw(0.2, -1.0, -1.0)  # margin 2.6; |G_d| <= 0.93 from 0.05 rad/s on; |f_dv|, |f_v| below the guard
    value, info = stability_penalty(stable, speeds, cfg)
    assert value.item() == 0.0 and info["n_no_equilibrium"].item() == 0.0
    ovm = OVM(ModelContext(ovm_params={"v0": 40.0, "tau": 1.0, "s0": 2.0}))  # margin -1 everywhere
    value, info = stability_penalty(ovm, speeds, cfg)
    assert value.item() > 0.0 and info["n_no_equilibrium"].item() == 0.0
    if kind == "jacobian":
        assert value.item() == pytest.approx(0.5 + 1.0) and info["mean_margin"].item() == pytest.approx(-1.0)
    else:
        assert info["max_gain"].item() > 1.0
    if kind == "combined":  # the rollout part plus the Jacobian part relu(0.5 - (-1)), f_v = -1 inside the guard
        assert info["jacobian"].item() == pytest.approx(1.5) and info["guard"].item() == 0.0
        assert value.item() == pytest.approx(info["rollout"].item() + 1.5, rel=1e-12) and info["needle"].item() == 1.0


@pytest.mark.parametrize("aggregate", ["max", "mean"])
def test_linear_gain_of_a_memoryless_law_is_the_discrete_transfer(aggregate):
    law = LinearLaw(0.5, -0.2, -0.5)  # margin -0.55: gains above 1 below 0.74 rad/s
    cfg = PenaltyConfig(kind="linear_gain", gain_margin=0.05, aggregate=aggregate)
    speeds = torch.tensor([7.0, 19.0], dtype=torch.float64)
    value, info = linear_gain_penalty(law, speeds, cfg)
    # the 25 frequencies of the audit; the margin shrinks as (w / 0.1)^2 below 0.1 rad/s
    one, omega = torch.ones(2, dtype=torch.float64), torch.logspace(math.log10(0.02), math.log10(2.0), 25, dtype=torch.float64)
    margin = 0.05 * torch.clamp((omega / 0.1) ** 2, max=1.0)
    assert margin[0].item() == pytest.approx(0.002) and margin[-1].item() == 0.05
    gain = transfer_discrete(0.5 * one, -0.2 * one, -0.5 * one, omega, 0.1).abs()
    excess = gain - 1.0 + margin  # [2 equilibria, 25 frequencies]
    # max: the largest excess of every equilibrium, then the mean over the equilibria; mean: M3
    expected = torch.relu(excess.max(dim=1).values).mean() if aggregate == "max" else torch.relu(excess).mean()
    assert value.item() == pytest.approx(expected.item(), rel=1e-12)
    assert info["max_gain"].item() == pytest.approx(gain.max().item(), rel=1e-12)
    if aggregate == "max":  # the narrow amplification is not averaged over the 25 frequencies
        assert value.item() > 2.0 * torch.relu(excess).mean().item()


@pytest.mark.parametrize("aggregate", ["max", "mean"])
def test_rollout_gain_approaches_the_linear_gain(aggregate):
    """At the frequencies whose period fits into the measured 20 s, the rollout gain of a linear law
    is the discrete transfer function (the transient of the warm-up has decayed). ``max``: the
    largest gain of every equilibrium, then the mean over the equilibria; ``mean``: over both."""
    law = LinearLaw(0.5, -0.2, -0.5)
    speeds = torch.tensor([15.0, 20.0], dtype=torch.float64)  # the same derivatives at both
    cfg = PenaltyConfig(kind="gain", omegas=(0.2, 0.4, 0.8), gain_margin=10.0, aggregate=aggregate)  # relu(gain + 9)
    value, info = gain_penalty(law, speeds, cfg)
    one, omega = torch.ones(1, dtype=torch.float64), torch.tensor(cfg.omegas, dtype=torch.float64)
    gain = transfer_discrete(0.5 * one, -0.2 * one, -0.5 * one, omega, 0.1).abs()
    expected = gain.max() if aggregate == "max" else gain.mean()
    assert value.item() - 9.0 == pytest.approx(expected.item(), rel=2e-3)
    assert info["max_gain"].item() == pytest.approx(gain.max().item(), rel=2e-3)


class Braking(CFModel):
    """Brakes at every state: no equilibrium."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))

    def forward(self, state_history: Tensor) -> Tensor:
        return self.c.expand(state_history.shape[0])


def test_speeds_without_equilibrium_are_skipped_and_counted():
    speeds = torch.tensor([10.0, 20.0], dtype=torch.float64)
    for kind in WITH_MONOTONE:
        # the behaviour of the specification: a speed without equilibrium contributes nothing
        value, info = stability_penalty(Braking(), speeds, settings(kind, existence_weight=0.0))
        assert value.item() == 0.0 and not value.requires_grad and info["n_no_equilibrium"].item() == 2.0
        # with the existence term it is penalised: the law must accelerate at 200 m
        braking = Braking()
        f_far = braking(torch.tensor([[[200.0, 0.0, 10.0]], [[200.0, 0.0, 20.0]]], dtype=torch.float64))
        f_near = braking(torch.tensor([[[1.0, 0.0, 10.0]], [[1.0, 0.0, 20.0]]], dtype=torch.float64))
        expected = (torch.relu(f_near + 0.1) + torch.relu(0.1 - f_far)).mean()
        value, info = stability_penalty(braking, speeds, settings(kind))
        assert value.item() == pytest.approx(expected.item(), rel=1e-12) and value.item() > 0.0
        assert info["existence"].item() == pytest.approx(expected.item(), rel=1e-12)
        # outside the speed range in which an equilibrium is demanded nothing is added
        outside = settings(kind, exist_v_min=12.0, exist_v_max=18.0)
        assert stability_penalty(braking, speeds, outside)[0].item() == 0.0
        half = settings(kind, exist_v_min=5.0, exist_v_max=15.0)
        first = torch.relu(f_near[0] + 0.1) + torch.relu(0.1 - f_far[0])
        assert stability_penalty(braking, speeds, half)[0].item() == pytest.approx(first.item(), rel=1e-12)
    indifferent = LinearLaw(0.0, -1.0, 0.0)  # zero acceleration at every spacing: nominal spacing, margin 0
    value, info = jacobian_penalty(indifferent, speeds, PenaltyConfig(kind="jacobian"))
    assert info["n_no_equilibrium"].item() == 0.0 and value.item() == pytest.approx(0.5)


# LinearLaw(0.2, -1, -1): f(s, 0, v) = 0.2 (s - 2) - v, equilibrium s_e = 2 + 5 v, margin 2.6 (no stability term)
STABLE = (0.2, -1.0, -1.0)
BAND_SPEEDS = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0], dtype=torch.float64)  # 10-20 m/s have a band


def stable_band(below: float, above: float) -> dict:
    """Band from ``s_e + below`` to ``s_e + above`` of the stable linear law, listed at 10 and 20 m/s."""
    return band_mapping([10.0, 20.0], [52.0 + below, 102.0 + below], [52.0 + above, 102.0 + above])


@pytest.mark.parametrize("kind", WITH_MONOTONE)
def test_band_existence_term_at_the_band_edges(kind):
    law = LinearLaw(*STABLE)
    cases = (
        (-10.0, 10.0, 0.0),  # brakes by 2 m/s^2 at the lower edge, accelerates by 2 at the upper edge
        (1.0, 11.0, 0.3),  # both edges above the equilibrium: relu(0.2 + 0.1) at the lower edge
        (-0.25, 0.25, 0.1),  # narrow band: brakes and accelerates by 0.05 only, relu(0.05) twice
        (-11.0, -1.0, 0.3),  # both edges below: relu(0.1 - (-0.2)) at the upper edge
    )
    for below, above, per_speed in cases:
        cfg = settings(kind, band=stable_band(below, above))
        value, info = stability_penalty(law, BAND_SPEEDS, cfg)
        # the mean over the three speeds that have a band (every one has the same excess); the
        # stability terms vanish, and every equilibrium counts as there is one at every speed
        assert value.item() == pytest.approx(per_speed, abs=1e-12)
        assert info["existence"].item() == pytest.approx(per_speed, abs=1e-12)
        doubled = stability_penalty(law, BAND_SPEEDS, dataclasses.replace(cfg, existence_weight=2.0))[0]
        assert doubled.item() == pytest.approx(2.0 * per_speed, abs=1e-12)
        # fixed (D74): the law has an equilibrium at every speed, the term vanishes; no band: the same
        assert stability_penalty(law, BAND_SPEEDS, dataclasses.replace(cfg, existence="fixed"))[0].item() == 0.0
        assert stability_penalty(law, BAND_SPEEDS, dataclasses.replace(cfg, band=None))[0].item() == 0.0
    # speeds without band carry no existence term: none of them has a band here
    cfg = settings(kind, band=stable_band(1.0, 11.0))
    assert stability_penalty(law, torch.tensor([5.0, 25.0], dtype=torch.float64), cfg)[0].item() == 0.0


def test_band_existence_term_falls_back_to_d74_without_band():
    braking, speeds = Braking(), torch.tensor([10.0, 20.0], dtype=torch.float64)
    usable = torch.zeros(2, dtype=torch.bool)
    d74 = (torch.relu(braking.c + 0.1) + torch.relu(0.1 - braking.c)).item()  # brakes at 1 m and at 200 m
    for existence in ("band", "fixed"):
        cfg = PenaltyConfig(kind="jacobian", existence=existence)
        assert existence_penalty(braking, speeds, usable, cfg).item() == pytest.approx(d74, rel=1e-12)
    # with a band that covers 10 m/s only, the band term is the mean over that speed alone
    cfg = PenaltyConfig(kind="jacobian", band=band_mapping([5.0, 15.0], [20.0, 30.0], [40.0, 50.0]))
    assert existence_penalty(braking, speeds, usable, cfg).item() == pytest.approx(d74)


@pytest.mark.parametrize("existence", ["band", "fixed"])
def test_existence_kind_is_the_existence_term_alone(existence):
    model, speeds = TinyMLP(), torch.tensor([8.0, 14.0, 21.0, 27.0], dtype=torch.float64)
    s_e = find_equilibria(model, speeds).s.tolist()
    # listed at 8, 14, 21 m/s: 0.3 m around the equilibrium at 8 and 14 m/s, above it at 21 m/s; 27 m/s has no band
    low, high = [s_e[0] - 0.3, s_e[1] - 0.3, s_e[2] + 5.0], [s_e[0] + 0.3, s_e[1] + 0.3, s_e[2] + 8.0]
    band = band_mapping(speeds[:3].tolist(), low, high)
    cfg = PenaltyConfig(kind="existence", band=band, existence=existence)
    assert PenaltyConfig(kind="existence").n_equilibria == 16
    value, info = stability_penalty(model, speeds, cfg)
    eq = find_equilibria(model, speeds, band=penalty_band(model, cfg))
    assert value.item() == existence_penalty(model, speeds, eq.usable, cfg).item()
    assert set(info) == {"n_no_equilibrium", "existence"} and info["existence"].item() == value.item()
    assert info["n_no_equilibrium"].item() == 1.0  # 21 m/s: the equilibrium lies below the band
    if existence == "band":  # the edges of every speed with band; fixed: every speed has an equilibrium, no term
        assert value.item() > 0.0
        check_gradient(model, existence_only_penalty, speeds, cfg)
    else:
        assert value.item() == 0.0


def test_band_existence_term_gradient():
    model, speeds = TinyMLP(), torch.tensor([8.0, 14.0, 21.0, 27.0], dtype=torch.float64)
    eq = find_equilibria(model, speeds)
    assert eq.found.all()
    # 0.3 m around the equilibria: the law brakes and accelerates by less than m_e at the edges
    s_e = eq.s.tolist()
    band = band_mapping(speeds.tolist(), [s - 0.3 for s in s_e], [s + 0.3 for s in s_e])
    cfg = PenaltyConfig(kind="jacobian", band=band)
    near = model(torch.tensor([[[s - 0.3, 0.0, v]] for s, v in zip(s_e, speeds.tolist())], dtype=torch.float64))
    far = model(torch.tensor([[[s + 0.3, 0.0, v]] for s, v in zip(s_e, speeds.tolist())], dtype=torch.float64))
    assert (near > -0.1).all() and (far < 0.1).all()  # both relu terms are active at every speed
    value = existence_penalty(model, speeds, eq.usable, cfg)
    assert value.item() == pytest.approx((torch.relu(near + 0.1) + torch.relu(0.1 - far)).mean().item(), rel=1e-12)
    check_value_gradient(model, lambda: existence_penalty(model, speeds, eq.usable, cfg))


class SpeedLaw(CFModel):
    """``a = 0.02 v (s - 2 - 1.5 v) - 0.3 dv``: one equilibrium ``s_e = 2 + 1.5 v`` whose derivatives
    depend on the speed (margin ``0.0009 v^2 - 0.022 v``)."""

    def forward(self, state_history: Tensor) -> Tensor:
        s, dv, v = state_history[..., -1, :].unbind(-1)
        return 0.02 * v * (s - 2.0 - 1.5 * v) - 0.3 * dv


# listed at 8, 16, 24 m/s: s_e -+ 2 m at 8 and 16 m/s, far above s_e at 24 m/s. Interpolated: 10 and
# 14 m/s hold their equilibrium, 20 and 22 m/s do not; 6 and 28 m/s have no band
SPEED_BAND = band_mapping([8.0, 16.0, 24.0], [12.0, 24.0, 60.0], [16.0, 28.0, 80.0])
MIXED_SPEEDS = torch.tensor([6.0, 10.0, 14.0, 20.0, 22.0, 28.0], dtype=torch.float64)
ANCHORED = torch.tensor([True, True, True, False, False, True])


@pytest.mark.parametrize("kind", KINDS)
def test_stability_terms_use_the_anchored_equilibria_only(kind):
    law = SpeedLaw()
    eq = find_equilibria(law, MIXED_SPEEDS, band=penalty_band(law, settings(kind, band=SPEED_BAND)))
    assert eq.status == ["ok", "ok", "ok", "outside", "outside", "ok"] and torch.equal(eq.anchored, ANCHORED)
    value, info = stability_penalty(law, MIXED_SPEEDS, settings(kind, band=SPEED_BAND, existence_weight=0.0))
    assert info["n_no_equilibrium"].item() == 2.0
    # the stability terms of the four anchored speeds alone, found without band (the same equilibria)
    subset, subset_info = stability_penalty(law, MIXED_SPEEDS[ANCHORED], settings(kind, existence_weight=0.0))
    everywhere = stability_penalty(law, MIXED_SPEEDS, settings(kind, existence_weight=0.0))[0]
    assert subset_info["n_no_equilibrium"].item() == 0.0 and value.item() > 0.0
    assert value.item() == pytest.approx(subset.item(), rel=1e-9) and abs(value.item() - everywhere.item()) > 1e-3
    # the existence term acts where the band misses the equilibrium: the law accelerates at its lower edge
    with_existence, info = stability_penalty(law, MIXED_SPEEDS, settings(kind, band=SPEED_BAND))
    low, high, has_band = penalty_band(law, PenaltyConfig(band=SPEED_BAND)).at(MIXED_SPEEDS)
    f_low, f_high = (law.acc(s, torch.zeros_like(s), MIXED_SPEEDS) for s in (low, high))
    terms = torch.relu(f_low + 0.1) + torch.relu(0.1 - f_high)
    assert terms[has_band & ANCHORED].max().item() == 0.0 and (terms[has_band & ~ANCHORED] > 4.0).all()
    assert info["existence"].item() == pytest.approx(terms[has_band].mean().item(), rel=1e-12)
    assert with_existence.item() == pytest.approx(value.item() + info["existence"].item(), rel=1e-12)
