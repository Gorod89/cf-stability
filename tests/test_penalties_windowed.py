"""The analytic windowed-gain penalty of the recurrent laws (``windowed_gain``, D128, docs/m11_contract.md): the
penalty of a linear law with memory equals the gain, the low-frequency coefficient and the poles of its known
transfer function; gradients by central differences; the derivative of the spectral radius at nearly defective
poles; ``M_w`` equal to that of ``scripts/analysis/full_history_audit.py`` on stored runs; the existence term and
the anchoring of every penalty; the configuration (late keys of the config hash) and the training loop (no CUDA
graph for the penalty steps)."""

import copy
import dataclasses
import importlib.util
import math
import os
import warnings
from typing import Callable

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cf_stability.data.schema import DT, Event
from cf_stability.models import load_model
from cf_stability.models.base import CFModel, InputScaler
from cf_stability.models.idm import idm_acc, idm_equilibrium_spacing
from cf_stability.stability.analytic import closed_loop_poles, transfer_discrete
from cf_stability.stability.equilibrium import V_GRID, Equilibria, find_equilibria
from cf_stability.stability.penalties import (
    CAPTURABLE,
    DEFAULT_N_EQUILIBRIA,
    INFO_KEYS,
    KINDS,
    PenaltyConfig,
    existence_penalty,
    penalty_band,
    spectral_radius,
    stability_penalty,
    windowed_gain_penalty,
    windowed_response,
)
from cf_stability.train.closed_loop import rollout_memoryless, rollout_model
from cf_stability.train.tensors import BandConfig, training_context
from cf_stability.train.trainer import RolloutConfig, TrainConfig, train_model
from cf_stability.utils import REPO_ROOT, config_hash, read_json

OMEGA = torch.logspace(math.log10(0.02), math.log10(2.0), 25, dtype=torch.float64)  # the audit frequencies
SCALER = ((30.0, 0.0, 15.0), (15.0, 1.5, 6.0))
IDM_P = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
GPU_ONLY = pytest.mark.skipif(
    not os.environ.get("CF_GPU_TESTS") or not torch.cuda.is_available(), reason="GPU test: set CF_GPU_TESTS=1"
)


def windowed(**more) -> PenaltyConfig:
    return PenaltyConfig(kind="windowed_gain", **more)


class GeometricLaw(CFModel):
    """``a = sum_j c_s[j] (s - s0) + c_dv[j] dv + c_v[j] v`` of the state ``j`` steps ago, ``j = 0..W-1``, with the
    geometric weights ``c_x[j] = g_x (1 - r_x) r_x^j / (1 - r_x^W)``: a linear law with memory whose static gains are
    ``g_x`` and whose window polynomials ``F_x(z) = g_x (1 - r_x) (1 - (r_x / z)^W) / ((1 - r_x^W) (1 - r_x / z))`` are
    known in closed form."""

    def __init__(self, gains: tuple[float, float, float], ratios: tuple[float, float, float], window: int = 30,
                 s0: float = 2.0) -> None:  # fmt: skip
        super().__init__()
        self.window, self.s0, self.gains, self.ratios = int(window), float(s0), gains, ratios
        lag = torch.arange(self.window - 1, -1, -1, dtype=torch.float64)  # steps ago of the window entries
        weights = [g * (1.0 - r) * r**lag / (1.0 - r**self.window) for g, r in zip(gains, ratios)]
        self.register_buffer("weights", torch.stack(weights, dim=-1))  # [W, 3], oldest entry first

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[:, -self.window :].to(torch.float64)
        x = x - torch.tensor([self.s0, 0.0, 0.0], dtype=torch.float64)
        return (x * self.weights).sum(dim=(1, 2))

    # ------------------------------------------------------------------ the known linearisation, in closed form
    def F(self, z: Tensor) -> list[Tensor]:
        W = self.window
        return [g * (1.0 - r) * (1.0 - (r / z) ** W) / ((1.0 - r**W) * (1.0 - r / z)) for g, r in zip(self.gains, self.ratios)]

    def gain(self, omega: Tensor) -> Tensor:
        """``|G_w(e^(i w dt))|`` of equation (A2) with the closed-form window polynomials."""
        z = torch.exp(1j * omega.to(torch.complex128) * self.dt)
        F_s, F_dv, F_v = self.F(z)
        dt = self.dt
        return ((dt**2 * F_s * z - dt * F_dv * (z - 1)) / ((z - 1) ** 2 + dt**2 * F_s * z - dt * (F_dv + F_v) * (z - 1))).abs()

    def margin_window(self) -> float:
        """``M_w = M + 2 (f_v m_s - f_s m_v) - dt f_s f_v`` with the lag moments ``m_x = dt sum_j j c_x[j]`` of the geometric
        weights in closed form (``sum_{j<W} j r^j = r (1 - W r^(W-1) + (W - 1) r^W) / (1 - r)^2``)."""
        W, dt = self.window, self.dt
        f_s, f_dv, f_v = self.gains
        moments = [dt * g * (1 - r) / (1 - r**W) * r * (1 - W * r ** (W - 1) + (W - 1) * r**W) / (1 - r) ** 2
                   for g, r in zip(self.gains, self.ratios)]  # fmt: skip
        m_s, _, m_v = moments
        return f_v**2 + 2 * f_v * f_dv - 2 * f_s + 2 * (f_v * m_s - f_s * m_v) - dt * f_s * f_v

    def radius(self) -> float:
        """Largest modulus of the roots of ``z^(W-1) ((z - 1)^2 + dt^2 F_s(z) z - dt (F_dv(z) + F_v(z)) (z - 1))``."""
        W, dt = self.window, self.dt
        c = self.weights.numpy()[::-1]  # c[j] acts on the state j steps ago
        coefficients = np.zeros(W + 2)  # of z^d at index d
        coefficients[[W + 1, W, W - 1]] += (1.0, -2.0, 1.0)
        for j in range(W):  # z^(W-1) z^-j = z^(W-1-j)
            coefficients[W - j] += dt**2 * c[j, 0] - dt * (c[j, 1] + c[j, 2])
            coefficients[W - 1 - j] += dt * (c[j, 1] + c[j, 2])
        return float(np.abs(np.roots(coefficients[::-1])).max())


# static gains (f_s, f_dv, f_v) and ratios: poles inside the unit circle, a low-frequency gain above 1 (M_w < 0)
AMPLIFYING = GeometricLaw((0.05, -0.17, -0.2), (0.95, 0.6, 0.3))  # M = 0.008, M_w = -0.012
# a damped law with memory: M_w > 0, every pole inside
DAMPED = GeometricLaw((0.08, -0.6, -0.6), (0.5, 0.3, 0.4), window=10)
SPEEDS = torch.tensor([8.0, 15.0, 23.0], dtype=torch.float64)


def penalty_of(law: GeometricLaw, cfg: PenaltyConfig) -> float:
    """The penalty of ``windowed_gain`` from the closed forms of ``law`` (the same derivatives at every speed)."""
    gain = law.gain(OMEGA)
    gain_term = torch.relu(gain - 1.0 + cfg.window_gain_margin).square().mean().item()
    return gain_term + max(0.0, cfg.margin - law.margin_window()) + max(0.0, law.radius() - cfg.pole_margin)


# ------------------------------------------------------------------------------------------ configuration


def test_config_and_hash_neutral_keys():
    cfg = windowed()
    assert "windowed_gain" in KINDS and cfg.n_equilibria == DEFAULT_N_EQUILIBRIA["windowed_gain"] == 16
    assert (cfg.window_gain_margin, cfg.margin, cfg.pole_margin, cfg.every) == (0.02, 0.5, 0.995, 1)
    assert (cfg.linear_omega_min, cfg.linear_omega_max, cfg.linear_n_omega) == (0.02, 2.0, 25)  # the audit frequencies
    assert "windowed_gain" not in CAPTURABLE and set(CAPTURABLE) == set(KINDS) - {"windowed_gain"}
    assert INFO_KEYS["windowed_gain"][0] == "n_no_equilibrium" and "existence" in INFO_KEYS["windowed_gain"]
    assert PenaltyConfig.from_mapping({"kind": "windowed_gain", "pole_margin": 0.99}).pole_margin == 0.99
    for bad, message in (
        ({"kind": "gain", "pole_margin": 0.99}, "belong to kind windowed_gain"),
        ({"kind": "jacobian", "window_gain_margin": 0.05}, "belong to kind windowed_gain"),
        ({"kind": "windowed_gain", "pole_margin": 0.0}, "pole_margin"),
        ({"kind": "windowed_gain", "pole_margin": 1.5}, "pole_margin"),
        ({"kind": "windowed_gain", "window_gain_margin": math.inf}, "window_gain_margin"),
    ):
        with pytest.raises(ValueError, match=message):
            PenaltyConfig(**bad)
    # late keys of the config (D110's mechanism, cf_stability/utils.py): at their defaults the hash is that of a config
    # without them, so the hashes of the earlier runs stay valid; any other value enters the hash
    base = {"train": {"penalty": {"kind": "gain", "weight": 0.1}}, "seed": 0}
    with_keys = copy.deepcopy(base)
    with_keys["train"]["penalty"].update(window_gain_margin=0.02, pole_margin=0.995)
    assert config_hash(with_keys) == config_hash(base)
    for key, value in (("window_gain_margin", 0.05), ("pole_margin", 0.99)):
        changed = copy.deepcopy(with_keys)
        changed["train"]["penalty"][key] = value
        assert config_hash(changed) != config_hash(base)
    # configs/train.yaml holds the two keys at these defaults
    text = (REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8")
    assert "window_gain_margin: 0.02" in text and "pole_margin: 0.995" in text


# ------------------------------------------------------------------------- a linear law with a known transfer


@pytest.mark.parametrize("law", [AMPLIFYING, DAMPED], ids=["amplifying", "damped"])
def test_response_of_a_linear_law_with_memory_is_its_known_transfer_function(law):
    eq = find_equilibria(law, SPEEDS)
    assert eq.found.all()
    response = windowed_response(law, eq.s, eq.v, windowed())
    expected = law.weights.expand(len(SPEEDS), -1, -1)
    torch.testing.assert_close(response["jacobian"], expected, rtol=1e-12, atol=1e-15)  # one backward pass
    torch.testing.assert_close(response["omega"], OMEGA, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(response["gain"], law.gain(OMEGA).expand(len(SPEEDS), -1), rtol=1e-10, atol=1e-12)
    assert response["margin_window"].tolist() == pytest.approx([law.margin_window()] * len(SPEEDS), rel=1e-10)
    assert response["radius"].tolist() == pytest.approx([law.radius()] * len(SPEEDS), rel=1e-10)
    if law is AMPLIFYING:  # the memory moves M_w below the memoryless margin and below 0: |G_w| > 1 at low frequency
        f_s, f_dv, f_v = law.gains
        assert law.margin_window() < 0.0 < f_v**2 + 2 * f_v * f_dv - 2 * f_s and response["gain"][0, 0] > 1.0
    assert law.radius() < 1.0


def test_penalty_of_a_linear_law_with_memory_is_the_gain_of_its_transfer_function():
    law = AMPLIFYING
    # with the margins m_w = 1, M_w > -100 and rho < 1 the penalty is the mean squared gain over the audit frequencies
    cfg = windowed(window_gain_margin=1.0, margin=-100.0, pole_margin=1.0)
    value, info = windowed_gain_penalty(law, SPEEDS, cfg)
    assert value.item() == pytest.approx(law.gain(OMEGA).square().mean().item(), rel=1e-10)
    assert info["max_gain"].item() == pytest.approx(law.gain(OMEGA).max().item(), rel=1e-10)
    # the margins of D128 (the pole bound lowered to 0.98, below the radius 0.984): every term is active and the
    # penalty is that of the closed forms
    cfg = windowed(pole_margin=0.98)
    value, info = windowed_gain_penalty(law, SPEEDS, cfg)
    assert min(info[k].item() for k in ("gain_term", "margin_term", "pole_term")) > 0.0
    assert value.item() == pytest.approx(penalty_of(law, cfg), rel=1e-10)
    assert info["min_margin_window"].item() == pytest.approx(law.margin_window(), rel=1e-10)
    assert info["max_radius"].item() == pytest.approx(law.radius(), rel=1e-10)
    assert info["n_no_equilibrium"].item() == 0.0 and info["existence"].item() == 0.0  # no band: no existence term
    assert stability_penalty(law, SPEEDS, cfg)[0].item() == value.item()
    assert value.item() == pytest.approx(
        info["gain_term"].item() + info["margin_term"].item() + info["pole_term"].item(), rel=1e-12
    )


def test_windowed_gain_is_the_gain_of_the_rollout():
    """The law is linear: the follower speed of the rollout of the project (rollout_model, semi-implicit Euler, the
    window of states) oscillates with the amplitude |G_w| A once the transient has decayed."""
    law, amplitude = AMPLIFYING, 0.2
    omega = torch.tensor([0.1, 0.4, 1.2], dtype=torch.float64)
    eq = find_equilibria(law, [15.0])
    n_steps = 6000  # 600 s; the transient decays as rho^k with rho = 0.984
    t = law.dt * torch.arange(n_steps + law.window, dtype=torch.float64).clamp(min=0.0)
    gains = []
    for w in omega:
        t_k = (t - (law.window - 1) * law.dt).clamp(min=0.0)
        v_lead = (eq.v[0] + amplitude * torch.sin(w * t_k))[None]
        x_lead = eq.s[:, None] + law.dt * (torch.cumsum(v_lead, dim=-1) - v_lead[:, :1])
        res = rollout_model(law, x_lead, v_lead, eq.s[:, None].expand(-1, law.window), eq.v[:, None].expand(-1, law.window),
                            dt=law.dt)  # fmt: skip
        n_meas = 4000
        tm = t_k[-n_meas:]
        design = torch.stack((torch.sin(w * tm), torch.cos(w * tm), torch.ones_like(tm)), dim=-1)
        coef = torch.linalg.lstsq(design, (res.v[0, -n_meas:] - eq.v[0])[:, None]).solution[:, 0]
        gains.append(torch.hypot(coef[0], coef[1]).item() / amplitude)
    assert gains == pytest.approx(law.gain(omega).tolist(), rel=1e-6)


def test_memoryless_law_gives_the_discrete_transfer_function():
    """Window 1: G_w is the transfer function of the memoryless law under the integrator, M_w = M - dt f_s f_v and the
    poles are the roots of the quadratic of the discrete loop."""
    law = GeometricLaw((0.08, -0.6, -0.6), (0.5, 0.5, 0.5), window=1)
    eq = find_equilibria(law, SPEEDS)
    response = windowed_response(law, eq.s, eq.v, windowed())
    one = torch.ones(len(SPEEDS), dtype=torch.float64)
    f_s, f_dv, f_v = law.gains
    torch.testing.assert_close(response["gain"], transfer_discrete(f_s * one, f_dv * one, f_v * one, OMEGA, 0.1).abs())
    assert response["margin_window"][0].item() == pytest.approx(f_v**2 + 2 * f_v * f_dv - 2 * f_s - 0.1 * f_s * f_v)
    q = f_dv + f_v
    roots = np.roots([1.0, -(2.0 + 0.1 * q - 0.01 * f_s), 1.0 + 0.1 * q])
    assert response["radius"][0].item() == pytest.approx(np.abs(roots).max(), rel=1e-12)
    # a damped memoryless law: |G_w| < 0.98 at every audit frequency, M_w > 0.5, rho < 0.995: the penalty is zero
    value, info = windowed_gain_penalty(law, SPEEDS, windowed())
    assert value.item() == 0.0 and info["max_gain"].item() < 0.98 and info["max_radius"].item() < 0.995


# ------------------------------------------------------------------------------------------------ gradients


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
    grads = torch.autograd.grad(value, params, allow_unused=True)  # the output bias does not reach the derivatives
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


GRADIENT_SPEEDS = torch.tensor([10.0, 22.0], dtype=torch.float64)
TERMS = {  # the configuration that leaves one term active (or all three)
    "all": {"window_gain_margin": 0.5, "margin": 5.0, "pole_margin": 0.5},
    "gain_term": {"window_gain_margin": 0.5, "margin": -100.0, "pole_margin": 1.0},
    "margin_term": {"window_gain_margin": -10.0, "margin": 5.0, "pole_margin": 1.0},
    "pole_term": {"window_gain_margin": -10.0, "margin": -100.0, "pole_margin": 0.5},
}


@pytest.mark.parametrize("term", list(TERMS))
def test_windowed_gain_penalty_gradient(term):
    model, cfg = TinyGRU(), windowed(**TERMS[term])
    eq = find_equilibria(model, GRADIENT_SPEEDS)
    assert eq.found.all()
    value, info = windowed_gain_penalty(model, GRADIENT_SPEEDS, cfg, eq)
    active = {key for key in ("gain_term", "margin_term", "pole_term") if info[key].item() > 0.0}
    assert active == ({"gain_term", "margin_term", "pole_term"} if term == "all" else {term})
    check_value_gradient(model, lambda: windowed_gain_penalty(model, GRADIENT_SPEEDS, cfg, eq)[0])


def test_spectral_radius_derivative_equals_that_of_the_eigenvalues():
    """Generic poles: the derivative of the dominant pole alone is the derivative of torch.linalg.eigvals."""
    generator = torch.Generator().manual_seed(3)
    jac = 0.05 * torch.randn(4, 30, 3, generator=generator, dtype=torch.float64)
    jac[:, -1] += torch.tensor([0.05, -0.3, -0.2], dtype=torch.float64)
    jac.requires_grad_(True)
    (mine,) = torch.autograd.grad(spectral_radius(jac, 0.1).sum(), jac)
    reference = closed_loop_poles(jac, 0.1).abs().amax(dim=1)
    (theirs,) = torch.autograd.grad(reference.sum(), jac)
    assert spectral_radius(jac, 0.1).tolist() == pytest.approx(reference.tolist(), rel=1e-13)
    torch.testing.assert_close(mine, theirs, rtol=1e-8, atol=1e-10)


def test_spectral_radius_derivative_at_nearly_defective_poles():
    """A law that reads only its newest state in a window of 30 has a Jordan block of 29 poles at zero, whose
    eigenvectors are parallel; the derivative of the dominant pole is still finite and exact (central differences,
    also for the entries of the old states, which the law ignores)."""
    jac = torch.zeros(1, 30, 3, dtype=torch.float64)
    jac[0, -1] = torch.tensor([0.05, -0.15, -0.2], dtype=torch.float64)
    jac.requires_grad_(True)
    (grad,) = torch.autograd.grad(spectral_radius(jac, 0.1).sum(), jac)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0.0
    h, fd = 1e-7, torch.zeros_like(jac)
    with torch.no_grad():
        for index in [(0, 29, 0), (0, 29, 1), (0, 29, 2), (0, 28, 0), (0, 20, 2), (0, 0, 1)]:
            up, down = jac.clone(), jac.clone()
            up[index] += h
            down[index] -= h
            fd[index] = (spectral_radius(up, 0.1) - spectral_radius(down, 0.1)).sum() / (2 * h)
            assert grad[index].item() == pytest.approx(fd[index].item(), rel=1e-5, abs=1e-9)
    # the memoryless pole of the quadratic: the derivative with respect to the newest entry in closed form
    f = jac[0, -1].detach().clone().requires_grad_(True)
    q = f[1] + f[2]
    b, c = -(2.0 + 0.1 * q - 0.01 * f[0]), 1.0 + 0.1 * q
    root = (-b + torch.sqrt((b**2 - 4.0 * c).to(torch.complex128))) / 2.0
    (expected,) = torch.autograd.grad(root.abs(), f)
    torch.testing.assert_close(grad[0, -1], expected, rtol=1e-7, atol=1e-12)


# ---------------------------------------------------------------- the M_w of full_history_audit.py on stored runs


def script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


STORED = ["e1/follownet_highd/gru/driver_fold0_seed0", "e2_gain_w0.1/follownet_highd/lstm/driver_fold0_seed0",
          "e1/follownet_highd/perl/driver_fold0_seed0"]  # fmt: skip


@pytest.mark.parametrize("run", STORED)
def test_margin_of_the_penalty_equals_that_of_the_full_history_audit(run):
    run_dir = REPO_ROOT / "runs" / run
    if not ((run_dir / "model.pt").is_file() and (run_dir / "stability.json").is_file()):
        pytest.skip(f"stored run {run} absent")
    audit = read_json(run_dir / "stability.json")["audit"]
    model = copy.deepcopy(load_model(run_dir / "model.pt")).double().cpu().eval()
    reference = script("full_history_audit").analyse(model, audit)["equilibria"]
    assert len(reference) > 3
    s = torch.tensor([r["s"] for r in reference], dtype=torch.float64)
    v = torch.tensor([r["v"] for r in reference], dtype=torch.float64)
    response = windowed_response(model.train(), s, v, windowed())
    expected = [r["margin_window"] for r in reference]
    assert response["margin_window"].tolist() == pytest.approx(expected, rel=1e-9, abs=1e-14)
    assert response["radius"].tolist() == pytest.approx([r["spectral_radius"] for r in reference], rel=1e-12)
    torch.testing.assert_close(response["gain"], torch.tensor([r["gain_windowed"] for r in reference], dtype=torch.float64),
                               rtol=1e-9, atol=1e-12)  # fmt: skip
    # the penalty at the stored equilibria (inside the band and outside it): its smallest M_w and largest radius
    n = len(reference)
    eq = Equilibria(v=v, s=s, found=torch.ones(n, dtype=torch.bool), indifferent=torch.zeros(n, dtype=torch.bool),
                    n_crossings=torch.ones(n, dtype=torch.long))  # fmt: skip
    value, info = windowed_gain_penalty(model, v, windowed(), eq)
    assert info["min_margin_window"].item() == pytest.approx(min(expected), rel=1e-9, abs=1e-14)
    assert info["max_radius"].item() == pytest.approx(max(r["spectral_radius"] for r in reference), rel=1e-12)
    stored = run_dir / "full_history.json"
    if stored.is_file():  # the file written by the audit script holds the same values
        on_file = {round(r["v"], 6): r["margin_window"] for r in read_json(stored)["equilibria"]}
        for r in reference:
            assert on_file[round(r["v"], 6)] == pytest.approx(r["margin_window"], rel=1e-9, abs=1e-14)


# ------------------------------------------------------------------------------ band, anchoring and existence


def band_mapping(v, s_low, s_high) -> dict:
    median = [0.5 * (lo + hi) for lo, hi in zip(s_low, s_high)]
    return {"v": list(v), "s_low": list(s_low), "s_median": median, "s_high": list(s_high), "n": [200] * len(v)}


def test_anchored_equilibria_and_the_existence_term_as_in_every_penalty():
    """The equilibrium s_e = 2 + 0.2 v / 0.05 of AMPLIFYING (f_v / f_s = -4 s): inside the band at 8 and 15 m/s, not
    at 23 m/s (the band misses it): that speed leaves the windowed terms and enters the existence term."""
    law = AMPLIFYING
    s_e = [2.0 + 4.0 * v for v in SPEEDS.tolist()]  # 34, 62, 94 m
    band = band_mapping(SPEEDS.tolist(), [s_e[0] - 3, s_e[1] - 3, s_e[2] + 10], [s_e[0] + 3, s_e[1] + 3, s_e[2] + 30])
    cfg = windowed(band=band)
    eq = find_equilibria(law, SPEEDS, band=penalty_band(law, cfg))
    assert eq.status == ["ok", "ok", "outside"]
    value, info = windowed_gain_penalty(law, SPEEDS, cfg)
    existence = existence_penalty(law, SPEEDS, eq.usable, cfg)
    assert info["n_no_equilibrium"].item() == 1.0 and existence.item() > 0.0
    assert info["existence"].item() == pytest.approx(existence.item(), rel=1e-12)
    assert value.item() == pytest.approx(penalty_of(law, cfg) + existence.item(), rel=1e-10)  # the same derivatives
    # without the existence term and without anchored equilibria: zero, with NaN info of the windowed terms
    nothing = band_mapping(SPEEDS.tolist(), [s + 10 for s in s_e], [s + 20 for s in s_e])
    value, info = windowed_gain_penalty(law, SPEEDS, windowed(band=nothing, existence_weight=0.0))
    assert value.item() == 0.0 and info["n_no_equilibrium"].item() == 3.0 and math.isnan(info["max_gain"].item())


# ----------------------------------------------------------------------------------------- training loop


def make_event(i: int, rng: np.random.Generator, n: int = 150) -> Event:
    t = DT * np.arange(n)
    base, amp, period = rng.uniform(8.0, 20.0), rng.uniform(1.0, 4.0), rng.uniform(6.0, 14.0)
    v_lead = base + amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    x_lead = np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    v0 = v_lead[0] + rng.uniform(-2.0, 2.0)
    s0 = idm_equilibrium_spacing(torch.tensor(v0), IDM_P["v0"], IDM_P["T"], IDM_P["s0"]).item() + rng.uniform(-3.0, 5.0)
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, **IDM_P), torch.tensor(x_lead), torch.tensor(v_lead),
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


SMALL_BAND = BandConfig(min_samples=20)
M2_HISTORY_KEYS = {
    "epoch", "loss", "loss_acc", "loss_rollout", "loss_extra", "loss_penalty", "val_rmse_s", "val_rmse_v",
    "val_collision_rate", "time_train_s", "time_val_s",
}  # fmt: skip


def train_config(**kwargs) -> TrainConfig:
    base = dict(
        lr=3e-3, batch_size=256, max_epochs=2, steps_per_epoch=5, patience=100,
        rollout=RolloutConfig(weight=1.0, horizon=30, segments=64), device="cpu", band=SMALL_BAND,
    )  # fmt: skip
    return TrainConfig(**{**base, **kwargs})


def test_windowed_gain_kind_trains(data):
    """``kind: windowed_gain`` on a GRU: its info in the history, the validation penalty of the epoch choice (D89) at
    the grid speeds, the penalty on every ``every``-th step multiplied by ``every``."""
    train, val = data
    torch.manual_seed(0)
    model = WindowGRU(train)
    penalty = windowed(weight=0.5, every=2, n_equilibria=4)
    res = train_model(model, train, val, train_config(penalty=penalty), seed=0)
    assert res["penalty_band"] is True and res["penalty_steps"] == 5 and res["cuda_graph"] is False
    keys = {f"{prefix}_{key}" for prefix in ("penalty", "val_penalty") for key in INFO_KEYS["windowed_gain"]}
    for h in res["history"]:
        assert set(h) == M2_HISTORY_KEYS | {"val_penalty"} | keys
        # NaN in an epoch none of whose penalty steps drew a speed with an anchored equilibrium (epoch 2 here)
        anchored = math.isfinite(h["penalty_max_gain"])
        terms = ("gain_term", "margin_term", "pole_term", "min_margin_window", "max_radius")
        assert all(math.isfinite(h[f"penalty_{k}"]) is anchored for k in terms) and math.isfinite(h["penalty_existence"])
        n = 3 if h["epoch"] == 1 else 2  # steps 0, 2, 4 of epoch 1 and 6, 8 of epoch 2
        parts = 0.5 * 2 * n / 5 * h["loss_penalty"]
        assert h["loss"] == pytest.approx(h["loss_acc"] + h["loss_rollout"] + h["loss_extra"] + parts, rel=1e-12)
    assert math.isfinite(res["history"][0]["penalty_gain_term"]) and res["history"][0]["penalty_margin_term"] > 0.0
    grid = torch.tensor([v for v in V_GRID if 5.0 <= v <= 30.0], dtype=torch.float32)
    settings = dataclasses.replace(penalty, band=training_context(train, 0, SMALL_BAND)["band"])
    value, info = stability_penalty(model, grid, settings)
    best = res["history"][res["best_epoch"] - 1]
    assert best["val_penalty"] == pytest.approx(value.item(), rel=1e-5)
    assert best["val_penalty_margin_term"] == pytest.approx(info["margin_term"].item(), rel=1e-5)


@GPU_ONLY
def test_windowed_gain_steps_are_eager_and_the_plain_steps_from_the_graph(data):
    """On CUDA with ``cuda_graph``: no capture of the penalty steps is attempted (a capture that met
    torch.linalg.eig would invalidate the CUDA context), the steps without penalty are replayed from their graph;
    bitwise the weights of the fully eager training."""
    train, val = data
    runs = []
    # a first eager training in this process (discarded): the first one on CUDA differs from the later ones in the
    # last bits for every penalty kind on this machine (the lazy initialisation of the kernels)
    for graph in (False, False, True):
        torch.manual_seed(0)
        model = WindowGRU(train)
        cfg = train_config(device="cuda", batch_size=64, steps_per_epoch=10, cuda_graph=graph,
                           penalty=windowed(n_equilibria=4, every=3))  # fmt: skip
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = train_model(model, train, val, cfg, seed=0)
        assert not any("capture" in str(w.message) for w in caught)
        assert res["penalty_graph"] is False and res["plain_graph"] is graph and res["cuda_graph"] is graph
        history = [{k: v for k, v in h.items() if not k.startswith("time")} for h in res["history"]]
        runs.append((repr(history), {k: v.detach().clone() for k, v in model.state_dict().items()}))
    (h_eager, w_eager), (h_graph, w_graph) = runs[1:]
    assert h_eager == h_graph and all(torch.equal(w_eager[k], w_graph[k]) for k in w_eager)
    # every = 1: every step is a penalty step and nothing is replayed
    torch.manual_seed(0)
    res = train_model(WindowGRU(train), train, val, train_config(device="cuda", batch_size=64, steps_per_epoch=4,
                      max_epochs=1, penalty=windowed(n_equilibria=4)), seed=0)  # fmt: skip
    assert res["cuda_graph"] is False and res["penalty_graph"] is False and res["plain_graph"] is False
