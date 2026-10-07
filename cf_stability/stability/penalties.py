"""Differentiable stability penalties for training (docs/m3_contract.md, section 7; docs/m4_contract.md, 1.3;
docs/m7_contract.md, section 3).

Every penalty takes the model in its training state (dtype, device, mode) and a set of speeds;
the equilibrium spacings come from :func:`find_equilibria` without gradient, the penalty is
differentiable with respect to the parameters at fixed equilibria. With the spacing band of the
training data (``PenaltyConfig.band``) the equilibria are searched inside the band first, the
stability terms are taken at the anchored equilibria only and the existence term demands a
crossing inside the band.

``combined`` (D110) has two parts that the trainer weights and schedules separately
(:func:`combined_parts`): the rollout part (the gain penalty with its existence term, on every
``every``-th step, weight ``weight``) and the Jacobian part (the Jacobian terms of the memoryless
view of the model plus a needle guard, on every step, weight ``jacobian_weight``).

``monotone`` (D117, docs/m8_contract.md, section 4) is the control of the rational (monotonicity)
constraints of RACER (Jiang and Di, arXiv 2312.07003) in the convention ``dv = v - v_lead``:
``relu(-f_s) + relu(f_dv) + relu(f_v)`` at the anchored equilibria, sampled as for ``jacobian``, plus
the existence term of every penalty (D74, D80); no string term and no margin.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.stability.analytic import criterion, partials, transfer_windowed
from cf_stability.stability.equilibrium import (
    BAND_KEYS,
    NOMINAL_S0,
    NOMINAL_T,
    Band,
    Equilibria,
    constant_history,
    find_equilibria,
    model_like,
)
from cf_stability.train.closed_loop import rollout_model

# existence: the existence term alone; combined: gain plus the Jacobian part of the memoryless view (D110);
# monotone: the monotonicity terms of RACER alone, no string term (D117)
KINDS = ("none", "jacobian", "gain", "linear_gain", "existence", "combined", "monotone")
CAPTURABLE = KINDS  # kinds whose training step can be replayed from a CUDA graph: all (D90)
DEFAULT_N_EQUILIBRIA = {
    "none": 0, "jacobian": 16, "gain": 3, "linear_gain": 16, "existence": 16, "combined": 16, "monotone": 16,
}  # fmt: skip
INFO_KEYS: dict[str, tuple[str, ...]] = {
    "none": (),
    "jacobian": ("n_no_equilibrium", "mean_margin", "existence"),
    # mean_margin: the string-stability margin M at the anchored equilibria, reported, not penalised
    "monotone": ("n_no_equilibrium", "mean_margin", "existence"),
    "gain": ("n_no_equilibrium", "max_gain", "existence"),
    "linear_gain": ("n_no_equilibrium", "max_gain", "existence"),
    "existence": ("n_no_equilibrium", "existence"),
    # rollout part (its value, max_gain, existence: NaN on a step without it); Jacobian part (its unweighted
    # value with the guard, mean_margin, the guard terms, needle = largest |f_dv|, |f_v|: NaN while it is off)
    "combined": ("n_no_equilibrium", "rollout", "max_gain", "existence", "jacobian", "mean_margin", "guard", "needle"),
}
JACOBIAN_PART_KEYS = ("jacobian", "mean_margin", "guard", "needle")  # info of the Jacobian part of combined
AMPLITUDE_EPS = 1e-12  # (m/s)^2 under the square root of the fitted amplitude: finite gradient at zero response
EXISTENCE_SPACINGS = (1.0, 200.0)  # m, the ends of the scan of find_equilibria

Info = dict[str, Tensor]


@dataclass
class PenaltyConfig:
    """``train.penalty``: which penalty, its weight ``lambda`` and its settings."""

    kind: str = "none"  # none | jacobian | gain | linear_gain | existence | combined | monotone
    weight: float = 1.0  # lambda (combined: of the rollout part)
    margin: float = 0.5  # m of the Jacobian penalty (not used by monotone)
    gain_margin: float = 0.05  # m_g of the gain penalties
    n_equilibria: int | None = None  # speeds per epoch; None: 16 (jacobian, linear_gain), 3 (gain)
    v_min: float = 5.0  # m/s
    v_max: float = 30.0
    omegas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)  # rad/s
    horizon_s: float = 40.0  # rollout of the gain penalty after the warm-up
    measure_s: float = 20.0  # last part of the rollout that gives the gain
    amplitude: float = 0.2  # m/s, amplitude of the leader speed
    # A penalty that is evaluated at the equilibria only can be satisfied by having none. Without
    # band (D74): at every speed in [exist_v_min, exist_v_max] without equilibrium the law must
    # brake at the smallest and accelerate at the largest spacing of the scan:
    # relu(f(1 m, 0, v) + m_e) + relu(m_e - f(200 m, 0, v)). With band: see `existence`.
    existence_weight: float = 1.0  # relative to the stability terms; 0 switches the term off
    existence_margin: float = 0.1  # m_e, m/s^2
    exist_v_min: float | None = None  # None: v_min; the trainer sets the speed range of the training data
    exist_v_max: float | None = None
    # linear_gain only: the frequencies of the audit, because a law penalised on a few frequencies
    # moves its amplification below the lowest of them; the margin shrinks towards w = 0, where
    # every gain tends to 1: m_g(w) = gain_margin * min(1, (w / gain_margin_ref)^2)
    linear_omega_min: float = 0.02  # rad/s
    linear_omega_max: float = 2.0
    linear_n_omega: int = 25  # log-spaced
    gain_margin_ref: float = 0.1  # rad/s
    # band: brake at the lower and accelerate at the upper edge of the spacing band at every sampled
    # speed that has one (D80); fixed: the term of D74; band falls back to fixed without a band
    existence: str = "band"
    # linear_gain (D81) and gain (D90): max: mean over the equilibria of relu(max_w (gain - 1 + m_g)),
    # so that a narrow amplification is not averaged away; mean: mean over equilibria and frequencies (M3)
    aggregate: str = "max"
    # choice of the best epoch under a penalty (D89): an epoch is feasible when the penalty at the
    # grid speeds is at most `tolerance`; before the first feasible epoch a fall of that penalty by
    # the share `progress` restarts the patience as well
    tolerance: float = 0.01
    progress: float = 0.05
    # the penalty enters the loss on every `every`-th training step, multiplied by `every`: its mean
    # weight over the steps stays `weight` (docs/m4_contract.md, 1.6); 1: every step. combined: the
    # rollout part only, the Jacobian part enters every step
    every: int = 1
    # spacing band of the training data, the "band" mapping of the training context (plain lists,
    # JSON-serialisable; set by the trainer); None: no band, the equilibria of M3
    band: Mapping[str, Any] | None = None
    # combined only (D110): weight of the Jacobian part (Jacobian terms of the memoryless view at the
    # anchored equilibria, every training step; 0: off) and the threshold of its needle guard
    # relu(|f_dv| - guard) + relu(|f_v| - guard) in 1/s (0: off). Late keys of the config: the
    # config hash leaves them out at these defaults (cf_stability/utils.py, LATE_KEYS)
    jacobian_weight: float = 0.0
    guard: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"penalty kind must be one of {KINDS}, got {self.kind!r}")
        if self.existence not in ("band", "fixed") or self.aggregate not in ("max", "mean"):
            raise ValueError("existence must be band or fixed, aggregate max or mean")
        if self.band is not None:
            missing = [key for key in BAND_KEYS if key not in self.band]
            if missing:
                raise ValueError(f"penalty band lacks {missing}")
        if self.n_equilibria is None:
            self.n_equilibria = DEFAULT_N_EQUILIBRIA[self.kind]
        self.n_equilibria = int(self.n_equilibria)
        self.omegas = tuple(float(w) for w in self.omegas)
        if not 0.0 < self.measure_s <= self.horizon_s:
            raise ValueError("need 0 < measure_s <= horizon_s")
        self.every = int(self.every)
        if self.every < 1:
            raise ValueError(f"every must be a positive number of steps, got {self.every}")
        self.jacobian_weight, self.guard = float(self.jacobian_weight), float(self.guard)
        if not (self.jacobian_weight >= 0.0 and self.guard >= 0.0):
            raise ValueError("jacobian_weight and guard must be >= 0 (0: off)")
        if self.kind != "combined" and (self.jacobian_weight or self.guard):
            raise ValueError(f"jacobian_weight and guard belong to kind combined (D110), not to {self.kind!r}")
        if self.guard and not self.jacobian_weight:
            raise ValueError("the needle guard is part of the Jacobian part of combined: it needs jacobian_weight > 0")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "PenaltyConfig":
        values = dict(mapping or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown penalty keys: {sorted(unknown)}")
        return cls(**values)

    @property
    def active(self) -> bool:
        return self.kind != "none"


def sample_speeds(
    cfg: PenaltyConfig, generator: torch.Generator, device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:  # fmt: skip
    """``cfg.n_equilibria`` speeds uniform in ``[v_min, v_max]``, drawn on the CPU from ``generator``."""
    u = torch.rand(cfg.n_equilibria, generator=generator, dtype=torch.float64)
    return (cfg.v_min + (cfg.v_max - cfg.v_min) * u).to(device=device, dtype=dtype)


def _zero(model: CFModel) -> Tensor:
    device, dtype = model_like(model)
    return torch.zeros((), device=device, dtype=dtype)


@functools.lru_cache(maxsize=32)
def _band_on(columns: tuple[tuple[float, ...], ...], device: torch.device, dtype: torch.dtype) -> Band:
    """The band of ``columns`` (``BAND_KEYS``) on ``device``, created once: a host-to-device copy cannot run
    inside a CUDA graph."""
    return Band.from_mapping(dict(zip(BAND_KEYS, columns)), device, dtype)


def penalty_band(model: CFModel, cfg: PenaltyConfig) -> Band | None:
    """``cfg.band`` as a :class:`Band` on the device and in the dtype of ``model`` (None without band).

    The tensors are created by the first call per device and dtype (the eager warm-up steps of
    the trainer) and reused afterwards, also inside a CUDA graph capture.
    """
    if cfg.band is None:
        return None
    columns = tuple(tuple(float(x) for x in cfg.band[key]) for key in BAND_KEYS)
    return _band_on(columns, *model_like(model))


def _equilibria(model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None) -> Equilibria:
    """``equilibria``, or those of ``model`` at ``speeds`` found with the band of ``cfg``."""
    return find_equilibria(model, speeds, band=penalty_band(model, cfg)) if equilibria is None else equilibria


def _anchored(eq: Equilibria) -> tuple[Tensor, Tensor, Tensor]:
    """Detached spacing and speed at every speed and the mask of the anchored equilibria.

    Speeds without an anchored equilibrium get the nominal spacing: finite values that are masked
    out. No shape depends on the data, so that the penalty can be captured in a CUDA graph.
    """
    anchored = eq.anchored
    s = torch.where(anchored, eq.s, NOMINAL_S0 + NOMINAL_T * eq.v)
    return s.detach(), eq.v.detach(), anchored


def _none_in(mask: Tensor) -> bool:
    """No true entry in ``mask``, checked only outside a CUDA graph capture (the check synchronises)."""
    capturing = mask.is_cuda and torch.cuda.is_current_stream_capturing()
    return not capturing and not bool(mask.any())


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values [n, ...]`` over the rows where ``mask [n]`` holds; 0 without any."""
    mask = mask.reshape(mask.shape + (1,) * (values.dim() - 1)).expand_as(values)
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp(min=1)


def existence_penalty(model: CFModel, v: Tensor, usable: Tensor, cfg: PenaltyConfig) -> Tensor:
    """``existence_weight`` x the term that demands an equilibrium at the speeds ``v``.

    With a band (``cfg.band`` and ``existence = "band"``, D80): mean over the speeds that have a
    band of ``relu(f(s_low(v), 0, v) + m_e) + relu(m_e - f(s_high(v), 0, v))``, whatever their
    equilibria: a law that brakes at the lower edge and accelerates at the upper edge has an
    upward crossing inside the band. Otherwise (D74): mean over the speeds without a usable
    equilibrium (``usable`` false) inside ``[exist_v_min, exist_v_max]`` of
    ``relu(f(1 m, 0, v) + m_e) + relu(m_e - f(200 m, 0, v))``.

    Evaluated at every speed and masked, so that no shape depends on the data (CUDA graph).
    """
    if cfg.existence_weight <= 0:
        return _zero(model)
    band = penalty_band(model, cfg) if cfg.existence == "band" else None
    if band is None:
        lo = cfg.v_min if cfg.exist_v_min is None else cfg.exist_v_min
        hi = cfg.v_max if cfg.exist_v_max is None else cfg.exist_v_max
        mask = ~usable & (v >= lo) & (v <= hi)
        near, far = (torch.full_like(v, s) for s in EXISTENCE_SPACINGS)
    else:
        near, far, mask = band.at(v)
    with torch.backends.cudnn.flags(enabled=False):
        f = model(constant_history(model, torch.cat((near, far)), torch.cat((v, v))))
    f_near, f_far = f[: len(v)], f[len(v) :]
    terms = torch.relu(f_near + cfg.existence_margin) + torch.relu(cfg.existence_margin - f_far)
    return cfg.existence_weight * _masked_mean(terms, mask)


def memoryless_partials(model: CFModel, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(f_s, f_dv, f_v)`` of the memoryless view of ``model`` at the constant states ``(s, 0, v)``, each
    ``[n]``, differentiable with respect to the parameters (docs/m3_contract.md, section 1; D110).

    The window is filled with the constant state, as the equilibrium finder evaluates the law
    (:func:`~cf_stability.stability.equilibrium.steady_acc`), and differentiated with respect to a
    common shift of all its entries: the sums over the window positions of the history Jacobian,
    i.e. the static gains, the limit ``w -> 0`` of the linearised window response. For a memoryless
    model (window 1) they are its partial derivatives.
    """
    return partials(model, s, v, create_graph=True)


def _jacobian_terms(f_s: Tensor, f_dv: Tensor, f_v: Tensor, cfg: PenaltyConfig) -> tuple[Tensor, Tensor]:
    """``relu(m - M) + relu(-f_s) + relu(f_dv) + relu(f_v + f_dv)`` per equilibrium and the margin ``M``."""
    margin = criterion(f_s, f_dv, f_v)["margin"]
    relu = torch.relu
    return relu(cfg.margin - margin) + relu(-f_s) + relu(f_dv) + relu(f_v + f_dv), margin


def jacobian_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """Mean over the anchored equilibria of ``relu(m - M) + relu(-f_s) + relu(f_dv) + relu(f_v + f_dv)``,
    plus the existence term (:func:`existence_penalty`). The derivatives are those of the memoryless
    view (:func:`memoryless_partials`).

    ``equilibria`` (at ``speeds``) are computed with the band of ``cfg`` when not given. ``info``:
    ``n_no_equilibrium`` (speeds without anchored equilibrium), ``mean_margin``, ``existence`` (detached).
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    s, v, anchored = _anchored(eq)
    nan = torch.full((), float("nan"), dtype=torch.float64, device=s.device)
    existence = existence_penalty(model, v, eq.usable, cfg)
    info = {
        "n_no_equilibrium": (~anchored).sum().double(),
        "mean_margin": nan,
        "existence": existence.detach().double(),
    }
    if _none_in(anchored):
        return existence, info
    terms, margin = _jacobian_terms(*memoryless_partials(model, s, v), cfg)
    info["mean_margin"] = torch.where(anchored.any(), _masked_mean(margin.detach().double(), anchored), nan)
    return _masked_mean(terms, anchored) + existence, info


def monotone_terms(f_s: Tensor, f_dv: Tensor, f_v: Tensor) -> Tensor:
    """``relu(-f_s) + relu(f_dv) + relu(f_v)`` per equilibrium: the rational constraints of RACER (Jiang and
    Di, arXiv 2312.07003: the acceleration rises with the spacing, falls with the speed difference
    ``dv = v - v_lead`` and with the own speed), with no string term and no margin (D117)."""
    relu = torch.relu
    return relu(-f_s) + relu(f_dv) + relu(f_v)


def monotone_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """Mean over the anchored equilibria of :func:`monotone_terms`, plus the existence term
    (:func:`existence_penalty`): :func:`jacobian_penalty` with the local terms of RACER in place of the
    margin and the local terms of the specification (D117). The derivatives are those of the memoryless
    view (:func:`memoryless_partials`); speeds without an anchored equilibrium are masked out.

    ``equilibria`` (at ``speeds``) are computed with the band of ``cfg`` when not given. ``info``:
    ``n_no_equilibrium``, ``mean_margin`` (the string-stability margin ``M`` at the anchored equilibria:
    reported, not penalised), ``existence`` (detached).
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    s, v, anchored = _anchored(eq)
    nan = torch.full((), float("nan"), dtype=torch.float64, device=s.device)
    existence = existence_penalty(model, v, eq.usable, cfg)
    info = {
        "n_no_equilibrium": (~anchored).sum().double(),
        "mean_margin": nan,
        "existence": existence.detach().double(),
    }
    if _none_in(anchored):
        return existence, info
    f_s, f_dv, f_v = memoryless_partials(model, s, v)
    margin = criterion(f_s.detach(), f_dv.detach(), f_v.detach())["margin"]
    info["mean_margin"] = torch.where(anchored.any(), _masked_mean(margin.double(), anchored), nan)
    return _masked_mean(monotone_terms(f_s, f_dv, f_v), anchored) + existence, info


def memoryless_jacobian_part(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """The Jacobian part of ``combined`` (D110), unweighted: mean over the anchored equilibria of the terms
    of :func:`jacobian_penalty` (margin ``m``, local terms) with the derivatives of the memoryless view
    (:func:`memoryless_partials`), plus with ``guard > 0`` the needle guard
    ``relu(|f_dv| - guard) + relu(|f_v| - guard)``. No existence term: in ``combined`` it belongs to the
    rollout part.

    ``equilibria`` (at ``speeds``) are computed with the band of ``cfg`` when not given. ``info``
    (detached): ``jacobian`` (the value), ``mean_margin``, ``guard`` (mean of the guard terms; NaN with
    ``guard = 0``), ``needle`` (largest ``|f_dv|``, ``|f_v|`` at the anchored equilibria, 1/s). Speeds
    without an anchored equilibrium are masked out as in :func:`jacobian_penalty` (CUDA graph).
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    s, v, anchored = _anchored(eq)
    zero, nan = _zero(model), torch.full((), float("nan"), dtype=torch.float64, device=s.device)
    guard_info = zero.double() if cfg.guard > 0 else nan
    info = {"jacobian": zero.double(), "mean_margin": nan, "guard": guard_info, "needle": nan}
    if _none_in(anchored):
        return zero, info
    f_s, f_dv, f_v = memoryless_partials(model, s, v)
    terms, margin = _jacobian_terms(f_s, f_dv, f_v, cfg)
    some = anchored.any()
    info["mean_margin"] = torch.where(some, _masked_mean(margin.detach().double(), anchored), nan)
    needle = torch.maximum(f_dv.abs(), f_v.abs()).detach().double()
    info["needle"] = torch.where(some, torch.where(anchored, needle, -torch.inf).max(), nan)
    if cfg.guard > 0:
        guard = torch.relu(f_dv.abs() - cfg.guard) + torch.relu(f_v.abs() - cfg.guard)
        info["guard"] = _masked_mean(guard.detach().double(), anchored)
        terms = terms + guard
    value = _masked_mean(terms, anchored)
    info["jacobian"] = value.detach().double()
    return value, info


def _projection(omega: Tensor, t: Tensor) -> Tensor:
    """Least-squares projection ``[B, 3, M]`` of a series sampled at ``t [M]`` on ``sin(w t), cos(w t), 1``."""
    phase = omega[:, None] * t[None, :]
    design = torch.stack((torch.sin(phase), torch.cos(phase), torch.ones_like(phase)), dim=-1)
    return torch.linalg.pinv(design)


def gain_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """Excess ``gain - 1 + m_g`` of two-vehicle rollouts at the anchored equilibria and the frequencies.

    ``aggregate = "max"``: mean over the equilibria of ``relu(max_w excess)``; ``"mean"``: mean
    over equilibria and frequencies of ``relu(excess)`` (M3). Leader speed ``v_e + A sin(w t)``
    from ``t = 0``, leader position by the integration scheme; the follower history holds the
    equilibrium for ``window`` samples before (the warm-up), then the follower is rolled out for
    ``horizon_s``. The gain is the amplitude of the follower speed fitted on ``sin, cos, 1`` over
    the last ``measure_s`` divided by ``A``. ``info``: ``n_no_equilibrium``, ``max_gain``,
    ``existence`` (detached).

    Every speed is rolled out, those without anchored equilibrium at the nominal spacing, masked
    out of the penalty and of ``max_gain``: no shape depends on the data, so that the penalty step
    can be captured in a CUDA graph (D90). The rollout uses cuDNN as the rollout loss of the
    training step does; only a backward in eval mode, which cuDNN RNNs lack, runs without it.
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    s, v, anchored = _anchored(eq)
    nan = torch.full((), float("nan"), dtype=torch.float64, device=s.device)
    existence = existence_penalty(model, v, eq.usable, cfg)
    info = {"n_no_equilibrium": (~anchored).sum().double(), "max_gain": nan, "existence": existence.detach().double()}
    if _none_in(anchored):
        return existence, info
    device, dtype, dt, window = s.device, s.dtype, model.dt, model.window
    n_steps, n_meas, n_w = round(cfg.horizon_s / dt), round(cfg.measure_s / dt), len(cfg.omegas)
    # sample k is at t = (k - window + 1) dt: the warm-up ends at t = 0, where the excitation starts
    omega_w, t, projection = _gain_design(cfg.omegas, dt, window, n_steps, n_meas, device, dtype)
    omega = omega_w.repeat(len(s))  # [n x n_w], the frequencies of one speed together
    s0, v0 = s.repeat_interleave(n_w), v.repeat_interleave(n_w)
    v_lead = v0.double()[:, None] + cfg.amplitude * torch.sin(omega[:, None] * t[None, :])
    x_lead = dt * (torch.cumsum(v_lead, dim=-1) - v_lead[:, :1])  # x[k] = x[k-1] + dt v[k]
    cudnn_ok = model.training or not torch.is_grad_enabled()
    with contextlib.nullcontext() if cudnn_ok else torch.backends.cudnn.flags(enabled=False):
        res = rollout_model(
            model, x_lead.to(dtype), v_lead.to(dtype), s0[:, None].expand(-1, window), v0[:, None].expand(-1, window),
            dt=dt,
        )  # fmt: skip
    deviation = (res.v[:, -n_meas:] - v0[:, None]).reshape(len(s), n_w, n_meas)
    coef = torch.einsum("wkm,nwm->nwk", projection, deviation)
    gain = torch.sqrt(coef[..., 0] ** 2 + coef[..., 1] ** 2 + AMPLITUDE_EPS) / cfg.amplitude  # [n, n_w]
    largest = torch.where(anchored[:, None], gain.detach().double(), -torch.inf).max()
    info["max_gain"] = torch.where(anchored.any(), largest, nan)
    excess = gain - 1.0 + cfg.gain_margin
    terms = torch.relu(excess.amax(dim=1)) if cfg.aggregate == "max" else torch.relu(excess)
    return _masked_mean(terms, anchored) + existence, info


def window_jacobian(model: CFModel, s: Tensor, v: Tensor) -> Tensor:
    """:func:`~cf_stability.stability.analytic.history_jacobian` with ``create_graph``: ``[n, window, 3]``,
    differentiable with respect to the parameters."""
    history = constant_history(model, s.detach(), v.detach()).clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        (jac,) = torch.autograd.grad(model(history).sum(), history, create_graph=True, allow_unused=True)
    return torch.zeros_like(history) if jac is None else jac


@functools.lru_cache(maxsize=None)
def _gain_design(
    omegas: tuple[float, ...], dt: float, window: int, n_steps: int, n_meas: int, device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    """Frequencies ``[n_w]``, sample times ``[window + n_steps]`` and the least-squares projection
    ``[n_w, 3, n_meas]`` of the measured part of the gain penalty, created once per device: neither
    a host-to-device copy nor the pseudo-inverse (SVD) can run inside a CUDA graph."""
    omega = torch.tensor(omegas, dtype=torch.float64, device=device)
    t = dt * torch.arange(1 - window, n_steps + 1, dtype=torch.float64, device=device).clamp(min=0.0)
    return omega, t, _projection(omega, t[-n_meas:]).to(dtype)


@functools.lru_cache(maxsize=None)
def _linear_grid(
    omega_min: float, omega_max: float, n: int, margin: float, ref: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Frequencies of the linearised gain penalty and their margins, created once per device."""
    omega = torch.logspace(math.log10(omega_min), math.log10(omega_max), n, dtype=torch.float64, device=device)
    return omega, margin * torch.clamp((omega / ref) ** 2, max=1.0)


def linear_gain_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """Excess ``|G_d(w)| - 1 + m_g(w)`` of the linearised window response at the anchored equilibria.

    ``aggregate = "max"``: mean over the equilibria of ``relu(max_w excess)``; ``"mean"``: mean
    over equilibria and frequencies of ``relu(excess)``. Frequencies ``linear_n_omega`` log-spaced
    in ``[linear_omega_min, linear_omega_max]``, ``m_g(w) = gain_margin * min(1, (w / gain_margin_ref)^2)``.
    No rollout; the transfer function is evaluated in complex128. ``info``: ``n_no_equilibrium``,
    ``max_gain``, ``existence`` (detached). Speeds without an anchored equilibrium as in
    :func:`jacobian_penalty`.
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    s, v, anchored = _anchored(eq)
    nan = torch.full((), float("nan"), dtype=torch.float64, device=s.device)
    existence = existence_penalty(model, v, eq.usable, cfg)
    info = {"n_no_equilibrium": (~anchored).sum().double(), "max_gain": nan, "existence": existence.detach().double()}
    if _none_in(anchored):
        return existence, info
    omega, margin = _linear_grid(
        cfg.linear_omega_min, cfg.linear_omega_max, cfg.linear_n_omega, cfg.gain_margin, cfg.gain_margin_ref, s.device
    )
    gain = transfer_windowed(window_jacobian(model, s, v).double(), omega, model.dt).abs()
    info["max_gain"] = torch.where(anchored.any(), torch.where(anchored[:, None], gain.detach(), -torch.inf).max(), nan)
    excess = gain - 1.0 + margin  # [n, n_w]
    terms = torch.relu(excess.amax(dim=1)) if cfg.aggregate == "max" else torch.relu(excess)
    return _masked_mean(terms, anchored).to(s.dtype) + existence, info


def existence_only_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """The existence term of ``cfg`` alone (:func:`existence_penalty`: band edges or D74), no stability
    term: an equilibrium inside the band without any demand on its stability. ``info``:
    ``n_no_equilibrium`` (speeds without anchored equilibrium), ``existence`` (detached)."""
    eq = _equilibria(model, speeds, cfg, equilibria)
    existence = existence_penalty(model, eq.v.detach(), eq.usable, cfg)
    return existence, {"n_no_equilibrium": (~eq.anchored).sum().double(), "existence": existence.detach().double()}


def combined_parts(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None, *, rollout: bool = True
) -> tuple[Tensor, Tensor, Info]:
    """The two parts of ``combined`` (D110), unweighted, at the equilibria of ``speeds`` (found once, with
    the band of ``cfg``, when not given): ``(rollout part, Jacobian part, info)``.

    Rollout part: :func:`gain_penalty` with its existence term (the penalty of the recurrent models in
    E2, D90); zero without ``rollout``. Jacobian part: :func:`memoryless_jacobian_part`; zero while
    ``jacobian_weight`` is 0. The trainer adds ``weight * every`` times the rollout part on every
    ``every``-th step and ``jacobian_weight`` times the Jacobian part on every step. ``info``: the keys
    of ``INFO_KEYS["combined"]``, NaN for a part that is not evaluated.
    """
    eq = _equilibria(model, speeds, cfg, equilibria)
    nan = torch.full((), float("nan"), dtype=torch.float64, device=eq.v.device)
    if rollout:
        rollout_value, info = gain_penalty(model, speeds, cfg, eq)
        info["rollout"] = rollout_value.detach().double()
    else:
        rollout_value = _zero(model)
        info = {"n_no_equilibrium": (~eq.anchored).sum().double(), "rollout": nan, "max_gain": nan, "existence": nan}
    if cfg.jacobian_weight > 0:
        jacobian_value, jacobian_info = memoryless_jacobian_part(model, speeds, cfg, eq)
    else:
        jacobian_value, jacobian_info = _zero(model), {key: nan for key in JACOBIAN_PART_KEYS}
    return rollout_value, jacobian_value, {**info, **jacobian_info}


def combined_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """``combined`` (D110) as one number: the rollout part plus the Jacobian part, both unweighted
    (:func:`combined_parts`). It is the penalty of the choice of the best epoch (D89)."""
    rollout_value, jacobian_value, info = combined_parts(model, speeds, cfg, equilibria)
    return rollout_value + jacobian_value, info


PENALTIES: dict[str, Callable[..., tuple[Tensor, Info]]] = {
    "jacobian": jacobian_penalty,
    "gain": gain_penalty,
    "linear_gain": linear_gain_penalty,
    "existence": existence_only_penalty,
    "combined": combined_penalty,
    "monotone": monotone_penalty,
}


def stability_penalty(
    model: CFModel, speeds: Tensor, cfg: PenaltyConfig, equilibria: Equilibria | None = None
) -> tuple[Tensor, Info]:
    """The penalty ``cfg.kind`` (zero and empty ``info`` for ``"none"``)."""
    if not cfg.active:
        return _zero(model), {}
    return PENALTIES[cfg.kind](model, speeds, cfg, equilibria)
