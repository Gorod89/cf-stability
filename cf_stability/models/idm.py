"""Intelligent Driver Model (Treiber, Hennecke, Helbing 2000) in torch."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel, ModelContext

IDM_PARAM_NAMES = ("v0", "T", "s0", "a", "b")
IDM_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (10.0, 45.0),
    "T": (0.3, 3.0),
    "s0": (0.5, 10.0),
    "a": (0.2, 4.0),
    "b": (0.5, 6.0),
}
BOUND_MARGIN = 1e-6  # share of the range between a bounded initial value and its bound (finite logits)

ParamValues = Mapping[str, float] | Sequence[float] | Tensor


def params_tensor(params: ParamValues, names: Sequence[str], dtype: torch.dtype) -> Tensor:
    """Parameter vector ``[len(names)]``; an empty mapping gives NaN (set later by ``fit`` or a checkpoint)."""
    if isinstance(params, Mapping):
        params = [float(params[name]) for name in names] if params else [float("nan")] * len(names)
    theta = torch.as_tensor(params, dtype=dtype).detach().clone()
    if theta.shape != (len(names),):
        raise ValueError(f"expected {len(names)} parameters, got shape {tuple(theta.shape)}")
    return theta


def idm_acc(
    s: Tensor,
    dv: Tensor,
    v: Tensor,
    v0: Tensor | float,
    T: Tensor | float,
    s0: Tensor | float,
    a: Tensor | float,
    b: Tensor | float,
    delta: float = 4.0,
    s_eps: float = 0.01,
) -> Tensor:
    """IDM acceleration for ``dv = v - v_lead``; all arguments broadcast.

    The spacing is clamped to ``>= s_eps`` so that rollouts continue after a collision.
    """
    s_star = s0 + torch.clamp(v * T + v * dv / (2.0 * (a * b) ** 0.5), min=0.0)
    return a * (1.0 - (v / v0) ** delta - (s_star / torch.clamp(s, min=s_eps)) ** 2)


def idm_equilibrium_spacing(
    v: Tensor, v0: Tensor | float, T: Tensor | float, s0: Tensor | float, delta: float = 4.0
) -> Tensor:
    """Spacing with zero acceleration at ``dv = 0``; ``inf`` for ``v >= v0``."""
    return (s0 + v * T) / torch.sqrt(torch.clamp(1.0 - (v / v0) ** delta, min=0.0))


def idm_equilibrium_partials(
    v: Tensor,
    v0: Tensor | float,
    T: Tensor | float,
    s0: Tensor | float,
    a: Tensor | float,
    b: Tensor | float,
    delta: float = 4.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(s_e, f_s, f_dv, f_v)`` of the IDM at its equilibrium for the speed ``v`` (closed form, ``dv = v - v_lead``).

    With ``s_star = s0 + v T``: ``f_s = 2 a s_star^2 / s_e^3``,
    ``f_dv = -a s_star v / (s_e^2 sqrt(a b))``, ``f_v = -a (delta v^(delta-1) / v0^delta + 2 s_star T / s_e^2)``.
    NaN where ``v >= v0`` (no equilibrium). All arguments broadcast.
    """
    free = 1.0 - (v / v0) ** delta
    s_star = s0 + v * T
    s_e = s_star / torch.sqrt(torch.clamp(free, min=1e-12))
    f_s = 2.0 * a * s_star**2 / s_e**3
    f_dv = -a * s_star * v / (s_e**2 * (a * b) ** 0.5)
    f_v = -a * (delta * v ** (delta - 1.0) / v0**delta + 2.0 * s_star * T / s_e**2)
    nan = torch.full_like(f_s, float("nan"))
    exists = free > 0
    return tuple(torch.where(exists, x, nan) for x in (s_e, f_s, f_dv, f_v))  # type: ignore[return-value]


def idm_margin(v: Tensor, v0: Tensor, T: Tensor, s0: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """String-stability margin ``f_v^2 + 2 f_v f_dv - 2 f_s`` of the IDM at the speed ``v``; NaN without equilibrium."""
    _, f_s, f_dv, f_v = idm_equilibrium_partials(v, v0, T, s0, a, b)
    return f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s


class IDM(CFModel):
    """IDM with the parameter vector ``theta = (v0, T, s0, a, b)`` held in one ``nn.Parameter`` ``raw``.

    Plain: ``raw`` is ``theta``. Bounded: ``raw`` holds logits that a sigmoid maps into
    ``IDM_BOUNDS``. Without ``params`` the values come from ``context.idm_params``; when that
    is empty they are NaN until :meth:`fit` or a checkpoint sets them.
    """

    name = "idm"
    trainable = False

    def __init__(
        self,
        params: ParamValues | None = None,
        *,
        context: ModelContext | None = None,
        learnable: bool = False,
        bounded: bool = False,
        delta: float = 4.0,
        s_eps: float = 0.01,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if params is None:
            params = context.idm_params if context is not None else {}
        theta = params_tensor(params, IDM_PARAM_NAMES, dtype)
        self.learnable, self.bounded = bool(learnable), bool(bounded)
        self.trainable = self.learnable
        self.delta, self.s_eps = float(delta), float(s_eps)
        if self.bounded:
            for k, bound in enumerate(("lower", "upper")):
                self.register_buffer(bound, torch.tensor([IDM_BOUNDS[n][k] for n in IDM_PARAM_NAMES], dtype=dtype))
        self.raw = nn.Parameter(self._to_raw(theta), requires_grad=self.learnable)

    def _to_raw(self, theta: Tensor) -> Tensor:
        if not self.bounded:
            return theta
        u = (theta - self.lower) / (self.upper - self.lower)
        return torch.logit(u.clamp(BOUND_MARGIN, 1.0 - BOUND_MARGIN))

    @property
    def theta(self) -> Tensor:
        """``(v0, T, s0, a, b)``; the leaf parameter itself unless bounded."""
        if self.bounded:
            return torch.lerp(self.lower, self.upper, torch.sigmoid(self.raw))  # lerp is exact at both ends
        return self.raw

    def set_params(self, params: ParamValues) -> None:
        theta = params_tensor(params, IDM_PARAM_NAMES, self.raw.dtype).to(self.raw.device)
        with torch.no_grad():
            self.raw.copy_(self._to_raw(theta))

    def acc(self, s: Tensor, dv: Tensor, v: Tensor) -> Tensor:
        v0, T, s0, a, b = self.theta.unbind()
        return idm_acc(s, dv, v, v0, T, s0, a, b, self.delta, self.s_eps)

    def forward(self, state_history: Tensor) -> Tensor:
        last = state_history[..., -1, :]
        return self.acc(last[..., 0], last[..., 1], last[..., 2])

    def params_dict(self) -> dict[str, float]:
        return dict(zip(IDM_PARAM_NAMES, self.theta.detach().cpu().tolist()))

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Parameters from ``context.idm_params`` when given, else the global calibration on ``events``."""
        if context.idm_params:
            summary = {"params": {n: float(context.idm_params[n]) for n in IDM_PARAM_NAMES}, "source": "context"}
        else:
            from cf_stability.train.calibration import calibrate_global  # that module imports this one

            summary = calibrate_global(events)
        self.set_params(summary["params"])
        return summary

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "learnable": self.learnable,
            "bounded": self.bounded,
            "delta": self.delta,
            "s_eps": self.s_eps,
        }
