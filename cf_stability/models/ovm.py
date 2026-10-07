"""Optimal velocity model with Newell's triangular speed-gap relation, in relaxation form.

``a = (min(v0, (s - s0) / tau) - v) / tau``. On the congested branch ``f_s = 1 / tau^2``,
``f_v = -1 / tau`` and ``f_dv = 0``, so ``f_v^2 + 2 f_v f_dv - 2 f_s = -1 / tau^2 < 0``: the model
is string unstable at every congested equilibrium, whatever its parameters (the known-unstable
reference of the stability analysis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch import Tensor

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import params_tensor

if TYPE_CHECKING:
    from cf_stability.train.calibration import CalibrationConfig

OVM_PARAM_NAMES = ("v0", "tau", "s0")
OVM_BOUNDS: dict[str, tuple[float, float]] = {"v0": (10.0, 45.0), "tau": (0.3, 3.0), "s0": (0.5, 10.0)}


def ovm_acc(s: Tensor, dv: Tensor, v: Tensor, v0: Tensor | float, tau: Tensor | float, s0: Tensor | float) -> Tensor:
    """``(min(v0, (s - s0) / tau) - v) / tau``: relaxation towards the speed the gap allows.

    ``dv`` is not used (kept for the common ``(s, dv, v)`` signature); the result broadcasts
    ``s``, ``v`` and the parameters.
    """
    target = (s - s0) / tau
    return (torch.minimum(target, torch.as_tensor(v0, dtype=target.dtype, device=target.device)) - v) / tau


def ovm_acc_factory(params: Tensor) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Acceleration law of OVM parameters ``[..., 3]`` (broadcast against the states)."""
    v0, tau, s0 = params.unbind(-1)
    return lambda s, dv, v: ovm_acc(s, dv, v, v0, tau, s0)


def calibrate_ovm(events: Sequence[Event], cfg: CalibrationConfig | None = None) -> dict[str, Any]:
    """Global calibration with the objective and the optimiser of the IDM (``calibrate_global``)."""
    from cf_stability.train.calibration import calibrate_global_model  # that module imports this package

    return calibrate_global_model(ovm_acc_factory, OVM_PARAM_NAMES, OVM_BOUNDS, events, cfg)


class OVM(CFModel):
    """OVM with the parameters ``theta = (v0, tau, s0)`` as a buffer.

    The values come from ``context.ovm_params``; when that is empty they are NaN until
    :meth:`fit` or a checkpoint sets them.
    """

    name = "ovm"
    trainable = False

    def __init__(self, context: ModelContext | None = None) -> None:
        super().__init__()
        params = context.ovm_params if context is not None else {}
        self.register_buffer("theta", params_tensor(params, OVM_PARAM_NAMES, torch.float64))

    def acc(self, s: Tensor, dv: Tensor, v: Tensor) -> Tensor:
        v0, tau, s0 = self.theta.unbind()
        return ovm_acc(s, dv, v, v0, tau, s0)

    def forward(self, state_history: Tensor) -> Tensor:
        last = state_history[..., -1, :]
        return self.acc(last[..., 0], last[..., 1], last[..., 2])

    def params_dict(self) -> dict[str, float]:
        return dict(zip(OVM_PARAM_NAMES, self.theta.cpu().tolist()))

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Parameters from ``context.ovm_params`` when given, else the global calibration on ``events``."""
        if context.ovm_params:
            summary = {"params": {n: float(context.ovm_params[n]) for n in OVM_PARAM_NAMES}, "source": "context"}
        else:
            summary = calibrate_ovm(events)
        self.theta.copy_(params_tensor(summary["params"], OVM_PARAM_NAMES, self.theta.dtype))
        return summary

    def config(self) -> dict[str, Any]:
        return {"name": self.name}
