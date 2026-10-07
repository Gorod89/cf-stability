"""Physics-enhanced residual learning (Long et al. 2023): Newell plus an LSTM residual."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import params_tensor
from cf_stability.models.newell import NEWELL_PARAM_NAMES, newell_acc, newell_fit
from cf_stability.models.recurrent import LSTM


class PERL(CFModel):
    """``a = newell_acc(history, w) + lstm_residual(history)`` over the last ``window`` states.

    The wave speed ``w`` is a buffer (``context.newell_params`` or :meth:`fit`); the head of the
    residual starts at zero, so that the untrained model equals Newell.
    """

    name = "perl"

    def __init__(self, context: ModelContext | None = None, *, hidden_size: int = 64, window: int = 30) -> None:
        super().__init__()
        context = context or ModelContext()
        self.window = int(window)
        self.residual = LSTM(context, hidden_size=hidden_size, window=window)
        nn.init.zeros_(self.residual.head.weight)
        nn.init.zeros_(self.residual.head.bias)
        self.register_buffer("newell_theta", params_tensor(context.newell_params, NEWELL_PARAM_NAMES, torch.float32))

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[:, -self.window :].to(self.residual.head.weight.dtype)
        return newell_acc(x, self.newell_theta[0], self.dt) + self.residual(x)

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Wave speed from ``context.newell_params`` when given, else the calibration on ``events``."""
        summary = newell_fit(events, context, self.window)
        self.newell_theta.copy_(params_tensor(summary["params"], NEWELL_PARAM_NAMES, self.newell_theta.dtype))
        return summary

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "hidden_size": self.residual.hidden_size, "window": self.window}
