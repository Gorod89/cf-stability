"""IDM plus a bounded residual with a certified Jacobian bound."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from cf_stability.models.base import CFModel, InputScaler, ModelContext
from cf_stability.models.idm import IDM
from cf_stability.models.mlp import tanh_mlp


def _antisymmetric_init(hidden: nn.Linear, out: nn.Linear) -> None:
    """Duplicate the units of ``hidden`` and give ``out`` opposite weights on the copies: output 0 for every input.

    Spectral normalisation divides a weight by its norm, so scaling the weights cannot make the
    initial output small; cancelling pairs of units can.
    """
    half = hidden.out_features // 2
    with torch.no_grad():
        hidden.weight[half:] = hidden.weight[:half]
        hidden.bias[half:] = hidden.bias[:half]
        out.weight[:, half:] = -out.weight[:, :half]
        out.bias.zero_()


class ResidualIDM(CFModel):
    """``a = idm(theta) + r_max * tanh(lipschitz * g(x_tilde))`` on the last state.

    ``g`` is a tanh MLP with spectral normalisation on every linear layer and ``x_tilde`` comes
    from the fixed physical scaler, so that the Jacobian bound of the residual does not depend on
    the dataset. The IDM parameters come from ``context.idm_params``; they are frozen, or
    learnable in bounded form with ``idm_learnable``. The untrained residual is zero.
    """

    name = "residual_idm"

    def __init__(
        self,
        context: ModelContext | None = None,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        r_max: float = 1.0,
        lipschitz: float = 1.0,
        idm_learnable: bool = False,
    ) -> None:
        super().__init__()
        context = context or ModelContext()
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if self.hidden_sizes[-1] % 2:
            raise ValueError(f"the last hidden layer needs an even width, got {self.hidden_sizes[-1]}")
        self.r_max, self.lipschitz, self.idm_learnable = float(r_max), float(lipschitz), bool(idm_learnable)
        self.scaler = InputScaler.physical()
        self.idm = IDM(context=context, learnable=idm_learnable, bounded=idm_learnable, dtype=torch.float32)
        self.g = tanh_mlp((3, *self.hidden_sizes, 1))
        linears = self._linears()
        _antisymmetric_init(linears[-2], linears[-1])
        for layer in linears:
            spectral_norm(layer)

    def _linears(self) -> list[nn.Linear]:
        return [m for m in self.g if isinstance(m, nn.Linear)]

    def _dtype(self) -> torch.dtype:
        return self.g[0].bias.dtype

    def residual(self, state_history: Tensor) -> Tensor:
        """``r_max * tanh(lipschitz * g(x_tilde))`` of the last state, ``[B]``."""
        x = self.scaler(state_history[..., -1, :].to(self._dtype()))
        return self.r_max * torch.tanh(self.lipschitz * self.g(x).squeeze(-1))

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history.to(self._dtype())
        return self.idm(x) + self.residual(x)

    def layer_norms(self) -> list[float]:
        """Exact spectral norms (SVD) of the effective weights of ``g``, read in eval mode."""
        was_training = self.training
        self.eval()  # reading a weight in training mode would run a power iteration
        try:
            with torch.no_grad():
                return [float(torch.linalg.matrix_norm(m.weight.double(), ord=2)) for m in self._linears()]
        finally:
            self.train(was_training)

    def jacobian_bound(self) -> Tensor:
        """Bound of ``|d r / d(s, dv, v)|`` ``[3]``: ``r_max * lipschitz * prod(layer_norms) / scale``."""
        return self.r_max * self.lipschitz * math.prod(self.layer_norms()) / self.scaler.scale

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_sizes": list(self.hidden_sizes),
            "r_max": self.r_max,
            "lipschitz": self.lipschitz,
            "idm_learnable": self.idm_learnable,
        }
