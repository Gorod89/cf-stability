"""Physics-informed deep learning: MLP with an IDM collocation loss."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import IDM
from cf_stability.models.mlp import MLP


class PIDL(CFModel):
    """Prediction = MLP; ``extra_loss = alpha * MSE(mlp(x_c) - idm(x_c))`` on collocation states
    ``x_c`` drawn uniformly from the box of the training states.

    The IDM parameters come from ``context.idm_params`` (global calibration of the training
    part); they are fixed, or learnable in bounded form with ``idm_learnable``. The IDM
    acceleration is clipped to ``physics_clip`` (the range of the closed-loop rollouts): in the
    corners of the box (small gap at high speed) the unclipped IDM returns tens of m/s^2 of
    braking, which would dominate the loss.
    """

    name = "pidl"

    def __init__(
        self,
        context: ModelContext | None = None,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        alpha: float = 1.0,
        idm_learnable: bool = False,
        physics_clip: Sequence[float] = (-8.0, 4.0),
    ) -> None:
        super().__init__()
        context = context or ModelContext()
        self.alpha, self.idm_learnable, self.seed = float(alpha), bool(idm_learnable), int(context.seed)
        self.physics_clip = (float(physics_clip[0]), float(physics_clip[1]))
        self.mlp = MLP(context, hidden_sizes=hidden_sizes)
        self.idm = IDM(context=context, learnable=idm_learnable, bounded=idm_learnable, dtype=torch.float32)
        self.register_buffer("box_low", torch.tensor(context.box_low, dtype=torch.float32))
        self.register_buffer("box_high", torch.tensor(context.box_high, dtype=torch.float32))
        self._generator: torch.Generator | None = None

    def forward(self, state_history: Tensor) -> Tensor:
        return self.mlp(state_history)

    def extra_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        state = batch["state"]
        if self._generator is None or self._generator.device != state.device:
            self._generator = torch.Generator(device=state.device)
            self._generator.manual_seed(self.seed)
        u = torch.rand(state.shape[0], 3, generator=self._generator, device=state.device, dtype=self.box_low.dtype)
        x_c = torch.lerp(self.box_low, self.box_high, u)[:, None, :]
        a_idm = torch.clamp(self.idm(x_c), *self.physics_clip)
        if not self.idm_learnable:
            a_idm = a_idm.detach()
        return self.alpha * torch.mean((self.mlp(x_c) - a_idm) ** 2)

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_sizes": list(self.mlp.hidden_sizes),
            "alpha": self.alpha,
            "idm_learnable": self.idm_learnable,
            "physics_clip": list(self.physics_clip),
        }
