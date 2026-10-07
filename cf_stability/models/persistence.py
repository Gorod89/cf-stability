"""Persistence baseline: the follower keeps its current acceleration."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cf_stability.data.schema import DT
from cf_stability.models.base import CFModel


class Persistence(CFModel):
    """``a_hat = (v[-1] - v[-2]) / dt``; zero with a single history step (decision D13)."""

    name = "persistence"
    window = 2
    trainable = False

    def __init__(self, dt: float = DT) -> None:
        super().__init__()
        self.dt = float(dt)

    def forward(self, state_history: Tensor) -> Tensor:
        v = state_history[..., 2]
        if v.shape[-1] < 2:
            return torch.zeros_like(v[..., -1])
        return (v[..., -1] - v[..., -2]) / self.dt

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "dt": self.dt}
