"""k-nearest-neighbour regression in the standardised state space."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.data.schema import Event
from cf_stability.models.base import CFModel, InputScaler, ModelContext


class KNN(CFModel):
    """Mean acceleration of the ``k`` training samples nearest to the last state (Euclidean
    distance of the standardised ``(s, dv, v)``, data scaler), brute force in chunks of queries."""

    name = "knn"
    trainable = False
    differentiable = False  # piecewise constant in the state

    def __init__(
        self, context: ModelContext | None = None, *, k: int = 10, max_points: int = 200_000, chunk: int = 1024
    ) -> None:
        super().__init__()
        context = context or ModelContext()
        self.k, self.max_points, self.chunk = int(k), int(max_points), int(chunk)
        self.scaler = InputScaler(context.center, context.scale)
        self.register_buffer("points", torch.zeros(0, 3))  # standardised states [N, 3]
        self.register_buffer("targets", torch.zeros(0))  # their accelerations [N]

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Store at most ``max_points`` samples of ``events``, drawn with ``context.seed``."""
        states = np.concatenate([np.stack((ev.s, ev.dv, ev.v), axis=-1) for ev in events])
        acc = np.concatenate([ev.a for ev in events])
        n_samples = len(acc)
        if n_samples > self.max_points:
            keep = np.sort(np.random.default_rng(context.seed).choice(n_samples, self.max_points, replace=False))
            states, acc = states[keep], acc[keep]
        like = self.points
        self.points = self.scaler(torch.as_tensor(states, device=like.device)).to(like.dtype)
        self.targets = torch.as_tensor(acc, dtype=like.dtype, device=like.device)
        return {"n_samples": n_samples, "n_points": len(acc)}

    def forward(self, state_history: Tensor) -> Tensor:
        if self.points.shape[0] < self.k:
            raise RuntimeError(f"KNN needs at least k = {self.k} fitted points, has {self.points.shape[0]}")
        query = self.scaler(state_history[..., -1, :]).to(self.points.dtype)
        flat = query.reshape(-1, 3)
        sq_norm = (self.points * self.points).sum(-1)
        out = []
        for start in range(0, flat.shape[0], self.chunk):
            # squared distances without |q|^2, which does not change the ranking for a query
            dist = sq_norm - 2.0 * flat[start : start + self.chunk] @ self.points.T
            nearest = torch.topk(dist, self.k, dim=-1, largest=False).indices
            out.append(self.targets[nearest].mean(-1))
        return torch.cat(out).reshape(query.shape[:-1])

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args: Any, **kwargs: Any) -> None:
        # the number of stored points is known from the checkpoint only
        for name in ("points", "targets"):
            if prefix + name in state_dict:
                setattr(self, name, getattr(self, name).new_empty(state_dict[prefix + name].shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k, "max_points": self.max_points, "chunk": self.chunk}
