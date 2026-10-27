"""k-nearest-neighbour regression in the standardised state space.

The fitted samples are training data, not parameters: a checkpoint of this model holds the scaler, the number of
samples the fit saw and the seed of the sub-sample, never the samples themselves (D137: the highD licence does
not allow the events to be redistributed, and a checkpoint with the samples is the data). A loaded model is
re-fitted from the training events of its run (:func:`cf_stability.eval.knn_refit.refit_from_run`, called by
:func:`cf_stability.models.load_model` when the events are available); the sub-sample is deterministic in the
seed and the number of samples, so the re-fit reproduces the stored points exactly.
"""

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
        # the samples are not part of the state dict (D137); they are re-fitted from the events of the run
        self.register_buffer("points", torch.zeros(0, 3), persistent=False)  # standardised states [N, 3]
        self.register_buffer("targets", torch.zeros(0), persistent=False)  # their accelerations [N]
        self.register_buffer("n_samples", torch.zeros((), dtype=torch.int64))  # samples of the training events
        self.register_buffer("seed", torch.zeros((), dtype=torch.int64))  # seed of the sub-sample

    @staticmethod
    def samples(events: Sequence[Event]) -> tuple[np.ndarray, np.ndarray]:
        """States ``(s, dv, v)`` ``[N, 3]`` and accelerations ``[N]`` of ``events`` in their order."""
        states = np.concatenate([np.stack((ev.s, ev.dv, ev.v), axis=-1) for ev in events])
        return states, np.concatenate([ev.a for ev in events])

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Store at most ``max_points`` samples of ``events``, drawn with ``context.seed``."""
        states, acc = self.samples(events)
        n_samples = len(acc)
        if n_samples > self.max_points:
            keep = np.sort(np.random.default_rng(context.seed).choice(n_samples, self.max_points, replace=False))
            states, acc = states[keep], acc[keep]
        like = self.points
        self.points = self.scaler(torch.as_tensor(states, device=like.device)).to(like.dtype)
        self.targets = torch.as_tensor(acc, dtype=like.dtype, device=like.device)
        self.n_samples.fill_(n_samples)
        self.seed.fill_(int(context.seed))
        return {"n_samples": n_samples, "n_points": len(acc)}

    @property
    def fitted(self) -> bool:
        return self.points.shape[0] >= self.k

    def refit(self, events: Sequence[Event]) -> dict[str, Any]:
        """Re-fit a loaded model from the training events of its run: the same seed, and the same number of
        samples as the fit the checkpoint records (otherwise the sub-sample would differ from the run's)."""
        n_samples = sum(len(ev.a) for ev in events)
        if int(self.n_samples) and n_samples != int(self.n_samples):
            raise ValueError(
                f"KNN refit: the events hold {n_samples} samples, the checkpoint was fitted on {int(self.n_samples)}; "
                "these are not the training events of the run (data view, split, fold or event extraction differ)"
            )
        return self.fit(events, ModelContext(seed=int(self.seed)))

    def forward(self, state_history: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError(
                f"KNN needs at least k = {self.k} fitted points, has {self.points.shape[0]}: a checkpoint holds no "
                "samples (D137); re-fit the model from the training events of its run (KNN.refit, "
                "cf_stability.eval.knn_refit.refit_from_run), which needs the event data"
            )
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
        # a checkpoint written before D137 carries the samples: take them, so that such a local run still loads
        for name in ("points", "targets"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict.pop(key).to(getattr(self, name).dtype))
        for name in ("n_samples", "seed"):  # absent before D137
            state_dict.setdefault(prefix + name, getattr(self, name).clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k, "max_points": self.max_points, "chunk": self.chunk}
