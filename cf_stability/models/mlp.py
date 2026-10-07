"""Feed-forward network on the last state."""

from __future__ import annotations

from typing import Any, Sequence

from torch import Tensor, nn

from cf_stability.models.base import CFModel, InputScaler, ModelContext


def tanh_mlp(sizes: Sequence[int]) -> nn.Sequential:
    """Linear layers ``sizes[0] -> ... -> sizes[-1]`` with tanh in between and a linear output."""
    layers: list[nn.Module] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    return nn.Sequential(*layers[:-1])


class MLP(CFModel):
    """``3 -> hidden_sizes -> 1`` tanh network on the standardised last state (data scaler)."""

    name = "mlp"

    def __init__(self, context: ModelContext | None = None, *, hidden_sizes: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        context = context or ModelContext()
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.scaler = InputScaler(context.center, context.scale)
        self.net = tanh_mlp((3, *self.hidden_sizes, 1))

    def forward(self, state_history: Tensor) -> Tensor:
        x = self.scaler(state_history[..., -1, :]).to(self.net[0].weight.dtype)
        return self.net(x).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "hidden_sizes": list(self.hidden_sizes)}
