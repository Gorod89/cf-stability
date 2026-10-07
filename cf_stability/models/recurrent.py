"""Recurrent networks (GRU, LSTM) over a window of states."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from cf_stability.models.base import CFModel, InputScaler, ModelContext

RNN_CELLS: dict[str, type[nn.RNNBase]] = {"gru": nn.GRU, "lstm": nn.LSTM}


class Recurrent(CFModel):
    """One-layer GRU or LSTM over the last ``window`` standardised states (data scaler),
    linear head on the last hidden state."""

    def __init__(
        self, context: ModelContext | None = None, *, cell: str = "lstm", hidden_size: int = 64, window: int = 30
    ) -> None:
        super().__init__()
        if cell not in RNN_CELLS:
            raise ValueError(f"cell must be one of {sorted(RNN_CELLS)}, got {cell!r}")
        context = context or ModelContext()
        self.name = cell
        self.hidden_size, self.window = int(hidden_size), int(window)
        self.scaler = InputScaler(context.center, context.scale)
        self.rnn = RNN_CELLS[cell](3, self.hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, state_history: Tensor) -> Tensor:
        """``[B, steps, 3] -> [B]``; a history shorter than ``window`` is used as it is."""
        x = self.scaler(state_history[:, -self.window :, :]).to(self.head.weight.dtype)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "hidden_size": self.hidden_size, "window": self.window}


class GRU(Recurrent):
    name = "gru"

    def __init__(self, context: ModelContext | None = None, *, hidden_size: int = 64, window: int = 30) -> None:
        super().__init__(context, cell="gru", hidden_size=hidden_size, window=window)


class LSTM(Recurrent):
    name = "lstm"

    def __init__(self, context: ModelContext | None = None, *, hidden_size: int = 64, window: int = 30) -> None:
        super().__init__(context, cell="lstm", hidden_size=hidden_size, window=window)
