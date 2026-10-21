"""Common interface of car-following models (docs/m2_contract.md, section 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from cf_stability.data.schema import DT, Event

# fixed physical scales of (s, dv, v); used where a bound must not depend on the dataset
PHYSICAL_SCALE = (20.0, 2.0, 10.0)


class InputScaler(nn.Module):
    """``(x - center) / scale`` for states ``(s, dv, v)``; the values are buffers of the model."""

    def __init__(self, center: Sequence[float] = (0.0, 0.0, 0.0), scale: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float64).clone())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64).clone())

    @classmethod
    def physical(cls) -> "InputScaler":
        return cls((0.0, 0.0, 0.0), PHYSICAL_SCALE)

    def forward(self, state: Tensor) -> Tensor:
        return (state - self.center.to(state.dtype)) / self.scale.to(state.dtype)


@dataclass
class ModelContext:
    """What a model needs from the training part of a fold."""

    center: Sequence[float] = (0.0, 0.0, 0.0)  # mean of (s, dv, v) over the training samples
    scale: Sequence[float] = (1.0, 1.0, 1.0)  # their standard deviations
    box_low: Sequence[float] = (0.5, -5.0, 0.0)  # 1 % quantiles of the training states
    box_high: Sequence[float] = (120.0, 5.0, 40.0)  # 99 % quantiles
    idm_params: Mapping[str, float] = field(default_factory=dict)  # global calibration of the training part
    newell_params: Mapping[str, float] = field(default_factory=dict)
    ovm_params: Mapping[str, float] = field(default_factory=dict)
    dt: float = DT
    seed: int = 0


class CFModel(nn.Module):
    """Car-following model: state history ``(s, dv, v)`` -> follower acceleration.

    ``forward`` takes ``[batch, steps, 3]`` with ``steps >= window`` (only the last
    ``window`` steps are used) and returns ``[batch]`` accelerations in m/s^2.
    States are raw SI values, ``dv = v - v_lead`` as everywhere in the project.
    """

    name: str = "model"
    window: int = 1
    dt: float = DT
    trainable: bool = True
    differentiable: bool = True  # False: the stability analysis uses finite differences (kNN)

    @property
    def memoryless(self) -> bool:
        return self.window == 1

    def forward(self, state_history: Tensor) -> Tensor:
        raise NotImplementedError

    def acc(self, s: Tensor, dv: Tensor, v: Tensor) -> Tensor:
        """Memoryless call on tensors of any (equal) shape, as a one-step history."""
        state = torch.stack((s, dv, v), dim=-1).reshape(-1, 1, 3)
        return self.forward(state).reshape(s.shape)

    def extra_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Additional training loss of the model; ``batch["state"]`` is ``[B, window, 3]``."""
        return batch["state"].new_zeros(())

    def fit(self, events: Sequence[Event], context: ModelContext) -> dict[str, Any]:
        """Fitting without gradient steps (calibration, neighbour index); returns a summary."""
        return {}

    def config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild the model from a checkpoint."""
        raise NotImplementedError

    # ------------------------------------------------------------------ stateful models (docs/m12_contract.md, D135)
    # A stateful model carries a state along the vehicle besides its history window: ``forward(history, state)``
    # returns ``(acceleration, state)``, and ``forward(history)`` (``state=None``) stays the call of every existing
    # caller. The callers that carry the state (the trainer, the evaluation, the audit and gain rollouts, the platoon
    # test, the corridor law) ask ``stateful``; every model without a state keeps ``stateful = False`` and its
    # behaviour, and the methods below return no state for it.

    stateful: bool = False

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> Any:
        """State of ``batch`` vehicles at their first call (a stateless model: None)."""
        return None

    def steady_state(self, state_history: Tensor) -> Any:
        """State of vehicles that have driven at the constant histories ``[B, steps, 3]`` for a long time (the warm-up
        on an equilibrium; a stateless model: None)."""
        return None

    def step(self, state_history: Tensor, state: Any = None) -> tuple[Tensor, Any]:
        """``(acceleration, state)`` of one call: ``forward(history, state)`` with a state, ``(forward(history), None)``
        without one (a stateless model, or a stateful one called as before)."""
        if state is None:
            return self.forward(state_history), None
        return self.forward(state_history, state)


class SteadyStateView(CFModel):
    """A stateful model with its state at the steady state of the history (``model.steady_state``): the memoryless law
    whose zeros are the equilibria of the stateful model, for the equilibrium search and the linearisation at a constant
    history (D135). Window, time step and differentiability of the model; the model is a submodule (``.to()``,
    ``.double()`` and ``.eval()`` reach it)."""

    def __init__(self, model: CFModel) -> None:
        super().__init__()
        self.model = model
        self.name = f"{model.name}_steady"
        self.dt = model.dt
        self.differentiable = bool(getattr(model, "differentiable", True))

    @property
    def window(self) -> int:  # type: ignore[override]
        return int(self.model.window)

    def forward(self, state_history: Tensor) -> Tensor:
        acc, _ = self.model(state_history, self.model.steady_state(state_history))
        return acc

    def config(self) -> dict[str, Any]:
        raise TypeError("a steady-state view is not saved; save the stateful model")


def steady_view(model: CFModel) -> CFModel:
    """:class:`SteadyStateView` of a stateful model, the model itself otherwise."""
    return SteadyStateView(model) if getattr(model, "stateful", False) else model
