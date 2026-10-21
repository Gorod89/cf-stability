"""Two-timescale certified hybrid with a latent carried along the vehicle (docs/m12_contract.md, D135, algorithm A2').

The model of D129 (:class:`~cf_stability.models.residual_idm_tt.ResidualIDMTT`: the certified ResidualIDM of E4 whose
core time gap follows a slow latent ``z``, the residual with ``z`` as a fourth input) with two changes:

1. *Range.* The latent follows the target ``z_target = z_lo + (z_hi - z_lo) sigmoid(g_phi(h_t))`` in
   ``(z_lo, z_hi) = (-0.2, 1.0)`` instead of ``tanh(g_phi(h_t))``; ``T(z) = T_c exp(z ln t_factor)`` as before (T from
   0.92 T_c to 1.5 T_c), so that the exact certificate of D129 can hold at every reachable ``z`` (the margin core cannot
   be certified below T = 0.92 T_c). The bias of the head starts at ``ln(-z_lo / z_hi)``: ``z_target = 0`` before
   training, so that the untrained model is the margin core, as in D129. ``LATENT_SWEEP``: the 21 values of
   ``[z_lo, z_hi]`` at which the certified budget of D86 is taken and the certificate is checked
   (``cf_stability/stability/certificate.py``).
2. *Persistence.* The model is stateful (``stateful``, :meth:`initial_state` ``z = 0``): ``forward(history, state)``
   advances the latent once per call with the target of the window's last step,
   ``z_{t+1} = z_t + (dt / tau_z)(z_target - z_t)``, and returns ``(acceleration, z_{t+1})``; the acceleration uses the
   advanced latent (as the latent of D129 includes the last state of the window). ``forward(history)``
   (``state=None``) is the model of D129 with the target above: the latent integrated over the window from ``z = 0`` at
   its first state (the window-restarted latent, which every existing caller gets). The latent of a vehicle is a
   convex combination of 0 and of targets in ``(z_lo, z_hi)``, so it stays in ``[z_lo, z_hi]``; for a constant history
   it converges to the target of that history with the time constant ``tau_z`` (:meth:`steady_state`).

``h_t`` is the hidden state of the one-layer GRU over the window (as in D129: the fast part is a feature of the window,
the slow latent is the state of the vehicle). :meth:`window_drive` (the target at the window's last step) and
:meth:`carry` (the latent after a sequence of targets, in closed form) let the trainer run the teacher-forced pass of
the latent over an event in one batch (``cf_stability/train/stateful.py``). ``frozen_z`` (the audits and the
certificate at frozen latents) makes the model memoryless and stateless, as in D129.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor

from cf_stability.models.base import ModelContext
from cf_stability.models.residual_idm_tt import ResidualIDMTT

LATENT_RANGE: tuple[float, float] = (-0.2, 1.0)  # D135: T from T_c 1.5^-0.2 = 0.922 T_c to 1.5 T_c
N_SWEEP = 21  # frozen latents of the budget and of the certificate


def latent_sweep(z_lo: float, z_hi: float, n: int = N_SWEEP) -> tuple[float, ...]:
    """``n`` equally spaced values of ``[z_lo, z_hi]`` (both included), rounded to 10 decimals."""
    return tuple(round(z_lo + (z_hi - z_lo) * k / (n - 1), 10) for k in range(n))


LATENT_SWEEP_TT2: tuple[float, ...] = latent_sweep(*LATENT_RANGE)  # -0.2, -0.14, ..., 0.94, 1.0


class ResidualIDMTT2(ResidualIDMTT):
    """Certified hybrid with a slow latent time gap carried along the vehicle (D135); see the module docstring."""

    name = "residual_idm_tt2"

    def __init__(
        self,
        context: ModelContext | None = None,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        r_max: float = 1.0,
        lipschitz: float = 1.0,
        idm_learnable: bool = False,
        tau_z: float = 30.0,
        latent_hidden: int = 16,
        window: int = 30,
        t_factor: float = 1.5,
        z_scale: float = 1.0,
        z_lo: float = LATENT_RANGE[0],
        z_hi: float = LATENT_RANGE[1],
    ) -> None:
        super().__init__(
            context, hidden_sizes=hidden_sizes, r_max=r_max, lipschitz=lipschitz, idm_learnable=idm_learnable,
            tau_z=tau_z, latent_hidden=latent_hidden, window=window, t_factor=t_factor, z_scale=z_scale,
        )  # fmt: skip
        if not float(z_lo) < 0.0 < float(z_hi):
            raise ValueError(f"need z_lo < 0 < z_hi (z = 0 is the initial latent and T_c), got {z_lo}, {z_hi}")
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.LATENT_SWEEP = latent_sweep(self.z_lo, self.z_hi)
        with torch.no_grad():  # sigmoid(b) = -z_lo / (z_hi - z_lo): z_target = 0, the untrained model is the core
            self.latent_head.bias.fill_(math.log(-self.z_lo / self.z_hi))

    # ------------------------------------------------------------------------------------------- latent

    @property
    def stateful(self) -> bool:  # type: ignore[override]
        """The latent is carried along the vehicle; a frozen latent has no state (the model is memoryless)."""
        return self.frozen_z is None

    def latent_target(self, g: Tensor) -> Tensor:
        """``z_lo + (z_hi - z_lo) sigmoid(g)``."""
        return self.z_lo + (self.z_hi - self.z_lo) * torch.sigmoid(g)

    def drive(self, state_history: Tensor) -> Tensor:
        """The targets ``[B, L]`` over the last ``min(steps, window)`` states (the window-restarted latent of
        ``state=None`` integrates them from ``z = 0``)."""
        x = self.latent_scaler(state_history[:, -self._window :, :]).to(self.latent_head.weight.dtype)
        out, _ = self.latent_rnn(x)
        return self.latent_target(self.latent_head(out).squeeze(-1))

    def window_drive(self, state_history: Tensor) -> Tensor:
        """The target ``[B]`` at the last state of the window: what one call of the persistent latent moves to."""
        x = self.latent_scaler(state_history[:, -self._window :, :]).to(self.latent_head.weight.dtype)
        out, _ = self.latent_rnn(x)
        return self.latent_target(self.latent_head(out[:, -1]).squeeze(-1))

    def advance(self, state: Tensor, drive: Tensor) -> Tensor:
        """One step of the latent: ``z + (dt / tau_z)(drive - z)``."""
        return state + self.rate * (drive - state)

    def carry(self, drives: Tensor, counts: Tensor, state: Tensor) -> Tensor:
        """The latent ``[S]`` after ``counts[r]`` steps from ``state[r]`` with the targets ``drives[r, :counts[r]]``
        (``[S, n]``; the entries from ``counts[r]`` on are not read):
        ``(1 - c)^k z_0 + sum_{i < k} c (1 - c)^(k - 1 - i) drives_i`` with ``c = dt / tau_z``, the closed form of
        :meth:`advance` applied ``k`` times. Fixed shapes: the trainer's pass can be captured in a CUDA graph."""
        c, n = self.rate, drives.shape[-1]
        lag = counts.to(torch.float64)[:, None] - 1.0 - torch.arange(n, dtype=torch.float64, device=drives.device)
        weights = torch.where(lag >= 0, c * (1.0 - c) ** lag.clamp(min=0.0), torch.zeros_like(lag))
        keep = (1.0 - c) ** counts.to(torch.float64)
        return keep.to(drives.dtype) * state.to(drives.dtype) + (weights.to(drives.dtype) * drives).sum(-1)

    def carry_path(self, drives: Tensor, state: Tensor) -> Tensor:
        """The latent ``[S, n]`` after each of the ``n`` steps from ``state [S]`` with the targets ``drives [S, n]``:
        ``z_k = (1 - c)^(k + 1) z_0 + sum_{i <= k} c (1 - c)^(k - i) drives_i`` (the evaluation's teacher-forced
        path; one product with a lower-triangular matrix)."""
        c, n = self.rate, drives.shape[-1]
        k = torch.arange(n, dtype=torch.float64, device=drives.device)
        lag = k[:, None] - k[None, :]  # [step, target]
        weights = torch.where(lag >= 0, c * (1.0 - c) ** lag.clamp(min=0.0), torch.zeros_like(lag))
        keep = (1.0 - c) ** (k + 1.0)
        return state.to(drives.dtype)[:, None] * keep.to(drives.dtype) + drives @ weights.T.to(drives.dtype)

    def latent_range(self) -> tuple[float, float]:
        """``(z_lo, z_hi)``: the range of the persistent latent (and of the budget and the certificate)."""
        return self.z_lo, self.z_hi

    def latent_reach(self, steps: int | None = None) -> float:
        """Largest ``|z|``: of the persistent latent (``steps`` None: ``max(|z_lo|, |z_hi|)``), or of the latent
        integrated from ``z = 0`` over ``steps`` states (``state=None``): ``(1 - (1 - c)^steps) max(|z_lo|, |z_hi|)``."""
        bound = max(abs(self.z_lo), abs(self.z_hi))
        return bound if steps is None else (1.0 - (1.0 - self.rate) ** int(steps)) * bound

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> Tensor:
        """``z = 0`` for ``batch`` vehicles (dtype of the latent head, on its device unless ``device``)."""
        weight = self.latent_head.weight
        return torch.zeros(int(batch), dtype=weight.dtype, device=weight.device if device is None else device)

    def steady_state(self, state_history: Tensor) -> Tensor | None:
        """The latent of a vehicle that has driven at the constant history for long: the target of the window (the
        fixed point of :meth:`advance`); None with a frozen latent."""
        if self.frozen_z is not None:
            return None
        return self.window_drive(state_history.to(self._dtype()))

    # -------------------------------------------------------------------------------------------- hybrid

    def forward(self, state_history: Tensor, state: Any = None) -> Any:  # type: ignore[override]
        """Without ``state``: the acceleration of the window-restarted latent (D129, the call of every existing
        caller). With ``state`` ``[B]``: ``(acceleration, state)`` with the latent advanced once by the target of the
        window's last step (a frozen latent: its value, the state returned unchanged)."""
        if state is None:
            return super().forward(state_history)
        x = state_history.to(self._dtype())
        last = x[..., -1, :]
        if self.frozen_z is not None:
            z = self.latent(x)
            return self.core_acc(last, z) + self.residual_at(last, z), state
        z = self.advance(state, self.window_drive(x))
        return self.core_acc(last, z) + self.residual_at(last, z), z

    def config(self) -> dict[str, Any]:
        return {**super().config(), "name": self.name, "z_lo": self.z_lo, "z_hi": self.z_hi}
