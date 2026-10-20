"""Two-timescale certified hybrid (docs/m11_contract.md, D129, algorithm A2).

The certified ResidualIDM of E4 (margin core, Lipschitz-bounded residual) whose core time gap follows a slow
latent ``z`` in [-1, 1]::

    a = idm(s, dv, v; v0, T(z), s0, a, b) + r_max * tanh(lipschitz * g(s / 20, dv / 2, v / 10, z / z_scale))
    T(z) = T_c * exp(z * ln(t_factor))                 (t_factor 1.5: T from T_c / 1.5 to 1.5 T_c)
    z_{t+1} = z_t + (dt / tau_z) * (tanh(g_phi(h_t)) - z_t),   z_0 = 0,

with ``h_t`` the hidden state of a one-layer GRU (hidden ``latent_hidden``) over the states of the model's window
(data scaler of the training part) and ``g_phi`` a linear head. The latent is integrated over the window that the
model is called with (``window`` states, the last state last), so the model is a function of its window like the
GRU of the zoo: the training rollouts, the audit, the platoon test and the corridor (the history of every vehicle)
need nothing of their own. With ``c = dt / tau_z`` the closed form is ``z_L = sum_t c (1 - c)^(L - t) tanh(g_t)``
over the ``L <= window`` states, so ``|z| <= 1 - (1 - c)^window`` (:meth:`ResidualIDMTT.latent_reach`) for every
input: the reachable range of the online latent. Gradients flow through ``z`` (no detach).

``frozen_z`` (None: the online latent) freezes the latent at a value for the audit and the certificate: the model is
then the memoryless hybrid with the core of ``T(frozen_z)`` and the residual at that ``z`` (``window`` 1).
:meth:`ResidualIDMTT.at_z` returns the same hybrid as a :class:`FrozenLatentHybrid`, a ResidualIDM whose ``idm`` is
the core at ``T(z)``: the certificate of ``cf_stability/stability/certificate.py`` applies to it as to the E4 hybrid.

The residual takes ``z`` as a fourth input of its own scale (``z_scale``, 1 by default): spectral normalisation
bounds the first layer over the four inputs together, so the bound of ``|dr/d(s, dv, v)|``
(``jacobian_bound``, that of ResidualIDM) is the same at every ``z``, and ``|dr/dz| <= r_max lipschitz
prod(norms) / z_scale`` (``latent_bound``). The untrained model is the margin core itself: the residual is zero
(antisymmetric start of ResidualIDM) and the head of the latent is zero, so that ``z = 0`` and ``T = T_c``.
:meth:`ResidualIDMTT.from_residual_idm` builds the model from a trained ResidualIDM (its core, scaler and residual,
the weights of ``z`` zero): at ``frozen_z = 0`` it computes the outputs and the certificate of that hybrid.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from cf_stability.models.base import CFModel, InputScaler, ModelContext
from cf_stability.models.idm import IDM, IDM_PARAM_NAMES, idm_acc
from cf_stability.models.mlp import tanh_mlp
from cf_stability.models.residual_idm import ResidualIDM, _antisymmetric_init

T_INDEX = IDM_PARAM_NAMES.index("T")


class ResidualIDMTT(ResidualIDM):
    """Certified hybrid with a slow latent time gap (D129); see the module docstring."""

    name = "residual_idm_tt"

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
    ) -> None:
        CFModel.__init__(self)  # the layers of ResidualIDM are built here with a fourth input of the residual
        context = context or ModelContext()
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if self.hidden_sizes[-1] % 2:
            raise ValueError(f"the last hidden layer needs an even width, got {self.hidden_sizes[-1]}")
        if not tau_z > 0 or not window >= 1 or not t_factor > 1 or not z_scale > 0:
            raise ValueError(f"need tau_z > 0, window >= 1, t_factor > 1, z_scale > 0, got {tau_z}, {window}, "
                             f"{t_factor}, {z_scale}")  # fmt: skip
        self.r_max, self.lipschitz, self.idm_learnable = float(r_max), float(lipschitz), bool(idm_learnable)
        self.tau_z, self.latent_hidden, self._window = float(tau_z), int(latent_hidden), int(window)
        self.t_factor, self.z_scale = float(t_factor), float(z_scale)
        self.frozen_z: float | None = None
        self.scaler = InputScaler.physical()  # residual: fixed physical scales (the bound must not depend on the data)
        self.idm = IDM(context=context, learnable=idm_learnable, bounded=idm_learnable, dtype=torch.float32)
        self.g = tanh_mlp((4, *self.hidden_sizes, 1))
        linears = self._linears()
        _antisymmetric_init(linears[-2], linears[-1])
        for layer in linears:
            spectral_norm(layer)
        self.latent_scaler = InputScaler(context.center, context.scale)  # latent: the data scaler, as the GRU
        self.latent_rnn = nn.GRU(3, self.latent_hidden, num_layers=1, batch_first=True)
        self.latent_head = nn.Linear(self.latent_hidden, 1)
        with torch.no_grad():  # z = 0 at the start: the untrained model is the margin core
            self.latent_head.weight.zero_()
            self.latent_head.bias.zero_()

    # ------------------------------------------------------------------------------------------- latent

    @property
    def window(self) -> int:  # type: ignore[override]
        """States the model reads: its window with the online latent, 1 with a frozen one (memoryless)."""
        return 1 if self.frozen_z is not None else self._window

    @property
    def rate(self) -> float:
        """``c = dt / tau_z``, the step of the latent per state."""
        return self.dt / self.tau_z

    def latent_reach(self, steps: int | None = None) -> float:
        """Largest ``|z|`` of the online latent after ``steps`` states (the window by default):
        ``1 - (1 - c)^steps`` (``|tanh| < 1`` and ``z_0 = 0``)."""
        steps = self._window if steps is None else int(steps)
        return 1.0 - (1.0 - self.rate) ** steps

    def drive(self, state_history: Tensor) -> Tensor:
        """``tanh(g_phi(h_t))`` ``[B, L]`` over the last ``min(steps, window)`` states of the history."""
        x = self.latent_scaler(state_history[:, -self._window :, :]).to(self.latent_head.weight.dtype)
        out, _ = self.latent_rnn(x)
        return torch.tanh(self.latent_head(out).squeeze(-1))

    def latent_path(self, state_history: Tensor) -> Tensor:
        """``z_1, ..., z_L`` ``[B, L]`` of the recursion ``z_{t+1} = z_t + c (tanh(g_t) - z_t)`` from ``z_0 = 0``
        (step by step; :meth:`latent` uses the closed form)."""
        u = self.drive(state_history)
        z, path = torch.zeros_like(u[:, 0]), []
        for t in range(u.shape[1]):
            z = z + self.rate * (u[:, t] - z)
            path.append(z)
        return torch.stack(path, dim=1)

    def latent(self, state_history: Tensor) -> Tensor:
        """The latent ``[B]`` at the last state: ``frozen_z``, or ``z_L = sum_t c (1 - c)^(L - t) tanh(g_t)``."""
        if self.frozen_z is not None:
            last = state_history[..., -1, 0]
            return torch.full_like(last, float(self.frozen_z), dtype=self._dtype())
        u = self.drive(state_history)
        n = u.shape[1]
        c = self.rate
        weights = c * (1.0 - c) ** torch.arange(n - 1, -1, -1, dtype=torch.float64, device=u.device)
        return u @ weights.to(u.dtype)

    # -------------------------------------------------------------------------------------------- hybrid

    def time_gap(self, z: Tensor) -> Tensor:
        """``T(z) = T_c exp(z ln t_factor)``."""
        return self.idm.theta[T_INDEX] * torch.exp(z * math.log(self.t_factor))

    def core_acc(self, last: Tensor, z: Tensor) -> Tensor:
        """IDM of the core with the time gap ``T(z)`` at the states ``last [B, 3]``."""
        v0, _, s0, a, b = self.idm.theta.unbind()
        return idm_acc(last[..., 0], last[..., 1], last[..., 2], v0, self.time_gap(z), s0, a, b,
                       self.idm.delta, self.idm.s_eps)  # fmt: skip

    def residual_at(self, last: Tensor, z: Tensor) -> Tensor:
        """``r_max * tanh(lipschitz * g(x_tilde, z / z_scale))`` ``[B]`` of the states ``last [B, 3]``."""
        x = torch.cat((self.scaler(last.to(self._dtype())), (z / self.z_scale).to(self._dtype())[..., None]), dim=-1)
        return self.r_max * torch.tanh(self.lipschitz * self.g(x).squeeze(-1))

    def residual(self, state_history: Tensor) -> Tensor:
        """The residual at the last state with the latent of the history (``frozen_z`` when set)."""
        x = state_history.to(self._dtype())
        return self.residual_at(x[..., -1, :], self.latent(x))

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history.to(self._dtype())
        last, z = x[..., -1, :], self.latent(x)
        return self.core_acc(last, z) + self.residual_at(last, z)

    def latent_bound(self) -> float:
        """Bound of ``|dr/dz|``: ``r_max * lipschitz * prod(layer_norms) / z_scale``."""
        return self.r_max * self.lipschitz * math.prod(self.layer_norms()) / self.z_scale

    def core_params(self, z: float = 0.0) -> dict[str, float]:
        """The parameters of the core at the frozen latent ``z`` (``T(z)``, the others fixed), in float64."""
        params = dict(zip(IDM_PARAM_NAMES, self.idm.theta.detach().double().cpu().tolist()))
        params["T"] = params["T"] * math.exp(float(z) * math.log(self.t_factor))
        return params

    def at_z(self, z: float) -> "FrozenLatentHybrid":
        """The hybrid at the frozen latent ``z`` as a ResidualIDM (copies of the core at ``T(z)``, the scaler and the
        residual): what the certificate of the E4 hybrid needs."""
        return FrozenLatentHybrid(self, z)

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_sizes": list(self.hidden_sizes),
            "r_max": self.r_max,
            "lipschitz": self.lipschitz,
            "idm_learnable": self.idm_learnable,
            "tau_z": self.tau_z,
            "latent_hidden": self.latent_hidden,
            "window": self._window,
            "t_factor": self.t_factor,
            "z_scale": self.z_scale,
        }

    @classmethod
    def from_residual_idm(cls, source: ResidualIDM, context: ModelContext | None = None, **kwargs: Any) -> "ResidualIDMTT":
        """The two-timescale model with the core, the residual scaler and the residual of ``source`` (the weights of
        ``z`` in the first layer zero, the stored vectors of the spectral normalisation extended by 0, so that the
        normalised weights are those of ``source``). ``kwargs``: the latent settings; the residual settings are those
        of ``source``. At ``frozen_z = 0`` the model computes ``source``."""
        model = cls(context, hidden_sizes=source.hidden_sizes, r_max=source.r_max, lipschitz=source.lipschitz,
                    idm_learnable=source.idm_learnable, **kwargs)  # fmt: skip
        for part in (model.g, model.idm, model.latent_rnn, model.latent_head):  # the scalers keep their float64
            part.to(source._dtype())
        state = {k: v.clone() for k, v in source.state_dict().items() if k.startswith(("g.", "scaler.", "idm."))}
        first = "g.0.parametrizations.weight"
        weight = state[f"{first}.original"]
        state[f"{first}.original"] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
        v = state[f"{first}.0._v"]
        state[f"{first}.0._v"] = torch.cat((v, torch.zeros_like(v[:1])))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith(("latent_",)) for k in missing):
            raise ValueError(f"state of {type(source).__name__} does not fit: missing {missing}, unexpected {unexpected}")
        model.train(source.training)
        return model


class FrozenLatentHybrid(ResidualIDM):
    """:class:`ResidualIDMTT` at a frozen latent ``z`` as a memoryless ResidualIDM: ``idm`` is the core with ``T(z)``
    (plain parameters in the dtype of the source), the residual is that of the source at ``z``. The bound of
    ``|dr/d(s, dv, v)|`` (``jacobian_bound``) is that of the source. Not saved (no checkpoint is written of it)."""

    name = "residual_idm_tt_frozen"

    def __init__(self, source: ResidualIDMTT, z: float) -> None:
        CFModel.__init__(self)
        self.z, self.z_scale = float(z), float(source.z_scale)
        self.hidden_sizes = source.hidden_sizes
        self.r_max, self.lipschitz, self.idm_learnable = source.r_max, source.lipschitz, False
        dtype = source.idm.raw.dtype
        self.idm = IDM(source.core_params(z), delta=source.idm.delta, s_eps=source.idm.s_eps, dtype=dtype)
        self.idm.to(source.idm.raw.device)
        self.scaler = copy.deepcopy(source.scaler)
        self.g = copy.deepcopy(source.g)
        self.train(source.training)

    def residual(self, state_history: Tensor) -> Tensor:
        last = state_history[..., -1, :].to(self._dtype())
        z = torch.full_like(last[..., :1], self.z / self.z_scale)
        x = torch.cat((self.scaler(last), z), dim=-1)
        return self.r_max * torch.tanh(self.lipschitz * self.g(x).squeeze(-1))

    def config(self) -> dict[str, Any]:
        raise TypeError("a frozen view of residual_idm_tt is not saved; save the ResidualIDMTT model")
