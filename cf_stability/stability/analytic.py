"""Linearisation of a car-following law and the analytic stability criterion (docs/m3_contract.md, 1 and 3).

Convention ``dv = v - v_lead``. For deviations from an equilibrium the follower obeys
``du/dt = f_s (y_lead - y) + f_dv (u - u_lead) + f_v u``, which gives the transfer function
from the leader speed to the follower speed

    G(i w) = (f_s - i w f_dv) / (f_s - w^2 - i w (f_dv + f_v)).
"""

from __future__ import annotations

import torch
from torch import Tensor

from cf_stability.models.base import PHYSICAL_SCALE, CFModel
from cf_stability.stability.equilibrium import constant_history, steady_acc

FD_STEP = 0.25  # finite-difference step of non-differentiable models, in units of the scaler


def _fd_steps(model: CFModel, like: Tensor) -> Tensor:
    scaler = getattr(model, "scaler", None)
    scale = scaler.scale if scaler is not None else torch.tensor(PHYSICAL_SCALE)
    return FD_STEP * scale.to(like)


def partials(model: CFModel, s: Tensor, v: Tensor, *, create_graph: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """``(f_s, f_dv, f_v)`` at the constant states ``(s, 0, v)``, each ``[n]``.

    Derivatives with respect to a shift of the whole history (the sum over the window of the
    derivatives with respect to the single steps). Autograd, or central differences for a model
    with ``differentiable = False``. ``create_graph`` keeps the result differentiable with
    respect to the parameters of the model.
    """
    if not getattr(model, "differentiable", True):
        steps = _fd_steps(model, s)
        zero = torch.zeros_like(s)
        out = []
        with torch.no_grad():
            for k, (ds, ddv, dvv) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                h = steps[k]
                up = steady_acc(model, s + ds * h, v + dvv * h, zero + ddv * h)
                down = steady_acc(model, s - ds * h, v - dvv * h, zero - ddv * h)
                out.append((up - down) / (2.0 * h))
        return out[0], out[1], out[2]
    s = s.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    dv = torch.zeros_like(s).requires_grad_(True)
    # cuDNN RNNs have no backward in eval mode and no double backward
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        acc = steady_acc(model, s, v, dv)
        f_s, f_dv, f_v = torch.autograd.grad(acc.sum(), (s, dv, v), create_graph=create_graph, allow_unused=True)
    zero = torch.zeros_like(acc)
    return tuple(zero if g is None else g for g in (f_s, f_dv, f_v))  # type: ignore[return-value]


def history_jacobian(model: CFModel, s: Tensor, v: Tensor) -> Tensor:
    """Jacobian ``[n, window, 3]`` of the acceleration with respect to every entry of the constant history."""
    history = constant_history(model, s.detach(), v.detach()).clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        (jac,) = torch.autograd.grad(model(history).sum(), history, allow_unused=True)
    return torch.zeros_like(history) if jac is None else jac


def criterion(f_s: Tensor, f_dv: Tensor, f_v: Tensor) -> dict[str, Tensor]:
    """Flags and margin of the analytic criterion.

    ``margin = f_v^2 + 2 f_v f_dv - 2 f_s``; string stable when ``margin >= 0``; when it is
    negative the gain exceeds 1 for ``0 < w < band_upper = sqrt(-margin)``.
    """
    margin = f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s
    return {
        "local_stable": (f_s > 0) & (f_v + f_dv < 0),
        "rational": (f_s >= 0) & (f_dv <= 0) & (f_v <= 0),
        "margin": margin,
        "string_stable": margin >= 0,
        "band_upper": torch.sqrt(torch.clamp(-margin, min=0.0)),
    }


def _columns(f_s: Tensor, f_dv: Tensor, f_v: Tensor, omega: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ctype = torch.complex128 if omega.dtype == torch.float64 else torch.complex64
    return f_s[..., None].to(ctype), f_dv[..., None].to(ctype), f_v[..., None].to(ctype), omega.to(ctype)


def transfer_continuous(f_s: Tensor, f_dv: Tensor, f_v: Tensor, omega: Tensor) -> Tensor:
    """``G(i w)`` of the continuous-time linearisation, complex ``[n, n_w]``."""
    f_s, f_dv, f_v, w = _columns(f_s, f_dv, f_v, omega)
    return (f_s - 1j * w * f_dv) / (f_s - w**2 - 1j * w * (f_dv + f_v))


def _discrete(F_s: Tensor, F_dv: Tensor, F_v: Tensor, z: Tensor, dt: float) -> Tensor:
    """Transfer function of ``u[k+1] = u[k] + dt a[k]``, ``y[k+1] = y[k] + dt u[k+1]``."""
    return (dt**2 * F_s * z - dt * F_dv * (z - 1.0)) / ((z - 1.0) ** 2 + dt**2 * F_s * z - dt * (F_dv + F_v) * (z - 1.0))


def transfer_discrete(f_s: Tensor, f_dv: Tensor, f_v: Tensor, omega: Tensor, dt: float) -> Tensor:
    """Gain of the memoryless linearisation under the integration scheme of the project, ``[n, n_w]``."""
    f_s, f_dv, f_v, w = _columns(f_s, f_dv, f_v, omega)
    return _discrete(f_s, f_dv, f_v, torch.exp(1j * w * dt), dt)


def transfer_windowed(jacobian: Tensor, omega: Tensor, dt: float) -> Tensor:
    """Gain of the linearised law with its history window, ``[n, n_w]``.

    ``jacobian [n, W, 3]`` from :func:`history_jacobian`; the entry ``W - 1 - j`` acts on the state
    ``j`` steps ago: ``F_x(z) = sum_j J_x[W - 1 - j] z^-j``.
    """
    ctype = torch.complex128 if omega.dtype == torch.float64 else torch.complex64
    window = jacobian.shape[1]
    z = torch.exp(1j * omega.to(ctype) * dt)  # [n_w]
    lag = torch.arange(window - 1, -1, -1, device=jacobian.device)  # steps ago of every history entry
    powers = z[None, :] ** (-lag[:, None].to(ctype))  # [W, n_w]
    F = torch.einsum("nwx,wk->nkx", jacobian.to(ctype), powers)  # [n, n_w, 3]
    return _discrete(F[..., 0], F[..., 1], F[..., 2], z[None, :], dt)
