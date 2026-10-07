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


def windowed_state_space(jacobian: Tensor, dt: float) -> tuple[Tensor, Tensor, Tensor]:
    """State-space form ``(A [n, 2W+1, 2W+1], B [n, 2W+1], C [2W+1])`` of the linearised two-vehicle loop of a
    law with a history window of ``W`` states (M9, the full-history linearisation).

    Deviations from the equilibrium: gap ``e``, follower speed ``u``, leader speed ``w`` (the input); the
    law reads ``ds = e``, ``ddv = u - w`` (``dv = v - v_lead``) and ``dv_own = u`` at every window position.
    ``jacobian [n, W, 3]`` comes from :func:`history_jacobian` (entry ``i`` acts on the state ``W - 1 - i``
    steps ago). The state at step ``k`` is

        X[k] = (e[k], u[k-W+1], ..., u[k], w[k-W+1], ..., w[k]),

    which holds the whole window: the gaps of the earlier positions follow from the integration scheme,
    ``e[m] = e[k] - dt sum_{l=m+1..k} (w[l] - u[l])``. One step of the semi-implicit Euler of the rollout
    (``rollout_model``; the leader position by the same scheme) with the leader speed ``w[k+1]`` as input:

        a[k]   = sum_i J_s[i] e[k-W+1+i] + J_dv[i] (u - w)[k-W+1+i] + J_v[i] u[k-W+1+i]
        u[k+1] = u[k] + dt a[k],    e[k+1] = e[k] + dt (w[k+1] - u[k+1]),    u, w: shift registers,

    i.e. ``X[k+1] = A X[k] + B w[k+1]`` and ``u[k] = C X[k]``. The transfer function from the leader speed to
    the follower speed is ``G(z) = z C (zI - A)^-1 B``, equal to :func:`transfer_windowed`. The rows of the
    leader speeds depend on the input only: ``A`` is block upper triangular, its eigenvalues are those of
    the block of ``(e, u)`` (:func:`closed_loop_poles`) and ``W`` zeros. Float64 throughout.
    """
    jac = jacobian.to(torch.float64)
    n, window = jac.shape[0], jac.shape[1]
    j_s, j_dv, j_v = jac.unbind(-1)  # [n, W]
    # the gap at window position i is e[k] - dt sum_{i' > i} (w - u)[i']: its weight moves to the later speeds
    earlier = torch.cumsum(j_s, dim=1) - j_s  # sum_{i < i'} J_s[i]
    c_e = j_s.sum(dim=1)  # [n]
    c_u = j_dv + j_v + dt * earlier  # [n, W]: a[k] = c_e e + c_u . u_window + c_w . w_window
    c_w = -j_dv - dt * earlier
    size = 2 * window + 1
    e, u = 0, 1 + torch.arange(window, device=jac.device)  # indices of the state
    w, last = 1 + window + torch.arange(window, device=jac.device), window  # u[k] is the entry `window`
    A = torch.zeros(n, size, size, dtype=torch.float64, device=jac.device)
    A[:, u[:-1], u[1:]] = 1.0  # shift registers
    A[:, w[:-1], w[1:]] = 1.0
    A[:, last, e] = dt * c_e  # u[k+1] = u[k] + dt a[k]
    A[:, last, u] = dt * c_u
    A[:, last, last] += 1.0
    A[:, last, w] = dt * c_w
    A[:, e, :] = -dt * A[:, last, :]  # e[k+1] = e[k] - dt u[k+1] + dt w[k+1]
    A[:, e, e] += 1.0
    B = torch.zeros(n, size, dtype=torch.float64, device=jac.device)
    B[:, e] = dt  # e[k+1] gains dt w[k+1]
    B[:, 2 * window] = 1.0  # the newest leader speed of the register
    C = torch.zeros(size, dtype=torch.float64, device=jac.device)
    C[last] = 1.0
    return A, B, C


def closed_loop_poles(jacobian: Tensor, dt: float) -> Tensor:
    """Poles ``[n, W+1]`` (complex128) of the linearised two-vehicle loop of a windowed law: the eigenvalues of
    the block of ``(e, u)`` of :func:`windowed_state_space` (the other ``W`` eigenvalues, of the shift register of
    the leader speeds, are 0). They are the roots of ``z^(W-1) ((z - 1)^2 + dt^2 F_s(z) z - dt (F_dv(z) + F_v(z))
    (z - 1))``, the denominator of :func:`transfer_windowed`. All of them inside the unit circle: the full history
    model is locally stable at the equilibrium (small deviations die out, with the leader at its equilibrium)."""
    A, _, _ = windowed_state_space(jacobian, dt)
    block = jacobian.shape[1] + 1
    return torch.linalg.eigvals(A[:, :block, :block])


def windowed_margin(jacobian: Tensor, dt: float) -> Tensor:
    """``M_w [n]``, the coefficient of the low-frequency expansion ``|G(e^(i w dt))|^2 = 1 - w^2 M_w / f_s^2 + O(w^4)``
    of :func:`transfer_windowed` (M9):

        M_w = M + 2 (f_v m_s - f_s m_v) - dt f_s f_v,    m_x = sum_j (j dt) J_x[W - 1 - j],

    with ``f_x = sum_j J_x[j]`` and ``M = f_v^2 + 2 f_v f_dv - 2 f_s`` of the memoryless view and ``m_x`` the first
    moments of the history Jacobian over the lag (``j`` steps ago). The memoryless ``M`` is the coefficient only
    when the lag moments satisfy ``f_v m_s = f_s m_v`` (no memory, or a delay common to spacing and speed); the
    last term is the semi-implicit Euler step (``M - dt f_s f_v`` for a memoryless law)."""
    jac = jacobian.to(torch.float64)
    window = jac.shape[1]
    lag = dt * torch.arange(window - 1, -1, -1, dtype=torch.float64, device=jac.device)  # time ago of entry i
    f_s, f_dv, f_v = jac.sum(dim=1).unbind(-1)
    m_s, _, m_v = (jac * lag[None, :, None]).sum(dim=1).unbind(-1)
    margin = f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s
    return margin + 2.0 * (f_v * m_s - f_s * m_v) - dt * f_s * f_v
