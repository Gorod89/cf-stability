"""Stability certificate of ResidualIDM (docs/m3_contract.md, section 8; docs/m4_contract.md, 3.1 and 3.3).

``f = f_idm + r`` with ``|r| <= r_max`` and ``|dr/dx_j| <= B_j`` (``ResidualIDM.jacobian_bound``).
At an equilibrium the partial derivatives of the hybrid are those of the IDM plus ``d_j`` with
``|d_j| <= B_j``; the minimum of the criterion over that box is a lower bound of the margin of the
hybrid (the guaranteed margin), and the box gives guaranteed signs of ``f_s`` and ``f_v + f_dv``.
The certificate holds at a speed when the guaranteed margin is ``>= 0`` and both signs are
guaranteed (local and string stability). All computations run in float64 on a CPU copy in eval mode.
With the spacing band of the data (``Band``) the a posteriori forms use the anchored equilibria
(inside the band, or at a speed without band).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.models.idm import IDM, idm_margin
from cf_stability.models.residual_idm import ResidualIDM
from cf_stability.stability.analytic import criterion, partials
from cf_stability.stability.equilibrium import V_GRID, Band, Equilibria, find_equilibria

S_MIN, S_MAX = 1.0, 200.0  # spacings searched by find_equilibria
N_ZOOM = 3  # rescans of the a priori minimum around the best spacing, 2 grid cells wide each
N_BISECT = 60

Speeds = Sequence[float] | Tensor


def _float64(model: ResidualIDM) -> ResidualIDM:
    return copy.deepcopy(model).cpu().double().eval()


def _speeds(v: Speeds) -> Tensor:
    return torch.as_tensor(v, dtype=torch.float64).reshape(-1).cpu()


def _equilibria(m: ResidualIDM, v: Tensor, band: Band | None) -> tuple[Equilibria, Tensor]:
    """Equilibria of the hybrid and the mask of those the certificate may use: the anchored ones
    (every usable one without band). ``Band.at`` follows the dtype of the speeds (float64 here)."""
    eq = find_equilibria(m, v, band=band)
    return eq, eq.anchored


def core_margin(model: ResidualIDM, v: Speeds = V_GRID) -> Tensor:
    """Closed-form string-stability margin ``[n]`` of the IDM core at its own equilibria, NaN for ``v >= v0``."""
    theta = model.idm.theta.detach().double().cpu()
    return idm_margin(_speeds(v), *theta.unbind())


def residual_bounds(model: ResidualIDM) -> Tensor:
    """Bounds ``[3]`` of ``|dr/d(s, dv, v)|`` (``model.jacobian_bound()``, float64)."""
    return model.jacobian_bound().double().cpu()


def guaranteed_margin(f_s: Tensor, f_dv: Tensor, f_v: Tensor, bounds: Tensor) -> Tensor:
    """Exact minimum of ``(f_v + d_v)^2 + 2 (f_v + d_v)(f_dv + d_dv) - 2 (f_s + d_s)`` over ``|d_j| <= B_j``.

    ``bounds [..., 3]`` (order ``s, dv, v``) broadcasts against the derivatives. ``d_s = B_s``;
    ``p^2 + 2 p q`` (``p = f_v + d_v``, ``q = f_dv + d_dv``) is linear in ``q``, so its minimum lies on
    an edge of ``q``, and convex in ``p``, so there it lies at ``p = -q`` clamped to the interval of ``p``.
    """
    b_s, b_dv, b_v = bounds[..., 0], bounds[..., 1], bounds[..., 2]
    p_lo, p_hi = f_v - b_v, f_v + b_v
    edges = []
    for q in (f_dv - b_dv, f_dv + b_dv):
        p = torch.minimum(torch.maximum(-q, p_lo), p_hi)
        edges.append(p * p + 2.0 * p * q)
    return torch.minimum(*edges) - 2.0 * (f_s + b_s)


def _guarantees(f_s: Tensor, f_dv: Tensor, f_v: Tensor, bounds: Tensor) -> dict[str, Tensor]:
    """Guaranteed margin and signs over the box of ``bounds``; NaN derivatives never hold."""
    margin = guaranteed_margin(f_s, f_dv, f_v, bounds)
    fs_positive = f_s - bounds[..., 0] > 0
    sum_negative = f_v + f_dv + bounds[..., 1] + bounds[..., 2] < 0
    return {
        "guaranteed_margin": margin,
        "fs_positive": fs_positive,
        "sum_negative": sum_negative,
        "holds": (margin >= 0) & fs_positive & sum_negative,
    }


def _feasible_interval(idm: IDM, v: Tensor, r_max: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Spacings in ``[S_MIN, S_MAX]`` with ``|f_idm(s, 0, v)| <= r_max`` (an interval: ``f_idm`` rises with ``s``).

    ``(s*/s)^2 = 1 - (v/v0)^delta -+ r_max / a`` with ``s* = s0 + v T`` gives the ends in closed form.
    """
    v0, T, s0, a, _ = idm.theta.detach().double().cpu().unbind()
    s_star = s0 + v * T
    free = 1.0 - (v / v0) ** idm.delta
    low_arg, high_arg = free + r_max / a, free - r_max / a
    s_low = torch.where(low_arg > 0, s_star / torch.sqrt(low_arg.clamp(min=1e-300)), torch.inf).clamp(min=S_MIN)
    s_high = torch.where(high_arg > 0, s_star / torch.sqrt(high_arg.clamp(min=1e-300)), torch.inf).clamp(max=S_MAX)
    return s_low, s_high, s_low <= s_high


def _a_priori(idm: IDM, v: Tensor, r_max: Tensor, bounds: Tensor, n_scan: int) -> dict[str, Tensor]:
    """Minimum over the feasible spacings of the guaranteed margin, and the signs at every one of them.

    The spacings are scanned on ``n_scan`` points and the scan is repeated ``N_ZOOM`` times around
    the minimum; the ends of the interval are scan points, which makes the signs exact (the IDM
    derivatives are monotone in ``s``).
    """
    s_low, s_high, feasible = _feasible_interval(idm, v, r_max)
    lo, hi = torch.where(feasible, s_low, S_MIN), torch.where(feasible, s_high, S_MIN)
    frac = torch.linspace(0.0, 1.0, n_scan, dtype=torch.float64)
    n = len(v)
    best, best_s = torch.full((n,), torch.inf, dtype=torch.float64), torch.full((n,), torch.nan, dtype=torch.float64)
    signs: dict[str, Tensor] = {}
    for scan in range(N_ZOOM + 1):
        s = lo[:, None] + (hi - lo)[:, None] * frac
        f = partials(idm, s.reshape(-1), v.repeat_interleave(n_scan))
        g = _guarantees(*(x.reshape(n, n_scan) for x in f), bounds[:, None, :])
        value, k = g["guaranteed_margin"].min(dim=1)
        better = value < best
        best, best_s = torch.where(better, value, best), torch.where(better, s[torch.arange(n), k], best_s)
        if scan == 0:
            signs = {name: g[name].all(dim=1) for name in ("fs_positive", "sum_negative")}
        step = (hi - lo) / (n_scan - 1)
        centre = s[torch.arange(n), k]
        lo, hi = torch.maximum(centre - step, lo), torch.minimum(centre + step, hi)
    nan = torch.full_like(best, torch.nan)
    margin = torch.where(feasible, best, nan)
    fs_positive, sum_negative = signs["fs_positive"] & feasible, signs["sum_negative"] & feasible
    return {
        "s_low": torch.where(feasible, s_low, nan),
        "s_high": torch.where(feasible, s_high, nan),
        "feasible": feasible,
        "guaranteed_margin": margin,
        "s_worst": torch.where(feasible, best_s, nan),
        "fs_positive": fs_positive,
        "sum_negative": sum_negative,
        "holds": (margin >= 0) & fs_positive & sum_negative,
    }


def _largest(holds_at: Callable[[Tensor], Tensor], n: int) -> Tensor:
    """Largest ``beta >= 0`` per entry with ``holds_at(beta)`` true (monotone), 0 where it fails at 0."""
    zero = torch.zeros(n, dtype=torch.float64)
    feasible = holds_at(zero)
    lo, hi = zero.clone(), torch.ones(n, dtype=torch.float64)
    for _ in range(64):  # doubling until the certificate fails everywhere
        grow = feasible & holds_at(hi)
        if not grow.any():
            break
        lo, hi = torch.where(grow, hi, lo), torch.where(grow, 2.0 * hi, hi)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        ok = holds_at(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    return torch.where(feasible, lo, zero)


def max_residual_budget(
    model: ResidualIDM,
    v: Speeds = V_GRID,
    *,
    a_priori: bool = False,
    n_scan: int = 400,
    keep_r_max: bool = False,
    band: Band | None = None,
) -> Tensor:
    """Largest ``r_max * lipschitz`` per speed ``[n]`` for which the certificate would hold.

    The bounds scale with the product (the layer norms of the weights stay). At the equilibria the
    IDM derivatives are those at the hybrid's current equilibria (anchored ones with ``band``; 0
    without equilibrium). A priori the feasible spacings grow with ``r_max``: by default ``r_max``
    and ``lipschitz`` scale together (the ratio of the model is kept); with ``keep_r_max`` the
    feasible spacings are those of the model's ``r_max`` and only ``lipschitz`` scales, which is
    the budget of a model whose ``r_max`` is fixed (D86). A budget without any feasible spacing
    counts as holding (no equilibrium can exist), which keeps the condition monotone where the
    IDM itself has no equilibrium.
    """
    m, v = _float64(model), _speeds(v)
    unit = math.prod(m.layer_norms()) / m.scaler.scale.double().cpu()  # bounds per unit of r_max * lipschitz
    if a_priori:
        ratio = m.r_max / m.lipschitz

        def holds_at(beta: Tensor) -> Tensor:
            r_max = torch.full_like(beta, m.r_max) if keep_r_max else torch.sqrt(beta * ratio)
            out = _a_priori(m.idm, v, r_max, beta[:, None] * unit, n_scan)
            return out["holds"] | ~out["feasible"]

    else:
        eq, usable = _equilibria(m, v, band)
        f = partials(m.idm, eq.s, eq.v)

        def holds_at(beta: Tensor) -> Tensor:
            return _guarantees(*f, beta[:, None] * unit)["holds"] & usable

    return _largest(holds_at, len(v))


def enforce_budget(model: ResidualIDM, safety: float, v: Speeds = V_GRID, n_scan: int = 400) -> dict[str, float]:
    """Certified budget of D86: ``b`` = the largest ``r_max * lipschitz`` with the model's ``r_max`` for
    which the a priori certificate holds at every speed of ``v``; sets ``lipschitz = safety * b / r_max``.

    The a priori certificate depends on the weights only through the layer norms, so it survives
    the training that follows. Raises ``ValueError`` when ``b = 0``: the IDM core is then not string
    stable (or its signs are not guaranteed) at some spacing that ``r_max`` admits as an equilibrium.
    """
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"certificate.safety must lie in (0, 1], got {safety}")
    speeds = _speeds(v)
    per_speed = max_residual_budget(model, speeds, a_priori=True, n_scan=n_scan, keep_r_max=True)
    admissible = float(per_speed.min())
    if not admissible > 0.0:
        failing = [float(x) for x in speeds[per_speed <= 0.0]]
        core = {k: round(x, 4) for k, x in model.idm.params_dict().items()}
        raise ValueError(
            f"no admissible residual budget: with r_max = {model.r_max:g} the a priori certificate fails for every "
            f"lipschitz > 0 at v = {failing} m/s (IDM core {core}); a core with a larger margin "
            f"(calibration.stability_margin) or a smaller model.r_max is needed"
        )
    model.lipschitz = safety * admissible / model.r_max
    return {"admissible": admissible, "safety": float(safety), "r_max": model.r_max, "lipschitz": model.lipschitz}


def _header(m: ResidualIDM, v: Tensor) -> dict[str, Any]:
    return {
        "v": v,
        "bounds": residual_bounds(m),
        "layer_norms": m.layer_norms(),
        "r_max": m.r_max,
        "lipschitz": m.lipschitz,
        "budget": m.r_max * m.lipschitz,
        "idm_params": m.idm.params_dict(),
    }


def certificate_at_equilibria(model: ResidualIDM, v: Speeds = V_GRID, band: Band | None = None) -> dict[str, Any]:
    """A posteriori certificate: IDM derivatives at the equilibria of the hybrid, per speed.

    With ``band`` the equilibria are searched inside the band and only the anchored ones count.
    Speeds without such an equilibrium (``found`` false) do not hold.
    """
    m, v = _float64(model), _speeds(v)
    eq, found = _equilibria(m, v, band)
    f_s, f_dv, f_v = partials(m.idm, eq.s, eq.v)
    g = _guarantees(f_s, f_dv, f_v, residual_bounds(m))
    g["holds"] = g["holds"] & found
    return {
        **_header(m, v),
        "s": eq.s,
        "found": found,
        "status": eq.status,
        "f_s": f_s,
        "f_dv": f_dv,
        "f_v": f_v,
        "margin_idm": criterion(f_s, f_dv, f_v)["margin"],
        **g,
        "n_holds": int(g["holds"].sum()),
        "max_budget": max_residual_budget(m, v, band=band),
    }


def certificate_a_priori(
    model: ResidualIDM, v: Speeds = V_GRID, n_scan: int = 400, *, keep_r_max: bool = False
) -> dict[str, Any]:
    """A priori certificate: minimum over every spacing that can be an equilibrium of a residual with
    ``|r| <= r_max`` (``|f_idm(s, 0, v)| <= r_max``, ``s`` in [1, 200] m), per speed.

    It depends on the weights only through the layer norms of the bound, which spectral
    normalisation keeps near 1, and therefore survives fine-tuning. ``max_budget`` follows
    :func:`max_residual_budget` with ``keep_r_max``.
    """
    m, v = _float64(model), _speeds(v)
    out = _a_priori(m.idm, v, torch.full_like(v, m.r_max), residual_bounds(m).expand(len(v), 3), n_scan)
    return {
        **_header(m, v),
        **out,
        "n_holds": int(out["holds"].sum()),
        "max_budget": max_residual_budget(m, v, a_priori=True, n_scan=n_scan, keep_r_max=keep_r_max),
    }


def residual_of(model: ResidualIDM) -> dict[str, Any]:
    """The residual of ``model`` as the a priori certificate sees it (float64, eval mode): ``r_max``,
    ``lipschitz``, their product, the bounds ``[3]`` of ``|dr/d(s, dv, v)|`` and the layer norms."""
    m = _float64(model)
    return {
        "r_max": m.r_max,
        "lipschitz": m.lipschitz,
        "product": m.r_max * m.lipschitz,
        "bounds": residual_bounds(m),
        "layer_norms": m.layer_norms(),
    }


def certify_core(
    params: Mapping[str, float], r_max: float, bounds: Tensor, v: Speeds = V_GRID, n_scan: int = 400,
    delta: float = 4.0,
) -> dict[str, Tensor]:  # fmt: skip
    """A priori certificate of the hybrid with the IDM core ``params`` and any residual with ``|r| <= r_max``
    and ``|dr/d(s, dv, v)| <= bounds`` (D71, D86), per speed of ``v``: the entries of
    :func:`certificate_a_priori` for a ResidualIDM with this core and this residual (``holds``,
    ``guaranteed_margin``, ``s_low``, ``s_high``, ``feasible``, ``s_worst``, ``fs_positive``,
    ``sum_negative``), without its budget. The residual enters through ``r_max`` and ``bounds`` only, so
    one certified residual certifies (or not) any number of cores (D118)."""
    idm = IDM({name: float(params[name]) for name in ("v0", "T", "s0", "a", "b")}, dtype=torch.float64, delta=delta)
    v = _speeds(v)
    bounds = torch.as_tensor(bounds, dtype=torch.float64).cpu().reshape(1, 3).expand(len(v), 3)
    return _a_priori(idm, v, torch.full_like(v, float(r_max)), bounds, n_scan)


def empirical_margin(model: ResidualIDM, v: Speeds = V_GRID, band: Band | None = None) -> Tensor:
    """Margin ``[n]`` of the hybrid itself at its equilibria (``partials``; anchored ones with ``band``),
    NaN without such an equilibrium."""
    m, v = _float64(model), _speeds(v)
    eq, found = _equilibria(m, v, band)
    margin = criterion(*partials(m, eq.s, eq.v))["margin"]
    return torch.where(found, margin, torch.full_like(margin, torch.nan))


def to_json(value: Any) -> Any:
    """Plain JSON values: tensors and arrays as (nested) lists, non-finite numbers as ``None``."""
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
