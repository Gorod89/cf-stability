"""Stability certificate of ResidualIDM (docs/m3_contract.md, section 8; docs/m4_contract.md, 3.1 and 3.3).

``f = f_idm + r`` with ``|r| <= r_max`` and ``|dr/dx_j| <= B_j`` (``ResidualIDM.jacobian_bound``).
At an equilibrium the partial derivatives of the hybrid are those of the IDM plus ``d_j`` with
``|d_j| <= B_j``; the minimum of the criterion over that box is a lower bound of the margin of the
hybrid (the guaranteed margin), and the box gives guaranteed signs of ``f_s`` and ``f_v + f_dv``.
The certificate holds at a speed when the guaranteed margin is ``>= 0`` and both signs are
guaranteed (local and string stability). All computations run in float64 on a CPU copy in eval mode.
With the spacing band of the data (``Band``) the a posteriori forms use the anchored equilibria
(inside the band, or at a speed without band). The a priori minimum over the feasible spacings is
scanned (``_a_priori``, the stored certificates) or found exactly (:func:`a_priori_exact`, M9 review);
:func:`feasible_ends` gives the conditions under which an equilibrium exists among those spacings.
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


def feasible_ends(idm: IDM, v: Tensor, r_max: Tensor | float) -> dict[str, Tensor]:
    """Ends of ``I(v) = {s > 0 : |f_idm(s, 0, v)| <= r_max}`` before the clamping to ``[S_MIN, S_MAX]``, and the
    existence of an equilibrium of the hybrid in it (M9 review). All entries ``[n]``.

    ``f_idm(s, 0, v) = a (1 - (v/v0)^delta) - a (s*/s)^2`` with ``s* = s0 + v T`` rises with ``s`` towards
    ``a (1 - (v/v0)^delta)``; ``(s*/s)^2 = 1 - (v/v0)^delta -+ r_max / a`` gives ``s_low`` and ``s_high``, ``inf``
    where the right side is not positive (``s_low``: no feasible spacing; ``s_high``: no upper end).
    ``limit_margin = a (1 - (v/v0)^delta) - r_max`` (m/s^2). ``bounded`` (``s_high`` finite, i.e.
    ``limit_margin > 0``): ``f_idm = -r_max`` at ``s_low`` and ``+r_max`` at ``s_high``, so for every residual with
    ``|r| <= r_max`` the hybrid ``f_idm + r`` (continuous in ``s``) is ``<= 0`` at ``s_low`` and ``>= 0`` at ``s_high``
    and has an equilibrium in ``I(v)`` (intermediate values; none outside, where ``|f_idm| > r_max``). Otherwise
    ``I(v)`` is unbounded and no equilibrium need exist: ``f_idm = 0.1 - 144 / s^2`` and the constant ``r = -0.2``,
    ``r_max = 0.3``, give ``f_idm + r < 0`` at every spacing. ``within``: ``S_MIN <= s_low`` and ``s_high <= S_MAX``,
    so the clamped interval of the certificate is all of ``I(v)``: the a priori certificate then covers every
    equilibrium, and one exists.
    """
    v0, T, s0, a, _ = idm.theta.detach().double().cpu().unbind()
    s_star = s0 + v * T
    free = 1.0 - (v / v0) ** idm.delta
    low_arg, high_arg = free + r_max / a, free - r_max / a
    s_low = torch.where(low_arg > 0, s_star / torch.sqrt(low_arg.clamp(min=1e-300)), torch.inf)
    s_high = torch.where(high_arg > 0, s_star / torch.sqrt(high_arg.clamp(min=1e-300)), torch.inf)
    return {
        "s_low": s_low,
        "s_high": s_high,
        "limit_margin": a * free - r_max,
        "bounded": high_arg > 0,
        "within": (s_low >= S_MIN) & (s_high <= S_MAX),
    }


def _feasible_interval(idm: IDM, v: Tensor, r_max: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Spacings in ``[S_MIN, S_MAX]`` with ``|f_idm(s, 0, v)| <= r_max`` (an interval: ``f_idm`` rises with ``s``):
    the closed-form ends of :func:`feasible_ends` clamped to ``[S_MIN, S_MAX]``."""
    ends = feasible_ends(idm, v, r_max)
    s_low, s_high = ends["s_low"].clamp(min=S_MIN), ends["s_high"].clamp(max=S_MAX)
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


def _root_real_parts(coef: np.ndarray) -> np.ndarray:
    """Real parts ``[n, d]`` of the roots of the polynomials ``coef [n, d + 1]`` (highest degree first; NaN pads a
    row of lower degree): ``numpy.roots`` row by row, the eigenvalues of the companion matrix, in one batched
    ``numpy.linalg.eigvals`` for the rows whose leading coefficient is not 0."""
    n, d = coef.shape[0], coef.shape[1] - 1
    out = np.full((n, d), np.nan)
    finite = np.isfinite(coef).all(axis=1)
    full = finite & (coef[:, 0] != 0.0)
    if full.any():
        c = coef[full]
        companion = np.zeros((len(c), d, d))
        companion[:, 0, :] = -c[:, 1:] / c[:, :1]
        companion[:, np.arange(1, d), np.arange(d - 1)] = 1.0
        out[full] = np.linalg.eigvals(companion).real
    for k in np.flatnonzero(finite & ~full):
        roots = np.roots(coef[k])
        out[k, : len(roots)] = roots.real
    return out


def a_priori_exact(idm: IDM, v: Tensor, r_max: Tensor | float, bounds: Tensor) -> dict[str, Tensor]:
    """:func:`_a_priori` with the minimum of the guaranteed margin over the feasible spacings found exactly (M9
    review): the same entries, without a scan.

    With ``x = 1/s`` and ``s* = s0 + v T`` the IDM derivatives at ``(s, 0, v)`` (the closed forms of
    ``idm_equilibrium_partials`` at any spacing) are ``F_s = c_s x^3``, ``F_dv = -c_dv x^2`` and
    ``F_v = -k_v - c_v x^2`` with ``c_s = 2 a s*^2``, ``c_dv = a s* v / sqrt(a b)``, ``c_v = 2 a s* T`` and
    ``k_v = a delta v^(delta-1) / v0^delta``.
    :func:`guaranteed_margin` is the smaller over the edges ``q = F_dv + e B_dv`` (``e = -1, +1``) of
    ``p^2 + 2 p q - 2 (F_s + B_s)`` with ``p = clamp(-q, F_v - B_v, F_v + B_v)``. For one edge and one regime of the
    clamp this is a polynomial ``g(x) = alpha x^4 + beta x^3 + gamma x^2 + const``, ``beta = -2 c_s``:

    * ``p = -q`` (value ``-q^2``): ``alpha = -c_dv^2``, ``gamma = 2 e B_dv c_dv``;
    * ``p = F_v -+ B_v = P - c_v x^2``, ``P = -k_v -+ B_v``: ``alpha = c_v^2 + 2 c_v c_dv``,
      ``gamma = -2 P (c_v + c_dv) - 2 c_v e B_dv``.

    The regime of an edge changes where ``-q = F_v -+ B_v``, ``x^2 = (e B_dv - k_v -+ B_v) / (c_dv + c_v)`` (four
    breakpoints). Between consecutive breakpoints the guaranteed margin is the smaller of two such polynomials, and
    the minimum of either over a closed piece lies at its ends or at a root of ``g'(x) = 4 alpha x^3 + 3 beta x^2 +
    2 gamma x`` inside it. Hence the minimum over ``[1/s_high, 1/s_low]`` is attained at an end of the interval, at a
    breakpoint or at a real root of the derivative of one of the six pieces (two edges x three regimes) that lies in
    the interval. These candidates (the real part of every root, :func:`_root_real_parts`: a spurious candidate only
    costs an evaluation; candidates outside the interval are moved into it, whose ends are candidates anyway) are
    evaluated with :func:`partials` and :func:`guaranteed_margin` as in :func:`_a_priori`, and their minimum is the
    minimum over the interval, to rounding (an error ``eps`` of a root moves the value by ``O(eps^2)``). The interval
    and its clamping are those of :func:`_feasible_interval`; the signs are taken at its two ends (``F_s - B_s``
    falls and ``F_v + F_dv + B_v + B_dv`` rises with ``s``, so the ends decide them on the whole interval).
    ``r_max`` broadcasts to ``[n]`` and ``bounds`` to ``[n, 3]``; ``idm`` is used in float64 (a copy otherwise).
    """
    if idm.raw.dtype != torch.float64:
        idm = copy.deepcopy(idm).cpu().double()
    v = _speeds(v)
    n = len(v)
    r_max = torch.as_tensor(r_max, dtype=torch.float64).cpu().expand(n)
    bounds = torch.as_tensor(bounds, dtype=torch.float64).cpu().expand(n, 3)
    s_low, s_high, feasible = _feasible_interval(idm, v, r_max)
    lo, hi = torch.where(feasible, s_low, S_MIN), torch.where(feasible, s_high, S_MIN)
    v0, T, s0, a, b = idm.theta.detach().double().cpu().unbind()
    s_star = s0 + v * T
    c_s, c_dv, c_v = 2.0 * a * s_star**2, a * s_star * v / torch.sqrt(a * b), 2.0 * a * s_star * T
    k_v = a * idm.delta * v ** (idm.delta - 1.0) / v0**idm.delta
    _, b_dv, b_v = bounds.unbind(-1)
    breakpoints, pieces = [], []  # x of the breakpoints; (alpha, beta, gamma) of the pieces
    for e in (-1.0, 1.0):
        edge = e * b_dv  # q = edge - c_dv x^2
        pieces.append((-(c_dv**2), -2.0 * c_s, 2.0 * edge * c_dv))  # p = -q
        for p_end in (-k_v - b_v, -k_v + b_v):  # p = p_end - c_v x^2, the lower and the upper end of p
            pieces.append((c_v**2 + 2.0 * c_v * c_dv, -2.0 * c_s, -2.0 * p_end * (c_v + c_dv) - 2.0 * c_v * edge))
            y = (edge + p_end) / (c_dv + c_v)  # x^2 where -q = p_end - c_v x^2
            breakpoints.append(torch.where(y > 0, torch.sqrt(y.clamp(min=1e-300)), torch.nan))
    derivative = torch.stack([torch.stack((4.0 * al, 3.0 * be, 2.0 * ga, torch.zeros_like(al)), dim=-1)
                              for al, be, ga in pieces], dim=1)  # [n, 6, 4]  # fmt: skip
    roots = torch.from_numpy(_root_real_parts(derivative.reshape(-1, 4).numpy())).reshape(n, -1)
    x = torch.cat((torch.stack(breakpoints, dim=1), roots), dim=1)
    s = 1.0 / x
    s = torch.where(torch.isfinite(s) & (s > 0), s, lo[:, None])
    s = torch.cat((lo[:, None], hi[:, None], torch.minimum(torch.maximum(s, lo[:, None]), hi[:, None])), dim=1)
    k_cand = s.shape[1]
    f = partials(idm, s.reshape(-1), v.repeat_interleave(k_cand))
    g = _guarantees(*(t.reshape(n, k_cand) for t in f), bounds[:, None, :])
    value, k = g["guaranteed_margin"].min(dim=1)
    nan = torch.full_like(value, torch.nan)
    margin = torch.where(feasible, value, nan)
    fs_positive = g["fs_positive"][:, :2].all(dim=1) & feasible
    sum_negative = g["sum_negative"][:, :2].all(dim=1) & feasible
    return {
        "s_low": torch.where(feasible, s_low, nan),
        "s_high": torch.where(feasible, s_high, nan),
        "feasible": feasible,
        "guaranteed_margin": margin,
        "s_worst": torch.where(feasible, s[torch.arange(n), k], nan),
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
