"""Robust (non-equilibrium) certificate of a car-following law (docs/m11_contract.md, D130).

For deviations from a trajectory the follower obeys, in the convention ``dv = v - v_lead`` of the project
(:mod:`cf_stability.stability.analytic`),

    e' = w - u,    u' = f_s e + f_dv (u - w) + f_v u,    output u,

with ``e`` the spacing error, ``u`` the speed error and ``w`` the leader's speed perturbation (the contract writes the
``f_dv`` term as ``f_dv (w - u)``, the same system in the convention ``dv = v_lead - v``). Frozen, it has the transfer
function ``G(s) = (f_s - s f_dv) / (s^2 + f_s - s (f_dv + f_v))`` of ``analytic.py``. When the derivatives move inside a
box ``f_s in [a_1, b_1]``, ``f_dv in [a_3, b_3]``, ``f_v in [a_2, b_2]`` the system is a polytopic LPV system: its
matrices ``A = [[0, -1], [f_s, f_dv + f_v]]``, ``B = [1, -f_dv]^T``, ``C = [0, 1]`` are affine in the derivatives, so the
polytope of systems is the convex hull of the eight corner systems. The bounded-real lemma with one common storage
function ``V = x^T P x`` (``P`` 2 x 2, three parameters),

    M_i(P, gamma) = [[A_i^T P + P A_i + C^T C, P B_i], [B_i^T P, -gamma^2]] < 0   at every vertex i,

is affine in ``(A, B)`` for fixed ``P``: it then holds on the whole polytope and ``dV/dt + u^2 - gamma^2 w^2 < 0`` along
every trajectory whose derivatives stay in the box, so the L2 gain from ``w`` to ``u`` is below ``gamma`` (incremental
string stability over the region when ``gamma <= 1``). No SDP solver: ``P = L L^T`` (``L`` lower triangular), ``gamma``
by bisection, and at every ``gamma`` the largest eigenvalue of the vertex matrices minimised over ``L`` with scipy
(Nelder-Mead from several starts, then BFGS from the best point). The bisection is bracketed from below by the largest
vertex H-infinity norm and from above by the smallest gain that a single ``P`` certifies by the Schur complement
(:func:`schur_gain`, minimised over ``L`` the same way), whose ``P`` warm-starts the eigenvalue minimisation. Every
certificate is verified afterwards in float64: all vertex eigenvalues ``<= -VERTEX_TOL`` and ``P`` positive definite.

Every vertex with ``f_s > 0`` has ``G(0) = 1``: the L2 gain of the polytope is at least 1 and ``gamma < 1`` cannot be
certified; a box is string stable in this sense when its infimum is 1, i.e. a certificate exists at ``1 + GAMMA_TOL``
(:func:`string_stable`). A vertex whose ``A`` is not Hurwitz makes the LMI infeasible for every ``gamma`` (``A^T P + P A
< -C^T C <= 0`` needs a Hurwitz ``A``).

:func:`law_boxes` builds the boxes of a law over a region: the IDM core's derivative ranges over the spacings of the
region (exact: at ``dv = 0`` the IDM derivatives are monotone in the spacing, so the ends of an interval give the
range), plus the residual budget ``|dr/dx_j| <= B_j`` for the hybrids, or empirical ranges of the partials sampled at
states of the region for laws without a core (a necessary-condition check only: the sampled range is inside the
true one).
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize

VERTEX_TOL = 1e-9  # a verified certificate: every vertex eigenvalue <= -VERTEX_TOL and P positive definite
GAMMA_TOL = 1e-3  # string stable over a box: a verified certificate at gamma = 1 + GAMMA_TOL
GAMMA_MAX = 1e4  # no certificate up to this gain: "infeasible"
BISECTION_RTOL = 1e-4
SCHUR_BRACKET = 1e-3  # the bisection searches down to this relative distance below the smallest Schur gain found
N_RANDOM_STARTS = 4
HINF_GRID = np.geomspace(1e-4, 1e2, 4001)  # rad/s, the frequency sweep of the vertex H-infinity norms
INFEASIBLE = "infeasible"


# ------------------------------------------------------------------------------------------------- boxes


@dataclasses.dataclass(frozen=True)
class Box:
    """Derivative box of a law over a region: closed intervals ``(low, high)`` of ``f_s``, ``f_dv`` and ``f_v``."""

    f_s: tuple[float, float]
    f_dv: tuple[float, float]
    f_v: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("f_s", "f_dv", "f_v"):
            low, high = (float(x) for x in getattr(self, name))
            if not (math.isfinite(low) and math.isfinite(high)) or low > high:
                raise ValueError(f"{name}: a finite interval low <= high, got {(low, high)}")
            object.__setattr__(self, name, (low, high))

    def vertices(self) -> np.ndarray:
        """The eight corners ``[8, 3]`` in the order ``(f_s, f_dv, f_v)``."""
        return np.array(list(itertools.product(self.f_s, self.f_dv, self.f_v)), dtype=np.float64)

    def widened(self, bounds: Sequence[float]) -> "Box":
        """The box plus ``[-B_j, B_j]`` per derivative, ``bounds`` in the order ``(s, dv, v)``."""
        b_s, b_dv, b_v = (float(b) for b in bounds)
        return Box((self.f_s[0] - b_s, self.f_s[1] + b_s), (self.f_dv[0] - b_dv, self.f_dv[1] + b_dv),
                   (self.f_v[0] - b_v, self.f_v[1] + b_v))  # fmt: skip

    @classmethod
    def of(cls, f_s: np.ndarray, f_dv: np.ndarray, f_v: np.ndarray) -> "Box":
        """The smallest box holding the derivatives ``f_s, f_dv, f_v`` (arrays of equal length, the finite ones)."""
        arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (f_s, f_dv, f_v)]
        keep = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
        if not keep.any():
            raise ValueError("no finite derivatives")
        return cls(*((float(x[keep].min()), float(x[keep].max())) for x in arrays))

    def as_dict(self) -> dict[str, float]:
        return {f"{name}_{end}": value for name in ("f_s", "f_dv", "f_v")
                for end, value in zip(("low", "high"), getattr(self, name))}  # fmt: skip


def vertex_systems(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``A [n, 2, 2]``, ``B [n, 2]`` and ``C [2]`` of the perturbation dynamics at the derivatives ``vertices [n, 3]``
    (``f_s, f_dv, f_v``)."""
    p = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    f_s, f_dv, f_v = p[:, 0], p[:, 1], p[:, 2]
    A = np.zeros((len(p), 2, 2))
    A[:, 0, 1] = -1.0
    A[:, 1, 0] = f_s
    A[:, 1, 1] = f_dv + f_v
    B = np.stack((np.ones(len(p)), -f_dv), axis=1)
    return A, B, np.array([0.0, 1.0])


def transfer(vertices: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``G(i w) [n, n_w]`` of the frozen systems (equal to ``analytic.transfer_continuous``)."""
    p = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    f_s, f_dv, f_v = (p[:, k : k + 1] for k in range(3))
    s = 1j * np.asarray(omega, dtype=np.float64)[None, :]
    return (f_s - s * f_dv) / (s**2 + f_s - s * (f_dv + f_v))


def hurwitz(vertices: np.ndarray) -> np.ndarray:
    """``[n]``: ``A`` is Hurwitz (``f_s > 0`` and ``f_dv + f_v < 0``) at every vertex."""
    p = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    return (p[:, 0] > 0.0) & (p[:, 1] + p[:, 2] < 0.0)


def hinf_norms(vertices: np.ndarray, omega: np.ndarray = HINF_GRID) -> np.ndarray:
    """H-infinity norm ``[n]`` of every frozen (Hurwitz) vertex system: the largest ``|G(i w)|`` over a log grid,
    refined by a bounded scalar search around the best grid point; ``inf`` for a vertex that is not Hurwitz."""
    p = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    out = np.full(len(p), np.inf)
    gains = np.abs(transfer(p, omega))
    for i in np.flatnonzero(hurwitz(p)):
        k = int(np.argmax(gains[i]))
        best = float(gains[i, k])
        lo, hi = omega[max(k - 1, 0)], omega[min(k + 1, len(omega) - 1)]
        if hi > lo:
            res = optimize.minimize_scalar(lambda w: -abs(transfer(p[i : i + 1], np.array([w]))[0, 0]),
                                           bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})  # fmt: skip
            best = max(best, -float(res.fun))
        out[i] = max(best, float(np.abs(transfer(p[i : i + 1], np.array([0.0])))[0, 0]))
    return out


# ------------------------------------------------------------------------------------------- the LMI


def lmi_matrices(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, gamma: float) -> np.ndarray:
    """The vertex matrices ``[n, 3, 3]`` of the bounded-real lemma at ``P`` and ``gamma``."""
    n = A.shape[0]
    M = np.zeros((n, 3, 3))
    M[:, :2, :2] = np.transpose(A, (0, 2, 1)) @ P + P @ A + np.outer(C, C)[None]
    PB = B @ P.T  # [n, 2]: P B_i
    M[:, :2, 2] = PB
    M[:, 2, :2] = PB
    M[:, 2, 2] = -gamma**2
    return M


def _P_of(x: np.ndarray) -> np.ndarray:
    """``P = L L^T`` of the lower triangular ``L = [[x0, 0], [x1, x2]]``."""
    L = np.array([[x[0], 0.0], [x[1], x[2]]])
    return L @ L.T


def max_eigenvalue(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, gamma: float) -> float:
    """Largest eigenvalue of the vertex matrices (float64, ``numpy.linalg.eigvalsh``); ``inf`` for a ``P`` that is not
    finite."""
    if not np.all(np.isfinite(P)):
        return math.inf
    try:
        return float(np.linalg.eigvalsh(lmi_matrices(P, A, B, C, gamma)).max())
    except np.linalg.LinAlgError:
        return math.inf


def verify(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, gamma: float, tol: float = VERTEX_TOL) -> bool:
    """The certificate ``(P, gamma)`` holds in float64: ``P`` symmetric positive definite and every eigenvalue of every
    vertex matrix ``<= -tol``."""
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (2, 2) or not np.all(np.isfinite(P)) or abs(P[0, 1] - P[1, 0]) > 1e-12 * max(1.0, np.abs(P).max()):
        return False
    if np.linalg.eigvalsh(P).min() <= 0.0:
        return False
    return max_eigenvalue(P, A, B, C, gamma) <= -tol


def _starts(A: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Starting points of ``L``: the identity, the Cholesky factor of the Lyapunov solution of the mean system
    (``A_c^T P + P A_c = -I``) at three scales, and random factors."""
    starts = [np.array([1.0, 0.0, 1.0])]
    A_c = A.mean(axis=0)
    try:
        from scipy.linalg import solve_continuous_lyapunov

        P0 = solve_continuous_lyapunov(A_c.T, -np.eye(2))
        P0 = 0.5 * (P0 + P0.T)
        if np.linalg.eigvalsh(P0).min() > 0:
            L0 = np.linalg.cholesky(P0)
            for scale in (0.1, 1.0, 10.0):
                L = L0 * math.sqrt(scale)
                starts.append(np.array([L[0, 0], L[1, 0], L[1, 1]]))
    except (ValueError, np.linalg.LinAlgError):
        pass
    for _ in range(N_RANDOM_STARTS):
        starts.append(np.exp(rng.normal(0.0, 1.0, 3)) * rng.choice([-1.0, 1.0], 3) * np.array([1.0, 0.3, 1.0]))
    return starts


def minimise_eigenvalue(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, gamma: float, starts: Sequence[np.ndarray],
    stop_below: float = -10 * VERTEX_TOL,
) -> tuple[float, np.ndarray]:  # fmt: skip
    """Smallest largest vertex eigenvalue found over ``L`` (Nelder-Mead from every start, then BFGS from the best
    point) and its ``L``. Stops early once a value below ``stop_below`` is found (enough for feasibility)."""

    def objective(x: np.ndarray) -> float:
        value = max_eigenvalue(_P_of(x), A, B, C, gamma)
        return value if math.isfinite(value) else 1e300

    best_x, best = np.asarray(starts[0], dtype=np.float64), math.inf
    for start in starts:
        res = optimize.minimize(objective, np.asarray(start, dtype=np.float64), method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000, "maxfev": 8000})  # fmt: skip
        if res.fun < best:
            best, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
        if best < stop_below:
            return best, best_x
    res = optimize.minimize(objective, best_x, method="BFGS", options={"gtol": 1e-12, "maxiter": 500})
    if res.fun < best:
        best, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
    return best, best_x


def feasible_at(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, gamma: float, starts: Sequence[np.ndarray]
) -> tuple[bool, np.ndarray | None, float]:
    """``(verified, P, smallest largest eigenvalue)`` of the LMI at ``gamma``; ``P`` None when not verified."""
    value, x = minimise_eigenvalue(A, B, C, gamma, starts)
    P = _P_of(x)
    if verify(P, A, B, C, gamma):
        return True, P, value
    return False, None, value


def _L_of(P: np.ndarray) -> np.ndarray:
    """The parameters ``x`` of the lower triangular Cholesky factor of a positive definite ``P``."""
    l11 = math.sqrt(P[0, 0])
    return np.array([l11, P[1, 0] / l11, math.sqrt(max(P[1, 1] - P[1, 0] ** 2 / P[0, 0], 1e-300))])


def schur_gain(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """The smallest ``gamma`` for which ``P`` satisfies the LMI at every vertex: with ``N_i = A_i^T P + P A_i + C^T C``
    (which must be negative definite) the Schur complement gives ``gamma^2 > max_i B_i^T P (-N_i)^-1 P B_i``; ``inf``
    when some ``N_i`` is not negative definite."""
    if not np.all(np.isfinite(P)):
        return math.inf
    N = np.transpose(A, (0, 2, 1)) @ P + P @ A + np.outer(C, C)[None]
    try:
        if not np.linalg.eigvalsh(N).max() < 0.0:
            return math.inf
        PB = B @ P.T
        value = np.einsum("ni,ni->n", PB, np.linalg.solve(-N, PB[..., None])[..., 0])
    except np.linalg.LinAlgError:  # numerically singular N: no certified gain from this P
        return math.inf
    top = float(value.max())
    return float(math.sqrt(max(top, 0.0))) if math.isfinite(top) else math.inf


class _Reached(Exception):
    """Raised inside an objective once its value is at or below the target (ends the search early)."""

    def __init__(self, x: np.ndarray, value: float) -> None:
        super().__init__()
        self.x, self.value = x, value


def minimise_schur_gain(A: np.ndarray, B: np.ndarray, C: np.ndarray, starts: Sequence[np.ndarray],
                        target: float | None = None) -> tuple[float, np.ndarray]:  # fmt: skip
    """The smallest :func:`schur_gain` found over ``L`` (Nelder-Mead from every start, then BFGS from the best point);
    it brackets the bisection of :func:`robust_certificate` from above. ``target``: stop as soon as a gain at or below
    it is found. ``(gain, x)``; ``inf`` when no start reaches a ``P`` with every ``N_i`` negative definite."""

    def objective(x: np.ndarray) -> float:
        P = _P_of(x)
        if not np.all(np.isfinite(P)):
            return 1e300
        N = np.transpose(A, (0, 2, 1)) @ P + P @ A + np.outer(C, C)[None]
        try:
            worst = float(np.linalg.eigvalsh(N).max())
        except np.linalg.LinAlgError:
            return 1e300
        if not worst < 0.0:  # the penalty leads the search into the region where the gain is defined
            return 1e6 * (1.0 + worst) if math.isfinite(worst) else 1e300
        value = schur_gain(P, A, B, C)
        if not math.isfinite(value):
            return 1e6
        if target is not None and value <= target:
            raise _Reached(np.array(x, dtype=np.float64), value)
        return value

    best, best_x = math.inf, np.asarray(starts[0], dtype=np.float64)
    try:
        for start in starts:
            res = optimize.minimize(objective, np.asarray(start, dtype=np.float64), method="Nelder-Mead",
                                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000})  # fmt: skip
            if res.fun < best:
                best, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
        res = optimize.minimize(objective, best_x, method="BFGS", options={"gtol": 1e-12, "maxiter": 500})
        if res.fun < best:
            best, best_x = float(res.fun), np.asarray(res.x, dtype=np.float64)
    except _Reached as reached:
        return reached.value, reached.x
    return (best if best < 1e5 else math.inf), best_x


def robust_certificate(
    box: Box | np.ndarray, *, gamma_max: float = GAMMA_MAX, rtol: float = BISECTION_RTOL, seed: int = 0
) -> dict[str, Any]:
    """Smallest verified ``gamma`` of the bounded-real LMI over a box (or over the convex hull of the derivative
    points ``[n, 3]``), by bisection to a relative tolerance ``rtol``: at every ``gamma`` the largest vertex eigenvalue
    minimised over ``L`` (:func:`feasible_at`, Nelder-Mead and BFGS) and the result verified in float64. The bracket:
    below, the largest vertex H-infinity norm (the common storage function can only add to it), raised to
    ``(1 - SCHUR_BRACKET)`` times the smallest Schur-complement gain over ``L`` (:func:`minimise_schur_gain`) when no
    certificate exists there; above, the first verified gain at or above that Schur gain (then doubling up to
    ``gamma_max``), whose ``P`` warm-starts the eigenvalue minimisation.

    ``{status, gamma, P, max_eigenvalue, lower_bound, schur_gain, n_vertices, hurwitz, string_stable, vertex_hinf}``;
    ``status`` ``certified`` with the verified ``gamma`` and ``P``, or ``infeasible`` (a vertex that is not Hurwitz, no
    ``P`` with a negative definite ``A^T P + P A + C^T C`` at every vertex, or no certificate up to ``gamma_max``)
    with ``gamma`` None."""
    vertices = box.vertices() if isinstance(box, Box) else np.atleast_2d(np.asarray(box, dtype=np.float64))
    A, B, C = vertex_systems(vertices)
    out: dict[str, Any] = {"status": INFEASIBLE, "gamma": None, "P": None, "max_eigenvalue": None,
                           "n_vertices": int(len(vertices)), "hurwitz": bool(hurwitz(vertices).all()),
                           "string_stable": False, "schur_gain": None}  # fmt: skip
    norms = hinf_norms(vertices)
    out["vertex_hinf"] = float(norms.max())
    out["lower_bound"] = float(norms.max())
    if not out["hurwitz"]:
        out["reason"] = "a vertex is not Hurwitz (f_s <= 0 or f_dv + f_v >= 0)"
        return out
    rng = np.random.default_rng(seed)
    starts = _starts(A, rng)
    # the set of P with gamma(P) <= g is convex (an LMI in P): the identity and the Lyapunov solutions of the mean
    # system suffice as starts; the random ones only when no P with N_i < 0 is found from them
    bound, x_schur = minimise_schur_gain(A, B, C, starts[:4])
    if not math.isfinite(bound):
        bound, x_schur = minimise_schur_gain(A, B, C, starts)
    out["schur_gain"] = None if not math.isfinite(bound) else bound
    low = float(norms.max())
    if not math.isfinite(bound):
        # no P found with A^T P + P A + C^T C < 0 at every vertex: try the eigenvalue minimisation at gamma_max
        ok, P, value = feasible_at(A, B, C, gamma_max, starts)
        if not ok:
            out["reason"] = "no common quadratic storage function (A^T P + P A + C^T C < 0 at every vertex)"
            return out
        found = (gamma_max, P, value)
    else:
        found = None
        high = max(bound * (1.0 + 1e-9), low * (1.0 + rtol))
        warm = [x_schur, *starts]
        while high <= gamma_max:
            ok, P, value = feasible_at(A, B, C, high, warm)
            if ok:
                found = (high, P, value)
                break
            low = max(low, high)
            high = high * (1.0 + 10 * rtol) if high < bound * 1.01 else 2.0 * high  # near the bound: small steps
        if found is None:
            out["reason"] = f"no verified certificate up to gamma = {gamma_max:g}"
            return out
    high, P, value = found
    warm = [_L_of(P), x_schur]  # the bisection below the first verified gain: warm starts only
    if math.isfinite(bound):
        # the smallest Schur gain is the infimum when its search found the optimum (the gain of a P is
        # quasi-convex in P): search down to SCHUR_BRACKET below it, unless a certificate exists there already
        trial = max(low, bound * (1.0 - SCHUR_BRACKET))
        if trial > low:
            ok, P_low, value_low = feasible_at(A, B, C, trial, warm)
            if ok:
                high, P, value = trial, P_low, value_low
                warm[0] = _L_of(P)
            else:
                low = trial
    while high - low > rtol * high:
        mid = 0.5 * (low + high)
        ok, P_mid, value_mid = feasible_at(A, B, C, mid, warm)
        if ok:
            high, P, value = mid, P_mid, value_mid
            warm[0] = _L_of(P)
        else:
            low = mid
    out.update(status="certified", gamma=float(high), P=P, max_eigenvalue=float(max_eigenvalue(P, A, B, C, high)))
    out["string_stable"] = bool(high <= 1.0 + GAMMA_TOL)
    return out


def robust_gain(box: Box | np.ndarray, **options: Any) -> float | str:
    """The verified ``gamma`` of :func:`robust_certificate`, or ``"infeasible"``."""
    result = robust_certificate(box, **options)
    return result["gamma"] if result["status"] == "certified" else INFEASIBLE


def string_stable(box: Box | np.ndarray, *, tol: float = GAMMA_TOL, seed: int = 0) -> tuple[bool, np.ndarray | None]:
    """A verified certificate at ``gamma = 1 + tol`` (the infimum of the gain is 1 for every box of Hurwitz vertices
    with ``f_s > 0``); ``(holds, P)``."""
    vertices = box.vertices() if isinstance(box, Box) else np.atleast_2d(np.asarray(box, dtype=np.float64))
    if not hurwitz(vertices).all():
        return False, None
    A, B, C = vertex_systems(vertices)
    if hinf_norms(vertices).max() > 1.0 + tol:
        return False, None
    starts = _starts(A, np.random.default_rng(seed))[:3]  # the identity and the Lyapunov solutions of the mean system
    bound, x = minimise_schur_gain(A, B, C, starts, target=1.0 + 0.5 * tol)
    if not bound <= 1.0 + tol:  # the smallest gain any P certifies is above 1 + tol (or no P has N_i < 0)
        return False, None
    ok, P, _ = feasible_at(A, B, C, 1.0 + tol, [x, *starts])
    return ok, P


# --------------------------------------------------------------------------------------- budget allocation


def widen(region: Box | np.ndarray, bounds: Sequence[float]) -> Box | np.ndarray:
    """A box widened by ``[-B_j, B_j]``, or the vertices ``[n, 3]`` of a polytope summed with the corners of that box
    (the Minkowski sum, whose convex hull the summed vertices span)."""
    if isinstance(region, Box):
        return region.widened(bounds)
    b = np.asarray([float(x) for x in bounds])
    corners = np.array(list(itertools.product(*((-x, x) for x in b))))
    points = np.atleast_2d(np.asarray(region, dtype=np.float64))
    return (points[:, None, :] + corners[None, :, :]).reshape(-1, 3) if b.any() else points


def allocate_budget(
    cores: Box | np.ndarray | Sequence[Box | np.ndarray], *, tol: float = GAMMA_TOL, cycles: int = 6,
    step_rtol: float = 1e-3, b_max: float = 10.0, seed: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Residual budgets ``(B_s, B_dv, B_v)`` that maximise ``B_s + B_dv + B_v`` subject to the certificate at
    ``gamma = 1 + tol`` on every core region widened by them (:func:`widen`; a box, the vertices of a polytope, or a
    list of them, e.g. one per speed: all must hold). Coordinate ascent: every coordinate in turn raised to the largest
    value that keeps the certificate, by doubling and bisection to ``step_rtol``; ``cycles`` rounds or until no
    coordinate grows. ``{feasible, bounds, total, cycles, tests}``; ``feasible`` False when a core region itself has no
    certificate (every budget 0)."""
    regions = [cores] if isinstance(cores, Box) or (isinstance(cores, np.ndarray) and cores.ndim == 2) else list(cores)
    count = {"tests": 0}

    def test(bounds: np.ndarray) -> bool:
        count["tests"] += 1
        return all(string_stable(widen(region, bounds), tol=tol, seed=seed)[0] for region in regions)

    bounds = np.zeros(3)
    if not test(bounds):
        return {"feasible": False, "bounds": bounds.tolist(), "total": 0.0, "cycles": 0, "tests": count["tests"]}
    done = 0
    for done in range(1, cycles + 1):
        grown = False
        for j in range(3):
            def ok(value: float, j: int = j) -> bool:
                trial = bounds.copy()
                trial[j] = value
                return test(trial)

            low = bounds[j]
            high = max(2.0 * low, 1e-4)
            while high <= b_max and ok(high):
                low, high = high, 2.0 * high
            if high > b_max:
                high = b_max
                if ok(high):
                    low = high
            while high - low > step_rtol * max(high, 1e-6) and high > low:
                mid = 0.5 * (low + high)
                low, high = (mid, high) if ok(mid) else (low, mid)
            if low > bounds[j] * (1.0 + step_rtol) + 1e-12:
                grown = True
            bounds[j] = low
        if not grown:
            break
    return {"feasible": True, "bounds": bounds.tolist(), "total": float(bounds.sum()), "cycles": done,
            "tests": count["tests"]}  # fmt: skip


def idm_curve_polytope(params: Mapping[str, float], v: float, s_low: float, s_high: float, n_tangents: int = 8,
                       delta: float = 4.0) -> np.ndarray:  # fmt: skip
    """Vertices ``[k, 3]`` (``f_s, f_dv, f_v``) of a polytope that holds the derivative curve of an IDM at the states
    ``(s, 0, v)``, ``s`` in ``[s_low, s_high]``: an outer approximation of the curve's convex hull, much tighter than the
    box of its ranges (the derivatives move together with ``s``).

    With ``y = 1 / s^2`` and ``s* = s0 + v T`` the curve is ``(c_s y^1.5, -c_dv y, -k_v - c_v y)`` (``c_s = 2 a s*^2``,
    ``c_dv = a s* v / sqrt(a b)``, ``c_v = 2 a s* T``, ``k_v = a delta v^(delta-1) / v0^delta``): it lies in a plane and
    is convex in it (``y^1.5``), so the chord between its ends lies above it and the tangents at ``n_tangents + 1``
    points below it; the ends and the intersections of consecutive tangents are the vertices of a polygon that contains
    it. A single spacing gives its point."""
    v0, T, s0, a, b = (float(params[k]) for k in ("v0", "T", "s0", "a", "b"))
    s_star = s0 + v * T
    c_s, c_dv, c_v = 2.0 * a * s_star**2, a * s_star * v / math.sqrt(a * b), 2.0 * a * s_star * T
    k_v = a * delta * v ** (delta - 1.0) / v0**delta
    r = np.sqrt(np.linspace(1.0 / s_high**2, 1.0 / s_low**2, n_tangents + 1)) if s_high > s_low else         np.array([1.0 / s_low])  # r = sqrt(y) = 1 / s  # fmt: skip
    y = [r[0] ** 2]
    f = [c_s * r[0] ** 3]
    for r0, r1 in zip(r[:-1], r[1:]):
        y_cross = (r0 * r0 + r0 * r1 + r1 * r1) / 3.0  # where the tangents at r0^2 and r1^2 meet
        y.append(y_cross)
        f.append(c_s * (-0.5 * r0**3 + 1.5 * r0 * y_cross))
    if len(r) > 1:
        y.append(r[-1] ** 2)
        f.append(c_s * r[-1] ** 3)
    y_a, f_a = np.asarray(y), np.asarray(f)
    return np.column_stack((f_a, -c_dv * y_a, -k_v - c_v * y_a))


def hull_vertices(points: np.ndarray) -> np.ndarray:
    """The vertices of the convex hull of ``points [n, 3]`` (``scipy.spatial.ConvexHull``); the points themselves when
    they are too few or flat for a hull in three dimensions."""
    points = np.unique(np.atleast_2d(np.asarray(points, dtype=np.float64)), axis=0)
    if len(points) <= 4:
        return points
    from scipy.spatial import ConvexHull, QhullError

    try:
        return points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return points


# ------------------------------------------------------------------------------------- boxes of a law


def idm_box(params: Mapping[str, float], v: float, s_low: float, s_high: float, delta: float = 4.0) -> Box:
    """Exact derivative box of an IDM (``params``: ``v0, T, s0, a, b``) at the states ``(s, 0, v)``, ``s`` in
    ``[s_low, s_high]``: at ``dv = 0`` ``f_s = c_s / s^3`` falls, ``f_dv = -c_dv / s^2`` and ``f_v = -k_v - c_v / s^2``
    rise with ``s``, so the two ends give the ranges (the derivatives by autograd of
    :func:`cf_stability.models.idm.idm_acc`, as everywhere in the project)."""
    f_s, f_dv, f_v = idm_partials(params, np.array([s_low, s_high]), np.array([v, v]), delta=delta)
    return Box.of(f_s, f_dv, f_v)


def idm_partials(
    params: Mapping[str, float], s: np.ndarray, v: np.ndarray, dv: np.ndarray | None = None, delta: float = 4.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(f_s, f_dv, f_v)`` of an IDM at the states ``(s, dv, v)`` (``dv = 0`` by default), float64 autograd."""
    import torch

    from cf_stability.models.idm import idm_acc

    s_t = torch.as_tensor(np.asarray(s, dtype=np.float64)).clone().requires_grad_(True)
    v_t = torch.as_tensor(np.asarray(v, dtype=np.float64)).clone().requires_grad_(True)
    dv_np = np.zeros_like(np.asarray(s, dtype=np.float64)) if dv is None else np.asarray(dv, dtype=np.float64)
    dv_t = torch.as_tensor(dv_np).clone().requires_grad_(True)
    p = {key: float(params[key]) for key in ("v0", "T", "s0", "a", "b")}
    with torch.enable_grad():
        acc = idm_acc(s_t, dv_t, v_t, p["v0"], p["T"], p["s0"], p["a"], p["b"], delta=delta)
        grads = torch.autograd.grad(acc.sum(), (s_t, dv_t, v_t), allow_unused=True)
    zero = np.zeros_like(dv_np)
    return tuple(zero if g is None else g.detach().numpy() for g in grads)  # type: ignore[return-value]


def model_partials(
    model: Any, s: np.ndarray, v: np.ndarray, dv: np.ndarray | None = None, device: str = "cpu"
) -> tuple[np.ndarray, ...]:
    """``(f_s, f_dv, f_v)`` of a model at the states ``(s, dv, v)`` in its memoryless view (the window filled with the
    state, derivatives with respect to a common shift of the whole window, :func:`analytic.partials`): autograd in
    float64 (on ``device``), central differences of the audit's step for a model with ``differentiable = False``."""
    import torch

    from cf_stability.stability.analytic import _fd_steps
    from cf_stability.stability.equilibrium import steady_acc

    m = copy.deepcopy(model).to(device).double().eval()
    s_t = torch.as_tensor(np.asarray(s, dtype=np.float64), device=device)
    v_t = torch.as_tensor(np.asarray(v, dtype=np.float64), device=device)
    dv_t = torch.zeros_like(s_t) if dv is None else torch.as_tensor(np.asarray(dv, dtype=np.float64), device=device)
    if not getattr(m, "differentiable", True):
        steps = _fd_steps(m, s_t)
        out = []
        with torch.no_grad():
            for k, (ds, ddv, dvv) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                h = steps[k]
                up = steady_acc(m, s_t + ds * h, v_t + dvv * h, dv_t + ddv * h)
                down = steady_acc(m, s_t - ds * h, v_t - dvv * h, dv_t - ddv * h)
                out.append(((up - down) / (2.0 * h)).cpu().numpy())
        return tuple(out)
    s_t, v_t, dv_t = (x.clone().requires_grad_(True) for x in (s_t, v_t, dv_t))
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        acc = steady_acc(m, s_t, v_t, dv_t)
        grads = torch.autograd.grad(acc.sum(), (s_t, dv_t, v_t), allow_unused=True)
    zero = np.zeros(len(s_t))
    return tuple(zero if g is None else g.detach().cpu().numpy() for g in grads)


def region_band(context: Mapping[str, Any], speeds: Sequence[float]) -> dict[float, tuple[float, float]]:
    """The spacing band of a run's training context (``context["band"]``, D78) at the grid ``speeds`` where it exists:
    ``{v: (s_low, s_high)}``, the edges interpolated linearly in the speed as the audit does (``Band.at``)."""
    import torch

    from cf_stability.stability.equilibrium import Band

    band = Band.from_mapping(context["band"])
    v = torch.as_tensor([float(x) for x in speeds], dtype=torch.float64)
    low, high, has = band.at(v)
    return {float(vi): (float(lo), float(hi)) for vi, lo, hi, h in zip(v.tolist(), low.tolist(), high.tolist(),
                                                                        has.tolist()) if h}  # fmt: skip


def region_feasible(params: Mapping[str, float], r_max: float, speeds: Sequence[float], delta: float = 4.0
                    ) -> dict[float, tuple[float, float]]:  # fmt: skip
    """The feasible gaps ``I(v)`` of the certificate (``certificate.feasible_ends`` clamped to ``[S_MIN, S_MAX]`` as
    ``_feasible_interval``) of an IDM core and ``r_max`` at the grid ``speeds`` where ``I(v)`` is not empty; with
    ``r_max = 0`` the equilibrium spacing alone (both ends equal)."""
    import torch

    from cf_stability.models.idm import IDM
    from cf_stability.stability.certificate import _feasible_interval

    idm = IDM({key: float(params[key]) for key in ("v0", "T", "s0", "a", "b")}, dtype=torch.float64, delta=delta)
    v = torch.as_tensor([float(x) for x in speeds], dtype=torch.float64)
    low, high, ok = _feasible_interval(idm, v, torch.full_like(v, float(r_max)))
    return {float(vi): (float(lo), float(hi)) for vi, lo, hi, k in zip(v.tolist(), low.tolist(), high.tolist(),
                                                                        ok.tolist()) if k}  # fmt: skip


def law_boxes(
    region: Mapping[float, tuple[float, float]], *, core: Mapping[str, float] | None = None,
    bounds: Sequence[float] | None = None, model: Any = None, n_states: int = 20000, seed: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Derivative boxes of a law over a region ``{v: (s_low, s_high)}`` of states ``(s, 0, v)``: per speed and one box
    over the whole region (every speed together).

    With ``core`` (IDM parameters): the exact ranges of the core over the spacings (:func:`idm_box`), widened by the
    residual budget ``bounds`` (``|dr/d(s, dv, v)| <= B``) when given. With ``model`` instead: the empirical ranges of
    its memoryless partials (:func:`model_partials`) at ``n_states`` states drawn uniformly over the region (a speed of
    the region uniformly, then a spacing uniformly in its interval), a necessary-condition check. ``{per_speed: {v:
    Box}, whole: Box, kind, n_states}``."""
    speeds = sorted(region)
    if not speeds:
        raise ValueError("an empty region")
    per_speed: dict[float, Box] = {}
    if core is not None:
        for v in speeds:
            box = idm_box(core, v, *region[v])
            per_speed[v] = box.widened(bounds) if bounds is not None else box
        kind = "core + residual budget" if bounds is not None else "core"
        n = 0
    elif model is not None:
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(speeds), n_states)
        v = np.asarray(speeds, dtype=np.float64)[pick]
        lo = np.array([region[x][0] for x in speeds])[pick]
        hi = np.array([region[x][1] for x in speeds])[pick]
        s = lo + (hi - lo) * rng.random(n_states)
        f_s, f_dv, f_v = model_partials(model, s, v)
        for k, speed in enumerate(speeds):
            mine = pick == k
            if mine.any():
                per_speed[speed] = Box.of(f_s[mine], f_dv[mine], f_v[mine])
        kind, n = "empirical (sampled partials)", n_states
    else:
        raise ValueError("law_boxes needs core or model")
    whole = Box(*(
        (min(getattr(b, name)[0] for b in per_speed.values()), max(getattr(b, name)[1] for b in per_speed.values()))
        for name in ("f_s", "f_dv", "f_v")
    ))  # fmt: skip
    return {"per_speed": per_speed, "whole": whole, "kind": kind, "n_states": n}
