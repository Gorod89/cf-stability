"""Equilibria of a car-following law on the speed grid (docs/m3_contract.md, section 2), optionally anchored
to the spacing band of the training data (docs/m4_contract.md, 1.1-1.2)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from cf_stability.models.base import CFModel

V_GRID: tuple[float, ...] = tuple(float(v) for v in range(5, 31))
NOMINAL_S0, NOMINAL_T = 2.0, 1.5  # spacing used where every spacing is an equilibrium
INDIFFERENT_TOL = 1e-9  # m/s^2
BAND_KEYS = ("v", "s_low", "s_median", "s_high")  # columns of ``context["band"]`` used here


def model_like(model: CFModel) -> tuple[torch.device, torch.dtype]:
    """Device and floating dtype of the model (CPU, float64 for a model without tensors)."""
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float64


def constant_history(model: CFModel, s: Tensor, v: Tensor, dv: Tensor | None = None) -> Tensor:
    """History ``[n, window, 3]`` holding the state ``(s, dv, v)`` at every step (``dv = 0`` by default)."""
    dv = torch.zeros_like(s) if dv is None else dv
    state = torch.stack(torch.broadcast_tensors(s, dv, v), dim=-1)
    return state.reshape(-1, 1, 3).expand(-1, model.window, -1)


def steady_acc(model: CFModel, s: Tensor, v: Tensor, dv: Tensor | None = None) -> Tensor:
    """``f(s, dv, v)`` of a constant history, flattened to ``[n]``."""
    return model(constant_history(model, s, v, dv))


@dataclass(frozen=True, eq=False)
class Band:
    """Spacing band of the near-steady training samples per speed (``context["band"]``, D78).

    Tensors ``[k]``, ``k >= 2``, on one device and dtype; ``v`` strictly increasing. Between the
    listed speeds the band is interpolated linearly in the speed; it is defined on
    ``[v[0], v[-1]]`` only.
    """

    v: Tensor
    s_low: Tensor
    s_median: Tensor
    s_high: Tensor

    @classmethod
    def from_mapping(
        cls, band: Mapping[str, Sequence[float]], device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "Band":  # fmt: skip
        """Band of the lists ``v, s_low, s_median, s_high`` of ``band`` (other keys are ignored).

        Checked on the host, before any tensor exists: no synchronisation with the device.
        """
        columns = [[float(x) for x in band[key]] for key in BAND_KEYS]
        v, s_low, _, s_high = columns
        if len(v) < 2 or any(len(column) != len(v) for column in columns):
            raise ValueError("a band needs at least two speeds and one value of every column per speed")
        if any(b <= a for a, b in zip(v, v[1:])) or not all(lo <= hi for lo, hi in zip(s_low, s_high)):
            raise ValueError("band speeds must increase strictly and s_low <= s_high")
        return cls(*(torch.tensor(column, dtype=dtype, device=device) for column in columns))

    @classmethod
    def from_context(cls, context: Mapping[str, Any] | None, model: CFModel) -> "Band | None":
        """``context["band"]`` on the device and dtype of ``model``; None when the context has no band."""
        band = None if context is None else context.get("band")
        return None if band is None else cls.from_mapping(band, *model_like(model))

    def at(self, speeds: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Lower edge, upper edge and ``has_band`` at ``speeds [n]``, in the dtype and on the device of ``speeds``.

        Outside ``[v[0], v[-1]]`` ``has_band`` is False and the edges are those of the nearest listed
        speed: finite values for masked arithmetic. Fixed shapes and no synchronisation, so that
        the call can be captured in a CUDA graph.
        """
        v, low, high = (x.to(speeds) for x in (self.v, self.s_low, self.s_high))
        inside = torch.minimum(torch.maximum(speeds, v[0]), v[-1])
        k = torch.searchsorted(v, inside, right=True).clamp(1, len(v) - 1)  # interval v[k - 1] .. v[k]
        w = (inside - v[k - 1]) / (v[k] - v[k - 1])
        has_band = (speeds >= v[0]) & (speeds <= v[-1])
        # lerp is exact at both ends: a listed speed gets exactly its listed edges
        return torch.lerp(low[k - 1], low[k], w), torch.lerp(high[k - 1], high[k], w), has_band


@dataclass
class Equilibria:
    """Equilibrium spacing per speed. ``s`` is NaN where ``status`` is ``"none"``."""

    v: Tensor  # [n]
    s: Tensor  # [n]
    found: Tensor  # [n] bool: an upward zero crossing exists (inside or outside the band)
    indifferent: Tensor  # [n] bool: zero acceleration at every spacing, nominal spacing in ``s``
    n_crossings: Tensor  # [n] upward zero crossings on the scan of the interval that gave the equilibrium
    has_band: Tensor | None = None  # [n] bool: the speed has a band (None: no band anywhere)
    in_band: Tensor | None = None  # [n] bool: the equilibrium lies inside the band (indifferent: every spacing does)

    def __post_init__(self) -> None:
        if self.has_band is None:
            self.has_band = torch.zeros_like(self.found)
        if self.in_band is None:
            self.in_band = torch.zeros_like(self.found)

    @property
    def usable(self) -> Tensor:
        """Speeds at which the law can be linearised and excited (``found`` or ``indifferent``)."""
        return self.found | self.indifferent

    @property
    def anchored(self) -> Tensor:
        """Usable equilibria inside the band, or at a speed without band."""
        return self.usable & (self.in_band | ~self.has_band)

    @property
    def status(self) -> list[str]:
        """``indifferent``, ``none``, ``outside`` (equilibrium outside the band of a speed that has one),
        ``ok`` or ``multiple`` (several upward crossings in the interval that gave the equilibrium)."""
        out = []
        rows = zip(*(x.tolist() for x in (self.found, self.indifferent, self.n_crossings, self.has_band, self.in_band)))
        for found, indifferent, n, has_band, in_band in rows:
            if indifferent:
                out.append("indifferent")
            elif not found:
                out.append("none")
            elif has_band and not in_band:
                out.append("outside")
            else:
                out.append("ok" if n == 1 else "multiple")
        return out


def _first_upward(f: Tensor) -> tuple[Tensor, Tensor]:
    """Number of upward zero crossings (``f < 0`` then ``f >= 0``) per row of ``f [n, m]`` and the
    index of the first one (0 if none)."""
    up = (f[:, :-1] < 0) & (f[:, 1:] >= 0)
    return up.sum(dim=1), torch.argmax(up.to(torch.int8), dim=1)


@torch.no_grad()
def find_equilibria(
    model: CFModel,
    v: Sequence[float] | Tensor = V_GRID,
    *,
    s_min: float = 1.0,
    s_max: float = 200.0,
    n_scan: int = 400,
    n_bisect: int = 50,
    band: Band | None = None,
) -> Equilibria:
    """First upward zero crossing of ``f(s, 0, v)`` for every speed.

    ``f`` is scanned on ``n_scan`` spacings of ``[s_min, s_max]`` and the crossing is refined by
    bisection. With ``band``, a speed that has one is scanned on ``n_scan`` spacings of
    ``[s_low(v), s_high(v)]`` first; without an upward crossing there, the first one of
    ``[s_min, s_max]`` is an equilibrium outside the band. Speeds without band get the scan of
    ``[s_min, s_max]`` alone, with the result of a call without band. Both scans run at every
    speed and are selected with masks: no shape depends on the values (CUDA graph capture). The
    model is used as it is (mode, device, dtype); gradients are not tracked.
    """
    device, dtype = model_like(model)
    v = torch.as_tensor(v, dtype=dtype, device=device).reshape(-1)
    n = v.shape[0]
    v_scan = v.repeat_interleave(n_scan)
    grid = torch.linspace(s_min, s_max, n_scan, dtype=dtype, device=device)
    f = steady_acc(model, grid.repeat(n), v_scan).reshape(n, n_scan)
    n_crossings, first = _first_upward(f)
    found = n_crossings > 0
    lo, hi = grid[first], grid[first + 1]
    has_band = in_band = torch.zeros_like(found)
    if band is not None:
        s_low, s_high, has_band = band.at(v)
        frac = torch.linspace(0.0, 1.0, n_scan, dtype=dtype, device=device)
        band_grid = torch.lerp(s_low[:, None], s_high[:, None], frac)  # [n, n_scan], the edges exactly
        n_band, first_band = _first_upward(steady_acc(model, band_grid.reshape(-1), v_scan).reshape(n, n_scan))
        in_band = has_band & (n_band > 0)
        lo = torch.where(in_band, band_grid.gather(1, first_band[:, None]).squeeze(1), lo)
        hi = torch.where(in_band, band_grid.gather(1, first_band[:, None] + 1).squeeze(1), hi)
        n_crossings = torch.where(in_band, n_band, n_crossings)
        found = found | in_band
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        below = steady_acc(model, mid, v) < 0
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    indifferent = (f.abs() < INDIFFERENT_TOL).all(dim=1)
    s = torch.where(found, 0.5 * (lo + hi), torch.full_like(v, float("nan")))
    s = torch.where(indifferent, NOMINAL_S0 + NOMINAL_T * v, s)
    return Equilibria(
        v=v, s=s, found=found & ~indifferent, indifferent=indifferent, n_crossings=n_crossings,
        has_band=has_band, in_band=torch.where(indifferent, has_band, in_band),
    )  # fmt: skip
