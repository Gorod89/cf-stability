"""Equilibria on the speed grid (cf_stability/stability/equilibrium.py), with and without the spacing band."""

from typing import Callable, Sequence

import pytest
import torch
from torch import Tensor

from cf_stability.models import IDM, OVM
from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import idm_acc, idm_equilibrium_spacing
from cf_stability.stability.equilibrium import NOMINAL_S0, NOMINAL_T, V_GRID, Band, Equilibria, find_equilibria

IDM_PARAMS = {"v0": 25.0, "T": 1.2, "s0": 2.0, "a": 1.0, "b": 1.5}
GRID = torch.tensor(V_GRID, dtype=torch.float64)


class Law(CFModel):
    """``fn(s, dv, v)`` of the state ``lag`` steps ago (window ``lag + 1``)."""

    def __init__(self, fn: Callable[[Tensor, Tensor, Tensor], Tensor], lag: int = 0) -> None:
        super().__init__()
        self.fn, self.window = fn, lag + 1

    def forward(self, state_history: Tensor) -> Tensor:
        state = state_history[..., -self.window, :]
        return self.fn(state[..., 0], state[..., 1], state[..., 2])


def idm_law(s: Tensor, dv: Tensor, v: Tensor) -> Tensor:
    return idm_acc(s, dv, v, **IDM_PARAMS)


def idm_closed_form(v: Tensor) -> Tensor:
    return idm_equilibrium_spacing(v, IDM_PARAMS["v0"], IDM_PARAMS["T"], IDM_PARAMS["s0"])


@pytest.mark.parametrize("dtype, rtol", [(torch.float64, 0.0), (torch.float32, 1e-5)])
def test_idm_matches_closed_form(dtype, rtol):
    eq = find_equilibria(IDM(IDM_PARAMS, dtype=dtype))
    below = GRID < IDM_PARAMS["v0"]
    assert eq.s.dtype == dtype
    torch.testing.assert_close(eq.s[below].double(), idm_closed_form(GRID[below]), rtol=rtol, atol=1e-9)
    assert [st for st, b in zip(eq.status, below) if b] == ["ok"] * int(below.sum())
    # at and above v0 the IDM brakes at every spacing
    assert [st for st, b in zip(eq.status, below) if not b] == ["none"] * int((~below).sum())
    assert eq.s[~below].isnan().all() and not eq.usable[~below].any()


def test_ovm_matches_closed_form():
    params = {"v0": 25.5, "tau": 1.2, "s0": 3.0}
    eq = find_equilibria(OVM(ModelContext(ovm_params=params)))
    below = GRID < params["v0"]
    torch.testing.assert_close(eq.s[below], params["s0"] + params["tau"] * GRID[below], rtol=0.0, atol=1e-9)
    assert set(st for st, b in zip(eq.status, below) if b) == {"ok"}
    assert set(st for st, b in zip(eq.status, below) if not b) == {"none"}


def test_zero_law_is_indifferent_with_nominal_spacing():
    eq = find_equilibria(Law(lambda s, dv, v: torch.zeros_like(s)))
    assert eq.status == ["indifferent"] * len(V_GRID)
    assert eq.usable.all() and not eq.found.any()
    torch.testing.assert_close(eq.s, NOMINAL_S0 + NOMINAL_T * GRID)


def test_two_upward_crossings_give_the_first():
    # upward crossings at 20 m and 100 m, downward at 60 m
    eq = find_equilibria(Law(lambda s, dv, v: 1e-5 * (s - 20.0) * (s - 60.0) * (s - 100.0)))
    assert eq.status == ["multiple"] * len(V_GRID)
    assert eq.n_crossings.tolist() == [2] * len(V_GRID)
    torch.testing.assert_close(eq.s, torch.full_like(eq.s, 20.0), rtol=0.0, atol=1e-9)


def test_windowed_model_uses_a_constant_history():
    law = Law(idm_law, lag=4)
    assert law.window == 5
    eq = find_equilibria(law, [8.0, 16.0, 24.0])
    torch.testing.assert_close(eq.s, idm_closed_form(eq.v), rtol=0.0, atol=1e-9)
    assert eq.status == ["ok"] * 3


def band_mapping(v: Sequence[float], s_low: Sequence[float], s_high: Sequence[float]) -> dict:
    """A band as ``training_context`` lists it (the median and the counts are not used here)."""
    median = [0.5 * (lo + hi) for lo, hi in zip(s_low, s_high)]
    return {"v": list(v), "s_low": list(s_low), "s_median": median, "s_high": list(s_high), "n": [200] * len(v)}


def idm_band(speeds: Sequence[float], below: float, above: float, dtype: torch.dtype = torch.float64) -> Band:
    """Band from ``s_e(v) - below`` to ``s_e(v) + above`` of the closed-form IDM equilibrium at the listed speeds."""
    s_e = idm_closed_form(torch.tensor(speeds, dtype=torch.float64)).tolist()
    return Band.from_mapping(band_mapping(speeds, [s - below for s in s_e], [s + above for s in s_e]), dtype=dtype)


def assert_same(a: Equilibria, b: Equilibria, rows: Tensor) -> None:
    """The equilibria of ``a`` and ``b`` agree bit by bit at ``rows``."""
    torch.testing.assert_close(a.s[rows], b.s[rows], rtol=0.0, atol=0.0, equal_nan=True)
    for name in ("found", "indifferent", "n_crossings"):
        assert torch.equal(getattr(a, name)[rows], getattr(b, name)[rows]), name
    index = rows.nonzero().squeeze(-1).tolist()
    assert [a.status[i] for i in index] == [b.status[i] for i in index]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_band_interpolates_linearly_and_ends_at_the_listed_speeds(dtype):
    band = Band.from_mapping(band_mapping([10.0, 20.0, 30.0], [10.0, 30.0, 35.0], [20.0, 50.0, 60.0]), dtype=dtype)
    speeds = torch.tensor([5.0, 9.99, 10.0, 12.5, 20.0, 25.0, 30.0, 30.5], dtype=dtype)
    low, high, has_band = band.at(speeds)
    assert low.dtype == high.dtype == dtype and has_band.dtype == torch.bool
    assert has_band.tolist() == [False, False, True, True, True, True, True, False]
    # listed speeds exactly, linear in between; outside the band the nearest end (finite, masked by has_band)
    assert low.tolist() == [10.0, 10.0, 10.0, 15.0, 30.0, 32.5, 35.0, 35.0]
    assert high.tolist() == [20.0, 20.0, 20.0, 27.5, 50.0, 55.0, 60.0, 60.0]
    assert band.s_median.tolist() == [15.0, 40.0, 47.5]


def test_band_from_context_and_malformed_bands():
    model = IDM(IDM_PARAMS, dtype=torch.float32)
    mapping = band_mapping([10.0, 20.0], [15.0, 25.0], [20.0, 30.0])
    assert Band.from_context(None, model) is None and Band.from_context({"box_low": [0, 0, 0]}, model) is None
    assert Band.from_context({"band": None}, model) is None
    band = Band.from_context({"band": mapping}, model)
    assert band.v.dtype == torch.float32 and band.v.device == next(model.parameters()).device
    assert band.s_low.tolist() == [15.0, 25.0] and band.s_high.tolist() == [20.0, 30.0]
    for bad in (
        band_mapping([10.0], [15.0], [20.0]),  # one speed
        band_mapping([20.0, 10.0], [15.0, 25.0], [20.0, 30.0]),  # decreasing speeds
        band_mapping([10.0, 20.0], [15.0, 35.0], [20.0, 30.0]),  # s_low > s_high
    ):
        with pytest.raises(ValueError):
            Band.from_mapping(bad)


@pytest.mark.parametrize("dtype, rtol", [(torch.float64, 0.0), (torch.float32, 1e-5)])
def test_idm_with_band_inside_outside_and_none(dtype, rtol):
    # listed at every speed: s_e - 2 m .. s_e + 2 m up to 14 m/s (inside), s_e + 5 m .. s_e + 15 m
    # from 15 m/s (outside); from v0 = 25 m/s on the IDM has no equilibrium at all
    inside, above = [float(v) for v in range(5, 15)], [float(v) for v in range(15, 25)]
    s_e = idm_closed_form(torch.tensor(inside + above, dtype=torch.float64))
    low = s_e + torch.tensor([-2.0] * 10 + [5.0] * 10, dtype=torch.float64)
    high = low + torch.tensor([4.0] * 10 + [10.0] * 10, dtype=torch.float64)
    speeds = inside + above + [25.0, 26.0, 30.0]
    band = Band.from_mapping(band_mapping(speeds, low.tolist() + [50.0] * 3, high.tolist() + [60.0] * 3), dtype=dtype)
    eq = find_equilibria(IDM(IDM_PARAMS, dtype=dtype), speeds, band=band)
    assert eq.s.dtype == dtype and eq.has_band.all()
    assert eq.status == ["ok"] * 10 + ["outside"] * 10 + ["none"] * 3
    assert eq.in_band.tolist() == [True] * 10 + [False] * 13
    assert eq.usable.tolist() == [True] * 20 + [False] * 3
    assert eq.anchored.tolist() == [True] * 10 + [False] * 13
    # inside the band and, outside it, the equilibrium of the full scan
    torch.testing.assert_close(eq.s[:20].double(), s_e, rtol=rtol, atol=1e-9)
    assert eq.s[20:].isnan().all() and eq.n_crossings.tolist() == [1] * 20 + [0] * 3
    assert ((eq.s[:10] >= band.s_low[:10]) & (eq.s[:10] <= band.s_high[:10])).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_speeds_without_band_keep_the_equilibria_of_the_call_without_band(dtype):
    model = IDM(IDM_PARAMS, dtype=dtype)
    band = idm_band([10.0, 14.0, 20.0], below=1.0, above=1.0, dtype=dtype)  # defined on [10, 20] m/s only
    plain, anchored = find_equilibria(model), find_equilibria(model, band=band)
    grid = torch.tensor(V_GRID, dtype=dtype)
    outside = (grid < 10.0) | (grid > 20.0)
    assert anchored.has_band.tolist() == (~outside).tolist() and not plain.has_band.any() and not plain.in_band.any()
    assert_same(plain, anchored, outside)
    assert torch.equal(anchored.anchored[outside], plain.usable[outside])
    assert torch.equal(plain.anchored, plain.usable)  # without band every usable equilibrium is anchored


def test_band_is_interpolated_between_the_listed_speeds():
    """Between 10 and 14 m/s the band is interpolated linearly: a narrow band then lies above the
    convex equilibrium curve of the IDM, while one listed at every speed holds it."""
    speeds = [11.0, 12.0, 13.0]
    coarse_band = idm_band([10.0, 14.0], below=0.01, above=0.01)
    low, high, has_band = coarse_band.at(torch.tensor(speeds, dtype=torch.float64))
    assert has_band.all() and (idm_closed_form(torch.tensor(speeds, dtype=torch.float64)) < low).all()
    torch.testing.assert_close(high - low, torch.full_like(low, 0.02), rtol=0.0, atol=1e-12)
    coarse = find_equilibria(IDM(IDM_PARAMS), speeds, band=coarse_band)
    assert coarse.status == ["outside"] * 3 and not coarse.anchored.any()
    fine = find_equilibria(IDM(IDM_PARAMS), speeds, band=idm_band([10.0, 11.0, 12.0, 13.0, 14.0], 0.01, 0.01))
    assert fine.status == ["ok"] * 3 and fine.anchored.all()
    torch.testing.assert_close(fine.s, coarse.s, rtol=1e-12, atol=0.0)  # the same crossing, found in or out of the band


def test_band_selects_among_several_crossings():
    # upward crossings at 20 m and 100 m, downward at 60 m, at every speed
    law = Law(lambda s, dv, v: 1e-5 * (s - 20.0) * (s - 60.0) * (s - 100.0))
    speeds = [10.0, 20.0]

    def with_band(low: float, high: float) -> Equilibria:
        return find_equilibria(law, speeds, band=Band.from_mapping(band_mapping(speeds, [low] * 2, [high] * 2)))

    both = with_band(10.0, 110.0)  # several crossings inside: the first one, "multiple"
    assert both.status == ["multiple"] * 2 and both.n_crossings.tolist() == [2, 2] and both.anchored.all()
    torch.testing.assert_close(both.s, torch.full_like(both.s, 20.0), rtol=0.0, atol=1e-9)
    upper = with_band(50.0, 110.0)  # the crossing inside the band, although the full scan finds 20 m first
    assert upper.status == ["ok"] * 2 and upper.n_crossings.tolist() == [1, 1] and upper.in_band.all()
    torch.testing.assert_close(upper.s, torch.full_like(upper.s, 100.0), rtol=0.0, atol=1e-9)
    between = with_band(30.0, 50.0)  # no crossing inside: the first one of the full scan, outside the band
    assert between.status == ["outside"] * 2 and between.n_crossings.tolist() == [2, 2]
    assert between.usable.all() and not between.anchored.any()
    torch.testing.assert_close(between.s, torch.full_like(between.s, 20.0), rtol=0.0, atol=1e-9)


def test_indifferent_law_with_band_is_anchored_at_the_nominal_spacing():
    band = Band.from_mapping(band_mapping([10.0, 20.0], [5.0, 10.0], [8.0, 12.0]))
    eq = find_equilibria(Law(lambda s, dv, v: torch.zeros_like(s)), [5.0, 15.0], band=band)
    assert eq.status == ["indifferent"] * 2 and eq.has_band.tolist() == [False, True]
    assert eq.in_band.tolist() == [False, True] and eq.anchored.all()  # every spacing, the band too, is an equilibrium
    torch.testing.assert_close(eq.s, NOMINAL_S0 + NOMINAL_T * eq.v)


def test_equilibria_without_band_fields():
    """Equilibria built without the band fields (as before M4) have no band: anchored = usable."""
    eq = Equilibria(
        v=torch.tensor([10.0, 20.0]), s=torch.tensor([15.0, float("nan")]), found=torch.tensor([True, False]),
        indifferent=torch.tensor([False, False]), n_crossings=torch.tensor([1, 0]),
    )  # fmt: skip
    assert eq.status == ["ok", "none"] and eq.anchored.tolist() == [True, False] and not eq.has_band.any()
