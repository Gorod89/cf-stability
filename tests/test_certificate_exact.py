"""Exact a priori certificate (M9 review): a_priori_exact against a brute force and against the scan of _a_priori, the
existence of the equilibria (feasible_ends; the reviewer's counterexample) and scripts/analysis/certificate_exact.py on
a synthetic tree of certificate.json files."""

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest
import torch

from cf_stability.models.idm import IDM
from cf_stability.stability.analytic import partials
from cf_stability.stability.certificate import (
    S_MAX,
    _a_priori,
    _feasible_interval,
    _root_real_parts,
    a_priori_exact,
    certify_core,
    feasible_ends,
    guaranteed_margin,
    to_json,
)
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.utils import REPO_ROOT, write_json

N_BRUTE = 200_000
SPEEDS = (5.0, 8.0, 12.5, 17.0, 21.5, 26.0, 29.0)
CORE_M02 = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}  # HighD fold 0 with margin 0.2 (test_certificate)
CORE_M05 = {"v0": 35.1, "T": 0.94, "s0": 4.71, "a": 2.74, "b": 0.5}  # ... with margin 0.5
CORE_FREE = {"v0": 29.57, "T": 0.676, "s0": 2.015, "a": 0.351, "b": 0.5}  # ... free
CORE_TWO = {"v0": 28.4, "T": 2.28, "s0": 0.6, "a": 0.91, "b": 0.58}  # two basins of the margin on I(v) = [s_low, 200] m
CASES = {  # core, r_max, bounds (B_s, B_dv, B_v)
    "certified": (CORE_M02, 0.3, (0.005, 0.05, 0.01)),  # residual of E4: the minimum at the upper end of I(v)
    "wide bounds": (CORE_M05, 1.0, (0.05, 0.6, 0.3)),  # interior minima
    "clamp regimes": (CORE_TWO, 1.48, (0.018, 0.7, 0.226)),  # p = F_v - B_v, -q and F_v + B_v all inside I(v)
}


def as_tensors(case: str) -> tuple[IDM, float, torch.Tensor]:
    core, r_max, bounds = CASES[case]
    return IDM(core, dtype=torch.float64), r_max, torch.tensor(bounds, dtype=torch.float64)


def gaps(idm: IDM, v: float, r_max: float, n: int) -> torch.Tensor:
    """``n`` spacings of I(v), uniform in ``x = 1/s`` (where the pieces are polynomials; 200 000 spacings uniform in
    ``s`` resolve an interior minimum on [12, 200] m only to about 1e-8 relative)."""
    lo, hi, feasible = _feasible_interval(idm, torch.tensor([v], dtype=torch.float64), r_max)
    assert feasible.item()
    return 1.0 / torch.linspace(1.0 / hi.item(), 1.0 / lo.item(), n, dtype=torch.float64)


def brute_force(idm: IDM, speeds, r_max: float, bounds: torch.Tensor) -> torch.Tensor:
    out = []
    for v in speeds:
        s = gaps(idm, float(v), r_max, N_BRUTE)
        out.append(guaranteed_margin(*partials(idm, s, torch.full_like(s, float(v))), bounds).min())
    return torch.stack(out)


def regimes(idm: IDM, v: float, r_max: float, bounds: torch.Tensor, edge: float) -> set[str]:
    """Regimes of the clamp ``p = clamp(-q, F_v - B_v, F_v + B_v)`` on the edge ``q = F_dv + edge B_dv`` over I(v)."""
    s = gaps(idm, v, r_max, 20_001)
    _, f_dv, f_v = partials(idm, s, torch.full_like(s, v))
    q = f_dv + edge * bounds[1]
    low, high = -q < f_v - bounds[2], -q > f_v + bounds[2]
    return {name for name, mask in (("lower", low), ("-q", ~low & ~high), ("upper", high)) if mask.any()}


@pytest.mark.parametrize("case", list(CASES))
def test_exact_minimum_equals_a_brute_force_minimum(case):
    idm, r_max, bounds = as_tensors(case)
    v = torch.tensor(SPEEDS, dtype=torch.float64)
    exact = a_priori_exact(idm, v, r_max, bounds)
    margin = exact["guaranteed_margin"]
    assert exact["feasible"].all()
    brute = brute_force(idm, SPEEDS, r_max, bounds)
    assert (margin <= brute + 1e-12).all()  # a value of the margin on I(v) ...
    assert ((brute - margin) <= 1e-9 * margin.abs()).all()  # ... that no spacing of the brute force undercuts
    s = exact["s_worst"]  # ... attained at s_worst
    assert ((exact["s_low"] <= s) & (s <= exact["s_high"])).all()
    torch.testing.assert_close(guaranteed_margin(*partials(idm, s, v), bounds), margin, rtol=1e-15, atol=1e-15)
    scan = _a_priori(idm, v, torch.full_like(v, r_max), bounds.expand(len(v), 3), 400)
    assert (scan["guaranteed_margin"] >= margin - 1e-12).all()  # the scan of the stored certificates is never lower
    for key in ("s_low", "s_high", "feasible", "fs_positive", "sum_negative"):
        assert torch.equal(exact[key], scan[key])
    if case == "certified":  # minimum at the upper end, a point of the scan too; the certificate holds
        assert torch.equal(s, exact["s_high"]) and exact["holds"].all() and torch.equal(exact["holds"], scan["holds"])
        torch.testing.assert_close(margin, scan["guaranteed_margin"], rtol=1e-14, atol=0)
    if case == "clamp regimes":  # every regime of the clamp occurs inside I(v) on one edge, the other edge clamped
        for speed in SPEEDS:
            assert regimes(idm, speed, r_max, bounds, 1.0) == {"lower", "-q", "upper"}
            assert regimes(idm, speed, r_max, bounds, -1.0) == {"upper"}
        assert (s > exact["s_low"]).any() and (s == S_MAX).any()  # interior minima and the far end


def test_infeasible_speeds_and_a_float32_core():
    idm = IDM(CORE_FREE, dtype=torch.float64)  # v0 29.57 m/s: no feasible spacing well above v0 with a small r_max
    v = torch.tensor([10.0, 45.0], dtype=torch.float64)
    bounds = torch.tensor([0.01, 0.1, 0.02], dtype=torch.float64)
    exact = a_priori_exact(idm, v, 0.05, bounds)
    scan = _a_priori(idm, v, torch.full_like(v, 0.05), bounds.expand(2, 3), 100)
    assert exact["feasible"].tolist() == [True, False] and not exact["holds"][1]
    for key in ("guaranteed_margin", "s_worst", "s_low", "s_high"):
        assert math.isnan(exact[key][1]) and math.isnan(scan[key][1])
    same = a_priori_exact(IDM(CORE_FREE, dtype=torch.float32), v, 0.05, bounds)  # used in float64, as converted
    float64_core = IDM(IDM(CORE_FREE, dtype=torch.float32).params_dict(), dtype=torch.float64)
    torch.testing.assert_close(same, a_priori_exact(float64_core, v, 0.05, bounds), rtol=0, atol=0, equal_nan=True)


def test_roots_are_those_of_numpy():
    gen = np.random.default_rng(0)
    coef = gen.normal(size=(40, 4))
    coef[::5, 3] = 0.0  # derivatives of the pieces: no constant term
    coef[1::7, 0] = 0.0  # a lower degree: numpy.roots row by row
    coef[2, :] = 0.0
    out = _root_real_parts(coef)
    for row, real in zip(coef, out):
        expected = np.sort(np.roots(row).real)
        assert np.allclose(np.sort(real[np.isfinite(real)]), expected, rtol=1e-10, atol=1e-12)
        assert np.isnan(real).sum() == 3 - len(expected)


def test_a_coarse_scan_misses_the_minimum_that_the_exact_method_finds():
    """Two basins of the guaranteed margin on I(8 m/s) = [11.6, 200] m: the 20-point scan with its zooms settles in the
    shallower one at 31.7 m (-0.314 s^-2); the minimum, -0.384 s^-2, lies at 13.9 m between its first two points."""
    idm, r_max, bounds = as_tensors("clamp regimes")
    v = torch.tensor([8.0], dtype=torch.float64)
    exact = a_priori_exact(idm, v, r_max, bounds)
    coarse = _a_priori(idm, v, torch.full_like(v, r_max), bounds.expand(1, 3), 20)
    assert exact["guaranteed_margin"].item() < coarse["guaranteed_margin"].item() - 0.05
    assert exact["s_worst"].item() == pytest.approx(13.92, abs=0.01)
    assert coarse["s_worst"].item() == pytest.approx(31.7, abs=0.1)
    brute = brute_force(idm, [8.0], r_max, bounds).item()
    assert abs(brute - exact["guaranteed_margin"].item()) <= 1e-9 * abs(brute)
    s = gaps(idm, 8.0, r_max, 20_001)
    g = guaranteed_margin(*partials(idm, s, torch.full_like(s, 8.0)), bounds)
    inner = (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
    assert sorted(round(x, 1) for x in s[1:-1][inner].tolist()) == [13.9, 31.7]  # two local minima


def test_an_unbounded_interval_guarantees_no_equilibrium():
    """The reviewer's counterexample: a = 1, 1 - (v/v0)^4 = 0.1 and s* = 12 m at v = 10 m/s give
    f_idm(s, 0, v) = 0.1 - 144 / s^2; the constant residual r = -0.2 (|r| <= r_max = 0.3, derivative bounds 0) leaves
    f_idm + r < 0 at every spacing. I(v) = [18.97 m, inf): the certificate on [18.97, 200] m holds without any
    equilibrium; feasible_ends reports the infinite upper end."""
    v = torch.tensor([10.0], dtype=torch.float64)
    idm = IDM({"v0": 10.0 / 0.9**0.25, "T": 1.0, "s0": 2.0, "a": 1.0, "b": 1.0}, dtype=torch.float64)
    s = torch.logspace(0, 6, 1001, dtype=torch.float64)
    f = idm.acc(s, torch.zeros_like(s), torch.full_like(s, 10.0))
    torch.testing.assert_close(f, 0.1 - 144.0 / s**2, rtol=0, atol=1e-12)
    assert (f - 0.2 < 0).all()
    ends = feasible_ends(idm, v, 0.3)
    assert ends["limit_margin"].item() == pytest.approx(-0.2)
    assert ends["s_low"].item() == pytest.approx(12.0 / math.sqrt(0.4))
    assert math.isinf(ends["s_high"].item()) and not ends["bounded"].item() and not ends["within"].item()
    prior = a_priori_exact(idm, v, 0.3, torch.zeros(3, dtype=torch.float64))
    assert prior["feasible"].item() and prior["s_high"].item() == S_MAX and prior["holds"].item()
    ends = feasible_ends(idm, v, 0.05)  # r_max below a (1 - (v/v0)^4): finite ends with f_idm = -+ r_max
    assert ends["bounded"].item() and ends["within"].item() and ends["limit_margin"].item() == pytest.approx(0.05)
    at_ends = idm.acc(torch.cat((ends["s_low"], ends["s_high"])), torch.zeros(2, dtype=torch.float64), v.expand(2))
    torch.testing.assert_close(at_ends, torch.tensor([-0.05, 0.05], dtype=torch.float64), rtol=0, atol=1e-12)
    clamped = _feasible_interval(idm, v, torch.tensor([0.05], dtype=torch.float64))
    assert clamped[0].item() == ends["s_low"].item() and clamped[1].item() == ends["s_high"].item()


# ------------------------------------------------------------------------------------- certificate_exact.py


def script():
    path = REPO_ROOT / "scripts" / "analysis" / "certificate_exact.py"
    spec = importlib.util.spec_from_file_location("certificate_exact", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def certificate(core: dict, r_max: float, bounds: tuple[float, float, float], n_scan: int = 50) -> dict:
    """The entries of certificate.json that the analysis reads, as scripts/certificate.py writes them."""
    prior = certify_core(core, r_max, torch.tensor(bounds, dtype=torch.float64), n_scan=n_scan)
    per_speed = [
        {"v": v, "a_priori_holds": prior["holds"][k], "a_priori_margin": prior["guaranteed_margin"][k],
         "a_priori_s_low": prior["s_low"][k], "a_priori_s_high": prior["s_high"][k]}
        for k, v in enumerate(V_GRID)
    ]  # fmt: skip
    return to_json({
        "model": "residual_idm", "applicable": True, "core": {"params": core},
        "residual": {"r_max": r_max, "lipschitz": 1.0, "product": r_max, "bounds": list(bounds)},
        "a_priori": {"holds": prior["holds"].all(), "n_hold": prior["holds"].sum(), "n_grid": len(V_GRID)},
        "per_speed": per_speed, "config": {"certificate": {"n_scan": n_scan, "speeds": None}},
    })  # fmt: skip


def test_the_script_writes_the_tables(tmp_path):
    ce = script()
    root = tmp_path / "runs"
    runs = {
        ("e4_free_r0.3", "follownet_highd", "driver_fold0_seed0"): certificate(CORE_FREE, 0.3, (0.01, 0.1, 0.02)),
        ("e4_stable", "follownet_highd", "driver_fold0_seed0"): certificate(CORE_M02, 0.3, (0.005, 0.05, 0.01)),
        ("e4_stable", "follownet_highd", "driver_fold1_seed2"): certificate(CORE_M02, 0.3, (0.0051, 0.051, 0.0102)),
    }
    for (experiment, data, run), payload in runs.items():
        write_json(root / experiment / data / "residual_idm" / run / "certificate.json", payload)
    write_json(root / "e4_free_ft" / "ngsim_i80" / "mlp" / "driver_fold0_seed0" / "certificate.json",
               {"model": "mlp", "applicable": False})  # fmt: skip
    out = tmp_path / "tables"
    assert ce.main(["--runs-root", str(root), "--out", str(out), "--sweep-step", "0.5"]) == 0
    table = pd.read_csv(out / "certificate_exact.csv").set_index("experiment")
    assert list(table.index) == ["e4_stable", "e4_free_r0.3"]  # the order of DESIGN
    stable, free = table.loc["e4_stable"], table.loc["e4_free_r0.3"]
    assert stable["runs"] == 2 and stable["speeds"] == len(V_GRID) and stable["cells"] == 2 * len(V_GRID)
    assert stable["runs_holding_exact"] == stable["runs_holding_scan"] == stable["runs_holding_sweep"] == 2
    assert stable["holds_differ"] == 0 and stable["max_abs_diff"] < 1e-12 and stable["rescan_max_abs_diff"] == 0.0
    assert stable["worst_at_upper_end"] == 1.0 and free["worst_inside"] > 0
    assert stable["exact_min"] > 0 and stable["sweep_min"] <= stable["exact_min"] + 1e-12
    assert stable["limit_margin_min"] > 0 and stable["within_share"] == 1.0 and stable["s_high_unclamped_max"] < 200
    assert pd.isna(stable["outside"])
    assert free["runs_holding_exact"] == 0 and free["limit_margin_min"] < 0 and free["within_share"] < 1.0
    assert math.isinf(free["s_high_unclamped_max"]) and free["outside"].startswith("upper end infinite at ")
    assert free["exact_min"] <= free["scan_min"] + 1e-12
    per_run = pd.read_csv(out / "per_run" / "certificate_exact_runs.csv")
    assert len(per_run) == 3 and per_run["sweep_speeds"].eq(51).all()
    assert per_run.set_index("run").loc["driver_fold1_seed2", ["fold", "seed"]].tolist() == [1, 2]
    assert (per_run["sweep_min"] <= per_run["exact_min"] + 1e-12).all()
    md = (out / "certificate_exact.md").read_text(encoding="utf-8")
    assert md.startswith("# Exact a priori certificate") and md.count("\n| e4_") == 2
    assert "1 x no a_priori entry (model mlp)" in md and "| e4_stable |" in md
    assert "| 2 | 26 |" in md  # runs and grid speeds of e4_stable; the holding counts are plain integers (of runs)
    with pytest.raises(ValueError, match="divide 1 m/s"):
        ce.sweep_speeds(5.0, 30.0, 0.3)
    assert ce.speed_ranges([5.0, 6.0, 7.0, 9.0, 12.5]) == "5-7, 9, 12.5"
