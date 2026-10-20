"""Certificate of the two-timescale hybrid at frozen latents (docs/m11_contract.md, D129).

The certificate of ResidualIDMTT at a frozen latent z is that of the ResidualIDM with the core at T(z) and the
residual bounds of the model; at z = 0 it is the certificate of the E4 hybrid the model was built from. The budget of
a latent model is taken over the range its online latent can reach. The exact machinery (a_priori_exact, as
scripts/analysis/certificate_exact.py) gives the certificate at every z of the 21-value sweep from the stored inputs.
The scripts: certificate.json of a latent model with the per-z block, and the audit at a frozen latent."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cf_stability.models import IDM, ResidualIDM, ResidualIDMTT, save_model
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import idm_equilibrium_spacing
from cf_stability.stability.certificate import (
    LATENT_SWEEP,
    _a_priori,
    a_priori_exact,
    a_priori_exact_latent,
    at_latent,
    certificate_a_priori,
    certificate_at_equilibria,
    core_margin,
    empirical_margin,
    enforce_budget,
    has_latent,
    latent_core_params,
    max_residual_budget,
    reachable_latent,
)
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.utils import REPO_ROOT, read_json, write_json

IDM_P = {"v0": 33.0, "T": 1.5, "s0": 2.0, "a": 2.0, "b": 1.0}  # string stable on the whole speed grid
CONTEXT = ModelContext(center=(30.0, 0.0, 15.0), scale=(15.0, 1.5, 6.0), idm_params=IDM_P)


def hybrid(seed: int = 0, lipschitz: float = 0.01) -> ResidualIDM:
    torch.manual_seed(seed)
    model = ResidualIDM(CONTEXT, r_max=0.3, lipschitz=lipschitz)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.g.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=gen))
    return model.eval()


def tt_of(source: ResidualIDM, tau_z: float = 15.0) -> ResidualIDMTT:
    model = ResidualIDMTT.from_residual_idm(source, CONTEXT, tau_z=tau_z).eval()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in (*model.latent_rnn.parameters(), *model.latent_head.parameters()):
            p.add_(torch.randn(p.shape, generator=gen))
    return model


def test_sweep_values():
    assert len(LATENT_SWEEP) == 21 and LATENT_SWEEP[0] == -1.0 and LATENT_SWEEP[10] == 0.0 and LATENT_SWEEP[-1] == 1.0
    assert all(b - a == pytest.approx(0.1) for a, b in zip(LATENT_SWEEP, LATENT_SWEEP[1:]))
    assert LATENT_SWEEP[5] == -0.5 and LATENT_SWEEP[15] == 0.5


def test_frozen_zero_gives_the_certificate_of_the_e4_hybrid():
    source = hybrid()
    model = tt_of(source)
    assert has_latent(model) and not has_latent(source)
    with pytest.raises(ValueError, match="frozen latent"):
        certificate_a_priori(model)  # an online latent is not certified as it is
    with pytest.raises(ValueError, match="no latent"):
        certificate_a_priori(source, z=0.0)
    a, b = certificate_a_priori(source, keep_r_max=True), certificate_a_priori(model, keep_r_max=True, z=0.0)
    for key in ("guaranteed_margin", "s_low", "s_high", "s_worst", "max_budget", "bounds"):
        assert torch.allclose(a[key], b[key], rtol=1e-12, atol=1e-15, equal_nan=True), key
    assert torch.equal(a["holds"], b["holds"]) and a["n_holds"] == b["n_holds"] == len(V_GRID)
    p, q = certificate_at_equilibria(source), certificate_at_equilibria(model, z=0.0)
    for key in ("s", "guaranteed_margin", "f_s", "f_dv", "f_v", "max_budget"):
        assert torch.allclose(p[key], q[key], rtol=1e-12, atol=1e-12, equal_nan=True), key
    assert torch.equal(p["holds"], q["holds"])
    assert torch.allclose(empirical_margin(source), empirical_margin(model, z=0.0), rtol=1e-9, atol=1e-12)
    assert torch.allclose(core_margin(source), core_margin(model, z=0.0), rtol=0, atol=0, equal_nan=True)
    model.frozen_z = 0.0  # frozen_z is used when z is not given
    assert torch.equal(certificate_a_priori(model, keep_r_max=True)["holds"], a["holds"])


def test_certificate_at_a_frozen_latent_is_that_of_its_core():
    model = tt_of(hybrid(2))
    bounds = model.jacobian_bound().double()
    for z in (-1.0, -0.4, 0.3, 1.0):
        core = IDM(latent_core_params(IDM_P, z), dtype=torch.float64)
        ref = _a_priori(core, torch.tensor(V_GRID, dtype=torch.float64), torch.full((len(V_GRID),), 0.3,
                        dtype=torch.float64), bounds.expand(len(V_GRID), 3), 400)  # fmt: skip
        got = certificate_a_priori(model, z=z)
        assert torch.allclose(got["guaranteed_margin"], ref["guaranteed_margin"], rtol=1e-5, atol=1e-7, equal_nan=True)
        assert torch.equal(got["holds"], ref["holds"]) and torch.allclose(got["bounds"], bounds)
        assert at_latent(model, z).idm.params_dict()["T"] == pytest.approx(IDM_P["T"] * 1.5**z, rel=1e-6)


def test_exact_certificate_at_every_z_of_the_sweep():
    """a_priori_exact_latent from the stored inputs (core of z = 0, r_max, bounds) equals a_priori_exact of the frozen
    views, is the minimum of the scan up to rounding, and holds exactly where the budget of the model is admissible at
    that z: the machinery of certificate_exact.py at every z."""
    model = tt_of(hybrid(3, lipschitz=0.02))  # r_max * lipschitz = 0.006
    params = model.core_params(0.0)
    bounds = model.jacobian_bound().double()
    v = torch.tensor(V_GRID, dtype=torch.float64)
    exact = a_priori_exact_latent(params, model.r_max, bounds, LATENT_SWEEP, V_GRID)
    assert len(exact) == 21
    holds = []
    r_max = torch.full((len(V_GRID),), model.r_max, dtype=torch.float64)
    for z, ex in zip(LATENT_SWEEP, exact):
        view = model.at_z(z).double()
        ref = a_priori_exact(view.idm, v, view.r_max, bounds)
        assert torch.allclose(ex["guaranteed_margin"], ref["guaranteed_margin"], rtol=1e-6, atol=1e-9, equal_nan=True)
        scan = _a_priori(view.idm, v, r_max, bounds.expand(len(V_GRID), 3), 400)["guaranteed_margin"]  # the scan
        finite = torch.isfinite(scan)
        assert torch.equal(finite, torch.isfinite(ex["guaranteed_margin"]))
        assert (ex["guaranteed_margin"][finite] <= scan[finite] + 1e-7).all()  # exact <= scan
        holds.append(bool(ex["holds"].all()))
    assert any(holds) and not all(holds)  # this core: the budget shrinks with T, the sweep fails at large z
    # the exact certificate holds where the budget of the model is admissible at that z (every 4th z: the bisection of
    # the budget is the costly part); away from the edge, where scan and exact agree
    product = model.r_max * model.lipschitz
    for z, h in list(zip(LATENT_SWEEP, holds))[::4]:
        b = float(max_residual_budget(model, V_GRID, a_priori=True, keep_r_max=True, z=z).min())
        if abs(b / product - 1.0) > 0.02:
            assert h == (b >= product), z
    assert certificate_a_priori(model, z=LATENT_SWEEP[0])["holds"].all() == holds[0]  # the public entry point


def test_budget_of_a_latent_model_is_taken_over_its_reachable_range():
    source = hybrid(4)
    model = tt_of(source, tau_z=15.0)
    reach = model.latent_reach()
    grid = reachable_latent(model)
    assert len(grid) == 21 and grid[0] == pytest.approx(-reach) and grid[-1] == pytest.approx(reach) and 0.0 in grid
    assert reachable_latent(source) == (0.0,)
    per_z = [float(max_residual_budget(model, V_GRID, a_priori=True, keep_r_max=True, z=z).min()) for z in grid]
    out = enforce_budget(model, 0.9)
    assert out["admissible"] == pytest.approx(min(per_z), rel=1e-12)
    assert out["latent"]["admissible_per_z"] == pytest.approx(per_z, rel=1e-12) and out["latent"]["reach"] == reach
    assert model.lipschitz == pytest.approx(0.9 * min(per_z) / 0.3)
    for z in grid:  # the certificate holds at every frozen latent the online latent can reach
        assert certificate_a_priori(model, z=z)["holds"].all()
    # the budget of the E4 hybrid is the special case without latent (no latent key in its record)
    plain = enforce_budget(hybrid(4), 0.9)
    assert "latent" not in plain and plain["admissible"] == pytest.approx(per_z[10], rel=1e-9)
    # a latent that cannot move (tau_z huge) has the budget of the E4 hybrid
    slow = tt_of(hybrid(4), tau_z=1e9)
    assert enforce_budget(slow, 0.9)["admissible"] == pytest.approx(plain["admissible"], rel=1e-6)


def test_no_budget_raises_with_the_latent_range():
    weak = {"v0": 33.0, "T": 0.6, "s0": 2.0, "a": 1.0, "b": 2.0}  # not string stable at some speed of the grid
    model = ResidualIDMTT(ModelContext(idm_params=weak), r_max=0.3, tau_z=15.0)
    with pytest.raises(ValueError, match="frozen latent"):
        enforce_budget(model, 0.9)


# ------------------------------------------------------------------------------------------ scripts


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def run(script: str, tmp_path: Path, *overrides: str) -> list[str]:
    args = [sys.executable, str(REPO_ROOT / "scripts" / script), *overrides, f"hydra.run.dir={quoted(tmp_path / 'out')}"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=900)  # fmt: skip
    assert proc.returncode == 0 and "Traceback" not in proc.stderr, proc.stdout + proc.stderr
    return proc.stdout.strip().splitlines()


SCRIPT_SPEEDS = (8.0, 15.0, 25.0)  # the script tests run on a few speeds (the full grid: the tests above)
SPEEDS_OVERRIDE = "certificate.speeds=[8.0,15.0,25.0]"


def band() -> dict:
    v = torch.arange(10.0, 26.0, dtype=torch.float64)
    s_e = idm_equilibrium_spacing(v, IDM_P["v0"], IDM_P["T"], IDM_P["s0"])
    columns = {"v": v, "s_low": 0.8 * s_e, "s_median": s_e, "s_high": 1.2 * s_e}
    return {**{k: c.tolist() for k, c in columns.items()}, "n": [500] * len(v), "settings": {}}


@pytest.fixture(scope="module")
def latent_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("latent")
    run_dir = root / "runs" / "exp" / "synthetic" / "residual_idm_tt" / "driver_fold0_seed0"
    run_dir.mkdir(parents=True)
    model = tt_of(hybrid(5), tau_z=30.0)
    save_model(model, run_dir / "model.pt")
    context = {"band": band(), "box_low": [1.0, -3.0, 4.0], "box_high": [80.0, 3.0, 31.0]}
    write_json(run_dir / "metrics.json", {"context": context})
    return run_dir


def test_certificate_script_writes_the_latent_sweep(latent_run, tmp_path):
    lines = run("certificate.py", tmp_path, f"run={quoted(latent_run)}", "certificate.n_scan=200", SPEEDS_OVERRIDE)
    assert len(lines) == 1 and "latent sweep a priori" in lines[0] and "reachable |z| <= 0.095" in lines[0]
    text = (latent_run / "certificate.json").read_text(encoding="utf-8")
    cert = json.loads(text)
    assert "NaN" not in text and cert["applicable"] and cert["frozen_z"] == 0.0 and cert["model"] == "residual_idm_tt"
    latent = cert["latent"]
    assert latent["z"] == list(LATENT_SWEEP) and len(latent["per_z"]) == 21 and latent["tau_z"] == 30.0
    assert latent["reach"] == pytest.approx(1 - (1 - 0.1 / 30.0) ** 30)
    zero = latent["per_z"][10]
    assert zero["z"] == 0.0 and zero["a_priori"] == cert["a_priori"] and zero["per_speed"] == cert["per_speed"]
    assert cert["core"]["params"]["T"] == pytest.approx(IDM_P["T"], rel=1e-6)
    for entry in latent["per_z"]:
        assert entry["core"]["params"]["T"] == pytest.approx(IDM_P["T"] * 1.5 ** entry["z"], rel=1e-6)
        assert [r["v"] for r in entry["per_speed"]] == list(SCRIPT_SPEEDS)
    summary = latent["summary"]
    assert summary["n_z"] == 21 and summary["n_a_priori_holds"] == sum(e["a_priori"]["holds"] for e in latent["per_z"])
    assert len(latent["reachable"]["z"]) == 21 and latent["reachable"]["z"][-1] == pytest.approx(latent["reach"])
    # the stored inputs reproduce the scan of every z with the exact machinery (certificate_exact.py)
    bounds = torch.tensor(cert["residual"]["bounds"], dtype=torch.float64)
    exact = a_priori_exact_latent(cert["core"]["params"], cert["residual"]["r_max"], bounds, latent["z"],
                                  SCRIPT_SPEEDS)  # fmt: skip
    for entry, ex in zip(latent["per_z"], exact):
        scan = torch.tensor([math.nan if r["a_priori_margin"] is None else r["a_priori_margin"]
                             for r in entry["per_speed"]], dtype=torch.float64)  # fmt: skip
        ok = torch.isfinite(scan)
        assert (ex["guaranteed_margin"][ok] <= scan[ok] + 1e-7).all()
        assert (scan[ok] - ex["guaranteed_margin"][ok]).max() < 1e-3


def test_audit_script_at_a_frozen_latent(latent_run, tmp_path):
    lines = run("audit_stability.py", tmp_path, f"run={quoted(latent_run)}", "frozen_z=-0.5",
                "stability.device=cpu", "stability.speeds=[10.0,15.0,20.0]")  # fmt: skip
    assert len(lines) == 1 and "FAILED" not in lines[0]
    audit = read_json(latent_run / "stability_z-0.5.json")
    assert audit["frozen_z"] == -0.5 and audit["audit"]["window"] == 1 and audit["audit"]["analytic_gain"] == "discrete"
    assert not (latent_run / "stability.json").exists()
    # the equilibria of the frozen audit are those of the view at z = -0.5
    model = ResidualIDMTT.from_residual_idm(hybrid(5), CONTEXT, tau_z=30.0)
    s = [r["s"] for r in audit["audit"]["equilibria"]]
    assert all(x is not None for x in s)
    from cf_stability.stability.equilibrium import Band, find_equilibria

    view = model.at_z(-0.5).double()
    eq = find_equilibria(view, (10.0, 15.0, 20.0), band=Band.from_context({"band": band()}, view))
    assert torch.allclose(torch.tensor(s, dtype=torch.float64), eq.s, rtol=1e-6)
    other = latent_run.parent.parent / "residual_idm" / "driver_fold0_seed0"  # a model without latent fails
    other.mkdir(parents=True, exist_ok=True)
    save_model(hybrid(6), other / "model.pt")
    lines = run("audit_stability.py", tmp_path, f"run={quoted(other)}", "frozen_z=0", "stability.device=cpu",
                "stability.speeds=[10.0]")  # fmt: skip
    assert "FAILED" in lines[0] and "has no latent" in read_json(other / "stability_z0.json")["error"]


def _analysis(name: str):
    import importlib.util

    folder = REPO_ROOT / "scripts" / "analysis"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))  # certificate_tt imports certificate_exact from its own folder
    spec = importlib.util.spec_from_file_location(f"m11_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_certificate_tt_runs_the_exact_machinery_at_every_z(latent_run, tmp_path):
    """scripts/analysis/certificate_tt.py: at z = 0 the summary of certificate_exact.analyse on the stored payload; at
    every z the core of T(z) and the stored scan of that z (no stored core differs from T(z))."""
    if not (latent_run / "certificate.json").exists():
        run("certificate.py", tmp_path, f"run={quoted(latent_run)}", "certificate.n_scan=200", SPEEDS_OVERRIDE)
    tt, exact = _analysis("certificate_tt"), _analysis("certificate_exact")
    payload = read_json(latent_run / "certificate.json")
    sweep = exact.sweep_speeds(5.0, 30.0, 0.5)
    records = tt.analyse_run(payload, sweep)
    assert [r["z"] for r in records] == list(LATENT_SWEEP)
    plain, _ = exact.analyse(payload, sweep)
    zero = records[10]
    for key in ("exact_min", "scan_min", "sweep_min", "n_holds_exact", "sweep_holds_all", "limit_margin_min"):
        assert zero[key] == plain[key], key
    reach = payload["latent"]["reach"]
    for r in records:
        assert r["core_mismatch"] < 1e-6 and r["reachable"] == (abs(r["z"]) <= reach)
        assert r["T"] == pytest.approx(IDM_P["T"] * 1.5 ** r["z"], rel=1e-6)
        assert r["holds_exact_all"] == r["stored_a_priori_holds"] or r["n_holds_differ"] > 0
        # the stored scan of every z is reproduced from the stored inputs, up to the float32 rounding of T(z) in the
        # frozen view (the core of the model is float32; the analysis takes T(z) = T_c 1.5^z in float64)
        assert r["rescan_max_abs_diff"] < 1e-6


def test_m11_pairs_branch():
    pairs = _analysis("m11_pairs")
    import pandas as pd

    rows = [{"candidate": pairs.HET_FD, "reference": pairs.HET, "metric": "macro_error", "corridor": c,
             "difference": d, "difference_low": d - 0.01, "difference_high": d + 0.005} for c, d in (("I-80", -0.03),
                                                                                                   ("US-101", -0.02))]  # fmt: skip
    assert "confirmed branch" in pairs.h_n5_branch(pd.DataFrame(rows)) and "not the" not in pairs.h_n5_branch(
        pd.DataFrame(rows))
    rows[1]["difference_high"] = 0.001
    assert "not the confirmed branch" in pairs.h_n5_branch(pd.DataFrame(rows))
    assert "no rows" in pairs.h_n5_branch(pd.DataFrame(columns=["candidate", "reference", "metric"]))
