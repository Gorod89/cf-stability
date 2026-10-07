"""Analyses of M9 on synthetic inputs: the full-history linearisation (cf_stability/stability/analytic.py:
windowed_state_space, closed_loop_poles, windowed_margin; scripts/analysis/full_history_audit.py), the threshold
sensitivity of the audit (scripts/analysis/threshold_sensitivity.py) and the zero crossings of f(s, 0, v)
(scripts/analysis/equilibrium_roots.py)."""

import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.stability.analytic import (
    closed_loop_poles,
    criterion,
    history_jacobian,
    partials,
    transfer_discrete,
    transfer_windowed,
    windowed_margin,
    windowed_state_space,
)
from cf_stability.stability.audit import AuditConfig, audit_model, summarise_audit
from cf_stability.stability.equilibrium import find_equilibria
from cf_stability.stability.frequency import FrequencyConfig
from cf_stability.train.closed_loop import rollout_model
from cf_stability.utils import REPO_ROOT, read_json, write_json

DT = 0.1
OMEGA = torch.logspace(-2, math.log10(2.0), 25, dtype=torch.float64)
SMALL = AuditConfig(
    device="cpu", speeds=(8.0, 15.0, 22.0),
    frequency=FrequencyConfig(n_omega=4, omega_min=0.1, omega_max=1.0, min_discard_s=20.0, min_measure_s=20.0),
)  # fmt: skip
CONTEXT = {"box_low": [1.0, -3.0, 6.0], "box_high": [100.0, 3.0, 18.0]}  # speeds 8 and 15 in support, 22 not


def script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class DelayedLaw(CFModel):
    """``a = c_s (s - s0)`` of the state ``lag_s`` steps ago ``- c_s T v - c_dv dv`` of the state ``lag_v`` steps ago:
    a linear law with memory whose spacing and speed are read with different lags."""

    def __init__(self, c_s: float, T: float, c_dv: float, lag_s: int = 0, lag_v: int = 0, s0: float = 2.0) -> None:
        super().__init__()
        self.c_s, self.T, self.c_dv, self.s0, self.lag_s, self.lag_v = c_s, T, c_dv, s0, lag_s, lag_v
        self.window = max(lag_s, lag_v) + 1

    def forward(self, state_history: Tensor) -> Tensor:
        spacing, speed = state_history[..., -1 - self.lag_s, :], state_history[..., -1 - self.lag_v, :]
        return self.c_s * (spacing[..., 0] - self.s0) - self.c_s * self.T * speed[..., 2] - self.c_dv * speed[..., 1]


STABLE = (0.2, 2.0, 0.6, 5, 0)  # poles inside, M > 0
COUNTEREXAMPLE = (0.3, 1.5, 0.6, 20, 3)  # poles inside, M > 0 but M_w < 0: the low-frequency gain exceeds 1
LOCALLY_UNSTABLE = (0.6, 1.2, 0.8, 15, 15)  # a common lag of 1.5 s: poles outside the unit circle


def random_jacobian(n: int, window: int, seed: int = 0, scale: float = 0.05) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    jac = scale * torch.randn(n, window, 3, generator=generator, dtype=torch.float64)
    jac[:, -1] += torch.tensor([0.05, -0.3, -0.2], dtype=torch.float64)  # f_s > 0, f_dv + f_v < 0 at the latest state
    return jac


# --------------------------------------------------------------------------------- analytic.py (state space)


@pytest.mark.parametrize("window", [1, 3, 30])
def test_state_space_gives_the_windowed_transfer_function(window):
    jac = random_jacobian(4, window)
    A, B, C = windowed_state_space(jac, DT)
    size = 2 * window + 1
    assert A.shape == (4, size, size) and B.shape == (4, size) and C.shape == (size,)
    z = torch.exp(1j * OMEGA * DT)
    eye = torch.eye(A.shape[1], dtype=torch.complex128)
    G = torch.stack([
        torch.stack([zk * (C.to(eye) @ torch.linalg.solve(zk * eye - A[i].to(eye), B[i].to(eye))) for zk in z])
        for i in range(len(jac))
    ])  # fmt: skip
    torch.testing.assert_close(G, transfer_windowed(jac, OMEGA, DT), rtol=1e-10, atol=1e-12)
    # the rows of the leader speeds depend on the leader speeds only (a shift register fed by the input): A is block
    # upper triangular, its eigenvalues are those of the block of (e, u), the poles, and W zeros
    assert (A[:, window + 1 :, : window + 1] == 0).all()
    shift = A[0, window + 1 :, window + 1 :]
    assert torch.equal(torch.linalg.matrix_power(shift, window), torch.zeros_like(shift))  # nilpotent
    assert closed_loop_poles(jac, DT).shape == (4, window + 1)


@pytest.mark.parametrize("window", [1, 4, 30])
def test_poles_are_the_roots_of_the_denominator(window):
    jac = random_jacobian(3, window, seed=1)
    poles = closed_loop_poles(jac, DT)
    for i in range(len(jac)):
        # z^(W-1) ((z - 1)^2 + dt^2 F_s(z) z - dt (F_dv(z) + F_v(z)) (z - 1)), F_x(z) = sum_j J_x[W-1-j] z^-j
        coefficients = np.zeros(window + 2)  # of z^d at index d
        coefficients[[window + 1, window, window - 1]] += (1.0, -2.0, 1.0)
        j_s, j_dv, j_v = (jac[i, :, k].numpy() for k in range(3))
        for k in range(window):
            coefficients[k + 1] += DT**2 * j_s[k] - DT * (j_dv[k] + j_v[k])
            coefficients[k] += DT * (j_dv[k] + j_v[k])
        roots = np.roots(coefficients[::-1])
        found = poles[i].numpy()
        assert max(min(abs(r - p) for p in found) for r in roots) < 1e-9
    if window == 1:  # the quadratic of the memoryless discrete loop
        f_s, f_dv, f_v = jac[:, 0].unbind(-1)
        q = f_dv + f_v
        b, c = -(2.0 + DT * q - DT**2 * f_s), 1.0 + DT * q
        root = torch.sqrt((b**2 - 4.0 * c).to(torch.complex128))
        expected = torch.stack(((-b + root) / 2.0, (-b - root) / 2.0), dim=-1)
        torch.testing.assert_close(poles.abs().sort(dim=1).values, expected.abs().sort(dim=1).values)


@pytest.mark.parametrize("params, settles", [(STABLE, True), (COUNTEREXAMPLE, True), (LOCALLY_UNSTABLE, False)])
def test_poles_decide_whether_a_rollout_settles(params, settles):
    """The rollout of the project (rollout_model, semi-implicit Euler, the window of states) after a gap offset of
    1 m: it settles when every pole lies inside the unit circle and grows otherwise."""
    law = DelayedLaw(*params).double()
    eq = find_equilibria(law, [10.0, 20.0])
    radius = closed_loop_poles(history_jacobian(law, eq.s, eq.v), law.dt).abs().amax(1)
    assert bool((radius < 1.0).all()) == settles
    n = 3000
    v_lead = eq.v[:, None].expand(-1, n).clone()
    x_lead = eq.s[:, None] + 1.0 + law.dt * (torch.cumsum(v_lead, -1) - v_lead[:, :1])
    s_hist = (eq.s + 1.0)[:, None].expand(-1, law.window).clone()
    res = rollout_model(law, x_lead, v_lead, s_hist, eq.v[:, None].expand(-1, law.window).clone(), dt=law.dt)
    late = (res.s[:, -300:] - eq.s[:, None]).abs().amax(1)
    assert bool((late < 1e-3).all()) if settles else bool((late > 1.0).all())


def test_windowed_margin_is_the_low_frequency_coefficient():
    jac = random_jacobian(6, 30, seed=2, scale=0.002)
    assert (jac.sum(1)[:, 0] > 0.02).all()  # f_s well away from 0: the expansion holds at 1e-4 rad/s
    f_s = jac.sum(1)[:, 0]
    omega = torch.tensor([1e-4], dtype=torch.float64)
    coefficient = (1.0 - transfer_windowed(jac, omega, DT).abs()[:, 0] ** 2) * f_s**2 / omega**2
    torch.testing.assert_close(windowed_margin(jac, DT), coefficient, rtol=2e-3, atol=1e-6)
    # memoryless: M - dt f_s f_v, the Euler step; a lag common to all inputs changes nothing
    f = jac.sum(1)
    margin = criterion(*f.unbind(-1))["margin"]
    for lag in (0, 7):
        delayed = torch.zeros(6, 10, 3, dtype=torch.float64)
        delayed[:, -1 - lag] = f
        torch.testing.assert_close(windowed_margin(delayed, DT), margin - DT * f[:, 0] * f[:, 2])
    memoryless = transfer_discrete(*f.unbind(-1), omega, DT).abs()[:, 0]
    torch.testing.assert_close((1.0 - memoryless**2) * f_s**2 / omega**2, margin - DT * f[:, 0] * f[:, 2], rtol=2e-3,
                               atol=1e-6)  # fmt: skip


def test_the_memoryless_margin_misses_a_lag_of_the_spacing():
    """The counterexample of the review: M > 0 (string stable by the memoryless view) while the gain of the law with its
    history exceeds 1 at low frequency, as M_w < 0 says and as the rollout of the audit measures."""
    law = DelayedLaw(*COUNTEREXAMPLE).double()
    eq = find_equilibria(law, [12.0])
    jac = history_jacobian(law, eq.s, eq.v)
    margin = criterion(*partials(law, eq.s, eq.v))["margin"]
    assert margin.item() > 0.1 and windowed_margin(jac, law.dt).item() < -0.2
    omega = torch.tensor([0.05, 0.1], dtype=torch.float64)
    windowed = transfer_windowed(jac, omega, law.dt).abs()[0]
    memoryless = transfer_discrete(*partials(law, eq.s, eq.v), omega, law.dt).abs()[0]
    assert (windowed > 1.0).all() and windowed[1] > 1.01 and (memoryless < 1.0).all()


# ------------------------------------------------------------------------------- full_history_audit.py


def audit_of(params, context=CONTEXT) -> dict:
    return audit_model(DelayedLaw(*params), context, SMALL)


def test_analyse_matches_the_stored_audit_of_a_law_with_memory():
    fh = script("full_history_audit")
    audit = audit_of(COUNTEREXAMPLE)
    law = DelayedLaw(*COUNTEREXAMPLE).double()
    payload = fh.analyse(law, audit)
    records = payload["equilibria"]
    assert [r["v"] for r in records] == [8.0, 15.0, 22.0] and payload["threshold"] == 1.02
    assert payload["omega"] == audit["omega"] and payload["lowest_omega"] == pytest.approx(0.1)
    for r, stored in zip(records, audit["equilibria"]):
        assert r["poles_stable"] and r["spectral_radius"] < 1.0 and len(r["poles"]) == law.window + 1
        assert r["max_gain_windowed"] == pytest.approx(stored["analytic_max_gain"]["windowed"], rel=1e-10)
        # a linear law with its poles inside: the rollout of the audit is the windowed response
        np.testing.assert_allclose(stored["gain"], r["gain_windowed"], atol=2e-3)
        assert r["margin"] > 0.1 and r["margin_window"] < -0.2 and r["lowfreq_windowed_above_one"]
        assert r["windowed_unstable"] and r["numerical_unstable"] and np.array(r["jacobian"]).shape == (law.window, 3)
    every, support = payload["summary"]["all"], payload["summary"]["support"]
    assert every["n_equilibria"] == 3 and support["n_equilibria"] == 2
    assert every["poles_stable"] == 1.0 and every["gain_agreement"] == 1.0 and every["windowed_unstable"] == 1.0
    assert every["lowfreq_sign_agreement"] == 0.0 and every["lowfreq_sign_agreement_window"] == 1.0
    assert every["margin_sign_agreement"] == 0.0 and every["limit_check_window"] == 1.0
    assert payload["stored_max_abs_diff"] < 1e-9


def test_the_full_history_script_writes_runs_table_and_figure(tmp_path, monkeypatch):
    fh = script("full_history_audit")
    root = tmp_path / "runs"
    laws = {"e1": STABLE, "e2_gain_w0.1": LOCALLY_UNSTABLE, "e2_combined_j0.1": COUNTEREXAMPLE}
    old = time.time() - 100.0
    for experiment, params in laws.items():
        audit = audit_of(params)
        for k in range(2):
            run = root / experiment / "follownet_highd" / "gru" / f"driver_fold{k}_seed0"
            run.mkdir(parents=True)
            (run / "model.pt").write_bytes(experiment.encode())
            os.utime(run / "model.pt", (old, old))
            write_json(run / "stability.json", {"audit": audit, "model": "gru"})
    stale = root / "e1" / "follownet_highd" / "gru" / "driver_fold2_seed0"  # its model is newer than its audit
    stale.mkdir()
    write_json(stale / "stability.json", {"audit": audit_of(STABLE), "model": "gru"})
    os.utime(stale / "stability.json", (old, old))
    (stale / "model.pt").write_bytes(b"e1")
    monkeypatch.setattr(fh, "load_model", lambda path: DelayedLaw(*laws[Path(path).read_bytes().decode()]))
    out, figures = tmp_path / "tables", tmp_path / "figures"
    args = ["--runs-root", str(root), "--architectures", "gru", "--out", str(out), "--figure-dir", str(figures)]
    assert fh.main(args) == 0
    payload = read_json(root / "e2_gain_w0.1" / "follownet_highd" / "gru" / "driver_fold1_seed0" / "full_history.json")
    assert payload["summary"]["all"]["poles_stable"] == 0.0 and not (stale / "full_history.json").exists()
    table = pd.read_csv(out / "full_history.csv")
    assert list(table["arm"]) == ["E1", "E1", "E2", "E2", "E2 combined", "E2 combined"]
    assert list(table["part"]) == ["all", "support"] * 3 and set(table["runs"]) == {2}
    assert set(table["runs_expected"]) == {2, 3}  # the stale run is counted but not analysed
    rows = table.set_index(["arm", "part"])
    assert rows.loc[("E1", "all"), "poles_stable"] == 1.0 and rows.loc[("E2", "all"), "poles_stable"] == 0.0
    assert rows.loc[("E2 combined", "all"), "lowfreq_sign_agreement"] == 0.0
    assert rows.loc[("E2 combined", "all"), "lowfreq_sign_agreement_window"] == 1.0
    assert rows.loc[("E1", "support"), "speeds"] == 4 and rows.loc[("E1", "all"), "speeds"] == 6
    md = (out / "full_history.md").read_text(encoding="utf-8")
    assert md.startswith("# Full-history linear analysis") and "older than model.pt" in md
    assert md.count("\n| gru |") == 12  # two tables of six rows
    for suffix in ("png", "pdf", "txt"):
        assert (figures / f"full_history_gain.{suffix}").is_file()
    assert "lowest audit frequency" in (figures / "full_history_gain.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------- threshold_sensitivity.py


def record(v: float, status: str, max_gain: float | None, in_support: bool = True, in_band: bool | None = True) -> dict:
    """An audit record with the fields summarise_audit reads; the numerical flag at 1.02."""
    usable = status != "none"
    return {
        "v": v, "s": 20.0 if usable else None, "status": status, "n_crossings": int(usable), "in_support": in_support,
        "in_band": in_band, "band_low": 10.0, "band_high": 30.0, "f_s": 0.1, "f_dv": -0.5, "f_v": -0.2,
        "margin": 0.1 if usable else None, "band_upper": 0.0, "local_stable": True if usable else None,
        "rational": True if usable else None, "string_stable": True if usable else None, "gain": None, "phase": None,
        "residual": None, "clipped": [False] if usable else None, "stopped": [False] if usable else None,
        "collided": [False] if usable else None, "max_gain": max_gain, "omega_at_max": 0.1 if max_gain else None,
        "unstable": None if max_gain is None else max_gain > 1.02, "analytic_max_gain": None,
        "analytic_unstable": None, "marginal": False if usable else None,
    }  # fmt: skip


def threshold_audit(shift: float = 0.0) -> dict:
    records = [
        record(5.0, "ok", 1.00), record(6.0, "ok", 1.005), record(7.0, "ok", 1.01), record(8.0, "ok", 1.015),
        record(9.0, "outside", 1.03, in_band=False), record(10.0, "ok", 1.06 + shift), record(11.0, "ok", None),
        record(12.0, "none", None, in_band=False), record(13.0, "ok", 1.02), record(14.0, "ok", 1.5, in_support=False),
    ]  # fmt: skip
    audit = {"support_v": [5.0, 13.0], "equilibria": records, "config": {"frequency": {"threshold": 1.02}}}
    audit["summary"] = summarise_audit(audit)
    return audit


def test_reclassification_at_the_thresholds_is_strict():
    ts = script("threshold_sensitivity")
    audit = threshold_audit()
    expected = {1.00: 6 / 7, 1.01: 4 / 7, 1.02: 2 / 7, 1.05: 1 / 7}  # 7 equilibria in support with a defined flag
    for threshold, share in expected.items():
        values = ts.statistics(ts.reclassified(audit, threshold)["summary"])
        assert values["unstable_eq"] == pytest.approx(share)
    reference = ts.statistics(ts.reclassified(audit, 1.02)["summary"])
    assert reference == ts.statistics(audit["summary"])  # the stored summary
    assert reference["band_unstable"] == pytest.approx(1 / 9) and reference["band_not_stable"] == pytest.approx(4 / 9)
    assert reference["band_outside"] == reference["band_none"] == reference["band_undefined"] == pytest.approx(1 / 9)
    at_one = ts.statistics(ts.reclassified(audit, 1.00)["summary"])
    assert at_one["band_unstable"] == pytest.approx(5 / 9) and at_one["band_stable"] == pytest.approx(1 / 9)
    assert ts.count_flagged(audit, 1.01) == 4  # the speed out of support is not counted
    assert audit["equilibria"][6]["unstable"] is None  # no largest gain: no flag at any threshold
    assert ts.reclassified(audit, 1.0)["equilibria"][6]["unstable"] is None


def test_the_threshold_script_writes_the_table(tmp_path):
    ts = script("threshold_sensitivity")
    root, old = tmp_path / "runs", time.time() - 100.0
    for k, shift in enumerate((0.0, -0.02, -0.04)):  # the speed at 1.06 drops to 1.04 and 1.02 in two runs
        run = root / "e1" / "follownet_highd" / "mlp" / f"driver_fold{k}_seed0"
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"fake")
        os.utime(run / "model.pt", (old, old))
        write_json(run / "stability.json", {"audit": threshold_audit(shift), "model": "mlp"})
    runs, problems, expected = ts.collect(root, "e1", "follownet_highd", ["mlp"])
    assert len(runs) == 3 * 4 and not problems and expected == {"mlp": 3}
    assert runs["reproduction_error"].dropna().tolist() == [0.0, 0.0, 0.0]
    table = ts.summarise(runs, ["mlp"], expected).set_index("threshold")
    assert list(table.index) == [1.00, 1.01, 1.02, 1.05] and (table["runs"] == 3).all()
    assert table.loc[1.05, "unstable_eq"] == pytest.approx(1 / 21)  # only the first run keeps a gain above 1.05
    assert table.loc[1.02, "unstable_eq"] == pytest.approx((2 + 2 + 1) / 21)
    assert table.loc[1.05, "unstable_eq_change"] == pytest.approx(table.loc[1.05, "unstable_eq"] - (5 / 21))
    assert table.loc[1.02, "flagged"] == 5 and table.loc[1.00, "h1_1_share_rule"] == "met"
    md = ts.markdown(table.reset_index(), runs, problems, "e1", "follownet_highd")
    assert "exactly at the threshold counts as not unstable" in md and md.count("\n| mlp |") == 4
    assert "3 of 3 runs" in md
    out = tmp_path / "tables"
    assert ts.main(["--runs-root", str(root), "--architectures", "mlp", "--out", str(out)]) == 0
    assert len(pd.read_csv(out / "threshold_sensitivity.csv")) == 4 and (out / "threshold_sensitivity.md").is_file()


# ---------------------------------------------------------------------------------- equilibrium_roots.py


class RootsLaw(CFModel):
    """``f(s, 0, v)`` with known roots per speed (window 3, the last state): no root at 5 m/s, one at 20 m (10 m/s),
    roots 5 (down), 30 (up), 100 m (down) at 20 m/s, roots 10 (up), 40 (down), 150 m (up) at 30 m/s."""

    window = 3

    def forward(self, state_history: Tensor) -> Tensor:
        s, v = state_history[..., -1, 0], state_history[..., -1, 2]
        one = (s - 20.0) / 10.0
        three_down = -(s - 5.0) * (s - 30.0) * (s - 100.0) / 1e4
        three_up = (s - 10.0) * (s - 40.0) * (s - 150.0) / 1e4
        return torch.where(v < 7.5, -torch.ones_like(s), torch.where(v < 15.0, one, torch.where(v < 25.0, three_down,
                                                                                                      three_up)))


def roots_audit() -> dict:
    eq = find_equilibria(RootsLaw(), [5.0, 10.0, 20.0, 30.0])
    status = ["none", "ok", "ok", "multiple"]
    assert eq.found.tolist() == [False, True, True, True] and eq.n_crossings.tolist() == [0, 1, 1, 2]
    s = [None if math.isnan(x) else x for x in eq.s.tolist()]
    return {"equilibria": [{"v": v, "status": st, "s": x, "in_support": v < 25.0}
                           for v, st, x in zip(eq.v.tolist(), status, s)]}  # fmt: skip


def test_crossings_and_uniqueness_of_the_audit_equilibrium():
    er = script("equilibrium_roots")
    audit = roots_audit()
    assert [r["s"] for r in audit["equilibria"]][1:] == pytest.approx([20.0, 30.0, 10.0])
    records = er.speed_records(RootsLaw(), audit)
    assert [r["n_up"] for r in records] == [0, 1, 1, 2] and [r["n_down"] for r in records] == [0, 0, 2, 1]
    assert [r["unique"] for r in records] == [None, True, True, False]
    assert [r["down_below"] for r in records] == [False, False, True, False]
    assert [r["down_above"] for r in records] == [False, False, True, True]
    assert all(r["bracketed"] for r in records[1:])
    shares = er.shares(records, "all")
    assert shares["up_0"] == shares["up_2plus"] == 0.25 and shares["up_1"] == 0.5 and shares["any_down"] == 0.5
    assert shares["unique"] == pytest.approx(2 / 3) and shares["not_bracketed"] == 0.0
    assert shares["audit_none_scan_up"] == 0.0 and shares["n_audit_equilibria"] == 3
    support = er.shares(records, "support")  # 5, 10 and 20 m/s
    assert support["n_speeds"] == 3 and support["unique"] == 1.0 and support["up_2plus"] == 0.0
    gaps = er.gap_grid()
    assert len(gaps) == 400 and gaps[0] == pytest.approx(1.0) and gaps[-1] == pytest.approx(200.0)
    assert np.allclose(gaps[1:] / gaps[:-1], gaps[1] / gaps[0])  # log-spaced


def test_the_roots_script_writes_the_table(tmp_path, monkeypatch):
    er = script("equilibrium_roots")
    root, old = tmp_path / "runs", time.time() - 100.0
    for k in range(2):
        run = root / "e1" / "follownet_highd" / "lstm" / f"driver_fold{k}_seed0"
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"fake")
        os.utime(run / "model.pt", (old, old))
        write_json(run / "stability.json", {"audit": roots_audit(), "model": "lstm"})
    monkeypatch.setattr(er, "load_model", lambda path: RootsLaw())
    out = tmp_path / "tables"
    assert er.main(["--runs-root", str(root), "--architectures", "lstm", "--out", str(out)]) == 0
    table = pd.read_csv(out / "equilibrium_roots.csv").set_index("part")
    assert list(table.index) == ["all", "support"] and (table["runs"] == 2).all()
    assert table.loc["all", "unique"] == pytest.approx(2 / 3) and table.loc["all", "up_2plus"] == 0.25
    assert table.loc["all", "unique_low"] == pytest.approx(2 / 3)  # identical runs: a degenerate interval
    md = (out / "equilibrium_roots.md").read_text(encoding="utf-8")
    assert md.startswith("# Multiple equilibria") and md.count("\n| lstm |") == 2
