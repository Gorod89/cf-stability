"""Analyses of the windowed-gain penalty of M11 (D128) on synthetic inputs: the arm set m11 of
scripts/analysis/full_history_audit.py, the long-window check (scripts/analysis/long_window_check.py) and the tables
with the branch of the decision rule of H-N3 (scripts/analysis/windowed_gain.py)."""

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
from cf_stability.stability.analytic import history_jacobian, transfer_windowed
from cf_stability.stability.audit import AuditConfig, audit_model
from cf_stability.stability.frequency import FrequencyConfig
from cf_stability.utils import REPO_ROOT, read_json, write_json

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
    """``a = c_s (s - s0)`` of the state ``lag_s`` steps ago ``- c_s T v - c_dv dv`` of the state ``lag_v`` steps ago."""

    def __init__(self, c_s: float, T: float, c_dv: float, lag_s: int = 0, lag_v: int = 0, s0: float = 2.0) -> None:
        super().__init__()
        self.c_s, self.T, self.c_dv, self.s0, self.lag_s, self.lag_v = c_s, T, c_dv, s0, lag_s, lag_v
        self.window = max(lag_s, lag_v) + 1

    def forward(self, state_history: Tensor) -> Tensor:
        spacing, speed = state_history[..., -1 - self.lag_s, :], state_history[..., -1 - self.lag_v, :]
        return self.c_s * (spacing[..., 0] - self.s0) - self.c_s * self.T * speed[..., 2] - self.c_dv * speed[..., 1]


STABLE = (0.2, 2.0, 0.6, 5, 0)  # poles inside, M > 0
COUNTEREXAMPLE = (0.3, 1.5, 0.6, 20, 3)  # poles inside, M_w < 0: the low-frequency gain exceeds 1
AMPLIFIER = (0.1, 1.0, 0.05, 0, 0)  # memoryless, M = -0.18, M / f_s^2 = -18: |G| about 1.09 at 0.1 rad/s


def audit_of(params) -> dict:
    return audit_model(DelayedLaw(*params), CONTEXT, SMALL)


def fake_runs(root: Path, laws: dict[str, tuple], runs: dict[str, list[str]]) -> float:
    """Run directories ``root/<experiment>/follownet_highd/gru/<run>`` with an audit of the law of the experiment and
    a model.pt that names it (older than the audit)."""
    old = time.time() - 100.0
    for experiment, names in runs.items():
        audit = audit_of(laws[experiment])
        for name in names:
            run = root / experiment / "follownet_highd" / "gru" / name
            run.mkdir(parents=True)
            (run / "model.pt").write_bytes(experiment.encode())
            os.utime(run / "model.pt", (old, old))
            write_json(run / "stability.json", {"audit": audit, "model": "gru"})
    return old


# ------------------------------------------------------------------------- full_history_audit.py --arm-set m11


def test_the_m11_arm_set_selects_its_runs_and_leaves_the_files_of_m9(tmp_path, monkeypatch):
    fh = script("full_history_audit")
    root = tmp_path / "runs"
    laws = {"e1": STABLE, "e2_windowed_w1": COUNTEREXAMPLE, "e2_windowed_w10": STABLE}
    fake_runs(root, laws, {
        "e1": ["driver_fold0_seed0", "driver_fold1_seed0", "driver_fold0_seed1"],  # seed 1: not an M11 run
        "e2_windowed_w1": ["driver_fold0_seed0", "driver_fold1_seed0"],
        "e2_windowed_w10": ["driver_fold0_seed0", "driver_fold1_seed0"],  # pilot weight: fold 0 only
    })  # fmt: skip
    arms = {"E1 (seed 0)": {"gru": "e1"}, "E2 windowed (chosen weight)": {"gru": "e2_windowed_w1"},
            "E2 windowed w10 (pilot)": {"gru": "e2_windowed_w10"}}  # fmt: skip
    monkeypatch.setattr(fh, "M11_ARMS", arms)
    monkeypatch.setattr(fh, "M11_SEEDS", {arm: (0,) for arm in arms})
    monkeypatch.setattr(fh, "M11_FOLDS", {"E2 windowed w10 (pilot)": (0,)})
    monkeypatch.setattr(fh, "load_model", lambda path: DelayedLaw(*laws[Path(path).read_bytes().decode()]))
    out = tmp_path / "m11"
    assert fh.main(["--arm-set", "m11", "--runs-root", str(root), "--architectures", "gru", "--out", str(out)]) == 0
    table = pd.read_csv(out / "full_history_windowed.csv")
    assert list(dict.fromkeys(table["arm"])) == list(arms) and list(table["runs"]) == [2, 2, 2, 2, 1, 1]
    rows = table.set_index(["arm", "part"])
    assert rows.loc[("E2 windowed (chosen weight)", "all"), "lowfreq_sign_agreement_window"] == 1.0
    assert rows.loc[("E2 windowed (chosen weight)", "all"), "margin_sign_agreement"] == 0.0  # the memory of D66
    # full_history.json in the runs of the new experiments only; the E1 runs keep theirs (none here)
    group = root / "e2_windowed_w1" / "follownet_highd" / "gru"
    assert all((group / name / "full_history.json").is_file() for name in ("driver_fold0_seed0", "driver_fold1_seed0"))
    assert not list((root / "e1").rglob("full_history.json"))
    assert not (root / "e2_windowed_w10" / "follownet_highd" / "gru" / "driver_fold1_seed0" / "full_history.json").exists()
    md = (out / "full_history_windowed.md").read_text(encoding="utf-8")
    assert md.startswith("# Full-history linear analysis of the windowed-gain penalty (M11, D128)")
    assert "windowed_gain" in md and md.count("\n| gru |") == 12
    # the default arm set is that of M9 (its table and its arms unchanged)
    assert list(fh.ARMS)[:2] == ["E1", "E2"] and fh.ARMS["E2"]["gru"] == "e2_gain_w0.1"
    assert fh.run_directories(root, "e1", "follownet_highd", "gru") == sorted((root / "e1" / "follownet_highd" / "gru")
                                                                              .iterdir())  # fmt: skip


# --------------------------------------------------------------------------------------- long_window_check.py


def test_long_window_gain_of_a_linear_law_is_its_windowed_transfer_function():
    """A linear law with its poles inside: after 128 s the transient has decayed and the rollout gain of the check is
    the windowed gain at its three frequencies; the check holds when the gain stays below 1.02 inside the band."""
    lw = script("long_window_check")
    # the counterexample amplifies below 0.39 rad/s, at most 1.026 (at 0.28 rad/s): above 1 but below 1.02 here
    for params, threshold, holds in ((STABLE, 1.02, True), (COUNTEREXAMPLE, 1.02, True), (COUNTEREXAMPLE, 1.0, False)):
        law = DelayedLaw(*params).double()
        audit = audit_of(params)
        result = lw.check(law, audit, torch.device("cpu"), threshold=threshold)
        records = result["equilibria"]
        assert [r["v"] for r in records] == [8.0, 15.0, 22.0] and result["omega"] == [0.02, 0.05, 0.1]
        s = torch.tensor([r["s"] for r in records], dtype=torch.float64)
        v = torch.tensor([r["v"] for r in records], dtype=torch.float64)
        expected = transfer_windowed(history_jacobian(law, s, v), torch.tensor(lw.OMEGAS, dtype=torch.float64), 0.1).abs()
        np.testing.assert_allclose([r["gain"] for r in records], expected.numpy(), rtol=1e-6)
        inside = result["summary"]["inside"]
        assert inside["n"] == 2 and result["holds"] is holds and inside["holds"] is holds  # 22 m/s: not in support
        assert inside["max_gain"] == pytest.approx(expected[:2].max().item(), rel=1e-6)


def test_summary_of_the_long_window_check():
    lw = script("long_window_check")

    def rec(v, status, gains, support=True):
        largest = max(gains) if None not in gains else None
        return {"v": v, "status": status, "in_support": support, "gain": gains, "max_gain": largest,
                "above": None if largest is None else largest > 1.02, "audit_max_gain_low": 1.0}  # fmt: skip

    records = [rec(5.0, "ok", [1.0, 1.01, 0.9]), rec(6.0, "multiple", [1.0, 1.0, 1.0]),
               rec(7.0, "outside", [1.5, 1.1, 1.0]), rec(8.0, "ok", [3.0, 3.0, 3.0], support=False)]  # fmt: skip
    inside = lw.summarise(records, "inside")
    assert inside["n"] == 2 and inside["holds"] is True and inside["share_above"] == 0.0
    assert inside["max_gain_0.02"] == 1.0 and inside["max_gain_0.05"] == 1.01 and inside["max_gain"] == 1.01
    outside = lw.summarise(records, "outside")
    assert outside["n"] == 1 and outside["holds"] is False and outside["max_gain"] == 1.5
    assert lw.summarise(records + [rec(9.0, "ok", [1.0, 1.03, 1.0])], "inside")["holds"] is False
    assert lw.summarise(records + [rec(9.0, "ok", [1.0, None, 1.0])], "inside")["holds"] is False  # undefined gain
    assert lw.summarise([], "inside")["holds"] is None
    assert lw.settings().horizon_s == 380.0 and lw.settings().measure_s == 252.0 and lw.settings().omegas == lw.OMEGAS


def test_the_long_window_script_writes_runs_and_table(tmp_path, monkeypatch):
    lw = script("long_window_check")
    root = tmp_path / "runs"
    laws = {"e1": AMPLIFIER, "e2_windowed_w1": STABLE}
    fake_runs(root, laws, {"e1": ["driver_fold0_seed0"], "e2_windowed_w1": ["driver_fold0_seed0"]})
    monkeypatch.setattr(lw, "ARMS", {"E1": {"gru": "e1"}, "windowed w1": {"gru": "e2_windowed_w1"},
                                     "windowed w10": {"gru": "e2_windowed_w10"}})  # fmt: skip
    monkeypatch.setattr(lw, "load_model", lambda path: DelayedLaw(*laws[Path(path).read_bytes().decode()]))
    out = tmp_path / "m11"
    assert lw.main(["--runs-root", str(root), "--out", str(out), "--device", "cpu"]) == 0
    table = pd.read_csv(out / "long_window.csv")
    assert list(table["arm"]) == ["E1", "windowed w1"] and list(table["holds"]) == [False, True]
    assert (root / "e2_windowed_w1" / "follownet_highd" / "gru" / "driver_fold0_seed0" / "long_window.json").is_file()
    assert not (root / "e1" / "follownet_highd" / "gru" / "driver_fold0_seed0" / "long_window.json").exists()
    details = read_json(out / "long_window_runs.json")  # every run, the reference ones included
    assert sorted(details) == ["e1/follownet_highd/gru/driver_fold0_seed0", "e2_windowed_w1/follownet_highd/gru/driver_fold0_seed0"]
    assert details["e1/follownet_highd/gru/driver_fold0_seed0"]["holds"] is False
    md = (out / "long_window.md").read_text(encoding="utf-8")
    assert "e2_windowed_w10/follownet_highd/gru/driver_fold0_seed0: run missing" in md


# ------------------------------------------------------------------------------------------ windowed_gain.py


def test_branch_of_the_decision_rule_of_h_n3(tmp_path):
    wg = script("windowed_gain")
    maker = wg.WindowedTables(wg.tables_config(tmp_path, tmp_path))

    def cond(stable, change, holds, complete=True):
        return {"stable": stable, "rmse_change": change, "long_window": holds, "complete": complete}

    cases = [
        ({"gru": cond(0.95, 0.05, True), "lstm": cond(0.92, 0.10, True)}, "'confirmed'"),
        ({"gru": cond(0.95, 0.15, True), "lstm": cond(0.92, 0.05, False)}, "'partial'"),
        ({"gru": cond(0.95, 0.05, False), "lstm": cond(0.92, 0.05, True)}, "long-window check fails"),
        ({"gru": cond(0.85, 0.05, True), "lstm": cond(0.92, 0.05, True)}, "'refuted'"),
        ({"gru": cond(0.95, 0.05, True)}, "no branch"),
    ]
    for conditions, text in cases:
        lines = maker.branch({**conditions, "perl": cond(0.1, 0.5, False)})
        assert text in lines[-1], (conditions, lines[-1])
        assert any(line.startswith("- perl (not judged)") for line in lines)
    incomplete = maker.branch({"gru": cond(0.95, 0.05, True, False), "lstm": cond(0.92, 0.05, True)})
    assert "(incomplete design)" in incomplete[-1] and "'confirmed'" in incomplete[-1]


def test_run_helpers_of_the_windowed_tables(tmp_path):
    wg = script("windowed_gain")
    run = tmp_path / "run"
    run.mkdir()
    assert wg.long_window(run)["long_window_holds"] is None and wg.best_epoch_penalty(run) == {}
    write_json(run / "long_window.json", {"summary": {"inside": {"holds": False, "max_gain": 1.3, "share_above": 0.5}}})
    assert wg.long_window(run) == {"long_window_holds": False, "long_window_max_gain": 1.3, "long_window_share": 0.5}
    history = [{"val_penalty": 0.5, "val_penalty_margin_term": 0.5}, {"val_penalty": 0.004, "val_penalty_margin_term": 0.0,
               "val_penalty_max_gain": 1.1}]  # fmt: skip
    write_json(run / "metrics.json", {"best_epoch": 2, "training": {"history": history, "wall_time_s": 7200.0}})
    parts = wg.best_epoch_penalty(run)
    assert parts["val_penalty"] == 0.004 and parts["val_margin_term"] == 0.0 and parts["val_max_gain"] == 1.1
    assert parts["training_h"] == 2.0 and math.isnan(parts["val_pole_term"])
    # without any run the script writes empty tables and names what is missing
    out = tmp_path / "m11"
    assert wg.main(["tables", "--runs-root", str(tmp_path / "runs"), "--out", str(out)]) == 0
    assert (out / "e2_windowed_pilot.md").is_file() and (out / "e2_windowed.md").is_file()
    assert "no branch" in (out / "e2_windowed_branch.md").read_text(encoding="utf-8")
    assert read_json(out / "e2_windowed_chosen.json") == {}
