"""Tables of the M4 report (cf_stability/eval/tables.py, scripts/make_tables.py) on hand-made run trees.

The trees follow a reduced design (2 folds x 2 seeds; MLP, GRU, LSTM and the IDM) with values that
are constant over the runs where the intervals should collapse, so that means, intervals and
verdicts can be computed by hand. The recurrent runs carry a full_history.json whose speeds give
the shares of their audit (M9: the shares with the poles of the full-history loop).
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cf_stability.eval.tables import TablesConfig, choose_weight, full_history_shares, make_tables, verdict_overall
from cf_stability.utils import REPO_ROOT, write_json

DATA, FT = "follownet_highd", "ngsim_i80"
FOLDS, SEEDS = (0, 1), (0, 1)
BASE = {"d0": 2.0, "d1": 4.0, "d2": 2.0, "d3": 4.0, "n0": 3.0, "n1": 5.0, "n2": 3.0, "n3": 5.0}
DRIVERS = {DATA: {0: ("d0", "d1"), 1: ("d2", "d3")}, FT: {0: ("n0", "n1"), 1: ("n2", "n3")}}
E1 = {  # model: (RMSE factor, stable, unstable, outside, unstable among the equilibria found)
    "mlp": (0.8, 0.3, 0.6, 0.1, 0.8), "gru": (0.9, 0.1, 0.8, 0.1, 0.2), "lstm": (1.1, 0.1, 0.9, 0.0, 0.9),
    "idm": (1.0, 0.8, 0.2, 0.0, 0.2), "residual_idm": (1.0, 0.5, 0.5, 0.0, 0.5),
}  # fmt: skip  # the GRU: not stable inside the band at 90 % of the speeds, yet few unstable equilibria
SWEEP = {  # (architecture, weight): (stable, validation RMSE, feasible best epoch)
    ("mlp", 0.1): (0.95, 2.5, True), ("mlp", 1.0): (0.97, 2.6, True),
    ("gru", 0.1): (0.5, 2.0, False), ("gru", 1.0): (0.8, 2.1, True),  # no weight reaches 0.9
    ("lstm", 0.1): (0.92, 2.9, None), ("lstm", 1.0): (0.95, 2.8, True),
}  # fmt: skip
CHOSEN = {"mlp": ("jacobian", 0.1, 1.05, 0.3), "gru": ("gain", 1.0, 1.3, 0.2), "lstm": ("gain", 1.0, 1.15, 0.4)}
MAX_GAIN = {(0, 0): 1.1, (0, 1): 1.2, (1, 0): 1.3, (1, 1): 1.4}
LOWFREQ = {  # D110, (architecture, Jacobian weight): (stable, validation RMSE, RMSE factor against E1)
    ("gru", 0.1): (0.95, 2.0, 1.05), ("gru", 1.0): (0.97, 2.2, 1.02),  # both stable: the smaller validation RMSE
    ("lstm", 0.1): (0.5, 2.9, 1.2), ("lstm", 1.0): (0.6, 3.0, 1.3),  # none stable: the largest share
}  # fmt: skip
LOWFREQ_CHOSEN = {"gru": 0.1, "lstm": 1.0}
RECURRENT = ("gru", "lstm", "perl")  # runs with a full_history.json (M9)
N_BAND = 100  # grid speeds in support with a band of every synthetic audit
# M9, (experiment, data, model) -> numerically stable equilibria whose full-history loop has a pole outside the unit
# circle: (inside the band, outside it, at speeds without band); every other recurrent run has none
POLES = {
    ("e1", DATA, "gru"): (5, 0, 140),  # unstable among equilibria 0.2 -> 0.5625: H1.1 would hold on it
    ("e2_gain_w1", DATA, "lstm"): (30, 0, 0),  # not stable 0.05 -> 0.35: H1.2 would be refuted on it
    ("e2_gain_long_w0.1", DATA, "gru"): (7, 2, 0),  # not stable 0.23 -> 0.30, unstable among equilibria 0.05 -> 0.14
    ("e5", "openacc_acc", "gru"): (10, 0, 0),  # both 0.5 -> 0.6
}  # fmt: skip


def shares(stable: float, unstable: float, outside: float = 0.0) -> dict:
    none = round(1.0 - stable - unstable - outside, 12)
    return {"stable": stable, "unstable": unstable, "outside": outside, "none": none, "indifferent": 0.0,
            "undefined": 0.0}  # fmt: skip


def full_history(stable: float, unstable: float, outside: float, unstable_eq: float, poles=(0, 0, 0)) -> dict:
    """full_history.json of a recurrent run whose audit has the band shares ``stable``, ``unstable`` and ``outside`` of
    N_BAND speeds and the share ``unstable_eq`` among its equilibria in support: the equilibria inside the band, those
    outside it (as many numerically unstable as ``unstable_eq`` needs) and, when ``unstable_eq`` is below the band share
    unstable, numerically stable equilibria at speeds in support without band; one speed out of support. ``poles``: of
    the numerically stable equilibria inside the band, outside it and without band, how many have a pole outside the
    unit circle; every other numerically unstable one has one as well, which changes no share."""
    n_stable, n_unstable, n_outside = (round(N_BAND * x) for x in (stable, unstable, outside))
    found = n_stable + n_unstable + n_outside
    if unstable_eq * found >= n_unstable - 1e-9:
        extra, outside_unstable = 0, round(unstable_eq * found) - n_unstable
    else:
        extra, outside_unstable = round(n_unstable / unstable_eq) - found, 0
    assert 0 <= outside_unstable <= n_outside and poles[1] <= n_outside - outside_unstable
    assert poles[0] <= n_stable and poles[2] <= extra

    def speed(in_band, status, unstable_flag, pole_inside, in_support=True):
        return {"v": 5.0, "s": 20.0, "status": status, "in_support": in_support, "in_band": in_band,
                "numerical_unstable": unstable_flag, "poles_stable": pole_inside}  # fmt: skip

    records = [speed(True, ("ok", "multiple")[k % 2], False, k >= poles[0]) for k in range(n_stable)]
    records += [speed(True, "ok", True, k % 2 == 1) for k in range(n_unstable)]
    records += [speed(False, "outside", True, True) for _ in range(outside_unstable)]
    records += [speed(False, "outside", False, k >= poles[1]) for k in range(n_outside - outside_unstable)]
    records += [speed(None, "ok", False, k >= poles[2]) for k in range(extra)]
    records.append(speed(True, "ok", False, False, in_support=False))  # out of support: no share sees it
    return {"model": "gru", "window": 30, "equilibria": records}


def write_run(
    root: Path, experiment: str, data: str, model: str, fold: int, seed: int, *, factor: float = 1.0,
    stable: float = 0.5, unstable: float = 0.5, outside: float = 0.0, unstable_eq: float | None = None,
    val: float = 2.0, feasible=None,
    growth: float = 0.4, acc: float = 0.4, human: float = 0.4, hysteresis: float = 100.0, collided=(),
    transfer: dict | None = None, certificate: dict | None = None,
) -> Path:  # fmt: skip
    run = root / experiment / data / model / f"driver_fold{fold}_seed{seed}"
    run.mkdir(parents=True)
    write_json(run / "metrics.json", {
        "config_hash": f"{experiment}-{model}-{fold}-{seed}", "val": {"rmse_s_mean": val},
        "test": {"rmse_s_mean": val + 0.1}, "training": {"epochs": 3, "best_feasible": feasible},
        "train_config": {"penalty": {"kind": "none", "weight": 1.0}},
    })  # fmt: skip
    unstable_eq = unstable if unstable_eq is None else unstable_eq
    write_json(run / "stability.json", {"audit": {"summary": {
        "band_numerical": {"support": shares(stable, unstable, outside)}, "band_sign": {"support": shares(0.5, 0.5)},
        "share_unstable_numerical": {"all": 0.0, "support": unstable_eq}, "n_band": {"all": N_BAND, "support": N_BAND},
        "share_unstable_sign": {"all": 0.0, "support": 0.7}, "max_gain": MAX_GAIN[fold, seed],
    }}})  # fmt: skip
    if model in RECURRENT:
        poles = POLES.get((experiment, data, model), (0, 0, 0))
        write_json(run / "full_history.json", full_history(stable, unstable, outside, unstable_eq, poles))
    write_json(run / "platoon.json", {"summary": {
        "growth_error_mean": growth, "growth_error_acc": acc, "growth_error_human": human,
        "hysteresis_area_pulse": hysteresis,
    }})  # fmt: skip
    drivers = DRIVERS.get(data, {0: ("o0", "o1"), 1: ("o2", "o3")})[fold]
    pd.DataFrame({
        "event_id": [f"{d}|e" for d in drivers], "follower_id": list(drivers), "site": "a", "n_scored": 100,
        "rmse_s": [factor * BASE.get(d, 2.0) for d in drivers], "rmse_v": 0.5,
        "collided": [d in collided for d in drivers],
    }).to_parquet(run / "test_events.parquet", index=False)  # fmt: skip
    if transfer is not None:
        targets = {}
        for target, (per_driver, degradation) in transfer.items():
            targets[target] = {"relative_degradation": degradation}
            pd.DataFrame({
                "event_id": [f"{d}|e" for d in per_driver], "follower_id": list(per_driver),
                "rmse_s": [v[0] for v in per_driver.values()], "rmse_v": 1.0, "collided": False, "n_scored": 400,
                "rmse_s_h": [v[1] for v in per_driver.values()], "rmse_v_h": 0.8, "collided_h": False,
                "n_scored_h": 121,
            }).to_parquet(run / f"transfer_{target}.parquet", index=False)  # fmt: skip
        write_json(run / "transfer.json", {"targets": targets})
    if certificate is not None:
        write_json(run / "certificate.json", certificate)
    return run


def certificate(a_priori: bool, at_equilibria: bool, n_hold: int, applicable: bool = True) -> dict:
    if not applicable:
        return {"model": "mlp", "applicable": False}
    return {"model": "residual_idm", "applicable": True, "a_priori": {"holds": a_priori, "n_hold": 26, "n_grid": 26},
            "at_equilibria": {"holds": at_equilibria, "n_hold": n_hold, "n_equilibria": 24}}  # fmt: skip


def build_tree(root: Path) -> Path:
    """Every run of the reduced design (see the module docstring)."""
    for fold in FOLDS:
        degradation = {0: 1.0, 1: 1.2}[fold]
        for model, (factor, stable, unstable, outside, unstable_eq) in E1.items():
            for seed in SEEDS if model != "idm" else SEEDS[:1]:
                transfer = None
                if seed == 0:
                    transfer = {"ngsim_i80": ({"t0": (10.0, 4.0), "t1": (12.0, 6.0)}, degradation)}
                collided = ("d0",) if model == "mlp" and seed == 0 else ()
                write_run(root, "e1", DATA, model, fold, seed, factor=factor, stable=stable, unstable=unstable,
                          outside=outside, unstable_eq=unstable_eq, collided=collided, transfer=transfer)  # fmt: skip
        for (arch, weight), (stable, val, feasible) in SWEEP.items():
            kind, chosen_weight, change, growth = CHOSEN[arch]
            chosen = weight == chosen_weight
            for seed in SEEDS if chosen else SEEDS[:1]:
                transfer = None
                if chosen and seed == 0:
                    transfer = {"ngsim_i80": ({"t0": (9.0, 3.0), "t1": (11.0, 5.0)}, 0.5)}
                write_run(root, f"e2_{kind}_w{weight:g}", DATA, arch, fold, seed, factor=E1[arch][0] * change,
                          stable=stable, unstable=1.0 - stable, val=val, feasible=feasible, growth=growth,
                          hysteresis=80.0, transfer=transfer)  # fmt: skip
        for seed in SEEDS:  # E4
            write_run(root, "e4_free_ft", FT, "residual_idm", fold, seed, stable=0.6, unstable=0.4,
                      certificate=certificate(False, False, 20))  # fmt: skip
            write_run(root, "e4_free_ft", FT, "mlp", fold, seed, stable=0.3, unstable=0.5, outside=0.2,
                      unstable_eq=0.1, certificate=certificate(False, False, 0, applicable=False))  # fmt: skip
            write_run(root, "e4_stable", DATA, "residual_idm", fold, seed, stable=0.98, unstable=0.02,
                      certificate=certificate(True, False, 24))  # fmt: skip
            write_run(root, "e4_stable_ft", FT, "residual_idm", fold, seed, stable=0.95, unstable=0.05,
                      unstable_eq=0.5, certificate=certificate(True, True, 24))  # fmt: skip
        for view in ("openacc_acc", "openacc_human"):  # E5
            for model, acc, human in (("idm", 0.4, 0.4), ("mlp", 0.5, 0.6), ("gru", 0.5, 0.6)):
                write_run(root, "e5", view, model, fold, 0, acc=acc, human=human)
            write_run(root, "e5_jacobian_w0.1", view, "mlp", fold, 0, acc=0.3, human=0.3)
            write_run(root, "e5_gain_w1", view, "gru", fold, 0, acc=0.475, human=0.39)
        for (arch, weight), (stable, val, change) in LOWFREQ.items():  # the pilot on fold 0, the chosen weight on all
            if fold == 0 or weight == LOWFREQ_CHOSEN[arch]:
                write_run(root, f"e2_combined_j{weight:g}", DATA, arch, fold, 0, factor=E1[arch][0] * change,
                          stable=stable, unstable=1.0 - stable, val=val)  # fmt: skip
    # the control of M8 (D117): the monotonicity terms only, fold 0, seed 0
    write_run(root, "e2_monotone_w1", DATA, "mlp", 0, 0, factor=E1["mlp"][0] * 1.1, stable=0.35, unstable=0.6,
              outside=0.05, unstable_eq=0.7)  # fmt: skip
    # the long-window arm of the revision (e2_horizon): fold 0, seed 0, the GRU stabilised at a cost
    write_run(root, "e2_gain_long_w0.1", DATA, "gru", 0, 0, factor=E1["gru"][0] * 1.16, stable=0.77, unstable=0.0,
              outside=0.23, unstable_eq=0.05, collided=("d0",))  # fmt: skip
    write_run(root, "e2_gain_long_w0.1", DATA, "lstm", 0, 0, factor=E1["lstm"][0] * 1.2, stable=0.5, unstable=0.3,
              outside=0.2, unstable_eq=0.3)  # fmt: skip
    return root


def config(root: Path, **changes) -> TablesConfig:
    design = dict(
        n_resamples=200, folds=FOLDS, seeds=SEEDS, learned=("mlp", "gru", "lstm"), baselines=("idm",),
        penalties={"mlp": "jacobian", "gru": "gain", "lstm": "gain"}, weights=(0.1, 1.0), targets=("ngsim_i80",),
        e5_models=("idm", "mlp", "gru"), e5_penalised=("mlp", "gru"), lowfreq_architectures=("gru", "lstm"),
        verdicts={"h1_3_min_pairs": 2},  # two folds: two pairs at most
        monotone_architectures=("mlp",),
    )  # fmt: skip
    return TablesConfig(runs_root=root, out_dir=root / "_tables" / "m4", **{**design, **changes})


def read(out: Path, name: str) -> dict[str, dict]:
    """Rows of ``<name>.csv`` keyed by their first columns."""
    frame = pd.read_csv(out / f"{name}.csv")
    keys = {"e1": ["model"], "e2_sweep": ["architecture", "weight"], "e2": ["architecture"],
            "e2_lowfreq": ["architecture", "weight"], "e3": ["architecture", "penalty", "target"],
            "e4": ["experiment", "model"], "e5": ["view", "model", "penalty"],
            "verdicts": ["hypothesis", "unit"]}[name]  # fmt: skip
    return {tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]: row for row in frame.to_dict("records")}


@pytest.fixture(scope="module")
def complete(tmp_path_factory) -> tuple[Path, list[str]]:
    root = build_tree(tmp_path_factory.mktemp("tables") / "runs")
    return root, make_tables(config(root))


def test_e1_complete(complete):
    root, lines = complete
    out = root / "_tables" / "m4"
    assert lines[0].startswith(
        "TABLE e1: 4 rows from 14/14 runs, 0 missing; H1.1 does not hold (mlp holds, gru does not hold, lstm does not "
        "hold; on band share not stable: holds) -> "
    )
    e1 = read(out, "e1")
    mlp, gru, lstm, idm = (e1[m] for m in ("mlp", "gru", "lstm", "idm"))
    assert (mlp["runs"], mlp["drivers"], idm["runs"]) == (4, 4, 2)
    assert mlp["rmse_s"] == pytest.approx(2.4) and idm["rmse_s"] == pytest.approx(3.0)
    assert 1.6 <= mlp["rmse_s_low"] <= 2.4 <= mlp["rmse_s_high"] <= 3.2
    assert mlp["collision_rate"] == pytest.approx(0.125)  # driver d0 collides with one of its two seeds
    for model, change in (("mlp", -0.2), ("gru", -0.1), ("lstm", 0.1)):  # the same factor for every driver
        row = e1[model]
        assert (row["rmse_vs_ref"], row["rmse_vs_ref_low"], row["rmse_vs_ref_high"]) == pytest.approx((change,) * 3)
        assert row["rmse_vs_ref_p"] == pytest.approx(0.125)  # four drivers, one sign: 2 / 2^4
    assert (mlp["stable"], mlp["unstable"], mlp["outside"], mlp["none"]) == pytest.approx((0.3, 0.6, 0.1, 0.0))
    assert (mlp["unstable_low"], mlp["unstable_high"]) == pytest.approx((0.6, 0.6))
    assert (mlp["not_stable"], mlp["not_stable_low"]) == pytest.approx((0.7, 0.7))
    assert mlp["unstable_sign"] == pytest.approx(0.5) and mlp["max_gain_median"] == pytest.approx(1.25)
    # among the equilibria found: the MLP 0.8, the GRU 0.2 although not stable inside the band at 0.9
    assert (mlp["unstable_eq"], mlp["unstable_eq_low"], mlp["unstable_eq_high"]) == pytest.approx((0.8,) * 3)
    assert (gru["unstable_eq"], gru["not_stable"]) == pytest.approx((0.2, 0.9))
    assert (mlp["unstable_eq_sign"], mlp["unstable_eq_sign_low"]) == pytest.approx((0.7, 0.7))
    # H1.1 on the equilibria found: the GRU fails; the LSTM fails on the RMSE (above the IDM) either way
    assert (mlp["h1_1"], gru["h1_1"], lstm["h1_1"]) == ("holds", "does not hold", "does not hold")
    assert (mlp["h1_1_information"], gru["h1_1_information"], lstm["h1_1_information"]) == (
        "holds", "holds", "does not hold",
    )  # fmt: skip
    assert pd.isna(idm["h1_1"]) and mlp["complete"] and idm["complete"]
    # M9: 145 of the 400 equilibria of the GRU are numerically stable with a pole outside the unit circle, 5 of them
    # inside the band; the verdicts and the summary line stay on the numerical rule
    assert (gru["unstable_eq_full"], gru["unstable_eq_full_low"], gru["unstable_eq_full_high"]) == pytest.approx(
        (225 / 400,) * 3
    )  # fmt: skip
    assert (gru["not_stable_full"], gru["not_stable_full_low"], gru["not_stable_full_high"]) == pytest.approx(
        (0.95,) * 3
    )  # fmt: skip
    assert (lstm["unstable_eq_full"], lstm["not_stable_full"]) == pytest.approx((0.9, 0.9))  # no pole outside
    assert pd.isna(mlp["unstable_eq_full"]) and pd.isna(idm["not_stable_full"])  # memoryless: blank
    assert (gru["h1_1_poles"], lstm["h1_1_poles"]) == ("holds", "does not hold") and pd.isna(mlp["h1_1_poles"])
    md = (out / "e1.md").read_text(encoding="utf-8")
    assert "- Runs: 14 of 14 expected runs exist" in md and "unit driver" in md and "200 resamples" in md
    assert "| mlp | 4 | 4 | 2.40 [" in md and "-20.0 % [-20.0 %, -20.0 %]" in md and "| 0.80 [0.80, 0.80] |" in md
    assert "H1.1 overall (holds for at least 2 of 3): does not hold" in md
    assert "For information, the same rule on band share not stable: holds" in md
    header = "| unstable among equilibria | unstable among equilibria (poles incl.) | unstable among equilibria (sign) |"
    assert header in md and "| not stable | not stable (poles incl.) |" in md and "(poles incl.) (information) |" in md
    assert "| 0.20 [0.20, 0.20] | 0.56 [0.56, 0.56] |" in md and "Section 3.2 of the paper" in md
    assert "noted in missing.txt as 'full_history.json missing'" in md and "unit driver (event on highD)" in md
    verdicts = read(out, "verdicts")
    assert verdicts["H1.1", "gru"]["information"].startswith("band share not stable 0.90 [0.90, 0.90]: holds")
    assert verdicts["H1.1", "overall"]["information"] == "band share not stable: holds"
    assert (out / "missing.txt").read_text(encoding="utf-8") == ""


def test_e2_monotone(complete):
    """D117: the monotone arm of fold 0 next to E1 and the chosen E2 weight of the same fold and seed, in m8_dir."""
    root, lines = complete
    out = root / "_tables" / "m8"
    frame = pd.read_csv(out / "e2_monotone.csv").set_index("arm")
    assert list(frame.index) == ["E1", "E2 (jacobian, weight 0.1)", "monotone (D117)"]
    assert (frame["experiment"].tolist(), frame["runs"].tolist()) == (["e1", "e2_jacobian_w0.1", "e2_monotone_w1"],
                                                                      [1, 1, 1])  # fmt: skip
    e1, e2, monotone = (frame.loc[arm] for arm in frame.index)
    assert e1["rmse_s"] == pytest.approx(2.4) and e1["drivers"] == 2 and pd.isna(e1["rmse_change"])
    assert e2["rmse_change"] == pytest.approx(0.05) and monotone["rmse_change"] == pytest.approx(0.10)
    assert monotone["rmse_change_pairs"] == 2 and monotone["stable"] == pytest.approx(0.35)
    assert monotone["not_stable"] == pytest.approx(0.65) and monotone["unstable"] == pytest.approx(0.6)
    assert monotone["unstable_eq"] == pytest.approx(0.7) and monotone["max_gain"] == pytest.approx(1.1)
    assert e2["not_stable"] == pytest.approx(0.05) and bool(monotone["complete"])
    md = (out / "e2_monotone.md").read_text(encoding="utf-8")
    assert "| mlp | monotone (D117) | 1 | 2 | 2.64 [" in md and "+10.0 % [" in md
    assert any(line.startswith("TABLE e2_monotone: 3 rows from 3/3 runs, 0 missing") for line in lines), lines
    assert not (root / "_tables" / "m4" / "e2_monotone.csv").exists()


def test_e2_horizon(complete):
    """The long-window arm of the revision: E1, the chosen E2 weight and the long rollout of fold 0, seed 0, per
    recurrent architecture, in out_dir (m4)."""
    root, lines = complete
    out = root / "_tables" / "m4"
    frame = pd.read_csv(out / "e2_horizon.csv")
    assert frame["architecture"].tolist() == ["gru"] * 3 + ["lstm"] * 3
    gru = frame[frame["architecture"] == "gru"].set_index("arm")
    assert list(gru.index) == ["E1", "E2 (gain, weight 1, rollout 40 s, last 20 s)",
                               "long window (rollout 380 s, last 252 s)"]  # fmt: skip
    assert gru["experiment"].tolist() == ["e1", "e2_gain_w1", "e2_gain_long_w0.1"] and gru["runs"].tolist() == [1, 1, 1]
    e1, e2, long = (gru.loc[arm] for arm in gru.index)
    assert e1["rmse_s"] == pytest.approx(2.7) and long["rmse_s"] == pytest.approx(2.7 * 1.16)
    assert (long["rmse_change"], long["rmse_change_low"], long["rmse_change_high"]) == pytest.approx((0.16,) * 3)
    assert e2["rmse_change"] == pytest.approx(0.3) and pd.isna(e1["rmse_change"])
    assert (long["stable"], long["unstable"], long["outside"], long["none"]) == pytest.approx((0.77, 0.0, 0.23, 0.0))
    assert (long["not_stable"], long["unstable_eq"]) == pytest.approx((0.23, 0.05))
    assert pd.isna(long["lowfreq_unstable"])  # the synthetic audits carry no frequency responses
    assert long["max_gain"] == pytest.approx(1.1) and long["epochs"] == 3 and pd.isna(long["best_epoch"])
    assert bool(long["complete"]) and bool(e1["complete"])
    # M9: 7 stable speeds inside the band and 2 outside it with a pole outside the unit circle
    assert (long["not_stable_full"], long["unstable_eq_full"]) == pytest.approx((0.30, 0.14))
    assert (e1["not_stable_full"], e1["unstable_eq_full"], e2["not_stable_full"]) == pytest.approx((0.95, 0.5625, 0.2))
    md = (out / "e2_horizon.md").read_text(encoding="utf-8")
    assert "| gru | long window (rollout 380 s, last 252 s) | 1 | 2 | 3.13 [" in md and "+16.0 % [" in md
    assert "gain above threshold at omega <= 0.1" in md and "| 0.23 [0.23, 0.23] | 0.30 [0.30, 0.30] |" in md
    assert any(line.startswith("TABLE e2_horizon: 6 rows from 6/6 runs, 0 missing") for line in lines), lines


def test_e2_sweep_and_the_chosen_weights(complete):
    root, lines = complete
    out = root / "_tables" / "m4"
    assert lines[1].startswith("TABLE e2_sweep: 6 rows from 12/12 runs, 0 missing; chosen: mlp 0.1, gru 1, lstm 1")
    sweep = read(out, "e2_sweep")
    assert (sweep["mlp", 0.1]["val_rmse_s"], sweep["mlp", 0.1]["test_rmse_s"]) == pytest.approx((2.5, 2.6))
    assert sweep["mlp", 0.1]["stable"] == pytest.approx(0.95) and sweep["mlp", 0.1]["feasible"] == 2
    assert (sweep["gru", 0.1]["feasible"], sweep["lstm", 0.1]["feasible"]) == (0, 0)
    assert [key for key, row in sweep.items() if row["chosen"]] == [("mlp", 0.1), ("gru", 1.0), ("lstm", 1.0)]
    chosen = json.loads((out / "chosen_weights.json").read_text(encoding="utf-8"))
    assert {arch: entry["weight"] for arch, entry in chosen.items()} == {"mlp": 0.1, "gru": 1.0, "lstm": 1.0}
    assert chosen["gru"]["rule"].startswith("no weight reaches") and chosen["mlp"]["rule"].startswith("stable >= 0.9")
    assert chosen["mlp"]["experiment"] == "e2_jacobian_w0.1" and all(entry["complete"] for entry in chosen.values())


def test_choose_weight():
    def sweep(*rows):
        return pd.DataFrame(rows, columns=["weight", "stable", "val_rmse_s"])

    assert choose_weight(sweep((0.01, 0.5, 2.0), (0.1, 0.91, 2.4), (1.0, 0.99, 2.3), (10.0, 0.95, 2.1)), 0.9)[0] == 10.0
    weight, rule = choose_weight(sweep((0.01, 0.5, 2.0), (0.1, 0.85, 2.4), (1.0, 0.7, 2.3)), 0.9)
    assert weight == 0.1 and rule.startswith("no weight reaches")
    assert choose_weight(sweep((0.1, 0.8, 2.4), (1.0, 0.8, 2.3), (10.0, 0.8, 2.3)), 0.9)[0] == 1.0  # ties: RMSE, weight
    assert choose_weight(sweep((0.1, float("nan"), 2.0), (1.0, 0.95, float("nan"))), 0.9)[0] == 1.0
    assert choose_weight(sweep((0.1, float("nan"), 2.0)), 0.9) == (None, "no audited run")
    assert verdict_overall(["holds", "", "holds"], 2) == "holds"
    assert verdict_overall(["does not hold", "", ""], 2) == "open"
    assert verdict_overall(["does not hold", "does not hold", ""], 2) == "does not hold"


def test_e2_h1_2(complete):
    root, lines = complete
    assert lines[2].endswith("H1.2 mlp confirmed, gru refuted, lstm open -> " + str(root / "_tables" / "m4" / "e2.md"))
    e2 = read(root / "_tables" / "m4", "e2")
    for arch, change, not_stable, verdict in (("mlp", 0.05, 0.05, "confirmed"), ("gru", 0.3, 0.2, "refuted"),
                                              ("lstm", 0.15, 0.05, "open")):  # fmt: skip
        row = e2[arch]
        assert (row["runs_e1"], row["runs_e2"], row["rmse_change_pairs"]) == (4, 4, 4)
        assert (row["rmse_change"], row["rmse_change_low"], row["rmse_change_high"]) == pytest.approx((change,) * 3)
        assert (row["not_stable_e2"], row["not_stable_e2_high"]) == pytest.approx((not_stable,) * 2)
        assert row["rmse_change_p"] == pytest.approx(0.125) and row["rmse_change_p_holm"] == pytest.approx(0.375)
        assert row["h1_2"] == verdict and row["complete"]
    assert e2["mlp"]["not_stable_e1"] == pytest.approx(0.7) and e2["mlp"]["weight"] == 0.1
    mlp = e2["mlp"]  # among the equilibria found, next to the band shares; the verdicts stay on "not stable"
    shares = (mlp["unstable_eq_e1"], mlp["unstable_eq_e2"], mlp["unstable_eq_e2_high"])
    assert shares == pytest.approx((0.8, 0.05, 0.05))
    assert (e2["mlp"]["growth_error_e1"], e2["mlp"]["growth_error_e2"]) == pytest.approx((0.4, 0.3))
    assert (e2["mlp"]["growth_error_change"], e2["gru"]["growth_error_change"]) == pytest.approx((-0.25, -0.5))
    assert e2["lstm"]["hysteresis_change"] == pytest.approx(-0.2) and e2["lstm"]["growth_error_pairs"] == 4
    # M9: the poles of the full-history loop; H1.2 stays open for the LSTM, on not stable (poles incl.) it is refuted
    lstm, gru = e2["lstm"], e2["gru"]
    assert (lstm["not_stable_full_e2"], lstm["not_stable_full_e2_low"], lstm["not_stable_full_e2_high"]) == (
        pytest.approx((0.35,) * 3)
    )  # fmt: skip
    assert (lstm["unstable_eq_full_e2"], lstm["not_stable_full_e1"], lstm["unstable_eq_full_e1"]) == pytest.approx(
        (0.35, 0.9, 0.9)
    )  # fmt: skip
    assert (lstm["h1_2"], lstm["h1_2_poles"], gru["h1_2_poles"]) == ("open", "refuted", "refuted")
    assert (gru["not_stable_full_e1"], gru["unstable_eq_full_e1"]) == pytest.approx((0.95, 225 / 400))
    assert (gru["not_stable_full_e2"], gru["unstable_eq_full_e2"]) == pytest.approx((0.2, 0.2))
    assert pd.isna(e2["mlp"]["not_stable_full_e2"]) and pd.isna(e2["mlp"]["h1_2_poles"])
    md = (root / "_tables" / "m4" / "e2.md").read_text(encoding="utf-8")
    assert "| not stable E1 | not stable E1 (poles incl.) | not stable E2 | not stable E2 (poles incl.) |" in md
    assert "| H1.2 | H1.2 on band share not stable (poles incl.) (information) | complete |" in md
    verdicts = read(root / "_tables" / "m4", "verdicts")
    assert verdicts["H1.2", "lstm"]["verdict"] == "open"


def test_e3_transfer(complete):
    root, _ = complete
    e3 = read(root / "_tables" / "m4", "e3")
    none, chosen = e3["mlp", "none", "ngsim_i80"], e3["mlp", "chosen", "ngsim_i80"]
    assert (none["runs"], none["drivers"]) == (2, 2)
    assert none["source_rmse_s"] == pytest.approx(2.4) and chosen["source_rmse_s"] == pytest.approx(2.52)
    assert (none["rmse_s_h"], none["rmse_s_full"]) == pytest.approx((5.0, 11.0))
    assert (none["rmse_s_h_low"], none["rmse_s_h_high"]) == pytest.approx((4.0, 6.0))  # two drivers
    assert none["degradation"] == pytest.approx(5.0 / 2.4 - 1.0) and none["degradation_runs"] == pytest.approx(1.1)
    assert chosen["degradation"] == pytest.approx(4.0 / 2.52 - 1.0) and chosen["degradation_runs"] == pytest.approx(0.5)
    # with - without penalty, the two drivers paired: t0 alone, t1 alone, or both
    assert chosen["degradation_change"] == pytest.approx((4.0 / 2.52 - 1.0) - (5.0 / 2.4 - 1.0))
    assert chosen["degradation_change_low"] == pytest.approx((5.0 / 2.52) - (6.0 / 2.4))
    assert chosen["degradation_change_high"] == pytest.approx((3.0 / 2.52) - (4.0 / 2.4))
    assert pd.isna(none["degradation_change"]) and ("idm", "none", "ngsim_i80") in e3
    assert ("idm", "chosen", "ngsim_i80") not in e3  # a baseline has no penalty


def test_e2_lowfreq_h1_2_combined(complete):
    """D110: the weight of the pilot (fold 0) by the rule of D85, on every fold; the verdict of H1.2 on it."""
    root, lines = complete
    out = root / "_tables" / "m4"
    assert lines[3].startswith("TABLE e2_lowfreq: 4 rows from 12/12 runs, 0 missing; H1.2 (combined) gru confirmed, "
                               "lstm refuted")  # fmt: skip
    table = read(out, "e2_lowfreq")
    gru, other = table["gru", 0.1], table["gru", 1.0]
    assert gru["chosen"] and not other["chosen"] and gru["rule"].startswith("stable >= 0.9")
    assert (gru["runs"], gru["runs_expected"], other["runs"], other["runs_expected"]) == (2, 2, 1, 1)
    assert (gru["rmse_change"], gru["rmse_change_low"], gru["rmse_change_high"]) == pytest.approx((0.05,) * 3)
    assert (gru["not_stable"], gru["not_stable_high"], gru["h1_2"]) == (pytest.approx(0.05), pytest.approx(0.05),
                                                                        "confirmed")  # fmt: skip
    assert other["rmse_change"] == pytest.approx(0.02) and pd.isna(other["h1_2"])  # fold 0 only, no verdict
    # M9: no pole outside the unit circle at the stable speeds: the shares with the poles equal those without
    assert (gru["not_stable_full"], gru["unstable_eq_full"], other["not_stable_full"]) == pytest.approx((0.05, 0.05, 0.03))
    lstm = table["lstm", 1.0]
    assert lstm["chosen"] and lstm["rule"].startswith("no weight reaches") and lstm["h1_2"] == "refuted"
    assert lstm["rmse_change"] == pytest.approx(0.3) and gru["drivers"] == 4 and other["drivers"] == 2
    verdicts = read(out, "verdicts")
    assert verdicts["H1.2 (combined)", "gru"]["verdict"] == "confirmed" and verdicts["H1.2 (combined)", "gru"]["complete"]
    assert verdicts["H1.2 (combined)", "lstm"]["basis"].startswith("weight 1 (no weight reaches stable >= 0.9")
    assert verdicts["H1.2", "gru"]["verdict"] == "refuted"  # the verdict of E2 stays


def test_e2_lowfreq_without_runs_and_configured(tmp_path):
    root = build_tree(tmp_path / "runs")
    shutil.rmtree(root / "e2_combined_j0.1")
    shutil.rmtree(root / "e2_combined_j1")
    lines = make_tables(config(root, tables=("e2_lowfreq",)))
    assert lines[0].startswith("TABLE e2_lowfreq: 4 rows from 4/8 runs, 4 missing; H1.2 (combined) n/a: no weight")
    missing = (root / "_tables" / "m4" / "missing.txt").read_text(encoding="utf-8").splitlines()
    assert "[e2_lowfreq] e2_combined_j0.1/follownet_highd/gru/driver_fold0_seed0: run missing" in missing
    verdicts = read(root / "_tables" / "m4", "verdicts")
    assert pd.isna(verdicts["H1.2 (combined)", "lstm"]["verdict"]) and not verdicts["H1.2 (combined)", "lstm"]["complete"]
    root = build_tree(tmp_path / "again")
    make_tables(config(root, tables=("e2_lowfreq",), lowfreq_chosen={"gru": 1.0}))  # the configuration decides
    table = read(root / "_tables" / "m4", "e2_lowfreq")
    assert table["gru", 1.0]["chosen"] and table["gru", 1.0]["runs_expected"] == 2 and table["gru", 1.0]["runs"] == 1
    missing = (root / "_tables" / "m4" / "missing.txt").read_text(encoding="utf-8")
    assert "e2_combined_j1/follownet_highd/gru/driver_fold1_seed0: run missing" in missing


def test_e4_h1_5(complete):
    root, lines = complete
    assert lines[5].split(" -> ")[0].endswith("; H1.5 holds")
    e4 = read(root / "_tables" / "m4", "e4")
    stable, stable_ft = e4["e4_stable", "residual_idm"], e4["e4_stable_ft", "residual_idm"]
    free, mlp = e4["e4_free_ft", "residual_idm"], e4["e4_free_ft", "mlp"]
    assert (stable["certificates"], stable["a_priori_holds"], stable["at_equilibria_holds"]) == (4, 4, 0)
    assert stable["all_equilibria_certified"] == 4 and stable_ft["at_equilibria_holds"] == 4
    assert (free["a_priori_holds"], free["all_equilibria_certified"]) == (0, 0)
    assert mlp["certificates"] == 0 and pd.isna(mlp["a_priori_holds"])  # the MLP has no certificate
    assert stable_ft["unstable_high"] == pytest.approx(0.05) and mlp["unstable_low"] == pytest.approx(0.5)
    # the shares among the equilibria found would give the opposite verdict: H1.5 stays on the band share
    assert (stable_ft["unstable_eq"], mlp["unstable_eq"]) == pytest.approx((0.5, 0.1))
    assert e4["e1", "mlp"]["unstable"] == pytest.approx(0.6) and pd.isna(e4["e1", "mlp"]["certificates"])
    assert stable_ft["rmse_s"] == pytest.approx(4.0) and stable["rmse_s"] == pytest.approx(3.0)  # NGSIM, HighD drivers
    verdicts = read(root / "_tables" / "m4", "verdicts")
    assert verdicts["H1.5", "overall"]["verdict"] == "holds" and verdicts["H1.5", "overall"]["complete"]


def test_e5_h1_3(complete):
    root, lines = complete
    assert lines[6].split(" -> ")[0].endswith("H1.3 openacc_acc open, openacc_human confirmed")
    e5 = read(root / "_tables" / "m4", "e5")
    for view, model, reduction, verdict in (
        ("openacc_acc", "mlp", 0.4, "confirmed"), ("openacc_acc", "gru", 0.05, "refuted"),
        ("openacc_human", "mlp", 0.5, "confirmed"), ("openacc_human", "gru", 0.35, "confirmed"),
    ):  # fmt: skip
        row = e5[view, model, "chosen"]
        assert (row["reduction"], row["reduction_low"], row["reduction_high"]) == pytest.approx((reduction,) * 3)
        assert row["h1_3"] == verdict and row["reduction_pairs"] == 2
    pooled_acc, pooled_human = e5["openacc_acc", "pooled", "chosen"], e5["openacc_human", "pooled", "chosen"]
    assert pooled_acc["reduction"] == pytest.approx(1.0 - 0.3875 / 0.5) and pooled_acc["h1_3"] == "open"
    assert pooled_human["reduction"] == pytest.approx(1.0 - 0.345 / 0.6) and pooled_human["h1_3"] == "confirmed"
    assert 0.35 - 1e-9 <= pooled_human["reduction_low"] <= pooled_human["reduction_high"] <= 0.5 + 1e-9
    assert e5["openacc_acc", "mlp", "none"]["growth_error"] == pytest.approx(0.5)  # the ACC profiles for the ACC view
    assert e5["openacc_human", "mlp", "none"]["growth_error"] == pytest.approx(0.6)
    assert e5["openacc_acc", "idm", "none"]["runs"] == 2 and pd.isna(e5["openacc_acc", "idm", "none"]["reduction"])
    # M9: the GRU without penalty on the ACC view has 10 stable speeds whose loop is locally unstable
    none, chosen = e5["openacc_acc", "gru", "none"], e5["openacc_acc", "gru", "chosen"]
    assert (none["not_stable"], none["not_stable_full"], none["unstable_eq"], none["unstable_eq_full"]) == pytest.approx(
        (0.5, 0.6, 0.5, 0.6)
    )  # fmt: skip
    assert (chosen["not_stable_full"], chosen["unstable_eq_full"]) == pytest.approx((0.5, 0.5))
    assert pd.isna(e5["openacc_acc", "mlp", "chosen"]["not_stable_full"]) and pd.isna(pooled_acc["unstable_eq_full"])
    md = (root / "_tables" / "m4" / "e5.md").read_text(encoding="utf-8")
    assert "| not stable | not stable (poles incl.) | unstable among equilibria | unstable among equilibria (poles" in md


def test_missing_runs_files_and_values(tmp_path):
    root = build_tree(tmp_path / "runs")
    shutil.rmtree(root / "e1" / DATA / "mlp" / "driver_fold1_seed1")
    (root / "e1" / DATA / "gru" / "driver_fold0_seed1" / "stability.json").unlink()
    write_json(root / "e1" / DATA / "lstm" / "driver_fold0_seed0" / "stability.json", {"error": "RuntimeError: broken"})
    shutil.rmtree(root / "e4_stable_ft")
    (root / "e5_gain_w1" / "openacc_acc" / "gru" / "driver_fold1_seed0" / "platoon.json").unlink()
    write_json(root / "e5" / "openacc_human" / "mlp" / "driver_fold0_seed0" / "platoon.json", {"summary": {}})
    lines = make_tables(config(root))
    out = root / "_tables" / "m4"
    missing = (out / "missing.txt").read_text(encoding="utf-8").splitlines()
    for expected in (
        "[e1] e1/follownet_highd/mlp/driver_fold1_seed1: run missing",
        "[e1] e1/follownet_highd/gru/driver_fold0_seed1: stability.json missing",
        "[e1] e1/follownet_highd/lstm/driver_fold0_seed0: stability.json RuntimeError: broken",
        "[e4] e4_stable_ft/ngsim_i80/residual_idm: run missing (all 4: folds 0, 1, seeds 0, 1)",  # one line
        "[e5] e5_gain_w1/openacc_acc/gru/driver_fold1_seed0: platoon.json missing",
        "[e5] e5/openacc_human/mlp/driver_fold0_seed0: no value platoon_prefix_growth_error_human in platoon.json",
    ):  # fmt: skip
        assert expected in missing, expected
    assert not any("e4_stable_ft/ngsim_i80/residual_idm/driver_fold" in line for line in missing)
    assert lines[0].startswith("TABLE e1: 4 rows from 13/14 runs, 3 missing; H1.1 does not hold (incomplete)")
    e1 = read(out, "e1")
    assert (e1["mlp"]["runs"], e1["mlp"]["audited"], e1["gru"]["audited"], e1["lstm"]["audited"]) == (3, 3, 3, 3)
    assert e1["mlp"]["drivers"] == 4 and not e1["mlp"]["complete"]  # d2 keeps its seed 0
    assert e1["mlp"]["rmse_vs_ref"] == pytest.approx(-0.2) and e1["mlp"]["h1_1"] == "holds"
    e4 = read(out, "e4")
    assert e4["e4_stable_ft", "residual_idm"]["runs"] == 0 and pd.isna(e4["e4_stable_ft", "residual_idm"]["unstable"])
    verdicts = read(out, "verdicts")
    assert pd.isna(verdicts["H1.5", "overall"]["verdict"]) and not verdicts["H1.1", "mlp"]["complete"]
    e5 = read(out, "e5")
    gru = e5["openacc_acc", "gru", "chosen"]
    assert gru["reduction_pairs"] == 1 and not gru["complete"]
    assert e5["openacc_human", "mlp", "chosen"]["reduction_pairs"] == 1
    assert "|  |" in (out / "e4.md").read_text(encoding="utf-8")  # empty cells, no error


def test_the_poles_need_a_full_history_that_matches_the_audit(tmp_path):
    """M9: a recurrent run without full_history.json, or whose file no longer gives the shares of its audit, is noted
    and left out of the shares with the poles; the memoryless laws have none and are not noted; the verdicts stay."""
    root = build_tree(tmp_path / "runs")
    (root / "e1" / DATA / "gru" / "driver_fold1_seed1" / "full_history.json").unlink()
    stale = root / "e2_gain_w1" / DATA / "lstm" / "driver_fold0_seed0"
    write_json(stale / "full_history.json", full_history(0.9, 0.1, 0.0, 0.1))  # the audit has 0.95 / 0.05
    (root / "e2_gain_w1" / DATA / "gru" / "driver_fold1_seed0" / "full_history.json").write_text("{", encoding="utf-8")
    lines = make_tables(config(root, tables=("e1", "e2")))
    out = root / "_tables" / "m4"
    missing = (out / "missing.txt").read_text(encoding="utf-8").splitlines()
    assert "[e1] e1/follownet_highd/gru/driver_fold1_seed1: full_history.json missing" in missing
    assert "[e2] e1/follownet_highd/gru/driver_fold1_seed1: full_history.json missing" in missing  # E1 runs of e2
    assert "[e2] e2_gain_w1/follownet_highd/gru/driver_fold1_seed0: full_history.json unreadable (JSONDecodeError)" in (
        missing
    )  # fmt: skip
    assert ("[e2] e2_gain_w1/follownet_highd/lstm/driver_fold0_seed0: full_history.json does not match stability.json "
            "(share_unstable_numerical 0.1000 from the file, 0.0500 in the audit; rerun "
            "scripts/analysis/full_history_audit.py)") in missing  # fmt: skip
    assert len(missing) == 4 and not any("/mlp/" in line or "/idm/" in line for line in missing)
    assert lines[0].startswith("TABLE e1: 4 rows from 14/14 runs, 1 missing; H1.1 does not hold (mlp holds, gru does "
                               "not hold, lstm does not hold; on band share not stable: holds)")  # fmt: skip
    assert lines[1].startswith("TABLE e2: 3 rows from 24/24 runs, 3 missing; H1.2 mlp confirmed, gru refuted, lstm open")
    e1, e2 = read(out, "e1"), read(out, "e2")
    assert e1["gru"]["unstable_eq_full"] == pytest.approx(225 / 400) and e1["gru"]["h1_1"] == "does not hold"
    assert e1["gru"]["complete"] and e1["gru"]["audited"] == 4  # the shares of the numerical rule keep every run
    assert e2["lstm"]["not_stable_full_e2"] == pytest.approx(0.35) and e2["lstm"]["h1_2_poles"] == "refuted"


def test_full_history_shares():
    """M9: the speeds in support; a pole outside decides a speed without numerical verdict; the band share counts
    only the stable equilibria inside the band with every pole inside, over the speeds that have a band."""

    def speed(status, in_band, numerical, poles, in_support=True):
        return {"status": status, "in_support": in_support, "in_band": in_band, "numerical_unstable": numerical,
                "poles_stable": poles}  # fmt: skip

    payload = {"equilibria": [
        speed("ok", True, False, True), speed("multiple", True, False, False), speed("ok", True, True, False),
        speed("outside", False, False, False), speed("ok", None, False, True), speed("ok", True, None, False),
        speed("ok", True, None, True), speed("ok", True, False, False, in_support=False),
    ]}  # fmt: skip
    shares = full_history_shares(payload, 8)
    assert shares["unstable_eq"] == pytest.approx(1 / 5) and shares["stable"] == pytest.approx(2 / 8)
    assert shares["unstable_eq_full"] == pytest.approx(4 / 6)  # the speed without verdict and a pole outside counts
    assert shares["not_stable_full"] == pytest.approx(1 - 1 / 8)
    empty = full_history_shares({"equilibria": []}, float("nan"))
    assert all(math.isnan(value) for value in empty.values())
    assert full_history_shares({"equilibria": []}, 4)["not_stable_full"] == 1.0


def test_a_single_run(tmp_path):
    root = build_tree(tmp_path / "runs")
    lines = make_tables(config(root, folds=(0,), seeds=(0,)))
    out = root / "_tables" / "m4"
    e1 = read(out, "e1")
    mlp = e1["mlp"]
    assert (mlp["runs"], mlp["drivers"], mlp["rmse_s"]) == (1, 2, pytest.approx(2.4))  # fold 0: drivers d0, d1
    assert (mlp["unstable"], mlp["unstable_low"], mlp["unstable_high"]) == pytest.approx((0.6,) * 3)
    assert mlp["max_gain_median"] == pytest.approx(1.1) and mlp["complete"] and mlp["h1_1"] == "holds"
    assert lines[0].startswith("TABLE e1: 4 rows from 4/4 runs, 0 missing; H1.1 does not hold (mlp holds, ")
    assert json.loads((out / "chosen_weights.json").read_text(encoding="utf-8"))["gru"]["weight"] == 1.0
    assert read(out, "e2")["gru"]["h1_2"] == "refuted" and read(out, "e2")["mlp"]["rmse_change_pairs"] == 2
    pooled = read(out, "e5")["openacc_human", "pooled", "chosen"]  # a reduction of 42.5 %, but one pair per model
    assert pooled["reduction"] == pytest.approx(1.0 - 0.345 / 0.6) and pooled["reduction_pairs"] == 2
    assert pooled["h1_3"] == "confirmed"
    human_mlp = read(out, "verdicts")["H1.3", "openacc_human/mlp"]
    assert human_mlp["verdict"] == "open" and human_mlp["basis"].endswith(
        "over 1 pairs (collision-free prefix); whole curves (D107) +50.0 % [+50.0 %, +50.0 %] over 1 pairs; collided "
        "profiles 0/0 without, 0/0 with penalty; open: fewer than 2 pairs")  # fmt: skip


def curve(flag: int, offset: float, collide_at: int | None = None, n: int = 8) -> dict:
    """A record of the platoon test with its curves (D109): the empirical curve 1.0 at the leader and four
    followers, the model ``offset`` above it at every follower; ``collide_at``: the first follower whose gap
    reaches 0 (every one behind it as well). The growth error of D107 in the record is ``2 offset`` (None in the
    columns after a collision); the prefix growth error is ``offset`` with at least three followers ahead of
    the collision."""
    gaps = [None] + [(-1.0 if collide_at is not None and k >= collide_at else 2.0) for k in range(1, n)]
    return {"acc_flag": flag, "empirical_std": [1.0] * 5, "speed_std": [1.0] + [1.0 + offset] * (n - 1),
            "min_gap": gaps, "collided": collide_at is not None, "growth_error": 2.0 * offset, "start": "equilibrium"}  # fmt: skip


def platoon_curves(profiles: dict[str, dict], hysteresis: float | None = 100.0) -> dict:
    """platoon.json with per-profile records and a pulse; the stored summary (the rule before M6) must not be used."""
    records = {**profiles, "pulse": {"collided": hysteresis is None, "speed_std": [1.0] * 8, "hysteresis_area": hysteresis,
                                     "min_gap": [None] + [1.0] * 7}}  # fmt: skip
    return {"profiles": records, "summary": {"growth_error_human": 9.9, "hysteresis_area_pulse": 1.0}}


def test_h1_3_on_the_collision_free_prefix(tmp_path):
    """D109: the growth error of a profile on the followers ahead of its first collided position (at least three),
    the whole-curve growth error of D107 (none after a collision) beside it; comparisons over the pairs of runs in
    which both have a value; the first collided position and the collision-free share per run."""
    root = build_tree(tmp_path / "runs")
    human = {  # E5, human view, GRU: (fold, penalised) -> its two human profiles; the ACC profile never counts here
        (0, False): {"h1.csv": curve(0, 0.6, 4), "h2.csv": curve(0, 0.9, 2)},  # both collide: prefixes of 3 and 1
        (1, False): {"h1.csv": curve(0, 0.6), "h2.csv": curve(0, 0.8, 5)},
        (0, True): {"h1.csv": curve(0, 0.3), "h2.csv": curve(0, 0.5)},
        (1, True): {"h1.csv": curve(0, 0.3), "h2.csv": curve(0, 0.3)},
    }  # fmt: skip
    for (fold, penalised), profiles in human.items():
        experiment = "e5_gain_w1" if penalised else "e5"
        run = root / experiment / "openacc_human" / "gru" / f"driver_fold{fold}_seed0"
        write_json(run / "platoon.json", platoon_curves({**profiles, "a1.csv": curve(1, 0.2)}))
    e1_run = root / "e1" / DATA / "mlp" / "driver_fold0_seed0"  # E2: one E1 run with a collision, no hysteresis
    write_json(e1_run / "platoon.json", platoon_curves({"h1.csv": curve(0, 0.4), "a1.csv": curve(1, 9.0, 3)}, None))
    make_tables(config(root, verdicts={"h1_3_min_pairs": 2}))
    out = root / "_tables" / "m4"
    assert (out / "missing.txt").read_text(encoding="utf-8") == ""  # no growth error by rule is no missing value
    e5 = read(out, "e5")
    none, chosen = e5["openacc_human", "gru", "none"], e5["openacc_human", "gru", "chosen"]
    assert none["growth_error"] == pytest.approx(0.65) and none["growth_runs"] == 2  # folds: 0.6, (0.6 + 0.8) / 2
    assert none["growth_error_full"] == pytest.approx(1.2) and none["growth_runs_full"] == 1  # fold 0: all collided
    assert (none["collided"], none["profiles"], chosen["collided"], chosen["profiles"]) == (3, 4, 0, 4)
    assert none["first_collided"] == pytest.approx(((4 + 2) / 2 + (8 + 5) / 2) / 2)  # 8: one past vehicle 7
    assert none["collision_free"] == pytest.approx(0.25) and chosen["first_collided"] == pytest.approx(8.0)
    assert chosen["growth_error"] == pytest.approx(0.35) and chosen["reduction_pairs"] == 2
    assert chosen["reduction"] == pytest.approx(1.0 - 0.35 / 0.65) and chosen["h1_3"] == "confirmed"
    assert chosen["reduction_low"] > 0.3 and none["complete"]
    assert chosen["reduction_full"] == pytest.approx(0.5) and chosen["reduction_full_pairs"] == 1  # fold 1 only
    basis = read(out, "verdicts")["H1.3", "openacc_human/gru"]["basis"]
    assert basis.startswith("reduction +46.2 % [") and basis.endswith(
        "over 2 pairs (collision-free prefix); whole curves (D107) +50.0 % [+50.0 %, +50.0 %] over 1 pairs; collided "
        "profiles 3/4 without, 0/4 with penalty; first collided position 4.8 without, 8.0 with penalty")  # fmt: skip
    pooled = e5["openacc_human", "pooled", "chosen"]  # MLP 2 pairs + GRU 2 pairs
    assert pooled["reduction_pairs"] == 4 and pooled["reduction_full_pairs"] == 3
    assert (pooled["collided_without"], pooled["profiles_without"]) == (3, 4)
    e2 = read(out, "e2")["mlp"]  # the collided ACC profile has two followers ahead of its collision: left out
    assert e2["growth_error_e1"] == pytest.approx(0.4) and e2["growth_error_pairs"] == 4
    assert e2["growth_error_full_e1"] == pytest.approx((0.8 + 3 * 0.4) / 4)  # D107: the stored 2 x 0.4 of h1
    assert e2["growth_error_full_change"] == pytest.approx(0.3 / 0.5 - 1.0)
    assert (e2["growth_collided_e1"], e2["growth_profiles_e1"], e2["growth_collided_e2"]) == (1, 2, 0)
    assert (e2["hysteresis_pairs"], e2["hysteresis_collided_e1"], e2["hysteresis_collided_e2"]) == (3, 1, 0)
    assert e2["hysteresis_e1"] == pytest.approx(100.0) and e2["growth_error_change"] == pytest.approx(-0.25)


def test_configured_weight_and_a_subset_of_tables(tmp_path):
    root = build_tree(tmp_path / "runs")
    lines = make_tables(config(root, tables=("e2",), chosen_weights={"gru": 0.1}))
    out = root / "_tables" / "m4"
    # E1 and E2 runs of three architectures; the GRU with weight 0.1 has seed 0 only
    assert len(lines) == 1 and lines[0].startswith("TABLE e2: 3 rows from 22/24 runs, 2 missing")
    e2 = read(out, "e2")
    assert e2["gru"]["weight"] == 0.1 and e2["gru"]["runs_e2"] == 2 and not e2["gru"]["complete"]
    assert not (out / "e1.md").exists() and not (out / "chosen_weights.json").exists()
    missing = (out / "missing.txt").read_text(encoding="utf-8")
    assert "[e2] e2_gain_w0.1/follownet_highd/gru/driver_fold0_seed1: run missing" in missing


def test_the_bases_of_the_verdicts_are_configured(tmp_path):
    """Swapped bases swap the verdicts: H1.1 on the band share holds (MLP and GRU), on the equilibria it does
    not; H1.5 on the equilibria found fails (certified hybrid 0.5, MLP 0.1)."""
    root = build_tree(tmp_path / "runs")
    bases = {"h1_1": "not_stable", "h1_1_information": "unstable_eq", "h1_5": "unstable_eq"}
    lines = make_tables(config(root, tables=("e1", "e4"), bases=bases))
    expected = "H1.1 holds (mlp holds, gru holds, lstm does not hold; on unstable among equilibria: does not hold)"
    assert expected in lines[0]
    assert "H1.5 does not hold" in lines[1]
    e1 = read(root / "_tables" / "m4", "e1")
    assert (e1["gru"]["h1_1"], e1["gru"]["h1_1_information"]) == ("holds", "does not hold")
    with pytest.raises(ValueError, match="bases.h1_2"):
        config(root, bases={"h1_2": "unstable_eq_sign"})  # e2 has no share of the sign criterion
    with pytest.raises(ValueError, match="unknown keys"):
        TablesConfig.from_mapping({"basis": {}}, root, root / "out")


def test_make_tables_script(tmp_path):
    root = build_tree(tmp_path / "runs")
    design = [
        "folds=[0,1]", "seeds=[0,1]", "learned=[mlp,gru,lstm]", "baselines=[idm]", "weights=[0.1,1.0]",
        "targets=[ngsim_i80]", "e5_models=[idm,mlp,gru]", "e5_penalised=[mlp,gru]", "n_resamples=200",
        "lowfreq_architectures=[gru,lstm]", "monotone_architectures=[mlp]",
    ]  # fmt: skip
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "make_tables.py"), *design,
        f"paths.runs_root='{root.as_posix()}'", f"hydra.run.dir='{(tmp_path / 'outputs').as_posix()}'",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    tables = ("e1", "e2_sweep", "e2", "e2_lowfreq", "e3", "e4", "e5")
    assert [line.split(":")[0] for line in lines] == [f"TABLE {name}" for name in (*tables, "e2_monotone", "e2_horizon")]
    out = root / "_tables" / "m4"
    names = {"missing.txt", "chosen_weights.json",
             *(f"{n}.{s}" for n in (*tables, "e2_horizon", "verdicts") for s in ("md", "csv"))}  # fmt: skip
    assert {p.name for p in out.iterdir()} == names
    assert {p.name for p in (root / "_tables" / "m8").iterdir()} == {"e2_monotone.md", "e2_monotone.csv"}  # M8
    mlp = read(out, "e1")["mlp"]
    assert mlp["rmse_s"] == pytest.approx(2.4) and math.isfinite(mlp["rmse_s_low"])
