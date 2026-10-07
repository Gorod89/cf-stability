"""Collection of run results (cf_stability/eval/collect.py, scripts/collect_results.py) on a hand-made runs tree."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cf_stability.eval.collect import (
    BASE_COLUMNS, collect_run, collect_runs, first_collided_position, prefix_growth_error, run_directories,
)  # fmt: skip
from cf_stability.utils import REPO_ROOT, write_json

SUMMARY = {"n_events": 10, "collision_rate": 0.1, "open_loop_rmse_a": 0.05, "rmse_s_mean": 2.5, "rmse_s_median": 2.0,
           "rmse_s_pooled": 3.0, "rmse_v_mean": 0.5, "rmse_v_median": 0.4, "rmse_v_pooled": 0.6}  # fmt: skip


def metrics(kind: str, weight: float, **extra) -> dict:
    return {
        "config_hash": "abc123", "config": {"init_from": None, "train": {"penalty": {"kind": "none", "weight": 1.0}}},
        "train_config": {"penalty": {"kind": kind, "weight": weight}}, "best_epoch": 7,
        "training": {"epochs": 12, "stop_reason": "early_stopping"}, "wall_time_s": 600.0,
        "n_trainable_parameters": 4545, "val": {**SUMMARY, "rmse_s_mean": 2.4}, "test": SUMMARY, **extra,
    }  # fmt: skip


def shares(**values) -> dict:
    keys = ("stable", "unstable", "outside", "none", "indifferent", "undefined")
    return {key: values.get(key, 0.0) for key in keys}


STABILITY = {
    "audit": {
        "summary": {
            "band_numerical": {"all": shares(stable=0.5, unstable=0.5), "support": shares(stable=0.75, outside=0.25)},
            "band_sign": {"all": shares(unstable=1.0), "support": shares(stable=0.5, unstable=0.5)},
            "grid_numerical": {"all": shares(stable=0.4, none=0.6), "support": shares(stable=0.9, none=0.1)},
            "grid_sign": {"all": shares(stable=0.4, none=0.6), "support": shares(unstable=0.9, none=0.1)},
            "max_gain": 1.25, "max_gain_omega": 0.1, "max_gain_v": 18.0,
            "agreement_gain": {"all": 0.9, "support": 1.0}, "agreement_sign": {"all": 0.6, "support": 0.5},
            "share_unstable_numerical": {"all": 0.3, "support": 0.35},
            "share_unstable_sign": {"all": 0.7, "support": 0.65},
            "n_grid": 26, "n_equilibria": 20, "n_multiple": 1, "n_none": 6, "n_indifferent": 0, "n_usable": 20,
            "n_in_support": 22, "n_band": {"all": 20, "support": 18},
        }
    }
}  # fmt: skip
PLATOON = {
    "summary": {
        "growth_error": {"ZalaZone/handling_part30.csv": 0.2, "ZalaZone/handling_part32.csv": 0.4},
        "growth_error_mean": 0.3, "growth_error_human": 0.2, "growth_error_acc": 0.4, "n_profiles": 3, "n_collided": 1,
        "n_start_equilibrium": 2,
        "std_ratio": {"ZalaZone/handling_part30.csv": 1.5, "ZalaZone/handling_part32.csv": None, "pulse": 0.8},
        "hysteresis_area_pulse": 42.0,
    }
}  # fmt: skip
TRANSFER = {"targets": {"ngsim_i80": {"n_events": 50, "full": {"rmse_s_mean": 6.0}, "horizon": {"rmse_s_mean": 3.0},
                                      "relative_degradation": 0.2}}}  # fmt: skip
CERTIFICATE = {
    "model": "residual_idm", "applicable": True,
    "core": {"params": {"v0": 30.0}, "margin_min": 0.21, "margin_max": 0.5},
    "residual": {"r_max": 1.0, "lipschitz": 0.05, "product": 0.05, "layer_norms": [1.0, 1.0]},
    "a_priori": {"holds": True, "n_hold": 26, "n_grid": 26, "guaranteed_margin_min": 0.01},
    "at_equilibria": {"holds": False, "n_hold": 19, "n_equilibria": 20, "guaranteed_margin_min": -0.01},
    "per_speed": [{"v": 5.0}],
}  # fmt: skip


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    full = root / "e2_jacobian_w0.1" / "follownet_highd" / "mlp" / "driver_fold0_seed0"
    write_json(full / "metrics.json", metrics("jacobian", 0.1))
    write_json(full / "stability.json", STABILITY)
    write_json(full / "platoon.json", PLATOON)
    write_json(full / "transfer.json", TRANSFER)
    write_json(full / "certificate.json", {"model": "mlp", "applicable": False})
    certified = root / "e2_jacobian_w0.1" / "follownet_highd" / "residual_idm" / "driver_fold1_seed2"
    budget = {"admissible": 0.06, "safety": 0.9, "r_max": 1.0, "lipschitz": 0.054}
    write_json(certified / "metrics.json", metrics("jacobian", 0.1, init_from="runs/x", certificate_budget=budget))
    write_json(certified / "stability.json", {"error": "RuntimeError: boom\nsecond line", "audit": None})
    write_json(certified / "certificate.json", CERTIFICATE)
    audit_only = root / "e2_jacobian_w10" / "follownet_highd" / "mlp" / "driver_fold0_seed0"
    write_json(audit_only / "stability.json", STABILITY)
    corrupt = root / "e1" / "follownet_highd" / "mlp" / "site_fold3_seed0"
    corrupt.mkdir(parents=True)
    (corrupt / "metrics.json").write_text("{cut short", encoding="utf-8")
    logs, notes = root / "e2_jacobian_w0.1" / "_logs" / "a" / "driver_fold0_seed0", corrupt.parent / "notes"
    for other in (logs, notes):
        other.mkdir(parents=True)  # a queue's logs and a directory that is no run
    return root


def test_run_directories(runs):
    found = run_directories(runs, ["e2_*", "e1", "e2_jacobian_w0.1", "missing"])
    assert [(e, p.relative_to(runs).as_posix()) for e, p in found] == [
        ("e2_jacobian_w0.1", "e2_jacobian_w0.1/follownet_highd/mlp/driver_fold0_seed0"),
        ("e2_jacobian_w0.1", "e2_jacobian_w0.1/follownet_highd/residual_idm/driver_fold1_seed2"),
        ("e2_jacobian_w10", "e2_jacobian_w10/follownet_highd/mlp/driver_fold0_seed0"),
        ("e1", "e1/follownet_highd/mlp/site_fold3_seed0"),
    ]  # every run once, logs and other directories left out  # fmt: skip


def test_collect_runs(runs):
    table = collect_runs(runs, ["e2_*", "e1"])
    assert list(table.columns[: len(BASE_COLUMNS)]) == BASE_COLUMNS
    assert list(table["experiment"]) == ["e1", "e2_jacobian_w0.1", "e2_jacobian_w0.1", "e2_jacobian_w10"]
    corrupt, full, certified, audit_only = (row for _, row in table.iterrows())

    def values(row: pd.Series, *columns: str) -> tuple:
        return tuple(row[c] for c in columns)

    assert values(full, "data", "model", "split", "fold", "seed") == ("follownet_highd", "mlp", "driver", 0, 0)
    assert values(full, "penalty_kind", "penalty_weight", "epochs", "best_epoch") == ("jacobian", 0.1, 12, 7)
    assert values(full, "val_rmse_s_mean", "test_rmse_s_mean", "test_collision_rate") == (2.4, 2.5, 0.1)
    assert values(full, "band_numerical_stable", "band_numerical_outside", "band_sign_unstable") == (0.75, 0.25, 0.5)
    assert values(full, "grid_numerical_stable", "grid_sign_unstable", "grid_sign_none") == (0.9, 0.9, 0.1)
    assert values(full, "max_gain", "agreement_gain", "agreement_sign") == (1.25, 1.0, 0.5)
    assert values(full, "share_unstable_numerical", "share_unstable_sign") == (0.35, 0.65)  # the support parts
    assert values(full, "n_grid", "n_band_all", "n_band_support", "n_in_support") == (26, 20, 18, 22)
    platoon = ("growth_error_mean", "growth_error_human", "n_collided", "n_start_equilibrium", "std_ratio_pulse")
    assert values(full, *(f"platoon_{key}" for key in platoon)) == (0.3, 0.2, 1, 2, 0.8)
    assert full["platoon_growth_error_ZalaZone_handling_part32"] == 0.4
    assert full["platoon_hysteresis_area_pulse"] == 42.0 and pd.isna(full["platoon_std_ratio_ZalaZone_handling_part32"])
    transfer = ("transfer_ngsim_i80_full_rmse_s_mean", "transfer_ngsim_i80_horizon_rmse_s_mean")
    assert values(full, *transfer, "transfer_ngsim_i80_relative_degradation") == (6.0, 3.0, 0.2)
    assert full["certificate_applicable"] == False  # noqa: E712
    assert pd.isna(full["init_from"]) and pd.isna(full["certificate_a_priori_holds"]) and pd.isna(full["audit_error"])

    assert certified["audit_error"] == "RuntimeError: boom" and pd.isna(certified["band_numerical_stable"])
    assert certified["init_from"] == "runs/x" and (certified["fold"], certified["seed"]) == (1, 2)
    assert (certified["budget_admissible"], certified["budget_lipschitz"]) == (0.06, 0.054)
    assert certified["certificate_applicable"] == True and certified["certificate_a_priori_holds"] == True  # noqa: E712
    assert certified["certificate_residual_product"] == 0.05 and certified["certificate_core_params_v0"] == 30.0
    assert certified["certificate_at_equilibria_n_hold"] == 19 and pd.isna(certified["platoon_growth_error_mean"])
    assert "certificate_residual_layer_norms" not in table.columns and "certificate_per_speed" not in table.columns

    assert pd.isna(audit_only["config_hash"]) and pd.isna(audit_only["test_rmse_s_mean"])
    assert audit_only["max_gain"] == 1.25
    assert corrupt["metrics_error"].startswith("unreadable") and corrupt["split"] == "site" and corrupt["fold"] == 3
    assert pd.isna(corrupt["epochs"]) and pd.isna(corrupt["max_gain"])

    empty = collect_runs(runs, ["missing*"])
    assert empty.empty and list(empty.columns) == BASE_COLUMNS
    alone = collect_run(runs / "e2_jacobian_w10" / "follownet_highd" / "mlp" / "driver_fold0_seed0")
    assert alone["experiment"] == "e2_jacobian_w10" and alone["max_gain"] == 1.25
    unpenalised = runs / "e1" / "follownet_highd" / "idm" / "driver_fold0_seed0"
    write_json(unpenalised / "metrics.json", metrics("none", 1.0))
    row = collect_run(unpenalised)
    assert row["penalty_kind"] == "none" and row["penalty_weight"] is None  # no penalty, no weight


def test_platoon_columns_are_rebuilt_from_the_profiles(tmp_path):
    """A platoon.json written before the rule of M6 holds the growth error of a collided platoon in its summary;
    the columns come from the per-profile records: the collided profile has no growth error and no std ratio."""
    run = tmp_path / "e5" / "openacc_human" / "gru" / "driver_fold0_seed0"
    profile = {"speed_std": [1.0, 2.0, 4.0], "start": "equilibrium"}
    write_json(run / "platoon.json", {
        "profiles": {
            "Vicolungo/a.csv": {**profile, "acc_flag": 0, "growth_error": 0.2, "collided": False},
            "Vicolungo/b.csv": {**profile, "acc_flag": 0, "growth_error": 19.0, "collided": True},
            "ZalaZone/c.csv": {**profile, "acc_flag": 1, "growth_error": 0.4, "collided": False},
            "pulse": {**profile, "collided": True, "hysteresis_area": 12.0},
        },
        "summary": {"growth_error_mean": 6.53, "growth_error_human": 9.6, "n_collided": 2},  # the rule before M6
    })  # fmt: skip
    row = collect_run(run)
    assert row["platoon_growth_error_human"] == 0.2 and row["platoon_growth_error_mean"] == pytest.approx(0.3)
    assert row["platoon_growth_error_acc"] == 0.4 and row["platoon_growth_error_Vicolungo_b"] is None
    assert (row["platoon_growth_collided_human"], row["platoon_growth_profiles_human"]) == (1, 2)
    kinds = ("growth_collided_mean", "growth_profiles_mean", "growth_collided_acc", "growth_profiles_acc")
    assert tuple(row[f"platoon_{key}"] for key in kinds) == (1, 3, 0, 1)
    assert (row["platoon_n_collided"], row["platoon_n_profiles"], row["platoon_n_start_equilibrium"]) == (2, 4, 4)
    assert row["platoon_std_ratio_Vicolungo_a"] == 2.0 and row["platoon_std_ratio_pulse"] is None
    assert row["platoon_hysteresis_area_pulse"] == 12.0
    write_json(run / "platoon.json", {"profiles": {"pulse": {"collided": False}}})  # no speed_std: unreadable
    broken = collect_run(run)
    assert broken["platoon_error"].startswith("unreadable profiles") and "platoon_growth_error_mean" not in broken


def record(collide_at: int | None, *, empirical=(1.0, 1.0, 1.2, 1.4, 1.6, 1.8), n: int = 8, flag: int = 0,
           first_in_time: int | None = None) -> dict:  # fmt: skip
    """A platoon record of D109: model curve 0.5 above the data at every follower (the leader shares the data),
    ``collide_at``: the follower whose gap reaches 0 (and every one behind it, as in the platoon test);
    ``first_in_time``: the collision_vehicle of the file (the first collision in time, not in the platoon)."""
    model = [empirical[0]] + [(1.0 if v is None else v) + 0.5 for v in empirical[1:]] + [9.0] * (n - len(empirical))
    gaps = [None] + [(-1.0 if collide_at is not None and k >= collide_at else 3.0) for k in range(1, n)]
    collided = collide_at is not None
    return {"speed_std": model, "min_gap": gaps, "empirical_std": list(empirical), "acc_flag": flag,
            "collided": collided, "collision_vehicle": (first_in_time or collide_at) if collided else None,
            "growth_error": None if collided else 0.123}  # fmt: skip


def test_prefix_growth_error_of_a_profile():
    """D109: the followers 1..min(P, c - 1) of the empirical curve, at least three of them; RMSE over them divided
    by the largest empirical value among them; the first collided position from min_gap."""
    free = record(None)
    assert first_collided_position(free) == 8  # one past the last vehicle (vehicles 0..7)
    error, positions = prefix_growth_error(free)  # followers 1..5, all 0.5 above the data, largest value 1.8
    assert positions == 5 and error == pytest.approx(0.5 / 1.8)
    late = record(4, first_in_time=6)  # vehicle 6 collides first in time; vehicle 4 is the first in the platoon
    assert first_collided_position(late) == 4
    error, positions = prefix_growth_error(late)  # followers 1..3, largest 1.4
    assert positions == 3 and error == pytest.approx(0.5 / 1.4)
    assert prefix_growth_error(record(3)) == (None, 2)  # two collision-free followers: no growth error
    assert prefix_growth_error(record(None), min_positions=6) == (None, 5)
    gaps = record(None, empirical=(1.0, 1.0, None, 1.4, 1.6, 2.0))  # a vehicle without speed in the data
    error, positions = prefix_growth_error(gaps)
    assert positions == 4 and error == pytest.approx(0.5 / 2.0)
    unknown = {**record(4), "min_gap": None}  # a collided record without min_gap: the position is unknown
    assert first_collided_position(unknown) is None and prefix_growth_error(unknown) == (None, 0)
    assert first_collided_position({**record(None), "min_gap": None}) == 8
    assert prefix_growth_error({"speed_std": [1.0, 2.0], "collided": False}) == (None, 0)  # no empirical curve


def test_prefix_columns_of_a_run(tmp_path):
    run = tmp_path / "e5" / "openacc_human" / "gru" / "driver_fold0_seed0"
    write_json(run / "platoon.json", {"profiles": {
        "Vicolungo/a.csv": record(None), "Vicolungo/b.csv": record(4), "Vicolungo/c.csv": record(2),
        "ZalaZone/d.csv": record(5, flag=1),
        "pulse": {"speed_std": [1.0] * 8, "min_gap": [None] + [1.0] * 7, "collided": False, "hysteresis_area": 1.0},
    }})  # fmt: skip
    row = collect_run(run)
    human = [0.5 / 1.8, 0.5 / 1.4]  # a: five followers, b: three, c: one (none)
    assert row["platoon_prefix_growth_error_human"] == pytest.approx(sum(human) / 2)
    assert row["platoon_prefix_profiles_human"] == 2 and row["platoon_prefix_positions_Vicolungo_c"] == 1
    assert row["platoon_prefix_growth_error_acc"] == pytest.approx(0.5 / 1.6)  # followers 1..4
    assert row["platoon_prefix_growth_error_mean"] == pytest.approx((sum(human) + 0.5 / 1.6) / 3)
    assert row["platoon_first_collided_human"] == pytest.approx((8 + 4 + 2) / 3)  # the pulse is no OpenACC profile
    assert row["platoon_first_collided_mean"] == pytest.approx((8 + 4 + 2 + 5) / 4)
    assert row["platoon_collision_free_human"] == pytest.approx(1 / 3) and row["platoon_collision_free_acc"] == 0.0
    assert row["platoon_growth_error_human"] == 0.123 and row["platoon_growth_collided_human"] == 2  # D107 stays
    assert row["platoon_prefix_growth_error_Vicolungo_c"] is None and row["platoon_first_collided_ZalaZone_d"] == 5
    write_json(run / "platoon.json", {"summary": {"growth_error_human": 0.4, "growth_error_mean": 0.3}})
    old = collect_run(run)  # no records: the stored growth errors stand for the prefix ones, positions unknown
    assert old["platoon_prefix_growth_error_human"] == 0.4 and old["platoon_prefix_growth_error_mean"] == 0.3
    assert pd.isna(old["platoon_first_collided_human"]) and pd.isna(old["platoon_prefix_profiles_human"])


def test_collect_results_script(runs, tmp_path):
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "collect_results.py"), "experiments=[e2_*,e1]", "name=table",
        f"paths.runs_root='{runs.as_posix()}'", f"hydra.run.dir='{(tmp_path / 'outputs').as_posix()}'",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("TABLE table: 4 runs of 3 experiments")
    expected = collect_runs(runs, ["e2_*", "e1"])
    for suffix, read in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        written = read(runs / "_tables" / f"table{suffix}")
        assert list(written.columns) == list(expected.columns) and len(written) == 4
        for column in ("test_rmse_s_mean", "band_numerical_stable", "platoon_growth_error_mean", "budget_admissible"):
            a, b = written[column].astype(float).tolist(), expected[column].astype(float).tolist()
            assert all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b)), column
    assert collect_runs(runs, ["_tables"]).empty  # the tables are no experiment
