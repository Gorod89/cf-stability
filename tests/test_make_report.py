"""Report of the project (cf_stability/eval/report.py, cf_stability/eval/figures.py, scripts/make_report.py).

The hand-made tree holds the CSV tables of M4 and M5 as scripts/make_tables.py and scripts/corridor_metrics.py
write them (a few rows each, the corridor column and the tables of M7 included), chosen_weights.json and the
run files of the figures: E1 and E2 runs with gain curves, the existence arm, E5 platoon files, a corridor with
a ground truth and two laws, and docs with a decisions log and two milestone reports. Checked: every table in
CSV and LaTeX (valid: as many cells as columns per row, rows ending in ``\\\\``, special characters escaped),
every figure in PNG and PDF, report.md with every section and its numbers from the tables, the tables of M7 in
their sections, notes for rows without runs and for a corridor without ground truth, the manifest, notes
instead of exceptions for missing inputs, the parsing of the docs, the refresh commands and the script as a
child process.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cf_stability.eval import report as report_module
from cf_stability.eval.report import (
    FIGURES, M4_TABLES, M5_TABLES, NEW_TABLES, SECTIONS, ReportConfig, ReportMaker, decision_entries, first_sentence,
    make_report, milestone_commands, profile_label, std_before_collision,
)  # fmt: skip
from cf_stability.utils import REPO_ROOT, read_json, write_json
from test_corridor_macro import constant_paths, sample

DATA = "follownet_highd"
FOLDS = (0, 1)
OMEGA = list(np.geomspace(0.02, 2.0, 5))
TABLES = (*M4_TABLES, *M5_TABLES, *NEW_TABLES)
M8_TEST_TABLES = ("band_width", "power")  # the tables of M8 of the tree: one of another work package, one known (D122)
ALL_TABLES = (*TABLES, *M8_TEST_TABLES)
PROFILES = ("Vicolungo/JRC-VC_280219_part2.csv", "ZalaZone/handling_part13.csv", "pulse")
CHOSEN = {"mlp": {"kind": "jacobian", "weight": 0.1, "experiment": "e2_jacobian_w0.1"},
          "gru": {"kind": "gain", "weight": 1.0, "experiment": "e2_gain_w1"}}  # fmt: skip


def ci(value: float, width: float = 0.1) -> dict[str, float]:
    return {"": value, "_low": value - width, "_high": value + width}


def frame(rows: list[dict]) -> pd.DataFrame:
    """Rows whose values may be ``ci`` dicts (a value with its interval) as flat columns."""
    flat = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, dict):
                out.update({f"{key}{end}": v for end, v in value.items()})
            else:
                out[key] = value
        flat.append(out)
    return pd.DataFrame(flat)


# ------------------------------------------------------------------------------------------ the tables


def write_m4_tables(folder: Path) -> None:
    folder.mkdir(parents=True)
    e1 = []
    for model, rmse, stable, unstable, outside, unstable_eq in (
        ("idm", 3.0, 0.8, 0.15, 0.05, 0.15), ("mlp", 2.8, 0.1, 0.9, 0.0, 0.9), ("gru", 2.1, 0.1, 0.5, 0.3, 0.85),
        ("residual_idm", 2.7, 0.7, 0.3, 0.0, 0.3), ("persistence", 5.9, 0.0, 0.0, 0.0, 0.0),
    ):  # fmt: skip
        e1.append({"model": model, "runs": 4, "drivers": 2, "rmse_s": ci(rmse), "collision_rate": 0.0,
                   "rmse_vs_ref": ci(rmse / 3.0 - 1.0, 0.01), "rmse_vs_ref_p": 0.0001, "unstable_eq": ci(unstable_eq),
                   "stable": stable, "unstable": ci(unstable), "outside": outside, "none": 0.0,
                   "not_stable": ci(1 - stable), "max_gain_median": 1.2, "h1_1": "holds" if model == "mlp" else None})
    frame(e1).to_csv(folder / "e1.csv", index=False)
    sweep = [
        {"architecture": "mlp", "kind": "jacobian", "weight": w, "runs": 2, "val_rmse_s": 2.8 + w / 10,
         "test_rmse_s": 2.85 + w / 10, "stable": s, "feasible": 2, "chosen": w == 0.1}
        for w, s in ((0.01, 0.4), (0.1, 0.95), (1.0, 1.0))
    ] + [
        {"architecture": "gru", "kind": "gain", "weight": w, "runs": 2, "val_rmse_s": 2.2 + w / 10,
         "test_rmse_s": 2.25 + w / 10, "stable": s, "feasible": 0, "chosen": w == 1.0}
        for w, s in ((0.1, 0.2), (1.0, 0.25), (10.0, 0.25))
    ]  # fmt: skip  # GRU weights 1 and 10 nearly at one point: one shared label
    pd.DataFrame(sweep).to_csv(folder / "e2_sweep.csv", index=False)
    frame([{"architecture": a, "kind": k, "weight": w, "runs_e1": 4, "runs_e2": 4, "rmse_change": ci(0.05, 0.01),
            "rmse_change_p_holm": 0.02, "not_stable_e1": ci(0.9), "not_stable_e2": ci(0.1), "h1_2": "confirmed"}
           for a, k, w in (("mlp", "jacobian", 0.1), ("gru", "gain", 1.0))]).to_csv(folder / "e2.csv", index=False)
    frame([{"architecture": "mlp", "penalty": "none", "weight": None, "target": "ngsim_i80", "drivers": 2,
            "source_rmse_s": 2.8, "rmse_s_h": ci(4.0), "rmse_s_full": ci(10.0), "degradation": ci(1.5, 0.2),
            "degradation_change": ci(-0.1, 0.05), "runs": 2}]).to_csv(folder / "e3.csv", index=False)
    frame([{"model": "residual_idm", "variant": "certified", "data": "ngsim_i80", "runs": 4, "a_priori_holds": 4,
            "at_equilibria_holds": 4, "all_equilibria_certified": 4, "unstable": ci(0.0, 0.0),
            "rmse_s": ci(3.1)}]).to_csv(folder / "e4.csv", index=False)
    frame([{"view": "openacc_human", "model": "mlp", "penalty": "jacobian 0.1", "runs": 2, "rmse_s": ci(1.0),
            "first_collided": ci(12.5, 2.0), "collision_free": ci(0.4, 0.1), "growth_error": 0.2,
            "reduction": ci(0.3, 0.2), "reduction_p": 0.04, "h1_3": "open", "reduction_full_pairs": 1,
            "reduction_full": ci(0.1, 0.0)}]).to_csv(folder / "e5.csv", index=False)  # fmt: skip
    frame([{"architecture": "gru", "weight": w, "chosen": w == 1.0, "runs": 2 if w == 1.0 else 1, "rmse_s": ci(2.3),
            "rmse_change": ci(0.05, 0.01), "stable": 0.95, "not_stable": ci(0.05, 0.02), "unstable_eq": ci(0.04, 0.02),
            "max_gain_median": 1.01, "collided": 0, "profiles": 4, "h1_2": "confirmed" if w == 1.0 else None}
           for w in (0.1, 1.0)]).to_csv(folder / "e2_lowfreq.csv", index=False)  # fmt: skip
    pd.DataFrame([
        {"hypothesis": "H1.1", "unit": "mlp", "verdict": "holds", "complete": True,
         "basis": "unstable among equilibria 0.90 [0.80, 1.00] & RMSE vs idm -6.7 % (score_x ~ 1)"},
        {"hypothesis": "H1.2", "unit": "mlp", "verdict": "confirmed", "complete": True, "basis": "band share 0.05"},
        {"hypothesis": "H1.2 (combined)", "unit": "gru", "verdict": "confirmed", "complete": True,
         "basis": "weight 1 (stable >= 0.9: smallest validation RMSE)"},
        {"hypothesis": "H1.3", "unit": "openacc_human/mlp", "verdict": "open", "complete": False, "basis": "reduction"},
        {"hypothesis": "H1.5", "unit": "overall", "verdict": "holds", "complete": True, "basis": "certified"},
    ]).to_csv(folder / "verdicts.csv", index=False)  # fmt: skip
    write_json(folder / "chosen_weights.json", CHOSEN)


def write_m5_tables(folder: Path) -> None:
    folder.mkdir(parents=True)
    laws = [{"corridor": "SC", "law": "ground truth", "kind": "data", "scenario": "sc", "runs": None,
             "throughput_vph": 3000.0, "mean_speed": 11.0}]  # fmt: skip
    for law, error, dynamic in (("law_a", 0.1, 0.2), ("law_b", 0.4, 0.5)):
        laws.append({"corridor": "SC", "law": law, "kind": "idm", "scenario": None, "runs": 1,
                     "macro_error": ci(error, 0.02), "macro_error_dynamic": ci(dynamic, 0.03), "n_components": 8.0,
                     "collisions_per_1000_vkm": ci(0.0 if law == "law_a" else 2.5, 0.0),
                     "runs_with_collisions": 0 if law == "law_a" else 1, "inserted_share": 1.0,
                     "throughput_vph": ci(2900.0, 50.0), "mean_speed": ci(10.5, 0.2)})  # fmt: skip
    frame(laws).to_csv(folder / "laws.csv", index=False)
    components = []
    for law, sign in (("law_a", 1.0), ("law_b", -1.0)):
        row: dict = {"corridor": "SC", "law": law, "runs": 1}
        for key in ("anchored_throughput", "anchored_mean_speed", "anchored_queue_discharge_flow",
                    "anchored_travel_time", "dynamic_fd", "dynamic_wave_speed", "dynamic_waves",
                    "dynamic_wave_amplitude", "macro_error", "macro_error_dynamic"):  # fmt: skip
            row[key] = ci(sign * 0.12, 0.02)
        components.append(row)
    frame(components).to_csv(folder / "components.csv", index=False)
    frame([{"corridor": "SC", "law": law, "members": 2, "audited": 2, "unstable_eq": x, "not_stable": x, "kind": "idm",
            "runs": 1, "macro_error": ci(e, 0.02), "macro_error_dynamic": ci(d, 0.03), "collisions_per_1000_vkm": c,
            "collides": c > 0}
           for law, x, e, d, c in (("law_a", 0.0, 0.1, 0.2, 0.0), ("law_b", 0.8, 0.4, 0.5, 2.5))]).to_csv(
        folder / "instability.csv", index=False)  # fmt: skip
    pd.DataFrame([
        {"corridor": "SC", "error": error, "measure": "unstable among equilibria", "laws": "all laws",
         "method": method, "n": 2, "estimate": 0.9, "estimate_low": 0.1, "estimate_high": 1.0, "n_valid": 50,
         "p_permutation": 0.0004}
        for error in ("macro error", "macro error (dynamic)") for method in ("spearman", "pearson")
    ]).to_csv(folder / "instability_correlation.csv", index=False)  # fmt: skip
    frame([{"corridor": "SC", "metric": m, "header": h, "pairs": 1, "mean_reference": 3000.0, "mean_candidate": 2950.0,
            "relative_difference": ci(-0.017, 0.01), "wilcoxon_p": 0.5, "p_value": p, "equivalent": p < 0.05}
           for m, h, p in (("throughput_vph", "throughput (veh/h)", 0.01), ("mean_speed", "mean speed (m/s)", 0.3))
           ]).to_csv(folder / "tost.csv", index=False)  # fmt: skip
    frame([{"corridor": "SC", "network": "law_b", "metric": m, "header": m.replace("_", " "), "pairs": 1,
            "abs_error_network": 0.4, "abs_error_reference": 0.1, "degradation": ci(0.3, 0.05), "degraded": True,
            "strong": True, "h12_1": "confirmed", "complete": True}
           for m in ("throughput", "travel_time", "wave_speed")]).to_csv(folder / "h12_1.csv", index=False)  # fmt: skip
    frame([{"role": role, "experiment": e, "model": m, "runs": 2, "drivers": 2, "rmse_s": ci(r), "rmse_runs": r,
            "rmse_vs_reference": ci(change, 0.01) if role != "reference" else {"": None, "_low": None, "_high": None},
            "rmse_vs_reference_p": 0.5, "superior": change < 0 if role != "reference" else None}
           for role, e, m, r, change in (("reference", "e3_reference", "idm", 4.7, 0.0),
                                         ("candidate", "e4_stable_ft", "residual_idm", 4.9, 0.05))]).to_csv(
        folder / "h12_2.csv", index=False)  # fmt: skip
    pd.DataFrame([
        {"hypothesis": "H12.1", "unit": "law_b", "corridors": "SC", "verdict": "confirmed", "complete": True,
         "basis": "degradation vs law_a: throughput +0.300 [+0.250, +0.350] degraded by >= 0.15"},
        {"hypothesis": "H12.1", "unit": "overall", "corridors": "SC only", "verdict": "open", "complete": True,
         "basis": "confirmed for >= 2 of 3 networks on every corridor: SC: law_b confirmed"},
        {"hypothesis": "H12.2", "unit": "law_b", "corridors": "SC only", "verdict": "open", "complete": True,
         "basis": "fails: micro part"},
        {"hypothesis": "H12.3", "unit": "Spearman", "corridors": "SC", "verdict": "open", "complete": True,
         "basis": "r +0.900 [+0.100, +1.000] over 2 laws, p (permutation) 0.0004"},
        {"hypothesis": "H12.3", "unit": "overall", "corridors": "SC only", "verdict": "open", "complete": True,
         "basis": "Spearman: SC open"},
    ]).to_csv(folder / "verdicts.csv", index=False)  # fmt: skip
    frame([{"corridor": "SC", "r_max": r, "core": "certified", "runs_highd": 4, "a_priori_highd": 4, "runs_ngsim": 4,
            "a_priori_ngsim": 4, "unstable_eq_ngsim": ci(0.0, 0.0), "rmse_highd": ci(2.5), "rmse_ngsim": ci(3.6),
            "runs_corridor": 1, "macro_error": ci(0.1, 0.02), "macro_error_dynamic": ci(0.2, 0.03),
            "collisions_per_1000_vkm": 0.0} for r in (0.3, 0.5)]).to_csv(folder / "e4_rmax.csv", index=False)  # fmt: skip
    frame([{"variant": variant, "law": law, "runs": 2, "macro_error": ci(error, 0.01), "rank": rank,
            "ranking": ranking, "order_holds": ranking == "law_a < law_b", "collisions_per_1000_vkm": 0.0}
           for variant, ranking, laws_errors in (("baseline", "law_a < law_b", (("law_a", 0.1, 1), ("law_b", 0.4, 2))),
                                                 ("v1", "law_b < law_a", (("law_a", 0.5, 2), ("law_b", 0.3, 1))))
           for law, error, rank in laws_errors]).to_csv(folder / "sensitivity.csv", index=False)  # fmt: skip


# --------------------------------------------------------------------------------------- the run files


def write_run(root: Path, experiment: str, data: str, model: str, fold: int, *, rmse: float = 2.0,
              stable: float = 0.5, gain: float | None = None, platoon: dict | None = None) -> Path:  # fmt: skip
    """A run directory with metrics.json, stability.json (summary and the gain curves of three speeds in
    support when ``gain`` is given), test_events.parquet and optionally platoon.json."""
    run = root / experiment / data / model / f"driver_fold{fold}_seed0"
    run.mkdir(parents=True)
    write_json(run / "metrics.json", {"config_hash": f"{experiment}-{model}-{fold}", "test": {"rmse_s_mean": rmse}})
    equilibria = [] if gain is None else [
        {"in_support": True, "gain": [gain * (1.0 + 0.05 * k) / (1.0 + w) for w in OMEGA]} for k in range(3)
    ] + [{"in_support": False, "gain": [9.0] * len(OMEGA)}]  # fmt: skip
    write_json(run / "stability.json", {"audit": {"omega": OMEGA, "equilibria": equilibria, "summary": {
        "band_numerical": {"support": {"stable": stable, "unstable": 1.0 - stable, "outside": 0.0, "none": 0.0}},
        "share_unstable_numerical": {"support": 1.0 - stable}, "max_gain": 1.1,
    }}})  # fmt: skip
    pd.DataFrame({
        "event_id": ["d0|e", "d1|e"], "follower_id": ["d0", "d1"], "site": "a", "n_scored": 100,
        "rmse_s": [rmse, rmse + 1.0], "rmse_v": 0.5, "collided": False,
    }).to_parquet(run / "test_events.parquet", index=False)  # fmt: skip
    if platoon is not None:
        write_json(run / "platoon.json", platoon)
    return run


def platoon(scale: float, collide: bool = False) -> dict:
    """platoon.json of E5: three profiles, six platoon positions; ``collide``: the gap of position 3 reaches 0."""
    std = [scale * (1.0 + 0.2 * k) for k in range(6)]
    gaps = [None, 5.0, 5.0, -0.5 if collide else 5.0, 5.0, 5.0]
    common = {"speed_std": std, "min_gap": gaps, "collided": collide, "collision_vehicle": 3 if collide else None}
    return {"profiles": {
        PROFILES[0]: {**common, "acc_flag": 0, "empirical_std": [1.0, 1.1, 1.3]},
        PROFILES[1]: {**common, "acc_flag": 1, "empirical_std": [0.5, 0.6, 0.65]},
        PROFILES[2]: dict(common),
    }}  # fmt: skip


def write_runs(root: Path) -> None:
    for fold in FOLDS:
        for model, stable in (("idm", 0.8), ("mlp", 0.1), ("gru", 0.1)):
            gain = 1.3 if model != "idm" else 0.9
            write_run(root, "e1", DATA, model, fold, rmse=2.0 + stable, stable=stable, gain=gain)
        write_run(root, "e2_jacobian_w0.1", DATA, "mlp", fold, stable=0.95, gain=0.95)
        write_run(root, "e2_gain_w1", DATA, "gru", fold, stable=0.25, gain=1.05)
        write_run(root, "e2_existence", DATA, "gru", fold, rmse=2.2, stable=0.0,
                  platoon={"summary": {"n_collided": 2, "n_profiles": 6}})  # fmt: skip
        for view in ("openacc_acc", "openacc_human"):
            for model, scale in (("idm", 1.0), ("mlp", 1.5)):
                write_run(root, "e5", view, model, fold, platoon=platoon(scale, collide=model == "mlp" and fold == 1))
            write_run(root, "e5_jacobian_w0.1", view, "mlp", fold, platoon=platoon(1.2))


def write_corridor(root: Path) -> None:
    def save(directory: Path, names: tuple[str, str], arrays: tuple[dict, dict]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / names[0], **arrays[0])
        np.savez_compressed(directory / names[1], **arrays[1])

    def flow(speed_1: float, speed_2: float) -> tuple[dict, dict]:
        return sample(constant_paths(speed_1, 2.0, 1, 0.37, t_end=300.0)
                      + constant_paths(speed_2, 3.0, 2, 1.13, t_end=300.0))  # fmt: skip

    scenario = root / "scenarios" / "sc"
    save(scenario, ("ground_truth.npz", "vehicles_truth.npz"), flow(10.0, 15.0))
    write_json(scenario / "scenario.json", {"scenario": "sc", "geometry": {"x_in": 20, "x_out": 500},
                                            "window": [60, 300], "config_hash": "abc"})  # fmt: skip
    for law, speeds in (("law_a", (10.0, 15.0)), ("law_b", (8.0, 12.0))):
        run = root / "sc" / law / "seed0"
        save(run, ("trajectories.npz", "vehicles.npz"), flow(*speeds))
        write_json(run / "run.json", {"law": law})
        write_json(run / "macro.json", {"config_hash": "m1"})


def write_docs(docs: Path, configs: Path) -> None:
    docs.mkdir(parents=True)
    (docs / "decisions.md").write_text("\n".join([
        "# Decisions log", "", "## Environment", "", "| Id | Date | Decision | Reason |", "|---|---|---|---|",
        "| E1 | 2026-09-27 | Python 3.11. | Asked. |", "", "## Data (M1)", "", "| Id | Date | Decision | Reason |",
        "|---|---|---|---|", "| D1 | 2026-09-27 | Positions at the bumpers, e.g. the rear one. Second sentence. | x |",
        "| D2 | 2026-09-27 | **Deviation:** speeds as measured. | y |", "| D3 | 2026-09-28 | A model choice. | z |",
        "", "## Report (M6)", "", "| Id | Date | Decision | Reason |", "|---|---|---|---|",
        "| D4 | 2026-10-01 | Components reported. | w |", "",
    ]), encoding="utf-8")  # fmt: skip
    fence = "```"
    (docs / "m1_report.md").write_text(
        f"# Milestone M1 report: data\n\nD1 and D2.\n\n## 1. Commands run\n\n{fence}bash\npython scripts/a.py\n"
        f"{fence}\n\n{fence}bash\nnot this\n{fence}\n", encoding="utf-8")  # fmt: skip
    (docs / "m2_report.md").write_text(
        f"# Milestone M2 report: models\n\nD3, after D1.\n\n## 1. Commands run\n\n{fence}powershell\n"
        f"python scripts/b.py x=1\n{fence}\n", encoding="utf-8")  # fmt: skip
    configs.mkdir(parents=True)
    (configs / "make_tables.yaml").write_text(
        f"data: {DATA}\nfolds: [0, 1]\nseeds: [0]\nn_resamples: 50\nlevel: 0.95\nseed: 0\nreference: idm\n",
        encoding="utf-8")  # fmt: skip
    (configs / "corridor_metrics.yaml").write_text(
        "design:\n  scenarios: [sc]\n  corridors: {sc: SC}\n  laws: [law_a, law_b]\n  seeds: [0]\n  reference: law_a\n"
        "  candidate: law_b\n  margin: 0.1\n  alpha: 0.05\n  n_resamples: 50\n  level: 0.95\n"
        "  sensitivity_order: [law_a, law_b]\n", encoding="utf-8")  # fmt: skip


def write_m8(root: Path) -> None:
    """The supplement of M8: a table of another work package (any columns, with its Markdown), the power table of D122 and
    one figure with its caption."""
    folder = root / "_tables" / "m8"
    folder.mkdir(parents=True)
    frame([{"data": "follownet_highd", "speed": v, "rel_width": ci(0.5 + v / 100, 0.05), "n_drivers": 25,
            "dispersion": 1.5} for v in (10.0, 20.0)]).to_csv(folder / "band_width.csv", index=False)  # fmt: skip
    (folder / "band_width.md").write_text(
        "\n".join(["# Width of the spacing band", "", "- Relative width of the band per speed.", ""]), encoding="utf-8")
    frame([{"corridor": "SC", "metric": "macro_error", "header": "macro error", "law_a": "law_a", "law_b": "law_b",
            "pairs": 3, "mean_a": 0.18, "mean_b": 0.12, "sd_a": 0.014, "sd_b": 0.046, "sd_difference": 0.048,
            "n_one_scenario": 79, "n_per_scenario": 27, "power_design": 0.39, "mdd_relative": 0.166,
            "observed": ci(-0.316, 0.1), "sentence": "On SC, ten seeds detect 16.6 % (power 0.39 for 10 %)."}]).to_csv(
        folder / "power.csv", index=False)  # fmt: skip
    figures_dir = root / "_report" / "supplement" / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "band_width.png").write_bytes(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))
    (figures_dir / "band_width.pdf").write_bytes(b"%PDF-1.4")
    (figures_dir / "band_width.txt").write_text("Relative width of the spacing band against the speed.", encoding="utf-8")


def build_tree(base: Path) -> dict[str, Path]:
    root = base / "runs"
    write_m4_tables(root / "_tables" / "m4")
    write_m5_tables(root / "_tables" / "m5")
    write_m8(root)
    write_runs(root)
    write_corridor(root / "corridor")
    write_docs(base / "docs", base / "configs")
    return {"base": base, "root": root, "docs": base / "docs", "configs": base / "configs"}


def config(paths: dict[str, Path], out: Path, **changes) -> ReportConfig:
    root = paths["root"]
    values = dict(
        runs_root=root, out_dir=out, tables_m4=root / "_tables" / "m4", tables_m5=root / "_tables" / "m5",
        corridor_root=root / "corridor", docs_dir=paths["docs"], configs_dir=paths["configs"],
        gain_architectures=("idm", "mlp", "gru"), existence_architectures=("gru",), e5_models=("idm", "mlp"),
        e5_penalised=("mlp",), scenario="sc", figure_laws=("law_a", "law_b"), contour_lanes=(1, 2, 3), dpi=50,
        supplement_figures=root / "_report" / "supplement" / "figures", m8_tables=("band_width", "power"),
        supplement_expected=("band_width",),
    )  # fmt: skip
    return ReportConfig(**{**values, **changes})


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict[str, Path]:
    return build_tree(tmp_path_factory.mktemp("report"))


@pytest.fixture(scope="module")
def made(tree) -> tuple[ReportConfig, list[str]]:
    cfg = config(tree, tree["base"] / "out")
    return cfg, make_report(cfg, {"note": "test"})


# ------------------------------------------------------------------------------------------- LaTeX


def unescaped(text: str, char: str) -> int:
    """Occurrences of ``char`` not escaped by a backslash."""
    return len(re.findall(rf"(?<!\\){re.escape(char)}", text))


def check_latex(path: Path) -> None:
    """A table of latex_table: one tabular, as many cells as columns in every row (multicolumn spans
    counted), every row ending in ``\\\\``, balanced braces, no unescaped special character."""
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("%")]
    assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1, path.name
    assert r"\toprule" in text and r"\midrule" in text and r"\bottomrule" in text
    spec = re.search(r"\\begin\{tabular\}\{([lrc]+)\}", text)
    assert spec, path.name
    n_columns = len(spec.group(1))
    start = next(i for i, line in enumerate(lines) if line.startswith(r"\toprule")) + 1
    end = next(i for i, line in enumerate(lines) if line.startswith(r"\bottomrule"))
    rows = [line for line in lines[start:end] if not line.startswith((r"\midrule", r"\cmidrule"))]
    assert rows, path.name
    for row in rows:
        assert row.endswith(r" \\"), (path.name, row)
        cells = row[: -len(r" \\")].split(" & ")
        spans = [int(m.group(1)) if (m := re.match(r"\\multicolumn\{(\d+)\}", cell)) else 1 for cell in cells]
        assert sum(spans) == n_columns, (path.name, row)
        assert unescaped(row, "&") == len(cells) - 1, (path.name, row)
        for char in ("_", "%", "#", "~"):
            assert unescaped(row, char) == 0, (path.name, char, row)
    body = "\n".join(line for line in lines if not line.endswith("{%"))  # the resizebox line ends in a comment
    assert unescaped(body, "%") == 0, path.name
    assert unescaped(text, "{") == unescaped(text, "}"), path.name


# ------------------------------------------------------------------------------------------- tests


def test_every_table_in_both_formats(made):
    cfg, lines = made
    folder = cfg.out_dir / "tables"
    for name in ALL_TABLES:
        assert (folder / f"{name}.csv").exists() and (folder / f"{name}.tex").exists(), name
        check_latex(folder / f"{name}.tex")
    assert f"TABLES {len(ALL_TABLES)} (CSV, LaTeX)" in lines[0]
    # the CSV of an M4 or M5 table is its input with full precision
    pd.testing.assert_frame_equal(pd.read_csv(folder / "e1.csv"), pd.read_csv(cfg.tables_m4 / "e1.csv"))
    components = pd.read_csv(cfg.tables_m5 / "components.csv")
    pd.testing.assert_frame_equal(pd.read_csv(folder / "m5_components.csv"), components)
    laws = pd.read_csv(folder / "laws.csv")
    assert list(laws["law"]) == ["law_a", "law_b"]  # the ground truth is in laws_raw only
    assert "ground truth" in set(pd.read_csv(folder / "laws_raw.csv")["law"])
    dynamic = pd.read_csv(folder / "macro_error_dynamic_correlation.csv")
    assert len(dynamic) == 2 and set(dynamic["method"]) == {"spearman", "pearson"}


def test_latex_rounds_and_names_the_design(made):
    cfg, _ = made
    e1 = (cfg.out_dir / "tables" / "e1.tex").read_text(encoding="utf-8")
    assert r"\caption{E1 on follownet\_highd: 20 runs (2 folds x 1 seeds" in e1
    assert "95 \\% percentile bootstrap intervals, 50 resamples" in e1
    assert "RMSE s (m)" in e1 and r"\resizebox{\linewidth}{!}{%" in e1  # units in the heads, a wide table
    assert "2.80 [2.70, 2.90]" in e1 and "$-$6.7 \\% [$-$7.7 \\%, $-$5.7 \\%]" in e1 and "$<$0.001" in e1
    verdicts = (cfg.out_dir / "tables" / "verdicts.tex").read_text(encoding="utf-8")
    assert r"0.90 [0.80, 1.00] \& RMSE vs idm -6.7 \% (score\_x \textasciitilde{} 1)" in verdicts
    components = (cfg.out_dir / "tables" / "m5_components.tex").read_text(encoding="utf-8")
    assert r"\multicolumn{2}{c}{} & \multicolumn{4}{c}{anchored} & \multicolumn{4}{c}{dynamic} \\" in components
    assert r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}" in components  # the corridor and the law before the groups
    assert "wave speed (xcorr)" in components and "travel time (W1)" in components
    correlation = (cfg.out_dir / "tables" / "instability_correlation.tex").read_text(encoding="utf-8")
    assert "p (permutation)" in correlation and "$<$0.001" in correlation


def test_every_figure_in_both_formats(made):
    cfg, lines = made
    folder = cfg.out_dir / "figures"
    for name in FIGURES:
        assert (folder / f"{name}.png").read_bytes()[:4] == b"\x89PNG", name
        assert (folder / f"{name}.pdf").read_bytes()[:4] == b"%PDF", name
    assert f"FIGURES {len(FIGURES)} of {len(FIGURES)} (PNG, PDF)" in lines[1]


def test_report_has_every_section_and_numbers_from_the_tables(made):
    cfg, lines = made
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    positions = [text.index(f"## {section}\n") for section in SECTIONS]
    assert positions == sorted(positions)
    for name in FIGURES:
        assert f"](figures/{name}.png)" in text and f"(figures/{name}.pdf)" in text
    for name in ALL_TABLES:
        assert f"[tables/{name}.tex](tables/{name}.tex)" in text
    assert "- H1.1, mlp: **holds**;" in text and "- H1.3, openacc_human/mlp: **open** (incomplete)" in text
    assert "Chosen weights (D85, chosen_weights.json): gru gain 1, mlp jacobian 0.1." in text
    assert "TOST: SC: 1 of 2 metrics equivalent." in text
    assert "| e1 | follownet_highd | gru, idm, mlp | 0, 1 | 0 | 6 | 6 |" in text  # section 1: counts
    assert "| sc | 2 | 2 | 2 | 2 |" in text
    assert "(3 of 18 simulated platoons collided)" in text  # three profiles: the MLP of fold 1 in the matching view
    section_9 = text[text.index(f"## {SECTIONS[8]}"): text.index(f"## {SECTIONS[9]}")]
    assert "4 D-entries" in section_9 and "E1" not in section_9.split("\n", 2)[2]
    order = [section_9.index(text) for text in (
        "### M1 (data)", "- D1 (2026-09-27): Positions at the bumpers, e.g. the rear one.", "### M2 (models)",
        "- D3 (2026-09-28)", "### M6 (Report)", "- D4 (2026-10-01): Components reported.",
    )]  # fmt: skip
    assert order == sorted(order)
    section_10 = text[text.index(f"## {SECTIONS[9]}"):]
    assert section_10.index("python scripts/a.py") < section_10.index("python scripts/b.py x=1")
    assert "not this" not in section_10 and "python scripts/make_report.py refresh=true" in section_10
    assert lines[2].endswith("3 notes on missing inputs")  # the empty lane 3 of the three contours


def test_the_tables_of_m7_in_their_sections(made):
    """Section 2: the verdicts of M4 and of H12 one after the other; section 4: the low-frequency arm; section 6:
    the residual amplitude; section 8: the verdict lines of H12, h12_1, h12_2 and the sensitivity statement."""
    cfg, _ = made
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")

    def section(k: int) -> str:
        return text[text.index(f"## {SECTIONS[k]}\n"): text.index(f"## {SECTIONS[k + 1]}\n")]

    verdicts = section(1)
    assert verdicts.index("#### Verdicts of the hypotheses") < verdicts.index("#### Verdicts of H12.1-H12.3 (corridor)")
    assert "| H12.1 | overall | SC only | open |" in verdicts and "[tables/verdicts_h12.tex]" in verdicts
    e2 = section(3)
    assert "- H1.2 (combined), gru: **confirmed**; weight 1" in e2 and "[tables/e2_lowfreq.tex]" in e2
    assert e2.index("#### E2: the existence term alone (D93)") < e2.index("#### E2: combined penalty of the recurrent")
    assert "[tables/e4_rmax.tex]" in section(5) and "| SC | 0.5 | certified | 4 | 4 | 4 | 4 |" in section(5)
    e5 = section(6)
    assert "| first collided position | collision-free share |" in e5 and "12.5 [10.5, 14.5]" in e5
    corridor = section(7)
    for name in ("h12_1", "h12_2", "sensitivity"):
        assert f"[tables/{name}.tex]" in corridor, name
    assert "- H12.1, law_b (SC): **confirmed**; degradation vs law_a" in corridor
    assert "- H12.3, overall (SC only): **open**; Spearman: SC open." in corridor
    assert ("Ranking law_a < law_b by macro error: it holds in 1 of the 2 variants with runs of every law (baseline: "
            "law_a < law_b, v1: law_b < law_a). The ranking does not survive every variant.") in corridor  # fmt: skip
    assert "| SC | law_a | 1 |" in corridor and "| p (permutation) |" in corridor


def test_supplement_of_m8(made):
    """Section 11: every table of runs/_tables/m8 in the order of the configuration (another work package's table with every
    column of its CSV and the title of its Markdown, a known table with its columns and its sentences), every figure
    of the supplement with its caption and PDF; CSV and LaTeX in tables/."""
    cfg, lines = made
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    supplement = text[text.index(f"## {SECTIONS[10]}\n"): text.index("## Notes on missing inputs")]
    assert supplement.index("#### Width of the spacing band") < supplement.index("#### Seed-to-seed spread and power")
    assert "| data | speed | rel width | n drivers | dispersion |" in supplement
    assert "| follownet_highd | 10.000 | 0.600 [0.550, 0.650] | 25 | 1.500 |" in supplement
    assert "| mean (law_a) | mean (law_b) | SD (law_a) | SD (law_b) |" in supplement
    assert "- On SC, ten seeds detect 16.6 % (power 0.39 for 10 %)." in supplement
    link = Path(os.path.relpath(cfg.supplement_dir / "band_width.png", cfg.out_dir)).as_posix()  # from report.md
    assert link == "../runs/_report/supplement/figures/band_width.png"
    assert f"![band_width]({link})" in supplement
    assert "Figure `band_width`: Relative width of the spacing band against the speed. ([PDF]" in supplement
    assert f"([PDF]({link[:-4]}.pdf))" in supplement
    for name in M8_TEST_TABLES:
        check_latex(cfg.out_dir / "tables" / f"{name}.tex")
    power = (cfg.out_dir / "tables" / "power.tex").read_text(encoding="utf-8")
    assert r"\caption{Two laws paired by seed" in power and "+16.6 \\%" in power
    assert lines[3].startswith("SUPPLEMENT (M8) 2 tables")


def test_rows_without_runs_and_a_missing_corridor_get_notes(tree, tmp_path):
    m4, m5, configs = tmp_path / "m4", tmp_path / "m5", tmp_path / "configs"
    shutil.copytree(tree["root"] / "_tables" / "m4", m4)
    shutil.copytree(tree["root"] / "_tables" / "m5", m5)
    shutil.copytree(tree["configs"], configs)
    lowfreq = pd.read_csv(m4 / "e2_lowfreq.csv")
    lowfreq.assign(runs=0).to_csv(m4 / "e2_lowfreq.csv", index=False)  # the arm before its queue
    rmax = pd.read_csv(m5 / "e4_rmax.csv")
    rmax.loc[rmax["r_max"] == 0.5, ["runs_highd", "runs_ngsim", "runs_corridor"]] = 0
    rmax.to_csv(m5 / "e4_rmax.csv", index=False)
    design = (configs / "corridor_metrics.yaml").read_text(encoding="utf-8")
    (configs / "corridor_metrics.yaml").write_text(
        design.replace("scenarios: [sc]", "scenarios: [sc, us101_p0]").replace("{sc: SC}", "{sc: SC, us101: US-101}"),
        encoding="utf-8")  # fmt: skip
    cfg = config(tree, tmp_path / "out", tables_m4=m4, tables_m5=m5, configs_dir=configs)
    make_report(cfg)
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    assert "> Table e2_lowfreq: no runs yet (empty cells)." in text and "- table e2_lowfreq: no runs yet" in text
    assert "- table e4_rmax: rows without runs yet: 0.5 certified" in text
    assert "- corridor: US-101 not in the tables yet (no ground truth with a macro.json)" in text


def test_contours_report_an_empty_lane(made):
    cfg, _ = made
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    for label in ("ground truth", "law_a", "law_b"):
        assert f"- figure corridor_speed_contours: {label}: no vehicle in lane(s) [3]" in text


def test_manifest(made, tree):
    cfg, _ = made
    manifest = read_json(cfg.out_dir / "manifest.json")
    keys = {"date", "config", "config_hash", "runs", "corridor", "inputs", "software", "outputs", "notes"}
    assert keys <= set(manifest)
    assert manifest["config"] == {"note": "test"}
    assert manifest["runs"]["e1"]["runs"] == 6 and manifest["runs"]["e1"]["trained"] == 6
    assert manifest["runs"]["e5"]["event_sets"] == ["openacc_acc", "openacc_human"]
    assert "corridor" not in manifest["runs"] and "_tables" not in manifest["runs"]
    assert manifest["corridor"]["scenarios"]["sc"]["runs"] == 2
    assert manifest["corridor"]["scenarios"]["sc"]["scenario_config_hash"] == "abc"
    assert manifest["corridor"]["macro_config_hashes"] == ["m1"]
    assert "_tables/m4/e1.csv" in manifest["inputs"] and "_tables/m4/chosen_weights.json" in manifest["inputs"]
    assert len(manifest["inputs"]["_tables/m5/components.csv"]["sha256"]) == 16
    assert manifest["software"]["numpy"] == np.__version__
    assert sorted(manifest["outputs"]["tables"]) == sorted(ALL_TABLES)
    assert set(manifest["outputs"]["figures"].values()) == {"written"}
    assert manifest["outputs"]["supplement"] == {"tables": ["band_width", "power"], "figures": ["band_width"]}
    assert "_tables/m8/power.csv" in manifest["inputs"] and "_report/supplement/figures/band_width.png" in manifest["inputs"]


def test_missing_inputs_give_notes_and_no_exception(tmp_path):
    empty = {"root": tmp_path / "runs", "docs": tmp_path / "docs", "configs": tmp_path / "configs"}
    cfg = config(empty, tmp_path / "out")
    lines = make_report(cfg)
    for name in TABLES:
        table = pd.read_csv(cfg.out_dir / "tables" / f"{name}.csv")
        if name == "e2_existence":  # one row per architecture: no runs, empty cells
            assert table["runs"].tolist() == [0] and table["rmse_s"].isna().all()
            check_latex(cfg.out_dir / "tables" / f"{name}.tex")
        else:
            assert table.empty, name
            check_latex_or_empty(cfg.out_dir / "tables" / f"{name}.tex")
    assert lines[1].startswith(f"FIGURES 0 of {len(FIGURES)}")
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    assert all(f"## {section}" in text for section in SECTIONS)
    for name in FIGURES:
        assert f"> Figure `{name}` missing:" in text
    assert "- table e1: runs/_tables/m4/e1.csv missing" not in text  # paths are given relative to the runs root
    assert "- table e1: _tables/m4/e1.csv missing" in text
    assert "> Table e1: input missing (empty cells)." in text
    assert "- section 9:" in text and "- section 10:" in text
    assert read_json(cfg.out_dir / "manifest.json")["runs"] == {}
    # the supplement of M8 without inputs: a note per expected table and figure, no table of M8
    for name in ("band_width", "power"):
        assert f"- table {name} (M8): _tables/m8/{name}.csv missing: no inputs yet" in text
        assert not (cfg.out_dir / "tables" / f"{name}.csv").exists()
    assert "- figure band_width (M8): missing in" in text
    assert "> No table of M8 yet (notes below)." in text and "> No figure of the supplement yet" in text


def check_latex_or_empty(path: Path) -> None:
    """A table without rows: header only, still a valid tabular."""
    text = path.read_text(encoding="utf-8")
    assert text.count(r"\begin{tabular}") == 1 and r"\midrule" + "\n" + r"\bottomrule" in text
    header = text.split(r"\toprule" + "\n", 1)[1].split("\n")[0]
    assert header.endswith(r" \\")


def test_partly_missing_inputs(tree, tmp_path):
    """No components.csv, an instability table without the dynamic error, no E5 runs: notes, the rest stays."""
    m5 = tmp_path / "m5"
    m5.mkdir()
    for name in ("laws", "instability_correlation", "tost"):
        (m5 / f"{name}.csv").write_bytes((tree["root"] / "_tables" / "m5" / f"{name}.csv").read_bytes())
    instability = pd.read_csv(tree["root"] / "_tables" / "m5" / "instability.csv")
    instability.drop(columns=[c for c in instability if c.startswith("macro_error_dynamic")]).to_csv(
        m5 / "instability.csv", index=False)  # fmt: skip
    cfg = config(tree, tmp_path / "out", tables_m5=m5, e5_models=("knn",), e5_penalised=())
    lines = make_report(cfg)
    assert "missing: growth_curves" in lines[1]
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    assert "- table m5_components: m5/components.csv missing" not in text  # outside the runs root: absolute path
    assert "components.csv missing" in text
    assert "- table macro_error_dynamic: instability.csv has no column macro_error_dynamic" in text
    assert "> Figure `growth_curves` missing: no platoon.json of E5." in text
    assert (cfg.out_dir / "figures" / "macro_error_vs_instability.png").exists()  # its second panel says why


def test_std_before_collision_and_profile_labels():
    profile = {"speed_std": [1.0, 2.0, 3.0, 4.0], "min_gap": [None, 5.0, -1.0, 3.0], "collided": True}
    np.testing.assert_array_equal(std_before_collision(profile), [1.0, 2.0, np.nan, np.nan])
    profile = {"speed_std": [1.0, 2.0, 3.0], "collided": True, "collision_vehicle": 1}
    np.testing.assert_array_equal(std_before_collision(profile), [1.0, np.nan, np.nan])
    profile = {"speed_std": [1.0, None, 3.0], "min_gap": [None, 1.0, 2.0], "collided": False}
    np.testing.assert_array_equal(std_before_collision(profile), [1.0, np.nan, 3.0])
    assert profile_label("Vicolungo/JRC-VC_280219_part4_highway.csv", "human") == "Vicolungo part 4 (human)"
    assert profile_label("ZalaZone/dynamic.csv") == "dynamic"
    assert profile_label("pulse", "ACC laws") == "braking pulse (ACC laws)"


def test_decisions_and_commands(tree):
    entries = decision_entries(tree["docs"] / "decisions.md", tree["docs"])
    assert [(e["id"], e["milestone"]) for e in entries] == [("D1", 1), ("D2", 1), ("D3", 2), ("D4", 6)]
    assert entries[0]["decision"] == "Positions at the bumpers, e.g. the rear one."
    assert entries[1]["decision"] == "Deviation: speeds as measured."
    assert entries[3]["label"] == "M6 (Report)" and entries[2]["label"] == "M2 (models)"
    assert [(e["id"], e["milestone"]) for e in decision_entries(tree["docs"] / "decisions.md")][2] == ("D3", 1)
    assert decision_entries(tree["docs"] / "missing.md") == []
    assert first_sentence("Done (cf. D3). Then more.") == "Done (cf. D3)."
    assert milestone_commands(tree["docs"]) == [("M1", "bash", ["python scripts/a.py"]),
                                                ("M2", "powershell", ["python scripts/b.py x=1"])]  # fmt: skip


def test_decisions_with_escaped_pipes(tmp_path):
    """A cell of the log may hold escaped pipes (absolute values in a table): they do not end the cell."""
    (tmp_path / "decisions.md").write_text("\r\n".join([
        "## Revision (M7)", "", "| Id | Date | Decision | Reason |", "|---|---|---|---|",
        "| D5 | 2026-10-05 | A guard `relu(\\|f_dv\\| - c)` on the gain. Then more. | Needles of `\\|f\\|` > 3. |", "",
    ]), encoding="utf-8")  # fmt: skip
    entries = decision_entries(tmp_path / "decisions.md")
    assert [(e["id"], e["milestone"]) for e in entries] == [("D5", 7)]
    assert entries[0]["decision"] == "A guard `relu(|f_dv| - c)` on the gain."


def test_refresh_reruns_both_table_scripts(tree, tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        failed = "corridor_metrics.py" in command[1]
        stderr = "Error: boom\n" if failed else ""
        return subprocess.CompletedProcess(command, 1 if failed else 0, stdout="", stderr=stderr)

    monkeypatch.setattr(report_module.subprocess, "run", fake_run)
    maker = ReportMaker(config(tree, tmp_path / "out", refresh=True))
    maker.refresh()
    assert [Path(c[1]).name for c in calls] == ["make_tables.py", "corridor_metrics.py"]
    assert f"paths.out_dir='{(tree['root'] / '_tables' / 'm4').as_posix()}'" in calls[0]
    assert "tables=true" in calls[1] and f"paths.corridor_root='{(tree['root'] / 'corridor').as_posix()}'" in calls[1]
    assert maker.refreshed == ["REFRESH make_tables: exit 0", "REFRESH corridor_metrics: exit 1"]
    assert maker.notes == ["refresh: scripts/corridor_metrics.py failed (exit 1): Error: boom"]


def test_make_report_script(tree, tmp_path):
    out = tmp_path / "out"
    q = lambda path: f"'{Path(path).as_posix()}'"  # noqa: E731  # non-ASCII paths: quoted for hydra
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "make_report.py"), f"paths.runs_root={q(tree['root'])}",
        f"paths.out_dir={q(out)}", f"paths.docs_dir={q(tree['docs'])}", f"paths.configs_dir={q(tree['configs'])}",
        f"hydra.run.dir={q(tmp_path / 'hydra')}", "scenario=sc", "figure_laws=[law_a,law_b]",
        "gain_architectures=[idm,mlp,gru]", "existence_architectures=[gru]", "e5_models=[idm,mlp]",
        "e5_penalised=[mlp]", "contour_lanes=[1,2]", "dpi=50", "m8_tables=[band_width,power]",
        "supplement_expected=[band_width]",
        f"paths.supplement_figures={q(tree['root'] / '_report' / 'supplement' / 'figures')}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    printed = proc.stdout.strip().splitlines()
    assert printed[0].startswith(f"TABLES {len(ALL_TABLES)} (CSV, LaTeX)")
    assert printed[1].startswith(f"FIGURES {len(FIGURES)} of {len(FIGURES)}")
    assert printed[2].endswith("0 notes on missing inputs")
    assert printed[3].startswith("SUPPLEMENT (M8) 2 tables of _tables/m8, 1 figures")
    manifest = read_json(out / "manifest.json")
    assert manifest["config"]["scenario"] == "sc" and manifest["config"]["paths"]["out_dir"] == str(out)
    assert manifest["config"]["contour_lanes"] == [1, 2]
