"""The corridor runs with the safety filter against the runs without it (M11, D130).

    python scripts/analysis/safety_filter.py            # runs/_tables/m11/safety_filter.{csv,md}

Runs: ``runs/corridor/<scenario>/<law>_filter/seed<s>`` (``sim.safety_filter`` on, ``cf_stability/corridor/safety.py``;
the law files ``<law>_filter`` are copies of the laws, the same members and draws) against
``runs/corridor/<scenario>/<law>/seed<s>`` (the runs of the corridor, filter off), paired by scenario and seed; the
metrics of their ``macro.json`` (``scripts/corridor_metrics.py``) and the filter's activity of their ``run.json``. Per law
and scenario: the means over the seeds without and with the filter of the contact rate (contact episodes per 1000
vehicle-km inside the analysis window), the episodes, the macro error, the dynamic macro error, the inserted share and
the insertion delay, and the paired differences (filter - no filter) with their 95 % percentile bootstrap intervals over
the seeds (1000 resamples, seed 0). Rule of D130: the filter works when the contact rate of the three networks (mlp,
gru, lstm) falls to zero on both scenarios and the macro error of the certified hybrid changes by at most 0.02 (the
mean paired change); the networks' change of the macro error is reported.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.corridor_tables import flatten_macro  # noqa: E402
from cf_stability.eval.stats import bootstrap_ci  # noqa: E402
from cf_stability.eval.tables import Column, Table, markdown  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

LAWS = ("mlp", "gru", "lstm", "residual_idm_certified")
NETWORKS = ("mlp", "gru", "lstm")
SCENARIOS = ("i80_p1", "us101_p1")
SEEDS = (0, 1, 2, 3, 4)
SUFFIX = "_filter"
METRICS = {  # key of the flat macro.json (or run.json): header, digits
    "collisions_per_1000_vkm": ("contacts / 1000 veh-km", 3), "n_collisions": ("contact episodes", 1),
    "macro_error": ("macro error", 3), "macro_error_dynamic": ("macro error (dynamic)", 3),
    "inserted_share": ("inserted share", 3), "mean_depart_delay_s": ("insertion delay (s)", 1),
}
MAX_CHANGE = 0.02


def run_values(run: Path) -> dict[str, Any] | None:
    """The flat metrics of a run's ``macro.json`` and the filter counts of its ``run.json`` (None without macro.json)."""
    if not (run / "macro.json").is_file():
        return None
    row: dict[str, Any] = flatten_macro(read_json(run / "macro.json"))
    info = read_json(run / "run.json") if (run / "run.json").is_file() else {}
    row["filter_active_share"] = info.get("safety_filter_active_share", math.nan)
    row["filter"] = (info.get("config") or {}).get("sim", {}).get("safety_filter")
    return row


def collect(corridor_root: Path, laws: Sequence[str], scenarios: Sequence[str], seeds: Sequence[int]
            ) -> tuple[pd.DataFrame, list[str]]:  # fmt: skip
    rows, missing = [], []
    for scenario in scenarios:
        for law in laws:
            for seed in seeds:
                pair = {}
                for side, name in (("off", law), ("on", law + SUFFIX)):
                    values = run_values(corridor_root / scenario / name / f"seed{seed}")
                    if values is None:
                        missing.append(f"{scenario}/{name}/seed{seed}: macro.json missing")
                    pair[side] = values
                if pair["on"] is not None and pair["on"]["filter"] is None:
                    missing.append(f"{scenario}/{law}{SUFFIX}/seed{seed}: run.json without the filter")
                row = {"scenario": scenario, "law": law, "seed": seed}
                for side in ("off", "on"):
                    for key in (*METRICS, "filter_active_share"):
                        row[f"{key}_{side}"] = math.nan if pair[side] is None else pair[side].get(key, math.nan)
                rows.append(row)
    return pd.DataFrame(rows), missing


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scenario, law), part in runs.groupby(["scenario", "law"], sort=False):
        row: dict[str, Any] = {"scenario": scenario, "law": law}
        both = part.dropna(subset=["macro_error_off", "macro_error_on"])
        row["pairs"] = len(both)
        for key in METRICS:
            off, on = part[f"{key}_off"].to_numpy(float), part[f"{key}_on"].to_numpy(float)
            row[f"{key}_off"] = float(np.nanmean(off)) if np.isfinite(off).any() else math.nan
            row[f"{key}_on"] = float(np.nanmean(on)) if np.isfinite(on).any() else math.nan
            ci = bootstrap_ci(on - off, n_resamples=1000, seed=0)
            row[f"{key}_change"], row[f"{key}_change_low"], row[f"{key}_change_high"] = ci["estimate"], ci["low"], ci["high"]
        row["contact_free_runs"] = int((part["n_collisions_on"] == 0).sum())
        row["filter_active_share"] = float(np.nanmean(part["filter_active_share_on"]))
        out.append(row)
    return pd.DataFrame(out)


def verdict_lines(table: pd.DataFrame) -> list[str]:
    nets = table[table["law"].isin(NETWORKS)]
    zero = bool(len(nets) and (nets["collisions_per_1000_vkm_on"] == 0).all())
    e = table[table["law"] == "residual_idm_certified"]
    small = bool(len(e) and (e["macro_error_change"].abs() <= MAX_CHANGE).all())
    rates = ", ".join(f"{r['law']} {r['scenario']} {r['collisions_per_1000_vkm_on']:.3f}" for r in nets.to_dict("records"))
    changes = ", ".join(f"{r['scenario']} {r['macro_error_change']:+.4f}" for r in e.to_dict("records"))
    branch = "works" if zero and small else "does not work"
    return [
        f"contact rate of the three networks with the filter (per 1000 veh-km, mean over the seeds): {rates}; all zero: "
        f"{zero}.",
        f"macro error of the certified hybrid, change with the filter (mean paired): {changes}; within "
        f"{MAX_CHANGE:g}: {small}.",
        f"rule of D130: the filter {branch}.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corridor-root", default="runs/corridor")
    parser.add_argument("--out", default="runs/_tables/m11")
    args = parser.parse_args(argv)
    runs, missing = collect(resolve_path(args.corridor_root), LAWS, SCENARIOS, SEEDS)
    table = summarise(runs)
    lines = verdict_lines(table)
    columns = [Column("scenario", "scenario", "text"), Column("law", "law", "text"), Column("pairs", "pairs", "int")]
    for key, (header, digits) in METRICS.items():
        columns += [Column(f"{key}_off", f"{header}, off", digits=digits), Column(f"{key}_on", f"{header}, on",
                                                                                  digits=digits),
                    Column(f"{key}_change", f"{header}, change", digits=digits, ci=True)]  # fmt: skip
    columns += [Column("contact_free_runs", "runs without contact (on)", "int"),
                Column("filter_active_share", "filter active (share of vehicle steps)", digits=4)]  # fmt: skip
    notes = [
        "Safety filter of D130 (cf_stability/corridor/safety.py, sim.safety_filter of configs/run_corridor.yaml): the "
        "law's acceleration clipped to ((v_lead - v) + alpha h) / (tau_s + [v > v_lead] v / b) with the barrier h = s - "
        "s_min - v tau_s - max(0, v^2 - v_lead^2) / (2 b), s_min 2 m, tau_s 0.5 s, b 4 m/s^2, alpha 1 1/s, before the "
        "clip of the loop; the runs with the filter belong to the law files <law>_filter (copies of the laws: the same "
        "members and draws for a seed, the same scenario and SUMO seed), the runs without it are those of the corridor.",
        "Per law and scenario the means over the seeds 0-4 without (off) and with (on) the filter and the paired change "
        "(on - off) with its 95 % percentile bootstrap interval over the seeds (1000 resamples, seed 0). contacts: "
        "contact episodes of followers per 1000 vehicle-km inside the analysis window; macro error and its dynamic part "
        "as in laws (runs/_tables/m5); inserted share and insertion delay of the demand; filter active: the share of "
        "the controlled vehicle steps at which the bound was below the law's acceleration (run.json).",
        *lines,
        *(f"missing: {m}" for m in missing),
    ]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "safety_filter.csv", index=False)
    (out / "safety_filter.md").write_text(markdown(Table("safety_filter", "Corridor runs with the safety filter "
                                                         "(D130)", notes, table, columns)), encoding="utf-8")  # fmt: skip
    run_columns = [Column("scenario", "scenario", "text"), Column("law", "law", "text"), Column("seed", "seed", "int")]
    for key, (header, digits) in METRICS.items():
        run_columns += [Column(f"{key}_off", f"{header}, off", digits=digits),
                        Column(f"{key}_on", f"{header}, on", digits=digits)]  # fmt: skip
    run_columns.append(Column("filter_active_share_on", "filter active (on)", digits=4))
    runs.to_csv(out / "safety_filter_runs.csv", index=False)
    (out / "safety_filter_runs.md").write_text(markdown(Table(
        "safety_filter_runs", "Corridor runs with and without the safety filter, per run (D130)",
        ["The pairs of safety_filter, one row per scenario, law and seed (off: runs/corridor/<scenario>/<law>/seed<s>, "
         "on: runs/corridor/<scenario>/<law>_filter/seed<s>)."], runs, run_columns)), encoding="utf-8")  # fmt: skip
    for line in lines:
        print(line)
    print(f"safety_filter: {len(runs)} pairs, {len(missing)} missing -> {out / 'safety_filter.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
