"""Exact a priori certificate of the two-timescale certified hybrid at 21 frozen latents (docs/m11_contract.md, D129).

    python scripts/analysis/certificate_tt.py                       # e4_tt (before) and e4_tt_ft (after fine-tuning)
    python scripts/analysis/certificate_tt.py --sweep-step 0.1      # a coarser speed sweep

For every run of the experiments (``--experiments``; ``runs/<experiment>/<data>/residual_idm_tt/<run>``) the
``certificate.json`` of ``scripts/certificate.py`` holds the core of ``z = 0`` (``core.params``), the residual
(``residual.r_max``, ``residual.bounds``: the same at every ``z``) and, under ``latent.per_z``, the scanned a priori
certificate per grid speed at every ``z`` of ``LATENT_SWEEP`` (-1, -0.9, ..., 1). At every ``z`` this analysis runs the
machinery of ``scripts/analysis/certificate_exact.py`` (``analyse``: ``a_priori_exact`` at the grid speeds and on the
sweep of 2 501 speeds 5.00-30.00 m/s, the existence of the equilibria, the scan recomputed from the stored inputs) on
the core of that ``z``: ``T(z) = T_c exp(z ln t_factor)``, the other parameters and the residual unchanged
(``latent_core_params``). Nothing is retrained; no model is loaded (CPU, float64).

Writes ``runs/_tables/m11/certificate_tt.{csv,md}`` (one row per experiment and ``z``) and
``runs/_tables/m11/per_run/certificate_tt_runs.csv`` (one row per run and ``z``).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from certificate_exact import SWEEP, analyse, sweep_speeds  # noqa: E402
from cf_stability.eval.tables import Column, Table, write_table  # noqa: E402
from cf_stability.stability.certificate import LATENT_SWEEP, latent_core_params  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

NAME = "certificate_tt"
MODEL = "residual_idm_tt"
STAGES = {"e4_tt": "before fine-tuning (follownet_highd)", "e4_tt_ft": "after fine-tuning (ngsim_i80)"}


def runs_of(runs_root: Path, experiments: Sequence[str]) -> list[tuple[str, Path]]:
    found = []
    for experiment in experiments:
        for path in sorted((runs_root / experiment).glob(f"*/{MODEL}/*/certificate.json")):
            found.append((experiment, path))
    return found


def analyse_run(payload: dict[str, Any], sweep: Any) -> list[dict[str, Any]]:
    """One record per ``z`` of the stored sweep: the summary of ``analyse`` on the core of that ``z``."""
    latent = payload["latent"]
    t_factor = float(latent.get("t_factor", 1.5))
    reach = float(latent["reach"])
    out = []
    for entry in latent["per_z"]:
        z = float(entry["z"])
        params = latent_core_params(payload["core"]["params"], z, t_factor)
        stored = entry["core"]["params"]
        mismatch = max(abs(params[k] - float(stored[k])) / max(abs(float(stored[k])), 1e-12) for k in params)
        view = {**payload, "core": {"params": params}, "per_speed": entry["per_speed"]}
        summary, _ = analyse(view, sweep)
        out.append({"z": z, "t_ratio": math.exp(z * math.log(t_factor)), "T": params["T"],
                    "reachable": abs(z) <= reach + 1e-12, "reach": reach, "core_mismatch": mismatch,
                    "stored_a_priori_holds": bool(entry["a_priori"]["holds"]), **summary})  # fmt: skip
    return out


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, z), group in runs.groupby(["experiment", "z"], sort=False):
        rows.append({
            "experiment": experiment, "stage": STAGES.get(experiment, ""), "z": z,
            "t_ratio": float(group["t_ratio"].iloc[0]), "reachable": bool(group["reachable"].all()),
            "runs": len(group), "runs_holding_exact": int(group["holds_exact_all"].sum()),
            "runs_holding_sweep": int(group["sweep_holds_all"].sum()),
            "runs_holding_scan": int(group["holds_scan_all"].sum()),
            "exact_min": group["exact_min"].min(), "scan_min": group["scan_min"].min(),
            "sweep_min": group["sweep_min"].min(), "max_abs_diff": group["max_abs_diff"].max(),
            "holds_differ": int(group["n_holds_differ"].sum()), "limit_margin_min": group["limit_margin_min"].min(),
            "within_share": float(group["within_share"].mean()), "rescan_max_abs_diff": group["rescan_max_abs_diff"].max(),
            "core_mismatch": group["core_mismatch"].max(), "bound_s": group["bound_s"].max(),
            "bound_dv": group["bound_dv"].max(), "bound_v": group["bound_v"].max(), "product": group["product"].max(),
            "reach": float(group["reach"].max()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--experiments", nargs="+", default=list(STAGES))
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--sweep-step", type=float, default=SWEEP[2])
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    sweep = sweep_speeds(SWEEP[0], SWEEP[1], args.sweep_step)
    records, problems = [], []
    for experiment, path in runs_of(resolve_path(args.runs_root), args.experiments):
        payload = read_json(path)
        run = path.parent.name
        if not isinstance(payload.get("latent"), dict):
            problems.append(f"{experiment}/{run}: no latent block ({payload.get('error') or 'not a latent model'})")
            continue
        for record in analyse_run(payload, sweep):
            records.append({"experiment": experiment, "data": path.parents[2].name, "run": run, **record})
    runs = pd.DataFrame(records)
    out = resolve_path(args.out)
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "per_run" / f"{NAME}_runs.csv", index=False)
    table = summarise(runs) if len(runs) else pd.DataFrame()
    holding = {e: [float(z) for z, g in runs[runs["experiment"] == e].groupby("z") if g["holds_exact_all"].all()
                   and g["sweep_holds_all"].all()] for e in args.experiments} if len(runs) else {}  # fmt: skip
    notes = [
        f"Runs: runs/<experiment>/<data>/{MODEL}/<run> of {', '.join(args.experiments)} ({runs['run'].nunique() if len(runs) else 0} "
        "runs per experiment at most); the stored certificate.json of scripts/certificate.py (no model is loaded). "
        "z: the frozen latent (LATENT_SWEEP: -1, -0.9, ..., 1); T(z) / T_c = exp(z ln 1.5); reachable: |z| <= the largest "
        "|z| of the online latent, 1 - (1 - dt / tau_z)^window (tau_z 15 s, window 30: 0.182), the range on which the "
        "training took the certified budget (D129).",
        "At every z the machinery of scripts/analysis/certificate_exact.py (analyse) on the core of that z (T(z), the "
        "other parameters of core.params) with residual.r_max and residual.bounds (the same at every z): exact = "
        "a_priori_exact at the grid speeds 5, 6, ..., 30 m/s, sweep = a_priori_exact at 2 501 speeds 5.00-30.00 m/s, "
        "scan = the stored scan of latent.per_z (400 spacings, 3 zooms). runs holding: runs whose certificate holds at "
        "every grid speed (exact, scan) or on the whole sweep; exact, scan, sweep min: the smallest guaranteed margin "
        "(s⁻²) over the runs and speeds; max |exact - scan| and holds differ as in certificate_exact; min a(1-(v/v0)^δ) - "
        "r_max: existence of the equilibria (positive: every residual with |r| <= r_max gives the hybrid an equilibrium "
        "in I(v)); rescan: the scan recomputed from the stored inputs against the stored one (largest difference).",
        "z at which every run holds on the grid and on the sweep: " + "; ".join(
            f"{e}: {', '.join(f'{z:g}' for z in zs) or 'none'} ({len(zs)} of {len(LATENT_SWEEP)})" for e, zs in holding.items()),
    ]  # fmt: skip
    if problems:
        notes.append(f"Problems: {len(problems)} ({'; '.join(problems[:5])}).")
    columns = [
        Column("experiment", "experiment", "text"), Column("z", "z", digits=1), Column("t_ratio", "T(z) / T_c", digits=3),
        Column("reachable", "reachable", "flag"), Column("runs", "runs", "int"),
        Column("runs_holding_exact", "runs holding, exact", "int"), Column("runs_holding_sweep", "runs holding, sweep", "int"),
        Column("runs_holding_scan", "runs holding, scan", "int"), Column("exact_min", "exact min (s⁻²)", digits=4),
        Column("sweep_min", "sweep min (s⁻²)", digits=4), Column("scan_min", "scan min (s⁻²)", digits=4),
        Column("max_abs_diff", "max \\|exact - scan\\|", digits=5), Column("holds_differ", "holds differ", "int"),
        Column("limit_margin_min", "min a(1-(v/v0)^δ) - r_max (m/s²)", digits=3),
        Column("rescan_max_abs_diff", "rescan", digits=6),
    ]  # fmt: skip
    title = "Exact a priori certificate of the two-timescale hybrid at 21 frozen latents (D129)"
    write_table(Table(NAME, title, notes, table, columns), out)
    for e, zs in holding.items():
        print(f"  {e:<10} holds (exact grid and sweep, every run) at {len(zs)} of {len(LATENT_SWEEP)} z: "
              f"{', '.join(f'{z:g}' for z in zs)}")  # fmt: skip
    for line in problems:
        print(f"  problem: {line}")
    print(f"{NAME}: {len(runs)} (run, z) rows, {time.perf_counter() - t_start:.0f} s -> {out / (NAME + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
