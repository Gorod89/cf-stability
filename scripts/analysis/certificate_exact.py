"""Exact a priori certificate of the ResidualIDM runs and the existence of their equilibria (M9, review).

    python scripts/analysis/certificate_exact.py                     # every certificate.json under runs/
    python scripts/analysis/certificate_exact.py --sweep-step 0.1    # a coarser speed sweep

The a priori certificate of ``certificate.json`` (``scripts/certificate.py``: ``certificate_a_priori`` of
``cf_stability/stability/certificate.py``) takes the minimum of the guaranteed margin over the feasible spacings
``I(v) = {s in [1, 200] m : |f_idm(s, 0, v)| <= r_max}`` from a scan of 400 spacings with 3 zooms. This analysis takes
it exactly with ``a_priori_exact`` (the margin at the ends of ``I(v)``, at the breakpoints of the box minimum and at
the critical points of its polynomial pieces in ``x = 1/s``) from the stored core (``core.params``; delta 4, the IDM of
ResidualIDM), ``residual.r_max`` and ``residual.bounds``. No model is loaded and nothing is retrained (CPU, float64);
the stored certificates are only read. For every run directory ``runs/<experiment>/<data>/<model>/<run>`` whose
``certificate.json`` has an ``a_priori`` entry (the MLP files have none):

* per grid speed (``per_speed[].v``, 5, 6, ..., 30 m/s): the stored scan minimum (``a_priori_margin``), the exact
  minimum, their difference, and ``holds`` of both (margin >= 0 and the signs ``f_s - B_s > 0`` and
  ``f_v + f_dv + B_v + B_dv < 0`` at the ends of ``I(v)``); a check: the scan recomputed from the stored inputs
  (``_a_priori``, ``n_scan`` of the file) against the stored one;
* a sweep of the speeds 5.00, 5.01, ..., 30.00 m/s (2 501) with the exact minimum at each: its smallest value and
  whether the certificate holds at every speed of the sweep;
* the existence of the equilibria (``feasible_ends``): ``a (1 - (v/v0)^delta) - r_max`` (positive: ``I(v)`` has a
  finite upper end and every residual with ``|r| <= r_max`` gives the hybrid an equilibrium in it), the ends of
  ``I(v)`` before the clamping to [1, 200] m and whether ``I(v)`` lies within [1, 200] m (then the certificate covers
  every equilibrium, and one exists).

Writes ``runs/_tables/m9/certificate_exact.{csv,md}`` (one row per experiment) and
``runs/_tables/m9/per_run/certificate_exact_runs.csv`` (one row per run, with the speeds where ``I(v)`` leaves
[1, 200] m; in a subfolder so that the report, which turns every CSV of m9 into a table, leaves it out).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from cf_stability.eval.tables import Column, Table, markdown  # noqa: E402
from cf_stability.models.idm import IDM, IDM_PARAM_NAMES  # noqa: E402
from cf_stability.stability.certificate import S_MAX, S_MIN, _a_priori, a_priori_exact, feasible_ends  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

INPUT = "certificate.json"
NAME = "certificate_exact"
DELTA = 4.0  # exponent of the IDM core of ResidualIDM (built with the IDM default, cf_stability/models/residual_idm.py)
N_SCAN = 400  # spacings of the stored scan when the file does not say (configs/certificate.yaml)
SWEEP = (5.0, 30.0, 0.01)  # m/s: first and last speed and step of the sweep
CERTIFIED = "core with margin 0.2, certified budget"
DESIGN: dict[str, str] = {  # experiment -> design, in the order of the table; other experiments follow, sorted
    "e4_stable_r0.1": f"{CERTIFIED} (D111)",
    "e4_stable_ft_r0.1": f"{CERTIFIED}, fine-tuned (D111)",
    "e4_stable_r0.2": f"{CERTIFIED} (D111)",
    "e4_stable_ft_r0.2": f"{CERTIFIED}, fine-tuned (D111)",
    "e4_stable": f"{CERTIFIED} (D86)",
    "e4_stable_ft": f"{CERTIFIED}, fine-tuned (D86)",
    "e4_stable_r0.5": f"{CERTIFIED} (D111)",
    "e4_stable_ft_r0.5": f"{CERTIFIED}, fine-tuned (D111)",
    "m8_temporal_stable_ft": f"{CERTIFIED}, fine-tuned on I-80 period 0 (D120)",
    "e4_margin_free_r0.3": "core with margin 0.2, no certificate (variant D)",
    "e4_margin_free_r0.3_ft": "core with margin 0.2, no certificate, fine-tuned (variant D)",
    "e4_free_r0.3": "free core, no certificate (D111)",
    "e4_free_r0.3_ft": "free core, no certificate, fine-tuned (D111)",
    "e4_free_ft": "free core of E1, no certificate, fine-tuned",
    "m4_e4pilot_r0.1": f"pilot of D86, {CERTIFIED}",
    "m4_e4pilot_r0.2": f"pilot of D86, {CERTIFIED}",
    "m4_e4pilot_r0.3": f"pilot of D86, {CERTIFIED}",
    "m8_smoke": "smoke test of m8_temporal_stable_ft (2 epochs)",
}
OUTSIDE = (  # how I(v) leaves [S_MIN, S_MAX], with the per-speed record that says so
    ("upper end infinite", lambda r: not r["bounded"]),
    (f"upper end above {S_MAX:g} m", lambda r: r["bounded"] and r["s_high_unclamped"] > S_MAX),
    (f"lower end below {S_MIN:g} m", lambda r: r["s_low_unclamped"] < S_MIN),
)


def _num(x: Any) -> float:
    """Float, NaN for None (the JSON null of a non-finite value)."""
    return math.nan if x is None else float(x)


def _smallest(values: Tensor, at: Tensor) -> tuple[float, float]:
    """Smallest finite value of ``values`` and the entry of ``at`` where it is attained; NaN, NaN without any."""
    finite = torch.isfinite(values)
    if not finite.any():
        return math.nan, math.nan
    k = int(torch.where(finite, values, torch.inf).argmin())
    return float(values[k]), float(at[k])


def sweep_speeds(first: float, last: float, step: float) -> Tensor:
    """``first, first + step, ..., last`` (m/s) as integers divided by ``1 / step``: the grid speeds are among them bit
    for bit. ``1 / step`` must be an integer."""
    scale = round(1.0 / step)
    if scale < 1 or abs(scale * step - 1.0) > 1e-9:
        raise ValueError(f"the sweep step must divide 1 m/s, got {step}")
    return torch.arange(round(first * scale), round(last * scale) + 1, dtype=torch.float64) / scale


def speed_ranges(speeds: Sequence[float]) -> str:
    """``5-7, 9, 12.5`` m/s: runs of consecutive whole speeds joined."""
    parts: list[list[float]] = []
    for v in sorted(speeds):
        if parts and float(v).is_integer() and float(parts[-1][-1]).is_integer() and v == parts[-1][-1] + 1.0:
            parts[-1].append(v)
        else:
            parts.append([v])
    return ", ".join(f"{p[0]:g}" if len(p) == 1 else f"{p[0]:g}-{p[-1]:g}" for p in parts)


def outside_text(records: Sequence[Mapping[str, Any]], runs: int | None = None) -> str:
    """Where ``I(v)`` leaves [1, 200] m: per kind of :data:`OUTSIDE` the speeds; with ``runs`` (records of that many
    runs) the number of runs per speed, ``all`` when every run. Empty when every ``I(v)`` lies within."""
    texts = []
    for kind, test in OUTSIDE:
        counts: dict[float, int] = {}
        for r in records:
            if test(r):
                counts[r["v"]] = counts.get(r["v"], 0) + 1
        if not counts:
            continue
        if runs is None:
            texts.append(f"{kind} at {speed_ranges(list(counts))} m/s")
            continue
        groups: dict[int, list[float]] = {}
        for v, n in counts.items():
            groups.setdefault(n, []).append(v)
        pieces = [f"{speed_ranges(vs)} m/s ({'all' if n == runs else n} runs)" for n, vs in
                  sorted(groups.items(), key=lambda item: min(item[1]))]  # fmt: skip
        texts.append(f"{kind} at {', '.join(pieces)}")
    return "; ".join(texts)


def run_label(path: Path, runs_root: Path) -> tuple[str, str, str, str]:
    """``(experiment, data, model, run)`` of ``runs/<experiment>/<data>/<model>/<run>/certificate.json``."""
    parts = path.relative_to(runs_root).parts
    return parts[0], parts[1], parts[2], parts[3]


def certificates(runs_root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """The ``certificate.json`` files of the run directories with an ``a_priori`` entry (directories of ``runs_root``
    that start with ``_`` are left out) and the files left out, with the reason."""
    found, skipped = [], []
    for path in sorted(runs_root.glob(f"*/*/*/*/{INPUT}")):
        label = "/".join(path.relative_to(runs_root).parts[:-1])
        if label.startswith("_"):
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError) as exc:
            skipped.append(f"{label}: unreadable ({type(exc).__name__})")
            continue
        if not isinstance(payload.get("a_priori"), Mapping):
            reason = "error" if payload.get("error") else f"no a_priori entry (model {payload.get('model')})"
            skipped.append(f"{label}: {reason}")
            continue
        found.append((path, payload))
    return found, skipped


def analyse(payload: Mapping[str, Any], sweep: Tensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The exact a priori certificate of one ``certificate.json`` (its core, residual and per-speed scan): the summary
    of the run and one record per grid speed."""
    params = payload["core"]["params"]
    idm = IDM({name: float(params[name]) for name in IDM_PARAM_NAMES}, dtype=torch.float64, delta=DELTA)
    residual = payload["residual"]
    r_max = float(residual["r_max"])
    bounds = torch.tensor([float(b) for b in residual["bounds"]], dtype=torch.float64)
    per_speed = payload["per_speed"]
    v = torch.tensor([float(r["v"]) for r in per_speed], dtype=torch.float64)
    scan = torch.tensor([_num(r.get("a_priori_margin")) for r in per_speed], dtype=torch.float64)
    holds_scan = torch.tensor([bool(r.get("a_priori_holds")) for r in per_speed])
    exact = a_priori_exact(idm, v, r_max, bounds)
    margin = exact["guaranteed_margin"]
    n_scan = int(((payload.get("config") or {}).get("certificate") or {}).get("n_scan") or N_SCAN)
    rescan = _a_priori(idm, v, torch.full_like(v, r_max), bounds.expand(len(v), 3), n_scan)["guaranteed_margin"]
    ends = feasible_ends(idm, v, r_max)
    swept = a_priori_exact(idm, sweep, r_max, bounds)
    both = torch.isfinite(margin) & torch.isfinite(scan)
    records = [
        {
            "v": float(v[k]), "scan": float(scan[k]), "exact": float(margin[k]), "holds_scan": bool(holds_scan[k]),
            "holds_exact": bool(exact["holds"][k]), "s_worst": float(exact["s_worst"][k]),
            "s_low": float(exact["s_low"][k]), "s_high": float(exact["s_high"][k]),
            "limit_margin": float(ends["limit_margin"][k]), "s_low_unclamped": float(ends["s_low"][k]),
            "s_high_unclamped": float(ends["s_high"][k]), "bounded": bool(ends["bounded"][k]),
            "within": bool(ends["within"][k]),
        }
        for k in range(len(v))
    ]  # fmt: skip
    exact_min, v_exact_min = _smallest(margin, v)
    sweep_min, v_sweep_min = _smallest(swept["guaranteed_margin"], sweep)
    limit_min, v_limit_min = _smallest(ends["limit_margin"], v)
    at_upper = exact["s_worst"] == exact["s_high"]  # the minimiser at an end of I(v) (a candidate of its own)
    at_lower = ~at_upper & (exact["s_worst"] == exact["s_low"])
    summary = {
        "r_max": r_max, "lipschitz": _num(residual.get("lipschitz")), "product": _num(residual.get("product")),
        "bound_s": float(bounds[0]), "bound_dv": float(bounds[1]), "bound_v": float(bounds[2]),
        **{name: float(params[name]) for name in IDM_PARAM_NAMES},
        "speeds": len(v), "scan_min": _smallest(scan, v)[0], "exact_min": exact_min, "v_exact_min": v_exact_min,
        "s_worst_exact_min": records[int(torch.where(torch.isfinite(margin), margin, torch.inf).argmin())]["s_worst"],
        "max_abs_diff": float((margin - scan)[both].abs().max()) if both.any() else math.nan,
        "feasible_mismatch": int((torch.isfinite(margin) != torch.isfinite(scan)).sum()),
        "n_worst_upper_end": int(at_upper.sum()), "n_worst_lower_end": int(at_lower.sum()),
        "n_worst_inside": int((exact["feasible"] & ~at_upper & ~at_lower).sum()),
        "n_holds_scan": int(holds_scan.sum()), "n_holds_exact": int(exact["holds"].sum()),
        "n_holds_differ": int((holds_scan != exact["holds"]).sum()),
        "holds_scan_all": bool(holds_scan.all()), "holds_exact_all": bool(exact["holds"].all()),
        "rescan_max_abs_diff": float((rescan - scan).nan_to_num(0.0).abs().max()),
        "sweep_speeds": len(sweep), "sweep_min": sweep_min, "v_sweep_min": v_sweep_min,
        "sweep_n_fail": int((~swept["holds"]).sum()), "sweep_holds_all": bool(swept["holds"].all()),
        "limit_margin_min": limit_min, "v_limit_margin_min": v_limit_min, "bounded_all": bool(ends["bounded"].all()),
        "within_share": float(ends["within"].double().mean()), "n_outside": int((~ends["within"]).sum()),
        "s_low_unclamped_min": float(ends["s_low"].min()), "s_high_unclamped_max": float(ends["s_high"].max()),
        "outside": outside_text(records),
    }  # fmt: skip
    return summary, records


def collect(runs_root: Path, sweep: Tensor) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """One row per run (:func:`analyse`), one per run and grid speed, the problems and the files left out."""
    found, skipped = certificates(runs_root)
    rows, cells, problems = [], [], []
    for path, payload in found:
        experiment, data, model, run = run_label(path, runs_root)
        key = {"experiment": experiment, "data": data, "model": model, "run": run}
        try:
            summary, records = analyse(payload, sweep)
        except (KeyError, TypeError, ValueError) as exc:  # a malformed file must not stop the others
            problems.append(f"{experiment}/{data}/{model}/{run}: {type(exc).__name__}: {exc}")
            continue
        fold_seed = re.search(r"fold(\d+)_seed(\d+)", run)
        fold, seed = (int(fold_seed.group(1)), int(fold_seed.group(2))) if fold_seed else (None, None)
        rows.append({**key, "design": DESIGN.get(experiment, ""), "fold": fold, "seed": seed, **summary})
        cells += [{**key, **r} for r in records]
        if summary["feasible_mismatch"]:
            problems.append(f"{experiment}/{data}/{model}/{run}: {summary['feasible_mismatch']} speeds feasible for "
                            "one of exact and scan only")  # fmt: skip
    order = {name: k for k, name in enumerate(DESIGN)}
    runs = pd.DataFrame(rows)
    if len(runs):
        runs["_order"] = runs["experiment"].map(order).fillna(len(order))
        runs = runs.sort_values(["_order", "experiment", "data", "run"], kind="stable").drop(columns="_order")
    return runs.reset_index(drop=True), pd.DataFrame(cells), problems, skipped


def summarise(runs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment (in the order of the runs): the numbers of the table."""
    out = []
    for experiment, group in runs.groupby("experiment", sort=False):
        mine = cells[cells["experiment"] == experiment].to_dict("records")
        r_max = group["r_max"].unique()
        out.append({
            "experiment": experiment, "design": group["design"].iloc[0],
            "data": ", ".join(sorted(group["data"].unique())),
            "r_max": float(r_max[0]) if len(r_max) == 1 else math.nan, "runs": len(group),
            "speeds": int(group["speeds"].max()), "cells": int(group["speeds"].sum()),
            "exact_min": group["exact_min"].min(), "scan_min": group["scan_min"].min(),
            "max_abs_diff": group["max_abs_diff"].max(), "holds_differ": int(group["n_holds_differ"].sum()),
            "worst_at_upper_end": float(group["n_worst_upper_end"].sum() / group["speeds"].sum()),
            "worst_at_lower_end": float(group["n_worst_lower_end"].sum() / group["speeds"].sum()),
            "worst_inside": float(group["n_worst_inside"].sum() / group["speeds"].sum()),
            "runs_holding_exact": int(group["holds_exact_all"].sum()),
            "runs_holding_scan": int(group["holds_scan_all"].sum()),
            "sweep_speeds": int(group["sweep_speeds"].max()), "sweep_min": group["sweep_min"].min(),
            "runs_holding_sweep": int(group["sweep_holds_all"].sum()),
            "sweep_holds_all": bool(group["sweep_holds_all"].all()),
            "limit_margin_min": group["limit_margin_min"].min(),
            "within_share": float((group["within_share"] * group["speeds"]).sum() / group["speeds"].sum()),
            "s_low_unclamped_min": group["s_low_unclamped_min"].min(),
            "s_high_unclamped_max": group["s_high_unclamped_max"].max(),
            "outside": outside_text(mine, runs=len(group)),
            "rescan_max_abs_diff": group["rescan_max_abs_diff"].max(),
        })  # fmt: skip
    return pd.DataFrame(out)


def _sci(value: Any) -> str:
    return "" if value is None or not np.isfinite(value) else f"{float(value):.1e}"


def _of(row: Mapping[str, Any], key: str) -> str:
    return f"{int(row[key])} / {int(row['runs'])}"


def markdown_text(
    table: pd.DataFrame, runs: pd.DataFrame, problems: Sequence[str], skipped: Sequence[str], sweep: Tensor
) -> str:  # fmt: skip
    """The Markdown of the table (title, notes, table): the format of the other tables of runs/_tables/m9."""
    shown = table.copy()  # every column as in the CSV, so that a generic reader reproduces the cells from it
    if len(table):  # an infinite upper end is blank in the Markdown (inf in the CSV; the last column names the speeds)
        shown["s_high_unclamped_max"] = [x if np.isfinite(x) else np.nan for x in table["s_high_unclamped_max"]]
    columns = [
        Column("experiment", "experiment", "text"), Column("design", "design", "text"), Column("data", "data", "text"),
        Column("r_max", "r_max (m/s²)", digits=1), Column("runs", "runs", "int"),
        Column("speeds", "grid speeds", "int"), Column("exact_min", "exact min (s⁻²)", digits=4),
        Column("scan_min", "scan min (s⁻²)", digits=4), Column("max_abs_diff", "max \\|exact - scan\\| (s⁻²)", digits=4),
        Column("holds_differ", "holds differ", "int"),
        Column("worst_at_upper_end", "minimum at the upper end of I(v)", digits=3),
        Column("runs_holding_exact", "runs holding at every grid speed, exact", "int"),
        Column("runs_holding_scan", "runs holding at every grid speed, scan", "int"),
        Column("sweep_min", "sweep min (s⁻²)", digits=4),
        Column("runs_holding_sweep", "runs holding on the whole sweep", "int"),
        Column("limit_margin_min", "min a(1-(v/v0)^δ) - r_max (m/s²)", digits=3),
        Column("within_share", "I(v) within [1, 200] m", digits=3),
        Column("s_low_unclamped_min", "min s_low (m)", digits=2),
        Column("s_high_unclamped_max", "max s_high (m)", digits=1),
        Column("outside", "I(v) outside [1, 200] m", "text"),
    ]  # fmt: skip
    first, last = float(sweep[0]), float(sweep[-1])
    step = float(sweep[1] - sweep[0]) if len(sweep) > 1 else 1.0
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    n_cells = int(table["cells"].sum()) if len(table) else 0
    rescan = float(table["rescan_max_abs_diff"].max()) if len(table) else math.nan
    where = {key: int(runs[f"n_worst_{key}"].sum()) if len(runs) else 0 for key in ("upper_end", "lower_end", "inside")}
    notes = [
        "Runs: every run directory runs/<experiment>/<data>/<model>/<run> whose certificate.json has an a_priori "
        f"entry (ResidualIDM; {len(runs)} runs of {len(table)} experiments, {n_cells} (run, grid speed) cells; "
        f"{len(skipped)} files without it left out: {_skipped_text(skipped)}). data: the event set of the runs; r_max "
        "(m/s²): the amplitude of the residual; runs: runs of the experiment; grid speeds: speeds of per_speed per run "
        "(5, 6, ..., 30 m/s). Inputs: the stored IDM core (core.params, δ = 4 as in ResidualIDM), residual.r_max, "
        "residual.bounds (B_s, B_dv, B_v) and per_speed; no model is loaded, nothing is retrained (CPU, float64). "
        "design: core with margin 0.2 = calibrated with stability_margin 0.2, free core = without it; certified "
        "budget = the budget of D86 enforced in training (lipschitz = 0.9 b / r_max with the admissible budget b of "
        "the core), no certificate = lipschitz 1 of the model config; fine-tuned runs keep the core and budget of "
        "their source run.",
        "Guaranteed margin (s⁻²): the minimum of f_v² + 2 f_v f_dv - 2 f_s over the feasible spacings I(v) = {s in "
        "[1, 200] m : \\|f_idm(s, 0, v)\\| <= r_max} and over the box \\|d_j\\| <= B_j of the residual derivatives "
        "added to the IDM derivatives. scan: the stored a_priori_margin (400 spacings including both ends of I(v), "
        "then 3 rescans around the best one, 2 cells wide). exact: a_priori_exact (cf_stability/stability/"
        "certificate.py): with x = 1/s the box minimum is the pointwise minimum of polynomials of degree <= 4 in x (2 "
        "edges of the box in f_dv x 3 regimes of the clamp in f_v, which change at 4 breakpoints with explicit x²), so "
        "its minimum over I(v) lies at an end, at a breakpoint or at a real root of the derivative of a piece; the "
        "margin is evaluated there with partials and guaranteed_margin, as by the scan. exact min, scan min: the "
        "smallest over the runs and grid speeds; max \\|exact - scan\\|: the largest difference over the runs and grid "
        "speeds (the exact minimum is never above the scan, up to rounding). minimum at the upper end of I(v): share "
        "of the (run, grid speed) cells whose exact minimiser is the upper end s_high, a point of the scan (over all "
        f"cells: {where['upper_end']} at the upper end, {where['lower_end']} at the lower end, {where['inside']} "
        "inside).",
        "holds: guaranteed margin >= 0, f_s - B_s > 0 and f_v + f_dv + B_v + B_dv < 0 (the signs are monotone in s and "
        "are taken at the ends of I(v), as stored). holds differ: (run, grid speed) cells where holds of the exact "
        "minimum differs from the stored one. runs holding at every grid speed (exact, scan): the number of runs whose "
        "certificate holds at all grid speeds (of the runs of the row).",
        f"Sweep: the exact minimum at {len(sweep)} speeds {first:.{decimals}f}, {first + step:.{decimals}f}, ..., "
        f"{last:.{decimals}f} m/s per run; sweep min: the smallest over the speeds and the runs (s⁻²); runs holding on the "
        "whole sweep: the number of runs whose certificate holds at every speed of the sweep (of the runs of the row).",
        "Existence: f_idm(s, 0, v) rises with s to a(1 - (v/v0)^δ). min a(1-(v/v0)^δ) - r_max (m/s²): the smallest "
        "over the runs and grid speeds. Where it is positive I(v) has a finite upper end s_high with f_idm = r_max "
        "there (f_idm = -r_max at s_low), so every residual with \\|r\\| <= r_max gives the hybrid an equilibrium in "
        "I(v) (intermediate values) and none outside; where it is <= 0 I(v) is unbounded and no equilibrium need "
        "exist (f = 0.1 - 144/s², r = -0.2, r_max = 0.3: f + r < 0 at every spacing). I(v) within [1, 200] m: share of "
        "the (run, grid speed) cells whose I(v) before the clamping to [1, 200] m lies in it (then the certificate "
        "covers every equilibrium and one exists). min s_low, max s_high (m): the smallest lower and largest upper end "
        "of I(v) before the clamping (max s_high blank when some I(v) has no upper end: inf in the CSV). Both ends rise with v, so the end speeds of the grid give "
        "the extremes of the sweep too. I(v) outside [1, 200] m: the grid speeds where it is not within, with the "
        "number of runs.",
        f"Check: the scan recomputed from the stored inputs (_a_priori, n_scan of the file) equals the stored "
        f"a_priori_margin to {_sci(rescan) or 'n/a'} (largest absolute difference over all runs and grid speeds): the "
        "stored core, residual and δ = 4 are those of the certificates. Per run (core, bounds, the minima and the "
        f"speeds where they are attained, the existence quantities): per_run/{NAME}_runs.csv.",
    ]
    if problems:
        notes.append(f"Problems: {len(problems)} (first: {problems[0]}).")
    title = "Exact a priori certificate and existence of the equilibria (M9)"
    return markdown(Table(NAME, title, notes, shown, columns))


def _skipped_text(skipped: Sequence[str]) -> str:
    """``25 x no a_priori entry (model mlp)``, grouped by reason."""
    reasons: dict[str, int] = {}
    for line in skipped:
        reason = line.split(": ", 1)[-1]
        reasons[reason] = reasons.get(reason, 0) + 1
    return "; ".join(f"{n} x {reason}" for reason, n in reasons.items()) or "none"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/_tables/m9", help=f"directory of {NAME}.{{csv,md}} and {NAME}_runs.csv")
    parser.add_argument("--sweep-step", type=float, default=SWEEP[2], help="m/s; 1 / step must be an integer")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    runs_root = resolve_path(args.runs_root)
    sweep = sweep_speeds(SWEEP[0], SWEEP[1], args.sweep_step)
    runs, cells, problems, skipped = collect(runs_root, sweep)
    table = summarise(runs, cells) if len(runs) else pd.DataFrame()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{NAME}.csv", index=False)
    (out / "per_run").mkdir(exist_ok=True)
    runs.to_csv(out / "per_run" / f"{NAME}_runs.csv", index=False)
    (out / f"{NAME}.md").write_text(markdown_text(table, runs, problems, skipped, sweep), encoding="utf-8")
    for row in table.to_dict("records"):
        print(f"  {row['experiment']:<24} {row['runs']:>3} runs  exact min {row['exact_min']:+.4f}  scan min "
              f"{row['scan_min']:+.4f}  max|diff| {_sci(row['max_abs_diff'])}  holds {_of(row, 'runs_holding_exact')}  "
              f"sweep {row['sweep_min']:+.4f} ({_of(row, 'runs_holding_sweep')})  "
              f"limit {row['limit_margin_min']:+.3f}  within {row['within_share']:.3f}")  # fmt: skip
    for line in problems[:20]:
        print(f"  problem: {line}")
    print(f"{NAME}: {len(runs)} runs, {len(skipped)} files left out, {len(problems)} problems, "
          f"{time.perf_counter() - t_start:.0f} s -> {out / (NAME + '.csv')}, .md, {NAME}_runs.csv")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
