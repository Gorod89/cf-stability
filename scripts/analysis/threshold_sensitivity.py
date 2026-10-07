"""Threshold sensitivity of the audit's numerical rule (M9).

    python scripts/analysis/threshold_sensitivity.py            # runs/_tables/m9/threshold_sensitivity.{csv,md}

The audit calls an equilibrium string unstable when the largest numerical gain over its 25 frequencies
exceeds 1.02 (``FrequencyConfig.threshold``, strict: ``max_gain > threshold``). The stored audits keep the
gain of every speed and frequency, so the verdicts at other thresholds follow without new rollouts: every
record of an audit with a defined maximum gain gets ``unstable = max_gain > threshold``, a record without
one (no equilibrium, or a rollout that broke down: NaN gain) keeps ``unstable = None``, and the summary is
recomputed by the audit's own ``summarise_audit``. For every E1 run (every model of E1: the six learned
ones five folds x five seeds, the five baselines five folds x seed 0; 175 runs) and the thresholds 1.00,
1.01, 1.02 (the reference) and 1.05: the H1.1 statistic (``share_unstable_numerical["support"]``: the
string-unstable equilibria among the equilibria the audit analysed, inside the band or outside it, speeds in
support; D92) and the band shares (``band_numerical["support"]``: unstable, not stable = 1 - stable, and the
others). Per architecture and threshold: the mean over the runs with its 95 % percentile bootstrap interval
over the runs (1000 resamples, seed 0), the change of the H1.1 statistic against 1.02 paired over the runs,
and the band-dependent part of the H1.1 rule (share >= 0.5 with the lower end > 0.3). At 1.02 the recomputed
summaries must equal the stored ones (checked and reported).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.stats import bootstrap_ci  # noqa: E402
from cf_stability.eval.tables import DEFAULT_VERDICTS  # noqa: E402
from cf_stability.stability.audit import summarise_audit  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

ARCHITECTURES = ("mlp", "pidl", "residual_idm", "gru", "lstm", "perl", "idm", "knn", "newell", "ovm", "persistence")
THRESHOLDS = (1.00, 1.01, 1.02, 1.05)
REFERENCE = 1.02
AUDIT = "stability.json"
BAND = ("stable", "unstable", "outside", "none", "indifferent", "undefined")
STATISTICS = ("unstable_eq", "band_unstable", "band_not_stable", *(f"band_{key}" for key in BAND if key != "unstable"))
WITH_CI = ("unstable_eq", "band_unstable", "band_not_stable")
N_RESAMPLES, SEED = 1000, 0


def run_directories(runs_root: Path, experiment: str, data: str, architecture: str) -> list[Path]:
    return sorted(p.parent for p in (runs_root / experiment / data / architecture).glob("*/model.pt"))


def reclassified(audit: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    """``audit`` with the numerical flag of every record recomputed at ``threshold`` (the rule of the audit,
    ``max_gain > threshold``; None where the maximum gain is undefined) and its summary recomputed."""
    out = copy.deepcopy(dict(audit))
    for record in out["equilibria"]:
        max_gain = record.get("max_gain")
        finite = max_gain is not None and math.isfinite(float(max_gain))
        record["unstable"] = bool(float(max_gain) > threshold) if finite else None
    out["summary"] = summarise_audit(out)
    return out


def statistics(summary: Mapping[str, Any]) -> dict[str, float]:
    """The H1.1 statistic and the band shares of the speeds in support (NaN where the audit has none)."""
    shares = (summary.get("band_numerical") or {}).get("support")
    h11 = (summary.get("share_unstable_numerical") or {}).get("support")
    out = {"unstable_eq": np.nan if h11 is None else float(h11)}
    for key in BAND:
        out[f"band_{key}"] = np.nan if shares is None else float(shares[key])
    out["band_not_stable"] = np.nan if shares is None else 1.0 - float(shares["stable"])
    return out


def count_flagged(audit: Mapping[str, Any], threshold: float) -> int:
    """Equilibria in support with a defined maximum gain above ``threshold`` (the speeds that the rule flags)."""
    return sum(
        1 for r in audit["equilibria"]
        if r.get("in_support") and r.get("status") != "none" and r.get("max_gain") is not None
        and float(r["max_gain"]) > threshold
    )  # fmt: skip


def audit_values(run_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Per threshold the statistics of one run (and the reproduction error at the reference), or the problem."""
    path = run_dir / AUDIT
    if not path.is_file():
        return [], f"{AUDIT} missing"
    payload = read_json(path)
    if payload.get("error") or not isinstance(payload.get("audit"), Mapping):
        return [], f"{AUDIT}: {str(payload.get('error', 'no audit')).splitlines()[0]}"
    if (run_dir / "model.pt").stat().st_mtime > path.stat().st_mtime:
        return [], f"{AUDIT} older than model.pt"
    audit = payload["audit"]
    stored_threshold = float(((audit.get("config") or {}).get("frequency") or {}).get("threshold", REFERENCE))
    stored = statistics(audit["summary"])
    exact = sum(1 for r in audit["equilibria"] if r.get("max_gain") is not None
                and any(float(r["max_gain"]) == t for t in THRESHOLDS))  # fmt: skip
    rows = []
    for threshold in THRESHOLDS:
        values = statistics(reclassified(audit, threshold)["summary"])
        row = {"threshold": threshold, **values, "flagged": count_flagged(audit, threshold),
               "stored_threshold": stored_threshold, "gains_at_a_threshold": exact}  # fmt: skip
        if threshold == stored_threshold:
            diffs = [abs(values[k] - stored[k]) for k in values if np.isfinite(values[k]) or np.isfinite(stored[k])]
            row["reproduction_error"] = max((d if np.isfinite(d) else np.inf for d in diffs), default=0.0)
        rows.append(row)
    return rows, None


def collect(
    runs_root: Path, experiment: str, data: str, architectures: Sequence[str]
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """One row per run and threshold; the problems; the number of runs per architecture."""
    rows, problems, expected = [], [], {}
    for architecture in architectures:
        runs = run_directories(runs_root, experiment, data, architecture)
        expected[architecture] = len(runs)
        for run_dir in runs:
            values, problem = audit_values(run_dir)
            if problem:
                problems.append(f"{architecture}/{run_dir.name}: {problem}")
                continue
            rows += [{"architecture": architecture, "run": run_dir.name, **row} for row in values]
    columns = ["architecture", "run", "threshold", *STATISTICS, "flagged", "stored_threshold", "gains_at_a_threshold",
               "reproduction_error"]  # fmt: skip
    return pd.DataFrame(rows, columns=columns), problems, expected


def summarise(runs: pd.DataFrame, architectures: Sequence[str], expected: Mapping[str, int]) -> pd.DataFrame:
    """Per architecture and threshold: means with bootstrap intervals over the runs, the paired change of the H1.1
    statistic against the reference, the band-dependent part of the H1.1 rule, the flagged equilibria."""
    out = []
    low_min, share_min = DEFAULT_VERDICTS["h1_1_unstable_low_min"], DEFAULT_VERDICTS["h1_1_unstable_min"]
    for architecture in architectures:
        mine = runs[runs.architecture == architecture]
        reference = mine[np.isclose(mine.threshold, REFERENCE)].set_index("run")
        for threshold in THRESHOLDS:
            part = mine[np.isclose(mine.threshold, threshold)].set_index("run")
            row: dict[str, Any] = {"architecture": architecture, "threshold": threshold, "runs": len(part),
                                   "runs_expected": expected.get(architecture, 0),
                                   "flagged": int(part["flagged"].sum())}  # fmt: skip
            for key in STATISTICS:
                values = part[key].to_numpy(dtype=np.float64)
                if key in WITH_CI:
                    ci = bootstrap_ci(values, part.index, n_resamples=N_RESAMPLES, seed=SEED)
                    row.update({key: ci["estimate"], f"{key}_low": ci["low"], f"{key}_high": ci["high"]})
                else:
                    row[key] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            common = part.index.intersection(reference.index)
            change = (part.loc[common, "unstable_eq"] - reference.loc[common, "unstable_eq"]).to_numpy(float)
            ci = bootstrap_ci(change, common, n_resamples=N_RESAMPLES, seed=SEED)
            row.update({"unstable_eq_change": ci["estimate"], "unstable_eq_change_low": ci["low"],
                        "unstable_eq_change_high": ci["high"], "pairs": int(np.isfinite(change).sum())})  # fmt: skip
            if np.isfinite(row["unstable_eq"]) and np.isfinite(row["unstable_eq_low"]):
                met = row["unstable_eq"] >= share_min and row["unstable_eq_low"] > low_min
                row["h1_1_share_rule"] = "met" if met else "not met"
            else:
                row["h1_1_share_rule"] = ""
            errors = pd.to_numeric(part.get("reproduction_error"), errors="coerce")
            row["reproduction_error"] = float(errors.max()) if errors.notna().any() else np.nan
            out.append(row)
    return pd.DataFrame(out)


def _ci(row: Mapping[str, Any], key: str, digits: int = 2, signed: bool = False) -> str:
    value, low, high = row[key], row[f"{key}_low"], row[f"{key}_high"]
    if not np.isfinite(value):
        return "n/a"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return f"{value:{spec}} [{low:{spec}}, {high:{spec}}]" if np.isfinite(low) else f"{value:{spec}}"


def markdown(table: pd.DataFrame, runs: pd.DataFrame, problems: Sequence[str], experiment: str, data: str) -> str:
    error = pd.to_numeric(runs["reproduction_error"], errors="coerce") if len(runs) else pd.Series(dtype=float)
    reproduced = int((error <= 1e-12).sum())
    checked = int(error.notna().sum())
    at_threshold = int(runs.drop_duplicates(["architecture", "run"])["gains_at_a_threshold"].sum()) if len(runs) else 0
    stored = sorted(set(runs["stored_threshold"])) if len(runs) else []
    lines = [
        "# Threshold sensitivity of the audit's numerical rule (M9)",
        "",
        f"- Runs: every E1 run on {data} ({experiment}: the learned models five folds x five seeds, the baselines "
        f"five folds x seed 0); the stored audits (stability.json, threshold {', '.join(f'{t:g}' for t in stored)}), "
        "no new rollouts. Per run and threshold the numerical flag of every equilibrium is recomputed from the stored "
        "largest gain over the 25 audit frequencies and the summary by the audit's own summarise_audit.",
        "- Rule: unstable = (largest gain > threshold), a strict inequality as in the audit "
        "(FrequencyConfig.threshold): a largest gain exactly at the threshold counts as not unstable (stable). A "
        "speed whose largest gain is "
        "undefined (no equilibrium; a rollout that broke down, NaN gain) has no flag: it is left out of the H1.1 "
        "share and counted as undefined in the band shares. Largest gains of the stored audits exactly equal to "
        f"one of the thresholds: {at_threshold}.",
        "- H1.1: string-unstable equilibria among the equilibria the audit analysed, inside the band or outside it, "
        "speeds in support (share_unstable_numerical, D92). Band shares: band_numerical of the grid speeds in support "
        "(unstable inside the band; not stable = 1 - stable; outside and none do not depend on the threshold). "
        "flagged: equilibria in support above the threshold, summed over the runs.",
        "- Unit run; mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, "
        f"seed 0). Change vs {REFERENCE:g}: the H1.1 statistic minus that of the same run at {REFERENCE:g}, mean over "
        "the runs with its interval. H1.1 share rule: share >= "
        f"{DEFAULT_VERDICTS['h1_1_unstable_min']:g} with the lower end > {DEFAULT_VERDICTS['h1_1_unstable_low_min']:g} "
        "(cf_stability/eval/tables.py; the verdict of H1.1 concerns mlp, gru and lstm and also needs the RMSE part).",
        f"- Check: at the stored threshold the recomputed H1.1 statistic and band shares equal the stored summaries "
        f"for {reproduced} of {checked} runs (largest difference {error.max() if checked else float('nan'):.1e}).",
    ]
    if problems:
        lines.append(f"- Missing or unusable audits: {len(problems)} (first: {problems[0]}).")
    lines += [
        "",
        f"| architecture | threshold | runs | H1.1: unstable among equilibria | change vs {REFERENCE:g} | "
        "band unstable | band not stable | stable | outside | none | flagged | H1.1 share rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        change = "" if np.isclose(row["threshold"], REFERENCE) else _ci(row, "unstable_eq_change", signed=True)
        lines.append(
            f"| {row['architecture']} | {row['threshold']:.2f} | {row['runs']} | {_ci(row, 'unstable_eq')} | "
            f"{change} | {_ci(row, 'band_unstable')} | {_ci(row, 'band_not_stable')} | {row['band_stable']:.2f} | "
            f"{row['band_outside']:.2f} | {row['band_none']:.2f} | {row['flagged']} | {row['h1_1_share_rule']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("--data", default="follownet_highd")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--out", default="runs/_tables/m9", help="directory of threshold_sensitivity.{csv,md}")
    args = parser.parse_args(argv)
    runs_root = resolve_path(args.runs_root)
    runs, problems, expected = collect(runs_root, args.experiment, args.data, args.architectures)
    table = summarise(runs, args.architectures, expected)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "threshold_sensitivity.csv", index=False)
    text = markdown(table, runs, problems, args.experiment, args.data)
    (out / "threshold_sensitivity.md").write_text(text, encoding="utf-8")
    for line in problems[:20]:
        print(f"  missing: {line}")
    n_runs = len(runs.drop_duplicates(["architecture", "run"])) if len(runs) else 0
    print(f"threshold_sensitivity: {n_runs} runs of {sum(expected.values())} x {len(THRESHOLDS)} thresholds, "
          f"{len(problems)} missing or unusable -> {out / 'threshold_sensitivity.csv'}, .md")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
