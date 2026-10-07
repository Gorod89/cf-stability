"""Band sensitivity of the audit (docs/m8_contract.md, section 6; D119).

    python scripts/analysis/band_sensitivity.py                 # the table, from the audits of the three bands
    python scripts/analysis/band_sensitivity.py --check         # first: do other bands move the equilibria?

The spacing band of D78 takes the 5/50/95 % quantiles of the near-steady training samples per grid
speed. For every E1 run of the learned models (mlp, pidl, residual_idm, gru, lstm, perl; 150 runs) the
table compares the band shares of the audit (``band_numerical["support"]``: stable / unstable inside the
band, outside, none) and the H1.1 statistic (unstable among the equilibria, ``share_unstable_numerical
["support"]``, D92) for the bands 1/99, 5/95 (``stability.json``) and 10/90 (``stability_q01_99.json``,
``stability_q10_90.json`` of ``configs/queue/audit_bands.yaml``). Per architecture and band: the mean over
the runs with its 95 % percentile bootstrap interval over the runs (1000 resamples, seed 0), the change of
the H1.1 statistic against 5/95 paired over the runs, and whether the band-dependent part of the H1.1 rule
holds (share >= 0.5 with the lower end > 0.3, the thresholds of ``cf_stability/eval/tables.py``). Writes
``runs/_tables/m8/band_sensitivity.{csv,md}``.

``--check`` answers whether the stored audits could simply be reclassified: for every run it rebuilds the
three bands from the training events of the run (``tensors.training_context``, as the training computed
the band of D78), searches the equilibria of the model with each band (``find_equilibria``, as the audit
does) and counts the speeds in support where the equilibrium of another band is not the stored one (the
anchored search of D79 takes the first upward crossing inside the band, else the first one of
[1, 200] m). Every such speed needs a new audit. Prints the counts; writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.stats import bootstrap_ci  # noqa: E402
from cf_stability.eval.tables import DEFAULT_VERDICTS  # noqa: E402
from cf_stability.stability.audit import BAND_VARIANTS, band_output  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

ARCHITECTURES = ("mlp", "pidl", "residual_idm", "gru", "lstm", "perl")
REFERENCE = "5/95"
BANDS: dict[str, tuple[str, tuple[float, float, float]]] = {  # label -> (audit file of a run, quantiles)
    "1/99": (band_output(BAND_VARIANTS["q01_99"]), BAND_VARIANTS["q01_99"]),
    REFERENCE: ("stability.json", (0.05, 0.5, 0.95)),
    "10/90": (band_output(BAND_VARIANTS["q10_90"]), BAND_VARIANTS["q10_90"]),
}
SHARES = ("stable", "unstable", "outside", "none", "indifferent", "undefined")
STATISTICS = (*SHARES, "unstable_eq")  # unstable_eq: the H1.1 statistic (unstable among the equilibria)
N_RESAMPLES, SEED = 1000, 0


def run_directories(runs_root: Path, experiment: str, data: str, architecture: str) -> list[Path]:
    return sorted(p.parent for p in (runs_root / experiment / data / architecture).glob("*/model.pt"))


def audit_values(run_dir: Path, band: str) -> dict[str, float] | str:
    """The shares of one audit of a run, or the reason why there are none."""
    name, quantiles = BANDS[band]
    path = run_dir / name
    if not path.is_file():
        return f"{name} missing"
    payload = read_json(path)
    if payload.get("error") or "audit" not in payload:
        return f"{name}: {str(payload.get('error', 'no audit')).splitlines()[0]}"
    if band != REFERENCE:  # the band of the file must be the one of the label
        used = (payload.get("band_override") or {}).get("quantiles")
        if used is None or not np.allclose(used, quantiles):
            return f"{name}: band quantiles {used}, expected {list(quantiles)}"
    if (run_dir / "model.pt").stat().st_mtime > path.stat().st_mtime:
        return f"{name} older than model.pt"
    summary = payload["audit"]["summary"]
    shares = summary["band_numerical"]["support"]
    if shares is None:
        return f"{name}: no speed in support has a band"
    unstable_eq = summary["share_unstable_numerical"]["support"]
    return {**{key: float(shares[key]) for key in SHARES}, "unstable_eq": unstable_eq}


def collect(
    runs_root: Path, experiment: str, data: str, architectures: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """One row per run and band with the shares; the problems (missing or broken audits)."""
    rows, problems = [], []
    for architecture in architectures:
        for run_dir in run_directories(runs_root, experiment, data, architecture):
            for band in BANDS:
                values = audit_values(run_dir, band)
                if isinstance(values, str):
                    problems.append(f"{architecture}/{run_dir.name}: {values}")
                    continue
                rows.append({"architecture": architecture, "run": run_dir.name, "band": band, **values})
    columns = ["architecture", "run", "band", *STATISTICS]
    return pd.DataFrame(rows, columns=columns), problems


def summarise(runs: pd.DataFrame, architectures: Sequence[str], expected: Mapping[str, int]) -> pd.DataFrame:
    """Per architecture and band: means with bootstrap intervals over the runs, the paired change of the H1.1
    statistic against the reference band, the band-dependent part of the H1.1 rule."""
    out = []
    low_min, share_min = DEFAULT_VERDICTS["h1_1_unstable_low_min"], DEFAULT_VERDICTS["h1_1_unstable_min"]
    for architecture in architectures:
        mine = runs[runs.architecture == architecture]
        reference = mine[mine.band == REFERENCE].set_index("run")
        for band, (_, quantiles) in BANDS.items():
            part = mine[mine.band == band].set_index("run")
            row: dict[str, Any] = {
                "architecture": architecture, "band": band, "quantiles": "/".join(f"{q:g}" for q in quantiles),
                "runs": len(part), "runs_expected": expected.get(architecture, 0),
            }  # fmt: skip
            for key in STATISTICS:
                ci = bootstrap_ci(part[key].to_numpy(dtype=np.float64), part.index, n_resamples=N_RESAMPLES, seed=SEED)
                row.update({key: ci["estimate"], f"{key}_low": ci["low"], f"{key}_high": ci["high"]})
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
            out.append(row)
    return pd.DataFrame(out)


def _ci(row: Mapping[str, Any], key: str, digits: int = 2, signed: bool = False) -> str:
    value, low, high = (row[key], row[f"{key}_low"], row[f"{key}_high"])
    if not np.isfinite(value):
        return "n/a"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return f"{value:{spec}} [{low:{spec}}, {high:{spec}}]" if np.isfinite(low) else f"{value:{spec}}"


def markdown(table: pd.DataFrame, problems: Sequence[str], experiment: str, data: str) -> str:
    lines = [
        "# Band sensitivity of the audit (D119)",
        "",
        f"- Runs: {experiment} on {data}, learned models, five folds x five seeds; one audit per band "
        "(1/99: stability_q01_99.json, 5/95: stability.json, the band of D78, 10/90: stability_q10_90.json).",
        "- Band: quantiles of the spacing of the near-steady training samples (|dv| < 0.5 m/s, |a| < 0.3 m/s^2, "
        "speed within 0.5 m/s) per grid speed, computed with cf_stability/train/tensors.py; the equilibria are "
        "searched inside the band first (D79), so another band can move the equilibrium itself, not only its "
        "status.",
        "- stable, unstable, outside, none: band_numerical of the grid speeds in support (numerical rule); "
        "H1.1 = unstable among the equilibria (share_unstable_numerical, D92): the equilibrium inside the band "
        "where there is one, the first one of [1, 200] m otherwise.",
        "- Unit run; mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 "
        "resamples, seed 0). Change vs 5/95: the H1.1 statistic minus that of the same run with the band of "
        "D78, mean over the runs with its interval.",
        f"- H1.1 share rule: the band-dependent part of the verdict of H1.1, share >= "
        f"{DEFAULT_VERDICTS['h1_1_unstable_min']:g} with the lower end > {DEFAULT_VERDICTS['h1_1_unstable_low_min']:g} "
        "(cf_stability/eval/tables.py; the RMSE part does not depend on the band).",
    ]
    if problems:
        lines.append(f"- Missing or unusable audits: {len(problems)} (first: {problems[0]}).")
    lines += [
        "",
        "| architecture | band | runs | stable | unstable | outside | none | H1.1: unstable among equilibria | "
        "change vs 5/95 | H1.1 share rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        change = "" if row["band"] == REFERENCE else _ci(row, "unstable_eq_change", signed=True)
        lines.append(
            f"| {row['architecture']} | {row['band']} | {row['runs']} | {_ci(row, 'stable')} | "
            f"{_ci(row, 'unstable')} | "
            f"{_ci(row, 'outside')} | {_ci(row, 'none')} | {_ci(row, 'unstable_eq')} | {change} | "
            f"{row['h1_1_share_rule']} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- --check


def equilibrium_shifts(runs_root: Path, experiment: str, data: str, architectures: Sequence[str]) -> Counter:
    """Speeds in support whose equilibrium under another band differs from the one the stored audit analysed."""
    import torch

    from cf_stability.data.views import run_training_events
    from cf_stability.models import load_model
    from cf_stability.stability.equilibrium import V_GRID, Band, find_equilibria
    from cf_stability.train.tensors import BandConfig, training_context

    bands: dict[tuple[str, str], dict[str, Any] | None] = {}  # (training events, band) -> band of the context
    counts: Counter = Counter()
    for architecture in architectures:
        for run_dir in run_directories(runs_root, experiment, data, architecture):
            metrics, audit = read_json(run_dir / "metrics.json"), read_json(run_dir / "stability.json")["audit"]
            config = metrics["config"]
            events_key = repr((config["data"], config["split"], config["fold"], config.get("max_train_events"),
                               config["seed"] if config.get("max_train_events") is not None else None))  # fmt: skip
            model = copy.deepcopy(load_model(run_dir / "model.pt")).double().cpu().eval()
            settings = metrics["train_config"]["band"]
            for label, (_, quantiles) in BANDS.items():
                key = (events_key, label)
                if key not in bands:
                    band_cfg = BandConfig.from_mapping({**settings, "quantiles": list(quantiles)})
                    bands[key] = training_context(run_training_events(config), int(config["seed"]), band_cfg)["band"]
                if label == REFERENCE and bands[key] != metrics["context"]["band"]:
                    counts[architecture, label, "band differs from the stored one"] += 1
                band = None if bands[key] is None else Band.from_mapping(bands[key], "cpu", torch.float64)
                eq = find_equilibria(model, V_GRID, band=band)
                for record, status, s in zip(audit["equilibria"], eq.status, eq.s.tolist()):
                    if not record["in_support"]:
                        continue
                    same = record["s"] is not None and abs(s - record["s"]) <= 1e-6 * max(1.0, abs(s))
                    if status == "none" or same:
                        counts[architecture, label, "kept"] += 1
                    else:
                        counts[architecture, label, "moved"] += 1
                        counts[architecture, label, f"runs:{run_dir.name}"] = 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("--data", default="follownet_highd")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--out", default="runs/_tables/m8", help="directory of band_sensitivity.{csv,md}")
    parser.add_argument("--check", action="store_true", help="count the speeds whose equilibrium moves; no table")
    args = parser.parse_args(argv)
    runs_root = resolve_path(args.runs_root)
    if args.check:
        counts = equilibrium_shifts(runs_root, args.experiment, args.data, args.architectures)
        for architecture in args.architectures:
            for label in BANDS:
                kept, moved = counts[architecture, label, "kept"], counts[architecture, label, "moved"]
                runs = sum(1 for key in counts if key[:2] == (architecture, label) and str(key[2]).startswith("runs:"))
                extra = counts[architecture, label, "band differs from the stored one"]
                note = f"  ({extra} runs: recomputed 5/95 band differs from the stored one)" if extra else ""
                print(f"{architecture:<13} {label:>5}: equilibrium kept at {kept} speeds, moved at {moved} speeds "
                      f"of {runs} runs{note}")  # fmt: skip
        moved = sum(v for k, v in counts.items() if k[2] == "moved")
        print(f"reclassification of the stored audits is {'exact' if moved == 0 else 'NOT exact'}: {moved} speeds move")
        return 0
    expected = {a: len(run_directories(runs_root, args.experiment, args.data, a)) for a in args.architectures}
    runs, problems = collect(runs_root, args.experiment, args.data, args.architectures)
    table = summarise(runs, args.architectures, expected)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "band_sensitivity.csv", index=False)
    (out / "band_sensitivity.md").write_text(markdown(table, problems, args.experiment, args.data), encoding="utf-8")
    for line in problems[:20]:
        print(f"  missing: {line}")
    if len(problems) > 20:
        print(f"  ... {len(problems) - 20} more")
    print(f"band_sensitivity: {len(runs)} audits of {sum(expected.values())} runs x {len(BANDS)} bands, "
          f"{len(problems)} missing or unusable -> {out / 'band_sensitivity.csv'}, .md")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
