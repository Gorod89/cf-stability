"""Multiple equilibria: the upward and downward zero crossings of f(s, 0, v) (M9).

    python scripts/analysis/equilibrium_roots.py               # runs/_tables/m9/equilibrium_roots.{csv,md}

The equilibrium finder (``find_equilibria``, docs/m3_contract.md, section 2) takes the first upward zero
crossing of ``f(s, 0, v)`` on 400 linearly spaced gaps of [1, 200] m (or inside the spacing band first,
D79). This script asks whether that root is the only one. For every E1 run of the learned models (mlp,
pidl, residual_idm, gru, lstm, perl; five folds x five seeds, 150 runs) and every speed of the grid
(5-30 m/s) it evaluates ``f(s, 0, v)`` of the memoryless view (the window filled with the constant state,
as the finder does; float64 on the CPU) on 400 log-spaced gaps over [1, 200] m and counts the sign changes
between neighbouring gaps: upward crossings (``f < 0`` then ``f >= 0``, the finder's convention) and
downward crossings (``f >= 0`` then ``f < 0``). Per run the shares of the grid speeds with 0, 1, 2 or more
upward crossings and with any downward crossing, and, among the speeds where the stored audit has an
equilibrium (status ok, multiple or outside), the share where that equilibrium is the only upward crossing
of the scan (exactly one, and its bracket contains the audit's spacing). Per architecture and part of the
speeds (all, in support): the mean over the runs with its 95 % percentile bootstrap interval over the runs
(1000 resamples, seed 0).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cf_stability.eval.stats import bootstrap_ci  # noqa: E402
from cf_stability.models import load_model  # noqa: E402
from cf_stability.models.base import CFModel  # noqa: E402
from cf_stability.stability.equilibrium import V_GRID, steady_acc  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

ARCHITECTURES = ("mlp", "pidl", "residual_idm", "gru", "lstm", "perl")
S_MIN, S_MAX, N_GAPS = 1.0, 200.0, 400
AUDIT = "stability.json"
EQUILIBRIUM = ("ok", "multiple", "outside")  # statuses with an equilibrium (FOUND of cf_stability.stability.audit)
PARTS = ("all", "support")
SHARES = ("up_0", "up_1", "up_2plus", "any_down", "unique")  # unique: among the speeds with an audit equilibrium
# down_below: a downward crossing below the first upward one (or without any: the law accelerates at small gaps);
# down_above: one above it (the law brakes again at larger gaps)
DIAGNOSTICS = ("down_below", "down_above", "not_bracketed", "audit_none_scan_up")
N_RESAMPLES, SEED = 1000, 0
CHUNK = 4096  # constant histories per call of the model


def gap_grid(n: int = N_GAPS, s_min: float = S_MIN, s_max: float = S_MAX) -> np.ndarray:
    return np.geomspace(s_min, s_max, n)


@torch.no_grad()
def scan(model: CFModel, speeds: Sequence[float], gaps: np.ndarray) -> np.ndarray:
    """``f(s, 0, v)`` of the constant history ``[n_speeds, n_gaps]`` (float64, the model as given)."""
    s = torch.as_tensor(np.tile(gaps, len(speeds)), dtype=torch.float64)
    v = torch.as_tensor(np.repeat(np.asarray(speeds, dtype=np.float64), len(gaps)), dtype=torch.float64)
    with torch.backends.cudnn.flags(enabled=False):
        f = torch.cat([steady_acc(model, s[k : k + CHUNK], v[k : k + CHUNK]) for k in range(0, len(s), CHUNK)])
    return f.double().cpu().numpy().reshape(len(speeds), len(gaps))


def crossings(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks ``[n, m - 1]`` of the upward (``f < 0`` then ``f >= 0``) and downward (``f >= 0`` then ``f < 0``) sign
    changes between neighbouring gaps."""
    negative = f < 0.0
    return negative[:, :-1] & ~negative[:, 1:], ~negative[:, :-1] & negative[:, 1:]


def speed_records(model: CFModel, audit: Mapping[str, Any], gaps: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Per grid speed of the audit: the numbers of upward and downward crossings of the scan, their brackets, the
    audit's status and spacing and whether that spacing is the only upward crossing (or lies in none)."""
    gaps = gap_grid() if gaps is None else gaps
    records = audit["equilibria"]
    f = scan(model, [float(r["v"]) for r in records], gaps)
    up, down = crossings(f)
    out = []
    for k, r in enumerate(records):
        ups, downs = np.flatnonzero(up[k]), np.flatnonzero(down[k])
        brackets = [(float(gaps[i]), float(gaps[i + 1])) for i in ups]
        first_up = ups[0] if len(ups) else len(gaps)
        s_audit = r.get("s") if r.get("status") in EQUILIBRIUM else None
        tolerance = 1e-9
        inside = None if s_audit is None else [lo - tolerance <= s_audit <= hi + tolerance for lo, hi in brackets]
        out.append({
            "v": float(r["v"]), "status": r.get("status"), "in_support": r.get("in_support"), "s_audit": s_audit,
            "n_up": len(brackets), "n_down": len(downs), "up_brackets": brackets,
            "down_below": bool((downs < first_up).any()), "down_above": bool((downs > first_up).any()),
            "unique": None if s_audit is None else len(brackets) == 1 and inside[0],
            "bracketed": None if s_audit is None else any(inside),
            "f_first": float(f[k, 0]), "f_last": float(f[k, -1]),
        })  # fmt: skip
    return out


def shares(records: Sequence[Mapping[str, Any]], part: str) -> dict[str, Any]:
    """Shares of the speeds of ``part`` (``all`` or ``support``) with 0, 1, 2+ upward crossings and any downward one;
    among those with an audit equilibrium the unique ones and those not bracketed by an upward crossing; among the
    speeds without one in the audit (status none) those where the scan finds an upward crossing."""
    rows = [r for r in records if part == "all" or r.get("in_support")]
    n = len(rows)
    with_eq = [r for r in rows if r["s_audit"] is not None]
    none = [r for r in rows if r["status"] == "none"]

    def share(values: Sequence[bool], total: int) -> float:
        return sum(values) / total if total else np.nan

    return {
        "n_speeds": n, "n_audit_equilibria": len(with_eq), "n_audit_none": len(none),
        "up_0": share([r["n_up"] == 0 for r in rows], n), "up_1": share([r["n_up"] == 1 for r in rows], n),
        "up_2plus": share([r["n_up"] >= 2 for r in rows], n), "any_down": share([r["n_down"] > 0 for r in rows], n),
        "down_below": share([r["down_below"] for r in rows], n),
        "down_above": share([r["down_above"] for r in rows], n),
        "unique": share([bool(r["unique"]) for r in with_eq], len(with_eq)),
        "not_bracketed": share([not r["bracketed"] for r in with_eq], len(with_eq)),
        "audit_none_scan_up": share([r["n_up"] > 0 for r in none], len(none)),
    }  # fmt: skip


def run_directories(runs_root: Path, experiment: str, data: str, architecture: str) -> list[Path]:
    return sorted(p.parent for p in (runs_root / experiment / data / architecture).glob("*/model.pt"))


def run_values(run_dir: Path, gaps: np.ndarray | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """The per-speed records of one run, or the reason why its audit cannot be used."""
    path = run_dir / AUDIT
    if not path.is_file():
        return [], f"{AUDIT} missing"
    payload = read_json(path)
    if payload.get("error") or not isinstance(payload.get("audit"), Mapping):
        return [], f"{AUDIT}: {str(payload.get('error', 'no audit')).splitlines()[0]}"
    if (run_dir / "model.pt").stat().st_mtime > path.stat().st_mtime:
        return [], f"{AUDIT} older than model.pt"
    model = copy.deepcopy(load_model(run_dir / "model.pt")).double().cpu().eval()
    return speed_records(model, payload["audit"], gaps), None


def collect(
    runs_root: Path, experiment: str, data: str, architectures: Sequence[str], gaps: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    """One row per run and part with the shares, one row per run and speed, the problems, the runs per architecture."""
    rows, speeds, problems, expected = [], [], [], {}
    for architecture in architectures:
        runs = run_directories(runs_root, experiment, data, architecture)
        expected[architecture] = len(runs)
        for run_dir in runs:
            records, problem = run_values(run_dir, gaps)
            if problem:
                problems.append(f"{architecture}/{run_dir.name}: {problem}")
                continue
            for part in PARTS:
                rows.append({"architecture": architecture, "run": run_dir.name, "part": part, **shares(records, part)})
            speeds += [{"architecture": architecture, "run": run_dir.name,
                        **{k: v for k, v in r.items() if k != "up_brackets"}} for r in records]  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(speeds), problems, expected


def summarise(rows: pd.DataFrame, architectures: Sequence[str], expected: Mapping[str, int]) -> pd.DataFrame:
    """Per architecture and part: the mean over the runs of every share with its bootstrap interval over the runs."""
    out = []
    for architecture in architectures:
        for part in PARTS:
            group = rows[(rows.architecture == architecture) & (rows.part == part)] if len(rows) else rows
            row: dict[str, Any] = {
                "architecture": architecture, "part": part, "runs": len(group),
                "runs_expected": expected.get(architecture, 0),
                "speeds": int(group["n_speeds"].sum()) if len(group) else 0,
                "audit_equilibria": int(group["n_audit_equilibria"].sum()) if len(group) else 0,
            }  # fmt: skip
            for key in (*SHARES, *DIAGNOSTICS):
                values = group[key].to_numpy(dtype=np.float64) if len(group) else np.array([])
                ci = bootstrap_ci(values, group["run"].to_numpy() if len(group) else None, n_resamples=N_RESAMPLES,
                                  seed=SEED)  # fmt: skip
                row.update({key: ci["estimate"], f"{key}_low": ci["low"], f"{key}_high": ci["high"]})
            out.append(row)
    return pd.DataFrame(out)


def _ci(row: Mapping[str, Any], key: str, digits: int = 2) -> str:
    value, low, high = row[key], row[f"{key}_low"], row[f"{key}_high"]
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]" if np.isfinite(low) else f"{value:.{digits}f}"


def markdown(table: pd.DataFrame, problems: Sequence[str], experiment: str, data: str, gaps: np.ndarray) -> str:
    ratio = gaps[1] / gaps[0]
    lines = [
        "# Multiple equilibria: zero crossings of f(s, 0, v) (M9)",
        "",
        f"- Runs: {experiment} on {data}, the learned models, five folds x five seeds; every grid speed 5-30 m/s "
        "(part all) or those in the speed range of the training data (support). The model as the equilibrium finder "
        "evaluates it: f(s, 0, v) of the memoryless view (the window filled with the constant state), float64.",
        f"- Scan: {len(gaps)} log-spaced gaps over [{gaps[0]:g}, {gaps[-1]:g}] m (neighbours {100 * (ratio - 1):.2f} % "
        f"apart: {gaps[1] - gaps[0]:.3f} m at {gaps[0]:g} m, {gaps[-1] - gaps[-2]:.2f} m at {gaps[-1]:g} m; the finder "
        "scans 400 linearly spaced gaps, 0.50 m apart). Upward crossing: f < 0 at a gap and f >= 0 at the next (the "
        "finder's convention); downward: f >= 0 then f < 0. Two roots between neighbouring gaps are not seen.",
        "- up 0 / 1 / 2+: shares of the speeds with no, one, two or more upward crossings; any down: with at least "
        "one downward crossing, a root where f falls through zero (f_s < 0 there: an equilibrium that a gap "
        "disturbance moves away from). down below: a downward crossing below the first upward one, or without any "
        "upward one (f >= 0 at the smallest gaps: the law accelerates when closer than its equilibrium); down above: "
        "one above the first upward crossing (the law brakes again at larger gaps, f < 0 towards 200 m).",
        "- unique: among the speeds where the stored audit has an equilibrium (status ok, multiple or outside, inside "
        "or outside the band), the share where the scan has exactly one upward crossing and its bracket holds the "
        "audit's spacing. not bracketed: the audit's spacing lies in no upward bracket of the scan (two roots within "
        "one step of the scan, or a root seen by the finder's finer steps at large gaps). audit none, scan up: among "
        "the speeds without equilibrium in the audit (status none), those where the scan finds an upward crossing.",
        "- Unit run; mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, "
        "seed 0).",
    ]
    if problems:
        lines.append(f"- Missing or unusable audits: {len(problems)} (first: {problems[0]}).")
    lines += [
        "",
        "| architecture | part | runs | speeds | up 0 | up 1 | up 2+ | any down | down below | down above | "
        "audit equilibria | unique | not bracketed | audit none, scan up |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['architecture']} | {row['part']} | {row['runs']} | {row['speeds']} | {_ci(row, 'up_0')} | "
            f"{_ci(row, 'up_1')} | {_ci(row, 'up_2plus')} | {_ci(row, 'any_down')} | {_ci(row, 'down_below')} | "
            f"{_ci(row, 'down_above')} | {row['audit_equilibria']} | {_ci(row, 'unique')} | "
            f"{_ci(row, 'not_bracketed')} | {_ci(row, 'audit_none_scan_up')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("--data", default="follownet_highd")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--out", default="runs/_tables/m9", help="directory of equilibrium_roots.{csv,md}")
    args = parser.parse_args(argv)
    gaps = gap_grid()
    runs_root = resolve_path(args.runs_root)
    rows, speeds, problems, expected = collect(runs_root, args.experiment, args.data, args.architectures, gaps)
    table = summarise(rows, args.architectures, expected)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "equilibrium_roots.csv", index=False)
    text = markdown(table, problems, args.experiment, args.data, gaps)
    (out / "equilibrium_roots.md").write_text(text, encoding="utf-8")
    for line in problems[:20]:
        print(f"  missing: {line}")
    n_runs = len(rows[rows.part == "all"]) if len(rows) else 0
    print(f"equilibrium_roots: {n_runs} runs of {sum(expected.values())}, {len(problems)} missing or unusable -> "
          f"{out / 'equilibrium_roots.csv'}, .md")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
