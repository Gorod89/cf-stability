"""Long-window check of the windowed-gain penalty (M11, docs/m11_contract.md, D128; hypothesis H-N3).

    python scripts/analysis/long_window_check.py                 # long_window.json per run and the table
    python scripts/analysis/long_window_check.py --device cpu    # on the CPU (default: cuda when available)

For the fold-0, seed-0 runs of the pilot of the windowed-gain penalty (``e2_windowed_w{0.1,1,10}``: GRU, LSTM, PERL)
and, for reference, of E1, of the chosen weight of E2 (``e2_gain_w0.1``: the rollout gain penalty measured over the
last 20 s of 40 s) and of the long-window arm of the revision (``e2_gain_long_w0.1``, GRU and LSTM): the rollout gain
of the gain penalty (``rollout_gain`` of ``cf_stability/stability/penalties.py``) with the rollout of
``configs/queue/e2_horizon_pilot.yaml``, 380 s after the warm-up of one window at the equilibrium, the amplitude of the
follower speed fitted on sin, cos and 1 over the last 252 s (two whole periods of 0.05 rad/s), leader amplitude
0.2 m/s, at omega 0.02, 0.05 and 0.1 rad/s. Equilibria: those of the stored audit (``stability.json``: status ok,
multiple or outside, at the grid speeds), the model in float64 and in eval mode. The check of D128 holds for a run
when the gain stays below 1.02 at the three frequencies at every equilibrium inside the band at a grid speed in the
support of the training data (status ok or multiple: the speeds of the stable band share of the audit); the
equilibria outside the band are reported beside it. Writes ``long_window.json`` next to ``stability.json`` of the runs
of the new experiments only (``WRITE``: ``e2_windowed*``; the reference runs of E1 and E2 are left as they are), the
per-equilibrium results of every run to ``runs/_tables/m11/long_window_runs.json`` and the table
``runs/_tables/m11/long_window.{csv,md}`` (``cf_stability/eval/tables.py`` conventions).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cf_stability.eval.tables import Column, Table, write_table  # noqa: E402
from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.penalties import PenaltyConfig, rollout_gain  # noqa: E402
from cf_stability.utils import read_json, resolve_path, write_json  # noqa: E402

OUTPUT, AUDIT = "long_window.json", "stability.json"
ARCHITECTURES = ("gru", "lstm", "perl")
WEIGHTS = ("0.1", "1", "10")  # the pilot of the windowed-gain penalty (configs/queue/e2_windowed_pilot.yaml)
ARMS: dict[str, dict[str, str]] = {  # arm -> architecture -> experiment
    "E1": {a: "e1" for a in ARCHITECTURES},
    "E2 (gain 0.1, rollout 40 s)": {a: "e2_gain_w0.1" for a in ARCHITECTURES},
    "E2 long window (gain 0.1, rollout 380 s)": {"gru": "e2_gain_long_w0.1", "lstm": "e2_gain_long_w0.1"},
    **{f"windowed w{w}": {a: f"e2_windowed_w{w}" for a in ARCHITECTURES} for w in WEIGHTS},
}
RUN = "driver_fold0_seed0"
WRITE = ("e2_windowed",)  # experiments (prefixes) whose runs get long_window.json
OMEGAS = (0.02, 0.05, 0.1)  # rad/s
HORIZON_S, MEASURE_S, AMPLITUDE = 380.0, 252.0, 0.2  # as configs/queue/e2_horizon_pilot.yaml; m/s
THRESHOLD = 1.02
INSIDE, EQUILIBRIUM = ("ok", "multiple"), ("ok", "multiple", "outside")


def settings() -> PenaltyConfig:
    """The rollout of the check: the gain penalty with the long window at the three frequencies."""
    return PenaltyConfig(kind="gain", omegas=OMEGAS, horizon_s=HORIZON_S, measure_s=MEASURE_S, amplitude=AMPLITUDE)


def _num(x: Any) -> float | None:
    x = None if x is None else float(x)
    return x if x is not None and math.isfinite(x) else None


def check(model: Any, audit: Mapping[str, Any], device: torch.device, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Rollout gains of ``model`` (cast to float64 on ``device``, eval mode) at the equilibria of ``audit`` (the
    ``audit`` block of ``stability.json``): per equilibrium the gains at ``OMEGAS``, and the summary of the check."""
    t_start = time.perf_counter()
    records = [r for r in audit["equilibria"] if r.get("status") in EQUILIBRIUM and r.get("s") is not None]
    cfg = settings()
    out: list[dict[str, Any]] = []
    if records:
        model = copy.deepcopy(model).double().to(device).eval()
        s = torch.tensor([float(r["s"]) for r in records], dtype=torch.float64, device=device)
        v = torch.tensor([float(r["v"]) for r in records], dtype=torch.float64, device=device)
        with torch.no_grad():
            gain = rollout_gain(model, s, v, cfg).cpu()  # [n, 3]
        omega_audit = np.asarray(audit.get("omega") or [], dtype=float)
        low = omega_audit <= max(OMEGAS)
        for k, r in enumerate(records):
            gains = [_num(g) for g in gain[k].tolist()]
            numerical = r.get("gain") if isinstance(r.get("gain"), list) else None
            audit_low = None
            if numerical is not None and len(numerical) == len(omega_audit) and low.any():
                values = np.asarray([np.nan if g is None else g for g in numerical], dtype=float)[low]
                audit_low = _num(np.nanmax(values)) if np.isfinite(values).any() else None
            finite = [g for g in gains if g is not None]
            largest = max(finite) if len(finite) == len(gains) else None
            out.append({
                "v": float(r["v"]), "s": float(r["s"]), "status": r["status"], "in_support": bool(r.get("in_support")),
                "in_band": r.get("in_band"), "gain": gains, "max_gain": largest,
                "above": None if largest is None else largest > threshold, "audit_max_gain_low": audit_low,
            })  # fmt: skip
    summary = {part: summarise(out, part, threshold) for part in ("inside", "outside", "all")}
    return {
        "omega": list(OMEGAS), "horizon_s": HORIZON_S, "measure_s": MEASURE_S, "amplitude": AMPLITUDE,
        "threshold": threshold, "device": str(device), "equilibria": out, "summary": summary,
        "holds": summary["inside"]["holds"],
        "method": "rollout_gain of cf_stability/stability/penalties.py (the rollout of the gain penalty) in float64, "
                  "eval mode, at the equilibria of stability.json",
        "wall_time_s": time.perf_counter() - t_start,
    }  # fmt: skip


def summarise(records: Sequence[Mapping[str, Any]], part: str, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Over the equilibria at grid speeds in support of ``part`` (``inside``: status ok or multiple; ``outside``: an
    equilibrium outside the band; ``all``): their number, the largest gain per frequency and over the frequencies,
    the share above ``threshold`` and whether the check holds (no gain above it; None without equilibria)."""
    statuses = {"inside": INSIDE, "outside": ("outside",), "all": EQUILIBRIUM}[part]
    rows = [r for r in records if r["in_support"] and r["status"] in statuses]
    out: dict[str, Any] = {"n": len(rows)}
    for j, w in enumerate(OMEGAS):
        values = [r["gain"][j] for r in rows if r["gain"][j] is not None]
        out[f"max_gain_{w:g}"] = max(values) if values else None
    maxima = [r["max_gain"] for r in rows if r["max_gain"] is not None]
    out["max_gain"] = max(maxima) if maxima else None
    out["n_above"] = sum(bool(r["above"]) for r in rows)
    out["n_undefined"] = sum(r["max_gain"] is None for r in rows)
    out["share_above"] = out["n_above"] / len(rows) if rows else None
    out["holds"] = None if not rows else (out["n_above"] == 0 and out["n_undefined"] == 0)
    audit = [r["audit_max_gain_low"] for r in rows if r["audit_max_gain_low"] is not None]
    out["audit_max_gain_low"] = max(audit) if audit else None
    return out


def check_run(run_dir: Path, device: torch.device, write: bool = True) -> dict[str, Any] | str:
    """:func:`check` of one run directory, written to ``long_window.json``; the reason as a string when the run has
    no usable audit."""
    path = run_dir / AUDIT
    if not (run_dir / "model.pt").is_file() or not path.is_file():
        return "model.pt or stability.json missing"
    payload = read_json(path)
    if payload.get("error") or not isinstance(payload.get("audit"), Mapping):
        return f"{AUDIT}: no audit"
    if (run_dir / "model.pt").stat().st_mtime > path.stat().st_mtime:
        return f"{AUDIT} older than model.pt"
    result = check(load_model(run_dir / "model.pt"), payload["audit"], device)
    result["run"] = str(run_dir)
    if write:
        write_json(run_dir / OUTPUT, result)
    return result


def collect(
    runs_root: Path, data: str, device: torch.device, write: bool = True, details: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, list[str]]:  # fmt: skip
    """One row per run of ``ARMS`` (fold 0, seed 0) and the problems; ``details`` (when given) receives the result of
    every run under ``<experiment>/<data>/<architecture>/<run>``."""
    rows, problems = [], []
    for arm, experiments in ARMS.items():
        for architecture, experiment in experiments.items():
            run_dir = runs_root / experiment / data / architecture / RUN
            if not run_dir.is_dir():
                problems.append(f"{experiment}/{data}/{architecture}/{RUN}: run missing")
                continue
            mine = write and any(experiment.startswith(prefix) for prefix in WRITE)
            result = check_run(run_dir, device, mine)
            if isinstance(result, str):
                problems.append(f"{experiment}/{data}/{architecture}/{RUN}: {result}")
                continue
            if details is not None:
                details[f"{experiment}/{data}/{architecture}/{RUN}"] = result
            inside, outside = result["summary"]["inside"], result["summary"]["outside"]
            row = {"architecture": architecture, "arm": arm, "experiment": experiment, "run": RUN,
                   "speeds_inside": inside["n"], "holds": inside["holds"], "share_above": inside["share_above"],
                   "max_gain": inside["max_gain"], "audit_max_gain_low": inside["audit_max_gain_low"],
                   "speeds_outside": outside["n"], "max_gain_outside": outside["max_gain"],
                   "wall_time_s": result["wall_time_s"]}  # fmt: skip
            row.update({f"max_gain_{w:g}": inside[f"max_gain_{w:g}"] for w in OMEGAS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        order = {a: k for k, a in enumerate(ARCHITECTURES)}
        arms = {a: k for k, a in enumerate(ARMS)}
        frame = frame.sort_values(by=["architecture", "arm"], key=lambda c: c.map(order if c.name == "architecture"
                                                                                  else arms), kind="stable")
        frame = frame.reset_index(drop=True)
    return frame, problems


def table(frame: pd.DataFrame, problems: Sequence[str], data: str) -> Table:
    columns = [
        Column("architecture", "architecture", "text"), Column("arm", "arm", "text"),
        Column("speeds_inside", "speeds (inside band, support)", "int"),
        *(Column(f"max_gain_{w:g}", f"max gain at {w:g} rad/s", digits=3) for w in OMEGAS),
        Column("max_gain", "max gain", digits=3), Column("share_above", f"share > {THRESHOLD:g}"),
        Column("holds", f"check holds (all < {THRESHOLD:g})", "flag"),
        Column("audit_max_gain_low", "audit: max gain at omega <= 0.1", digits=3),
        Column("speeds_outside", "speeds (outside band)", "int"), Column("max_gain_outside", "max gain outside", digits=3),
    ]  # fmt: skip
    notes = [
        f"Long-window check of D128 (M11, hypothesis H-N3): the rollout gain of the gain penalty with the rollout of "
        f"e2_horizon_pilot, {HORIZON_S:g} s after a warm-up of one window at the equilibrium, the follower speed fitted "
        f"on sin, cos and 1 over the last {MEASURE_S:g} s (two periods of 0.05 rad/s), leader amplitude {AMPLITUDE:g} "
        f"m/s, at omega {', '.join(f'{w:g}' for w in OMEGAS)} rad/s; float64, eval mode; {data}, fold 0, seed 0.",
        "Equilibria: those of the stored audit (stability.json) at the grid speeds in the support of the training "
        "data; inside band: status ok or multiple (the speeds of the stable band share); outside band: reported "
        "beside. max gain at a frequency: the largest over the speeds inside the band; share: of those speeds with "
        f"a gain above {THRESHOLD:g} at one of the three frequencies; the check holds when there is none.",
        "audit: the largest numerical gain of the audit (discard max(60 s, one period), whole periods >= max(60 s, two "
        "periods)) at its frequencies <= 0.1 rad/s at the same speeds, for reference.",
        "Arms: E1 (no penalty); E2 (gain penalty, weight 0.1, rollout 40 s, last 20 s); E2 long window (the same "
        "with the rollout of this check, the revision of 7 October 2026); windowed w<weight>: the windowed-gain "
        "penalty of D128 at the pilot weights. Per run: long_window_runs.json next to this table (and long_window.json "
        "next to stability.json of the runs of the windowed penalty).",
    ]
    if problems:
        notes.append(f"Problems: {len(problems)} (first: {problems[0]}).")
    return Table("long_window", "Long-window check of the windowed-gain penalty (M11, D128)", notes, frame, columns)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--data", default="follownet_highd")
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-json", action="store_true", help="do not write long_window.json into the runs")
    args = parser.parse_args(argv)
    details: dict[str, Any] = {}
    frame, problems = collect(resolve_path(args.runs_root), args.data, torch.device(args.device), not args.no_json,
                              details)  # fmt: skip
    out = resolve_path(args.out)
    write_table(table(frame, problems, args.data), out)
    write_json(out / "long_window_runs.json", details)
    for line in problems:
        print(f"  problem: {line}")
    holds = ", ".join(f"{r['architecture']} {r['arm']}: {'holds' if r['holds'] else 'fails'} (max {r['max_gain']})"
                      for r in frame.to_dict("records") if r["arm"].startswith("windowed"))  # fmt: skip
    print(f"long_window: {len(frame)} runs, {len(problems)} problems -> {out / 'long_window.md'}; {holds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
