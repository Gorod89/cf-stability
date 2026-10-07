"""Full-history linear analysis of the recurrent laws (M9, review P0).

    python scripts/analysis/full_history_audit.py              # full_history.json per run, the table, the figure
    python scripts/analysis/full_history_audit.py --no-figure  # without the figure

The audit classifies GRU, LSTM and PERL by the numerical frequency response of the full rollout and
reports the Jacobian criterion only for the memoryless view (the window filled with the constant
state, the derivatives summed over the window). These laws re-run their network over the last
``W = 30`` states at every step (no hidden state is carried between steps, ``rollout_model``), so the
history Jacobian ``J [W, 3]`` of the acceleration (``history_jacobian``) is their exact
linearisation. The state-space form of the linearised two-vehicle loop (leader speed in, follower
speed out, semi-implicit Euler at ``dt = 0.1 s``) is ``windowed_state_space`` of
``cf_stability/stability/analytic.py``: state ``(e[k], u[k-W+1..k], w[k-W+1..k])`` with the gap,
follower and leader speed deviations.

For every run of the arms ``ARMS`` (E1; the chosen E2 weight, gain penalty 0.1; the combined penalty of
D110, Jacobian weight 0.1 for GRU and LSTM, 1 for PERL) and every grid speed with an equilibrium in the
stored audit (status ok, multiple or outside: inside or outside the band), at the stored equilibrium:

* (a) the poles of the closed loop (``closed_loop_poles``, the ``W + 1`` eigenvalues of the block of the gap
  and the follower speeds) and whether all lie inside the unit circle: local stability of the full
  history model;
* (b) the windowed gain ``|G(e^(i w dt))|`` (``transfer_windowed``) at the 25 frequencies of the audit and its
  maximum, with the verdict of the audit's rule (maximum above 1.02, strict);
* (c) the memoryless-view margin ``M = f_v^2 + 2 f_v f_dv - 2 f_s`` for reference, and the coefficient
  ``M_w`` of the low-frequency expansion of the windowed gain (``windowed_margin``; ``|G|^2 = 1 - w^2 M_w /
  f_s^2 + ...``), which differs from ``M`` when spacing and speed are read with different lags.

Writes ``full_history.json`` next to ``stability.json`` of every run, the table
``runs/_tables/m9/full_history.{csv,md}`` (per architecture, arm and part of the speeds: the share of
the speeds whose full-history poles are all inside the unit circle, the share whose windowed gain exceeds
1.02 at some audit frequency, the agreement of that verdict with the numerical verdict of the audit, the
agreement of the sign of ``M`` and of ``M_w`` with the windowed gain at the lowest audit frequency being
above 1; mean over the runs with the 95 % percentile bootstrap interval over the runs, 1000 resamples,
seed 0) and the figure ``runs/_report/supplement/figures/full_history_gain.{png,pdf,txt}`` (windowed
against numerical gain at 0.02 rad/s, one panel per architecture, E1 and E2). Nothing is retrained:
the stored models and audits only (CPU, float64).
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

from cf_stability.eval.stats import bootstrap_ci  # noqa: E402
from cf_stability.models import load_model  # noqa: E402
from cf_stability.models.base import CFModel  # noqa: E402
from cf_stability.stability.analytic import (  # noqa: E402
    closed_loop_poles,
    criterion,
    history_jacobian,
    transfer_windowed,
    windowed_margin,
)
from cf_stability.utils import read_json, resolve_path, write_json  # noqa: E402

ARCHITECTURES = ("gru", "lstm", "perl")
ARMS: dict[str, dict[str, str]] = {  # arm -> architecture -> experiment
    "E1": {"gru": "e1", "lstm": "e1", "perl": "e1"},
    "E2": {"gru": "e2_gain_w0.1", "lstm": "e2_gain_w0.1", "perl": "e2_gain_w0.1"},  # chosen weight (D85)
    "E2 combined": {"gru": "e2_combined_j0.1", "lstm": "e2_combined_j0.1", "perl": "e2_combined_j1"},  # D110
}
FIGURE_ARMS = ("E1", "E2")
OUTPUT = "full_history.json"
AUDIT = "stability.json"
EQUILIBRIUM = ("ok", "multiple", "outside")  # statuses with an equilibrium (FOUND of cf_stability.stability.audit)
FLAGS = ("clipped", "stopped", "collided")  # rollout flags of the audit (cf_stability.stability.frequency)
PARTS = ("all", "support")
THRESHOLD = 1.02  # the audit's numerical rule when its config does not say (FrequencyConfig.threshold)
LIMIT_OMEGA = 1e-4  # rad/s: check of the low-frequency limit, |G_w| > 1 there exactly when M_w < 0
# per-speed boolean statistics of the table (share of the speeds where the flag holds, among those where it is defined)
STATISTICS: dict[str, str] = {
    "poles_stable": "all full-history poles inside the unit circle",
    "local_stable_memoryless": "memoryless view locally stable (f_s > 0, f_v + f_dv < 0)",
    "windowed_unstable": "windowed gain above the threshold at some audit frequency",
    "numerical_unstable": "numerical verdict of the audit (max gain above the threshold)",
    "gain_agreement": "windowed verdict = numerical verdict",
    "lowfreq_sign_agreement": "(M < 0) = (windowed gain at the lowest frequency > 1)",
    "lowfreq_sign_agreement_window": "(M_w < 0) = (windowed gain at the lowest frequency > 1)",
    "margin_sign_agreement": "(M < 0) = (M_w < 0): the memoryless criterion gives the low-frequency limit",
    "limit_check_window": f"(M_w < 0) = (windowed gain at {LIMIT_OMEGA:g} rad/s > 1), a check of M_w",
}
N_RESAMPLES, SEED = 1000, 0


def _num(x: Any) -> float | None:
    """Finite float, else None (JSON null)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _mean(values: Sequence[bool | None]) -> tuple[float | None, int]:
    """Share of True among the defined values and their number."""
    defined = [bool(v) for v in values if v is not None]
    return (sum(defined) / len(defined) if defined else None), len(defined)


def _equal(a: bool | None, b: bool | None) -> bool | None:
    return None if a is None or b is None else a == b


def analyse(model: CFModel, audit: Mapping[str, Any], threshold: float | None = None) -> dict[str, Any]:
    """Full-history linear analysis of ``model`` at the equilibria of its stored ``audit`` (the ``audit`` block of
    ``stability.json``). ``model`` is used as given (cast to float64 on the CPU by the caller). Returns the content of
    ``full_history.json``: per speed with an equilibrium the history Jacobian, the poles, the windowed gains at the
    audit frequencies and the verdicts; the shares of :data:`STATISTICS` per part (all speeds, speeds in support)."""
    t_start = time.perf_counter()
    if threshold is None:
        threshold = float(((audit.get("config") or {}).get("frequency") or {}).get("threshold", THRESHOLD))
    omega_list = [float(w) for w in audit["omega"]]
    records = [r for r in audit["equilibria"] if r.get("status") in EQUILIBRIUM and r.get("s") is not None]
    dt, window = float(model.dt), int(model.window)
    out_records: list[dict[str, Any]] = []
    if records:
        s = torch.tensor([float(r["s"]) for r in records], dtype=torch.float64)
        v = torch.tensor([float(r["v"]) for r in records], dtype=torch.float64)
        jacobian = history_jacobian(model, s, v).detach().to(torch.float64).cpu()  # [n, W, 3]
        poles = closed_loop_poles(jacobian, dt)  # [n, W + 1]
        radius = poles.abs().amax(dim=1)
        omega = torch.tensor(omega_list, dtype=torch.float64)
        gain = transfer_windowed(jacobian, omega, dt).abs()  # [n, n_w]
        gain_limit = transfer_windowed(jacobian, torch.tensor([LIMIT_OMEGA], dtype=torch.float64), dt).abs()[:, 0]
        f_s, f_dv, f_v = jacobian.sum(dim=1).unbind(-1)
        crit = criterion(f_s, f_dv, f_v)
        margin_window = windowed_margin(jacobian, dt)
        lag = dt * torch.arange(window - 1, -1, -1, dtype=torch.float64)  # time ago of every window entry
        moments = (jacobian * lag[None, :, None]).sum(dim=1)  # [n, 3]
        for k, r in enumerate(records):
            order = torch.argsort(poles[k].abs(), descending=True)
            gains = gain[k].tolist()
            k_max = int(np.nanargmax(gains)) if np.isfinite(gains).any() else 0
            max_gain = _num(max(gains)) if np.isfinite(gains).all() else None
            numerical = r.get("gain") if isinstance(r.get("gain"), list) else []
            stored = (r.get("analytic_max_gain") or {}).get("windowed")
            flags_low = [bool((r.get(name) or [False])[0]) for name in FLAGS if isinstance(r.get(name), list)]
            margin, margin_w = _num(crit["margin"][k]), _num(margin_window[k])
            low = _num(gains[0])
            out_records.append({
                "v": float(r["v"]), "s": float(r["s"]), "status": r["status"], "in_support": r.get("in_support"),
                "in_band": r.get("in_band"),
                "f_s": _num(f_s[k]), "f_dv": _num(f_dv[k]), "f_v": _num(f_v[k]), "margin": margin,
                "margin_window": margin_w,
                "lag_moments": {"s": _num(moments[k, 0]), "dv": _num(moments[k, 1]), "v": _num(moments[k, 2])},
                "local_stable_memoryless": bool(crit["local_stable"][k]),
                "spectral_radius": _num(radius[k]), "poles_stable": bool(radius[k] < 1.0),
                "poles": [[_num(p.real), _num(p.imag)] for p in poles[k][order].tolist()],
                "gain_windowed": [_num(g) for g in gains], "max_gain_windowed": max_gain,
                "omega_at_max_windowed": omega_list[k_max] if max_gain is not None else None,
                "windowed_unstable": None if max_gain is None else max_gain > threshold,
                "lowfreq_windowed_above_one": None if low is None else low > 1.0,
                "gain_windowed_limit": _num(gain_limit[k]),
                "gain_numerical_lowest": _num(numerical[0]) if numerical else None,
                "fit_residual_lowest": _num(r["residual"][0]) if isinstance(r.get("residual"), list) and r["residual"]
                else None,
                "flagged_lowest": any(flags_low) if flags_low else None,
                "flagged_any": any(any(r.get(name) or ()) for name in FLAGS),
                "numerical_max_gain": _num(r.get("max_gain")), "numerical_unstable": r.get("unstable"),
                "stored_max_gain_windowed": _num(stored),
                "jacobian": [[_num(x) for x in row] for row in jacobian[k].tolist()],
            })  # fmt: skip
    summary = {part: summarise_records(out_records, part) for part in PARTS}
    diffs = [abs(r["max_gain_windowed"] - r["stored_max_gain_windowed"]) for r in out_records
             if r["max_gain_windowed"] is not None and r["stored_max_gain_windowed"] is not None]  # fmt: skip
    return {
        "model": getattr(model, "name", None), "window": window, "dt": dt, "omega": omega_list,
        "lowest_omega": omega_list[0] if omega_list else None, "threshold": threshold,
        "method": "history Jacobian at the stored equilibria (float64, CPU); poles: eigenvalues of the (e, u) block "
                  "of windowed_state_space; gains: transfer_windowed; M_w: windowed_margin "
                  "(cf_stability/stability/analytic.py)",
        "equilibria": out_records, "summary": summary,
        "stored_max_abs_diff": max(diffs) if diffs else None,
        "wall_time_s": time.perf_counter() - t_start,
    }  # fmt: skip


def summarise_records(records: Sequence[Mapping[str, Any]], part: str) -> dict[str, Any]:
    """Shares of :data:`STATISTICS` over the speeds of ``part`` (``all`` or ``support``: ``in_support`` true), each
    over the speeds where it is defined, with the numbers of speeds."""
    rows = [r for r in records if part == "all" or r.get("in_support")]

    def negative(value: float | None) -> bool | None:
        return None if value is None else value < 0.0

    def above_one(value: float | None) -> bool | None:
        return None if value is None else value > 1.0

    values = {
        "poles_stable": [r["poles_stable"] for r in rows],
        "local_stable_memoryless": [r["local_stable_memoryless"] for r in rows],
        "windowed_unstable": [r["windowed_unstable"] for r in rows],
        "numerical_unstable": [r["numerical_unstable"] for r in rows],
        "gain_agreement": [_equal(r["windowed_unstable"], r["numerical_unstable"]) for r in rows],
        "lowfreq_sign_agreement": [_equal(negative(r["margin"]), r["lowfreq_windowed_above_one"]) for r in rows],
        "lowfreq_sign_agreement_window": [
            _equal(negative(r["margin_window"]), r["lowfreq_windowed_above_one"]) for r in rows
        ],
        "margin_sign_agreement": [_equal(negative(r["margin"]), negative(r["margin_window"])) for r in rows],
        "limit_check_window": [_equal(negative(r["margin_window"]), above_one(r["gain_windowed_limit"])) for r in rows],
    }
    out: dict[str, Any] = {"n_equilibria": len(rows)}
    for name, flags in values.items():
        out[name], out[f"n_{name}"] = _mean(flags)
    out["n_poles_unstable_numerical_stable"] = sum(
        1 for r in rows if not r["poles_stable"] and r["numerical_unstable"] is False
    )
    return out


def run_directories(runs_root: Path, experiment: str, data: str, architecture: str) -> list[Path]:
    return sorted(p.parent for p in (runs_root / experiment / data / architecture).glob("*/model.pt"))


def load_audit(run_dir: Path) -> dict[str, Any] | str:
    """The ``audit`` block of the run's ``stability.json``, or the reason why it cannot be used."""
    path = run_dir / AUDIT
    if not path.is_file():
        return f"{AUDIT} missing"
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        return f"{AUDIT} unreadable ({type(exc).__name__})"
    if payload.get("error") or not isinstance(payload.get("audit"), Mapping):
        return f"{AUDIT}: {str(payload.get('error', 'no audit')).splitlines()[0]}"
    if (run_dir / "model.pt").stat().st_mtime > path.stat().st_mtime:
        return f"{AUDIT} older than model.pt"
    return payload["audit"]


def analyse_run(run_dir: Path, write: bool = True) -> dict[str, Any] | str:
    """:func:`analyse` of one run directory (model.pt and stability.json), written to ``full_history.json``; the
    reason as a string when the audit cannot be used."""
    audit = load_audit(run_dir)
    if isinstance(audit, str):
        return audit
    model = copy.deepcopy(load_model(run_dir / "model.pt")).double().cpu().eval()
    payload = analyse(model, audit)
    payload["run"] = str(run_dir)
    payload["source"] = {"audit": AUDIT, "audit_mtime": (run_dir / AUDIT).stat().st_mtime,
                         "model_mtime": (run_dir / "model.pt").stat().st_mtime,
                         "audit_agreement_gain": audit.get("summary", {}).get("agreement_gain")}  # fmt: skip
    if write:
        write_json(run_dir / OUTPUT, payload)
    return payload


def collect(
    runs_root: Path, data: str, architectures: Sequence[str], arms: Mapping[str, Mapping[str, str]] = ARMS,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[tuple[str, str], int]]:
    """Per run and part the shares (one row each), per speed the points of the figure, the problems and the
    numbers of runs per architecture and arm."""
    rows, points, problems, expected = [], [], [], {}
    for arm, experiments in arms.items():
        for architecture in architectures:
            experiment = experiments.get(architecture)
            if experiment is None:
                continue
            runs = run_directories(runs_root, experiment, data, architecture)
            expected[architecture, arm] = len(runs)
            for run_dir in runs:
                payload = analyse_run(run_dir, write)
                if isinstance(payload, str):
                    problems.append(f"{experiment}/{architecture}/{run_dir.name}: {payload}")
                    continue
                agreement = (payload["source"].get("audit_agreement_gain") or {})
                for part in PARTS:
                    summary = payload["summary"][part]
                    stored = agreement.get(part)
                    rows.append({
                        "architecture": architecture, "arm": arm, "experiment": experiment, "run": run_dir.name,
                        "part": part, **summary, "stored_max_abs_diff": payload["stored_max_abs_diff"],
                        "audit_agreement_gain": stored, "threshold": payload["threshold"],
                        "lowest_omega": payload["lowest_omega"],
                    })  # fmt: skip
                for r in payload["equilibria"]:
                    points.append({
                        "architecture": architecture, "arm": arm, "experiment": experiment, "run": run_dir.name,
                        "v": r["v"], "in_support": r["in_support"], "gain_windowed_lowest": r["gain_windowed"][0],
                        "gain_numerical_lowest": r["gain_numerical_lowest"], "poles_stable": r["poles_stable"],
                        "spectral_radius": r["spectral_radius"], "flagged_lowest": r["flagged_lowest"],
                        "fit_residual_lowest": r["fit_residual_lowest"],
                        "margin": r["margin"], "margin_window": r["margin_window"],
                    })  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(points), problems, expected


def lowest_differences(points: pd.DataFrame, part: str) -> tuple[int, float, float]:
    """Speeds whose poles are all inside and whose rollout at the lowest frequency was not clipped, stopped or
    collided, and the median and 90 % quantile over them of |numerical - windowed| gain at that frequency."""
    if points.empty:
        return 0, np.nan, np.nan
    sel = points[points["poles_stable"].astype(bool) & ~points["flagged_lowest"].fillna(True).astype(bool)]
    if part == "support":
        sel = sel[sel["in_support"].fillna(False).astype(bool)]
    diff = (sel["gain_numerical_lowest"].astype(float) - sel["gain_windowed_lowest"].astype(float)).abs().dropna()
    if not len(diff):
        return 0, np.nan, np.nan
    return len(diff), float(diff.median()), float(diff.quantile(0.9))


def summarise(
    rows: pd.DataFrame, expected: Mapping[tuple[str, str], int], points: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per architecture, arm and part: the mean over the runs of every share of :data:`STATISTICS` with its 95 %
    percentile bootstrap interval over the runs, the numbers of runs and speeds, |numerical - windowed| gain at the
    lowest frequency over the locally stable speeds (pooled; :func:`lowest_differences`), the consistency checks."""
    out = []
    if rows.empty:
        return pd.DataFrame(out)
    for (architecture, arm, part), group in rows.groupby(["architecture", "arm", "part"], sort=False):
        row: dict[str, Any] = {
            "architecture": architecture, "arm": arm, "experiment": group["experiment"].iloc[0], "part": part,
            "runs": int(group["run"].nunique()), "runs_expected": expected.get((architecture, arm), 0),
            "speeds": int(group["n_equilibria"].sum()),
        }  # fmt: skip
        for name in STATISTICS:
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=np.float64)
            ci = bootstrap_ci(values, group["run"].to_numpy(), n_resamples=N_RESAMPLES, seed=SEED)
            row.update({name: ci["estimate"], f"{name}_low": ci["low"], f"{name}_high": ci["high"]})
        row["poles_unstable_numerical_stable"] = int(group["n_poles_unstable_numerical_stable"].sum())
        mine = points[(points["architecture"] == architecture) & (points["arm"] == arm)] if points is not None else None
        n, median, q90 = lowest_differences(mine if mine is not None else pd.DataFrame(), part)
        row.update(lowest_speeds=n, lowest_abs_diff_median=median, lowest_abs_diff_q90=q90)
        row["stored_max_abs_diff"] = pd.to_numeric(group["stored_max_abs_diff"], errors="coerce").max()
        stored = pd.to_numeric(group["audit_agreement_gain"], errors="coerce")
        mine_agreement = pd.to_numeric(group["gain_agreement"], errors="coerce")
        row["audit_agreement_max_abs_diff"] = (
            float((stored - mine_agreement).abs().max()) if stored.notna().any() else np.nan
        )
        out.append(row)
    order = {arch: k for k, arch in enumerate(ARCHITECTURES)}
    frame = pd.DataFrame(out)
    arm_order = {arm: k for k, arm in enumerate(ARMS)}
    frame["_a"], frame["_b"] = frame["architecture"].map(order).fillna(99), frame["arm"].map(arm_order).fillna(99)
    frame["_c"] = frame["part"].map({p: k for k, p in enumerate(PARTS)})
    return frame.sort_values(["_a", "_b", "_c"], kind="stable").drop(columns=["_a", "_b", "_c"]).reset_index(drop=True)


def _ci(row: Mapping[str, Any], key: str, digits: int = 2) -> str:
    value, low, high = row[key], row[f"{key}_low"], row[f"{key}_high"]
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]" if np.isfinite(low) else f"{value:.{digits}f}"


def markdown(table: pd.DataFrame, problems: Sequence[str], data: str, threshold: float, lowest: float) -> str:
    def largest(key: str) -> float:
        return float(pd.to_numeric(table[key], errors="coerce").max()) if len(table) and key in table else np.nan

    def smallest(key: str) -> float:
        return float(pd.to_numeric(table[key], errors="coerce").min()) if len(table) and key in table else np.nan

    arms = "; ".join(f"{arm}: " + ", ".join(f"{a} {e}" for a, e in experiments.items())
                     for arm, experiments in ARMS.items())  # fmt: skip
    lines = [
        "# Full-history linear analysis of the recurrent laws (M9)",
        "",
        f"- Runs: GRU, LSTM and PERL on {data}; arms {arms} (E1 and E2 five folds x five seeds, the combined arm "
        "five folds x seed 0). Speeds: every grid speed with an equilibrium in the stored audit (status ok, "
        "multiple or outside: inside or outside the band), at the stored equilibrium; part all = every such speed, "
        "support = those in the speed range of the training data.",
        "- Linearisation: the history Jacobian J [30, 3] of the acceleration with respect to the 30 states of the "
        "window (history_jacobian). The networks are re-run over the last 30 states at every step, without a hidden "
        "state carried between steps, so J is the exact linearisation of the rollout. State-space form of the "
        "two-vehicle loop with the leader speed as input and the follower speed as output, semi-implicit Euler at "
        "0.1 s as in the rollout (windowed_state_space, cf_stability/stability/analytic.py): state (e[k], "
        "u[k-29..k], w[k-29..k]) of the gap, follower and leader speed deviations.",
        "- poles inside: share of the speeds whose 31 closed-loop poles (the eigenvalues of the block of e and u; "
        "the other 30 are 0) all lie strictly inside the unit circle: local stability of the full history model. "
        "memoryless locally stable (reference): f_s > 0 and f_v + f_dv < 0 of the memoryless view (derivatives "
        "summed over the window).",
        f"- windowed > {threshold:g}: share of the speeds whose windowed gain |G(e^(iω dt))| (transfer_windowed) "
        f"exceeds {threshold:g} at some of the 25 audit frequencies (0.02-2 rad/s; strict inequality, the rule of the "
        "audit). numerical unstable (reference): the audit's verdict from the rollouts. agreement: share of the "
        "speeds where the two verdicts are equal (a speed whose rollout broke down has no numerical verdict and is "
        f"left out). \\|num - win\\| at {lowest:g}: median (90 % quantile) over the speeds with all poles inside and "
        f"an unflagged rollout at {lowest:g} rad/s (pooled over the runs) of the difference between the audit's "
        "numerical gain and the windowed gain at that frequency.",
        f"- Low frequency: sign(M) vs \\|G_w({lowest:g})\\| is the share of the speeds where M < 0 (M = f_v² + "
        "2 f_v f_dv - 2 f_s of the memoryless view) exactly when the windowed gain at the lowest audit frequency "
        f"{lowest:g} rad/s is above 1; sign(M_w) the same for M_w = M + 2 (f_v m_s - f_s m_v) - dt f_s f_v, the "
        "exact coefficient of the low-frequency expansion of the windowed gain, |G|² = 1 - ω² M_w / f_s² + O(ω⁴) "
        "(windowed_margin; m_x = sum over the lags of lag x J_x, the first moments of the history Jacobian; the last "
        "term is the Euler step); sign(M) = sign(M_w): the memoryless criterion gives the ω -> 0 limit of the "
        "windowed gain. M is the coefficient only when f_v m_s = f_s m_v (no memory, or a lag common to spacing and "
        f"speed). At {lowest:g} rad/s the higher-order terms can decide the side of 1 when |M_w| is small: PERL "
        "(Newell's feed-forward of the leader acceleration has m_dv = -m_v = 1, so M_w ~ M + 2 f_s) under the "
        "combined penalty.",
        "- Unit run: the shares per run over its speeds, mean over the runs with the 95 % percentile bootstrap "
        "interval over the runs (1000 resamples, seed 0); a run without speeds in the part is left out of it.",
        f"- Checks: the recomputed maximum windowed gain equals the stored analytic_max_gain.windowed of the audits "
        f"to {largest('stored_max_abs_diff'):.1e} (largest absolute difference); the agreement column equals the "
        f"audit's agreement_gain to {largest('audit_agreement_max_abs_diff'):.1e}; at {LIMIT_OMEGA:g} rad/s the "
        f"windowed gain is above 1 exactly when M_w < 0 at a share {smallest('limit_check_window'):.3f} or more of the "
        "speeds (mean over the runs, smallest row). Per-run details (poles, gains, Jacobian): full_history.json next "
        "to stability.json.",
    ]
    if problems:
        lines.append(f"- Problems: {len(problems)} (first: {problems[0]}).")
    lines += [
        "",
        "Poles and gain verdicts:",
        "",
        "| architecture | arm | part | runs | speeds | poles inside | memoryless locally stable | "
        f"windowed > {threshold:g} | numerical unstable | agreement windowed vs numerical | "
        f"\\|num - win\\| at {lowest:g} |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        diff = (f"{row['lowest_abs_diff_median']:.1e} ({row['lowest_abs_diff_q90']:.1e})"
                if np.isfinite(row["lowest_abs_diff_median"]) else "n/a")  # fmt: skip
        lines.append(
            f"| {row['architecture']} | {row['arm']} | {row['part']} | {row['runs']} | {row['speeds']} | "
            f"{_ci(row, 'poles_stable')} | {row['local_stable_memoryless']:.2f} | {_ci(row, 'windowed_unstable')} | "
            f"{row['numerical_unstable']:.2f} | {_ci(row, 'gain_agreement')} | {diff} |"
        )
    lines += [
        "",
        "Low frequency:",
        "",
        f"| architecture | arm | part | sign(M) vs \\|G_w({lowest:g})\\| | sign(M_w) vs \\|G_w({lowest:g})\\| | "
        "sign(M) = sign(M_w) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['architecture']} | {row['arm']} | {row['part']} | {_ci(row, 'lowfreq_sign_agreement')} | "
            f"{_ci(row, 'lowfreq_sign_agreement_window')} | {_ci(row, 'margin_sign_agreement')} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ figure


def gain_figure(
    points: pd.DataFrame, architectures: Sequence[str], arms: Sequence[str] = FIGURE_ARMS, lowest: float = 0.02
) -> tuple[Any, dict]:
    """Numerical against windowed gain at the lowest audit frequency, one panel per architecture, E1 and E2 in two
    colours and two shapes; open markers: a full-history pole on or outside the unit circle. Equal linear axes per
    panel over the range of its windowed gains (and 1); a numerical gain beyond the limits is drawn on the upper
    (lower) edge as a triangle pointing up (down). Returns the figure and the counts for the caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from cf_stability.eval import figures
    from cf_stability.eval.figures import FULL_WIDTH, PALETTE, WITH, WITHOUT

    figures.apply_style()
    colours = {"E1": WITHOUT, "E2": WITH}
    shapes = {"E1": "o", "E2": "s"}
    names = {"E1": "E1, no penalty", "E2": "E2, gain penalty 0.1 (chosen)"}
    used = points[points["arm"].isin(arms)].copy()
    used = used[np.isfinite(used["gain_windowed_lowest"].astype(float))
                & np.isfinite(used["gain_numerical_lowest"].astype(float))]  # fmt: skip
    present = [a for a in architectures if (used["architecture"] == a).any()]
    if not present:
        raise ValueError("no points to draw")
    stable = used["poles_stable"].astype(bool)
    counts: dict[str, Any] = {"points": len(used), "above": 0, "below": 0, "poles_outside": int((~stable).sum()),
                              "limits": {}}  # fmt: skip
    fig, axes = plt.subplots(1, len(present) + 1, figsize=(FULL_WIDTH, FULL_WIDTH / (len(present) + 1) + 0.35),
                             squeeze=False)  # fmt: skip
    for ax, arch in zip(axes.flat, present):
        part = used[used["architecture"] == arch]
        x_all = np.append(part["gain_windowed_lowest"].to_numpy(float), 1.0)
        low, high = float(x_all.min()), float(x_all.max())
        pad = 0.06 * (high - low if high > low else 1.0)
        limits = (low - pad, high + pad)
        counts["limits"][arch] = limits
        for arm in arms:
            sel = part[part["arm"] == arm]
            x = sel["gain_windowed_lowest"].to_numpy(float)
            y = sel["gain_numerical_lowest"].to_numpy(float)
            inside = sel["poles_stable"].astype(bool).to_numpy()
            above, below = y > limits[1], y < limits[0]
            counts["above"] += int(above.sum())
            counts["below"] += int(below.sum())
            y_drawn = np.clip(y, *limits)
            for mask, filled in ((inside, True), (~inside, False)):
                for clip, marker in ((~above & ~below, shapes[arm]), (above, "^"), (below, "v")):
                    keep = mask & clip
                    if not keep.any():
                        continue
                    style = dict(mfc=colours[arm], mec="none", alpha=0.55) if filled else dict(
                        mfc="none", mec=colours[arm], mew=0.8, alpha=0.9)  # fmt: skip
                    ax.plot(x[keep], y_drawn[keep], ls="none", marker=marker, ms=3.4, zorder=3, clip_on=False, **style)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_box_aspect(1)
        ax.plot(limits, limits, color=PALETTE["dark"], lw=0.8, zorder=1)
        for line in (ax.axhline, ax.axvline):
            line(1.0, color=PALETTE["grey"], ls=":", lw=0.8, zorder=0)
        figures._label(ax, figures.name(arch))
    fig.supxlabel(f"windowed (full-history) |G| at {lowest:g} rad/s", fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(f"numerical |G| at {lowest:g} rad/s", fontsize=plt.rcParams["axes.labelsize"])
    legend_ax = axes.flat[len(present)]
    legend_ax.axis("off")
    handles = [Line2D([], [], ls="none", marker=shapes[a], color=colours[a], mec="none", label=names[a]) for a in arms]
    handles += [
        Line2D([], [], ls="none", marker="o", mfc="none", mec=PALETTE["dark"], label="a pole on or\noutside |z| = 1"),
        Line2D([], [], ls="none", marker="^", color=PALETTE["dark"], label="numerical |G| beyond\nthe axis (edge)"),
        Line2D([], [], color=PALETTE["dark"], lw=0.8, label="numerical = windowed"),
        Line2D([], [], color=PALETTE["grey"], ls=":", lw=0.8, label="|G| = 1"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=plt.rcParams["legend.fontsize"])
    return fig, counts


def off_diagonal(points: pd.DataFrame, arms: Sequence[str] = FIGURE_ARMS, tolerance: float = 0.05) -> dict[str, Any]:
    """Among the drawn speeds whose poles are all inside and whose rollout at the lowest frequency is not flagged:
    their number, those whose numerical and windowed gains differ by more than ``tolerance``, the smallest audit fit
    residual among the latter and the median one among the others."""
    sel = points[points["arm"].isin(arms) & points["poles_stable"].astype(bool)
                 & ~points["flagged_lowest"].fillna(True).astype(bool)]  # fmt: skip
    diff = (sel["gain_numerical_lowest"].astype(float) - sel["gain_windowed_lowest"].astype(float)).abs()
    residual = pd.to_numeric(sel["fit_residual_lowest"], errors="coerce")
    far = diff > tolerance
    return {"n": int(diff.notna().sum()), "far": int(far.sum()), "tolerance": tolerance,
            "far_residual_min": float(residual[far].min()) if far.any() else np.nan,
            "near_residual_median": float(residual[~far].median()) if (~far).any() else np.nan}  # fmt: skip


def caption(
    counts: Mapping[str, Any], table: pd.DataFrame, data: str, lowest: float = 0.02,
    far: Mapping[str, Any] | None = None,
) -> str:  # fmt: skip
    from cf_stability.eval.figures import name

    def share(arch: str, arm: str, key: str) -> str:
        sel = table[(table["architecture"] == arch) & (table["arm"] == arm) & (table["part"] == "all")]
        return "n/a" if sel.empty or not np.isfinite(sel[key].iloc[0]) else f"{sel[key].iloc[0]:.2f}"

    inside = ", ".join(f"{name(arch)} {share(arch, 'E1', 'poles_stable')} / {share(arch, 'E2', 'poles_stable')}"
                       for arch in ARCHITECTURES)  # fmt: skip
    e1 = table[(table["arm"] == "E1") & (table["part"] == "all")]
    medians = ", ".join(
        f"{name(arch)} {row['lowest_abs_diff_median']:.0e} / {row['lowest_abs_diff_q90']:.0e}"
        for arch in ARCHITECTURES for _, row in e1[e1["architecture"] == arch].iterrows()
    )  # fmt: skip
    outliers = "" if not far or not far["n"] else (
        f" {far['far']} of these {far['n']} points (E1 and E2) differ by more than {far['tolerance']:g}; their "
        f"rollouts are not sinusoidal (fit residual of the audit {far['far_residual_min']:.2f} or more, against a "
        f"median of {far['near_residual_median']:.0e} elsewhere): a finite-amplitude response, the gap moving by "
        f"|1 - G| A / ω with A / ω = {0.2 / lowest:g} m at A = 0.2 m/s and {lowest:g} rad/s."
    )  # fmt: skip
    return (
        "Full-history linearisation of the recurrent laws against the audit (M9). Every grid speed with an equilibrium "
        f"of every E1 run (no penalty) and E2 run (gain penalty, chosen weight 0.1) of GRU, LSTM and PERL on {data} "
        f"(five folds x five seeds): the gain of the follower speed at {lowest:g} rad/s, the lowest audit frequency, "
        "from the numerical rollout of the audit (y) against the gain |G(e^(iω dt))| of the linearised two-vehicle "
        "loop with the whole 30-state history (x; the history Jacobian of the network, semi-implicit Euler at 0.1 s, "
        "transfer_windowed). Filled markers: all 31 closed-loop poles inside the unit circle; open markers: a pole on "
        "or outside it (the loop is not locally stable, the rollout grows instead of settling and the two gains need "
        f"not agree; {counts['poles_outside']} of {counts['points']} points). Axes per panel over the range of its "
        f"windowed gains; {counts['above']} numerical gains above the upper limit and {counts['below']} below the "
        "lower one are drawn on the edge as triangles. Solid line: numerical = windowed; dotted: |G| = 1. Where all "
        "poles are inside and the rollout is not clipped, stopped or collided, |numerical - windowed| has median / "
        f"90 % quantile (E1) {medians}.{outliers} Share of the speeds with all poles inside, E1 / E2 (mean over the "
        f"runs): {inside}. The combined arm (D110) and the intervals: table full_history (runs/_tables/m9)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--data", default="follownet_highd")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--out", default="runs/_tables/m9", help="directory of full_history.{csv,md}")
    parser.add_argument("--figure-dir", default="runs/_report/supplement/figures")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-json", action="store_true", help="do not write full_history.json into the runs")
    args = parser.parse_args(argv)
    runs_root = resolve_path(args.runs_root)
    rows, points, problems, expected = collect(runs_root, args.data, args.architectures, write=not args.no_json)
    table = summarise(rows, expected, points)
    thresholds = sorted(set(rows["threshold"])) if len(rows) else [THRESHOLD]
    lowest_values = sorted(set(rows["lowest_omega"])) if len(rows) else [0.02]
    if len(thresholds) > 1 or len(lowest_values) > 1:
        problems.append(f"audits with different thresholds {thresholds} or lowest frequencies {lowest_values}")
    threshold, lowest = float(thresholds[0]), float(lowest_values[0])
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "full_history.csv", index=False)
    (out / "full_history.md").write_text(markdown(table, problems, args.data, threshold, lowest), encoding="utf-8")
    for line in problems[:20]:
        print(f"  problem: {line}")
    if not args.no_figure and len(points):
        from cf_stability.eval.figures import save

        fig, counts = gain_figure(points, args.architectures, lowest=lowest)
        figure_dir = resolve_path(args.figure_dir)
        save(fig, figure_dir, "full_history_gain")
        text = caption(counts, table, args.data, lowest, off_diagonal(points)) + "\n"
        (figure_dir / "full_history_gain.txt").write_text(text, encoding="utf-8")
        print(f"figure: {figure_dir / 'full_history_gain.png'} (+ .pdf, .txt)")
    analysed = len(rows[rows["part"] == "all"]) if len(rows) else 0
    print(f"full_history: {analysed} runs analysed, {len(problems)} problems -> {out / 'full_history.csv'}, .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
