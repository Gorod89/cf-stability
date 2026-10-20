"""Tables of the windowed-gain penalty of the recurrent laws (M11, docs/m11_contract.md, D128: algorithm A1, H-N3).

    python scripts/analysis/windowed_gain.py choose    # the pilot table and the weights chosen by the rule of D85
    python scripts/analysis/windowed_gain.py tables    # the pilot, the five-fold table and the branch of H-N3

Tables into ``runs/_tables/m11/`` (CSV and Markdown twins, ``cf_stability/eval/tables.py`` conventions; the run rows
and intervals of its ``TableMaker``):

* ``e2_windowed_pilot``: per architecture (GRU, LSTM, PERL) and weight of the pilot (``e2_windowed_w{0.1,1,10}``,
  ``follownet_highd``, fold 0, seed 0): validation and test spacing RMSE, the change against the E1 run of the same
  fold and seed, the band shares of the audit, the shares with the poles (``full_history.json``), the epochs, the
  feasible epochs (D89), the parts of the validation penalty at the best epoch, the long-window check
  (``long_window.json``, ``scripts/analysis/long_window_check.py``) and the weight chosen by the rule of D85 (the
  smallest validation RMSE among the weights whose stable band share is at least 0.9, else the largest share),
  written to ``e2_windowed_chosen.json``;
* ``e2_windowed``: per architecture the E1 runs (seed 0, five folds) and the windowed penalty at the chosen weight
  (``e2_windowed_w<chosen>``, five folds): the spacing RMSE of the test parts with its change against E1 paired over
  the drivers (the events on highD), the band shares and the poles with intervals over the runs, the long-window
  check of the run of fold 0, the platoon collisions; and, for information, the fine-tuned runs on ``ngsim_i80``
  (``m11_ft_windowed_w<chosen>`` against ``m5_ft``, the E1 models fine-tuned in M5); the branch of the decision rule
  of H-N3 in ``e2_windowed_branch.md`` (GRU and LSTM; PERL reported, not judged).

The rule of H-N3 (5-fold means on highD against the E1 runs of the same architecture): confirmed when GRU and LSTM
reach a stable band share >= 0.9 at a test spacing RMSE increase <= 10 % and the long-window check holds; partial
when the share is reached at a larger increase; refuted otherwise. The script names the branch the numbers fall in;
the verdict is not pronounced here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.tables import (  # noqa: E402
    AUDIT,
    EMPTY,
    EVENTS,
    METRICS,
    PLATOON,
    Column,
    Table,
    TableMaker,
    TablesConfig,
    _number,
    _put,
    choose_weight,
    write_table,
)
from cf_stability.utils import read_json, resolve_path  # noqa: E402

ARCHITECTURES = ("gru", "lstm", "perl")
JUDGED = ("gru", "lstm")  # H-N3: PERL is reported, not judged
WEIGHTS = (0.1, 1.0, 10.0)
EXPERIMENT = "e2_windowed_w{weight:g}"
FINE_TUNED = "m11_ft_windowed_w{weight:g}"  # configs/queue/m11_ft_windowed.yaml
FINE_TUNED_REFERENCE = "m5_ft"  # the E1 models of seed 0 fine-tuned on ngsim_i80 in M5 (configs/queue/m5_laws.yaml)
DATA, FINE_TUNE_DATA = "follownet_highd", "ngsim_i80"
PILOT_FOLDS, FOLDS, SEEDS = (0,), (0, 1, 2, 3, 4), (0,)
STABLE_MIN = 0.9  # D85
RULE = {"stable_min": 0.9, "rmse_change_max": 0.10}  # H-N3 (D128)
LONG_WINDOW = "long_window.json"
CHOSEN_FILE = "e2_windowed_chosen.json"
PENALTY_PARTS = ("margin_term", "gain_term", "pole_term", "existence", "min_margin_window", "max_radius", "max_gain")


def tables_config(runs_root: Path, out_dir: Path) -> TablesConfig:
    """The settings of the M4 tables (intervals: 1000 resamples, 95 %, seed 0) on the design of this milestone."""
    return TablesConfig(runs_root=runs_root, out_dir=out_dir, data=DATA, folds=FOLDS, seeds=SEEDS)


def best_epoch_penalty(run_dir: Path) -> dict[str, float]:
    """The parts of the validation penalty at the best epoch (``val_penalty_<part>`` of the history) and the
    numbers of feasible epochs; empty without metrics.json."""
    path = run_dir / METRICS
    if not path.is_file():
        return {}
    metrics = read_json(path)
    training = metrics.get("training") or {}
    history, best = training.get("history") or [], metrics.get("best_epoch")
    out: dict[str, float] = {}
    if best and 0 < best <= len(history):
        entry = history[best - 1]
        out["val_penalty"] = _number(entry.get("val_penalty"))
        out.update({f"val_{key}": _number(entry.get(f"val_penalty_{key}")) for key in PENALTY_PARTS})
    out["training_h"] = _number(training.get("wall_time_s")) / 3600.0
    return out


def long_window(run_dir: Path) -> dict[str, Any]:
    """``holds`` and ``max_gain`` (equilibria inside the band at speeds in support) of ``long_window.json``."""
    path = run_dir / LONG_WINDOW
    if not path.is_file():
        return {"long_window_holds": None, "long_window_max_gain": np.nan, "long_window_share": np.nan}
    inside = (read_json(path).get("summary") or {}).get("inside") or {}
    return {"long_window_holds": inside.get("holds"), "long_window_max_gain": _number(inside.get("max_gain")),
            "long_window_share": _number(inside.get("share_above"))}  # fmt: skip


class WindowedTables(TableMaker):
    """The tables of A1 on the run rows and intervals of :class:`~cf_stability.eval.tables.TableMaker`."""

    def __init__(self, cfg: TablesConfig, chosen: Mapping[str, float] | None = None) -> None:
        super().__init__(cfg)
        self.given = dict(chosen or {})
        self.choice: dict[str, dict[str, Any]] = {}

    # ----------------------------------------------------------------------------------------- pilot
    def table_pilot(self) -> Table:
        c, t = self.cfg, "e2_windowed_pilot"
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows = []
        for arch in ARCHITECTURES:
            base = self.runs(t, "e1", c.data, arch, SEEDS, (METRICS, EVENTS), folds=PILOT_FOLDS)
            base_drivers = self.drivers(t, base)
            for weight in WEIGHTS:
                experiment = EXPERIMENT.format(weight=weight)
                runs = self.runs(t, experiment, c.data, arch, SEEDS, files, folds=PILOT_FOLDS)
                row: dict[str, Any] = {"architecture": arch, "weight": weight, "experiment": experiment,
                                       "runs": self.trained(runs), "runs_expected": len(PILOT_FOLDS) * len(SEEDS)}
                row["val_rmse_s"] = self.values(t, runs, "val_rmse_s_mean", METRICS).mean()
                drivers = self.drivers(t, runs)
                self.rmse(row, drivers)
                self.versus(row, "rmse_change", base_drivers, drivers)
                self.shares(t, row, runs)
                self.poles(t, row, runs)
                row["epochs"] = self.values(t, runs, "epochs").mean()
                row["best_epoch"] = self.values(t, runs, "best_epoch").mean()
                row["n_feasible"] = self.values(t, runs, "n_feasible").mean()
                row["collided"], row["profiles"] = self.collided(t, runs, "mean")
                for run_dir in runs["run_dir"] if len(runs) else []:  # one run per row (fold 0, seed 0)
                    row.update(best_epoch_penalty(Path(run_dir)))
                    row.update(long_window(Path(run_dir)))
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame["chosen"], frame["rule"] = False, ""
        self.choice = {}
        for arch in ARCHITECTURES:
            group = frame[frame["architecture"] == arch]
            weight, rule = choose_weight(group, STABLE_MIN)
            if arch in self.given:
                weight, rule = float(self.given[arch]), "from the configuration"
            if weight is None:
                self.note(t, f"{arch}: no weight can be chosen ({rule})")
                continue
            best = group[np.isclose(group["weight"], weight)]
            if len(best):
                frame.loc[best.index[0], ["chosen", "rule"]] = [True, rule]
            entry = best.iloc[0] if len(best) else {}
            self.choice[arch] = {"weight": weight, "experiment": EXPERIMENT.format(weight=weight), "rule": rule,
                                 "stable": _number(entry.get("stable")), "val_rmse_s": _number(entry.get("val_rmse_s")),
                                 "complete": bool(len(group) == len(WEIGHTS) and (group["audited"] == 1).all())}  # fmt: skip
        columns = [
            Column("architecture", "architecture", "text"), Column("weight", "weight", "weight"),
            Column("runs", "runs", "int"), Column("val_rmse_s", "val RMSE s (m)"),
            Column("rmse_s", "test RMSE s (m)", ci=True), Column("rmse_change", "test RMSE change vs E1", "pct", 1, ci=True),
            Column("stable", "stable"), Column("unstable", "unstable"), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable_full", "not stable (poles incl.)"), Column("unstable_eq", "unstable among equilibria"),
            Column("max_gain_median", "max gain", digits=3), Column("epochs", "epochs", "int"),
            Column("best_epoch", "best epoch", "int"), Column("n_feasible", "feasible epochs", "int"),
            Column("val_penalty", "val penalty (best epoch)", digits=3),
            Column("val_margin_term", "relu(0.5 - M_w)", digits=3), Column("val_gain_term", "gain term", digits=4),
            Column("val_pole_term", "pole term", digits=4), Column("val_min_margin_window", "min M_w", digits=4),
            Column("val_max_radius", "max radius", digits=4), Column("long_window_max_gain", "long window: max gain",
                                                                     digits=3),
            Column("long_window_holds", "long window holds", "flag"), Column("collided", "collided profiles", "int"),
            Column("training_h", "training (h)", digits=2), Column("chosen", "chosen", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"Pilot of the windowed-gain penalty (D128, kind windowed_gain, every 4th step x 4, 16 speeds per epoch): "
            f"weights {', '.join(f'{w:g}' for w in WEIGHTS)}, {c.data}, fold 0, seed 0; one run per row.",
            "val / test RMSE s: mean spacing RMSE of the validation / test events (m), the test one with its interval "
            "over the drivers (events); change vs E1: relative change of the mean against the E1 run of fold 0, seed "
            "0, paired over the events.",
            "Band shares of the audit (band_numerical of the grid speeds in support): stable, unstable, outside, none; "
            "not stable (poles incl.): with a pole of the full-history loop on or outside the unit circle counted "
            "unstable (full_history.json); unstable among equilibria: share_unstable_numerical; max gain: the largest "
            "numerical gain of the audit.",
            "Validation penalty of the best epoch (unweighted, at the grid speeds 5-30 m/s): its value, the mean "
            "relu(0.5 - M_w), the mean gain term mean_w relu(|G_w| - 0.98)^2, the mean pole term relu(rho - 0.995), "
            "the smallest M_w (1/s^2) and the largest spectral radius over the anchored equilibria; feasible epochs: "
            "validation penalty <= 0.01 (D89).",
            "Long window: the rollout gain over the last 252 s of a 380-s rollout at 0.02, 0.05 and 0.1 rad/s at the "
            "equilibria inside the band (long_window.json, scripts/analysis/long_window_check.py); holds: below 1.02 "
            "everywhere.",
            f"Chosen (D85): among the weights with stable >= {STABLE_MIN:g} the smallest validation RMSE; without such "
            f"a weight the largest stable share; written to {CHOSEN_FILE}.",
        )
        chosen = ", ".join(f"{a} {e['weight']:g}" for a, e in self.choice.items()) or "none"
        return Table(t, "Windowed-gain penalty, pilot: choice of the weight (M11, D128)", notes, frame, columns,
                     [f"Chosen weights: {chosen}."], f"chosen: {chosen}")  # fmt: skip

    # ------------------------------------------------------------------------------------ five folds
    def table_folds(self) -> tuple[Table, list[str]]:
        c, t = self.cfg, "e2_windowed"
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows, branch_lines = [], []
        conditions: dict[str, dict[str, Any]] = {}
        for arch in ARCHITECTURES:
            entry = self.choice.get(arch)
            base = self.runs(t, "e1", c.data, arch, SEEDS, files)
            base_drivers = self.drivers(t, base)
            arms = [("E1 (no penalty)", "e1", base, None)]
            if entry is None:
                self.note(t, f"{arch}: no chosen weight (table e2_windowed_pilot), no row of the windowed penalty")
            else:
                experiment = EXPERIMENT.format(weight=entry["weight"])
                arms.append((f"windowed, weight {entry['weight']:g}", experiment,
                             self.runs(t, experiment, c.data, arch, SEEDS, files), entry["weight"]))  # fmt: skip
            for arm, experiment, runs, weight in arms:
                row: dict[str, Any] = {"architecture": arch, "data": c.data, "arm": arm, "experiment": experiment,
                                       "weight": weight, "runs": self.trained(runs),
                                       "runs_expected": len(c.folds) * len(SEEDS)}  # fmt: skip
                drivers = base_drivers if experiment == "e1" else self.drivers(t, runs)
                self.rmse(row, drivers)
                if experiment != "e1":
                    self.versus(row, "rmse_change", base_drivers, drivers)
                self.shares(t, row, runs)
                _put(row, "stable_ci", self.ci(self.values(t, runs, "band_numerical_stable", AUDIT)))
                self.poles(t, row, runs)
                row["collided"], row["profiles"] = self.collided(t, runs, "mean")
                row["epochs"] = self.values(t, runs, "epochs").mean()
                row["training_h"] = self.values(t, runs, "wall_time_s").mean() / 3600.0
                fold0 = [Path(d) for d in runs["run_dir"]] if len(runs) else []
                fold0 = [d for d in fold0 if d.name == "driver_fold0_seed0"]
                row.update(long_window(fold0[0]) if fold0 else long_window(Path("-")))
                row["complete"] = row["runs"] == row["audited"] == row["runs_expected"]
                rows.append(row)
                if weight is not None:
                    conditions[arch] = {
                        "stable": row.get("stable_ci"), "rmse_change": row.get("rmse_change"),
                        "long_window": row.get("long_window_holds"), "complete": row["complete"],
                    }  # fmt: skip
            if entry is not None:  # fine-tuned on ngsim_i80 (information: the members of the corridor law)
                reference = self.runs(t, FINE_TUNED_REFERENCE, FINE_TUNE_DATA, arch, SEEDS, (METRICS, EVENTS, AUDIT))
                reference_drivers = self.drivers(t, reference)
                tuned = self.runs(t, FINE_TUNED.format(weight=entry["weight"]), FINE_TUNE_DATA, arch, SEEDS,
                                  (METRICS, EVENTS, AUDIT))  # fmt: skip
                for arm, experiment, runs in (("E1 fine-tuned (m5_ft)", FINE_TUNED_REFERENCE, reference),
                                              (f"windowed {entry['weight']:g} fine-tuned",
                                               FINE_TUNED.format(weight=entry["weight"]), tuned)):  # fmt: skip
                    row = {"architecture": arch, "data": FINE_TUNE_DATA, "arm": arm, "experiment": experiment,
                           "weight": None if experiment == FINE_TUNED_REFERENCE else entry["weight"],
                           "runs": self.trained(runs), "runs_expected": len(c.folds) * len(SEEDS)}  # fmt: skip
                    drivers = reference_drivers if experiment == FINE_TUNED_REFERENCE else self.drivers(t, runs)
                    self.rmse(row, drivers)
                    if experiment != FINE_TUNED_REFERENCE:
                        self.versus(row, "rmse_change", reference_drivers, drivers)
                    self.shares(t, row, runs)
                    _put(row, "stable_ci", self.ci(self.values(t, runs, "band_numerical_stable", AUDIT)))
                    row["epochs"] = self.values(t, runs, "epochs").mean()
                    row["training_h"] = self.values(t, runs, "wall_time_s").mean() / 3600.0
                    row["complete"] = row["runs"] == row["audited"] == row["runs_expected"]
                    rows.append(row)
        frame = pd.DataFrame(rows)
        for key in ("rmse_change", "rmse_change_low", "rmse_change_high", "rmse_change_p", "rmse_change_pairs"):
            if key not in frame.columns:
                frame[key] = np.nan
        branch_lines = self.branch(conditions)
        columns = [
            Column("architecture", "architecture", "text"), Column("data", "data", "text"), Column("arm", "arm", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_change", "RMSE change vs reference", "pct", 1, ci=True),
            Column("rmse_change_p", "p", "p"), Column("audited", "audited", "int"),
            Column("stable_ci", "stable", ci=True), Column("unstable", "unstable", ci=True), Column("outside", "outside"),
            Column("none", "none"), Column("not_stable_full", "not stable (poles incl.)", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("max_gain_median", "max gain (median)", digits=3),
            Column("long_window_max_gain", "long window: max gain (fold 0)", digits=3),
            Column("long_window_holds", "long window holds", "flag"), Column("collided", "collided profiles", "int"),
            Column("profiles", "profiles", "int"), Column("epochs", "epochs", "int"),
            Column("training_h", "training (h)", digits=2), Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"Windowed-gain penalty at the weight chosen on the pilot (D85; table e2_windowed_pilot) against E1, "
            f"{c.data}, folds {', '.join(map(str, c.folds))}, seed {SEEDS[0]}; for information the members of the "
            f"corridor laws: the same models fine-tuned on {FINE_TUNE_DATA} with the penalty continued "
            f"({FINE_TUNED.format(weight=0).replace('w0', 'w<weight>')}) against the E1 models fine-tuned in M5 "
            f"({FINE_TUNED_REFERENCE}).",
            "RMSE s: spacing RMSE of the test parts (m), unit driver (event on highD), interval over the drivers; RMSE "
            "change: relative change of the mean RMSE against the reference row of the same data (E1, or m5_ft), "
            "paired over the common drivers; p: Wilcoxon signed-rank test.",
            "Band shares (band_numerical, speeds in support): stable with its interval over the runs (the share of "
            "H-N3), unstable, outside, none; not stable (poles incl.) and unstable among equilibria as in the M4 "
            "tables; max gain: median over the runs of the largest numerical gain of the audit; collided: OpenACC "
            "platoon profiles that collided, of profiles, summed over the runs.",
            "Long window: the check of D128 on the run of fold 0 (the pilot run of the chosen weight; table "
            "long_window).",
            self.poles_note(),
        )
        return Table(t, "Windowed-gain penalty (A1) on five folds against E1 (M11, D128, H-N3)", notes, frame,
                     columns, branch_lines), branch_lines  # fmt: skip

    def branch(self, conditions: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """The branch of the decision rule of H-N3 the numbers fall in, with the conditions per judged architecture."""
        lines = ["Decision rule of H-N3 (D128): confirmed when GRU and LSTM reach a stable band share >= 0.9 at a test "
                 "spacing RMSE increase <= 10 % (5-fold means against E1) and the long-window check holds; partial when "
                 "the share is reached at a larger increase; refuted otherwise. PERL is reported, not judged.", ""]
        status = {}
        for arch in (*JUDGED, "perl"):
            cond = conditions.get(arch)
            if cond is None:
                lines.append(f"- {arch}: no run of the chosen weight")
                status[arch] = None
                continue
            stable, change, holds = _number(cond["stable"]), _number(cond["rmse_change"]), cond["long_window"]
            share_ok = math.isfinite(stable) and stable >= RULE["stable_min"]
            rmse_ok = math.isfinite(change) and change <= RULE["rmse_change_max"]
            status[arch] = {"share": share_ok, "rmse": rmse_ok, "long": holds is True, "complete": cond["complete"]}
            lines.append(f"- {arch}{' (not judged)' if arch not in JUDGED else ''}: stable share {stable:.3f} "
                         f"({'>=' if share_ok else '<'} 0.9), RMSE change {100 * change:+.1f} % ({'<=' if rmse_ok else '>'}"
                         f" 10 %), long-window check {'holds' if holds is True else 'fails' if holds is False else 'n/a'}"
                         f"{'' if cond['complete'] else ' (incomplete)'}")  # fmt: skip
        judged = [status.get(a) for a in JUDGED]
        if any(s is None for s in judged):
            text = "no branch: a judged architecture has no runs"
        elif all(s["share"] and s["rmse"] and s["long"] for s in judged):
            text = "the numbers fall in the branch 'confirmed'"
        elif all(s["share"] for s in judged) and not all(s["rmse"] for s in judged):
            text = "the numbers fall in the branch 'partial' (the share is reached at a larger increase)"
        else:
            text = "the numbers fall in the branch 'refuted' (otherwise)"
            if all(s["share"] and s["rmse"] for s in judged):
                text += ": share and RMSE are met but the long-window check fails"
        complete = all(s is not None and s["complete"] for s in judged)
        lines += ["", f"Branch: {text}{'' if complete else ' (incomplete design)'}."]
        return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("choose", "tables"))
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--chosen", nargs="*", default=[], help="arch=weight: replaces the choice of the pilot")
    args = parser.parse_args(argv)
    out = resolve_path(args.out)
    given = {item.split("=")[0]: float(item.split("=")[1]) for item in args.chosen}
    maker = WindowedTables(tables_config(resolve_path(args.runs_root), out), given)
    pilot = maker.table_pilot()
    write_table(pilot, out)
    (out / CHOSEN_FILE).write_text(json.dumps(maker.choice, indent=2) + "\n", encoding="utf-8")
    print(f"TABLE e2_windowed_pilot: {len(pilot.frame)} rows; {pilot.summary} -> {out / 'e2_windowed_pilot.md'}")
    if args.command == "tables":
        folds, branch = maker.table_folds()
        write_table(folds, out)
        (out / "e2_windowed_branch.md").write_text("# H-N3: the branch of the decision rule (D128)\n\n"
                                                   + "\n".join(branch) + "\n", encoding="utf-8")  # fmt: skip
        print(f"TABLE e2_windowed: {len(folds.frame)} rows -> {out / 'e2_windowed.md'}; {branch[-1]}")
    (out / "missing_windowed.txt").write_text("".join(f"{line}\n" for line in maker.missing), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
