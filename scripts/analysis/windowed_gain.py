"""Tables of the windowed-gain penalty of the recurrent laws (M11, docs/m11_contract.md, D128: algorithm A1, H-N3; M12,
docs/m12_contract.md, D134: algorithm A1', H-N3').

    python scripts/analysis/windowed_gain.py choose               # the pilot table and the weights chosen by D85
    python scripts/analysis/windowed_gain.py tables               # the pilot, the five-fold table and the branch
    python scripts/analysis/windowed_gain.py tables --arm wmax    # the same for the maximum over the frequencies (M12)

Tables into ``runs/_tables/m11/`` (``--arm windowed``, the default) or ``runs/_tables/m12/`` (``--arm wmax``) (CSV and
Markdown twins, ``cf_stability/eval/tables.py`` conventions; the run rows and intervals of its ``TableMaker``):

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

``--arm wmax`` (M12, D134): the same tables for the penalty with the maximum over the frequencies
(``train.penalty.frequency_reduce: max``): ``e2_wmax_pilot`` (weights 0.1, 1, 10, 100: ``e2_wmax_w<weight>``, with the
frequency of the largest excess at the best epoch, ``val_penalty_argmax_omega``), ``e2_wmax_chosen.json``, ``e2_wmax``
(five folds; next to E1 and the chosen weight of the maximum, the windowed penalty of M11 at its chosen weights, the
rows "before", ``e2_windowed_w<weight>`` and ``m11_ft_windowed_w<weight>``) and ``e2_wmax_branch.md``, the branch of the
rule of H-N3' (the rule of H-N3 on the runs of the maximum); ``missing_wmax.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

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
RULE = {"stable_min": 0.9, "rmse_change_max": 0.10}  # H-N3 (D128); H-N3' (D134) the same
LONG_WINDOW = "long_window.json"
CHOSEN_FILE = "e2_windowed_chosen.json"
PENALTY_PARTS = ("margin_term", "gain_term", "pole_term", "existence", "min_margin_window", "max_radius", "max_gain")
# M12 (D134): the penalty with the maximum over the frequencies (configs/queue/e2_wmax_pilot.yaml, e2_wmax.yaml and
# m12_ft_wmax.yaml); the rows "before" are the windowed penalty of M11 at the weights D85 chose on its pilot
WMAX_WEIGHTS = (0.1, 1.0, 10.0, 100.0)
WMAX_EXPERIMENT = "e2_wmax_w{weight:g}"
WMAX_FINE_TUNED = "m12_ft_wmax_w{weight:g}"
WMAX_CHOSEN_FILE = "e2_wmax_chosen.json"
M11_CHOSEN = {"gru": 0.1, "lstm": 10.0, "perl": 1.0}  # D128 (runs/_tables/m11/e2_windowed_chosen.json)
WMAX_PENALTY_PARTS = (*PENALTY_PARTS, "argmax_omega")


class Arm(NamedTuple):
    """The penalty the tables describe: ``windowed`` (M11, D128, H-N3: the gain term is the mean over the frequencies)
    or ``wmax`` (M12, D134, H-N3': the maximum). The texts of ``windowed`` are those of the tables of M11. (A named
    tuple, not a dataclass: the tests load this script without registering it as a module.)"""

    name: str
    weights: tuple[float, ...]
    experiment: str
    fine_tuned: str
    out: str
    pilot: str  # table names
    folds: str
    chosen_file: str
    branch_file: str
    missing_file: str
    hypothesis: str
    decision: str
    milestone: str
    pilot_text: str  # the penalty in the first line of the pilot table
    gain_term: str  # the gain term in the notes
    label: str  # rows of the chosen weight: "<label>, weight <w>", "<label> <w> fine-tuned"
    penalty_parts: tuple[str, ...]
    before: Mapping[str, float] = {}  # wmax: the weights M11 chose (the rows "before"); never mutated


WINDOWED = Arm(
    name="windowed", weights=WEIGHTS, experiment=EXPERIMENT, fine_tuned=FINE_TUNED, out="runs/_tables/m11",
    pilot="e2_windowed_pilot", folds="e2_windowed", chosen_file=CHOSEN_FILE, branch_file="e2_windowed_branch.md",
    missing_file="missing_windowed.txt", hypothesis="H-N3", decision="D128", milestone="M11",
    pilot_text="Pilot of the windowed-gain penalty (D128, kind windowed_gain, every 4th step x 4, 16 speeds per epoch)",
    gain_term="mean_w relu(|G_w| - 0.98)^2", label="windowed", penalty_parts=PENALTY_PARTS,
)  # fmt: skip
WMAX = Arm(
    name="wmax", weights=WMAX_WEIGHTS, experiment=WMAX_EXPERIMENT, fine_tuned=WMAX_FINE_TUNED, out="runs/_tables/m12",
    pilot="e2_wmax_pilot", folds="e2_wmax", chosen_file=WMAX_CHOSEN_FILE, branch_file="e2_wmax_branch.md",
    missing_file="missing_wmax.txt", hypothesis="H-N3'", decision="D134", milestone="M12",
    pilot_text="Pilot of the windowed-gain penalty with the maximum over the frequencies (D134, kind windowed_gain, "
               "frequency_reduce max, every 4th step x 4, 16 speeds per epoch)",
    gain_term="max_w relu(|G_w| - 0.98)^2", label="windowed max", penalty_parts=WMAX_PENALTY_PARTS,
    before=dict(M11_CHOSEN),
)  # fmt: skip
ARMS = {arm.name: arm for arm in (WINDOWED, WMAX)}


def tables_config(runs_root: Path, out_dir: Path) -> TablesConfig:
    """The settings of the M4 tables (intervals: 1000 resamples, 95 %, seed 0) on the design of this milestone."""
    return TablesConfig(runs_root=runs_root, out_dir=out_dir, data=DATA, folds=FOLDS, seeds=SEEDS)


def best_epoch_penalty(run_dir: Path, parts: Sequence[str] = PENALTY_PARTS) -> dict[str, float]:
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
        out.update({f"val_{key}": _number(entry.get(f"val_penalty_{key}")) for key in parts})
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
    """The tables of A1 (and, with ``arm`` WMAX, of A1') on the run rows and intervals of
    :class:`~cf_stability.eval.tables.TableMaker`."""

    def __init__(self, cfg: TablesConfig, chosen: Mapping[str, float] | None = None, arm: Arm = WINDOWED) -> None:
        super().__init__(cfg)
        self.given = dict(chosen or {})
        self.arm = arm
        self.choice: dict[str, dict[str, Any]] = {}

    # ----------------------------------------------------------------------------------------- pilot
    def table_pilot(self) -> Table:
        c, a = self.cfg, self.arm
        t = a.pilot
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows = []
        for arch in ARCHITECTURES:
            base = self.runs(t, "e1", c.data, arch, SEEDS, (METRICS, EVENTS), folds=PILOT_FOLDS)
            base_drivers = self.drivers(t, base)
            for weight in a.weights:
                experiment = a.experiment.format(weight=weight)
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
                    row.update(best_epoch_penalty(Path(run_dir), a.penalty_parts))
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
            self.choice[arch] = {"weight": weight, "experiment": a.experiment.format(weight=weight), "rule": rule,
                                 "stable": _number(entry.get("stable")), "val_rmse_s": _number(entry.get("val_rmse_s")),
                                 "complete": bool(len(group) == len(a.weights) and (group["audited"] == 1).all())}  # fmt: skip
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
        if a is not WINDOWED:  # D134: the frequency of the largest excess at the best epoch (new in M12)
            index = next(k for k, col in enumerate(columns) if col.key == "val_max_radius") + 1
            columns.insert(index, Column("val_max_gain", "max windowed gain (val)", digits=3))
            columns.insert(index + 1, Column("val_argmax_omega", "at omega (rad/s)", digits=3))
        notes = self.header(
            t,
            f"{a.pilot_text}: weights {', '.join(f'{w:g}' for w in a.weights)}, {c.data}, fold 0, seed 0; one run per "
            "row.",
            "val / test RMSE s: mean spacing RMSE of the validation / test events (m), the test one with its interval "
            "over the drivers (events); change vs E1: relative change of the mean against the E1 run of fold 0, seed "
            "0, paired over the events.",
            "Band shares of the audit (band_numerical of the grid speeds in support): stable, unstable, outside, none; "
            "not stable (poles incl.): with a pole of the full-history loop on or outside the unit circle counted "
            "unstable (full_history.json); unstable among equilibria: share_unstable_numerical; max gain: the largest "
            "numerical gain of the audit.",
            "Validation penalty of the best epoch (unweighted, at the grid speeds 5-30 m/s): its value, the mean "
            f"relu(0.5 - M_w), the mean gain term {a.gain_term}, the mean pole term relu(rho - 0.995), "
            "the smallest M_w (1/s^2) and the largest spectral radius over the anchored equilibria; feasible epochs: "
            "validation penalty <= 0.01 (D89)."
            + ("" if a is WINDOWED else " max windowed gain (val) and at omega: the largest windowed gain over the "
               "anchored equilibria and the frequencies at the best epoch and its frequency (val_penalty_max_gain, "
               "val_penalty_argmax_omega of the history: the frequency of the largest excess)."),
            "Long window: the rollout gain over the last 252 s of a 380-s rollout at 0.02, 0.05 and 0.1 rad/s at the "
            "equilibria inside the band (long_window.json, scripts/analysis/long_window_check.py); holds: below 1.02 "
            "everywhere.",
            f"Chosen (D85): among the weights with stable >= {STABLE_MIN:g} the smallest validation RMSE; without such "
            f"a weight the largest stable share; written to {a.chosen_file}.",
        )
        chosen = ", ".join(f"{arch} {e['weight']:g}" for arch, e in self.choice.items()) or "none"
        title = ("Windowed-gain penalty, pilot: choice of the weight (M11, D128)" if a is WINDOWED else
                 "Windowed-gain penalty with the maximum over the frequencies, pilot: choice of the weight (M12, D134)")
        return Table(t, title, notes, frame, columns, [f"Chosen weights: {chosen}."], f"chosen: {chosen}")

    # ------------------------------------------------------------------------------------ five folds
    def table_folds(self) -> tuple[Table, list[str]]:
        c, a = self.cfg, self.arm
        t = a.folds
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows, branch_lines = [], []
        conditions: dict[str, dict[str, Any]] = {}
        for arch in ARCHITECTURES:
            entry = self.choice.get(arch)
            base = self.runs(t, "e1", c.data, arch, SEEDS, files)
            base_drivers = self.drivers(t, base)
            arms = [("E1 (no penalty)", "e1", base, None, False)]
            if arch in a.before:  # M12: the windowed penalty of M11 at its chosen weight, the rows "before"
                w = a.before[arch]
                arms.append((f"windowed mean (M11), weight {w:g}", WINDOWED.experiment.format(weight=w),
                             self.runs(t, WINDOWED.experiment.format(weight=w), c.data, arch, SEEDS, files), w, False))
            if entry is None:
                self.note(t, f"{arch}: no chosen weight (table {a.pilot}), no row of the windowed penalty")
            else:
                experiment = a.experiment.format(weight=entry["weight"])
                arms.append((f"{a.label}, weight {entry['weight']:g}", experiment,
                             self.runs(t, experiment, c.data, arch, SEEDS, files), entry["weight"], True))  # fmt: skip
            for arm, experiment, runs, weight, judged in arms:
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
                if judged:
                    conditions[arch] = {
                        "stable": row.get("stable_ci"), "rmse_change": row.get("rmse_change"),
                        "long_window": row.get("long_window_holds"), "complete": row["complete"],
                    }  # fmt: skip
            if entry is not None:  # fine-tuned on ngsim_i80 (information: the members of the corridor law)
                reference = self.runs(t, FINE_TUNED_REFERENCE, FINE_TUNE_DATA, arch, SEEDS, (METRICS, EVENTS, AUDIT))
                reference_drivers = self.drivers(t, reference)
                tuned_arms = [("E1 fine-tuned (m5_ft)", FINE_TUNED_REFERENCE, reference, None)]
                if arch in a.before:
                    w = a.before[arch]
                    name = WINDOWED.fine_tuned.format(weight=w)
                    tuned_arms.append((f"windowed mean (M11) {w:g} fine-tuned", name,
                                       self.runs(t, name, FINE_TUNE_DATA, arch, SEEDS, (METRICS, EVENTS, AUDIT)), w))
                name = a.fine_tuned.format(weight=entry["weight"])
                tuned_arms.append((f"{a.label} {entry['weight']:g} fine-tuned", name,
                                   self.runs(t, name, FINE_TUNE_DATA, arch, SEEDS, (METRICS, EVENTS, AUDIT)),
                                   entry["weight"]))  # fmt: skip
                for arm, experiment, runs, weight in tuned_arms:
                    row = {"architecture": arch, "data": FINE_TUNE_DATA, "arm": arm, "experiment": experiment,
                           "weight": weight, "runs": self.trained(runs),
                           "runs_expected": len(c.folds) * len(SEEDS)}  # fmt: skip
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
        if a is WINDOWED:
            first = (f"Windowed-gain penalty at the weight chosen on the pilot (D85; table e2_windowed_pilot) against E1, "
                     f"{c.data}, folds {', '.join(map(str, c.folds))}, seed {SEEDS[0]}; for information the members of "
                     f"the corridor laws: the same models fine-tuned on {FINE_TUNE_DATA} with the penalty continued "
                     f"({FINE_TUNED.format(weight=0).replace('w0', 'w<weight>')}) against the E1 models fine-tuned in "
                     f"M5 ({FINE_TUNED_REFERENCE}).")  # fmt: skip
        else:
            before = ", ".join(f"{arch} {w:g}" for arch, w in a.before.items())
            first = (f"Windowed-gain penalty with the maximum over the frequencies ({a.decision}) at the weight chosen "
                     f"on its pilot (D85; table {a.pilot}) against E1, {c.data}, folds {', '.join(map(str, c.folds))}, "
                     f"seed {SEEDS[0]}; the rows 'windowed mean (M11)' are the penalty of D128 (the mean over the "
                     f"frequencies) at the weights chosen on its pilot ({before}), before the change. For information "
                     f"the members of the corridor laws: the same models fine-tuned on {FINE_TUNE_DATA} with the "
                     f"penalty continued ({a.fine_tuned.format(weight=0).replace('w0', 'w<weight>')}; M11: "
                     f"{WINDOWED.fine_tuned.format(weight=0).replace('w0', 'w<weight>')}) against the E1 models "
                     f"fine-tuned in M5 ({FINE_TUNED_REFERENCE}).")  # fmt: skip
        notes = self.header(
            t,
            first,
            "RMSE s: spacing RMSE of the test parts (m), unit driver (event on highD), interval over the drivers; RMSE "
            "change: relative change of the mean RMSE against the reference row of the same data (E1, or m5_ft), "
            "paired over the common drivers; p: Wilcoxon signed-rank test.",
            "Band shares (band_numerical, speeds in support): stable with its interval over the runs (the share of "
            f"{a.hypothesis}), unstable, outside, none; not stable (poles incl.) and unstable among equilibria as in "
            "the M4 tables; max gain: median over the runs of the largest numerical gain of the audit; collided: "
            "OpenACC platoon profiles that collided, of profiles, summed over the runs.",
            f"Long window: the check of {a.decision if a is not WINDOWED else 'D128'} on the run of fold 0 (the pilot "
            f"run of the chosen weight; table {'long_window' if a is WINDOWED else 'long_window_wmax'}).",
            self.poles_note(),
        )
        title = ("Windowed-gain penalty (A1) on five folds against E1 (M11, D128, H-N3)" if a is WINDOWED else
                 "Windowed-gain penalty with the maximum over the frequencies (A1') on five folds against E1 and the "
                 "mean of M11 (M12, D134, H-N3')")
        return Table(t, title, notes, frame, columns, branch_lines), branch_lines

    def branch(self, conditions: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """The branch of the decision rule of H-N3 (H-N3' for the arm wmax) the numbers fall in, with the conditions
        per judged architecture."""
        a = self.arm
        lines = [f"Decision rule of {a.hypothesis} ({a.decision}): confirmed when GRU and LSTM reach a stable band "
                 "share >= 0.9 at a test spacing RMSE increase <= 10 % (5-fold means against E1) and the long-window "
                 "check holds; partial when the share is reached at a larger increase; refuted otherwise. PERL is "
                 "reported, not judged.", ""]  # fmt: skip
        if a is not WINDOWED:
            lines[0] = lines[0].replace("(5-fold means against E1)", "(5-fold means against E1; the penalty with the "
                                        "maximum over the frequencies at the weights chosen on its pilot)")  # fmt: skip
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
        judged = [status.get(arch) for arch in JUDGED]
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
    parser.add_argument("--arm", choices=tuple(ARMS), default="windowed",
                        help="windowed: the mean over the frequencies (M11, D128); wmax: the maximum (M12, D134)")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default=None, help="default: runs/_tables/m11 (windowed), runs/_tables/m12 (wmax)")
    parser.add_argument("--chosen", nargs="*", default=[], help="arch=weight: replaces the choice of the pilot")
    args = parser.parse_args(argv)
    arm = ARMS[args.arm]
    out = resolve_path(args.out or arm.out)
    given = {item.split("=")[0]: float(item.split("=")[1]) for item in args.chosen}
    maker = WindowedTables(tables_config(resolve_path(args.runs_root), out), given, arm)
    pilot = maker.table_pilot()
    write_table(pilot, out)
    (out / arm.chosen_file).write_text(json.dumps(maker.choice, indent=2) + "\n", encoding="utf-8")
    print(f"TABLE {arm.pilot}: {len(pilot.frame)} rows; {pilot.summary} -> {out / (arm.pilot + '.md')}")
    if args.command == "tables":
        folds, branch = maker.table_folds()
        write_table(folds, out)
        (out / arm.branch_file).write_text(f"# {arm.hypothesis}: the branch of the decision rule ({arm.decision})\n\n"
                                           + "\n".join(branch) + "\n", encoding="utf-8")  # fmt: skip
        print(f"TABLE {arm.folds}: {len(folds.frame)} rows -> {out / (arm.folds + '.md')}; {branch[-1]}")
    (out / arm.missing_file).write_text("".join(f"{line}\n" for line in maker.missing), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
