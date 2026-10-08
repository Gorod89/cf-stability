"""Tables of the M4 report from the runs of E1-E5 (docs/m4_contract.md, section 4; D84-D91).

:func:`make_tables` writes every table as Markdown (rounded; the header lines name the units and their
numbers) and as CSV (full precision) to ``<out_dir>/<table>.md|csv``, the weights chosen on the sweep
(D85) to ``chosen_weights.json``, the verdicts of the hypotheses to ``verdicts.md|csv`` and every
missing run, file or value to ``missing.txt``. Nothing missing raises: it leaves an empty cell, and a
verdict drawn from fewer runs than the design has is marked incomplete.

Units (D88): RMSE-type numbers use the driver (``driver_table``: per driver the mean over its events
and over the seeds, or over the folds for the transfer targets; on highD every event is its own driver,
the events carry no driver identifier, D19); numbers of a model (stability shares,
largest gain, growth error, hysteresis area) use the run (fold and seed). Every interval is a
percentile bootstrap over these units (``n_resamples`` resamples, seed ``seed``); paired tests are
Wilcoxon signed-rank tests of the differences over the common units. The growth error of a profile is
taken on the collision-free prefix of the simulated platoon (D109; ``cf_stability.eval.collect``), the
growth error over the whole curve of D107 (none for a platoon that collides) is reported beside it:
growth errors and hysteresis areas are compared over the pairs of runs in which both have a value, and
the tables give the numbers of pairs, of collided profiles, the mean first collided position and the
collision-free share of the profiles. The arm ``e2_lowfreq`` (D110) has its own table and the verdict rows
"H1.2 (combined)". The control of M8 with the monotonicity terms only (D117, ``e2_monotone``) has a table without
verdict in ``m8_dir`` (``runs/_tables/m8``): its fold-0 runs next to E1 and the chosen E2 weight of the same fold.

Local instability (M9, review): for the recurrent laws (``RECURRENT``) the tables e1, e2, e2_lowfreq, e5 and
e2_horizon add to the shares of the numerical rule (D92) the same shares with the local stability of the
full-history loop (the poles of ``full_history.json``, ``scripts/analysis/full_history_audit.py``): a speed
whose loop has a pole on or outside the unit circle is not stable whatever its measured gain
(``unstable_eq_full``, ``not_stable_full``; :func:`full_history_shares`). The verdicts stay on the
pre-specified shares; e1 and e2 show the rule of H1.1 and H1.2 on the new shares for information.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from cf_stability.eval.collect import BASE_COLUMNS, collect_run
from cf_stability.eval.stats import bootstrap_ci, driver_table, holm, paired_comparison

TABLES = ("e1", "e2_sweep", "e2", "e2_lowfreq", "e3", "e4", "e5", "e2_monotone", "e2_horizon")
M8_TABLES = ("e2_monotone",)  # written to m8_dir (runs/_tables/m8)
METRICS, AUDIT, PLATOON, TRANSFER, CERTIFICATE = (
    "metrics.json", "stability.json", "platoon.json", "transfer.json", "certificate.json"
)  # fmt: skip
EVENTS = "test_events.parquet"
FULL_HISTORY = "full_history.json"  # scripts/analysis/full_history_audit.py: poles of the full-history loop (M9)
RECURRENT = ("gru", "lstm", "perl")  # the laws with memory that full_history_audit.py linearises over their window
INSIDE = ("ok", "multiple")  # statuses of an equilibrium inside the band where the speed has one (stability.audit)
POLES = {"h1_1": "unstable_eq_full", "h1_2": "not_stable_full"}  # the shares of the verdicts with the poles (M9)
ERROR_COLUMN = {
    METRICS: "metrics_error", AUDIT: "audit_error", PLATOON: "platoon_error", TRANSFER: "transfer_error",
    CERTIFICATE: "certificate_error",
}  # fmt: skip
DEFAULT_PENALTIES = {  # D90: Jacobian penalty for the memoryless laws, rollout gain penalty for the recurrent ones
    "mlp": "jacobian", "pidl": "jacobian", "residual_idm": "jacobian", "gru": "gain", "lstm": "gain", "perl": "gain",
}  # fmt: skip
DEFAULT_VERDICTS = {
    "h1_1_unstable_min": 0.5, "h1_1_unstable_low_min": 0.3, "h1_1_min_models": 2,
    "h1_2_not_stable_high_max": 0.10, "h1_2_rmse_change_high_max": 0.10,
    "h1_2_rmse_change_low_refute": 0.20, "h1_2_not_stable_low_refute": 0.25,
    "h1_3_confirm_min": 0.30, "h1_3_refute_below": 0.10, "h1_3_min_pairs": 3,
    "h1_5_certified_unstable_high_max": 0.10, "h1_5_mlp_unstable_low_min": 0.30,
}  # fmt: skip
DEFAULT_BASES = {  # the share each verdict uses; a key of the rows of its table, with an interval
    "h1_1": "unstable_eq",  # e1: string-unstable equilibria among the equilibria found (share_unstable_numerical)
    "h1_1_information": "not_stable",  # e1: the verdict this band share would give, shown next to it
    "h1_2": "not_stable",  # e2: band share "not stable" with the penalty
    "h1_5": "unstable",  # e4: band share "unstable"
}
SHARE_NAMES = {
    "unstable_eq": "unstable among equilibria", "unstable_eq_sign": "unstable among equilibria (sign)",
    "unstable": "band share unstable", "not_stable": "band share not stable",
    "unstable_eq_full": "unstable among equilibria (poles incl.)",  # M9: with the poles of full_history.json
    "not_stable_full": "band share not stable (poles incl.)",
}  # fmt: skip
BASES = {  # the shares a verdict can use
    "h1_1": ("unstable_eq", "unstable_eq_sign", "unstable", "not_stable"),
    "h1_1_information": ("unstable_eq", "unstable_eq_sign", "unstable", "not_stable"),
    "h1_2": ("not_stable", "unstable", "unstable_eq"), "h1_5": ("unstable", "not_stable", "unstable_eq"),
}  # fmt: skip
EMPTY = {"estimate": np.nan, "low": np.nan, "high": np.nan, "n": 0}
NO_PAIRS = {"n": 0, "mean_a": np.nan, "mean_b": np.nan, "relative_change": np.nan, "ci_low": np.nan,
            "ci_high": np.nan, "p_value": np.nan}  # fmt: skip


def experiment_name(prefix: str, kind: str, weight: float) -> str:
    """``e2_jacobian_w0.1``: the experiment of a penalty and weight (docs/m4_contract.md, section 0)."""
    return f"{prefix}_{kind}_w{weight:g}"


@dataclasses.dataclass(frozen=True)
class TablesConfig:
    """Design of the experiments and settings of the tables (``configs/make_tables.yaml``)."""

    runs_root: Path
    out_dir: Path
    tables: tuple[str, ...] = TABLES
    n_resamples: int = 1000
    level: float = 0.95
    seed: int = 0
    data: str = "follownet_highd"
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    learned: tuple[str, ...] = ("mlp", "pidl", "gru", "lstm", "perl", "residual_idm")
    baselines: tuple[str, ...] = ("persistence", "newell", "ovm", "idm", "knn")
    reference: str = "idm"
    h1_1_models: tuple[str, ...] = ("mlp", "gru", "lstm")
    penalties: Mapping[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_PENALTIES))
    weights: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    stable_min: float = 0.9
    chosen_weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    targets: tuple[str, ...] = ("ngsim_i80", "ngsim_us101", "waymo")
    fine_tune_data: str = "ngsim_i80"
    views: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {"openacc_acc": "acc", "openacc_human": "human"}
    )
    e5_models: tuple[str, ...] = ("idm", "mlp", "gru", "lstm")
    e5_penalised: tuple[str, ...] = ("mlp", "gru", "lstm")
    lowfreq_architectures: tuple[str, ...] = ("gru", "lstm", "perl")  # D110: rollout gain + Jacobian penalty
    lowfreq_weights: tuple[float, ...] = (0.1, 1.0)  # train.penalty.jacobian_weight of the pilot
    lowfreq_experiment: str = "e2_combined_j{weight:g}"
    lowfreq_pilot_folds: tuple[int, ...] = (0,)  # every weight; the chosen one on every fold
    lowfreq_chosen: Mapping[str, float] = dataclasses.field(default_factory=dict)  # replaces the D85 choice
    monotone_experiment: str = "e2_monotone_w1"  # D117: the monotonicity terms of RACER only, weight 1
    monotone_architectures: tuple[str, ...] = ("mlp", "pidl", "residual_idm")
    monotone_folds: tuple[int, ...] = (0,)  # with the first of seeds; E1 and the chosen E2 weight of the same runs
    horizon_experiment: str = "e2_gain_long_w0.1"  # revision of 2026-10-07: the rollout gain penalty with a long
    horizon_architectures: tuple[str, ...] = ("gru", "lstm")  # rollout (380 s, gain over the last 252 s), E2 weight
    horizon_folds: tuple[int, ...] = (0,)  # with the first of seeds; E1 and the chosen E2 weight of the same runs
    horizon_omega_max: float = 0.1  # rad/s: the low-frequency band the 20-s window of E2 cannot resolve (<= 0.1)
    verdicts: Mapping[str, float] = dataclasses.field(default_factory=lambda: dict(DEFAULT_VERDICTS))
    bases: Mapping[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_BASES))
    m8_dir: Path | None = None  # the tables of M8_TABLES; None: <out_dir>/../m8

    def __post_init__(self) -> None:
        object.__setattr__(self, "bases", {**DEFAULT_BASES, **self.bases})  # given keys replace the defaults
        object.__setattr__(self, "verdicts", {**DEFAULT_VERDICTS, **self.verdicts})
        for verdict, share in self.bases.items():
            if share not in BASES.get(verdict, ()):
                raise ValueError(f"bases.{verdict} must be one of {BASES.get(verdict, ())}, got {share!r}")
        unknown = set(self.tables) - set(TABLES)
        if unknown:
            raise ValueError(f"unknown tables {sorted(unknown)}, known: {list(TABLES)}")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], runs_root: str | Path, out_dir: str | Path, m8_dir: str | Path | None = None
    ) -> TablesConfig:
        names = {f.name for f in dataclasses.fields(cls)} - {"runs_root", "out_dir", "m8_dir"}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"unknown keys of the table config: {sorted(unknown)}")
        values = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
        for key in ("penalties", "chosen_weights", "views", "lowfreq_chosen"):
            if key in values:
                values[key] = dict(values[key] or {})
        values["verdicts"] = {**DEFAULT_VERDICTS, **(raw.get("verdicts") or {})}
        values["bases"] = {**DEFAULT_BASES, **(raw.get("bases") or {})}
        return cls(runs_root=Path(runs_root), out_dir=Path(out_dir), m8_dir=None if m8_dir is None else Path(m8_dir),
                   **values)  # fmt: skip

    @property
    def m8(self) -> Path:
        return self.m8_dir if self.m8_dir is not None else self.out_dir.parent / "m8"

    def folder(self, table: str) -> Path:
        """The folder of a table: ``m8`` for the tables of M8, else ``out_dir``."""
        return self.m8 if table in M8_TABLES else self.out_dir


# --------------------------------------------------------------------------------------------- rendering


@dataclasses.dataclass(frozen=True)
class Column:
    """A column of the Markdown table: ``key`` of the frame (with ``ci``: ``key [key_low, key_high]``)."""

    key: str
    header: str
    kind: str = "num"  # text | flag | int | num | pct | p | weight
    digits: int = 2
    ci: bool = False


@dataclasses.dataclass
class Table:
    name: str
    title: str
    notes: list[str]
    frame: pd.DataFrame
    columns: list[Column]
    footer: list[str] = dataclasses.field(default_factory=list)
    summary: str = ""  # the end of the printed line: verdicts, choices


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value: Any, kind: str, digits: int = 2) -> str:
    if _blank(value):
        return ""
    if kind == "text":
        return str(value).replace("|", "\\|")
    if kind == "flag":
        return "yes" if value is True or value == 1 else ""
    if kind == "weight":
        return f"{float(value):g}"
    if kind == "int":
        return str(int(value))
    if kind == "pct":
        return f"{100.0 * float(value):+.{digits}f} %"
    if kind == "p":
        return "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"
    return f"{float(value):.{digits}f}"


def markdown(table: Table) -> str:
    """The table in Markdown: title, header lines, rounded cells, footer."""
    lines = [f"# {table.title}", "", *(f"- {note}" for note in table.notes), ""]
    lines.append("| " + " | ".join(c.header for c in table.columns) + " |")
    lines.append("|" + "|".join("---" if c.kind in ("text", "flag") else "---:" for c in table.columns) + "|")
    for row in table.frame.to_dict("records"):
        cells = []
        for c in table.columns:
            cell = _cell(row.get(c.key), c.kind, c.digits)
            low, high = (_cell(row.get(f"{c.key}_{end}"), c.kind, c.digits) for end in ("low", "high"))
            cells.append(f"{cell} [{low}, {high}]" if c.ci and cell and low and high else cell)
        lines.append("| " + " | ".join(cells) + " |")
    if table.footer:
        lines += ["", *table.footer]
    return "\n".join(lines) + "\n"


def write_table(table: Table, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.frame.to_csv(out_dir / f"{table.name}.csv", index=False)
    (out_dir / f"{table.name}.md").write_text(markdown(table), encoding="utf-8")


def _ci_text(row: Mapping[str, Any], key: str, kind: str = "num", digits: int = 2) -> str:
    value = _cell(row.get(key), kind, digits)
    low, high = _cell(row.get(f"{key}_low"), kind, digits), _cell(row.get(f"{key}_high"), kind, digits)
    return f"{value} [{low}, {high}]" if value and low and high else value or "n/a"


# --------------------------------------------------------------------------------------------- verdicts


def _present(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return not any(_blank(row.get(key)) for key in keys)


def verdict_h1_1(row: Mapping[str, Any], v: Mapping[str, float], share: str = "unstable_eq") -> str:
    """The share ``share`` (by default the unstable equilibria among the equilibria found) at least
    ``h1_1_unstable_min`` with the lower end above ``h1_1_unstable_low_min``, and the RMSE below the
    reference (upper end of the relative difference below 0)."""
    if not _present(row, (share, f"{share}_low", "rmse_vs_ref_high")):
        return ""
    unstable = row[share] >= v["h1_1_unstable_min"] and row[f"{share}_low"] > v["h1_1_unstable_low_min"]
    return "holds" if unstable and row["rmse_vs_ref_high"] < 0.0 else "does not hold"


def verdict_overall(verdicts: Sequence[str], need: int) -> str:
    """``holds`` when at least ``need`` of the verdicts hold, ``does not hold`` when that is out of reach."""
    n_hold, n_not = verdicts.count("holds"), verdicts.count("does not hold")
    if n_hold >= need:
        return "holds"
    return "does not hold" if n_not > len(verdicts) - need else "open"


def verdict_h1_2(row: Mapping[str, Any], v: Mapping[str, float], share: str = "not_stable") -> str:
    """Confirmed: upper end of the share ``share`` with penalty (by default the band share ``not stable``)
    below ``h1_2_not_stable_high_max`` and upper end of the RMSE change at most ``h1_2_rmse_change_high_max``;
    refuted: lower end of the RMSE change above ``h1_2_rmse_change_low_refute`` or lower end of the share above
    ``h1_2_not_stable_low_refute``."""
    low, high = f"{share}_e2_low", f"{share}_e2_high"
    if not _present(row, (low, high, "rmse_change_low", "rmse_change_high")):
        return ""
    if row[high] < v["h1_2_not_stable_high_max"] and row["rmse_change_high"] <= v["h1_2_rmse_change_high_max"]:
        return "confirmed"
    worse = row["rmse_change_low"] > v["h1_2_rmse_change_low_refute"]
    return "refuted" if worse or row[low] > v["h1_2_not_stable_low_refute"] else "open"


def verdict_h1_3(row: Mapping[str, Any], v: Mapping[str, float]) -> str:
    """Open with fewer than ``h1_3_min_pairs`` pairs of runs that both have a growth error (``reduction_pairs``;
    a platoon that collides has none); else confirmed: relative reduction of the growth error at least
    ``h1_3_confirm_min`` with an interval that excludes 0; refuted: reduction below ``h1_3_refute_below``."""
    pairs = row.get("reduction_pairs")
    if _blank(pairs) or pairs < v["h1_3_min_pairs"]:
        return "open"
    if not _present(row, ("reduction", "reduction_low", "reduction_high")):
        return ""
    if row["reduction"] >= v["h1_3_confirm_min"] and row["reduction_low"] > 0.0:
        return "confirmed"
    return "refuted" if row["reduction"] < v["h1_3_refute_below"] else "open"


def verdict_h1_5(
    certified: Mapping[str, Any], mlp: Mapping[str, Any], v: Mapping[str, float], share: str = "unstable"
) -> str:
    """After fine-tuning: upper end of the share ``share`` (by default the band share ``unstable``) of the
    certified hybrid below ``h1_5_certified_unstable_high_max`` and lower end of that of the MLP above
    ``h1_5_mlp_unstable_low_min``."""
    if not (_present(certified, (f"{share}_high",)) and _present(mlp, (f"{share}_low",))):
        return ""
    holds = (certified[f"{share}_high"] < v["h1_5_certified_unstable_high_max"]
             and mlp[f"{share}_low"] > v["h1_5_mlp_unstable_low_min"])  # fmt: skip
    return "holds" if holds else "does not hold"


def choose_weight(sweep: pd.DataFrame, stable_min: float) -> tuple[float | None, str]:
    """D85: among the weights whose mean share ``stable`` is at least ``stable_min`` the one with the smallest
    mean validation RMSE; without such a weight the largest share (ties: smaller validation RMSE, then
    smaller weight). ``sweep`` has the columns ``weight``, ``stable``, ``val_rmse_s``; weights without a
    share are left out."""
    known = sweep.dropna(subset=["stable"])
    if known.empty:
        return None, "no audited run"
    reaching = known[(known["stable"] >= stable_min) & known["val_rmse_s"].notna()]
    if len(reaching):
        best = reaching.sort_values(["val_rmse_s", "weight"], kind="stable").iloc[0]
        return float(best["weight"]), f"stable >= {stable_min:g}: smallest validation RMSE"
    order = known.sort_values(["stable", "val_rmse_s", "weight"], ascending=[False, True, True], kind="stable")
    return float(order.iloc[0]["weight"]), f"no weight reaches stable >= {stable_min:g}: largest stable share"


def full_history_shares(payload: Mapping[str, Any], n_band: Any) -> dict[str, float]:
    """The shares of one run with local instability added (M9), from its ``full_history.json`` (``payload``: per
    grid speed with an equilibrium the numerical verdict of the audit and whether every pole of the full-history
    loop lies inside the unit circle), over the speeds in support as the audit takes them:

    * ``unstable_eq_full``: among the speeds with an equilibrium (those of ``share_unstable_numerical``), the share
      whose numerical verdict is unstable or whose loop has a pole on or outside the unit circle (a pole outside
      decides a speed without numerical verdict as well);
    * ``not_stable_full``: 1 - the share of the ``n_band`` speeds that have a band (``n_band.support``, those of
      ``band_numerical``) whose equilibrium lies inside the band, is stable by the numerical rule and has every
      pole inside.

    With the shares of the numerical rule recomputed from the file (``unstable_eq``, ``stable``), which must equal
    those of the audit. NaN where a share has no speed."""
    unstable_full: list[bool] = []
    unstable: list[bool] = []
    stable = stable_full = 0
    for r in payload.get("equilibria") or ():
        if not isinstance(r, Mapping) or not r.get("in_support"):
            continue
        numerical, poles = r.get("numerical_unstable"), r.get("poles_stable")
        if numerical is not None:
            unstable.append(bool(numerical))
        if numerical is True or poles is False:
            unstable_full.append(True)
        elif numerical is False and poles is True:
            unstable_full.append(False)
        if r.get("in_band") is not None and r.get("status") in INSIDE and numerical is False:
            stable += 1
            stable_full += poles is True
    n_band = _number(n_band)
    banded = math.isfinite(n_band) and n_band > 0

    def share(flags: list[bool]) -> float:
        return sum(flags) / len(flags) if flags else math.nan

    return {
        "unstable_eq_full": share(unstable_full), "not_stable_full": 1.0 - stable_full / n_band if banded else math.nan,
        "unstable_eq": share(unstable), "stable": stable / n_band if banded else math.nan,
    }  # fmt: skip


# --------------------------------------------------------------------------------------------- the maker


def _put(row: dict[str, Any], key: str, ci: Mapping[str, Any]) -> None:
    row[key], row[f"{key}_low"], row[f"{key}_high"] = ci["estimate"], ci["low"], ci["high"]


def _number(value: Any) -> float:
    """A finite float, NaN for anything else (None, text, infinity)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _nanmean(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.mean(finite)) if finite else math.nan


def _column(runs: pd.DataFrame, name: str) -> pd.Series:
    return runs[name] if name in runs.columns else pd.Series(np.nan, index=runs.index, dtype=object)


class TableMaker:
    """Builds the tables; remembers the rows of the runs and the per-event frames it has read."""

    def __init__(self, cfg: TablesConfig) -> None:
        self.cfg, self.v = cfg, cfg.verdicts
        self.missing: list[str] = []
        self.verdicts: list[dict[str, Any]] = []
        self.counts: dict[str, list[int]] = {}  # table -> [expected runs, existing runs]
        self.chosen: dict[str, float] = {}
        self.sweep_choice: dict[str, dict[str, Any]] = {}
        self._rows: dict[Path, dict[str, Any]] = {}
        self._frames: dict[Path, pd.DataFrame | None] = {}
        self._full: dict[Path, dict[str, float] | str] = {}  # run -> full_history_shares, or why it has none

    # ------------------------------------------------------------------------------------- reading
    def label(self, path: Path) -> str:
        try:
            return path.relative_to(self.cfg.runs_root).as_posix()
        except ValueError:
            return path.as_posix()

    def note(self, table: str, text: str) -> None:
        self.missing.append(f"[{table}] {text}")

    def runs(
        self, table: str, experiment: str, data: str, model: str, seeds: Sequence[int], files: Sequence[str],
        folds: Sequence[int] | None = None,
    ) -> pd.DataFrame:  # fmt: skip
        """One row (``collect_run``) per expected run (every fold of ``folds``, by default those of the
        design, and every seed of ``seeds``) that exists; missing runs and files and files that hold an
        error are noted (one line for a group none of whose runs exists)."""
        counts = self.counts.setdefault(table, [0, 0])
        expected = list(itertools.product(self.cfg.folds if folds is None else folds, seeds))
        group = self.cfg.runs_root / experiment / data / model
        if len(expected) > 1 and not any((group / f"driver_fold{f}_seed{s}").is_dir() for f, s in expected):
            counts[0] += len(expected)
            folds_text = ", ".join(str(f) for f in sorted({f for f, _ in expected}))
            seeds_text = ", ".join(str(s) for s in sorted({s for _, s in expected}))
            self.note(table, f"{self.label(group)}: run missing (all {len(expected)}: folds {folds_text}, seeds "
                             f"{seeds_text})")  # fmt: skip
            expected = []
        rows = []
        for fold, seed in expected:
            run = group / f"driver_fold{fold}_seed{seed}"
            counts[0] += 1
            if not run.is_dir():
                self.note(table, f"{self.label(run)}: run missing")
                continue
            counts[1] += 1
            absent = [name for name in files if not (run / name).exists()]
            if absent:
                self.note(table, f"{self.label(run)}: {', '.join(absent)} missing")
            if run not in self._rows:
                self._rows[run] = collect_run(run, experiment)
            row = self._rows[run]
            for name in files:
                error = row.get(ERROR_COLUMN.get(name, ""))
                if isinstance(error, str):
                    self.note(table, f"{self.label(run)}: {name} {error}")
            rows.append(row)
        frame = pd.DataFrame(rows)
        absent = {name: pd.Series(np.nan, index=frame.index, dtype=object) for name in BASE_COLUMNS
                  if name not in frame.columns}  # fmt: skip
        return pd.concat([frame, pd.DataFrame(absent, index=frame.index)], axis=1) if absent else frame

    def values(self, table: str, runs: pd.DataFrame, column: str, source: str | None = None) -> pd.Series:
        """``column`` as floats indexed by (fold, seed); with ``source``, a run whose ``source`` file exists
        and holds no error but has no value is noted."""
        if runs.empty:
            return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["fold", "seed"]))
        values = pd.to_numeric(_column(runs, column), errors="coerce").astype(float)
        if source is not None:
            errors = _column(runs, ERROR_COLUMN[source])
            for run_dir, value, error in zip(runs["run_dir"], values, errors):
                if math.isnan(value) and (Path(run_dir) / source).exists() and not isinstance(error, str):
                    self.note(table, f"{self.label(Path(run_dir))}: no value {column} in {source}")
        index = pd.MultiIndex.from_arrays([runs["fold"].to_numpy(), runs["seed"].to_numpy()], names=["fold", "seed"])
        return pd.Series(values.to_numpy(), index=index)

    def frames(self, table: str, runs: pd.DataFrame, name: str = EVENTS) -> list[pd.DataFrame]:
        """The per-event parquet ``name`` of every run that has it (missing ones are noted by :meth:`runs`)."""
        out = []
        for run_dir in runs["run_dir"] if len(runs) else []:
            path = Path(run_dir) / name
            if path not in self._frames:
                self._frames[path] = None
                if path.exists():
                    try:
                        self._frames[path] = pd.read_parquet(path)
                    except Exception as exc:  # a file cut short counts as missing
                        self.note(table, f"{self.label(path)}: unreadable ({type(exc).__name__})")
            if self._frames[path] is not None:
                out.append(self._frames[path])
        return out

    def drivers(self, table: str, runs: pd.DataFrame, name: str = EVENTS) -> pd.DataFrame | None:
        frames = self.frames(table, runs, name)
        return driver_table(frames) if frames else None

    # ---------------------------------------------------------------------------------- statistics
    def ci(self, values: Any, statistic: Callable[[np.ndarray], float] = np.mean) -> dict[str, Any]:
        c = self.cfg
        return bootstrap_ci(np.asarray(values, dtype=float), None, statistic, c.n_resamples, c.level, c.seed)

    def compare(self, a: pd.Series | None, b: pd.Series | None) -> dict[str, Any]:
        if a is None or b is None or a.empty or b.empty:
            return dict(NO_PAIRS)
        return paired_comparison(a, b, n_resamples=self.cfg.n_resamples, level=self.cfg.level, seed=self.cfg.seed)

    def rmse(self, row: dict[str, Any], drivers: pd.DataFrame | None) -> None:
        """Mean spacing RMSE over the drivers with its interval, number of drivers and collision rate."""
        if drivers is None or drivers.empty:
            _put(row, "rmse_s", EMPTY)
            row["drivers"], row["collision_rate"] = 0, np.nan
            return
        _put(row, "rmse_s", self.ci(drivers["rmse_s"]))
        row["drivers"] = len(drivers)
        row["collision_rate"] = float(drivers["collided"].mean()) if "collided" in drivers else np.nan

    def versus(self, row: dict[str, Any], key: str, reference: pd.DataFrame | None, other: pd.DataFrame | None) -> None:
        """Relative change of the mean spacing RMSE of ``other`` against ``reference``, paired over the drivers."""
        a = None if reference is None else reference["rmse_s"]
        cmp = self.compare(a, None if other is None else other["rmse_s"])
        row[key], row[f"{key}_low"], row[f"{key}_high"] = cmp["relative_change"], cmp["ci_low"], cmp["ci_high"]
        row[f"{key}_p"], row[f"{key}_pairs"] = cmp["p_value"], cmp["n"]

    def unstable_eq(self, table: str, runs: pd.DataFrame) -> dict[str, Any]:
        """Share of the string-unstable equilibria among the equilibria the audit analysed (inside the band or
        outside it; ``share_unstable_numerical``, speeds in support), mean over the runs with interval."""
        return self.ci(self.values(table, runs, "share_unstable_numerical", AUDIT))

    def shares(self, table: str, row: dict[str, Any], runs: pd.DataFrame) -> None:
        """Shares of ``band_numerical`` (support) and of the sign criterion, mean over the runs; intervals over
        the runs for ``unstable`` and ``not stable``; the unstable equilibria among the equilibria found, by
        the numerical rule and by the sign criterion, with intervals; median of the largest gain."""
        stable = self.values(table, runs, "band_numerical_stable", AUDIT)
        row["audited"] = int(stable.notna().sum())
        for share in ("stable", "outside", "none"):
            row[share] = self.values(table, runs, f"band_numerical_{share}").mean()
        _put(row, "unstable", self.ci(self.values(table, runs, "band_numerical_unstable")))
        _put(row, "not_stable", self.ci(1.0 - stable))
        row["unstable_sign"] = self.values(table, runs, "band_sign_unstable").mean()
        _put(row, "unstable_eq", self.unstable_eq(table, runs))
        _put(row, "unstable_eq_sign", self.ci(self.values(table, runs, "share_unstable_sign")))
        row["max_gain_median"] = self.values(table, runs, "max_gain").median()

    def full_history(self, run: Mapping[str, Any]) -> dict[str, float] | str:
        """:func:`full_history_shares` of a run (its row of ``collect_run``) checked against its audit: the shares of
        the numerical rule recomputed from ``full_history.json`` must equal ``share_unstable_numerical`` and the band
        share stable of ``stability.json`` (else the audit changed after the file was written). The reason as a
        string when the run has none, empty for a run without a usable audit (noted by :meth:`runs`)."""
        run_dir = Path(run["run_dir"])
        if isinstance(run.get(ERROR_COLUMN[AUDIT]), str) or not (run_dir / AUDIT).exists():
            return ""
        path = run_dir / FULL_HISTORY
        if not path.exists():
            return f"{FULL_HISTORY} missing"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"{FULL_HISTORY} unreadable ({type(exc).__name__})"
        shares = full_history_shares(payload, run.get("n_band_support"))
        stored = {"unstable_eq": "share_unstable_numerical", "stable": "band_numerical_stable"}
        for key, column in stored.items():
            mine, theirs = shares[key], _number(run.get(column))
            if not (math.isnan(mine) and math.isnan(theirs)) and not math.isclose(mine, theirs, abs_tol=1e-9):
                return (f"{FULL_HISTORY} does not match {AUDIT} ({column} {mine:.4f} from the file, {theirs:.4f} in "
                        "the audit; rerun scripts/analysis/full_history_audit.py)")  # fmt: skip
        return shares

    def poles(self, table: str, row: dict[str, Any], runs: pd.DataFrame, suffix: str = "") -> None:
        """The shares with local instability added (M9; :func:`full_history_shares`) of the runs of a recurrent law,
        mean over the runs with interval, as ``unstable_eq_full<suffix>`` and ``not_stable_full<suffix>``; empty for
        the memoryless laws. A run without a usable ``full_history.json`` is noted and left out."""
        values: dict[str, list[float]] = {key: [] for key in POLES.values()}
        for run in runs.to_dict("records") if len(runs) else []:
            if run.get("model") not in RECURRENT:
                continue
            run_dir = Path(run["run_dir"])
            if run_dir not in self._full:
                self._full[run_dir] = self.full_history(run)
            shares = self._full[run_dir]
            if isinstance(shares, str):
                if shares:
                    self.note(table, f"{self.label(run_dir)}: {shares}")
                continue
            for key, found in values.items():
                found.append(shares[key])
        for key, found in values.items():
            _put(row, f"{key}{suffix}", self.ci(found) if found else EMPTY)

    @staticmethod
    def poles_note() -> str:
        """The header line of the shares with local instability added (M9)."""
        laws = ", ".join(name.upper() for name in RECURRENT)
        return (
            f"Columns (poles incl.): the shares with the local stability of the recurrent laws ({laws}) added to the "
            "numerical rule, from the poles of their two-vehicle loop linearised over the whole window at the "
            "equilibria of the audit (Section 3.2 of the paper; scripts/analysis/full_history_audit.py, "
            "full_history.json of the run): a speed whose loop has a pole on or outside the unit circle is not "
            "stable whatever its measured gain. unstable among equilibria (poles incl.): numerical verdict unstable "
            "or a pole outside, over the speeds of share_unstable_numerical; not stable (poles incl.): 1 - the share "
            "of the speeds of band_numerical whose equilibrium lies inside the band, is stable by the numerical rule "
            "and has every pole inside; unit run, mean over the runs. Blank for the memoryless laws and for runs "
            "without full_history.json (noted in missing.txt as 'full_history.json missing')."
        )

    @staticmethod
    def trained(runs: pd.DataFrame) -> int:
        return int(runs["config_hash"].notna().sum()) if len(runs) else 0

    def collided(self, table: str, runs: pd.DataFrame, kind: str) -> tuple[int, int]:
        """Profiles of the growth error ``kind`` (``mean``: every OpenACC profile, ``human``, ``acc``) whose
        platoon collided, and all of them, summed over the runs."""
        n = self.values(table, runs, f"platoon_growth_collided_{kind}").fillna(0.0)
        total = self.values(table, runs, f"platoon_growth_profiles_{kind}").fillna(0.0)
        return int(n.sum()), int(total.sum())

    @staticmethod
    def platoon_runs(runs: pd.DataFrame) -> int:
        """Runs with a readable platoon.json (missing and broken ones are noted by :meth:`runs`)."""
        if runs.empty:
            return 0
        errors = _column(runs, ERROR_COLUMN[PLATOON])
        return sum((Path(run_dir) / PLATOON).exists() and not isinstance(error, str)
                   for run_dir, error in zip(runs["run_dir"], errors))  # fmt: skip

    def growth(self, table: str, runs: pd.DataFrame, kind: str, prefix: bool = True) -> pd.Series:
        """Growth errors of the runs (index fold, seed) for the profiles ``kind`` (``mean``: every OpenACC
        profile, ``human``, ``acc``): on the collision-free prefix of the platoon (D109,
        ``platoon_prefix_growth_error_<kind>``) or, with ``prefix=False``, over the whole curves (D107,
        ``platoon_growth_error_<kind>``: none when every profile of the kind collided). A run with a readable
        platoon.json and no prefix value is noted, unless none of its profiles has a collision-free prefix
        of three positions (``platoon_prefix_profiles_<kind>`` = 0)."""
        column = f"platoon_{'prefix_' if prefix else ''}growth_error_{kind}"
        values = self.values(table, runs, column)
        if runs.empty or not prefix:
            return values
        counted = pd.to_numeric(_column(runs, f"platoon_prefix_profiles_{kind}"), errors="coerce")
        errors = _column(runs, ERROR_COLUMN[PLATOON])
        for run_dir, value, n, error in zip(runs["run_dir"], values, counted, errors):
            readable = (Path(run_dir) / PLATOON).exists() and not isinstance(error, str)
            if math.isnan(value) and readable and n != 0:  # NaN != 0: a file without the prefix columns
                self.note(table, f"{self.label(Path(run_dir))}: no value {column} in {PLATOON}")
        return values

    def positions(self, table: str, row: dict[str, Any], runs: pd.DataFrame, kind: str, suffix: str = "") -> None:
        """The two descriptive outcomes of D109 with their intervals over the runs: the mean first collided
        position over the profiles ``kind`` (51 without collision: one past the last vehicle) and the
        collision-free share of the profiles."""
        _put(row, f"first_collided{suffix}", self.ci(self.values(table, runs, f"platoon_first_collided_{kind}")))
        _put(row, f"collision_free{suffix}", self.ci(self.values(table, runs, f"platoon_collision_free_{kind}")))

    def verdict(
        self, hypothesis: str, unit: str, verdict: str, complete: bool, basis: str, information: str = ""
    ) -> None:
        self.verdicts.append({
            "hypothesis": hypothesis, "unit": unit, "verdict": verdict, "complete": complete, "basis": basis,
            "information": information,
        })  # fmt: skip

    def header(self, table: str, *lines: str) -> list[str]:
        c = self.cfg
        expected, existing = self.counts.get(table, [0, 0])
        return [
            f"Runs: {existing} of {expected} expected runs exist; missing runs, files and values: missing.txt.",
            *lines,
            f"Intervals: {100 * c.level:g} % percentile bootstrap over the units, {c.n_resamples} resamples, "
            f"seed {c.seed}; complete: every run of the design is there and audited.",
        ]

    # ---------------------------------------------------------------------------------------- E1
    def table_e1(self) -> Table:
        c, t, v = self.cfg, "e1", self.v
        models = [*c.learned, *[m for m in c.baselines if m not in c.learned]]
        loaded = {}
        for model in models:
            seeds = c.seeds if model in c.learned else c.seeds[:1]
            runs = self.runs(t, "e1", c.data, model, seeds, (METRICS, AUDIT, EVENTS))
            loaded[model] = (runs, self.drivers(t, runs), len(c.folds) * len(seeds))
        reference = loaded[c.reference][1] if c.reference in loaded else None
        reference_complete = c.reference in loaded and self.trained(loaded[c.reference][0]) == loaded[c.reference][2]
        basis, information = c.bases["h1_1"], c.bases["h1_1_information"]
        rows = []
        for model in models:
            runs, drivers, expected = loaded[model]
            row: dict[str, Any] = {"model": model, "runs": self.trained(runs), "runs_expected": expected}
            self.rmse(row, drivers)
            if model != c.reference:
                self.versus(row, "rmse_vs_ref", reference, drivers)
            self.shares(t, row, runs)
            self.poles(t, row, runs)
            row["complete"] = row["runs"] == row["audited"] == expected
            if model in c.h1_1_models:
                row["h1_1"] = verdict_h1_1(row, v, basis)
                row["h1_1_information"] = verdict_h1_1(row, v, information)
                row["h1_1_poles"] = verdict_h1_1(row, v, POLES["h1_1"])  # M9, for information
                row["complete"] = row["complete"] and reference_complete
                versus = _ci_text(row, "rmse_vs_ref", "pct", 1)
                text = f"{SHARE_NAMES[basis]} {_ci_text(row, basis)}, RMSE vs {c.reference} {versus}"
                about = f"{SHARE_NAMES[information]} {_ci_text(row, information)}: {row['h1_1_information'] or 'n/a'}"
                self.verdict("H1.1", model, row["h1_1"], row["complete"], text, about)
            rows.append(row)
        chosen = [r for r in rows if r["model"] in c.h1_1_models]
        need = int(v["h1_1_min_models"])
        overall = verdict_overall([r.get("h1_1", "") for r in chosen], need)
        overall_information = verdict_overall([r.get("h1_1_information", "") for r in chosen], need)
        complete = all(r["complete"] for r in chosen)
        self.verdict("H1.1", "overall", overall, complete, f"holds for at least {need} of {len(chosen)}",
                     f"{SHARE_NAMES[information]}: {overall_information}")  # fmt: skip
        columns = [
            Column("model", "model", "text"), Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("collision_rate", "collisions", digits=4),
            Column("rmse_vs_ref", f"RMSE vs {c.reference}", "pct", 1, ci=True), Column("rmse_vs_ref_p", "p", "p"),
            Column("audited", "audited", "int"), Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("unstable_eq_full", "unstable among equilibria (poles incl.)", ci=True),
            Column("unstable_eq_sign", "unstable among equilibria (sign)", ci=True), Column("stable", "stable"),
            Column("unstable", "unstable", ci=True), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable", "not stable", ci=True), Column("not_stable_full", "not stable (poles incl.)", ci=True),
            Column("unstable_sign", "unstable (sign)"),
            Column("max_gain_median", "max gain (median)", digits=3), Column("h1_1", "H1.1", "text"),
            Column("h1_1_information", f"H1.1 on {SHARE_NAMES[information]} (information)", "text"),
            Column("h1_1_poles", f"H1.1 on {SHARE_NAMES[POLES['h1_1']]} (information)", "text"),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"E1: models without penalty on {c.data}; learned models {len(c.folds)} folds x {len(c.seeds)} seeds, "
            f"baselines {len(c.folds)} folds x seed {c.seeds[0]}.",
            "RMSE s: spacing RMSE of the test parts (m); unit driver (event on highD): per driver the mean over its "
            "events and the seeds, mean over the drivers (column drivers). collisions: mean over the drivers of their "
            "collision rate.",
            f"RMSE vs {c.reference}: relative difference of the mean RMSE to the {c.reference}, paired over the common "
            "drivers (on highD: the events, which carry no driver identifier); p: Wilcoxon signed-rank test.",
            "unstable among equilibria: share of the string-unstable equilibria among the equilibria the audit "
            "analysed, inside the band or outside it (share_unstable_numerical; sign criterion: "
            "share_unstable_sign), speeds in support; unit run (column audited), mean over the runs.",
            "Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable), sign criterion "
            "band_sign; unit run, mean over the runs. max gain: median over the runs.",
            self.poles_note(),
            f"H1.1 (models {', '.join(c.h1_1_models)}): {SHARE_NAMES[basis]} >= {v['h1_1_unstable_min']:g} with "
            f"lower end > {v['h1_1_unstable_low_min']:g}, and RMSE below the {c.reference} (upper end < 0). For "
            f"information: the same rule on {SHARE_NAMES[information]}, and on {SHARE_NAMES[POLES['h1_1']]} "
            "(recurrent laws); the verdicts stay on the pre-specified share (D92: the numerical rule).",
        )
        state = "" if complete else " (incomplete)"
        footer = [
            f"H1.1 overall (holds for at least {need} of {len(chosen)}): {overall}{state}",
            f"For information, the same rule on {SHARE_NAMES[information]}: {overall_information}{state}",
        ]
        verdicts = ", ".join(f"{r['model']} {r.get('h1_1') or 'n/a'}" for r in chosen)
        summary = f"H1.1 {overall}{state} ({verdicts}; on {SHARE_NAMES[information]}: {overall_information})"
        return Table(t, "E1: models without penalty (H1.1)", notes, pd.DataFrame(rows), columns, footer, summary)

    # ------------------------------------------------------------------------------------ E2 sweep
    def table_e2_sweep(self, write_choice: bool = True) -> Table:
        c, t = self.cfg, "e2_sweep"
        rows = []
        for arch in c.learned:
            kind = c.penalties.get(arch)
            for weight in c.weights:
                experiment = experiment_name("e2", kind, weight)
                runs = self.runs(t, experiment, c.data, arch, c.seeds[:1], (METRICS, AUDIT))
                stable = self.values(t, runs, "band_numerical_stable", AUDIT)
                rows.append({
                    "architecture": arch, "kind": kind, "weight": weight, "experiment": experiment,
                    "runs": self.trained(runs), "runs_expected": len(c.folds),
                    "val_rmse_s": self.values(t, runs, "val_rmse_s_mean", METRICS).mean(),
                    "test_rmse_s": self.values(t, runs, "test_rmse_s_mean").mean(),
                    "audited": int(stable.notna().sum()), "stable": stable.mean(),
                    "feasible": int(_column(runs, "best_feasible").eq(True).sum()),
                })  # fmt: skip
        frame = pd.DataFrame(rows)
        frame["chosen"] = False
        self.sweep_choice = {}
        for arch in c.learned:
            group = frame[frame["architecture"] == arch]
            weight, rule = choose_weight(group, c.stable_min)
            if weight is None:
                self.note(t, f"{arch}: no weight can be chosen ({rule})")
                continue
            best = group[group["weight"] == weight].iloc[0]
            frame.loc[best.name, "chosen"] = True
            complete = bool((group["runs"] == len(c.folds)).all() and (group["audited"] == len(c.folds)).all())
            self.sweep_choice[arch] = {
                "kind": best["kind"], "weight": weight, "experiment": best["experiment"], "rule": rule,
                "stable": best["stable"], "val_rmse_s": best["val_rmse_s"], "runs": int(best["runs"]),
                "complete": complete,
            }  # fmt: skip
            if arch in c.chosen_weights:
                self.sweep_choice[arch]["override"] = float(c.chosen_weights[arch])
        self.chosen = {arch: entry["weight"] for arch, entry in self.sweep_choice.items()}
        self.chosen.update({arch: float(weight) for arch, weight in c.chosen_weights.items()})
        if write_choice:
            c.out_dir.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self.sweep_choice, indent=2, default=lambda x: None if _blank(x) else float(x))
            (c.out_dir / "chosen_weights.json").write_text(text + "\n", encoding="utf-8")
        columns = [
            Column("architecture", "architecture", "text"), Column("kind", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("runs", "runs", "int"), Column("val_rmse_s", "val RMSE s (m)"),
            Column("test_rmse_s", "test RMSE s (m)"), Column("audited", "audited", "int"), Column("stable", "stable"),
            Column("feasible", "feasible best epoch", "int"), Column("chosen", "chosen", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"E2 sweep: penalty weights {', '.join(f'{w:g}' for w in c.weights)} on {c.data}, {len(c.folds)} folds, "
            f"seed {c.seeds[0]}; unit run, means over the folds (column runs; audited for the share).",
            "val / test RMSE s: mean over the runs of the mean spacing RMSE of the validation / test events (m). "
            "stable: band_numerical of the grid speeds in support. feasible best epoch: runs whose best epoch "
            "meets the penalty (D89).",
            f"Chosen (D85): among the weights with stable >= {c.stable_min:g} the smallest validation RMSE; without "
            "such a weight the largest stable share. Written to chosen_weights.json.",
        )
        chosen = ", ".join(f"{arch} {weight:g}" for arch, weight in self.chosen.items()) or "none"
        overrides = [f"{arch} {weight:g} (sweep: {self.sweep_choice.get(arch, {}).get('weight')})"
                     for arch, weight in c.chosen_weights.items()]  # fmt: skip
        footer = [f"Chosen weights: {chosen}."]
        if overrides:
            footer.append(f"From the configuration: {'; '.join(overrides)}.")
        title = "E2 sweep: choice of the penalty weight (D85)"
        return Table(t, title, notes, frame, columns, footer, f"chosen: {chosen}")

    # ------------------------------------------------------------------------------------------ E2
    def table_e2(self) -> Table:
        c, t, v = self.cfg, "e2", self.v
        rows = []
        for arch in c.learned:
            kind, weight = c.penalties.get(arch), self.chosen.get(arch)
            row: dict[str, Any] = {"architecture": arch, "kind": kind, "weight": weight}
            if weight is None:
                self.note(t, f"{arch}: no chosen weight")
                rows.append(row)
                continue
            experiment = experiment_name("e2", kind, weight)
            files = (METRICS, AUDIT, EVENTS, PLATOON)
            base = self.runs(t, "e1", c.data, arch, c.seeds, files)
            pen = self.runs(t, experiment, c.data, arch, c.seeds, files)
            row.update(experiment=experiment, runs_e1=self.trained(base), runs_e2=self.trained(pen),
                       runs_expected=len(c.folds) * len(c.seeds))  # fmt: skip
            self.versus(row, "rmse_change", self.drivers(t, base), self.drivers(t, pen))
            stable_e1 = self.values(t, base, "band_numerical_stable", AUDIT)
            stable_e2 = self.values(t, pen, "band_numerical_stable", AUDIT)
            _put(row, "not_stable_e1", self.ci(1.0 - stable_e1))
            _put(row, "not_stable_e2", self.ci(1.0 - stable_e2))
            _put(row, "unstable_e1", self.ci(self.values(t, base, "band_numerical_unstable")))
            _put(row, "unstable_e2", self.ci(self.values(t, pen, "band_numerical_unstable")))
            _put(row, "unstable_eq_e1", self.unstable_eq(t, base))
            _put(row, "unstable_eq_e2", self.unstable_eq(t, pen))
            self.poles(t, row, base, "_e1")  # M9: not_stable_full_e1, unstable_eq_full_e1
            self.poles(t, row, pen, "_e2")
            hysteresis = "platoon_hysteresis_area_pulse"  # missing when follower 1 collided behind the pulse
            compared = {
                "growth_error": (self.growth(t, base, "mean"), self.growth(t, pen, "mean")),  # D109: the prefix
                "growth_error_full": (self.growth(t, base, "mean", prefix=False),
                                      self.growth(t, pen, "mean", prefix=False)),  # D107: whole curves
                "hysteresis": (self.values(t, base, hysteresis), self.values(t, pen, hysteresis)),
            }  # fmt: skip
            for key, (a, b) in compared.items():
                cmp = self.compare(a, b)  # the runs paired by fold and seed in which both have a value
                row[f"{key}_e1"], row[f"{key}_e2"], row[f"{key}_pairs"] = cmp["mean_a"], cmp["mean_b"], cmp["n"]
                change = (cmp["relative_change"], cmp["ci_low"], cmp["ci_high"])
                row[f"{key}_change"], row[f"{key}_change_low"], row[f"{key}_change_high"] = change
                row[f"{key}_p"] = cmp["p_value"]
            for arm, runs in (("e1", base), ("e2", pen)):
                values = compared["hysteresis"][0 if arm == "e1" else 1]
                row[f"hysteresis_collided_{arm}"] = self.platoon_runs(runs) - int(values.notna().sum())
            row["growth_collided_e1"], row["growth_profiles_e1"] = self.collided(t, base, "mean")
            row["growth_collided_e2"], row["growth_profiles_e2"] = self.collided(t, pen, "mean")
            expected = row["runs_expected"]
            row["complete"] = (row["runs_e1"] == row["runs_e2"] == expected
                               and stable_e1.notna().sum() == stable_e2.notna().sum() == expected)  # fmt: skip
            share = c.bases["h1_2"]
            row["h1_2"] = verdict_h1_2(row, v, share)
            row["h1_2_poles"] = verdict_h1_2(row, v, POLES["h1_2"])  # M9, for information
            change = _ci_text(row, "rmse_change", "pct", 1)
            basis = f"{SHARE_NAMES[share]} {_ci_text(row, f'{share}_e2')}, RMSE change {change}"
            self.verdict("H1.2", arch, row["h1_2"], row["complete"], basis)
            rows.append(row)
        frame = pd.DataFrame(rows)
        if "rmse_change_p" not in frame.columns:
            frame["rmse_change_p"] = np.nan
        frame["rmse_change_p_holm"] = holm(frame["rmse_change_p"].astype(float))
        columns = [
            Column("architecture", "architecture", "text"), Column("kind", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("runs_e1", "runs E1", "int"),
            Column("runs_e2", "runs E2", "int"), Column("rmse_change_pairs", "drivers", "int"),
            Column("rmse_change", "RMSE change", "pct", 1, ci=True), Column("rmse_change_p", "p", "p"),
            Column("rmse_change_p_holm", "p (Holm)", "p"), Column("not_stable_e1", "not stable E1", ci=True),
            Column("not_stable_full_e1", "not stable E1 (poles incl.)", ci=True),
            Column("not_stable_e2", "not stable E2", ci=True),
            Column("not_stable_full_e2", "not stable E2 (poles incl.)", ci=True),
            Column("unstable_eq_e1", "unstable among equilibria E1", ci=True),
            Column("unstable_eq_full_e1", "unstable among equilibria E1 (poles incl.)", ci=True),
            Column("unstable_eq_e2", "unstable among equilibria E2", ci=True),
            Column("unstable_eq_full_e2", "unstable among equilibria E2 (poles incl.)", ci=True),
            Column("growth_error_pairs", "growth pairs", "int"),
            Column("growth_collided_e1", "collided E1", "int"), Column("growth_collided_e2", "collided E2", "int"),
            Column("growth_error_e1", "growth error E1", digits=3),
            Column("growth_error_e2", "growth error E2", digits=3),
            Column("growth_error_change", "change", "pct", 1, ci=True),
            Column("growth_error_full_pairs", "whole-curve pairs (D107)", "int"),
            Column("growth_error_full_e1", "whole-curve E1", digits=3),
            Column("growth_error_full_e2", "whole-curve E2", digits=3),
            Column("growth_error_full_change", "change", "pct", 1, ci=True),
            Column("hysteresis_pairs", "hysteresis pairs", "int"),
            Column("hysteresis_collided_e1", "collided E1", "int"),
            Column("hysteresis_collided_e2", "collided E2", "int"),
            Column("hysteresis_e1", "hysteresis E1", digits=1), Column("hysteresis_e2", "hysteresis E2", digits=1),
            Column("hysteresis_change", "change", "pct", 1, ci=True),
            Column("h1_2", "H1.2", "text"),
            Column("h1_2_poles", f"H1.2 on {SHARE_NAMES[POLES['h1_2']]} (information)", "text"),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"E2: the chosen weight of every architecture (E2 runs) against E1, {len(c.folds)} folds x "
            f"{len(c.seeds)} seeds each.",
            "RMSE change: relative change of the mean spacing RMSE of the test parts, unit driver (event on highD): "
            "per driver the mean over its events and seeds, paired over the common drivers (on highD: the events, "
            "which carry no driver identifier), their number in column drivers; p: Wilcoxon signed-rank test; "
            "p (Holm): corrected over the architectures.",
            "not stable: 1 - stable (band_numerical, speeds in support); unstable among equilibria: share of the "
            "string-unstable equilibria among the equilibria the audit analysed (share_unstable_numerical); unit "
            "run, mean over the runs.",
            self.poles_note(),
            "growth error (D109, platoon_prefix_growth_error_mean): per OpenACC profile the growth error on the "
            "collision-free prefix of the platoon (the followers ahead of the first collided position that the "
            "empirical curve has, at least 3), per run the mean over the profiles that have one; whole-curve (D107, "
            "platoon_growth_error_mean): over the whole empirical curve, none for a platoon that collided. "
            "Hysteresis area of the braking pulse (m^2/s): none when follower 1 collided. All three: unit run, means "
            "over the pairs of runs (fold and seed) in which both have a value (pairs), relative change with "
            "interval. collided E1/E2: OpenACC profiles whose platoon collided, summed over the runs (of "
            "growth_profiles_e1/e2 in the CSV), and runs whose follower 1 collided behind the pulse.",
            f"H1.2 on the {SHARE_NAMES[c.bases['h1_2']]} with penalty (E2): confirmed when its upper end < "
            f"{v['h1_2_not_stable_high_max']:g} and the upper end of the RMSE change <= "
            f"{100 * v['h1_2_rmse_change_high_max']:+g} %; refuted when the lower end of the RMSE change > "
            f"{100 * v['h1_2_rmse_change_low_refute']:+g} % or its lower end > {v['h1_2_not_stable_low_refute']:g}; "
            f"otherwise open. For information: the same rule on the {SHARE_NAMES[POLES['h1_2']]} with penalty "
            "(recurrent laws); the verdicts stay on the pre-specified share (D92: the numerical rule).",
        )
        summary = ", ".join(f"{r['architecture']} {r.get('h1_2') or 'n/a'}" for r in rows)
        return Table(t, "E2: stability penalty with the chosen weight (H1.2)", notes, frame, columns, [],
                     f"H1.2 {summary}")  # fmt: skip

    # ---------------------------------------------------------------------------------- E2 low frequency
    def lowfreq_choice(self, arch: str) -> tuple[float | None, str]:
        """The weight of the combined penalty of ``arch`` (D110): the configured one, else the rule of D85 on the
        runs of the pilot (``lowfreq_pilot_folds``, first seed), read without notes."""
        c = self.cfg
        if arch in c.lowfreq_chosen:
            return float(c.lowfreq_chosen[arch]), "from the configuration"
        rows = []
        for weight in c.lowfreq_weights:
            experiment = c.lowfreq_experiment.format(weight=weight)
            stable, val = [], []
            for fold in c.lowfreq_pilot_folds:
                run = c.runs_root / experiment / c.data / arch / f"driver_fold{fold}_seed{c.seeds[0]}"
                if run.is_dir():
                    if run not in self._rows:
                        self._rows[run] = collect_run(run, experiment)
                    stable.append(_number(self._rows[run].get("band_numerical_stable")))
                    val.append(_number(self._rows[run].get("val_rmse_s_mean")))
            rows.append({"weight": weight, "stable": _nanmean(stable), "val_rmse_s": _nanmean(val)})
        return choose_weight(pd.DataFrame(rows, columns=["weight", "stable", "val_rmse_s"]), c.stable_min)

    def table_e2_lowfreq(self) -> Table:
        c, t, v = self.cfg, "e2_lowfreq", self.v
        share = c.bases["h1_2"]
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows, seeds = [], c.seeds[:1]
        for arch in c.lowfreq_architectures:
            chosen, rule = self.lowfreq_choice(arch)
            weights = list(c.lowfreq_weights)
            if chosen is not None and not any(math.isclose(w, chosen) for w in weights):
                weights.append(chosen)  # a configured weight outside the pilot
            for weight in weights:
                is_chosen = chosen is not None and math.isclose(weight, chosen)
                folds = c.folds if is_chosen else c.lowfreq_pilot_folds
                experiment = c.lowfreq_experiment.format(weight=weight)
                runs = self.runs(t, experiment, c.data, arch, seeds, files, folds=folds)
                base = self.runs(t, "e1", c.data, arch, seeds, (METRICS, EVENTS), folds=folds)
                row: dict[str, Any] = {
                    "architecture": arch, "weight": weight, "experiment": experiment, "chosen": is_chosen,
                    "rule": rule if is_chosen else "", "runs": self.trained(runs),
                    "runs_expected": len(folds) * len(seeds), "folds": len(folds),
                }  # fmt: skip
                drivers = self.drivers(t, runs)
                self.rmse(row, drivers)
                self.versus(row, "rmse_change", self.drivers(t, base), drivers)
                self.shares(t, row, runs)
                self.poles(t, row, runs)
                row["collided"], row["profiles"] = self.collided(t, runs, "mean")
                row["complete"] = row["runs"] == row["audited"] == row["runs_expected"]
                row["h1_2"] = None  # the verdict belongs to the chosen weight
                if is_chosen:
                    view = {**row, f"{share}_e2": row.get(share), f"{share}_e2_low": row.get(f"{share}_low"),
                            f"{share}_e2_high": row.get(f"{share}_high")}  # fmt: skip
                    row["h1_2"] = verdict_h1_2(view, v, share)
                rows.append(row)
            mine = [r for r in rows if r["architecture"] == arch and r["chosen"]]
            if mine:
                row = mine[0]
                basis = (f"weight {row['weight']:g} ({rule}): {SHARE_NAMES[share]} {_ci_text(row, share)}, RMSE "
                         f"change {_ci_text(row, 'rmse_change', 'pct', 1)} against E1 (seed {seeds[0]})")  # fmt: skip
                self.verdict("H1.2 (combined)", arch, row["h1_2"], row["complete"], basis)
            else:
                self.verdict("H1.2 (combined)", arch, "", False, f"no weight chosen ({rule})")
        frame = pd.DataFrame(rows)
        columns = [
            Column("architecture", "architecture", "text"), Column("weight", "Jacobian weight", "weight"),
            Column("chosen", "chosen", "flag"), Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_change", "RMSE change vs E1", "pct", 1, ci=True),
            Column("rmse_change_p", "p", "p"), Column("audited", "audited", "int"), Column("stable", "stable"),
            Column("unstable", "unstable", ci=True), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable", "not stable", ci=True), Column("not_stable_full", "not stable (poles incl.)", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("unstable_eq_full", "unstable among equilibria (poles incl.)", ci=True),
            Column("max_gain_median", "max gain (median)", digits=3), Column("collided", "collided", "int"),
            Column("profiles", "profiles", "int"), Column("h1_2", "H1.2 (combined)", "text"),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"Low-frequency arm of E2 (D110): {', '.join(c.lowfreq_architectures)} on {c.data} with the combined "
            "penalty (rollout gain penalty of E2 plus the Jacobian penalty of the memoryless view and the needle "
            f"guard), experiments {c.lowfreq_experiment.format(weight=c.lowfreq_weights[0])} etc.; the pilot runs "
            f"every weight on fold(s) {', '.join(map(str, c.lowfreq_pilot_folds))}, the weight chosen by the rule "
            f"of D85 on the pilot runs (or given in the configuration) on {len(c.folds)} folds; seed {seeds[0]}.",
            "RMSE s: spacing RMSE of the test parts (m), unit driver (event on highD); RMSE change vs E1: relative "
            f"change of the mean RMSE against the E1 runs of the same folds and seed {seeds[0]}, paired over the "
            "common drivers (on highD: the events, which carry no driver identifier); p: Wilcoxon signed-rank test.",
            "Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among "
            "equilibria: share_unstable_numerical; unit run. max gain: median over the runs. collided: OpenACC "
            "platoon profiles that collided, of profiles, summed over the runs.",
            self.poles_note(),
            f"H1.2 (combined), on the chosen weight: the rule of H1.2 (E2): confirmed when the upper end of the "
            f"{SHARE_NAMES[share]} < {v['h1_2_not_stable_high_max']:g} and the upper end of the RMSE change <= "
            f"{100 * v['h1_2_rmse_change_high_max']:+g} %; refuted when the lower end of the RMSE change > "
            f"{100 * v['h1_2_rmse_change_low_refute']:+g} % or the lower end of the share > "
            f"{v['h1_2_not_stable_low_refute']:g}; otherwise open. The verdict of H1.2 of table e2 stays.",
        )
        verdicts = [f"{r['architecture']} {r.get('h1_2') or 'n/a'}" for r in rows if r["chosen"]]
        summary = f"H1.2 (combined) {', '.join(verdicts) or 'n/a: no weight chosen'}"
        return Table(t, "E2, low-frequency arm: combined penalty of the recurrent models (D110)", notes, frame,
                     columns, [], summary)  # fmt: skip

    # ---------------------------------------------------------------------------------- E2 monotone (M8)
    def table_e2_monotone(self) -> Table:
        """D117: the monotonicity terms of RACER alone against E1 and the chosen E2 weight, on the same fold(s)."""
        c, t = self.cfg, "e2_monotone"
        seeds, folds = c.seeds[:1], c.monotone_folds
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows = []
        for arch in c.monotone_architectures:
            base = self.runs(t, "e1", c.data, arch, seeds, files, folds=folds)
            arms: list[tuple[str, str, pd.DataFrame]] = [("E1", "e1", base)]
            kind, weight = c.penalties.get(arch), self.chosen.get(arch)
            if weight is None:
                self.note(t, f"{arch}: no chosen weight of E2 (table e2_sweep), no E2 row")
            else:
                experiment = experiment_name("e2", kind, weight)
                arms.append((f"E2 ({kind}, weight {weight:g})", experiment,
                             self.runs(t, experiment, c.data, arch, seeds, files, folds=folds)))  # fmt: skip
            arms.append(("monotone (D117)", c.monotone_experiment,
                         self.runs(t, c.monotone_experiment, c.data, arch, seeds, files, folds=folds)))  # fmt: skip
            base_drivers = self.drivers(t, base)
            for arm, experiment, runs in arms:
                row: dict[str, Any] = {"architecture": arch, "arm": arm, "experiment": experiment,
                                       "runs": self.trained(runs), "runs_expected": len(folds) * len(seeds)}  # fmt: skip
                drivers = base_drivers if experiment == "e1" else self.drivers(t, runs)
                self.rmse(row, drivers)
                if experiment != "e1":
                    self.versus(row, "rmse_change", base_drivers, drivers)
                self.shares(t, row, runs)
                row["max_gain"] = self.values(t, runs, "max_gain").max() if len(runs) else np.nan
                row["collided"], row["profiles"] = self.collided(t, runs, "mean")
                row["complete"] = row["runs"] == row["audited"] == row["runs_expected"]
                rows.append(row)
        frame = pd.DataFrame(rows)
        for key in ("rmse_change", "rmse_change_low", "rmse_change_high", "rmse_change_p", "rmse_change_pairs"):
            if key not in frame.columns:
                frame[key] = np.nan
        columns = [
            Column("architecture", "architecture", "text"), Column("arm", "arm", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_change", "RMSE change vs E1", "pct", 1, ci=True),
            Column("rmse_change_p", "p", "p"), Column("audited", "audited", "int"), Column("stable", "stable"),
            Column("unstable", "unstable", ci=True), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable", "not stable", ci=True), Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("max_gain", "max gain", digits=3), Column("collided", "collided profiles", "int"),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"Control of M8 (D117): penalty kind monotone = relu(-f_s) + relu(f_dv) + relu(f_v) at the anchored "
            f"equilibria of E2, without string term (the monotonicity constraints of RACER, dv = v - v_lead), "
            f"experiment {c.monotone_experiment}; next to E1 and the chosen weight of E2 (D85) of the same runs: "
            f"{c.data}, fold(s) {', '.join(map(str, folds))}, seed {seeds[0]}.",
            "RMSE s: spacing RMSE of the test part (m), unit driver (event on highD), with its interval over the "
            "drivers; RMSE change vs E1: relative change of the mean RMSE against E1 of the same fold and seed, paired "
            "over the drivers (on highD: the events, which carry no driver identifier); p: Wilcoxon signed-rank test.",
            "Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among "
            "equilibria: share_unstable_numerical; max gain: largest measured gain of the audit; collided: OpenACC "
            "platoon profiles that collided, of profiles (CSV). Unit run: one run per row, so the intervals of the "
            "shares are the values themselves.",
            "Missing runs and files: runs/_tables/m4/missing.txt.",
        )
        return Table(t, "E2 control: monotonicity terms only (D117)", notes, frame, columns)

    # ------------------------------------------------------------------------------ E2 long window (M9)
    def lowfreq_unstable(self, table: str, runs: pd.DataFrame) -> dict[str, Any]:
        """Share of the audited equilibria (speeds in support with a measured response) whose gain exceeds the
        threshold of the audit at a frequency at most ``horizon_omega_max`` rad/s, mean over the runs with its
        interval (the band the 20-s window of the E2 penalty cannot resolve)."""
        shares = []
        for run_dir in runs["run_dir"] if len(runs) else []:
            path = Path(run_dir) / AUDIT
            if not path.exists():
                continue
            try:
                audit = json.loads(path.read_text(encoding="utf-8")).get("audit") or {}
            except (OSError, ValueError) as exc:
                self.note(table, f"{self.label(path)}: unreadable ({type(exc).__name__})")
                continue
            if not isinstance(audit.get("omega"), list) or not isinstance(audit.get("equilibria"), list):
                continue  # an audit without frequency responses (noted by the shares when a value is missing)
            omega, equilibria = np.asarray(audit["omega"], dtype=float), audit["equilibria"]
            low = omega <= self.cfg.horizon_omega_max
            frequency = (audit.get("config") or {}).get("frequency") or {}
            threshold = float(frequency.get("threshold", 1.02))
            counted, above = 0, 0
            for eq in equilibria:
                if not isinstance(eq, dict) or not eq.get("in_support") or not isinstance(eq.get("gain"), list):
                    continue
                gain = np.asarray(eq["gain"], dtype=float)
                if len(gain) != len(omega) or not np.isfinite(gain[low]).any():
                    continue
                counted += 1
                above += bool(np.nanmax(gain[low]) > threshold)
            if counted:
                shares.append(above / counted)
        return self.ci(shares) if shares else dict(EMPTY)

    def table_e2_horizon(self) -> Table:
        """Revision of 2026-10-07: the rollout gain penalty with a rollout long enough to measure its lowest
        frequency (0.05 rad/s) over two whole periods, against E1 and the chosen E2 weight on the same fold(s)."""
        c, t = self.cfg, "e2_horizon"
        seeds, folds = c.seeds[:1], c.horizon_folds
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows = []
        for arch in c.horizon_architectures:
            base = self.runs(t, "e1", c.data, arch, seeds, files, folds=folds)
            arms: list[tuple[str, str, pd.DataFrame]] = [("E1", "e1", base)]
            kind, weight = c.penalties.get(arch), self.chosen.get(arch)
            if weight is None:
                self.note(t, f"{arch}: no chosen weight of E2 (table e2_sweep), no E2 row")
            else:
                experiment = experiment_name("e2", kind, weight)
                arms.append((f"E2 ({kind}, weight {weight:g}, rollout 40 s, last 20 s)", experiment,
                             self.runs(t, experiment, c.data, arch, seeds, files, folds=folds)))  # fmt: skip
            arms.append(("long window (rollout 380 s, last 252 s)", c.horizon_experiment,
                         self.runs(t, c.horizon_experiment, c.data, arch, seeds, files, folds=folds)))  # fmt: skip
            base_drivers = self.drivers(t, base)
            for arm, experiment, runs in arms:
                row: dict[str, Any] = {"architecture": arch, "arm": arm, "experiment": experiment,
                                       "runs": self.trained(runs), "runs_expected": len(folds) * len(seeds)}  # fmt: skip
                drivers = base_drivers if experiment == "e1" else self.drivers(t, runs)
                self.rmse(row, drivers)
                if experiment != "e1":
                    self.versus(row, "rmse_change", base_drivers, drivers)
                self.shares(t, row, runs)
                self.poles(t, row, runs)
                _put(row, "lowfreq_unstable", self.lowfreq_unstable(t, runs))
                row["max_gain"] = self.values(t, runs, "max_gain").max() if len(runs) else np.nan
                row["max_gain_omega"] = self.values(t, runs, "max_gain_omega").median() if len(runs) else np.nan
                row["epochs"] = self.values(t, runs, "epochs").mean() if len(runs) else np.nan
                row["best_epoch"] = self.values(t, runs, "best_epoch").mean() if len(runs) else np.nan
                row["wall_time_h"] = self.values(t, runs, "wall_time_s").mean() / 3600.0 if len(runs) else np.nan
                row["collided"], row["profiles"] = self.collided(t, runs, "mean")
                row["complete"] = row["runs"] == row["audited"] == row["runs_expected"]
                rows.append(row)
        frame = pd.DataFrame(rows)
        for key in ("rmse_change", "rmse_change_low", "rmse_change_high", "rmse_change_p", "rmse_change_pairs"):
            if key not in frame.columns:
                frame[key] = np.nan
        columns = [
            Column("architecture", "architecture", "text"), Column("arm", "arm", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_change", "RMSE change vs E1", "pct", 1, ci=True),
            Column("rmse_change_p", "p", "p"), Column("audited", "audited", "int"), Column("stable", "stable"),
            Column("unstable", "unstable", ci=True), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable", "not stable", ci=True), Column("not_stable_full", "not stable (poles incl.)", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("unstable_eq_full", "unstable among equilibria (poles incl.)", ci=True),
            Column("lowfreq_unstable", f"gain above threshold at omega <= {c.horizon_omega_max:g}", ci=True),
            Column("max_gain", "max gain", digits=3), Column("max_gain_omega", "omega of max gain (rad/s)", digits=3),
            Column("epochs", "epochs", "int"), Column("best_epoch", "best epoch", "int"),
            Column("wall_time_h", "training (h)", digits=1), Column("collided", "collided profiles", "int"),
            Column("profiles", "profiles", "int"), Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"Long-window arm of E2 (revision of 7 October 2026): the rollout gain penalty of E2 at the chosen weight "
            f"(D85) with a rollout of 380 s whose gain is measured over the last 252 s, two whole periods of the lowest "
            f"penalty frequency 0.05 rad/s after a warm-up of one period, instead of 40 s and the last 20 s "
            f"(experiment {c.horizon_experiment}); next to E1 and the chosen weight of E2 of the same runs: "
            f"{c.data}, fold(s) {', '.join(map(str, folds))}, seed {seeds[0]}.",
            "RMSE s: spacing RMSE of the test part (m), unit driver (event on highD), with its interval over the "
            "drivers; RMSE change vs E1: relative change of the mean RMSE against E1 of the same fold and seed, paired "
            "over the drivers (on highD: the events, which carry no driver identifier); p: Wilcoxon signed-rank test.",
            "Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among "
            "equilibria: share_unstable_numerical; gain above threshold at low frequency: share of the audited "
            f"equilibria (speeds in support) whose measured gain exceeds the threshold of the audit at a frequency "
            f"at most {c.horizon_omega_max:g} rad/s, the band a 20-s window cannot resolve; max gain: largest gain "
            "of the audit, with the frequency at which it occurs; epochs and best epoch: of the training (patience "
            "10); training: wall time of the training in hours; collided: OpenACC platoon profiles that collided, of "
            "profiles. Unit run: one run per row, so the intervals of the shares are the values themselves.",
            self.poles_note(),
            "Missing runs and files: runs/_tables/m4/missing.txt.",
        )
        return Table(t, "E2, long-window arm: rollout gain penalty measured over two periods of 0.05 rad/s", notes,
                     frame, columns)  # fmt: skip

    # ------------------------------------------------------------------------------------------ E3
    def table_e3(self) -> Table:
        c, t = self.cfg, "e3"
        rows, sources = [], {}
        files = (TRANSFER, EVENTS, *(f"transfer_{target}.parquet" for target in c.targets))
        for arch in [*c.learned, *[m for m in c.baselines if m not in c.learned]]:
            variants: list[tuple[str, str, float | None]] = [("none", "e1", None)]
            if arch in c.learned:
                weight = self.chosen.get(arch)
                if weight is None:
                    self.note(t, f"{arch}: no chosen weight")
                else:
                    variants.append(("chosen", experiment_name("e2", c.penalties.get(arch), weight), weight))
            for penalty, experiment, weight in variants:
                runs = self.runs(t, experiment, c.data, arch, c.seeds[:1], files)
                source = self.drivers(t, runs)
                src = float(source["rmse_s"].mean()) if source is not None else np.nan
                n_transfer = sum((Path(r) / TRANSFER).exists() for r in runs["run_dir"]) if len(runs) else 0
                for target in c.targets:
                    drivers = self.drivers(t, runs, f"transfer_{target}.parquet")
                    has = drivers is not None and not drivers.empty
                    row: dict[str, Any] = {
                        "architecture": arch, "penalty": penalty, "weight": weight, "experiment": experiment,
                        "target": target, "runs": n_transfer, "runs_expected": len(c.folds),
                        "drivers": len(drivers) if has else 0, "source_rmse_s": src,
                    }  # fmt: skip
                    _put(row, "rmse_s_h", self.ci(drivers["rmse_s_h"]) if has else EMPTY)
                    _put(row, "rmse_s_full", self.ci(drivers["rmse_s"]) if has else EMPTY)
                    ratio = (lambda x, s=src: float(x.mean() / s - 1.0))  # noqa: E731
                    degradation = self.ci(drivers["rmse_s_h"], ratio) if has and math.isfinite(src) else EMPTY
                    _put(row, "degradation", degradation)
                    runs_mean = self.values(t, runs, f"transfer_{target}_relative_degradation", TRANSFER).mean()
                    row["degradation_runs"] = runs_mean
                    sources[(arch, penalty, target)] = (drivers if has else None, src, len(rows))
                    rows.append(row)
            for target in c.targets:  # difference of the relative degradations, with minus without penalty
                none, chosen = sources.get((arch, "none", target)), sources.get((arch, "chosen", target))
                if none is None or chosen is None or none[0] is None or chosen[0] is None:
                    continue
                if not (math.isfinite(none[1]) and math.isfinite(chosen[1])):
                    continue
                pairs = pd.concat({"none": none[0]["rmse_s_h"], "chosen": chosen[0]["rmse_s_h"]}, axis=1, join="inner")

                def difference(x: np.ndarray, a: float = none[1], b: float = chosen[1]) -> float:
                    return float((x[:, 1].mean() / b - 1.0) - (x[:, 0].mean() / a - 1.0))

                _put(rows[chosen[2]], "degradation_change", self.ci(pairs.dropna().to_numpy(), difference))
        columns = [
            Column("architecture", "architecture", "text"), Column("penalty", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("target", "target", "text"), Column("runs", "runs", "int"),
            Column("drivers", "drivers", "int"), Column("source_rmse_s", "source RMSE s (m)"),
            Column("rmse_s_h", "RMSE s, 15 s (m)", ci=True), Column("rmse_s_full", "RMSE s, whole (m)", ci=True),
            Column("degradation", "relative degradation", "pct", 1, ci=True),
            Column("degradation_runs", "mean over runs", "pct", 1),
            Column("degradation_change", "with - without penalty", "pct", 1, ci=True),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"E3: transfer of the models of seed {c.seeds[0]} ({len(c.folds)} folds) of {c.data} to "
            f"{', '.join(c.targets)}, without penalty (E1) and with the chosen weight (E2).",
            "RMSE s: spacing RMSE of the whole target events and of their first 15 s; unit driver of the target "
            "(transfer_<target>.parquet, per driver the mean over its events and the folds), mean over the "
            "drivers.",
            "relative degradation: mean RMSE of the first 15 s over the mean RMSE of the source test parts (unit "
            "driver, source RMSE s) minus 1, interval over the drivers of the target with the source fixed; mean "
            "over runs: relative_degradation of transfer.json (event means), mean over the folds.",
            "with - without penalty: difference of the relative degradations, the drivers of the target paired.",
        )
        return Table(t, "E3: transfer to other data (D84)", notes, pd.DataFrame(rows), columns, [], "")

    # ------------------------------------------------------------------------------------------ E4
    def table_e4(self) -> Table:
        c, t, v = self.cfg, "e4", self.v
        ft = c.fine_tune_data
        spec = [
            ("residual_idm", "E1, free core", "e1", c.data),
            ("residual_idm", "fine-tuned, free core", "e4_free_ft", ft),
            ("mlp", "E1", "e1", c.data), ("mlp", "fine-tuned", "e4_free_ft", ft),
            ("residual_idm", "certified", "e4_stable", c.data),
            ("residual_idm", "certified, fine-tuned", "e4_stable_ft", ft),
        ]  # fmt: skip
        rows = []
        for model, label, experiment, data in spec:
            certificate = experiment != "e1"
            files = (METRICS, AUDIT, EVENTS) + ((CERTIFICATE,) if certificate else ())
            runs = self.runs(t, experiment, data, model, c.seeds, files)
            row: dict[str, Any] = {
                "model": model, "variant": label, "experiment": experiment, "data": data, "runs": self.trained(runs),
                "runs_expected": len(c.folds) * len(c.seeds),
            }  # fmt: skip
            if certificate and len(runs):
                applicable = _column(runs, "certificate_applicable").eq(True)
                row["certificates"] = int(applicable.sum())
                if row["certificates"]:
                    held = runs[applicable]
                    row["a_priori_holds"] = int(_column(held, "certificate_a_priori_holds").eq(True).sum())
                    row["at_equilibria_holds"] = int(_column(held, "certificate_at_equilibria_holds").eq(True).sum())
                    n_hold = pd.to_numeric(_column(held, "certificate_at_equilibria_n_hold"), errors="coerce")
                    n_eq = pd.to_numeric(_column(held, "certificate_at_equilibria_n_equilibria"), errors="coerce")
                    row["all_equilibria_certified"] = int(((n_hold == n_eq) & n_eq.notna()).sum())
            unstable = self.values(t, runs, "band_numerical_unstable", AUDIT)
            _put(row, "unstable", self.ci(unstable))
            _put(row, "not_stable", self.ci(1.0 - self.values(t, runs, "band_numerical_stable")))
            _put(row, "unstable_eq", self.unstable_eq(t, runs))
            row["complete"] = row["runs"] == unstable.notna().sum() == row["runs_expected"]
            self.rmse(row, self.drivers(t, runs))
            rows.append(row)
        certified = next(r for r in rows if r["experiment"] == "e4_stable_ft")
        mlp = next(r for r in rows if r["experiment"] == "e4_free_ft" and r["model"] == "mlp")
        share = c.bases["h1_5"]
        verdict = verdict_h1_5(certified, mlp, v, share)
        basis = (f"certified, fine-tuned: {SHARE_NAMES[share]} {_ci_text(certified, share)}; fine-tuned MLP: "
                 f"{_ci_text(mlp, share)}")  # fmt: skip
        complete = certified["complete"] and mlp["complete"]
        self.verdict("H1.5", "overall", verdict, complete, basis)
        columns = [
            Column("model", "model", "text"), Column("variant", "variant", "text"), Column("data", "data", "text"),
            Column("runs", "runs", "int"), Column("certificates", "certificates", "int"),
            Column("a_priori_holds", "a priori holds", "int"),
            Column("at_equilibria_holds", "at equilibria holds", "int"),
            Column("all_equilibria_certified", "all equilibria certified", "int"),
            Column("unstable", "unstable", ci=True), Column("not_stable", "not stable", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("drivers", "drivers", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            t,
            f"E4: certified hybrid (e4_stable) and its fine-tuning on {ft} (e4_stable_ft), the ResidualIDM with "
            f"free core and the MLP of E1 fine-tuned on {ft} (e4_free_ft); {len(c.folds)} folds x {len(c.seeds)} "
            "seeds each.",
            "Certificates (certificate.json, the applicable ones): runs in which the a priori certificate holds, "
            "in which the certificate at the equilibria holds (its holds), and in which every equilibrium found is "
            "certified (n_hold = n_equilibria).",
            "Shares: band_numerical of the grid speeds in support; unstable among equilibria: share of the "
            "string-unstable equilibria among the equilibria the audit analysed (share_unstable_numerical); unit "
            "run, mean over the runs. RMSE s: spacing RMSE of the test part of the data of the run (m), unit "
            "driver, mean over the drivers.",
            f"H1.5 on the {SHARE_NAMES[c.bases['h1_5']]}: after fine-tuning its upper end for the certified hybrid "
            f"< {v['h1_5_certified_unstable_high_max']:g} and its lower end for the fine-tuned MLP > "
            f"{v['h1_5_mlp_unstable_low_min']:g}.",
        )
        state = "" if complete else " (incomplete)"
        footer = [f"H1.5: {verdict or 'n/a'}{state} ({basis})."]
        title = "E4: certified hybrid and fine-tuning (H1.5)"
        return Table(t, title, notes, pd.DataFrame(rows), columns, footer, f"H1.5 {verdict or 'n/a'}{state}")

    # ------------------------------------------------------------------------------------------ E5
    def e5_experiment(self, view: str, model: str) -> str | None:
        """The penalised experiment of ``model``: that of the chosen weight, else the only one that exists."""
        kind, weight = self.cfg.penalties.get(model), self.chosen.get(model)
        if kind is None:
            return None
        if weight is not None:
            return experiment_name("e5", kind, weight)
        found = sorted(p.name for p in self.cfg.runs_root.glob(f"e5_{kind}_w*") if (p / view / model).is_dir())
        return found[0] if len(found) == 1 else None

    def e5_row(self, t: str, view: str, model: str, penalty: str, experiment: str, runs: pd.DataFrame,
               mode: str) -> dict[str, Any]:  # fmt: skip
        row: dict[str, Any] = {"view": view, "model": model, "penalty": penalty, "experiment": experiment,
                               "runs": self.trained(runs), "runs_expected": len(self.cfg.folds)}  # fmt: skip
        self.rmse(row, self.drivers(t, runs))
        self.shares(t, row, runs)
        self.poles(t, row, runs)
        growth = self.growth(t, runs, mode)  # D109: the collision-free prefix of every profile of the mode
        row["growth_error"] = growth.mean()
        row["growth_runs"] = int(growth.notna().sum())
        full = self.growth(t, runs, mode, prefix=False)  # D107: missing when every profile of the mode collided
        row["growth_error_full"] = full.mean()
        row["growth_runs_full"] = int(full.notna().sum())
        self.positions(t, row, runs, mode)
        row["collided"], row["profiles"] = self.collided(t, runs, mode)
        row["complete"] = row["runs"] == self.platoon_runs(runs) == row["runs_expected"]
        return row

    def reduction(self, row: dict[str, Any], a: pd.Series, b: pd.Series, key: str = "reduction") -> None:
        """Relative reduction ``1 - mean(b) / mean(a)`` of the growth error, paired over the runs in which both
        have a growth error (``<key>_pairs``), with interval and p."""
        cmp = self.compare(a, b)
        row[key], row[f"{key}_low"], row[f"{key}_high"] = -cmp["relative_change"], -cmp["ci_high"], -cmp["ci_low"]
        row[f"{key}_p"], row[f"{key}_pairs"] = cmp["p_value"], cmp["n"]

    def h1_3_basis(self, row: Mapping[str, Any]) -> str:
        """The reduction on the collision-free prefix and its pairs, the whole-curve reduction of D107, the
        collided profiles, the mean first collided position and the collision-free share without and with
        penalty; why it is open."""
        pairs = 0 if _blank(row.get("reduction_pairs")) else int(row["reduction_pairs"])
        full = 0 if _blank(row.get("reduction_full_pairs")) else int(row["reduction_full_pairs"])
        text = (f"reduction {_ci_text(row, 'reduction', 'pct', 1)} over {pairs} pairs (collision-free prefix); "
                f"whole curves (D107) {_ci_text(row, 'reduction_full', 'pct', 1)} over {full} pairs; collided "
                f"profiles {row.get('collided_without', 0)}/{row.get('profiles_without', 0)} without, "
                f"{row.get('collided', 0)}/{row.get('profiles', 0)} with penalty")  # fmt: skip
        if not (_blank(row.get("first_collided_without")) and _blank(row.get("first_collided"))):
            text += (f"; first collided position {_cell(row.get('first_collided_without'), 'num', 1) or 'n/a'} "
                     f"without, {_cell(row.get('first_collided'), 'num', 1) or 'n/a'} with penalty")  # fmt: skip
        if pairs < self.v["h1_3_min_pairs"]:
            text += f"; open: fewer than {self.v['h1_3_min_pairs']:g} pairs"
        return text

    def table_e5(self) -> Table:
        c, t, v = self.cfg, "e5", self.v
        rows = []
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        for view, mode in c.views.items():
            prefix, whole = f"platoon_prefix_growth_error_{mode}", f"platoon_growth_error_{mode}"  # D109, D107
            base, base_rows = {}, {}
            for model in [*c.e5_models, *[m for m in c.e5_penalised if m not in c.e5_models]]:
                base[model] = self.runs(t, "e5", view, model, c.seeds[:1], files)
                base_rows[model] = self.e5_row(t, view, model, "none", "e5", base[model], mode)
                if model in c.e5_models:
                    rows.append(base_rows[model])
            pooled: dict[str, list[pd.Series]] = {"a": [], "b": [], "a_full": [], "b_full": []}
            complete: list[bool] = []
            counts = dict.fromkeys(("collided_without", "profiles_without", "collided", "profiles"), 0)
            for model in c.e5_penalised:
                experiment = self.e5_experiment(view, model)
                if experiment is None:
                    self.note(t, f"{view}/{model}: no penalised experiment (no chosen weight, none or several e5_*)")
                    complete.append(False)
                    continue
                runs = self.runs(t, experiment, view, model, c.seeds[:1], files)
                row = self.e5_row(t, view, model, "chosen", experiment, runs, mode)
                without = base_rows[model]
                row.update(collided_without=without["collided"], profiles_without=without["profiles"],
                           first_collided_without=without["first_collided"],
                           collision_free_without=without["collision_free"])  # fmt: skip
                series = {"a": self.values(t, base[model], prefix), "b": self.values(t, runs, prefix),
                          "a_full": self.values(t, base[model], whole), "b_full": self.values(t, runs, whole)}  # fmt: skip
                self.reduction(row, series["a"], series["b"])
                self.reduction(row, series["a_full"], series["b_full"], "reduction_full")
                row["complete"] = row["complete"] and without["complete"]
                row["h1_3"] = verdict_h1_3(row, v)
                self.verdict("H1.3", f"{view}/{model}", row["h1_3"], row["complete"], self.h1_3_basis(row))
                rows.append(row)
                for key, values in series.items():
                    pooled[key].append(pd.concat({model: values}, names=["model"]))
                complete.append(row["complete"])
                for key in counts:
                    counts[key] += row[key]
            row = {"view": view, "model": "pooled", "penalty": "chosen", "experiment": "",
                   "complete": bool(complete) and all(complete), **counts}  # fmt: skip
            if pooled["a"]:
                self.reduction(row, pd.concat(pooled["a"]), pd.concat(pooled["b"]))
                self.reduction(row, pd.concat(pooled["a_full"]), pd.concat(pooled["b_full"]), "reduction_full")
            row["h1_3"] = verdict_h1_3(row, v)
            self.verdict("H1.3", f"{view}/pooled", row["h1_3"], row["complete"], self.h1_3_basis(row))
            rows.append(row)
        columns = [
            Column("view", "view", "text"), Column("model", "model", "text"), Column("penalty", "penalty", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("audited", "audited", "int"), Column("stable", "stable"), Column("unstable", "unstable", ci=True),
            Column("outside", "outside"), Column("none", "none"), Column("not_stable", "not stable", ci=True),
            Column("not_stable_full", "not stable (poles incl.)", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("unstable_eq_full", "unstable among equilibria (poles incl.)", ci=True),
            Column("max_gain_median", "max gain (median)", digits=3), Column("collided", "collided", "int"),
            Column("profiles", "profiles", "int"),
            Column("first_collided", "first collided position", digits=1, ci=True),
            Column("collision_free", "collision-free share", digits=2, ci=True),
            Column("growth_runs", "runs with growth error", "int"),
            Column("growth_error", "growth error", digits=3), Column("reduction_pairs", "pairs", "int"),
            Column("reduction", "reduction", "pct", 1, ci=True), Column("reduction_p", "p", "p"),
            Column("h1_3", "H1.3", "text"),
            Column("growth_runs_full", "whole-curve runs (D107)", "int"),
            Column("growth_error_full", "whole-curve growth error", digits=3),
            Column("reduction_full_pairs", "whole-curve pairs", "int"),
            Column("reduction_full", "whole-curve reduction", "pct", 1, ci=True),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        modes = ", ".join(f"{mode} for {view}" for view, mode in c.views.items())
        notes = self.header(
            t,
            f"E5: models of the OpenACC views {', '.join(c.views)}, {len(c.folds)} folds, seed {c.seeds[0]}; "
            "without penalty (e5) and with the chosen weight (e5_<penalty>_w<weight>).",
            "RMSE s: spacing RMSE of the test parts (m), unit driver, mean over the drivers. Shares: "
            "band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among equilibria: "
            "share_unstable_numerical; unit run. max gain: median over the runs.",
            self.poles_note(),
            f"growth error (D109): on the platoon profiles of the driving mode of the view ({modes}); per profile "
            "on the collision-free prefix of the simulated platoon: the followers ahead of its first collided "
            "position that the empirical curve has (at least 3, otherwise none), RMSE over them divided by the "
            "largest empirical value among them; per run the mean over the profiles that have one; unit run, mean "
            "over the runs with a value. collided: profiles of the mode whose platoon collided, of profiles, summed "
            "over the runs; first collided position: lowest follower whose gap reached 0, 51 without collision, "
            "per run the mean over the profiles of the mode; collision-free share: of these profiles; both: mean "
            "over the runs with interval.",
            "reduction: 1 - growth error with penalty / without, paired by fold over the folds in which both runs "
            "have a growth error (pairs; pooled: over the architectures of a view); p: Wilcoxon signed-rank test. "
            "whole-curve (D107): the same on the growth error over the whole empirical curve, none for a platoon "
            "that collided (secondary).",
            f"H1.3 (on the collision-free prefix): open with fewer than {v['h1_3_min_pairs']:g} pairs; otherwise "
            f"confirmed when the reduction >= {100 * v['h1_3_confirm_min']:g} % and its interval excludes 0, "
            f"refuted when it is below {100 * v['h1_3_refute_below']:g} %, else open.",
        )
        pooled = ", ".join(f"{r['view']} {r.get('h1_3') or 'n/a'}" for r in rows if r["model"] == "pooled")
        title = "E5: OpenACC control and growth error (H1.3)"
        return Table(t, title, notes, pd.DataFrame(rows), columns, [], f"H1.3 {pooled}")

    # ------------------------------------------------------------------------------------ verdicts
    def table_verdicts(self) -> Table:
        columns = [Column("hypothesis", "hypothesis", "text"), Column("unit", "unit", "text"),
                   Column("verdict", "verdict", "text"), Column("complete", "complete", "flag"),
                   Column("basis", "basis", "text"), Column("information", "information", "text")]  # fmt: skip
        b = self.cfg.bases
        notes = [
            "Verdicts of the tables e1 (H1.1), e2 (H1.2), e2_lowfreq (H1.2 (combined), D110), e4 (H1.5) and e5 "
            "(H1.3); an empty verdict lacks data; complete: drawn from every run of the design.",
            f"Bases: H1.1 {SHARE_NAMES[b['h1_1']]} and the RMSE against the reference; H1.2 and H1.2 (combined) "
            f"{SHARE_NAMES[b['h1_2']]} with penalty and the RMSE change; H1.3 the reduction of the growth error on "
            f"the collision-free prefix of the platoons (D109; open with fewer than {self.v['h1_3_min_pairs']:g} "
            f"pairs of runs that both have one); H1.5 {SHARE_NAMES[b['h1_5']]}. information: the verdict of H1.1 on "
            f"{SHARE_NAMES[b['h1_1_information']]}.",
        ]
        names = ["hypothesis", "unit", "verdict", "complete", "basis", "information"]
        frame = pd.DataFrame(self.verdicts, columns=names)
        return Table("verdicts", "Verdicts of the hypotheses", notes, frame, columns)


def make_tables(cfg: TablesConfig) -> list[str]:
    """Write the tables of ``cfg.tables`` (and the verdicts, the chosen weights, missing.txt); one line per table."""
    maker = TableMaker(cfg)
    builders = {
        "e1": maker.table_e1, "e2_sweep": maker.table_e2_sweep, "e2": maker.table_e2,
        "e2_lowfreq": maker.table_e2_lowfreq, "e3": maker.table_e3, "e4": maker.table_e4, "e5": maker.table_e5,
        "e2_monotone": maker.table_e2_monotone, "e2_horizon": maker.table_e2_horizon,
    }  # fmt: skip
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [name for name in TABLES if name in cfg.tables]
    if "e2_sweep" not in wanted and any(name in wanted for name in ("e2", "e3", "e5", "e2_monotone", "e2_horizon")):
        maker.table_e2_sweep(write_choice=False)  # the chosen weights without the table
        maker.missing = [line for line in maker.missing if not line.startswith("[e2_sweep]")]
        maker.counts.pop("e2_sweep", None)
    lines = []
    for name in wanted:
        table = builders[name]()
        write_table(table, cfg.folder(name))
        expected, existing = maker.counts.get(name, [0, 0])
        n_missing = sum(line.startswith(f"[{name}]") for line in maker.missing)
        summary = f"; {table.summary}" if table.summary else ""
        lines.append(f"TABLE {name}: {len(table.frame)} rows from {existing}/{expected} runs, {n_missing} missing"
                     f"{summary} -> {cfg.folder(name) / name}.md")  # fmt: skip
    if maker.verdicts:
        write_table(maker.table_verdicts(), cfg.out_dir)
    (cfg.out_dir / "missing.txt").write_text("".join(f"{line}\n" for line in maker.missing), encoding="utf-8")
    return lines
