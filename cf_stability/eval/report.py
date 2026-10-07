"""Report of the project (docs/m6_contract.md; section 7 of the specification; the tables of M7).

:func:`make_report` reads the tables of M4 and M5 (``runs/_tables/m4|m5/*.csv``, written by
``scripts/make_tables.py`` and ``scripts/corridor_metrics.py tables=true``, which ``refresh`` reruns
first) and the run files the figures need, and writes to ``out_dir`` (``runs/_report``):

* ``tables/<name>.csv`` (full precision) and ``tables/<name>.tex`` (booktabs, rounded, units in the
  column heads, a caption naming the design) of every table of M4 and M5, the verdicts of M4 and of H12
  (``verdicts_h12``), the existence arm of E2, the tables of the macro error by components and the tables
  of the arms of M7 (``e2_lowfreq``, ``e4_rmax``, ``sensitivity``, ``h12_1``, ``h12_2``); a table whose
  rows have no runs yet gets a note;
* ``figures/<name>.png`` (200 dpi) and ``figures/<name>.pdf``: the seven figures of section 3;
* ``report.md``: the sections 1-10 of the contract with the tables inline and the figures linked; every
  number in it comes from a table or from the run files;
* ``manifest.json``: run counts per experiment, config hashes and file hashes of the inputs, software
  versions, date.

Section 11 "Supplement (M8)" (docs/m8_contract.md, section 11): every table of ``runs/_tables/m8/`` (its CSV
copied, its LaTeX written to ``tables/``; the tables of M8 that this code knows get their columns, any other
one gets every column of its CSV) and every figure of ``runs/_report/supplement/figures/`` with the caption of
its ``<name>.txt``. An expected table or figure of M8 without its file gets a note.

Nothing missing raises: a missing table is written with its header and no rows, a missing figure is
left out, and both get a note in ``report.md`` (section "Notes on missing inputs").
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from cf_stability.eval import figures
from cf_stability.eval.collect import RUN_NAME
from cf_stability.eval.corridor_tables import COMPONENT_HEADERS, RAW_METRICS
from cf_stability.eval.tables import (
    AUDIT, EVENTS, METRICS, PLATOON, Column, Table, TableMaker, TablesConfig, _cell, experiment_name, markdown,
)  # fmt: skip
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

M4_TABLES = ("e1", "e2_sweep", "e2", "e2_lowfreq", "e3", "e4", "e5", "verdicts")
M5_TABLES = (
    "laws", "laws_raw", "instability", "instability_correlation", "tost", "h12_1", "h12_2", "verdicts_h12", "e4_rmax",
    "sensitivity",
)  # fmt: skip
NEW_TABLES = ("e2_existence", "m5_components", "macro_error_dynamic", "macro_error_dynamic_correlation")
FIGURES = (
    "rmse_vs_instability", "e2_tradeoff", "gain_curves", "growth_curves", "corridor_speed_contours",
    "fundamental_diagrams", "macro_error_vs_instability",
)  # fmt: skip
SECTIONS = (
    "1. Data and runs", "2. Verdicts of the hypotheses", "3. E1: models without penalty (H1.1)",
    "4. E2: stability penalties (H1.2)", "5. E3: transfer to other data", "6. E4: certified hybrid (H1.5)",
    "7. E5: OpenACC and the growth of oscillations (H1.3)", "8. Corridor (M5)", "9. Deviations from the specification",
    "10. How to reproduce", "11. Supplement (M8)",
)  # fmt: skip
M8_TABLES = (  # docs/m8_contract.md: the tables of runs/_tables/m8 (work packages T, E and C2), in the order of the contract
    "lowfreq_expansion", "certificate_tightness", "band_width", "e2_monotone", "band_sensitivity", "temporal",
    "asymmetry", "asymmetry_contrasts", "correlation_pooled", "correlation_pooled_laws", "power",
)  # fmt: skip
SUPPLEMENT_FIGURES = (  # the figures of runs/_report/supplement/figures (work package T); a pattern needs one match
    "lowfreq_expansion", "certificate_tightness", "band_width", "stability_map_*", "contours_*", "fd_*",
)  # fmt: skip
SOFTWARE = ("numpy", "pandas", "scipy", "matplotlib", "torch", "hydra-core", "omegaconf", "pyarrow", "eclipse-sumo")
LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}  # fmt: skip
ALIGN_LEFT = ("text", "flag")


@dataclasses.dataclass(frozen=True)
class ReportConfig:
    """Paths and the choices of the figures (``configs/make_report.yaml``)."""

    runs_root: Path
    out_dir: Path
    tables_m4: Path
    tables_m5: Path
    corridor_root: Path
    docs_dir: Path
    configs_dir: Path = REPO_ROOT / "configs"
    tables_m8: Path | None = None  # None: <runs_root>/_tables/m8
    supplement_figures: Path | None = None  # None: <out_dir>/supplement/figures
    m8_tables: tuple[str, ...] = M8_TABLES  # expected tables of M8 (a missing one gets a note)
    supplement_expected: tuple[str, ...] = SUPPLEMENT_FIGURES  # expected figures or patterns of M8
    refresh: bool = False
    gain_architectures: tuple[str, ...] = ("idm", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl")
    existence_experiment: str = "e2_existence"
    existence_architectures: tuple[str, ...] = ("gru", "lstm", "perl")
    e5_views: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {"openacc_acc": "acc", "openacc_human": "human"}
    )
    e5_models: tuple[str, ...] = ("idm", "mlp", "gru", "lstm")
    e5_penalised: tuple[str, ...] = ("mlp", "gru", "lstm")
    pulse_view: str = "openacc_human"
    scenario: str = "i80_p1"
    figure_laws: tuple[str, ...] = ("idm_global", "residual_idm_certified", "mlp", "gru", "lstm")
    figure_seed: int = 0
    contour_dx: float = 20.0
    contour_dt: float = 2.0
    contour_lanes: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    fd_dx: float = 100.0
    fd_dt: float = 30.0
    fd_bin: float = 10.0
    gain_threshold: float = 1.02
    dpi: int = 200
    font_size: float = 9.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], paths: Mapping[str, Path]) -> ReportConfig:
        names = {f.name for f in dataclasses.fields(cls)} - set(paths)
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"unknown keys of the report config: {sorted(unknown)}")
        values = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
        if "e5_views" in values:
            values["e5_views"] = dict(values["e5_views"])
        return cls(**{key: Path(value) for key, value in paths.items() if value is not None}, **values)

    @property
    def m8_dir(self) -> Path:
        return self.tables_m8 if self.tables_m8 is not None else self.runs_root / "_tables" / "m8"

    @property
    def supplement_dir(self) -> Path:
        return self.supplement_figures if self.supplement_figures is not None else self.out_dir / "supplement" / "figures"


@dataclasses.dataclass
class ReportTable:
    """A table of the report: the frame of its CSV, the columns of its LaTeX and Markdown, its caption."""

    name: str
    title: str
    caption: str
    frame: pd.DataFrame
    columns: list[Column]
    source: str = ""
    groups: list[tuple[str, int]] | None = None  # LaTeX header groups over the columns: (label, span)
    note: str | None = None  # why the table is empty or incomplete


# ----------------------------------------------------------------------------------------------- LaTeX


def latex_escape(text: Any) -> str:
    """Text with the special characters of LaTeX escaped; ``^2`` (units) becomes a superscript."""
    escaped = "".join(LATEX_SPECIAL.get(ch, ch) for ch in str(text))
    return escaped.replace(r"\textasciicircum{}2", r"$^2$")


def _latex_cell(row: Mapping[str, Any], column: Column) -> str:
    text = _cell(row.get(column.key), column.kind, column.digits)
    if column.ci and text:
        low, high = (_cell(row.get(f"{column.key}_{end}"), column.kind, column.digits) for end in ("low", "high"))
        if low and high:
            text = f"{text} [{low}, {high}]"
    if column.kind == "p" and text.startswith("<"):
        return "$<$" + latex_escape(text[1:])
    if column.kind in ALIGN_LEFT:
        return latex_escape(text)
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", latex_escape(text))  # minus signs, not hyphens


def latex_table(table: ReportTable) -> str:
    """The table as a LaTeX ``table`` environment: booktabs rules, one row per line ending in ``\\\\``, the
    caption with the design; tables wider than eight columns are scaled to the line width (graphicx)."""
    columns = table.columns
    wide = len(columns) > 8
    align = "".join("l" if c.kind in ALIGN_LEFT else "r" for c in columns)
    lines = [
        f"% {table.name}: generated by scripts/make_report.py from {table.source or 'the tables of the project'}.",
        "% Needs \\usepackage{booktabs}" + (" and \\usepackage{graphicx}." if wide else "."),
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(table.caption)}}}",
        rf"\label{{tab:{table.name.replace('_', '-')}}}",
        r"\small",
    ]
    if wide:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if table.groups:
        cells, rules, start = [], [], 1
        for label, span in table.groups:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{latex_escape(label)}}}" if span > 1 else latex_escape(label))
            if label and span > 1:
                rules.append(rf"\cmidrule(lr){{{start}-{start + span - 1}}}")
            start += span
        lines.append(" & ".join(cells) + r" \\")
        if rules:
            lines.append(" ".join(rules))
    lines.append(" & ".join(latex_escape(c.header) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for row in table.frame.to_dict("records"):
        lines.append(" & ".join(_latex_cell(row, c) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        lines.append("}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def markdown_table(table: ReportTable, level: int = 4) -> str:
    """The table in Markdown (``cf_stability.eval.tables.markdown``), its title one heading level down."""
    text = markdown(Table(table.name, table.title, [table.caption], table.frame, table.columns))
    return "#" * level + text[1:] if text.startswith("# ") else text


# --------------------------------------------------------------------------------------- docs parsing


def _number(name: str) -> int:
    return int(re.sub(r"\D", "", name) or 0)


def milestone_reports(docs_dir: Path) -> list[tuple[int, str, str]]:
    """``(k, title, text)`` of ``docs/m<k>_report.md`` in the order of the milestones."""
    out = []
    for path in sorted(docs_dir.glob("m*_report.md"), key=lambda p: _number(p.stem)):
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[0] if text else ""
        out.append((_number(path.stem.split("_")[0]), first.split(":", 1)[1].strip() if ":" in first else "", text))
    return out


def decision_entries(path: Path, docs_dir: Path | None = None) -> list[dict[str, Any]]:
    """The D-entries of ``docs/decisions.md`` (id, date, first sentence of the decision) with their milestone,
    sorted by milestone and number. The milestone is the one of the section head ``## <name> (M<k>)``, or a
    later one: a section can hold the entries of several milestones (``Data (M1)`` those of M1 to M3), and
    the ids are numbered in time, so an entry belongs at least to the last milestone whose report (in
    ``docs_dir``) cites as its first new entry an id at or below it."""
    if not path.exists():
        return []
    entries, section, names = [], None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        head = re.match(r"^##\s+(.*?)\s*\(M(\d+)\)\s*$", line)
        if head:
            section = int(head.group(2))
            names.setdefault(section, head.group(1))
            continue
        if line.startswith("## "):
            section = None
            continue
        # a cell ends at the first pipe that is not escaped (``\|`` inside a cell, e.g. ``\|f_dv\|``)
        row = re.match(r"^\|\s*(D\d+)\s*\|\s*([^|]*?)\s*\|\s*((?:\\\||[^|])*?)\s*\|\s*(.*?)\s*\|\s*$", line)
        if row and section is not None:
            decision = first_sentence(row.group(3)).replace("\\|", "|")  # outside a table the pipe needs no escape
            entries.append({"id": row.group(1), "number": _number(row.group(1)), "date": row.group(2),
                            "section": section, "decision": decision})  # fmt: skip
    starts, known = [], 0  # (k, first new id cited by the report of M<k>)
    titles = {}
    for k, title, text in milestone_reports(docs_dir) if docs_dir is not None and docs_dir.is_dir() else []:
        titles[k] = title
        cited = {_number(m) for m in re.findall(r"\bD\d+\b", text)}
        new = [n for n in cited if n > known]
        if new:
            starts.append((k, min(new)))
            known = max(known, *new)
    for entry in entries:
        by_number = max((k for k, start in starts if start <= entry["number"]), default=entry["section"])
        entry["milestone"] = max(entry["section"], by_number)
        title = titles.get(entry["milestone"]) or names.get(entry["milestone"], "")
        entry["label"] = f"M{entry['milestone']}" + (f" ({title})" if title else "")
    return sorted(entries, key=lambda e: (e["milestone"], e["number"]))


def first_sentence(text: str) -> str:
    """The first sentence of a cell of the decisions log, without Markdown emphasis."""
    text = text.replace("**", "").strip()
    masked = re.sub(r"\b(e\.g|i\.e|cf|vs|approx|incl)\.", lambda m: m.group(0).replace(".", "\0"), text)
    match = re.search(r"\.(\s|$)", masked)
    sentence = masked[: match.start() + 1] if match else masked
    return sentence.replace("\0", ".")


def milestone_commands(docs_dir: Path) -> list[tuple[str, str, list[str]]]:
    """``(milestone, language, lines)`` of the code block under "Commands run" of every milestone report."""
    out = []
    for k, _, text in milestone_reports(docs_dir) if docs_dir.is_dir() else []:
        start = re.search(r"^##\s+\d*\.?\s*Commands run.*$", text, flags=re.MULTILINE)
        if not start:
            continue
        block = re.search(r"^```(\w*)\n(.*?)^```", text[start.end():], flags=re.MULTILINE | re.DOTALL)
        if block:
            out.append((f"M{k}", block.group(1) or "bash", block.group(2).rstrip("\n").splitlines()))
    return out


# ------------------------------------------------------------------------------------------- the maker


def _q(path: Path) -> str:
    return f"'{Path(path).as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def _flag(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def profile_label(profile: str, mode: str | None = None) -> str:
    """Short name of a platoon profile: site and part of an OpenACC file ("Vicolungo part 2"), or the pulse."""
    if profile == "pulse":
        text = "braking pulse"
    else:
        path = Path(profile)
        part = re.search(r"part(\d+)", path.stem)
        text = f"{path.parent.name} part {part.group(1)}" if part else path.stem.replace("_", " ")
    return text if not mode else f"{text} ({mode})"


def std_before_collision(profile: Mapping[str, Any]) -> np.ndarray:
    """Speed std per platoon position of one simulated platoon, NaN from the first follower whose gap reached
    zero onward (``min_gap``; ``collision_vehicle`` without it): what follows a collision is not car following."""
    std = np.asarray([np.nan if v is None else v for v in profile.get("speed_std") or []], dtype=float)
    gaps = np.asarray([np.nan if v is None else v for v in profile.get("min_gap") or []], dtype=float)
    if gaps.shape == std.shape and gaps.size > 1:
        hit = np.flatnonzero(gaps[1:] <= 0.0)
        if hit.size:
            std[1 + int(hit[0]):] = np.nan
    elif _flag(profile.get("collided")) and isinstance(profile.get("collision_vehicle"), int):
        std[int(profile["collision_vehicle"]):] = np.nan
    return std


class ReportMaker:
    """Builds the tables, the figures, report.md and the manifest; every missing input becomes a note."""

    def __init__(self, cfg: ReportConfig) -> None:
        self.cfg = cfg
        self.notes: list[str] = []
        self.tables: dict[str, ReportTable] = {}
        self.figures: dict[str, str | None] = {}  # name -> None when written, else the reason it is missing
        self.refreshed: list[str] = []
        self.m8_names: list[str] = []  # the tables of M8 in the report, in order
        self.platoons = [0, 0]  # simulated platoons of figure 4: collided, all
        self.m4_design = self._yaml(cfg.configs_dir / "make_tables.yaml")
        self.m5_design = (self._yaml(cfg.configs_dir / "corridor_metrics.yaml").get("design") or {})
        chosen = cfg.tables_m4 / "chosen_weights.json"
        self.chosen: dict[str, Any] = {}
        if chosen.exists():
            try:
                self.chosen = read_json(chosen)
            except (OSError, ValueError):
                self.note("E2", f"{self.label(chosen)} unreadable")
        else:
            self.note("E2", f"{self.label(chosen)} missing: no chosen weights")
        self._maker: TableMaker | None = None

    # ------------------------------------------------------------------------------------ helpers
    def note(self, where: str, text: str) -> None:
        self.notes.append(f"{where}: {text}")

    def label(self, path: Path) -> str:
        for root in (self.cfg.runs_root, REPO_ROOT):
            try:
                return Path(path).relative_to(root).as_posix()
            except ValueError:
                continue
        return Path(path).as_posix()

    @staticmethod
    def _yaml(path: Path) -> dict[str, Any]:
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}

    def csv(self, folder: Path, name: str, where: str) -> pd.DataFrame | None:
        path = folder / f"{name}.csv"
        if not path.exists():
            self.note(where, f"{self.label(path)} missing")
            return None
        try:
            return pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            self.note(where, f"{self.label(path)} unreadable ({type(exc).__name__})")
            return None

    @property
    def maker(self) -> TableMaker:
        """The table code of M4, for the numbers the M4 tables do not hold (existence arm, E1 of seed 0)."""
        if self._maker is None:
            d = self.m4_design
            tables_cfg = TablesConfig(
                runs_root=self.cfg.runs_root, out_dir=self.cfg.out_dir / "tables",
                data=d.get("data", "follownet_highd"), folds=tuple(d.get("folds", (0, 1, 2, 3, 4))),
                seeds=tuple(d.get("seeds", (0, 1, 2, 3, 4))), n_resamples=int(d.get("n_resamples", 1000)),
                level=float(d.get("level", 0.95)), seed=int(d.get("seed", 0)),
            )  # fmt: skip
            self._maker = TableMaker(tables_cfg)
        return self._maker

    def design_text(self, kind: str) -> str:
        """Intervals of the tables of M4 or M5, from their configuration files."""
        d = self.m4_design if kind == "m4" else self.m5_design
        return (f"{100 * float(d.get('level', 0.95)):g} % percentile bootstrap intervals, "
                f"{int(d.get('n_resamples', 1000))} resamples")  # fmt: skip

    @staticmethod
    def runs_in(frame: pd.DataFrame | None, *keys: str) -> int:
        if frame is None:
            return 0
        return int(sum(pd.to_numeric(frame[k], errors="coerce").fillna(0).sum() for k in keys if k in frame))

    def runless(self, table: ReportTable, counts: tuple[str, ...], labels: tuple[str, ...]) -> None:
        """A note for the rows of ``table`` without runs (the arms of M7 before their queues): the table note
        when no row has a run, else one note naming the rows (``labels``, each once)."""
        frame = table.frame
        present = [k for k in counts if k in frame]
        if frame.empty or not present:
            return
        total = sum(pd.to_numeric(frame[k], errors="coerce").fillna(0) for k in present)
        empty = frame[total.to_numpy() == 0]
        if len(empty) == len(frame):
            table.note = table.note or "no runs yet"
        elif len(empty):
            names = [" ".join(_cell(r.get(k), "weight" if k in ("weight", "r_max") else "text") for k in labels
                              if k in r and _cell(r.get(k), "text")) for r in empty.to_dict("records")]  # fmt: skip
            self.note(f"table {table.name}", f"rows without runs yet: {', '.join(dict.fromkeys(names))}")

    def corridors(self, frame: pd.DataFrame | None) -> list[str]:
        """The corridors of a table of M5 (its column ``corridor``), in order."""
        if frame is None or "corridor" not in frame:
            return []
        return list(dict.fromkeys(str(c) for c in frame["corridor"].dropna()))

    def design_corridors(self) -> list[str]:
        """The corridors of the design of M5 (configs/corridor_metrics.yaml: scenarios and the corridor names of
        their prefixes)."""
        d = self.m5_design
        names = d.get("corridors") or {}
        return list(dict.fromkeys(names.get(str(s).split("_", 1)[0], str(s).split("_", 1)[0])
                                  for s in d.get("scenarios") or []))  # fmt: skip

    # ------------------------------------------------------------------------------------- refresh
    def refresh(self) -> None:
        """Rerun ``scripts/make_tables.py`` and ``scripts/corridor_metrics.py tables=true`` (child processes)."""
        c = self.cfg
        commands = {
            "make_tables": [sys.executable, str(REPO_ROOT / "scripts" / "make_tables.py"),
                            f"paths.runs_root={_q(c.runs_root)}", f"paths.out_dir={_q(c.tables_m4)}",
                            f"paths.out_dir_m8={_q(c.m8_dir)}"],
            "corridor_metrics": [sys.executable, str(REPO_ROOT / "scripts" / "corridor_metrics.py"), "tables=true",
                                 f"paths.corridor_root={_q(c.corridor_root)}", f"paths.runs_root={_q(c.runs_root)}",
                                 f"paths.out_dir={_q(c.tables_m5)}", f"paths.out_dir_m8={_q(c.m8_dir)}"],
        }  # fmt: skip
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        for script, command in commands.items():
            proc = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")  # fmt: skip
            self.refreshed.append(f"REFRESH {script}: exit {proc.returncode}")
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
                self.note("refresh", f"scripts/{script}.py failed (exit {proc.returncode}): {tail[0]}")

    # -------------------------------------------------------------------------------------- tables
    def add(self, table: ReportTable) -> None:
        self.tables[table.name] = table

    def m4_table(self, name: str, title: str, caption: str, columns: list[Column]) -> None:
        frame = self.csv(self.cfg.tables_m4, name, f"table {name}")
        self.add(ReportTable(name, title, caption, frame if frame is not None else pd.DataFrame(), columns,
                             f"runs/_tables/m4/{name}.csv",
                             note=None if frame is not None else "input missing"))  # fmt: skip

    def tables_m4(self) -> None:
        d, design = self.m4_design, self.design_text("m4")
        folds, seeds = len(d.get("folds", [])), len(d.get("seeds", []))
        reference = d.get("reference", "idm")
        e1 = self.csv(self.cfg.tables_m4, "e1", "table e1")
        self.m4_table("e1", "E1: models without penalty", (
            f"E1 on {d.get('data', 'follownet_highd')}: {self.runs_in(e1, 'runs')} runs "
            f"({folds} folds x {seeds} seeds, "
            f"baselines {folds} folds). RMSE s: spacing RMSE of the test parts (m), unit driver; shares of the grid "
            f"speeds in support, unit run; RMSE vs {reference}: relative difference, paired over drivers; {design}."
        ), [
            Column("model", "model", "text"), Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("collision_rate", "collisions", digits=4),
            Column("rmse_vs_ref", f"RMSE vs {reference}", "pct", 1, ci=True), Column("rmse_vs_ref_p", "p", "p"),
            Column("unstable_eq", "unstable among equilibria", ci=True), Column("stable", "stable"),
            Column("unstable", "unstable", ci=True), Column("outside", "outside"), Column("none", "none"),
            Column("not_stable", "not stable", ci=True), Column("max_gain_median", "max gain (median)", digits=3),
            Column("h1_1", "H1.1", "text"),
        ])  # fmt: skip
        sweep = self.csv(self.cfg.tables_m4, "e2_sweep", "table e2_sweep")
        self.m4_table("e2_sweep", "E2: sweep of the penalty weight", (
            f"E2 sweep: {self.runs_in(sweep, 'runs')} runs ({folds} folds, seed 0, four weights per architecture). "
            "RMSE s (m): mean over the runs of the mean spacing RMSE of the validation and test events; stable: band "
            "share of the grid speeds in support; feasible: runs whose best epoch meets the penalty (D89); chosen: D85."
        ), [
            Column("architecture", "architecture", "text"), Column("kind", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("runs", "runs", "int"),
            Column("val_rmse_s", "val. RMSE s (m)"), Column("test_rmse_s", "test RMSE s (m)"),
            Column("stable", "stable"),
            Column("feasible", "feasible", "int"), Column("chosen", "chosen", "flag"),
        ])  # fmt: skip
        e2 = self.csv(self.cfg.tables_m4, "e2", "table e2")
        self.m4_table("e2", "E2: chosen weights against E1 (H1.2)", (
            f"E2 with the chosen weights: {self.runs_in(e2, 'runs_e2')} runs against "
            f"{self.runs_in(e2, 'runs_e1')} of E1 "
            f"({folds} folds x {seeds} seeds). RMSE change: relative change of the spacing RMSE, unit driver, paired; "
            "p (Holm) over the architectures; shares, growth error and hysteresis area (m^2/s): unit run. Growth "
            "error (D109): per OpenACC profile on the collision-free prefix of the platoon (at least 3 positions), "
            "mean over the profiles of a run that have one; whole curve (D107): over the profiles whose platoon did "
            "not collide; each compared over the pairs of runs in which both have a value; collided: OpenACC "
            "profiles whose platoon collided (growth error) and runs whose first follower collided behind the "
            f"braking pulse (hysteresis), summed over the runs; {design}."
        ), [
            Column("architecture", "architecture", "text"), Column("kind", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("runs_e2", "runs", "int"),
            Column("rmse_change", "RMSE change", "pct", 1, ci=True), Column("rmse_change_p_holm", "p (Holm)", "p"),
            Column("not_stable_e1", "not stable E1", ci=True), Column("not_stable_e2", "not stable E2", ci=True),
            Column("unstable_eq_e1", "unstable among eq. E1", ci=True),
            Column("unstable_eq_e2", "unstable among eq. E2", ci=True),
            Column("growth_error_pairs", "growth pairs", "int"), Column("growth_collided_e1", "collided E1", "int"),
            Column("growth_collided_e2", "collided E2", "int"),
            Column("growth_error_e1", "growth error E1", digits=3),
            Column("growth_error_e2", "growth error E2", digits=3),
            Column("growth_error_change", "growth change", "pct", 1, ci=True),
            Column("growth_error_full_pairs", "whole-curve pairs", "int"),
            Column("growth_error_full_change", "whole-curve change", "pct", 1, ci=True),
            Column("hysteresis_pairs", "hysteresis pairs", "int"),
            Column("hysteresis_collided_e1", "collided E1", "int"),
            Column("hysteresis_collided_e2", "collided E2", "int"),
            Column("hysteresis_e1", "hysteresis E1 (m^2/s)", digits=1),
            Column("hysteresis_e2", "hysteresis E2 (m^2/s)", digits=1), Column("h1_2", "H1.2", "text"),
        ])  # fmt: skip
        e3 = self.csv(self.cfg.tables_m4, "e3", "table e3")
        self.m4_table("e3", "E3: transfer to other data", (
            f"E3: models of seed 0 ({folds} folds) on all events of the targets; "
            f"{self.runs_in(e3, 'runs')} evaluations. "
            "RMSE s (m) of the first 15 s and of the whole events, unit driver of the target; relative degradation "
            f"against the source test RMSE; {design}."
        ), [
            Column("architecture", "architecture", "text"), Column("penalty", "penalty", "text"),
            Column("weight", "weight", "weight"), Column("target", "target", "text"),
            Column("drivers", "drivers", "int"),
            Column("source_rmse_s", "source RMSE s (m)"), Column("rmse_s_h", "RMSE s, 15 s (m)", ci=True),
            Column("rmse_s_full", "RMSE s, whole (m)", ci=True),
            Column("degradation", "degradation", "pct", 1, ci=True),
            Column("degradation_change", "with - without penalty", "pct", 1, ci=True),
        ])  # fmt: skip
        e4 = self.csv(self.cfg.tables_m4, "e4", "table e4")
        self.m4_table("e4", "E4: certified hybrid and fine-tuning (H1.5)", (
            f"E4: {self.runs_in(e4, 'runs')} runs ({folds} folds x {seeds} seeds per row). Certificates: runs in which "
            "it holds; shares of the grid speeds in support, unit run; RMSE s: spacing RMSE of the test part of the "
            f"data of the run (m), unit driver; {design}."
        ), [
            Column("model", "model", "text"), Column("variant", "variant", "text"), Column("data", "data", "text"),
            Column("runs", "runs", "int"), Column("a_priori_holds", "a priori holds", "int"),
            Column("at_equilibria_holds", "at equilibria holds", "int"),
            Column("all_equilibria_certified", "all equilibria certified", "int"),
            Column("unstable", "unstable", ci=True), Column("not_stable", "not stable", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True), Column("rmse_s", "RMSE s (m)", ci=True),
        ])  # fmt: skip
        lowfreq = self.csv(self.cfg.tables_m4, "e2_lowfreq", "table e2_lowfreq")
        pilot = ", ".join(str(f) for f in d.get("lowfreq_pilot_folds", [0]))
        first_seed = (d.get("seeds") or [0])[0]
        self.m4_table("e2_lowfreq", "E2: combined penalty of the recurrent models (D110)", (
            f"Low-frequency arm of E2 (D110): {self.runs_in(lowfreq, 'runs')} runs of the combined penalty (rollout "
            "gain penalty of E2, Jacobian penalty of the memoryless view, needle guard) per architecture and "
            f"Jacobian weight (the pilot on fold {pilot}, the chosen weight on all {folds} folds, seed {first_seed}). "
            "RMSE s (m): unit driver; RMSE change against E1 of the same folds and seed, paired over drivers; shares "
            f"of the grid speeds in support, unit run; H1.2 (combined): the rule of H1.2 on the chosen weight; {design}."
        ), [
            Column("architecture", "architecture", "text"), Column("weight", "Jacobian weight", "weight"),
            Column("chosen", "chosen", "flag"), Column("runs", "runs", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("rmse_change", "RMSE change vs E1", "pct", 1, ci=True), Column("stable", "stable"),
            Column("not_stable", "not stable", ci=True), Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("max_gain_median", "max gain (median)", digits=3), Column("collided", "collided", "int"),
            Column("profiles", "profiles", "int"), Column("h1_2", "H1.2 (combined)", "text"),
        ])  # fmt: skip
        self.runless(self.tables["e2_lowfreq"], ("runs",), ("architecture", "weight"))
        e5 = self.csv(self.cfg.tables_m4, "e5", "table e5")
        self.m4_table("e5", "E5: OpenACC views (H1.3)", (
            f"E5 on the OpenACC views: {self.runs_in(e5, 'runs')} runs ({folds} folds, seed 0). "
            "RMSE s (m): unit driver; "
            "shares of the grid speeds in support and growth error of the driving mode of the view, unit run; the "
            "growth error of a profile is taken on the collision-free prefix of its simulated platoon (D109: the "
            "followers ahead of the first collided position, at least 3), the growth error of a run is the mean over "
            "the profiles of the mode that have one (collided: of profiles, summed over the runs; first collided "
            "position: one past the last vehicle without collision; collision-free share of the profiles); "
            "reduction: 1 - growth error with / without penalty, paired by fold over the folds in which both runs "
            f"have a growth error (pairs; H1.3 open with fewer than "
            f"{int((d.get('verdicts') or {}).get('h1_3_min_pairs', 3))} pairs); whole curve: the same on the growth "
            f"error of D107 (none after a collision); {design}."
        ), [
            Column("view", "view", "text"), Column("model", "model", "text"), Column("penalty", "penalty", "text"),
            Column("runs", "runs", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("unstable_eq", "unstable among equilibria", ci=True), Column("not_stable", "not stable", ci=True),
            Column("max_gain_median", "max gain (median)", digits=3), Column("collided", "collided", "int"),
            Column("profiles", "profiles", "int"),
            Column("first_collided", "first collided position", digits=1, ci=True),
            Column("collision_free", "collision-free share", digits=2, ci=True),
            Column("growth_error", "growth error", digits=3),
            Column("reduction_pairs", "pairs", "int"), Column("reduction", "reduction", "pct", 1, ci=True),
            Column("reduction_p", "p", "p"), Column("h1_3", "H1.3", "text"),
            Column("reduction_full_pairs", "whole-curve pairs", "int"),
            Column("reduction_full", "whole-curve reduction", "pct", 1, ci=True),
        ])  # fmt: skip
        self.m4_table("verdicts", "Verdicts of the hypotheses", (
            "Verdicts of H1.1 (E1), H1.2 (E2), H1.2 (combined) (the low-frequency arm of E2, D110), H1.3 (E5, on the "
            "collision-free prefix of the platoons, D109) and H1.5 (E4) with their basis (configs/make_tables.yaml, "
            "bases); complete: drawn from every run of the design."
        ), [
            Column("hypothesis", "hypothesis", "text"), Column("unit", "unit", "text"),
            Column("verdict", "verdict", "text"),
            Column("complete", "complete", "flag"), Column("basis", "basis", "text"),
        ])  # fmt: skip

    def table_existence(self) -> None:
        """The existence arm of E2 (D93): the existence term alone against E1, with the table code of M4."""
        c, t, maker = self.cfg, "e2_existence", self.maker
        files = (METRICS, AUDIT, EVENTS, PLATOON)
        rows = []
        for arch in c.existence_architectures:
            runs = maker.runs(t, c.existence_experiment, maker.cfg.data, arch, maker.cfg.seeds[:1], files)
            base = maker.runs(t, "e1", maker.cfg.data, arch, maker.cfg.seeds[:1], (METRICS, EVENTS))
            row: dict[str, Any] = {"architecture": arch, "runs": maker.trained(runs)}
            drivers = maker.drivers(t, runs)
            maker.rmse(row, drivers)
            maker.versus(row, "rmse_vs_e1", maker.drivers(t, base), drivers)
            maker.shares(t, row, runs)
            row["collided_profiles"] = maker.values(t, runs, "platoon_n_collided", PLATOON).mean()
            row["profiles"] = maker.values(t, runs, "platoon_n_profiles").mean()
            rows.append(row)
        for line in maker.missing:
            if line.startswith(f"[{t}]"):
                self.note("table e2_existence", line[len(t) + 3:])
        maker.missing = [line for line in maker.missing if not line.startswith(f"[{t}]")]
        frame = pd.DataFrame(rows)
        self.add(ReportTable(t, "E2: the existence term alone (D93)", (
            f"Arm {c.existence_experiment} of E2: {self.runs_in(frame, 'runs')} runs (existence term only, "
            f"{len(maker.cfg.folds)} folds, seed 0) against E1 of seed 0. RMSE s (m): unit driver; RMSE vs E1: "
            "relative difference, paired over drivers; shares of the grid speeds in support, unit run; collided "
            f"profiles: platoon profiles with a collision, mean over the runs; {self.design_text('m4')}."
        ), frame, [
            Column("architecture", "architecture", "text"), Column("runs", "runs", "int"),
            Column("drivers", "drivers", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("rmse_vs_e1", "RMSE vs E1", "pct", 1, ci=True), Column("rmse_vs_e1_p", "p", "p"),
            Column("stable", "stable"), Column("unstable", "unstable"), Column("outside", "outside"),
            Column("none", "none"), Column("unstable_eq", "unstable among equilibria", ci=True),
            Column("max_gain_median", "max gain (median)", digits=2),
            Column("collided_profiles", "collided profiles", digits=1),
        ], "the runs of the arm and of E1 (cf_stability.eval.tables)"))  # fmt: skip

    def tables_m5(self) -> None:
        c, d, design = self.cfg, self.m5_design, self.design_text("m5")
        laws = self.csv(c.tables_m5, "laws", "table laws")
        per_law = laws[~laws["law"].eq("ground truth")] if laws is not None and "law" in laws else laws
        runs = self.runs_in(per_law, "runs")
        corridors = self.corridors(per_law)
        if per_law is not None and "corridor" in per_law:
            parts = []
            for corridor in corridors:
                mine = per_law[per_law["corridor"] == corridor]
                truths = laws[laws["law"].eq("ground truth") & laws["corridor"].eq(corridor)]
                parts.append(f"{corridor}: {int((pd.to_numeric(mine['runs'], errors='coerce') > 0).sum())} of "
                             f"{len(mine)} laws with runs, {len(truths)} scenarios")  # fmt: skip
            shape = "; ".join(parts)
        else:
            shape = (f"{len(d.get('laws', []))} laws x {len(d.get('scenarios', []))} scenarios"
                     if per_law is not None else "no table")  # fmt: skip
        common = f"{runs} runs ({shape}; {len(d.get('seeds', []))} seeds per scenario); unit run; {design}"
        absent = [corridor for corridor in self.design_corridors() if corridors and corridor not in corridors]
        if absent:
            self.note("corridor", f"{', '.join(absent)} not in the tables yet (no ground truth with a macro.json); the "
                                  "H12 verdicts name the corridors they use")  # fmt: skip
        self.add(ReportTable("laws", "Corridor: macro error of the laws", (
            f"Corridor laws: {common}. Macro error: mean of the absolute relative components (dynamic: FD, wave "
            "speed, waves, wave amplitude); collisions per 1000 vehicle-km inside the analysis window."
        ), per_law if per_law is not None else pd.DataFrame(), [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("scenario_set", "scenarios", "text"), Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("n_components", "components", digits=1),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3, ci=True),
            Column("runs_with_collisions", "runs with collisions", "int"),
            Column("inserted_share", "inserted", digits=3),
        ], "runs/_tables/m5/laws.csv", note=None if laws is not None else "input missing"))  # fmt: skip
        self.runless(self.tables["laws"], ("runs",), ("corridor", "law"))
        raw_keys = ("throughput_vph", "mean_speed", "queue_discharge_flow", "capacity_drop", "n_waves",
                    "wave_speed_xcorr", "wave_amplitude", "travel_time_mean", "collisions_per_1000_vkm")  # fmt: skip
        wanted = ["corridor", "law", "kind", "scenario", "runs",
                  *(k2 for k in RAW_METRICS for k2 in (k, f"{k}_low", f"{k}_high"))]  # fmt: skip
        raw_frame = laws[[k for k in wanted if k in laws.columns]] if laws is not None else pd.DataFrame()
        self.add(ReportTable("laws_raw", "Corridor: raw metrics per law and the ground truth", (
            f"Raw metrics of the corridor: per law the mean over its runs ({common}); ground truth: the data per "
            "scenario (no interval). Flows in veh/h at the throughput detector, speeds in m/s, travel time in s."
        ), raw_frame, [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"),
            Column("scenario", "scenario", "text"), Column("runs", "runs", "int"),
            *(Column(k, RAW_METRICS[k][0], digits=RAW_METRICS[k][1], ci=True) for k in raw_keys if k in RAW_METRICS),
        ], "runs/_tables/m5/laws.csv", note=None if laws is not None else "input missing"))  # fmt: skip
        instability = self.csv(c.tables_m5, "instability", "table instability")
        self.add(ReportTable("instability", "Corridor: instability of the members and macro error", (
            f"Instability of the members of every law (audits of the member runs; idm_heterogeneous, "
            f"idm_heterogeneous_all, the cores of residual_idm_certified_het and an IDM of calibrations without member "
            f"runs: exact gain of their parameter sets) and the macro errors ({common}); the residual amplitudes of "
            "D111 are in table e4_rmax; the temporal hold-out laws of D120 run on their scenarios only and are not in "
            "the correlations."
        ), instability if instability is not None else pd.DataFrame(), [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("members", "members", "int"),
            Column("audited", "audited", "int"), Column("unstable_eq", "unstable among equilibria", digits=3),
            Column("not_stable", "not stable", digits=3), Column("runs", "runs", "int"),
            Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3),
            Column("collides", "collides", "flag"), Column("scenario_set", "scenarios", "text"),
            Column("in_correlation", "in the correlation", "flag"),
        ], "runs/_tables/m5/instability.csv", note=None if instability is not None else "input missing"))  # fmt: skip
        corr = self.csv(c.tables_m5, "instability_correlation", "table instability_correlation")
        corr_columns = [
            Column("corridor", "corridor", "text"), Column("error", "error", "text"),
            Column("measure", "instability", "text"), Column("laws", "laws", "text"),
            Column("method", "method", "text"), Column("n", "n", "int"),
            Column("estimate", "correlation", digits=3, ci=True),
            Column("n_valid", "valid resamples", "int"), Column("p_permutation", "p (permutation)", "p"),
        ]  # fmt: skip
        self.add(ReportTable("instability_correlation", "Corridor: correlation of instability and macro error", (
            "Spearman and Pearson correlation per corridor over the laws of the mean macro error (all components or "
            "the dynamic ones) with the instability of the members; percentile bootstrap over the laws "
            f"({int(d.get('n_resamples', 1000))} resamples), two-sided permutation p-value "
            f"({int(d.get('n_permutations', 10000))} permutations of the instability over the laws, D108), all laws "
            "and the laws without collisions."
        ), corr if corr is not None else pd.DataFrame(), corr_columns, "runs/_tables/m5/instability_correlation.csv",
            note=None if corr is not None else "input missing"))  # fmt: skip
        tost = self.csv(c.tables_m5, "tost", "table tost")
        candidate, reference = d.get("candidate", "candidate"), d.get("reference", "reference")
        self.add(ReportTable("tost", f"Corridor: equivalence of {candidate} and {reference}", (
            f"TOST of {candidate} against {reference} per corridor, paired by scenario and seed (pairs); relative "
            f"difference with its {design.split(',')[0]}; margin +-{100 * float(d.get('margin', 0.1)):g} % of the mean "
            f"of {reference}, alpha {float(d.get('alpha', 0.05)):g}."
        ), tost if tost is not None else pd.DataFrame(), [
            Column("corridor", "corridor", "text"), Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("mean_reference", reference, digits=3), Column("mean_candidate", candidate, digits=3),
            Column("relative_difference", "relative difference", "pct", 1, ci=True),
            Column("wilcoxon_p", "p (Wilcoxon)", "p"), Column("p_value", "p (TOST)", "p"),
            Column("equivalent", "equivalent", "flag"),
        ], "runs/_tables/m5/tost.csv", note=None if tost is not None else "input missing"))  # fmt: skip
        self.table_components(common)
        self.table_dynamic(instability, corr, common)
        self.tables_h12()
        self.tables_arms()

    def m5_table(self, name: str, source: str, title: str, caption: str, columns: list[Column]) -> ReportTable:
        """A table of M5 from ``runs/_tables/m5/<source>.csv`` (``name`` in the report)."""
        frame = self.csv(self.cfg.tables_m5, source, f"table {name}")
        table = ReportTable(name, title, caption, frame if frame is not None else pd.DataFrame(), columns,
                            f"runs/_tables/m5/{source}.csv", note=None if frame is not None else "input missing")  # fmt: skip
        self.add(table)
        return table

    def tables_h12(self) -> None:
        """The verdicts of H12.1-H12.3 and their tables h12_1 and h12_2 (D108)."""
        d, design = self.m5_design, self.design_text("m5")
        reference, candidate = d.get("reference", "idm_global"), d.get("candidate", "residual_idm_certified")
        networks = ", ".join(d.get("networks") or ["mlp", "gru", "lstm"])
        self.m5_table("verdicts_h12", "verdicts", "Verdicts of H12.1-H12.3 (corridor)", (
            "Verdicts of H12.1-H12.3 of Part A of the plan (D108) per network and corridor, overall (naming the "
            "corridors used), for the certified hybrid, and per correlation; complete: drawn from every run of the "
            "design on every corridor of the configuration."
        ), [
            Column("hypothesis", "hypothesis", "text"), Column("unit", "unit", "text"),
            Column("corridors", "corridors", "text"), Column("verdict", "verdict", "text"),
            Column("complete", "complete", "flag"), Column("basis", "basis", "text"),
        ])  # fmt: skip
        self.m5_table("h12_1", "h12_1", "Corridor: degradation of the macro triple by the pure networks (H12.1)", (
            f"H12.1 (D108): {networks} against {reference} per corridor; degradation |e_network| - |e_{reference}| of "
            "the signed relative errors of throughput, travel time (W1) and wave speed (cross-correlation), paired by "
            f"scenario and seed (pairs), mean with its {design.split(',')[0]}; degraded: lower end > 0; degraded by >= "
            f"{float(d.get('degradation_min', 0.15)):g}: and the mean at least that."
        ), [
            Column("corridor", "corridor", "text"), Column("network", "network", "text"),
            Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("abs_error_network", "|e| network", digits=3),
            Column("abs_error_reference", f"|e| {reference}", digits=3),
            Column("degradation", "degradation", digits=3, ci=True), Column("degraded", "degraded", "flag"),
            Column("strong", f"degraded by >= {float(d.get('degradation_min', 0.15)):g}", "flag"),
            Column("h12_1", "H12.1", "text"),
        ])  # fmt: skip
        micro_reference = str(d.get("micro_reference", "e3_reference/idm")).replace("/", " ")
        self.m5_table("h12_2", "h12_2", "NGSIM test parts: micro comparison with the IDM (H12.2)", (
            f"H12.2, micro part (D108): spacing RMSE of the test parts of {d.get('micro_data', 'ngsim_i80')} (m), "
            f"unit driver, {candidate} (fine-tuned certified hybrid) and the reference rows against {micro_reference} "
            f"of the same folds, paired over the drivers (pairs); superior: upper end of the relative difference "
            f"below 0; mean over runs: of rmse_s_mean; {design}."
        ), [
            Column("role", "role", "text"), Column("experiment", "experiment", "text"), Column("model", "model", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"), Column("rmse_s", "RMSE s (m)", ci=True),
            Column("rmse_runs", "mean over runs (m)"), Column("rmse_vs_reference", "RMSE vs IDM", "pct", 1, ci=True),
            Column("rmse_vs_reference_p", "p", "p"), Column("superior", "superior", "flag"),
        ])  # fmt: skip

    def tables_arms(self) -> None:
        """The tables of the arms of M7 that come from the corridor tables: e4_rmax (D111), sensitivity (D113)."""
        d, design = self.m5_design, self.design_text("m5")
        points = ", ".join(f"{float(r.get('r_max', 0)):g} {r.get('core', '')}" for r in d.get("rmax") or []
                           if isinstance(r, Mapping)) or "the configured amplitudes"  # fmt: skip
        rmax = self.m5_table("e4_rmax", "e4_rmax", "E4: residual amplitude, certificate, accuracy and corridor (D111)", (
            f"Residual-amplitude sweep of the hybrid (D111): per r_max and core ({points}) the runs on HighD and after "
            "fine-tuning on NGSIM I-80 (folds x seeds of the configuration): runs whose a priori certificate holds, "
            "unstable among equilibria (unit run), spacing RMSE of the test parts (m, unit driver), and the corridor "
            f"numbers of the law (unit run: scenario and seed); {design}."
        ), [
            Column("corridor", "corridor", "text"), Column("r_max", "r_max", "weight"), Column("core", "core", "text"),
            Column("runs_highd", "runs HighD", "int"), Column("a_priori_highd", "a priori holds", "int"),
            Column("runs_ngsim", "runs NGSIM", "int"), Column("a_priori_ngsim", "after fine-tuning", "int"),
            Column("unstable_eq_ngsim", "unstable among eq. (NGSIM)", ci=True),
            Column("rmse_highd", "RMSE s HighD (m)", ci=True), Column("rmse_ngsim", "RMSE s NGSIM (m)", ci=True),
            Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3),
        ])  # fmt: skip
        self.runless(rmax, ("runs_highd", "runs_ngsim", "runs_corridor"), ("r_max", "core"))
        order = " < ".join(d.get("sensitivity_order") or ["residual_idm_certified", "idm_global", "gru"])
        sensitivity = self.m5_table("sensitivity", "sensitivity", "Corridor: sensitivity to the boundary and the "
                                    "lane-change model (D113)", (  # fmt: skip
            f"Variants of {d.get('sensitivity_scenario', 'i80_p1')} (D113) and the scenario itself (baseline, the same "
            "seeds): per variant and law the macro error, the macro triple (signed relative errors) and the "
            f"collisions, unit run (seed); rank of the laws by macro error per variant and whether it is {order}; "
            f"{design}."
        ), [
            Column("variant", "variant", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("macro_error", "macro error", digits=3, ci=True),
            Column("error_throughput", "throughput", digits=3, ci=True),
            Column("error_travel_time", "travel time (W1)", digits=3, ci=True),
            Column("error_wave_speed", "wave speed (xcorr)", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3), Column("rank", "rank", "int"),
            Column("order_holds", order, "flag"),
        ])  # fmt: skip
        self.runless(sensitivity, ("runs",), ("variant",))

    def table_components(self, common: str) -> None:
        components = self.csv(self.cfg.tables_m5, "components", "table m5_components")
        keys = [k for k in (components.columns if components is not None else [])
                if k.startswith(("anchored_", "dynamic_")) and not k.endswith(("_low", "_high"))]  # fmt: skip
        anchored = [k for k in keys if k.startswith("anchored_")]
        dynamic = [k for k in keys if k.startswith("dynamic_")]

        def header(key: str) -> str:
            component = key.split("_", 1)[1]
            return COMPONENT_HEADERS.get("n_waves" if component == "waves" else component, component)

        columns = [Column("corridor", "corridor", "text"), Column("law", "law", "text"),
                   *(Column(k, header(k), digits=3, ci=True) for k in anchored + dynamic)]  # fmt: skip
        groups = [("", 2), ("anchored", len(anchored)), ("dynamic", len(dynamic))] if keys else None
        self.add(ReportTable("m5_components", "Corridor: components of the macro error", (
            f"Components of the macro-error vector per law, signed relative errors against the ground truth "
            f"({common}); anchored: follow the demand and the downstream boundary of the data; dynamic: decided by "
            "the law. FD: RMSE of the binned flow / mean flow of the truth; waves: (run - truth) / max(truth, 1); "
            "travel time: Wasserstein-1 / mean of the truth; wave speed: cross-correlation."
        ), components if components is not None else pd.DataFrame(), columns, "runs/_tables/m5/components.csv", groups,
            None if components is not None else "input missing"))  # fmt: skip

    def table_dynamic(self, instability: pd.DataFrame | None, corr: pd.DataFrame | None, common: str) -> None:
        has = instability is not None and "macro_error_dynamic" in instability
        if instability is not None and not has:
            self.note("table macro_error_dynamic", "instability.csv has no column macro_error_dynamic")
        frame = instability[[k for k in ("corridor", "law", "unstable_eq", "not_stable", "runs", "macro_error_dynamic",
                                          "macro_error_dynamic_low", "macro_error_dynamic_high", "collides")
                             if k in instability]] if has else pd.DataFrame()  # fmt: skip
        self.add(ReportTable("macro_error_dynamic", "Corridor: dynamic macro error", (
            f"Dynamic macro error per law: mean of the absolute dynamic components (FD, wave speed, waves, wave "
            f"amplitude) with its interval ({common}); instability of the members as in table instability."
        ), frame, [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("unstable_eq", "unstable among equilibria", digits=3), Column("not_stable", "not stable", digits=3),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("collides", "collides", "flag"),
        ], "runs/_tables/m5/instability.csv", note=None if has else "input missing"))  # fmt: skip
        dynamic = None
        if corr is not None and "error" in corr:
            dynamic = corr[corr["error"].astype(str).str.contains("dynamic")]
        if corr is not None and (dynamic is None or dynamic.empty):
            self.note("table macro_error_dynamic_correlation", "instability_correlation.csv has no dynamic rows")
        title = "Corridor: correlation of instability and dynamic macro error"
        self.add(ReportTable("macro_error_dynamic_correlation", title, (
            "Spearman and Pearson correlation per corridor over the laws of the dynamic macro error with the "
            "instability of the members, percentile bootstrap over the laws "
            f"({int(self.m5_design.get('n_resamples', 1000))} resamples), permutation p-value, all laws and the laws "
            "without collisions."
        ), dynamic if dynamic is not None else pd.DataFrame(), [
            Column("corridor", "corridor", "text"), Column("measure", "instability", "text"),
            Column("laws", "laws", "text"), Column("method", "method", "text"),
            Column("n", "n", "int"), Column("estimate", "correlation", digits=3, ci=True),
            Column("n_valid", "valid resamples", "int"), Column("p_permutation", "p (permutation)", "p"),
        ], "runs/_tables/m5/instability_correlation.csv",
            note=None if dynamic is not None and len(dynamic) else "input missing"))  # fmt: skip

    # ------------------------------------------------------------------------------- M8 supplement
    M8_KNOWN: dict[str, tuple[str, str, list[Column]]] = {
        "e2_monotone": ("E2 control: monotonicity terms only (D117)", (
            "Penalty kind monotone (relu(-f_s) + relu(f_dv) + relu(f_v) at the anchored equilibria of E2, no string "
            "term) next to E1 and the chosen E2 weight, fold 0, seed 0: spacing RMSE of the test part (m, unit driver) "
            "and its change against E1 paired over the drivers, band shares of the audit, unstable among equilibria, "
            "largest measured gain."
        ), [
            Column("architecture", "architecture", "text"), Column("arm", "arm", "text"), Column("runs", "runs", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_change", "RMSE change vs E1", "pct", 1, ci=True),
            Column("rmse_change_p", "p", "p"), Column("stable", "stable"), Column("unstable", "unstable"),
            Column("outside", "outside"), Column("none", "none"), Column("not_stable", "not stable"),
            Column("unstable_eq", "unstable among equilibria"), Column("max_gain", "max gain", digits=3),
            Column("collided", "collided profiles", "int"),
        ]),
        "temporal": ("Temporal hold-out on I-80 (D120)", (
            "Laws fine-tuned on period 0 only against the same laws of D97 (fine-tuned on all periods) on periods 1 and "
            "2: macro error and dynamic macro error (unit run) and their paired difference (scenario and seed)."
        ), [
            Column("law", "law (period 0)", "text"), Column("versus", "law of D97", "text"), Column("runs", "runs", "int"),
            Column("macro_error_law", "macro error", digits=3, ci=True),
            Column("macro_error_versus", "macro error D97", digits=3, ci=True),
            Column("macro_error_difference", "difference", digits=3, ci=True),
            Column("macro_error_relative", "relative", "pct", 1, ci=True),
            Column("macro_error_dynamic_law", "dynamic", digits=3, ci=True),
            Column("macro_error_dynamic_versus", "dynamic D97", digits=3, ci=True),
            Column("collisions_per_1000_vkm_law", "collisions / 1000 veh-km", digits=3),
            Column("outcome", "period 0 only", "text"),
        ]),
        "asymmetry": ("Acceleration asymmetry and oscillation spectrum (D121)", (
            "Per corridor and law, unit run: asymmetry index (mean acceleration above 0.1 m/s^2 over the mean magnitude "
            "of the decelerations below -0.1 m/s^2, 1 s differences of the speeds), shares of the vehicle-seconds "
            "accelerating and decelerating, peak frequency and centroid in 0.002-0.05 Hz and band RMS of the Welch "
            "spectrum of the detector speeds (2 s samples, 128 s segments), signed relative errors against the ground "
            "truth of the run's scenario; ground truth per scenario."
        ), [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("scenario", "scenario", "text"),
            Column("runs", "runs", "int"), Column("asymmetry_index", "asymmetry index", digits=3, ci=True),
            Column("accelerating_share", "accelerating", digits=3, ci=True),
            Column("decelerating_share", "decelerating", digits=3, ci=True),
            Column("peak_frequency", "peak (Hz)", digits=4), Column("centroid", "centroid (Hz)", digits=4, ci=True),
            Column("band_rms", "band RMS (m/s)", digits=3, ci=True),
            Column("error_asymmetry_index", "error of the index", "pct", 1, ci=True),
            Column("error_peak_frequency", "error of the peak", "pct", 1),
            Column("error_centroid", "error of the centroid", "pct", 1, ci=True),
        ]),
        "asymmetry_contrasts": ("Asymmetry: penalised and certified laws against the free laws (D121)", (
            "Paired by scenario and seed: differences (law - free law) of the asymmetry index, of the distance of the "
            "index to the ground truth (|e_law| - |e_free|, negative: closer), of the accelerating share, of the "
            "spectral centroid and of the band RMS."
        ), [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("versus", "free law", "text"),
            Column("pairs", "pairs", "int"),
            Column("asymmetry_index_difference", "index difference", digits=3, ci=True),
            Column("index_changed", "changed", "flag"),
            Column("closer_asymmetry_index", "|error| difference", digits=3, ci=True),
            Column("index_closer", "to the truth", "text"),
            Column("accelerating_share_difference", "accelerating share", digits=3, ci=True),
            Column("centroid_difference", "centroid (Hz)", digits=4, ci=True),
            Column("band_rms_difference", "band RMS (m/s)", digits=3, ci=True),
        ]),
        "correlation_pooled": ("H12.3 over all laws, per corridor and pooled (D122)", (
            "Spearman correlation of the mean macro error of a law with the share of unstable equilibria of its members, "
            "per corridor over the laws and pooled with the corridor as a stratum (ranks within the corridor), "
            "bootstrap over the laws within the corridors, permutation p-value within the corridors; all laws with "
            "runs (D122) and the laws of the verdicts of H12.3 (D108)."
        ), [
            Column("scope", "corridor", "text"), Column("error", "error", "text"), Column("laws", "laws", "text"),
            Column("n", "n", "int"), Column("estimate", "Spearman", digits=3, ci=True),
            Column("p_permutation", "p (permutation)", "p"), Column("h12_3_rule", "rule of H12.3", "text"),
        ]),
        "correlation_pooled_laws": ("The laws of the pooled correlation (D122)", (
            "Per corridor and law the values of the pooled correlation: share of unstable equilibria among the "
            "equilibria of the members (audits; exact gain of the parameter sets for the closed-form IDM laws) and the "
            "mean macro errors over the runs."
        ), [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"),
            Column("unstable_eq", "unstable among equilibria", digits=3), Column("macro_error", "macro error", digits=3),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3),
        ]),
        "power": ("Seed-to-seed spread and power (D122)", (
            "Two laws paired by seed within every scenario: standard deviation over the seeds within a scenario "
            "(pooled over the scenarios), seeds needed to detect a 10 % difference at alpha 0.05 with power 0.8 (paired "
            "t-test), power of the design and the smallest difference it detects, and the observed difference."
        ), [
            Column("corridor", "corridor", "text"), Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("mean_a", "mean (first law)", digits=3), Column("mean_b", "mean (second law)", digits=3),
            Column("sd_a", "SD (first)", digits=4), Column("sd_b", "SD (second)", digits=4),
            Column("sd_difference", "SD of the difference", digits=4),
            Column("n_one_scenario", "seeds, one scenario", "int"),
            Column("n_per_scenario", "seeds per scenario", "int"), Column("power_design", "power of the design", digits=3),
            Column("mdd_relative", "detectable", "pct", 1),
            Column("observed", "observed difference", "pct", 1, ci=True),
        ]),
    }  # fmt: skip

    @staticmethod
    def generic_columns(frame: pd.DataFrame) -> list[Column]:
        """Every column of a CSV of another work package: text, flag, int or number (3 decimals); ``<key>_low`` and
        ``<key>_high`` next to ``<key>`` become its interval."""
        names, columns = list(frame.columns), []
        for name in names:
            base = name.rsplit("_", 1)[0]
            if name.endswith(("_low", "_high")) and base in names:
                continue
            series = frame[name]
            if pd.api.types.is_bool_dtype(series):
                kind = "flag"
            elif pd.api.types.is_integer_dtype(series):
                kind = "int"
            elif pd.api.types.is_numeric_dtype(series):
                kind = "num"
            else:
                kind = "text"
            ci = kind == "num" and f"{name}_low" in names and f"{name}_high" in names
            columns.append(Column(name, name.replace("_", " "), kind, 3, ci=ci))
        return columns

    @staticmethod
    def md_title(path: Path, name: str) -> tuple[str, str]:
        """Title and first note of the Markdown file of a table (``# title``, ``- note``)."""
        title, first = name.replace("_", " "), ""
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            title = next((line[2:].strip() for line in lines if line.startswith("# ")), title)
            first = next((line[2:].strip() for line in lines if line.startswith("- ")), "")
        return title, first

    def tables_m8(self) -> None:
        """Every table of runs/_tables/m8 (CSV and LaTeX); an expected table without its CSV gets a note."""
        folder = self.cfg.m8_dir
        found = sorted(folder.glob("*.csv")) if folder.is_dir() else []
        names = [path.stem for path in found]
        for name in self.cfg.m8_tables:
            if name not in names:
                self.note(f"table {name} (M8)", f"{self.label(folder / (name + '.csv'))} missing: no inputs yet")
        order = {name: k for k, name in enumerate(self.cfg.m8_tables)}
        # the tables of the revision (runs/_tables/m9: full-history analysis, threshold sensitivity, equilibrium
        # roots) follow those of M8 with generic columns
        folder_m9 = folder.parent / "m9"
        found_m9 = sorted(folder_m9.glob("*.csv")) if folder_m9.is_dir() else []
        for path in sorted(found, key=lambda p: (order.get(p.stem, len(order)), p.stem)) + found_m9:
            folder = path.parent
            frame = self.csv(folder, path.stem, f"table {path.stem} (M8)")
            if frame is None:
                continue
            name = path.stem if path.stem not in self.tables else f"m8_{path.stem}"
            title, first = self.md_title(path.with_suffix(".md"), path.stem)
            if path.stem in self.M8_KNOWN:
                title, caption, columns = self.M8_KNOWN[path.stem]
                if path.stem == "power" and {"law_a", "law_b"} <= set(frame.columns) and len(frame):
                    first_law, second_law = str(frame["law_a"].iloc[0]), str(frame["law_b"].iloc[0])
                    columns = [Column(c.key, c.header.replace("first law", first_law).replace("second law", second_law)
                                      .replace("first", first_law).replace("second", second_law), c.kind, c.digits, c.ci)
                               for c in columns]  # fmt: skip
                missing = [c.key for c in columns if c.key not in frame.columns]
                if missing:
                    self.note(f"table {path.stem} (M8)", f"columns {', '.join(missing)} missing in the CSV")
            else:
                caption = first if first.startswith(title) else (f"{title}. {first}" if first else title)
                columns = self.generic_columns(frame)
            self.add(ReportTable(name, title, caption, frame, columns, f"runs/_tables/{folder.name}/{path.name}",
                                 note=None if len(frame) else "no rows"))  # fmt: skip
            self.m8_names.append(name)

    def supplement(self) -> list[tuple[str, str | None, Path]]:
        """The figures of the supplement: (name, caption or None, PNG), in the order of the expected names; an
        expected figure (or pattern) without a file and a figure without caption or PDF get notes."""
        import fnmatch

        folder = self.cfg.supplement_dir
        pngs = sorted(folder.glob("*.png")) if folder.is_dir() else []
        expected = list(self.cfg.supplement_expected)
        for pattern in expected:
            if not any(fnmatch.fnmatch(png.stem, pattern) for png in pngs):
                self.note(f"figure {pattern} (M8)", f"missing in {self.label(folder)}: no inputs yet")

        def rank(png: Path) -> tuple[int, str]:
            return next((k for k, p in enumerate(expected) if fnmatch.fnmatch(png.stem, p)), len(expected)), png.stem

        out = []
        for png in sorted(pngs, key=rank):
            text = png.with_suffix(".txt")
            caption = text.read_text(encoding="utf-8").strip() if text.exists() else None
            if caption is None:
                self.note(f"figure {png.stem} (M8)", f"caption {text.name} missing")
            if not png.with_suffix(".pdf").exists():
                self.note(f"figure {png.stem} (M8)", "PDF missing")
            out.append((png.stem, caption, png))
        return out

    def section_supplement(self, figures_m8: list[tuple[str, str | None, Path]]) -> list[str]:
        lines = [
            "Tables of `runs/_tables/m8/` (CSV and LaTeX in `tables/`) and the figures of `supplement/figures/` with "
            "their captions (docs/m8_contract.md). The loophole table of the theory notes is a document "
            "(`docs/theory_notes.md`), not a generated table.", "",
        ]  # fmt: skip
        if not self.m8_names:
            lines += ["> No table of M8 yet (notes below).", ""]
        for name in self.m8_names:
            lines += self.table_md(name)
            table = self.tables[name]
            if "sentence" in table.frame and table.frame["sentence"].notna().any():
                lines += ["Sentences for the paper:", "", *(f"- {s}" for s in table.frame["sentence"].dropna()), ""]
        if not figures_m8:
            lines += ["> No figure of the supplement yet (notes below).", ""]
        for name, caption, png in figures_m8:
            link = Path(os.path.relpath(png, self.cfg.out_dir)).as_posix()
            text = caption or name.replace("_", " ")
            lines += [f"![{name}]({link})", "", f"Figure `{name}`: {text} ([PDF]({link[:-4]}.pdf)).", ""]
        return lines

    def write_tables(self) -> None:
        folder = self.cfg.out_dir / "tables"
        folder.mkdir(parents=True, exist_ok=True)
        for table in self.tables.values():
            frame = table.frame
            if frame.empty and not len(frame.columns):  # a missing input: the header of its columns
                frame = pd.DataFrame(columns=[c.key for c in table.columns])
                table.frame = frame
            frame.to_csv(folder / f"{table.name}.csv", index=False)
            (folder / f"{table.name}.tex").write_text(latex_table(table), encoding="utf-8")
            if table.note:
                self.note(f"table {table.name}", table.note)

    # ------------------------------------------------------------------------------------- figures
    def figure(self, name: str, build: Callable[[], Any]) -> None:
        """Build and save one figure; a missing input or an error of the drawing becomes a note."""
        try:
            fig = build()
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            self.figures[name] = str(exc) or type(exc).__name__
            self.note(f"figure {name}", self.figures[name])
            return
        except Exception as exc:  # an unexpected failure must not stop the report
            self.figures[name] = f"{type(exc).__name__}: {exc}"
            self.note(f"figure {name}", f"{self.figures[name]} ({traceback.format_exc(limit=2).splitlines()[-1]})")
            return
        figures.save(fig, self.cfg.out_dir / "figures", name, self.cfg.dpi)
        self.figures[name] = None

    def run_dir(self, experiment: str, data: str, model: str, fold: int = 0, seed: int = 0) -> Path:
        return self.cfg.runs_root / experiment / data / model / f"driver_fold{fold}_seed{seed}"

    def chosen_experiment(self, arch: str, prefix: str = "e2") -> str | None:
        entry = self.chosen.get(arch) or {}
        weight = entry.get("override", entry.get("weight"))
        if weight is None or not entry.get("kind"):
            return None
        return experiment_name(prefix, entry["kind"], float(weight))

    def fig_rmse_vs_instability(self) -> Any:
        e1 = self.tables.get("e1")
        if e1 is None or e1.frame.empty:
            raise ValueError("table e1 missing")
        return figures.rmse_vs_instability(e1.frame, self.m4_design.get("reference", "idm"))

    def fig_e2_tradeoff(self) -> Any:
        sweep = self.tables.get("e2_sweep")
        if sweep is None or sweep.frame.empty:
            raise ValueError("table e2_sweep missing")
        maker, points = self.maker, {}
        archs = list(dict.fromkeys(sweep.frame["architecture"]))
        for arch in archs:  # E1 of seed 0, as the sweep: mean over the folds of the run summaries
            runs = maker.runs("figure", "e1", maker.cfg.data, arch, maker.cfg.seeds[:1], (METRICS, AUDIT))
            points[arch] = (maker.values("figure", runs, "test_rmse_s_mean").mean(),
                            maker.values("figure", runs, "band_numerical_stable").mean())  # fmt: skip
        for line in [line for line in maker.missing if line.startswith("[figure]")]:
            self.note("figure e2_tradeoff", f"E1 of seed 0, {line[len('[figure] '):]}")
        maker.missing = [line for line in maker.missing if not line.startswith("[figure]")]
        return figures.e2_tradeoff(sweep.frame, points, archs)

    def gains(self, run: Path) -> tuple[list[float], list[list[float]]] | None:
        path = run / "stability.json"
        if not path.exists():
            return None
        try:
            audit = read_json(path).get("audit") or {}
        except (OSError, ValueError):
            return None
        omega = audit.get("omega") or []
        curves = [r["gain"] for r in audit.get("equilibria") or []
                  if r.get("in_support") and isinstance(r.get("gain"), list)
                  and len(r["gain"]) == len(omega)]  # fmt: skip
        return omega, curves

    def fig_gain_curves(self) -> Any:
        data = self.maker.cfg.data
        curves: dict[str, dict[str, Any]] = {}
        for arch in self.cfg.gain_architectures:
            without = self.gains(self.run_dir("e1", data, arch))
            experiment = self.chosen_experiment(arch)
            with_ = self.gains(self.run_dir(experiment, data, arch)) if experiment else None
            if without is None:
                self.note("figure gain_curves", f"{self.label(self.run_dir('e1', data, arch))}/stability.json missing")
            if experiment and with_ is None:
                missing = self.label(self.run_dir(experiment, data, arch))
                self.note("figure gain_curves", f"{missing}/stability.json missing")
            omega = (without or with_ or ([], []))[0]
            if omega:
                curves[arch] = {"omega": omega, "without": (without or (0, []))[1], "with": (with_ or (0, []))[1]}
        return figures.gain_curves(curves, self.cfg.gain_threshold)

    def platoon_runs(self, experiment: str, view: str, model: str) -> list[dict[str, Any]]:
        out = []
        for fold in self.maker.cfg.folds:
            path = self.run_dir(experiment, view, model, fold) / "platoon.json"
            try:
                profiles = (read_json(path).get("profiles") or {}) if path.exists() else None
            except (OSError, ValueError):
                profiles = None
            if profiles is None:
                self.note("figure growth_curves", f"{self.label(path)} missing")
            else:
                out.append(profiles)
        return out

    def fig_growth_curves(self) -> Any:
        c = self.cfg
        laws: dict[str, list[tuple[str, str, bool, list[dict[str, Any]]]]] = {}
        for view in c.e5_views:
            laws[view] = [(figures.name(m), m, False, self.platoon_runs("e5", view, m)) for m in c.e5_models]
            for m in c.e5_penalised:
                experiment = self.chosen_experiment(m, "e5")
                if experiment is None:
                    self.note("figure growth_curves", f"{m}: no chosen weight, no penalised E5 law")
                    continue
                laws[view].append((f"{figures.name(m)} + penalty", m, True, self.platoon_runs(experiment, view, m)))
        names: dict[str, dict[str, Any]] = {}
        for view_laws in laws.values():
            for _, _, _, runs in view_laws:
                for run in runs:
                    for profile, values in run.items():
                        names.setdefault(profile, values)
        if not names:
            raise ValueError("no platoon.json of E5")
        modes = {mode: view for view, mode in c.e5_views.items()}
        panels = []
        self.platoons = [0, 0]
        order = sorted(names.items(), key=lambda kv: (kv[0] == "pulse", kv[1].get("acc_flag") != 0, kv[0]))
        for profile, first in order:
            if profile == "pulse":
                view = c.pulse_view
                label = profile_label(profile, f"{c.e5_views.get(view, view)} laws".replace("acc", "ACC"))
            else:
                mode = "human" if first.get("acc_flag") == 0 else "acc"
                view = modes.get(mode, next(iter(c.e5_views)))
                label = profile_label(profile, mode.replace("acc", "ACC"))
            models = {}
            for law_label, model, penalised, runs in laws.get(view, []):
                found = [run[profile] for run in runs if profile in run and run[profile].get("speed_std")]
                self.platoons[0] += sum(_flag(p.get("collided")) for p in found)
                self.platoons[1] += len(found)
                stds = [std_before_collision(p) for p in found]
                if stds and len({len(s) for s in stds}) == 1:  # the mean ends where the platoon of a fold collided
                    models[law_label] = {"model": model, "penalised": penalised, "std": np.vstack(stds).mean(axis=0)}
            empirical = first.get("empirical_std")
            empirical = None if empirical is None else [np.nan if v is None else v for v in empirical]
            panels.append({"label": label, "models": models, "empirical": empirical})
        return figures.growth_curves(panels)

    def corridor_inputs(self) -> list[dict[str, Any]]:
        """Trajectories of the ground truth and of the laws of the figures (prepared as for the metrics)."""
        from cf_stability.corridor.macro import (
            Geometry, MacroConfig, geometry_from_scenario, load_npz, prepare_trajectories,
        )  # fmt: skip

        c = self.cfg
        scenario_dir = c.corridor_root / "scenarios" / c.scenario
        scenario_json = scenario_dir / "scenario.json"
        geometry = Geometry()
        if scenario_json.exists():
            geometry = geometry_from_scenario(read_json(scenario_json), geometry)
        else:
            self.note("corridor figures", f"{self.label(scenario_json)} missing: default geometry and window")
        items = [("ground truth", "ground truth", scenario_dir / "ground_truth.npz",
                  scenario_dir / "vehicles_truth.npz")]  # fmt: skip
        for law in c.figure_laws:
            run = c.corridor_root / c.scenario / law / f"seed{c.figure_seed}"
            items.append((law, law, run / "trajectories.npz", run / "vehicles.npz"))
        out = []
        for label, key, trajectories, vehicles in items:
            if not trajectories.exists():
                self.note("corridor figures", f"{self.label(trajectories)} missing")
                continue
            try:
                arrays = load_npz(trajectories)
                fleet = load_npz(vehicles) if vehicles.exists() else None
                prepared = prepare_trajectories(arrays, fleet, geometry, MacroConfig())
            except (OSError, ValueError, KeyError) as exc:
                self.note("corridor figures", f"{self.label(trajectories)} unreadable ({type(exc).__name__})")
                continue
            out.append({"label": label, "key": key, "geometry": geometry, "prepared": prepared})
        return out

    def fig_speed_contours(self, inputs: list[dict[str, Any]]) -> Any:
        from cf_stability.corridor.macro import edie_grid, interval_edges
        from cf_stability.corridor.waves import speed_field

        if not inputs:
            raise ValueError("no trajectories of the corridor")
        c = self.cfg
        t_end = max(float(np.nanmax(i["prepared"]["t"])) for i in inputs if len(i["prepared"]["t"]))
        fields = []
        for item in inputs:
            p, g = item["prepared"], item["geometry"]
            grid = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                             interval_edges(g.x_in, g.x_out, c.contour_dx), interval_edges(0.0, t_end, c.contour_dt),
                             lanes=c.contour_lanes)  # fmt: skip
            speed = speed_field(grid["total"]["distance"], grid["total"]["time"], smooth=0)
            empty = [int(lane) for i, lane in enumerate(grid["lanes"]) if grid["per_lane"]["time"][i].sum() == 0]
            if empty:
                self.note("figure corridor_speed_contours", f"{item['label']}: no vehicle in lane(s) {empty}")
            fields.append({"label": item["label"], "speed": speed, "x_edges": grid["x_edges"],
                           "t_edges": grid["t_edges"]})  # fmt: skip
        truth = next((f["speed"] for f in fields if f["label"] == "ground truth"), fields[0]["speed"])
        vmax = float(np.nanpercentile(truth, 99)) if np.isfinite(truth).any() else 30.0
        return figures.speed_contours(fields, inputs[0]["geometry"].window, math.ceil(vmax))

    def fig_fundamental_diagrams(self, inputs: list[dict[str, Any]]) -> Any:
        from cf_stability.corridor.macro import edie_grid, fundamental_diagram, interval_edges, section_edges

        if not inputs:
            raise ValueError("no trajectories of the corridor")
        c = self.cfg
        panels, truth_curve = [], None
        for item in inputs:
            p, g = item["prepared"], item["geometry"]
            w0, w1 = (float(w) for w in g.window)
            grid = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                             section_edges(g.x_in, g.x_out, c.fd_dx), interval_edges(w0, w1, c.fd_dt))  # fmt: skip
            flow, density = grid["total"]["flow"].ravel() * 3600.0, grid["total"]["density"].ravel() * 1000.0
            keep = np.isfinite(flow) & np.isfinite(density) & (grid["total"]["time"].ravel() > 0)
            panels.append({"label": item["label"], "key": item["key"], "density": density[keep], "flow": flow[keep]})
            if item["key"] == "ground truth":
                fd = fundamental_diagram(flow[keep], density[keep], c.fd_bin)
                truth_curve = (fd["density_bins"], fd["flow"])
        if truth_curve is None:
            self.note("figure fundamental_diagrams", "no ground truth: no binned curve")
        return figures.fundamental_diagrams(panels, truth_curve, c.fd_bin)

    def fig_macro_error_vs_instability(self) -> Any:
        instability = self.tables.get("instability")
        corr = self.tables.get("instability_correlation")
        if instability is None or instability.frame.empty:
            raise ValueError("table instability missing")
        frame = instability.frame
        if "in_correlation" in frame:  # the laws of the correlation (not the temporal hold-out of D120)
            frame = frame[~frame["in_correlation"].map(lambda v: str(v).lower() == "false")]
        corridors = self.corridors(frame)
        corridor = corridors[0] if corridors else None  # one point per law: the first corridor of the table
        if corridor is not None:
            frame = frame[frame["corridor"] == corridor]
        panels = []
        errors = (("macro_error", "macro error (relative)", "macro error"),
                  ("macro_error_dynamic", "dynamic macro error (relative)", "macro error (dynamic)"))  # fmt: skip
        for key, ylabel, error in errors:
            if key not in frame:
                panels.append({"frame": pd.DataFrame(columns=["law", "x", "y"]), "ylabel": ylabel,
                               "empty": f"no column {key}"})  # fmt: skip
                continue
            part = frame[frame["unstable_eq"].notna() & frame[key].notna()]
            points = pd.DataFrame({"law": part["law"], "x": part["unstable_eq"], "y": part[key],
                                   "y_low": part.get(f"{key}_low"), "y_high": part.get(f"{key}_high"),
                                   "collides": part.get("collides")})  # fmt: skip
            texts = []
            if corr is not None and not corr.frame.empty:
                rows = corr.frame
                if "error" in rows:
                    rows = rows[rows["error"] == error]
                if corridor is not None and "corridor" in rows:
                    rows = rows[rows["corridor"] == corridor]
                rows = rows[(rows["measure"] == "unstable among equilibria") & (rows["laws"] == "all laws")]
                for _, r in rows.iterrows():
                    symbol = "Spearman ρ" if r["method"] == "spearman" else "Pearson r"
                    texts.append(f"{symbol} = {r['estimate']:.2f} [{r['estimate_low']:.2f}, {r['estimate_high']:.2f}]")
            panels.append({"frame": points.reset_index(drop=True), "ylabel": ylabel, "correlations": texts})
        return figures.macro_error_vs_instability(panels)

    def make_figures(self) -> None:
        figures.apply_style(self.cfg.font_size)
        self.figure("rmse_vs_instability", self.fig_rmse_vs_instability)
        self.figure("e2_tradeoff", self.fig_e2_tradeoff)
        self.figure("gain_curves", self.fig_gain_curves)
        self.figure("growth_curves", self.fig_growth_curves)
        inputs: list[dict[str, Any]] = []
        try:
            inputs = self.corridor_inputs()
        except Exception as exc:  # the corridor figures are left out, the others stay
            self.note("corridor figures", f"{type(exc).__name__}: {exc}")
        self.figure("corridor_speed_contours", lambda: self.fig_speed_contours(inputs))
        self.figure("fundamental_diagrams", lambda: self.fig_fundamental_diagrams(inputs))
        self.figure("macro_error_vs_instability", self.fig_macro_error_vs_instability)

    # ------------------------------------------------------------------------------------ manifest
    def scan_runs(self) -> dict[str, dict[str, Any]]:
        """Run directories ``<experiment>/<data>/<model>/<split>_fold<k>_seed<s>`` under ``runs_root``."""
        out: dict[str, dict[str, Any]] = {}
        root = self.cfg.runs_root
        for experiment in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            if experiment.name.startswith("_") or experiment.resolve() == self.cfg.corridor_root.resolve():
                continue
            runs = [r for r in experiment.glob("*/*/*") if r.is_dir() and RUN_NAME.fullmatch(r.name)
                    and not r.parent.parent.name.startswith("_")]  # fmt: skip
            if not runs:
                continue
            hashes, trained = [], 0
            for run in runs:
                metrics = run / "metrics.json"
                if metrics.exists():
                    trained += 1
                    try:
                        hashes.append(str(read_json(metrics).get("config_hash")))
                    except (OSError, ValueError):
                        hashes.append("unreadable")
            names = [RUN_NAME.fullmatch(r.name) for r in runs]
            out[experiment.name] = {
                "runs": len(runs), "trained": trained,
                "event_sets": sorted({r.parent.parent.name for r in runs}),
                "models": sorted({r.parent.name for r in runs}),
                "folds": sorted({int(m["fold"]) for m in names}), "seeds": sorted({int(m["seed"]) for m in names}),
                "config_hashes": len(set(hashes)), "config_fingerprint": config_hash(sorted(hashes)),
            }  # fmt: skip
        return out

    def scan_corridor(self) -> dict[str, Any]:
        root = self.cfg.corridor_root
        out: dict[str, Any] = {"runs": 0, "with_run_json": 0, "with_macro_json": 0, "macro_config_hashes": [],
                               "scenarios": {}}  # fmt: skip
        if not root.is_dir():
            return out
        hashes: set[str] = set()
        for scenario in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("scenarios", "laws")
                               and not p.name.startswith("_")):  # fmt: skip
            seeds = [s for s in scenario.glob("*/seed*") if s.is_dir()]
            counts = {"runs": len(seeds), "laws": sorted({s.parent.name for s in seeds}),
                      "with_run_json": sum((s / "run.json").exists() for s in seeds),
                      "with_macro_json": sum((s / "macro.json").exists() for s in seeds)}  # fmt: skip
            for s in seeds:
                if (s / "macro.json").exists():
                    try:
                        hashes.add(str(read_json(s / "macro.json").get("config_hash")))
                    except (OSError, ValueError):
                        hashes.add("unreadable")
            truth = root / "scenarios" / scenario.name / "scenario.json"
            if truth.exists():
                try:
                    counts["scenario_config_hash"] = read_json(truth).get("config_hash")
                except (OSError, ValueError):
                    pass
            out["scenarios"][scenario.name] = counts
            for key in ("runs", "with_run_json", "with_macro_json"):
                out[key] += counts[key]
        out["macro_config_hashes"] = sorted(hashes)
        return out

    def software(self) -> dict[str, str | None]:
        versions: dict[str, str | None] = {"python": platform.python_version()}
        for package in SOFTWARE:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return versions

    def manifest(
        self, runs: Mapping[str, Any], corridor: Mapping[str, Any], config: Mapping[str, Any],
        figures_m8: list[tuple[str, str | None, Path]] | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        inputs = {}
        for folder in (self.cfg.tables_m4, self.cfg.tables_m5, self.cfg.m8_dir):
            for path in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.json")) if folder.is_dir() else []:
                inputs[self.label(path)] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for _, _, png in figures_m8 or []:
            inputs[self.label(png)] = {"sha256": _sha256(png), "bytes": png.stat().st_size}
        for path in (self.cfg.docs_dir / "decisions.md", self.cfg.configs_dir / "make_tables.yaml",
                     self.cfg.configs_dir / "corridor_metrics.yaml"):  # fmt: skip
            if path.exists():
                inputs[self.label(path)] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        return {
            "date": datetime.now().isoformat(timespec="seconds"),
            "config": config, "config_hash": config_hash(config),
            "runs": dict(runs), "corridor": dict(corridor), "inputs": inputs, "software": self.software(),
            "outputs": {"tables": sorted(self.tables), "figures": {k: v or "written" for k, v in self.figures.items()},
                        "supplement": {"tables": list(self.m8_names),
                                       "figures": [name for name, _, _ in figures_m8 or []]}},  # fmt: skip
            "notes": self.notes,
        }

    # ------------------------------------------------------------------------------------ report.md
    def verdict_lines(self, hypothesis: str, table: str = "verdicts") -> list[str]:
        verdicts = self.tables.get(table)
        if verdicts is None or verdicts.frame.empty or "hypothesis" not in verdicts.frame:
            return [f"- {hypothesis}: no verdict (table {table} missing)."]
        rows = verdicts.frame[verdicts.frame["hypothesis"] == hypothesis]
        lines = []
        for _, row in rows.iterrows():
            complete = "" if _flag(row.get("complete")) else " (incomplete)"
            verdict = row.get("verdict") if isinstance(row.get("verdict"), str) else "n/a"
            basis = row.get("basis") if isinstance(row.get("basis"), str) else ""
            where = f" ({row['corridors']})" if isinstance(row.get("corridors"), str) else ""
            lines.append(f"- {hypothesis}, {row.get('unit', '')}{where}: **{verdict}**{complete}; {basis}.")
        return lines or [f"- {hypothesis}: no verdict in the table."]

    def sensitivity_lines(self) -> list[str]:
        """Whether the ranking of the laws by macro error survives every variant of table sensitivity (D113)."""
        table = self.tables.get("sensitivity")
        order = " < ".join(self.m5_design.get("sensitivity_order") or ["residual_idm_certified", "idm_global", "gru"])
        if table is None or table.frame.empty or "order_holds" not in table.frame:
            return [f"Ranking {order}: no sensitivity table."]
        per_variant = table.frame.groupby("variant", sort=False)["order_holds"].first()
        rankings = table.frame.groupby("variant", sort=False)["ranking"].first()
        decided = {v: _flag(h) for v, h in per_variant.items() if isinstance(rankings.get(v), str) and rankings[v]}
        holds = [v for v, h in decided.items() if h]
        pending = [v for v in per_variant.index if v not in decided]
        text = (f"Ranking {order} by macro error: it holds in {len(holds)} of the {len(decided)} variants with runs of "
                f"every law ({', '.join(f'{v}: {rankings[v]}' for v in decided) or 'none'}).")  # fmt: skip
        if pending:
            text += f" Not yet decidable (no runs of every law): {', '.join(pending)}."
        if decided and not pending:
            text += f" The ranking {'survives' if len(holds) == len(decided) else 'does not survive'} every variant."
        return [text]

    def table_md(self, name: str) -> list[str]:
        table = self.tables.get(name)
        if table is None:
            return [f"> Table {name}: not built."]
        lines = [markdown_table(table).rstrip(), ""]
        if table.note:
            lines.insert(0, f"> Table {name}: {table.note} (empty cells).")
        lines += [f"CSV: [tables/{name}.csv](tables/{name}.csv), LaTeX: [tables/{name}.tex](tables/{name}.tex).", ""]
        return lines

    def figure_md(self, name: str, caption: str) -> list[str]:
        reason = self.figures.get(name, "not built")
        if reason is None:
            return [f"![{caption}](figures/{name}.png)", "",
                    f"Figure `{name}`: {caption} ([PDF](figures/{name}.pdf))."]  # fmt: skip
        return [f"> Figure `{name}` missing: {reason}."]

    def section_data(self, runs: Mapping[str, Any], corridor: Mapping[str, Any]) -> list[str]:
        lines = ["Run directories found under the runs root "
                 "(`<experiment>/<event set>/<model>/<split>_fold<k>_seed<s>`; "
                 "trained: with metrics.json; configs: hash of the sorted config hashes of their metrics.json, as in "
                 "manifest.json).", "",
                 "| experiment | event sets | models | folds | seeds | runs | trained | configs |",
                 "|---|---|---|---|---|---:|---:|---|"]  # fmt: skip
        for experiment, info in runs.items():
            lines.append(
                f"| {experiment} | {', '.join(info['event_sets'])} | {', '.join(info['models'])} | "
                f"{', '.join(map(str, info['folds']))} | {', '.join(map(str, info['seeds']))} | {info['runs']} | "
                f"{info['trained']} | `{info['config_fingerprint']}` |"
            )
        total = sum(info["runs"] for info in runs.values())
        trained = sum(info["trained"] for info in runs.values())
        lines += ["", f"In all {total} run directories in {len(runs)} experiments, {trained} with metrics.json.", ""]
        lines += ["Corridor runs (`<scenario>/<law>/seed<s>`):", "",
                  "| scenario | laws | runs | run.json | macro.json |",
                  "|---|---:|---:|---:|---:|"]  # fmt: skip
        for scenario, info in (corridor.get("scenarios") or {}).items():
            lines.append(f"| {scenario} | {len(info['laws'])} | {info['runs']} | {info['with_run_json']} | "
                         f"{info['with_macro_json']} |")  # fmt: skip
        hashes = corridor.get("macro_config_hashes") or []
        lines += ["", f"Config hashes of the macro.json files: {', '.join(hashes) if hashes else 'none'}."]
        return lines

    def section_deviations(self) -> list[str]:
        entries = decision_entries(self.cfg.docs_dir / "decisions.md", self.cfg.docs_dir)
        if not entries:
            self.note("section 9", f"{self.label(self.cfg.docs_dir / 'decisions.md')} missing or without D-entries")
            return ["> docs/decisions.md missing or without D-entries."]
        lines = [f"{len(entries)} D-entries of `docs/decisions.md`, the first sentence of each decision, by milestone: "
                 "the one of its section in the log, or a later one when the entry's number is at or above the first "
                 "new entry that a later milestone report cites (the entries are numbered in time)."]  # fmt: skip
        label = None
        for entry in entries:
            if entry["label"] != label:
                label = entry["label"]
                lines += ["", f"### {label}", ""]
            lines.append(f"- {entry['id']} ({entry['date']}): {entry['decision']}")
        return lines

    def section_commands(self) -> list[str]:
        blocks = milestone_commands(self.cfg.docs_dir)
        if not blocks:
            self.note("section 10", "no milestone report with a section 'Commands run'")
        lines = ["The commands of every milestone in order, from the section \"Commands run\" of its report; then "
                 "this report."]  # fmt: skip
        for milestone, language, commands in blocks:
            lines += ["", f"### {milestone}", "", f"```{language}", *commands, "```"]
        lines += ["", "### This report", "", "```bash",
                  "python scripts/corridor_asymmetry.py workers=4   # asymmetry.json of every corridor run and ground truth (M8)",
                  "python scripts/make_report.py refresh=true       # first rerun the table scripts (runs/_tables/m4, m5, m8)",
                  "python scripts/make_report.py                    # runs/_report/ from the existing tables",
                  "```"]  # fmt: skip
        return lines

    def report_md(
        self, runs: Mapping[str, Any], corridor: Mapping[str, Any],
        figures_m8: list[tuple[str, str | None, Path]] | None = None,
    ) -> str:  # fmt: skip
        weights = {arch: e.get("override", e.get("weight")) for arch, e in self.chosen.items()}
        chosen = ", ".join(f"{arch} {self.chosen[arch].get('kind')} {w:g}" for arch, w in weights.items()
                           if isinstance(w, (int, float))) or "none"  # fmt: skip
        tost = self.tables.get("tost")
        equivalent = "no TOST table"
        if tost is not None and "equivalent" in tost.frame and len(tost.frame):
            parts = []
            groups = tost.frame.groupby("corridor", sort=False) if "corridor" in tost.frame else [("", tost.frame)]
            for name, part in groups:
                parts.append(f"{name + ': ' if name else ''}{int(part['equivalent'].map(_flag).sum())} of {len(part)} "
                             "metrics equivalent")  # fmt: skip
            equivalent = "; ".join(parts)
        first = (self.corridors(self.tables["instability"].frame) or [""])[0] if "instability" in self.tables else ""
        out = [
            "# Report: string stability of learned car-following models", "",
            f"Generated by `python scripts/make_report.py` on {datetime.now().isoformat(timespec='seconds')}. Tables "
            "(CSV, LaTeX) in `tables/`, figures (PNG, PDF) in `figures/`, inputs, hashes and versions in "
            "`manifest.json`. Every number below is read from a table or from the run files.", "",
            f"## {SECTIONS[0]}", "", *self.section_data(runs, corridor), "",
            f"## {SECTIONS[1]}", "", "Hypotheses of the experiments (M4; the low-frequency arm of E2 and H1.3 on the "
            "collision-free prefix of M7):", "", *self.table_md("verdicts"),
            "Hypotheses of the corridor (H12.1-H12.3, D108):", "", *self.table_md("verdicts_h12"), "",
            f"## {SECTIONS[2]}", "", *self.verdict_lines("H1.1"), "", *self.table_md("e1"),
            *self.figure_md("rmse_vs_instability", "E1: spacing RMSE against the share of unstable equilibria, and the "
                            "band shares per model"), "",  # fmt: skip
            f"## {SECTIONS[3]}", "", f"Chosen weights (D85, chosen_weights.json): {chosen}.", "",
            *self.table_md("e2_sweep"), *self.figure_md("e2_tradeoff", "E2 sweep: test RMSE against the share stable "
                                                         "inside the band, per architecture"), "",  # fmt: skip
            *self.verdict_lines("H1.2"), "", *self.table_md("e2"),
            *self.figure_md("gain_curves", "|G(ω)| of the frequency response of the fold-0 runs, E1 and the chosen E2 "
                            "weight"), "",  # fmt: skip
            "The existence arm (D93): the existence term alone, without stability terms.", "",
            *self.table_md("e2_existence"), "",
            "The low-frequency arm (D110): the rollout gain penalty with the Jacobian penalty of the memoryless view "
            "and the needle guard for the recurrent models; the verdict of H1.2 above stays.", "",
            *self.verdict_lines("H1.2 (combined)"), "", *self.table_md("e2_lowfreq"), "",
            f"## {SECTIONS[4]}", "", *self.table_md("e3"), "",
            f"## {SECTIONS[5]}", "", *self.verdict_lines("H1.5"), "", *self.table_md("e4"),
            "The residual amplitude of the certified hybrid (D111): certificate, accuracy and corridor.", "",
            *self.table_md("e4_rmax"), "",
            f"## {SECTIONS[6]}", "", *self.verdict_lines("H1.3"), "", *self.table_md("e5"),
            *self.figure_md("growth_curves", "platoon test: speed standard deviation per position, data and E5 laws of "
                            "the matching driving mode, mean over the folds; a model curve ends at the first position "
                            "whose gap reached zero in the platoon of a fold "
                            f"({self.platoons[0]} of {self.platoons[1]} simulated platoons collided)"), "",  # fmt: skip
            f"## {SECTIONS[7]}", "", *self.table_md("laws"), *self.table_md("m5_components"),
            *self.table_md("macro_error_dynamic"), *self.table_md("laws_raw"),
            *self.figure_md("corridor_speed_contours", f"speed fields of {self.cfg.scenario}, ground truth and laws "
                            f"(seed {self.cfg.figure_seed})"), "",  # fmt: skip
            *self.figure_md("fundamental_diagrams", f"fundamental diagrams of {self.cfg.scenario}"), "",
            *self.table_md("instability"), *self.figure_md("macro_error_vs_instability", "macro error against the "
                                                          "instability of the members, per law"
                                                          + (f" ({first})" if first else "")), "",  # fmt: skip
            *self.table_md("instability_correlation"), *self.table_md("macro_error_dynamic_correlation"),
            f"TOST: {equivalent}.", "", *self.table_md("tost"),
            "Verdicts of the corridor hypotheses (D108):", "", *self.verdict_lines("H12.1", "verdicts_h12"),
            *self.verdict_lines("H12.2", "verdicts_h12"), *self.verdict_lines("H12.3", "verdicts_h12"), "",
            *self.table_md("h12_1"), *self.table_md("h12_2"),
            "Sensitivity to the downstream boundary and the lane-change model (D113).", "",
            *self.sensitivity_lines(), "", *self.table_md("sensitivity"), "",
            f"## {SECTIONS[8]}", "", *self.section_deviations(), "",
            f"## {SECTIONS[9]}", "", *self.section_commands(), "",
            f"## {SECTIONS[10]}", "", *self.section_supplement(figures_m8 or []), "",
            "## Notes on missing inputs", "",
            *([f"- {note}" for note in self.notes] or ["None."]),
        ]
        return "\n".join(out) + "\n"


def make_report(cfg: ReportConfig, config: Mapping[str, Any] | None = None) -> list[str]:
    """Write the tables, figures, report.md and manifest.json; one printed line per output group."""
    maker = ReportMaker(cfg)
    if cfg.refresh:
        maker.refresh()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    maker.tables_m4()
    maker.table_existence()
    maker.tables_m5()
    maker.tables_m8()
    maker.write_tables()
    maker.make_figures()
    figures_m8 = maker.supplement()
    runs, corridor = maker.scan_runs(), maker.scan_corridor()
    text = maker.report_md(runs, corridor, figures_m8)  # adds the notes of sections 9 and 10 before the manifest
    (cfg.out_dir / "report.md").write_text(text, encoding="utf-8")
    write_json(cfg.out_dir / "manifest.json", maker.manifest(runs, corridor, dict(config or {}), figures_m8))
    written = [name for name, reason in maker.figures.items() if reason is None]
    missing = [name for name in FIGURES if name not in written]
    lines = [*maker.refreshed,
             f"TABLES {len(maker.tables)} (CSV, LaTeX) -> {cfg.out_dir / 'tables'}",
             f"FIGURES {len(written)} of {len(FIGURES)} (PNG, PDF) -> {cfg.out_dir / 'figures'}"
             + (f"; missing: {', '.join(missing)}" if missing else ""),
             f"REPORT {cfg.out_dir / 'report.md'}: {len(maker.notes)} notes on missing inputs",
             f"SUPPLEMENT (M8) {len(maker.m8_names)} tables of {maker.label(cfg.m8_dir)}, {len(figures_m8)} figures of "
             f"{maker.label(cfg.supplement_dir)}"]  # fmt: skip
    return lines
