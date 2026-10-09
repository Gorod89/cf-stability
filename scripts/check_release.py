"""Completeness of a checkout for level 2 of the reproduction (README.md, section Reproduce).

    python scripts/check_release.py          # exit status 1 when something is missing
    python scripts/check_release.py --all    # list every missing file (default: the first 10 of a check)

Level 1 reads the published tables, figures and report; level 2 recomputes them from the run outputs of the
repository. This script checks that a checkout holds both, without running anything, and prints one line per
check (OK, FAIL, WARN, or INFO for what is only reported) with the missing files under a failed check:

* docs/paper_materials.md: every file of the repository that it names as the source of a table or figure of the
  manuscript (column "built from"), the scripts of its commands, and the numbering of its items without gaps;
* runs/_report: report.md, manifest.json and every table (CSV, LaTeX), figure (PNG, PDF) and supplementary
  figure (PNG, PDF, caption) that the manifest of the report lists; runs/_tables/m4, m5, m8, m9 present;
* runs/corridor/laws: the law file of every law of configs/corridor_metrics.yaml (design: laws, rmax_laws,
  temporal_laws, ablation_laws) and every file a law file names (the npz tables of the heterogeneous laws, the
  member runs and their models, the calibrations);
* runs/corridor/scenarios: scenario.json, ground_truth.npz, macro.json and asymmetry.json of every scenario of
  the manifest of the report;
* data/calibration and data/splits: present, every JSON readable; configs: every YAML readable;
* the run outputs, counted against the manifest of the report: every trained run (model.pt) with metrics.json
  and test_events.parquet; per scenario the corridor runs with run.json, macro.json and asymmetry.json; the
  audits and the other per-run files against their counts in the release of 8 October 2026 (RELEASE); the count
  of trajectories.npz (level 3) is reported only;
* the inputs of the corridor figures (figures 5 and 6 of the report, Figures S10-S21 of the supplement): every
  corridor run of the scenarios of the report holds a fields.npz (cf_stability/corridor/fields.py,
  scripts/export_fields.py) of the current format, written for this run with the grid settings of the current
  configs, with the grids the figures draw (the diagram cells of every run, the speed field of the figure seeds),
  or else its trajectories.npz; a run with neither, or a stale or incomplete fields.npz, fails the check (the
  figure builders would stop on it in their strict mode).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # cf_stability (the grid settings of the fields), as the other scripts import it
RUNS = ROOT / "runs"
REPORT = RUNS / "_report"
MATERIALS = ROOT / "docs" / "paper_materials.md"
NOT_TRAINING = {"_superseded", "_report", "_tables", "corridor"}  # directories of runs/ without training runs
RELEASE = {  # per-run files of the release of 8 October 2026 (outside runs/_superseded): the least counts
    "stability.json": 968, "platoon.json": 536, "certificate.json": 359, "full_history.json": 207,
    "transfer_*.parquet": 255,
}
CONFIGS = ("make_tables.yaml", "corridor_metrics.yaml", "make_report.yaml", "build_corridor.yaml", "run_corridor.yaml",
           "train.yaml")  # the configurations of the commands of levels 2 and 3
LAW_SETS = ("laws", "rmax_laws", "temporal_laws", "ablation_laws")  # the laws of configs/corridor_metrics.yaml
SCENARIO_FILES = ("scenario.json", "ground_truth.npz", "macro.json", "asymmetry.json")
FIELDS, TRAJECTORIES = "fields.npz", "trajectories.npz"  # the inputs of a corridor panel (either one)


class Checks:
    """The lines of the checks and their counts."""

    def __init__(self, show_all: bool) -> None:
        self.show_all = show_all
        self.counts = {"OK": 0, "FAIL": 0, "WARN": 0, "INFO": 0}

    def line(self, status: str, name: str, detail: str, missing: list[str] | tuple = ()) -> None:
        self.counts[status] += 1
        print(f"{status:<4}  {name:<26} {detail}", flush=True)
        shown = list(missing) if self.show_all else list(missing)[:10]
        for m in shown:
            print(f"      missing: {m}")
        if len(missing) > len(shown):
            print(f"      ... and {len(missing) - len(shown)} more (--all)")

    def files(self, name: str, paths: list[str], what: str) -> list[str]:
        """OK when every path (relative to the repository root) exists, else FAIL with the missing ones."""
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            self.line("FAIL", name, f"{len(missing)} of {len(paths)} {what} missing", missing)
        else:
            self.line("OK", name, f"{len(paths)} {what}")
        return missing


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_materials(chk: Checks) -> None:
    """docs/paper_materials.md: the sources and scripts it names, and its numbering."""
    if not MATERIALS.exists():
        chk.line("FAIL", "paper materials", f"{rel(MATERIALS)} missing (python scripts/paper_assets.py --manifest)")
        return
    sources, scripts, items, bad = set(), set(), {}, []
    for line in MATERIALS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (Table|Figure) (S?)(\d+) \|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 6:
            bad.append(line[:60])
            continue
        items.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3)))
        for token in re.findall(r"`([^`]+)`", cells[4]):  # built from
            if re.match(r"(runs|docs|data)/", token) and not re.search(r"[<>*{]", token):
                sources.add(token)
        scripts |= set(re.findall(r"`python (scripts/\S+\.py)", cells[5]))  # commands
    gaps = [f"{kind} {s}{k}" for (kind, s), ns in items.items() for k in range(1, max(ns) + 1) if ns.count(k) != 1]
    n = sum(len(ns) for ns in items.values())
    numbering = ", ".join(f"{kind}s {s}1-{s}{max(ns)}" for (kind, s), ns in sorted(items.items(), key=lambda x: x[0][1]))
    if bad or gaps or not n:
        chk.line("FAIL", "paper materials", f"{rel(MATERIALS)}: {n} items; malformed rows or numbering gaps",
                 bad + [f"item {g}" for g in gaps])
    else:
        chk.line("OK", "paper materials", f"{rel(MATERIALS)}: {n} items ({numbering})")
    chk.files("paper sources", sorted(sources), "source files of the tables and figures")
    chk.files("paper commands", sorted(scripts), "scripts of the commands")


def check_report(chk: Checks) -> dict:
    """runs/_report and the table folders; returns the manifest of the report ({} when unreadable)."""
    manifest = read_json(REPORT / "manifest.json")
    if manifest is None:
        chk.line("FAIL", "report manifest", f"{rel(REPORT / 'manifest.json')} missing or unreadable")
        return {}
    out = manifest.get("outputs") or {}
    supp = (out.get("supplement") or {}).get("figures") or []
    paths = [rel(REPORT / "report.md"), rel(REPORT / "manifest.json")]
    paths += [rel(REPORT / "tables" / f"{t}.{x}") for t in out.get("tables") or [] for x in ("csv", "tex")]
    paths += [rel(REPORT / "figures" / f"{f}.{x}") for f, state in (out.get("figures") or {}).items()
              if state == "written" for x in ("png", "pdf")]
    paths += [rel(REPORT / "supplement" / "figures" / f"{f}.{x}") for f in supp for x in ("png", "pdf", "txt")]
    if not out.get("tables") or not out.get("figures") or not supp:
        chk.line("FAIL", "report", "manifest.json lists no tables, figures or supplementary figures")
    else:
        chk.files("report", paths, f"files of runs/_report ({len(out['tables'])} tables, {len(out['figures'])} "
                                   f"figures, {len(supp)} supplementary figures)")
    folders = [RUNS / "_tables" / m for m in ("m4", "m5", "m8", "m9")]
    empty = [rel(f) for f in folders if not any(f.glob("*.csv"))]
    n_csv = sum(len(list(f.glob("*.csv"))) for f in folders)
    n_md = sum(len(list(f.glob("*.md"))) for f in folders)
    if empty:
        chk.line("FAIL", "tables", "table folders missing or without CSV", empty)
    else:
        chk.line("OK", "tables", f"runs/_tables/m4, m5, m8, m9: {n_csv} CSV, {n_md} Markdown tables")
    return manifest


def law_paths(law: dict) -> set[str]:
    """Files a law file names: its npz table and the repository paths of its members and sources."""
    out: set[str] = set()

    def walk(x, key: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str):
            if key == "table" and x.endswith(".npz"):
                out.add(f"runs/corridor/laws/{x}")
            elif re.match(r"(runs|data)/", x) and not re.search(r"[<>*{]", x):  # patterns name no file
                out.add(x)

    walk(law)
    return out


def check_laws(chk: Checks) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "corridor_metrics.yaml").read_text(encoding="utf-8"))
    design = cfg.get("design") or {}
    laws = [law for key in LAW_SETS for law in design.get(key) or []]
    files = [f"runs/corridor/laws/{law}.json" for law in laws]
    missing = chk.files("law files", files, f"law files of the design ({', '.join(LAW_SETS)})")
    named: set[str] = set()
    unreadable = []
    for f in files:
        if f in missing:
            continue
        law = read_json(ROOT / f)
        if law is None:
            unreadable.append(f)
            continue
        named |= law_paths(law)
    if unreadable:
        chk.line("FAIL", "law files readable", f"{len(unreadable)} unreadable", unreadable)
    npz = sorted(p for p in named if p.endswith(".npz"))
    chk.files("law inputs", sorted(named), f"files named by the law files (npz tables: "
                                           f"{', '.join(Path(p).stem for p in npz) or 'none'}; members, calibrations)")


def check_scenarios(chk: Checks, manifest: dict) -> list[str]:
    scenarios = sorted(((manifest.get("corridor") or {}).get("scenarios") or {}))
    if not scenarios:
        chk.line("FAIL", "scenarios", "the manifest of the report lists no corridor scenario")
        return []
    base = RUNS / "corridor" / "scenarios"
    chk.files("scenarios", [rel(base / s / f) for s in scenarios for f in SCENARIO_FILES],
              f"files of {len(scenarios)} scenarios ({', '.join(SCENARIO_FILES)})")
    extra = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in scenarios) if base.exists() else []
    if extra:
        chk.line("INFO", "scenarios (other)", f"not in the manifest of the report: {', '.join(extra)}")
    return scenarios


def check_inputs(chk: Checks) -> None:
    """data/calibration, data/splits and configs."""
    for folder in ("data/calibration", "data/splits"):
        path = ROOT / folder
        jsons = sorted(path.rglob("*.json")) if path.exists() else []
        if not jsons:
            chk.line("FAIL", folder, "missing or without JSON files")
            continue
        bad = [rel(p) for p in jsons if read_json(p) is None]
        others = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix not in (".json", "") and p.name != ".gitkeep")
        if bad:
            chk.line("FAIL", folder, f"{len(bad)} of {len(jsons)} JSON files unreadable", bad)
        else:
            chk.line("OK", folder, f"{len(jsons)} JSON files" + (f", {others} other files" if others else ""))
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    bad = []
    for p in configs:
        try:
            yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            bad.append(rel(p))
    absent = [f"configs/{c}" for c in CONFIGS if not (ROOT / "configs" / c).exists()]
    if bad or absent or not configs:
        chk.line("FAIL", "configs", f"{len(configs)} YAML files; unreadable or missing", absent + bad)
    else:
        chk.line("OK", "configs", f"{len(configs)} YAML files readable")


def training_runs() -> list[Path]:
    """Directories of the trained runs (model.pt), outside the folders without training runs."""
    out = []
    for top in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name not in NOT_TRAINING):
        out += sorted(p.parent for p in top.rglob("model.pt"))
    return out


def check_runs(chk: Checks, manifest: dict, scenarios: list[str]) -> None:
    runs = training_runs()
    expected = sum(int(v.get("trained", 0)) for v in (manifest.get("runs") or {}).values())
    detail = f"{len(runs)} trained runs (model.pt), the manifest of the report counts {expected}"
    chk.line("OK" if len(runs) == expected and runs else "FAIL", "training runs", detail)
    for name in ("metrics.json", "test_events.parquet"):
        missing = [rel(r / name) for r in runs if not (r / name).exists()]
        chk.line("FAIL" if missing else "OK", name, f"{len(runs) - len(missing)} of {len(runs)} trained runs", missing)
    tops = [p for p in RUNS.iterdir() if p.is_dir() and p.name not in NOT_TRAINING]
    for pattern, least in RELEASE.items():
        n = sum(1 for top in tops for _ in top.rglob(pattern))
        status = "OK" if n >= least else "FAIL"
        lacking = [rel(r) for r in runs if not any(r.glob(pattern))] if status == "FAIL" else []
        chk.line(status, pattern, f"{n} (the release: {least})",
                 [f"{x} has none (some runs have none by design)" for x in lacking])
    corridor = manifest.get("corridor") or {}
    totals = {"run.json": 0, "macro.json": 0, "asymmetry.json": 0, TRAJECTORIES: 0}
    missing: list[str] = []
    for s in scenarios:
        expect = corridor["scenarios"][s]
        dirs = sorted(p.parent for p in (RUNS / "corridor" / s).glob("*/seed*/run.json"))
        if len(dirs) != int(expect.get("with_run_json", expect.get("runs", 0))):
            missing.append(f"runs/corridor/{s}: {len(dirs)} runs with run.json, the manifest of the report counts "
                           f"{expect.get('with_run_json', expect.get('runs'))}")
        for d in dirs:
            for name in totals:
                if (d / name).exists():
                    totals[name] += 1
                elif name in ("macro.json", "asymmetry.json"):
                    missing.append(rel(d / name))
    want = int(corridor.get("runs", 0))
    detail = (f"{totals['run.json']} runs (the manifest of the report: {want}); macro.json {totals['macro.json']}, "
              f"asymmetry.json {totals['asymmetry.json']}")
    chk.line("FAIL" if missing or totals["run.json"] != want else "OK", "corridor runs", detail, missing)
    chk.line("INFO", TRAJECTORIES, f"{totals[TRAJECTORIES]} corridor runs (not part of the release; level 3)")


def fields_status(path: Path, fmt: int, expected_hash: str, parts: tuple[str, str, int], contours: bool) -> str | None:
    """Why the fields.npz at ``path`` does not serve the figures of the run ``parts`` (scenario, law, seed): None when
    it does, else the reason (unreadable, another format, another run, other grid settings, a grid missing); the
    speed-field grid is needed only with ``contours`` (the seeds the figures draw)."""
    try:
        with np.load(path, allow_pickle=False) as data:
            header = (int(data["format"]), str(data["settings_hash"]), str(data["scenario"]), str(data["law"]),
                      int(data["seed"]))  # fmt: skip
            kinds = {str(name).split("_")[0] for name in np.atleast_1d(data["grids"])}
    except Exception as exc:  # a file that is not a fields.npz: whatever numpy or zipfile raise
        return f"unreadable ({type(exc).__name__})"
    if header[0] != fmt:
        return f"format {header[0]}, this code reads format {fmt}"
    if header[2:] != parts:
        return "belongs to the run {}/{}/seed{}".format(*header[2:])
    if header[1] != expected_hash:
        return "stale: written with other grid settings (python scripts/export_fields.py, needs trajectories.npz)"
    lacking = sorted({"fd", *(("contours",) if contours else ())} - kinds)
    if lacking:
        return f"holds no {', '.join(lacking)} grid"
    return None


def check_fields(chk: Checks, scenarios: list[str]) -> None:
    """The inputs of the corridor figures: every run of the scenarios of the report has a fields.npz that serves its
    figures (:func:`fields_status`) or its trajectories.npz (level 3)."""
    try:
        from cf_stability.corridor.fields import CONTOUR_SEEDS, FORMAT, figure_specs, settings_hash
        from cf_stability.corridor.macro import Geometry, geometry_from_scenario
    except ImportError as exc:  # the package is needed for the grid settings: pip install -e . --no-deps
        chk.line("FAIL", FIELDS, f"cf_stability does not import ({exc}): the fields cannot be checked")
        return
    try:
        report = yaml.safe_load((ROOT / "configs" / "make_report.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        report = {}
    contour_seeds = {*CONTOUR_SEEDS, int(report.get("figure_seed", 0))}  # the seeds whose speed field is drawn
    counts = {FIELDS: 0, TRAJECTORIES: 0}
    missing: list[str] = []
    for s in scenarios:
        scenario_file = RUNS / "corridor" / "scenarios" / s / "scenario.json"
        scenario_json = read_json(scenario_file)
        if scenario_json is None:
            missing.append(f"{rel(scenario_file)} missing or unreadable: the fields of {s} cannot be checked")
            continue
        geometry = geometry_from_scenario(scenario_json, Geometry())
        specs = figure_specs(s, scenario_json)
        expected = {c: settings_hash([spec for spec in specs if c or spec.kind != "contours"], geometry)
                    for c in (True, False)}  # fmt: skip
        for run_json in sorted((RUNS / "corridor" / s).glob("*/seed*/run.json")):
            run = run_json.parent
            seed = int(run.name[4:]) if run.name[4:].isdigit() else -1
            contours = seed in contour_seeds
            path = run / FIELDS
            if path.exists():
                problem = fields_status(path, FORMAT, expected[contours], (s, run.parent.name, seed), contours)
                if problem:
                    missing.append(f"{rel(path)}: {problem}")
                else:
                    counts[FIELDS] += 1
            elif (run / TRAJECTORIES).exists():
                counts[TRAJECTORIES] += 1
            else:
                missing.append(f"{rel(run)}: neither {FIELDS} nor {TRAJECTORIES} (python scripts/export_fields.py "
                               f"writes {FIELDS} where {TRAJECTORIES} exists)")
    detail = (f"{counts[FIELDS]} corridor runs with a {FIELDS} that holds the grids of the figures (the diagram cells; "
              f"the speed field for seed{'s' if len(contour_seeds) > 1 else ''} "
              f"{', '.join(str(k) for k in sorted(contour_seeds))})")
    if counts[TRAJECTORIES]:
        detail += f", {counts[TRAJECTORIES]} with {TRAJECTORIES} only (level 3)"
    chk.line("FAIL" if missing or not sum(counts.values()) else "OK", FIELDS, detail, missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="list every missing file")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # paths with characters the console cannot print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    chk = Checks(args.all)
    check_materials(chk)
    manifest = check_report(chk)
    check_laws(chk)
    scenarios = check_scenarios(chk, manifest)
    check_inputs(chk)
    check_runs(chk, manifest, scenarios)
    check_fields(chk, scenarios)
    c = chk.counts
    print(f"{sum(c.values())} checks: {c['OK']} OK, {c['FAIL']} FAIL, {c['WARN']} WARN, {c['INFO']} INFO"
          + (" -- the checkout is incomplete for level 2" if c["FAIL"] else ""))
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
