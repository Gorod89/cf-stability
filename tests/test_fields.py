"""Edie fields of the corridor figures (cf_stability/corridor/fields.py, scripts/export_fields.py) and the corridor
figures of the report and of the supplement without trajectories (cf_stability/eval/report.py, figures_supplement.py).

Checked on hand-made corridors: grids over a whole run, cut or padded to a common end, equal the grids computed to that
end bit for bit; the export and load round trip (header, arrays, fixed bytes, the contour grid only for the figure
seed); the update lines of the script (OK, KEPT, SKIPPED, FAILED); the sources of a panel and the errors that name the
run and the file; figures 5 and 6 and the supplementary grids drawn from fields.npz are byte-identical to those drawn
from trajectories.npz; strict builds raise for a panel without input, lenient ones mark the figure incomplete; the
grids of the export are those the figure builders ask for.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from cf_stability.corridor.fields import (
    FIELDS_FILE, FORMAT, GridSpec, MissingInputError, PanelSource, compute_grids, export_fields, fd_points,
    figure_specs, load_fields, panel_grids, speed_panels, update_fields,
)  # fmt: skip
from cf_stability.corridor.macro import (
    Geometry, MacroConfig, edie_grid, geometry_from_scenario, interval_edges, load_npz, prepare_trajectories,
    scenario_macro_config,
)  # fmt: skip
from cf_stability.corridor.waves import speed_field
from cf_stability.eval.figures_supplement import corridor_specs as supplement_specs
from cf_stability.eval.figures_supplement import make_supplement
from cf_stability.eval.report import FIGURES, ReportConfig, make_report
from cf_stability.eval.report import corridor_specs as report_specs
from cf_stability.utils import REPO_ROOT, read_json, write_json
from test_corridor_macro import constant_paths, sample, wave_paths
from test_figures_supplement import build_tree as build_supplement_tree
from test_figures_supplement import config as supplement_config
from test_make_report import build_tree as build_report_tree
from test_make_report import config as report_config

SCENARIO = {"scenario": "sc", "geometry": {"x_in": 20, "x_out": 500}, "window": [60, 300], "config_hash": "abc"}
GEOMETRY = geometry_from_scenario(SCENARIO, Geometry())
SPECS = (GridSpec.contours(20.0, 2.0, (1, 2, 3)), GridSpec.fd(100.0, 30.0))


def flow(speed_1: float, speed_2: float, t_end: float = 300.0) -> tuple[dict, dict]:
    return sample(constant_paths(speed_1, 2.0, 1, 0.37, t_end=t_end) + constant_paths(speed_2, 3.0, 2, 1.13, t_end=t_end))


def write_run(root: Path, law: str, seed: int, arrays: tuple[dict, dict], run_json: bool = True) -> Path:
    run = root / "sc" / law / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    np.savez(run / "trajectories.npz", **arrays[0])
    np.savez(run / "vehicles.npz", **arrays[1])
    if run_json:
        write_json(run / "run.json", {"law": law, "config_hash": f"hash-{law}-{seed}"})
    return run


@pytest.fixture()
def corridor(tmp_path) -> Path:
    """A scenario ``sc`` with a ground truth and runs of two laws (seeds 0 and 1 of law_a, seed 0 of law_b)."""
    root = tmp_path / "corridor"
    scenario = root / "scenarios" / "sc"
    scenario.mkdir(parents=True)
    truth = flow(10.0, 15.0)
    np.savez(scenario / "ground_truth.npz", **truth[0])
    np.savez(scenario / "vehicles_truth.npz", **truth[1])
    write_json(scenario / "scenario.json", SCENARIO)
    write_run(root, "law_a", 0, flow(10.0, 15.0))
    write_run(root, "law_a", 1, flow(11.0, 14.0))
    write_run(root, "law_b", 0, flow(8.0, 12.0, t_end=320.0))
    return root


def configs(tmp_path: Path, lanes: tuple[int, ...] = (1, 2, 3)) -> Path:
    """configs/ of the export: the wave lanes of the metrics and the report on scenario sc with the same cells."""
    folder = tmp_path / "configs"
    folder.mkdir(exist_ok=True)
    (folder / "corridor_metrics.yaml").write_text(yaml.safe_dump({"macro": {"waves": {"lanes": list(lanes)}}}),
                                                  encoding="utf-8")  # fmt: skip
    (folder / "make_report.yaml").write_text(yaml.safe_dump({"scenario": "sc", "contour_lanes": list(lanes)}),
                                             encoding="utf-8")  # fmt: skip
    return folder


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-identical arrays (NaN where NaN, the sign of zeros, the dtype)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True) and \
        np.array_equal(np.signbit(a), np.signbit(b))  # fmt: skip


# ---------------------------------------------------------------------------------------------- the grids


def test_whole_run_grids_cut_to_a_common_end_equal_grids_computed_to_it():
    """The figures cut every panel to the last time of any panel: the cells of a grid over the whole run, cut or
    padded with empty cells, are the sums of a grid computed to that end, bit for bit (speed and empty lanes too)."""
    runs = [sample(wave_paths([(150.0, 30.0)], t_end=200.0)), sample(wave_paths([(100.0, 50.0)], t_end=230.0)),
            flow(9.0, 13.0, t_end=210.0)]  # fmt: skip
    lanes = (1, 2, 3, 4)  # lane 4 is empty everywhere
    spec = GridSpec.contours(20.0, 2.0, lanes)
    prepared = [prepare_trajectories(t, v, GEOMETRY, MacroConfig()) for t, v in runs]
    t_end = max(float(np.nanmax(p["t"])) for p in prepared)
    grids = [compute_grids(t, v, GEOMETRY, [spec])[spec] for t, v in runs]
    assert len({g["distance"].shape[1] for g in grids}) > 1  # the runs end at different times
    t_edges, panels = speed_panels(grids, spec.dt)
    assert same(t_edges, interval_edges(0.0, t_end, 2.0))
    for p, (speed, empty), expected in zip(prepared, panels, ([4], [4], [3, 4])):  # three lanes, three, two
        old = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                        interval_edges(20.0, 500.0, 20.0), interval_edges(0.0, t_end, 2.0), lanes=lanes)  # fmt: skip
        assert same(speed, speed_field(old["total"]["distance"], old["total"]["time"], smooth=0))
        assert empty == [int(lane) for i, lane in enumerate(old["lanes"]) if old["per_lane"]["time"][i].sum() == 0]
        assert empty == expected
    with pytest.raises(ValueError, match="no vehicle"):
        speed_panels([{**grids[0], "t_max": float("nan")}], 2.0)


def test_fd_points_are_the_edie_cells_with_vehicle_time():
    trajectories, vehicles = flow(10.0, 15.0)
    spec = GridSpec.fd(100.0, 30.0)
    grid = compute_grids(trajectories, vehicles, GEOMETRY, [spec])[spec]
    density, flow_vph = fd_points(grid)
    p = prepare_trajectories(trajectories, vehicles, GEOMETRY, MacroConfig())
    from cf_stability.corridor.macro import section_edges

    old = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                    section_edges(20.0, 500.0, 100.0), interval_edges(60.0, 300.0, 30.0))  # fmt: skip
    keep = old["total"]["time"].ravel() > 0
    assert same(flow_vph, old["total"]["flow"].ravel()[keep] * 3600.0)
    assert same(density, old["total"]["density"].ravel()[keep] * 1000.0)
    assert flow_vph.mean() == pytest.approx(3000.0, rel=0.02)  # 1800 + 1200 veh/h in uniform flow


# ---------------------------------------------------------------------------------------------- the file


def test_export_and_load_round_trip(corridor):
    run = corridor / "sc" / "law_b" / "seed0"
    path = export_fields(run, SPECS, SCENARIO)
    assert path == run / FIELDS_FILE
    stored = load_fields(run)
    assert stored.header == {
        "format": FORMAT, "scenario": "sc", "law": "law_b", "seed": 0, "run_config_hash": "hash-law_b-0",
        "settings_hash": stored.header["settings_hash"], "x_in": 20.0, "x_out": 500.0, "window": (60.0, 300.0),
    }  # fmt: skip
    assert list(stored.grids) == ["contours", "fd"]
    computed = compute_grids(load_npz(run / "trajectories.npz"), load_npz(run / "vehicles.npz"), GEOMETRY, SPECS)
    for spec in SPECS:
        grid = stored.grid(spec, GEOMETRY)
        assert grid["settings"] == spec.settings(GEOMETRY)
        for key in ("x_edges", "t_edges", "lanes", "distance", "time", "occupied"):
            if key in computed[spec]:
                assert same(grid[key], computed[spec][key]), key
        assert grid["distance"].dtype == np.float64 and grid["time"].dtype == np.float64
    assert stored.grids["contours"]["t_max"] == computed[SPECS[0]]["t_max"] > 300.0
    assert stored.grid(GridSpec.contours(20.0, 2.0, (1, 2)), GEOMETRY) is None  # other lanes: another grid
    first = path.read_bytes()
    export_fields(run, SPECS, SCENARIO)
    assert path.read_bytes() == first  # fixed entry dates: the same content gives the same bytes
    with np.load(path) as data:  # a plain npz
        assert data["contours_distance"].shape == computed[SPECS[0]]["distance"].shape
    export_fields(run, SPECS, SCENARIO, contours=False)  # the seeds the figures do not draw: the diagrams only
    assert list(load_fields(path).grids) == ["fd"]


def test_update_fields_writes_keeps_and_skips(corridor, tmp_path):
    folder = configs(tmp_path)
    (corridor / "sc" / "law_c" / "seed0").mkdir(parents=True)  # a run without trajectories: skipped
    lines = list(update_fields(corridor, configs_dir=folder))
    assert [line.split(":")[0] for line in lines] == ["OK sc/law_a/seed0", "OK sc/law_a/seed1", "OK sc/law_b/seed0",
                                                      "SKIPPED sc/law_c/seed0"]  # fmt: skip
    assert "contours, fd (" in lines[0] and ": fd (" in lines[1] and "trajectories.npz, run.json missing" in lines[3]
    assert list(load_fields(corridor / "sc" / "law_a" / "seed1").grids) == ["fd"]
    assert [line.split(" ", 1)[0] for line in update_fields(corridor, configs_dir=folder)] == ["KEPT"] * 3 + ["SKIPPED"]
    os.utime(corridor / "sc" / "law_a" / "seed0" / "trajectories.npz")  # a newer input: written again
    assert [line[:2] for line in update_fields(corridor, law="law_a", configs_dir=folder)] == ["OK", "KE"]
    lines = list(update_fields(corridor, law="law_a", seed=1, contour_seeds=None, configs_dir=folder))
    assert lines[0].startswith("OK sc/law_a/seed1: contours, fd")  # every seed: another settings hash
    assert list(update_fields(corridor, law="law_b", force=True, configs_dir=folder))[0].startswith("OK ")
    (corridor / "scenarios" / "sc" / "scenario.json").unlink()
    assert list(update_fields(corridor, law="law_b", force=True, configs_dir=folder))[0].startswith(
        "FAILED sc/law_b/seed0: FileNotFoundError")  # fmt: skip


def test_panel_sources_and_their_errors(corridor):
    run = corridor / "sc" / "law_a" / "seed0"
    panel = PanelSource.run(run)
    computed = panel_grids(panel, GEOMETRY, SPECS, "trajectories")
    with pytest.raises(MissingInputError, match=r"law_a/seed0/fields.npz missing \(corridor source: fields\)"):
        panel_grids(panel, GEOMETRY, SPECS, "fields")
    export_fields(run, SPECS, SCENARIO)
    from_fields = panel_grids(panel, GEOMETRY, SPECS, "fields")
    for spec in SPECS:
        assert same(from_fields[spec]["time"], computed[spec]["time"])
    np.savez(run / "trajectories.npz", **flow(5.0, 6.0)[0])  # auto prefers the trajectories when both exist
    assert not same(panel_grids(panel, GEOMETRY, SPECS, "auto")[SPECS[0]]["time"], computed[SPECS[0]]["time"])
    (run / "trajectories.npz").unlink()
    assert same(panel_grids(panel, GEOMETRY, SPECS, "auto")[SPECS[0]]["time"], computed[SPECS[0]]["time"])
    with pytest.raises(MissingInputError, match=r"trajectories.npz missing \(corridor source: trajectories\)"):
        panel_grids(panel, GEOMETRY, SPECS, "trajectories")
    with pytest.raises(MissingInputError, match="holds no contours grid of 20 m x 2 s cells over lanes 1, 2 "):
        panel_grids(panel, GEOMETRY, [GridSpec.contours(20.0, 2.0, (1, 2))], "auto")
    with pytest.raises(MissingInputError, match="holds no fd grid"):  # another analysis window
        panel_grids(panel, Geometry(20.0, 500.0, (90.0, 300.0)), [SPECS[1]], "auto")
    other = corridor / "sc" / "law_b" / "seed0"
    (other / "trajectories.npz").unlink()
    shutil.copy(run / FIELDS_FILE, other / FIELDS_FILE)  # the file of another run
    with pytest.raises(MissingInputError, match=r"belongs to the run sc/law_a/seed0"):
        panel_grids(PanelSource.run(other), GEOMETRY, SPECS, "auto")
    (other / FIELDS_FILE).unlink()
    with pytest.raises(MissingInputError) as error:
        panel_grids(PanelSource.run(other), GEOMETRY, SPECS, "auto")
    assert error.value.describe(lambda p: p.relative_to(corridor).as_posix()) == \
        "sc/law_b/seed0 has neither trajectories.npz nor fields.npz"  # fmt: skip
    (other / FIELDS_FILE).write_bytes(b"not a zip")
    with pytest.raises(MissingInputError, match="unreadable"):
        panel_grids(PanelSource.run(other), GEOMETRY, SPECS, "auto")
    truth = PanelSource.truth(corridor / "scenarios" / "sc")
    assert panel_grids(truth, GEOMETRY, SPECS, "fields")[SPECS[1]]["time"].sum() > 0  # a truth: always its npz
    (corridor / "scenarios" / "sc" / "ground_truth.npz").unlink()
    with pytest.raises(MissingInputError, match="ground_truth.npz missing"):
        panel_grids(truth, GEOMETRY, SPECS, "auto")


def test_figure_specs_are_those_the_figures_ask_for():
    """The export of the repository's configs holds the grids that report.py (figures 5 and 6 on its scenario) and
    figures_supplement.py (every scenario, with the metric settings of the scenario) ask for."""
    report = ReportConfig.from_mapping(
        {k: v for k, v in yaml.safe_load((REPO_ROOT / "configs" / "make_report.yaml").read_text(encoding="utf-8"))
         .items() if k not in ("paths", "hydra")},
        {"runs_root": Path("r"), "out_dir": Path("o"), "tables_m4": Path("a"), "tables_m5": Path("b"),
         "corridor_root": Path("c"), "docs_dir": Path("d")},
    )  # fmt: skip
    macro = yaml.safe_load((REPO_ROOT / "configs" / "corridor_metrics.yaml").read_text(encoding="utf-8"))["macro"]
    for scenario, lanes in ((report.scenario, None), ("us101_p1", [1, 2, 3, 4, 5])):
        scenario_json = {"x_in": 20.0, "x_out": 500.0, "analysis_window": [180.0, 840.0]}
        if lanes:
            scenario_json["metrics"] = {"waves": {"lanes": lanes}}
        specs = figure_specs(scenario, scenario_json)
        mcfg = scenario_macro_config(MacroConfig.from_mapping(macro), scenario_json)
        assert set(supplement_specs(mcfg)) <= set(specs)
        if scenario == report.scenario:
            assert set(report_specs(report)) <= set(specs)
        assert len(specs) == 2  # with the repository's configs the report's cells are those of the supplement
    assert figure_specs("us101_p1", {"metrics": {"waves": {"lanes": [1, 2, 3, 4, 5]}}})[0].lanes == (1, 2, 3, 4, 5)


def test_export_script(corridor, tmp_path):
    folder = configs(tmp_path)
    command = [sys.executable, str(REPO_ROOT / "scripts" / "export_fields.py"), "--corridor-root", str(corridor),
               "--configs-dir", str(folder)]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.strip().splitlines()
    assert [line.split(" ", 1)[0] for line in lines[:3]] == ["OK", "OK", "OK"]
    assert lines[-1].startswith("FIELDS: written 3, kept 0, skipped 0, failed 0; 3 fields.npz files of the selection")
    proc = subprocess.run([*command, "--law", "law_a", "--seed", "1", "--contour-seeds", "all"], capture_output=True,
                          text=True, encoding="utf-8", env=env, timeout=600)  # fmt: skip
    assert proc.returncode == 0 and proc.stdout.startswith("OK sc/law_a/seed1: contours, fd"), proc.stdout


# ------------------------------------------------------------------------------------------- the figures


def published(tree_root: Path, target: Path) -> Path:
    """A copy of a tree without the files that are not published (the trajectories, vehicles and collisions of the
    runs; the ground truths keep theirs)."""

    def ignore(directory: str, names: list[str]) -> list[str]:
        if "scenarios" in Path(directory).parts:
            return []
        return [n for n in names if n in ("trajectories.npz", "vehicles.npz", "collisions.npz")]

    shutil.copytree(tree_root, target, ignore=ignore)
    return target


@pytest.fixture(scope="module")
def report_tree(tmp_path_factory) -> dict[str, Path]:
    tree = build_report_tree(tmp_path_factory.mktemp("report_fields"))
    cfg = report_config(tree, tree["base"] / "unused")
    specs = report_specs(cfg)
    scenario_json = read_json(cfg.corridor_root / "scenarios" / "sc" / "scenario.json")
    for law in cfg.figure_laws:
        export_fields(cfg.corridor_root / "sc" / law / "seed0", specs, scenario_json)
    tree["published"] = published(tree["root"], tree["base"] / "published" / "runs")
    return tree


def test_report_figures_from_fields_are_byte_identical(report_tree):
    tree = report_tree
    outputs = {}
    for name, root, source in (("trajectories", tree["root"], "trajectories"), ("fields", tree["root"], "fields"),
                               ("published", tree["published"], "auto")):  # fmt: skip
        paths = {**tree, "root": root}
        cfg = report_config(paths, tree["base"] / f"out_{name}", corridor_source=source)
        lines = make_report(cfg)
        assert lines[1].startswith(f"FIGURES {len(FIGURES)} of {len(FIGURES)} (PNG, PDF)"), (name, lines[1])
        outputs[name] = cfg.out_dir
    assert not list((tree["published"] / "corridor" / "sc").rglob("trajectories.npz"))
    for figure in ("corridor_speed_contours", "fundamental_diagrams"):
        reference = (outputs["trajectories"] / "figures" / f"{figure}.png").read_bytes()
        for name in ("fields", "published"):
            assert (outputs[name] / "figures" / f"{figure}.png").read_bytes() == reference, (figure, name)
    notes = [(out / "report.md").read_text(encoding="utf-8").split("## Notes on missing inputs")[1]
             for out in outputs.values()]  # fmt: skip
    assert notes[0] == notes[1] == notes[2] and "law_b: no vehicle in lane(s) [3]" in notes[0]


def test_report_strict_raises_and_lenient_marks_incomplete(report_tree, tmp_path):
    root = published(report_tree["root"], tmp_path / "runs")
    (root / "corridor" / "sc" / "law_b" / "seed0" / FIELDS_FILE).unlink()  # run.json and macro.json only
    paths = {**report_tree, "root": root}
    with pytest.raises(MissingInputError, match=r"corridor figures of sc: .*sc/law_b/seed0 has neither "
                                                 r"trajectories.npz nor fields.npz"):  # fmt: skip
        make_report(report_config(paths, tmp_path / "strict"))
    assert not (tmp_path / "strict").exists()  # stopped before any output
    cfg = report_config(paths, tmp_path / "lenient", strict=False)
    lines = make_report(cfg)
    assert lines[1].startswith(f"FIGURES {len(FIGURES) - 2} of {len(FIGURES)} (PNG, PDF)")
    assert lines[1].endswith("incomplete (written without some panels, notes in report.md): corridor_speed_contours, "
                             "fundamental_diagrams")  # fmt: skip
    text = (cfg.out_dir / "report.md").read_text(encoding="utf-8")
    assert "- corridor figures: corridor/sc/law_b/seed0 has neither trajectories.npz nor fields.npz" in text
    assert ("> Figure `corridor_speed_contours` is incomplete (inputs missing): corridor/sc/law_b/seed0 has neither "
            "trajectories.npz nor fields.npz.") in text  # fmt: skip
    manifest = read_json(cfg.out_dir / "manifest.json")["outputs"]["figures"]
    assert manifest["fundamental_diagrams"].startswith("written, incomplete: corridor/sc/law_b/seed0")
    assert manifest["gain_curves"] == "written"
    # a fields.npz without the grid the figure asks for: strict names the file and the grid
    with pytest.raises(MissingInputError, match=r"law_a/seed0/fields.npz holds no contours grid of 20 m x 2 s cells "
                                                 r"over lanes 1, 2 "):  # fmt: skip
        make_report(report_config(paths, tmp_path / "lanes", contour_lanes=(1, 2)))


@pytest.fixture(scope="module")
def supplement_tree(tmp_path_factory) -> dict[str, Path]:
    tree = build_supplement_tree(tmp_path_factory.mktemp("supplement_fields"))
    cfg = supplement_config(tree, tree["base"] / "unused")
    macro = yaml.safe_load((tree["configs"] / "corridor_metrics.yaml").read_text(encoding="utf-8"))["macro"]
    scenario_json = read_json(cfg.corridor_root / "scenarios" / "i80_p0" / "scenario.json")
    specs = supplement_specs(scenario_macro_config(MacroConfig.from_mapping(macro), scenario_json))
    for law in ("law_a", "law_b", "law_d"):
        export_fields(cfg.corridor_root / "i80_p0" / law / "seed0", specs, scenario_json)
    tree["published"] = published(tree["root"], tree["base"] / "published" / "runs")
    return tree


def test_supplement_grids_from_fields_are_byte_identical(supplement_tree):
    tree = supplement_tree
    outputs = {}
    for name, root, source in (("trajectories", tree["root"], "trajectories"), ("fields", tree["root"], "fields"),
                               ("published", tree["published"], "auto")):  # fmt: skip
        cfg = supplement_config({**tree, "root": root}, tree["base"] / f"out_{name}", outputs=("corridor",),
                                corridors=("i80",), corridor_source=source)  # fmt: skip
        lines = make_supplement(cfg)
        assert lines[-1].startswith("SUPPLEMENT: 2 figures, 0 tables, "), (name, lines[-1])
        outputs[name] = cfg.out_dir / "figures"
    for figure in ("contours_i80_p0", "fd_i80_p0"):
        for suffix in (".png", ".txt"):
            reference = (outputs["trajectories"] / f"{figure}{suffix}").read_bytes()
            for name in ("fields", "published"):
                assert (outputs[name] / f"{figure}{suffix}").read_bytes() == reference, (figure, suffix, name)


def test_supplement_strict_raises_and_lenient_marks_incomplete(supplement_tree, tmp_path):
    root = published(supplement_tree["root"], tmp_path / "runs")
    (root / "corridor" / "i80_p0" / "law_b" / "seed0" / FIELDS_FILE).unlink()
    tree = {**supplement_tree, "root": root}
    with pytest.raises(MissingInputError, match=r"corridor i80_p0: .*i80_p0/law_b/seed0 has neither .*--no-strict"):
        make_supplement(supplement_config(tree, tmp_path / "strict", outputs=("band_width", "corridor"),
                                          corridors=("i80",)))  # fmt: skip
    assert not (tmp_path / "strict").exists()  # stopped before any output (the band widths come first otherwise)
    cfg = supplement_config(tree, tmp_path / "lenient", outputs=("corridor",), corridors=("i80",), strict=False)
    lines = make_supplement(cfg)
    assert lines[-1].startswith("SUPPLEMENT: 2 figures (2 incomplete: contours_i80_p0, fd_i80_p0), 0 tables, ")
    assert any(line.startswith("FIGURE fd_i80_p0: ") and "INCOMPLETE: drawn without 1 panel(s)" in line
               for line in lines)  # fmt: skip
    caption = (cfg.out_dir / "figures" / "contours_i80_p0.txt").read_text(encoding="utf-8")
    assert "(2): law_a, law_d." in caption and "law_b/seed0 has neither trajectories.npz nor fields.npz" in caption
    # a main law without any run in a scenario that exists
    shutil.rmtree(root / "corridor" / "i80_p0" / "law_a")
    with pytest.raises(MissingInputError, match=r"law_a/seed0 missing \(a main law\)"):
        make_supplement(supplement_config(tree, tmp_path / "main", outputs=("corridor",), corridors=("i80",)))
