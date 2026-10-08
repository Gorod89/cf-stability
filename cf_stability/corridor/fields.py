"""Edie fields of the corridor figures (``fields.npz``): what the speed contours and the fundamental diagrams of the
report (figures 5 and 6, :mod:`cf_stability.eval.report`) and of the supplement (``contours_*``, ``fd_*``,
:mod:`cf_stability.eval.figures_supplement`) take from a run, so that they are drawn without ``trajectories.npz``
(3.5 GB over the runs, not published).

A figure asks for grids (:class:`GridSpec`): ``contours``, the x-t cells of ``dx`` m from ``x_in`` (whole cells inside
the section) by ``dt`` s from t = 0 over the whole run, lanes ``lanes`` together; and ``fd``, the cells of the metrics
(``section_edges`` of ``dx``, whole intervals of ``dt`` inside the analysis window, every lane). :func:`compute_grid`
sums the Edie distance and time of the trajectories prepared as the metrics prepare them (``prepare_trajectories`` with
the settings of the spec); :func:`panel_grids` computes them from ``trajectories.npz`` where the run has it and reads
them from its ``fields.npz`` otherwise (``source``); :func:`speed_panels` and :func:`fd_points` turn them into what the
figures draw. Both ways give bit-identical arrays: ``fields.npz`` holds the float64 sums of :func:`compute_grid`, and
the cells of a contour grid that covers its whole run, cut or padded with empty cells to the common end of a figure,
hold the sums of a grid computed to that end (the same pieces, summed in the same order).

``fields.npz`` (:func:`export_fields`; compressed, format :data:`FORMAT`):

* header: ``format``, ``scenario``, ``law``, ``seed``, ``run_config_hash`` (``config_hash`` of ``run.json``, "" without
  one), ``settings_hash`` (of the format and the settings of the grids: another setting makes the file stale),
  ``x_in``, ``x_out`` (m), ``window`` (the analysis window, s), ``grids`` (the names of the grids);
* per grid ``<name>`` (``contours``, ``fd``; ``contours_2``, ... for further settings): ``<name>_settings`` (JSON:
  kind, dx, dt, lanes, x_in, x_out, the window of a ``fd`` grid, the preparation), ``<name>_x_edges``,
  ``<name>_t_edges`` (float64), ``<name>_lanes`` (int64), ``<name>_distance``, ``<name>_time`` (float64 ``[x cell,
  t cell]``: vehicle-metres and vehicle-seconds of the lanes together); a contour grid also ``<name>_occupied`` (bool
  ``[lane, t cell]``: the lane has vehicle time in the interval) and ``<name>_t_max`` (s: the last time of the
  prepared rows, which fixes the common end).

The sums stay float64: they are sums of float64 pieces with full mantissas; rounded to float32 they move the speed of
98 % of the cells of the figures and the colour level of about one cell in 200 000, and the PNG of figure 5 is no
longer the same (seed-0 runs, 8 October 2026). Contour grids (0.1-0.2 MB) are written for the seeds the figures draw
(:data:`CONTOUR_SEEDS`), the cells of the diagrams (5 KB with the header) for every run; a figure that asks a
``fields.npz`` for a grid it does not hold gets :class:`MissingInputError`. :func:`update_fields`
(``scripts/export_fields.py``) writes the file of every selected run with ``trajectories.npz`` and keeps one that is
up to date.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from cf_stability.corridor.macro import (
    RUN_FILES, TRUTH_FILES, Geometry, MacroConfig, edie_grid, find_items, geometry_from_scenario, interval_edges,
    load_npz, prepare_trajectories, scenario_macro_config, section_edges,
)  # fmt: skip
from cf_stability.corridor.waves import speed_field
from cf_stability.utils import REPO_ROOT, config_hash, read_json

FIELDS_FILE = "fields.npz"
FORMAT = 1  # raise it when the content changes: every fields.npz written before is stale
KINDS = ("contours", "fd")
SOURCES = ("auto", "trajectories", "fields")  # auto: trajectories.npz where the run has it, else fields.npz
CONTOUR_SEEDS = (0,)  # the seed the corridor figures draw (figure_seed of configs/make_report.yaml, the supplement's)
CONFIGS = REPO_ROOT / "configs"
GRID_ARRAYS = ("x_edges", "t_edges", "lanes", "distance", "time", "occupied", "t_max")
LOAD_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


class MissingInputError(Exception):
    """A panel of a corridor figure without its input: ``path`` (the run directory or the file looked for) and
    ``reason`` ("missing", "has neither trajectories.npz nor fields.npz", ...)."""

    def __init__(self, path: str | Path, reason: str, context: str = "") -> None:
        self.path, self.reason, self.context = Path(path), reason, context
        super().__init__(f"{context}{self.describe()}")

    def describe(self, label: Any = None) -> str:
        """``<path> <reason>``, the path through ``label`` (e.g. relative to the runs root)."""
        return f"{(label or (lambda p: Path(p).as_posix()))(self.path)} {self.reason}"

    def within(self, where: str, lenient: str = "strict=false") -> MissingInputError:
        """The error with ``where`` before it and the remedies after it (the message of a strict build; ``lenient``:
        the option that draws the figure without the panel)."""
        hint = (f" (scripts/export_fields.py writes fields.npz where trajectories.npz exists; {lenient} draws the figure "
                "without the panel)")  # fmt: skip
        return MissingInputError(self.path, self.reason + hint, f"{where}: ")


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """Cells of one Edie grid of a corridor figure and the preparation of the trajectories (``max_gap_s``,
    ``boundary_points``, ``boundary_max_gap_s`` of :func:`prepare_trajectories`); ``lanes`` None: every lane."""

    kind: str
    dx: float
    dt: float
    lanes: tuple[int, ...] | None = None
    max_gap_s: float = 1.5
    boundary_points: bool = True
    boundary_max_gap_s: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"grid kind {self.kind!r} (known: {list(KINDS)})")
        if self.kind == "fd" and self.lanes is not None:
            raise ValueError("a fd grid takes every lane")
        lanes = None if self.lanes is None else tuple(int(lane) for lane in self.lanes)
        for name, value in (("dx", float(self.dx)), ("dt", float(self.dt)), ("lanes", lanes),
                            ("max_gap_s", float(self.max_gap_s)), ("boundary_points", bool(self.boundary_points)),
                            ("boundary_max_gap_s", float(self.boundary_max_gap_s))):  # fmt: skip
            object.__setattr__(self, name, value)  # canonical values: 20 and 20.0 are the same spec

    @classmethod
    def contours(cls, dx: float, dt: float, lanes: Sequence[int], prepare: MacroConfig | None = None) -> GridSpec:
        m = prepare or MacroConfig()
        return cls("contours", dx, dt, tuple(lanes), m.max_gap_s, m.boundary_points, m.boundary_max_gap_s)

    @classmethod
    def fd(cls, dx: float, dt: float, prepare: MacroConfig | None = None) -> GridSpec:
        m = prepare or MacroConfig()
        return cls("fd", dx, dt, None, m.max_gap_s, m.boundary_points, m.boundary_max_gap_s)

    @property
    def preparation(self) -> MacroConfig:
        """The settings of :func:`prepare_trajectories` (the others are not used there)."""
        return MacroConfig(max_gap_s=self.max_gap_s, boundary_points=self.boundary_points,
                           boundary_max_gap_s=self.boundary_max_gap_s)  # fmt: skip

    def settings(self, geometry: Geometry) -> dict[str, Any]:
        """Everything the grid depends on: the cells, the section (and the window of a ``fd`` grid), the preparation."""
        lanes = None if self.lanes is None else list(self.lanes)
        out = {"kind": self.kind, "dx": self.dx, "dt": self.dt, "lanes": lanes,
               "x_in": float(geometry.x_in), "x_out": float(geometry.x_out),
               "prepare": {"max_gap_s": self.max_gap_s, "boundary_points": self.boundary_points,
                           "boundary_max_gap_s": self.boundary_max_gap_s}}  # fmt: skip
        if self.kind == "fd":
            out["window"] = [float(w) for w in geometry.window]
        return out

    def describe(self) -> str:
        lanes = "every lane" if self.lanes is None else "lanes " + ", ".join(str(lane) for lane in self.lanes)
        return f"{self.kind} grid of {self.dx:g} m x {self.dt:g} s cells over {lanes}"


def describe_settings(settings: Mapping[str, Any]) -> str:
    """:meth:`GridSpec.describe` of stored settings."""
    lanes = settings.get("lanes")
    lanes = "every lane" if lanes is None else "lanes " + ", ".join(str(lane) for lane in lanes)
    return f"{settings.get('kind')} grid of {settings.get('dx'):g} m x {settings.get('dt'):g} s cells over {lanes}"


# ------------------------------------------------------------------------------------------------- the grids


def contour_t_edges(t_max: float, dt: float) -> np.ndarray:
    """Edges ``0, dt, ...`` of the intervals up to the first one past ``t_max`` (none without samples): the values of
    ``interval_edges(0, t_end, dt)`` for every ``t_end`` the figure can take."""
    n = int(math.floor(t_max / dt + 1e-9)) + 1 if math.isfinite(t_max) else 0
    return 0.0 + dt * np.arange(n + 1, dtype=np.float64)


def compute_grid(prepared: Mapping[str, np.ndarray], geometry: Geometry, spec: GridSpec) -> dict[str, Any]:
    """The grid ``spec`` of trajectories prepared with its settings (:func:`prepare_trajectories`), lanes together."""
    arrays = {"vehicle": prepared["piece"], "t": prepared["t"], "x": prepared["x"], "lane": prepared["lane"]}
    if spec.kind == "contours":
        t = np.asarray(prepared["t"], dtype=np.float64)
        t_max = float(t.max()) if len(t) else math.nan
        x_edges = interval_edges(geometry.x_in, geometry.x_out, spec.dx)
        grid = edie_grid(arrays, x_edges, contour_t_edges(t_max, spec.dt), lanes=spec.lanes)
        return {"x_edges": grid["x_edges"], "t_edges": grid["t_edges"], "lanes": np.asarray(grid["lanes"], np.int64),
                "distance": grid["total"]["distance"], "time": grid["total"]["time"],
                "occupied": grid["per_lane"]["time"].sum(axis=1) > 0, "t_max": t_max}  # fmt: skip
    w0, w1 = (float(w) for w in geometry.window)
    grid = edie_grid(arrays, section_edges(geometry.x_in, geometry.x_out, spec.dx), interval_edges(w0, w1, spec.dt))
    return {"x_edges": grid["x_edges"], "t_edges": grid["t_edges"], "lanes": np.asarray(grid["lanes"], np.int64),
            "distance": grid["total"]["distance"], "time": grid["total"]["time"]}  # fmt: skip


def compute_grids(
    arrays: Mapping[str, Any], vehicles: Mapping[str, Any] | None, geometry: Geometry, specs: Sequence[GridSpec]
) -> dict[GridSpec, dict[str, Any]]:  # fmt: skip
    """The grids ``specs`` of the arrays of ``trajectories.npz`` (and ``vehicles.npz``), prepared once per setting."""
    prepared: dict[tuple, dict[str, np.ndarray]] = {}
    out = {}
    for spec in specs:
        key = (spec.max_gap_s, spec.boundary_points, spec.boundary_max_gap_s)
        if key not in prepared:
            prepared[key] = prepare_trajectories(arrays, vehicles, geometry, spec.preparation)
        out[spec] = compute_grid(prepared[key], geometry, spec)
    return out


def _cells(values: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` time cells of ``values [x, t]``, empty cells after its own end."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] >= n:
        return values[:, :n]
    return np.concatenate([values, np.zeros((values.shape[0], n - values.shape[1]))], axis=1)


def speed_panels(
    grids: Sequence[Mapping[str, Any]], dt: float
) -> tuple[np.ndarray, list[tuple[np.ndarray, list[int]]]]:  # fmt: skip
    """Contour grids of one figure on one time axis: the edges of the whole intervals of ``dt`` from 0 to the last
    time of any grid (``interval_edges``), and per grid the speed of its raw cells on that axis (``speed_field``; the
    cells past its own end empty) with the lanes that hold no vehicle time on the axis."""
    ends = [float(g["t_max"]) for g in grids if math.isfinite(float(g["t_max"]))]
    if not ends:
        raise ValueError("no vehicle in any panel of the corridor")
    t_edges = interval_edges(0.0, max(ends), dt)
    n = len(t_edges) - 1
    out = []
    for g in grids:
        speed = speed_field(_cells(g["distance"], n), _cells(g["time"], n), smooth=0)
        occupied = np.asarray(g["occupied"], dtype=bool)
        empty = [int(lane) for lane, cells in zip(g["lanes"], occupied) if not cells[:n].any()]
        out.append((speed, empty))
    return t_edges, out


def fd_points(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Density (veh/km) and flow (veh/h) of the cells of a ``fd`` grid that hold vehicle time, lanes together: the
    points of a fundamental diagram (Edie: flow = distance / area, density = time / area, as :func:`edie_grid`)."""
    area = np.diff(grid["x_edges"])[:, None] * np.diff(grid["t_edges"])[None, :]
    flow, density = (grid["distance"] / area).ravel() * 3600.0, (grid["time"] / area).ravel() * 1000.0
    keep = np.isfinite(flow) & np.isfinite(density) & (np.asarray(grid["time"]).ravel() > 0)
    return density[keep], flow[keep]


# ------------------------------------------------------------------------------------------------ the file


@dataclasses.dataclass
class Fields:
    """A ``fields.npz``: the header and the grids by name (each with its ``settings`` and its arrays)."""

    path: Path
    header: dict[str, Any]
    grids: dict[str, dict[str, Any]]

    def grid(self, spec: GridSpec, geometry: Geometry) -> dict[str, Any] | None:
        """The grid of ``spec`` on ``geometry``, None when the file holds none with exactly these settings."""
        wanted = spec.settings(geometry)
        return next((g for g in self.grids.values() if g["settings"] == wanted), None)

    def holds(self) -> str:
        return "; ".join(describe_settings(g["settings"]) for g in self.grids.values()) or "no grid"


def settings_hash(specs: Sequence[GridSpec], geometry: Geometry) -> str:
    return config_hash({"format": FORMAT, "grids": [spec.settings(geometry) for spec in specs]})


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """A compressed ``.npz`` with fixed entry dates (the same content gives the same bytes), written through a
    temporary file and a rename."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temporary, path)


def run_parts(run_dir: Path) -> tuple[str, str, int]:
    """Scenario, law and seed of a run directory ``<scenario>/<law>/seed<k>``."""
    run_dir = Path(run_dir)
    if not run_dir.name.startswith("seed") or not run_dir.name[4:].isdigit():
        raise ValueError(f"{run_dir} is no run directory <scenario>/<law>/seed<k>")
    return run_dir.parent.parent.name, run_dir.parent.name, int(run_dir.name[4:])


def export_fields(
    run_dir: str | Path, specs: Sequence[GridSpec], scenario_json: Mapping[str, Any], *, contours: bool = True
) -> Path:  # fmt: skip
    """Write ``<run_dir>/fields.npz``: the grids ``specs`` of the run's ``trajectories.npz`` and ``vehicles.npz`` on the
    geometry of its ``scenario_json`` (the contour grids only with ``contours``); returns the path."""
    run_dir = Path(run_dir)
    scenario, law, seed = run_parts(run_dir)
    geometry = geometry_from_scenario(scenario_json, Geometry())
    stored = [spec for spec in dict.fromkeys(specs) if contours or spec.kind != "contours"]
    trajectories, vehicles = (run_dir / name for name in RUN_FILES)
    grids = compute_grids(load_npz(trajectories), load_npz(vehicles) if vehicles.exists() else None, geometry, stored)
    run_json = run_dir / "run.json"
    run_hash = str(read_json(run_json).get("config_hash") or "") if run_json.exists() else ""
    names, counts = [], Counter()
    payload: dict[str, Any] = {
        "format": np.int64(FORMAT), "scenario": np.str_(scenario), "law": np.str_(law), "seed": np.int64(seed),
        "run_config_hash": np.str_(run_hash), "settings_hash": np.str_(settings_hash(stored, geometry)),
        "x_in": np.float64(geometry.x_in), "x_out": np.float64(geometry.x_out),
        "window": np.asarray(geometry.window, dtype=np.float64),
    }  # fmt: skip
    for spec in stored:
        counts[spec.kind] += 1
        name = spec.kind if counts[spec.kind] == 1 else f"{spec.kind}_{counts[spec.kind]}"
        names.append(name)
        payload[f"{name}_settings"] = np.str_(json.dumps(spec.settings(geometry), sort_keys=True))
        for key in GRID_ARRAYS:
            if key in grids[spec]:
                payload[f"{name}_{key}"] = np.asarray(grids[spec][key])
    payload["grids"] = np.asarray(names, dtype=np.str_)
    path = run_dir / FIELDS_FILE
    _save_npz(path, payload)
    return path


def load_fields(path: str | Path) -> Fields:
    """A ``fields.npz`` (or the one of a run directory); raises ValueError for another format."""
    path = Path(path)
    if path.is_dir():
        path = path / FIELDS_FILE
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if int(arrays["format"]) != FORMAT:
        raise ValueError(f"{path} has format {int(arrays['format'])}, this code reads format {FORMAT}")
    header = {
        "format": int(arrays["format"]), "scenario": str(arrays["scenario"]), "law": str(arrays["law"]),
        "seed": int(arrays["seed"]), "run_config_hash": str(arrays["run_config_hash"]),
        "settings_hash": str(arrays["settings_hash"]), "x_in": float(arrays["x_in"]), "x_out": float(arrays["x_out"]),
        "window": tuple(float(w) for w in arrays["window"]),
    }  # fmt: skip
    grids = {}
    for name in (str(n) for n in np.atleast_1d(arrays["grids"])):
        grid: dict[str, Any] = {"name": name, "settings": json.loads(str(arrays[f"{name}_settings"]))}
        for key in GRID_ARRAYS:
            if f"{name}_{key}" in arrays:
                grid[key] = arrays[f"{name}_{key}"]
        if "t_max" in grid:
            grid["t_max"] = float(grid["t_max"])
        grids[name] = grid
    return Fields(path, header, grids)


def _checked(grid: Mapping[str, Any], spec: GridSpec, geometry: Geometry) -> bool:
    """The stored edges are those of the settings (the cells are the ones the figure asks for)."""
    if spec.kind == "contours":
        x_edges = interval_edges(geometry.x_in, geometry.x_out, spec.dx)
        t_edges = 0.0 + spec.dt * np.arange(len(grid["t_edges"]), dtype=np.float64)
    else:
        x_edges = section_edges(geometry.x_in, geometry.x_out, spec.dx)
        t_edges = interval_edges(float(geometry.window[0]), float(geometry.window[1]), spec.dt)
    shape = (len(x_edges) - 1, len(t_edges) - 1)
    return (np.array_equal(grid["x_edges"], x_edges) and np.array_equal(grid["t_edges"], t_edges)
            and grid["distance"].shape == shape and grid["time"].shape == shape)  # fmt: skip


# ----------------------------------------------------------------------------------------------- the panels


@dataclasses.dataclass(frozen=True)
class PanelSource:
    """The files of one panel: a ground truth (``fields`` None: always its npz files) or a run directory."""

    trajectories: Path
    vehicles: Path
    fields: Path | None = None

    @classmethod
    def truth(cls, scenario_dir: str | Path) -> PanelSource:
        return cls(*(Path(scenario_dir) / name for name in TRUTH_FILES))

    @classmethod
    def run(cls, run_dir: str | Path) -> PanelSource:
        run_dir = Path(run_dir)
        return cls(run_dir / RUN_FILES[0], run_dir / RUN_FILES[1], run_dir / FIELDS_FILE)

    @property
    def directory(self) -> Path:
        return self.trajectories.parent

    def uses_trajectories(self, source: str) -> bool:
        """Whether the panel is drawn from its trajectories (``source`` auto: when the file exists)."""
        if source not in SOURCES:
            raise ValueError(f"corridor source {source!r} (known: {list(SOURCES)})")
        if self.fields is None or source == "trajectories":
            return True
        return source == "auto" and self.trajectories.exists()


def panel_grids(
    panel: PanelSource, geometry: Geometry, specs: Sequence[GridSpec], source: str = "auto"
) -> dict[GridSpec, dict[str, Any]]:  # fmt: skip
    """The grids ``specs`` of one panel: computed from its trajectories (``source`` auto: where the run has
    ``trajectories.npz``; trajectories: always) or read from the run's ``fields.npz`` (auto: without trajectories;
    fields: always). Raises :class:`MissingInputError` naming the run or the file when they cannot be had."""
    if panel.uses_trajectories(source):
        if not panel.trajectories.exists():
            reason = "missing" if panel.fields is None else "missing (corridor source: trajectories)"
            raise MissingInputError(panel.trajectories, reason)
        try:
            arrays = load_npz(panel.trajectories)
            fleet = load_npz(panel.vehicles) if panel.vehicles.exists() else None
        except LOAD_ERRORS as exc:
            raise MissingInputError(panel.trajectories, f"unreadable ({type(exc).__name__})") from exc
        return compute_grids(arrays, fleet, geometry, specs)
    assert panel.fields is not None
    if not panel.fields.exists():
        if source == "auto":
            names = (panel.trajectories.name, panel.fields.name)
            raise MissingInputError(panel.directory, "has neither {} nor {}".format(*names))
        raise MissingInputError(panel.fields, "missing (corridor source: fields)")
    try:
        stored = load_fields(panel.fields)
    except LOAD_ERRORS as exc:
        raise MissingInputError(panel.fields, f"unreadable ({type(exc).__name__}: {exc})") from exc
    try:
        expected = run_parts(panel.directory)
    except ValueError:
        expected = None
    found = (stored.header["scenario"], stored.header["law"], stored.header["seed"])
    if expected is not None and found != expected:
        raise MissingInputError(panel.fields, "belongs to the run {}/{}/seed{}".format(*found))
    out = {}
    for spec in specs:
        grid = stored.grid(spec, geometry)
        if grid is None:
            where = f"the section {geometry.x_in:g}-{geometry.x_out:g} m"
            raise MissingInputError(panel.fields, f"holds no {spec.describe()} on {where} with these settings (it holds: "
                                                  f"{stored.holds()}); {panel.trajectories.name} is needed for it")
        if not _checked(grid, spec, geometry):
            raise MissingInputError(panel.fields, f"holds a {spec.describe()} whose edges differ from its settings")
        out[spec] = grid
    return out


# ------------------------------------------------------------------------------------------------ the export


def _yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def figure_specs(scenario: str, scenario_json: Mapping[str, Any], configs_dir: str | Path = CONFIGS) -> list[GridSpec]:
    """The grids the corridor figures draw of a run of ``scenario``: the speed field (the wave cells and lanes) and the
    cells of the metrics (diagrams) of the scenario's metric settings (``macro`` of ``configs/corridor_metrics.yaml``
    with the scenario's ``metrics`` over it), as ``scripts/make_figures_supplement.py`` draws them; and for the
    scenario of the report (``configs/make_report.yaml``) the cells of its figures 5 and 6 (``contour_*``, ``fd_*``;
    the default preparation, as ``report.py``) where they differ."""
    configs_dir = Path(configs_dir)
    mcfg = scenario_macro_config(MacroConfig.from_mapping(_yaml(configs_dir / "corridor_metrics.yaml").get("macro")),
                                 scenario_json)  # fmt: skip
    specs = [GridSpec.contours(mcfg.waves.dx, mcfg.waves.dt, mcfg.waves.lanes, mcfg),
             GridSpec.fd(mcfg.cell_dx, mcfg.cell_dt, mcfg)]  # fmt: skip
    report = _yaml(configs_dir / "make_report.yaml")
    if report.get("scenario", "i80_p1") == scenario:  # the defaults of ReportConfig
        specs += [GridSpec.contours(report.get("contour_dx", 20.0), report.get("contour_dt", 2.0),
                                    report.get("contour_lanes", (1, 2, 3, 4, 5, 6))),
                  GridSpec.fd(report.get("fd_dx", 100.0), report.get("fd_dt", 30.0))]  # fmt: skip
    return list(dict.fromkeys(specs))


def _inputs(run_dir: Path, scenario_file: Path) -> list[Path]:
    """Files whose change makes ``fields.npz`` stale."""
    return [*(run_dir / name for name in RUN_FILES), run_dir / "run.json", scenario_file]


def is_current(run_dir: Path, expected_hash: str, inputs: Sequence[Path]) -> bool:
    """``fields.npz`` exists, has the format and the settings hash ``expected_hash``, belongs to the run and is not
    older than any of ``inputs``."""
    path = run_dir / FIELDS_FILE
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            header = (int(data["format"]), str(data["settings_hash"]), str(data["scenario"]), str(data["law"]),
                      int(data["seed"]))  # fmt: skip
    except (*LOAD_ERRORS, TypeError):
        return False
    if header != (FORMAT, expected_hash, *run_parts(run_dir)):
        return False
    written = path.stat().st_mtime
    return all(p.stat().st_mtime <= written for p in inputs if p.exists())


def update_fields(
    corridor_root: str | Path, *, scenario: str | None = None, law: str | None = None, seed: int | None = None,
    force: bool = False, contour_seeds: Sequence[int] | None = CONTOUR_SEEDS, scenarios_root: str | Path | None = None,
    configs_dir: str | Path = CONFIGS,
) -> Iterator[str]:  # fmt: skip
    """Write the ``fields.npz`` of every selected run (``find_items`` of the metrics; ground truths are not selected:
    their npz files are published) with the grids of :func:`figure_specs` (contour grids for ``contour_seeds``, None:
    every seed); yields one line per run: ``OK``, ``KEPT`` (up to date, unless ``force``), ``SKIPPED`` (no
    ``trajectories.npz`` or ``run.json``: its ``fields.npz`` is left as it is) or ``FAILED``."""
    root = Path(corridor_root)
    scenarios = Path(scenarios_root) if scenarios_root is not None else root / "scenarios"
    settings: dict[str, tuple[dict[str, Any], list[GridSpec], Geometry]] = {}
    for item in find_items(root, scenario, law, seed):
        if item.is_truth:
            continue
        run_dir = item.directory
        missing = [name for name in (RUN_FILES[0], "run.json") if not (run_dir / name).exists()]
        if missing:
            kept = f" ({FIELDS_FILE} left as it is)" if (run_dir / FIELDS_FILE).exists() else ""
            yield f"SKIPPED {item.label}: {', '.join(missing)} missing{kept}"
            continue
        start = time.perf_counter()
        scenario_file = scenarios / item.scenario / "scenario.json"
        try:
            if item.scenario not in settings:
                scenario_json = read_json(scenario_file)
                settings[item.scenario] = (scenario_json, figure_specs(item.scenario, scenario_json, configs_dir),
                                           geometry_from_scenario(scenario_json, Geometry()))  # fmt: skip
            scenario_json, specs, geometry = settings[item.scenario]
            contours = contour_seeds is None or item.seed in contour_seeds
            stored = [spec for spec in specs if contours or spec.kind != "contours"]
            if not force and is_current(run_dir, settings_hash(stored, geometry), _inputs(run_dir, scenario_file)):
                yield f"KEPT {item.label}: {run_dir / FIELDS_FILE} is up to date"
                continue
            path = export_fields(run_dir, specs, scenario_json, contours=contours)
        except Exception as exc:  # one broken run must not stop the others
            message = str(exc).splitlines()[0] if str(exc) else ""
            yield f"FAILED {item.label}: {type(exc).__name__}: {message}"
            continue
        kinds, seconds = ", ".join(spec.kind for spec in stored), time.perf_counter() - start
        yield f"OK {item.label}: {kinds} ({path.stat().st_size / 1024:.0f} KB, {seconds:.1f} s) -> {path}"
