"""OpenACC loader (JRC car-following campaigns, CC BY 4.0; format in docs/data_contract.md, 4a).

Every csv file is one platoon run. ``IVS<i>`` is the gap between vehicle ``i`` and vehicle
``i + 1``, i.e. the spacing of follower ``i + 1`` (D14). A driver is the pair (vehicle,
driving mode) (D15); the mode of a sample comes from ``Driver<i>`` or the header flag (D16).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events
from cf_stability.data.processing import resample_nearest, resample_uniform
from cf_stability.data.schema import DT, EventSet
from cf_stability.utils import resolve_path

DATASET = "openacc"
# site name -> campaign folder under raw_dir
CAMPAIGN_DIRS = {
    "AstaZero": "AstaZero",
    "Casale": "Casale",
    "Cherasco": "Cherasco",
    "JRC": "JRC low speed",
    "Vicolungo": "Vicolungo",
    "ZalaZone": "ZalaZone",
}
MODES = ("ACC", "Human", "unknown")
HEADER_MODES = {0: "Human", 1: "ACC"}  # header flag ACC -> mode when Driver<i> is absent

_COLUMN = re.compile(r"^(Speed|Lat|Lon|Alt|E|N|U|VE|VN|VU|IVS|Driver)(\d+)$")
_KINDS = {"Speed": "speed", "IVS": "ivs", "Driver": "driver", "E": "e", "N": "n"}
_MAX_HEADER_LINES = 20
_SITE_OF_DIR = {folder: site for site, folder in CAMPAIGN_DIRS.items()}


@dataclass
class OpenACCRun:
    """One OpenACC file: header metadata and the per-vehicle columns on the source time base."""

    path: Path
    date: str | None  # ISO date
    vehicle_order: list[str]  # names, whitespace removed; vehicle i is vehicle_order[i - 1]
    n_vehicles: int | None
    acc_flag: int | None  # 0 manual, 1 ACC, 2 mixed, None unknown
    distance_setting: str | None
    time: np.ndarray  # [n] s, common time frame of the run
    # vehicle index (1-based) -> the columns present among speed, ivs, driver (object), e, n
    vehicles: dict[int, dict[str, np.ndarray]]

    @property
    def campaign(self) -> str:
        return self.path.parent.name

    @property
    def site(self) -> str:
        """Site name of the campaign folder (``JRC low speed`` -> ``JRC``)."""
        return _SITE_OF_DIR.get(self.campaign, self.campaign)

    def vehicle_name(self, index: int) -> str:
        return self.vehicle_order[index - 1]


def parse_openacc_csv(path: str | Path) -> OpenACCRun:
    """Parse one OpenACC csv file; table columns are identified by name, never by position."""
    path = Path(path)
    header, names, header_line = _read_header(path)
    selected: dict[str, tuple[str, int]] = {}
    for name in names:
        match = _COLUMN.match(name.strip())
        if match and match.group(1) in _KINDS:
            selected[name] = (_KINDS[match.group(1)], int(match.group(2)))
    time_column = next(name for name in names if name.strip() == "Time")
    frame = pd.read_csv(
        path,
        skiprows=header_line,
        header=0,
        index_col=False,  # tolerates trailing commas in the data rows
        usecols=[time_column, *selected],
        dtype={name: (str if kind == "driver" else np.float64) for name, (kind, _) in selected.items()}
        | {time_column: np.float64},
        encoding="utf-8",
    )
    vehicles: dict[int, dict[str, np.ndarray]] = {}
    for name, (kind, index) in selected.items():
        column = frame[name]
        if kind == "driver":
            text = column.str.strip()
            values = text.where(text != "").to_numpy(dtype=object, na_value=None)
        else:
            values = column.to_numpy(dtype=np.float64)
        vehicles.setdefault(index, {})[kind] = values
    order = header.get("Vehicle_order", [])
    return OpenACCRun(
        path=path,
        date=_iso_date(header.get("Date", [])),
        vehicle_order=[re.sub(r"\s+", "", name) for name in order],
        n_vehicles=_int_or_none(header.get("Number_of_vehicles")),
        acc_flag=_int_or_none(header.get("ACC")),
        distance_setting=(header.get("Distance_setting") or [None])[0],
        time=frame[time_column].to_numpy(dtype=np.float64),
        vehicles=dict(sorted(vehicles.items())),
    )


def vehicle_pairs(run: OpenACCRun) -> list[int]:
    """Leader indices ``i`` of the usable pairs: ``IVS<i>``, ``Speed<i>`` and ``Speed<i+1>`` exist."""
    return [
        i
        for i, columns in run.vehicles.items()
        if "ivs" in columns and "speed" in columns and "speed" in run.vehicles.get(i + 1, {})
    ]


def sample_modes(run: OpenACCRun, index: int) -> np.ndarray:
    """Driving mode of vehicle ``index`` per source sample (D16).

    ``Driver<i>`` decides where the column exists (an empty or unrecognised cell is
    ``unknown``); without the column the header flag decides (1 ACC, 0 Human, else unknown).
    """
    driver = run.vehicles[index].get("driver")
    if driver is None:
        return np.full(run.time.shape, HEADER_MODES.get(run.acc_flag, "unknown"), dtype=object)
    lower = np.array([str(value).lower() for value in driver], dtype=object)
    return np.where(lower == "acc", "ACC", np.where(lower == "human", "Human", "unknown")).astype(object)




def run_to_pairs(run: OpenACCRun, site: str, modes: Sequence[str] = MODES, dt: float = DT) -> list[PairSeries]:
    """One PairSeries per (follower ``i + 1`` behind leader ``i``, driving mode) on the uniform grid.

    Follower samples of other modes are NaN, so that the extractor cuts there. Positions have
    no common frame: ``x_lead`` integrates the leader speed and ``x_follower = x_lead - IVS``.
    """
    pairs_of_run = vehicle_pairs(run)
    columns = {}
    for i in pairs_of_run:
        columns[f"ivs{i}"] = run.vehicles[i]["ivs"]
        columns[f"speed{i}"] = run.vehicles[i]["speed"]
        columns[f"speed{i + 1}"] = run.vehicles[i + 1]["speed"]
    grid, series = resample_uniform(run.time, columns, dt=dt)
    pairs = []
    for i in pairs_of_run:
        follower, leader = run.vehicle_name(i + 1), run.vehicle_name(i)
        v, v_lead, ivs = series[f"speed{i + 1}"], series[f"speed{i}"], series[f"ivs{i}"]
        x_lead = cumulative_trapezoid(np.nan_to_num(v_lead, nan=0.0), dx=dt, initial=0.0)
        x_follower = x_lead - ivs
        mode = resample_nearest(run.time, sample_modes(run, i + 1), grid)
        observed = np.isfinite(v) & np.isfinite(x_follower)
        for m in modes:
            here = mode == m
            if not np.any(here & observed):
                continue
            pairs.append(
                PairSeries(
                    dataset=DATASET,
                    site=site,
                    follower_id=f"{DATASET}/{site}/{follower}:{m}",
                    t=grid,
                    x_follower=np.where(here, x_follower, np.nan),
                    v=np.where(here, v, np.nan),
                    leader_id=np.full(grid.shape, f"{DATASET}/{site}/{leader}", dtype=object),
                    x_lead=x_lead,
                    v_lead=v_lead,
                    run=run.path.stem,
                    meta={
                        "campaign": run.campaign,
                        "file": run.path.name,
                        "vehicle": follower,
                        "leader_vehicle": leader,
                        "driver_mode": m,
                        "acc_flag": run.acc_flag,
                        "distance_setting": run.distance_setting,
                        "platoon_index": i + 1,
                        "date": run.date,
                    },
                )
            )
    return pairs


def data_files(folder: Path) -> list[Path]:
    """The run files of a campaign folder (first line ``Date,...``; skips the specification tables)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"OpenACC campaign folder not found: {folder}")
    files = []
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            if fh.readline().startswith("Date"):
                files.append(path)
    return files


def load_runs(data_cfg: Mapping) -> Iterator[OpenACCRun]:
    """Parsed runs of the configured campaigns (``campaigns`` are site names), in file order."""
    raw_dir = resolve_path(data_cfg.get("raw_dir", "data/raw/openacc"))
    for site in data_cfg.get("campaigns") or CAMPAIGN_DIRS:
        for path in data_files(raw_dir / CAMPAIGN_DIRS.get(site, site)):
            yield parse_openacc_csv(path)


def build_events(data_cfg: Mapping, extraction: ExtractionConfig, stats: Counter | None = None) -> EventSet:
    """Events of all configured campaigns and driving modes.

    Extra ``stats``: ``files``, ``pairs``, ``events_site_<site>``, ``events_mode_<mode>``,
    ``nonpositive_ivs_samples`` (grid samples of the selected modes with ``IVS <= 0``).
    """
    stats = Counter() if stats is None else stats
    modes = list(data_cfg.get("modes") or MODES)
    events = []
    for run in load_runs(data_cfg):
        stats["files"] += 1
        stats["pairs"] += len(vehicle_pairs(run))
        for pair in run_to_pairs(run, run.site, modes, extraction.dt):
            gap = pair.x_lead - pair.x_follower
            stats["nonpositive_ivs_samples"] += int(np.count_nonzero(np.isfinite(gap) & (gap <= 0.0)))
            new = extract_events(pair, extraction, stats)
            stats[f"events_site_{run.site}"] += len(new)
            stats[f"events_mode_{pair.meta['driver_mode']}"] += len(new)
            events.extend(new)
    return EventSet(events)


def _read_header(path: Path) -> tuple[dict[str, list[str]], list[str], int]:
    """Metadata lines (key -> non-empty cells), raw table column names, index of the table header line."""
    header: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for number, line in enumerate(fh):
            cells = line.rstrip("\r\n").split(",")
            key = cells[0].strip()
            if key == "Time":
                return header, cells, number
            header[key] = [cell.strip() for cell in cells[1:] if cell.strip()]
            if number >= _MAX_HEADER_LINES:
                break
    raise ValueError(f"{path}: no table header line starting with 'Time'")


def _int_or_none(cells: list[str] | None) -> int | None:
    try:
        return int(cells[0]) if cells else None
    except ValueError:
        return None


def _iso_date(cells: list[str]) -> str | None:
    """``Date,DD,MM,YYYY`` -> ``YYYY-MM-DD``."""
    try:
        day, month, year = (int(cell) for cell in cells[:3])
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"
