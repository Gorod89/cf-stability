"""OpenACC loader: parser layouts, pair assignment (D14), driving modes (D16), cuts."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cf_stability.data.extraction import ExtractionConfig, extract_events
from cf_stability.data.openacc import (
    CAMPAIGN_DIRS,
    build_events,
    data_files,
    parse_openacc_csv,
    run_to_pairs,
    sample_modes,
    vehicle_pairs,
)
from cf_stability.utils import resolve_path

T = np.round(np.arange(0.0, 100.0, 0.1), 1)
CFG = ExtractionConfig(max_duration=60.0)


def speed(t: np.ndarray, phase: float) -> np.ndarray:
    return 15.0 + 2.0 * np.sin(0.2 * t + phase)


IVS1 = 20.0 + 3.0 * np.sin(0.1 * T)
IVS2 = 40.0 + 3.0 * np.cos(0.1 * T)


def write_run(path: Path, meta: list[str], columns: dict, trailing_comma: bool = False) -> Path:
    """OpenACC-like csv: metadata lines, header, rows; NaN and None become empty cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    lines = [*meta, ",".join(names)]
    for k in range(len(columns["Time"])):
        cells = []
        for name in names:
            value = columns[name][k]
            if value is None or (not isinstance(value, str) and np.isnan(value)):
                cells.append("")
            else:
                cells.append(value if isinstance(value, str) else repr(float(value)))
        lines.append(",".join(cells) + ("," if trailing_comma else ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def gnss(i: int, n: int, names: tuple[str, ...]) -> dict:
    return {f"{name}{i}": np.full(n, 1.0 + i) for name in names}


def astazero_like(path: Path) -> Path:
    """Driver columns after the vehicle blocks, IVS last, header flag 0 but ACC samples."""
    n = T.size
    columns = {"Time": T}
    for i in (1, 2, 3):
        columns[f"Speed{i}"] = speed(T, -0.3 * i)
        columns |= gnss(i, n, ("Lat", "Lon", "Alt", "E", "N", "U"))
    columns |= {f"Driver{i}": ["ACC"] * n for i in (1, 2, 3)}
    columns |= {"IVS1": IVS1, "IVS2": IVS2}
    meta = [
        "Date,4,7,2019,,,,,,",
        "Vehicle_order,Audi (A8),Audi(A6),BMW(X5),,,,",
        "Number_of_vehicles,3,,,,,",
        "ACC,0,,,,,",
        "Distance_setting,none",
    ]
    return write_run(path, meta, columns)


def casale_like(path: Path, t: np.ndarray = T, ivs: np.ndarray = IVS1, driver: list | None = None) -> Path:
    """Leader without Alt/U, only Driver2."""
    n = t.size
    columns = {"Time": t, "Speed1": speed(t, 0.0)} | gnss(1, n, ("Lat", "Lon", "E", "N"))
    columns |= {"Speed2": speed(t, -0.3)} | gnss(2, n, ("Lat", "Lon", "Alt", "E", "N", "U"))
    columns |= {"IVS1": ivs, "Driver2": driver if driver is not None else ["ACC"] * n}
    meta = ["Date,27,10,2020", "Vehicle_order,Rexton,Hyundai", "Number_of_vehicles,2", "ACC,1", "Distance_setting,min"]
    return write_run(path, meta, columns)


def zalazone_like(path: Path) -> Path:
    """Vehicle 3 listed but without columns, no Driver columns, empty cells, trailing commas."""
    n = T.size
    columns = {"Time": T}
    for i in (1, 2, 4, 5):
        columns[f"Speed{i}"] = speed(T, -0.3 * i)
        columns |= gnss(i, n, ("E", "N", "Lon", "Lat", "Alt"))
    columns["Speed5"] = np.where((T >= 50.0) & (T < 52.0), np.nan, columns["Speed5"])
    columns |= {"IVS1": IVS1, "IVS4": IVS2}
    meta = [
        "Date,8,10,2019",
        "Vehicle_order,SMART_TARGET,BMW_I3,MERCEDES_GLE450,JAGUAR_I_PACE,TESLA_MODELX,",
        "Number_of_vehicles,5",
        "ACC,1",
        "Distance_setting,S",
    ]
    return write_run(path, meta, columns, trailing_comma=True)


def jrc_like(path: Path) -> Path:
    """Empty header flag and no Driver columns: mode unknown."""
    n = T.size
    columns = {"Time": T, "Speed1": speed(T, 0.0)} | gnss(1, n, ("Lat", "Lon", "Alt", "E", "N", "U"))
    columns |= {"Speed2": speed(T, -0.3)} | gnss(2, n, ("Lat", "Lon", "Alt", "E", "N", "U"))
    columns["IVS1"] = IVS1
    meta = ["Date,30,9,2020", "Vehicle_order,Hyundai,Ford", "Number_of_vehicles,2", "ACC,", "Distance_setting,shortest"]
    return write_run(path, meta, columns)


def events_of(run, site: str = "Test", modes=("ACC", "Human", "unknown")):
    stats: Counter = Counter()
    events = [ev for pair in run_to_pairs(run, site, modes) for ev in extract_events(pair, CFG, stats)]
    return events, stats


def test_parse_metadata_and_columns(tmp_path: Path) -> None:
    run = parse_openacc_csv(astazero_like(tmp_path / "ASta_040719_platoon1.csv"))
    assert run.date == "2019-07-04"
    assert run.vehicle_order == ["Audi(A8)", "Audi(A6)", "BMW(X5)"]
    assert (run.n_vehicles, run.acc_flag, run.distance_setting) == (3, 0, "none")
    assert sorted(run.vehicles) == [1, 2, 3]
    assert set(run.vehicles[1]) == {"speed", "e", "n", "driver", "ivs"}
    assert set(run.vehicles[3]) == {"speed", "e", "n", "driver"}
    np.testing.assert_allclose(run.time, T)
    np.testing.assert_allclose(run.vehicles[2]["ivs"], IVS2)

    run = parse_openacc_csv(casale_like(tmp_path / "part3.csv"))
    assert set(run.vehicles[1]) == {"speed", "e", "n", "ivs"}
    assert set(run.vehicles[2]) == {"speed", "e", "n", "driver"}

    run = parse_openacc_csv(zalazone_like(tmp_path / "dynamic_part1.csv"))
    assert run.vehicle_order[-1] == "TESLA_MODELX" and len(run.vehicle_order) == 5
    assert sorted(run.vehicles) == [1, 2, 4, 5]
    assert np.isnan(run.vehicles[5]["speed"][500]) and np.isfinite(run.vehicles[5]["speed"][499])

    run = parse_openacc_csv(jrc_like(tmp_path / "part1.csv"))
    assert run.acc_flag is None and run.distance_setting == "shortest"


def test_ivs_belongs_to_follower_i_plus_1(tmp_path: Path) -> None:
    run = parse_openacc_csv(astazero_like(tmp_path / "ASta_040719_platoon1.csv"))
    assert vehicle_pairs(run) == [1, 2]
    events, _ = events_of(run, "AstaZero")
    by_follower = {ev.meta["vehicle"]: ev for ev in events if ev.meta["t0"] == 0.0}
    first, second = by_follower["Audi(A6)"], by_follower["BMW(X5)"]
    assert first.leader_id == "openacc/AstaZero/Audi(A8)" and first.meta["platoon_index"] == 2
    assert second.leader_id == "openacc/AstaZero/Audi(A6)" and second.meta["platoon_index"] == 3
    np.testing.assert_allclose(first.s, IVS1[: len(first)], atol=1e-9)
    np.testing.assert_allclose(second.s, IVS2[: len(second)], atol=1e-9)
    np.testing.assert_allclose(second.v, speed(T, -0.9)[: len(second)], atol=1e-9)
    np.testing.assert_allclose(second.v_lead, speed(T, -0.6)[: len(second)], atol=1e-9)
    # positions: trapezoidal integral of the leader speed, follower = leader - IVS
    np.testing.assert_allclose(np.diff(first.x_lead), 0.05 * (first.v_lead[1:] + first.v_lead[:-1]), atol=1e-9)
    assert first.follower_id == "openacc/AstaZero/Audi(A6):ACC"
    assert first.event_id == "openacc/AstaZero/Audi(A6):ACC|Audi(A8)|ASta_040719_platoon1|0"
    assert first.meta["driver_mode"] == "ACC" and first.meta["acc_flag"] == 0 and first.meta["date"] == "2019-07-04"
    assert first.meta["a_source"] == "savgol"
    # 100 s runs are cut into two 50 s chunks (max_duration 60 s)
    assert len(events) == 4 and all(len(ev) == 500 for ev in events)


def test_driving_modes(tmp_path: Path) -> None:
    driver = np.where(T < 40.0, "ACC", np.where(T < 70.0, "Human", "ACC")).tolist()
    run = parse_openacc_csv(casale_like(tmp_path / "part3.csv", driver=driver))
    pairs = run_to_pairs(run, "Casale", ("ACC", "Human"))
    assert [p.follower_id for p in pairs] == ["openacc/Casale/Hyundai:ACC", "openacc/Casale/Hyundai:Human"]
    acc, human = pairs
    assert np.all(np.isnan(acc.v[(T >= 40.0) & (T < 70.0)])) and np.all(np.isfinite(acc.v[T < 40.0]))
    assert np.all(np.isnan(human.v[(T < 40.0) | (T >= 70.0)]))
    events, _ = events_of(run, "Casale")
    spans = sorted((ev.meta["driver_mode"], round(ev.meta["t0"], 3), len(ev)) for ev in events)
    assert spans == [("ACC", 0.0, 400), ("ACC", 70.0, 300), ("Human", 40.0, 300)]
    assert {ev.event_id for ev in events} == {
        "openacc/Casale/Hyundai:ACC|Rexton|part3|0",
        "openacc/Casale/Hyundai:ACC|Rexton|part3|1",
        "openacc/Casale/Hyundai:Human|Rexton|part3|0",
    }
    assert run_to_pairs(run, "Casale", ("unknown",)) == []

    # without Driver<i> the header flag decides: 1 ACC, 0 Human, empty or 2 unknown
    zala = parse_openacc_csv(zalazone_like(tmp_path / "dynamic_part1.csv"))
    assert set(sample_modes(zala, 2)) == {"ACC"}
    assert set(sample_modes(dataclasses.replace(zala, acc_flag=0), 2)) == {"Human"}
    assert set(sample_modes(dataclasses.replace(zala, acc_flag=2), 2)) == {"unknown"}
    jrc = parse_openacc_csv(jrc_like(tmp_path / "part1.csv"))
    assert {ev.follower_id for ev in events_of(jrc, "JRC")[0]} == {"openacc/JRC/Ford:unknown"}
    # an existing Driver column wins over the header flag (AstaZero: flag 0, samples ACC)
    asta = parse_openacc_csv(astazero_like(tmp_path / "ASta_040719_platoon1.csv"))
    assert set(sample_modes(asta, 2)) == {"ACC"}


def test_index_holes_and_empty_cells(tmp_path: Path) -> None:
    run = parse_openacc_csv(zalazone_like(tmp_path / "dynamic_part1.csv"))
    assert vehicle_pairs(run) == [1, 4]
    events, _ = events_of(run, "ZalaZone")
    pairs = {(ev.meta["leader_vehicle"], ev.meta["vehicle"], ev.meta["platoon_index"]) for ev in events}
    assert pairs == {("SMART_TARGET", "BMW_I3", 2), ("JAGUAR_I_PACE", "TESLA_MODELX", 5)}
    tesla = sorted((round(ev.meta["t0"], 3), len(ev)) for ev in events if ev.meta["vehicle"] == "TESLA_MODELX")
    assert tesla == [(0.0, 500), (52.0, 480)]  # the empty Speed5 cells (50-52 s) end an event


def test_time_gap_gives_separate_events(tmp_path: Path) -> None:
    t = np.round(np.r_[np.arange(0.0, 30.0, 0.1), np.arange(60.0, 100.0, 0.1)], 1)
    run = parse_openacc_csv(casale_like(tmp_path / "part3.csv", t=t, ivs=20.0 + np.sin(0.1 * t)))
    events, _ = events_of(run)
    assert sorted((round(ev.meta["t0"], 3), len(ev)) for ev in events) == [(0.0, 300), (60.0, 400)]


def test_negative_ivs_never_reaches_an_event(tmp_path: Path) -> None:
    ivs = np.where((T >= 30.0) & (T < 31.0), -0.5, IVS1)
    path = casale_like(tmp_path / "Casale" / "part3.csv", ivs=ivs)
    casale_like(tmp_path / "Casale" / "part4.csv")
    (tmp_path / "Casale" / "Veh_specifications.csv").write_text("Vehicles,Max power (kW)\nRexton,133\n", encoding="utf-8")
    assert [p.name for p in data_files(tmp_path / "Casale")] == ["part3.csv", "part4.csv"]
    stats: Counter = Counter()
    events = build_events({"raw_dir": str(tmp_path), "campaigns": ["Casale"]}, CFG, stats)
    events.validate()
    first = sorted((round(ev.meta["t0"], 3), len(ev)) for ev in events if ev.meta["file"] == path.name)
    assert first == [(0.0, 300), (31.0, 345), (65.5, 345)]
    assert min(ev.s.min() for ev in events) > 0.0
    assert stats["files"] == 2 and stats["pairs"] == 2 and stats["nonpositive_ivs_samples"] == 10
    assert stats["events_site_Casale"] == len(events) == stats["events_mode_ACC"] == 5


RAW = resolve_path("data/raw/openacc")


@pytest.mark.data
@pytest.mark.skipif(not RAW.is_dir(), reason="OpenACC not downloaded")
def test_real_files_parse() -> None:
    n_files = 0
    for folder in CAMPAIGN_DIRS.values():
        for path in data_files(RAW / folder):
            run = parse_openacc_csv(path)
            n_files += 1
            assert run.time.size > 0 and np.all(np.diff(run.time) > 0), path
            assert max(run.vehicles) <= len(run.vehicle_order), path
    assert n_files == 139


@pytest.mark.data
@pytest.mark.skipif(not (RAW / "Casale").is_dir(), reason="OpenACC not downloaded")
def test_real_casale_build_events() -> None:
    stats: Counter = Counter()
    events = build_events({"raw_dir": "data/raw/openacc", "campaigns": ["Casale"]}, CFG, stats)
    events.validate()
    assert len(events) > 0 and stats["files"] == 9
    assert {ev.site for ev in events} == {"Casale"}
