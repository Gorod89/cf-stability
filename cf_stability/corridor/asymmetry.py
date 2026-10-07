"""Acceleration asymmetry and oscillation spectrum of the corridor runs and ground truths (docs/m8_contract.md,
section 9; D121).

Per run (``trajectories.npz``: one row per vehicle and whole second inside the section) and per ground truth
(``ground_truth.npz`` of the scenario), inside the analysis window of the scenario and on the lanes of its wave
field (``macro.waves.lanes``, the main lanes: I-80 1-6, US-101 1-5, as the speed field of the waves):

* **Acceleration asymmetry.** The acceleration of a vehicle over a whole second is the difference of its speeds at
  two consecutive whole seconds, ``a = v(t + 1) - v(t)`` (the mean acceleration over that second; the files hold
  speeds at whole seconds only), for the pairs whose two samples lie inside the window, on a lane of the field
  (the lane of the first sample). Asymmetry index = mean of ``a`` over the samples with ``a > threshold`` divided
  by the mean of ``|a|`` over the samples with ``a < -threshold`` (``threshold`` 0.1 m/s^2); the shares of the
  samples (vehicle-seconds) spent accelerating (``a > threshold``), decelerating (``a < -threshold``) and in
  between.
* **Oscillation spectrum.** The speed series of every virtual detector of the metrics (``macro.detectors`` with
  the throughput and queue detectors, as in ``macro.json``): the mean speed of the vehicles passing it on the
  lanes of the field per interval of ``sample_s`` (2 s) inside the window (passages linear between the 1 Hz
  samples, as the detectors of the metrics); an interval without a passage takes the value interpolated linearly
  in time between its neighbours (the nearest one at the ends), and a detector with passages in fewer than
  ``min_coverage`` of the intervals is left out. Welch power spectral density of every series (``scipy.signal.
  welch``: Hann window of ``window_s`` = 128 s, i.e. 64 samples, half overlap, mean removed per segment, density
  scaling, (m/s)^2/Hz), averaged over the detectors used; summarised in the band ``f_min``-``f_max``
  (0.002-0.05 Hz) by the frequency of its peak, the spectral centroid ``sum f P / sum P`` and the power in the
  band (``sum P df``, the speed variance of the band, with its root). With 2 s samples and 128 s segments the
  frequencies are multiples of 1/128 Hz: the band holds the six lines 0.0078-0.0469 Hz (periods 128-21 s).

``asymmetry.json`` is written next to ``macro.json`` (runs: ``<corridor_root>/<scenario>/<law>/seed<s>/``;
ground truths: ``<corridor_root>/scenarios/<scenario>/``). A file that holds the hash of its configuration and
is newer than its inputs (the two npz files and ``scenario.json``) is kept. This module never imports libsumo.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from cf_stability.corridor.macro import (
    Geometry,
    MacroConfig,
    MacroItem,
    crossings,
    find_items,
    geometry_from_scenario,
    interval_edges,
    load_npz,
    prepare_trajectories,
    scenario_directory,
    scenario_macro_config,
)
from cf_stability.utils import config_hash, read_json, write_json

# version of the definitions, hashed with the parameters: raise it when the code changes a quantity
ASYMMETRY_VERSION = 1
FILE = "asymmetry.json"


@dataclasses.dataclass(frozen=True)
class AsymmetryConfig:
    """Parameters of D121 (``configs/corridor_metrics.yaml``, ``asymmetry``)."""

    threshold: float = 0.1  # m/s^2: accelerating above it, decelerating below its negative
    sample_s: float = 2.0  # s: intervals of the speed series of the detectors
    window_s: float = 128.0  # s: Welch segment (64 samples of 2 s)
    f_min: float = 0.002  # Hz: band of the summary
    f_max: float = 0.05
    min_coverage: float = 0.5  # share of the intervals with a passage for a detector to enter the mean spectrum

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AsymmetryConfig:
        values = dict(raw or {})
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown asymmetry keys: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------------------- acceleration


def acceleration_samples(
    trajectories: Mapping[str, Any], window: Sequence[float], lanes: Sequence[int] | None = None
) -> np.ndarray:
    """Accelerations over the whole seconds ``[t, t + 1]`` of every vehicle (``v(t + 1) - v(t)``, m/s^2) whose two
    samples lie inside ``window`` and whose first sample is on one of ``lanes`` (all lanes when None)."""
    t = np.asarray(trajectories.get("t", np.zeros(0)), dtype=np.float64).reshape(-1)
    v = np.asarray(trajectories.get("v", np.zeros(0)), dtype=np.float64).reshape(-1)
    veh = np.asarray(trajectories.get("vehicle", np.zeros(0)), dtype=np.int64).reshape(-1)
    lane = np.asarray(trajectories.get("lane", np.zeros(len(t))), dtype=np.int64).reshape(-1)
    if len(t) < 2:
        return np.zeros(0)
    order = np.lexsort((t, veh))
    t, v, veh, lane = t[order], v[order], veh[order], lane[order]
    w0, w1 = (float(w) for w in window)
    keep = (veh[1:] == veh[:-1]) & (np.abs(t[1:] - t[:-1] - 1.0) < 1e-6)
    keep &= (t[:-1] >= w0) & (t[1:] <= w1) & np.isfinite(v[1:]) & np.isfinite(v[:-1])
    if lanes is not None:
        keep &= np.isin(lane[:-1], np.asarray(list(lanes), dtype=np.int64))
    return (v[1:] - v[:-1])[keep]


def asymmetry_of(a: np.ndarray, threshold: float = 0.1) -> dict[str, Any]:
    """Asymmetry index (mean acceleration above ``threshold`` over the mean magnitude of the decelerations below
    ``-threshold``) and the shares of the samples accelerating, decelerating and in between; None where a value
    cannot be formed."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = len(a)
    up, down = a[a > threshold], a[a < -threshold]
    mean_up = float(up.mean()) if len(up) else None
    mean_down = float(-down.mean()) if len(down) else None
    index = mean_up / mean_down if mean_up is not None and mean_down else None
    return {
        "n_samples": int(n),
        "threshold": float(threshold),
        "asymmetry_index": index,
        "mean_acceleration": mean_up,
        "mean_deceleration": mean_down,
        "accelerating_share": len(up) / n if n else None,
        "decelerating_share": len(down) / n if n else None,
        "cruising_share": (n - len(up) - len(down)) / n if n else None,
        "acceleration_p90": float(np.quantile(up, 0.9)) if len(up) else None,
        "deceleration_p90": float(np.quantile(-down, 0.9)) if len(down) else None,
    }


# ------------------------------------------------------------------------------------------- spectrum


def detector_speed_series(
    prepared: Mapping[str, np.ndarray], x: float, t_edges: np.ndarray, lanes: Sequence[int] | None = None
) -> tuple[np.ndarray, float, int]:
    """Mean speed of the vehicles passing ``x`` (on ``lanes``) per interval of ``t_edges``; intervals without a
    passage filled by linear interpolation in time (the nearest value at the ends). ``(series, coverage,
    passages)``; the series is all NaN without any passage."""
    n = len(t_edges) - 1
    c = crossings(prepared, x)
    keep = np.ones(len(c["t"]), dtype=bool)
    if lanes is not None:
        keep = np.isin(c["lane"], np.asarray(list(lanes), dtype=np.int64))
    tc, vc = c["t"][keep], c["v"][keep]
    idx = np.searchsorted(t_edges, tc, side="right") - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(vc)
    count = np.bincount(idx[ok], minlength=n)[:n] if n > 0 else np.zeros(0, np.int64)
    total = np.bincount(idx[ok], weights=vc[ok], minlength=n)[:n] if n > 0 else np.zeros(0)
    has = count > 0
    series = np.full(n, np.nan)
    if has.any():
        centres = 0.5 * (t_edges[:-1] + t_edges[1:])
        series = np.interp(centres, centres[has], total[has] / count[has])
    return series, float(has.mean()) if n else 0.0, int(ok.sum())


def welch_psd(series: np.ndarray, sample_s: float, window_s: float) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Welch power spectral density of a series of ``sample_s`` samples: Hann window of ``window_s``, half overlap,
    the mean of every segment removed, density scaling; ``(frequency (Hz), psd, segments)``, None when the series
    is shorter than one segment."""
    from scipy import signal

    nperseg = int(round(window_s / sample_s))
    series = np.asarray(series, dtype=np.float64)
    if nperseg < 2 or len(series) < nperseg or not np.isfinite(series).all():
        return None
    noverlap = nperseg // 2
    f, p = signal.welch(series, fs=1.0 / sample_s, window="hann", nperseg=nperseg, noverlap=noverlap,
                        detrend="constant", scaling="density", average="mean")  # fmt: skip
    segments = 1 + (len(series) - nperseg) // (nperseg - noverlap)
    return f, p, int(segments)


def band_summary(f: np.ndarray, p: np.ndarray, f_min: float, f_max: float) -> dict[str, Any]:
    """Peak frequency, spectral centroid and power of the spectrum ``p`` over ``f_min <= f <= f_max``."""
    f, p = np.asarray(f, dtype=np.float64), np.asarray(p, dtype=np.float64)
    band = (f >= f_min) & (f <= f_max) & np.isfinite(p)
    out: dict[str, Any] = {"n_lines": int(band.sum()), "peak_frequency": None, "peak_period_s": None,
                           "centroid": None, "band_power": None, "band_rms": None}  # fmt: skip
    if not band.any():
        return out
    fb, pb = f[band], p[band]
    df = float(f[1] - f[0]) if len(f) > 1 else 0.0
    peak = float(fb[int(np.argmax(pb))])
    power = float(pb.sum() * df)
    out.update(peak_frequency=peak, peak_period_s=1.0 / peak if peak > 0 else None,
               centroid=float((fb * pb).sum() / pb.sum()) if pb.sum() > 0 else None, band_power=power,
               band_rms=math.sqrt(power) if power >= 0 else None)  # fmt: skip
    return out


def spectrum_of(
    prepared: Mapping[str, np.ndarray], detectors: Sequence[float], window: Sequence[float],
    lanes: Sequence[int] | None, cfg: AsymmetryConfig,
) -> dict[str, Any]:  # fmt: skip
    """Mean Welch spectrum of the speed series of ``detectors`` inside ``window`` and its band summary."""
    w0, w1 = (float(w) for w in window)
    t_edges = interval_edges(w0, w1, cfg.sample_s)
    per_detector, spectra = {}, []
    frequency: np.ndarray | None = None
    segments = 0
    for x in detectors:
        series, coverage, passages = detector_speed_series(prepared, float(x), t_edges, lanes)
        used = coverage >= cfg.min_coverage and np.isfinite(series).all()
        psd = welch_psd(series, cfg.sample_s, cfg.window_s) if used else None
        entry: dict[str, Any] = {"coverage": coverage, "passages": passages, "used": psd is not None}
        if psd is not None:
            frequency, p, segments = psd
            spectra.append(p)
            entry.update(band_summary(frequency, p, cfg.f_min, cfg.f_max))
            entry["mean_speed"] = float(series.mean())
            entry["std_speed"] = float(series.std())
        per_detector[f"{float(x):g}"] = entry
    out: dict[str, Any] = {
        "sample_s": cfg.sample_s, "window_s": cfg.window_s, "band_hz": [cfg.f_min, cfg.f_max],
        "n_samples": int(len(t_edges) - 1), "n_segments": segments, "detectors": per_detector,
        "n_detectors": len(spectra), "frequency": None, "psd": None,
    }  # fmt: skip
    if spectra and frequency is not None:
        mean = np.mean(spectra, axis=0)
        out.update(frequency=[float(v) for v in frequency], psd=[float(v) for v in mean])
        out.update(band_summary(frequency, mean, cfg.f_min, cfg.f_max))
    else:
        out.update(band_summary(np.zeros(0), np.zeros(0), cfg.f_min, cfg.f_max))
    return out


# ---------------------------------------------------------------------------------------------- items


def detector_positions(macro_cfg: MacroConfig) -> list[float]:
    """The detectors of ``macro.json``: ``detectors`` with the throughput and the queue detector."""
    return sorted({*macro_cfg.detectors, macro_cfg.throughput_x, macro_cfg.queue_x})


def asymmetry_metrics(
    trajectories: Mapping[str, Any], vehicles: Mapping[str, Any] | None, geometry: Geometry, macro_cfg: MacroConfig,
    cfg: AsymmetryConfig,
) -> dict[str, Any]:  # fmt: skip
    """Content of ``asymmetry.json`` without the header and the hash: acceleration asymmetry and spectrum."""
    window = [float(w) for w in geometry.window]
    lanes = [int(lane) for lane in macro_cfg.waves.lanes] if macro_cfg.waves.lanes else None
    accel = asymmetry_of(acceleration_samples(trajectories, window, lanes), cfg.threshold)
    prepared = prepare_trajectories(trajectories, vehicles, geometry, macro_cfg)
    spectrum = spectrum_of(prepared, detector_positions(macro_cfg), window, lanes, cfg)
    return {"window": window, "lanes": lanes, "acceleration": accel, "spectrum": spectrum}


def hashed(cfg: AsymmetryConfig, macro_cfg: MacroConfig) -> dict[str, Any]:
    """The configuration of an ``asymmetry.json``: the parameters, the detectors and lanes, and the version."""
    return {**cfg.as_dict(), "detectors": detector_positions(macro_cfg), "lanes": list(macro_cfg.waves.lanes),
            "max_gap_s": macro_cfg.max_gap_s, "boundary_points": macro_cfg.boundary_points,
            "boundary_max_gap_s": macro_cfg.boundary_max_gap_s, "version": ASYMMETRY_VERSION}  # fmt: skip


def _inputs(item: MacroItem, corridor_root: Path) -> list[Path]:
    return [*item.data_files(), scenario_directory(corridor_root, item.scenario) / "scenario.json"]


def _scenario(corridor_root: Path, scenario: str) -> dict[str, Any]:
    path = scenario_directory(corridor_root, scenario) / "scenario.json"
    return read_json(path) if path.exists() else {}


def _settings(
    item: MacroItem, corridor_root: Path, macro_cfg: MacroConfig, default: Geometry | None
) -> tuple[Geometry, MacroConfig]:
    scenario = _scenario(corridor_root, item.scenario)
    return geometry_from_scenario(scenario, default), scenario_macro_config(macro_cfg, scenario)


def is_current(item: MacroItem, corridor_root: Path, hash_: str, window: Sequence[float]) -> bool:
    """``asymmetry.json`` holds ``hash_`` and the analysis window of the scenario now, and is not older than any of
    its inputs."""
    path = item.directory / FILE
    if not path.exists():
        return False
    try:
        payload = read_json(path)
        if payload.get("config_hash") != hash_ or [float(w) for w in payload.get("window") or []] != list(window):
            return False
    except (OSError, ValueError, UnicodeDecodeError, AttributeError, TypeError):
        return False
    written = path.stat().st_mtime
    return all(p.stat().st_mtime <= written for p in _inputs(item, corridor_root) if p.exists())


def item_asymmetry(
    item: MacroItem, corridor_root: str | Path, macro_cfg: MacroConfig, cfg: AsymmetryConfig,
    default: Geometry | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Content of the ``asymmetry.json`` of a run or ground truth."""
    root = Path(corridor_root)
    geometry, scenario_cfg = _settings(item, root, macro_cfg, default)
    trajectories_file, vehicles_file = item.data_files()
    metrics = asymmetry_metrics(load_npz(trajectories_file), load_npz(vehicles_file), geometry, scenario_cfg, cfg)
    config = hashed(cfg, scenario_cfg)
    header = {"kind": "truth" if item.is_truth else "run", "scenario": item.scenario, "law": item.law,
              "seed": item.seed}  # fmt: skip
    return {**header, **metrics, "config": config, "config_hash": config_hash(config)}


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def summary_line(item: MacroItem, payload: Mapping[str, Any], seconds: float) -> str:
    accel, spectrum = payload.get("acceleration") or {}, payload.get("spectrum") or {}

    def number(value: Any, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    return (f"OK {item.label}: asymmetry {number(accel.get('asymmetry_index'), '.3f')} (accelerating "
            f"{number(accel.get('accelerating_share'), '.3f')}, decelerating {number(accel.get('decelerating_share'), '.3f')}"
            f", {accel.get('n_samples', 0)} samples), peak {number(spectrum.get('peak_frequency'), '.4f')} Hz, centroid "
            f"{number(spectrum.get('centroid'), '.4f')} Hz, {spectrum.get('n_detectors', 0)} detectors ({seconds:.1f} s) "
            f"-> {item.directory / FILE}")  # fmt: skip


def _work(item: MacroItem, root: str, macro_raw: Mapping[str, Any], cfg_raw: Mapping[str, Any],
          default: tuple[float, float, tuple[float, float]] | None) -> str:  # fmt: skip
    """One file (in a worker process); the printed line."""
    start = time.perf_counter()
    try:
        geometry = Geometry(default[0], default[1], tuple(default[2])) if default else None
        payload = item_asymmetry(item, root, MacroConfig.from_mapping(macro_raw), AsymmetryConfig.from_mapping(cfg_raw),
                                 geometry)  # fmt: skip
        _write_atomic(item.directory / FILE, payload)
    except Exception as exc:  # one broken file must not stop the others
        message = str(exc).splitlines()[0] if str(exc) else ""
        return f"FAILED {item.label}: {type(exc).__name__}: {message}"
    return summary_line(item, payload, time.perf_counter() - start)


def update_asymmetry(
    corridor_root: str | Path, macro_cfg: MacroConfig, cfg: AsymmetryConfig, default: Geometry | None = None, *,
    scenario: str | None = None, law: str | None = None, seed: int | None = None, force: bool = False,
    workers: int = 1,
) -> Iterator[str]:  # fmt: skip
    """Write the ``asymmetry.json`` of every selected ground truth and run (``find_items`` of the metrics); one line
    per file (``OK``, ``KEPT``, ``SKIPPED``, ``FAILED``). With ``workers`` > 1 the files are computed in a pool of
    processes; a failing file does not stop the others."""
    root = Path(corridor_root)
    todo: list[MacroItem] = []
    for item in find_items(root, scenario, law, seed):
        missing = [path.name for path in item.data_files() if not path.exists()]
        if not item.is_truth and not missing and not (item.directory / "run.json").exists():
            missing.append("run.json")
        if missing:
            yield f"SKIPPED {item.label}: {', '.join(missing)} missing"
            continue
        try:
            geometry, scenario_cfg = _settings(item, root, macro_cfg, default)
            hash_ = config_hash(hashed(cfg, scenario_cfg))
            if not force and is_current(item, root, hash_, [float(w) for w in geometry.window]):
                yield f"KEPT {item.label}: {item.directory / FILE} is up to date"
                continue
        except Exception as exc:
            yield f"FAILED {item.label}: {type(exc).__name__}: {exc}"
            continue
        todo.append(item)
    default_tuple = None if default is None else (default.x_in, default.x_out, tuple(default.window))
    macro_raw = macro_cfg.as_dict()
    if workers <= 1 or len(todo) <= 1:
        for item in todo:
            yield _work(item, str(root), macro_raw, cfg.as_dict(), default_tuple)
        return
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_work, item, str(root), macro_raw, cfg.as_dict(), default_tuple) for item in todo]
        for future in as_completed(futures):
            yield future.result()
