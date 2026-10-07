"""Tables of the corridor (docs/m5_contract.md, section 6; D100; docs/m7_contract.md, D108-D114).

:func:`make_corridor_tables` reads the ``macro.json`` of every run of the design
(``<corridor_root>/<scenario>/<law>/seed<s>/``) and of the ground truths
(``<corridor_root>/scenarios/<scenario>/``), the laws (``<corridor_root>/laws/<law>.json`` and ``.npz``)
and the ``stability.json`` of their member runs, and writes to ``out_dir``:

* ``laws.md|csv``: per corridor and law the mean over its runs (unit: run = scenario and seed) of the
  macro error, of the dynamic macro error, of their components and of the raw metrics, with percentile
  bootstrap intervals over the runs, the collisions and the insertion share; the ground-truth values per
  scenario.
* ``components.md|csv``: per corridor and law the eight components, the anchored and the dynamic ones
  marked (M6), with the macro error and the dynamic macro error.
* ``instability.md|csv`` and ``instability_correlation.csv``: the unstable-equilibrium fraction of the
  members of every law and its mean macro errors per corridor (all components and dynamic); Spearman and
  Pearson correlation over the laws with bootstrap intervals (resampling the laws) and permutation p-values
  (D108), with all laws and without the laws that collide, for both errors.
* ``tost.md|csv``: per corridor the candidate law against the reference, paired by scenario and seed:
  relative difference of every raw metric and the two one-sided tests for the relative margin.
* ``h12_1.md|csv``, ``h12_2.md|csv``, ``verdicts.md|csv``: the verdicts of H12.1-H12.3 (D108): the
  degradations of the macro triple of the pure networks against the IDM, the micro comparison of the
  certified hybrid with the IDM on the NGSIM test parts (read with the table code of M4), and the verdict
  rows of the three hypotheses (with the TOST and the correlations).
* ``e4_rmax.md|csv``: the residual-amplitude sweep of the certified hybrid (D111): certificates, shares
  and RMSE of its training runs and its corridor numbers.
* ``sensitivity.md|csv``: the variants of one scenario (D113): macro error and the macro triple of three
  laws per variant and their ranking.
* ``missing.txt``: every missing run, file or value. Nothing missing raises: it leaves an empty cell.

Corridors (D108): the corridor of a scenario is named by the prefix of the scenario name (``i80_p1`` ->
``i80`` -> ``I-80``, ``corridors``). The corridor of the first scenario is always in the tables; any other
one enters when a ground truth of its scenarios has a ``macro.json``, and otherwise gets one line in
missing.txt and the verdicts name the corridors they use ("I-80 only").

Laws of M8 (docs/m8_contract.md, sections 5 and 7): ``residual_idm_certified_het`` (D118) is a law of the
design (laws, components, instability with the exact gain of its cores, the correlation of H12.3); the temporal
hold-out laws ``*_p0`` (D120) run on periods 1 and 2 of I-80 only (``law_scenarios``) and have rows in the laws,
components and instability tables but stay out of the correlation and the verdicts. Both are optional: a law of
``optional_laws`` without a law file and without runs is left out of every table with one note.

:func:`make_m8_corridor_tables` writes the corridor tables of M8 to ``m8_dir`` (``runs/_tables/m8``), with the
notes of their missing inputs in ``missing_corridor.txt``:

* ``asymmetry.md|csv`` (D121): per corridor and law the means over its runs of the acceleration asymmetry index,
  the shares of time accelerating and decelerating, the peak frequency, the spectral centroid and the band RMS of
  the detector speed spectrum (``asymmetry.json``, ``cf_stability/corridor/asymmetry.py``) with intervals over the
  runs, the signed relative errors of the index, of the peak frequency and of the centroid against the ground
  truth of the run's scenario, and the ground truth rows; ``asymmetry_contrasts.md|csv``: the penalised and the
  certified laws against their free counterparts, paired by scenario and seed.
* ``correlation_pooled.md|csv`` (D122): H12.3 over every law with runs (``pooled_laws``: the laws of the design and
  the residual amplitudes of D111) per corridor and pooled with the corridor as a stratum (ranks within the corridor
  scaled to (0, 1), Pearson of the pooled ranks; bootstrap over the laws within the corridors; permutations within
  the corridors).
* ``power.md|csv`` (D122): the spread over seeds within a scenario of the macro error, the throughput and the travel
  time of two laws and the seeds per scenario that detect a relative difference between them (paired by seed).
* ``temporal.md|csv`` (D120): the laws fine-tuned on period 0 against the same laws of D97 on periods 1-2.

The review of M8 adds three tables to ``m8_dir`` (same notes file) and the rows of a factorial ablation:

* ``correlation_clustered.md|csv``: H12.3 over the rows of correlation_pooled as dependent units: percentile intervals
  of a cluster bootstrap over the laws (the two corridor rows of a law move together) and over the architecture families
  (``families``), the permutation of the instability of whole laws (linked across the corridors), leave-one-family-out
  and the learned laws only (without ``physics_families``); per corridor and pooled, for both macro errors
  (``cluster_resamples``, ``cluster_seed``; ``cf_stability.eval.stats.clustered_spearman``).
* ``contacts_absolute.md|csv``: per corridor and law the contact episodes and vehicles in contact per run with their
  exposure (vehicle-km and vehicles of the window, against the ground truth) and the demand (inserted share, shortfall
  in the window and over the run, insertion delay; ``run.json``), the rate per 1000 vehicle-km as the mean of the runs
  and pooled.
* ``h12_2_error.md|csv``: exploratory, no verdict: |e| of the candidate minus |e| of the reference for every component
  of the macro-error vector and both macro errors, paired by scenario and seed, with intervals and Wilcoxon p-values.
* ``ablation_laws`` (``idm_core_margin``, ``idm_margin_i80``, ``residual_idm_margin_free_r0.3``): optional laws with rows
  in laws, components, instability, asymmetry and contacts_absolute, and rows of ``rmax_ablation`` in e4_rmax; not in
  the correlations nor the verdicts.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from cf_stability.corridor.macro import ANCHORED, COMPONENTS, DYNAMIC, macro_error_dynamic
from cf_stability.eval.collect import collect_run
from cf_stability.eval.stats import (
    bootstrap_ci, clustered_spearman, holm, minimal_detectable, paired_comparison, paired_t_power, pooled_sd,
    seeds_needed, stratified_permutation_p, stratified_spearman_ci, tost_relative,
)  # fmt: skip
from cf_stability.eval.tables import (
    AUDIT, CERTIFICATE, EVENTS, METRICS, Column, Table, TableMaker, TablesConfig, _column, markdown,
)  # fmt: skip
from cf_stability.utils import REPO_ROOT, read_json

HET_LAW = "residual_idm_certified_het"  # D118: certified per-event cores with the residual of a fold member
LAWS = (
    "idm_global", "idm_heterogeneous", "knn", "mlp", "pidl", "gru", "lstm", "perl", "residual_idm",
    "residual_idm_certified", "mlp_penalty", "gru_penalty", "lstm_penalty", "idm_heterogeneous_all", HET_LAW,
)  # fmt: skip
TEMPORAL_LAWS = ("idm_global_p0", "residual_idm_certified_p0", "mlp_p0", "gru_p0")  # D120: fine-tuned on period 0
TEMPORAL_PAIRS = {  # D120: the law of period 0 -> the same law of D97 (fine-tuned on all periods)
    "idm_global_p0": "idm_global", "residual_idm_certified_p0": "residual_idm_certified", "mlp_p0": "mlp",
    "gru_p0": "gru",
}  # fmt: skip
TEMPORAL_SCENARIOS = ("i80_p1", "i80_p2")
ASYMMETRY_CONTRASTS = (  # D121: penalised or certified law -> the free law it is compared with
    ("mlp_penalty", "mlp"), ("gru_penalty", "gru"), ("lstm_penalty", "lstm"), ("residual_idm_certified", "residual_idm"),
    ("residual_idm_certified", "residual_idm_free_r0.3"),
)  # fmt: skip
POWER_METRICS = {"macro_error": ("macro error", 3), "throughput_vph": ("throughput (veh/h)", 0),
                 "travel_time_mean": ("travel time mean (s)", 1)}  # fmt: skip
M8_TABLES = ("asymmetry", "asymmetry_contrasts", "correlation_pooled", "power", "temporal")
ASYMMETRY_VALUES = {  # flat key of a run's asymmetry values: (header, digits)
    "asymmetry_index": ("asymmetry index", 3), "accelerating_share": ("accelerating share", 3),
    "decelerating_share": ("decelerating share", 3), "mean_acceleration": ("mean acceleration (m/s^2)", 3),
    "mean_deceleration": ("mean deceleration (m/s^2)", 3), "peak_frequency": ("peak frequency (Hz)", 4),
    "centroid": ("spectral centroid (Hz)", 4), "band_rms": ("band RMS (m/s)", 3),
}  # fmt: skip
ASYMMETRY_ERRORS = {"asymmetry_index": "index", "peak_frequency": "peak frequency", "centroid": "centroid"}
RMAX_LAWS = (  # D111: in the laws and components tables and in e4_rmax, not in the instability correlation
    "residual_idm_certified_r0.1", "residual_idm_certified_r0.2", "residual_idm_certified_r0.5",
    "residual_idm_free_r0.3",
)  # fmt: skip
RMAX = (  # D111: one row per r_max: the run of HighD, its fine-tuning on NGSIM I-80 and the corridor law
    {"r_max": 0.1, "core": "certified", "highd": "e4_stable_r0.1", "ngsim": "e4_stable_ft_r0.1",
     "law": "residual_idm_certified_r0.1"},
    {"r_max": 0.2, "core": "certified", "highd": "e4_stable_r0.2", "ngsim": "e4_stable_ft_r0.2",
     "law": "residual_idm_certified_r0.2"},
    {"r_max": 0.3, "core": "certified", "highd": "e4_stable", "ngsim": "e4_stable_ft", "law": "residual_idm_certified"},
    {"r_max": 0.5, "core": "certified", "highd": "e4_stable_r0.5", "ngsim": "e4_stable_ft_r0.5",
     "law": "residual_idm_certified_r0.5"},
    {"r_max": 0.3, "core": "free", "highd": "e4_free_r0.3", "ngsim": "e4_free_r0.3_ft", "law": "residual_idm_free_r0.3"},
    {"r_max": 1.0, "core": "free", "highd": "e1", "ngsim": "e4_free_ft", "law": "residual_idm"},
)  # fmt: skip
# review of M8: the factorial ablation of the certified hybrid (the core alone, an I-80 core with the margin, the margin
# core with a residual without certificate): rows of laws, components, instability, asymmetry and e4_rmax, optional
# (in the tables once a law file or a run exists), not in the correlations nor the verdicts of H12
ABLATION_LAWS = ("idm_core_margin", "idm_margin_i80", "residual_idm_margin_free_r0.3")
ABLATION_TEXT = {  # the note of the tables on every law of the ablation
    "idm_core_margin": "the IDM cores of the fold members of residual_idm_certified (margin 0.2, calibrated on "
                       "follownet_highd) with the residual switched off, one core per member drawn per vehicle as "
                       "residual_idm_certified draws its members",
    "idm_margin_i80": "the global IDM of ngsim_i80 calibrated per fold with the stability margin 0.2",
    "residual_idm_margin_free_r0.3": "the margin core with a residual of r_max 0.3 without certificate, fine-tuned on "
                                     "ngsim_i80",
}  # fmt: skip
RMAX_ABLATION = (  # extra rows of e4_rmax; r_max 0: no residual; highd/ngsim None: no training runs of the row
    {"r_max": 0.0, "core": "certified core alone", "highd": None, "ngsim": None, "law": "idm_core_margin"},
    {"r_max": 0.0, "core": "I-80 margin core", "highd": None, "ngsim": None, "law": "idm_margin_i80"},
    {"r_max": 0.3, "core": "margin, no certificate", "highd": "e4_margin_free_r0.3", "ngsim": "e4_margin_free_r0.3_ft",
     "law": "residual_idm_margin_free_r0.3"},
)  # fmt: skip
FAMILIES = {  # review of M8: the architecture families, the clusters of correlation_clustered (a law of none: its own)
    "IDM": ("idm_global", "idm_heterogeneous", "idm_heterogeneous_all", "idm_core_margin", "idm_margin_i80"),
    "ResidualIDM": ("residual_idm", "residual_idm_certified", "residual_idm_certified_r0.1", "residual_idm_certified_r0.2",
                    "residual_idm_certified_r0.5", "residual_idm_free_r0.3", HET_LAW, "residual_idm_margin_free_r0.3"),
    "k-NN": ("knn",), "MLP": ("mlp", "mlp_penalty"), "PIDL": ("pidl",), "GRU": ("gru", "gru_penalty"),
    "LSTM": ("lstm", "lstm_penalty"), "PERL": ("perl",),
}  # fmt: skip
PHYSICS_FAMILIES = ("IDM", "ResidualIDM")  # left out of the subset "learned laws only"
M9_TABLES = ("correlation_clustered", "contacts_absolute", "h12_2_error")  # review of M8: in m8_dir, after M8_TABLES
EXPOSURE_KEYS = (  # per run: macro.json (window) and run.json (whole run) values of the exposure and of the demand
    "vehicle_km", "n_vehicles_window", "n_planned_window", "n_vehicles", "run_n_planned", "run_n_inserted",
    "run_depart_delay_s",
)  # fmt: skip
RAW_METRICS: dict[str, tuple[str, int]] = {  # flat key of a macro.json: (header with unit, digits)
    "throughput_vph": ("throughput (veh/h)", 0),
    "mean_speed": ("mean speed (m/s)", 2),
    "queue_discharge_flow": ("queue discharge (veh/h)", 0),
    "flow_peak_2min": ("peak 2-min flow (veh/h)", 0),
    "capacity_drop": ("capacity drop", 3),
    "congested_share": ("congested share", 3),
    "fd_scatter": ("FD scatter (veh/h)", 0),
    "n_waves": ("waves", 1),
    "wave_speed_xcorr": ("wave speed, cross-correlation (m/s)", 2),
    "wave_speed": ("wave speed, leading edges (m/s)", 2),
    "wave_amplitude": ("wave amplitude (m/s)", 2),
    "travel_time_mean": ("travel time mean (s)", 1),
    "travel_time_median": ("travel time median (s)", 1),
    "collisions_per_1000_vkm": ("collisions / 1000 veh-km", 3),
    "vehicles_in_contact_share": ("vehicles in contact", 3),
    "inserted_share": ("inserted share", 3),
    "mean_depart_delay_s": ("insertion delay (s)", 2),
}
COMPONENT_HEADERS = {
    "throughput": "throughput", "mean_speed": "mean speed", "queue_discharge_flow": "queue discharge",
    "fd": "FD", "wave_speed": "wave speed (xcorr)", "n_waves": "waves", "wave_amplitude": "wave amplitude",
    "travel_time": "travel time (W1)",
}  # fmt: skip
# D108: the macro triple (one anchored flow quantity, one delay quantity, one dynamic quantity): component of
# the macro-error vector, raw metric of the TOST, and the side of the relative difference that is worse
TRIPLE = {
    "throughput": ("throughput_vph", "lower"), "travel_time": ("travel_time_mean", "longer"),
    "wave_speed": ("wave_speed_xcorr", "slower"),
}  # fmt: skip
DIRECTIONS = {"lower": ("higher", "lower"), "longer": ("longer", "shorter"), "slower": ("faster", "slower")}
MEASURES = {"unstable_eq": "unstable among equilibria", "not_stable": "band share not stable"}
ERRORS = {"macro_error": "macro error", "macro_error_dynamic": "macro error (dynamic)"}
SENSITIVITY_VARIANTS = ("gain0.01", "gain0.04", "nofeedback", "lc_low", "lc_high")


def _plain(value: Any) -> Any:
    """Lists (of a configuration) as tuples, mappings as dicts of the same."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


@dataclasses.dataclass(frozen=True)
class CorridorTablesConfig:
    """Design and settings of the tables (``configs/corridor_metrics.yaml``, ``design``)."""

    corridor_root: Path
    runs_root: Path
    out_dir: Path
    m8_dir: Path | None = None  # the corridor tables of M8; None: <out_dir>/../m8
    scenarios: tuple[str, ...] = ("i80_p0", "i80_p1", "i80_p2", "us101_p0", "us101_p1", "us101_p2")
    corridors: Mapping[str, str] = dataclasses.field(default_factory=lambda: {"i80": "I-80", "us101": "US-101"})
    laws: tuple[str, ...] = LAWS
    rmax_laws: tuple[str, ...] = RMAX_LAWS
    temporal_laws: tuple[str, ...] = TEMPORAL_LAWS  # D120: rows of laws, components, instability; not in H12
    ablation_laws: tuple[str, ...] = ABLATION_LAWS  # review of M8: rows of laws, components, instability, e4_rmax
    optional_laws: tuple[str, ...] = (HET_LAW, *TEMPORAL_LAWS, *ABLATION_LAWS)  # in the tables once a law file or a
    # run exists
    law_corridors: Mapping[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {"idm_heterogeneous_all": ("I-80",), **{law: ("I-80",) for law in TEMPORAL_LAWS}}
    )
    law_scenarios: Mapping[str, tuple[str, ...]] = dataclasses.field(  # laws that run on some scenarios only
        default_factory=lambda: {law: TEMPORAL_SCENARIOS for law in TEMPORAL_LAWS}
    )
    temporal_pairs: Mapping[str, str] = dataclasses.field(default_factory=lambda: dict(TEMPORAL_PAIRS))
    temporal_scenarios: tuple[str, ...] = TEMPORAL_SCENARIOS
    pooled_laws: tuple[str, ...] | None = None  # D122 correlation_pooled; None: laws and rmax_laws
    power_laws: tuple[str, ...] = ("idm_global", "residual_idm_certified")  # D122 power: the two laws compared
    power_metrics: tuple[str, ...] = tuple(POWER_METRICS)
    power_difference: float = 0.10  # relative difference to detect
    power_alpha: float = 0.05
    power_target: float = 0.8
    asymmetry_contrasts: tuple[tuple[str, str], ...] = ASYMMETRY_CONTRASTS  # D121: (law, free law it is compared with)
    seeds: tuple[int, ...] = tuple(range(10))
    reference: str = "idm_global"
    candidate: str = "residual_idm_certified"
    margin: float = 0.10
    alpha: float = 0.05
    test: str = "t"
    n_resamples: int = 1000
    level: float = 0.95
    seed: int = 0
    collision_free_max: float = 0.0
    gain_threshold: float = 1.02
    omega_min: float = 0.02
    omega_max: float = 2.0
    n_omega: int = 25
    dt: float = 0.1
    v_grid: tuple[float, ...] = tuple(float(v) for v in range(5, 31))
    # D108: verdicts of H12.1-H12.3
    networks: tuple[str, ...] = ("mlp", "gru", "lstm")
    degradation_min: float = 0.15
    networks_min: int = 2
    metrics_min: int = 2
    correlation_confirm: float = 0.6
    correlation_refute: float = 0.3
    correlation_alpha: float = 0.05
    correlation_min_laws: int = 10
    n_permutations: int = 10000
    micro_data: str = "ngsim_i80"
    micro_reference: str = "e3_reference/idm"
    micro_candidate: str = "e4_stable_ft/residual_idm"
    micro_others: tuple[str, ...] = ("e4_free_ft/residual_idm", "e4_free_ft/mlp")
    run_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    run_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # D111: the residual-amplitude sweep of the certified hybrid; extra rows of the ablation (review of M8)
    rmax: tuple[Mapping[str, Any], ...] = RMAX
    rmax_ablation: tuple[Mapping[str, Any], ...] = RMAX_ABLATION
    rmax_data: str = "follownet_highd"
    rmax_model: str = "residual_idm"
    # review of M8: clustered inference of H12.3 (correlation_clustered)
    families: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=lambda: dict(FAMILIES))
    physics_families: tuple[str, ...] = PHYSICS_FAMILIES
    cluster_resamples: int = 5000
    cluster_seed: int = 20261007
    # D113: variants of one scenario
    sensitivity_scenario: str = "i80_p1"
    sensitivity_variants: tuple[str, ...] = SENSITIVITY_VARIANTS
    sensitivity_laws: tuple[str, ...] = ("idm_global", "residual_idm_certified", "gru")
    sensitivity_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sensitivity_order: tuple[str, ...] = ("residual_idm_certified", "idm_global", "gru")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], corridor_root: str | Path, runs_root: str | Path, out_dir: str | Path,
        m8_dir: str | Path | None = None,
    ) -> CorridorTablesConfig:  # fmt: skip
        names = {f.name for f in dataclasses.fields(cls)} - {"corridor_root", "runs_root", "out_dir", "m8_dir"}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"unknown keys of the corridor tables: {sorted(unknown)}")
        values = {key: _plain(value) for key, value in raw.items()}
        return cls(corridor_root=Path(corridor_root), runs_root=Path(runs_root), out_dir=Path(out_dir),
                   m8_dir=None if m8_dir is None else Path(m8_dir), **values)  # fmt: skip

    def omegas(self) -> np.ndarray:
        return np.geomspace(self.omega_min, self.omega_max, self.n_omega)

    @property
    def m8(self) -> Path:
        return self.m8_dir if self.m8_dir is not None else self.out_dir.parent / "m8"

    def table_laws(self) -> tuple[str, ...]:
        """The laws with rows in the laws and components tables: the design, the residual amplitudes, the temporal
        hold-out, the ablation of the review (each once, in this order)."""
        return tuple(dict.fromkeys((*self.laws, *self.rmax_laws, *self.temporal_laws, *self.ablation_laws)))

    def family_of(self, law: str) -> str | None:
        """The architecture family of a law (``families``); None for a law of none."""
        return next((family for family, laws in self.families.items() if law in laws), None)

    def pooled(self) -> tuple[str, ...]:
        """The laws of correlation_pooled (D122): ``pooled_laws`` or the design and the residual amplitudes."""
        laws = self.pooled_laws if self.pooled_laws is not None else (*self.laws, *self.rmax_laws)
        return tuple(dict.fromkeys(laws))


# --------------------------------------------------------------------------------------------- helpers


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def flatten_macro(macro: Mapping[str, Any]) -> dict[str, float]:
    """The scalar metrics of a ``macro.json`` (None -> NaN): raw metrics, macro error and its components."""
    waves, travel = macro.get("waves") or {}, macro.get("travel_time") or {}
    error = macro.get("macro_error") or {}
    components = error.get("components") or {}
    row = {key: _float(macro.get(key)) for key in (
        "throughput_vph", "mean_speed", "queue_discharge_flow", "flow_peak_2min", "capacity_drop", "congested_share",
        "fd_scatter", "collisions_per_1000_vkm", "vehicles_in_contact_share", "inserted_share", "mean_depart_delay_s",
        "vehicle_km",
    )}  # fmt: skip
    # episodes attributed to the window; macro.json files of the first loop counted removed vehicles
    row["n_collisions"] = _float(macro.get("collision_episodes", macro.get("n_collisions")))
    row.update(n_waves=_float(waves.get("n_waves")), wave_speed=_float(waves.get("wave_speed")),
               wave_speed_xcorr=_float(waves.get("wave_speed_xcorr")),
               wave_amplitude=_float(waves.get("wave_amplitude")))  # fmt: skip
    for key in ("n", "mean", "median", "p10", "p90"):
        row[f"travel_time_{key}"] = _float(travel.get(key))
    row["macro_error"] = _float(error.get("value"))
    row["n_components"] = _float(error.get("n_components"))
    for name in COMPONENTS:
        row[f"error_{name}"] = _float(components.get(name))
    # the dynamic part (M6); a macro.json written before it gets it from its components
    dynamic = macro.get("macro_error_dynamic") or macro_error_dynamic(error)
    row["macro_error_dynamic"] = _float(dynamic.get("value"))
    row["n_components_dynamic"] = _float(dynamic.get("n_components"))
    return row


def exposure_values(macro: Mapping[str, Any] | None, info: Mapping[str, Any] | None = None) -> dict[str, float]:
    """The exposure and the demand of a run (review of M8): from ``macro.json`` the vehicle-km inside the section and
    the analysis window, the vehicles with time or a sample in the window, the vehicles planned to depart inside the
    window and the vehicles with samples; from ``run.json`` (``info``) the vehicles planned and inserted over the whole
    run and their mean insertion delay. None -> NaN."""
    macro, info = macro or {}, info or {}
    return {
        "vehicle_km": _float(macro.get("vehicle_km")), "n_vehicles_window": _float(macro.get("n_vehicles_window")),
        "n_planned_window": _float(macro.get("n_planned")), "n_vehicles": _float(macro.get("n_vehicles")),
        "run_n_planned": _float(info.get("n_planned")), "run_n_inserted": _float(info.get("n_inserted")),
        "run_depart_delay_s": _float(info.get("mean_depart_delay_s")),
    }  # fmt: skip


def _rate_per_1000(rows: np.ndarray) -> float:
    """Contact episodes per 1000 vehicle-km of the runs together: 1000 sum(episodes) / sum(vehicle-km)."""
    distance = float(rows[:, 1].sum())
    return 1000.0 * float(rows[:, 0].sum()) / distance if distance > 0 else math.nan


def _bracket(low: Any, high: Any, digits: int = 3) -> str:
    """``[low, high]`` rounded; empty without both ends."""
    low, high = _float(low), _float(high)
    return "" if math.isnan(low) or math.isnan(high) else f"[{low:.{digits}f}, {high:.{digits}f}]"


def flatten_asymmetry(payload: Mapping[str, Any] | None) -> dict[str, float]:
    """The scalar values of an ``asymmetry.json`` (None -> NaN): the acceleration asymmetry and the summary of the
    mean detector spectrum."""
    payload = payload or {}
    accel, spectrum = payload.get("acceleration") or {}, payload.get("spectrum") or {}
    row = {key: _float(accel.get(key)) for key in ("asymmetry_index", "accelerating_share", "decelerating_share",
                                                    "mean_acceleration", "mean_deceleration", "n_samples")}  # fmt: skip
    row.update({key: _float(spectrum.get(key)) for key in ("peak_frequency", "centroid", "band_rms", "band_power",
                                                            "n_detectors")})  # fmt: skip
    return row


def relative_error(value: Any, truth: Any) -> float:
    """``(value - truth) / |truth|``; NaN when a value is missing or the truth is 0."""
    value, truth = _float(value), _float(truth)
    if math.isnan(value) or math.isnan(truth) or truth == 0.0:
        return math.nan
    return (value - truth) / abs(truth)


def window_text(window: Any) -> str | None:
    """``"180-840"`` for the analysis window ``[180, 840]`` of a macro.json; None without one."""
    try:
        start, end = (float(w) for w in window)
    except (TypeError, ValueError):
        return None
    return f"{start:g}-{end:g}"


def _rank(values: np.ndarray) -> np.ndarray:
    return sps.rankdata(values)


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Pearson or Spearman correlation; NaN with fewer than three pairs or a constant variable."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return math.nan
    if method == "spearman":
        x, y = _rank(x), _rank(y)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def correlation_ci(
    x: Sequence[float], y: Sequence[float], method: str, n_resamples: int = 1000, level: float = 0.95, seed: int = 0
) -> dict[str, Any]:
    """Correlation over the pairs with its percentile bootstrap interval, resampling the pairs (the laws).

    Resamples in which a variable is constant have no correlation and are left out (``n_valid``)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = {"method": method, "n": int(len(x)), "estimate": correlation(x, y, method), "low": math.nan,
           "high": math.nan, "n_valid": 0}  # fmt: skip
    if len(x) < 3:
        return out
    rng = np.random.default_rng(seed)
    boot = np.array([correlation(x[i], y[i], method) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_resamples))])
    valid = boot[np.isfinite(boot)]
    out["n_valid"] = int(len(valid))
    if len(valid):
        alpha = 1.0 - level
        out["low"], out["high"] = (float(q) for q in np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0]))
    return out


def permutation_p(
    x: Sequence[float], y: Sequence[float], method: str, n_permutations: int = 10000, seed: int = 0
) -> float:
    """Two-sided permutation p-value of the correlation of ``x`` and ``y`` (D108): the values of ``x`` (the
    instability) are permuted over the pairs (the laws) ``n_permutations`` times, and
    ``p = (1 + #{|r_perm| >= |r|}) / (1 + n_permutations)`` (a Monte Carlo p-value is never 0). Spearman permutes
    the ranks. NaN with fewer than three pairs or a constant variable."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = correlation(x, y, method)
    if len(x) < 3 or not math.isfinite(observed) or n_permutations < 1:
        return math.nan
    if method == "spearman":
        x, y = _rank(x), _rank(y)
    shuffled = np.random.default_rng(seed).permuted(np.tile(x, (n_permutations, 1)), axis=1)
    r = (shuffled - x.mean()) @ (y - y.mean()) / (len(x) * x.std() * y.std())
    hits = int(np.sum(np.abs(r) >= abs(observed) - 1e-12))
    return (hits + 1) / (n_permutations + 1)


def idm_unstable_share(
    params: Mapping[str, Any], support_v: Sequence[float], v_grid: Sequence[float], omegas: np.ndarray,
    dt: float = 0.1, threshold: float = 1.02,
) -> dict[str, Any]:  # fmt: skip
    """Share of the grid speeds in ``support_v`` at which the exact discrete-time gain of an IDM exceeds
    ``threshold`` on ``omegas``, per parameter set (``T``, ``s0``, ``a`` arrays; ``v0``, ``b`` scalars or
    arrays), mean over the sets. Speeds without equilibrium (``v >= v0``) are left out of a set's share;
    a set without any equilibrium in the support has no share."""
    import torch  # only this table needs torch

    from cf_stability.models.idm import idm_equilibrium_partials
    from cf_stability.stability.analytic import transfer_discrete

    def column(name: str) -> torch.Tensor:
        value = torch.as_tensor(np.asarray(params[name], dtype=np.float64))
        return value.reshape(-1, 1) if value.ndim else value

    speeds = [float(v) for v in v_grid if support_v[0] <= v <= support_v[1]]
    T = column("T")
    n_sets = int(T.shape[0])
    out = {"share": math.nan, "n_sets": n_sets, "n_sets_with_equilibrium": 0, "n_speeds": len(speeds),
           "max_gain": math.nan}  # fmt: skip
    if not speeds or not n_sets:
        return out
    v = torch.tensor(speeds, dtype=torch.float64)[None, :]
    s_e, f_s, f_dv, f_v = idm_equilibrium_partials(v, column("v0"), T, column("s0"), column("a"), column("b"))
    shape = torch.broadcast_shapes(s_e.shape, f_s.shape)
    s_e, f_s, f_dv, f_v = (x.expand(shape) for x in (s_e, f_s, f_dv, f_v))
    omega = torch.as_tensor(omegas, dtype=torch.float64)
    gain = transfer_discrete(f_s.reshape(-1), f_dv.reshape(-1), f_v.reshape(-1), omega, dt).abs()
    max_gain = gain.max(dim=-1).values.reshape(shape)
    exists = torch.isfinite(s_e) & torch.isfinite(max_gain)
    unstable = exists & (max_gain > threshold)
    n_eq = exists.sum(dim=1)
    has = n_eq > 0
    if has.any():
        shares = unstable.sum(dim=1)[has].double() / n_eq[has].double()
        out.update(share=float(shares.mean()), n_sets_with_equilibrium=int(has.sum()),
                   max_gain=float(max_gain[exists].max()))  # fmt: skip
    return out


# --------------------------------------------------------------------------------------------- verdicts


def verdict_h12_1(metrics: Sequence[Mapping[str, Any]], need: int = 2) -> str:
    """H12.1 for one network and corridor (D108) from its three metrics (``degraded``: lower end of the
    degradation above 0, ``strong``: degraded with a point estimate of at least the threshold; None without
    an interval): confirmed with at least ``need`` metrics strongly degraded, refuted with every metric known
    and none degraded, open otherwise; empty when a missing metric leaves it undecided."""
    strong = sum(m.get("strong") is True for m in metrics)
    if strong >= need:
        return "confirmed"
    if any(m.get("degraded") is None for m in metrics):
        return ""
    return "refuted" if not any(m["degraded"] for m in metrics) else "open"


def verdict_every(verdicts: Sequence[str]) -> str:
    """The verdict of a hypothesis over the corridors: confirmed (refuted) when it is confirmed (refuted) on
    every corridor, empty when one has none, open otherwise."""
    if not verdicts or any(not v for v in verdicts):
        return ""
    if all(v == "confirmed" for v in verdicts):
        return "confirmed"
    return "refuted" if all(v == "refuted" for v in verdicts) else "open"


def verdict_correlation(row: Mapping[str, Any], cfg: CorridorTablesConfig) -> str:
    """H12.3 on one correlation row (D108): confirmed when r > ``correlation_confirm``, the permutation p-value
    below ``correlation_alpha`` and at least ``correlation_min_laws`` laws; refuted when r <
    ``correlation_refute``; open otherwise; empty without a correlation."""
    r, p, n = _float(row.get("estimate")), _float(row.get("p_permutation")), _float(row.get("n"))
    if math.isnan(r):
        return ""
    if r > cfg.correlation_confirm and p < cfg.correlation_alpha and n >= cfg.correlation_min_laws:
        return "confirmed"
    return "refuted" if r < cfg.correlation_refute else "open"


def corridor_text(corridors: Sequence[str]) -> str:
    """``I-80 only`` for one corridor, ``I-80 and US-101`` for two (D108)."""
    if len(corridors) == 1:
        return f"{corridors[0]} only"
    return ", ".join(corridors[:-1]) + f" and {corridors[-1]}" if corridors else "none"


def _interval(row: Mapping[str, Any], key: str, digits: int = 1, pct: bool = True) -> str:
    """``value [low, high]`` of ``key`` in percent (or plain with ``pct=False``), ``n/a`` without a value."""
    def text(value: Any) -> str:
        value = _float(value)
        if math.isnan(value):
            return "n/a"
        return f"{100.0 * value:+.{digits}f} %" if pct else f"{value:+.{digits}f}"

    value, low, high = (text(row.get(k)) for k in (key, f"{key}_low", f"{key}_high"))
    return value if "n/a" in (low, high) else f"{value} [{low}, {high}]"


# --------------------------------------------------------------------------------------------- the maker


class CorridorTableMaker:
    """Builds the tables; remembers the runs it has read, the verdicts and every missing item."""

    def __init__(self, cfg: CorridorTablesConfig) -> None:
        self.cfg = cfg
        self.missing: list[str] = []
        self.verdicts: list[dict[str, Any]] = []
        self._runs: pd.DataFrame | None = None
        self._truths: dict[str, dict[str, Any]] = {}
        self._corridors: list[str] | None = None
        self._instability: dict[str, dict[str, Any]] = {}
        self._maker: TableMaker | None = None
        self._hashes: set[Any] = set()
        self._present: dict[str, bool] = {}
        self._asymmetry: dict[tuple[str, str], pd.DataFrame] = {}
        self._asymmetry_truths: dict[str, dict[str, Any]] = {}
        self._asymmetry_hashes: set[Any] = set()
        self._asymmetry_config: dict[str, Any] | None = None
        self._truth_exposure: dict[str, dict[str, float]] = {}  # the exposure of the ground truths (contacts_absolute)
        self.pooled_laws_frame: pd.DataFrame | None = None  # the values behind correlation_pooled, one row per law
        self.tost_frame: pd.DataFrame | None = None
        self.correlation_frame: pd.DataFrame | None = None

    def note(self, table: str, text: str) -> None:
        self.missing.append(f"[{table}] {text}")

    def label(self, path: Path) -> str:
        for root in (self.cfg.corridor_root, self.cfg.runs_root):
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    # ------------------------------------------------------------------------------------ corridors
    def corridor_of(self, scenario: str) -> str:
        prefix = scenario.split("_", 1)[0]
        return self.cfg.corridors.get(prefix, prefix)

    def scenarios_of(self, corridor: str) -> list[str]:
        return [s for s in self.cfg.scenarios if self.corridor_of(s) == corridor]

    def all_corridors(self) -> list[str]:
        return list(dict.fromkeys(self.corridor_of(s) for s in self.cfg.scenarios))

    def corridors(self) -> list[str]:
        """The corridors in the tables: the first one always, any other once a ground truth of its scenarios
        has a macro.json (one line in missing.txt otherwise)."""
        if self._corridors is None:
            self._corridors = []
            for k, corridor in enumerate(self.all_corridors()):
                scenarios = self.scenarios_of(corridor)
                truths = [self.cfg.corridor_root / "scenarios" / s / "macro.json" for s in scenarios]
                if k == 0 or any(path.exists() for path in truths):
                    self._corridors.append(corridor)
                else:
                    self.note("runs", f"corridor {corridor} ({', '.join(scenarios)}): no ground truth with a "
                                      "macro.json, not in the tables")  # fmt: skip
        return self._corridors

    def present(self, law: str) -> bool:
        """A law of ``optional_laws`` (D118, D120) is in the tables once its law file or a run of it exists (one note
        otherwise); every other law always is."""
        if law not in self.cfg.optional_laws:
            return True
        if law not in self._present:
            root = self.cfg.corridor_root
            scenarios = dict.fromkeys((*self.cfg.scenarios, *self.cfg.law_scenarios.get(law, ())))
            found = (root / "laws" / f"{law}.json").exists() or any((root / s / law).is_dir() for s in scenarios)
            self._present[law] = found
            if not found:
                origin = "a law of the review of M8" if law in self.cfg.ablation_laws else "a law of M8"
                self.note("runs", f"{law}: no law file and no runs yet ({origin}): not in the tables")
        return self._present[law]

    def laws_on(self, corridor: str, laws: Sequence[str]) -> list[str]:
        """The laws of ``laws`` that run on ``corridor`` (``law_corridors``) and are present."""
        restricted = self.cfg.law_corridors
        return [law for law in laws if (law not in restricted or corridor in restricted[law]) and self.present(law)]

    def scenarios_for(self, corridor: str, law: str) -> list[str]:
        """The scenarios of ``corridor`` on which ``law`` runs (``law_scenarios``: the temporal hold-out on periods 1
        and 2 of I-80, D120)."""
        allowed = self.cfg.law_scenarios.get(law)
        return [s for s in self.scenarios_of(corridor) if allowed is None or s in allowed]

    def scenario_set(self, corridor: str, law: str) -> str:
        """``all`` for a law on every scenario of the corridor, else its scenarios."""
        mine = self.scenarios_for(corridor, law)
        return "all" if mine == self.scenarios_of(corridor) else ", ".join(mine) or "none"

    def complete_design(self) -> bool:
        """Every corridor of the configuration is in the tables."""
        return len(self.corridors()) == len(self.all_corridors())

    # ------------------------------------------------------------------------------------- reading
    def _read(self, table: str, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            self.note(table, f"{self.label(path)} missing")
            return None
        try:
            payload = read_json(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.note(table, f"{self.label(path)} unreadable ({type(exc).__name__})")
            return None
        if not isinstance(payload, dict):
            self.note(table, f"{self.label(path)} unreadable (not a mapping)")
            return None
        return payload

    def truth(self, scenario: str, table: str = "laws") -> dict[str, Any]:
        """The flat metrics of the ground truth of a scenario (read once; a missing file is noted)."""
        if scenario not in self._truths:
            macro = self._read(table, self.cfg.corridor_root / "scenarios" / scenario / "macro.json")
            values = {} if macro is None else {**flatten_macro(macro), "window": window_text(macro.get("window"))}
            self._truth_exposure[scenario] = exposure_values(macro)  # apart: the truth rows of laws.csv stay as they are
            self._truths[scenario] = {"law": "ground truth", "scenario": scenario, "kind": "truth",
                                      "corridor": self.corridor_of(scenario), **values}  # fmt: skip
        return self._truths[scenario]

    def truths(self) -> list[dict[str, Any]]:
        """One row per scenario of the corridors in the tables with the flat metrics of its ground truth."""
        return [self.truth(s) for corridor in self.corridors() for s in self.scenarios_of(corridor)]

    def read_runs(
        self, table: str, corridor: str, law: str, scenarios: Sequence[str], seeds: Sequence[int]
    ) -> list[dict[str, Any]]:
        """One row per run (scenario, seed) of ``law``; ``present`` marks the runs whose ``macro.json`` was
        read and whose analysis window is that of the ground truth of their scenario. Missing runs and files
        are noted (one line when no run of the law exists on these scenarios)."""
        expected = [(s, seed) for s in scenarios for seed in seeds]
        root = self.cfg.corridor_root
        rows = [{"corridor": corridor, "law": law, "scenario": s, "seed": seed, "present": False} for s, seed in expected]
        if len(expected) > 1 and not any((root / s / law / f"seed{seed}").is_dir() for s, seed in expected):
            self.note(table, f"{corridor}/{law}: run missing (all {len(expected)}: {', '.join(scenarios)} x seeds "
                             f"{', '.join(map(str, seeds))})")  # fmt: skip
            return rows
        for row in rows:
            run = root / row["scenario"] / law / f"seed{row['seed']}"
            if not run.is_dir():
                self.note(table, f"{self.label(run)}: run missing")
                continue
            macro = self._read(table, run / "macro.json")
            if macro is None:
                continue
            row.update(flatten_macro(macro), present=True, window=window_text(macro.get("window")))
            self._hashes.add(macro.get("config_hash"))
            info = self._read(table, run / "run.json") or {}  # the boundary of the loop
            row.update(g_mean=_float(info.get("g_mean")), dN_mean=_float(info.get("dN_mean")))
            row.update(exposure_values(macro, info))  # contacts_absolute (review of M8)
            truth_window = self.truth(row["scenario"], table).get("window")
            if truth_window is not None and row["window"] != truth_window:  # stale: left out of every table
                row["present"] = False
                self.note(table, f"{row['scenario']}/{law}/seed{row['seed']}: window {row['window']} s, ground truth "
                                 f"{truth_window} s: left out, rerun the metrics")  # fmt: skip
        return rows

    @staticmethod
    def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        keys = ["corridor", "law", "scenario", "seed", "present", *RAW_METRICS, "macro_error", "n_components",
                "macro_error_dynamic", "n_components_dynamic", "n_collisions", "window", "g_mean", "dN_mean",
                *(f"error_{n}" for n in COMPONENTS), *EXPOSURE_KEYS]  # fmt: skip
        absent = {key: np.nan for key in keys if key not in frame.columns}
        frame = pd.concat([frame, pd.DataFrame(absent, index=frame.index)], axis=1) if absent else frame
        frame["present"] = frame["present"].astype(bool)
        return frame

    def runs(self) -> pd.DataFrame:
        """One row per run of the design (every corridor in the tables, its laws, scenarios and seeds) with its
        flat metrics; ``present`` marks the runs whose ``macro.json`` was read. Noted under [runs]."""
        if self._runs is not None:
            return self._runs
        c, rows = self.cfg, []
        self.truths()  # the ground truths first: their windows decide which runs are stale
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.table_laws()):
                rows += self.read_runs("runs", corridor, law, self.scenarios_for(corridor, law), c.seeds)
        if len(self._hashes) > 1:
            self.note("runs", f"the macro.json files hold {len(self._hashes)} different config hashes: rerun the "
                              "metrics")  # fmt: skip
        self._runs = self.frame(rows)
        return self._runs

    def mine(self, corridor: str, law: str) -> pd.DataFrame:
        runs = self.runs()
        return runs[(runs["corridor"] == corridor) & (runs["law"] == law) & runs["present"]]

    def windows(self) -> str:
        """The analysis windows of the ground truths and of the runs in the tables, e.g. ``180-840 s``."""
        runs = self.runs()
        found = {row.get("window") for row in self.truths()} | set(runs.loc[runs["present"], "window"].dropna())
        found.discard(None)
        return ", ".join(f"{w} s" for w in sorted(found)) or "none read"

    def law_spec(self, table: str, law: str) -> dict[str, Any] | None:
        return self._read(table, self.cfg.corridor_root / "laws" / f"{law}.json")

    # ---------------------------------------------------------------------------------- statistics
    def ci(self, values: Any) -> dict[str, Any]:
        c = self.cfg
        return bootstrap_ci(np.asarray(values, dtype=float), None, np.mean, c.n_resamples, c.level, c.seed)

    def put(self, row: dict[str, Any], key: str, values: Any) -> None:
        ci = self.ci(values)
        row[key], row[f"{key}_low"], row[f"{key}_high"] = ci["estimate"], ci["low"], ci["high"]

    def design_text(self) -> str:
        c, parts = self.cfg, []
        for corridor in self.corridors():
            laws = self.laws_on(corridor, c.table_laws())
            full = [law for law in laws if self.scenario_set(corridor, law) == "all"]
            text = f"{corridor}: {len(full)} laws x {len(self.scenarios_of(corridor))} scenarios x {len(c.seeds)} seeds"
            partial: dict[str, list[str]] = {}
            for law in laws:
                if law not in full:
                    partial.setdefault(self.scenario_set(corridor, law), []).append(law)
            for scenarios, group in partial.items():
                text += (f" + {len(group)} laws ({', '.join(group)}) x {len(scenarios.split(', '))} scenarios "
                         f"({scenarios}) x {len(c.seeds)} seeds")  # fmt: skip
            parts.append(text)
        absent = [corridor for corridor in self.all_corridors() if corridor not in self.corridors()]
        if absent:
            parts.append(f"not in the tables (no ground truth yet): {', '.join(absent)}")
        return "; ".join(parts)

    def ablation_note(self) -> list[str]:
        """One note line on the laws of the ablation of the review in the tables (none: no line)."""
        present = [law for law in self.cfg.ablation_laws if any(law in self.laws_on(k, [law]) for k in self.corridors())]
        if not present:
            return []
        laws = "; ".join(f"{law}: {ABLATION_TEXT.get(law, 'configs/corridor/laws.yaml')}" for law in present)
        return [f"Factorial ablation of the certified hybrid (review of M8; rows only, not in the correlations nor the "
                f"verdicts of H12): {laws}."]  # fmt: skip

    def header(self, *lines: str) -> list[str]:
        c, runs = self.cfg, self.runs()
        return [
            f"Runs: {int(runs['present'].sum())} of {len(runs)} expected runs have a macro.json ({self.design_text()}); "
            "missing runs, files and values: missing.txt.",
            f"Analysis window (scenario.json): {self.windows()}.",
            *lines,
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]

    # ------------------------------------------------------------------------------------------ laws
    def table_laws(self) -> tuple[Table, Table, pd.DataFrame]:
        c, t = self.cfg, "laws"
        rows, specs = [], {}
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.table_laws()):
                if law not in specs:
                    specs[law] = self.law_spec(t, law)
                mine = self.mine(corridor, law)
                expected = len(self.scenarios_for(corridor, law)) * len(c.seeds)
                row: dict[str, Any] = {"corridor": corridor, "law": law, "kind": (specs[law] or {}).get("kind"),
                                       "runs": len(mine), "runs_expected": expected,
                                       "scenario_set": self.scenario_set(corridor, law)}  # fmt: skip
                for key in ("macro_error", "macro_error_dynamic", *(f"error_{n}" for n in COMPONENTS), *RAW_METRICS,
                            "g_mean", "dN_mean"):  # fmt: skip
                    self.put(row, key, mine[key])
                row["n_components"] = mine["n_components"].mean() if len(mine) else np.nan
                row["n_components_dynamic"] = mine["n_components_dynamic"].mean() if len(mine) else np.nan
                row["runs_with_collisions"] = int((mine["n_collisions"] > 0).sum()) if len(mine) else 0
                rows.append(row)
        frame = pd.DataFrame(rows)
        truths = pd.DataFrame(self.truths())
        error_columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("scenario_set", "scenarios", "text"), Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("n_components", "components", digits=1),
            *(Column(f"error_{n}", COMPONENT_HEADERS[n], digits=3, ci=True) for n in COMPONENTS),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3, ci=True),
            Column("runs_with_collisions", "runs with collisions", "int"),
            Column("vehicles_in_contact_share", "vehicles in contact", digits=3, ci=True),
            Column("inserted_share", "inserted share", digits=3),
            Column("g_mean", "boundary gain g", digits=3, ci=True), Column("dN_mean", "dN (veh)", digits=1, ci=True),
        ]  # fmt: skip
        raw_columns = [Column("corridor", "corridor", "text"), Column("law", "law", "text"),
                       Column("scenario", "scenario", "text"), Column("scenario_set", "scenarios", "text"),
                       Column("runs", "runs", "int"),
                       *(Column(key, header, digits=digits, ci=True) for key, (header, digits) in RAW_METRICS.items())]  # fmt: skip
        notes = self.header(
            "Unit: run (scenario and seed); every value is the mean over the runs of the law on the corridor with its "
            "interval over the runs. Metrics of docs/m5_contract.md, section 5, inside the analysis window.",
            "Macro error: mean of the absolute components that exist (components: mean number). Components are "
            "signed relative errors against the ground truth of the run's scenario, (run - truth) / |truth|; "
            "wave speed: the cross-correlation estimate (the speed of the leading edges of the stop waves is in "
            "the raw metrics beside it, not in the error); waves: (run - truth) / max(truth, 1); FD: RMSE of the "
            "binned flow over the common bins / mean cell flow of the truth; travel time: Wasserstein-1 distance / "
            "mean travel time of the truth. Macro error (dynamic): the same mean over the dynamic components (FD, "
            "wave speed, waves, wave amplitude) only; components.md lists them with the anchored ones.",
            "collisions: contact episodes of followers per 1000 vehicle-km inside the window (without the time of an "
            "episode, the episodes of a vehicle count in proportion to its time in the section inside the window); "
            "runs with collisions: runs with at least one; vehicles in contact: share of the vehicles of the window "
            "with at least one episode.",
            "boundary gain g and dN: g_mean and dN_mean of run.json (time means inside the window of the gain of the "
            "downstream boundary and of the simulated minus the observed passages of x_out), mean over the runs.",
            f"Laws: those of D97, D114 and D118 ({', '.join(c.laws)}), the residual amplitudes of D111 "
            f"({', '.join(c.rmax_laws)}) and the temporal hold-out of D120 ({', '.join(c.temporal_laws)}: fine-tuned on "
            f"period 0, run on {', '.join(c.temporal_scenarios)} only, column scenarios); a law listed in law_corridors "
            "runs on those corridors only; a law of M8 without runs yet is left out (missing.txt).",
            *self.ablation_note(),
        )
        errors = Table(t, "Laws of the corridor: macro error against the ground truth", notes, frame, error_columns)
        raw_frame = pd.concat([frame.assign(scenario="all"), truths], ignore_index=True)
        raw_notes = [
            "Raw metrics: mean over the runs of every law (scenario all) on its corridor with its interval; ground "
            "truth: the values of the data per scenario (no interval).",
            "Flows in veh/h over the whole cross-section at the throughput detector, speeds in m/s, travel time from "
            "x_in + 10 m to x_out - 10 m. Capacity drop = 1 - queue discharge / peak 2-min flow; congested share: "
            "share of the 30 s intervals with a mean speed below the threshold at the queue detector (queue "
            "discharge: mean flow at the throughput detector over those intervals).",
            "Wave speed, cross-correlation: lag of the largest correlation of the speed series of cells a fixed "
            "distance apart, median over the pairs (the component of the macro error); leading edges: median over "
            "the stop waves of the line through their leading edges.",
        ]
        for key in ("runs", "runs_expected", "runs_with_collisions"):  # counts stay integers next to the truths
            if key in raw_frame:
                raw_frame[key] = raw_frame[key].astype("Int64")
        raw = Table(t, "Laws of the corridor: raw metrics", raw_notes, raw_frame, raw_columns)
        return errors, raw, raw_frame

    # ----------------------------------------------------------------------------------- instability
    def _member_run(self, run: str) -> Path | None:
        path = Path(run)
        if path.is_absolute():
            return path
        for base in (self.cfg.runs_root, self.cfg.runs_root.parent, REPO_ROOT):
            if (base / path).is_dir():
                return base / path
        return None

    def instability_of(self, t: str, law: str) -> dict[str, Any]:
        """Unstable-equilibrium fraction and band share "not stable" of the members of a law (read once: the
        members are the same on every corridor)."""
        if law not in self._instability:
            self._instability[law] = self._instability_of(t, law, self.law_spec(t, law))
        return self._instability[law]

    def _instability_of(self, t: str, law: str, spec: Mapping[str, Any] | None) -> dict[str, Any]:
        c = self.cfg
        out: dict[str, Any] = {"members": 0, "audited": 0, "unstable_eq": np.nan, "not_stable": np.nan}
        if spec is None:
            return out
        kind = spec.get("kind") or ("idm_heterogeneous" if law.startswith("idm_heterogeneous") else "models")
        out["kind"] = kind
        laws_dir = c.corridor_root / "laws"
        if kind == "idm" and not spec.get("members"):  # D120: global calibrations without member runs
            params = spec.get("params") or []
            support = spec.get("support_v")
            if not params or support is None:
                self.note(t, f"{law}: {'params' if not params else 'support_v'} missing in the law file")
                return out
            columns = {key: np.array([float(p[key]) for p in params]) for key in ("v0", "T", "s0", "a", "b")}
            share = idm_unstable_share(columns, [float(s) for s in support[:2]], c.v_grid, c.omegas(), c.dt,
                                       c.gain_threshold)  # fmt: skip
            out.update(members=share["n_sets"], audited=share["n_sets_with_equilibrium"], unstable_eq=share["share"],
                       max_gain=share["max_gain"], speeds=share["n_speeds"])  # fmt: skip
            self.note(t, f"{law}: band share not stable not defined for a closed-form law (contract, section 6)")
            return out
        if str(kind).startswith("idm_heterogeneous") or kind == "residual_heterogeneous":
            # idm_heterogeneous (D63), idm_heterogeneous_all (D114); residual_heterogeneous (D118): its cores
            table = laws_dir / str(spec.get("table") or f"{law}.npz")
            if not table.exists():
                self.note(t, f"{self.label(table)} missing")
                return out
            try:
                with np.load(table, allow_pickle=False) as data:
                    params = {key: data[key] for key in data.files}
            except (OSError, ValueError) as exc:
                self.note(t, f"{self.label(table)} unreadable ({type(exc).__name__})")
                return out
            params.update({k: spec[k] for k in ("v0", "b") if k not in params and spec.get(k) is not None})
            support = spec.get("support_v") if spec.get("support_v") is not None else params.get("support_v")
            absent = [k for k in ("T", "s0", "a", "v0", "b") if k not in params]
            if absent or support is None:
                self.note(t, f"{law}: {', '.join(absent + ([] if support is not None else ['support_v']))} missing")
                return out
            share = idm_unstable_share(params, [float(s) for s in np.asarray(support).reshape(-1)[:2]], c.v_grid,
                                       c.omegas(), c.dt, c.gain_threshold)  # fmt: skip
            out.update(members=share["n_sets"], audited=share["n_sets_with_equilibrium"], unstable_eq=share["share"],
                       max_gain=share["max_gain"], speeds=share["n_speeds"])  # fmt: skip
            self.note(t, f"{law}: band share not stable not defined for a closed-form law (contract, section 6)")
            return out
        members = spec.get("members") or []
        out["members"] = len(members)
        unstable, not_stable = [], []
        for member in members:
            run = self._member_run(str(member.get("run"))) if member.get("run") else None
            if run is None or not run.is_dir():
                self.note(t, f"{law}: member run {member.get('run')} missing")
                continue
            if not (run / "stability.json").exists():
                self.note(t, f"{law}: {self.label(run)}/stability.json missing")
                continue
            row = collect_run(run)
            share, stable = _float(row.get("share_unstable_numerical")), _float(row.get("band_numerical_stable"))
            if isinstance(row.get("audit_error"), str) or (math.isnan(share) and math.isnan(stable)):
                self.note(t, f"{law}: {self.label(run)}/stability.json holds no shares")
                continue
            unstable.append(share)
            not_stable.append(1.0 - stable)
        out["audited"] = len(unstable)
        if unstable:
            out["unstable_eq"] = float(np.nanmean(unstable)) if np.isfinite(unstable).any() else np.nan
            out["not_stable"] = float(np.nanmean(not_stable)) if np.isfinite(not_stable).any() else np.nan
        return out

    def table_instability(self) -> tuple[Table, Table]:
        c, t = self.cfg, "instability"
        rows = []
        for corridor in self.corridors():
            for law in self.laws_on(corridor, tuple(dict.fromkeys((*c.laws, *c.temporal_laws, *c.ablation_laws)))):
                mine = self.mine(corridor, law)
                row: dict[str, Any] = {"corridor": corridor, "law": law, **self.instability_of(t, law), "runs": len(mine),
                                       "scenario_set": self.scenario_set(corridor, law),
                                       "in_correlation": law in c.laws}  # fmt: skip
                for key in ERRORS:
                    self.put(row, key, mine[key])
                collisions = mine["collisions_per_1000_vkm"].mean() if len(mine) else np.nan
                row["collisions_per_1000_vkm"] = collisions
                row["collides"] = bool(collisions > c.collision_free_max) if np.isfinite(collisions) else np.nan
                rows.append(row)
        frame = pd.DataFrame(rows)
        correlations = []
        for corridor in self.corridors():
            mine = frame[(frame["corridor"] == corridor) & frame["in_correlation"]] if len(frame) else frame
            for key, error in ERRORS.items():
                for measure, name in MEASURES.items():
                    for subset in ("all laws", "laws without collisions"):
                        part = mine if subset == "all laws" else mine[mine["collides"].eq(False)]
                        x = part[measure].astype(float) if len(part) else pd.Series(dtype=float)
                        y = part[key].astype(float) if len(part) else pd.Series(dtype=float)
                        for method in ("spearman", "pearson"):
                            r = correlation_ci(x, y, method, c.n_resamples, c.level, c.seed)
                            p = permutation_p(x, y, method, c.n_permutations, c.seed)
                            correlations.append({"corridor": corridor, "error": error, "measure": name, "laws": subset,
                                                 "method": method, "n": r["n"], "estimate": r["estimate"],
                                                 "estimate_low": r["low"], "estimate_high": r["high"],
                                                 "n_valid": r["n_valid"], "p_permutation": p})  # fmt: skip
        corr = pd.DataFrame(correlations, columns=["corridor", "error", "measure", "laws", "method", "n", "estimate",
                                                   "estimate_low", "estimate_high", "n_valid", "p_permutation"])  # fmt: skip
        self.correlation_frame = corr
        columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("kind", "kind", "text"),
            Column("members", "members", "int"), Column("audited", "audited", "int"),
            Column("unstable_eq", "unstable among equilibria", digits=3),
            Column("not_stable", "band share not stable", digits=3), Column("runs", "runs", "int"),
            Column("scenario_set", "scenarios", "text"),
            Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3),
            Column("collides", "collides", "flag"), Column("in_correlation", "in the correlation", "flag"),
        ]  # fmt: skip
        notes = self.header(
            "Unstable among equilibria: mean over the member runs of the audit share share_unstable_numerical "
            "(string-unstable equilibria among the equilibria found, speeds in support); band share not stable: "
            "mean over the members of 1 - band_numerical stable (speeds in support). idm_heterogeneous and "
            "idm_heterogeneous_all: per parameter set (members) the share of the grid speeds in the support of the "
            f"data at which the exact discrete-time gain (dt {c.dt:g} s, {c.n_omega} frequencies {c.omega_min:g}-"
            f"{c.omega_max:g} rad/s) exceeds {c.gain_threshold:g}, speeds without equilibrium left out, mean over the "
            "sets (audited: sets with an equilibrium in the support); the same for the certified per-event cores of "
            f"{HET_LAW} (D118; members: its cores, the residual left out) and for an IDM law of calibrations without "
            "member runs (idm_global_p0, D120). The members are the same on every corridor.",
            "macro error and macro error (dynamic: FD, wave speed, waves, wave amplitude only): mean over the runs of "
            f"the law on the corridor (unit run) with its interval. collides: mean collisions per 1000 vehicle-km "
            f"above {c.collision_free_max:g}. The residual amplitudes of D111 are not in this table (correlation_pooled "
            f"of M8 has them); the temporal hold-out laws of D120 ({', '.join(c.temporal_laws)}) have rows over their "
            "scenarios (column scenarios) but are not in the correlations (in the correlation).",
            f"Correlations below: per corridor over the laws, {c.n_resamples} bootstrap resamples of the laws "
            "(resamples with a constant variable have no correlation: valid resamples); p (permutation): two-sided, "
            f"{c.n_permutations} permutations of the instability over the laws, (1 + hits) / (1 + permutations).",
            *(f"{line} Their instability: the exact gain of the parameter sets for the IDM laws (members: the parameter "
              "sets), the member audits for the hybrid." for line in self.ablation_note()),
        )
        table = Table(t, "Instability of the members and macro error of the laws", notes, frame, columns)
        corr_columns = [
            Column("corridor", "corridor", "text"), Column("error", "error", "text"),
            Column("measure", "instability measure", "text"), Column("laws", "subset", "text"),
            Column("method", "method", "text"), Column("n", "laws", "int"),
            Column("estimate", "correlation", digits=3, ci=True), Column("n_valid", "valid resamples", "int"),
            Column("p_permutation", "p (permutation)", "p"),
        ]  # fmt: skip
        corr_notes = [
            "Correlation of the mean macro error (all eight components, or the dynamic four) of a law with the "
            "instability of its members, per corridor over the laws that have both; "
            f"{100 * c.level:g} % percentile interval from {c.n_resamples} resamples of the laws; p (permutation): "
            f"two-sided permutation test, {c.n_permutations} permutations of the instability over the laws.",
        ]
        corr_table = Table("instability_correlation", "Correlation of instability and macro error over the laws",
                           corr_notes, corr, corr_columns)  # fmt: skip
        return table, corr_table

    # ------------------------------------------------------------------------------------ components
    def table_components(self) -> Table:
        """Per corridor and law the eight components of the macro-error vector, anchored and dynamic."""
        c = self.cfg
        rows = []
        parts = [*((f"anchored_{n}", f"error_{n}", f"{COMPONENT_HEADERS[n]} (anchored)") for n in ANCHORED),
                 *((f"dynamic_{n}", f"error_{source}", f"{COMPONENT_HEADERS[source]} (dynamic)")
                   for n, source in DYNAMIC.items())]  # fmt: skip
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.table_laws()):
                mine = self.mine(corridor, law)
                row: dict[str, Any] = {"corridor": corridor, "law": law, "runs": len(mine),
                                       "scenario_set": self.scenario_set(corridor, law)}  # fmt: skip
                for key, source in [*((key, source) for key, source, _ in parts), *((k, k) for k in ERRORS)]:
                    self.put(row, key, mine[source])
                rows.append(row)
        columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("scenario_set", "scenarios", "text"),
            *(Column(key, header, digits=3, ci=True) for key, _, header in parts),
            *(Column(key, header, digits=3, ci=True) for key, header in ERRORS.items()),
        ]  # fmt: skip
        notes = self.header(
            "Components of the macro-error vector against the ground truth of the run's scenario, signed: "
            "(run - truth) / |truth|; waves: (run - truth) / max(truth, 1); FD: RMSE of the binned flow over the "
            "common bins / mean cell flow of the truth; travel time: Wasserstein-1 distance / mean travel time of "
            "the truth; wave speed: the cross-correlation estimate. Unit: run; mean over the runs of the law on the "
            "corridor with its interval.",
            "Anchored: throughput, mean speed, queue discharge and travel time follow largely the demand and the "
            "downstream boundary of the data, which every law shares. Dynamic: FD, wave speed, waves and wave "
            "amplitude are what a law decides. macro error: mean of the absolute values of all eight components; "
            "macro error (dynamic): of the four dynamic ones (components that exist).",
            *self.ablation_note(),
        )
        title = "Components of the macro error: anchored and dynamic"
        return Table("components", title, notes, pd.DataFrame(rows), columns)

    # ------------------------------------------------------------------------------------------ TOST
    def table_tost(self) -> Table:
        c, t = self.cfg, "tost"
        index = ["scenario", "seed"]
        rows = []
        for corridor in self.corridors():
            ref = self.mine(corridor, c.reference).set_index(index)
            cand = self.mine(corridor, c.candidate).set_index(index)
            common = len(ref.index.intersection(cand.index))
            if common < 2:
                self.note(t, f"{corridor}: {common} pairs of runs of {c.reference} and {c.candidate} (scenario and "
                             "seed): no test")  # fmt: skip
            for key, (header, _) in RAW_METRICS.items():
                a, b = ref[key].astype(float), cand[key].astype(float)
                pairs = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
                row: dict[str, Any] = {"corridor": corridor, "metric": key, "header": header, "pairs": len(pairs),
                                       "mean_reference": pairs["a"].mean() if len(pairs) else np.nan,
                                       "mean_candidate": pairs["b"].mean() if len(pairs) else np.nan}  # fmt: skip
                if len(pairs) < 2:
                    if common >= 2:  # the runs are there, this metric has no values in them
                        self.note(t, f"{corridor}: {key}: {len(pairs)} pairs with a value, no test")
                elif pairs["a"].mean() == 0:
                    self.note(t, f"{corridor}: {key}: mean of {c.reference} is 0, no relative margin")
                else:
                    # the margins (1 -+ m) mean(a) of tost_relative assume a positive reference: a negative one (wave
                    # speed, upstream) is tested on the magnitudes, which is the same equivalence interval
                    sign = -1.0 if pairs["a"].mean() < 0 else 1.0
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # SciPy warns on degenerate samples; the NaN it returns says it
                        cmp = paired_comparison(pairs["a"], pairs["b"], n_resamples=c.n_resamples, level=c.level,
                                                seed=c.seed)  # fmt: skip
                        tost = tost_relative(sign * pairs["a"], sign * pairs["b"], c.margin, alpha=c.alpha, test=c.test)
                    row.update(relative_difference=cmp["relative_change"], relative_difference_low=cmp["ci_low"],
                               relative_difference_high=cmp["ci_high"], wilcoxon_p=cmp["p_value"],
                               p_lower=tost["p_lower"], p_upper=tost["p_upper"], p_value=tost["p_value"],
                               equivalent=bool(np.isfinite(tost["p_value"]) and tost["equivalent"]))  # fmt: skip
                rows.append(row)
        frame = pd.DataFrame(rows)
        for key in ("corridor", "metric", "header", "pairs", "mean_reference", "mean_candidate", "relative_difference",
                    "relative_difference_low", "relative_difference_high", "wilcoxon_p", "p_lower", "p_upper",
                    "p_value", "equivalent"):  # fmt: skip
            if key not in frame.columns:
                frame[key] = np.nan
        self.tost_frame = frame
        columns = [
            Column("corridor", "corridor", "text"), Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("mean_reference", c.reference, digits=3), Column("mean_candidate", c.candidate, digits=3),
            Column("relative_difference", "relative difference", "pct", 1, ci=True),
            Column("wilcoxon_p", "p (Wilcoxon)", "p"), Column("p_lower", "p lower", "p"),
            Column("p_upper", "p upper", "p"), Column("p_value", "p TOST", "p"),
            Column("equivalent", f"equivalent within {100 * c.margin:g} %", "flag"),
        ]  # fmt: skip
        notes = self.header(
            f"TOST: {c.candidate} against {c.reference} per corridor, paired by scenario and seed (pairs); relative "
            f"difference mean({c.candidate}) / mean({c.reference}) - 1 with its interval over the pairs and the "
            "Wilcoxon signed-rank test.",
            f"Two one-sided paired {'t' if c.test == 't' else 'Wilcoxon'} tests of the margin +-{100 * c.margin:g} % "
            f"of the mean of {c.reference} (cf_stability.eval.stats.tost_relative); equivalent when the larger "
            f"p-value (p TOST) is below {c.alpha:g}. A metric with a negative reference mean (wave speed) is tested "
            "on its magnitude; one whose reference mean is 0 has no relative margin.",
        )
        return Table(t, f"Equivalence of {c.candidate} and {c.reference} (TOST)", notes, frame, columns)

    # ------------------------------------------------------------------------------------------ H12
    def table_h12_1(self) -> Table:
        """H12.1 (D108): degradation |e_net| - |e_reference| of the macro triple, paired by scenario and seed."""
        c, t = self.cfg, "h12_1"
        rows, by_network = [], {}
        index = ["scenario", "seed"]
        for corridor in self.corridors():
            expected = len(self.scenarios_of(corridor)) * len(c.seeds)
            reference = self.mine(corridor, c.reference).set_index(index)
            for network in self.laws_on(corridor, c.networks):
                runs = self.mine(corridor, network).set_index(index)
                metrics = []
                for component in TRIPLE:
                    key = f"error_{component}"
                    pairs = pd.concat({"net": runs[key].astype(float), "ref": reference[key].astype(float)}, axis=1,
                                      join="inner").dropna()  # fmt: skip
                    row: dict[str, Any] = {"corridor": corridor, "network": network, "metric": component,
                                           "header": COMPONENT_HEADERS[component], "pairs": len(pairs),
                                           "abs_error_network": pairs["net"].abs().mean() if len(pairs) else np.nan,
                                           "abs_error_reference": pairs["ref"].abs().mean() if len(pairs) else np.nan}  # fmt: skip
                    self.put(row, "degradation", (pairs["net"].abs() - pairs["ref"].abs()).to_numpy())
                    has = len(pairs) >= 2 and math.isfinite(_float(row["degradation_low"]))
                    if len(pairs) < 2:
                        self.note(t, f"{corridor}/{network}: {component}: {len(pairs)} pairs with {c.reference}, no "
                                     "interval")  # fmt: skip
                    row["degraded"] = bool(row["degradation_low"] > 0.0) if has else None
                    row["strong"] = bool(row["degraded"] and row["degradation"] >= c.degradation_min) if has else None
                    metrics.append(row)
                verdict = verdict_h12_1(metrics, c.metrics_min)
                complete = len(runs) == len(reference) == expected  # every run of both laws has a macro.json
                for row in metrics:
                    row.update(h12_1=verdict, complete=complete)
                rows += metrics
                by_network[corridor, network] = (verdict, complete, metrics)
        frame = pd.DataFrame(rows, columns=[
            "corridor", "network", "metric", "header", "pairs", "abs_error_network", "abs_error_reference",
            "degradation", "degradation_low", "degradation_high", "degraded", "strong", "h12_1", "complete"])  # fmt: skip
        self.h12_1_verdicts(by_network)
        columns = [
            Column("corridor", "corridor", "text"), Column("network", "network", "text"),
            Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("abs_error_network", "|e| network", digits=3), Column("abs_error_reference", f"|e| {c.reference}",
                                                                         digits=3),
            Column("degradation", "degradation", digits=3, ci=True), Column("degraded", "degraded", "flag"),
            Column("strong", f"degraded by >= {c.degradation_min:g}", "flag"), Column("h12_1", "H12.1", "text"),
            Column("complete", "complete", "flag"),
        ]  # fmt: skip
        notes = self.header(
            f"H12.1 (D108): the pure networks {', '.join(c.networks)} (laws of D97) against {c.reference} on the macro "
            "triple: throughput (anchored flow quantity), travel time (Wasserstein-1, delay quantity), wave speed by "
            "cross-correlation (dynamic quantity), the signed relative errors e of the macro-error vector (D99).",
            f"degradation = |e_network| - |e_{c.reference}|, paired by scenario and seed (unit run, pairs), mean with "
            f"its interval; degraded: lower end > 0; degraded by >= {c.degradation_min:g}: degraded and the mean at "
            f"least {c.degradation_min:g}. |e|: means over the pairs.",
            f"Verdict per network and corridor: confirmed with at least {c.metrics_min} of the 3 metrics degraded by >= "
            f"{c.degradation_min:g}, refuted with no metric degraded, otherwise open. Overall (verdicts.md): confirmed "
            f"when confirmed for at least {c.networks_min} of {len(c.networks)} networks on every corridor in the "
            f"tables, refuted when refuted for at least {c.networks_min} on every corridor, otherwise open.",
        )
        return Table(t, "H12.1: degradation of the macro triple by the pure networks", notes, frame, columns)

    def h12_1_verdicts(self, by_network: Mapping[tuple[str, str], tuple[str, bool, list[dict[str, Any]]]]) -> None:
        c = self.cfg
        per_corridor = []
        for corridor in self.corridors():
            verdicts = []
            for network in self.laws_on(corridor, c.networks):
                verdict, complete, metrics = by_network[corridor, network]

                def state(m: Mapping[str, Any]) -> str:
                    if m["degraded"] is None:
                        return "no interval"
                    if not m["degraded"]:
                        return "not degraded"
                    return f"degraded by >= {c.degradation_min:g}" if m["strong"] else "degraded"

                basis = "; ".join(f"{m['metric'].replace('_', ' ')} {_interval(m, 'degradation', 3, pct=False)} "
                                  f"{state(m)}" for m in metrics)  # fmt: skip
                self.verdict("H12.1", network, corridor, verdict, complete, f"degradation vs {c.reference}: {basis}")
                verdicts.append((network, verdict, complete))
            n_conf = sum(v == "confirmed" for _, v, _ in verdicts)
            n_ref = sum(v == "refuted" for _, v, _ in verdicts)
            undecided = any(not v for _, v, _ in verdicts)
            if n_conf >= c.networks_min:
                state = "confirmed"
            elif n_ref >= c.networks_min:
                state = "refuted"
            else:
                state = "" if undecided else "open"
            per_corridor.append((corridor, state, verdicts))
        overall = verdict_every([state for _, state, _ in per_corridor])
        complete = self.complete_design() and all(done for _, _, v in per_corridor for _, _, done in v)
        basis = "; ".join(f"{corridor}: " + ", ".join(f"{n} {v or 'n/a'}" for n, v, _ in verdicts)
                          for corridor, _, verdicts in per_corridor)  # fmt: skip
        rule = f"confirmed for >= {c.networks_min} of {len(c.networks)} networks on every corridor"
        self.verdict("H12.1", "overall", corridor_text(self.corridors()), overall, complete, f"{rule}: {basis}")

    def micro_maker(self) -> TableMaker:
        """The table code of M4 for the training runs (H12.2 micro part, e4_rmax)."""
        if self._maker is None:
            c = self.cfg
            self._maker = TableMaker(TablesConfig(
                runs_root=c.runs_root, out_dir=c.out_dir, data=c.micro_data, folds=c.run_folds, seeds=c.run_seeds,
                n_resamples=c.n_resamples, level=c.level, seed=c.seed,
            ))  # fmt: skip
        return self._maker

    def take_notes(self, table: str) -> None:
        """The notes of the M4 table code for ``table`` into missing.txt."""
        maker = self.micro_maker()
        self.missing += [line for line in maker.missing if line.startswith(f"[{table}]")]
        maker.missing = [line for line in maker.missing if not line.startswith(f"[{table}]")]

    def table_h12_2(self) -> Table:
        """H12.2 micro part (D108): spacing RMSE on the NGSIM test parts against the IDM of the same folds."""
        c, t, maker = self.cfg, "h12_2", self.micro_maker()
        items = [("reference", c.micro_reference), ("candidate", c.micro_candidate),
                 *(("reference row", other) for other in c.micro_others)]  # fmt: skip
        loaded = {}
        for role, item in items:
            experiment, model = item.split("/", 1)
            seeds = c.run_seeds[:1] if role == "reference" else c.run_seeds
            runs = maker.runs(t, experiment, c.micro_data, model, seeds, (METRICS, EVENTS))
            loaded[item] = (role, experiment, model, runs, maker.drivers(t, runs), len(c.run_folds) * len(seeds))
        reference_drivers = loaded[c.micro_reference][4]
        rows = []
        for item, (role, experiment, model, runs, drivers, expected) in loaded.items():
            row: dict[str, Any] = {"role": role, "experiment": experiment, "model": model, "data": c.micro_data,
                                   "runs": maker.trained(runs), "runs_expected": expected}  # fmt: skip
            maker.rmse(row, drivers)
            row["rmse_runs"] = maker.values(t, runs, "test_rmse_s_mean").mean()
            if role != "reference":
                maker.versus(row, "rmse_vs_reference", reference_drivers, drivers)
                row["reference_rmse_s"] = (reference_drivers["rmse_s"].reindex(drivers.index).mean()
                                           if reference_drivers is not None and drivers is not None else np.nan)  # fmt: skip
                high = _float(row.get("rmse_vs_reference_high"))
                row["superior"] = bool(high < 0.0) if math.isfinite(high) else None
            row["complete"] = row["runs"] == expected
            rows.append(row)
        self.take_notes(t)
        frame = pd.DataFrame(rows, columns=[
            "role", "experiment", "model", "data", "runs", "runs_expected", "drivers", "rmse_s", "rmse_s_low",
            "rmse_s_high", "collision_rate", "rmse_runs", "reference_rmse_s", "rmse_vs_reference",
            "rmse_vs_reference_low", "rmse_vs_reference_high", "rmse_vs_reference_p", "rmse_vs_reference_pairs",
            "superior", "complete"])  # fmt: skip
        self.h12_2_verdict(frame)
        columns = [
            Column("role", "role", "text"), Column("experiment", "experiment", "text"), Column("model", "model", "text"),
            Column("runs", "runs", "int"), Column("drivers", "drivers", "int"),
            Column("rmse_s", "RMSE s (m)", ci=True), Column("rmse_runs", "mean over runs (m)"),
            Column("rmse_vs_reference_pairs", "pairs", "int"),
            Column("reference_rmse_s", "IDM on the pairs (m)"),
            Column("rmse_vs_reference", "RMSE vs IDM", "pct", 1, ci=True), Column("rmse_vs_reference_p", "p", "p"),
            Column("superior", "superior", "flag"), Column("complete", "complete", "flag"),
        ]  # fmt: skip
        reference = c.micro_reference.replace("/", " ")
        notes = [
            f"Runs: {sum(int(r['runs']) for r in rows)} of {sum(int(r['runs_expected']) for r in rows)} expected runs "
            "exist; missing runs and files: missing.txt.",
            f"H12.2, micro part (D108): closed-loop spacing RMSE on the test parts of {c.micro_data} (m), unit driver "
            "(per driver the mean over its events and the seeds, folds "
            f"{', '.join(map(str, c.run_folds))}), {c.micro_candidate.replace('/', ' ')} (certified hybrid after "
            f"fine-tuning) against {reference} of the same folds (the same split): relative difference of the mean "
            "RMSE, paired over the common drivers (pairs), interval over the drivers; p: Wilcoxon signed-rank test. "
            "superior: upper end below 0.",
            f"Reference rows (no verdict): {', '.join(o.replace('/', ' ') for o in c.micro_others)} against the same "
            "IDM. mean over runs: mean over the runs of rmse_s_mean of metrics.json (event means).",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]
        return Table(t, "H12.2: micro comparison with the IDM on the NGSIM test parts", notes, frame, columns)

    def h12_2_verdict(self, micro: pd.DataFrame) -> None:
        """H12.2 (D108): the macro part from the TOST of the macro triple on every corridor, the micro part from
        the candidate row of the micro comparison."""
        c = self.cfg
        tost = self.tost_frame if self.tost_frame is not None else pd.DataFrame(columns=["corridor", "metric"])
        holds, worse, texts, complete = [], [], [], self.complete_design()
        expected_pairs = {corridor: len(self.scenarios_of(corridor)) * len(c.seeds) for corridor in self.corridors()}
        for corridor in self.corridors():
            mine = tost[tost["corridor"] == corridor].set_index("metric")
            parts, equivalent, bad = [], [], []
            for component, (metric, side) in TRIPLE.items():
                row = mine.loc[metric].to_dict() if metric in mine.index else {}
                tested = math.isfinite(_float(row.get("p_value")))
                low, high = _float(row.get("relative_difference_low")), _float(row.get("relative_difference_high"))
                worse_here = (low > c.margin) if side == "longer" else (high < -c.margin)
                equivalent.append(bool(row.get("equivalent")) if tested else None)
                bad.append(bool(worse_here) if tested else None)
                up, down = DIRECTIONS[side]
                direction = up if _float(row.get("relative_difference")) > 0 else down
                state = ("equivalent" if equivalent[-1] else f"not equivalent ({direction})") if tested else "no test"
                if bad[-1]:
                    state += f", worse by > {100 * c.margin:g} %"
                parts.append(f"{component.replace('_', ' ')} {_interval(row, 'relative_difference')} {state}")
                complete = complete and row.get("pairs") == expected_pairs[corridor]
            holds.append(None if None in equivalent else all(equivalent))
            worse.append(True if any(b is True for b in bad) else (None if None in bad else False))
            texts.append(f"{corridor}: " + ", ".join(parts))
        candidate = micro[micro["role"] == "candidate"]
        cand = candidate.iloc[0].to_dict() if len(candidate) else {}
        superior = cand.get("superior") if isinstance(cand.get("superior"), bool) else None
        complete = complete and bool(micro["complete"].astype(bool).all()) if len(micro) else False
        drivers = _float(cand.get("rmse_vs_reference_pairs"))
        micro_text = (f"micro: RMSE vs {c.micro_reference.split('/')[-1]} {_interval(cand, 'rmse_vs_reference')} over "
                      f"{0 if math.isnan(drivers) else int(drivers)} drivers, "
                      f"{'superior' if superior else 'not superior' if superior is False else 'n/a'}")  # fmt: skip
        macro_holds = None if not holds or None in holds else all(holds)
        if any(w is True for w in worse):
            verdict, why = "refuted", f"macro part worse by > {100 * c.margin:g} %"
        elif macro_holds and superior:
            verdict, why = "confirmed", "both parts hold"
        elif macro_holds is None or superior is None:
            verdict, why = "", "a part lacks data"
        else:
            failing = [name for name, ok in (("macro part (TOST)", macro_holds), ("micro part", superior)) if not ok]
            verdict, why = "open", f"fails: {' and '.join(failing)}"
        others = micro[micro["role"] == "reference row"]
        information = "; ".join(f"{r['experiment']} {r['model']}: RMSE vs IDM {_interval(r, 'rmse_vs_reference')}"
                                for r in others.to_dict("records"))  # fmt: skip
        basis = f"{why}; macro (TOST +-{100 * c.margin:g} %): {'; '.join(texts) or 'no corridor'}; {micro_text}"
        self.verdict("H12.2", c.candidate, corridor_text(self.corridors()), verdict, complete, basis,
                     f"reference rows (no verdict): {information}" if information else "")

    def h12_3_verdicts(self) -> None:
        """H12.3 (D108) from the correlation table: per corridor the Spearman row decides; Pearson and the
        dynamic macro error get rows for information; overall: the Spearman verdicts of every corridor."""
        c = self.cfg
        corr = self.correlation_frame if self.correlation_frame is not None else pd.DataFrame()
        deciding = []
        rows = [("macro error", "spearman", "Spearman", True), ("macro error", "pearson", "Pearson", False),
                ("macro error (dynamic)", "spearman", "Spearman, dynamic macro error", False),
                ("macro error (dynamic)", "pearson", "Pearson, dynamic macro error", False)]  # fmt: skip
        for corridor in self.corridors():
            n_laws = len(self.laws_on(corridor, c.laws))
            for error, method, unit, decides in rows:
                pick = corr[(corr["corridor"] == corridor) & (corr["error"] == error) & (corr["method"] == method)
                            & (corr["measure"] == MEASURES["unstable_eq"]) & (corr["laws"] == "all laws")] \
                    if len(corr) else corr  # fmt: skip
                row = pick.iloc[0].to_dict() if len(pick) else {}
                verdict = verdict_correlation(row, c)
                n = int(_float(row.get("n"))) if math.isfinite(_float(row.get("n"))) else 0
                p = _float(row.get("p_permutation"))
                basis = (f"r {_interval(row, 'estimate', 3, pct=False)} over {n} laws, p (permutation) "
                         f"{'n/a' if math.isnan(p) else f'{p:.4f}'}")  # fmt: skip
                label = unit if decides else f"{unit} (does not decide)"
                self.verdict("H12.3", label, corridor, verdict, n == n_laws, basis,
                             "decides" if decides else "information")  # fmt: skip
                if decides:
                    deciding.append((corridor, verdict, n == n_laws))
        overall = verdict_every([v for _, v, _ in deciding])
        complete = self.complete_design() and all(done for _, _, done in deciding)
        basis = (f"Spearman, r > {c.correlation_confirm:g}, p < {c.correlation_alpha:g} and >= "
                 f"{c.correlation_min_laws} laws on every corridor: "
                 + ", ".join(f"{corridor} {v or 'n/a'}" for corridor, v, _ in deciding))  # fmt: skip
        self.verdict("H12.3", "overall", corridor_text(self.corridors()), overall, complete, basis)

    def verdict(self, hypothesis: str, unit: str, corridors: str, verdict: str, complete: bool, basis: str,
                information: str = "") -> None:  # fmt: skip
        self.verdicts.append({"hypothesis": hypothesis, "unit": unit, "corridors": corridors, "verdict": verdict,
                              "complete": bool(complete), "basis": basis, "information": information})  # fmt: skip

    def table_verdicts(self) -> Table:
        c = self.cfg
        columns = [Column("hypothesis", "hypothesis", "text"), Column("unit", "unit", "text"),
                   Column("corridors", "corridors", "text"), Column("verdict", "verdict", "text"),
                   Column("complete", "complete", "flag"), Column("basis", "basis", "text"),
                   Column("information", "information", "text")]  # fmt: skip
        names = ["hypothesis", "unit", "corridors", "verdict", "complete", "basis", "information"]
        notes = [
            "Verdicts of H12.1-H12.3 of Part A of the plan (D108); an empty verdict lacks data; complete: drawn from "
            "every run of the design on every corridor of the configuration (the corridors used are named).",
            f"H12.1: table h12_1. H12.2: macro part = TOST of {c.candidate} against {c.reference} on the macro triple "
            f"(table tost: throughput, mean travel time, wave speed by cross-correlation), holds when all three are "
            f"equivalent within +-{100 * c.margin:g} % on every corridor, worse by > {100 * c.margin:g} % when the "
            "interval of a relative difference lies beyond the margin on the bad side (lower throughput, longer "
            "travel time, slower wave speed in magnitude) on a corridor; micro part = table h12_2; confirmed when "
            "both parts hold, refuted when the macro part is worse, otherwise open.",
            f"H12.3: Spearman over the laws of the macro error against the share of unstable equilibria (table "
            f"instability_correlation, all laws): confirmed when r > {c.correlation_confirm:g}, p (permutation) < "
            f"{c.correlation_alpha:g} and at least {c.correlation_min_laws} laws, refuted when r < "
            f"{c.correlation_refute:g}, otherwise open; overall on every corridor. Pearson and the dynamic macro "
            "error: the same rule, for information.",
        ]
        frame = pd.DataFrame(self.verdicts, columns=names)
        return Table("verdicts", "Verdicts of H12.1-H12.3 (corridor)", notes, frame, columns)

    # ---------------------------------------------------------------------------------------- e4_rmax
    def table_e4_rmax(self) -> Table:
        """D111: the certificate-accuracy-corridor curve of the residual amplitude."""
        c, t, maker = self.cfg, "e4_rmax", self.micro_maker()
        expected = len(c.run_folds) * len(c.run_seeds)
        training = []
        specs = [*((spec, False) for spec in c.rmax), *((spec, True) for spec in c.rmax_ablation)]
        for spec, extra in specs:  # the ablation of the review: extra rows once its law is in the tables
            if extra and not any(spec["law"] in self.laws_on(k, [spec["law"]]) for k in self.corridors()):
                continue
            trained = bool(spec.get("highd") or spec.get("ngsim"))
            row: dict[str, Any] = {"r_max": float(spec["r_max"]), "core": spec["core"], "highd": spec.get("highd"),
                                   "ngsim": spec.get("ngsim"), "law": spec["law"],
                                   "runs_expected": expected if trained else None}  # fmt: skip
            for part, experiment, data in (("highd", spec.get("highd"), c.rmax_data),
                                           ("ngsim", spec.get("ngsim"), c.micro_data)):  # fmt: skip
                if not experiment:  # a row without training runs of its own (a law without residual): empty cells
                    continue
                certified = experiment != "e1"  # E1 has no certificate step
                files = (METRICS, AUDIT, EVENTS) + ((CERTIFICATE,) if certified else ())
                runs = maker.runs(t, experiment, data, c.rmax_model, c.run_seeds, files)
                row[f"runs_{part}"] = maker.trained(runs)
                if certified and len(runs):
                    applicable = _column(runs, "certificate_applicable").eq(True)
                    row[f"certificates_{part}"] = int(applicable.sum())
                    row[f"a_priori_{part}"] = int(_column(runs[applicable], "certificate_a_priori_holds").eq(True).sum())
                share = maker.unstable_eq(t, runs)
                row[f"unstable_eq_{part}"], row[f"unstable_eq_{part}_low"], row[f"unstable_eq_{part}_high"] = (
                    share["estimate"], share["low"], share["high"])  # fmt: skip
                rmse: dict[str, Any] = {}
                maker.rmse(rmse, maker.drivers(t, runs))
                row[f"rmse_{part}"], row[f"rmse_{part}_low"], row[f"rmse_{part}_high"] = (
                    rmse["rmse_s"], rmse["rmse_s_low"], rmse["rmse_s_high"])  # fmt: skip
                row[f"drivers_{part}"] = rmse["drivers"]
            training.append(row)
        self.take_notes(t)
        rows = []
        for corridor in self.corridors():
            for base in training:
                if base["law"] not in self.laws_on(corridor, [base["law"]]):
                    continue
                row = {"corridor": corridor, **base}
                known = (*c.laws, *c.rmax_laws, *c.ablation_laws)
                mine = self.mine(corridor, base["law"]) if base["law"] in known else self.frame([])
                row["runs_corridor"] = len(mine)
                for key in ("macro_error", "macro_error_dynamic", "collisions_per_1000_vkm"):
                    self.put(row, key, mine[key] if len(mine) else [])
                rows.append(row)
        frame = pd.DataFrame(rows)
        extra = {spec["law"] for spec in c.rmax_ablation} - {spec["law"] for spec in c.rmax}
        if len(frame) and frame["law"].isin(extra).any():  # counts of the rows without training runs stay empty and
            regular = ~frame["law"].isin(extra)  # those of the other rows integers, as without these rows
            for key in [k for k in frame.columns if k == "runs_expected" or k.startswith(("runs_", "certificates_",
                                                                                         "a_priori_", "drivers_"))]:
                if frame[key].isna().any() and frame.loc[regular, key].notna().all():
                    frame[key] = frame[key].astype("Int64")
        columns = [
            Column("corridor", "corridor", "text"), Column("r_max", "r_max", "weight"), Column("core", "core", "text"),
            Column("runs_highd", "runs HighD", "int"), Column("a_priori_highd", "a priori holds", "int"),
            Column("runs_ngsim", "runs NGSIM", "int"), Column("a_priori_ngsim", "holds after fine-tuning", "int"),
            Column("unstable_eq_highd", "unstable among eq. HighD", ci=True),
            Column("unstable_eq_ngsim", "unstable among eq. NGSIM", ci=True),
            Column("rmse_highd", "RMSE s HighD (m)", ci=True), Column("rmse_ngsim", "RMSE s NGSIM (m)", ci=True),
            Column("runs_corridor", "corridor runs", "int"), Column("macro_error", "macro error", digits=3, ci=True),
            Column("macro_error_dynamic", "macro error (dynamic)", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3, ci=True),
        ]  # fmt: skip
        notes = [
            f"Residual-amplitude sweep of the hybrid (D111): per r_max the runs on {c.rmax_data} (certified: core "
            f"with margin, certified budget; free: no certificate) and their fine-tuning on {c.micro_data}, "
            f"{len(c.run_folds)} folds x {len(c.run_seeds)} seeds each ({expected} runs per experiment), and the "
            "corridor law of the fine-tuned runs. r_max 0.3 certified: e4_stable, e4_stable_ft, residual_idm_certified; "
            "free 1.0: the ResidualIDM of E1, e4_free_ft, residual_idm.",
            "a priori holds / holds after fine-tuning: runs whose a priori certificate (certificate.json) holds, on "
            "HighD and after the fine-tuning (no certificate step in E1). unstable among eq.: share_unstable_numerical, "
            "unit run, mean over the runs with interval. RMSE s: spacing RMSE of the test parts (m), unit driver.",
            "Corridor: macro error, dynamic macro error and collisions per 1000 vehicle-km of the law (table laws), "
            "unit run (scenario and seed), mean with interval.",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}; "
            "missing runs and files: missing.txt.",
        ]
        shown = [spec for spec in c.rmax_ablation if len(frame) and spec["law"] in set(frame["law"])]
        if shown:
            rows_text = "; ".join(
                f"{spec['law']} (r_max {float(spec['r_max']):g}, {spec['core']}"
                + (f": {spec.get('highd') or 'no run on ' + c.rmax_data}, {spec.get('ngsim') or 'no fine-tuning'})"
                   if spec.get("highd") or spec.get("ngsim") else ": no training run of its own, empty cells)")
                for spec in shown)  # fmt: skip
            notes.insert(3, f"Extra rows of the factorial ablation of the certified hybrid (review of M8; r_max 0: no "
                            f"residual; not in the verdicts): {rows_text}.")  # fmt: skip
        return Table(t, "E4: residual amplitude, certificate, accuracy and corridor (D111)", notes, frame, columns)

    # ------------------------------------------------------------------------------------ sensitivity
    def table_sensitivity(self) -> Table:
        """D113: per variant of the scenario and law the macro error, the macro triple and the collisions, and the
        ranking of the laws by macro error in every variant."""
        c, t = self.cfg, "sensitivity"
        corridor = self.corridor_of(c.sensitivity_scenario)
        rows = []
        variants = [("baseline", c.sensitivity_scenario),
                    *((v, f"{c.sensitivity_scenario}_{v}") for v in c.sensitivity_variants)]  # fmt: skip
        for variant, scenario in variants:
            built = (c.corridor_root / scenario).is_dir() or (c.corridor_root / "scenarios" / scenario).is_dir()
            if not built:
                self.note(t, f"{scenario}: scenario not built, no runs")
            group = []
            for law in c.sensitivity_laws:
                found = self.frame(self.read_runs(t, corridor, law, [scenario], c.sensitivity_seeds) if built else [])
                mine = found[found["present"]] if len(found) else found
                row: dict[str, Any] = {"variant": variant, "scenario": scenario, "law": law, "runs": len(mine),
                                       "runs_expected": len(c.sensitivity_seeds)}  # fmt: skip
                for key in ("macro_error", *(f"error_{n}" for n in TRIPLE), "collisions_per_1000_vkm"):
                    self.put(row, key, mine[key] if len(mine) else [])
                group.append(row)
            known = [r for r in group if math.isfinite(_float(r["macro_error"]))]
            order = [r["law"] for r in sorted(known, key=lambda r: r["macro_error"])]
            for row in group:
                row["rank"] = order.index(row["law"]) + 1 if row["law"] in order else np.nan
                row["ranking"] = " < ".join(order) if len(order) == len(group) else ""
                row["order_holds"] = (order == list(c.sensitivity_order)) if len(order) == len(group) else None
            rows += group
        frame = pd.DataFrame(rows)
        expected = " < ".join(c.sensitivity_order)
        columns = [
            Column("variant", "variant", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("macro_error", "macro error", digits=3, ci=True),
            *(Column(f"error_{n}", COMPONENT_HEADERS[n], digits=3, ci=True) for n in TRIPLE),
            Column("collisions_per_1000_vkm", "collisions / 1000 veh-km", digits=3, ci=True),
            Column("rank", "rank", "int"), Column("ranking", "ranking by macro error", "text"),
            Column("order_holds", f"{expected}", "flag"),
        ]  # fmt: skip
        notes = [
            f"Sensitivity of the corridor results (D113): variants of {c.sensitivity_scenario} (scenario "
            f"{c.sensitivity_scenario}_<variant>: {', '.join(c.sensitivity_variants)}) and the scenario itself "
            f"(baseline), laws {', '.join(c.sensitivity_laws)}, seeds {', '.join(map(str, c.sensitivity_seeds))} (the "
            "baseline with the same seeds).",
            "macro error and the macro triple (signed relative errors of throughput, travel time (W1) and wave speed "
            "by cross-correlation against the ground truth of the variant's scenario), collisions per 1000 "
            "vehicle-km: unit run (seed), mean with interval.",
            f"rank: 1 = smallest mean macro error within the variant; the last column marks the variants whose ranking "
            f"is {expected} (empty while a law has no runs).",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}; "
            "missing runs and files: missing.txt.",
        ]
        return Table(t, "Sensitivity to the downstream boundary and the lane-change model (D113)", notes, frame,
                     columns)  # fmt: skip

    # ----------------------------------------------------------------------------- M8: asymmetry (D121)
    def asymmetry_truth(self, scenario: str) -> dict[str, Any]:
        """The values of the ``asymmetry.json`` of the ground truth of a scenario (read once; missing: noted)."""
        if scenario not in self._asymmetry_truths:
            payload = self._read("asymmetry", self.cfg.corridor_root / "scenarios" / scenario / "asymmetry.json")
            values = {} if payload is None else {**flatten_asymmetry(payload), "window": window_text(payload.get("window"))}
            if payload is not None:
                self._asymmetry_hashes.add(payload.get("config_hash"))
                if self._asymmetry_config is None and isinstance(payload.get("config"), Mapping):
                    self._asymmetry_config = dict(payload["config"])
            self._asymmetry_truths[scenario] = values
        return self._asymmetry_truths[scenario]

    def asymmetry_runs(self, corridor: str, law: str) -> pd.DataFrame:
        """One row per run of ``law`` on its scenarios of ``corridor`` with the values of its ``asymmetry.json`` and their
        signed relative errors against the ground truth of the scenario (``error_<name>``); ``present`` marks the
        runs whose file was read and has the analysis window of the truth. Missing runs and files are noted."""
        key = (corridor, law)
        if key in self._asymmetry:
            return self._asymmetry[key]
        c, t, root = self.cfg, "asymmetry", self.cfg.corridor_root
        expected = [(s, seed) for s in self.scenarios_for(corridor, law) for seed in c.seeds]
        rows = [{"corridor": corridor, "law": law, "scenario": s, "seed": seed, "present": False} for s, seed in expected]
        if len(expected) > 1 and not any((root / s / law / f"seed{seed}").is_dir() for s, seed in expected):
            self.note(t, f"{corridor}/{law}: run missing (all {len(expected)})")
            rows = []
        for row in rows:
            run = root / row["scenario"] / law / f"seed{row['seed']}"
            if not run.is_dir():
                self.note(t, f"{self.label(run)}: run missing")
                continue
            payload = self._read(t, run / "asymmetry.json")
            if payload is None:
                continue
            self._asymmetry_hashes.add(payload.get("config_hash"))
            if self._asymmetry_config is None and isinstance(payload.get("config"), Mapping):
                self._asymmetry_config = dict(payload["config"])
            truth = self.asymmetry_truth(row["scenario"])
            window = window_text(payload.get("window"))
            if truth.get("window") is not None and window != truth["window"]:
                self.note(t, f"{row['scenario']}/{law}/seed{row['seed']}: window {window} s, ground truth "
                             f"{truth['window']} s: left out, rerun scripts/corridor_asymmetry.py")  # fmt: skip
                continue
            row.update(flatten_asymmetry(payload), present=True)
            for name in ASYMMETRY_ERRORS:
                row[f"error_{name}"] = relative_error(row.get(name), truth.get(name))
        columns = ["corridor", "law", "scenario", "seed", "present", *ASYMMETRY_VALUES,
                   *(f"error_{n}" for n in ASYMMETRY_ERRORS)]  # fmt: skip
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan if column != "present" else False
        frame["present"] = frame["present"].astype(bool)
        self._asymmetry[key] = frame[columns]
        return self._asymmetry[key]

    def table_asymmetry(self) -> tuple[Table, Table]:
        """D121: per corridor and law the means of the asymmetry and spectrum values with intervals over the runs,
        the signed relative errors against the truth, the ground truth rows; and the contrasts of the penalised and
        certified laws with their free counterparts."""
        c, t = self.cfg, "asymmetry"
        rows = []
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.table_laws()):
                runs = self.asymmetry_runs(corridor, law)
                mine = runs[runs["present"]]
                row: dict[str, Any] = {"corridor": corridor, "law": law, "scenario": "all", "runs": len(mine),
                                       "runs_expected": len(self.scenarios_for(corridor, law)) * len(c.seeds),
                                       "scenario_set": self.scenario_set(corridor, law)}  # fmt: skip
                for key in ASYMMETRY_VALUES:
                    self.put(row, key, mine[key])
                for name in ASYMMETRY_ERRORS:
                    self.put(row, f"error_{name}", mine[f"error_{name}"])
                    self.put(row, f"abs_error_{name}", mine[f"error_{name}"].abs())
                rows.append(row)
            truths = []
            for scenario in self.scenarios_of(corridor):
                values = self.asymmetry_truth(scenario)
                truths.append({"corridor": corridor, "law": "ground truth", "scenario": scenario,
                               **{key: values.get(key, np.nan) for key in ASYMMETRY_VALUES}})  # fmt: skip
            rows += truths
            if truths:
                rows.append({"corridor": corridor, "law": "ground truth", "scenario": "mean",
                             **{key: float(np.nanmean([r[key] for r in truths]))
                                if any(np.isfinite(r[key]) for r in truths) else np.nan for key in ASYMMETRY_VALUES}})  # fmt: skip
        hashes = self._asymmetry_hashes - {None}
        if len(hashes) > len(self.corridors()):  # one per corridor: the detectors and lanes of a site differ
            self.note(t, f"the asymmetry.json files hold {len(hashes)} config hashes for {len(self.corridors())} "
                         "corridors: rerun scripts/corridor_asymmetry.py")  # fmt: skip
        names = ["corridor", "law", "scenario", "runs", "runs_expected", "scenario_set"]
        for key in (*ASYMMETRY_VALUES, *(f"{p}_{n}" for n in ASYMMETRY_ERRORS for p in ("error", "abs_error"))):
            names += [key, f"{key}_low", f"{key}_high"]
        frame = pd.DataFrame(rows, columns=names)  # every column, also without runs
        for key in ("runs", "runs_expected"):
            frame[key] = frame[key].astype("Int64")
        a = self.cfg_asymmetry_text()
        columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("scenario", "scenario", "text"),
            Column("runs", "runs", "int"), Column("scenario_set", "scenarios", "text"),
            *(Column(key, header, digits=digits, ci=True) for key, (header, digits) in ASYMMETRY_VALUES.items()
              if key not in ("mean_acceleration", "mean_deceleration")),
            Column("error_asymmetry_index", "error of the index", "pct", 1, ci=True),
            Column("error_peak_frequency", "error of the peak frequency", "pct", 1, ci=True),
            Column("error_centroid", "error of the centroid", "pct", 1, ci=True),
        ]  # fmt: skip
        notes = [
            f"Runs: {int(frame['runs'].fillna(0).sum()) if 'runs' in frame else 0} runs with an asymmetry.json "
            "(scripts/corridor_asymmetry.py; D121); missing runs and files: missing_corridor.txt.",
            f"Acceleration asymmetry ({a}): a = v(t + 1) - v(t) of every vehicle over the whole seconds inside the "
            "analysis window, on the lanes of the wave field (main lanes); asymmetry index = mean a over a > threshold "
            "divided by the mean |a| over a < -threshold; accelerating / decelerating share: of the vehicle-seconds.",
            "Spectrum: Welch PSD of the speed series of the virtual detectors (mean speed of the passing vehicles per 2 "
            "s, gaps interpolated; Hann window of 128 s, half overlap), averaged over the detectors; peak frequency "
            "and spectral centroid in 0.002-0.05 Hz (the lines k / 128 Hz, k = 1..6); band RMS: root of the power in "
            "the band (m/s).",
            "Unit: run (scenario and seed); mean over the runs of the law on its scenarios of the corridor with its "
            "interval; errors: signed relative errors (run - truth) / |truth| against the ground truth of the run's "
            "scenario. Ground truth: per scenario, and their mean.",
            *self.ablation_note(),
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]
        table = Table(t, "Acceleration asymmetry and oscillation spectrum (D121)", notes, frame, columns)
        return table, self.table_asymmetry_contrasts()

    def cfg_asymmetry_text(self) -> str:
        """The settings of the asymmetry files that were read (their ``config``)."""
        settings = self._asymmetry_config or {}
        if not settings:
            return "no file read"
        return (f"threshold {_float(settings.get('threshold')):g} m/s^2, detector series of "
                f"{_float(settings.get('sample_s')):g} s, Welch segments of {_float(settings.get('window_s')):g} s, band "
                f"{_float(settings.get('f_min')):g}-{_float(settings.get('f_max')):g} Hz, detectors with passages in at "
                f"least {100 * _float(settings.get('min_coverage')):g} % of the intervals")  # fmt: skip

    def table_asymmetry_contrasts(self) -> Table:
        c, t = self.cfg, "asymmetry_contrasts"
        index = ["scenario", "seed"]
        keys = ("asymmetry_index", "accelerating_share", "decelerating_share", "peak_frequency", "centroid", "band_rms")
        rows = []
        for corridor in self.corridors():
            present = self.laws_on(corridor, c.table_laws())
            for law, free in c.asymmetry_contrasts:
                if law not in present or free not in present:
                    self.note(t, f"{corridor}: {law} or {free} not in the tables, no contrast")
                    continue
                a = self.asymmetry_runs(corridor, law)
                b = self.asymmetry_runs(corridor, free)
                a, b = a[a["present"]].set_index(index), b[b["present"]].set_index(index)
                pairs = a.join(b, how="inner", lsuffix="_law", rsuffix="_free")
                row: dict[str, Any] = {"corridor": corridor, "law": law, "versus": free, "pairs": len(pairs)}
                if len(pairs) < 2:
                    self.note(t, f"{corridor}: {law} against {free}: {len(pairs)} pairs, no interval")
                for key in keys:
                    row[f"{key}_law"] = pairs[f"{key}_law"].mean() if len(pairs) else np.nan
                    row[f"{key}_free"] = pairs[f"{key}_free"].mean() if len(pairs) else np.nan
                    self.put(row, f"{key}_difference", (pairs[f"{key}_law"] - pairs[f"{key}_free"]).to_numpy()
                             if len(pairs) else [])  # fmt: skip
                for name in ASYMMETRY_ERRORS:
                    closer = (pairs[f"error_{name}_law"].abs() - pairs[f"error_{name}_free"].abs()).to_numpy() \
                        if len(pairs) else []  # fmt: skip
                    self.put(row, f"closer_{name}", closer)
                low, high = _float(row.get("asymmetry_index_difference_low")), _float(row.get("asymmetry_index_difference_high"))
                row["index_changed"] = bool(low > 0 or high < 0) if math.isfinite(low) and math.isfinite(high) else None
                low, high = _float(row.get("closer_asymmetry_index_low")), _float(row.get("closer_asymmetry_index_high"))
                row["index_closer"] = ("closer" if high < 0 else "further" if low > 0 else "no difference") \
                    if math.isfinite(low) and math.isfinite(high) else None  # fmt: skip
                rows.append(row)
        names = ["corridor", "law", "versus", "pairs"]
        for key in keys:
            names += [f"{key}_law", f"{key}_free", *(f"{key}_difference{end}" for end in ("", "_low", "_high"))]
        names += [f"closer_{name}{end}" for name in ASYMMETRY_ERRORS for end in ("", "_low", "_high")]
        frame = pd.DataFrame(rows, columns=[*names, "index_changed", "index_closer"])
        columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("versus", "free law", "text"),
            Column("pairs", "pairs", "int"), Column("asymmetry_index_law", "index", digits=3),
            Column("asymmetry_index_free", "index (free)", digits=3),
            Column("asymmetry_index_difference", "index difference", digits=3, ci=True),
            Column("index_changed", "changed", "flag"),
            Column("closer_asymmetry_index", "|error| difference", digits=3, ci=True),
            Column("index_closer", "to the truth", "text"),
            Column("accelerating_share_difference", "accelerating share difference", digits=3, ci=True),
            Column("centroid_difference", "centroid difference (Hz)", digits=4, ci=True),
            Column("peak_frequency_difference", "peak frequency difference (Hz)", digits=4, ci=True),
            Column("band_rms_difference", "band RMS difference (m/s)", digits=3, ci=True),
        ]  # fmt: skip
        notes = [
            "Penalties and certificate against the free laws (D121): every pair (law, free law), paired by scenario and "
            "seed (pairs); differences law - free law of the asymmetry index, the accelerating share, the spectral "
            "centroid, the peak frequency and the band RMS, mean with interval over the pairs; changed: the interval "
            "of the index difference excludes 0.",
            "|error| difference: |e_law| - |e_free| of the signed relative error of the index against the ground truth "
            "(negative: the law is closer to the truth); to the truth: closer / further when its interval excludes 0.",
            f"Pairs: {', '.join(f'{a} vs {b}' for a, b in c.asymmetry_contrasts)}. Intervals: {100 * c.level:g} % "
            f"percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]
        return Table(t, "Asymmetry: penalised and certified laws against the free laws (D121)", notes, frame, columns)

    # -------------------------------------------------------------------- M8: pooled correlation (D122)
    def table_correlation_pooled(self) -> Table:
        """H12.3 over every law with runs, per corridor and pooled with the corridor as a stratum."""
        c, t = self.cfg, "correlation_pooled"
        records = []
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.pooled()):
                mine = self.mine(corridor, law)
                if not len(mine):
                    self.note(t, f"{corridor}/{law}: no run with a macro.json, not in the correlation")
                    continue
                instability = self.instability_of(t, law)
                records.append({"corridor": corridor, "law": law, "unstable_eq": instability["unstable_eq"],
                                **{key: float(mine[key].mean()) for key in ERRORS}})  # fmt: skip
        laws_frame = pd.DataFrame(records, columns=["corridor", "law", "unstable_eq", *ERRORS])
        self.pooled_laws_frame = laws_frame
        unknown = laws_frame[laws_frame["unstable_eq"].isna()]
        for law in dict.fromkeys(unknown["law"]):
            self.note(t, f"{law}: no instability (audits of the members or the closed form), not in the correlation")
        subsets = {"all laws with runs (D122)": c.pooled(), "laws of H12.3 (D108)": c.laws}
        rows = []
        for subset, laws in subsets.items():
            part = laws_frame[laws_frame["law"].isin(laws) & laws_frame["unstable_eq"].notna()]
            for key, error in ERRORS.items():
                scopes = [(corridor, part[part["corridor"] == corridor]) for corridor in self.corridors()]
                if len(self.corridors()) > 1:
                    scopes.append(("pooled (corridor as stratum)", part))
                for scope, mine in scopes:
                    mine = mine[np.isfinite(mine[key].astype(float))]
                    x, y, strata = mine["unstable_eq"].to_numpy(float), mine[key].to_numpy(float), mine["corridor"].to_numpy()
                    r = stratified_spearman_ci(x, y, strata, c.n_resamples, c.level, c.seed)
                    p = stratified_permutation_p(x, y, strata, c.n_permutations, c.seed)
                    row = {"scope": scope, "error": error, "laws": subset, "n": r["n"], "n_strata": r["n_strata"],
                           "estimate": r["estimate"], "estimate_low": r["low"], "estimate_high": r["high"],
                           "n_valid": r["n_valid"], "p_permutation": p,
                           "law_list": ", ".join(dict.fromkeys(mine["law"]))}  # fmt: skip
                    row["h12_3_rule"] = verdict_correlation(row, c)
                    rows.append(row)
        frame = pd.DataFrame(rows, columns=["scope", "error", "laws", "n", "n_strata", "estimate", "estimate_low",
                                            "estimate_high", "n_valid", "p_permutation", "h12_3_rule", "law_list"])  # fmt: skip
        columns = [
            Column("scope", "corridor", "text"), Column("error", "error", "text"), Column("laws", "laws", "text"),
            Column("n", "n", "int"), Column("estimate", "Spearman", digits=3, ci=True),
            Column("n_valid", "valid resamples", "int"), Column("p_permutation", "p (permutation)", "p"),
            Column("h12_3_rule", "rule of H12.3", "text"),
        ]  # fmt: skip
        pooled_laws = ", ".join(c.pooled())
        notes = [
            "H12.3 over the laws (D122): Spearman correlation of the mean macro error of a law on a corridor (all "
            "components, or the dynamic ones) with the share of unstable equilibria among the equilibria of its "
            "members (table instability; the residual amplitudes of D111 from their member audits, the closed-form "
            "laws from their parameter sets).",
            f"Laws: all laws with runs (D122: {pooled_laws}) and, for comparison, the laws of the verdicts of H12.3 "
            f"(D108: {', '.join(c.laws)}); a law without runs or without instability is left out (missing_corridor.txt).",
            "Per corridor over its laws (n laws); pooled with the corridor as a stratum: the ranks of the instability "
            "and of the error within every corridor, scaled to (0, 1) as (rank - 0.5) / n, then the Pearson "
            "correlation of the pooled ranks over the (law, corridor) pairs (n); bootstrap: the laws resampled within "
            "every corridor; p (permutation): two-sided, the instability permuted within every corridor, "
            f"{c.n_permutations} permutations, (1 + hits) / (1 + permutations).",
            f"rule of H12.3 (for information, the verdicts stay those of D108): r > {c.correlation_confirm:g}, p < "
            f"{c.correlation_alpha:g} and n >= {c.correlation_min_laws}: confirmed; r < {c.correlation_refute:g}: "
            "refuted; otherwise open.",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]
        return Table(t, "H12.3 over all laws, per corridor and pooled (D122)", notes, frame, columns)

    # ---------------------------------------------------------------------------------- M8: power (D122)
    def table_power(self) -> Table:
        """The spread over seeds within a scenario and the seeds per scenario that detect a relative difference."""
        c, t = self.cfg, "power"
        first, second = c.power_laws[0], c.power_laws[1]
        index = ["scenario", "seed"]
        rows = []
        for corridor in self.corridors():
            if not {first, second} <= set(self.laws_on(corridor, c.table_laws())):
                self.note(t, f"{corridor}: {first} or {second} not in the tables, no power")
                continue
            a_runs, b_runs = self.mine(corridor, first), self.mine(corridor, second)
            for metric in c.power_metrics:
                header, digits = POWER_METRICS.get(metric, (metric, 3))
                a = a_runs.set_index(index)[metric].astype(float)
                b = b_runs.set_index(index)[metric].astype(float)
                pairs = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
                scenario = pairs.index.get_level_values("scenario")
                k = len(set(scenario))
                n_seeds = int(pairs.groupby(level="scenario").size().min()) if len(pairs) else 0
                row: dict[str, Any] = {"corridor": corridor, "metric": metric, "header": header, "law_a": first,
                                       "law_b": second, "pairs": len(pairs), "scenarios": k, "seeds": n_seeds}  # fmt: skip
                if len(pairs) < 4 or k < 1:
                    self.note(t, f"{corridor}: {metric}: {len(pairs)} pairs of {first} and {second}, no power")
                    rows.append(row)
                    continue
                difference = pairs["b"] - pairs["a"]
                mean = 0.5 * (pairs["a"].mean() + pairs["b"].mean())
                sd_a = pooled_sd(a.to_numpy(), a.index.get_level_values("scenario"))
                sd_b = pooled_sd(b.to_numpy(), b.index.get_level_values("scenario"))
                sd_d = pooled_sd(difference.to_numpy(), scenario)
                delta = c.power_difference * abs(mean)
                if math.isnan(sd_d) or (sd_d == 0 and delta == 0):
                    effect = math.nan
                else:
                    effect = delta / sd_d if sd_d > 0 else math.inf
                within = pairs.groupby(level="scenario").apply(
                    lambda g: g["a"].corr(g["b"]) if len(g) > 2 and g["a"].std() > 0 and g["b"].std() > 0 else np.nan)
                cmp = paired_comparison(pairs["a"], pairs["b"], n_resamples=c.n_resamples, level=c.level, seed=c.seed)
                mdd = minimal_detectable(sd_d, n_seeds, k, c.power_alpha, c.power_target)
                row.update(
                    mean_a=pairs["a"].mean(), mean_b=pairs["b"].mean(), sd_a=sd_a, sd_b=sd_b,
                    cv_a=sd_a / abs(pairs["a"].mean()) if pairs["a"].mean() else np.nan,
                    cv_b=sd_b / abs(pairs["b"].mean()) if pairs["b"].mean() else np.nan,
                    sd_difference=sd_d, correlation_within=float(np.nanmean(within)) if within.notna().any() else np.nan,
                    delta=delta, effect=effect, n_one_scenario=seeds_needed(effect, 1, c.power_alpha, c.power_target),
                    n_per_scenario=seeds_needed(effect, k, c.power_alpha, c.power_target),
                    power_design=paired_t_power(effect, k * n_seeds, k * (n_seeds - 1), c.power_alpha),
                    mdd=mdd, mdd_relative=mdd / abs(mean) if mean else np.nan,
                    observed=cmp["relative_change"], observed_low=cmp["ci_low"], observed_high=cmp["ci_high"],
                )  # fmt: skip
                row["sentence"] = self.power_sentence(row, digits)
                rows.append(row)
        frame = pd.DataFrame(rows, columns=[
            "corridor", "metric", "header", "law_a", "law_b", "pairs", "scenarios", "seeds", "mean_a", "mean_b", "sd_a",
            "sd_b", "cv_a", "cv_b", "sd_difference", "correlation_within", "delta", "effect", "n_one_scenario",
            "n_per_scenario", "power_design", "mdd", "mdd_relative", "observed", "observed_low", "observed_high",
            "sentence"])  # fmt: skip
        for key in ("n_one_scenario", "n_per_scenario"):
            frame[key] = frame[key].astype("Int64")
        columns = [
            Column("corridor", "corridor", "text"), Column("header", "metric", "text"), Column("pairs", "pairs", "int"),
            Column("mean_a", f"mean {first}", digits=3), Column("mean_b", f"mean {second}", digits=3),
            Column("sd_a", f"SD {first}", digits=4), Column("sd_b", f"SD {second}", digits=4),
            Column("sd_difference", "SD of the difference", digits=4), Column("delta", "10 % of the mean", digits=4),
            Column("n_one_scenario", "seeds (one scenario)", "int"),
            Column("n_per_scenario", "seeds per scenario (all scenarios)", "int"),
            Column("power_design", "power of the design", digits=3),
            Column("mdd_relative", "detectable with the design", "pct", 1),
            Column("observed", f"observed {second} vs {first}", "pct", 1, ci=True),
        ]  # fmt: skip
        notes = [
            f"Power of the corridor design (D122): {first} and {second}, unit run (scenario and seed), paired by seed "
            "within every scenario (pairs).",
            "SD: standard deviation over the seeds within a scenario, pooled over the scenarios of the corridor "
            "(sqrt of the mean within-scenario variance); SD of the difference: the same for the paired difference "
            f"{second} - {first}.",
            f"Seeds needed to detect a difference of {100 * c.power_difference:g} % of the mean of the two laws "
            f"(10 % of the mean) at alpha {c.power_alpha:g} (two-sided) with power {c.power_target:g}: paired t-test "
            "(noncentral t) with the observed SD of the difference; one scenario: the seeds of a single scenario; all "
            "scenarios: seeds per scenario when every scenario of the corridor contributes its pairs (the scenario "
            "means removed, df = scenarios x (seeds - 1)).",
            "power of the design: of that test with the seeds of the design on every scenario; detectable: the "
            f"smallest difference (relative to the mean) the design detects with power {c.power_target:g}; observed: "
            f"mean({second}) / mean({first}) - 1 over the pairs with its interval.",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}.",
        ]
        footer = [f"- {s}" for s in frame["sentence"].dropna()]
        return Table(t, "Seed-to-seed spread and power of the corridor comparisons (D122)", notes, frame, columns,
                     ["Sentences for the paper:", "", *footer] if footer else [])  # fmt: skip

    def power_sentence(self, row: Mapping[str, Any], digits: int) -> str:
        c = self.cfg

        def number(value: Any, extra: int = 1) -> str:
            value = _float(value)
            return "n/a" if math.isnan(value) else f"{value:.{digits + extra}f}"

        def count(value: Any) -> str:
            return "more than any feasible number of" if value is None or _float(value) != _float(value) else str(int(value))

        return (
            f"On {row['corridor']}, the seed-to-seed standard deviation of the {row['header']} within a scenario is "
            f"{number(row['sd_a'])} for {row['law_a']} and {number(row['sd_b'])} for {row['law_b']} (of the paired "
            f"difference: {number(row['sd_difference'])}); detecting a {100 * c.power_difference:g} % difference "
            f"({number(row['delta'])}) between two laws at alpha {c.power_alpha:g} with power {c.power_target:g}, paired "
            f"by seed, needs {count(row['n_one_scenario'])} seeds in one scenario or {count(row['n_per_scenario'])} "
            f"seeds per scenario over the {row['scenarios']} scenarios of the corridor; the {row['seeds']} seeds per "
            f"scenario of the design detect {100 * _float(row['mdd_relative']):.1f} % with power {c.power_target:g} "
            f"(power {_float(row['power_design']):.2f} for {100 * c.power_difference:g} %)."
        )

    # ------------------------------------------------------------------------------- M8: temporal (D120)
    def table_temporal(self) -> Table:
        """D120: the laws fine-tuned on period 0 against the same laws of D97 on the periods they did not see."""
        c, t = self.cfg, "temporal"
        index = ["scenario", "seed"]
        corridor = self.corridor_of(c.temporal_scenarios[0]) if c.temporal_scenarios else None
        rows = []
        for law, base in c.temporal_pairs.items():
            row: dict[str, Any] = {"corridor": corridor, "law": law, "versus": base,
                                   "scenarios": ", ".join(c.temporal_scenarios),
                                   "runs_expected": len(c.temporal_scenarios) * len(c.seeds)}  # fmt: skip
            on = corridor in self.corridors() and law in self.laws_on(corridor, c.table_laws())
            if not on:
                self.note(t, f"{law}: no runs yet (D120), no comparison")
                rows.append({**row, "runs": 0, "runs_versus": 0})
                continue
            mine = self.mine(corridor, law)
            reference = self.mine(corridor, base)
            mine = mine[mine["scenario"].isin(c.temporal_scenarios)]
            reference = reference[reference["scenario"].isin(c.temporal_scenarios)]
            row.update(runs=len(mine), runs_versus=len(reference))
            for key in ("macro_error", "macro_error_dynamic", "collisions_per_1000_vkm"):
                self.put(row, f"{key}_law", mine[key])
                self.put(row, f"{key}_versus", reference[key])
            for key in ("macro_error", "macro_error_dynamic"):
                pairs = pd.concat({"law": mine.set_index(index)[key].astype(float),
                                   "versus": reference.set_index(index)[key].astype(float)}, axis=1,
                                  join="inner").dropna()  # fmt: skip
                row[f"{key}_pairs"] = len(pairs)
                self.put(row, f"{key}_difference", (pairs["law"] - pairs["versus"]).to_numpy())
                if len(pairs) >= 2:
                    cmp = paired_comparison(pairs["versus"], pairs["law"], n_resamples=c.n_resamples, level=c.level,
                                            seed=c.seed)  # fmt: skip
                    row.update({f"{key}_relative": cmp["relative_change"], f"{key}_relative_low": cmp["ci_low"],
                                f"{key}_relative_high": cmp["ci_high"], f"{key}_p": cmp["p_value"]})  # fmt: skip
            low = _float(row.get("macro_error_difference_low"))
            high = _float(row.get("macro_error_difference_high"))
            row["outcome"] = ("worse" if low > 0 else "better" if high < 0 else "no difference") \
                if math.isfinite(low) and math.isfinite(high) else None  # fmt: skip
            rows.append(row)
        keys = ["corridor", "law", "versus", "scenarios", "runs_expected", "runs", "runs_versus"]
        for key in ("macro_error", "macro_error_dynamic", "collisions_per_1000_vkm"):
            keys += [f"{key}_{side}{end}" for side in ("law", "versus") for end in ("", "_low", "_high")]
        for key in ("macro_error", "macro_error_dynamic"):
            keys += [f"{key}_pairs", *(f"{key}_difference{end}" for end in ("", "_low", "_high")),
                     *(f"{key}_relative{end}" for end in ("", "_low", "_high")), f"{key}_p"]  # fmt: skip
        frame = pd.DataFrame(rows, columns=[*keys, "outcome"])  # every column, also before the runs exist
        for key in ("runs", "runs_versus", "macro_error_pairs", "macro_error_dynamic_pairs"):
            frame[key] = frame[key].astype("Int64")
        columns = [
            Column("law", "law (period 0)", "text"), Column("versus", "law of D97", "text"),
            Column("runs", "runs", "int"), Column("runs_versus", "runs D97", "int"),
            Column("macro_error_law", "macro error", digits=3, ci=True),
            Column("macro_error_versus", "macro error D97", digits=3, ci=True),
            Column("macro_error_difference", "difference", digits=3, ci=True),
            Column("macro_error_relative", "relative", "pct", 1, ci=True), Column("macro_error_p", "p", "p"),
            Column("macro_error_dynamic_law", "dynamic", digits=3, ci=True),
            Column("macro_error_dynamic_versus", "dynamic D97", digits=3, ci=True),
            Column("macro_error_dynamic_difference", "dynamic difference", digits=3, ci=True),
            Column("collisions_per_1000_vkm_law", "collisions / 1000 veh-km", digits=3),
            Column("collisions_per_1000_vkm_versus", "collisions D97", digits=3),
            Column("outcome", "period 0 only", "text"),
        ]  # fmt: skip
        notes = [
            f"Temporal hold-out on I-80 (D120): the laws fine-tuned on period 0 only (view ngsim_i80_p0: "
            f"{', '.join(c.temporal_pairs)}) against the same laws of D97 fine-tuned on all periods, both on "
            f"{', '.join(c.temporal_scenarios)} (periods the period-0 laws did not see) x seeds "
            f"{', '.join(map(str, c.seeds))}.",
            "macro error and dynamic macro error: means over the runs (unit run) with intervals; difference: period-0 "
            "law - law of D97, paired by scenario and seed, mean with interval; relative: mean ratio - 1 with interval "
            "and the Wilcoxon signed-rank p; period 0 only: worse / better when the interval of the difference "
            "excludes 0.",
            f"Intervals: {100 * c.level:g} % percentile bootstrap, {c.n_resamples} resamples, seed {c.seed}; missing "
            "runs: missing_corridor.txt.",
        ]
        return Table(t, "Temporal hold-out: laws fine-tuned on period 0 against the laws of D97 (D120)", notes, frame,
                     columns)  # fmt: skip

    # --------------------------------------------------- review of M8: clustered inference of H12.3
    def table_correlation_clustered(self) -> Table:
        """H12.3 over the (law, corridor) rows of correlation_pooled with the rows as dependent units: cluster bootstraps
        over the laws and over the architecture families, the permutation of whole laws, leave-one-family-out and the
        learned laws only; per corridor and pooled, for both macro errors."""
        c, t = self.cfg, "correlation_clustered"
        if self.pooled_laws_frame is None:
            self.table_correlation_pooled()
        values = self.pooled_laws_frame if self.pooled_laws_frame is not None else pd.DataFrame(
            columns=["corridor", "law", "unstable_eq", *ERRORS])  # fmt: skip
        values = values[values["law"].isin(c.pooled()) & values["unstable_eq"].notna()].copy()
        for law in dict.fromkeys(values["law"]):
            if c.family_of(law) is None:
                self.note(t, f"{law}: in no family of `families`: a cluster of its own")
        values["family"] = [c.family_of(law) or law for law in values["law"]]
        physics = list(c.physics_families)
        rows = []
        for key, error in ERRORS.items():
            data = values[np.isfinite(values[key].astype(float))]
            scopes = [(corridor, data[data["corridor"] == corridor]) for corridor in self.corridors()]
            if len(self.corridors()) > 1:
                scopes.append(("pooled (corridor as stratum)", data))
            for scope, part in scopes:
                subsets = [("all laws with runs", "", part)]
                subsets += [(f"without {family}", family, part[part["family"] != family])
                            for family in dict.fromkeys(part["family"])]  # fmt: skip
                subsets.append((f"learned laws only (without {', '.join(physics)})", ", ".join(physics),
                                part[~part["family"].isin(physics)]))  # fmt: skip
                for subset, left_out, mine in subsets:
                    r = clustered_spearman(mine["unstable_eq"], mine[key], mine["corridor"], mine["law"],
                                           mine["family"], n_resamples=c.cluster_resamples,
                                           n_permutations=c.n_permutations, level=c.level, seed=c.cluster_seed)  # fmt: skip
                    row = {"scope": scope, "error": error, "subset": subset, "left_out": left_out, **r}
                    row["h12_3_rule"] = verdict_correlation({"estimate": r["estimate"], "p_permutation": r["p_linked"],
                                                             "n": r["n"]}, c)  # fmt: skip
                    for name in ("law", "family", "pairs"):
                        row[f"{name}_interval"] = _bracket(row[f"{name}_low"], row[f"{name}_high"])
                    row["law_list"] = ", ".join(dict.fromkeys(mine["law"]))
                    row["family_list"] = ", ".join(dict.fromkeys(mine["family"]))
                    rows.append(row)
        names = ["scope", "error", "subset", "left_out", "n", "n_laws", "n_families", "n_strata", "estimate", "law_low",
                 "law_high", "law_valid", "family_low", "family_high", "family_valid", "pairs_low", "pairs_high",
                 "pairs_valid", "p_linked", "h12_3_rule", "law_interval", "family_interval", "pairs_interval",
                 "law_list", "family_list"]  # fmt: skip
        frame = pd.DataFrame(rows, columns=names)
        columns = [
            Column("scope", "corridor", "text"), Column("error", "error", "text"), Column("subset", "laws", "text"),
            Column("n", "n", "int"), Column("n_laws", "laws (clusters)", "int"),
            Column("n_families", "families (clusters)", "int"), Column("estimate", "Spearman", digits=3),
            Column("law_interval", "law clusters", "text"), Column("family_interval", "family clusters", "text"),
            Column("pairs_interval", "pairs independent (D122)", "text"),
            Column("p_linked", "p (laws permuted)", "p"), Column("h12_3_rule", "rule of H12.3", "text"),
        ]  # fmt: skip
        present = set(values["law"])
        families = "; ".join(f"{family}: {', '.join(law for law in laws if law in present)}"
                             for family, laws in c.families.items() if present & set(laws))  # fmt: skip
        notes = [
            "Clustered inference of H12.3 (review of M8): the rows of correlation_pooled, one per law and corridor (the "
            "share of unstable equilibria among the equilibria of the law's members against its mean macro error over "
            "its runs; correlation_pooled_laws), are not independent: a law has the same instability on both "
            "corridors, and the variants of one architecture (penalised, certified, residual amplitudes) are related. "
            "Spearman: as correlation_pooled (ranks within every corridor scaled to (0, 1), Pearson of the pooled "
            "ranks; per corridor: the Spearman correlation over its laws).",
            f"law clusters: bootstrap of the laws as clusters (the rows of a law on both corridors move together); "
            f"family clusters: bootstrap of the architecture families as clusters (families: {families or 'none'}). A "
            "cluster drawn k times enters with all its rows k times; ranks within the corridors of every resample; "
            "resamples with a constant variable have no correlation. pairs independent (D122): the rows resampled within "
            "every corridor as in correlation_pooled, with the seed and the resamples of this table, for comparison. "
            "Per corridor the law clusters are the laws themselves.",
            f"Percentile intervals ({100 * c.level:g} %), {c.cluster_resamples} resamples each; one generator "
            f"numpy default_rng({c.cluster_seed}) per row draws the laws (resamples x laws, the clusters numbered in the "
            "order of their first appearance: corridor, then the law order of the design), then from the same stream "
            "the families, then the rows within the corridors.",
            f"p (laws permuted): two-sided permutation test of {c.n_permutations} permutations (own generator "
            f"default_rng({c.cluster_seed})) of the instability values over the laws as wholes: a law takes another "
            "law's value on every corridor it runs on (linked across the corridors), the ranks are recomputed; "
            "(1 + hits) / (1 + permutations). Per corridor it is the permutation of the instability over its laws.",
            f"Subsets: all laws with runs (those of correlation_pooled); without <family>: the rows of one family left "
            f"out, ranks recomputed (leave-one-family-out); learned laws only: without the families "
            f"{', '.join(physics)}. n: rows (law-corridor pairs).",
            f"rule of H12.3 (for information; the verdicts of D108 stay as reported): r > {c.correlation_confirm:g}, "
            f"p (laws permuted) < {c.correlation_alpha:g} and n >= {c.correlation_min_laws}: confirmed; r < "
            f"{c.correlation_refute:g}: refuted; otherwise open.",
        ]
        return Table(t, "H12.3 with clustered inference: laws and architecture families as units (review of M8)", notes,
                     frame, columns)  # fmt: skip

    # ------------------------------------------------------ review of M8: absolute contact exposure
    def table_contacts_absolute(self) -> Table:
        """Per corridor and law the contact episodes and the vehicles in contact per run with the exposure (vehicle-km,
        vehicles of the window) and the demand (inserted share, shortfall), so that the rate per 1000 vehicle-km is read
        with its denominator."""
        c, t = self.cfg, "contacts_absolute"
        rows = []
        for corridor in self.corridors():
            for law in self.laws_on(corridor, c.table_laws()):
                mine = self.mine(corridor, law)
                row: dict[str, Any] = {"corridor": corridor, "law": law, "runs": len(mine),
                                       "runs_expected": len(self.scenarios_for(corridor, law)) * len(c.seeds),
                                       "scenario_set": self.scenario_set(corridor, law)}  # fmt: skip
                episodes, distance = mine["n_collisions"].astype(float), mine["vehicle_km"].astype(float)
                share, window = mine["vehicles_in_contact_share"].astype(float), mine["n_vehicles_window"].astype(float)
                planned, inserted = mine["n_planned_window"].astype(float), mine["inserted_share"].astype(float)
                truth = mine["scenario"].map(lambda s: _float(self.truth(s).get("vehicle_km"))).astype(float)
                self.put(row, "collision_episodes", episodes)
                row["runs_with_contacts"] = int((episodes > 0).sum())
                self.put(row, "vehicles_in_contact_share", share)
                self.put(row, "vehicles_in_contact", share * window)
                self.put(row, "n_vehicles_window", window)
                self.put(row, "vehicle_km", distance)
                self.put(row, "vehicle_km_ratio", distance / truth)
                self.put(row, "collisions_per_1000_vkm", mine["collisions_per_1000_vkm"])
                rate = bootstrap_ci(np.column_stack([episodes.to_numpy(), distance.to_numpy()]), None, _rate_per_1000,
                                    c.n_resamples, c.level, c.seed)  # fmt: skip
                row["rate_pooled"], row["rate_pooled_low"], row["rate_pooled_high"] = rate["estimate"], rate["low"], \
                    rate["high"]  # fmt: skip
                self.put(row, "n_planned_window", planned)
                self.put(row, "inserted_share", inserted)
                self.put(row, "shortfall_window", planned * (1.0 - inserted))
                self.put(row, "n_planned_run", mine["run_n_planned"])
                self.put(row, "shortfall_run", mine["run_n_planned"] - mine["run_n_inserted"])
                self.put(row, "depart_delay_window", mine["mean_depart_delay_s"])
                self.put(row, "depart_delay_run", mine["run_depart_delay_s"])
                rows.append(row)
            for scenario in self.scenarios_of(corridor):
                self.truth(scenario)  # reads the ground truth once (and its exposure)
            truths = [self._truth_exposure.get(s, {}) for s in self.scenarios_of(corridor)]
            if truths:
                row = {"corridor": corridor, "law": "ground truth", "scenario_set": "mean of the scenarios"}
                for key, source in (("n_vehicles_window", "n_vehicles_window"), ("vehicle_km", "vehicle_km"),
                                    ("n_planned_window", "n_planned_window"), ("n_planned_run", "n_vehicles")):  # fmt: skip
                    known = [_float(item.get(source)) for item in truths]
                    known = [x for x in known if math.isfinite(x)]
                    row[key] = float(np.mean(known)) if known else np.nan
                rows.append(row)
        keys = ["corridor", "law", "runs", "runs_expected", "scenario_set", "runs_with_contacts"]
        for key in ("collision_episodes", "vehicles_in_contact_share", "vehicles_in_contact", "n_vehicles_window",
                    "vehicle_km", "vehicle_km_ratio", "collisions_per_1000_vkm", "rate_pooled", "n_planned_window",
                    "inserted_share", "shortfall_window", "n_planned_run", "shortfall_run", "depart_delay_window",
                    "depart_delay_run"):  # fmt: skip
            keys += [key, f"{key}_low", f"{key}_high"]
        frame = pd.DataFrame(rows, columns=keys)
        for key in ("runs", "runs_expected", "runs_with_contacts"):
            frame[key] = frame[key].astype("Int64")
        columns = [
            Column("corridor", "corridor", "text"), Column("law", "law", "text"), Column("runs", "runs", "int"),
            Column("collision_episodes", "contact episodes per run", digits=1, ci=True),
            Column("runs_with_contacts", "runs with contacts", "int"),
            Column("vehicles_in_contact", "vehicles in contact per run", digits=1, ci=True),
            Column("vehicles_in_contact_share", "share of the vehicles", digits=3, ci=True),
            Column("n_vehicles_window", "vehicles in the window", digits=0, ci=True),
            Column("vehicle_km", "veh-km per run", digits=1, ci=True),
            Column("vehicle_km_ratio", "veh-km / ground truth", digits=3, ci=True),
            Column("collisions_per_1000_vkm", "episodes / 1000 veh-km (mean of runs)", digits=1, ci=True),
            Column("rate_pooled", "episodes / 1000 veh-km (pooled)", digits=1, ci=True),
            Column("inserted_share", "inserted share (window)", digits=3, ci=True),
            Column("shortfall_window", "shortfall (window)", digits=1, ci=True),
            Column("shortfall_run", "shortfall (run)", digits=1, ci=True),
            Column("depart_delay_run", "insertion delay (s)", digits=1, ci=True),
        ]  # fmt: skip
        notes = self.header(
            "Absolute contact exposure (review of M8): per corridor and law, unit run (scenario and seed), the mean over "
            "the runs with its interval; the rate per 1000 vehicle-km of the laws table is read here with its "
            "numerator and its denominator.",
            "contact episodes per run: collision_episodes of macro.json, the contact episodes of followers that begin "
            "inside the analysis window (gap <= 0; run_corridor.yaml, sim.contact_gap); runs with contacts: runs with at "
            "least one; vehicles in contact per run: vehicles_in_contact_share x n_vehicles_window, the vehicles of the "
            "window (time or a sample inside it) with an episode that begins inside it; share of the vehicles: "
            "vehicles_in_contact_share.",
            "veh-km per run: vehicle_km of macro.json, the distance driven inside the section and the window (the "
            "denominator of the rate); veh-km / ground truth: per run divided by that of the ground truth of its "
            "scenario. episodes / 1000 veh-km: the mean of the rates of the runs (table laws) and the pooled rate 1000 "
            "sum(episodes) / sum(veh-km) over the runs, with its interval over the runs.",
            "Demand: inserted share (window): of the vehicles planned to depart inside the window (n_planned of "
            "macro.json) the share inserted at all; shortfall (window): n_planned x (1 - inserted share), planned in "
            "the window and never inserted; shortfall (run): n_planned - n_inserted of run.json over the whole run; "
            "insertion delay: mean_depart_delay_s of run.json (all inserted vehicles; the CSV has that of the window "
            "too: depart_delay_window). The CSV also holds n_planned_window and n_planned_run.",
            "ground truth: mean over the scenarios of the corridor of the vehicles of the window, the vehicle-km and the "
            "vehicles planned (window, n_planned_window; whole period, n_planned_run) of the data.",
            *self.ablation_note(),
        )
        return Table(t, "Contact episodes with their exposure and the demand shortfall (review of M8)", notes, frame,
                     columns)  # fmt: skip

    # ------------------------------------------------------- review of M8: H12.2 on the errors
    def table_h12_2_error(self) -> Table:
        """Exploratory: |e| of the candidate minus |e| of the reference for every component of the macro-error vector
        (the macro triple first) and both macro errors, paired by scenario and seed, per corridor."""
        c, t = self.cfg, "h12_2_error"
        index = ["scenario", "seed"]
        order = [*TRIPLE, *(name for name in COMPONENTS if name not in TRIPLE)]
        items = [(f"error_{n}", n, COMPONENT_HEADERS[n], "anchored" if n in ANCHORED else "dynamic") for n in order]
        items += [(key, key, header, "summary") for key, header in ERRORS.items()]
        rows = []
        for corridor in self.corridors():
            reference = self.mine(corridor, c.reference).set_index(index)
            candidate = self.mine(corridor, c.candidate).set_index(index)
            mine = []
            for key, name, header, kind in items:
                pairs = pd.concat({"a": reference[key].astype(float).abs(), "b": candidate[key].astype(float).abs()},
                                  axis=1, join="inner").dropna()  # fmt: skip
                row: dict[str, Any] = {"corridor": corridor, "component": name, "header": header, "kind": kind,
                                       "in_triple": name in TRIPLE, "pairs": len(pairs),
                                       "abs_error_reference": pairs["a"].mean() if len(pairs) else np.nan,
                                       "abs_error_candidate": pairs["b"].mean() if len(pairs) else np.nan}  # fmt: skip
                self.put(row, "difference", (pairs["b"] - pairs["a"]).to_numpy())
                if len(pairs) < 2:
                    self.note(t, f"{corridor}: {name}: {len(pairs)} pairs of {c.reference} and {c.candidate}, no test")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # SciPy warns on degenerate samples; the NaN it returns says it
                        cmp = paired_comparison(pairs["a"], pairs["b"], n_resamples=c.n_resamples, level=c.level,
                                                seed=c.seed)  # fmt: skip
                    relative = pairs["a"].mean() > 0  # the relative change needs a reference error
                    row.update(relative=cmp["relative_change"] if relative else np.nan,
                               relative_low=cmp["ci_low"] if relative else np.nan,
                               relative_high=cmp["ci_high"] if relative else np.nan, wilcoxon_p=cmp["p_value"])  # fmt: skip
                low, high = _float(row.get("difference_low")), _float(row.get("difference_high"))
                row["outcome"] = ("smaller error" if high < 0 else "larger error" if low > 0 else "no difference") \
                    if math.isfinite(low) and math.isfinite(high) else None  # fmt: skip
                mine.append(row)
            components = [row for row in mine if row["kind"] != "summary"]
            for row, p in zip(components, holm([_float(row.get("wilcoxon_p")) for row in components])):
                row["wilcoxon_p_holm"] = p
            rows += mine
        frame = pd.DataFrame(rows, columns=[
            "corridor", "component", "header", "kind", "in_triple", "pairs", "abs_error_reference", "abs_error_candidate",
            "difference", "difference_low", "difference_high", "relative", "relative_low", "relative_high", "wilcoxon_p",
            "wilcoxon_p_holm", "outcome"])  # fmt: skip
        columns = [
            Column("corridor", "corridor", "text"), Column("header", "component", "text"),
            Column("in_triple", "macro triple", "flag"), Column("kind", "kind", "text"), Column("pairs", "pairs", "int"),
            Column("abs_error_reference", f"\\|e\\| {c.reference}", digits=3),  # escaped: a pipe ends a cell
            Column("abs_error_candidate", f"\\|e\\| {c.candidate}", digits=3),
            Column("difference", "difference of \\|e\\|", digits=3, ci=True),
            Column("relative", "relative", "pct", 1, ci=True), Column("wilcoxon_p", "p (Wilcoxon)", "p"),
            Column("wilcoxon_p_holm", "p (Holm)", "p"), Column("outcome", f"{c.candidate}", "text"),
        ]  # fmt: skip
        notes = self.header(
            f"Exploratory comparison on the errors (review of M8), not a verdict: the pre-specified H12.2 used the TOST "
            f"of the raw metrics (table tost, D108: equivalence of {c.candidate} and {c.reference} within "
            f"+-{100 * c.margin:g} % of the mean of {c.reference}) and stays as reported (verdicts).",
            f"Per corridor, {c.candidate} against {c.reference}, paired by scenario and seed (pairs): the absolute "
            "value |e| of every component of the macro-error vector (signed relative errors against the ground truth of "
            "the run's scenario, as in components; the macro triple first: throughput, travel time (W1), wave speed by "
            "cross-correlation) and of the macro error and the dynamic macro error (means of the absolute components; "
            "kind summary).",
            f"difference of |e| = |e_{c.candidate}| - |e_{c.reference}| per pair, mean with its interval over the "
            f"pairs (negative: {c.candidate} closer to the data); relative: mean |e_{c.candidate}| / mean "
            f"|e_{c.reference}| - 1 with its interval; p (Wilcoxon): signed-rank test of the differences (two-sided); "
            "p (Holm): over the eight components of a corridor; last column: smaller / larger error when the interval "
            "of the difference excludes 0, else no difference.",
        )
        return Table(t, f"H12.2 on the errors: {c.candidate} against {c.reference} (exploratory, review of M8)", notes,
                     frame, columns)  # fmt: skip


def _secondary(table: Table) -> str:
    """Markdown of a second table in the same file: its title one level down."""
    text = markdown(table)
    return "#" + text if text.startswith("# ") else text


def _write(table: Table, out: Path) -> None:
    (out / f"{table.name}.md").write_text(markdown(table), encoding="utf-8")
    table.frame.to_csv(out / f"{table.name}.csv", index=False)


def clustered_summary(frame: pd.DataFrame) -> str:
    """The printed line of correlation_clustered: the pooled (or the only corridor's) Spearman of the macro error with
    its intervals, and the estimates without the family with the largest change and over the learned laws only."""
    if not len(frame):
        return "n/a"
    mine = frame[frame["error"] == ERRORS["macro_error"]]
    scope = "pooled (corridor as stratum)" if (mine["scope"] == "pooled (corridor as stratum)").any() else \
        (mine["scope"].iloc[0] if len(mine) else None)  # fmt: skip
    mine = mine[mine["scope"] == scope]
    every = mine[mine["subset"] == "all laws with runs"]
    if not len(every) or pd.isna(every["estimate"].iloc[0]):
        return "n/a"
    a = every.iloc[0]
    text = (f"{scope}: macro error {a['estimate']:.3f} (n {a['n']}), law clusters {a['law_interval'] or 'n/a'}, family "
            f"clusters {a['family_interval'] or 'n/a'}, p (laws permuted) {a['p_linked']:.4f}")  # fmt: skip
    drop = mine[mine["left_out"].astype(str).ne("") & mine["subset"].str.startswith("without")].dropna(subset=["estimate"])
    if len(drop):
        worst = drop.loc[(drop["estimate"] - a["estimate"]).abs().idxmax()]
        text += f"; {worst['subset']} {worst['estimate']:.3f} (n {worst['n']})"
    learned = mine[mine["subset"].str.startswith("learned")].dropna(subset=["estimate"])
    if len(learned):
        text += f"; learned laws only {learned['estimate'].iloc[0]:.3f} (n {learned['n'].iloc[0]})"
    return text


def make_corridor_tables(cfg: CorridorTablesConfig) -> list[str]:
    """Write laws, components, instability (with its correlations), tost, h12_1, h12_2, e4_rmax, sensitivity and
    the verdicts of H12 as Markdown and CSV, and missing.txt; one printed line per table."""
    maker = CorridorTableMaker(cfg)
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def count(*tables: str) -> int:
        return sum(any(line.startswith(f"[{t}]") for t in tables) for line in maker.missing)

    errors, raw, csv = maker.table_laws()
    (out / "laws.md").write_text(markdown(errors) + "\n" + _secondary(raw), encoding="utf-8")
    csv.to_csv(out / "laws.csv", index=False)
    present = errors.frame["runs"].astype(int) if len(errors.frame) else pd.Series(dtype=int)
    lines.append(f"TABLE laws: {len(errors.frame)} laws on {corridor_text(maker.corridors())}, {int(present.sum())} "
                 f"runs, {count('laws', 'runs')} missing -> {out / 'laws.md'}")  # fmt: skip

    components = maker.table_components()
    _write(components, out)
    lines.append(f"TABLE components: {len(components.frame)} laws -> {out / 'components.md'}")

    table, corr = maker.table_instability()
    (out / "instability.md").write_text(markdown(table) + "\n" + _secondary(corr), encoding="utf-8")
    table.frame.to_csv(out / "instability.csv", index=False)
    corr.frame.to_csv(out / "instability_correlation.csv", index=False)
    summary = []
    for corridor in maker.corridors():
        for error in ERRORS.values():  # Spearman on the unstable equilibria over all laws, for both errors
            pick = corr.frame[(corr.frame["corridor"] == corridor) & (corr.frame["error"] == error)
                              & (corr.frame["method"] == "spearman") & (corr.frame["laws"] == "all laws")
                              & (corr.frame["measure"] == MEASURES["unstable_eq"])]  # fmt: skip
            if len(pick) and pd.notna(pick["estimate"].iloc[0]):
                summary.append(f"{corridor} {error} {pick['estimate'].iloc[0]:.3f} (p {pick['p_permutation'].iloc[0]:.4f})")
    summary_text = f"; Spearman with the unstable equilibria: {', '.join(summary)}" if summary else ""
    lines.append(f"TABLE instability: {len(table.frame)} laws, {count('instability')} missing{summary_text} "
                 f"-> {out / 'instability.md'}")  # fmt: skip

    tost = maker.table_tost()
    _write(tost, out)
    equivalent = int(tost.frame["equivalent"].eq(True).sum()) if len(tost.frame) else 0
    pairs = int(tost.frame["pairs"].max() or 0) if len(tost.frame) else 0
    lines.append(f"TABLE tost: {len(tost.frame)} metrics, {pairs} pairs, {equivalent} equivalent within "
                 f"{100 * cfg.margin:g} % -> {out / 'tost.md'}")  # fmt: skip

    h12_1 = maker.table_h12_1()
    _write(h12_1, out)
    lines.append(f"TABLE h12_1: {len(h12_1.frame)} rows, {count('h12_1')} missing -> {out / 'h12_1.md'}")
    h12_2 = maker.table_h12_2()
    _write(h12_2, out)
    lines.append(f"TABLE h12_2: {len(h12_2.frame)} rows, {count('h12_2')} missing -> {out / 'h12_2.md'}")
    maker.h12_3_verdicts()

    rmax = maker.table_e4_rmax()
    _write(rmax, out)
    lines.append(f"TABLE e4_rmax: {len(rmax.frame)} rows, {count('e4_rmax')} missing -> {out / 'e4_rmax.md'}")
    sensitivity = maker.table_sensitivity()
    _write(sensitivity, out)
    lines.append(f"TABLE sensitivity: {len(sensitivity.frame)} rows, {count('sensitivity')} missing "
                 f"-> {out / 'sensitivity.md'}")  # fmt: skip

    verdicts = maker.table_verdicts()
    _write(verdicts, out)
    overall = ", ".join(f"{r['hypothesis']} {r['verdict'] or 'n/a'}" for r in maker.verdicts
                        if r["unit"] in ("overall", cfg.candidate))  # fmt: skip
    lines.append(f"TABLE verdicts: {overall} ({corridor_text(maker.corridors())}) -> {out / 'verdicts.md'}")

    lines += make_m8_corridor_tables(cfg, maker)
    m8_notes = tuple(f"[{name}]" for name in (*M8_TABLES, *M9_TABLES))
    (out / "missing.txt").write_text("".join(f"{line}\n" for line in maker.missing if not line.startswith(m8_notes)),
                                     encoding="utf-8")  # fmt: skip
    return lines


def make_m8_corridor_tables(cfg: CorridorTablesConfig, maker: CorridorTableMaker | None = None) -> list[str]:
    """Write the corridor tables of M8 (asymmetry and asymmetry_contrasts, correlation_pooled, power, temporal) and of
    the review of M8 (correlation_clustered, contacts_absolute, h12_2_error) as Markdown and CSV to ``cfg.m8`` with
    ``missing_corridor.txt`` (the notes of these tables and of the runs); one printed line per table. ``maker``: the
    maker of the M5 tables, whose runs are then read once."""
    maker = maker or CorridorTableMaker(cfg)
    out = cfg.m8
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def count(*tables: str) -> int:
        return sum(any(line.startswith(f"[{t}]") for t in tables) for line in maker.missing)

    asymmetry, contrasts = maker.table_asymmetry()
    _write(asymmetry, out)
    _write(contrasts, out)
    runs = int(asymmetry.frame["runs"].fillna(0).sum()) if "runs" in asymmetry.frame else 0
    changed = int(contrasts.frame["index_changed"].eq(True).sum()) if "index_changed" in contrasts.frame else 0
    lines.append(f"TABLE asymmetry: {runs} runs, {count('asymmetry')} missing -> {out / 'asymmetry.md'}")
    lines.append(f"TABLE asymmetry_contrasts: {len(contrasts.frame)} pairs of laws, the index changed in {changed} -> "
                 f"{out / 'asymmetry_contrasts.md'}")  # fmt: skip

    pooled = maker.table_correlation_pooled()
    values = maker.pooled_laws_frame if maker.pooled_laws_frame is not None else pd.DataFrame(
        columns=["corridor", "law", "unstable_eq", *ERRORS])  # fmt: skip
    laws_table = Table("correlation_pooled_laws", "The laws of the pooled correlation (D122)", [
        "Per corridor and law the values that enter correlation_pooled: the share of unstable equilibria among the "
        "equilibria of its members and its mean macro errors over its runs (unit run).",
    ], values, [Column("corridor", "corridor", "text"), Column("law", "law", "text"),
                Column("unstable_eq", "unstable among equilibria", digits=3),
                Column("macro_error", "macro error", digits=3),
                Column("macro_error_dynamic", "macro error (dynamic)", digits=3)])  # fmt: skip
    (out / "correlation_pooled.md").write_text(markdown(pooled) + "\n" + _secondary(laws_table), encoding="utf-8")
    pooled.frame.to_csv(out / "correlation_pooled.csv", index=False)
    values.to_csv(out / "correlation_pooled_laws.csv", index=False)
    pick = pooled.frame[(pooled.frame["error"] == ERRORS["macro_error"])
                        & (pooled.frame["laws"] == "all laws with runs (D122)")]  # fmt: skip
    summary = ", ".join(f"{r['scope']} {r['estimate']:.3f} (p {r['p_permutation']:.4f}, n {r['n']})"
                        for r in pick.to_dict("records") if pd.notna(r["estimate"]))  # fmt: skip
    lines.append(f"TABLE correlation_pooled: Spearman of the macro error: {summary or 'n/a'} -> "
                 f"{out / 'correlation_pooled.md'}")  # fmt: skip

    power = maker.table_power()
    _write(power, out)
    seeds = ", ".join(f"{r['corridor']} {r['metric']} {r['n_per_scenario']}" for r in power.frame.to_dict("records")
                      if pd.notna(r.get("n_per_scenario")))  # fmt: skip
    lines.append(f"TABLE power: seeds per scenario for 10 %: {seeds or 'n/a'} -> {out / 'power.md'}")

    temporal = maker.table_temporal()
    _write(temporal, out)
    outcomes = ", ".join(f"{r['law']} {r['outcome'] if isinstance(r.get('outcome'), str) else 'n/a'}"
                         for r in temporal.frame.to_dict("records"))  # fmt: skip
    lines.append(f"TABLE temporal: {outcomes}, {count('temporal')} missing -> {out / 'temporal.md'}")

    # the review of M8: clustered inference of H12.3, absolute contact exposure, H12.2 on the errors
    clustered = maker.table_correlation_clustered()
    _write(clustered, out)
    lines.append(f"TABLE correlation_clustered: {clustered_summary(clustered.frame)} -> {out / 'correlation_clustered.md'}")
    contacts = maker.table_contacts_absolute()
    _write(contacts, out)
    laws = contacts.frame[contacts.frame["law"] != "ground truth"] if len(contacts.frame) else contacts.frame
    lines.append(f"TABLE contacts_absolute: {len(laws)} laws, {count('contacts_absolute')} missing -> "
                 f"{out / 'contacts_absolute.md'}")  # fmt: skip
    errors = maker.table_h12_2_error()
    _write(errors, out)
    smaller = ", ".join(f"{r['corridor']} {r['component']}" for r in errors.frame.to_dict("records")
                        if bool(r.get("in_triple")) and r.get("outcome") == "smaller error")  # fmt: skip
    lines.append(f"TABLE h12_2_error: {len(errors.frame)} rows, macro triple with a smaller error: {smaller or 'none'} -> "
                 f"{out / 'h12_2_error.md'}")  # fmt: skip

    notes = tuple(f"[{name}]" for name in (*M8_TABLES, *M9_TABLES, "runs"))
    (out / "missing_corridor.txt").write_text("".join(f"{line}\n" for line in maker.missing if line.startswith(notes)),
                                              encoding="utf-8")  # fmt: skip
    return lines
