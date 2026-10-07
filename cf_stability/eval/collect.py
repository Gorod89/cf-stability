"""One table of the results of many runs (docs/m4_contract.md, section 2.3).

A run is a directory ``<runs_root>/<experiment>/<data>/<model>/<split>_fold<k>_seed<s>/``; its row holds
the training summary (``metrics.json``), the audit (``stability.json``, the ``support`` part of its
shares), the platoon test (``platoon.json``, its summary rebuilt from the per-profile records: a profile
whose platoon collided has no growth error (D107); the growth error on the collision-free prefix of the
platoon (D109)), the transfer (``transfer.json``) and the certificate (``certificate.json``). A missing
or unreadable file gives missing values (and ``<step>_error`` for an unreadable file or one that holds
an error), never an exception. The columns of the first three files always exist, so that tables of
different experiments have the same schema; transfer and certificate add a column per value they hold.
``penalty_weight`` is missing without a penalty.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from cf_stability.utils import read_json

RUN_NAME = re.compile(r"(?P<split>[A-Za-z0-9]+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)")
SUMMARY_KEYS = (
    "n_events", "collision_rate", "open_loop_rmse_a", "rmse_s_mean", "rmse_s_median", "rmse_s_pooled",
    "rmse_v_mean", "rmse_v_median", "rmse_v_pooled",
)  # fmt: skip
BAND_CATEGORIES = ("stable", "unstable", "outside", "none", "indifferent", "undefined")
GRID_CATEGORIES = ("stable", "unstable", "none", "indifferent", "undefined")
AUDIT_COUNTS = ("n_grid", "n_equilibria", "n_multiple", "n_none", "n_indifferent", "n_usable", "n_in_support")
PLATOON_KEYS = (
    "growth_error_mean", "growth_error_human", "growth_error_acc", "n_profiles", "n_collided", "n_start_equilibrium",
    "hysteresis_area_pulse",
)  # fmt: skip
GROWTH_KINDS = ("mean", "human", "acc")  # the profiles of the growth errors: all OpenACC ones, human, ACC
PREFIX_KEYS = ("prefix_growth_error", "prefix_profiles", "first_collided", "collision_free")  # D109, per kind
PREFIX_MIN_POSITIONS = 3  # D109: a collision-free prefix with fewer compared positions has no growth error
ID_COLUMNS = ["experiment", "data", "model", "split", "fold", "seed", "run_dir"]
BASE_COLUMNS = [
    *ID_COLUMNS,
    "config_hash", "penalty_kind", "penalty_weight", "init_from", "init_config_hash", "epochs", "best_epoch",
    "best_feasible", "n_feasible", "stop_reason", "wall_time_s", "n_trainable_parameters",
    *(f"{part}_{key}" for part in ("val", "test") for key in SUMMARY_KEYS),
    *(f"budget_{key}" for key in ("admissible", "safety", "r_max", "lipschitz")),
    *(f"band_{rule}_{c}" for rule in ("numerical", "sign") for c in BAND_CATEGORIES),
    *(f"grid_{rule}_{c}" for rule in ("numerical", "sign") for c in GRID_CATEGORIES),
    "max_gain", "max_gain_omega", "max_gain_v", "agreement_gain", "agreement_sign",
    "share_unstable_numerical", "share_unstable_sign", *AUDIT_COUNTS, "n_band_all", "n_band_support",
    *(f"platoon_{key}" for key in PLATOON_KEYS), "platoon_std_ratio_pulse",
    *(f"platoon_{key}_{kind}" for key in ("growth_profiles", "growth_collided") for kind in GROWTH_KINDS),
    *(f"platoon_{key}_{kind}" for key in PREFIX_KEYS for kind in GROWTH_KINDS),
    "metrics_error", "audit_error", "platoon_error", "transfer_error", "certificate_error",
]  # fmt: skip
FILES = {  # step -> file of the run directory
    "metrics": "metrics.json", "audit": "stability.json", "platoon": "platoon.json", "transfer": "transfer.json",
    "certificate": "certificate.json",
}  # fmt: skip


def run_directories(runs_root: str | Path, experiments: Iterable[str]) -> list[tuple[str, Path]]:
    """``(experiment, run directory)`` of every run of ``experiments`` (names or glob patterns of
    directories under ``runs_root``), sorted; directories starting with ``_`` (logs, tables) are skipped."""
    root, found, seen = Path(runs_root), [], set()
    for pattern in experiments:
        for experiment in sorted(root.glob(pattern)):
            if not experiment.is_dir() or experiment.name.startswith("_"):
                continue
            for run in sorted(experiment.glob("*/*/*")):
                names = (run.parent.parent.name, run.parent.name)
                if run.is_dir() and RUN_NAME.fullmatch(run.name) and not any(n.startswith("_") for n in names):
                    if run not in seen:
                        seen.add(run)
                        found.append((experiment.name, run))
    return found


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Scalar leaves of nested mappings as ``prefix_key_subkey``; lists are left out."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}", item, out)
    elif value is None or isinstance(value, (bool, int, float, str)):
        out[prefix] = value


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(key)
    return mapping


def _profile_name(profile: str) -> str:
    """Column name of a platoon profile: ``ZalaZone/handling_part30.csv`` -> ``ZalaZone_handling_part30``."""
    return re.sub(r"\.csv$", "", profile).replace("/", "_").replace("\\", "_")


def _load(run: Path, step: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """Content of the file of ``step``; None (with ``<step>_error``) when it is missing, unreadable or an error."""
    path = run / FILES[step]
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        row[f"{step}_error"] = f"unreadable: {type(exc).__name__}"
        return None
    if not isinstance(payload, dict):
        row[f"{step}_error"] = "unreadable: not a mapping"
        return None
    if payload.get("error"):
        row[f"{step}_error"] = str(payload["error"]).splitlines()[0]
        return None
    return payload


def _metrics(m: Mapping[str, Any], row: dict[str, Any]) -> None:
    penalty = _get(m, "train_config", "penalty") or _get(m, "config", "train", "penalty") or {}
    training = m.get("training") or {}
    active = penalty.get("kind") not in (None, "none")  # the weight of an inactive penalty is a default, no setting
    row.update(
        config_hash=m.get("config_hash"),
        penalty_kind=penalty.get("kind"),
        penalty_weight=penalty.get("weight") if active else None,
        init_from=m.get("init_from", _get(m, "config", "init_from")),
        init_config_hash=m.get("init_config_hash"),
        epochs=training.get("epochs"),
        best_epoch=m.get("best_epoch"),
        best_feasible=training.get("best_feasible"),  # D89: the best epoch meets the penalty (None without one)
        n_feasible=training.get("n_feasible"),
        stop_reason=training.get("stop_reason"),
        wall_time_s=m.get("wall_time_s"),
        n_trainable_parameters=m.get("n_trainable_parameters"),
    )
    for part in ("val", "test"):
        _flatten(part, m.get(part) or {}, row)
    _flatten("budget", m.get("certificate_budget") or {}, row)


def _audit(stability: Mapping[str, Any], row: dict[str, Any]) -> None:
    s = _get(stability, "audit", "summary") or {}
    for group, categories in (("band", BAND_CATEGORIES), ("grid", GRID_CATEGORIES)):
        for rule in ("numerical", "sign"):
            shares = _get(s, f"{group}_{rule}", "support") or {}
            for c in categories:
                row[f"{group}_{rule}_{c}"] = shares.get(c)
    for key in ("max_gain", "max_gain_omega", "max_gain_v", *AUDIT_COUNTS):
        row[key] = s.get(key)
    # shares among the equilibria the audit analysed (inside the band or outside it), speeds in support
    for key in ("agreement_gain", "agreement_sign", "share_unstable_numerical", "share_unstable_sign"):
        row[key] = _get(s, key, "support")
    for part in ("all", "support"):
        row[f"n_band_{part}"] = _get(s, "n_band", part)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first_collided_position(profile: Mapping[str, Any]) -> int | None:
    """First collided position of a simulated platoon (D109): the lowest follower index ``k >= 1`` (the
    leader is vehicle 0) whose gap reached 0 at any time of the run (``min_gap[k] <= 0``, the collision rule
    of the platoon test). Without collision the number of vehicles (51 for the leader and 50 followers: one
    past the last vehicle). Vehicles ahead of it never touched their leaders, and their leaders never did:
    their speeds are car following throughout. None when it cannot be known: a collided platoon whose
    record has no ``min_gap`` (``collision_vehicle`` is the first collision in time, not this position)."""
    n = len(profile.get("speed_std") or [])
    gaps = profile.get("min_gap")
    if isinstance(gaps, list) and n > 1 and len(gaps) == n:
        hit = next((k for k in range(1, n) if _finite(gaps[k]) and gaps[k] <= 0.0), None)
        return n if hit is None else hit
    if not profile.get("collided") and n:
        return n
    return None


def prefix_growth_error(
    profile: Mapping[str, Any], min_positions: int = PREFIX_MIN_POSITIONS
) -> tuple[float | None, int]:
    """Growth error of one OpenACC profile on the collision-free prefix of the platoon (D109) and the number
    of positions it compares.

    The empirical curve has the positions ``0..P`` (the leader 0 and the ``P`` followers of the OpenACC
    platoon); with the first collided position ``c`` (:func:`first_collided_position`) the followers
    ``1..min(P, c - 1)`` whose empirical value exists are compared: RMSE between the model and the empirical
    curve over them divided by the maximum of the empirical curve over them (the normalisation of
    ``cf_stability.stability.platoon.growth_error``). The leader is left out: model and data share its
    speed. None with fewer than ``min_positions`` positions, without curves, or with ``c`` unknown."""
    empirical, model = profile.get("empirical_std"), profile.get("speed_std")
    first = first_collided_position(profile)
    if not isinstance(empirical, list) or not isinstance(model, list) or first is None:
        return None, 0
    last = min(len(empirical) - 1, first - 1, len(model) - 1)
    positions = [k for k in range(1, last + 1) if _finite(empirical[k])]
    if len(positions) < min_positions or not all(_finite(model[k]) for k in positions):
        return None, len(positions)
    data = np.array([empirical[k] for k in positions], dtype=np.float64)
    simulated = np.array([model[k] for k in positions], dtype=np.float64)
    scale = data.max()
    if not scale > 0.0:
        return None, len(positions)
    return float(np.sqrt(np.mean((simulated - data) ** 2)) / scale), len(positions)


def _prefix(profiles: Mapping[str, Mapping[str, Any]], row: dict[str, Any]) -> None:
    """Columns of D109 from the per-profile records: per OpenACC profile (one with an empirical curve or a
    growth error) the prefix growth error, its positions and the first collided position; per kind (all
    OpenACC profiles, human ones with ACC flag 0, the others) the mean prefix growth error over the profiles
    that have one and their number, the mean first collided position and the collision-free share."""
    openacc = {name: p for name, p in profiles.items() if "empirical_std" in p or "growth_error" in p}
    errors, firsts = {}, {}
    for name, profile in openacc.items():
        errors[name], positions = prefix_growth_error(profile)
        firsts[name] = first_collided_position(profile)
        label = _profile_name(name)
        row[f"platoon_prefix_growth_error_{label}"] = errors[name]
        row[f"platoon_prefix_positions_{label}"] = positions
        row[f"platoon_first_collided_{label}"] = firsts[name]
    kinds = {"mean": list(openacc), "human": [n for n in openacc if openacc[n].get("acc_flag") == 0],
             "acc": [n for n in openacc if openacc[n].get("acc_flag") != 0]}  # fmt: skip
    for kind, names in kinds.items():
        values = [errors[n] for n in names if errors[n] is not None]
        known = [firsts[n] for n in names]
        row[f"platoon_prefix_growth_error_{kind}"] = float(np.mean(values)) if values else None
        row[f"platoon_prefix_profiles_{kind}"] = len(values)
        row[f"platoon_first_collided_{kind}"] = float(np.mean(known)) if known and None not in known else None
        free = [not bool(openacc[n].get("collided")) for n in names]
        row[f"platoon_collision_free_{kind}"] = float(np.mean(free)) if free else None


def _platoon(platoon: Mapping[str, Any], row: dict[str, Any]) -> None:
    """Columns of the platoon test, rebuilt from the per-profile records (``profiles``) with
    ``platoon_summary``, so that files written before its rule (a profile whose platoon collided has no
    growth error) give the numbers of that rule, and the prefix columns of D109 (:func:`_prefix`). Without
    records the stored ``summary``; its growth errors then stand for the prefix ones (a file without records
    has no curves; no run of the project has such a file) and the collision positions are unknown."""
    profiles = platoon.get("profiles")
    s = platoon.get("summary") or {}
    if isinstance(profiles, Mapping) and profiles:
        from cf_stability.stability.platoon import platoon_summary  # torch: only once a platoon test is read

        try:
            s = platoon_summary(profiles)
            _prefix(profiles, row)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            row["platoon_error"] = f"unreadable profiles: {type(exc).__name__}"
            return
    else:
        for kind in GROWTH_KINDS:
            row[f"platoon_prefix_growth_error_{kind}"] = s.get(f"growth_error_{kind}")
            for key in PREFIX_KEYS[1:]:
                row[f"platoon_{key}_{kind}"] = None
    for key in PLATOON_KEYS:
        row[f"platoon_{key}"] = s.get(key)
    for key in ("growth_profiles", "growth_collided"):
        for kind in GROWTH_KINDS:
            row[f"platoon_{key}_{kind}"] = (s.get(key) or {}).get(kind)
    for key in ("growth_error", "std_ratio"):
        for profile, value in (s.get(key) or {}).items():
            row[f"platoon_{key}_{_profile_name(profile)}"] = value


def collect_run(run: str | Path, experiment: str | None = None) -> dict[str, Any]:
    """Row of one run directory (see the module docstring)."""
    run = Path(run)
    match = RUN_NAME.fullmatch(run.name)
    row: dict[str, Any] = {
        "experiment": experiment if experiment is not None else run.parents[2].name,
        "data": run.parent.parent.name,
        "model": run.parent.name,
        "split": match["split"] if match else None,
        "fold": int(match["fold"]) if match else None,
        "seed": int(match["seed"]) if match else None,
        "run_dir": str(run),
    }
    if (metrics := _load(run, "metrics", row)) is not None:
        _metrics(metrics, row)
    if (stability := _load(run, "audit", row)) is not None:
        _audit(stability, row)
    if (platoon := _load(run, "platoon", row)) is not None:
        _platoon(platoon, row)
    if (transfer := _load(run, "transfer", row)) is not None:
        for target, values in (transfer.get("targets") or {}).items():
            _flatten(f"transfer_{target}", values, row)
    if (certificate := _load(run, "certificate", row)) is not None:
        row["certificate_applicable"] = certificate.get("applicable", True)
        for section in ("core", "residual", "a_priori", "at_equilibria"):
            _flatten(f"certificate_{section}", certificate.get(section) or {}, row)
    return row


def collect_runs(runs_root: str | Path, experiments: Iterable[str]) -> pd.DataFrame:
    """One row per run of ``experiments`` under ``runs_root`` (names or glob patterns such as ``e2_*``),
    sorted by experiment, data, model, split, fold and seed; the base columns first, then the others by name."""
    rows = [collect_run(run, experiment) for experiment, run in run_directories(runs_root, experiments)]
    extra = sorted({key for row in rows for key in row} - set(BASE_COLUMNS))
    table = pd.DataFrame(rows, columns=[*BASE_COLUMNS, *extra])
    if len(table):
        table = table.sort_values(ID_COLUMNS[:6], kind="stable").reset_index(drop=True)
    return table
