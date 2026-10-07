"""Laws that drive the vehicles of the corridor (docs/m5_contract.md, section 3).

Export (build process, :func:`export_laws`): ``<laws_root>/<law>.json`` =
``{law, kind, device, members: [{run, model}], params (kind idm), table (kinds idm_heterogeneous and
residual_heterogeneous), support_v, window}`` with the member runs of ``configs/corridor/laws.yaml``; a law is
written only when every member run exists (a law of fewer members would be another law, and its runs would go
stale when the rest arrive), else its line names the missing runs. Kind ``idm_heterogeneous`` writes per-event
estimates to ``<law>.npz`` (arrays ``T, s0, a``, scalars ``v0, b``): the interior ones (``selection: interior``,
the default, D63) or all of them, bounds-hitting ones included (``selection: all``, D114); ``source`` names them.
``support_v`` = [smallest 1 % speed, largest 99 % speed] of the training data of the members
(``metrics.json["context"]``); ``window`` = largest history window of the members; ``device`` = the device of the
inference fixed per law in the table (``cpu`` for the IDM laws).

M8 adds two sources (docs/m8_contract.md, sections 5 and 7):

* kind ``idm`` with ``calibration`` instead of ``members`` (D120, ``idm_global_p0``): the parameters of global IDM
  calibrations (``params`` of the JSON files that the path pattern names: ``{fold}`` gives one file per fold,
  ``*`` must match one file); no member runs (``members: []``), ``support_v`` from the members of
  ``support_from``. A missing or ambiguous file leaves the law out with a line that says so.
* kind ``residual_heterogeneous`` (D118, ``residual_idm_certified_het``): the ResidualIDM fold members of
  ``members`` give the residual, the certified per-event cores of ``cores`` (the JSON file of the certificate per
  core) give one IDM core per vehicle, in ``<law>.npz`` (arrays ``v0, T, s0, a, b`` and ``index``, the position of
  the core in the file). With ``recheck`` a core enters only when the a priori certificate
  (``cf_stability.stability.certificate``) also holds with the residual bound of every member (the largest one:
  the certificate is monotone in the bound); without it the law is written only when the residuals of the file's
  certificate (``settings.residual_runs``) cover every member (a core is paired only with the residuals it was
  certified with, D118). ``source`` records the counts, the folds and runs of the file's certificate and the
  bounds.

The review of M8 adds two more (the factorial ablation of the certified hybrid):

* kind ``idm`` with ``calibration`` and ``calibration_settings`` (``idm_margin_i80``): of the files of the pattern only
  those whose ``settings`` hold these values (e.g. ``stability_margin: 0.2`` among the per-fold calibrations of a data
  set), one per pattern; none or several: the law is not written and its line says so.
* kind ``idm`` with ``cores_of`` instead of ``members`` (``idm_core_margin``): the IDM parameters held by the
  checkpoints of these runs (the core of a ResidualIDM), one parameter set per run, drawn per vehicle as the law of the
  runs draws its members (``IDMLaw`` and ``ModelLaw`` draw alike from the generator of the seed), the residual switched
  off; ``members: []`` (the audits of the runs are those of the hybrid: the instability of the law is that of its cores
  in closed form), the runs in ``source``, ``support_v`` from the runs.

Simulation (:func:`load_law`): an object with ``n_members``, ``window``, ``assign(n, rng)`` (the member of
every vehicle of the demand, drawn in the order of the planned departures) and
``accelerations(rows, history)`` (unclipped accelerations of the vehicles ``rows`` for their histories
``[n, window, 3]`` of ``(s, dv, v)``). The heterogeneous hybrid draws the member of every vehicle exactly as
``ModelLaw`` (the same members as ``residual_idm_certified`` for a seed), then its core as ``IDMLaw`` draws a row
of its table, and evaluates per member in one batch ``idm(core of each vehicle) + residual of the member``, the
core in the float32 of the member's own core (``ResidualIDM.forward``).

Only numpy is imported at module level: the parent of the multi-run mode computes fingerprints without
pandas or torch, and a simulation process must import libsumo before them (docs/m5_contract.md, section 0).
"""

from __future__ import annotations

import glob
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_stability.utils import REPO_ROOT, config_hash, read_json, resolve_path, write_json

KINDS = ("models", "idm", "idm_heterogeneous", "residual_heterogeneous")
SELECTIONS = ("interior", "all")  # kind idm_heterogeneous: the per-event estimates drawn from (D63, D114)
DEVICES = ("cpu", "cuda")
IDM_KEYS = ("v0", "T", "s0", "a", "b")
IDM_DEFAULTS = {"delta": 4.0, "s_eps": 0.01}  # the IDM of the members and of the per-event calibration
CORE_LISTS = ("cores", "events", "per_event", "per_core", "records", "items")  # where a cores file lists its cores
CERTIFIED_KEYS = ("certified", "certificate_holds", "a_priori_holds", "holds", "certificate", "a_priori")
MARGIN_KEYS = ("margin", "guaranteed_margin_min", "guaranteed_margin", "margin_min", "core_margin_min")
SETTINGS_KEYS = ("settings", "config", "certificate", "meta")


def law_device(payload: Mapping[str, Any], requested: str | None = None) -> str:
    """Device of the inference of a law: ``requested`` when given, else that of the law file (``cpu`` for files
    without one); the IDM laws run on the CPU only."""
    device = str(requested or payload.get("device") or "cpu")
    if device not in DEVICES:
        raise ValueError(f"device {device!r}, expected one of {DEVICES}")
    if payload.get("kind") != "models" and device != "cpu":
        raise ValueError(f"law {payload.get('law')}: the IDM laws are computed on the CPU, not on {device}")
    return device


# ------------------------------------------------------------------------------------------------ export


def member_runs(spec: Mapping[str, Any], folds: Sequence[int]) -> list[str]:
    """Run directories (relative to the repository, as in the table) of the members of a law."""
    return [str(spec["members"]).format(fold=int(k)) for k in folds]


def existing_runs(runs: Sequence[str]) -> list[str]:
    return [run for run in runs if all((resolve_path(run) / f).is_file() for f in ("model.pt", "metrics.json"))]


def support_of(runs: Sequence[str]) -> list[float] | None:
    """[smallest 1 % speed, largest 99 % speed] of the training data of ``runs`` (``box_low/box_high[2]``)."""
    low, high = [], []
    for run in runs:
        context = read_json(resolve_path(run) / "metrics.json").get("context") or {}
        if context.get("box_low") is not None and context.get("box_high") is not None:
            low.append(float(context["box_low"][2]))
            high.append(float(context["box_high"][2]))
    return [min(low), max(high)] if low else None


def export_law(
    name: str, spec: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]], folds: Sequence[int], out_dir: Path
) -> tuple[dict[str, Any] | None, str]:
    """Write the law file of ``name`` (and its table); ``(payload or None, printed line)``. A law with a member
    run that does not exist (``model.pt`` and ``metrics.json``) is not written (a file of an earlier export
    stays)."""
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"law {name}: kind {kind!r}, expected one of {KINDS}")
    device = law_device({"law": name, "kind": kind, "device": spec.get("device")})
    if kind == "idm_heterogeneous":
        return _export_heterogeneous(name, spec, table, folds, out_dir)
    if kind == "idm" and spec.get("calibration"):
        return _export_calibrated(name, spec, table, folds, out_dir)
    if kind == "idm" and spec.get("cores_of"):
        return _export_cores(name, spec, folds, out_dir)
    if kind == "residual_heterogeneous":
        return _export_residual_heterogeneous(name, spec, folds, out_dir)
    runs = member_runs(spec, folds)
    present = existing_runs(runs)
    missing = [run for run in runs if run not in present]
    if missing:
        return None, (
            f"LAW {name:<24} not written: {len(missing)} of {len(runs)} member runs missing (model.pt, metrics.json): "
            f"{', '.join(missing)}"
        )
    from cf_stability.models import IDM, load_model

    models = [load_model(resolve_path(run) / "model.pt") for run in present]
    payload: dict[str, Any] = {
        "law": name,
        "kind": kind,
        "device": device,
        "members": [{"run": run, "model": f"{run}/model.pt"} for run in present],
        "support_v": support_of(present),
        "window": max(int(model.window) for model in models),
    }
    if kind == "idm":
        wrong = [run for run, model in zip(present, models) if not isinstance(model, IDM)]
        defaults = tuple(IDM_DEFAULTS.values())
        other = [
            run for run, model in zip(present, models) if run not in wrong and (model.delta, model.s_eps) != defaults
        ]
        if wrong or other:
            raise ValueError(f"law {name}: members {wrong + other} are not IDMs with {IDM_DEFAULTS}")
        payload["params"] = [model.params_dict() for model in models]
    write_json(out_dir / f"{name}.json", payload)
    return payload, _line(payload, f"{len(present)}/{len(runs)} members", out_dir)


def _export_heterogeneous(
    name: str, spec: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]], folds: Sequence[int], out_dir: Path
) -> tuple[dict[str, Any], str]:
    import pandas as pd

    selection = str(spec.get("selection") or "interior")
    if selection not in SELECTIONS:
        raise ValueError(f"law {name}: selection {selection!r}, expected one of {SELECTIONS}")
    estimates = pd.read_parquet(resolve_path(spec["estimates"]))
    bound_columns = [c for c in estimates.columns if c.startswith("at_bound_")]
    if not bound_columns:
        raise ValueError(f"law {name}: {spec['estimates']} has no at_bound_* columns")
    interior = estimates[~estimates[bound_columns].any(axis=1)]
    used = interior if selection == "interior" else estimates
    fixed = read_json(resolve_path(spec["spread"]))["fixed"]
    for key in ("v0", "b"):
        if not np.allclose(used[key].to_numpy(np.float64), float(fixed[key]), rtol=1e-9, atol=1e-9):
            raise ValueError(f"law {name}: {key} of the estimates is not the fixed value {fixed[key]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: used[key].to_numpy(np.float64) for key in ("T", "s0", "a")}
    np.savez(out_dir / f"{name}.npz", **arrays, v0=np.float64(fixed["v0"]), b=np.float64(fixed["b"]))
    source = spec.get("support_from")
    support_runs = existing_runs(member_runs(table[source], folds)) if source else []
    payload = {
        "law": name,
        "kind": "idm_heterogeneous",
        "device": "cpu",
        "members": [],
        "table": f"{name}.npz",
        "support_v": support_of(support_runs),
        "window": 1,
        "source": {
            "estimates": str(spec["estimates"]), "spread": str(spec["spread"]), "n_estimates": int(len(estimates)),
            "n_interior": int(len(interior)), "support_from": source, "support_runs": support_runs,
        },
    }  # fmt: skip
    if selection != "interior":  # the file of the interior law keeps its content (and its fingerprint)
        payload["source"].update(selection=selection, n_used=int(len(used)))
    write_json(out_dir / f"{name}.json", payload)
    if selection == "interior":
        what = f"{len(interior)} interior of {len(estimates)} per-event estimates"
    else:
        what = f"all {len(used)} per-event estimates ({len(used) - len(interior)} on a bound, {len(interior)} interior)"
    return payload, _line(payload, what, out_dir)


# ------------------------------------------------------------------------------------------ M8 sources


def _relative(path: str | Path) -> str:
    """``path`` relative to the repository (POSIX) when it lies inside it, else as given."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def match_files(pattern: str) -> list[Path]:
    """The files a path pattern names (relative to the repository or absolute; ``*``, ``?`` and ``[...]`` as in
    glob), sorted."""
    resolved = resolve_path(pattern)
    if not glob.has_magic(str(pattern)):
        return [resolved] if resolved.is_file() else []
    return sorted(Path(p) for p in glob.glob(str(resolved)) if Path(p).is_file())


def _one_file(pattern: str) -> tuple[Path | None, str | None]:
    """The single file of ``pattern``, or the reason there is none (missing, ambiguous)."""
    found = match_files(pattern)
    if not found:
        return None, f"{pattern}: no file"
    if len(found) > 1:
        return None, f"{pattern}: {len(found)} files ({', '.join(_relative(f) for f in found)}); name one in the law"
    return found[0], None


def calibration_params(path: str | Path) -> dict[str, float]:
    """The IDM parameters of a calibration file: ``params`` (``scripts/calibrate_idm.py``, the calibration cache of
    the training runs), ``global.params`` or the five keys at the top level."""
    payload = read_json(path)
    candidates = [payload.get("params"), (payload.get("global") or {}).get("params"), payload] \
        if isinstance(payload, Mapping) else []  # fmt: skip
    for candidate in candidates:
        if isinstance(candidate, Mapping) and all(k in candidate for k in IDM_KEYS):
            return {k: float(candidate[k]) for k in IDM_KEYS}
    raise ValueError(f"{_relative(path)} holds no IDM parameters ({', '.join(IDM_KEYS)})")


def _same_setting(value: Any, wanted: Any) -> bool:
    """A setting of a calibration file equals the wanted one: numbers within 1e-9, anything else exactly."""
    numbers = (int, float)
    if isinstance(value, numbers) and isinstance(wanted, numbers) and not isinstance(value, bool) \
            and not isinstance(wanted, bool):  # fmt: skip
        return abs(float(value) - float(wanted)) <= 1e-9
    return value == wanted


def calibration_matches(path: str | Path, settings: Mapping[str, Any]) -> bool:
    """The ``settings`` of the calibration file hold every key of ``settings`` with its value (a key missing in the
    file does not match; an unreadable file does not match)."""
    try:
        payload = read_json(path)
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    held = payload.get("settings") if isinstance(payload, Mapping) else None
    if not isinstance(held, Mapping):
        return False
    return all(key in held and _same_setting(held[key], value) for key, value in settings.items())


def _one_calibration(pattern: str, settings: Mapping[str, Any] | None) -> tuple[Path | None, str | None]:
    """The single calibration file of ``pattern`` whose settings hold ``settings`` (all of them without), or the
    reason there is none."""
    if not settings:
        return _one_file(pattern)
    found = match_files(pattern)
    wanted = ", ".join(f"settings.{key} = {value}" for key, value in settings.items())
    chosen = [path for path in found if calibration_matches(path, settings)]
    if not chosen:
        return None, f"{pattern}: no file with {wanted} (of {len(found)} file{'s' if len(found) != 1 else ''})"
    if len(chosen) > 1:
        return None, (f"{pattern}: {len(chosen)} files with {wanted} ({', '.join(_relative(f) for f in chosen)}); "
                      "name one in the law")  # fmt: skip
    return chosen[0], None


def _export_calibrated(
    name: str, spec: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]], folds: Sequence[int], out_dir: Path
) -> tuple[dict[str, Any] | None, str]:
    """Kind ``idm`` from global calibrations (D120): one member per calibration file of ``calibration``; with
    ``calibration_settings`` only the files whose ``settings`` hold these values (the margin of D72 of a per-fold
    calibration, review of M8)."""
    pattern = str(spec["calibration"])
    settings = dict(spec.get("calibration_settings") or {})
    patterns = [pattern.format(fold=int(k)) for k in folds] if "{fold}" in pattern else [pattern]
    files, problems = [], []
    for item in patterns:
        found, problem = _one_calibration(item, settings)
        if problem:
            problems.append(problem)
        else:
            files.append(found)
    if problems:
        return None, f"LAW {name:<24} not written: calibration missing or ambiguous: {'; '.join(problems)}"
    try:
        params = [calibration_params(path) for path in files]
    except (OSError, ValueError) as exc:
        return None, f"LAW {name:<24} not written: {exc}"
    source = spec.get("support_from")
    support_runs = existing_runs(member_runs(table[source], folds)) if source in table else []
    payload: dict[str, Any] = {
        "law": name, "kind": "idm", "device": "cpu", "members": [], "params": params,
        "support_v": support_of(support_runs), "window": 1,
        "source": {"calibration": [_relative(path) for path in files], "pattern": pattern, "support_from": source,
                   "support_runs": support_runs},
    }  # fmt: skip
    if settings:  # only then: the law files without a selection keep their content (and their fingerprint)
        payload["source"]["calibration_settings"] = settings
    write_json(out_dir / f"{name}.json", payload)
    what = f"{len(files)} calibration{'s' if len(files) > 1 else ''} ({', '.join(_relative(p) for p in files)})"
    if settings:
        what += f" with {', '.join(f'settings.{key} = {value}' for key, value in settings.items())}"
    return payload, _line(payload, what, out_dir)


def _export_cores(
    name: str, spec: Mapping[str, Any], folds: Sequence[int], out_dir: Path
) -> tuple[dict[str, Any] | None, str]:
    """Kind ``idm`` from the IDM cores of member runs (review of M8, ``idm_core_margin``): the IDM parameters held by
    the checkpoints of ``cores_of`` (the core of a ResidualIDM, the IDM itself), one parameter set per member, drawn per
    vehicle as the law of these runs draws its members (``IDMLaw`` and ``ModelLaw`` draw alike); the residual is not
    used. No member runs (``members: []``: the audits of the runs are those of the hybrid, the instability of the law is
    that of its cores in closed form); the runs are in ``source``; ``support_v`` from the runs."""
    runs = member_runs({"members": spec["cores_of"]}, folds)
    present = existing_runs(runs)
    missing = [run for run in runs if run not in present]
    if missing:
        return None, (
            f"LAW {name:<24} not written: {len(missing)} of {len(runs)} runs of the cores missing (model.pt, "
            f"metrics.json): {', '.join(missing)}"
        )
    from cf_stability.models import IDM, load_model

    params, kinds = [], []
    for run in present:
        model = load_model(resolve_path(run) / "model.pt")
        core = model if isinstance(model, IDM) else getattr(model, "idm", None)
        if not isinstance(core, IDM) or (core.delta, core.s_eps) != tuple(IDM_DEFAULTS.values()):
            raise ValueError(f"law {name}: {run} holds no IDM core with {IDM_DEFAULTS}")
        params.append(core.params_dict())
        kinds.append(type(model).__name__)
    payload: dict[str, Any] = {
        "law": name, "kind": "idm", "device": "cpu", "members": [], "params": params, "support_v": support_of(present),
        "window": 1,
        "source": {"cores_of": str(spec["cores_of"]), "core_runs": present, "models": sorted(set(kinds)),
                   "residual": "off"},
    }  # fmt: skip
    write_json(out_dir / f"{name}.json", payload)
    what = f"the IDM cores of {len(present)}/{len(runs)} runs ({', '.join(sorted(set(kinds)))}; residual off)"
    return payload, _line(payload, what, out_dir)


def _holds(record: Mapping[str, Any]) -> bool | None:
    """The certificate flag of a core record (``certified`` or the like, possibly a mapping with ``holds``)."""
    for key in CERTIFIED_KEYS:
        value = record.get(key)
        if isinstance(value, Mapping):
            value = value.get("holds", value.get("certified"))
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.lower() in ("yes", "no", "true", "false"):
            return value.lower() in ("yes", "true")
    return None


def _fold_of(payload: Any) -> int | None:
    """The fold of the residual named in the settings of a cores file (a key ``fold``/``residual_fold`` or a run
    directory ``..._fold<k>_...``), searched breadth first."""
    queue = [payload]
    while queue:
        item = queue.pop(0)
        if isinstance(item, Mapping):
            for key in ("residual_fold", "fold", "certificate_fold"):
                if isinstance(item.get(key), int) and not isinstance(item.get(key), bool):
                    return int(item[key])
            for value in item.values():
                if isinstance(value, str) and (match := re.search(r"fold(\d+)", value)):
                    return int(match[1])
            queue.extend(value for value in item.values() if isinstance(value, Mapping))
    return None


def read_cores(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The cores of the per-core certificate file (D118): ``[{index, params [5], certified, margin}]`` and
    ``{settings, fold, share_certified}``. Accepted layouts: a list of records, or a mapping with the records under
    one of ``CORE_LISTS`` (a list, a mapping of id to record, or columns), or columns at the top level; a record holds
    the parameters at its top level or under ``params``/``parameters``/``theta`` (a mapping or the five values in the
    order v0, T, s0, a, b), the flag under ``certified`` (or ``certificate_holds``, ``a_priori_holds``, ``holds``,
    ``certificate.holds``) and the margin under ``margin`` (or the like)."""
    payload = read_json(path)

    def columns(data: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        if all(isinstance(data.get(k), list) for k in IDM_KEYS):
            n = len(data["T"])
            return [{key: value[i] for key, value in data.items() if isinstance(value, list) and len(value) == n}
                    for i in range(n)]  # fmt: skip
        return None

    records: list[Any] | None = None
    settings: dict[str, Any] = {}
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        settings = next((dict(payload[k]) for k in SETTINGS_KEYS if isinstance(payload.get(k), Mapping)), {})
        for key in CORE_LISTS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
            elif isinstance(value, Mapping):
                records = columns(value) or [{"id": k, **v} for k, v in value.items() if isinstance(v, Mapping)]
            if records is not None:
                break
        if records is None:
            records = columns(payload)
    if records is None:
        raise ValueError(f"no list of cores (a list, or one of the keys {', '.join(CORE_LISTS)}, or the columns "
                         f"{', '.join(IDM_KEYS)})")  # fmt: skip
    cores = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"core {i} is not a mapping")
        params = next((record[k] for k in ("params", "parameters", "theta") if record.get(k) is not None), record)
        if isinstance(params, (list, tuple)) and len(params) == len(IDM_KEYS):
            params = dict(zip(IDM_KEYS, params))
        if not isinstance(params, Mapping) or not all(k in params for k in IDM_KEYS):
            raise ValueError(f"core {i}: no parameters {', '.join(IDM_KEYS)}")
        holds = _holds(record)
        if holds is None:
            raise ValueError(f"core {i}: no certificate flag ({', '.join(CERTIFIED_KEYS)})")
        margin = next((record[k] for k in MARGIN_KEYS if isinstance(record.get(k), (int, float))), None)
        cores.append({"index": i, "params": [float(params[k]) for k in IDM_KEYS], "certified": holds,
                      "margin": None if margin is None else float(margin)})  # fmt: skip
    summary = payload.get("summary") if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping) \
        else {}  # fmt: skip
    share = next((source.get("share_certified") for source in (payload if isinstance(payload, Mapping) else {}, summary,
                                                               settings) if source.get("share_certified") is not None),
                 None)  # fmt: skip
    folds = settings.get("folds")
    if isinstance(folds, list) and all(isinstance(k, int) and not isinstance(k, bool) for k in folds):
        folds = [int(k) for k in folds]  # the folds whose residuals the certificate of every core used (D118)
    else:
        fold = _fold_of(settings) if settings else None
        if fold is None and isinstance(payload, Mapping):
            fold = _fold_of({k: v for k, v in payload.items() if k not in CORE_LISTS})
        folds = None if fold is None else [fold]
    runs = settings.get("residual_runs")  # scripts/certify_cores.py: {fold: run}; a list is taken as well
    runs = list(runs.values()) if isinstance(runs, Mapping) else runs
    info = {"settings": settings, "folds": folds, "fold": folds[0] if folds else None,
            "residual_runs": [str(r) for r in runs] if isinstance(runs, list) else None,
            "share_certified": float(share) if isinstance(share, (int, float)) else None}  # fmt: skip
    return cores, info


def _same_runs(members: Sequence[str], runs: Sequence[str] | None) -> bool | None:
    """Whether every member run is among ``runs`` (paths compared after resolution); None without ``runs``."""
    if runs is None:
        return None
    known = {resolve_path(run).resolve() for run in runs}
    return all(resolve_path(member).resolve() in known for member in members)


def recheck_cores(
    params: np.ndarray, bound: Sequence[float], r_max: float, speeds: Sequence[float] | None = None, n_scan: int = 400
) -> np.ndarray:
    """Whether the a priori certificate of D71/D86 holds at every speed for the IDM core of every row of ``params``
    (``[n, 5]``: v0, T, s0, a, b) with a residual of amplitude ``r_max`` and Jacobian bound ``bound`` (s, dv, v):
    the computation of ``certificate_a_priori`` (``cf_stability.stability.certificate``) with the core in place of
    the model's own one."""
    import torch

    from cf_stability.models.idm import IDM
    from cf_stability.stability.certificate import _a_priori
    from cf_stability.stability.equilibrium import V_GRID

    v = torch.as_tensor(V_GRID if speeds is None else speeds, dtype=torch.float64).reshape(-1)
    bounds = torch.as_tensor(np.asarray(bound, dtype=np.float64)).reshape(1, 3).expand(len(v), 3)
    r = torch.full_like(v, float(r_max))
    out = np.zeros(len(params), dtype=bool)
    for i, row in enumerate(np.asarray(params, dtype=np.float64)):
        core = IDM(dict(zip(IDM_KEYS, (float(x) for x in row))), dtype=torch.float64)
        out[i] = bool(_a_priori(core, v, r, bounds, int(n_scan))["holds"].all())
    return out


def _export_residual_heterogeneous(
    name: str, spec: Mapping[str, Any], folds: Sequence[int], out_dir: Path
) -> tuple[dict[str, Any] | None, str]:
    """Kind ``residual_heterogeneous`` (D118): the fold members of ``members`` for the residual, the certified cores
    of ``cores`` for the IDM core of every vehicle."""
    runs = member_runs(spec, folds)
    present = existing_runs(runs)
    missing = [run for run in runs if run not in present]
    if missing:
        return None, (
            f"LAW {name:<24} not written: {len(missing)} of {len(runs)} member runs missing (model.pt, metrics.json): "
            f"{', '.join(missing)}"
        )
    path, problem = _one_file(str(spec["cores"]))
    if path is None:
        return None, f"LAW {name:<24} not written: certified cores missing or ambiguous: {problem}"
    try:
        cores, info = read_cores(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"LAW {name:<24} not written: {_relative(path)} unreadable: {exc}"
    from cf_stability.models import ResidualIDM, load_model

    models = [load_model(resolve_path(run) / "model.pt") for run in present]
    wrong = [run for run, model in zip(present, models) if not isinstance(model, ResidualIDM)]
    if wrong:
        raise ValueError(f"law {name}: members {wrong} are not ResidualIDM models")
    bounds = np.array([model.jacobian_bound().double().cpu().numpy() for model in models], dtype=np.float64)
    r_max = sorted({float(model.r_max) for model in models})
    certified = [core for core in cores if core["certified"]]
    recheck = bool(spec.get("recheck", True))
    covered = _same_runs(present, info["residual_runs"])  # D118: a core is paired only with the residuals it was
    if covered is False and not recheck:  # certified with; the recheck certifies it with every member's bound
        return None, (
            f"LAW {name:<24} not written: the certificate of {_relative(path)} used the residuals of "
            f"{info['residual_runs']}, not of every member; run scripts/certify_cores.py folds=[0,1,2,3,4] or set "
            "recheck: true"
        )
    used, holding = certified, None
    if recheck and certified:
        ok = recheck_cores(np.array([core["params"] for core in certified]), bounds.max(axis=0), max(r_max))
        used, holding = [core for core, flag in zip(certified, ok) if flag], int(ok.sum())
    counts = f"{len(cores)} cores, {len(certified)} certified in the file"
    if holding is not None:
        counts += f", {holding} hold with the largest residual bound of the {len(present)} members"
    if not used:
        return None, f"LAW {name:<24} not written: no usable core in {_relative(path)} ({counts})"
    arrays = np.array([core["params"] for core in used], dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **{key: arrays[:, j] for j, key in enumerate(IDM_KEYS)},
             index=np.array([core["index"] for core in used], dtype=np.int64))  # fmt: skip
    share = info["share_certified"] if info["share_certified"] is not None else len(certified) / max(len(cores), 1)
    payload: dict[str, Any] = {
        "law": name, "kind": "residual_heterogeneous", "device": "cpu",
        "members": [{"run": run, "model": f"{run}/model.pt"} for run in present],
        "table": f"{name}.npz", "support_v": support_of(present), "window": max(int(m.window) for m in models),
        "source": {
            "cores": _relative(path), "pattern": str(spec["cores"]), "n_cores": len(cores),
            "n_certified": len(certified), "share_certified": share, "n_used": len(used), "recheck": recheck,
            "n_hold_all_members": holding, "certificate_folds": info["folds"],
            "certificate_runs": info["residual_runs"], "members_covered": covered, "r_max": r_max,
            "residual_bounds": bounds.tolist(), "bound_used": bounds.max(axis=0).tolist(),
        },
    }  # fmt: skip
    write_json(out_dir / f"{name}.json", payload)
    folds = "n/a" if info["folds"] is None else ", ".join(map(str, info["folds"]))
    return payload, _line(payload, f"{len(used)} cores of {counts} (certificate of the file: residual of fold(s) "
                                   f"{folds}); {len(present)}/{len(runs)} members for the residual", out_dir)  # fmt: skip


def _line(payload: Mapping[str, Any], what: str, out_dir: Path) -> str:
    support = payload["support_v"]
    support_text = "n/a" if support is None else f"{support[0]:.1f}-{support[1]:.1f} m/s"
    return (
        f"LAW {payload['law']:<24} {payload['kind']}: {what}, window {payload['window']}, support {support_text}, "
        f"{payload['device']} -> {out_dir / (payload['law'] + '.json')}"
    )


def export_laws(laws_cfg: Mapping[str, Any], out_dir: str | Path, only: Sequence[str] | None = None) -> list[str]:
    """Export every law of ``laws_cfg`` (``{folds, table}``; ``only``: a subset); the printed lines."""
    out_dir = Path(out_dir)
    table, folds = laws_cfg["table"], [int(k) for k in laws_cfg["folds"]]
    unknown = sorted(set(only or ()) - set(table))
    if unknown:
        raise ValueError(f"unknown laws {unknown} (table: {sorted(table)})")
    return [export_law(name, table[name], table, folds, out_dir)[1] for name in table if not only or name in only]


# ---------------------------------------------------------------------------------------------- fingerprint


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def law_fingerprint(path: str | Path) -> str:
    """Hash of everything that decides the behaviour of a law: the law file, the bytes of the member
    checkpoints and the arrays of the table. A retrained member or a new export changes it."""
    path = Path(path)
    payload = read_json(path)
    parts: dict[str, Any] = {"law": payload, "models": []}
    for member in payload.get("members") or []:
        parts["models"].append(_file_digest(resolve_path(member["model"])))
    if payload.get("table"):
        with np.load(path.parent / payload["table"]) as data:
            parts["table"] = {
                key: hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest() for key in sorted(data.files)
            }
    return config_hash(parts)


# --------------------------------------------------------------------------------------------- driver laws


class IDMLaw:
    """IDM with a parameter set per vehicle: that of its member (kind ``idm``) or a row of the table drawn for
    it (kind ``idm_heterogeneous``, member 0). Computed with :func:`cf_stability.models.idm.idm_acc` in
    float64 on the CPU."""

    def __init__(self, name: str, *, member_params: np.ndarray | None = None, table: np.ndarray | None = None) -> None:
        if (member_params is None) == (table is None):
            raise ValueError("either member_params [members, 5] or table [rows, 5]")
        import torch

        from cf_stability.models.idm import idm_acc

        self.name, self._torch, self._idm_acc = name, torch, idm_acc
        self.member_params, self.table = member_params, table
        self.n_members = 1 if table is not None else len(member_params)
        self.window = 1
        self.params: np.ndarray | None = None  # [vehicles, 5] after assign

    def assign(self, n_vehicles: int, rng: np.random.Generator) -> np.ndarray:
        if self.table is not None:
            rows = rng.integers(len(self.table), size=n_vehicles)
            self.params = self.table[rows]
            return np.zeros(n_vehicles, dtype=np.int8)
        members = rng.integers(self.n_members, size=n_vehicles)
        self.params = self.member_params[members]
        return members.astype(np.int8)

    def accelerations(self, rows: np.ndarray, history: np.ndarray) -> np.ndarray:
        torch = self._torch
        last = torch.from_numpy(np.ascontiguousarray(history[:, -1, :], dtype=np.float64))
        p = torch.from_numpy(np.ascontiguousarray(self.params[rows], dtype=np.float64))
        acc = self._idm_acc(last[:, 0], last[:, 1], last[:, 2], *p.unbind(-1), **IDM_DEFAULTS)
        return acc.numpy()


class ModelLaw:
    """Checkpoints of the members; the vehicles of one member are evaluated in one batch per step."""

    def __init__(self, name: str, models: Sequence[Any], device: str) -> None:
        import torch

        self.name, self._torch, self.device = name, torch, torch.device(device)
        self.models = [model.to(self.device).eval() for model in models]
        self.n_members = len(self.models)
        self.window = max(int(model.window) for model in self.models)
        self.member: np.ndarray | None = None

    def assign(self, n_vehicles: int, rng: np.random.Generator) -> np.ndarray:
        self.member = rng.integers(self.n_members, size=n_vehicles).astype(np.int8)
        return self.member

    def accelerations(self, rows: np.ndarray, history: np.ndarray) -> np.ndarray:
        torch = self._torch
        out = np.empty(len(rows), dtype=np.float64)
        member = self.member[rows]
        with torch.inference_mode():
            for m, model in enumerate(self.models):
                idx = np.flatnonzero(member == m)
                if not len(idx):
                    continue
                batch = np.ascontiguousarray(history[idx, -int(model.window) :, :], dtype=np.float32)
                x = torch.from_numpy(batch).to(self.device, non_blocking=False)
                out[idx] = model(x).to("cpu", torch.float64).numpy()
        return out


class ResidualHetLaw:
    """Certified hybrid with one IDM core per vehicle (D118): ``a = idm(core of the vehicle) + residual of its
    member`` (``ResidualIDM.residual`` of the last state). The member of every vehicle is drawn as ``ModelLaw`` draws
    it (the same members as the law of the fold members alone for a seed), then its core as ``IDMLaw`` draws a row of
    its table; the vehicles of one member are evaluated in one batch, the core in the dtype of the member's own core
    (``ResidualIDM.forward``), so that a table holding a member's own core reproduces that member exactly."""

    def __init__(self, name: str, models: Sequence[Any], table: np.ndarray, device: str) -> None:
        import torch

        from cf_stability.models.idm import idm_acc

        if table.ndim != 2 or table.shape[1] != len(IDM_KEYS) or not len(table):
            raise ValueError(f"law {name}: the table of cores must be [cores, {len(IDM_KEYS)}], got {table.shape}")
        self.name, self._torch, self._idm_acc, self.device = name, torch, idm_acc, torch.device(device)
        self.models = [model.to(self.device).eval() for model in models]
        self.n_members = len(self.models)
        self.window = max(int(model.window) for model in self.models)
        self.table = np.asarray(table, dtype=np.float64)
        self.member: np.ndarray | None = None
        self.params: np.ndarray | None = None  # [vehicles, 5] after assign

    def assign(self, n_vehicles: int, rng: np.random.Generator) -> np.ndarray:
        self.member = rng.integers(self.n_members, size=n_vehicles).astype(np.int8)
        self.params = self.table[rng.integers(len(self.table), size=n_vehicles)]
        return self.member

    def accelerations(self, rows: np.ndarray, history: np.ndarray) -> np.ndarray:
        torch = self._torch
        out = np.empty(len(rows), dtype=np.float64)
        member = self.member[rows]
        with torch.inference_mode():
            for m, model in enumerate(self.models):
                idx = np.flatnonzero(member == m)
                if not len(idx):
                    continue
                dtype = model.idm.raw.dtype
                x = torch.from_numpy(np.ascontiguousarray(history[idx, -int(model.window) :, :], dtype=np.float32))
                x = x.to(self.device)
                last = x[:, -1, :].to(dtype)
                p = torch.as_tensor(self.params[rows[idx]], dtype=dtype, device=self.device)
                core = self._idm_acc(last[:, 0], last[:, 1], last[:, 2], *p.unbind(-1), model.idm.delta,
                                     model.idm.s_eps)  # fmt: skip
                out[idx] = (core + model.residual(x)).to("cpu", torch.float64).numpy()
        return out


def load_law(path: str | Path, device: str | None = None) -> tuple[IDMLaw | ModelLaw | ResidualHetLaw, str]:
    """The driver law of a law file and the device of its inference (:func:`law_device`)."""
    path = Path(path)
    payload = read_json(path)
    kind, name = payload.get("kind"), payload.get("law", path.stem)
    device = law_device(payload, device)
    if kind == "idm":
        params = np.array([[float(p[key]) for key in IDM_KEYS] for p in payload["params"]], dtype=np.float64)
        return IDMLaw(name, member_params=params), device
    if kind == "idm_heterogeneous":
        with np.load(path.parent / payload["table"]) as data:
            n = len(data["T"])
            columns = {"v0": np.full(n, float(data["v0"])), "b": np.full(n, float(data["b"]))}
            columns.update({key: np.asarray(data[key], dtype=np.float64) for key in ("T", "s0", "a")})
        return IDMLaw(name, table=np.stack([columns[key] for key in IDM_KEYS], axis=1)), device
    if kind not in ("models", "residual_heterogeneous"):
        raise ValueError(f"{path}: kind {kind!r}, expected one of {KINDS}")
    if not payload.get("members"):
        raise ValueError(f"{path}: no members")
    from cf_stability.models import load_model

    models = [load_model(resolve_path(member["model"])) for member in payload["members"]]
    if kind == "residual_heterogeneous":
        with np.load(path.parent / payload["table"]) as data:
            table = np.stack([np.asarray(data[key], dtype=np.float64).reshape(-1) for key in IDM_KEYS], axis=1)
        return ResidualHetLaw(name, models, table, device), device
    return ModelLaw(name, models, device), device
