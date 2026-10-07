"""Calibrate the IDM per event and per dataset (docs/data_contract.md, section 6).

    python scripts/calibrate_idm.py dataset=highd
    python scripts/calibrate_idm.py dataset=synthetic max_events=500 calibration.maxiter=50
    python scripts/calibrate_idm.py dataset=ngsim_i80 variant=fixed_v0_b "calibration.fixed={v0: global, b: global}"
    python scripts/calibrate_idm.py dataset=ngsim_i80 per_event_margin=0.2 global_fit=false
    python scripts/calibrate_idm.py dataset=ngsim_i80_p0 per_event=false

Writes ``data/calibration/<dataset>/{idm_per_event.parquet, idm_global.json, idm_spread.json}``
and ``runs/baselines/<dataset>/idm/metrics.json`` (in-sample closed-loop metrics). A variant
other than ``full`` writes ``idm_per_event_<variant>.parquet``, ``idm_spread_<variant>.json`` and
``runs/baselines/<dataset>/idm_<variant>/metrics.json``; the global fit is shared by the variants.

``dataset`` is a directory under ``paths.events_root``, or a view of an event set (D87, D120): a data
config ``configs/data/<dataset>.yaml`` that names ``events`` (and ``filter``) calibrates the events of
that directory that the filter keeps (``cf_stability/data/views.py``), with the outputs under
``<dataset>`` (e.g. ``ngsim_i80_p0``: the period-0 events of ``ngsim_i80``).

Per-event cores with a stability margin (D118): ``per_event_margin=<m>`` adds the term of D72 to the
objective of every event (``calibration.stability_weight`` x the shortfall of the analytic
string-stability margin below ``m`` summed over ``calibration.stability_speeds``, 5..30 m/s) and
writes ``data/calibration/<dataset>/idm_event_margin<m>_<key>.json`` (``..._<variant>_<key>.json`` for a
variant other than ``full``; ``key`` hashes the event ids and the settings) and
``runs/baselines/<dataset>/idm_event_margin<m>/metrics.json`` instead of the parquet and spread files. It
needs ``global_fit=false``: the global fit of the data set is not written again (a ``fixed`` value
``global`` reads the existing ``idm_global.json``).
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.data.schema import Event, EventSet  # noqa: E402
from cf_stability.data.views import keep_event, view_filter  # noqa: E402
from cf_stability.models.idm import IDM_PARAM_NAMES  # noqa: E402
from cf_stability.train.calibration import (  # noqa: E402
    MARGIN_TOLERANCE,
    CalibrationConfig,
    calibrate_global,
    calibrate_per_event,
    evaluate_idm,
    parameter_spread,
    resolve_device,
)
from cf_stability.utils import REPO_ROOT, config_hash, read_json, resolve_path, to_plain, write_json  # noqa: E402

VIEWS = REPO_ROOT / "configs" / "data"  # data configs; one that names `events` is a view of an event set

# what the margin-core file holds (D118); written into its settings
MARGIN_FORMAT = (
    "events: one record per event in the order of the event set: event_id, follower_id, params {v0, T, s0, a, b} "
    "(the IDM core of the event), objective (the minimised value: fit_objective + stability_weight x the summed "
    "shortfall of the margin), fit_objective (J = NRMSE(s) + NRMSE(v) of the closed loop over the event; "
    "collision: 10 + share of samples with s <= 0), margin_min (smallest analytic string-stability margin "
    "f_v^2 + 2 f_v f_dv - 2 f_s of the IDM at its equilibria over stability_speeds; -10 at a speed without "
    "equilibrium), margin_holds (margin_min >= stability_margin - margin_tolerance), rmse_s, rmse_v, nrmse_s, "
    "nrmse_v, collided, converged, n_generations, at_bound (estimated parameters within 1 % of the range from a "
    "bound)"
)


def dataset_events(dataset: str, events_root: str | Path) -> tuple[list[Event], Path, dict[str, Any] | None]:
    """The events of ``dataset``, the directory they come from and the view (None for a plain directory): a
    data config ``configs/data/<dataset>.yaml`` that names ``events`` is a view (D87, D120), whose filter keeps
    the events of that directory; otherwise ``dataset`` is the directory under ``events_root``."""
    path = VIEWS / f"{dataset}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    if not (isinstance(data, dict) and data.get("events")):
        directory = resolve_path(events_root) / dataset
        return EventSet.from_parquet(directory).events, directory, None
    directory = resolve_path(events_root) / data["events"]
    rule = view_filter(data)
    events = EventSet.from_parquet(directory).events
    kept = [ev for ev in events if keep_event(ev, rule)]
    if not kept:
        raise ValueError(f"dataset={dataset}: the view {path} keeps none of the {len(events)} events of {directory}")
    view = {"events": data["events"], "filter": rule, "kept": len(kept), "dropped": len(events) - len(kept)}
    return kept, directory, view


def closed_loop_summary(df: pd.DataFrame) -> dict[str, float]:
    """Mean, median and pooled RMSE of spacing and speed, collision rate."""
    out: dict[str, float] = {"n_events": int(len(df)), "collision_rate": float(df.collided.mean())}
    for name in ("rmse_s", "rmse_v"):
        out[f"{name}_mean"] = float(df[name].mean())
        out[f"{name}_median"] = float(df[name].median())
        out[f"{name}_pooled"] = float(np.sqrt((df[name] ** 2 * df.n_samples).sum() / df.n_samples.sum()))
    return out


def margin_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per event of a per-event fit with a margin (the ``events`` of the margin-core file)."""
    at_bound = [f"at_bound_{name}" for name in IDM_PARAM_NAMES]
    records = []
    for row in df.to_dict("records"):
        records.append({
            "event_id": row["event_id"],
            "follower_id": row["follower_id"],
            "params": {name: float(row[name]) for name in IDM_PARAM_NAMES},
            **{key: float(row[key]) for key in ("objective", "fit_objective", "margin_min")},
            "margin_holds": bool(row["margin_holds"]),
            **{key: float(row[key]) for key in ("rmse_s", "rmse_v", "nrmse_s", "nrmse_v")},
            "collided": bool(row["collided"]),
            "converged": bool(row["converged"]),
            "n_generations": int(row["n_generations"]),
            "at_bound": [name for name, column in zip(IDM_PARAM_NAMES, at_bound) if row[column]],
        })  # fmt: skip
    return records


def margin_summary(df: pd.DataFrame, at_bound: float, wall_time_s: float) -> dict[str, Any]:
    """Shares and quantiles of a per-event fit with a margin."""
    return {
        **closed_loop_summary(df),
        "n_margin_holds": int(df.margin_holds.sum()),
        "share_margin_holds": float(df.margin_holds.mean()),
        "margin_min": {q: float(df.margin_min.quantile(p)) for q, p in (("min", 0.0), ("q05", 0.05), ("median", 0.5))},
        "objective_median": float(df.objective.median()),
        "fit_objective_median": float(df.fit_objective.median()),
        "converged_share": float(df.converged.mean()),
        "at_bound_share": float(at_bound),
        "wall_time_s": wall_time_s,
        "seconds_per_event": wall_time_s / max(len(df), 1),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="calibrate_idm")
def main(cfg: DictConfig) -> None:
    t_start = time.perf_counter()
    plain = to_plain(cfg)
    header = {"config": plain, "config_hash": config_hash(plain), "dataset": cfg.dataset, "variant": cfg.variant}
    margin = cfg.get("per_event_margin")
    if margin is not None and (cfg.global_fit or not cfg.per_event):
        raise ValueError(
            "per_event_margin writes the per-event margin cores only (D118): set global_fit=false (idm_global.json "
            "is not written again) and keep per_event=true"
        )
    if margin is None:
        suffix = "" if cfg.variant == "full" else f"_{cfg.variant}"
    else:
        suffix = f"_event_margin{float(margin):g}" + ("" if cfg.variant == "full" else f"_{cfg.variant}")
    settings = dict(plain["calibration"])
    fixed = dict(settings.pop("fixed", None) or {})
    calib = CalibrationConfig.from_mapping(settings)
    device = resolve_device(calib.device)
    events, events_dir, view = dataset_events(cfg.dataset, cfg.paths.events_root)
    if view is not None:
        header["view"] = view
    if cfg.max_events is not None and cfg.max_events < len(events):
        keep = np.sort(np.random.default_rng(calib.seed).choice(len(events), size=cfg.max_events, replace=False))
        events = [events[i] for i in keep]
    out_dir = resolve_path(cfg.paths.calibration_root) / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {**header, "n_events": len(events), "device": str(device)}
    lines = [
        f"calibrate_idm  dataset={cfg.dataset}  variant={cfg.variant}  events={len(events)}  device={device}  "
        f"hash={config_hash(plain)}" + ("" if margin is None else f"  per-event margin {float(margin):g}")
    ]
    if view is not None:
        total = view["kept"] + view["dropped"]
        lines.append(f"  view     : {view['kept']} of {total} events of {view['events']}, {view['filter']}")

    # the global fit comes first: the per-event fit may keep some parameters at their global values
    if cfg.global_fit:
        t0 = time.perf_counter()
        fit = calibrate_global(events, calib)
        write_json(out_dir / "idm_global.json", {**header, **fit})
        df_global = evaluate_idm(
            events,
            fit["params"],
            collision_penalty=calib.collision_penalty,
            batch_events=calib.batch_events,
            device=device,
        )
        summary = closed_loop_summary(df_global)
        metrics["global"] = {
            **summary,
            "params": fit["params"],
            "objective": fit["objective"],
            "wall_time_s": time.perf_counter() - t0,
        }
        params = " ".join(f"{k}={v:.3g}" for k, v in fit["params"].items())
        lines.append(
            f"  global   : {params}  J {fit['objective']:.3f}  RMSE s {summary['rmse_s_median']:.2f}/"
            f"{summary['rmse_s_mean']:.2f} m  v {summary['rmse_v_median']:.2f}/{summary['rmse_v_mean']:.2f} m/s  "
            f"collisions {summary['collision_rate']:.1%}  ({metrics['global']['wall_time_s']:.1f} s)"
        )

    if cfg.per_event:
        t0 = time.perf_counter()
        if any(value == "global" for value in fixed.values()):
            global_params = read_json(out_dir / "idm_global.json")["params"]
            fixed = {name: global_params[name] if value == "global" else value for name, value in fixed.items()}
        calib = CalibrationConfig.from_mapping({**settings, "fixed": fixed})
        df = calibrate_per_event(events, calib, margin=None if margin is None else float(margin))
        wall = time.perf_counter() - t0
        summary = closed_loop_summary(df)
        at_bound = df[[c for c in df.columns if c.startswith("at_bound_")]].any(axis=1).mean()
        if margin is None:
            df.attrs["config_hash"] = header["config_hash"]
            df.to_parquet(out_dir / f"idm_per_event{suffix}.parquet", index=False)
            spread = {**header, **parameter_spread(df, calib.bounds, calib.fixed)}
            write_json(out_dir / f"idm_spread{suffix}.json", spread)
            written = ""
        else:
            # the settings of the per-event margin fit; the margin of the global fit does not apply here
            core_settings = {
                **{k: v for k, v in dataclasses.asdict(calib).items() if k not in ("device", "stability_margin")},
                "stability_margin": float(margin),
                "margin_tolerance": MARGIN_TOLERANCE,
                "variant": cfg.variant,
                "max_events": cfg.max_events,
            }
            key = config_hash({
                "dataset": cfg.dataset, "event_ids": sorted(ev.event_id for ev in events), "settings": core_settings,
            })  # fmt: skip
            manifest = events_dir / "manifest.json"
            name = f"idm_event_margin{float(margin):g}" + ("" if cfg.variant == "full" else f"_{cfg.variant}")
            path = out_dir / f"{name}_{key}.json"
            write_json(path, {
                **header,
                "kind": "idm_event_margin",
                "key": key,
                "settings": {
                    **core_settings,
                    "events_config_hash": read_json(manifest).get("config_hash") if manifest.exists() else None,
                    "format": MARGIN_FORMAT,
                },
                "n_events": len(events),
                "summary": margin_summary(df, at_bound, wall),
                "events": margin_records(df),
            })  # fmt: skip
            written = f"  {path.name}"
        metrics["per_event"] = {
            **summary,
            "fixed": calib.fixed,
            "objective_median": float(df.objective.median()),
            "converged_share": float(df.converged.mean()),
            "at_bound_share": float(at_bound),
            "wall_time_s": wall,
        }
        held = "  fixed " + " ".join(f"{k}={v:.3g}" for k, v in calib.fixed.items()) if calib.fixed else ""
        if margin is not None:
            metrics["per_event"].update(margin_summary(df, at_bound, wall))
            held += (
                f"  margin {float(margin):g} holds {df.margin_holds.mean():.1%} (fit J median "
                f"{df.fit_objective.median():.3f}, {wall / max(len(events), 1):.3f} s/event)"
            )
        lines.append(
            f"  per-event: J median {df.objective.median():.3f}  RMSE s {summary['rmse_s_median']:.2f}/"
            f"{summary['rmse_s_mean']:.2f} m  v {summary['rmse_v_median']:.2f}/{summary['rmse_v_mean']:.2f} m/s "
            f"(median/mean)  collisions {summary['collision_rate']:.1%}  converged {df.converged.mean():.1%}  "
            f"at a bound {at_bound:.1%}{held}  ({wall:.1f} s){written}"
        )

    metrics["wall_time_s"] = time.perf_counter() - t_start
    run_dir = resolve_path(cfg.paths.runs_root) / "baselines" / cfg.dataset / f"idm{suffix}"
    metrics_path = write_json(run_dir / "metrics.json", metrics)
    lines.append(f"  written  : {out_dir}  {metrics_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
