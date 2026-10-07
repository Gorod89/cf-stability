"""Transfer of trained models to other event sets (E3, D84; docs/m4_contract.md, section 3.2).

    python scripts/evaluate_transfer.py run=runs/e1/follownet_highd/mlp/driver_fold0_seed0
    python scripts/evaluate_transfer.py experiment=e1 data=follownet_highd [force=true] [targets=[waymo]]

Closed-loop evaluation of the model of a run (warm-up of its training) on every event of every
target set, twice: whole events, and the first ``horizon_s`` of every event (its first
``round(horizon_s / 0.1)`` samples). Events with fewer than ``warm-up + min_scored_s`` samples
after the cut are dropped and counted. Writes into the run directory

* ``transfer_<target>.parquet``: per event ``event_id, follower_id, rmse_s, rmse_v, collided,
  n_scored`` and the same with the suffix ``_h`` for the horizon (missing where dropped);
* ``transfer.json``: ``{run, model, warmup, source: {data, test}, targets: {<target>: {n_events,
  n_dropped: {full, horizon}, full, horizon, relative_degradation}}, config, config_hash,
  git_revision, wall_time_s}``, the summaries of :func:`summarise`, ``source.test`` that of
  ``metrics.json`` and ``relative_degradation = horizon rmse_s_mean / source test rmse_s_mean - 1``.

One line per run. A failing run does not stop the others (``FAILED`` line, error in
``transfer.json``); the exit status is then 1. In experiment mode runs whose ``transfer.json`` is
newer than ``model.pt`` are skipped unless ``force=true``.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.data.schema import ARRAY_FIELDS, DT, Event, EventSet  # noqa: E402
from cf_stability.models import CFModel, load_model  # noqa: E402
from cf_stability.train.evaluate import evaluate_closed_loop, summarise  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

OUTPUT = "transfer.json"
SETTINGS = ("targets", "horizon_s", "min_scored_s", "device", "batch_events")  # keys hashed into config_hash
METRICS = ("rmse_s", "rmse_v", "collided", "n_scored")
DTYPES = {"rmse_s": "float64", "rmse_v": "float64", "collided": "boolean", "n_scored": "Int64"}


def is_current(run_dir: Path) -> bool:
    out = run_dir / OUTPUT
    return out.exists() and out.stat().st_mtime > (run_dir / "model.pt").stat().st_mtime


def run_directories(cfg: DictConfig) -> list[Path]:
    """``run``, or the runs of ``experiment`` / ``data`` with a checkpoint (without a newer output unless ``force``)."""
    if cfg.run is not None:
        if not resolve_path(cfg.run).is_dir():
            raise FileNotFoundError(f"run directory {resolve_path(cfg.run)} does not exist")
        return [resolve_path(cfg.run)]
    data = cfg.get("data")
    if cfg.experiment is None or data is None:
        raise ValueError("set run=<run directory> or experiment=<name> data=<event set>")
    root = resolve_path(cfg.paths.runs_root) / cfg.experiment / data.name
    runs = sorted(path.parent for path in root.glob("*/*/model.pt"))
    return [run for run in runs if cfg.force or not is_current(run)]


def head(event: Event, n: int) -> Event:
    """The first ``n`` samples of ``event``."""
    return dataclasses.replace(event, **{name: getattr(event, name)[:n] for name in ARRAY_FIELDS})


def evaluate(
    model: CFModel, events: Sequence[Event], warmup: int, min_samples: int, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, float] | None, int]:
    """Per-event metrics of the events with at least ``min_samples`` samples, their summary (None when
    no event is left) and the number of events dropped."""
    kept = [ev for ev in events if len(ev) >= min_samples]
    if not kept:
        return pd.DataFrame({name: [] for name in ("event_id", *METRICS)}), None, len(events)
    df = evaluate_closed_loop(model, kept, warmup=warmup, batch_events=int(cfg.batch_events), device=cfg.device)
    return df, summarise(df), len(events) - len(kept)


def transfer_target(
    model: CFModel, events: Sequence[Event], warmup: int, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-event table (whole events and horizon, suffix ``_h``) and summaries on one target set."""
    min_samples = warmup + round(float(cfg.min_scored_s) / DT)
    full, full_summary, full_dropped = evaluate(model, events, warmup, min_samples, cfg)
    cut = [head(ev, round(float(cfg.horizon_s) / DT)) for ev in events]
    horizon, horizon_summary, horizon_dropped = evaluate(model, cut, warmup, min_samples, cfg)
    table = pd.DataFrame({"event_id": [ev.event_id for ev in events], "follower_id": [ev.follower_id for ev in events]})
    for df, suffix in ((full, ""), (horizon, "_h")):
        table = table.join(df.set_index("event_id")[list(METRICS)].add_suffix(suffix), on="event_id")
        for name, dtype in DTYPES.items():
            table[name + suffix] = table[name + suffix].astype(dtype)
    summary = {
        "n_events": len(events),
        "n_dropped": {"full": full_dropped, "horizon": horizon_dropped},
        "full": full_summary,
        "horizon": horizon_summary,
    }
    return table, summary


def relative_degradation(horizon: Mapping[str, float] | None, source_test: Mapping[str, float] | None) -> float | None:
    """``horizon rmse_s_mean / source test rmse_s_mean - 1``; None without either number."""
    if not horizon or not source_test or not source_test.get("rmse_s_mean"):
        return None
    return horizon["rmse_s_mean"] / source_test["rmse_s_mean"] - 1.0


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def summary_line(label: str, payload: Mapping[str, Any], wall: float) -> str:
    items = []
    for target, t in payload["targets"].items():
        full, horizon = ((t[k] or {}).get("rmse_s_mean") for k in ("full", "horizon"))
        degradation = _fmt(t["relative_degradation"], "+.0%")
        items.append(f"{target} {_fmt(full, '.2f')}/{_fmt(horizon, '.2f')} m ({degradation})")
    test = (payload["source"]["test"] or {}).get("rmse_s_mean")
    return (
        f"{label:<30} test {_fmt(test, '.2f')} m  " + "  ".join(items)
        + f"  (RMSE s full/horizon, relative degradation)  {wall:.0f} s"
    )  # fmt: skip


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate_transfer")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    header = {
        "config": plain,
        "config_hash": config_hash({key: plain[key] for key in SETTINGS}),
        "git_revision": git_revision(),
    }
    targets: dict[str, list[Event]] = {}  # loaded once, shared by the runs
    failed = 0
    for run_dir in run_directories(cfg):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        t_start = time.perf_counter()
        payload: dict[str, Any] = {**header, "run": str(run_dir), "model": None}
        try:
            metrics = read_json(run_dir / "metrics.json")
            model = load_model(run_dir / "model.pt")
            warmup = int(metrics["train_config"]["warmup"])
            payload.update(
                model=model.name, warmup=warmup, source={"data": metrics.get("data"), "test": metrics.get("test")},
                targets={},
            )  # fmt: skip
            for target in cfg.targets:
                if target not in targets:
                    targets[target] = EventSet.from_parquet(resolve_path(cfg.paths.events_root) / target).events
                table, summary = transfer_target(model, targets[target], warmup, cfg)
                summary["relative_degradation"] = relative_degradation(summary["horizon"], payload["source"]["test"])
                table.to_parquet(run_dir / f"transfer_{target}.parquet", index=False)
                payload["targets"][target] = summary
            line = summary_line(label, payload, time.perf_counter() - t_start)
        except Exception as exc:  # a failing run must not stop the others
            failed += 1
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
            line = f"{label:<30} FAILED  {payload['error'].splitlines()[0]}"
        payload["wall_time_s"] = time.perf_counter() - t_start
        write_json(run_dir / OUTPUT, payload)
        print(line, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
