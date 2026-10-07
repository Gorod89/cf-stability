"""Platoon growth test of trained models (docs/m4_contract.md, section 2.2; D83).

    python scripts/platoon_test.py run=runs/e1/follownet_highd/mlp/driver_fold0_seed0
    python scripts/platoon_test.py experiment=e1 data=follownet_highd [force=true]

Writes ``platoon.json`` into every tested run directory: resolved config, hash of the platoon
options (``stability``), git revision, model name, wall time, the result of ``platoon_test`` per
profile (start, hysteresis loop area of follower 1) and its summary (``platoon_summary``), or the
error of a failing model. The start ``anchored`` (the default) uses the spacing band of the run's
training data, ``metrics.json["context"]["band"]``; a run without it starts at the time gap. Prints
one line per model: mean growth error over all, the human and the ACC profiles, profiles with a
collision, profiles started at the equilibrium, speed std ratio of the last follower over the first
behind the braking pulse, hysteresis loop area of the pulse, wall time. In experiment mode runs
whose ``platoon.json`` is newer than ``model.pt`` are skipped unless ``force=true``.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.equilibrium import Band  # noqa: E402
from cf_stability.stability.platoon import platoon_summary, platoon_test  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

OUTPUT = "platoon.json"


def is_current(run_dir: Path) -> bool:
    out = run_dir / OUTPUT
    return out.exists() and out.stat().st_mtime > (run_dir / "model.pt").stat().st_mtime


def run_directories(cfg: DictConfig) -> list[Path]:
    """``run``, or the runs of ``experiment`` / ``data`` with a checkpoint (without a newer result unless ``force``)."""
    if cfg.run is not None:
        return [resolve_path(cfg.run)]
    data = cfg.get("data")
    if cfg.experiment is None or data is None:
        raise ValueError("set run=<run directory> or experiment=<name> data=<event set>")
    root = resolve_path(cfg.paths.runs_root) / cfg.experiment / data.name
    runs = sorted(path.parent for path in root.glob("*/*/model.pt"))
    return [run for run in runs if cfg.force or not is_current(run)]


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def summary_line(label: str, s: Mapping[str, Any], wall: float) -> str:
    return (
        f"{label:<30} growth error {_fmt(s['growth_error_mean'], '.3f')} "
        f"(human {_fmt(s['growth_error_human'], '.3f')}, ACC {_fmt(s['growth_error_acc'], '.3f')})  "
        f"collisions {s['n_collided']}/{s['n_profiles']}  start at equilibrium {s['n_start_equilibrium']}/"
        f"{s['n_profiles']}  pulse: std ratio {_fmt(s['std_ratio'].get('pulse'), '.2f')}, "
        f"hysteresis {_fmt(s['hysteresis_area_pulse'], '.1f')} m^2/s  {wall:.0f} s"
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="platoon_test")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    header = {"config": plain, "config_hash": config_hash(plain["stability"]), "git_revision": git_revision()}
    for run_dir in run_directories(cfg):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        t_start = time.perf_counter()
        payload: dict[str, Any] = {**header, "run": str(run_dir), "model": None}
        try:
            model = load_model(run_dir / "model.pt")
            payload["model"] = model.name
            metrics = run_dir / "metrics.json"
            band = Band.from_context(read_json(metrics).get("context") if metrics.exists() else None, model)
            payload["band"] = band is not None
            payload["profiles"] = platoon_test(model, plain["stability"], band=band)
            payload["summary"] = platoon_summary(payload["profiles"])
            line = summary_line(label, payload["summary"], time.perf_counter() - t_start)
        except Exception as exc:  # a failing model must not stop the others
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
            line = f"{label:<30} FAILED  {payload['error'].splitlines()[0]}"
        payload["wall_time_s"] = time.perf_counter() - t_start
        write_json(run_dir / OUTPUT, payload)
        print(line, flush=True)


if __name__ == "__main__":
    main()
