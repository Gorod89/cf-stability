"""Stability audit of trained models (docs/m3_contract.md, section 5).

    python scripts/audit_stability.py run=runs/m2/follownet_highd/mlp/driver_fold0_seed0
    python scripts/audit_stability.py experiment=m2 data=follownet_highd [force=true]
    python scripts/audit_stability.py experiment=e1 data=follownet_highd band_quantiles=[0.01,0.5,0.99]

Writes ``stability.json`` into every audited run directory (resolved config, hash of the stability
options, git revision, model name and the audit of ``audit_model``, or the error of a failing
model) and prints one line per model: shares of the speeds in support that have a band with a
stable / unstable equilibrium inside the band, an equilibrium outside it, none (numerical flag,
in percent; ``n/a`` without band), equilibria found / without solution / indifferent, largest
gain with its frequency and speed, agreement of the numerical and the analytic-gain flags, wall
time. In experiment mode runs whose output is newer than ``model.pt`` are skipped unless
``force=true``.

Spacing band of other quantiles (D119): ``band_quantiles=[low, median, high]`` recomputes the band of
the near-steady training samples per grid speed with these quantiles (``training_context`` of
``cf_stability/train/tensors.py``, the other band settings of the run's ``train_config``) from the
training events of the run, rebuilt from the config of its ``metrics.json`` (view, split, fold,
``max_train_events``: ``cf_stability/data/views.py``), audits the model with that band in place of the
band of the context, and writes ``stability_q<low>_<high>.json`` (``output`` overrides the name) with
the band under ``band_override``. The anchored search of D79 makes the equilibrium itself depend on the
band, so another band needs a new audit, not a reclassification of the stored one.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.data.views import run_training_events  # noqa: E402
from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.audit import AuditConfig, audit_model, band_output  # noqa: E402
from cf_stability.train.tensors import BandConfig, training_context  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

OUTPUT = "stability.json"


def output_name(cfg: DictConfig) -> str:
    """``output``, else ``stability.json`` or, with ``band_quantiles``, ``stability_q<low>_<high>.json``."""
    if cfg.get("output"):
        return str(cfg.output)
    quantiles = cfg.get("band_quantiles")
    return OUTPUT if quantiles is None else band_output(quantiles)


def is_current(run_dir: Path, name: str = OUTPUT) -> bool:
    out = run_dir / name
    return out.exists() and out.stat().st_mtime > (run_dir / "model.pt").stat().st_mtime


def run_directories(cfg: DictConfig) -> list[Path]:
    """``run``, or the runs of ``experiment`` / ``data`` with a checkpoint (without a newer audit unless ``force``)."""
    if cfg.run is not None:
        return [resolve_path(cfg.run)]
    data = cfg.get("data")
    if cfg.experiment is None or data is None:
        raise ValueError("set run=<run directory> or experiment=<name> data=<event set>")
    root = resolve_path(cfg.paths.runs_root) / cfg.experiment / data.name
    runs = sorted(path.parent for path in root.glob("*/*/model.pt"))
    name = output_name(cfg)
    return [run for run in runs if cfg.force or not is_current(run, name)]


def band_settings(metrics: Mapping[str, Any], quantiles: Sequence[float]) -> BandConfig:
    """The band settings of the run (its effective ``train_config``, else ``config.train``) with other quantiles."""
    band = (metrics.get("train_config") or {}).get("band") or metrics["config"]["train"]["band"]
    return BandConfig.from_mapping({**band, "quantiles": [float(q) for q in quantiles]})


def override_band(
    metrics: Mapping[str, Any], quantiles: Sequence[float], cache: dict[str, Any]
) -> tuple[dict[str, Any] | None, BandConfig]:
    """The spacing band of the run's training events with the quantiles ``quantiles``, as the training computed
    the band of D78 (``training_context``). Runs of the same training events and settings share it
    (``cache``): the seed enters the training events only through ``max_train_events``."""
    config, settings = metrics["config"], band_settings(metrics, quantiles)
    key = config_hash({
        "data": config["data"], "split": config["split"], "fold": config["fold"], "paths": config["paths"],
        "max_train_events": config.get("max_train_events"),
        "seed": config["seed"] if config.get("max_train_events") is not None else None,
        "band": dataclasses.asdict(settings),
    })  # fmt: skip
    if key not in cache:
        cache[key] = training_context(run_training_events(config), int(config["seed"]), settings)["band"]
    return cache[key], settings


BAND_LINE = ("stable", "unstable", "outside", "none")  # shares of band_numerical["support"] in the printed line


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def band_shares(shares: Mapping[str, float] | None) -> str:
    """``stable/unstable/outside/none`` in percent, ``n/a`` without speeds that have a band."""
    return "n/a" if shares is None else "/".join(f"{100.0 * shares[key]:.0f}" for key in BAND_LINE) + " %"


def summary_line(label: str, s: Mapping[str, Any], wall: float) -> str:
    return (
        f"{label:<30} band {band_shares(s['band_numerical']['support'])} (stable/unstable/outside/none, support)  "
        f"eq {s['n_equilibria']}/{s['n_none']}/{s['n_indifferent']} (found/none/indiff)  "
        f"max gain {_fmt(s['max_gain'], '.3f')} at {_fmt(s['max_gain_omega'], '.3f')} rad/s, "
        f"{_fmt(s['max_gain_v'], '.0f')} m/s  agreement {_fmt(s['agreement_gain']['all'], '.0%')}  {wall:.0f} s"
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="audit_stability")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    audit_cfg = AuditConfig.from_mapping(plain["stability"])
    quantiles = plain.get("band_quantiles")
    if quantiles is not None:
        BandConfig(quantiles=tuple(quantiles))  # checked before any run: three increasing numbers in [0, 1]
    hashed = plain["stability"] if quantiles is None else {"stability": plain["stability"], "band_quantiles": quantiles}
    header = {"config": plain, "config_hash": config_hash(hashed), "git_revision": git_revision()}
    name, cache = output_name(cfg), {}
    for run_dir in run_directories(cfg):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        t_start = time.perf_counter()
        payload: dict[str, Any] = {**header, "run": str(run_dir), "model": None}
        try:
            model = load_model(run_dir / "model.pt")
            payload["model"] = model.name
            metrics_path = run_dir / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.exists() else None
            context = None if metrics is None else metrics.get("context")
            if quantiles is not None:
                if metrics is None:
                    raise FileNotFoundError(f"band_quantiles needs the training config of the run: {metrics_path}")
                band, settings = override_band(metrics, quantiles, cache)
                reference = (context or {}).get("band")
                payload["band_override"] = {
                    "quantiles": [float(q) for q in quantiles],
                    "settings": dataclasses.asdict(settings),
                    "band": band,
                    "reference_settings": None if reference is None else reference.get("settings"),
                }
                context = {**(context or {}), "band": band}
            payload["audit"] = audit_model(model, context, audit_cfg)
            line = summary_line(label, payload["audit"]["summary"], time.perf_counter() - t_start)
        except Exception as exc:  # a failing model must not stop the others
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
            line = f"{label:<30} FAILED  {payload['error'].splitlines()[0]}"
        payload["wall_time_s"] = time.perf_counter() - t_start
        write_json(run_dir / name, payload)
        print(line, flush=True)


if __name__ == "__main__":
    main()
