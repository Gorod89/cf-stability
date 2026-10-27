"""Re-fit a kNN model from the training events of its run, and strip the samples from old checkpoints (D137).

A kNN checkpoint written before D137 held its sub-sample of the training events (``points``, ``targets``):
on highD these are trajectory samples that the licence of the data does not allow to redistribute. Since D137 a
checkpoint holds the scaler, the number of samples and the seed only, and a loaded model is re-fitted from the
training events of its run, which ``metrics.json`` identifies (data config, split, fold, seed, sub-sampling of the
events). The sub-sample is deterministic, so the re-fit reproduces the stored points bit for bit, which
:func:`strip_checkpoint` verifies before it rewrites a checkpoint.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cf_stability.data.schema import Event
from cf_stability.data.views import load_parts, subsample
from cf_stability.models import KNN, load_model, save_model
from cf_stability.utils import resolve_path

DEFAULT_PATHS = {"events_root": "data/events", "splits_root": "data/splits"}


def run_config(run_dir: str | Path) -> dict[str, Any]:
    """The ``config`` of ``metrics.json`` of a run (the composed config of ``scripts/train.py``)."""
    metrics = Path(run_dir) / "metrics.json"
    if not metrics.is_file():
        raise FileNotFoundError(f"{metrics}: no metrics.json, the training events of the run are unknown")
    with metrics.open(encoding="utf-8") as f:
        payload = json.load(f)
    config = payload.get("config")
    if not isinstance(config, Mapping) or "data" not in config:
        raise ValueError(f"{metrics}: metrics.json holds no training config")
    return dict(config)


def training_events(run_dir: str | Path, config: Mapping[str, Any] | None = None) -> list[Event]:
    """The training events of the run, as ``scripts/train.py`` built them: the fold of the split restricted to
    the data view, then the seeded sub-sample of ``max_train_events`` (``None``: all)."""
    config = dict(config or run_config(run_dir))
    paths = {**DEFAULT_PATHS, **dict(config.get("paths") or {})}
    parts, _view = load_parts(dict(config["data"]), config["split"], int(config["fold"]), paths["events_root"],
                              paths["splits_root"])  # fmt: skip
    return subsample(parts["train"], config.get("max_train_events"), int(config["seed"]))


def refit_from_run(model: KNN, run_dir: str | Path) -> dict[str, Any]:
    """Re-fit ``model`` (a loaded kNN without samples) from the training events of ``run_dir``."""
    if not isinstance(model, KNN):
        raise TypeError(f"refit_from_run: a KNN model is expected, not {type(model).__name__}")
    return model.refit(training_events(run_dir))


def events_available(run_dir: str | Path) -> bool:
    """Whether the event set and the split file of the run are on disk (level 3 of the reproduction)."""
    try:
        config = run_config(run_dir)
    except (FileNotFoundError, ValueError):
        return False
    paths = {**DEFAULT_PATHS, **dict(config.get("paths") or {})}
    data = dict(config["data"])
    events_dir = data.get("events") or data["name"]
    splits_prefix = data.get("splits") or events_dir
    return (resolve_path(paths["events_root"]) / events_dir).is_dir() and (
        resolve_path(paths["splits_root"]) / f"{splits_prefix}_{config['split']}.json"
    ).is_file()


def strip_checkpoint(run_dir: str | Path, verify: bool = True) -> dict[str, Any]:
    """Rewrite ``model.pt`` of a kNN run without its samples. With ``verify`` the samples re-fitted from the
    training events must equal the stored ones exactly, else nothing is written."""
    run_dir = Path(run_dir)
    path = run_dir / "model.pt"
    with path.open("rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=True)
    if checkpoint.get("name") != "knn":
        raise ValueError(f"{path}: a {checkpoint.get('name')} checkpoint, not knn")
    state = checkpoint["state_dict"]
    legacy = "points" in state
    model = load_model(path, refit=False)  # legacy: the stored samples; new: no samples
    info: dict[str, Any] = {"run": str(run_dir), "legacy": legacy, "n_points_stored": int(state["points"].shape[0])
                            if legacy else 0, "bytes_before": path.stat().st_size}  # fmt: skip
    if legacy and not int(model.n_samples):  # the fit of the run, recorded by scripts/train.py
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        fit = metrics.get("fit") or {}
        model.n_samples.fill_(int(fit.get("n_samples") or 0))
        model.seed.fill_(int(metrics.get("seed", 0)))
    if verify:  # the re-fit from the training events of the run; for a legacy checkpoint it must reproduce the samples
        stored_points, stored_targets = model.points.clone(), model.targets.clone()
        fresh = copy.deepcopy(model)
        fresh.points, fresh.targets = fresh.points.new_zeros(0, 3), fresh.targets.new_zeros(0)
        refit = refit_from_run(fresh, run_dir)
        same = torch.equal(fresh.points, stored_points) and torch.equal(fresh.targets, stored_targets)
        info.update({"refit": refit, "verified": bool(same) if legacy else True})
        if legacy and not same:
            raise RuntimeError(f"{path}: the re-fit from the training events does not reproduce the stored samples; "
                               "the checkpoint is left as it is")
    save_model(model, path)  # KNN.state_dict() holds no samples
    with path.open("rb") as f:
        written = torch.load(f, map_location="cpu", weights_only=True)["state_dict"]
    assert "points" not in written and "targets" not in written, path
    info["bytes_after"] = path.stat().st_size
    return info


def knn_runs(runs_root: str | Path = "runs") -> list[Path]:
    """Every run directory ``<runs_root>/<experiment>/<data>/knn/<run>`` with a ``model.pt``."""
    root = resolve_path(runs_root)
    return sorted(p.parent for p in root.glob("*/*/knn/*/model.pt") if not p.parent.parts[-4].startswith("_"))


__all__ = ["events_available", "knn_runs", "refit_from_run", "run_config", "strip_checkpoint", "training_events"]
