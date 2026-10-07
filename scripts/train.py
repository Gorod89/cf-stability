"""Fit / train one model on one fold and evaluate it in closed loop (docs/m2_contract.md, sections 3-5;
docs/m4_contract.md, section 3.1).

    python scripts/train.py data=follownet_highd model=mlp fold=0 seed=0
    python scripts/train.py -m model=mlp,gru seed=0,1,2,3,4
    python scripts/train.py model=gru max_train_events=500 train.max_epochs=2 experiment=smoke
    python scripts/train.py model=mlp train.penalty.kind=jacobian train.penalty.weight=0.1
    python scripts/train.py model=mlp train.penalty.kind=monotone experiment=e2_monotone_w1   # D117
    python scripts/train.py data=ngsim_i80_p0 model=mlp experiment=m8_temporal_ft \
        init_from=runs/e1/follownet_highd/mlp/driver_fold0_seed0                               # D120
    python scripts/train.py model=gru train.penalty.kind=combined train.penalty.weight=0.1 train.penalty.every=8 \
        train.penalty.jacobian_weight=1.0 train.penalty.guard=3.0 experiment=e2_combined_j1
    python scripts/train.py data=openacc_acc model=gru experiment=e5
    python scripts/train.py model=residual_idm calibration.stability_margin=0.2 certificate.enforce=true \
        experiment=e4_stable
    python scripts/train.py data=ngsim_i80 model=residual_idm experiment=e4_stable_ft \
        init_from=runs/e4_stable/follownet_highd/residual_idm/driver_fold0_seed0

Writes ``<runs_root>/<experiment>/<data>/<model>/<split>_fold<k>_seed<s>/`` with ``metrics.json``,
``model.pt`` (best weights) and ``test_events.parquet`` (one row per test event). The global
calibrations of the training part (IDM for ``idm``, ``pidl``, ``residual_idm``; Newell's wave speed
for ``newell``, ``perl``; OVM for ``ovm``) are cached in ``<calibration_root>/<data>/``, so that all
seeds of a fold share them. The ``calibration`` section holds options of the IDM calibration
(``CalibrationConfig``): ``stability_margin`` gives the stable core of D72; other keys can be added
with ``+calibration.<key>=...`` (short fits in tests). The cache key holds these settings.

A model config may carry ``train_overrides: {<train key>: value, ...}``: these values replace
those of ``train`` (nested keys are merged) before the training config is built, e.g.
``acc_weight: 0.0`` of ``residual_idm`` (D63). An override always wins: hydra cannot tell an
explicit ``train.acc_weight=1.0`` on the command line from the default, so a value set by the
model config is changed with ``model.train_overrides.acc_weight=1.0`` (or all of them removed with
``~model.train_overrides``); a new one is added with ``+model.train_overrides.<key>=...``. The
effective training config is written to ``metrics.json`` as ``train_config``.

Views of an event set (D87, D120; ``cf_stability/data/views.py``): a data config with ``events``
(directory under ``events_root``, default ``name``), ``splits`` (prefix of the split files, default the
events directory) and ``filter: {mode, exclude_files, follower_prefix}`` trains on the events of that set
whose follower drives in ``mode`` (``follower_id`` ends with ``:<mode>``), whose ``follower_id`` starts with
``follower_prefix`` and whose run file is not excluded (``meta["file"]``, or ``<campaign folder>/<file>`` for
an entry with a folder), in all three parts of the fold. The run directory and the calibration cache use
``data.name``.

Fine-tuning (D86): ``init_from=<run directory>`` (relative paths from the repository root) starts
from ``<run>/model.pt`` as it is (scaler, fitted parts, IDM core, ``lipschitz``): no calibration,
no ``fit``; the ``model`` group must name the model type of the checkpoint. Certified budget
(D86): ``certificate.enforce=true`` (``residual_idm`` with a fixed core, not with ``init_from``) sets
``lipschitz = certificate.safety * b / r_max`` after the calibration of the core, with ``b`` the
admissible budget of the a priori certificate at the model's ``r_max`` (``b = 0``: error).
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.data.schema import DT, Event  # noqa: E402
from cf_stability.data.views import PARTS, load_parts, subsample  # noqa: E402
from cf_stability.models import IDM, OVM, PERL, CFModel, Newell, build_model, load_model, save_model  # noqa: E402
from cf_stability.models.base import ModelContext  # noqa: E402
from cf_stability.models.newell import NEWELL_PARAM_NAMES, calibrate_newell  # noqa: E402
from cf_stability.models.ovm import calibrate_ovm  # noqa: E402
from cf_stability.stability.certificate import enforce_budget  # noqa: E402
from cf_stability.train.calibration import CalibrationConfig, calibrate_global, resolve_device  # noqa: E402
from cf_stability.train.evaluate import evaluate_closed_loop, open_loop_rmse, summarise  # noqa: E402
from cf_stability.train.tensors import training_context  # noqa: E402
from cf_stability.train.trainer import TrainConfig, seed_everything, train_model  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

NEEDS_IDM = ("idm", "pidl", "residual_idm")
NEEDS_NEWELL = ("newell", "perl")
NEEDS_OVM = ("ovm",)
CALIBRATIONS = {"idm": calibrate_global, "newell": calibrate_newell, "ovm": calibrate_ovm}


def calibration_config(law: str, cfg: DictConfig, device: torch.device) -> CalibrationConfig:
    """Settings of the global calibration of ``law``: the defaults, and for the IDM the ``calibration``
    section of the config (without other keys than ``stability_margin: null`` the defaults, so the
    cache files of the free fits stay valid)."""
    options = dict(to_plain(cfg.calibration) or {}) if law == "idm" else {}
    return CalibrationConfig.from_mapping({**options, "device": str(device)})


def fold_calibration(law: str, events: Sequence[Event], cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """Global calibration of ``law`` ("idm", "newell" or "ovm") on the training events, cached under a
    key that hashes the law, the event ids and the settings: all seeds of a fold share it. Newell's
    rollouts observe the window of the model."""
    calib = calibration_config(law, cfg, device)
    extra = {"window": int(cfg.model.window)} if law == "newell" else {}
    settings = {**{k: v for k, v in dataclasses.asdict(calib).items() if k != "device"}, **extra}
    key = config_hash({"law": law, "event_ids": sorted(ev.event_id for ev in events), "calibration": settings})
    name = f"{law}_global_{cfg.split}_fold{cfg.fold}_{key}.json"
    path = resolve_path(cfg.paths.calibration_root) / cfg.data.name / name
    if not path.exists():
        fit = CALIBRATIONS[law](events, calib, **extra)
        manifest = resolve_path(cfg.paths.events_root) / (cfg.data.get("events") or cfg.data.name) / "manifest.json"
        origin = {
            "data": cfg.data.name,
            "split": cfg.split,
            "fold": cfg.fold,
            "n_events": len(events),
            "events_config_hash": read_json(manifest).get("config_hash") if manifest.exists() else None,
        }
        write_json(path, {**origin, "settings": settings, **fit})
    return read_json(path)


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with the values of ``overrides``; nested mappings are merged key by key."""
    out = dict(base)
    for key, value in overrides.items():
        both_mappings = isinstance(value, Mapping) and isinstance(out.get(key), Mapping)
        out[key] = merge_overrides(out[key], value) if both_mappings else value
    return out


def check_options(cfg: DictConfig, model_cfg: Mapping[str, Any]) -> None:
    """Combinations of options that cannot work, before any data are read."""
    if not cfg.certificate.enforce:
        return
    if model_cfg["name"] != "residual_idm":
        raise ValueError(f"certificate.enforce=true needs model=residual_idm, got a {model_cfg['name']!r} model")
    if cfg.init_from is not None:
        raise ValueError("certificate.enforce=true cannot be combined with init_from: the budget is set at the start")
    if model_cfg.get("idm_learnable"):  # the budget holds for the calibrated core, not for one changed by training
        raise ValueError("certificate.enforce=true needs a fixed IDM core (model.idm_learnable=false)")


def carried_params(model: CFModel) -> dict[str, dict[str, float]]:
    """Physics parameters a model holds, in the slots of ``ModelContext``: the IDM core (``idm``, ``pidl``,
    ``residual_idm``), Newell's wave speed (``newell``, ``perl``), the OVM."""
    out: dict[str, dict[str, float]] = {"idm_params": {}, "newell_params": {}, "ovm_params": {}}
    core = model if isinstance(model, IDM) else getattr(model, "idm", None)
    if isinstance(core, IDM):
        out["idm_params"] = core.params_dict()
    elif isinstance(model, Newell):
        out["newell_params"] = model.params_dict()
    elif isinstance(model, PERL):
        out["newell_params"] = dict(zip(NEWELL_PARAM_NAMES, model.newell_theta.double().cpu().tolist()))
    elif isinstance(model, OVM):
        out["ovm_params"] = model.params_dict()
    return out


def load_source(init_from: str, model_name: str) -> tuple[CFModel, dict[str, Any]]:
    """Model and ``metrics.json`` of the run ``init_from``; its model type must be ``model_name``."""
    run = resolve_path(init_from)
    for name in ("model.pt", "metrics.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"init_from={init_from}: {run / name} does not exist")
    model = load_model(run / "model.pt")
    if model.name != model_name:
        raise ValueError(
            f"init_from={init_from} holds a {model.name!r} model, but the model group names {model_name!r}: "
            f"use model={model.name}"
        )
    return model, read_json(run / "metrics.json")


def summary_line(label: str, s: dict[str, float]) -> str:
    return (
        f"  {label:<9}: RMSE s {s['rmse_s_mean']:.2f}/{s['rmse_s_median']:.2f} m  v {s['rmse_v_mean']:.2f}/"
        f"{s['rmse_v_median']:.2f} m/s (mean/median)  collisions {s['collision_rate']:.1%}  "
        f"open-loop a {s['open_loop_rmse_a']:.3f} m/s^2"
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    t_start = time.perf_counter()
    plain = to_plain(cfg)
    model_cfg = dict(plain["model"])
    train_overrides = model_cfg.pop("train_overrides", None) or {}
    train_cfg = TrainConfig.from_mapping(merge_overrides(plain["train"], train_overrides))
    device = resolve_device(train_cfg.device)
    data_name, model_choice = cfg.data.name, HydraConfig.get().runtime.choices["model"]
    check_options(cfg, model_cfg)
    run_dir = (
        resolve_path(cfg.paths.runs_root) / cfg.experiment / data_name / model_choice
        / f"{cfg.split}_fold{cfg.fold}_seed{cfg.seed}"
    )  # fmt: skip

    parts, data_view = load_parts(to_plain(cfg.data), cfg.split, cfg.fold, cfg.paths.events_root, cfg.paths.splits_root)
    parts["train"] = subsample(parts["train"], cfg.max_train_events, cfg.seed)  # seeded sub-sample (smoke runs)

    stats = training_context(parts["train"], cfg.seed, train_cfg.band)
    idm_fit = newell_fit = ovm_fit = source = budget = None
    if cfg.init_from is not None:  # fine-tuning: the checkpoint as it is, no calibration and no fit
        model, source = load_source(cfg.init_from, model_cfg["name"])
        physics = carried_params(model)
        fit = {"source": "checkpoint", "params": {k: v for params in physics.values() for k, v in params.items()}}
    else:
        idm_fit = fold_calibration("idm", parts["train"], cfg, device) if cfg.model.name in NEEDS_IDM else None
        newell_fit = fold_calibration("newell", parts["train"], cfg, device) if cfg.model.name in NEEDS_NEWELL else None
        ovm_fit = fold_calibration("ovm", parts["train"], cfg, device) if cfg.model.name in NEEDS_OVM else None
        physics = {
            "idm_params": idm_fit["params"] if idm_fit else {},
            "newell_params": newell_fit["params"] if newell_fit else {},
            "ovm_params": ovm_fit["params"] if ovm_fit else {},
        }
        context = ModelContext(
            center=stats["center"], scale=stats["scale"], box_low=stats["box_low"], box_high=stats["box_high"],
            dt=DT, seed=cfg.seed, **physics,
        )  # fmt: skip
        seed_everything(cfg.seed)  # after the calibrations: the initial weights must not depend on the cache
        model = build_model(model_cfg, context)
        fit = model.fit(parts["train"], context)
        if cfg.certificate.enforce:  # lipschitz from the a priori certificate of the calibrated core (D86)
            budget = enforce_budget(model, float(cfg.certificate.safety))
    training = train_model(model, parts["train"], parts["val"], train_cfg, cfg.seed) if model.trainable else None

    frames, summaries = {}, {}
    for name in ("val", "test"):
        frames[name] = evaluate_closed_loop(model, parts[name], warmup=train_cfg.warmup, device=device)
        summaries[name] = {
            **summarise(frames[name]),
            "open_loop_rmse_a": open_loop_rmse(model, parts[name], train_cfg.warmup, device=device),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, run_dir / "model.pt")
    frames["test"].to_parquet(run_dir / "test_events.parquet", index=False)
    metrics = {
        "config": plain,
        "config_hash": config_hash(plain),
        "train_config": dataclasses.asdict(train_cfg),  # effective, after the model's train_overrides
        "git_revision": git_revision(),
        "data": data_name,
        "data_view": data_view,
        "model": model_choice,
        "model_config": model.config(),  # effective: that of the checkpoint with init_from, lipschitz of the budget
        "split": cfg.split,
        "fold": cfg.fold,
        "seed": cfg.seed,
        "init_from": cfg.init_from,
        "init_config_hash": source["config_hash"] if source else None,
        "parts": {name: {"n_events": len(evs), "n_samples": sum(len(ev) for ev in evs)} for name, evs in parts.items()},
        "context": {**{k: v for k, v in stats.items() if k != "seed"}, **physics},
        "idm_calibration": idm_fit,
        "newell_calibration": newell_fit,
        "ovm_calibration": ovm_fit,
        "certificate_budget": budget,
        "fit": fit,
        "best_epoch": training["best_epoch"] if training else None,
        "training": training,
        "val": summaries["val"],
        "test": summaries["test"],
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "wall_time_s": time.perf_counter() - t_start,
    }
    write_json(run_dir / "metrics.json", metrics)

    counts = " / ".join(f"{metrics['parts'][p]['n_events']}" for p in PARTS)
    view = ""
    if data_view["filter"]:
        rule, dropped = data_view["filter"], sum(part["dropped"] for part in data_view["parts"].values())
        prefix = f", follower prefix {rule['follower_prefix']}" if rule.get("follower_prefix") else ""
        view = (
            f"  view of {data_view['events']}: mode {rule['mode']}, {len(rule['exclude_files'])} files excluded"
            f"{prefix}, {dropped} events dropped"
        )
    lines = [
        f"train  data={data_name}  model={model_choice}  split={cfg.split}  fold={cfg.fold}  seed={cfg.seed}  "
        f"device={device}  hash={metrics['config_hash']}",
        f"  events   : {counts} (train/val/test)  parameters {metrics['n_trainable_parameters']} trainable{view}",
    ]
    if source:
        lines.append(f"  init     : {cfg.init_from} (config hash {source['config_hash']})")
    if fit.get("params"):
        lines.append("  fit      : " + "  ".join(f"{k}={v:.3g}" for k, v in fit["params"].items()))
    if budget:
        lines.append(
            f"  budget   : admissible r_max*lipschitz {budget['admissible']:.4g}, lipschitz {budget['lipschitz']:.4g} "
            f"(safety {budget['safety']:g}, r_max {budget['r_max']:g})"
        )
    if training:
        history, best = training["history"], training["best_epoch"]
        epoch_time = np.mean([h["time_train_s"] + h["time_val_s"] for h in history]) if history else float("nan")
        keys = ("loss_acc", "loss_rollout", "loss_extra") + (("loss_penalty",) if train_cfg.penalty.active else ())
        terms = "  ".join(f"{k[5:]} {history[best - 1][k]:.4f}" for k in keys) if best else ""  # epoch 0: no loss
        if best and train_cfg.penalty.kind == "combined":  # D110: the Jacobian part, mean over the steps, unweighted
            terms += f"  jacobian {history[best - 1]['penalty_jacobian']:.4f}"
        lines.append(
            f"  training : {training['epochs']} epochs, best {best} ({training['stop_reason']}), "
            f"{epoch_time:.1f} s/epoch" + (f"  best-epoch loss {terms}" if best else "")
        )
    lines += [summary_line("val", summaries["val"]), summary_line("test", summaries["test"])]
    lines.append(f"  written  : {run_dir}  ({metrics['wall_time_s']:.0f} s)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
