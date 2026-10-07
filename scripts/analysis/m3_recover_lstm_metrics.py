"""Recovery of metrics.json of the penalised LSTM (M3 preview, second pass).

The training ended on 2026-09-28 19:20 and wrote model.pt and test_events.parquet; the process then
hung in the child process `git rev-parse` and never wrote metrics.json. The evaluation is repeated
here from the saved weights with the calls of scripts/train.py; the training history is lost.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))

from cf_stability.data.schema import EventSet  # noqa: E402
from cf_stability.data.splits import fold_ids, load_split  # noqa: E402
from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.penalties import stability_penalty  # noqa: E402
from cf_stability.train.calibration import resolve_device  # noqa: E402
from cf_stability.train.evaluate import evaluate_closed_loop, open_loop_rmse, summarise  # noqa: E402
from cf_stability.train.tensors import training_context  # noqa: E402
from cf_stability.train.trainer import TrainConfig  # noqa: E402
from cf_stability.utils import config_hash, git_revision, resolve_path, to_plain, write_json  # noqa: E402

OVERRIDES = [
    "data=follownet_highd", "fold=0", "seed=0", "model=lstm", "train.penalty.kind=linear_gain",
    "train.penalty.weight=1.0", "experiment=m3_linear_gain",
]
PARTS = ("train", "val", "test")

t_start = time.perf_counter()
with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
    cfg = compose(config_name="train", overrides=OVERRIDES)
plain = to_plain(cfg)
model_cfg = dict(plain["model"])
model_cfg.pop("train_overrides", None)
train_cfg = TrainConfig.from_mapping(plain["train"])
device = resolve_device(train_cfg.device)
run_dir = resolve_path(cfg.paths.runs_root) / cfg.experiment / cfg.data.name / "lstm" / f"{cfg.split}_fold{cfg.fold}_seed{cfg.seed}"

event_set = EventSet.from_parquet(resolve_path(cfg.paths.events_root) / cfg.data.name)
split = load_split(resolve_path(cfg.paths.splits_root) / f"{cfg.data.name}_{cfg.split}.json")
parts = {name: event_set.subset(ids).events for name, ids in zip(PARTS, fold_ids(split, cfg.fold))}
stats = training_context(parts["train"], cfg.seed)

model = load_model(run_dir / "model.pt", device)
frames, summaries = {}, {}
for name in ("val", "test"):
    frames[name] = evaluate_closed_loop(model, parts[name], warmup=train_cfg.warmup, device=device)
    summaries[name] = {
        **summarise(frames[name]),
        "open_loop_rmse_a": open_loop_rmse(model, parts[name], train_cfg.warmup, device=device),
    }

# The stored frame of the test events comes from the run itself: it has to agree with the repetition.
stored = pd.read_parquet(run_dir / "test_events.parquet")
assert list(stored["event_id"]) == list(frames["test"]["event_id"])
difference = {c: float(np.nanmax(np.abs(stored[c].to_numpy(float) - frames["test"][c].to_numpy(float)))) for c in ("rmse_s", "rmse_v")}
same_collisions = bool((stored["collided"].to_numpy() == frames["test"]["collided"].to_numpy()).all())
stored_summary = summarise(stored)

# Penalty of the saved weights on the grid speeds of the audit (the training draws 16 speeds per epoch).
penalty_cfg = train_cfg.penalty
if penalty_cfg.exist_v_min is None and penalty_cfg.exist_v_max is None:
    penalty_cfg = dataclasses.replace(
        penalty_cfg,
        exist_v_min=max(penalty_cfg.v_min, float(stats["box_low"][2])),
        exist_v_max=min(penalty_cfg.v_max, float(stats["box_high"][2])),
    )
penalty_value = None
try:
    speeds = torch.arange(5.0, 30.5, 1.0, device=device, dtype=next(model.parameters()).dtype)
    value, info = stability_penalty(model, speeds, penalty_cfg)
    penalty_value = {"value": float(value), **{k: float(v) for k, v in info.items()}}
except Exception as error:  # the penalty interface is only needed for the report
    penalty_value = {"error": repr(error)}

metrics = {
    "config": plain,
    "config_hash": config_hash(plain),
    "train_config": dataclasses.asdict(train_cfg),
    "git_revision": git_revision(),
    "data": cfg.data.name,
    "model": "lstm",
    "split": cfg.split,
    "fold": cfg.fold,
    "seed": cfg.seed,
    "parts": {name: {"n_events": len(evs), "n_samples": sum(len(ev) for ev in evs)} for name, evs in parts.items()},
    "context": {**{k: v for k, v in stats.items() if k != "seed"}, "idm_params": {}, "newell_params": {}, "ovm_params": {}},
    "idm_calibration": None,
    "newell_calibration": None,
    "ovm_calibration": None,
    "fit": {},
    "best_epoch": None,
    "training": None,
    "val": summaries["val"],
    "test": summaries["test"],
    "n_parameters": sum(p.numel() for p in model.parameters()),
    "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "device": str(device),
    "wall_time_s": None,
    "recovered": {
        "reason": "the training process hung in the child process `git rev-parse` after writing model.pt and "
        "test_events.parquet (2026-09-28 19:20); metrics.json is rebuilt from the saved weights",
        "lost": ["training history", "best epoch", "wall time"],
        "training_started": "2026-09-28 18:53:58",
        "weights_written": "2026-09-28 19:20:45",
        "max_abs_difference_to_stored_test_frame": difference,
        "same_collisions_as_stored_test_frame": same_collisions,
        "stored_test_summary": stored_summary,
        "penalty_of_saved_weights": penalty_value,
        "recovery_time_s": time.perf_counter() - t_start,
    },
}
out = Path(sys.argv[2])
write_json(out, metrics)
print("written", out)
print("test     ", json.dumps(summaries["test"]))
print("val      ", json.dumps(summaries["val"]))
print("stored   ", json.dumps(stored_summary))
print("difference to stored frame", difference, "same collisions", same_collisions)
print("penalty  ", penalty_value)
