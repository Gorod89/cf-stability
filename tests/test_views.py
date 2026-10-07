"""Views of an event set (cf_stability/data/views.py; D87, D120): the filter keys, the parts of a fold, the
training part of a run rebuilt from its config, and the temporal view ngsim_i80_p0 through scripts/train.py."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.data.views import FILTER_KEYS, keep_event, load_parts, run_training_events, subsample, view_filter
from cf_stability.models.idm import idm_acc
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.utils import REPO_ROOT, read_json, write_json


def tiny_event(i: int, follower: str, n: int = 150) -> Event:
    t = DT * np.arange(n)
    v_lead = 12.0 + 3.0 * np.sin(2 * np.pi * t / 7.0 + i)
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, 30.0, 1.3, 2.0, 1.2, 1.8), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(22.0), torch.tensor(12.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"{follower}|L{i}|0", dataset="ngsim_i80", site="i80", follower_id=follower,
        leader_id=f"ngsim_i80/i80/L{i}", t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead,
        x_follower=x_lead - s,
    )  # fmt: skip


# two drivers of every period per fold: fold k holds p0_k, p0_(k+5), p1_k, p2_k
EVENTS = [(f"ngsim_i80/i80/p{p}_{i}", (i % 5)) for p in (0, 1, 2) for i in range(10 if p == 0 else 5)]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    events = [tiny_event(i, follower) for i, (follower, _) in enumerate(EVENTS)]
    EventSet(events).to_parquet(tmp_path / "events" / "ngsim_i80")
    folds = {ev.event_id: fold for ev, (_, fold) in zip(events, EVENTS)}
    split = {"kind": "driver", "n_folds": 5, "seed": 0, "folds": folds}
    write_json(tmp_path / "splits" / "ngsim_i80_driver.json", split)
    return tmp_path


def test_filter_keys_and_rules():
    assert FILTER_KEYS == ("mode", "exclude_files", "follower_prefix")
    assert view_filter({"name": "x"}) is None and view_filter({"name": "x", "filter": None}) is None
    # the views of D87 keep their rule (and metrics.json["data_view"] its form): no follower_prefix key
    acc = yaml.safe_load((REPO_ROOT / "configs" / "data" / "openacc_acc.yaml").read_text(encoding="utf-8"))
    assert set(view_filter(acc)) == {"mode", "exclude_files"} and view_filter(acc)["mode"] == "ACC"
    p0 = yaml.safe_load((REPO_ROOT / "configs" / "data" / "ngsim_i80_p0.yaml").read_text(encoding="utf-8"))
    assert p0["name"] == "ngsim_i80_p0" and p0["events"] == p0["splits"] == "ngsim_i80"
    rule = view_filter(p0)
    assert rule == {"mode": None, "exclude_files": [], "follower_prefix": "ngsim_i80/i80/p0_"}
    with pytest.raises(ValueError, match="unknown filter keys"):
        view_filter({"name": "x", "filter": {"prefix": "a"}})
    event = tiny_event(0, "ngsim_i80/i80/p0_17")
    assert keep_event(event, rule) and keep_event(event, None)
    assert not keep_event(dataclasses.replace(event, follower_id="ngsim_i80/i80/p1_17"), rule)
    assert not keep_event(dataclasses.replace(event, follower_id="ngsim_i80/i80/p10_3"), rule)  # the prefix ends with _
    both = {"mode": "ACC", "exclude_files": [], "follower_prefix": "openacc/"}  # every key must hold
    assert keep_event(dataclasses.replace(event, follower_id="openacc/Z/a:ACC"), both)
    assert not keep_event(dataclasses.replace(event, follower_id="openacc/Z/a:Human"), both)
    assert not keep_event(dataclasses.replace(event, follower_id="other/Z/a:ACC"), both)


def test_parts_of_the_period_zero_view(root):
    p0 = yaml.safe_load((REPO_ROOT / "configs" / "data" / "ngsim_i80_p0.yaml").read_text(encoding="utf-8"))
    parts, view = load_parts(p0, "driver", 0, root / "events", root / "splits")
    assert view["events"] == view["splits"] == "ngsim_i80" and view["filter"]["follower_prefix"] == "ngsim_i80/i80/p0_"
    # test fold 0, validation fold 1, training folds 2-4: two period-0 drivers per fold, the others dropped
    assert {name: (c["kept"], c["dropped"]) for name, c in view["parts"].items()} == {
        "train": (6, 6), "val": (2, 2), "test": (2, 2),
    }  # fmt: skip
    assert all(ev.follower_id.startswith("ngsim_i80/i80/p0_") for part in parts.values() for ev in part)
    full, _ = load_parts({"name": "ngsim_i80"}, "driver", 0, root / "events", root / "splits")
    assert [len(full[p]) for p in ("train", "val", "test")] == [12, 4, 4]
    with pytest.raises(ValueError, match="has no train / val / test events"):
        none = {**p0, "filter": {"follower_prefix": "ngsim_i80/i80/p9_"}}
        load_parts(none, "driver", 0, root / "events", root / "splits")


def test_training_events_of_a_run(root):
    """run_training_events rebuilds the training part from a stored config, the sub-sample of max_train_events
    included (the seeded draw of scripts/train.py)."""
    p0 = yaml.safe_load((REPO_ROOT / "configs" / "data" / "ngsim_i80_p0.yaml").read_text(encoding="utf-8"))
    paths = {"events_root": (root / "events").as_posix(), "splits_root": (root / "splits").as_posix()}
    config = {"data": p0, "split": "driver", "fold": 0, "seed": 4, "max_train_events": None, "paths": paths}
    train = load_parts(p0, "driver", 0, root / "events", root / "splits")[0]["train"]
    assert [ev.event_id for ev in run_training_events(config)] == [ev.event_id for ev in train]
    picked = run_training_events({**config, "max_train_events": 4})
    keep = np.sort(np.random.default_rng(4).choice(6, size=4, replace=False))
    assert [ev.event_id for ev in picked] == [train[i].event_id for i in keep]
    assert subsample(train, None, 0) == train and subsample(train, 6, 0) == train


def test_train_script_on_the_period_zero_view(root):
    """scripts/train.py data=ngsim_i80_p0: the view filters the three parts; the run directory and the calibration
    cache use the name of the view."""
    overrides = [
        "data=ngsim_i80_p0", "model=idm", "experiment=temporal", "+calibration.maxiter=5", "+calibration.restarts=1",
        f"paths.events_root='{(root / 'events').as_posix()}'", f"paths.splits_root='{(root / 'splits').as_posix()}'",
        f"paths.calibration_root='{(root / 'calibration').as_posix()}'",
        f"paths.runs_root='{(root / 'runs').as_posix()}'",
        f"hydra.run.dir='{(root / 'outputs').as_posix()}'", "train.device=cpu",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), *overrides], cwd=root, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "follower prefix ngsim_i80/i80/p0_" in proc.stdout and "10 events dropped" in proc.stdout
    metrics = read_json(root / "runs" / "temporal" / "ngsim_i80_p0" / "idm" / "driver_fold0_seed0" / "metrics.json")
    assert metrics["data"] == "ngsim_i80_p0"
    assert [metrics["parts"][p]["n_events"] for p in ("train", "val", "test")] == [6, 2, 2]
    assert metrics["data_view"]["filter"] == {"mode": None, "exclude_files": [], "follower_prefix": "ngsim_i80/i80/p0_"}
    (cache,) = (root / "calibration" / "ngsim_i80_p0").glob("idm_global_driver_fold0_*.json")
    assert read_json(cache)["n_events"] == 6
