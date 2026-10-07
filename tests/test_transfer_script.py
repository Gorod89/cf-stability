"""scripts/evaluate_transfer.py end to end on synthetic target sets (docs/m4_contract.md, 3.2)."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cf_stability.data.schema import ARRAY_FIELDS, DT, Event, EventSet
from cf_stability.models import IDM, MLP, load_model, save_model
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import idm_acc
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.evaluate import evaluate_closed_loop, summarise
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

IDM_P = {"v0": 30.0, "T": 1.3, "s0": 2.0, "a": 1.2, "b": 1.8}  # generates the events
MODEL_P = {"v0": 33.0, "T": 1.1, "s0": 2.5, "a": 1.0, "b": 2.0}  # the evaluated IDM: errors grow with the horizon
TARGETS = {"tgt_a": (150, 90, 60, 45, 35), "tgt_b": (120, 80)}  # samples per event; 35 < warm-up 30 + 1 s
HORIZON_S, WARMUP = 6.0, 30  # the horizon keeps the first 60 samples
SOURCE_TEST = {"n_events": 10, "rmse_s_mean": 0.8, "rmse_s_median": 0.7, "rmse_v_mean": 0.3}
COLUMNS = [
    "event_id", "follower_id", "rmse_s", "rmse_v", "collided", "n_scored",
    "rmse_s_h", "rmse_v_h", "collided_h", "n_scored_h",
]  # fmt: skip
TOP_KEYS = {"run", "model", "warmup", "source", "targets", "config", "config_hash", "git_revision", "wall_time_s"}


def target_event(name: str, i: int, n: int) -> Event:
    t = DT * np.arange(n)
    v_lead = 15.0 + 4.0 * np.sin(2 * np.pi * t / 9.0 + i)
    x_lead = 35.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, *IDM_P.values()), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(26.0), torch.tensor(15.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"{name}/x/{i}|L|0", dataset=name, site="x", follower_id=f"{name}/x/{i % 3}", leader_id=f"{name}/x/L",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def run_script(tmp_path: Path, *overrides: str) -> tuple[int, list[str]]:
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate_transfer.py"), *overrides,
        f"targets=[{','.join(TARGETS)}]", f"horizon_s={HORIZON_S}", "device=cpu",
        f"paths.runs_root={quoted(tmp_path / 'runs')}", f"paths.events_root={quoted(tmp_path / 'events')}",
        f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode in (0, 1) and "Traceback" not in proc.stderr, proc.stdout + proc.stderr
    return proc.returncode, proc.stdout.strip().splitlines()


@pytest.fixture()
def setup(tmp_path: Path) -> dict[str, list[Event]]:
    targets = {name: [target_event(name, i, n) for i, n in enumerate(lengths)] for name, lengths in TARGETS.items()}
    for name, events in targets.items():
        EventSet(events).to_parquet(tmp_path / "events" / name)
    runs = tmp_path / "runs" / "exp" / "synthetic"
    torch.manual_seed(0)
    models = {"idm": IDM(MODEL_P), "mlp": MLP(ModelContext(center=(25.0, 0.0, 15.0), scale=(10.0, 1.0, 4.0)))}
    for name, model in models.items():
        run = runs / name / "driver_fold0_seed0"
        run.mkdir(parents=True)
        save_model(model, run / "model.pt")
        write_json(run / "metrics.json", {"data": "src", "train_config": {"warmup": WARMUP}, "test": SOURCE_TEST})
    broken = runs / "broken" / "driver_fold0_seed0"
    broken.mkdir(parents=True)
    (broken / "model.pt").write_bytes(b"not a checkpoint")
    return targets


def head(event: Event, n: int) -> Event:
    return dataclasses.replace(event, **{name: getattr(event, name)[:n] for name in ARRAY_FIELDS})


def test_transfer_script(tmp_path, setup):
    status, lines = run_script(tmp_path, "experiment=exp", "data=synthetic")
    assert status == 1 and len(lines) == 3  # the broken run fails, the others are evaluated
    assert "FAILED" in lines[0] and lines[1].startswith("idm/driver_fold0_seed0") and lines[2].startswith("mlp/")
    runs = tmp_path / "runs" / "exp" / "synthetic"
    run = runs / "idm" / "driver_fold0_seed0"
    result = read_json(run / "transfer.json")
    assert set(result) == TOP_KEYS and result["model"] == "idm" and result["warmup"] == WARMUP
    assert result["source"] == {"data": "src", "test": SOURCE_TEST}
    settings = ("targets", "horizon_s", "min_scored_s", "device", "batch_events")
    assert result["config_hash"] == config_hash({key: result["config"][key] for key in settings})
    assert "error" in read_json(runs / "broken" / "driver_fold0_seed0" / "transfer.json")

    model = load_model(run / "model.pt")
    for name, events in setup.items():
        entry, table = result["targets"][name], pd.read_parquet(run / f"transfer_{name}.parquet")
        assert set(entry) == {"n_events", "n_dropped", "full", "horizon", "relative_degradation"}
        assert list(table.columns) == COLUMNS and list(table.event_id) == [ev.event_id for ev in events]
        kept = [ev for ev in events if len(ev) >= WARMUP + 10]
        dropped = len(events) - len(kept)
        assert entry["n_events"] == len(events) and entry["n_dropped"] == {"full": dropped, "horizon": dropped}
        # hand computation: whole events and their first HORIZON_S seconds
        full = evaluate_closed_loop(model, kept, warmup=WARMUP, device="cpu")
        cut = evaluate_closed_loop(model, [head(ev, round(HORIZON_S / DT)) for ev in kept], warmup=WARMUP, device="cpu")
        assert entry["full"] == pytest.approx(summarise(full), rel=1e-12)
        assert entry["horizon"] == pytest.approx(summarise(cut), rel=1e-12)
        assert entry["relative_degradation"] == pytest.approx(cut["rmse_s"].mean() / SOURCE_TEST["rmse_s_mean"] - 1)
        rows = table.set_index("event_id").loc[[ev.event_id for ev in kept]]
        np.testing.assert_allclose(rows["rmse_s"], full["rmse_s"], rtol=1e-12)
        np.testing.assert_allclose(rows["rmse_s_h"], cut["rmse_s"], rtol=1e-12)
        assert list(rows["n_scored_h"]) == [min(len(ev), 60) - WARMUP + 1 for ev in kept]  # the horizon cut
        assert list(rows["n_scored"]) == [len(ev) - WARMUP + 1 for ev in kept]
        assert str(table["collided"].dtype) == "boolean" and str(table["n_scored_h"].dtype) == "Int64"
        short = table[~table.event_id.isin([ev.event_id for ev in kept])]
        assert len(short) == dropped and short.drop(columns=["event_id", "follower_id"]).isna().all().all()
    assert result["targets"]["tgt_a"]["n_dropped"]["full"] == 1
    assert result["targets"]["tgt_a"]["horizon"]["rmse_s_mean"] < result["targets"]["tgt_a"]["full"]["rmse_s_mean"]
    assert read_json(runs / "mlp" / "driver_fold0_seed0" / "transfer.json")["model"] == "mlp"

    assert run_script(tmp_path, "experiment=exp", "data=synthetic") == (0, [])  # outputs newer than model.pt are kept
    status, lines = run_script(tmp_path, "experiment=exp", "data=synthetic", "force=true")
    assert status == 1 and len(lines) == 3
    status, lines = run_script(tmp_path, f"run={quoted(run)}")
    assert status == 0 and len(lines) == 1 and lines[0].startswith("idm/driver_fold0_seed0")
