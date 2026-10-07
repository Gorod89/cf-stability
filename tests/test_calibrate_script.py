"""The hydra scripts run end to end on a tiny event set, from a foreign working directory."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.models.idm import idm_acc
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.utils import REPO_ROOT, config_hash, read_json


def tiny_event(i: int, n: int = 160) -> Event:
    t = DT * np.arange(n)
    v_lead = 12.0 + 3.0 * np.sin(2 * np.pi * t / 7.0 + i)
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, 30.0, 1.3, 2.0, 1.2, 1.8), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(22.0), torch.tensor(12.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"tiny/a/{i}|L{i}|0", dataset="tiny", site="a", follower_id=f"tiny/a/{i}", leader_id=f"tiny/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def run_script(name: str, overrides: list[str], cwd: Path) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), *overrides],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.fixture()
def event_root(tmp_path: Path) -> Path:
    events = EventSet(tiny_event(i) for i in range(3))
    events.validate()
    events.to_parquet(tmp_path / "events" / "tiny")
    return tmp_path / "events"


def test_calibrate_idm_script(tmp_path, event_root):
    out = run_script(
        "calibrate_idm.py",
        [
            "dataset=tiny",
            f"paths.events_root={quoted(event_root)}",
            f"paths.calibration_root={quoted(tmp_path / 'calibration')}",
            f"paths.runs_root={quoted(tmp_path / 'runs')}",
            f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
            "calibration.maxiter=5",
        ],
        cwd=tmp_path,
    )
    assert "calibrate_idm" in out and "global" in out
    calib_dir = tmp_path / "calibration" / "tiny"
    payloads = [read_json(p) for p in (calib_dir / "idm_global.json", calib_dir / "idm_spread.json")]
    payloads.append(read_json(tmp_path / "runs" / "baselines" / "tiny" / "idm" / "metrics.json"))
    hashes = {p["config_hash"] for p in payloads}
    assert hashes == {config_hash(payloads[0]["config"])}
    assert payloads[0]["config"]["calibration"]["maxiter"] == 5 and payloads[0]["n_events"] == 3
    assert payloads[1]["n_events"] == 3 and set(payloads[2]) >= {"per_event", "global", "device", "wall_time_s"}
    df = pd.read_parquet(calib_dir / "idm_per_event.parquet")
    assert df.event_id.tolist() == [f"tiny/a/{i}|L{i}|0" for i in range(3)]
    assert df.attrs.get("config_hash") in hashes


def test_calibrate_idm_on_a_view(tmp_path):
    """D120: dataset=<view> (configs/data/ngsim_i80_p0.yaml: events ngsim_i80, follower ids ngsim_i80/i80/p0_*)
    calibrates the events the view keeps and writes under the name of the view."""
    import dataclasses

    events = [
        dataclasses.replace(ev, event_id=f"ngsim_i80/i80/p{i % 2}_{i}|L|0", dataset="ngsim_i80", site="i80",
                            follower_id=f"ngsim_i80/i80/p{i % 2}_{i}")
        for i, ev in enumerate(tiny_event(k) for k in range(5))
    ]  # fmt: skip
    EventSet(events).to_parquet(tmp_path / "events" / "ngsim_i80")
    out = run_script(
        "calibrate_idm.py",
        [
            "dataset=ngsim_i80_p0", "per_event=false", "calibration.maxiter=5", "calibration.restarts=1",
            f"paths.events_root={quoted(tmp_path / 'events')}",
            f"paths.calibration_root={quoted(tmp_path / 'calibration')}",
            f"paths.runs_root={quoted(tmp_path / 'runs')}", f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
        ],
        cwd=tmp_path,
    )  # fmt: skip
    fit = read_json(tmp_path / "calibration" / "ngsim_i80_p0" / "idm_global.json")
    assert fit["n_events"] == 3 and fit["view"]["kept"] == 3 and fit["view"]["dropped"] == 2  # events 0, 2, 4
    assert fit["view"]["events"] == "ngsim_i80" and fit["view"]["filter"]["follower_prefix"] == "ngsim_i80/i80/p0_"
    assert "view     : 3 of 5 events of ngsim_i80" in out
    assert read_json(tmp_path / "runs" / "baselines" / "ngsim_i80_p0" / "idm" / "metrics.json")["n_events"] == 3


def test_persistence_baseline_script(tmp_path, event_root):
    run_script(
        "persistence_baseline.py",
        [
            "dataset=tiny",
            f"paths.events_root={quoted(event_root)}",
            f"paths.runs_root={quoted(tmp_path / 'runs')}",
            f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
            "device=cpu",
        ],
        cwd=tmp_path,
    )
    metrics = read_json(tmp_path / "runs" / "baselines" / "tiny" / "persistence" / "metrics.json")
    assert metrics["config_hash"] == config_hash(metrics["config"]) and metrics["n_events"] == 3
    assert metrics["open_loop"]["stored_acceleration"]["n_samples"] == 3 * 159
    assert metrics["open_loop"]["persistence_model"]["n_samples"] == 3 * 159
    assert metrics["closed_loop"]["n_samples"] == 3 * 159 and metrics["closed_loop"]["rmse_s_mean"] > 0
