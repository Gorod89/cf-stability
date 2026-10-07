"""scripts/train.py end to end on tiny event sets, from a foreign working directory: train_overrides, views of
an event set, fine-tuning, stable core, certified budget (docs/m4_contract.md, 3.1), the penalty ``combined``
(docs/m7_contract.md, section 3)."""

import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.data.splits import fold_ids, write_splits
from cf_stability.models import load_model
from cf_stability.models.idm import idm_acc
from cf_stability.stability.certificate import certificate_a_priori
from cf_stability.train.calibration import CalibrationConfig
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.evaluate import evaluate_closed_loop, summarise
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

DATA = "synthetic"  # selects configs/data/synthetic.yaml; events and splits come from tmp_path
SUMMARY_KEYS = {
    "n_events", "collision_rate", "open_loop_rmse_a",
    *(f"rmse_{y}_{stat}" for y in ("s", "v") for stat in ("mean", "median", "pooled")),
}  # fmt: skip
SHORT_CALIBRATION = ("+calibration.maxiter=5", "+calibration.restarts=1")  # a few generations of the IDM fit
PLATOON_FILES = yaml.safe_load((REPO_ROOT / "configs" / "stability" / "platoon.yaml").read_text(encoding="utf-8"))[
    "profiles"
]
# global IDM fits on FollowNet HighD, fold 0 (docs/m3_report.md, 3.4), with a margin of at least 0.5 / 0.2
CORE_M05 = {"v0": 35.1, "T": 0.94, "s0": 4.71, "a": 2.74, "b": 0.5}
CORE_M02 = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}


def tiny_event(i: int, n: int = 150, v_mean: float = 12.0) -> Event:
    t = DT * np.arange(n)
    v_lead = v_mean + 3.0 * np.sin(2 * np.pi * t / 7.0 + i)
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, 30.0, 1.3, 2.0, 1.2, 1.8), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(22.0 + 1.5 * (v_mean - 12.0)), torch.tensor(v_mean),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"{DATA}/a/{i}|L{i}|0", dataset=DATA, site="a", follower_id=f"{DATA}/a/{i}",
        leader_id=f"{DATA}/a/L{i}", t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead,
        x_follower=x_lead - s,
    )  # fmt: skip


def quoted(path: Path | str) -> str:
    return f"'{Path(path).as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def train_process(
    tmp_path: Path, *extra: str, data: str = DATA, model: str = "mlp", max_epochs: int = 2, experiment: str = "test"
) -> subprocess.CompletedProcess:
    overrides = [
        f"data={data}", f"model={model}", f"experiment={experiment}",
        f"paths.events_root={quoted(tmp_path / 'events')}", f"paths.splits_root={quoted(tmp_path / 'splits')}",
        f"paths.calibration_root={quoted(tmp_path / 'calibration')}", f"paths.runs_root={quoted(tmp_path / 'runs')}",
        f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
        f"train.max_epochs={max_epochs}", "train.steps_per_epoch=5", "train.device=cpu", *extra,
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), *overrides],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip


def run_train(tmp_path: Path, *extra: str, **kwargs) -> str:
    proc = train_process(tmp_path, *extra, **kwargs)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def failed_train(tmp_path: Path, *extra: str, **kwargs) -> str:
    """Output of a run that must end with an error."""
    proc = train_process(tmp_path, *extra, **kwargs)
    assert proc.returncode != 0, proc.stdout
    return proc.stdout + proc.stderr


def write_events(root: Path, name: str, events: list[Event]) -> EventSet:
    event_set = EventSet(events)
    event_set.validate()
    event_set.to_parquet(root / "events" / name)
    write_splits(event_set.events_frame(), name, root / "splits")
    return event_set


@pytest.fixture()
def events(tmp_path: Path) -> EventSet:
    return write_events(tmp_path, DATA, [tiny_event(i) for i in range(10)])


def test_train_script(tmp_path, events):
    out = run_train(tmp_path)
    run_dir = tmp_path / "runs" / "test" / DATA / "mlp" / "driver_fold0_seed0"
    assert {p.name for p in run_dir.iterdir()} == {"metrics.json", "model.pt", "test_events.parquet"}
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["config_hash"] == config_hash(metrics["config"]) and metrics["config_hash"] in out
    assert [metrics["parts"][p]["n_events"] for p in ("train", "val", "test")] == [6, 2, 2]
    assert len(metrics["training"]["history"]) == 2 and metrics["best_epoch"] in (1, 2)
    assert set(metrics["val"]) == set(metrics["test"]) == SUMMARY_KEYS

    df = pd.read_parquet(run_dir / "test_events.parquet")
    test_ids = sorted(e for e, fold in read_json(tmp_path / "splits" / f"{DATA}_driver.json")["folds"].items() if fold == 0)
    assert sorted(df.event_id) == test_ids  # one row per test event
    # the checkpoint holds the evaluated weights
    again = evaluate_closed_loop(load_model(run_dir / "model.pt"), [events[e] for e in df.event_id], device="cpu")
    pd.testing.assert_frame_equal(again, df)
    # a second run with the same overrides reproduces the per-event results exactly
    run_train(tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(run_dir / "test_events.parquet"), df, check_exact=True)
    assert metrics["train_config"]["acc_weight"] == 1.0 and metrics["train_config"]["penalty"]["kind"] == "none"


def test_train_overrides_of_the_model_config(tmp_path, events):
    residual = yaml.safe_load((REPO_ROOT / "configs" / "model" / "residual_idm.yaml").read_text(encoding="utf-8"))
    assert residual["train_overrides"] == {"acc_weight": 0.0}  # D63
    overrides = "+model.train_overrides={acc_weight:0.0,penalty:{kind:jacobian,n_equilibria:4},rollout:{horizon:20}}"
    run_train(tmp_path, "train.acc_weight=1.0", "train.penalty.weight=0.1", overrides)
    metrics = read_json(tmp_path / "runs" / "test" / DATA / "mlp" / "driver_fold0_seed0" / "metrics.json")
    effective = metrics["train_config"]
    assert effective["acc_weight"] == 0.0 and effective["rollout"] == {"weight": 1.0, "horizon": 20, "segments": 256}
    assert effective["penalty"]["kind"] == "jacobian" and effective["penalty"]["weight"] == 0.1
    assert effective["penalty"]["n_equilibria"] == 4 and metrics["config"]["train"]["acc_weight"] == 1.0
    assert metrics["config"]["model"]["train_overrides"]["acc_weight"] == 0.0  # the raw config keeps them
    history = metrics["training"]["history"]
    assert all("loss_penalty" in h and "penalty_n_no_equilibrium" in h for h in history)
    assert all(h["loss"] == pytest.approx(h["loss_rollout"] + 0.1 * h["loss_penalty"], rel=1e-12) for h in history)


def test_combined_penalty_from_the_command_line(tmp_path, events):
    """D110 through scripts/train.py: the keys of the Jacobian part reach the penalty, the best-epoch loss
    shows the part, and the keys enter the config hash once they leave their defaults (0)."""
    combined = (
        "train.penalty.kind=combined", "train.penalty.weight=0.1", "train.penalty.every=3",
        "train.penalty.jacobian_weight=1.0", "train.penalty.guard=3.0", "train.penalty.n_equilibria=4",
        "train.penalty.horizon_s=4.0", "train.penalty.measure_s=2.0", "train.penalty.v_min=10.0",
        "train.penalty.v_max=14.0", "train.batch_size=128", "model.window=10", "model.hidden_size=8",
    )  # fmt: skip
    out = run_train(tmp_path, *combined, model="gru", experiment="combined")
    metrics = read_json(tmp_path / "runs" / "combined" / DATA / "gru" / "driver_fold0_seed0" / "metrics.json")
    penalty, training = metrics["train_config"]["penalty"], metrics["training"]
    assert [penalty[k] for k in ("kind", "jacobian_weight", "guard", "every")] == ["combined", 1.0, 3.0, 3]
    assert training["steps_per_epoch"] == 5 and training["penalty_steps"] == 4  # steps 0, 3, 6, 9 of 10
    assert all(math.isfinite(h["penalty_jacobian"]) and math.isfinite(h["val_penalty"]) for h in training["history"])
    assert metrics["config_hash"] == config_hash(metrics["config"]) and "  jacobian " in out
    defaults = copy.deepcopy(metrics["config"])
    defaults["train"]["penalty"].update(jacobian_weight=0.0, guard=0.0)
    assert config_hash(defaults) != metrics["config_hash"]
    out = failed_train(tmp_path, "train.penalty.kind=gain", "train.penalty.jacobian_weight=1.0")
    assert "belong to kind combined" in out


# ------------------------------------------------------------------ views of an event set (D87)
OPENACC_EVENTS = [  # vehicle, driving mode, run file, campaign folder, fold of the driver
    ("A0", "ACC", "run_a.csv", "ZalaZone", 0), ("A0", "ACC", "run_b.csv", "ZalaZone", 0),
    ("H0", "Human", "run_a.csv", "ZalaZone", 0), ("H0", "Human", "run_b.csv", "ZalaZone", 0),
    ("A1", "ACC", "run_a.csv", "ZalaZone", 1), ("A1", "ACC", "run_c.csv", "ZalaZone", 1),
    ("H1", "Human", "run_c.csv", "ZalaZone", 1), ("H1", "Human", "run_d.csv", "ZalaZone", 1),
    ("A2", "ACC", "handling_part30.csv", "ZalaZone", 2),  # leader profile of the platoon test: dropped
    ("A2", "ACC", "run_c.csv", "ZalaZone", 2),
    ("A3", "ACC", "handling_part30.csv", "Casale", 3),  # the same file name in another campaign: kept
    ("A3", "ACC", "run_d.csv", "Casale", 3),
    ("A4", "ACC", "JRC-VC_280219_part2.csv", "Vicolungo", 4),  # leader profile: dropped
    ("A4", "ACC", "run_e.csv", "Vicolungo", 4), ("A4", "ACC", "run_f.csv", "Vicolungo", 4),
]  # fmt: skip


def openacc_event(i: int, vehicle: str, mode: str, file: str, campaign: str) -> Event:
    """A tiny event labelled like an OpenACC one: the driver is (vehicle, mode), the run file is in ``meta``."""
    follower = f"openacc/{campaign}/{vehicle}:{mode}"
    return dataclasses.replace(
        tiny_event(i), event_id=f"{follower}|L|{i}", dataset="openacc", site=campaign, follower_id=follower,
        leader_id=f"openacc/{campaign}/L", meta={"campaign": campaign, "file": file, "driver_mode": mode},
    )  # fmt: skip


@pytest.fixture(scope="module")
def openacc(tmp_path_factory) -> Path:
    """OpenACC-like event set with a hand-made driver split (fold 0: test fold 0, validation fold 1, training
    folds 2-4; no human driver in training) and a run of the IDM on its ACC view."""
    root = tmp_path_factory.mktemp("openacc")
    events = [openacc_event(i, *row[:4]) for i, row in enumerate(OPENACC_EVENTS)]
    EventSet(events).to_parquet(root / "events" / "openacc")
    folds = {ev.event_id: row[4] for ev, row in zip(events, OPENACC_EVENTS)}
    write_json(root / "splits" / "openacc_driver.json", {"kind": "driver", "n_folds": 5, "seed": 0, "folds": folds})
    run_train(root, *SHORT_CALIBRATION, data="openacc_acc", model="idm")
    return root


def test_view_filters_the_driving_mode_and_the_run_files(openacc):
    for name in ("openacc_acc", "openacc_human"):  # views of openacc without the leader profiles of the platoon test
        view = yaml.safe_load((REPO_ROOT / "configs" / "data" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert view["name"] == name and view["events"] == view["splits"] == "openacc"
        assert view["filter"]["exclude_files"] == PLATOON_FILES and len(PLATOON_FILES) == 5
    run_dir = openacc / "runs" / "test" / "openacc_acc" / "idm" / "driver_fold0_seed0"
    metrics = read_json(run_dir / "metrics.json")
    parts = {"train": (5, 2), "val": (2, 2), "test": (2, 2)}  # kept, dropped
    assert metrics["data"] == "openacc_acc" and metrics["data_view"] == {
        "events": "openacc", "splits": "openacc", "filter": {"mode": "ACC", "exclude_files": PLATOON_FILES},
        "parts": {part: {"kept": kept, "dropped": dropped} for part, (kept, dropped) in parts.items()},
    }  # fmt: skip
    assert [metrics["parts"][p]["n_events"] for p in ("train", "val", "test")] == [5, 2, 2]
    test_ids = pd.read_parquet(run_dir / "test_events.parquet").event_id
    assert sorted(test_ids) == [f"openacc/ZalaZone/A0:ACC|L|{i}" for i in (0, 1)]
    # the calibration of the view has its own cache directory and saw the kept training events only
    (cache,) = (openacc / "calibration" / "openacc_acc").glob("idm_global_driver_fold0_*.json")
    assert read_json(cache)["n_events"] == 5 and metrics["idm_calibration"] == read_json(cache)
    assert sorted(p.name for p in (openacc / "calibration").iterdir()) == ["openacc_acc"]


def test_view_with_an_empty_part_is_an_error(openacc):
    out = failed_train(openacc, *SHORT_CALIBRATION, data="openacc_human", model="idm")
    assert "data=openacc_human: fold 0" in out and "has no train events" in out
    assert not (openacc / "runs" / "test" / "openacc_human").exists()


def test_stable_core_reaches_the_calibration(openacc):
    run_train(openacc, *SHORT_CALIBRATION, "calibration.stability_margin=0.05", data="openacc_acc", model="idm",
              experiment="stable")  # fmt: skip
    caches = [read_json(p) for p in (openacc / "calibration" / "openacc_acc").glob("idm_global_driver_fold0_*.json")]
    free, stable = sorted(caches, key=lambda c: c["settings"]["stability_margin"] is not None)
    assert free["settings"]["stability_margin"] is None and "margin_min" not in free
    assert stable["settings"]["stability_margin"] == stable["stability_margin"] == 0.05 and "margin_min" in stable
    assert {**stable["settings"], "stability_margin": None} == free["settings"]
    # without extra keys the settings, hence the cache keys of the free fits, are those of M2 and M3
    defaults = {k: v for k, v in dataclasses.asdict(CalibrationConfig()).items() if k != "device"}
    assert free["settings"] == {**json.loads(json.dumps(defaults)), "maxiter": 5, "restarts": 1}
    metrics = read_json(openacc / "runs" / "stable" / "openacc_acc" / "idm" / "driver_fold0_seed0" / "metrics.json")
    assert metrics["idm_calibration"] == stable and metrics["context"]["idm_params"] == stable["params"]


# ------------------------------------------------------------------ fine-tuning and certified budget (D86)
def test_fine_tuning_starts_from_the_checkpoint(tmp_path, events):
    run_train(tmp_path, max_epochs=1, experiment="source")
    source = tmp_path / "runs" / "source" / DATA / "mlp" / "driver_fold0_seed0"
    tuning = write_events(tmp_path, "synth_ft", [tiny_event(i, v_mean=20.0) for i in range(10)])
    try:  # relative paths are taken from the repository root, not from the working directory
        init_from = Path(os.path.relpath(source, REPO_ROOT))
    except ValueError:  # another drive
        init_from = source
    out = run_train(tmp_path, "data.name=synth_ft", f"init_from={quoted(init_from)}", max_epochs=0, experiment="ft")
    run_dir = tmp_path / "runs" / "ft" / "synth_ft" / "mlp" / "driver_fold0_seed0"
    metrics, source_metrics = read_json(run_dir / "metrics.json"), read_json(source / "metrics.json")
    assert metrics["init_from"] == init_from.as_posix() and metrics["init_config_hash"] == source_metrics["config_hash"]
    assert metrics["fit"] == {"source": "checkpoint", "params": {}} and metrics["best_epoch"] == 0
    assert metrics["context"]["center"] != source_metrics["context"]["center"]  # the context of the new data
    tuned, original = load_model(run_dir / "model.pt"), load_model(source / "model.pt")
    assert list(tuned.state_dict()) == list(original.state_dict())
    for name, value in tuned.state_dict().items():  # the weights and the scaler of the checkpoint
        assert torch.equal(value, original.state_dict()[name]), name
    val = [tuning[e] for e in fold_ids(read_json(tmp_path / "splits" / "synth_ft_driver.json"), 0)[1]]
    initial = summarise(evaluate_closed_loop(original, val, device="cpu"))["rmse_s_mean"]
    assert metrics["training"]["initial_val_rmse_s"] == pytest.approx(initial, rel=1e-12)
    assert f"init     : {init_from.as_posix()}" in out

    assert "holds a 'mlp' model" in failed_train(tmp_path, f"init_from={quoted(source)}", model="gru")
    out = failed_train(tmp_path, f"init_from={quoted(source)}", "certificate.enforce=true", model="residual_idm")
    assert "cannot be combined with init_from" in out


def test_certified_budget(tmp_path, events):
    run_train(tmp_path, *SHORT_CALIBRATION, model="idm")  # writes the cached IDM calibration of the fold
    (cache,) = (tmp_path / "calibration" / DATA).glob("idm_global_driver_fold0_*.json")
    calibration = read_json(cache)
    write_json(cache, {**calibration, "params": CORE_M05})  # the next runs take their core from the cache
    out = run_train(tmp_path, *SHORT_CALIBRATION, "certificate.enforce=true", model="residual_idm", max_epochs=1)
    run_dir = tmp_path / "runs" / "test" / DATA / "residual_idm" / "driver_fold0_seed0"
    metrics, model = read_json(run_dir / "metrics.json"), load_model(run_dir / "model.pt")
    budget = metrics["certificate_budget"]
    assert set(budget) == {"admissible", "safety", "r_max", "lipschitz"} and budget["admissible"] > 0
    assert budget["safety"] == 0.9 and budget["r_max"] == model.r_max == 1.0 and "budget   :" in out
    assert model.lipschitz == budget["lipschitz"] == pytest.approx(0.9 * budget["admissible"], rel=1e-12)
    assert metrics["model_config"] == model.config() and metrics["config"]["model"]["lipschitz"] == 1.0
    assert metrics["training"]["epochs"] == 1 and model.idm.params_dict() == pytest.approx(CORE_M05, rel=1e-6)
    assert certificate_a_priori(model)["holds"].all()  # the trained model is certified

    write_json(cache, {**calibration, "params": CORE_M02})  # with r_max = 1 this core admits no residual
    out = failed_train(tmp_path, *SHORT_CALIBRATION, "certificate.enforce=true", model="residual_idm")
    assert "no admissible residual budget" in out
    assert "needs model=residual_idm" in failed_train(tmp_path, "certificate.enforce=true", model="mlp")
    out = failed_train(tmp_path, "certificate.enforce=true", "model.idm_learnable=true", model="residual_idm")
    assert "needs a fixed IDM core" in out
