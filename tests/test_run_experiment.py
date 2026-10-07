"""Queue of experiment jobs (cf_stability/eval/experiment_queue.py, scripts/run_experiment.py).

Most tests run the queue in this process with the step scripts replaced by small fakes (a train
that writes ``metrics.json`` with the hash of its hydra config, steps that write their output
file), steered by environment variables; one test runs the real scripts on a tiny synthetic data
set, one kills a running queue and restarts it.
"""

import collections
import ctypes
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.data.splits import write_splits
from cf_stability.eval import experiment_queue as eq
from cf_stability.models import IDM, save_model
from cf_stability.models.idm import idm_acc
from cf_stability.stability.audit import BAND_VARIANTS, band_output
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.train.tensors import BandConfig, training_context
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

DATA = "synthetic"  # configs/data/synthetic.yaml; events and splits of the real runs come from tmp_path
SMALL_AUDIT = (
    "stability.device=cpu", "stability.speeds=[10.0,20.0]", "stability.frequency.n_omega=3",
    "stability.frequency.omega_min=0.2", "stability.frequency.min_discard_s=20.0",
    "stability.frequency.min_measure_s=20.0",
)  # fmt: skip
SMALL_PLATOON = (
    "stability.n_vehicles=4", "stability.profiles=[]", "stability.pulse.duration_s=20.0",
    "stability.empirical_growth=null", "stability.device=cpu",
)  # fmt: skip

FAKE_COMMON = '''
import json, os, subprocess, sys, time
from pathlib import Path

OUTPUTS = {"audit": "stability.json", "platoon": "platoon.json", "transfer": "transfer.json",
           "certificate": "certificate.json"}


def matches(variable, step, run):
    entries = [e.partition(":") for e in os.environ.get(variable, "").split(",") if e]
    return any(s == step and part in run for s, _, part in entries)


def act(step, run):
    """Trace the call; FAKE_FAIL: exit 1, FAKE_HANG: write the pids (own, grandchild) and sleep; the
    entries are <step>:<substring of the run directory>. True when FAKE_ERROR matches."""
    run = Path(run).as_posix()
    trace = Path(os.environ["FAKE_TRACE"], f"{time.time_ns()}_{os.getpid()}.txt")  # a file per call: no races
    trace.write_text(f"{step} {run}", encoding="utf-8")
    if matches("FAKE_FAIL", step, run):
        sys.exit(1)
    if matches("FAKE_HANG", step, run):
        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("fake_sleep.py"))])
        Path(os.environ["FAKE_PIDS"]).write_text(f"{os.getpid()} {child.pid}", encoding="utf-8")
        time.sleep(120)
    return matches("FAKE_ERROR", step, run)
'''
FAKE_TRAIN = '''
import sys
sys.path.insert(0, {repo!r})
import hydra
from hydra.core.hydra_config import HydraConfig
from cf_stability.utils import config_hash, resolve_path, to_plain, write_json
from fake_common import act


@hydra.main(version_base="1.3", config_path={configs!r}, config_name="train")
def main(cfg):  # the run directory and the hash as scripts/train.py forms them
    plain = to_plain(cfg)
    model = HydraConfig.get().runtime.choices["model"]
    name = f"{{cfg.split}}_fold{{cfg.fold}}_seed{{cfg.seed}}"
    run = resolve_path(cfg.paths.runs_root) / cfg.experiment / cfg.data.name / model / name
    act("train", run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "model.pt").write_bytes(b"fake")
    write_json(run / "metrics.json", {{"config": plain, "config_hash": config_hash(plain)}})


main()
'''
FAKE_STEP = '''
import json, sys
from pathlib import Path
from fake_common import OUTPUTS, act

STEP = Path(__file__).stem.removeprefix("fake_")
run = next(a for a in sys.argv[1:] if a.startswith("run=")).split("=", 1)[1].strip("'")
error = act(STEP, run)
payload = {"step": STEP, "argv": sys.argv[1:], **({"error": "RuntimeError: broken"} if error else {})}
Path(run, OUTPUTS[STEP]).write_text(json.dumps(payload), encoding="utf-8")
'''


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def tiny_event(i: int, n: int = 150) -> Event:
    """An IDM follower behind a sinusoidal leader (as in tests/test_train_script.py)."""
    t = DT * np.arange(n)
    v_lead = 12.0 + 3.0 * np.sin(2 * np.pi * t / 7.0 + i)
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, 30.0, 1.3, 2.0, 1.2, 1.8), torch.tensor(x_lead), torch.tensor(v_lead),
        torch.tensor(22.0), torch.tensor(12.0),
    )  # fmt: skip
    s, v = res.s.numpy(), res.v.numpy()
    return Event(
        event_id=f"{DATA}/a/{i}|L{i}|0", dataset=DATA, site="a", follower_id=f"{DATA}/a/{i}",
        leader_id=f"{DATA}/a/L{i}", t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead,
        x_follower=x_lead - s,
    )  # fmt: skip


@pytest.fixture()
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake step scripts in place of the real ones; the trace file lists their calls."""
    directory = tmp_path / "fakes"
    directory.mkdir()
    (directory / "fake_common.py").write_text(FAKE_COMMON, encoding="utf-8")
    (directory / "fake_sleep.py").write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
    train = FAKE_TRAIN.format(repo=str(REPO_ROOT), configs=str(REPO_ROOT / "configs"))
    (directory / "fake_train.py").write_text(train, encoding="utf-8")
    for step, (_, output) in eq.STEPS.items():
        if step != "train":
            (directory / f"fake_{step}.py").write_text(FAKE_STEP, encoding="utf-8")
        monkeypatch.setitem(eq.STEPS, step, (str(directory / f"fake_{step}.py"), output))
    (tmp_path / "trace").mkdir()
    monkeypatch.setenv("FAKE_TRACE", str(tmp_path / "trace"))
    monkeypatch.setenv("FAKE_PIDS", str(tmp_path / "pids.txt"))
    for name in ("FAKE_FAIL", "FAKE_ERROR", "FAKE_HANG"):
        monkeypatch.delenv(name, raising=False)
    return directory


def trace(tmp_path: Path) -> list[str]:
    """Calls of the fake steps since the last look, sorted, as ``<step> <experiment>/<run name>``."""
    calls = []
    for path in (tmp_path / "trace").glob("*.txt"):
        step, run = path.read_text(encoding="utf-8").split(" ", 1)
        calls.append(f"{step} {Path(run).parents[2].name}/{Path(run).name}")
        path.unlink()  # the next look sees the new calls only
    return sorted(calls)


def write_queue(tmp_path: Path, groups: list[dict], name: str = "q", **options) -> Path:
    common = [f"paths.runs_root={quoted(tmp_path / 'runs')}", f"hydra.run.dir={quoted(tmp_path / 'out')}"]
    queue = {"name": name, "workers": 1, "steps": ["train"], **options, "groups": groups, "common_overrides": common}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(queue, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def synthetic(model: str = "mlp", folds=(0,), seeds=(0,), **more) -> dict:
    return {"data": [DATA], "model": [model], "fold": list(folds), "seed": list(seeds), **more}


def run_dir(tmp_path: Path, experiment: str, fold: int, seed: int = 0, model: str = "mlp") -> Path:
    return tmp_path / "runs" / experiment / DATA / model / f"driver_fold{fold}_seed{seed}"


def queue_lines(capsys: pytest.CaptureFixture) -> list[str]:
    """The printed lines of the queue without its start line."""
    lines = capsys.readouterr().out.splitlines()
    return [line for line in lines if line.startswith(("JOB", "QUEUE", "WAIT", "DRY")) and ": start, " not in line]


def alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)  # query limited information, synchronize
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT: still running
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def wait_until(condition, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


# ----------------------------------------------------------------------------------------- queue file, jobs


def test_experiment_override_is_not_shadowed_by_a_config_group():
    """A directory configs/experiment/ would make `experiment` a hydra config group: `experiment=<name>`
    of train.py and of every step script would then fail. The queue files live in configs/queue/."""
    assert not (REPO_ROOT / "configs" / "experiment").exists()


def test_override_values():
    assert [eq.override_value(v) for v in ("mlp", "residual_idm", 0, 0.01, 10.0, True, None)] == [
        "mlp", "residual_idm", "0", "0.01", "10.0", "true", "null",
    ]  # fmt: skip
    assert eq.override_value("e2_jacobian_w0.1") == "'e2_jacobian_w0.1'" and eq.override_value("true") == "'true'"
    assert eq.override_value(Path("C:/Users/Город/runs")) == "'C:/Users/Город/runs'"
    with pytest.raises(eq.QueueError, match="backslashes"):
        eq.override_value("C:\\runs")


def test_jobs_templates_order_and_run_directories(tmp_path):
    source = tmp_path / "runs" / "src"
    penalty = {"train.penalty.kind": ["jacobian"], "train.penalty.weight": [0.01, 10.0]}
    path = write_queue(
        tmp_path,
        [
            {"experiment": "e2_jacobian_w{train.penalty.weight:g}", "overrides": synthetic(folds=(1, 0), **penalty)},
            {"experiment": "e0", "overrides": {**synthetic(folds=(1, 0)), "model": ["residual_idm", "mlp"]}},
            {
                "experiment": "ft", "steps": ["train", "certificate"], "overrides": synthetic(seeds=(2,)),
                "fixed": {"init_from": f"{source.as_posix()}/{DATA}/{{model}}/driver_fold{{fold}}_seed{{seed}}",
                          "calibration.stability_margin": 0.2},
            },
        ],
        order=["seed", "fold", "model"], steps=["train", "audit"], step_overrides={"audit": ["stability.device=cpu"]},
    )  # fmt: skip
    queue = eq.QueueConfig.from_file(path)
    jobs, runs_root = eq.load_jobs(queue)
    assert runs_root == tmp_path / "runs"
    assert [job.label for job in jobs] == [
        f"e2_jacobian_w0.01/{DATA}/mlp/fold0_seed0", f"e2_jacobian_w10/{DATA}/mlp/fold0_seed0",
        f"e0/{DATA}/mlp/fold0_seed0", f"e0/{DATA}/residual_idm/fold0_seed0",
        f"e2_jacobian_w0.01/{DATA}/mlp/fold1_seed0", f"e2_jacobian_w10/{DATA}/mlp/fold1_seed0",
        f"e0/{DATA}/mlp/fold1_seed0", f"e0/{DATA}/residual_idm/fold1_seed0", f"ft/{DATA}/mlp/fold0_seed2",
    ]  # seed slowest, then fold, then model; ties keep the order of the file  # fmt: skip
    first, ft = jobs[0], jobs[-1]
    assert first.run_dir == run_dir(tmp_path, "e2_jacobian_w0.01", 0) and first.steps == ("train", "audit")
    assert first.spec.arguments() == [
        f"data={DATA}", "model=mlp", "fold=0", "seed=0", "train.penalty.kind=jacobian", "train.penalty.weight=0.01",
        "experiment='e2_jacobian_w0.01'",
    ]  # fmt: skip
    assert first.train_args[-2:] == tuple(queue.common_overrides) and first.init_from is None
    assert ft.steps == ("train", "certificate") and ft.init_from == source / DATA / "mlp" / "driver_fold0_seed2"
    assert f"init_from={quoted(ft.init_from)}" in ft.train_args and "calibration.stability_margin=0.2" in ft.train_args
    assert eq.step_command(first, "audit", queue) == [
        sys.executable, str(REPO_ROOT / "scripts" / "audit_stability.py"), f"run={quoted(first.run_dir)}",
        "stability.device=cpu", *queue.common_overrides,
    ]  # fmt: skip
    train = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), *ft.train_args]
    assert eq.step_command(ft, "train", queue) == train
    assert len({job.config_hash for job in jobs}) == len(jobs)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"timout_h": 1.0}, "unknown keys"),
        ({"workers": "two"}, "workers: an integer"),
        ({"steps": ["audit", "train"]}, "train must be the first step"),
        ({"order": ["no_such_key"]}, "not a key of the train config"),
        ({"groups": [{"experiment": "e{no_such_field}", "overrides": synthetic()}]}, "template"),
        ({"groups": [{"experiment": "e", "overrides": synthetic()}] * 2}, "two jobs"),
        ({"groups": [{"experiment": "e", "overrides": synthetic(model="no_such_model")}]}, "no_such_model"),
        ({"groups": [{"experiment": "e", "overrides": {**synthetic(), "seed": []}}]}, "empty list"),
    ],
)  # fmt: skip
def test_errors_of_the_queue_file(tmp_path, change, message):
    groups = change.get("groups", [{"experiment": "e", "overrides": synthetic()}])
    queue_path = write_queue(tmp_path, groups, **{key: value for key, value in change.items() if key != "groups"})
    with pytest.raises(eq.QueueError, match=message):
        eq.load_jobs(eq.QueueConfig.from_file(queue_path))
    assert eq.main([str(queue_path)]) == 2 and not (tmp_path / "runs").exists()


def run_experiment_script():
    """``scripts/run_experiment.py`` as a module: its ``register_steps`` adds the band audits of D119."""
    spec = importlib.util.spec_from_file_location("run_experiment_script", REPO_ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_files_of_the_experiments(monkeypatch):
    """The queue files of section 4 of the contract: jobs, names, steps; one job per group composes."""
    monkeypatch.setattr(eq, "STEPS", dict(eq.STEPS))
    run_experiment_script().register_steps()  # audit_bands uses the steps of D119
    expected = {
        "preview": 5, "e1": 175, "e2_sweep": 120, "e2_chosen": 150, "e2_rollout": 1, "e4": 100, "e5": 40, "e5_penalised": 30,
        "e4_pilot": 4, "ngsim_idm": 5, "preview_gain": 3, "preview_existence": 2, "e2_existence": 15, "m5_laws": 40,
        "e2_lowfreq_pilot": 6, "e2_lowfreq": 15, "e4_rmax": 200,  # M7 (D110, D111)
        "e2_monotone": 3, "audit_bands": 150, "m8_temporal": 15,  # M8 (D117, D119, D120)
    }  # fmt: skip
    files = sorted((REPO_ROOT / "configs" / "queue").glob("*.yaml"))
    queues = {path.stem: eq.QueueConfig.from_file(path) for path in files}
    assert set(queues) == set(expected)
    specs = {name: eq.expand(queue) for name, queue in queues.items()}
    assert {name: len(s) for name, s in specs.items()} == expected
    experiments = {name: sorted({spec.experiment for spec in s}) for name, s in specs.items()}
    assert experiments["preview"] == ["m4_preview_jacobian", "m4_preview_linear_gain"]
    weights = ("0.01", "0.1", "1", "10")
    assert experiments["e2_sweep"] == [f"e2_{kind}_w{w}" for kind in ("gain", "jacobian") for w in weights]
    assert experiments["e2_rollout"] == ["e2_gain_spec_w1"]
    assert experiments["e4"] == ["e4_free_ft", "e4_stable", "e4_stable_ft"]
    assert experiments["e5"] == ["e5"]
    assert experiments["e5_penalised"] == ["e5_gain_w0.1", "e5_jacobian_w1"]  # chosen weights of the sweep (D85)
    for name in ("e1", "e2_chosen"):  # the transfer for seed 0 only
        assert all(("transfer" in spec.steps) == (spec.overrides["seed"] == 0) for spec in specs[name])
    assert all(spec.steps == ("train", "audit", "certificate") for spec in specs["e4"])
    ft = {spec.overrides["init_from"] for spec in specs["e4"] if spec.experiment.endswith("_ft")}
    assert "runs/e4_stable/follownet_highd/residual_idm/driver_fold4_seed3" in ft
    assert "runs/e1/follownet_highd/mlp/driver_fold0_seed1" in ft and len(ft) == 75
    # M7: the low-frequency arm of E2 (D110) and the residual-amplitude sweep of E4 (D111)
    assert experiments["e2_lowfreq_pilot"] == ["e2_combined_j0.1", "e2_combined_j1"]
    assert set(experiments["e2_lowfreq"]) <= {"e2_combined_j0.1", "e2_combined_j1"}  # the chosen weights
    for name in ("e2_lowfreq_pilot", "e2_lowfreq"):
        assert all(spec.steps == ("train", "audit", "platoon") for spec in specs[name])
        assert {spec.overrides["model"] for spec in specs[name]} == {"gru", "lstm", "perl"}
        assert all(spec.overrides["train.penalty.kind"] == "combined" and spec.overrides["train.penalty.weight"] == 0.1
                   and spec.overrides["train.penalty.every"] == 8 and spec.overrides["train.penalty.guard"] == 3.0
                   for spec in specs[name])  # fmt: skip
    assert sorted({spec.overrides["fold"] for spec in specs["e2_lowfreq"]}) == [0, 1, 2, 3, 4]
    rmax = [f"e4_{arm}_r{r}" for arm in ("stable", "stable_ft") for r in ("0.1", "0.2", "0.5")]
    assert experiments["e4_rmax"] == sorted([*rmax, "e4_free_r0.3", "e4_free_r0.3_ft"])
    assert all(spec.steps == ("train", "audit", "certificate") for spec in specs["e4_rmax"])
    counts = collections.Counter(spec.experiment for spec in specs["e4_rmax"])
    assert counts == {name: 25 for name in experiments["e4_rmax"]}
    for spec in specs["e4_rmax"]:
        o = spec.overrides
        run = f"follownet_highd/residual_idm/driver_fold{o['fold']}_seed{o['seed']}"
        if spec.experiment.startswith("e4_stable_r"):  # certified at its own amplitude, core with margin 0.2
            assert o["certificate.enforce"] is True and o["calibration.stability_margin"] == 0.2
            assert spec.experiment == f"e4_stable_r{o['model.r_max']:g}" and o["data"] == "follownet_highd"
        elif spec.experiment.startswith("e4_stable_ft_r"):  # fine-tuned from the run of the same amplitude
            assert o["init_from"] == f"runs/e4_stable_r{o['model.r_max']:g}/{run}" and o["data"] == "ngsim_i80"
            assert "certificate.enforce" not in o
        elif spec.experiment == "e4_free_r0.3":  # free core, no certificate
            assert o["certificate.enforce"] is False and o["model.r_max"] == 0.3
            assert "calibration.stability_margin" not in o
        else:
            assert spec.experiment == "e4_free_r0.3_ft" and o["model.r_max"] == 0.3
            assert o["init_from"] == f"runs/e4_free_r0.3/{run}"
    # M8: the RACER control (D117), the band audits of the E1 runs (D119), the temporal hold-out (D120)
    assert experiments["e2_monotone"] == ["e2_monotone_w1"]
    for spec in specs["e2_monotone"]:
        o = spec.overrides
        assert spec.steps == ("train", "audit", "platoon")
        assert (o["fold"], o["seed"], o["data"]) == (0, 0, "follownet_highd")
        assert o["train.penalty.kind"] == "monotone" and o["train.penalty.weight"] == 1.0
    assert sorted(spec.overrides["model"] for spec in specs["e2_monotone"]) == ["mlp", "pidl", "residual_idm"]
    assert experiments["audit_bands"] == ["e1"]
    assert all(spec.steps == ("audit_q01_99", "audit_q10_90") for spec in specs["audit_bands"])
    combinations = {(s.overrides["model"], s.overrides["fold"], s.overrides["seed"]) for s in specs["audit_bands"]}
    assert len(combinations) == 150
    assert {m for m, _, _ in combinations} == {"mlp", "pidl", "residual_idm", "gru", "lstm", "perl"}
    for step, (override,) in queues["audit_bands"].step_overrides.items():  # the quantiles give the output of the step
        key, value = override.split("=", 1)
        quantiles = tuple(yaml.safe_load(value))
        assert key == "band_quantiles" and quantiles == BAND_VARIANTS[step.removeprefix("audit_")]
        assert eq.STEPS[step] == ("scripts/audit_stability.py", band_output(quantiles))
    assert experiments["m8_temporal"] == ["m8_temporal_ft", "m8_temporal_stable_ft"]
    for spec in specs["m8_temporal"]:
        o = spec.overrides
        assert o["data"] == "ngsim_i80_p0" and o["seed"] == 0 and o["fold"] in range(5)
        source = f"follownet_highd/{o['model']}/driver_fold{o['fold']}_seed0"
        if spec.experiment == "m8_temporal_stable_ft":  # the certified hybrid of E4, as e4_stable_ft
            assert o["model"] == "residual_idm" and spec.steps == ("train", "audit", "certificate")
            assert o["init_from"] == f"runs/e4_stable/{source}" and o["model.r_max"] == 0.3
        else:
            assert o["model"] in ("mlp", "gru") and spec.steps == ("train", "audit")
            assert o["init_from"] == f"runs/e1/{source}"
    for name, queue in queues.items():  # the first value of every list composes
        groups = tuple({**g, "overrides": {k: v[:1] for k, v in g["overrides"].items()}} for g in queue.groups)
        jobs, _ = eq.load_jobs(eq.QueueConfig(**{**queue.__dict__, "groups": groups, "order": ()}))
        assert len(jobs) == len(groups), name


def test_fold_zero_of_the_low_frequency_arm_is_its_pilot():
    """configs/queue/e2_lowfreq.yaml repeats the pilot on five folds with the chosen weights (D110): its jobs of
    fold 0 are the pilot runs of the same weight (run directory and config hash), complete once the pilot
    has run, so the queue trains 12 further runs."""
    queues = REPO_ROOT / "configs" / "queue"
    pilot = eq.QueueConfig.from_file(queues / "e2_lowfreq_pilot.yaml")
    arm = eq.QueueConfig.from_file(queues / "e2_lowfreq.yaml")
    groups = tuple({**g, "overrides": {**g["overrides"], "fold": [0]}} for g in arm.groups)
    runs = {job.run_dir: job.config_hash for job in eq.load_jobs(pilot)[0]}
    jobs, _ = eq.load_jobs(eq.QueueConfig(**{**arm.__dict__, "groups": groups}))
    assert len(runs) == 6 and len(jobs) == 3 and all(runs.get(job.run_dir) == job.config_hash for job in jobs)


# --------------------------------------------------------------------------------------- running, restarts


def test_restart_skips_complete_steps_and_failures_do_not_stop_the_queue(tmp_path, fakes, capsys, monkeypatch):
    groups = [{"experiment": "q", "overrides": synthetic(folds=(0, 1, 2))}]
    path = write_queue(tmp_path, groups, workers=2, steps=["train", "audit", "platoon"], order=["fold"])
    monkeypatch.setenv("FAKE_FAIL", "audit:fold1")
    monkeypatch.setenv("FAKE_ERROR", "platoon:fold2")
    assert eq.main([str(path)]) == 0
    lines = queue_lines(capsys)
    states = {line.split()[3]: line.split()[2] for line in lines if line.startswith("JOB")}
    assert states == {f"q/{DATA}/mlp/fold{k}_seed0": state for k, state in enumerate(("done", "failed", "failed"))}
    assert lines[-1] == "QUEUE q: done 1, failed 2, skipped 0, timeout 0, blocked 0"
    assert any("audit: exit code 1" in line for line in lines)
    assert any("platoon: platoon.json holds an error: RuntimeError: broken" in line for line in lines)
    assert trace(tmp_path) == sorted(
        [f"train q/driver_fold{k}_seed0" for k in range(3)] + [f"audit q/driver_fold{k}_seed0" for k in range(3)]
        + ["platoon q/driver_fold0_seed0", "platoon q/driver_fold2_seed0"]
    )  # the failed audit ends its job: no platoon of fold 1  # fmt: skip

    status = read_json(tmp_path / "runs" / "q" / "_status.json")
    assert status["totals"] == {"n_jobs": 3, "done": 1, "failed": 2, "timeout": 0, "skipped": 0, "blocked": 0,
                                "running": 0, "pending": 0, "interrupted": 0}  # fmt: skip
    failed = status["jobs"][1]
    assert failed["state"] == "failed" and failed["failed_step"] == "audit" and failed["exit_code"] == 1
    assert failed["overrides"] == {"data": DATA, "model": "mlp", "fold": 1, "seed": 0}
    assert failed["run_dir"] == str(run_dir(tmp_path, "q", 1)) and failed["steps_run"] == ["train", "audit"]
    assert status["jobs"][2]["failed_step"] == "platoon" and status["jobs"][0]["wall_time_s"] > 0
    log = Path(failed["log"]).read_text(encoding="utf-8")
    assert "train: metrics.json missing" in log and "audit: stability.json missing" in log and "exit code 1" in log
    assert not list((tmp_path / "runs" / "q").glob("*.tmp"))  # the status is written through a rename

    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.delenv("FAKE_ERROR")
    assert eq.main([str(path)]) == 0  # restart: only what is missing runs
    lines = queue_lines(capsys)
    assert lines[-1] == "QUEUE q: done 2, failed 0, skipped 1, timeout 0, blocked 0"
    fold1, fold2 = "q/driver_fold1_seed0", "q/driver_fold2_seed0"
    assert trace(tmp_path) == [f"audit {fold1}", f"platoon {fold1}", f"platoon {fold2}"]
    assert eq.main([str(path)]) == 0 and trace(tmp_path) == []
    assert queue_lines(capsys)[-1] == "QUEUE q: done 0, failed 0, skipped 3, timeout 0, blocked 0"

    # a new train config (other hash) trains again, and the later steps follow the new model
    new_lr = {"train": ["train.lr=0.002"]}
    path = write_queue(tmp_path, groups, steps=["train", "audit", "platoon"], step_overrides=new_lr)
    assert eq.main([str(path), "--only", "fold1"]) == 0
    assert trace(tmp_path) == [f"audit {fold1}", f"platoon {fold1}", f"train {fold1}"]
    metrics = read_json(run_dir(tmp_path, "q", 1) / "metrics.json")
    assert metrics["config"]["train"]["lr"] == 0.002 and metrics["config_hash"] == config_hash(metrics["config"])


def test_timeout_ends_the_process_tree_and_the_queue_goes_on(tmp_path, fakes, capsys, monkeypatch):
    for fold in (0, 1):
        run_dir(tmp_path, "q", fold).mkdir(parents=True)
        (run_dir(tmp_path, "q", fold) / "model.pt").write_bytes(b"fake")
    path = write_queue(tmp_path, [{"experiment": "q", "overrides": synthetic(folds=(0, 1))}], steps=["audit"],
                       timeout_h=6.0 / 3600, order=["fold"])  # fmt: skip
    monkeypatch.setenv("FAKE_HANG", "audit:fold0")
    t_start = time.monotonic()
    assert eq.main([str(path)]) == 0
    assert time.monotonic() - t_start < 60
    lines = queue_lines(capsys)
    assert lines[0].startswith(f"JOB 1/2 timeout q/{DATA}/mlp/fold0_seed0") and "audit: no result after" in lines[0]
    assert lines[1].startswith(f"JOB 2/2 done q/{DATA}/mlp/fold1_seed0")
    assert lines[-1] == "QUEUE q: done 1, failed 0, skipped 0, timeout 1, blocked 0"
    pids = [int(pid) for pid in (tmp_path / "pids.txt").read_text(encoding="utf-8").split()]
    assert wait_until(lambda: not any(alive(pid) for pid in pids), timeout=10.0), "a step process survived its timeout"
    record = read_json(tmp_path / "runs" / "q" / "_status.json")["jobs"][0]
    assert record["state"] == "timeout" and record["failed_step"] == "audit" and record["exit_code"] is None
    assert "ended after the timeout" in Path(record["log"]).read_text(encoding="utf-8")


def test_init_from_waits_for_its_source_or_blocks(tmp_path, fakes, capsys):
    source, nowhere = run_dir(tmp_path, "src", 0).as_posix(), f"{tmp_path.as_posix()}/nowhere"
    groups = [
        {"experiment": "ft", "overrides": synthetic(folds=(0,)), "fixed": {"init_from": source}},
        {"experiment": "ft", "overrides": synthetic(folds=(1,)), "fixed": {"init_from": nowhere}},
        {"experiment": "src", "overrides": synthetic(folds=(0,))},
    ]
    path = write_queue(tmp_path, groups, workers=2, order=["experiment", "fold"])
    assert eq.main([str(path), "--dry-run"]) == 0
    lines = queue_lines(capsys)
    assert lines[0] == (
        f"DRY 1/3 run train after init_from (no model.pt yet) ft/{DATA}/mlp/fold0_seed0  data={DATA} model=mlp fold=0 "
        f"seed=0 init_from={quoted(Path(source))} experiment=ft"
    )  # fmt: skip
    assert lines[-1] == "QUEUE q (dry run): 3 jobs, to run 1, complete 0, waiting for init_from 2"
    assert not (tmp_path / "runs").exists() and trace(tmp_path) == []  # a dry run changes nothing

    assert eq.main([str(path)]) == 0
    lines = queue_lines(capsys)
    assert [line.split()[1:4] for line in lines[:-1]] == [
        ["2/3", "blocked", f"ft/{DATA}/mlp/fold1_seed0"], ["3/3", "done", f"src/{DATA}/mlp/fold0_seed0"],
        ["1/3", "done", f"ft/{DATA}/mlp/fold0_seed0"],
    ]  # the first job waited for its source, which was running when the job came round again  # fmt: skip
    assert "nowhere has no model.pt" in lines[0]
    assert lines[-1] == "QUEUE q: done 2, failed 0, skipped 0, timeout 0, blocked 1"
    tuned, started = run_dir(tmp_path, "ft", 0) / "metrics.json", Path(source) / "model.pt"
    assert tuned.stat().st_mtime >= started.stat().st_mtime


def test_max_jobs_only_memory_and_lock(tmp_path, fakes, capsys, monkeypatch):
    for fold in (0, 1, 2):
        run_dir(tmp_path, "q", fold).mkdir(parents=True)
        (run_dir(tmp_path, "q", fold) / "model.pt").write_bytes(b"fake")
    path = write_queue(tmp_path, [{"experiment": "q", "overrides": synthetic(folds=(0, 1, 2))}], steps=["audit"],
                       order=["fold"], min_free_ram_gb=4.0)  # fmt: skip
    free = iter([1.0, 2.0])  # two looks below the limit, then enough
    monkeypatch.setattr(eq, "free_memory_gb", lambda: next(free, 64.0))
    monkeypatch.setattr(eq, "MEMORY_POLL_S", 0.05)
    assert eq.main([str(path), "--max-jobs", "1"]) == 0
    lines = queue_lines(capsys)
    end = "QUEUE q: done 1, failed 0, skipped 0, timeout 0, blocked 0, not started 2"
    assert lines == ["WAIT free memory 1.0 GB < 4 GB", lines[1], end]
    assert lines[1].startswith(f"JOB 1/3 done q/{DATA}/mlp/fold0_seed0")
    assert trace(tmp_path) == ["audit q/driver_fold0_seed0"]
    status = read_json(tmp_path / "runs" / "q" / "_status.json")
    assert [job["state"] for job in status["jobs"]] == ["done", "pending", "pending"]

    assert eq.main([str(path), "--only", "fold2"]) == 0
    assert queue_lines(capsys)[0].startswith(f"JOB 1/1 done q/{DATA}/mlp/fold2_seed0")
    assert trace(tmp_path) == ["audit q/driver_fold2_seed0"]

    with eq.QueueLock(tmp_path / "runs" / "q" / "_queue.lock"):  # a second queue on the same file refuses to start
        assert eq.main([str(path)]) == 3
    assert "already running" in queue_lines(capsys)[-1] and trace(tmp_path) == []


# ------------------------------------------------------------------------------------ as a separate process


def run_queue_script(tmp_path: Path, path: Path, *args: str) -> subprocess.CompletedProcess:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), str(path), *args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


@pytest.mark.skipif(os.name != "nt", reason="the job object that ends the steps with the queue is Windows only")
def test_steps_end_with_a_killed_queue_and_a_restart_completes(tmp_path, fakes):
    """The queue binds its steps to its own life: killing the queue ends a hanging step, and a restart
    runs the step again. The fake steps come through a queue file whose scripts are the fakes."""
    run = run_dir(tmp_path, "q", 0)
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"fake")
    path = write_queue(tmp_path, [{"experiment": "q", "overrides": synthetic()}], steps=["audit"])
    wrapper = tmp_path / "queue_with_fakes.py"
    wrapper.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from cf_stability.eval import experiment_queue as eq\n"
        f"eq.STEPS['audit'] = ({str(fakes / 'fake_audit.py')!r}, 'stability.json')\n"
        "eq.bind_children_to_this_process()\n"
        "sys.exit(eq.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "FAKE_HANG": "audit:fold0"}
    command = [sys.executable, str(wrapper), str(path)]
    queue = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=subprocess.DEVNULL)
    try:
        assert wait_until(lambda: (tmp_path / "pids.txt").exists())
        status = read_json(tmp_path / "runs" / "q" / "_status.json")
        steps = [int(pid) for pid in (tmp_path / "pids.txt").read_text(encoding="utf-8").split()]
        assert all(alive(pid) for pid in steps) and status["jobs"][0]["state"] == "running"
        # the queue's interpreter only (the pid of the status), not its tree
        subprocess.run(["taskkill", "/F", "/PID", str(status["pid"])], capture_output=True, check=True)
        assert wait_until(lambda: not any(alive(pid) for pid in steps), timeout=15.0), "a step outlived its queue"
    finally:
        eq.kill_tree(queue)
    del env["FAKE_HANG"]
    subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True)
    assert read_json(run / "stability.json")["step"] == "audit"
    assert read_json(tmp_path / "runs" / "q" / "_status.json")["totals"]["done"] == 1


def test_queue_script_end_to_end_with_the_real_steps(tmp_path):
    """train, audit and platoon of a tiny MLP; a job whose training fails; a restart skips everything."""
    events = EventSet(tiny_event(i) for i in range(10))
    events.to_parquet(tmp_path / "events" / DATA)
    write_splits(events.events_frame(), DATA, tmp_path / "splits")
    train = [
        f"paths.events_root={quoted(tmp_path / 'events')}", f"paths.splits_root={quoted(tmp_path / 'splits')}",
        f"paths.calibration_root={quoted(tmp_path / 'calibration')}", "train.max_epochs=2", "train.steps_per_epoch=5",
        "train.device=cpu",
    ]  # fmt: skip
    good = {"experiment": "e_w{train.penalty.weight:g}", "overrides": synthetic(**{"train.penalty.weight": [0.5]})}
    broken = {"experiment": "broken", "steps": ["train"], "overrides": synthetic()}
    broken["fixed"] = {"train.penalty.kind": "bogus"}  # train.py stops at the unknown penalty
    path = write_queue(
        tmp_path, [good, broken], name="real", workers=2, steps=["train", "audit", "platoon"],
        step_overrides={"train": train, "audit": list(SMALL_AUDIT), "platoon": list(SMALL_PLATOON)},
    )  # fmt: skip
    out = run_queue_script(tmp_path, path).stdout.splitlines()
    assert sorted(line.split()[2:4] for line in out if line.startswith("JOB")) == [
        ["done", f"e_w0.5/{DATA}/mlp/fold0_seed0"], ["failed", f"broken/{DATA}/mlp/fold0_seed0"],
    ]  # fmt: skip
    assert out[-1] == "QUEUE real: done 1, failed 1, skipped 0, timeout 0, blocked 0"
    run = run_dir(tmp_path, "e_w0.5", 0)
    files = {"metrics.json", "model.pt", "test_events.parquet", "stability.json", "platoon.json"}
    assert {p.name for p in run.iterdir()} == files
    assert read_json(run / "metrics.json")["config"]["experiment"] == "e_w0.5"  # the hash matched: the job is done
    assert read_json(run / "stability.json")["audit"]["summary"]["n_grid"] == 2
    assert read_json(run / "platoon.json")["summary"]["n_profiles"] == 1
    status = read_json(tmp_path / "runs" / "real" / "_status.json")
    failed = next(job for job in status["jobs"] if job["experiment"] == "broken")
    assert failed["failed_step"] == "train" and failed["exit_code"] != 0
    assert "penalty kind" in Path(failed["log"]).read_text(encoding="utf-8")  # the traceback of the step is in its log

    mtimes = {p.name: p.stat().st_mtime for p in run.iterdir()}
    out = run_queue_script(tmp_path, path, "--only", "e_w0.5").stdout.splitlines()
    assert out[-2].startswith(f"JOB 1/1 skipped e_w0.5/{DATA}/mlp/fold0_seed0")
    assert out[-1] == "QUEUE real: done 0, failed 0, skipped 1, timeout 0, blocked 0"
    assert {p.name: p.stat().st_mtime for p in run.iterdir()} == mtimes


def test_band_audit_steps_through_the_queue_script(tmp_path):
    """D119: the steps audit_q01_99 and audit_q10_90 that scripts/run_experiment.py registers run the audit with
    the quantiles of their step overrides on an existing run (no training) and are complete by their own output
    files; stability.json is not touched and a restart skips the job."""
    events = EventSet(tiny_event(i) for i in range(10))
    events.to_parquet(tmp_path / "events" / DATA)
    write_splits(events.events_frame(), DATA, tmp_path / "splits")
    run = run_dir(tmp_path, "bands", 0, model="idm")
    run.mkdir(parents=True)
    save_model(IDM({"v0": 30.0, "T": 1.3, "s0": 2.0, "a": 1.2, "b": 1.8}), run / "model.pt")
    settings = {"dv_max": 0.5, "a_max": 0.3, "half_width": 0.5, "quantiles": [0.05, 0.5, 0.95], "min_samples": 20}
    paths = {"events_root": (tmp_path / "events").as_posix(), "splits_root": (tmp_path / "splits").as_posix()}
    config = {"data": {"name": DATA}, "split": "driver", "fold": 0, "seed": 0, "max_train_events": None, "paths": paths}
    context = training_context(list(events), 0, BandConfig.from_mapping(settings))
    write_json(run / "metrics.json", {"config": config, "train_config": {"band": settings}, "context": context})
    (run / "stability.json").write_text("{}", encoding="utf-8")
    overrides = {step: [*SMALL_AUDIT, f"band_quantiles=[{','.join(map(str, q))}]"] for step, q in (
        ("audit_q01_99", BAND_VARIANTS["q01_99"]), ("audit_q10_90", BAND_VARIANTS["q10_90"])
    )}  # fmt: skip
    path = write_queue(
        tmp_path, [{"experiment": "bands", "overrides": synthetic(model="idm")}], name="bands",
        steps=["audit_q01_99", "audit_q10_90"], step_overrides=overrides,
    )  # fmt: skip
    out = run_queue_script(tmp_path, path).stdout.splitlines()
    assert out[-1] == "QUEUE bands: done 1, failed 0, skipped 0, timeout 0, blocked 0", out
    for name, quantiles in BAND_VARIANTS.items():
        result = read_json(run / f"stability_{name}.json")
        assert result["band_override"]["quantiles"] == list(quantiles) and result["audit"]["summary"]["n_grid"] == 2
    assert (run / "stability.json").read_text(encoding="utf-8") == "{}"
    out = run_queue_script(tmp_path, path).stdout.splitlines()
    assert out[-1] == "QUEUE bands: done 0, failed 0, skipped 1, timeout 0, blocked 0"
