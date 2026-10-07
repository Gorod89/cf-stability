"""Queue of experiment jobs, every step in a separate process (docs/m4_contract.md, section 2.1; D82).

A queue file (plain YAML, read here and not by hydra) lists groups of jobs. A job is the chain of
its steps (``train``, ``audit``, ``platoon``, ``transfer``, ``certificate``: the scripts of section 0
of the contract) for one combination of overrides::

    name: e2_sweep                   # directory of the logs and of the status under paths.runs_root
    workers: 2                       # jobs at once
    timeout_h: 4.0                   # per step; the step's process tree is ended after it
    min_free_ram_gb: 6.0             # no new job is started below it
    order: [seed, fold, model]       # sort keys of the jobs (keys of the train config), first key slowest
    steps: [train, audit, platoon]   # default steps of a job
    common_overrides: []             # appended to the command of every step
    step_overrides: {audit: [...]}   # appended to the command of one step (train: also to its composed config)
    groups:
      - experiment: "e2_jacobian_w{train.penalty.weight:g}"
        steps: [train, audit, platoon, transfer]       # optional, replaces the default
        overrides: {model: [mlp, pidl], fold: [0, 1], train.penalty.weight: [0.1, 1.0]}   # Cartesian product
        fixed: {init_from: "runs/e1/follownet_highd/{model}/driver_fold{fold}_seed{seed}"}  # optional

``experiment`` and the string values of ``fixed`` are templates of ``str.format`` over the values of
the combination, dotted keys included (``{train.penalty.weight:g}``). ``train`` is called with the
overrides of the job, every later step with ``run=<run directory>``.

The state lives in the run directories: a step is complete when its output file exists, is not
older than ``model.pt`` and holds no ``error``; ``train`` is complete when ``metrics.json`` holds the
hash of the config composed from the job's overrides (and is not older than the ``model.pt`` of
``init_from``). Complete steps are skipped, so the queue can be ended at any moment and restarted;
``_status.json`` is a report only.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import itertools
import json
import os
import queue as queue_module
import re
import shutil
import signal
import string
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from cf_stability.utils import REPO_ROOT, config_hash, json_default, read_json, resolve_path, to_plain

STEPS: dict[str, tuple[str, str]] = {  # step -> (script relative to the repository root, output file of a run)
    "train": ("scripts/train.py", "metrics.json"),
    "audit": ("scripts/audit_stability.py", "stability.json"),
    "platoon": ("scripts/platoon_test.py", "platoon.json"),
    "transfer": ("scripts/evaluate_transfer.py", "transfer.json"),
    "certificate": ("scripts/certificate.py", "certificate.json"),
}
QUEUE_KEYS = frozenset({
    "name", "workers", "timeout_h", "min_free_ram_gb", "order", "steps", "groups", "common_overrides", "step_overrides",
})  # fmt: skip
GROUP_KEYS = frozenset({"experiment", "steps", "overrides", "fixed"})
STATES = ("done", "failed", "timeout", "skipped", "blocked", "running", "pending", "interrupted")
MEMORY_POLL_S = 30.0  # seconds between two looks at the free memory while it is short
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_WORDS = frozenset({"true", "false", "null", "nan", "inf"})  # identifiers that hydra reads as other types


class QueueError(ValueError):
    """An error of the queue file; nothing is run."""


class QueueLocked(RuntimeError):
    """Another process runs the same queue."""


# ----------------------------------------------------------------------------------------------- queue file


def _steps(value: Any, where: str) -> tuple[str, ...]:
    steps = [value] if isinstance(value, str) else list(value or [])
    unknown = [s for s in steps if s not in STEPS]
    if not steps or unknown or len(set(steps)) != len(steps):
        raise QueueError(f"{where}: distinct steps of {list(STEPS)}, got {value!r}")
    if "train" in steps and steps[0] != "train":
        raise QueueError(f"{where}: train must be the first step (the later steps use its model.pt)")
    return tuple(steps)


def _overrides(value: Any, where: str) -> tuple[str, ...]:
    items = list(value or [])
    if not all(isinstance(item, str) and "=" in item for item in items):
        raise QueueError(f"{where}: a list of hydra overrides key=value, got {value!r}")
    return tuple(items)


@dataclasses.dataclass(frozen=True)
class QueueConfig:
    """Content of a queue file (see the module docstring)."""

    name: str
    groups: tuple[dict[str, Any], ...]
    workers: int = 1
    timeout_h: float | None = None
    min_free_ram_gb: float = 0.0
    order: tuple[str, ...] = ()
    steps: tuple[str, ...] = ("train",)
    common_overrides: tuple[str, ...] = ()
    step_overrides: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any) -> QueueConfig:
        if not isinstance(raw, Mapping):
            raise QueueError("a queue file holds a mapping")
        unknown = set(raw) - QUEUE_KEYS
        if unknown:
            raise QueueError(f"unknown keys {sorted(unknown)} (known: {sorted(QUEUE_KEYS)})")
        name = raw.get("name")
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name.replace(".", "_")):
            raise QueueError(f"name: a plain name (letters, digits, _ - .), got {name!r}")
        steps = _steps(raw.get("steps", ["train"]), "steps")
        groups = []
        for k, group in enumerate(raw.get("groups") or []):
            where = f"groups[{k}]"
            if not isinstance(group, Mapping):
                raise QueueError(f"{where}: a mapping")
            if set(group) - GROUP_KEYS:
                unknown = sorted(set(group) - GROUP_KEYS)
                raise QueueError(f"{where}: unknown keys {unknown} (known: {sorted(GROUP_KEYS)})")
            if not isinstance(group.get("experiment"), str) or not group["experiment"]:
                raise QueueError(f"{where}.experiment: the name (or template) of the experiment")
            overrides, fixed = group.get("overrides"), group.get("fixed") or {}
            if not isinstance(overrides, Mapping) or not overrides or not isinstance(fixed, Mapping):
                raise QueueError(f"{where}: overrides (a mapping of keys to values or lists) and fixed (a mapping)")
            lists = {str(key): list(v) if isinstance(v, (list, tuple)) else [v] for key, v in overrides.items()}
            if any(not v for v in lists.values()):
                raise QueueError(f"{where}.overrides: an empty list")
            if set(lists) & set(fixed) or "experiment" in set(lists) | set(fixed):
                raise QueueError(f"{where}: a key in both overrides and fixed, or `experiment` outside `experiment`")
            groups.append(
                {
                    "experiment": group["experiment"],
                    "steps": _steps(group["steps"], f"{where}.steps") if "steps" in group else steps,
                    "overrides": lists,
                    "fixed": {str(key): value for key, value in fixed.items()},
                }
            )
        if not groups:
            raise QueueError("groups: at least one group")
        step_overrides = raw.get("step_overrides") or {}
        if not isinstance(step_overrides, Mapping) or set(step_overrides) - set(STEPS):
            raise QueueError(f"step_overrides: a mapping from steps of {list(STEPS)} to lists of overrides")
        try:
            workers = int(raw.get("workers", 1))
            timeout_h = None if raw.get("timeout_h") is None else float(raw["timeout_h"])
            min_free_ram_gb = float(raw.get("min_free_ram_gb") or 0.0)
        except (TypeError, ValueError):
            workers = timeout_h = min_free_ram_gb = None
        if workers is None or workers < 1 or (timeout_h is not None and not timeout_h > 0.0) or min_free_ram_gb is None:
            raise QueueError("workers: an integer >= 1, timeout_h: hours > 0 or null, min_free_ram_gb: GB")
        order = [raw["order"]] if isinstance(raw.get("order"), str) else list(raw.get("order") or [])
        return cls(
            name=name,
            groups=tuple(groups),
            workers=workers,
            timeout_h=timeout_h,
            min_free_ram_gb=min_free_ram_gb,
            order=tuple(str(key) for key in order),
            steps=steps,
            common_overrides=_overrides(raw.get("common_overrides"), "common_overrides"),
            step_overrides={step: _overrides(v, f"step_overrides.{step}") for step, v in step_overrides.items()},
        )

    @classmethod
    def from_file(cls, path: str | Path) -> QueueConfig:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise QueueError(f"{path}: {exc}") from None
        return cls.from_mapping(raw)


class _Template(string.Formatter):
    """``str.format`` whose fields are whole (dotted) names: ``{train.penalty.weight:g}``."""

    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> tuple[Any, str]:
        if field_name not in kwargs:
            raise KeyError(f"unknown field {{{field_name}}} (fields: {', '.join(kwargs)})")
        return kwargs[field_name], field_name


def fill(template: Any, fields: Mapping[str, Any]) -> Any:
    """``template`` filled with ``fields`` when it is a string, else unchanged."""
    return _Template().vformat(template, (), fields) if isinstance(template, str) else template


def override_value(value: Any) -> str:
    """``value`` in the hydra override grammar. Strings other than plain identifiers (paths, names
    with dots, non-ASCII characters) are quoted; paths are written with forward slashes."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str):
        raise QueueError(f"override values are numbers, booleans, null or strings, got {value!r}")
    if _IDENTIFIER.fullmatch(value) and value.lower() not in _WORDS:
        return value
    if "'" in value or "\\" in value:
        raise QueueError(f"quotes and backslashes are not supported in override values (use /): {value!r}")
    return f"'{value}'"


# ----------------------------------------------------------------------------------------------------- jobs


@dataclasses.dataclass(frozen=True)
class JobSpec:
    """One combination of a group, before the config is composed."""

    experiment: str
    steps: tuple[str, ...]
    overrides: dict[str, Any]  # values of the combination and of `fixed`, in command-line order

    def arguments(self) -> list[str]:
        items = [f"{key}={override_value(value)}" for key, value in self.overrides.items()]
        return [*items, f"experiment={override_value(self.experiment)}"]


@dataclasses.dataclass(frozen=True)
class Job:
    """A job with the run directory and config hash of its composed train config."""

    spec: JobSpec
    train_args: tuple[str, ...]  # command line of the train step
    run_dir: Path
    config_hash: str
    data: str  # data.name
    model: str  # choice of the config group model
    split: str
    fold: int
    seed: int
    init_from: Path | None
    sort_key: tuple[Any, ...]

    @property
    def experiment(self) -> str:
        return self.spec.experiment

    @property
    def steps(self) -> tuple[str, ...]:
        return self.spec.steps

    @property
    def overrides(self) -> dict[str, Any]:
        return self.spec.overrides

    @property
    def label(self) -> str:
        return f"{self.experiment}/{self.data}/{self.model}/fold{self.fold}_seed{self.seed}"

    @property
    def log_name(self) -> str:
        return f"{self.experiment}__{self.data}__{self.model}__{self.split}_fold{self.fold}_seed{self.seed}.log"


def expand(queue: QueueConfig) -> list[JobSpec]:
    """Jobs of the groups in file order; per group the Cartesian product with the first key slowest."""
    specs = []
    for k, group in enumerate(queue.groups):
        keys = list(group["overrides"])
        for values in itertools.product(*group["overrides"].values()):
            fields = dict(zip(keys, values))
            try:
                fixed = {key: fill(value, fields) for key, value in group["fixed"].items()}
                experiment = fill(group["experiment"], fields)
            except (KeyError, ValueError, IndexError) as exc:
                raise QueueError(f"groups[{k}]: template: {exc}") from None
            specs.append(JobSpec(experiment, group["steps"], {**fields, **fixed}))
    return specs


def _sort_value(plain: Mapping[str, Any], choices: Mapping[str, Any], key: str) -> tuple[int, Any]:
    if key in choices:
        value = choices[key]
    else:
        value = plain
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise QueueError(f"order: {key!r} is not a key of the train config")
            value = value[part]
    if value is None:
        return (0, 0.0)
    if isinstance(value, (bool, int, float)):
        return (1, float(value))
    return (2, str(value))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def load_jobs(queue: QueueConfig) -> tuple[list[Job], Path]:
    """Jobs of the queue in their order, and the runs root.

    Every job's train config is composed with ``hydra.compose`` (config directory ``configs``, config
    ``train``) from the overrides of the job, the train overrides and the common overrides: the run
    directory and the config hash follow from it as in ``scripts/train.py``.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    def composed(args: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = compose(config_name="train", overrides=list(args), return_hydra_config=True)
        choices = dict(cfg.hydra.runtime.choices)
        with open_dict(cfg):
            del cfg["hydra"]  # as hydra.main hands the config to the task function
        return to_plain(cfg), choices

    base = (*queue.step_overrides.get("train", ()), *queue.common_overrides)
    specs, jobs = expand(queue), []
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3", job_name="run_experiment"):
        try:
            runs_root = resolve_path(composed(base)[0]["paths"]["runs_root"])
        except Exception as exc:
            raise QueueError(f"train overrides {' '.join(base)}: {type(exc).__name__}: {exc}") from None
        for spec in specs:
            args = (*spec.arguments(), *base)
            try:
                plain, choices = composed(args)
            except Exception as exc:
                raise QueueError(f"job {' '.join(spec.arguments())}: {type(exc).__name__}: {exc}") from None
            run_dir = (
                resolve_path(plain["paths"]["runs_root"]) / str(plain["experiment"]) / str(plain["data"]["name"])
                / str(choices["model"]) / f"{plain['split']}_fold{plain['fold']}_seed{plain['seed']}"
            )  # fmt: skip
            init_from = plain.get("init_from")
            jobs.append(
                Job(
                    spec=spec,
                    train_args=args,
                    run_dir=run_dir,
                    config_hash=config_hash(plain),
                    data=str(plain["data"]["name"]),
                    model=str(choices["model"]),
                    split=str(plain["split"]),
                    fold=plain["fold"],
                    seed=plain["seed"],
                    init_from=None if init_from is None else resolve_path(init_from),
                    sort_key=tuple(_sort_value(plain, choices, key) for key in queue.order),
                )
            )
    jobs.sort(key=lambda job: job.sort_key)  # stable: ties keep the order of the file
    seen: dict[str, Job] = {}
    for job in jobs:
        other = seen.setdefault(_path_key(job.run_dir), job)
        if other is not job:
            first, second = (" ".join(j.spec.arguments()) for j in (other, job))
            raise QueueError(f"two jobs write {job.run_dir}: {first} and {second}")
    return jobs, runs_root


# ------------------------------------------------------------------------------------------ state of steps


def _read(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError, UnicodeDecodeError):  # a file cut short by a kill is not a result
        return None


def output_problem(job: Job, step: str) -> str | None:
    """None when ``step`` of ``job`` is complete, else what is missing."""
    name = STEPS[step][1]
    output, model = job.run_dir / name, job.run_dir / "model.pt"
    if not output.exists():
        return f"{name} missing"
    if not model.exists():
        return "model.pt missing"
    payload = _read(output)
    if not isinstance(payload, dict):
        return f"{name} unreadable"
    if step == "train":
        if payload.get("config_hash") != job.config_hash:
            return f"config hash {payload.get('config_hash')} of {name} differs from {job.config_hash}"
        source = None if job.init_from is None else job.init_from / "model.pt"
        if source is not None and source.exists() and output.stat().st_mtime < source.stat().st_mtime:
            return f"{name} older than {source}"
        return None
    if output.stat().st_mtime < model.stat().st_mtime:
        return f"{name} older than model.pt"
    if payload.get("error"):
        return f"{name} holds an error: {str(payload['error']).splitlines()[0]}"
    return None


def steps_to_run(job: Job) -> list[str]:
    """Steps of ``job`` that are not complete; after a new training every later step runs again."""
    todo: list[str] = []
    for step in job.steps:
        if todo[:1] == ["train"] or output_problem(job, step) is not None:
            todo.append(step)
    return todo


def step_command(job: Job, step: str, queue: QueueConfig) -> list[str]:
    """Command line of ``step``: the interpreter of the queue, the script and its overrides."""
    script = str(REPO_ROOT / STEPS[step][0])
    if step == "train":
        return [sys.executable, script, *job.train_args]
    return [
        sys.executable, script, f"run={override_value(job.run_dir)}", *queue.step_overrides.get(step, ()),
        *queue.common_overrides,
    ]  # fmt: skip


# ------------------------------------------------------------------------------------------------ processes


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        *((name, ctypes.c_ulonglong) for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual",
            "ullAvailVirtual", "ullAvailExtendedVirtual",
        )),
    ]  # fmt: skip


def free_memory_gb() -> float | None:
    """Free physical memory in GB (``GlobalMemoryStatusEx`` on Windows, ``sysconf`` elsewhere), None if unknown."""
    if os.name == "nt":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / 2**30
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, OSError, ValueError):
        return None


def kill_tree(proc: subprocess.Popen) -> None:
    """End ``proc`` together with the processes it started (the venv launcher starts the interpreter as
    its child): ``taskkill /T /F`` of the step's own process on Windows, its process group elsewhere."""
    if proc.poll() is None:
        if os.name == "nt":
            system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
            taskkill = str(system) if system.exists() else shutil.which("taskkill") or "taskkill"
            subprocess.run(
                [taskkill, "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=120, check=False,
            )  # fmt: skip
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _print(line: str) -> None:
    print(line, flush=True)  # the output of a queue is usually redirected to a file that is watched


def run_step(
    command: Sequence[str], log_path: Path, timeout_s: float | None, note: str = "",
    register: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[int | None, bool]:  # fmt: skip
    """Run ``command`` from the repository root with its output appended to ``log_path``;
    ``(exit code, timed out)``, the exit code None after a timeout."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1"}
    extra: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}  # own group: killpg
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {_now()}  {note}\n=== {subprocess.list2cmdline(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **extra
        )
        if register:
            register(proc)
        try:
            code, timed_out = proc.wait(timeout=timeout_s), False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            code, timed_out = None, True
        finally:
            if register:
                register(None)
        end = f"ended after the timeout of {timeout_s / 3600:g} h" if timed_out else f"exit code {code}"
        log.write(f"=== {_now()}  {end} ({time.perf_counter() - t_start:.0f} s)\n")
    return code, timed_out


def run_job(
    job: Job, queue: QueueConfig, log_path: Path,
    register: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Run the steps of ``job`` that are not complete, in order; the first failing step ends the job."""
    t_start = time.perf_counter()
    timeout_s = None if queue.timeout_h is None else 3600.0 * queue.timeout_h
    result: dict[str, Any] = {"state": "done", "exit_code": None, "failed_step": None, "error": None, "steps_run": []}
    retrain = False
    for step in job.steps:
        problem = output_problem(job, step)
        if problem is None and not retrain:
            continue
        retrain = retrain or step == "train"
        result["steps_run"].append(step)
        note = f"{step}: {problem or 'new model'}"
        code, timed_out = run_step(step_command(job, step, queue), log_path, timeout_s, note, register)
        result["exit_code"] = code
        if timed_out:
            result.update(state="timeout", failed_step=step, error=f"no result after {queue.timeout_h:g} h")
            break
        problem = f"exit code {code}" if code != 0 else output_problem(job, step)
        if problem is not None:
            result.update(state="failed", failed_step=step, error=problem)
            break
    result["wall_time_s"] = time.perf_counter() - t_start
    return result


def bind_children_to_this_process() -> bool:
    """Windows: put this process into a job object that ends every process in it when the last handle
    to it closes, i.e. when this process ends in whatever way. The steps started afterwards belong to
    it too, so a killed queue leaves no step behind that a restarted queue would run a second time.
    Best effort: False when it is not possible (other systems, or the call fails)."""
    if os.name != "nt":
        return False
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]  # fmt: skip

    class Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]  # fmt: skip

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return False
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    extended_limit_information = 9  # JOBOBJECTINFOCLASS
    size = ctypes.sizeof(info)
    if not kernel32.SetInformationJobObject(handle, extended_limit_information, ctypes.byref(info), size):
        return False
    # the handle stays open for the life of the process on purpose: closing it would end the steps
    return bool(kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()))


class QueueLock:
    """Exclusive lock of one queue for the life of the process; the system releases it however the
    process ends, so a second queue on the same file refuses to start instead of running the jobs twice."""

    def __init__(self, path: Path) -> None:
        self.path, self.file = path, None

    def __enter__(self) -> QueueLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise QueueLocked(f"the queue is already running (lock {self.path})") from None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)  # at once: the system may release it late
        except OSError:
            pass
        self.file.close()


def write_status(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON through a temporary file and a rename, so that the file is always whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader holds the file open (Windows): the report can wait
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ the queue


class QueueRunner:
    """Runs the jobs of a queue with up to ``workers`` at once (one thread per running job, every step
    in its own process). Only the main thread touches the records; the threads hand back results."""

    def __init__(
        self, queue: QueueConfig, jobs: Sequence[Job], queue_dir: Path, *, workers: int | None = None,
        max_jobs: int | None = None, source: str | Path | None = None, out: Callable[[str], None] = _print,
    ) -> None:  # fmt: skip
        self.queue, self.jobs, self.out = queue, list(jobs), out
        self.workers = queue.workers if workers is None else max(1, workers)
        self.max_jobs, self.source = max_jobs, source
        self.log_dir, self.status_path = queue_dir / "_logs", queue_dir / "_status.json"
        self.records = [self._record(k, job) for k, job in enumerate(self.jobs)]
        self.running: dict[int, threading.Thread] = {}
        self.finished: queue_module.Queue[tuple[int, dict[str, Any]]] = queue_module.Queue()
        self.processes: dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.started, self.waiting_for_memory = 0, False

    def _record(self, k: int, job: Job) -> dict[str, Any]:
        return {
            "index": k + 1, "label": job.label, "experiment": job.experiment, "overrides": job.overrides,
            "run_dir": str(job.run_dir), "steps": list(job.steps), "state": "pending", "exit_code": None,
            "failed_step": None, "error": None, "steps_run": [], "wall_time_s": None, "started": None, "ended": None,
            "log": str(self.log_dir / job.log_name),
        }  # fmt: skip

    def totals(self) -> dict[str, int]:
        counts = {state: 0 for state in STATES}
        for record in self.records:
            counts[record["state"]] += 1
        return {"n_jobs": len(self.records), **counts}

    def write_status(self) -> None:
        write_status(
            self.status_path,
            {
                "name": self.queue.name, "queue_file": None if self.source is None else str(self.source),
                "pid": os.getpid(), "workers": self.workers, "updated": _now(), "totals": self.totals(),
                "jobs": self.records,
            },
        )  # fmt: skip

    def _finish(self, k: int, result: dict[str, Any]) -> None:
        record = self.records[k]
        record.update(result, ended=_now())
        minutes = (record["wall_time_s"] or 0.0) / 60.0
        line = f"JOB {k + 1}/{len(self.jobs)} {record['state']} {record['label']} ({minutes:.1f} min)"
        if record["state"] in ("failed", "timeout"):
            line += f"  {record['failed_step']}: {record['error']}"
        elif record["state"] == "blocked":
            line += f"  {record['error']}"
        self.out(line)
        self.write_status()

    def _collect(self, block: bool, timeout: float | None = None) -> bool:
        """Hand the results of finished jobs to their records; wait for one when ``block``."""
        collected = False
        while True:
            try:
                k, result = self.finished.get(block=block and not collected, timeout=timeout)
            except queue_module.Empty:
                return collected
            self.running.pop(k).join()
            self._finish(k, result)
            collected = True

    def _register(self, k: int) -> Callable[[subprocess.Popen | None], None]:
        def register(proc: subprocess.Popen | None) -> None:
            with self.lock:
                if proc is None:
                    self.processes.pop(k, None)
                else:
                    self.processes[k] = proc

        return register

    def _start(self, k: int) -> None:
        job = self.jobs[k]
        self.records[k].update(state="running", started=_now())

        def work() -> None:
            try:
                result = run_job(job, self.queue, self.log_dir / job.log_name, self._register(k))
            except Exception as exc:  # an error of the queue itself ends this job only
                result = {"state": "failed", "failed_step": None, "error": f"{type(exc).__name__}: {exc}"}
            self.finished.put((k, result))

        thread = threading.Thread(target=work, name=f"job-{k + 1}", daemon=True)
        self.running[k] = thread
        self.started += 1
        self.write_status()
        thread.start()

    def _producer_pending(self, job: Job, pending: deque[int]) -> bool:
        """Whether a pending or running job of this queue writes the run directory ``job`` starts from."""
        source = _path_key(job.init_from)
        return any(_path_key(self.jobs[j].run_dir) == source for j in (*pending, *self.running))

    def _memory_ok(self) -> bool:
        free = free_memory_gb() if self.queue.min_free_ram_gb > 0 else None
        if free is None or free >= self.queue.min_free_ram_gb:
            self.waiting_for_memory = False
            return True
        if not self.waiting_for_memory:
            self.out(f"WAIT free memory {free:.1f} GB < {self.queue.min_free_ram_gb:g} GB")
        self.waiting_for_memory = True
        return False

    def run(self) -> dict[str, int]:
        """Run the queue; the totals of the states at the end."""
        pending = deque(range(len(self.jobs)))
        postponed: set[int] = set()
        stalled = 0  # jobs put back in a row, waiting for the model.pt of their init_from
        self.write_status()
        try:
            while pending or self.running:
                if self._collect(block=False):
                    stalled = 0
                limit = self.max_jobs is not None and self.started >= self.max_jobs
                if not pending or limit or len(self.running) >= self.workers or stalled >= len(pending):
                    if self.running:
                        self._collect(block=True)
                        stalled = 0
                        continue
                    if pending and not limit:  # every job left waits for a model.pt that nothing will write
                        for k in pending:
                            self._finish(k, self._blocked(k))
                        pending.clear()
                    break
                k = pending.popleft()
                job = self.jobs[k]
                todo = steps_to_run(job)
                if not todo:
                    self._finish(k, {"state": "skipped", "wall_time_s": 0.0})
                    stalled = 0
                    continue
                if "train" in todo and job.init_from is not None and not (job.init_from / "model.pt").exists():
                    if k not in postponed or self._producer_pending(job, pending):
                        postponed.add(k)  # to the end of the queue
                        pending.append(k)
                        stalled += 1
                    else:
                        self._finish(k, self._blocked(k))
                        stalled = 0
                    continue
                if not self._memory_ok():
                    pending.appendleft(k)
                    if self.running:
                        self._collect(block=True, timeout=MEMORY_POLL_S)
                    else:
                        time.sleep(MEMORY_POLL_S)
                    continue
                self._start(k)
                stalled = 0
        except BaseException:  # KeyboardInterrupt above all: end the steps this queue started
            with self.lock:
                processes = list(self.processes.values())
            for proc in processes:
                kill_tree(proc)
            for k in self.running:
                self.records[k].update(state="interrupted", ended=_now())
            self.write_status()
            raise
        return self.totals()

    def _blocked(self, k: int) -> dict[str, Any]:
        source = self.jobs[k].init_from
        return {"state": "blocked", "wall_time_s": 0.0, "error": f"init_from {source} has no model.pt"}


def dry_run(jobs: Sequence[Job], out: Callable[[str], None] = _print) -> dict[str, int]:
    """One line per job with the steps that would run and the overrides of the job; counts."""
    counts = {"run": 0, "complete": 0, "wait": 0}
    for i, job in enumerate(jobs, 1):
        todo = steps_to_run(job)
        head = f"DRY {i}/{len(jobs)}"
        if not todo:
            counts["complete"] += 1
            out(f"{head} complete {job.label}")
            continue
        waits = "train" in todo and job.init_from is not None and not (job.init_from / "model.pt").exists()
        counts["wait" if waits else "run"] += 1
        what = f"run {','.join(todo)}" + (" after init_from (no model.pt yet)" if waits else "")
        out(f"{head} {what} {job.label}  {' '.join(job.spec.arguments())}")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_experiment.py", description="Run the jobs of a queue file.")
    parser.add_argument("queue_file", help="plain YAML, e.g. configs/queue/e2_sweep.yaml")
    parser.add_argument("--workers", type=int, help="jobs at once (default: workers of the file)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and the steps that would run")
    parser.add_argument("--only", help="only the jobs whose label (experiment/data/model/fold<k>_seed<s>) contains it")
    parser.add_argument("--max-jobs", type=int, help="start at most this many jobs (complete ones do not count)")
    args = parser.parse_args(argv)
    path = Path(args.queue_file)
    if not path.is_absolute() and not path.exists():
        path = resolve_path(path)
    try:
        queue = QueueConfig.from_file(path)
        jobs, runs_root = load_jobs(queue)
    except QueueError as exc:
        _print(f"QUEUE {path.stem}: error in the queue file: {exc}")
        return 2
    if args.only:
        jobs = [job for job in jobs if args.only in job.label]
    if args.dry_run:
        counts = dry_run(jobs)
        _print(
            f"QUEUE {queue.name} (dry run): {len(jobs)} jobs, to run {counts['run']}, complete {counts['complete']}, "
            f"waiting for init_from {counts['wait']}"
        )
        return 0
    queue_dir = runs_root / queue.name
    runner = QueueRunner(queue, jobs, queue_dir, workers=args.workers, max_jobs=args.max_jobs, source=path)
    try:
        with QueueLock(queue_dir / "_queue.lock"):
            _print(f"QUEUE {queue.name}: start, {len(jobs)} jobs, {runner.workers} workers, {runner.status_path}")
            totals = runner.run()
    except QueueLocked as exc:
        _print(f"QUEUE {queue.name}: {exc}")
        return 3
    except KeyboardInterrupt:
        _print(f"QUEUE {queue.name}: interrupted, the running steps were ended")
        return 130
    line = ", ".join(f"{state} {totals[state]}" for state in ("done", "failed", "skipped", "timeout", "blocked"))
    rest = totals["pending"] + totals["interrupted"]
    _print(f"QUEUE {queue.name}: {line}" + (f", not started {rest}" if rest else ""))
    return 0
