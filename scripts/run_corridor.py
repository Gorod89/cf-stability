"""Runs of the corridor simulation (docs/m5_contract.md, section 3).

    python scripts/run_corridor.py scenario=i80_p1 law=idm_global seed=0
    python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2] laws=[idm_global,mlp] seeds=[0,1,2] workers=4
    python scripts/run_corridor.py scenario=us101_p1 law=idm_global seed=0

One run (``scenario``, ``law``, ``seed``) is one process: it writes
``<paths.runs_root>/<scenario>/<law>/seed<seed>/`` with ``trajectories.npz``, ``vehicles.npz``,
``collisions.npz`` (``t, vehicle, x, lane`` of the beginning of every collision episode) and, last,
``run.json`` (counts, throughput, the time mean, minimum and maximum of the buffer gain ``g`` and of the
exit difference ``dN`` inside the analysis window, config, config hash; section 1 of the contract;
``n_offramp``: vehicles that left by the off-ramp of US-101, D112) and prints one line. The scenarios are
those of ``scripts/build_corridor.py``: ``i80_p<k>``, ``us101_p<k>`` and the variants ``i80_p1_<variant>``.
The hashed part of the config (``config_hash``) is the scenario, the law, the seed, ``sim``, the device of
the inference, the config hash of the scenario and the fingerprint of the law (law file, bytes of the
member checkpoints, table); ``paths`` is not hashed. ``device``: null takes the device of the law file
(fixed per law in ``configs/corridor/laws.yaml``), ``cpu`` or ``cuda`` override it.

Multi-run mode (any of ``scenarios``, ``laws``, ``seeds`` set; the others default to ``[scenario]``,
``[law]``, ``[seed]``): the runs of the grid whose ``run.json`` does not carry the hash of their config
(all with ``force=true``) are run as child processes of this script, ``workers`` at once, each with
``timeout_h``; one printed line per run, a failing run does not stop the others, and the grid can be
ended and started again at any moment. The children get the command-line overrides of this call; their
output goes to ``<paths.runs_root>/_logs/<scenario>__<law>__seed<seed>.log``.

This is the simulation process of section 0: libsumo is imported before anything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_stability.corridor.sumo_env import import_libsumo  # noqa: E402

import_libsumo()  # before pandas and torch (docs/m5_contract.md, section 0)

import dataclasses  # noqa: E402
import functools  # noqa: E402
import itertools  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from typing import Any, Mapping  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.corridor.laws import law_device, law_fingerprint, load_law  # noqa: E402
from cf_stability.corridor.loop import Scenario, SimConfig, run_simulation  # noqa: E402
from cf_stability.eval.experiment_queue import bind_children_to_this_process, override_value, run_step  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

GRID_KEYS = ("scenarios", "laws", "seeds")
PARENT_KEYS = frozenset({*GRID_KEYS, "workers", "timeout_h", "force", "scenario", "law", "seed"})
HASHED_KEYS = ("scenario", "law", "seed", "sim", "device", "scenario_hash", "law_hash")
OUTPUTS = ("trajectories.npz", "vehicles.npz", "collisions.npz", "run.json")


def run_directory(cfg: Mapping[str, Any], scenario: str, law: str, seed: int) -> Path:
    return resolve_path(cfg["paths"]["runs_root"]) / scenario / law / f"seed{int(seed)}"


@functools.lru_cache(maxsize=None)
def _identity(path: Path) -> str:
    """Config hash of a scenario or fingerprint of a law, once per process (the grid asks for every seed)."""
    return read_json(path)["config_hash"] if path.name == "scenario.json" else law_fingerprint(path)


def run_config(cfg: Mapping[str, Any], scenario: str, law: str, seed: int) -> dict[str, Any]:
    """The config of one run (``run.json["config"]``); raises when the scenario or the law file does not exist."""
    scenario_file = resolve_path(cfg["paths"]["scenarios_root"]) / scenario / "scenario.json"
    law_file = resolve_path(cfg["paths"]["laws_root"]) / f"{law}.json"
    for path, what in ((scenario_file, "scenario"), (law_file, "law file")):
        if not path.is_file():
            raise FileNotFoundError(f"{what} {path} does not exist (scripts/build_corridor.py)")
    return {
        "scenario": scenario,
        "law": law,
        "seed": int(seed),
        "sim": dataclasses.asdict(SimConfig.from_mapping(cfg.get("sim"))),
        "device": law_device(read_json(law_file), cfg.get("device")),
        "scenario_hash": _identity(scenario_file),
        "law_hash": _identity(law_file),
        "paths": dict(cfg["paths"]),
    }


def hashed(config: Mapping[str, Any]) -> str:
    return config_hash({key: config[key] for key in HASHED_KEYS})


def is_complete(run_dir: Path, expected_hash: str) -> bool:
    """``run.json`` carries ``expected_hash`` and the arrays exist."""
    try:
        payload = read_json(run_dir / "run.json")
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return payload.get("config_hash") == expected_hash and all((run_dir / name).is_file() for name in OUTPUTS)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write through a temporary file and a rename: a killed run leaves no half-written file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def summary_line(payload: Mapping[str, Any], timing: Mapping[str, Any] | None = None) -> str:
    g = (_fmt(payload.get(f"g_{what}"), ".2f") for what in ("mean", "min", "max"))
    ramp = f" (+{payload['n_offramp']} by the off-ramp)" if payload.get("n_offramp") else ""
    line = (
        f"inserted {payload['n_inserted']}/{payload['n_planned']}, exited {payload['n_exited']}{ramp}, in network "
        f"{payload['n_in_network']}, collisions {payload['n_collisions']} (SUMO {payload['n_sumo_collisions']}), "
        f"teleports {payload['n_teleports']}, delay {_fmt(payload['mean_depart_delay_s'], '.2f')} s, "
        "g {} ({}-{}), ".format(*g) + f"dN {_fmt(payload.get('dN_mean'), '.1f')}; "
        f"{payload['vehicle_steps'] / 1e6:.2f} M vehicle steps, {payload['vehicle_steps_per_second'] / 1e3:.0f}k/s"
    )
    if timing:
        loop = timing["loop"]
        without_law = payload["vehicle_steps"] / max(loop - timing["law"], 1e-9)
        line += (
            f" ({without_law / 1e3:.0f}k/s without the law; sumo {timing['sumo']:.0f} s, read {timing['read']:.0f} s, "
            f"law {timing['law']:.0f} s, setSpeed {timing['write']:.0f} s)"
        )
        if timing.get("vanished"):
            line += f"  WARNING: {timing['vanished']} vehicles left the network unexplained"
    return line + f", {payload['config']['device']}, {payload['wall_time_s']:.0f} s"


def run_one(cfg: DictConfig) -> None:
    t_start = time.perf_counter()
    plain = to_plain(cfg)
    scenario_name, law_name, seed = str(cfg.scenario), str(cfg.law), int(cfg.seed)
    config = run_config(plain, scenario_name, law_name, seed)
    law, _ = load_law(resolve_path(plain["paths"]["laws_root"]) / f"{law_name}.json", config["device"])
    scenario = Scenario.load(resolve_path(plain["paths"]["scenarios_root"]) / scenario_name)
    result = run_simulation(scenario, law, seed, SimConfig.from_mapping(plain.get("sim")))

    run_dir = run_directory(plain, scenario_name, law_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_npz(run_dir / "trajectories.npz", result["trajectories"])
    save_npz(run_dir / "vehicles.npz", result["vehicles"])
    save_npz(run_dir / "collisions.npz", result["collisions"])
    payload = {
        "scenario": scenario_name,
        "law": law_name,
        "seed": seed,
        **result["stats"],
        "wall_time_s": time.perf_counter() - t_start,
        "sumo_version": result["sumo_version"],
        "config": config,
        "config_hash": hashed(config),
        "git_revision": git_revision(),
    }
    tmp = write_json(run_dir / f"run.json.{os.getpid()}.tmp", payload)
    os.replace(tmp, run_dir / "run.json")  # last: its presence with the hash marks the run complete
    label = f"{scenario_name}/{law_name}/seed{seed}"
    print(f"RUN {label}: {summary_line(payload, result['timing'])} -> {run_dir}", flush=True)


def child_overrides() -> list[str]:
    """The command-line overrides of this call without the keys of the grid and of the parent."""
    keep = []
    for item in HydraConfig.get().overrides.task:
        key = item.split("=", 1)[0].lstrip("+~")
        if key not in PARENT_KEYS:
            keep.append(item)
    return keep


def run_many(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    grid = {
        "scenarios": [str(s) for s in (cfg.scenarios if cfg.scenarios is not None else [cfg.scenario])],
        "laws": [str(s) for s in (cfg.laws if cfg.laws is not None else [cfg.law])],
        "seeds": [int(s) for s in (cfg.seeds if cfg.seeds is not None else [cfg.seed])],
    }
    runs_root = resolve_path(plain["paths"]["runs_root"])
    hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    base = child_overrides()
    timeout_s = None if cfg.timeout_h is None else 3600.0 * float(cfg.timeout_h)
    counts = {"done": 0, "complete": 0, "failed": 0}
    todo = []
    for scenario, law, seed in itertools.product(grid["scenarios"], grid["laws"], grid["seeds"]):
        label = f"{scenario}/{law}/seed{seed}"
        try:
            expected = hashed(run_config(plain, scenario, law, seed))
        except FileNotFoundError as exc:
            print(f"RUN {label} failed: {exc}", flush=True)
            counts["failed"] += 1
            continue
        run_dir = run_directory(plain, scenario, law, seed)
        if not cfg.force and is_complete(run_dir, expected):
            print(f"RUN {label} complete (config hash {expected})", flush=True)
            counts["complete"] += 1
            continue
        todo.append((label, scenario, law, seed, run_dir, expected))
    print(f"CORRIDOR {len(todo)} runs to do, {counts['complete']} complete, {int(cfg.workers)} workers", flush=True)
    if not todo:
        return
    bind_children_to_this_process()  # the runs end with this process, however it ends

    def work(label: str, scenario: str, law: str, seed: int, run_dir: Path, expected: str) -> str:
        name = f"{scenario}__{law}__seed{seed}"
        command = [
            sys.executable, str(Path(__file__).resolve()), *base, f"scenario={scenario}", f"law={law}", f"seed={seed}",
            f"hydra.run.dir={override_value(hydra_dir / name)}", "hydra.output_subdir=null",
        ]  # fmt: skip
        log = runs_root / "_logs" / f"{name}.log"
        t_start = time.perf_counter()
        code, timed_out = run_step(command, log, timeout_s, note=label)
        if timed_out:
            return f"RUN {label} failed: no result after {cfg.timeout_h:g} h (log {log})"
        if code != 0 or not is_complete(run_dir, expected):
            return f"RUN {label} failed: exit code {code} (log {log})"
        payload = read_json(run_dir / "run.json")
        return f"RUN {label} done: {summary_line(payload)} ({time.perf_counter() - t_start:.0f} s)"

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.workers))) as pool:
        futures = [pool.submit(work, *job) for job in todo]
        for future in as_completed(futures):
            line = future.result()
            counts["done" if " done: " in line else "failed"] += 1
            print(line, flush=True)
    print(f"CORRIDOR done {counts['done']}, complete {counts['complete']}, failed {counts['failed']}", flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run_corridor")
def main(cfg: DictConfig) -> None:
    if any(cfg.get(key) is not None for key in GRID_KEYS):
        run_many(cfg)
    else:
        run_one(cfg)


if __name__ == "__main__":
    main()
