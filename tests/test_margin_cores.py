"""Per-event IDM cores with a stability margin and their certificate under a shared residual (D118):
``scripts/calibrate_idm.py per_event_margin=<m>`` and ``scripts/certify_cores.py`` end to end on a tiny event
set, from a foreign working directory."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.models import ResidualIDM, save_model
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import IDM_PARAM_NAMES, idm_acc
from cf_stability.stability.certificate import certify_core, residual_of
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.train.calibration import CalibrationConfig, idm_margin_min
from cf_stability.train.closed_loop import rollout_memoryless
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

CORE_M02 = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}  # HighD fold 0, margin 0.2 (D86)
CORE_FREE = {"v0": 29.57, "T": 0.676, "s0": 2.015, "a": 0.351, "b": 0.5}  # the free core of the same fold
RECORD_KEYS = {
    "event_id", "follower_id", "params", "objective", "fit_objective", "margin_min", "margin_holds", "rmse_s",
    "rmse_v", "nrmse_s", "nrmse_v", "collided", "converged", "n_generations", "at_bound",
}  # fmt: skip


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


def run_script(name: str, overrides: list[str], cwd: Path, ok: bool = True) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), *overrides, f"hydra.run.dir={quoted(cwd / 'outputs')}"],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert (proc.returncode == 0) is ok, proc.stdout + proc.stderr
    return proc.stdout + ("" if ok else proc.stderr)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    events = EventSet(tiny_event(i) for i in range(3))
    events.validate()
    events.to_parquet(tmp_path / "events" / "tiny")
    return tmp_path


def calibrate(root: Path, *extra: str, ok: bool = True) -> str:
    overrides = [
        "dataset=tiny", f"paths.events_root={quoted(root / 'events')}",
        f"paths.calibration_root={quoted(root / 'calibration')}", f"paths.runs_root={quoted(root / 'runs')}",
        "calibration.maxiter=20", "calibration.restarts=1", "calibration.device=cpu", *extra,
    ]  # fmt: skip
    return run_script("calibrate_idm.py", overrides, root, ok)


def test_margin_cores_and_their_certificate(root):
    out = calibrate(root, "per_event_margin=0.2", "global_fit=false")
    (path,) = (root / "calibration" / "tiny").glob("idm_event_margin0.2_*.json")
    assert sorted(p.name for p in (root / "calibration" / "tiny").iterdir()) == [path.name]  # no parquet, no global
    assert path.name in out and "margin 0.2 holds" in out
    cores = read_json(path)
    assert cores["kind"] == "idm_event_margin" and cores["n_events"] == 3 and path.stem.endswith(cores["key"])
    assert cores["config_hash"] == config_hash(cores["config"]) and cores["config"]["per_event_margin"] == 0.2
    settings = cores["settings"]
    assert settings["stability_margin"] == 0.2 and settings["stability_weight"] == 10.0 and settings["maxiter"] == 20
    assert settings["stability_speeds"] == [float(v) for v in range(5, 31)] and "format" in settings
    assert "device" not in settings and settings["variant"] == "full" and settings["fixed"] == {}
    assert settings["margin_tolerance"] == 1e-4
    records = cores["events"]
    assert [r["event_id"] for r in records] == [f"tiny/a/{i}|L{i}|0" for i in range(3)]
    assert all(set(r) == RECORD_KEYS and set(r["params"]) == set(IDM_PARAM_NAMES) for r in records)
    params = torch.tensor([[r["params"][k] for k in IDM_PARAM_NAMES] for r in records], dtype=torch.float64)
    margins = idm_margin_min(params, CalibrationConfig().stability_speeds).tolist()
    assert [r["margin_min"] for r in records] == pytest.approx(margins, rel=1e-12)
    assert all(r["margin_holds"] == (r["margin_min"] >= 0.2 - 1e-4) for r in records)
    assert all(r["objective"] >= r["fit_objective"] for r in records)
    assert cores["summary"]["n_margin_holds"] == sum(r["margin_holds"] for r in records)
    metrics = read_json(root / "runs" / "baselines" / "tiny" / "idm_event_margin0.2" / "metrics.json")
    assert metrics["per_event"]["share_margin_holds"] == cores["summary"]["share_margin_holds"]
    # the per-event margin needs global_fit=false: idm_global.json is not written again
    assert "set global_fit=false" in calibrate(root, "per_event_margin=0.2", ok=False)
    # without the margin the per-event fit of M1, with its files
    calibrate(root, "global_fit=false")
    assert (root / "calibration" / "tiny" / "idm_per_event.parquet").is_file()

    # certificate of every core with the residual of two "fold" runs; the cores of the file plus two known cores
    write_json(path, {**cores, "events": [*records, *(
        {"event_id": name, "follower_id": name, "params": core, "margin_min": None, "margin_holds": None}
        for name, core in (("m02", CORE_M02), ("free", CORE_FREE))
    )]})  # fmt: skip
    residual_runs = root / "runs" / "e4" / "residual_idm"
    lipschitz = {0: 0.3, 1: 1.0}  # r_max * lipschitz 0.09 (fold 0) and 0.3 (fold 1)
    for fold, value in lipschitz.items():
        torch.manual_seed(fold)
        run = residual_runs / f"driver_fold{fold}_seed0"
        run.mkdir(parents=True)
        save_model(ResidualIDM(ModelContext(idm_params=CORE_M02), r_max=0.3, lipschitz=value), run / "model.pt")
    template = f"residual={quoted(residual_runs / 'driver_fold{fold}_seed0')}"
    calibration_root = f"paths.calibration_root={quoted(root / 'calibration')}"
    common = ["dataset=tiny", calibration_root, template, "certificate.n_scan=200"]
    out = run_script("certify_cores.py", common, root)  # fold 0 by default, the cores file found by its name
    certified = path.with_name(f"{path.stem}_certified.json")
    result = read_json(certified)
    assert result["kind"] == "idm_event_margin_certified" and result["settings"]["folds"] == [0]
    assert result["cores_file"] == path.as_posix() and result["cores_key"] == cores["key"]
    assert result["settings"]["residuals"]["0"]["r_max"] == 0.3 and "format" in result["settings"]
    assert result["settings"]["speeds"] == list(V_GRID)
    assert result["config_hash"] == config_hash(result["config"]["certificate"])
    by_id = {r["event_id"]: r for r in result["cores"]}
    assert len(by_id) == 5 and by_id["m02"]["certified"] and not by_id["free"]["certified"]
    bounds = torch.tensor(result["settings"]["residuals"]["0"]["bounds"], dtype=torch.float64)
    for record in result["cores"]:  # the stored verdict is the certificate of the core with that residual
        expected = certify_core(record["params"], 0.3, bounds, n_scan=200)
        assert record["certified"] == record["folds"]["0"]["holds"] == bool(expected["holds"].all())
        assert record["folds"]["0"]["n_hold"] == int(expected["holds"].sum())
        margin = torch.nanquantile(expected["guaranteed_margin"], 0.0).item()
        assert record["guaranteed_margin_min"] == pytest.approx(margin, rel=1e-12)
    n = sum(r["certified"] for r in result["cores"])
    assert result["summary"]["n_certified"] == n and result["summary"]["share_certified"] == pytest.approx(n / 5)
    assert f"certified for every fold: {n}/5" in out
    # every fold: the larger residual of fold 1 breaks the certificate of the margin-0.2 core of HighD
    out = run_script("certify_cores.py", [*common, "folds=[0,1]"], root)
    result = read_json(certified)
    m02 = {r["event_id"]: r for r in result["cores"]}["m02"]
    assert result["settings"]["folds"] == [0, 1] and m02["folds"]["0"]["holds"] and not m02["folds"]["1"]["holds"]
    assert not m02["certified"] and m02["guaranteed_margin_min"] == m02["folds"]["1"]["guaranteed_margin_min"] < 0
    assert residual_of(ResidualIDM(ModelContext(idm_params=CORE_M02), r_max=0.3, lipschitz=1.0))["product"] == 0.3
    assert "fold 1: certified" in out
