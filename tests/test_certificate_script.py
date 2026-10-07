"""scripts/certificate.py end to end (docs/m4_contract.md, 3.3): a ResidualIDM with a spacing band in its
context, a model where the certificate is not applicable, a broken checkpoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cf_stability.models import MLP, ResidualIDM, save_model
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import idm_equilibrium_spacing, idm_margin
from cf_stability.stability.certificate import max_residual_budget
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

IDM_P = {"v0": 33.0, "T": 1.5, "s0": 2.0, "a": 2.0, "b": 1.0}  # string stable on the whole speed grid
LIPSCHITZ = 0.01  # r_max * lipschitz below the admissible budget of this core (0.018)
TOP_KEYS = {
    "run", "model", "applicable", "core", "residual", "a_priori", "at_equilibria", "per_speed", "config",
    "config_hash", "git_revision", "wall_time_s",
}  # fmt: skip
PER_SPEED_KEYS = {
    "v", "core_margin", "a_priori_holds", "a_priori_margin", "a_priori_s_low", "a_priori_s_high", "a_priori_budget",
    "s", "status", "found", "holds", "guaranteed_margin", "empirical_margin", "budget",
}  # fmt: skip


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def run_script(tmp_path: Path, *overrides: str) -> tuple[int, list[str]]:
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "certificate.py"), *overrides, "certificate.n_scan=200",
        f"paths.runs_root={quoted(tmp_path / 'runs')}", f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode in (0, 1) and "Traceback" not in proc.stderr, proc.stdout + proc.stderr
    return proc.returncode, proc.stdout.strip().splitlines()


def band_below_the_equilibrium_from(v_split: float) -> dict:
    """Band of the context on 10-25 m/s: around the IDM equilibrium below ``v_split``, under it from there on."""
    v = torch.arange(10.0, 26.0, dtype=torch.float64)
    s_e = idm_equilibrium_spacing(v, IDM_P["v0"], IDM_P["T"], IDM_P["s0"])
    low, high = torch.where(v < v_split, 0.8, 0.5) * s_e, torch.where(v < v_split, 1.2, 0.9) * s_e
    columns = {"v": v, "s_low": low, "s_median": s_e, "s_high": high}
    return {**{k: c.tolist() for k, c in columns.items()}, "n": [500] * len(v), "settings": {}}


def test_certificate_script(tmp_path):
    runs = tmp_path / "runs" / "exp" / "synthetic"
    hybrid, mlp, broken = (runs / name / "driver_fold0_seed0" for name in ("residual_idm", "mlp", "broken"))
    for run in (hybrid, mlp, broken):
        run.mkdir(parents=True)
    torch.manual_seed(0)
    model = ResidualIDM(ModelContext(idm_params=IDM_P), lipschitz=LIPSCHITZ)
    save_model(model, hybrid / "model.pt")
    write_json(hybrid / "metrics.json", {"context": {"band": band_below_the_equilibrium_from(20)}})
    save_model(MLP(), mlp / "model.pt")
    (broken / "model.pt").write_bytes(b"not a checkpoint")

    status, lines = run_script(tmp_path, "experiment=exp", "data=synthetic")
    assert status == 1 and len(lines) == 3
    assert "FAILED" in lines[0] and lines[1] == f"{'mlp/driver_fold0_seed0':<30} not applicable (mlp)"
    assert lines[2].startswith("residual_idm/driver_fold0_seed0") and "a priori 26/26" in lines[2]
    assert read_json(mlp / "certificate.json")["applicable"] is False
    assert "error" in read_json(broken / "certificate.json")

    text = (hybrid / "certificate.json").read_text(encoding="utf-8")
    cert = json.loads(text)
    assert "NaN" not in text and set(cert) == TOP_KEYS and cert["model"] == "residual_idm" and cert["applicable"]
    assert cert["config_hash"] == config_hash(cert["config"]["certificate"])
    v = torch.tensor(V_GRID, dtype=torch.float64)
    margins = idm_margin(v, *torch.tensor(list(IDM_P.values()), dtype=torch.float64))
    assert cert["core"]["params"] == pytest.approx(IDM_P, rel=1e-6) and cert["core"]["n_equilibria"] == 26
    assert cert["core"]["margin_min"] == pytest.approx(float(margins.min()), rel=1e-5)
    assert cert["core"]["margin_max"] == pytest.approx(float(margins.max()), rel=1e-5)
    residual = cert["residual"]
    assert residual["r_max"] == 1.0 and residual["lipschitz"] == LIPSCHITZ and residual["product"] == LIPSCHITZ
    assert len(residual["bounds"]) == 3 and len(residual["layer_norms"]) == 3
    admissible = float(max_residual_budget(model, a_priori=True, keep_r_max=True, n_scan=200).min())
    assert cert["a_priori"] == {
        "holds": True, "n_hold": 26, "n_grid": 26, "guaranteed_margin_min": cert["a_priori"]["guaranteed_margin_min"],
        "admissible_budget": pytest.approx(admissible, rel=1e-12),
    }  # fmt: skip
    # a posteriori with the band of the context: the equilibria of 20-25 m/s lie outside it and do not count
    post = cert["at_equilibria"]
    assert post["band"] is True and post["n_equilibria"] == post["n_hold"] == 20 and post["holds"] is False
    assert post["empirical_margin_min"] >= post["guaranteed_margin_min"] >= cert["a_priori"]["guaranteed_margin_min"]
    per_speed = cert["per_speed"]
    assert [r["v"] for r in per_speed] == list(V_GRID) and all(set(r) == PER_SPEED_KEYS for r in per_speed)
    outside = [r for r in per_speed if 20 <= r["v"] <= 25]
    assert all(r["status"] == "outside" and not r["found"] and not r["holds"] for r in outside)
    assert all(r["empirical_margin"] is None and r["budget"] == 0.0 for r in outside)
    assert all(r["a_priori_holds"] and r["a_priori_budget"] >= LIPSCHITZ for r in per_speed)

    assert run_script(tmp_path, "experiment=exp", "data=synthetic") == (0, [])  # outputs newer than model.pt are kept
    status, lines = run_script(tmp_path, f"run={quoted(hybrid)}", "force=true")
    assert status == 0 and len(lines) == 1 and lines[0].startswith("residual_idm/driver_fold0_seed0")
