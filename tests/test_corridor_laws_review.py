"""Laws of the review of M8 (cf_stability/corridor/laws.py; the factorial ablation of the certified hybrid): the IDM
core of member runs with the residual switched off (idm_core_margin), the per-fold calibrations selected by their
settings (idm_margin_i80) and the entries of configs/corridor/laws.yaml; synthetic inputs only, and a short run of the
core law in a child process (no libsumo in this process)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cf_stability.corridor.laws import (
    IDM_KEYS,
    IDMLaw,
    ModelLaw,
    calibration_matches,
    export_laws,
    law_fingerprint,
    load_law,
)
from cf_stability.corridor.scenario import build_scenario
from cf_stability.models import IDM, MLP, ModelContext, ResidualIDM, load_model, save_model
from cf_stability.models.idm import idm_acc
from cf_stability.utils import REPO_ROOT, read_json, write_json

LAWS = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "laws.yaml").read_text(encoding="utf-8"))
I80 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "i80.yaml").read_text(encoding="utf-8"))
CORES = [  # two margin cores (as the fold members of e4_stable_ft hold them)
    {"v0": 35.213199615478516, "T": 0.9788587093353271, "s0": 3.193310499191284, "a": 1.5670632123947144,
     "b": 0.5000092387199402},
    {"v0": 35.14104461669922, "T": 0.9873303771018982, "s0": 3.2530648708343506, "a": 1.5770609378814697,
     "b": 0.5000016689300537},
]  # fmt: skip
OTHER = {"v0": 33.0, "T": 1.2, "s0": 2.5, "a": 1.2, "b": 0.8}


def residual_member(run: Path, core: dict, seed: int) -> ResidualIDM:
    """A ResidualIDM with the core ``core``, r_max 0.3 and a non-zero residual, saved as a member run."""
    torch.manual_seed(seed)
    model = ResidualIDM(ModelContext(idm_params=core), r_max=0.3, lipschitz=0.05)
    with torch.no_grad():
        for parameter in model.g.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    model.eval()
    run.mkdir(parents=True, exist_ok=True)
    save_model(model, run / "model.pt")
    write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.2], "box_high": [50.0, 3.0, 17.5]}})
    return load_model(run / "model.pt")


@pytest.fixture()
def members(tmp_path):
    runs = [residual_member(tmp_path / "runs" / f"fold{k}", core, seed=k) for k, core in enumerate(CORES)]
    return tmp_path, runs, (tmp_path / "runs" / "fold{fold}").as_posix()


# ----------------------------------------------------------------------------------------- idm_core_margin


def test_core_law_from_member_runs(members):
    """kind idm with cores_of: the IDM parameters of the members' cores (float32 values), one set per member, no member
    runs (the audits of the runs are those of the hybrid), the runs in source, the support of the runs."""
    root, runs, pattern = members
    out = root / "laws"
    table = {"core": {"kind": "idm", "device": "cpu", "cores_of": pattern},
             "core_missing": {"kind": "idm", "device": "cpu", "cores_of": (root / "none" / "fold{fold}").as_posix()}}  # fmt: skip
    lines = export_laws({"folds": [0, 1], "table": table}, out)
    assert "the IDM cores of 2/2 runs (ResidualIDM; residual off)" in lines[0] and "support 0.2-17.5 m/s" in lines[0]
    assert "not written: 2 of 2 runs of the cores missing" in lines[1] and not (out / "core_missing.json").exists()
    law = read_json(out / "core.json")
    assert law["kind"] == "idm" and law["members"] == [] and law["device"] == "cpu" and law["window"] == 1
    assert law["support_v"] == [0.2, 17.5] and law["source"]["residual"] == "off"
    assert law["source"]["core_runs"] == [pattern.format(fold=k) for k in range(2)]
    for params, model in zip(law["params"], runs):
        assert params == pytest.approx(model.idm.params_dict()) and params == pytest.approx(CORES[runs.index(model)])
    before = law_fingerprint(out / "core.json")  # the cores are in the law file, hence in its fingerprint
    assert law_fingerprint(out / "core.json") == before
    save_model(ResidualIDM(ModelContext(idm_params=OTHER), r_max=0.3), root / "runs" / "fold1" / "model.pt")
    export_laws({"folds": [0, 1], "table": table}, out, only=["core"])
    assert law_fingerprint(out / "core.json") != before
    mlp = root / "mlp" / "fold0"  # a model without an IDM core cannot give one
    mlp.mkdir(parents=True)
    save_model(MLP(ModelContext()), mlp / "model.pt")
    write_json(mlp / "metrics.json", {"context": {}})
    with pytest.raises(ValueError, match="no IDM core"):
        export_laws({"folds": [0], "table": {"bad": {"kind": "idm", "cores_of": (root / "mlp" / "fold{fold}").as_posix()}}},
                    out)  # fmt: skip


def test_core_law_draws_the_members_of_the_hybrid(members):
    """The core law gives every vehicle the core of the member that drives it in the law of the member runs (the same
    draws from the generator of the seed) and its acceleration is that core's IDM, the residual switched off."""
    root, runs, pattern = members
    out = root / "laws"
    export_laws({"folds": [0, 1], "table": {"core": {"kind": "idm", "device": "cpu", "cores_of": pattern},
                                            "hybrid": {"kind": "models", "device": "cpu", "members": pattern}}}, out)  # fmt: skip
    core, device = load_law(out / "core.json")
    hybrid, _ = load_law(out / "hybrid.json")
    assert isinstance(core, IDMLaw) and isinstance(hybrid, ModelLaw) and device == "cpu" and core.n_members == 2
    member = core.assign(200, np.random.default_rng(5))
    np.testing.assert_array_equal(member, hybrid.assign(200, np.random.default_rng(5)))
    rng = np.random.default_rng(1)
    history = np.stack([rng.uniform(2.0, 40.0, 200), rng.uniform(-2.0, 2.0, 200), rng.uniform(0.0, 17.0, 200)],
                       axis=1)[:, None, :]  # fmt: skip
    got = core.accelerations(np.arange(200), history)
    x = torch.as_tensor(history, dtype=torch.float32)
    residuals = []
    for k in range(200):
        model = runs[int(member[k])]
        p = torch.tensor([model.idm.params_dict()[key] for key in IDM_KEYS], dtype=torch.float64)
        last = torch.as_tensor(history[k, -1], dtype=torch.float64)
        assert got[k] == pytest.approx(float(idm_acc(*last.unbind(-1), *p.unbind(-1))), rel=1e-12, abs=1e-12)
        with torch.no_grad():  # the float32 core of the member: the hybrid without its residual
            assert got[k] == pytest.approx(float(model.idm(x[k : k + 1])[0]), rel=1e-4, abs=1e-4)
            residuals.append(float(model.residual(x[k : k + 1])[0]))
    assert max(abs(r) for r in residuals) > 1e-3  # the hybrid itself differs: the residual is what is switched off


# ------------------------------------------------------------------------------------------ idm_margin_i80


def test_calibration_selected_by_its_settings(tmp_path):
    """kind idm with calibration_settings: of the per-fold files only the one whose settings hold the values; none or
    two: the law is not written and the line names the pattern; without the key the file is the same as before."""
    calib = tmp_path / "calibration"
    free, margin = {"stability_margin": None, "seed": 0}, {"stability_margin": 0.2, "seed": 0}
    for k in range(3):
        write_json(calib / f"idm_global_driver_fold{k}_aaa{k}.json", {"fold": k, "settings": free, "params": OTHER})
    for k in range(2):
        write_json(calib / f"idm_global_driver_fold{k}_bbb{k}.json", {"fold": k, "settings": margin,
                                                                      "params": {**CORES[0], "T": 1.0 + k}})  # fmt: skip
    pattern = (calib / "idm_global_driver_fold{fold}_*.json").as_posix()
    spec = {"kind": "idm", "device": "cpu", "calibration": pattern, "calibration_settings": {"stability_margin": 0.2}}
    out = tmp_path / "laws"
    lines = export_laws({"folds": [0, 1], "table": {"margin": spec}}, out)
    assert "2 calibrations" in lines[0] and "with settings.stability_margin = 0.2" in lines[0], lines[0]
    law = read_json(out / "margin.json")
    assert [p["T"] for p in law["params"]] == [1.0, 2.0] and law["members"] == []
    assert law["source"]["calibration_settings"] == {"stability_margin": 0.2}
    assert [Path(p).name for p in law["source"]["calibration"]] == ["idm_global_driver_fold0_bbb0.json",
                                                                     "idm_global_driver_fold1_bbb1.json"]  # fmt: skip
    lines = export_laws({"folds": [0, 1, 2], "table": {"margin3": spec}}, out)  # fold 2 has no margin calibration
    assert "not written" in lines[0] and "no file with settings.stability_margin = 0.2 (of 1 file)" in lines[0]
    assert "fold2" in lines[0] and not (out / "margin3.json").exists()
    write_json(calib / "idm_global_driver_fold0_ccc0.json", {"settings": margin, "params": OTHER})
    lines = export_laws({"folds": [0, 1], "table": {"margin2": spec}}, out)
    assert "not written" in lines[0] and "2 files with settings.stability_margin = 0.2" in lines[0]
    plain = {k: v for k, v in spec.items() if k != "calibration_settings"}  # the behaviour of D120 is unchanged
    lines = export_laws({"folds": [2], "table": {"plain": plain}}, out)
    assert "1 calibration (" in lines[0] and "calibration_settings" not in read_json(out / "plain.json")["source"]
    assert calibration_matches(calib / "idm_global_driver_fold1_bbb1.json", {"stability_margin": 0.2 + 1e-12})
    assert not calibration_matches(calib / "idm_global_driver_fold1_aaa1.json", {"stability_margin": 0.2})
    assert not calibration_matches(calib / "idm_global_driver_fold1_aaa1.json", {"other": None})  # a missing key


# --------------------------------------------------------------------------------------------- the table


def test_review_laws_in_the_table():
    table = LAWS["table"]
    core = table["idm_core_margin"]
    assert core == {"kind": "idm", "device": "cpu", "cores_of": table["residual_idm_certified"]["members"]}
    margin = table["idm_margin_i80"]
    assert margin["kind"] == "idm" and margin["device"] == "cpu" and margin["support_from"] == "idm_global"
    assert margin["calibration"] == "data/calibration/ngsim_i80/idm_global_driver_fold{fold}_*.json"
    assert margin["calibration_settings"] == {"stability_margin": 0.2} and "members" not in margin
    free = table["residual_idm_margin_free_r0.3"]
    assert free == {"kind": "models", "device": "cpu",
                    "members": "runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold{fold}_seed0"}  # fmt: skip


def test_review_laws_from_a_run_tree(tmp_path):
    """The three laws of the table from synthetic inputs under tmp_path: written when their inputs exist, else one line
    that names what is missing."""
    names = ["idm_core_margin", "idm_margin_i80", "residual_idm_margin_free_r0.3", "idm_global"]
    table = {}
    for name in names:
        spec = dict(LAWS["table"][name])
        for key in ("members", "cores_of", "calibration"):
            if key in spec:
                spec[key] = (tmp_path / spec[key]).as_posix()
        table[name] = spec
    lines = export_laws({"folds": LAWS["folds"], "table": table}, tmp_path / "laws")
    assert "not written: 5 of 5 runs of the cores missing" in lines[0]
    assert "not written: calibration missing or ambiguous" in lines[1] and "no file with settings.stability_margin" in lines[1]
    assert "not written: 5 of 5 member runs missing" in lines[2]
    for fold in LAWS["folds"]:
        residual_member(Path(table["idm_core_margin"]["cores_of"].format(fold=fold)), CORES[fold % 2], seed=fold)
        residual_member(Path(table["residual_idm_margin_free_r0.3"]["members"].format(fold=fold)), CORES[0], seed=fold)
        run = Path(table["idm_global"]["members"].format(fold=fold))
        run.mkdir(parents=True)
        save_model(IDM(OTHER), run / "model.pt")
        write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.0], "box_high": [50.0, 3.0, 18.0]}})
        write_json(Path(table["idm_margin_i80"]["calibration"].format(fold=fold).replace("*", "m")),
                   {"settings": {"stability_margin": 0.2}, "params": {**OTHER, "T": 1.5}})  # fmt: skip
    lines = export_laws({"folds": LAWS["folds"], "table": table}, tmp_path / "laws")
    assert "the IDM cores of 5/5 runs" in lines[0] and "5 calibrations" in lines[1] and "5/5 members" in lines[2], lines
    margin = read_json(tmp_path / "laws" / "idm_margin_i80.json")
    assert margin["support_v"] == [0.0, 18.0] and all(p["T"] == 1.5 for p in margin["params"])


def test_core_law_run_in_a_child_process(members):
    """A short run of the core law through scripts/run_corridor.py on a synthetic scenario (libsumo in the child): every
    vehicle inserted, and the members of the vehicles those of the law of the member runs for the same seed."""
    root, runs, pattern = members
    laws = root / "laws"
    export_laws({"folds": [0, 1], "table": {"core": {"kind": "idm", "device": "cpu", "cores_of": pattern},
                                            "hybrid": {"kind": "models", "device": "cpu", "members": pattern}}}, laws)  # fmt: skip
    frames = []
    for lane in range(1, 7):
        for j in range(3):
            n = 300
            frames.append(pd.DataFrame({"track_id": f"p0_{lane * 10 + j}", "period": 0, "vehicle_id": lane * 10 + j,
                                        "frame_id": 1 + 80 * j + 5 * lane + np.arange(n), "lane_id": lane,
                                        "x": 20.2 + 12.0 * 0.1 * np.arange(n), "v": 12.0, "length": 4.5, "v_class": 2}))  # fmt: skip
    build_scenario(pd.concat(frames, ignore_index=True), {**I80, "name": "syn"}, 0, root / "scenarios" / "syn_p0")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    quoted = lambda p: f"'{p.as_posix()}'"  # noqa: E731  (hydra: non-ASCII paths quoted)
    vehicles = {}
    for name in ("core", "hybrid"):
        args = [sys.executable, str(REPO_ROOT / "scripts" / "run_corridor.py"), "scenario=syn_p0", f"law={name}",
                "seed=3", "sim.tail_s=15.0", f"paths.scenarios_root={quoted(root / 'scenarios')}",
                f"paths.laws_root={quoted(laws)}", f"paths.runs_root={quoted(root / 'runs_corridor')}",
                f"hydra.run.dir={quoted(root / 'outputs' / name)}"]  # fmt: skip
        proc = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=600)  # fmt: skip
        assert proc.returncode == 0, proc.stdout + proc.stderr
        run_dir = root / "runs_corridor" / "syn_p0" / name / "seed3"
        run = read_json(run_dir / "run.json")
        assert run["n_inserted"] == run["n_planned"] == 18 and run["n_collisions"] == 0
        with np.load(run_dir / "vehicles.npz") as data:
            vehicles[name] = data["member"].copy()
    np.testing.assert_array_equal(vehicles["core"], vehicles["hybrid"])
