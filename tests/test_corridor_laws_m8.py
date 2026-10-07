"""Laws of M8 (cf_stability/corridor/laws.py; docs/m8_contract.md, sections 5 and 7; D118, D120): the global IDM
from calibration files (idm_global_p0), the heterogeneous certified hybrid (residual_idm_certified_het) with its
reader of the per-core certificate file, its batched inference and a short run in a child process (no libsumo in this
process)."""

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
    ModelLaw,
    ResidualHetLaw,
    export_laws,
    law_device,
    law_fingerprint,
    load_law,
    read_cores,
    recheck_cores,
)
from cf_stability.corridor.scenario import build_scenario
from cf_stability.models import IDM, ModelContext, ResidualIDM, load_model, save_model
from cf_stability.models.idm import idm_acc
from cf_stability.utils import REPO_ROOT, read_json, write_json

LAWS = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "laws.yaml").read_text(encoding="utf-8"))
I80 = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "i80.yaml").read_text(encoding="utf-8"))
# the core of e4_stable_ft fold 0 (margin 0.2, certificate.json): the a priori certificate holds with r_max 0.3
CORE = {"v0": 35.213199615478516, "T": 0.9788587093353271, "s0": 3.193310499191284, "a": 1.5670632123947144,
        "b": 0.5000092387199402}  # fmt: skip
FREE = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 0.5, "b": 3.0}  # an IDM without margin: not certifiable
OTHER = {"v0": 33.0, "T": 1.2, "s0": 2.5, "a": 1.2, "b": 0.8}


def residual_member(run: Path, core: dict, seed: int, lipschitz: float = 0.3) -> ResidualIDM:
    """A ResidualIDM with core ``core``, r_max 0.3 and a non-zero residual, saved as a member run."""
    torch.manual_seed(seed)
    model = ResidualIDM(ModelContext(idm_params=core), r_max=0.3, lipschitz=lipschitz)
    with torch.no_grad():
        for parameter in model.g.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
        model.train()
        for _ in range(50):  # power iterations of the spectral normalisation for the new weights
            model(torch.zeros(1, 1, 3))
    model.eval()
    assert max(model.layer_norms()) < 1.01
    run.mkdir(parents=True, exist_ok=True)
    save_model(model, run / "model.pt")
    write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.2], "box_high": [50.0, 3.0, 17.5]}})
    return load_model(run / "model.pt")


def idm_member(run: Path, params: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    save_model(IDM(params), run / "model.pt")
    write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.0], "box_high": [50.0, 3.0, 18.5]}})


# ------------------------------------------------------------------------------------------- idm_global_p0


def test_idm_law_from_calibration_files(tmp_path):
    """D120: kind idm with `calibration`: one member per file (`{fold}`: per fold), parameters from `params` or
    `global.params`, support from `support_from`; a missing or ambiguous file leaves the law out with the reason."""
    calib = tmp_path / "calibration"
    write_json(calib / "view" / "idm_global.json", {"dataset": "view", "params": OTHER, "objective": 0.2})
    for k in range(2):
        write_json(calib / "folds" / f"idm_global_driver_fold{k}_abc{k}.json", {"global": {"params": {**OTHER, "T": 1.0 + k}}})
    write_json(calib / "two" / "idm_global_a.json", {"params": OTHER})
    write_json(calib / "two" / "idm_global_b.json", {"params": OTHER})
    write_json(calib / "bad" / "idm_global.json", {"params": {"v0": 30.0}})
    idm_member(tmp_path / "runs" / "fold0", CORE)
    idm_member(tmp_path / "runs" / "fold1", CORE)
    table = {
        "idm_global": {"kind": "idm", "device": "cpu", "members": (tmp_path / "runs" / "fold{fold}").as_posix()},
        "p0": {"kind": "idm", "device": "cpu", "calibration": (calib / "view" / "idm_global.json").as_posix(),
               "support_from": "idm_global"},
        "per_fold": {"kind": "idm", "calibration": (calib / "folds" / "idm_global_driver_fold{fold}_*.json").as_posix()},
        "absent": {"kind": "idm", "calibration": (calib / "none" / "idm_global.json").as_posix()},
        "ambiguous": {"kind": "idm", "calibration": (calib / "two" / "idm_global_*.json").as_posix()},
        "broken": {"kind": "idm", "calibration": (calib / "bad" / "idm_global.json").as_posix()},
    }  # fmt: skip
    out = tmp_path / "laws"
    lines = export_laws({"folds": [0, 1], "table": table}, out)
    assert "1 calibration (" in lines[1] and "support 0.0-18.5 m/s" in lines[1], lines[1]
    assert "2 calibrations" in lines[2]
    assert "not written: calibration missing or ambiguous" in lines[3] and "no file" in lines[3]
    assert "not written" in lines[4] and "2 files" in lines[4] and "name one" in lines[4]
    assert "not written" in lines[5] and "holds no IDM parameters" in lines[5]
    assert not any((out / f"{name}.json").exists() for name in ("absent", "ambiguous", "broken"))
    law = read_json(out / "p0.json")
    assert law["kind"] == "idm" and law["members"] == [] and law["device"] == "cpu" and law["window"] == 1
    assert law["params"] == [pytest.approx(OTHER)] and law["support_v"] == [0.0, 18.5]
    assert law["source"]["support_from"] == "idm_global" and law_device(law) == "cpu"
    folds = read_json(out / "per_fold.json")
    assert [p["T"] for p in folds["params"]] == [1.0, 2.0] and folds["support_v"] is None
    driver, device = load_law(out / "p0.json")
    assert device == "cpu" and driver.n_members == 1 and driver.window == 1
    driver.assign(3, np.random.default_rng(0))
    history = np.array([[[20.0, 0.5, 10.0]], [[8.0, -1.0, 5.0]], [[30.0, 0.0, 15.0]]])
    expected = IDM(OTHER).acc(*torch.as_tensor(history[:, 0, :]).unbind(-1)).numpy()
    np.testing.assert_allclose(driver.accelerations(np.arange(3), history), expected, rtol=1e-12)
    before = law_fingerprint(out / "p0.json")  # the parameters are part of the law file, hence of the fingerprint
    write_json(calib / "view" / "idm_global.json", {"params": {**OTHER, "a": 2.0}})
    export_laws({"folds": [0, 1], "table": table}, out, only=["p0"])
    assert law_fingerprint(out / "p0.json") != before


# ------------------------------------------------------------------------------------------ cores reader


def test_read_cores_layouts(tmp_path):
    """The reader of the per-core certificate file accepts records (parameters at the top level or nested, as a
    mapping or a list), columns and a mapping of id to record, with the flag under several names, and finds the fold
    of the certificate's residual in the settings."""
    records = [
        {"event": "e0", "params": CORE, "certified": True, "margin": 0.21},
        {"event": "e1", "parameters": [FREE[k] for k in IDM_KEYS], "certificate": {"holds": False}},
        {"event": "e2", **OTHER, "a_priori_holds": "yes", "guaranteed_margin_min": 0.01},
    ]
    layouts = {
        "list.json": records,
        "mapping.json": {"settings": {"residual_run": "runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0"},
                         "share_certified": 0.5, "cores": records},
        "columns.json": {"settings": {"fold": 2}, **{k: [CORE[k], FREE[k], OTHER[k]] for k in IDM_KEYS},
                         "certified": [True, False, True]},
        "by_id.json": {"config": {"residual_fold": 1}, "events": {f"e{i}": r for i, r in enumerate(records)}},
    }  # fmt: skip
    layouts["certify_cores.json"] = {"kind": "idm_event_margin_certified", "summary": {"share_certified": 0.43},
                                     "settings": {"folds": [0, 1, 2, 3, 4], "residual_runs": {"0": "runs/x_fold0", "1": "runs/x_fold1"}},
                                     "cores": [{**r, "certified": r.get("certified", False)} for r in records]}  # fmt: skip
    layouts["certify_cores.json"]["cores"][2]["certified"] = True
    folds = {"list.json": None, "mapping.json": 3, "columns.json": 2, "by_id.json": 1, "certify_cores.json": 0}
    for name, payload in layouts.items():
        write_json(tmp_path / name, payload)
        cores, info = read_cores(tmp_path / name)
        assert [c["certified"] for c in cores] == [True, False, True], name
        assert cores[0]["params"] == pytest.approx([CORE[k] for k in IDM_KEYS]), name
        assert cores[1]["params"] == pytest.approx([FREE[k] for k in IDM_KEYS]), name
        assert [c["index"] for c in cores] == [0, 1, 2] and info["fold"] == folds[name], name
    cores, info = read_cores(tmp_path / "certify_cores.json")
    assert info["folds"] == [0, 1, 2, 3, 4] and info["residual_runs"] == ["runs/x_fold0", "runs/x_fold1"]
    assert info["share_certified"] == 0.43
    write_json(tmp_path / "runs_list.json", {"settings": {"folds": [1], "residual_runs": ["runs/x_fold1"]}, "cores": records})
    assert read_cores(tmp_path / "runs_list.json")[1]["residual_runs"] == ["runs/x_fold1"]  # a list is taken as well
    cores, info = read_cores(tmp_path / "list.json")
    assert info["folds"] is None and info["residual_runs"] is None
    assert cores[0]["margin"] == 0.21 and cores[2]["margin"] == 0.01 and cores[1]["margin"] is None
    assert read_cores(tmp_path / "mapping.json")[1]["share_certified"] == 0.5
    write_json(tmp_path / "noflag.json", [{"params": CORE}])
    with pytest.raises(ValueError, match="flag"):
        read_cores(tmp_path / "noflag.json")
    write_json(tmp_path / "nolist.json", {"settings": {}})
    with pytest.raises(ValueError, match="no list of cores"):
        read_cores(tmp_path / "nolist.json")


def test_recheck_cores_against_the_certificate_code():
    """The recheck reproduces the a priori certificate of a run: the core of e4_stable_ft fold 0 with its own residual
    bound holds (certificate.json: 26 of 26 speeds), the IDM without margin does not; a larger bound can only fail."""
    bound = [0.005023677178014134, 0.05023677178014133, 0.010047354356028267]  # certificate.json, fold 0
    params = np.array([[CORE[k] for k in IDM_KEYS], [FREE[k] for k in IDM_KEYS]])
    assert recheck_cores(params, bound, 0.3).tolist() == [True, False]
    assert recheck_cores(params[:1], [10 * b for b in bound], 0.3).tolist() == [False]


# ------------------------------------------------------------------------------- residual_idm_certified_het


@pytest.fixture()
def het_tree(tmp_path):
    """Two ResidualIDM members with the core CORE (as the fold members of e4_stable_ft) and a cores file in the format
    of scripts/certify_cores.py (D118, training side) with the member core (certified), FREE (marked certified in the
    file, fails the recheck), OTHER (not certified); the certificate of the file used the residual of fold 0 only."""
    members = [residual_member(tmp_path / "runs" / f"fold{k}", CORE, seed=k, lipschitz=0.05) for k in range(2)]
    cores_file = tmp_path / "calibration" / "idm_event_margin0.2_abc_certified.json"
    write_json(cores_file, {
        "kind": "idm_event_margin_certified", "cores_file": "idm_event_margin0.2_abc.json", "cores_key": "abc",
        "settings": {"folds": [0], "residual_runs": {"0": (tmp_path / "runs" / "fold0").as_posix()}, "n_scan": 400},
        "summary": {"n_cores": 3, "n_certified": 2, "share_certified": 2 / 3},
        "cores": [{"event_id": "e0", "params": CORE, "margin_min": 0.2, "certified": True, "guaranteed_margin_min": 0.01,
                   "folds": {"0": {"holds": True}}},
                  {"event_id": "e1", "params": FREE, "margin_min": 0.2, "certified": True},
                  {"event_id": "e2", "params": OTHER, "margin_min": 0.1, "certified": False}],
    })  # fmt: skip
    spec = {"kind": "residual_heterogeneous", "device": "cpu", "members": (tmp_path / "runs" / "fold{fold}").as_posix(),
            "cores": (tmp_path / "calibration" / "idm_event_margin0.2_*_certified.json").as_posix(), "recheck": True}  # fmt: skip
    return tmp_path, members, spec


def test_residual_het_export(het_tree):
    root, members, spec = het_tree
    out = root / "laws"
    table = {"het": spec, "het_norecheck": {**spec, "recheck": False},
             "no_cores": {**spec, "cores": (root / "calibration" / "none_*.json").as_posix()},
             "no_members": {**spec, "members": (root / "none" / "fold{fold}").as_posix()}}  # fmt: skip
    lines = export_laws({"folds": [0, 1], "table": table}, out)
    assert "1 cores of 3 cores, 2 certified in the file, 1 hold with the largest residual bound of the 2 members" in lines[0]
    assert "residual of fold(s) 0" in lines[0] and "2/2 members" in lines[0], lines[0]
    # without the recheck the cores would meet the residual of fold 1, which their certificate did not use (D118)
    assert "not written: the certificate of" in lines[1] and "certify_cores.py folds=[0,1,2,3,4]" in lines[1], lines[1]
    assert "not written: certified cores missing or ambiguous" in lines[2]
    assert "not written: 2 of 2 member runs missing" in lines[3]
    law = read_json(out / "het.json")
    assert law["kind"] == "residual_heterogeneous" and law["device"] == "cpu" and law["table"] == "het.npz"
    assert len(law["members"]) == 2 and law["window"] == 1 and law["support_v"] == [0.2, 17.5]
    source = law["source"]
    assert source["n_cores"] == 3 and source["n_certified"] == 2 and source["n_used"] == 1
    assert source["n_hold_all_members"] == 1 and source["certificate_folds"] == [0] and source["r_max"] == [0.3]
    assert source["members_covered"] is False and source["share_certified"] == pytest.approx(2 / 3)
    bounds = np.array(source["residual_bounds"])
    assert bounds.shape == (2, 3) and np.allclose(source["bound_used"], bounds.max(axis=0))
    with np.load(out / "het.npz") as data:
        assert np.allclose([data[k][0] for k in IDM_KEYS], [CORE[k] for k in IDM_KEYS]) and data["index"].tolist() == [0]
    assert not (out / "het_norecheck.json").exists()
    every = {**read_json(root / "calibration" / "idm_event_margin0.2_abc_certified.json")}
    every["settings"] = {**every["settings"], "folds": [0, 1],
                         "residual_runs": {str(k): (root / "runs" / f"fold{k}").as_posix() for k in range(2)}}  # fmt: skip
    write_json(root / "calibration" / "idm_event_margin0.2_abc_certified.json", every)  # certified with both folds
    lines = export_laws({"folds": [0, 1], "table": {"het_norecheck": {**spec, "recheck": False}}}, out)
    assert "2 cores of 3 cores, 2 certified in the file (certificate of the file: residual of fold(s) 0, 1)" in lines[0]
    with np.load(out / "het_norecheck.npz") as data:
        assert data["index"].tolist() == [0, 1]
    assert read_json(out / "het_norecheck.json")["source"]["members_covered"] is True
    with pytest.raises(ValueError, match="CPU"):  # an IDM-type law is computed on the CPU
        law_device(law, "cuda")
    before = law_fingerprint(out / "het.json")  # the cores are part of the fingerprint
    with np.load(out / "het.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["T"] = arrays["T"] + 0.1
    np.savez(out / "het.npz", **arrays)
    assert law_fingerprint(out / "het.json") != before


def test_residual_het_inference(het_tree):
    """The members are drawn as by ModelLaw; with a table holding the members' own core the law is that of the fold
    members exactly; with several cores every vehicle gets idm(its core) + the residual of its member."""
    root, members, spec = het_tree
    rng = np.random.default_rng(0)
    states = np.stack([rng.uniform(2.0, 40.0, 64), rng.uniform(-2.0, 2.0, 64), rng.uniform(0.0, 17.0, 64)], axis=1)
    history = states[:, None, :]
    rows = np.arange(64)
    own = np.array([[float(np.float32(CORE[k])) for k in IDM_KEYS]])
    reference, het = ModelLaw("ref", members, "cpu"), ResidualHetLaw("het", members, own, "cpu")
    assert (reference.assign(64, np.random.default_rng(7)) == het.assign(64, np.random.default_rng(7))).all()
    np.testing.assert_array_equal(het.accelerations(rows, history), reference.accelerations(rows, history))
    table = np.array([[CORE[k] for k in IDM_KEYS], [OTHER[k] for k in IDM_KEYS], [FREE[k] for k in IDM_KEYS]])
    het = ResidualHetLaw("het", members, table, "cpu")
    member = het.assign(64, np.random.default_rng(11))
    assert set(np.unique(member)) == {0, 1} and len({tuple(p) for p in het.params}) == 3
    got = het.accelerations(rows[::-1], history[::-1])  # any order of the rows
    x = torch.as_tensor(history, dtype=torch.float32)
    with torch.no_grad():
        for k in range(64):
            p = torch.as_tensor(het.params[k], dtype=torch.float32)
            core = idm_acc(x[k, -1, 0], x[k, -1, 1], x[k, -1, 2], *p.unbind(-1))
            residual = members[int(member[k])].residual(x[k : k + 1])[0]
            assert got[63 - k] == pytest.approx(float(core + residual), rel=1e-6, abs=1e-6)
    with pytest.raises(ValueError, match="table of cores"):
        ResidualHetLaw("het", members, np.zeros((0, 5)), "cpu")
    out = root / "laws"
    export_laws({"folds": [0, 1], "table": {"het": spec}}, out)
    loaded, device = load_law(out / "het.json")
    assert isinstance(loaded, ResidualHetLaw) and device == "cpu" and loaded.n_members == 2
    assert loaded.table.shape == (1, 5)


def test_m8_laws_in_the_table():
    """The entries of configs/corridor/laws.yaml for D118 and D120."""
    table = LAWS["table"]
    het = table["residual_idm_certified_het"]
    assert het["kind"] == "residual_heterogeneous" and het["device"] == "cpu" and het["recheck"] is True
    assert het["members"] == table["residual_idm_certified"]["members"]
    assert het["cores"].startswith("data/calibration/ngsim_i80/idm_event_margin0.2_") and het["cores"].endswith(
        "_certified.json")
    assert table["idm_global_p0"]["kind"] == "idm" and "members" not in table["idm_global_p0"]
    assert table["idm_global_p0"]["calibration"].startswith("data/calibration/ngsim_i80_p0/")
    expected = {
        "residual_idm_certified_p0": "runs/m8_temporal_stable_ft/ngsim_i80_p0/residual_idm/driver_fold{fold}_seed0",
        "mlp_p0": "runs/m8_temporal_ft/ngsim_i80_p0/mlp/driver_fold{fold}_seed0",
        "gru_p0": "runs/m8_temporal_ft/ngsim_i80_p0/gru/driver_fold{fold}_seed0",
    }
    for name, members in expected.items():
        assert table[name]["kind"] == "models" and table[name]["members"] == members, name
    assert table["gru_p0"]["device"] == table["gru"]["device"] and table["mlp_p0"]["device"] == table["mlp"]["device"]


def test_residual_het_run_in_a_child_process(het_tree):
    """A short run of the heterogeneous hybrid through scripts/run_corridor.py on a synthetic scenario (libsumo in the
    child): every vehicle inserted, no collision, and the members of the vehicles those of the fold members' law."""
    root, members, spec = het_tree
    laws = root / "laws"
    table = {"het": spec, "ref": {"kind": "models", "device": "cpu", "members": spec["members"]}}
    export_laws({"folds": [0, 1], "table": table}, laws)
    frames = []
    for lane in range(1, 7):
        for j in range(3):
            n = 300
            x = 20.2 + 12.0 * 0.1 * np.arange(n)
            frames.append(pd.DataFrame({"track_id": f"p0_{lane * 10 + j}", "period": 0, "vehicle_id": lane * 10 + j,
                                        "frame_id": 1 + 80 * j + 5 * lane + np.arange(n), "lane_id": lane, "x": x,
                                        "v": 12.0, "length": 4.5, "v_class": 2}))  # fmt: skip
    build_scenario(pd.concat(frames, ignore_index=True), {**I80, "name": "syn"}, 0, root / "scenarios" / "syn_p0")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    quoted = lambda p: f"'{p.as_posix()}'"  # noqa: E731  (hydra: non-ASCII paths quoted)
    vehicles = {}
    for name in ("het", "ref"):
        args = [sys.executable, str(REPO_ROOT / "scripts" / "run_corridor.py"), "scenario=syn_p0", f"law={name}",
                "seed=4", "sim.tail_s=15.0", f"paths.scenarios_root={quoted(root / 'scenarios')}",
                f"paths.laws_root={quoted(laws)}", f"paths.runs_root={quoted(root / 'runs_corridor')}",
                f"hydra.run.dir={quoted(root / 'outputs' / name)}"]  # fmt: skip
        proc = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=600)  # fmt: skip
        assert proc.returncode == 0, proc.stdout + proc.stderr
        run_dir = root / "runs_corridor" / "syn_p0" / name / "seed4"
        run = read_json(run_dir / "run.json")
        assert run["n_inserted"] == run["n_planned"] == 18 and run["n_collisions"] == 0 and run["n_teleports"] == 0
        with np.load(run_dir / "vehicles.npz") as data:
            vehicles[name] = data["member"].copy()
    np.testing.assert_array_equal(vehicles["het"], vehicles["ref"])
