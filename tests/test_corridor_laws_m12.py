"""The corridor law of a stateful model (docs/m12_contract.md, D135): the law file of ``residual_idm_tt2`` members, its
fingerprint, ``ModelLaw`` with one state per vehicle, and the corridor loop on the synthetic scenario of
``tests/test_corridor_loop.py`` with and without the persistent latent (the loop test needs the eclipse-sumo package and
is skipped without it, as ``tests/test_safety.py``)."""

import importlib.util

import numpy as np
import pytest
import torch

from cf_stability.corridor.laws import ModelLaw, export_laws, law_fingerprint, load_law
from cf_stability.models import IDM, ResidualIDMTT2, save_model
from cf_stability.models.base import ModelContext
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

CORE = {"v0": 15.0, "T": 1.0, "s0": 2.0, "a": 1.5, "b": 2.0}  # the STABLE law of tests/test_corridor_loop.py
SUPPORT = {"context": {"box_low": [1.0, -3.0, 0.0], "box_high": [50.0, 3.0, 18.0]}}


def tt2_member(seed: int) -> ResidualIDMTT2:
    """A tt2 model on the core above whose latent moves fast (tau_z 2 s) to a target that depends on the window."""
    torch.manual_seed(seed)
    model = ResidualIDMTT2(ModelContext(center=(20.0, 0.0, 10.0), scale=(10.0, 1.0, 4.0), idm_params=CORE), r_max=0.3,
                           lipschitz=0.2, tau_z=2.0)  # fmt: skip
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (*model.latent_rnn.parameters(), *model.latent_head.parameters()):
            p.add_(torch.randn(p.shape, generator=gen))
        model.latent_head.bias.add_(2.0)  # targets near the top of the range: T up to 1.5 T_c
    return model.eval()


@pytest.fixture(scope="module")
def members(tmp_path_factory):
    root = tmp_path_factory.mktemp("m12_members")
    for k in range(2):
        for kind, model in (("tt2", tt2_member(k)), ("idm", IDM(CORE))):
            run = root / kind / f"fold{k}"
            run.mkdir(parents=True)
            save_model(model, run / "model.pt")
            write_json(run / "metrics.json", SUPPORT)
    return root


def test_law_file_of_stateful_members(members, tmp_path):
    table = {
        "tt2": {"kind": "models", "device": "cpu", "members": (members / "tt2" / "fold{fold}").as_posix()},
        "tt2_window": {"kind": "models", "device": "cpu", "members": (members / "tt2" / "fold{fold}").as_posix(),
                       "stateful": False},
        "plain": {"kind": "models", "device": "cpu", "members": (members / "idm" / "fold{fold}").as_posix()},
    }  # fmt: skip
    lines = export_laws({"folds": [0, 1], "table": table}, tmp_path)
    assert all("2/2 members" in line for line in lines)
    tt2, window, plain = (read_json(tmp_path / f"{name}.json") for name in ("tt2", "tt2_window", "plain"))
    assert tt2["stateful"] is True and window["stateful"] is False and "stateful" not in plain  # others unchanged
    assert tt2["window"] == 30
    fingerprints = {name: law_fingerprint(tmp_path / f"{name}.json") for name in ("tt2", "tt2_window")}
    assert fingerprints["tt2"] != fingerprints["tt2_window"]  # stateful enters the fingerprint (the run's law_hash)
    law, device = load_law(tmp_path / "tt2.json")
    assert isinstance(law, ModelLaw) and law.stateful and device == "cpu"
    law_w, _ = load_law(tmp_path / "tt2_window.json")
    assert not law_w.stateful
    law_p, _ = load_law(tmp_path / "plain.json")
    assert not law_p.stateful


def test_model_law_carries_the_latent_of_every_vehicle(members, tmp_path):
    """Two vehicles of the same member, one inserted later: each has its own latent, from z = 0 at its first step; the
    same calls without the state give the window-restarted latent of D129, other accelerations."""
    model = tt2_member(0)
    law = ModelLaw("tt2", [model], "cpu", stateful=True)
    law.assign(3, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    base = np.array([[25.0, 0.0, 10.0]])
    histories = [np.repeat(base + rng.normal(0.0, [1.0, 0.2, 0.2], size=(1, 3)), 30, axis=0) for _ in range(40)]
    z_ref = {0: torch.zeros(1), 2: torch.zeros(1)}
    for step, h in enumerate(histories):
        rows = np.array([0, 2]) if step >= 10 else np.array([0])  # vehicle 2 enters at step 10
        batch = np.stack([h] * len(rows))
        acc = law.accelerations(rows, batch)
        x = torch.from_numpy(batch.astype(np.float32))
        for i, row in enumerate(rows.tolist()):
            with torch.no_grad():
                ref, z_ref[row] = model(x[i : i + 1], z_ref[row])
            assert acc[i] == pytest.approx(float(ref), abs=1e-6)
    assert float(law.states[0][0]) == pytest.approx(float(z_ref[0]), abs=1e-7)
    assert float(law.states[0][2]) == pytest.approx(float(z_ref[2]), abs=1e-7)
    assert float(law.states[0][0]) != float(law.states[0][2]) and float(law.states[0][1]) == 0.0  # vehicle 1: none
    plain = ModelLaw("tt2", [model], "cpu", stateful=False)
    plain.assign(3, np.random.default_rng(0))
    batch = np.stack([histories[-1]] * 2)
    assert not np.allclose(plain.accelerations(np.array([0, 2]), batch), law.accelerations(np.array([0, 2]), batch))


def _loop_tests():
    path = REPO_ROOT / "tests" / "test_corridor_loop.py"
    spec = importlib.util.spec_from_file_location("corridor_loop_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stateful_corridor(tmp_path_factory, members):
    pytest.importorskip("sumo", reason="the corridor loop needs the eclipse-sumo package (not installed in the CI)")
    from cf_stability.corridor.scenario import build_scenario

    loop_tests = _loop_tests()
    root = tmp_path_factory.mktemp("m12_corridor")
    build_scenario(loop_tests.synthetic_tracks(), {**loop_tests.I80, "name": "syn"}, 0, root / "scenarios" / "syn_p0")
    table = {
        "tt2": {"kind": "models", "device": "cpu", "members": (members / "tt2" / "fold{fold}").as_posix()},
        "tt2_window": {"kind": "models", "device": "cpu", "members": (members / "tt2" / "fold{fold}").as_posix(),
                       "stateful": False},
    }  # fmt: skip
    export_laws({"folds": [0, 1], "table": table}, root / "laws")
    return root, loop_tests


def test_corridor_loop_carries_the_state_per_vehicle(stateful_corridor):
    root, loop_tests = stateful_corridor
    runs = {}
    for law in ("tt2", "tt2_window"):
        lines = loop_tests.run_script(root, "runs", "scenario=syn_p0", f"law={law}", "seed=3")
        assert len(lines) == 1 and lines[0].startswith(f"RUN syn_p0/{law}/seed3: inserted"), lines
        runs[law] = loop_tests.outputs(root / "runs" / "syn_p0" / law / "seed3")
    (run_a, traj_a, veh_a), (run_b, traj_b, veh_b) = runs["tt2"], runs["tt2_window"]
    for run, law in ((run_a, "tt2"), (run_b, "tt2_window")):  # the hash of the run config holds the law fingerprint
        assert run["config_hash"] == config_hash({key: run["config"][key] for key in loop_tests.HASHED})
        assert run["config"]["law_hash"] == law_fingerprint(root / "laws" / f"{law}.json")
    assert run_a["config_hash"] != run_b["config_hash"] and run_a["config"]["law_hash"] != run_b["config"]["law_hash"]
    assert np.array_equal(veh_a["member"], veh_b["member"])  # the same members for the seed
    assert len(traj_a["x"]) != len(traj_b["x"]) or not np.allclose(traj_a["x"], traj_b["x"])  # persistence matters
    assert run_a["n_inserted"] == run_b["n_inserted"] == loop_tests.N_SYN
