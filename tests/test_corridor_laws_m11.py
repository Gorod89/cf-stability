"""Corridor laws of M11, agent B (docs/m11_contract.md): the FD-matched member weights of the heterogeneous certified
hybrid (D132, ``residual_idm_certified_het_fd``: kind residual_heterogeneous with ``tilt``) and its fit
(``scripts/analysis/het_fd_tilt.py``), and the two-timescale certified hybrid (D129, ``residual_idm_tt``: kind models)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cf_stability.corridor.laws import IDM_KEYS, ModelLaw, ResidualHetLaw, export_laws, law_fingerprint, load_law, \
    tilt_weights  # fmt: skip
from cf_stability.models import ModelContext, ResidualIDM, ResidualIDMTT, load_model, save_model
from cf_stability.utils import REPO_ROOT, read_json, write_json

LAWS = yaml.safe_load((REPO_ROOT / "configs" / "corridor" / "laws.yaml").read_text(encoding="utf-8"))
# the core of e4_stable_ft fold 0 (margin 0.2): the a priori certificate holds with r_max 0.3
CORE = {"v0": 35.213199615478516, "T": 0.9788587093353271, "s0": 3.193310499191284, "a": 1.5670632123947144,
        "b": 0.5000092387199402}  # fmt: skip
VARIANTS = [{**CORE, "T": CORE["T"] * f, "s0": CORE["s0"] + d} for f in (1.0, 1.1, 1.2) for d in (0.0, 0.5)]
FREE = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a": 0.5, "b": 3.0}  # not certifiable


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"m11_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def residual_member(run: Path, seed: int, lipschitz: float = 0.05) -> ResidualIDM:
    torch.manual_seed(seed)
    model = ResidualIDM(ModelContext(idm_params=CORE), r_max=0.3, lipschitz=lipschitz)
    with torch.no_grad():
        for parameter in model.g.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
        model.train()
        for _ in range(50):  # power iterations of the spectral normalisation for the new weights
            model(torch.zeros(1, 1, 3))
    model.eval()
    run.mkdir(parents=True, exist_ok=True)
    save_model(model, run / "model.pt")
    write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.2], "box_high": [50.0, 3.0, 17.5]}})
    return load_model(run / "model.pt")


@pytest.fixture()
def tilt_tree(tmp_path):
    """Two ResidualIDM members, a cores file with the six certified VARIANTS and FREE (marked certified, fails the
    recheck), and the spec of the uniform law."""
    members = [residual_member(tmp_path / "runs" / f"fold{k}", seed=k) for k in range(2)]
    cores = [{"event_id": f"e{i}", "params": p, "certified": True} for i, p in enumerate(VARIANTS)]
    cores.append({"event_id": "free", "params": FREE, "certified": True})
    write_json(tmp_path / "calibration" / "idm_event_margin0.2_k_certified.json", {
        "kind": "idm_event_margin_certified", "settings": {"folds": [0, 1], "residual_runs": {
            str(k): (tmp_path / "runs" / f"fold{k}").as_posix() for k in range(2)}}, "cores": cores})  # fmt: skip
    spec = {"kind": "residual_heterogeneous", "device": "cpu", "members": (tmp_path / "runs" / "fold{fold}").as_posix(),
            "cores": (tmp_path / "calibration" / "idm_event_margin0.2_*_certified.json").as_posix(), "recheck": True}  # fmt: skip
    return tmp_path, members, spec


def test_tilted_export(tilt_tree):
    root, members, spec = tilt_tree
    out = root / "laws"
    export_laws({"folds": [0, 1], "table": {"het": spec}}, out)
    with np.load(out / "het.npz") as data:
        index = data["index"]
        params = np.stack([data[k] for k in IDM_KEYS], axis=1)
        assert "weight" not in data.files
    assert index.tolist() == list(range(6))  # the variants pass the recheck, FREE does not
    beta = (1.3, -0.4)
    write_json(root / "tilt.json", {"beta_T": beta[0], "beta_s": beta[1], "key": "k1", "cores_index": index.tolist()})
    write_json(root / "tilt_other.json", {"beta_T": 1.0, "beta_s": 0.0, "cores_index": [0, 1, 2]})
    table = {"het": spec, "het_fd": {**spec, "tilt": (root / "tilt.json").as_posix()},
             "other": {**spec, "tilt": (root / "tilt_other.json").as_posix()},
             "none": {**spec, "tilt": (root / "absent.json").as_posix()}}  # fmt: skip
    before = law_fingerprint(out / "het.json")
    lines = export_laws({"folds": [0, 1], "table": table}, out)
    assert law_fingerprint(out / "het.json") == before  # the uniform law keeps its file and fingerprint
    assert "tilted: beta_T +1.3000, beta_s -0.4000" in lines[1], lines[1]
    assert "not written: tilt" in lines[2] and "fitted on other cores" in lines[2]
    assert "not written: tilt" in lines[3] and "no file" in lines[3]
    law = read_json(out / "het_fd.json")
    tilt = law["source"]["tilt"]
    logit = beta[0] * params[:, 1] + beta[1] * params[:, 2]
    expected = np.exp(logit - logit.max())
    expected /= expected.sum()
    assert tilt["beta_T"] == beta[0] and tilt["beta_s"] == beta[1] and tilt["key"] == "k1"
    assert tilt["effective_cores"] == pytest.approx(1.0 / np.sum(expected**2))
    with np.load(out / "het_fd.npz") as data:
        np.testing.assert_allclose(data["weight"], expected, rtol=1e-12)
        assert data["index"].tolist() == index.tolist()
    record, problem = tilt_weights((root / "tilt.json").as_posix(), params, index)
    assert problem is None and np.allclose(record["weight"], expected)
    assert law_fingerprint(out / "het_fd.json") != before


def test_tilted_draws(tilt_tree):
    """Uniform: the draws of M8 (Generator.integers) bit for bit; tilted: the same members for a seed and the cores with
    the frequencies of the weights."""
    root, members, spec = tilt_tree
    table = np.array([[p[k] for k in IDM_KEYS] for p in VARIANTS])
    uniform = ResidualHetLaw("het", members, table, "cpu")
    member = uniform.assign(5000, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    old_member = rng.integers(2, size=5000).astype(np.int8)
    old_rows = rng.integers(len(table), size=5000)
    assert np.array_equal(member, old_member) and np.array_equal(uniform.params, table[old_rows])
    weights = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    tilted = ResidualHetLaw("het_fd", members, table, "cpu", weights * 3.0)  # normalised inside
    assert np.allclose(tilted.weights, weights)
    assert np.array_equal(tilted.assign(5000, np.random.default_rng(3)), member)
    reference = ModelLaw("ref", members, "cpu")
    assert np.array_equal(reference.assign(5000, np.random.default_rng(3)), member)
    n = 200_000
    tilted.assign(n, np.random.default_rng(4))
    rows = np.array([int(np.flatnonzero((table == p).all(axis=1))[0]) for p in tilted.params[:2000]])
    assert set(rows) <= set(range(6))
    counts = np.array([np.sum(np.all(tilted.params == table[i], axis=1)) for i in range(6)])
    assert counts.sum() == n
    sd = np.sqrt(weights * (1 - weights) / n)
    assert np.all(np.abs(counts / n - weights) <= 4 * sd), (counts / n, weights)
    for bad in (np.ones(5), -weights, np.zeros(6)):
        with pytest.raises(ValueError, match="weights"):
            ResidualHetLaw("bad", members, table, "cpu", bad)


def test_load_law_with_weights(tilt_tree):
    root, members, spec = tilt_tree
    out = root / "laws"
    export_laws({"folds": [0, 1], "table": {"het": spec}}, out)
    with np.load(out / "het.npz") as data:
        index = data["index"].tolist()
    write_json(root / "tilt.json", {"beta_T": 2.0, "beta_s": 0.3, "cores_index": index})
    export_laws({"folds": [0, 1], "table": {"het_fd": {**spec, "tilt": (root / "tilt.json").as_posix()}}}, out)
    uniform, _ = load_law(out / "het.json")
    tilted, device = load_law(out / "het_fd.json")
    assert uniform.weights is None and device == "cpu"
    with np.load(out / "het_fd.npz") as data:
        assert np.allclose(tilted.weights, data["weight"])
    assert np.array_equal(uniform.table, tilted.table)
    history = np.stack([np.full(20, 20.0), np.zeros(20), np.linspace(0.0, 15.0, 20)], axis=1)[:, None, :]
    tilted.assign(20, np.random.default_rng(0))
    assert np.isfinite(tilted.accelerations(np.arange(20), history)).all()


def test_m11_laws_in_the_table():
    table = LAWS["table"]
    het, fd = table["residual_idm_certified_het"], table["residual_idm_certified_het_fd"]
    assert fd["kind"] == "residual_heterogeneous" and fd["device"] == het["device"] == "cpu"
    assert {k: v for k, v in fd.items() if k != "tilt"} == het
    assert fd["tilt"] == "data/calibration/ngsim_i80/het_fd_tilt.json"
    tt = table["residual_idm_tt"]
    assert tt["kind"] == "models" and tt["members"] == "runs/e4_tt_ft/ngsim_i80/residual_idm_tt/driver_fold{fold}_seed0"


def test_two_timescale_members_run_as_models(tmp_path):
    """residual_idm_tt is a model with a window: ModelLaw draws its members as for residual_idm_certified and the latent
    runs on each vehicle's history (the acceleration of a vehicle is that of its member on its own window)."""
    context = ModelContext(center=(20.0, 0.0, 10.0), scale=(10.0, 1.0, 5.0), idm_params=CORE)
    members = []
    for k in range(2):
        torch.manual_seed(k)
        model = ResidualIDMTT(context, r_max=0.3, lipschitz=0.1, tau_z=15.0)
        with torch.no_grad():
            model.latent_head.weight.normal_()
            model.latent_head.bias.fill_(1.0)
        run = tmp_path / f"fold{k}"
        run.mkdir()
        save_model(model, run / "model.pt")
        write_json(run / "metrics.json", {"context": {"box_low": [1.0, -3.0, 0.2], "box_high": [50.0, 3.0, 17.5]}})
        members.append(load_model(run / "model.pt"))
    spec = {"kind": "models", "device": "cpu", "members": (tmp_path / "fold{fold}").as_posix()}
    out = tmp_path / "laws"
    lines = export_laws({"folds": [0, 1], "table": {"tt": spec}}, out)
    assert "models: 2/2 members, window 30" in lines[0], lines[0]
    law, _ = load_law(out / "tt.json")
    assert isinstance(law, ModelLaw) and law.window == 30
    member = law.assign(40, np.random.default_rng(5))
    rng = np.random.default_rng(6)
    history = np.stack([rng.uniform(10, 40, (40, 30)), rng.uniform(-1, 1, (40, 30)), rng.uniform(2, 15, (40, 30))], -1)
    got = law.accelerations(np.arange(40), history)
    with torch.no_grad():
        for i in range(40):
            x = torch.as_tensor(history[i : i + 1], dtype=torch.float32)
            assert got[i] == pytest.approx(float(members[member[i]](x)[0]), rel=1e-6, abs=1e-6)
            assert abs(float(members[member[i]].latent(x)[0])) <= members[0].latent_reach()


# ------------------------------------------------------------------------------------------ the FD fit


def test_fd_fit_helpers():
    fit = _load_script("het_fd_tilt")
    params = np.array([[35.0, 1.0, 2.0, 1.5, 1.5], [35.0, 2.0, 4.0, 1.5, 1.5], [35.0, 1.5, 3.0, 1.5, 1.5]])
    w = fit.weights(params, (0.0, 0.0))
    assert np.allclose(w, 1 / 3)
    w = fit.weights(params, (2.0, -1.0))
    logit = 2.0 * params[:, 1] - params[:, 2]
    assert np.allclose(w, np.exp(logit) / np.exp(logit).sum())
    geometry = {"x_in": 20.0, "x_out": 500.0, "edges": [{"x0": 20.0, "x1": 100.0, "lanes": 6},
                {"x0": 100.0, "x1": 204.0, "lanes": 7}, {"x0": 204.0, "x1": 500.0, "lanes": 6}]}  # fmt: skip
    lanes, cells = fit.lane_count(geometry, 100.0)
    assert cells == pytest.approx([6.0, 7.0, 6.04, 6.0, 6.0]) and lanes == pytest.approx(31.04 / 5)
    # one core, no residual: the flow at a density is k v with the equilibrium spacing of the IDM
    speeds = np.arange(0.0, 25.01, 0.1)
    gaps, _ = fit.equilibrium_gaps(params[:1], None, speeds)
    v0, T, s0 = params[0, :3]
    expected = (s0 + speeds * T) / np.sqrt(1 - (speeds / v0) ** 4)
    assert np.allclose(gaps[0], expected)
    density = np.array([300.0, 450.0])
    q = fit.mixture_flow(gaps, np.ones(1), speeds, 5.0, 6.0, density)
    target = 6.0 * 1000.0 / density
    v = np.interp(target, gaps[0] + 5.0, speeds)
    assert np.allclose(q, density * v * 3.6)


def test_fd_fit_equilibria_with_a_residual(tmp_path):
    """The equilibrium gap with a member's residual solves idm(s, 0, v) + r(s, 0, v) = 0."""
    fit = _load_script("het_fd_tilt")
    member = residual_member(tmp_path / "m", seed=0, lipschitz=0.2)
    params = np.array([[CORE[k] for k in IDM_KEYS], [VARIANTS[3][k] for k in IDM_KEYS]])
    speeds = np.arange(0.0, 20.01, 0.5)
    gaps, info = fit.equilibrium_gaps(params, [member], speeds)
    assert max(info["max_step"]) < 1e-7
    from cf_stability.models.idm import idm_acc

    m = member.double()
    for i in range(2):
        s = torch.as_tensor(gaps[i], dtype=torch.float64)
        v = torch.as_tensor(speeds, dtype=torch.float64)
        state = torch.stack((s, torch.zeros_like(s), v), -1)[:, None, :]
        with torch.no_grad():
            f = idm_acc(s, torch.zeros_like(s), v, *(float(params[i, j]) for j in range(5))) + m.residual(state)
        assert f.abs().max() < 1e-6


def test_fd_fit_recovers_the_tilt():
    """Flows generated by a mixture with known betas are fitted back from the uniform start (cores alone)."""
    fit = _load_script("het_fd_tilt")
    rng = np.random.default_rng(0)
    n = 400
    params = np.stack([rng.uniform(32, 40, n), rng.uniform(0.6, 2.5, n), rng.uniform(1.0, 5.0, n),
                       rng.uniform(1.0, 3.0, n), rng.uniform(0.5, 2.0, n)], axis=1)  # fmt: skip
    speeds = np.arange(0.0, 25.01, 0.1)
    gaps, _ = fit.equilibrium_gaps(params, None, speeds)
    density = np.arange(235.0, 606.0, 10.0)
    truth = fit.mixture_flow(gaps, fit.weights(params, (1.2, -0.6)), speeds, 5.0, 6.2, density)

    def objective(beta):
        return fit.fd_distance(fit.mixture_flow(gaps, fit.weights(params, beta), speeds, 5.0, 6.2, density), truth,
                               float(truth.mean()))[0]  # fmt: skip

    beta, info = fit.fit(objective, (0.0, 0.0))
    assert info["success"] and objective(beta) < 1e-6 * objective(np.zeros(2))
    assert beta == pytest.approx([1.2, -0.6], abs=0.05)
    squares, rel = fit.fd_distance(truth + 10.0, truth, 100.0)
    assert squares == pytest.approx(100.0 * len(truth)) and rel == pytest.approx(0.1)
    assert math.isfinite(objective(np.array([5.0, -5.0])))
