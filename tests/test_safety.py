"""Safety filter of the corridor (D130, cf_stability/corridor/safety.py) and its hook in the loop
(cf_stability/corridor/loop.py, behind sim.safety_filter of configs/run_corridor.yaml).

The loop runs in child processes on the synthetic scenario of tests/test_corridor_loop.py (libsumo is never imported
here); the band equilibria are those of the stored audits of the members of the four filtered laws (skipped where the
run directories are not on the machine)."""

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from cf_stability.corridor.safety import (
    LAW_SUFFIX,
    SafetyConfig,
    apply,
    barrier,
    bound,
    check_law_name,
    simulate_pair,
)
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

CFG = SafetyConfig()
FILTERED_LAWS = ("mlp", "gru", "lstm", "residual_idm_certified")
HASHED = ("scenario", "law", "seed", "sim", "device", "scenario_hash", "law_hash")


def test_defaults_and_mapping():
    assert dataclasses.asdict(CFG) == {"s_min": 2.0, "tau_s": 0.5, "b": 4.0, "alpha": 1.0}
    assert SafetyConfig.from_mapping(None) is None and SafetyConfig.from_mapping(False) is None
    assert SafetyConfig.from_mapping(True) == CFG
    assert SafetyConfig.from_mapping({"enabled": False}) is None
    assert SafetyConfig.from_mapping({"alpha": 2}).alpha == 2.0
    with pytest.raises(ValueError):
        SafetyConfig.from_mapping({"gain": 1.0})
    with pytest.raises(ValueError):
        SafetyConfig.from_mapping("on")


def test_clip_equals_the_analytic_bound():
    rng = np.random.default_rng(0)
    n = 2000
    s, v = rng.uniform(0.5, 60.0, n), rng.uniform(0.0, 30.0, n)
    dv = rng.uniform(-6.0, np.minimum(6.0, v), n)  # speeds of the loop: the leader's is never negative
    v_lead = v - dv
    h = s - 2.0 - 0.5 * v - np.maximum(0.0, v**2 - v_lead**2) / 8.0
    expected = ((v_lead - v) + 1.0 * h) / (0.5 + np.where(v > v_lead, v / 4.0, 0.0))
    np.testing.assert_allclose(barrier(s, dv, v, CFG), h, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(bound(s, dv, v, CFG), expected, rtol=1e-13, atol=1e-12)
    acc = rng.uniform(-8.0, 4.0, n)
    filtered, active = apply(acc, s, dv, v, CFG)
    limit = bound(s, dv, v, CFG)
    np.testing.assert_array_equal(filtered, np.minimum(acc, limit))  # the clip is the bound exactly
    np.testing.assert_array_equal(active, limit < acc)
    np.testing.assert_allclose(filtered, np.minimum(acc, expected), rtol=1e-13, atol=1e-12)
    # at the bound the barrier condition holds with equality under a constant leader speed: dh/dt + alpha h = 0
    eps = 1e-5
    away = np.abs(v - v_lead) > 1e-3  # away from the switch of max(0, .)

    def h_at(step: float, a: np.ndarray) -> np.ndarray:  # h after `step` seconds of the acceleration a
        return barrier(s + step * (v_lead - v), (v + step * a) - v_lead, v + step * a, CFG)

    for a, sign in ((expected, 0.0), (expected - 1.0, 1.0)):
        rate = (h_at(eps, a) - h_at(-eps, a)) / (2.0 * eps) + CFG.alpha * h  # central difference
        if sign == 0.0:
            np.testing.assert_allclose(rate[away], 0.0, atol=1e-6)
        else:
            assert (rate[away] > 0).all()  # below the bound the condition holds strictly


def test_inactive_without_leader_and_at_equilibria():
    v = np.linspace(0.0, 35.0, 36)
    no_leader = bound(np.full_like(v, 50.0), np.zeros_like(v), v, CFG)  # the loop's state without a leader in range
    assert (no_leader > 4.0).all()  # above the clip a_max of the loop
    # at an equilibrium (s, 0, v) with a law acceleration 0: active exactly when h <= 0
    s = 2.0 + 0.5 * v + np.array([-0.5, 0.5] * 18)
    filtered, active = apply(np.zeros_like(v), s, np.zeros_like(v), v, CFG)
    np.testing.assert_array_equal(active, barrier(s, 0.0, v, CFG) < 0)


def _band_equilibria(law: str) -> list[tuple[float, float, Path]]:
    payload = read_json(REPO_ROOT / "runs" / "corridor" / "laws" / f"{law}.json")
    out = []
    for member in payload["members"]:
        audit = read_json(REPO_ROOT / member["run"] / "stability.json")["audit"]
        out += [(r["s"], r["v"], REPO_ROOT / member["model"]) for r in audit["equilibria"]
                if r["status"] in ("ok", "multiple") and r["in_band"]]  # fmt: skip
    return out


@pytest.mark.parametrize("law", FILTERED_LAWS)
def test_inactive_at_the_band_equilibria_of_the_filtered_laws(law):
    """h > 0 and the bound above the law's acceleration at every band equilibrium of the members' audits, and at the
    states 0.25 m and 0.1 m/s around it: the filter leaves the audited linearisation unchanged."""
    law_file = REPO_ROOT / "runs" / "corridor" / "laws" / f"{law}.json"
    if not law_file.exists():
        pytest.skip(f"{law_file} is not on this machine")
    try:
        records = _band_equilibria(law)
    except FileNotFoundError as exc:
        pytest.skip(f"member audit missing: {exc}")
    import torch

    from cf_stability.models import load_model
    from cf_stability.stability.equilibrium import steady_acc

    assert records
    s = np.array([r[0] for r in records])
    v = np.array([r[1] for r in records])
    assert (barrier(s, np.zeros_like(s), v, CFG) > 0).all()
    for path in sorted({r[2] for r in records}):
        mine = [r for r in records if r[2] == path]
        model = load_model(path).double().eval()
        for ds, ddv in ((0.0, 0.0), (-0.25, 0.0), (0.25, 0.0), (0.0, 0.1), (0.0, -0.1)):
            s_k = torch.tensor([r[0] + ds for r in mine], dtype=torch.float64)
            v_k = torch.tensor([r[1] for r in mine], dtype=torch.float64)
            dv_k = torch.full_like(s_k, ddv)
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                acc = steady_acc(model, s_k, v_k, dv_k).numpy()
            _, active = apply(acc, s_k.numpy(), dv_k.numpy(), v_k.numpy(), CFG)
            assert not active.any(), (law, path, ds, ddv)


def braking(v0: float, b_lead: float, pre_s: float = 5.0, total_s: float = 80.0, v_end: float = 0.0) -> np.ndarray:
    t = np.arange(0.0, total_s + 0.05, 0.1)
    return np.where(t < pre_s, v0, np.maximum(v_end, v0 - b_lead * (t - pre_s)))


RECKLESS = {
    "accelerating": lambda s, dv, v: np.full_like(s, 2.0),
    "full throttle": lambda s, dv, v: np.full_like(s, 4.0),
    "cruising": lambda s, dv, v: np.zeros_like(s),
}


@pytest.mark.parametrize("name", RECKLESS)
def test_braking_leader_never_produces_a_contact(name):
    """Two-vehicle toy with the arithmetic of the loop: a follower whose law ignores the leader, started anywhere
    with h >= 0, behind a leader that brakes at 2 m/s^2 (the braking pulse of the platoon test) down to a lower
    speed or to a standstill: the filtered follower never touches it, the unfiltered one does."""
    law = RECKLESS[name]
    gaps = []
    for v0 in np.arange(1.0, 35.0, 2.0):
        for extra in (0.0, 0.5, 2.0, 10.0, 30.0):
            for v_end in (0.0, 0.5 * v0):
                run = simulate_pair(law, braking(v0, 2.0, v_end=v_end), 2.0 + 0.5 * v0 + extra, v0, CFG)
                gaps.append(run["s"].min())
    assert min(gaps) > 0.0
    unfiltered = simulate_pair(law, braking(20.0, 2.0), 12.0, 20.0, None)
    assert unfiltered["s"].min() <= 0.0


def test_law_names_of_filtered_runs():
    check_law_name("mlp", None)
    check_law_name("mlp" + LAW_SUFFIX, CFG)
    with pytest.raises(ValueError):
        check_law_name("mlp", CFG)  # would replace the unfiltered runs of mlp
    with pytest.raises(ValueError):
        check_law_name("mlp" + LAW_SUFFIX, None)


def test_the_key_of_the_run_config_is_off_by_default():
    sim = yaml.safe_load((REPO_ROOT / "configs" / "run_corridor.yaml").read_text(encoding="utf-8"))["sim"]
    assert sim["safety_filter"] is None


# ------------------------------------------------------------------------------- the hook in the loop (SUMO)


def _loop_tests():
    path = REPO_ROOT / "tests" / "test_corridor_loop.py"
    spec = importlib.util.spec_from_file_location("corridor_loop_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAILGATER = {"v0": 15.0, "T": 0.1, "s0": 0.5, "a": 3.0, "b": 3.0}  # equilibrium gap 2.2 m at 12 m/s: h < 0


@pytest.fixture(scope="module")
def filter_corridor(tmp_path_factory):
    from cf_stability.corridor.scenario import build_scenario

    loop_tests = _loop_tests()
    root = tmp_path_factory.mktemp("safety")
    build_scenario(loop_tests.synthetic_tracks(), {**loop_tests.I80, "name": "syn"}, 0, root / "scenarios" / "syn_p0")
    for name in ("tail", "tail" + LAW_SUFFIX):
        write_json(root / "laws" / f"{name}.json", {"law": name, "kind": "idm", "device": "cpu", "members": [],
                                                    "params": [TAILGATER], "support_v": None, "window": 1})  # fmt: skip
    return root, loop_tests


def _run(root: Path, runs: str, *overrides: str) -> subprocess.CompletedProcess:
    quoted = lambda p: f"'{p.as_posix()}'"  # noqa: E731
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_corridor.py"), *overrides, "sim.tail_s=20.0",
        f"paths.scenarios_root={quoted(root / 'scenarios')}", f"paths.laws_root={quoted(root / 'laws')}",
        f"paths.runs_root={quoted(root / runs)}", f"hydra.run.dir={quoted(root / 'outputs' / runs)}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=900)  # fmt: skip


def test_filter_hook_in_the_loop(filter_corridor):
    root, loop_tests = filter_corridor
    plain = _run(root, "runs", "scenario=syn_p0", "law=tail", "seed=1")
    assert plain.returncode == 0, plain.stdout + plain.stderr
    filtered = _run(root, "runs", "scenario=syn_p0", f"law=tail{LAW_SUFFIX}", "seed=1", "sim.safety_filter=true")
    assert filtered.returncode == 0, filtered.stdout + filtered.stderr
    run_a = read_json(root / "runs" / "syn_p0" / "tail" / "seed1" / "run.json")
    run_b = read_json(root / "runs" / "syn_p0" / f"tail{LAW_SUFFIX}" / "seed1" / "run.json")
    # the default leaves the run as before: no new key in its sim block or its statistics, the hash of its config
    assert "safety_filter" not in run_a["config"]["sim"] and set(run_a) == loop_tests.RUN_KEYS
    assert run_a["config_hash"] == config_hash({key: run_a["config"][key] for key in HASHED})
    # the filtered run: the parameters in the hashed sim block, the counts in the statistics
    assert run_b["config"]["sim"]["safety_filter"] == dataclasses.asdict(CFG)
    assert run_b["config_hash"] == config_hash({key: run_b["config"][key] for key in HASHED})
    assert set(run_b) - set(run_a) == {"safety_filter_steps", "safety_filter_active_steps", "safety_filter_active_share"}
    assert run_b["safety_filter_active_steps"] > 0 and 0 < run_b["safety_filter_active_share"] < 1
    sim_a = {k: v for k, v in run_b["config"]["sim"].items() if k != "safety_filter"}
    assert sim_a == run_a["config"]["sim"]
    with np.load(root / "runs" / "syn_p0" / "tail" / "seed1" / "trajectories.npz") as a, \
            np.load(root / "runs" / "syn_p0" / f"tail{LAW_SUFFIX}" / "seed1" / "trajectories.npz") as b:  # fmt: skip
        assert len(a["x"]) != len(b["x"]) or not np.array_equal(a["x"], b["x"])
    assert run_b["n_collisions"] <= run_a["n_collisions"]
    # a filtered run of a law without the suffix, or an unfiltered one of a law with it, is refused
    for args in (("law=tail", "sim.safety_filter=true"), (f"law=tail{LAW_SUFFIX}",)):
        refused = _run(root, "refused", "scenario=syn_p0", "seed=1", *args)
        assert refused.returncode != 0 and "safety_filter" in (refused.stdout + refused.stderr)
        assert not (root / "refused" / "syn_p0").exists() or not any((root / "refused" / "syn_p0").rglob("run.json"))
