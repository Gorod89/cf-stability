"""Predictive validity of D127 (scripts/analysis/predictive_validity.py) on hand-made laws and synthetic data: the
frozen gain along trajectories, the follower states of a run, the closed forms of parameter sets and the statistics."""

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest
import torch

from cf_stability.corridor.laws import IDMLaw, ModelLaw
from cf_stability.eval.corridor_tables import idm_unstable_share
from cf_stability.eval.stats import stratified_spearman
from cf_stability.models import IDM
from cf_stability.models.idm import idm_equilibrium_partials
from cf_stability.stability.analytic import transfer_discrete
from cf_stability.utils import REPO_ROOT

PARAMS = [{"v0": 30.0, "T": 1.2, "s0": 2.0, "a": 1.2, "b": 2.0}, {"v0": 20.0, "T": 0.6, "s0": 1.5, "a": 0.4, "b": 1.5}]
KEYS = ("v0", "T", "s0", "a", "b")
OMEGA = np.geomspace(0.02, 2.0, 25)


def script():
    path = REPO_ROOT / "scripts" / "analysis" / "predictive_validity.py"
    spec = importlib.util.spec_from_file_location("predictive_validity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pv = script()


def idm_law() -> IDMLaw:
    law = IDMLaw("toy", member_params=np.array([[p[k] for k in KEYS] for p in PARAMS]))
    law.assign(40, np.random.default_rng(0))
    return law


def test_frozen_gain_of_the_idm_at_an_equilibrium_is_the_analytic_gain():
    law = idm_law()
    rows = np.arange(40)
    params = law.params[rows]
    v = np.linspace(5.0, 18.0, 40)
    s_e, f_s, f_dv, f_v = (x.numpy() for x in idm_equilibrium_partials(torch.as_tensor(v), *(torch.as_tensor(
        params[:, k]) for k in range(5))))  # fmt: skip
    f = pv.law_partials(law, rows, s_e, np.zeros_like(v), v)
    np.testing.assert_allclose(f[0], f_s, rtol=1e-9)
    np.testing.assert_allclose(f[1], f_dv, rtol=1e-9)
    np.testing.assert_allclose(f[2], f_v, rtol=1e-9)
    analytic = transfer_discrete(*(torch.as_tensor(x) for x in (f_s, f_dv, f_v)), torch.as_tensor(OMEGA), 0.1).abs()
    np.testing.assert_allclose(pv.frozen_gain(*f), analytic.max(dim=1).values.numpy(), rtol=1e-9)
    # the exact gain of the corridor tables gives the same unstable share for a member's speeds
    share = idm_unstable_share({k: np.array([PARAMS[1][k]]) for k in KEYS}, [5.0, 18.0], [float(x) for x in
                               range(5, 19)], OMEGA, 0.1, 1.02)  # fmt: skip
    one = IDMLaw("one", member_params=np.array([[PARAMS[1][k] for k in KEYS]]))
    one.assign(14, np.random.default_rng(0))
    grid = np.arange(5.0, 19.0)
    s_g = idm_equilibrium_partials(torch.as_tensor(grid), *(PARAMS[1][k] for k in KEYS))[0].numpy()
    gains = pv.frozen_gain(*pv.law_partials(one, np.arange(14), s_g, np.zeros(14), grid))
    assert share["share"] == pytest.approx(np.mean(gains[np.isfinite(gains)] > 1.02))


def test_model_and_heterogeneous_laws_have_the_partials_of_their_vehicle():
    """A law of IDM checkpoints (ModelLaw) gives the partials of the member each vehicle drew; at a non-equilibrium
    state (dv != 0) too."""
    models = [IDM(p, dtype=torch.float64) for p in PARAMS]
    law = ModelLaw("toy_models", models, "cpu")
    law.assign(30, np.random.default_rng(3))
    reference = idm_law()
    reference.params = np.array([[PARAMS[m][k] for k in KEYS] for m in law.member])
    rows = np.arange(30)
    rng = np.random.default_rng(1)
    s, dv, v = rng.uniform(5, 40, 30), rng.uniform(-2, 2, 30), rng.uniform(2, 18, 30)
    for got, expected in zip(pv.law_partials(law, rows, s, dv, v), pv.law_partials(reference, rows, s, dv, v)):
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_follower_states_of_a_run():
    # lane 1: vehicles 0 (x 10), 1 (x 30), 2 (x 100); lane 2: vehicle 3 (x 20) ahead of nobody; second t = 0 and 1
    traj = {"t": np.array([0, 0, 0, 0, 1, 1], dtype=np.float32), "vehicle": np.array([0, 1, 2, 3, 0, 1]),
            "x": np.array([10.0, 30.0, 100.0, 20.0, 20.0, 21.5], dtype=np.float32),
            "lane": np.array([1, 1, 1, 2, 1, 1], dtype=np.int8), "v": np.array([10.0, 8.0, 12.0, 5.0, 9.0, 8.0])}  # fmt: skip
    lengths = np.array([4.0, 5.0, 4.5, 4.0])
    states = pv.follower_states(traj, lengths)
    # t = 0: 0 follows 1 (gap 30 - 5 - 10 = 15), 1 follows 2 at 65.5 m (beyond 50 m: left out); t = 1: gap 21.5-5-20<0
    assert states["vehicle"].tolist() == [0] and states["s"].tolist() == [15.0]
    assert states["dv"].tolist() == [2.0] and states["v"].tolist() == [10.0]


def test_closed_form_values_and_audit_values():
    context = {"box_low": [1.0, -3.0, 0.0], "box_high": [60.0, 3.0, 18.0],
               "band": {"v": [5.0, 18.0], "s_low": [5.0, 15.0], "s_median": [10.0, 30.0], "s_high": [20.0, 60.0]}}  # fmt: skip
    params = np.array([[p[k] for k in KEYS] for p in PARAMS])
    values = pv.closed_form_values(params, context)
    for i, p in enumerate(PARAMS):
        share = idm_unstable_share({k: np.array([p[k]]) for k in KEYS}, [0.0, 18.0], [float(v) for v in range(5, 31)],
                                   OMEGA, 0.1, 1.02)  # fmt: skip
        assert values["unstable_eq"][i] == pytest.approx(share["share"])
        assert values["max_gain"][i] == pytest.approx(share["max_gain"])
    assert values["max_gain_low"][1] <= values["max_gain"][1] + 1e-12
    assert 0.0 <= values["not_stable"][0] <= 1.0
    audit = {"omega": OMEGA.tolist(), "summary": {"band_numerical": {"support": {"stable": 0.25}}}, "equilibria": [
        {"in_support": True, "status": "ok", "gain": [1.0] * 8 + [1.3] + [0.9] * 16},
        {"in_support": True, "status": "outside", "gain": [None] * 12 + [2.0] * 13},
        {"in_support": False, "status": "ok", "gain": [5.0] * 25},
        {"in_support": True, "status": "none", "gain": None},
    ]}  # fmt: skip
    got = pv.audit_values(audit)
    assert got == {"not_stable": 0.75, "max_gain": 2.0, "max_gain_low": 1.3}  # the 9th frequency is 0.0928 rad/s


def test_transient_growth_and_member_means():
    assert pv.transient_growth({"speed_std": [1.0, 0.5, 0.8, 0.4], "min_gap": [None, 3.0, 2.0, 1.0]}) == pytest.approx(1.6)
    assert pv.transient_growth({"speed_std": [1.0, 0.5, 0.4], "min_gap": [None, 3.0, 0.0]}) == math.inf
    assert pv.transient_growth({"speed_std": [1.0, 0.5, 0.4], "collided": True}) == math.inf
    assert math.isnan(pv.transient_growth({"speed_std": [1.0, 0.0, 0.4]}))
    assert pv.law_mean([1.0, 2.0, math.nan]) == pytest.approx(1.5)
    assert pv.law_mean([1.0, math.inf]) == math.inf
    assert math.isnan(pv.law_mean([math.nan]))


def test_paired_difference_and_rank_check():
    rng = np.random.default_rng(0)
    laws = [f"law{k}" for k in range(15)]
    frame = pd.DataFrame([{"law": law, "corridor": c} for law in laws for c in ("I-80", "US-101")])
    frame["macro_error"] = rng.uniform(0, 1, len(frame))
    strata, clusters = frame["corridor"].to_numpy(), frame["law"].to_numpy()
    y = frame["macro_error"].to_numpy()
    draws = rng.integers(0, len(laws), size=(500, len(laws)))
    same = pv.paired_difference(y, y, y, strata, clusters, draws, laws)
    assert same["difference"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0
    noise = rng.uniform(0, 1, len(frame))
    better = pv.paired_difference(y ** 2, noise, y, strata, clusters, draws, laws)  # a monotone transform of y
    assert better["difference"] == pytest.approx(1.0 - stratified_spearman(noise, y, strata))
    assert better["low"] > 0
    values = {("E", "I-80"): 0.1, ("D", "I-80"): 0.2, ("E", "US-101"): 0.3, ("D", "US-101"): 0.3}
    assert pv.rank_check(values, ["I-80"], "E", "D") is True
    assert pv.rank_check(values, ["I-80", "US-101"], "E", "D") is False  # a tie is no better rank
    assert pv.rank_check({("E", "I-80"): 0.1}, ["I-80"], "E", "D") is None


def test_correlation_rows_on_synthetic_pairs():
    rng = np.random.default_rng(2)
    laws = [f"law{k}" for k in range(16)] + [pv.LAW_E, pv.LAW_D]
    rows = []
    for law in laws:
        latent = rng.uniform(0, 1)
        for corridor in ("I-80", "US-101"):
            rows.append({"law": law, "corridor": corridor, "macro_error": latent + 0.05 * rng.normal()})
    frame = pd.DataFrame(rows)
    frame.loc[frame["law"] == pv.LAW_E, "macro_error"] = 0.0
    frame.loc[frame["law"] == pv.LAW_D, "macro_error"] = 0.2
    for key in pv.PREDICTORS:
        frame[key] = rng.uniform(0, 1, len(frame))
    frame["frozen_share"] = frame["macro_error"] + 0.01 * rng.normal(size=len(frame))  # a good predictor of a pair
    frame["unstable_eq"] = frame["law"].map({law: rng.uniform(0, 1) for law in laws})  # a predictor of no use
    frame.loc[frame["law"] == pv.LAW_D, "frozen_share"] = 0.5  # D above E on both corridors
    out = {r["predictor"]: r for r in pv.correlation_rows(frame, "synthetic", 400, 999, 0)}
    good = out["frozen_share"]
    assert good["rho"] > 0.9 and good["difference_low"] > 0 and good["rank_e_better_than_d"] is True
    assert good["predicts_better"] is True and out["unstable_eq"]["predicts_better"] is None
    assert good["p"] < 0.01 and good["p_method"] == "within every corridor"
    assert out["max_gain"]["p_method"] == "laws permuted as wholes"
