"""IDM calibration on synthetic events generated with known parameters."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import differential_evolution

from cf_stability.data.schema import DT, Event
from cf_stability.models.idm import IDM_BOUNDS, IDM_PARAM_NAMES, idm_acc, idm_equilibrium_spacing
from cf_stability.models.idm import idm_margin
from cf_stability.train.calibration import (
    MARGIN_COLUMNS,
    MARGIN_TOLERANCE,
    CalibrationConfig,
    calibrate_global,
    calibrate_per_event,
    evaluate_idm,
    idm_margin_min,
    idm_objective,
    idm_stability_shortfall,
    metrics_from_parts,
    parameter_spread,
)
from cf_stability.train.closed_loop import closed_loop_metrics, pad_events, rollout_memoryless

TRUE = [
    {"v0": 33.0, "T": 1.2, "s0": 2.0, "a": 1.2, "b": 1.8},
    {"v0": 25.0, "T": 1.6, "s0": 3.0, "a": 0.9, "b": 1.4},
    {"v0": 30.0, "T": 0.9, "s0": 1.5, "a": 1.8, "b": 2.5},
    {"v0": 28.0, "T": 1.4, "s0": 2.5, "a": 1.5, "b": 2.0},
    {"v0": 22.0, "T": 2.0, "s0": 4.0, "a": 0.8, "b": 1.2},
    {"v0": 35.0, "T": 1.0, "s0": 1.0, "a": 2.2, "b": 3.0},
]
LENGTHS = [260, 200, 300, 220, 280, 240]
CPU = {"device": "cpu"}


def leader_speed(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two superposed oscillations and a stop-and-go episode (3 s standstill)."""
    t = DT * np.arange(n)
    v = 14.0 + 3.0 * np.sin(2 * np.pi * t / rng.uniform(8, 14) + rng.uniform(0, 2 * np.pi))
    v += 2.0 * np.sin(2 * np.pi * t / rng.uniform(3, 5))
    z = np.clip((np.abs(t - rng.uniform(0.35, 0.5) * t[-1]) - 1.5) / 8.0, 0.0, 1.0)
    return v * z * z * (3.0 - 2.0 * z)


def synthetic_event(i: int, params: dict, n: int = 250, noise: tuple[float, float] = (0.0, 0.0)) -> Event:
    rng = np.random.default_rng(i)
    v_lead = leader_speed(n, rng)
    x_lead = 40.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    p = [params[k] for k in IDM_PARAM_NAMES]
    v0 = torch.tensor(v_lead[0])
    res = rollout_memoryless(
        lambda s, dv, v: idm_acc(s, dv, v, *p), torch.tensor(x_lead), torch.tensor(v_lead),
        idm_equilibrium_spacing(v0, *p[:3]), v0,
    )  # fmt: skip
    s = res.s.numpy() + noise[0] * rng.standard_normal(n)
    v = np.clip(res.v.numpy() + noise[1] * rng.standard_normal(n), 0.0, None)
    ev = Event(
        event_id=f"syn/a/{i}|L{i}|0", dataset="syn", site="a", follower_id=f"syn/a/{i}", leader_id=f"syn/a/L{i}",
        t=DT * np.arange(n), s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead,
        x_follower=x_lead - s,
    )  # fmt: skip
    ev.validate()
    return ev


@pytest.fixture(scope="module")
def clean_events() -> list[Event]:
    return [synthetic_event(i, p, n) for i, (p, n) in enumerate(zip(TRUE, LENGTHS))]


@pytest.fixture(scope="module")
def per_event(clean_events) -> pd.DataFrame:
    return calibrate_per_event(clean_events, CalibrationConfig(maxiter=100, **CPU))


def test_per_event_calibration_fits(clean_events, per_event):
    assert list(per_event.columns) == [
        "event_id", "dataset", "site", "follower_id", *IDM_PARAM_NAMES, "objective", "nrmse_s", "nrmse_v",
        "rmse_s", "rmse_v", "collided", "converged", "n_generations",
        *(f"at_bound_{k}" for k in IDM_PARAM_NAMES), "n_samples",
    ]  # fmt: skip
    assert per_event.event_id.tolist() == [ev.event_id for ev in clean_events]
    assert per_event.n_samples.tolist() == LENGTHS
    assert (per_event.objective < 0.01).all()
    assert np.allclose(per_event.objective, per_event.nrmse_s + per_event.nrmse_v)
    assert not per_event.collided.any() and not per_event[[f"at_bound_{k}" for k in IDM_PARAM_NAMES]].any().any()
    # noise-free: J goes to 0, so only the absolute tolerance can stop the search
    assert per_event.converged.all() and (per_event.n_generations < 100).all()
    true = pd.DataFrame(TRUE)
    assert np.allclose(per_event[["T", "s0"]], true[["T", "s0"]], rtol=0.05)


def test_fixed_parameters_are_not_estimated(clean_events):
    events = clean_events[:3]
    fixed = {"v0": 30.0, "b": 2.0}
    cfg = CalibrationConfig(maxiter=100, restarts=1, fixed=fixed, **CPU)
    assert cfg.free_index == [1, 2, 3]
    df = calibrate_per_event(events, cfg)
    assert (df.v0 == 30.0).all() and (df.b == 2.0).all()
    assert not df[["at_bound_v0", "at_bound_b"]].any().any()
    # the fit is the best one among the vectors with these two values: never better than the free fit,
    # and equal to the objective evaluated at the stored vector
    free = calibrate_per_event(events, CalibrationConfig(maxiter=100, restarts=1, **CPU))
    assert (df.objective >= free.objective - 1e-6).all()
    assert np.allclose(evaluate_idm(events, df, **CPU).objective, df.objective, rtol=1e-12, atol=0)
    # fixing the two parameters at their true values recovers the other three
    exact = calibrate_per_event(events[:1], CalibrationConfig(maxiter=150, fixed={k: TRUE[0][k] for k in fixed}, **CPU))
    assert np.allclose(exact[["T", "s0", "a"]], pd.DataFrame(TRUE[:1])[["T", "s0", "a"]], rtol=0.02)
    spread = parameter_spread(df, fixed=fixed)
    assert spread["fixed"] == fixed and set(spread["parameters"]) == {"T", "s0", "a"}
    assert set(spread["correlation"]) == {"T", "s0", "a"}
    with pytest.raises(ValueError):
        CalibrationConfig(fixed={"x": 1.0})


def test_per_event_calibration_with_a_stability_margin(clean_events, per_event):
    """D118: the term of D72 in the objective of every event. The minimised objective is the fit plus 10 x the
    summed shortfall of the margin over 5..30 m/s; the fit is never better than the free fit of the event;
    the reported margin is the closed-form minimum over the speeds (-10 without equilibrium)."""
    events, margin = clean_events[:3], 0.2
    df = calibrate_per_event(events, CalibrationConfig(maxiter=100, **CPU), margin=margin)
    columns = list(per_event.columns)
    at = columns.index("objective") + 1
    assert list(df.columns) == [*columns[:at], *MARGIN_COLUMNS, *columns[at:]]
    params = torch.tensor(df[list(IDM_PARAM_NAMES)].to_numpy())
    speeds = CalibrationConfig().stability_speeds
    assert speeds == tuple(float(v) for v in range(5, 31))
    shortfall = idm_stability_shortfall(params, margin, speeds).numpy()
    assert np.allclose(df.objective, df.fit_objective + 10.0 * shortfall, rtol=1e-12, atol=0)
    assert np.allclose(evaluate_idm(events, df, **CPU).objective, df.fit_objective, rtol=1e-12, atol=0)
    margins = idm_margin(torch.tensor(speeds, dtype=torch.float64), *params[:, :, None].unbind(1))
    assert np.allclose(df.margin_min, torch.nan_to_num(margins, nan=-10.0).amin(-1).numpy(), rtol=1e-12, atol=0)
    assert np.allclose(df.margin_min, idm_margin_min(params, speeds).numpy(), rtol=0, atol=0)
    assert MARGIN_TOLERANCE == 1e-4 and (df.margin_holds == (df.margin_min >= margin - 1e-4)).all()
    # the margin is met (the true parameters of the second and the third event have no equilibrium above v0 <= 30)
    assert df.margin_holds.all() and (df.v0 > 30.0).all() and np.allclose(shortfall, 0.0)
    assert (df.fit_objective >= per_event.objective.iloc[:3].to_numpy() - 1e-6).all()
    free = per_event.iloc[:3]
    assert (idm_margin_min(torch.tensor(free[list(IDM_PARAM_NAMES)].to_numpy()), speeds) < margin).all()


def test_evaluate_idm(clean_events, per_event):
    at_truth = evaluate_idm(clean_events, pd.DataFrame(TRUE), **CPU)
    assert (at_truth.objective < 1e-9).all() and at_truth.event_id.tolist() == per_event.event_id.tolist()
    again = evaluate_idm(clean_events, per_event, **CPU)
    assert np.allclose(again.objective, per_event.objective, rtol=1e-12, atol=0)
    shared = evaluate_idm(clean_events, TRUE[0], **CPU)
    assert shared.objective.iloc[0] < 1e-9 and (shared.objective.iloc[1:] > 0.01).all()


def test_accumulated_loop_matches_rollout(clean_events):
    batch = pad_events(clean_events[:4])
    gen = torch.Generator().manual_seed(0)
    lo, hi = CalibrationConfig().bound_tensors()
    params = lo + (hi - lo) * torch.rand(4, 7, 5, generator=gen, dtype=torch.float64)
    for b in (batch, {**batch, "x_lead": batch["x_lead"] - 150.0 * (torch.arange(300) >= 120)}):  # 2nd: collisions
        J, parts = idm_objective(params, b)
        res = rollout_memoryless(
            lambda s, dv, v: idm_acc(s, dv, v, *params.unbind(-1)), b["x_lead"][:, None], b["v_lead"][:, None],
            b["s"][:, :1], b["v"][:, :1],
        )  # fmt: skip
        ref = closed_loop_metrics(res, b["s"][:, None], b["v"][:, None], b["mask"][:, None])
        acc = metrics_from_parts(parts)
        for key in ("rmse_s", "rmse_v", "nrmse_s", "nrmse_v", "collision_fraction"):
            assert torch.allclose(acc[key], ref[key], rtol=1e-12, atol=1e-14)
        assert torch.equal(acc["collided"], ref["collided"])
        expected = torch.where(ref["collided"], 10.0 + ref["collision_fraction"], ref["nrmse_s"] + ref["nrmse_v"])
        assert torch.allclose(J, expected, rtol=1e-12, atol=0)
    assert ref["collided"].all() and (J >= 10.0).all()


def test_not_worse_than_scipy():
    events = [synthetic_event(10 + i, TRUE[i], 250, noise=(0.2, 0.1)) for i in range(3)]
    cfg = CalibrationConfig(**CPU)
    batched = calibrate_per_event(events, cfg)
    for ev, j_batched in zip(events, batched.objective):
        batch = pad_events([ev])

        def f(x: np.ndarray) -> np.ndarray:  # vectorized SciPy call: x is (5, S)
            return idm_objective(torch.as_tensor(x.T[None]), batch)[0][0].numpy()

        res = differential_evolution(
            f, [IDM_BOUNDS[k] for k in IDM_PARAM_NAMES], maxiter=cfg.maxiter, popsize=cfg.popsize, tol=cfg.tol,
            mutation=cfg.mutation, recombination=cfg.recombination, polish=False, rng=0, updating="deferred",
            vectorized=True,
        )  # fmt: skip
        assert j_batched <= res.fun + 0.005
        assert 0.0 < j_batched < 0.2  # noise floor


def test_global_calibration_homogeneous_set():
    events = [synthetic_event(20 + i, TRUE[0], 220) for i in range(4)]
    out = calibrate_global(events, CalibrationConfig(maxiter=150, **CPU))
    assert out["objective"] < 0.02
    assert out["n_events"] == 4 and out["collision_rate"] == 0.0
    assert out["objective"] == pytest.approx(out["nrmse_s"] + out["nrmse_v"], rel=1e-9)
    assert set(out["params"]) == set(IDM_PARAM_NAMES)
    assert out["params"]["T"] == pytest.approx(TRUE[0]["T"], rel=0.05)


def test_reproducible_for_fixed_seed(clean_events):
    cfg = CalibrationConfig(maxiter=5, batch_events=2, device="auto")  # several batches, GPU when available
    first, second = calibrate_per_event(clean_events, cfg), calibrate_per_event(clean_events, cfg)
    pd.testing.assert_frame_equal(first, second)
    assert first.event_id.tolist() == [ev.event_id for ev in clean_events]
    other = calibrate_per_event(clean_events, CalibrationConfig(maxiter=5, batch_events=2, device="auto", seed=1))
    assert not np.allclose(first["T"], other["T"])


def test_config_from_mapping():
    cfg = CalibrationConfig.from_mapping({"bounds": {"v0": [15, 40]}, "mutation": [0.4, 0.9], "maxiter": 50})
    assert cfg.bounds["v0"] == (15.0, 40.0) and cfg.bounds["T"] == IDM_BOUNDS["T"]
    assert cfg.mutation == (0.4, 0.9) and cfg.maxiter == 50 and cfg.collision_penalty == 10.0
    with pytest.raises(ValueError):
        CalibrationConfig.from_mapping({"pop_size": 3})


def test_parameter_spread():
    values = {
        "v0": [10.0, 20.0, 30.0, 45.0],
        "T": [0.3, 1.0, 1.5, 2.0],
        "s0": [1.0, 2.0, 4.0, 3.0],
        "a": [0.5, 1.0, 1.5, 4.0],
        "b": [6.0, 2.0, 3.0, 1.0],
    }
    out = parameter_spread(pd.DataFrame(values))
    assert out["n_events"] == 4 and set(out["parameters"]) == set(IDM_PARAM_NAMES)
    v0 = out["parameters"]["v0"]
    assert set(v0) == {"mean", "std", "median", "q05", "q25", "q75", "q95", "cv", "share_at_lower", "share_at_upper"}
    x = np.array(values["v0"])
    assert v0["mean"] == pytest.approx(26.25) and v0["median"] == pytest.approx(25.0)
    assert v0["std"] == pytest.approx(x.std(ddof=1)) and v0["cv"] == pytest.approx(x.std(ddof=1) / 26.25)
    assert v0["q05"] == pytest.approx(np.quantile(x, 0.05)) and v0["q95"] == pytest.approx(np.quantile(x, 0.95))
    assert v0["share_at_lower"] == 0.25 and v0["share_at_upper"] == 0.25
    assert out["parameters"]["T"]["share_at_lower"] == 0.25 and out["parameters"]["s0"]["share_at_upper"] == 0.0
    corr = out["correlation"]
    assert all(corr[k][k] == pytest.approx(1.0) for k in IDM_PARAM_NAMES)
    assert corr["v0"]["T"] == pytest.approx(np.corrcoef(values["v0"], values["T"])[0, 1])
    assert corr["a"]["b"] == pytest.approx(corr["b"]["a"])
