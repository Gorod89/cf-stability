"""Empirical gain of D131 (scripts/analysis/empirical_gain.py) on synthetic data: the Welch estimate of a known transfer
function, the interpolation of the audits' gains, the excess log gain and the rule."""

import importlib.util
import math

import numpy as np
import pytest
import torch
from scipy import signal

from cf_stability.stability.analytic import transfer_discrete
from cf_stability.utils import REPO_ROOT


def script():
    path = REPO_ROOT / "scripts" / "analysis" / "empirical_gain.py"
    spec = importlib.util.spec_from_file_location("empirical_gain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eg = script()
DT = 0.1


def follower_of(v_lead: np.ndarray, f_s: float, f_dv: float, f_v: float, v_e: float, s_e: float) -> np.ndarray:
    """The linear follower of the discrete scheme of the project (``u[k+1] = u[k] + dt a[k]``, gap by the new speeds)
    around the equilibrium ``(s_e, v_e)``: its speed."""
    n = len(v_lead)
    v, s = np.full(n, v_e), np.full(n, s_e)
    for k in range(n - 1):
        a = f_s * (s[k] - s_e) + f_dv * (v[k] - v_lead[k]) + f_v * (v[k] - v_e)
        v[k + 1] = v[k] + DT * a
        s[k + 1] = s[k] + DT * (v_lead[k + 1] - v[k + 1])
    return v


def test_welch_estimate_of_a_known_linear_follower():
    rng = np.random.default_rng(0)
    n = 2401  # 240 s at 10 Hz: 11 segments of 40 s
    b, a = signal.butter(2, 0.08)
    v_lead = 12.0 + signal.lfilter(b, a, rng.normal(0, 3.0, n))  # broadband in 0.3-2 rad/s
    f_s, f_dv, f_v = 0.3, -0.9, -0.3
    v = follower_of(v_lead, f_s, f_dv, f_v, 12.0, 20.0)
    omega, gain = eg.event_gain(v_lead, v, 400)
    keep = (omega >= 0.3) & (omega <= 2.0)
    assert np.allclose(omega[keep] / (2 * np.pi), np.arange(2, 13) / 40.0)  # the Welch lines k / 40 Hz
    exact = transfer_discrete(*(torch.tensor([x], dtype=torch.float64) for x in (f_s, f_dv, f_v)),
                              torch.as_tensor(omega[keep]), DT).abs()[0].numpy()  # fmt: skip
    np.testing.assert_allclose(gain[keep], exact, rtol=0.05)  # noise free: the H1 estimate is the transfer function


def test_driver_gains_bins_and_cell_medians():
    class Event:
        def __init__(self, k, n, speed):
            t = np.arange(n) * DT
            self.event_id, self.follower_id = f"e{k}", f"f{k}"
            self.v_lead = speed + np.sin(0.8 * t) + 0.3 * np.sin(1.7 * t)
            self.v = speed + 0.9 * np.sin(0.8 * t - 0.3) + 0.2 * np.sin(1.7 * t - 0.5)
            self.duration = (n - 1) * DT

        def __len__(self):
            return len(self.v)

    events = [Event(0, 401, 7.0), Event(1, 300, 7.0), Event(2, 801, 12.0), Event(3, 401, 2.0)]
    frame, gains, omega = eg.driver_gains(events, 40.0, 40.0)
    assert frame["event_id"].tolist() == ["e0", "e2", "e3"]  # e1 is shorter than 40 s
    assert frame["bin"].tolist() == [1, 2, 0] and frame["segments"].tolist() == [1, 3, 1]
    assert gains.shape == (3, 11) and np.allclose(omega, 2 * np.pi * np.arange(2, 13) / 40.0)
    medians = eg.cell_medians(frame["bin"].to_numpy(), gains, [0, 1, 2, 3])
    np.testing.assert_allclose(medians[1], gains[0])
    assert np.isnan(medians[3]).all()


def test_interpolation_of_the_audit_gains():
    speeds = np.array([5.0, 6.0, 7.0])
    omega = np.geomspace(0.02, 2.0, 25)
    gain = np.vstack([omega**0.0 * 1.1, omega**0.2, omega**0.4])  # power laws: linear in log-log
    at_grid = eg.interpolate_log_gain(speeds, omega, gain, 6.0, omega[[3, 10]])
    np.testing.assert_allclose(at_grid, np.log(gain[1, [3, 10]]))
    w = np.array([0.5])
    mid = eg.interpolate_log_gain(speeds, omega, gain, 6.5, w)
    np.testing.assert_allclose(mid, 0.5 * (0.2 + 0.4) * np.log(w))  # linear in speed of exact log-log values
    assert np.isnan(eg.interpolate_log_gain(speeds, omega, gain, 4.0, w)).all()  # no extrapolation in speed
    holes = gain.copy()
    holes[2, 20] = np.nan
    target = np.array([np.sqrt(omega[20] * omega[21])])
    assert np.isnan(eg.interpolate_log_gain(speeds, omega, holes, 6.5, target)).all()
    audit = {"omega": omega.tolist(), "equilibria": [
        {"v": 5.0, "status": "ok", "gain": [1.0] * 25, "clipped": [False] * 24 + [True], "stopped": [False] * 25,
         "collided": [False] * 25},
        {"v": 6.0, "status": "none", "gain": None},
    ]}  # fmt: skip
    v, w_grid, g = eg.audit_gain_grid(audit)
    assert v.tolist() == [5.0, 6.0] and np.isnan(g[0, -1]) and np.isnan(g[1]).all() and g[0, 0] == 1.0


def test_excess_bootstrap_and_rule():
    median = np.array([[1.0, 1.0], [0.5, 0.5]])
    law = np.log(np.array([[2.0, 2.0], [0.5, np.nan]]))
    valid = np.isfinite(law) & np.isfinite(median)
    assert eg.excess(law, median, valid) == pytest.approx((2 * math.log(2.0) + 0.0) / 3)
    rng = np.random.default_rng(0)
    bins = np.array([0] * 30 + [1] * 30)
    gains = np.where(bins[:, None] == 0, 1.0, 0.5) * np.exp(rng.normal(0, 0.1, (60, 2)))
    folds = np.stack([law + rng.normal(0, 0.01, law.shape) for _ in range(5)])
    med = eg.cell_medians(bins, gains, [0, 1])
    medians_r = eg.resampled_medians(bins, gains, [0, 1], 300, rng)
    draws = rng.integers(0, 5, size=(300, 5))
    out = eg.bootstrap_excess(folds, med, medians_r, draws, valid)
    for name in ("events", "folds", "joint"):
        assert out[f"{name}_low"] <= out["excess"] <= out[f"{name}_high"]
    assert out["events_low"] > 0
    rows = [{"law": law, "excess": 0.3, "events_low": 0.1, "folds_low": 0.1, "joint_low": 0.1}
            for law in eg.NETWORKS]  # fmt: skip
    assert eg.rule(rows + [{"law": "idm", "excess": -0.1, "events_low": -0.2}], "events") == ("confirmed", 5, 5)
    assert eg.rule(rows + [{"law": "idm", "excess": 0.1, "events_low": 0.0}], "events")[0] == "open"
    low = [{**r, "excess": -0.1} for r in rows[:3]] + rows[3:]
    assert eg.rule(low + [{"law": "idm", "excess": -0.1, "events_low": -0.2}], "events") == ("refuted", 2, 2)
