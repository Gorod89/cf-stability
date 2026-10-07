"""Newell's model in the adapted form of PERL: delayed leader acceleration, calibration of w."""

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models import Newell
from cf_stability.models.base import ModelContext
from cf_stability.models.newell import NEWELL_BOUNDS, calibrate_newell, leader_acceleration, newell_acc
from cf_stability.train.calibration import CalibrationConfig
from cf_stability.train.closed_loop import rollout_model

W_TRUE = 5.0
SUMMARY_KEYS = {"params", "objective", "rmse_s", "rmse_v", "nrmse_s", "nrmse_v", "collision_rate", "n_events", "grid"}


def t64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def ramp_history(s_last: list[float]) -> torch.Tensor:
    """30 steps, one per entry of ``s_last`` (gap of the last step): the leader accelerates at
    +1 m/s^2 up to sample 14 and at -2 m/s^2 after it; the follower speed is arbitrary."""
    j = np.arange(30)
    v_lead = np.where(j <= 14, 10.0 + 0.1 * j, 11.4 - 0.2 * (j - 14))
    v = 12.0 + 0.5 * np.sin(j)
    history = t64(np.stack((np.full(30, 10.0), v - v_lead, v), -1)).repeat(len(s_last), 1, 1)
    history[:, -1, 0] = t64(s_last)
    return history


def test_leader_acceleration():
    # central differences inside the window (-0.5 at the kink), one-sided at both ends
    expected = np.r_[np.full(14, 1.0), -0.5, np.full(15, -2.0)]
    assert np.allclose(leader_acceleration(ramp_history([10.0]))[0].numpy(), expected, rtol=0, atol=1e-9)
    assert np.allclose(leader_acceleration(ramp_history([10.0])[:, 13:15]).numpy(), [[1.0, 1.0]], atol=1e-9)
    assert leader_acceleration(ramp_history([10.0])[:, :1]).tolist() == [[0.0]]


def test_output_is_the_leader_acceleration_one_wave_travel_time_ago():
    # delay = s / w with w = 10 m/s; fractional sample p = 29 - delay / dt
    s = [10.0, 20.0, 14.75, 15.5, 29.0]
    # p = 19 -> -2; p = 9 -> +1; p = 14.25 -> -0.5 + 0.25 * (-2 + 0.5); p = 13.5 -> 1 + 0.5 * (-0.5 - 1); p = 0 -> +1
    expected = [-2.0, 1.0, -0.875, 0.25, 1.0]
    out = newell_acc(ramp_history(s), 10.0)
    assert out.shape == (5,) and out.tolist() == pytest.approx(expected, abs=1e-9)
    assert torch.equal(newell_acc(ramp_history(s), torch.full((5,), 10.0, dtype=torch.float64)), out)
    assert newell_acc(ramp_history(s), t64(s)).tolist() == pytest.approx([-2.0] * 5, abs=1e-9)  # w per sample: 1 s


def test_delay_saturates_and_vanishes():
    history = ramp_history([20.0, 20.0, 0.0, -3.0])
    w = t64([1.0, 1e9, 10.0, 10.0])
    # 20 s > 2.9 s: the oldest sample (+1); w -> infinity or a gap <= 0: the latest sample (-2)
    assert newell_acc(history, w).tolist() == pytest.approx([1.0, -2.0, -2.0, -2.0], abs=1e-6)
    assert newell_acc(history[:, -1:], w).tolist() == [0.0] * 4  # one step: no leader acceleration


def test_independent_of_the_follower_speed():
    history = ramp_history([14.75, 20.0, 3.0])
    shift = 5.0 * torch.rand(3, 30, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    moved = history.clone()
    moved[..., 1] += shift  # v and dv move together: v_lead = v - dv is unchanged
    moved[..., 2] += shift
    assert torch.allclose(newell_acc(moved, 10.0), newell_acc(history, 10.0), rtol=0, atol=1e-9)


def test_dtypes_and_gradients():
    history = ramp_history([14.75, 20.0, 3.0])
    out64, out32 = newell_acc(history, 10.0), newell_acc(history.float(), 10.0)
    assert out64.dtype == torch.float64 and out32.dtype == torch.float32
    assert torch.allclose(out32.double(), out64, rtol=0, atol=1e-4)
    x = history.clone().requires_grad_()
    newell_acc(x, 10.0).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, -1, 0] != 0  # fractional delay across the kink: depends on the gap
    assert x.grad[..., 2].abs().sum() > 0 and torch.allclose(x.grad[..., 1], -x.grad[..., 2])  # through v - dv


def test_model_wrapper():
    model = Newell(ModelContext(newell_params={"w": 10.0}))
    assert model.window == 30 and not model.memoryless and not model.trainable
    assert model.config() == {"name": "newell", "window": 30} and model.params_dict() == {"w": 10.0}
    history = ramp_history([14.75, 20.0])
    longer = torch.cat((torch.randn(2, 5, 3, dtype=torch.float64), history), 1)
    assert torch.equal(model(longer), newell_acc(history, 10.0))  # the last 30 steps
    short = Newell(ModelContext(newell_params={"w": 10.0}), window=10)
    assert torch.equal(short(history), newell_acc(history[:, -10:], 10.0))
    assert torch.isnan(Newell().theta).all()  # set by fit or a checkpoint


def closed_loop_event(i: int, n: int = 150) -> Event:
    """Follower driven by Newell's law (``W_TRUE``) behind a mildly oscillating leader, after 3 s at 12 m/s."""
    rng = np.random.default_rng(i)
    t = DT * np.arange(n)
    wave = np.sin(2 * np.pi * (t - 3.0) / rng.uniform(6, 10) + rng.uniform(0, 2 * np.pi))
    v_lead = 12.0 + np.clip((t - 3.0) / 2.0, 0.0, 1.0) * wave
    x_lead = 30.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    law = Newell(ModelContext(newell_params={"w": W_TRUE}))
    warmup = torch.ones(1, 30, dtype=torch.float64)
    res = rollout_model(law, t64(x_lead)[None], t64(v_lead)[None], rng.uniform(8, 12) * warmup, 12.0 * warmup)
    s, v = res.s[0].numpy(), res.v[0].numpy()
    ev = Event(
        event_id=f"syn/a/{i}|L{i}|0", dataset="syn", site="a", follower_id=f"syn/a/{i}", leader_id=f"syn/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a[0].numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip
    ev.validate()
    return ev


def test_calibration_recovers_the_wave_speed():
    events = [closed_loop_event(i) for i in range(3)]
    assert max((ev.s / W_TRUE).max() for ev in events) < 2.9  # the delay stays inside the window
    out = calibrate_newell(events, CalibrationConfig(device="cpu"))
    assert set(out) == SUMMARY_KEYS and out["n_events"] == 3 and out["collision_rate"] == 0.0
    assert out["params"]["w"] == pytest.approx(W_TRUE, rel=0.05)
    assert out["objective"] < 1e-3 and out["objective"] == pytest.approx(out["nrmse_s"] + out["nrmse_v"])
    grid = out["grid"]
    assert len(grid["w"]) == len(grid["objective"]) == 40
    assert grid["w"][0] == pytest.approx(NEWELL_BOUNDS["w"][0]) and grid["w"][-1] == pytest.approx(NEWELL_BOUNDS["w"][1])
    best = int(np.argmin(grid["objective"]))
    assert grid["w"][best - 1] <= out["params"]["w"] <= grid["w"][best + 1]
    assert out["objective"] <= min(grid["objective"])
    with pytest.raises(ValueError, match="more than window"):
        calibrate_newell(events, CalibrationConfig(device="cpu"), window=150)


def test_newell_acc_of_a_gap_that_is_not_finite_is_nan_not_an_error():
    """A diverged rollout hands NaN or infinite gaps to the law: the result is NaN, the index stays valid."""
    history = torch.zeros(3, 30, 3, dtype=torch.float64)
    history[..., 2] = 20.0 + 0.1 * torch.arange(30, dtype=torch.float64)  # follower speed, the leader accelerates
    history[0, :, 0], history[1, :, 0], history[2, :, 0] = 25.0, float("nan"), float("inf")
    out = newell_acc(history, 5.0)
    assert torch.isfinite(out[0]) and torch.isnan(out[1]) and torch.isfinite(out[2])
