"""Platoon growth test: stable and unstable laws, exact follower rollouts, padding, collisions,
empirical curves, start at the equilibrium inside the band, hysteresis loop area, summary, end to end."""

import json
import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cf_stability.models.base import CFModel, InputScaler
from cf_stability.models.idm import idm_acc
from cf_stability.stability.analytic import transfer_discrete
from cf_stability.stability.equilibrium import Band
from cf_stability.stability.platoon import (
    braking_pulse,
    growth_error,
    hysteresis_area,
    openacc_profile,
    platoon_summary,
    platoon_test,
    simulate_platoon,
    simulate_platoons,
)
from cf_stability.train.closed_loop import rollout_model

DT = 0.1


def band(v: list[float], s_low: list[float], s_high: list[float]) -> Band:
    """Spacing band of a training context (D78) with the given edges."""
    median = [0.5 * (low + high) for low, high in zip(s_low, s_high)]
    return Band.from_mapping({"v": v, "s_low": s_low, "s_median": median, "s_high": s_high})


WIDE_BAND = band([5.0, 30.0], [10.0, 10.0], [40.0, 40.0])


class LinearLaw(CFModel):
    """``a = f_s (s - s0) + f_dv dv + f_v v`` of the mean of the last ``window`` states."""

    def __init__(self, f_s: float, f_dv: float, f_v: float, s0: float = 2.0, window: int = 1) -> None:
        super().__init__()
        self.register_buffer("coef", torch.tensor([f_s, f_dv, f_v], dtype=torch.float64))
        self.s0, self.window = s0, window

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[..., -self.window :, :].mean(dim=-2)
        return (x[..., 0] - self.s0) * self.coef[0] + x[..., 1] * self.coef[1] + x[..., 2] * self.coef[2]


class IDMPlusGRU(CFModel):
    """IDM plus a small GRU over a window of 5 states: a network in the batched model call."""

    window = 5

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.scaler = InputScaler((30.0, 0.0, 15.0), (15.0, 1.5, 6.0))
        self.gru, self.head = nn.GRU(3, 8, batch_first=True), nn.Linear(8, 1)

    def forward(self, state_history: Tensor) -> Tensor:
        x = state_history[:, -self.window :]
        last = x[:, -1]
        out, _ = self.gru(self.scaler(x).to(self.head.weight.dtype))
        idm = idm_acc(last[:, 0], last[:, 1], last[:, 2], 30.0, 1.5, 2.0, 1.0, 1.5)
        return idm + 0.2 * self.head(out[:, -1]).squeeze(-1)


class ConstantAcc(CFModel):
    def forward(self, state_history: Tensor) -> Tensor:
        return torch.ones(state_history.shape[0], dtype=state_history.dtype)


def max_discrete_gain(f_s: float, f_dv: float, f_v: float) -> float:
    omega = torch.linspace(1e-3, math.pi / DT, 4000, dtype=torch.float64)
    one = torch.ones(1, dtype=torch.float64)
    return transfer_discrete(f_s * one, f_dv * one, f_v * one, omega, DT).abs().max().item()


def test_braking_pulse():
    v = braking_pulse(20.0, 5.0, 2.0, 5.0, 30.0)
    assert v.shape == (301,) and v[0] == v[-1] == 20.0 and v.min() == 15.0
    assert (v == 15.0).sum() == 51  # 2.5 s .. 7.5 s
    assert torch.diff(v).abs().max().item() == pytest.approx(2.0 * DT)


def test_stable_law_does_not_amplify_unstable_law_does():
    pulse = braking_pulse(20.0, 5.0, 2.0, 5.0, 90.0)
    assert max_discrete_gain(0.2, -1.0, -0.6) <= 1.0 < max_discrete_gain(0.25, 0.0, -0.5)
    stable = simulate_platoon(LinearLaw(0.2, -1.0, -0.6), pulse, n_vehicles=12)
    assert stable["speed_std"].shape == (12,) and not stable["collided"]
    assert (stable["speed_std"][1:] <= stable["speed_std"][:-1] + 1e-9).all()
    unstable = simulate_platoon(LinearLaw(0.25, 0.0, -0.5), pulse, n_vehicles=12)  # OVM-like, margin -0.25
    assert (unstable["speed_std"][1:] > unstable["speed_std"][:-1]).all()
    assert unstable["speed_std"][-1] > 1.5 * unstable["speed_std"][0]


@pytest.mark.parametrize("model", [LinearLaw(0.3, -0.5, -0.4), LinearLaw(0.3, -0.5, -0.4, window=4), IDMPlusGRU()])
@pytest.mark.parametrize("start_gap", ["equilibrium", "time_gap", "anchored"])
def test_vehicle_follows_its_simulated_predecessor(model, start_gap):
    """All vehicles advance together; each one equals a direct two-vehicle rollout behind its predecessor."""
    window = model.window
    profile = 15.0 + 2.0 * torch.sin(0.3 * DT * torch.arange(300, dtype=torch.float64))
    sim = simulate_platoon(model, profile, n_vehicles=5, start_gap=start_gap, band=WIDE_BAND)
    assert sim["start_equilibrium"] and sim["start_gap_m"] == sim["start_gap"]
    if start_gap == "time_gap":
        assert sim["start_gap"] == 2.0 + 1.5 * 15.0 and sim["start"] == "time_gap"
    else:
        assert sim["start"] == "equilibrium"  # the equilibria lie inside the wide band
        if isinstance(model, LinearLaw):
            assert sim["start_gap"] == pytest.approx(2.0 + 0.4 / 0.3 * 15.0, rel=1e-12)
    assert torch.equal(sim["v"][0], profile) and torch.isnan(sim["s"][0]).all() and math.isnan(sim["min_gap"][0])
    v_start = profile[0]
    before = DT * v_start * torch.arange(window - 1, 0, -1, dtype=torch.float64)  # cruising before the profile
    for n in range(1, 5):
        v_prev = torch.cat((v_start.expand(window - 1), sim["v"][n - 1]))
        x_prev = torch.cat((sim["x"][n - 1][0] - before, sim["x"][n - 1]))
        direct = rollout_model(
            model.double(), x_prev[None], v_prev[None], torch.full((1, window), sim["start_gap"], dtype=torch.float64),
            v_start.expand(1, window),
        )  # fmt: skip
        for name in ("v", "x", "s"):
            assert torch.allclose(getattr(direct, name)[0, window - 1 :], sim[name][n], rtol=0, atol=1e-9)
    assert sim["min_gap"][1:].min() > 0 and 0 < sim["max_abs_acc"] < 4.0 and not sim["collided"]
    assert sim["collision_step"] is None and sim["collision_vehicle"] is None


def test_profiles_of_different_length_are_padded_and_masked():
    law = LinearLaw(0.3, -0.5, -0.4, window=3)
    long = 15.0 + 2.0 * torch.sin(0.3 * DT * torch.arange(250, dtype=torch.float64))
    short = braking_pulse(12.0, 4.0, 2.0, 1.0, 17.9)
    together = simulate_platoons(law, [long, short], n_vehicles=6)
    assert [len(sim["v"][0]) for sim in together] == [250, 180]
    for profile, sim in zip((long, short), together):
        alone = simulate_platoon(law, profile, n_vehicles=6)
        assert sim.keys() == alone.keys()
        for key, value in alone.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(sim[key], value, rtol=0, atol=0, equal_nan=True)
            else:
                assert sim[key] == value, key


def test_first_collision_is_reported():
    """Followers accelerating at 1 m/s^2 behind a leader at 10 m/s: only the first one closes its gap,
    ``s[k] = 17 - 0.005 k (k + 1)`` <= 0 first at k = 58."""
    leader = torch.full((100,), 10.0, dtype=torch.float64)
    sim = simulate_platoon(ConstantAcc(), leader, n_vehicles=4, start_gap="time_gap")
    assert sim["start_gap"] == 17.0 and not sim["start_equilibrium"]
    assert sim["collided"] and (sim["collision_step"], sim["collision_vehicle"]) == (58, 1)
    assert sim["s"][1, 57] > 0 >= sim["s"][1, 58]
    assert torch.allclose(sim["s"][2:], torch.tensor(17.0, dtype=torch.float64), rtol=0, atol=1e-9)


def test_anchored_start_uses_the_equilibrium_inside_the_band():
    law = LinearLaw(0.3, -1.5, -0.3)  # equilibrium spacing 2 m + 1 s * v
    speeds = band([10.0, 25.0], [8.0, 20.0], [16.0, 35.0])  # at 15 m/s: 12 .. 22.3 m, holds 17 m
    cruise = [torch.full((50,), v, dtype=torch.float64) for v in (15.0, 30.0)]  # 30 m/s: no band there
    sims = simulate_platoons(law, cruise, n_vehicles=3, start_gap="anchored", band=speeds)
    assert [sim["start"] for sim in sims] == ["equilibrium", "time_gap"]
    assert sims[0]["start_gap_m"] == pytest.approx(17.0, rel=1e-12) and sims[1]["start_gap_m"] == 2.0 + 1.5 * 30.0
    assert all(sim["start_equilibrium"] and sim["start_gap"] == sim["start_gap_m"] for sim in sims)
    assert torch.allclose(sims[0]["s"][1:], torch.tensor(17.0, dtype=torch.float64), rtol=0, atol=1e-9)  # steady

    elsewhere = band([10.0, 25.0], [20.0, 30.0], [30.0, 40.0])  # at 15 m/s: 23.3 .. 33.3 m, the law is at 17 m
    sim = simulate_platoon(law, cruise[0], 3, start_gap="anchored", band=elsewhere)
    assert sim["start"] == "time_gap" and sim["start_gap_m"] == 2.0 + 1.5 * 15.0 and sim["start_equilibrium"]
    indifferent = simulate_platoon(LinearLaw(0.0, 0.0, 0.0), cruise[0], 3, start_gap="anchored", band=speeds)
    assert indifferent["start"] == "time_gap" and indifferent["start_gap_m"] == 2.0 + 1.5 * 15.0
    equilibrium = simulate_platoon(law, cruise[0], 3, start_gap="equilibrium", band=elsewhere)  # ignores the band
    assert equilibrium["start"] == "equilibrium" and equilibrium["start_gap_m"] == pytest.approx(17.0, rel=1e-12)

    # without band the start is the time gap: everything equals the result of time_gap
    options = {"n_vehicles": 3, "gap_s0": 3.0, "gap_T": 1.0}
    for plain, time_gap in zip(
        simulate_platoons(law, cruise, start_gap="anchored", **options),
        simulate_platoons(law, cruise, start_gap="time_gap", **options),
    ):
        assert plain.keys() == time_gap.keys() and plain["start"] == "time_gap"
        for key, value in time_gap.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(plain[key], value, rtol=0, atol=0, equal_nan=True)
            else:
                assert plain[key] == value, key
    with pytest.raises(ValueError, match="start_gap"):
        simulate_platoon(law, cruise[0], 3, start_gap="halfway")


def test_string_stable_platoon_started_anchored_does_not_grow():
    """Behind the braking pulse a string-stable law started at its equilibrium inside the band damps the
    pulse along the platoon; started 10 m away from it (the time gap), every vehicle closes its gap
    first, and the speed std grows from vehicle 10 to vehicle 50 because of the start."""
    assert max_discrete_gain(0.3, -1.5, -0.3) <= 1.0 + 1e-9
    law, pulse = LinearLaw(0.3, -1.5, -0.3), braking_pulse(20.0, 5.0, 2.0, 5.0, 150.0)  # equilibrium 22 m at 20 m/s
    speeds = band([5.0, 30.0], [4.0, 24.0], [10.0, 40.0])  # at 20 m/s: 16 .. 28 m
    anchored = simulate_platoon(law, pulse, 51, start_gap="anchored", band=speeds)
    assert anchored["start"] == "equilibrium" and anchored["start_gap_m"] == pytest.approx(22.0, rel=1e-12)
    std = anchored["speed_std"]
    assert (std[11:] <= std[10:-1]).all() and std[50] < 0.6 * std[10] and not anchored["collided"]
    time_gap = simulate_platoon(law, pulse, 51, start_gap="time_gap", band=speeds)
    assert time_gap["start_gap_m"] == 32.0 and time_gap["speed_std"][50] > 2.0 * time_gap["speed_std"][10]


def test_growth_error():
    assert growth_error([1.0, 2.0, 3.0, 4.0, 9.0], [1.0, 2.0, np.nan, 5.0]) == pytest.approx(math.sqrt(1 / 3) / 5)
    assert growth_error(torch.tensor([2.0, 2.0, 7.0]), [1.0, 3.0]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="positions"):
        growth_error([1.0], [1.0, 2.0])


def test_hysteresis_area_of_hand_made_loops():
    # rectangle 4 m x 3 m/s, run through with several points per side, both senses
    s = [0.0, 2.0, 4.0, 4.0, 4.0, 2.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0, 1.5, 3.0, 3.0, 3.0, 1.5]
    assert hysteresis_area(s, v) == pytest.approx(12.0, rel=1e-12)
    assert hysteresis_area(s[::-1], v[::-1]) == pytest.approx(12.0, rel=1e-12)
    assert hysteresis_area(np.array(s) + 30.0, np.array(v) + 20.0) == pytest.approx(12.0, rel=1e-12)  # any origin
    # three sides only: the straight line back to the first point closes the rectangle
    assert hysteresis_area([0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 3.0, 3.0]) == pytest.approx(12.0, rel=1e-12)
    # degenerate loops enclose nothing: out and back along a line, a single point, nothing
    line = np.linspace(0.0, 1.0, 11)
    there_and_back = (np.concatenate((20 + 5 * line, 25 - 5 * line)), np.concatenate((10 + line, 11 - line)))
    assert hysteresis_area(*there_and_back) == pytest.approx(0.0, abs=1e-12)
    assert hysteresis_area([20.0], [10.0]) == 0.0 and hysteresis_area([], []) == 0.0
    # circle of radius 2 as a polygon of n points: (n / 2) r^2 sin(2 pi / n)
    theta = torch.linspace(0.0, 2 * math.pi, 101, dtype=torch.float64)[:-1]
    area = hysteresis_area(30.0 + 2.0 * torch.cos(theta), 15.0 + 2.0 * torch.sin(theta))
    assert area == pytest.approx(50 * 4.0 * math.sin(2 * math.pi / 100), rel=1e-12)
    # a figure eight: the two lobes are run through in opposite senses and cancel
    theta = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    assert hysteresis_area(np.sin(theta), np.sin(theta) * np.cos(theta)) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(hysteresis_area([1.0, np.nan, 2.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="length"):
        hysteresis_area([1.0, 2.0], [1.0])


def test_platoon_summary_of_hand_made_profiles():
    def profile(std, collided=False, **extra):
        return {"speed_std": std, "collided": collided, **extra}

    profiles = {
        "human_a": profile([1.0, 2.0, 3.0], growth_error=0.1, acc_flag=0),
        "human_b": profile([1.0, 4.0, 2.0], growth_error=0.3, acc_flag=0, collided=True),
        "acc": profile([1.0, 2.0, 2.0], growth_error=0.5, acc_flag=1),
        "mixed": profile([1.0, 0.0, 2.0], growth_error=0.7, acc_flag=2),
        "pulse": profile([1.0, 1.0, 0.5], hysteresis_area=12.5),
    }
    summary = platoon_summary(profiles)
    # human_b collided: no growth error and no std ratio, whatever its record holds; the means leave it out
    assert summary["growth_error"] == {"human_a": 0.1, "human_b": None, "acc": 0.5, "mixed": 0.7}
    assert summary["growth_error_mean"] == pytest.approx(1.3 / 3)
    assert summary["growth_error_human"] == pytest.approx(0.1)
    assert summary["growth_error_acc"] == pytest.approx(0.6)  # every flag but 0 (1 ACC, 2 mixed)
    assert summary["growth_profiles"] == {"mean": 4, "human": 2, "acc": 2}
    assert summary["growth_collided"] == {"mean": 1, "human": 1, "acc": 0}
    assert (summary["n_profiles"], summary["n_collided"], summary["hysteresis_area_pulse"]) == (5, 1, 12.5)
    assert summary["std_ratio"] == {"human_a": 1.5, "human_b": None, "acc": 1.0, "mixed": None, "pulse": 0.5}
    profiles["acc"]["growth_error"] = None  # a model curve with NaN: the means that hold it are undefined
    del profiles["pulse"]
    summary = platoon_summary(profiles)
    assert summary["growth_error_acc"] is None and summary["growth_error_mean"] is None
    assert summary["growth_error_human"] == pytest.approx(0.1) and summary["hysteresis_area_pulse"] is None
    profiles["human_a"]["collided"] = True  # every human platoon collided: no human mean
    assert platoon_summary(profiles)["growth_error_human"] is None
    empty = platoon_summary({"pulse": profile([1.0, 1.0], hysteresis_area=None)})
    assert empty["growth_error"] == {} and empty["growth_error_mean"] is None and empty["growth_error_human"] is None
    assert empty["growth_profiles"] == empty["growth_collided"] == {"mean": 0, "human": 0, "acc": 0}


def test_hysteresis_area_of_the_first_follower_none_after_its_collision():
    cfg = {
        "n_vehicles": 4, "pulse": {"v0": 10.0, "dv_pulse": 3.0, "b_pulse": 2.0, "hold_s": 1.0, "duration_s": 12.0},
        "start_gap": "time_gap", "gap_s0": 2.0, "gap_T": 1.5, "device": "cpu",
    }  # fmt: skip
    law = LinearLaw(0.2, -1.0, -0.6)
    out = platoon_test(law, cfg)["pulse"]
    pulse = braking_pulse(10.0, 3.0, 2.0, 1.0, 12.0)
    alone = simulate_platoon(law, pulse, 4, start_gap="time_gap", gap_s0=2.0, gap_T=1.5)
    assert out["hysteresis_area"] == pytest.approx(hysteresis_area(alone["s"][1], alone["v"][1]), rel=1e-12)
    assert out["hysteresis_area"] > 0.0 and not out["collided"]
    crash = platoon_test(ConstantAcc(), cfg)["pulse"]  # the followers accelerate into their leaders
    assert crash["collided"] and crash["collision_vehicle"] == 1 and crash["hysteresis_area"] is None


def write_run(path, acc_flag: int = 0) -> dict[str, np.ndarray]:
    """OpenACC-like file: the target car (vehicle 1) and vehicle 4 have no speed; vehicle 5 misses 0.5 s."""
    t = DT * np.arange(120)
    speeds = {2: 10.0 + np.sin(t), 3: 10.0 + 1.2 * np.sin(t - 0.5), 5: 10.0 + 1.5 * np.sin(t - 1.0)}
    speeds[3][100] = -0.05  # measurement noise around standstill: clipped at 0
    lines = [
        "Date,8,10,2019", "Vehicle_order,TARGET,CAR_A,CAR_B,CAR_C,CAR_D,", "Number_of_vehicles,5", f"ACC,{acc_flag}",
        "Distance_setting,S", "Time,Speed2,Speed3,Speed5,IVS2",
    ]  # fmt: skip
    for k in range(len(t)):
        v5 = "" if 30 <= k <= 33 else repr(float(speeds[5][k]))
        lines.append(f"{t[k]:.1f},{float(speeds[2][k])!r},{float(speeds[3][k])!r},{v5},20.0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return speeds


def test_openacc_profile(tmp_path):
    speeds = write_run(tmp_path / "Zala" / "run.csv")
    profile = openacc_profile(tmp_path / "Zala" / "run.csv")
    block = slice(34, 120)  # the longest stretch with every speed
    assert profile["vehicle_order"] == ["CAR_A", "CAR_B", "CAR_C", "CAR_D"]
    assert profile["vehicles_with_speed"] == [0, 1, 3] and profile["acc_flag"] == 0
    assert profile["t_start"] == pytest.approx(3.4) and profile["duration_s"] == pytest.approx(8.6)
    np.testing.assert_allclose(profile["v_lead"], speeds[2][block], rtol=1e-12)
    clipped = np.clip(speeds[3][block], 0.0, None)
    expected = [np.std(speeds[2][block]), np.std(clipped), np.nan, np.std(speeds[5][block])]
    np.testing.assert_allclose(profile["empirical_std"], expected, rtol=1e-12)


def test_platoon_test_end_to_end(tmp_path):
    write_run(tmp_path / "Zala" / "run.csv")
    references = tmp_path / "growth.yaml"
    references.write_text("jiang2015: null\nhand: {profile: pulse, std: [1.0, 1.1, 1.2, 1.3]}\n", encoding="utf-8")
    cfg = {
        "n_vehicles": 5, "dt": DT, "raw_dir": str(tmp_path), "profiles": ["Zala/run.csv"],
        "pulse": {"v0": 15.0, "dv_pulse": 3.0, "b_pulse": 2.0, "hold_s": 1.0, "duration_s": 20.0},
        "empirical_growth": str(references), "start_gap": "time_gap", "gap_s0": 3.0, "gap_T": 1.0, "device": "cpu",
    }  # fmt: skip
    law = LinearLaw(0.2, -1.0, -0.6)
    out = platoon_test(law, cfg)
    json.dumps(out, allow_nan=False)
    run, pulse = out["Zala/run.csv"], out["pulse"]
    assert len(run["speed_std"]) == len(pulse["speed_std"]) == 5 and run["min_gap"][0] is None
    assert run["empirical_std"][2] is None and 0.0 < run["growth_error"] < 1.0
    assert run["collided"] is False and run["collision_step"] is None and "growth_error" not in pulse
    assert pulse["start_gap"] == 3.0 + 1.0 * 15.0
    alone = simulate_platoon(law, braking_pulse(15.0, 3.0, 2.0, 1.0, 20.0), 5, start_gap="time_gap", gap_s0=3.0, gap_T=1.0)
    assert pulse["speed_std"] == alone["speed_std"].tolist()
    assert pulse["hysteresis_area"] == pytest.approx(hysteresis_area(alone["s"][1], alone["v"][1]), rel=1e-12)
    assert run["hysteresis_area"] > 0.0
    reference = growth_error(pulse["speed_std"], [1.0, 1.1, 1.2, 1.3])
    assert pulse["growth_error_reference"]["hand"] == pytest.approx(reference)
    summary = platoon_summary(out)
    assert summary["growth_error"] == {"Zala/run.csv": run["growth_error"]} and summary["growth_error_acc"] is None
    assert summary["growth_error_mean"] == summary["growth_error_human"] == pytest.approx(run["growth_error"])
    assert summary["hysteresis_area_pulse"] == pulse["hysteresis_area"] and summary["n_collided"] == 0
    assert summary["std_ratio"]["pulse"] == pytest.approx(pulse["speed_std"][-1] / pulse["speed_std"][1])
    assert run["start"] == pulse["start"] == "time_gap" and summary["n_start_equilibrium"] == 0

    # anchored: both leaders start inside the band, at the equilibrium 2 m + 3 s * v of the law
    speeds = band([5.0, 30.0], [10.0, 50.0], [40.0, 120.0])
    anchored = platoon_test(law, {**cfg, "start_gap": "anchored"}, band=speeds)
    first_speed = openacc_profile(tmp_path / "Zala" / "run.csv")["v_lead"][0]
    assert anchored["Zala/run.csv"]["start"] == anchored["pulse"]["start"] == "equilibrium"
    assert anchored["Zala/run.csv"]["start_gap_m"] == pytest.approx(2.0 + 3.0 * first_speed, rel=1e-12)
    assert anchored["pulse"]["start_gap_m"] == pytest.approx(47.0, rel=1e-12)
    assert platoon_summary(anchored)["n_start_equilibrium"] == 2
    assert platoon_test(law, {**cfg, "start_gap": "anchored"}) == out  # no band: the time gap

    crash = platoon_test(ConstantAcc(), cfg)  # the followers accelerate into their leaders: no growth error
    assert crash["Zala/run.csv"]["collided"] and crash["Zala/run.csv"]["growth_error"] is None
    assert crash["pulse"]["collided"] and crash["pulse"]["growth_error_reference"]["hand"] is None
    summary = platoon_summary(crash)
    assert summary["growth_error_mean"] is None and summary["growth_collided"] == {"mean": 1, "human": 1, "acc": 0}
    assert summary["std_ratio"] == {"Zala/run.csv": None, "pulse": None} and summary["n_collided"] == 2
