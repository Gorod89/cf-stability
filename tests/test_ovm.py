"""OVM (formula, signs, string instability, wrapper, calibration) and the generic global calibration."""

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models import OVM
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import IDM_PARAM_NAMES, idm_acc
from cf_stability.models.ovm import OVM_BOUNDS, OVM_PARAM_NAMES, calibrate_ovm, ovm_acc
from cf_stability.train.calibration import RESTART_SEED_STEP, CalibrationConfig, calibrate_global, idm_objective
from cf_stability.train.closed_loop import pad_events, rollout_memoryless
from cf_stability.train.de import differential_evolution_batched

P = {"v0": 30.0, "tau": 1.2, "s0": 2.0}
TRUE = {"v0": 24.0, "tau": 1.4, "s0": 3.0}
IDM_P = {"v0": 30.0, "T": 1.3, "s0": 2.0, "a": 1.2, "b": 1.8}


def t64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def event(i: int, acc_fn, n: int, v_init: float, s_init: float) -> Event:
    """Follower driven by ``acc_fn`` behind a leader oscillating between 11 and 29 m/s."""
    rng = np.random.default_rng(i)
    t = DT * np.arange(n)
    v_lead = 20.0 + 9.0 * np.sin(2 * np.pi * t / rng.uniform(15, 25) + rng.uniform(0, 2 * np.pi))
    x_lead = 40.0 + np.concatenate(([0.0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * DT)))
    res = rollout_memoryless(acc_fn, t64(x_lead), t64(v_lead), t64(s_init), t64(v_init))
    s, v = res.s.numpy(), res.v.numpy()
    ev = Event(
        event_id=f"syn/a/{i}|L{i}|0", dataset="syn", site="a", follower_id=f"syn/a/{i}", leader_id=f"syn/a/L{i}",
        t=t, s=s, dv=v - v_lead, v=v, a=res.a.numpy(), v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip
    ev.validate()
    return ev


def ovm_event(i: int, n: int = 200) -> Event:
    p = [TRUE[k] for k in OVM_PARAM_NAMES]
    return event(i, lambda s, dv, v: ovm_acc(s, dv, v, *p), n, 18.0, TRUE["s0"] + TRUE["tau"] * 18.0)


def test_reference_values():
    s, v = t64([20.0, 60.0, 1.0]), t64([10.0, 32.0, 5.0])
    # (min(30, 18 / 1.2) - 10) / 1.2 = 4.166667; (min(30, 48.33) - 32) / 1.2 = -1.666667;
    # (min(30, -1 / 1.2) - 5) / 1.2 = -4.861111
    expected = [4.1666667, -1.6666667, -4.8611111]
    assert ovm_acc(s, t64([3.0, -2.0, 0.0]), v, **P).tolist() == pytest.approx(expected, abs=1e-6)


def test_sign_conventions():
    s = torch.linspace(1.0, 100.0, 200, dtype=torch.float64)
    a_s = ovm_acc(s, t64(0.0), t64(15.0), **P)
    congested = (s - P["s0"]) / P["tau"] < P["v0"]
    assert (a_s.diff() >= 0).all() and (a_s.diff()[congested[1:]] > 0).all() and (~congested).any()
    v = torch.linspace(0.0, 40.0, 200, dtype=torch.float64)
    assert (ovm_acc(t64(30.0), t64(0.0), v, **P).diff() < 0).all()
    dv = torch.linspace(-5.0, 5.0, 50, dtype=torch.float64)
    a_dv = ovm_acc(torch.full_like(dv, 30.0), dv, torch.full_like(dv, 15.0), **P)
    assert (a_dv == a_dv[0]).all()


def test_string_unstable_on_the_congested_branch():
    s = torch.linspace(3.0, 35.0, 20, dtype=torch.float64, requires_grad=True)  # (s - s0) / tau < v0
    v = ((s.detach() - P["s0"]) / P["tau"]).requires_grad_()  # equilibrium speeds
    dv = torch.zeros_like(v, requires_grad=True)
    a = ovm_acc(s, dv, v, **P)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-12)
    f_s, f_dv, f_v = torch.autograd.grad(a.sum(), (s, dv, v), allow_unused=True)
    assert f_dv is None  # the law does not use dv
    criterion = f_v**2 - 2 * f_s  # f_v^2 + 2 f_v f_dv - 2 f_s with f_dv = 0
    assert torch.allclose(criterion, torch.full_like(criterion, -1.0 / P["tau"] ** 2), rtol=1e-12)


def test_model_wrapper():
    model = OVM(ModelContext(ovm_params=P))
    assert model.window == 1 and not model.trainable and model.config() == {"name": "ovm"}
    assert model.params_dict() == pytest.approx(P)
    x = torch.stack((t64([20.0, 60.0]), t64([1.0, 0.0]), t64([10.0, 32.0])), -1)[:, None, :]
    stale = x + 5.0
    out = model(torch.cat((stale, x), 1))  # only the last step matters
    assert torch.allclose(out, ovm_acc(x[:, 0, 0], x[:, 0, 1], x[:, 0, 2], **P), rtol=1e-14, atol=0)
    assert model(x.float()).dtype == torch.float32
    assert torch.isnan(OVM().theta).all()  # set by fit or a checkpoint


def test_calibration_recovers_the_parameters():
    events = [ovm_event(i) for i in range(3)]
    free = np.concatenate([(ev.s - TRUE["s0"]) / TRUE["tau"] > TRUE["v0"] for ev in events])
    assert 0.1 < free.mean() < 0.9  # both branches of the min are visited, so that v0 is identified
    out = calibrate_ovm(events, CalibrationConfig(maxiter=60, restarts=1, device="cpu"))
    assert set(out["params"]) == set(OVM_PARAM_NAMES) and out["collision_rate"] == 0.0
    assert out["objective"] < 0.01
    for name in OVM_PARAM_NAMES:
        lo, hi = OVM_BOUNDS[name]
        assert lo <= out["params"][name] <= hi
        assert out["params"][name] == pytest.approx(TRUE[name], rel=0.05)


def test_generic_global_calibration_reproduces_the_idm_fit():
    p = [IDM_P[k] for k in IDM_PARAM_NAMES]
    events = [event(i, lambda s, dv, v: idm_acc(s, dv, v, *p), n, 20.0, 30.0) for i, n in enumerate((180, 150, 210))]
    cfg = CalibrationConfig(maxiter=5, restarts=2, batch_events=2, device="cpu")
    out = calibrate_global(events, cfg)
    # reference: the pooled objective through idm_objective and the optimiser called directly
    order = np.argsort([len(ev) for ev in events], kind="stable")
    batches = [pad_events([events[j] for j in order[k : k + 2]]) for k in range(0, len(order), 2)]

    def energy(pop: torch.Tensor) -> torch.Tensor:
        parts = [idm_objective(pop, batch)[1] for batch in batches]
        total = {name: sum(part[name].sum(0) for part in parts) for name in ("sse_s", "sse_v", "ss_s", "ss_v")}
        collided = sum((part["n_coll"] > 0).sum(0) for part in parts)
        fit = torch.sqrt(total["sse_s"] / total["ss_s"]) + torch.sqrt(total["sse_v"] / total["ss_v"])
        return (fit + 10.0 * collided / len(events))[None, :]

    lower, upper = cfg.bound_tensors()
    runs = [
        differential_evolution_batched(
            energy, lower, upper, 1, popsize=15, maxiter=5, tol=cfg.global_tol, atol=0.0, mutation=(0.5, 1.0),
            recombination=0.7, seed=RESTART_SEED_STEP * r,
        )
        for r in range(2)
    ]  # fmt: skip
    best = min(runs, key=lambda run: float(run.fun[0]))
    assert out["params"] == dict(zip(IDM_PARAM_NAMES, best.x[0].tolist()))
    assert out["objective"] == float(best.fun[0]) and out["n_events"] == 3
    assert out["objective"] == pytest.approx(out["nrmse_s"] + out["nrmse_v"], rel=1e-9)
