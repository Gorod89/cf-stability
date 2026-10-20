"""Robust (non-equilibrium) certificate of D130 (cf_stability/stability/robust.py): the bounded-real LMI of the polytopic
perturbation dynamics without an SDP solver, its float64 verification, the boxes of a law and the budget allocation."""

import math

import numpy as np
import pytest
import torch

from cf_stability.models.idm import IDM, idm_equilibrium_partials
from cf_stability.stability.analytic import partials, transfer_continuous
from cf_stability.stability.robust import (
    GAMMA_TOL,
    INFEASIBLE,
    Box,
    allocate_budget,
    hinf_norms,
    idm_box,
    law_boxes,
    lmi_matrices,
    model_partials,
    region_feasible,
    robust_certificate,
    robust_gain,
    schur_gain,
    string_stable,
    transfer,
    verify,
    vertex_systems,
)

CORE = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}  # a margin core of the certified hybrid (fold 0)


def test_transfer_is_that_of_the_analytic_module():
    vertices = np.array([[0.5, -0.9, -0.3], [0.08, -0.4, -0.02]])
    omega = np.geomspace(0.01, 3.0, 17)
    expected = transfer_continuous(*(torch.as_tensor(vertices[:, k]) for k in range(3)), torch.as_tensor(omega))
    np.testing.assert_allclose(transfer(vertices, omega), expected.numpy(), rtol=1e-12)
    # the state space (A, B, C) has the same transfer function C (sI - A)^-1 B
    A, B, C = vertex_systems(vertices)
    for i in range(2):
        for w in (0.05, 0.7):
            g = C @ np.linalg.solve(1j * w * np.eye(2) - A[i], B[i])
            assert g == pytest.approx(transfer(vertices[i : i + 1], np.array([w]))[0, 0], rel=1e-12)


@pytest.mark.parametrize("vertex", [(0.504, -0.886, -0.307), (0.3, -1.0, -0.3), (1.2, -0.25, -0.05), (0.05, -2.0, -0.1)])
def test_known_lti_system_gives_its_hinf_norm(vertex):
    """One vertex: the bounded-real lemma is exact for an LTI system, gamma = its H-infinity norm (within 1 %)."""
    p = np.array([vertex])
    omega = np.geomspace(1e-4, 1e2, 200001)
    norm = float(np.abs(transfer(p, omega)).max())
    assert hinf_norms(p)[0] == pytest.approx(norm, rel=1e-6)
    result = robust_certificate(p)
    assert result["status"] == "certified"
    assert result["gamma"] == pytest.approx(norm, rel=0.01)
    assert result["gamma"] >= norm * (1 - 1e-9)  # a certificate never undercuts the H-infinity norm
    A, B, C = vertex_systems(p)
    assert verify(result["P"], A, B, C, result["gamma"])


def test_string_stable_vertex_has_gain_one():
    """M = f_v^2 + 2 f_v f_dv - 2 f_s > 0: |G| <= 1 with G(0) = 1, the certificate holds at 1 + tol, not below 1."""
    p = np.array([[0.3, -1.0, -0.3]])
    assert string_stable(p)[0]
    A, B, C = vertex_systems(p)
    ok_below = robust_certificate(p)["gamma"]
    assert ok_below >= 1.0 and ok_below <= 1.0 + GAMMA_TOL
    # an unstable vertex (M < 0) has no certificate at 1 + tol
    assert not string_stable(np.array([[0.504, -0.886, -0.307]]))[0]


@pytest.mark.parametrize("box", [
    Box((-0.1, 0.4), (-1.0, -0.5), (-0.3, -0.1)),     # f_s <= 0 at a vertex: the spacing error integrates
    Box((0.1, 0.4), (-0.5, 0.6), (-0.3, -0.1)),       # f_dv + f_v >= 0 at a vertex: negative damping
])  # fmt: skip
def test_unstable_box_is_infeasible(box):
    result = robust_certificate(box)
    assert result["status"] == INFEASIBLE and result["gamma"] is None and not result["hurwitz"]
    assert robust_gain(box) == INFEASIBLE
    assert not string_stable(box)[0]


def test_verification_rejects_a_wrong_P():
    p = np.array([[0.504, -0.886, -0.307]])
    A, B, C = vertex_systems(p)
    result = robust_certificate(p)
    P, gamma = result["P"], result["gamma"]
    assert verify(P, A, B, C, gamma)
    assert np.linalg.eigvalsh(lmi_matrices(P, A, B, C, gamma)).max() <= -1e-9
    assert not verify(-P, A, B, C, gamma)  # not positive definite
    assert not verify(np.eye(2) * 1e3, A, B, C, gamma)  # positive definite, but the LMI fails
    assert not verify(P, A, B, C, 0.9 * gamma)  # below the H-infinity norm no P can hold
    wrong = P.copy()
    wrong[0, 1] += 1.0  # not symmetric
    assert not verify(wrong, A, B, C, gamma)


def test_common_storage_function_over_a_box_bounds_every_frozen_system():
    box = Box((0.2, 0.4), (-1.3, -0.9), (-0.45, -0.3))
    result = robust_certificate(box)
    assert result["status"] == "certified"
    rng = np.random.default_rng(0)
    inside = np.column_stack([rng.uniform(*getattr(box, name), 200) for name in ("f_s", "f_dv", "f_v")])
    assert hinf_norms(inside).max() <= result["gamma"] * (1 + 1e-9)  # the polytope's gain bounds every member
    assert result["gamma"] >= result["lower_bound"] * (1 - 1e-9)


def test_idm_box_is_the_range_of_the_derivatives_over_the_spacings():
    v, s_low, s_high = 12.0, 9.0, 31.0
    box = idm_box(CORE, v, s_low, s_high)
    idm = IDM(CORE, dtype=torch.float64)
    s = torch.linspace(s_low, s_high, 501, dtype=torch.float64)
    f = partials(idm, s, torch.full_like(s, v))
    for (low, high), values in zip((box.f_s, box.f_dv, box.f_v), f):
        assert low == pytest.approx(float(values.min()), rel=1e-12, abs=1e-15)
        assert high == pytest.approx(float(values.max()), rel=1e-12, abs=1e-15)
    # at the equilibrium spacing the box holds the closed-form equilibrium derivatives
    s_e, f_s, f_dv, f_v = (float(x) for x in idm_equilibrium_partials(torch.tensor(v, dtype=torch.float64),
                                                                       *(CORE[k] for k in ("v0", "T", "s0", "a", "b"))))
    point = idm_box(CORE, v, s_e, s_e)
    assert point.f_s[0] == pytest.approx(f_s, rel=1e-10) and point.f_dv[0] == pytest.approx(f_dv, rel=1e-10)
    assert point.f_v[0] == pytest.approx(f_v, rel=1e-10)


def test_feasible_gaps_and_law_boxes():
    speeds = [5.0, 10.0, 15.0]
    region = region_feasible(CORE, 0.3, speeds)
    assert sorted(region) == speeds
    idm = IDM(CORE, dtype=torch.float64)
    for v, (lo, hi) in region.items():
        acc = idm.acc(torch.tensor([lo, hi], dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
                      torch.full((2,), v, dtype=torch.float64))  # fmt: skip
        np.testing.assert_allclose(acc.numpy(), [-0.3, 0.3], atol=1e-9)  # the ends of I(v): |f_idm| = r_max
    point = region_feasible(CORE, 0.0, speeds)  # r_max 0: the equilibrium spacing alone
    assert all(lo == pytest.approx(hi, rel=1e-9) for lo, hi in point.values())
    bounds = (0.005, 0.05, 0.01)
    boxes = law_boxes(region, core=CORE, bounds=bounds)
    for v in speeds:
        core_box = idm_box(CORE, v, *region[v])
        assert boxes["per_speed"][v].f_dv == pytest.approx((core_box.f_dv[0] - 0.05, core_box.f_dv[1] + 0.05))
    whole = boxes["whole"]
    assert whole.f_s[1] == pytest.approx(max(b.f_s[1] for b in boxes["per_speed"].values()))
    # empirical boxes of a model: the sampled partials of the IDM lie inside its exact box
    empirical = law_boxes(region, model=idm, n_states=400, seed=1)
    for v in speeds:
        exact, sampled = law_boxes(region, core=CORE)["per_speed"][v], empirical["per_speed"][v]
        for name in ("f_s", "f_dv", "f_v"):
            assert getattr(sampled, name)[0] >= getattr(exact, name)[0] - 1e-12
            assert getattr(sampled, name)[1] <= getattr(exact, name)[1] + 1e-12


def test_model_partials_at_a_non_equilibrium_state_are_those_of_the_idm():
    idm = IDM(CORE, dtype=torch.float64)
    s, v, dv = np.array([12.0, 30.0]), np.array([10.0, 14.0]), np.array([0.8, -1.5])
    f = model_partials(idm, s, v, dv)
    s_t, v_t, dv_t = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (s, v, dv))
    grads = torch.autograd.grad(idm.acc(s_t, dv_t, v_t).sum(), (s_t, dv_t, v_t))
    for got, expected in zip(f, grads):
        np.testing.assert_allclose(got, expected.numpy(), rtol=1e-12)


def test_budget_allocation_keeps_the_certificate():
    core = Box((0.30, 0.33), (-1.20, -1.10), (-0.42, -0.40))  # a narrow, strictly string stable box
    assert string_stable(core)[0]
    result = allocate_budget(core, cycles=1, step_rtol=0.05)
    assert result["feasible"] and result["total"] > 0
    bounds = np.array(result["bounds"])
    assert string_stable(core.widened(bounds))[0]
    for j in range(3):  # no coordinate can grow by 10 % more
        trial = bounds.copy()
        trial[j] = trial[j] * 1.1 + 1e-3
        assert not string_stable(core.widened(trial))[0]
    unstable = Box((0.5, 0.52), (-0.9, -0.88), (-0.31, -0.3))
    assert not allocate_budget(unstable)["feasible"]


def test_schur_gain_of_a_degenerate_storage_function_is_infinite():
    A, B, C = vertex_systems(np.array([[0.504, -0.886, -0.307], [0.3, -1.0, -0.3]]))
    for P in (np.zeros((2, 2)), np.array([[1.0, 0.0], [0.0, 0.0]]), np.full((2, 2), np.inf), np.full((2, 2), np.nan)):
        assert schur_gain(P, A, B, C) == math.inf  # no exception for a singular or non-finite P
    result = robust_certificate(np.array([[0.504, -0.886, -0.307], [0.3, -1.0, -0.3]]))
    assert schur_gain(result["P"], A, B, C) <= result["gamma"] * (1 + 1e-9)  # the verified P certifies its gamma
