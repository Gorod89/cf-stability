"""Certificate of ResidualIDM: exact box minimum, zero bounds, soundness after adversarial training;
budget with r_max kept and its enforcement (D86), anchored equilibria of the band."""

import copy
import json

import pytest
import torch

from cf_stability.models import ResidualIDM
from cf_stability.models.base import ModelContext
from cf_stability.models.idm import IDM, IDM_PARAM_NAMES, idm_equilibrium_spacing, idm_margin
from cf_stability.stability.analytic import criterion, partials
from cf_stability.stability.certificate import (
    _a_priori,
    _guarantees,
    certificate_a_priori,
    certificate_at_equilibria,
    certify_core,
    core_margin,
    empirical_margin,
    enforce_budget,
    guaranteed_margin,
    max_residual_budget,
    residual_bounds,
    residual_of,
    to_json,
)
from cf_stability.stability.equilibrium import V_GRID, Band, find_equilibria

IDM_P = {"v0": 33.0, "T": 1.5, "s0": 2.0, "a": 2.0, "b": 1.0}  # string stable on the whole speed grid
CONTEXT = ModelContext(idm_params=IDM_P)
ADVERSARIAL_SPEEDS = torch.tensor([6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 29.0])
# global IDM fits on FollowNet HighD, fold 0 (docs/m3_report.md, 3.4): free, and with a margin of at least 0.2 / 0.5
CORE_FREE = {"v0": 29.57, "T": 0.676, "s0": 2.015, "a": 0.351, "b": 0.5}
CORE_M02 = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}
CORE_M05 = {"v0": 35.1, "T": 0.94, "s0": 4.71, "a": 2.74, "b": 0.5}


def random_cases(n: int, seed: int, stationary_inside: bool) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64)

    f_s, f_dv, bounds = u(-0.1, 0.5), u(-1.0, 0.2), torch.stack((u(0.0, 0.2), u(0.0, 0.3), u(0.2, 0.5)), dim=-1)
    f_v = -f_dv + u(-0.1, 0.1) if stationary_inside else u(-1.0, 0.2)
    return f_s, f_dv, f_v, bounds


def brute_force(f_s: float, f_dv: float, f_v: float, b: torch.Tensor, n: int = 801) -> float:
    d_s = torch.tensor([-b[0], 0.0, b[0]], dtype=torch.float64)
    d_dv = torch.linspace(-b[1], b[1], n, dtype=torch.float64)
    d_v = torch.linspace(-b[2], b[2], n, dtype=torch.float64)
    p, q, s = f_v + d_v[:, None, None], f_dv + d_dv[None, :, None], f_s + d_s[None, None, :]
    return (p**2 + 2.0 * p * q - 2.0 * s).min().item()


@pytest.mark.parametrize("stationary_inside", [False, True])
def test_guaranteed_margin_is_the_minimum_over_the_box(stationary_inside):
    f_s, f_dv, f_v, bounds = random_cases(15, int(stationary_inside), stationary_inside)
    exact = guaranteed_margin(f_s, f_dv, f_v, bounds)
    for k in range(len(f_s)):
        brute = brute_force(f_s[k].item(), f_dv[k].item(), f_v[k].item(), bounds[k])
        assert exact[k].item() <= brute + 1e-12 and brute - exact[k].item() < 1e-6
    if stationary_inside:  # p = -q inside the interval of p on an edge of q
        edges = (f_dv - bounds[:, 1], f_dv + bounds[:, 1])
        inside = [(f_v - bounds[:, 2] < -q) & (-q < f_v + bounds[:, 2]) for q in edges]
        assert (inside[0] | inside[1]).sum() >= 10


def test_zero_bounds_give_the_margin_of_the_idm():
    f_s, f_dv, f_v, _ = random_cases(50, 3, False)
    zero = torch.zeros(3, dtype=torch.float64)
    margin = criterion(f_s, f_dv, f_v)["margin"]
    assert torch.allclose(guaranteed_margin(f_s, f_dv, f_v, zero), margin, rtol=0, atol=1e-15)
    model = ResidualIDM(CONTEXT, r_max=0.0).eval()
    cert = certificate_at_equilibria(model)
    assert torch.equal(cert["bounds"], zero) and cert["n_holds"] == len(V_GRID)
    idm = IDM(IDM_P)
    v = torch.tensor(V_GRID, dtype=torch.float64)
    s_e = idm_equilibrium_spacing(v, IDM_P["v0"], IDM_P["T"], IDM_P["s0"])
    reference = criterion(*partials(idm, s_e, v))["margin"]
    assert torch.allclose(cert["guaranteed_margin"], cert["margin_idm"], rtol=0, atol=1e-15)
    assert torch.allclose(cert["guaranteed_margin"], reference, rtol=0, atol=1e-6)  # float32 IDM of the hybrid
    assert torch.allclose(empirical_margin(model), reference, rtol=0, atol=1e-6)


def adversarial_training(model: ResidualIDM, steps: int, speeds: torch.Tensor) -> None:
    """Train the residual to minimise the margin of the hybrid at its equilibria (updated every 10 steps)."""
    opt = torch.optim.Adam(model.g.parameters(), lr=1e-2)
    model.train()
    for step in range(steps):
        if step % 10 == 0:
            eq = find_equilibria(model, speeds)
        f = partials(model, eq.s[eq.found], eq.v[eq.found], create_graph=True)
        loss = criterion(*f)["margin"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def regression_training(model: ResidualIDM, steps: int) -> None:
    """Continue on other data: fit random accelerations on random states."""
    gen = torch.Generator().manual_seed(7)
    x = torch.tensor([2.0, -4.0, 0.0]) + torch.tensor([80.0, 8.0, 35.0]) * torch.rand(512, 1, 3, generator=gen)
    target = torch.sin(x[:, 0, 0] / 7.0) + 0.5 * torch.cos(x[:, 0, 2])
    opt = torch.optim.Adam(model.g.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        loss = torch.mean((model(x) - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def check_soundness(model: ResidualIDM) -> dict:
    """Empirical margin >= guaranteed margin, a priori <= a posteriori, at the speeds with an equilibrium."""
    cert, prior = certificate_at_equilibria(model), certificate_a_priori(model, n_scan=200)
    found = cert["found"]
    assert found.sum() >= 20 and prior["feasible"].all()
    assert (empirical_margin(model)[found] >= cert["guaranteed_margin"][found] - 1e-9).all()
    assert (prior["guaranteed_margin"][found] <= cert["guaranteed_margin"][found] + 1e-9).all()
    assert not (prior["holds"] & ~cert["holds"])[found].any()
    json.dumps(to_json(cert), allow_nan=False), json.dumps(to_json(prior), allow_nan=False)
    return cert


def test_certificate_is_sound_after_adversarial_training_and_fine_tuning():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT, lipschitz=2.0)
    initial = certificate_at_equilibria(model)
    adversarial_training(model, 200, ADVERSARIAL_SPEEDS)
    cert = check_soundness(model)
    assert cert["found"].all()
    trained = torch.isin(cert["v"], ADVERSARIAL_SPEEDS.double())
    emp = empirical_margin(model)
    assert (emp[trained] < initial["margin_idm"][trained] - 0.05).all()  # the residual did its damage
    gap_after, gap_before = emp - cert["guaranteed_margin"], initial["margin_idm"] - cert["guaranteed_margin"]
    assert gap_after[trained].min() < 0.5 * gap_before[trained].min()  # ... close to the guaranteed bound
    regression_training(model, 150)
    check_soundness(model)


def test_largest_budget():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT, r_max=0.5, lipschitz=0.1).eval()  # budget 0.05
    unit = residual_bounds(model) / 0.05
    cert, prior = certificate_at_equilibria(model), certificate_a_priori(model, n_scan=200)
    assert cert["n_holds"] == len(V_GRID) and prior["n_holds"] >= 20
    for c in (cert, prior):  # the certificate holds at the model's own budget exactly where that is admissible
        assert torch.equal(c["max_budget"] >= 0.05, c["holds"])
    below, above = 0.999 * cert["max_budget"], 1.001 * cert["max_budget"]
    f = (cert["f_s"], cert["f_dv"], cert["f_v"])
    assert _guarantees(*f, below[:, None] * unit)["holds"].all()
    assert not _guarantees(*f, above[:, None] * unit)["holds"].any()
    v, ratio = torch.tensor(V_GRID, dtype=torch.float64), 0.5 / 0.1
    idm = model.idm.double()
    for scale, expected in ((0.999, True), (1.001, False)):
        beta = scale * prior["max_budget"]
        out = _a_priori(idm, v, torch.sqrt(beta * ratio), beta[:, None] * unit, 200)
        assert ((out["holds"] | ~out["feasible"]) == expected).all()
    assert torch.equal(max_residual_budget(model, a_priori=True, n_scan=200), prior["max_budget"])


def test_budget_with_r_max_kept():
    torch.manual_seed(0)
    model = ResidualIDM(CONTEXT, r_max=1.0, lipschitz=0.3).eval()
    budget = max_residual_budget(model, a_priori=True, keep_r_max=True, n_scan=200)
    assert (budget > 0).all()
    unit = residual_bounds(model) / 0.3  # bounds per unit of r_max * lipschitz
    v = torch.tensor(V_GRID, dtype=torch.float64)
    idm = model.idm.double()
    for scale, expected in ((0.999, True), (1.001, False)):  # the feasible spacings stay those of r_max = 1
        out = _a_priori(idm, v, torch.ones_like(v), (scale * budget)[:, None] * unit, 200)
        assert ((out["holds"] | ~out["feasible"]) == expected).all()
    model.lipschitz = 0.7  # the ratio r_max : lipschitz does not enter
    assert torch.equal(max_residual_budget(model, a_priori=True, keep_r_max=True, n_scan=200), budget)
    assert torch.equal(certificate_a_priori(model, n_scan=200, keep_r_max=True)["max_budget"], budget)


def test_enforced_budget_certifies_and_survives_training():
    torch.manual_seed(0)
    model = ResidualIDM(ModelContext(idm_params=CORE_M05))  # r_max = lipschitz = 1
    assert not certificate_a_priori(model, n_scan=200)["holds"].all()
    budget = enforce_budget(model, 0.9, n_scan=200)
    assert budget["r_max"] == model.r_max == 1.0 and budget["safety"] == 0.9 and budget["admissible"] > 0
    assert model.lipschitz == budget["lipschitz"] == pytest.approx(0.9 * budget["admissible"], rel=1e-12)
    assert certificate_a_priori(model, n_scan=200)["holds"].all()
    regression_training(model, 100)  # the a priori certificate depends on the weights through the layer norms only
    assert certificate_a_priori(model, n_scan=200)["holds"].all()


def test_budget_shrinks_with_the_margin_of_the_core():
    admissible = {}
    for margin, core in ((0.5, CORE_M05), (0.2, CORE_M02)):
        torch.manual_seed(0)
        admissible[margin] = enforce_budget(ResidualIDM(ModelContext(idm_params=core), r_max=0.5), 0.9)["admissible"]
    assert admissible[0.5] > admissible[0.2] > 0
    for core in (CORE_M02, CORE_FREE):  # r_max = 1 admits compressed spacings where these cores are string unstable
        torch.manual_seed(0)
        model = ResidualIDM(ModelContext(idm_params=core))
        with pytest.raises(ValueError, match="no admissible residual budget"):
            enforce_budget(model, 0.9, n_scan=200)
        assert model.lipschitz == 1.0
    with pytest.raises(ValueError, match="safety"):
        enforce_budget(ResidualIDM(CONTEXT), 1.5)


def test_one_residual_certifies_many_cores():
    """D118: ``certify_core`` of a core is ``certificate_a_priori`` of the ResidualIDM that carries this core and
    the residual (``residual_of``), speed by speed and bit by bit: the residual enters through r_max and its
    bounds only. With the residual of E4 (r_max 0.3, r_max * lipschitz 0.09 below the admissible 0.117 of a
    margin-0.2 core, D86) the margin cores hold and the free core of HighD does not."""
    torch.manual_seed(0)
    model = ResidualIDM(ModelContext(idm_params=CORE_M02), r_max=0.3, lipschitz=0.3)
    residual = residual_of(model)
    assert residual["r_max"] == 0.3 and residual["product"] == pytest.approx(0.09) and residual["bounds"].shape == (3,)
    assert torch.equal(residual["bounds"], residual_bounds(copy.deepcopy(model).double()))
    assert residual["layer_norms"] == pytest.approx(model.layer_norms(), rel=1e-6)
    holds = {}
    for name, core in (("m02", CORE_M02), ("m05", CORE_M05), ("free", CORE_FREE)):
        hybrid = copy.deepcopy(model).double()
        hybrid.idm.set_params(core)  # the core in float64, as certify_core takes it
        expected = certificate_a_priori(hybrid, n_scan=200, keep_r_max=True)
        out = certify_core(core, residual["r_max"], residual["bounds"], n_scan=200)
        for key in ("holds", "guaranteed_margin", "s_low", "s_high", "feasible", "s_worst"):
            torch.testing.assert_close(out[key], expected[key], rtol=0, atol=0, equal_nan=True)
        holds[name] = bool(out["holds"].all())
    assert holds == {"m02": True, "m05": True, "free": False}
    # a larger residual (r_max * lipschitz 0.3) breaks the certificate of the margin-0.2 core at compressed spacings
    wide = certify_core(CORE_M02, 0.3, 3.0 * residual["bounds"], n_scan=200)
    assert not wide["holds"].all() and wide["feasible"].all()


def test_band_restricts_the_certificate_to_the_anchored_equilibria():
    model = ResidualIDM(CONTEXT, r_max=0.0).eval()  # the hybrid is its IDM: equilibria in closed form
    listed = torch.arange(10.0, 26.0, dtype=torch.float64)  # band on 10-25 m/s, below the equilibrium from 20 m/s
    s_e = idm_equilibrium_spacing(listed, IDM_P["v0"], IDM_P["T"], IDM_P["s0"])
    low, high = torch.where(listed < 20, 0.8, 0.5) * s_e, torch.where(listed < 20, 1.2, 0.9) * s_e
    columns = {"v": listed, "s_low": low, "s_median": s_e, "s_high": high}
    band = Band.from_mapping({key: column.tolist() for key, column in columns.items()})
    cert, free = certificate_at_equilibria(model, band=band), certificate_at_equilibria(model)
    v = cert["v"]
    outside = (v >= 20) & (v <= 25)
    assert free["n_holds"] == len(V_GRID) and free["found"].all()
    assert torch.equal(cert["found"], ~outside) and torch.equal(cert["holds"], ~outside)
    assert cert["n_holds"] == len(V_GRID) - 6
    assert [s for s, out in zip(cert["status"], outside.tolist()) if out] == ["outside"] * 6
    assert (cert["max_budget"][outside] == 0).all() and (cert["max_budget"][~outside] > 0).all()
    anchored = empirical_margin(model, band=band)
    assert torch.isnan(anchored[outside]).all()
    assert torch.allclose(anchored[~outside], empirical_margin(model)[~outside], rtol=0, atol=1e-9)
    assert torch.allclose(cert["s"][~outside], free["s"][~outside], rtol=0, atol=1e-6)


def test_core_margin_is_the_closed_form_margin_of_the_idm():
    model = ResidualIDM(ModelContext(idm_params=CORE_FREE))
    v = torch.tensor(V_GRID, dtype=torch.float64)
    of_core = idm_margin(v, *model.idm.theta.detach().double().unbind())  # the float32 core
    of_fit = idm_margin(v, *torch.tensor([CORE_FREE[k] for k in IDM_PARAM_NAMES], dtype=torch.float64))
    margin = core_margin(model)
    assert torch.isnan(margin[-1]) and torch.equal(torch.isnan(margin), torch.isnan(of_core))  # v0 < 30 m/s
    assert torch.allclose(margin[:-1], of_core[:-1], rtol=0, atol=1e-12)
    assert torch.allclose(margin[:-1], of_fit[:-1], rtol=1e-5)
