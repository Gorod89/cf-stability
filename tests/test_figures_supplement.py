"""Supplementary figures and tables of M8 (cf_stability/eval/figures_supplement.py, scripts/make_figures_supplement.py;
docs/m8_contract.md, items 1, 2, 3 and 8).

The hand-made tree holds audits of E1 and of the chosen E2 weight whose gains are the exact memoryless response of
their partial derivatives (one law in the band -omega^2 < M < 0, one with memory, flagged and incomplete records),
certificates and audits of a certified hybrid (one certificate without per-speed bounds next to a saved untrained
ResidualIDM, so that the bounds are computed from the model, and one speed that violates its bound), an event set
with known spacings per driver and a corridor scenario with a ground truth and three laws. Checked: every figure
(PNG, PDF, caption) and table (CSV, Markdown) is written, their numbers, notes instead of exceptions for missing
inputs, and the script as a child process.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cf_stability.data.schema import Event, EventSet
from cf_stability.eval.figures_supplement import (
    NOTES_FILE, SupplementConfig, band_width_rows, certificate_points, exact_gain, expansion_gain, law_color,
    lowfreq_rows, make_supplement, map_category,
)  # fmt: skip
from cf_stability.models import ResidualIDM, save_model
from cf_stability.models.base import ModelContext
from cf_stability.stability.analytic import criterion, partials
from cf_stability.stability.equilibrium import find_equilibria
from cf_stability.utils import REPO_ROOT, write_json
from test_corridor_macro import constant_paths, sample

OMEGA = [float(w) for w in np.geomspace(0.02, 2.0, 25)]
SPEEDS = (10.0, 15.0, 20.0, 25.0)
STABLE = (0.4, -0.6, -0.5)  # f_s, f_dv, f_v: M = 0.05 > 0, slow pole 0.4 / 1.1 rad/s
BLIND = (0.0103, -0.5, -0.02)  # M = -0.0002: |G| < 1 at 0.02 rad/s, the expansion says > 1
IDM_P = {"v0": 33.0, "T": 1.5, "s0": 2.0, "a": 2.0, "b": 1.0}  # string stable on the speed grid
CHOSEN = {"mlp": {"kind": "jacobian", "weight": 1.0}, "gru": {"kind": "gain", "weight": 0.1}}
FIGURES = ("lowfreq_expansion", "certificate_tightness", "band_width", "stability_map_mlp", "stability_map_gru",
           "contours_i80_p0", "fd_i80_p0")  # fmt: skip
TABLES = ("lowfreq_expansion", "certificate_tightness", "band_width")


def gain_of(f_s: float, f_dv: float, f_v: float, scale: float = 1.0) -> list[float]:
    w = np.asarray(OMEGA)
    return (scale * np.abs((f_s - 1j * w * f_dv) / (f_s - w**2 - 1j * w * (f_dv + f_v)))).tolist()


def record(v: float, partial: tuple[float, float, float] | None = STABLE, status: str = "ok",
           flagged_at: int | None = None, scale: float = 1.0, margin: float | None = None,
           s: float | None = None) -> dict:  # fmt: skip
    """A record of audit.equilibria: the gains of the exact memoryless response of ``partial`` times ``scale``."""
    out = {"v": v, "s": s if s is not None else 2.0 + 1.2 * v, "status": status, "in_support": v <= 27.0,
           "in_band": status != "outside", "band_low": 0.8 * v, "band_high": 3.0 * v, "unstable": False}  # fmt: skip
    if status == "none":
        return {**out, "s": None, "f_s": None, "f_dv": None, "f_v": None, "margin": None, "gain": None,
                "clipped": None, "stopped": None, "collided": None, "unstable": None}  # fmt: skip
    f_s, f_dv, f_v = partial if partial is not None else (0.0, 0.0, 0.0)
    gain = gain_of(f_s, f_dv, f_v, scale) if partial is not None else [1.0] * len(OMEGA)
    flags = [k == flagged_at for k in range(len(OMEGA))]
    if margin is None:
        margin = f_v**2 + 2 * f_v * f_dv - 2 * f_s
    return {**out, "f_s": f_s, "f_dv": f_dv, "f_v": f_v, "gain": gain, "clipped": flags,
            "stopped": [False] * len(OMEGA), "collided": [False] * len(OMEGA), "max_gain": max(gain),
            "unstable": max(gain) > 1.02, "margin": margin}  # fmt: skip


def write_audit(run: Path, records: list[dict], band: bool = True) -> None:
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "stability.json", {"audit": {"omega": OMEGA, "support_v": [5.0, 27.0], "equilibria": records}})
    if band:
        v = [5.0, 10.0, 15.0, 20.0, 25.0]
        write_json(run / "metrics.json", {"context": {"band": {
            "v": v, "s_low": [0.8 * x for x in v], "s_median": [1.5 * x for x in v], "s_high": [3.0 * x for x in v]}}})


# ---------------------------------------------------------------------------------------------- the tree


def write_expansion_runs(root: Path) -> None:
    def run(experiment: str, model: str) -> Path:
        return root / experiment / "follownet_highd" / model / "driver_fold0_seed0"

    write_audit(run("e1", "idm"), [record(v) for v in SPEEDS[:3]] + [record(25.0, BLIND)])
    write_audit(run("e1", "mlp"), [
        record(5.0, status="none"), *(record(v) for v in SPEEDS), record(26.0, flagged_at=0),
        record(27.0, (0.3, -0.5, -0.05), status="outside", s=90.0), {**record(28.0), "gain": None},
        record(29.0, None, status="indifferent"),
    ])  # fmt: skip
    write_audit(run("e2_jacobian_w1", "mlp"), [record(v, (0.5, -0.7, -0.05)) for v in SPEEDS])
    write_audit(run("e1", "gru"), [record(v, (0.2, -0.4, 0.1), scale=1.1) for v in SPEEDS]  # memory: 10 % above (1)
                + [{**record(26.0, (0.2, 0.3, 0.1), scale=1.1), "residual": [0.5] * len(OMEGA)}])  # locally unstable
    # e2_gain_w0.1/gru (the chosen weight) has no audit: a note


def hybrid_audit(model: ResidualIDM, speeds: tuple[float, ...]) -> list[dict]:
    """Records of the audit of a hybrid without band: its equilibria and the margin of its own derivatives."""
    m = copy.deepcopy(model).double().eval()
    eq = find_equilibria(m, torch.tensor(speeds, dtype=torch.float64))
    margin = criterion(*partials(m, eq.s, eq.v))["margin"]
    return [{"v": float(v), "s": float(s), "status": status, "in_band": None, "margin": float(mg)}
            for v, s, status, mg in zip(eq.v.tolist(), eq.s.tolist(), eq.status, margin.tolist())]  # fmt: skip


def write_certificate_runs(root: Path) -> None:
    def run(experiment: str, data: str, fold: int) -> Path:
        return root / experiment / data / "residual_idm" / f"driver_fold{fold}_seed0"

    for experiment, data in (("e4_stable", "follownet_highd"), ("e4_stable_ft", "ngsim_i80")):
        records = [record(v, margin=0.1) for v in SPEEDS[:3]]
        records += [record(25.0, margin=0.06 if experiment == "e4_stable" else 0.1),
                    record(27.0, status="outside", margin=0.07, s=90.0)]  # fmt: skip
        write_audit(run(experiment, data, 0), records, band=False)
        per_speed = [{"v": r["v"], "s": r["s"], "status": r["status"], "found": r["status"] == "ok",
                      "a_priori_margin": 0.05, "guaranteed_margin": 0.08} for r in records]  # fmt: skip
        write_json(run(experiment, data, 0) / "certificate.json",
                   {"applicable": True, "residual": {"r_max": 0.3}, "per_speed": per_speed})  # fmt: skip
    # fold 1 of e4_stable: a global verdict only, next to the saved model; fold 1 of e4_stable_ft is missing
    hybrid = run("e4_stable", "follownet_highd", 1)
    hybrid.mkdir(parents=True)
    torch.manual_seed(0)
    model = ResidualIDM(ModelContext(idm_params=IDM_P), r_max=0.3, lipschitz=0.01)
    save_model(model, hybrid / "model.pt")
    write_audit(hybrid, hybrid_audit(model, SPEEDS), band=False)
    write_json(hybrid / "certificate.json", {"applicable": True, "residual": {"r_max": 0.3},
                                             "a_priori": {"holds": True}, "config": {"certificate": {"n_scan": 50}}})


def driver_event(index: int, driver: int, speed: float, spacing: float, n: int) -> Event:
    t = 0.1 * np.arange(n)
    x_lead = 100.0 + speed * t
    zeros = np.zeros(n)
    return Event(event_id=f"ev{index}", dataset="synthetic", site="s", follower_id=f"synthetic/s/d{driver}",
                 leader_id="l", t=t, s=np.full(n, spacing), dv=zeros, v=np.full(n, speed), a=zeros,
                 v_lead=np.full(n, speed), x_lead=x_lead, x_follower=x_lead - spacing)  # fmt: skip


def write_events(events_root: Path) -> None:
    """Ten drivers with 300 samples at 10 m/s (spacing 10 + d) and at 20 m/s (20 + 2 d), one driver with 10 samples."""
    events = [driver_event(d, d, 10.0, 10.0 + d, 300) for d in range(10)]
    events += [driver_event(10 + d, d, 20.0, 20.0 + 2 * d, 300) for d in range(10)]
    events.append(driver_event(20, 99, 10.0, 50.0, 10))
    EventSet(events).to_parquet(events_root / "follownet_highd")  # ngsim_i80 is missing: a note


def write_corridor(root: Path) -> None:
    def save(directory: Path, names: tuple[str, str], arrays: tuple[dict, dict]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / names[0], **arrays[0])
        np.savez_compressed(directory / names[1], **arrays[1])

    def flow(speed_1: float, speed_2: float) -> tuple[dict, dict]:
        return sample(constant_paths(speed_1, 2.0, 1, 0.37, t_end=300.0)
                      + constant_paths(speed_2, 3.0, 2, 1.13, t_end=300.0))  # fmt: skip

    scenario = root / "scenarios" / "i80_p0"
    save(scenario, ("ground_truth.npz", "vehicles_truth.npz"), flow(10.0, 15.0))
    write_json(scenario / "scenario.json", {"scenario": "i80_p0", "geometry": {"x_in": 20, "x_out": 500},
                                            "window": [60, 300]})  # fmt: skip
    for law, speeds in (("law_a", (10.0, 15.0)), ("law_b", (8.0, 12.0)), ("law_d", (12.0, 14.0))):
        save(root / "i80_p0" / law / "seed0", ("trajectories.npz", "vehicles.npz"), flow(*speeds))


def build_tree(base: Path) -> dict[str, Path]:
    root = base / "runs"
    (root / "_tables" / "m4").mkdir(parents=True)
    write_json(root / "_tables" / "m4" / "chosen_weights.json", CHOSEN)
    (root / "_tables" / "m5").mkdir(parents=True)
    pd.DataFrame({"corridor": ["I-80", "I-80", "I-80", "I-80", "US-101"],
                  "law": ["ground truth", "law_b", "law_a", "law_c", "law_x"]}).to_csv(
        root / "_tables" / "m5" / "laws.csv", index=False)  # fmt: skip
    write_expansion_runs(root)
    write_certificate_runs(root)
    write_events(base / "events")
    write_corridor(root / "corridor")
    configs = base / "configs"
    configs.mkdir()
    (configs / "corridor_metrics.yaml").write_text(
        "design:\n  corridors: {i80: I-80, us101: US-101}\nmacro:\n  waves:\n    lanes: [1, 2]\n", encoding="utf-8")
    return {"base": base, "root": root, "events": base / "events", "configs": configs}


def config(tree: dict[str, Path], out: Path, **changes) -> SupplementConfig:
    root = tree["root"]
    values = dict(
        runs_root=root, out_dir=out / "supplement", tables_dir=out / "m8", tables_m4=root / "_tables" / "m4",
        tables_m5=root / "_tables" / "m5", corridor_root=root / "corridor", events_root=tree["events"],
        configs_dir=tree["configs"], expansion_architectures=("idm", "mlp", "gru"), map_architectures=("mlp", "gru"),
        certificate_experiments=((0.3, "e4_stable", "e4_stable_ft"),), certificate_folds=(0, 1),
        certificate_seeds=(0,), corridors=("i80", "us101"), periods=(0,), main_laws=("law_a",), dpi=50,
    )  # fmt: skip
    return SupplementConfig(**{**values, **changes})


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict[str, Path]:
    return build_tree(tmp_path_factory.mktemp("supplement"))


@pytest.fixture(scope="module")
def made(tree, tmp_path_factory) -> tuple[SupplementConfig, list[str]]:
    cfg = config(tree, tmp_path_factory.mktemp("out"))
    return cfg, make_supplement(cfg)


def notes_of(cfg: SupplementConfig) -> str:
    return (cfg.out_dir / NOTES_FILE).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- outputs


def test_every_figure_and_table_in_both_formats(made):
    cfg, lines = made
    for name in FIGURES:
        for suffix in (".png", ".pdf", ".txt"):
            path = cfg.out_dir / "figures" / f"{name}{suffix}"
            assert path.exists() and path.stat().st_size > 0, path
        assert f"FIGURE {name}: " in "\n".join(lines)
    for name in TABLES:
        for suffix in (".csv", ".md"):
            assert (cfg.tables_dir / f"{name}{suffix}").exists(), name
    written = {p.stem for p in (cfg.out_dir / "figures").glob("*.png")}
    assert written == set(FIGURES)  # US-101 has no scenario: no corridor figures, notes instead
    assert lines[-1].startswith(f"SUPPLEMENT: {len(FIGURES)} figures, 3 tables, ")
    notes = notes_of(cfg)
    assert "- figure contours_us101_p0: not written: no trajectories of the corridor" in notes
    assert "- corridor us101_p0: corridor/scenarios/us101_p0/scenario.json missing" in notes
    for line in notes.splitlines()[1:]:
        assert f"NOTE {line[2:]}" in lines


def test_lowfreq_expansion_numbers(made):
    cfg, _ = made
    table = pd.read_csv(cfg.tables_dir / "lowfreq_expansion.csv")
    assert list(dict.fromkeys(zip(table["architecture"], table["arm"]))) == [
        ("idm", "E1"), ("mlp", "E1"), ("mlp", "E2"), ("gru", "E1")]  # fmt: skip
    assert np.allclose(table["omega"].unique(), OMEGA[:3])
    mlp = table[(table["architecture"] == "mlp") & (table["arm"] == "E1")].sort_values("omega")
    first, second = mlp.iloc[0], mlp.iloc[1]  # the CSV parser may change the last digit of a float key
    assert (first["equilibria"], first["flagged"], first["speeds"]) == (6, 1, 5)  # none, gain missing and
    assert (second["equilibria"], second["flagged"], second["speeds"]) == (6, 0, 6)  # indifferent are not counted
    assert first["abs_diff_median"] < 1e-4 and first["sign_agreement"] == 1.0  # omega far below the slow pole
    assert first["abs_diff_exact_median"] < 1e-12 and first["slow_pole_median"] == pytest.approx(0.4 / 1.1)
    idm = table[table["architecture"] == "idm"]
    assert (idm["sign_disagree"] == 1).all() and (idm["disagree_blind"] == 1).all()
    assert np.allclose(idm["sign_agreement"], 0.75)
    gru = table[table["architecture"] == "gru"]
    exact = np.asarray(gain_of(*(0.2, -0.4, 0.1)))[:3]
    assert np.allclose(gru["abs_diff_exact_median"], 0.1 * exact)  # the memory: 10 % above (1)
    assert (gru["locally_unstable"] == 1).all() and (gru["residual_large"] == 1).all()
    assert (table.loc[table["architecture"] != "gru", ["locally_unstable", "residual_large"]] == 0).all().all()
    notes = notes_of(cfg)
    assert "e2_gain_w0.1/follownet_highd/gru/driver_fold0_seed0/stability.json missing" in notes
    assert "e1/follownet_highd/mlp/driver_fold0_seed0/stability.json: 1 speed(s) left out, field gain missing" in notes
    assert "1 speed(s) left out, indifferent law" in notes
    md = (cfg.tables_dir / "lowfreq_expansion.md").read_text(encoding="utf-8")
    header = next(line for line in md.splitlines() if line.startswith("| architecture"))
    assert header.replace("\\|", "").count("|") == 17  # 16 columns: no unescaped pipe inside a header
    caption = (cfg.out_dir / "figures" / "lowfreq_expansion.txt").read_text(encoding="utf-8")
    assert "left out (1 of 19)" in caption and "stability.json missing" in caption


def test_expansion_formula():
    f_s, f_dv, f_v = np.array([0.4, 0.0103, 1e-3]), np.array([-0.6, -0.5, -0.5]), np.array([-0.05, -0.02, -0.1])
    omega = [1e-3, 0.02]
    expansion, exact = expansion_gain(f_s, f_dv, f_v, omega), exact_gain(f_s, f_dv, f_v, omega)
    assert expansion.shape == exact.shape == (3, 2)
    assert abs(expansion[0, 0] - exact[0, 0]) < 1e-10  # O(omega^4)
    assert expansion[1, 1] > 1.0 > exact[1, 1]  # -omega^2 < M < 0: the expansion is above 1, (1) below
    assert expansion[2, 1] == 0.0  # 1 - omega^2 M / f_s^2 < 0 is clipped
    points = pd.DataFrame({
        "architecture": "x", "arm": "E1", "experiment": "e1", "k": 0, "omega": 0.02, "flagged": [False, False, True],
        "gain_numerical": exact[:, 1], "gain_expansion": expansion[:, 1], "gain_exact": exact[:, 1],
        "margin": f_v**2 + 2 * f_v * f_dv - 2 * f_s, "slow_pole": [1.0, 1.0, 1.0], "local_stable": True,
        "fit_residual": [0.0, 0.2, 0.0],
    })  # fmt: skip
    row = lowfreq_rows(points).iloc[0]
    assert (row["equilibria"], row["flagged"], row["speeds"]) == (3, 1, 2)
    assert (row["sign_disagree"], row["disagree_blind"], row["locally_unstable"], row["residual_large"]) == (1, 1, 0, 1)


def test_certificate_tightness_numbers(made):
    cfg, _ = made
    table = pd.read_csv(cfg.tables_dir / "certificate_tightness.csv")
    rows = {(r["stage"], r["bound"]): r for r in table.to_dict("records")}
    assert set(rows) == {("before", "a priori"), ("before", "at the equilibria"), ("after", "a priori"),
                         ("after", "at the equilibria")}  # fmt: skip
    before = rows[("before", "at the equilibria")]
    assert before["runs"] == 2 and before["speeds"] == 4 + len(SPEEDS)  # the outside equilibrium is not certified
    assert before["negative"] == 1 and before["negative_share"] == pytest.approx(1 / 8)
    assert before["slack_min"] == pytest.approx(-0.02)
    prior = rows[("before", "a priori")]
    assert prior["speeds"] == 5 + len(SPEEDS) and prior["negative"] == 0  # every equilibrium, outside included
    after = rows[("after", "at the equilibria")]
    assert after["runs"] == 1 and after["slack_median"] == pytest.approx(0.02) and after["negative"] == 0
    assert rows[("after", "a priori")]["slack_median"] == pytest.approx(0.05)
    notes = notes_of(cfg)
    assert "e4_stable_ft/ngsim_i80: 1 of 2 runs without input: fold1_seed0 (certificate.json, stability.json)" in notes
    assert ("per-speed bounds not stored in certificate.json of 1 run(s), computed from model.pt with "
            "cf_stability.stability.certificate: e4_stable/follownet_highd/residual_idm/driver_fold1_seed0") in notes


def test_certificate_bounds_from_the_model_hold(tree):
    """The bounds computed from the saved untrained hybrid lie below its audited margin (the proposition)."""
    from cf_stability.eval.figures_supplement import computed_certificate_speeds
    from cf_stability.utils import read_json

    run = tree["root"] / "e4_stable" / "follownet_highd" / "residual_idm" / "driver_fold1_seed0"
    per_speed = computed_certificate_speeds(run, SPEEDS, n_scan=50)
    points, skipped = certificate_points(per_speed, read_json(run / "stability.json")["audit"])
    assert not skipped and len(points) == 2 * len(SPEEDS)
    for p in points:
        assert p["audited"] >= p["certified"] - 1e-12
    by_v = {(p["v"], p["bound"]): p["certified"] for p in points}
    assert all(by_v[(v, "a_priori")] <= by_v[(v, "at_equilibria")] + 1e-9 for v in SPEEDS)


def test_certificate_points_compare_the_same_equilibrium():
    audit = {"equilibria": [{"v": 10.0, "s": 12.0, "status": "ok", "margin": 0.1}]}
    per_speed = [{"v": 10.0, "s": 13.0, "found": True, "a_priori_margin": 0.05, "guaranteed_margin": 0.08}]
    points, skipped = certificate_points(per_speed, audit)
    assert [p["bound"] for p in points] == ["a_priori"]
    assert skipped == {"equilibrium of the certificate differs from that of the audit": 1}


def test_band_width_numbers(made):
    cfg, _ = made
    table = pd.read_csv(cfg.tables_dir / "band_width.csv")
    assert set(table["data"]) == {"follownet_highd"} and len(table) == 26
    at = table.set_index("v")
    pooled = np.concatenate([np.full(300, 10.0 + d) for d in range(10)] + [np.full(10, 50.0)])
    low, median, high = np.quantile(pooled, (0.05, 0.5, 0.95))
    row = at.loc[10.0]
    assert row["samples"] == 3010 and bool(row["band"])
    assert (row["s_low"], row["s_median"], row["s_high"]) == pytest.approx((low, median, high))
    assert row["relative_width"] == pytest.approx((high - low) / median)
    assert row["drivers"] == 10  # the driver with 10 samples is left out of the dispersion
    assert row["driver_std"] == pytest.approx(np.std(np.arange(10.0, 20.0), ddof=1))
    assert row["driver_std_relative"] == pytest.approx(row["driver_std"] / median)
    assert at.loc[20.0, "driver_std"] == pytest.approx(np.std(20.0 + 2 * np.arange(10.0), ddof=1))
    empty = at.loc[15.0]
    assert empty["samples"] == 0 and not bool(empty["band"]) and np.isnan(empty["relative_width"])
    assert empty["drivers"] == 0 and np.isnan(empty["driver_std"])
    notes = notes_of(cfg).replace(cfg.events_root.parent.as_posix() + "/", "")
    assert "band_width: events/ngsim_i80 not readable" in notes
    assert band_width_rows([], "x").empty


def test_stability_maps_and_categories(made):
    cfg, _ = made
    caption = (cfg.out_dir / "figures" / "stability_map_mlp.txt").read_text(encoding="utf-8")
    assert "E1, no penalty (e1) and E2, jacobian penalty, weight 1 (e2_jacobian_w1)" in caption
    assert "none 1" in caption and "outside 1" in caption and "indifferent 1" in caption
    gru = (cfg.out_dir / "figures" / "stability_map_gru.txt").read_text(encoding="utf-8")
    assert "e2_gain_w0.1/follownet_highd/gru/driver_fold0_seed0/stability.json missing" in gru
    assert [map_category(r) for r in (
        {"status": "ok", "unstable": True}, {"status": "multiple", "unstable": False},
        {"status": "ok", "unstable": None}, {"status": "outside", "unstable": True}, {"status": "none"},
        {"status": "indifferent"},
    )] == ["unstable", "stable", "undefined", "outside", "none", "indifferent"]  # fmt: skip


def test_corridor_figures_order_and_notes(made):
    cfg, _ = made
    caption = (cfg.out_dir / "figures" / "contours_i80_p0.txt").read_text(encoding="utf-8")
    assert "every law with a run of seed 0 (3): law_a, law_b, law_d." in caption  # main laws, laws.csv, the others
    assert "lanes 1, 2 together" in caption and "analysis window 60-300 s" in caption
    fd = (cfg.out_dir / "figures" / "fd_i80_p0.txt").read_text(encoding="utf-8")
    assert "the ground truth first, then every law with a run of seed 0 (3): law_a, law_b, law_d." in fd
    notes = notes_of(cfg)
    assert "- corridor i80_p0: no seed0 run of law_c" in notes
    assert "- corridor i80_p0: laws with a run but not in laws.csv: law_d" in notes
    assert "- corridor us101_p0: no seed0 run of law_a, law_x" in notes
    assert law_color("residual_idm_certified_r0.1") == law_color("residual_idm_certified") != law_color("unknown")


def test_too_many_laws_are_cut(tree, tmp_path):
    cfg = config(tree, tmp_path, outputs=("corridor",), corridors=("i80",), max_panels=3)
    lines = make_supplement(cfg)
    assert any("3 laws, 2 panels: left out law_d" in line for line in lines)
    caption = (cfg.out_dir / "figures" / "fd_i80_p0.txt").read_text(encoding="utf-8")
    assert "(2): law_a, law_b." in caption


def test_missing_inputs_give_notes_and_empty_tables(tmp_path):
    cfg = SupplementConfig.under(tmp_path / "runs", events_root=tmp_path / "events", configs_dir=tmp_path / "configs",
                                 dpi=50)  # fmt: skip
    lines = make_supplement(cfg)
    assert lines[-1].startswith("SUPPLEMENT: 0 figures, 3 tables, ")
    for name in TABLES:
        frame = pd.read_csv(cfg.tables_dir / f"{name}.csv")
        assert frame.empty and len(frame.columns) > 3, name  # the header without rows
        assert (cfg.tables_dir / f"{name}.md").exists()
    notes = notes_of(cfg)
    for text in ("_tables/m4/chosen_weights.json missing", "e1/follownet_highd/idm/driver_fold0_seed0/stability.json "
                 "missing", "e4_stable/follownet_highd: 25 of 25 runs without input", "figure lowfreq_expansion: not "
                 "written", "figure stability_map_perl: not written: no stability.json", "corridor_metrics.yaml not "
                 "readable", "_tables/m5/laws.csv not readable", "figure fd_us101_p2: not written"):  # fmt: skip
        assert text in notes, text
    assert not (cfg.out_dir / "figures").exists() or not list((cfg.out_dir / "figures").glob("*.png"))


def test_unknown_output_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown outputs"):
        make_supplement(SupplementConfig.under(tmp_path, outputs=("figure_9",)))


def test_script(tree, tmp_path, made):
    out = tmp_path / "out"
    args = [sys.executable, str(REPO_ROOT / "scripts" / "make_figures_supplement.py"), "--runs-root", str(tree["root"]),
            "--out-dir", str(out / "supplement"), "--tables-dir", str(out / "m8"), "--events-root", str(tree["events"]),
            "--configs-dir", str(tree["configs"]), "--only", "band_width", "--dpi", "50"]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)  # fmt: skip
    assert proc.returncode == 0 and "Traceback" not in proc.stderr, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert any(line.startswith("TABLE band_width: 26 rows") for line in lines)
    assert any(line.startswith("FIGURE band_width: ") for line in lines)
    assert lines[-1].startswith("SUPPLEMENT: 1 figures, 1 tables, ")
    for suffix in (".png", ".pdf", ".txt"):
        assert (out / "supplement" / "figures" / f"band_width{suffix}").exists()
    cfg, _ = made
    expected = pd.read_csv(cfg.tables_dir / "band_width.csv")
    pd.testing.assert_frame_equal(pd.read_csv(out / "m8" / "band_width.csv"), expected)
