"""Tables of the corridor (cf_stability/eval/corridor_tables.py) from a hand-made tree of runs.

Two scenarios of corridor A x three seeds; the laws' macro errors rise with the instability of their members so
that the Spearman correlation is 1; the certified hybrid repeats the metrics of idm_global run by run (the TOST
accepts equal samples); a law without its file, missing and unreadable macro.json files leave empty cells and
lines in missing.txt. The fraction of idm_heterogeneous is checked against an independent NumPy evaluation of
the exact discrete-time transfer function. The verdicts of H12 (D108) use laws of the tree as the networks, a
few NGSIM runs for the micro part, a second corridor B, the residual-amplitude table (D111) and the
sensitivity table (D113).
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from cf_stability.eval.corridor_tables import (
    CorridorTablesConfig,
    corridor_text,
    correlation,
    correlation_ci,
    flatten_macro,
    idm_unstable_share,
    make_corridor_tables,
    permutation_p,
    verdict_correlation,
    verdict_every,
    verdict_h12_1,
)
from cf_stability.utils import write_json

SCENARIOS, SEEDS = ("a_p0", "a_p1"), (0, 1, 2)
LAWS = ("idm_global", "idm_heterogeneous", "residual_idm_certified", "law_c", "law_d", "law_e", "law_f", "law_x")
# law: (members' shares unstable among equilibria, members' band shares stable, macro error, collisions)
SPEC = {
    "idm_global": ((0.0, 0.0), (1.0, 1.0), 0.10, 0.0),
    "residual_idm_certified": ((0.0, 0.0), (1.0, 0.8), 0.10, 0.0),
    "law_c": ((0.1, 0.3), (0.7, 0.9), 0.20, 0.0),
    "law_d": ((0.3, 0.5), (0.5, 0.7), 0.30, 0.0),
    "law_e": ((0.5, 0.7), (0.3, 0.5), 0.50, 0.0),
    "law_f": ((0.7, 0.9), (0.1, 0.3), 0.90, 2.5),  # collides
}
HETEROGENEOUS = {"T": np.array([1.556, 0.6]), "s0": np.array([1.696, 1.0]), "a": np.array([0.893, 0.25]),
                 "v0": np.float64(26.6), "b": np.float64(0.5)}  # fmt: skip
SUPPORT = [0.0, 18.05]
FOLDS, RUN_SEEDS = (0, 1), (0, 1)
MICRO = {  # H12.2: <experiment>/<model> -> spacing RMSE of every driver of its test parts
    "e3_reference/idm": 4.0, "e4_stable_ft/residual_idm": 3.6, "e4_free_ft/residual_idm": 3.8, "e4_free_ft/mlp": 4.4,
}  # fmt: skip
DESIGN = {  # the design of the tree (the networks of H12.1 are laws of the tree)
    "scenarios": list(SCENARIOS), "corridors": {"a": "A"}, "laws": list(LAWS), "rmax_laws": [], "seeds": list(SEEDS),
    "n_resamples": 200, "networks": ["law_e", "law_f", "law_c"], "correlation_min_laws": 5, "n_permutations": 2000,
    "run_folds": list(FOLDS), "run_seeds": list(RUN_SEEDS),
    "rmax": [{"r_max": 0.3, "core": "certified", "highd": "e4_stable", "ngsim": "e4_stable_ft",
              "law": "residual_idm_certified"},
             {"r_max": 0.1, "core": "certified", "highd": "e4_stable_r0.1", "ngsim": "e4_stable_ft_r0.1",
              "law": "residual_idm_certified_r0.1"}],
    "sensitivity_scenario": "a_p1", "sensitivity_variants": ["v1", "v2"],
    "sensitivity_laws": ["idm_global", "law_c", "law_e"], "sensitivity_seeds": [0, 1],
    "sensitivity_order": ["idm_global", "law_c", "law_e"],
}  # fmt: skip


def metrics(law: str, scenario: str, seed: int, error: float | None = None) -> dict:
    """macro.json of a run: the certified hybrid repeats idm_global; values vary over scenario and seed."""
    base = "idm_global" if law == "residual_idm_certified" else law
    k = int(scenario[-1]) + 0.5 * seed
    error = SPEC[law][2] + 0.01 * seed if error is None else error
    dynamic = 2.0 * error  # FD, wave speed, waves and wave amplitude: rises with the instability as well
    return {
        "window": [180.0, 840.0],
        "throughput_vph": 6000.0 + 100.0 * k + (0.0 if base == "idm_global" else 300.0), "mean_speed": 6.0 + 0.1 * k,
        "queue_discharge_flow": 5500.0 + 50.0 * k, "flow_peak_2min": 6200.0 + 40.0 * k, "capacity_drop": 0.1 + 0.01 * k,
        "congested_share": 0.9 + 0.02 * k, "fd_scatter": 800.0 + 10.0 * k,
        "waves": {"n_waves": 3 + seed, "wave_speed": -7.0 - 0.1 * k, "wave_speed_xcorr": -5.4 - 0.05 * k,
                  "wave_amplitude": 4.0 + 0.2 * k},
        "travel_time": {"n": 100, "mean": 70.0 + k, "median": 69.0 + k, "p10": 50.0, "p90": 90.0, "values": [70.0]},
        "collisions_per_1000_vkm": SPEC[law][3], "collision_episodes": 3.5 if SPEC[law][3] else 0.0,
        "vehicles_in_contact_share": 0.02 * SPEC[law][3], "inserted_share": 1.0, "mean_depart_delay_s": 0.0,
        "macro_error": {"components": {"throughput": 0.05, "travel_time": error, "fd": dynamic, "wave_speed": -dynamic,
                                       "n_waves": dynamic, "wave_amplitude": dynamic},
                        "value": error, "n_components": 6},
        "macro_error_dynamic": {"components": {"fd": dynamic, "wave_speed": -dynamic, "waves": dynamic,
                                               "wave_amplitude": dynamic}, "value": dynamic, "n_components": 4},
        "config_hash": "h",
    }  # fmt: skip


def write_training_run(runs, experiment: str, data: str, model: str, fold: int, seed: int, rmse: float,
                       unstable: float = 0.0, a_priori: bool | None = None) -> None:  # fmt: skip
    run = runs / experiment / data / model / f"driver_fold{fold}_seed{seed}"
    drivers = [f"{data}-{fold}-{k}" for k in range(3)]
    write_json(run / "metrics.json", {"config_hash": f"{experiment}-{fold}-{seed}", "test": {"rmse_s_mean": rmse}})
    write_json(run / "stability.json", {"audit": {"summary": {"share_unstable_numerical": {"support": unstable},
                                                              "band_numerical": {"support": {"stable": 1.0}}}}})  # fmt: skip
    pd.DataFrame({"event_id": [f"{d}|e" for d in drivers], "follower_id": drivers, "rmse_s": rmse, "rmse_v": 0.5,
                  "collided": False}).to_parquet(run / "test_events.parquet", index=False)  # fmt: skip
    if a_priori is not None:
        write_json(run / "certificate.json", {"model": model, "applicable": True, "a_priori": {"holds": a_priori}})


def build(root, design: dict | None = None) -> dict:
    """The tree of the module docstring under ``root``; the tables of ``design`` (DESIGN by default)."""
    corridor, runs = root / "runs" / "corridor", root / "runs"
    for law in SPEC:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                run = corridor / scenario / law / f"seed{seed}"
                run.mkdir(parents=True)
                if not (law == "law_c" and scenario == "a_p1" and seed == 2):  # a run without macro.json
                    write_json(run / "macro.json", metrics(law, scenario, seed))
                if not (law == "law_d" and seed == 2):  # runs without run.json: no boundary statistics
                    write_json(run / "run.json", {"g_mean": 0.9 + 0.01 * seed, "dN_mean": float(seed)})
    old = metrics("law_f", "a_p0", 0)  # a macro.json of the first loop: removed vehicles counted as n_collisions,
    old["n_collisions"] = old.pop("collision_episodes")  # and written before the dynamic error (from its components)
    del old["macro_error_dynamic"]
    write_json(corridor / "a_p0" / "law_f" / "seed0" / "macro.json", old)
    (corridor / "a_p0" / "law_d" / "seed0" / "macro.json").write_text("{not json", encoding="utf-8")
    write_json(corridor / "a_p1" / "law_e" / "seed1" / "macro.json", {**metrics("law_e", "a_p1", 1),
                                                                     "window": [120.0, 840.0]})  # fmt: skip
    for scenario in SCENARIOS:
        write_json(corridor / "scenarios" / scenario / "macro.json", {**metrics("idm_global", scenario, 0),
                                                                     "throughput_vph": 7000.0, "macro_error": None})  # fmt: skip
    laws = corridor / "laws"
    for law, (unstable, stable, _, _) in SPEC.items():
        members = []
        for k, (share, band) in enumerate(zip(unstable, stable)):
            member = runs / "members" / law / f"driver_fold{k}_seed0"
            write_json(member / "stability.json", {"audit": {"summary": {
                "share_unstable_numerical": {"all": 0.0, "support": share},
                "band_numerical": {"support": {"stable": band, "unstable": 1.0 - band}},
            }}})  # fmt: skip
            members.append({"run": f"members/{law}/driver_fold{k}_seed0", "model": law})
        write_json(laws / f"{law}.json", {"law": law, "kind": "idm" if law == "idm_global" else "models",
                                          "members": members})  # fmt: skip
    np.savez(laws / "idm_heterogeneous.npz", **HETEROGENEOUS)
    write_json(laws / "idm_heterogeneous.json", {"law": "idm_heterogeneous", "kind": "idm_heterogeneous", "members": [],
                                                 "table": "idm_heterogeneous.npz", "support_v": SUPPORT, "window": 1})  # fmt: skip
    for item, rmse in MICRO.items():  # H12.2 and e4_rmax: NGSIM runs (the IDM with the first seed), HighD runs
        experiment, model = item.split("/")
        for fold, seed in itertools.product(FOLDS, RUN_SEEDS[:1] if model == "idm" else RUN_SEEDS):
            certified = experiment == "e4_stable_ft"
            write_training_run(runs, experiment, "ngsim_i80", model, fold, seed, rmse, 0.0 if certified else 0.4,
                               a_priori=True if certified else None)  # fmt: skip
    for fold, seed in itertools.product(FOLDS, RUN_SEEDS):
        write_training_run(runs, "e4_stable", "follownet_highd", "residual_idm", fold, seed, 2.5, 0.1, a_priori=True)
    for law, error in (("idm_global", 0.3), ("law_c", 0.2), ("law_e", 0.9)):  # variant v1 of a_p1: law_c ahead
        for seed in (0, 1):
            write_json(corridor / "a_p1_v1" / law / f"seed{seed}" / "macro.json", metrics(law, "a_p1", seed, error))
    write_json(corridor / "scenarios" / "a_p1_v1" / "macro.json", metrics("idm_global", "a_p1", 0))
    out = root / "runs" / "_tables" / "m5"
    cfg = CorridorTablesConfig.from_mapping(design or DESIGN, corridor, runs, out)
    lines = make_corridor_tables(cfg)
    return {"out": out, "lines": lines, "cfg": cfg, "corridor": corridor, "runs": runs}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return build(tmp_path_factory.mktemp("m5"))


def numpy_share(params: dict, support, v_grid=range(5, 31), dt: float = 0.1, threshold: float = 1.02) -> float:
    """Independent evaluation: closed-form IDM partials and the transfer function of the semi-implicit Euler scheme."""
    omega = np.geomspace(0.02, 2.0, 25)
    z = np.exp(1j * omega * dt)
    shares = []
    for T, s0, a in zip(params["T"], params["s0"], params["a"]):
        v0, b = float(params["v0"]), float(params["b"])
        flags = []
        for v in (float(v) for v in v_grid if support[0] <= v <= support[1] and v < v0):
            s_star = s0 + v * T
            s_e = s_star / math.sqrt(1.0 - (v / v0) ** 4)
            f_s = 2 * a * s_star**2 / s_e**3
            f_dv = -a * s_star * v / (s_e**2 * math.sqrt(a * b))
            f_v = -a * (4 * v**3 / v0**4 + 2 * s_star * T / s_e**2)
            gain = (dt**2 * f_s * z - dt * f_dv * (z - 1)) / ((z - 1) ** 2 + dt**2 * f_s * z - dt * (f_dv + f_v) * (z - 1))
            flags.append(np.abs(gain).max() > threshold)
        shares.append(np.mean(flags))
    return float(np.mean(shares))


def test_heterogeneous_idm_share_matches_an_independent_evaluation():
    expected = numpy_share(HETEROGENEOUS, SUPPORT)
    assert 0.0 < expected < 0.5  # the first set is stable everywhere, the second at some speeds only
    out = idm_unstable_share(HETEROGENEOUS, SUPPORT, range(5, 31), np.geomspace(0.02, 2.0, 25))
    assert out["share"] == pytest.approx(expected) and out["n_sets"] == 2 and out["n_speeds"] == 14
    low_v0 = {**HETEROGENEOUS, "v0": np.float64(12.0)}  # speeds 12-18 have no equilibrium and are left out
    out = idm_unstable_share(low_v0, SUPPORT, range(5, 31), np.geomspace(0.02, 2.0, 25))
    assert out["share"] == pytest.approx(numpy_share(low_v0, SUPPORT)) and out["n_sets_with_equilibrium"] == 2
    none = idm_unstable_share({**HETEROGENEOUS, "v0": np.float64(4.0)}, SUPPORT, range(5, 31), np.geomspace(0.02, 2, 25))
    assert math.isnan(none["share"]) and none["n_sets_with_equilibrium"] == 0


def test_laws_table(tree):
    laws = pd.read_csv(tree["out"] / "laws.csv")
    rows = laws[laws["scenario"] == "all"].set_index("law")
    assert (rows["corridor"] == "A").all()
    c = rows.loc["law_c"]
    errors = [SPEC["law_c"][2] + 0.01 * seed for scenario in SCENARIOS for seed in SEEDS][:-1]  # a_p1/seed2 missing
    assert c["runs"] == 5 and c["runs_expected"] == 6 and c["macro_error"] == pytest.approx(np.mean(errors))
    assert c["macro_error_low"] <= c["macro_error"] <= c["macro_error_high"]
    assert c["error_travel_time"] == pytest.approx(np.mean(errors)) and c["error_throughput"] == pytest.approx(0.05)
    assert rows.loc["law_d", "runs"] == 5  # the unreadable macro.json
    g = rows.loc["idm_global"]
    assert g["throughput_vph"] == pytest.approx(np.mean([6000 + 100 * (i + 0.5 * s) for i in (0, 1) for s in SEEDS]))
    assert g["collisions_per_1000_vkm"] == 0.0 and g["runs_with_collisions"] == 0
    assert rows.loc["law_f", "runs_with_collisions"] == 6 and rows.loc["law_f", "collisions_per_1000_vkm"] == 2.5
    assert rows.loc["law_f", "vehicles_in_contact_share"] == pytest.approx(0.05)
    assert g["g_mean"] == pytest.approx(0.91) and g["dN_mean"] == pytest.approx(1.0)  # seeds 0-2, both scenarios
    assert rows.loc["law_d", "dN_mean"] == pytest.approx(2 / 3)  # a_p0/seed1, a_p1/seed0, a_p1/seed1: seed 2 none
    assert rows.loc["law_x", "runs"] == 0 and pd.isna(rows.loc["law_x", "macro_error"])  # nothing: empty cells
    truths = laws[laws["law"] == "ground truth"]
    assert truths["scenario"].tolist() == list(SCENARIOS) and (truths["throughput_vph"] == 7000.0).all()
    text = (tree["out"] / "laws.md").read_text(encoding="utf-8")
    assert text.count("\n# ") == 0 and "## Laws of the corridor: raw metrics" in text and "| A | law_x | 0 |" in text
    assert "| wave speed, cross-correlation (m/s) | wave speed, leading edges (m/s) |" in text
    assert "| vehicles in contact | inserted share | boundary gain g | dN (veh) |" in text
    assert "| peak 2-min flow (veh/h) | capacity drop | congested share |" in text
    assert "Analysis window (scenario.json): 180-840 s." in text  # the run of another window is left out
    assert "(A: 8 laws x 2 scenarios x 3 seeds)" in text
    assert rows.loc["law_e", "runs"] == 5
    assert g["flow_peak_2min"] == pytest.approx(np.mean([6200 + 40 * (i + 0.5 * s) for i in (0, 1) for s in SEEDS]))
    assert g["wave_speed_xcorr"] == pytest.approx(np.mean([-5.4 - 0.05 * (i + 0.5 * s) for i in (0, 1) for s in SEEDS]))
    assert g["congested_share"] == pytest.approx(np.mean([0.9 + 0.02 * (i + 0.5 * s) for i in (0, 1) for s in SEEDS]))


def test_instability_table_and_correlation(tree):
    table = pd.read_csv(tree["out"] / "instability.csv").set_index("law")
    assert table.loc["law_c", "unstable_eq"] == pytest.approx(0.2) and table.loc["law_c", "not_stable"] == pytest.approx(0.2)
    assert table.loc["idm_global", "unstable_eq"] == 0.0 and table.loc["idm_global", "audited"] == 2
    assert table.loc["residual_idm_certified", "not_stable"] == pytest.approx(0.1)
    het = table.loc["idm_heterogeneous"]
    assert het["unstable_eq"] == pytest.approx(numpy_share(HETEROGENEOUS, SUPPORT)) and het["members"] == 2
    assert pd.isna(het["not_stable"]) and het["runs"] == 0
    assert bool(table.loc["law_f", "collides"]) and not bool(table.loc["law_e", "collides"])
    assert pd.isna(table.loc["law_x", "unstable_eq"]) and table.loc["law_c", "corridor"] == "A"
    corr = pd.read_csv(tree["out"] / "instability_correlation.csv")
    corr = corr[corr["error"] == "macro error"]  # the dynamic error: test_components_table_and_dynamic_error
    spearman = corr[(corr["method"] == "spearman") & (corr["measure"] == "unstable among equilibria")].set_index("laws")
    assert spearman.loc["all laws", "n"] == 6 and spearman.loc["all laws", "estimate"] == pytest.approx(1.0)
    assert spearman.loc["laws without collisions", "n"] == 5
    assert spearman.loc["laws without collisions", "estimate"] == pytest.approx(1.0)
    # six laws, the two IDM-like laws tied in both variables: 2 of the 720 orders of the instability give |r| = 1
    assert spearman.loc["all laws", "p_permutation"] == pytest.approx(2 / 720, abs=0.004)
    pearson = corr[(corr["method"] == "pearson") & (corr["measure"] == "unstable among equilibria")].iloc[0]
    x, y = table.loc[["idm_global", "residual_idm_certified", "law_c", "law_d", "law_e", "law_f"], ["unstable_eq", "macro_error"]].T.values
    assert pearson["estimate"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert pearson["estimate_low"] <= pearson["estimate"] <= pearson["estimate_high"] and pearson["n_valid"] > 150
    assert 0.0 < pearson["p_permutation"] <= 1.0
    text = (tree["out"] / "instability.md").read_text(encoding="utf-8")
    assert "## Correlation of instability and macro error over the laws" in text and "| p (permutation) |" in text


def test_permutation_p_value():
    """Monte Carlo p-value of the permutation test against the exact one from all orders of six values."""
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=6), rng.normal(size=6)
    y = y + 1.2 * x
    for method in ("spearman", "pearson"):
        observed = abs(correlation(x, y, method))
        exact = np.mean([abs(correlation(np.array(p), y, method)) >= observed - 1e-12 for p in itertools.permutations(x)])
        assert permutation_p(x, y, method, 20000, seed=1) == pytest.approx(exact, abs=0.01), method
    assert permutation_p(x, y, "spearman", 200, seed=1) == permutation_p(x, y, "spearman", 200, seed=1)
    assert permutation_p(x, y, "spearman", 10) >= 1 / 11  # never 0: (1 + hits) / (1 + permutations)
    assert math.isnan(permutation_p([1.0, 2.0], [1.0, 2.0], "pearson"))  # fewer than three pairs
    assert math.isnan(permutation_p([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "spearman"))  # a constant variable
    assert permutation_p([1.0, 2.0, np.nan, 3.0, 4.0], [1.0, 2.0, 5.0, 3.0, 4.0], "pearson", 100) <= 1.0


def test_components_table_and_dynamic_error(tree):
    comp = pd.read_csv(tree["out"] / "components.csv").set_index("law")
    c = comp.loc["law_c"]
    errors = np.array([SPEC["law_c"][2] + 0.01 * seed for scenario in SCENARIOS for seed in SEEDS][:-1])
    assert c["runs"] == 5 and c["anchored_throughput"] == pytest.approx(0.05)
    assert c["anchored_travel_time"] == pytest.approx(errors.mean()) and pd.isna(c["anchored_mean_speed"])
    assert c["dynamic_fd"] == pytest.approx(2 * errors.mean()) and c["dynamic_wave_speed"] == pytest.approx(-2 * errors.mean())
    assert c["dynamic_waves"] == pytest.approx(2 * errors.mean()) and c["dynamic_wave_amplitude"] == pytest.approx(2 * errors.mean())
    assert c["macro_error"] == pytest.approx(errors.mean()) and c["macro_error_dynamic"] == pytest.approx(2 * errors.mean())
    assert c["dynamic_fd_low"] <= c["dynamic_fd"] <= c["dynamic_fd_high"]
    f_errors = np.array([SPEC["law_f"][2] + 0.01 * seed for scenario in SCENARIOS for seed in SEEDS])
    assert comp.loc["law_f", "macro_error_dynamic"] == pytest.approx(2 * f_errors.mean())  # one from its components
    text = (tree["out"] / "components.md").read_text(encoding="utf-8")
    assert ("| corridor | law | runs | scenarios | throughput (anchored) | mean speed (anchored) | queue discharge (anchored) | "
            "travel time (W1) (anchored) | FD (dynamic) | wave speed (xcorr) (dynamic) | waves (dynamic) | wave "
            "amplitude (dynamic) | macro error | macro error (dynamic) |") in text  # fmt: skip
    assert "| macro error | macro error (dynamic) | components |" in (tree["out"] / "laws.md").read_text(encoding="utf-8")
    table = pd.read_csv(tree["out"] / "instability.csv").set_index("law")
    assert table.loc["law_c", "macro_error_dynamic"] == pytest.approx(2 * errors.mean())
    corr = pd.read_csv(tree["out"] / "instability_correlation.csv")
    dynamic = corr[corr["error"] == "macro error (dynamic)"]
    assert len(dynamic) == 8 and len(corr) == 16  # two measures x two subsets x two methods, for both errors
    spearman = dynamic[(dynamic["method"] == "spearman") & (dynamic["measure"] == "unstable among equilibria")]
    assert spearman.set_index("laws")["estimate"].tolist() == [pytest.approx(1.0), pytest.approx(1.0)]
    assert spearman.set_index("laws").loc["laws without collisions", "n"] == 5
    assert "| A | macro error (dynamic) | unstable among equilibria | all laws | spearman | 6 | 1.000 [" in \
        (tree["out"] / "instability.md").read_text(encoding="utf-8")


def test_tost_accepts_equal_samples(tree):
    tost = pd.read_csv(tree["out"] / "tost.csv").set_index("metric")
    tested = tost.dropna(subset=["p_value"])
    assert set(tested.index) == {
        "throughput_vph", "mean_speed", "queue_discharge_flow", "flow_peak_2min", "capacity_drop", "congested_share",
        "fd_scatter", "n_waves", "wave_speed_xcorr", "wave_speed", "wave_amplitude", "travel_time_mean",
        "travel_time_median", "inserted_share",
    }  # fmt: skip
    assert tested["equivalent"].astype(bool).all() and (tested["pairs"] == 6).all() and (tost["corridor"] == "A").all()
    assert (tested["relative_difference"].abs() < 1e-12).all()
    for key in ("wave_speed", "wave_speed_xcorr"):  # a negative reference is tested on its magnitude
        assert tost.loc[key, "p_value"] < 0.05
    assert pd.isna(tost.loc["collisions_per_1000_vkm", "p_value"])  # reference mean 0: no relative margin


def test_h12_1_degradations(tree):
    """D108: |e_net| - |e_idm| of throughput, travel time and wave speed, paired by scenario and seed."""
    h12 = pd.read_csv(tree["out"] / "h12_1.csv").set_index(["network", "metric"])
    e_travel = h12.loc[("law_e", "travel_time")]  # 0.5 + 0.01 s against 0.1 + 0.01 s: 0.4 in every pair
    assert e_travel["pairs"] == 5 and e_travel["degradation"] == pytest.approx(0.4)
    assert (e_travel["degradation_low"], e_travel["degradation_high"]) == pytest.approx((0.4, 0.4))
    assert bool(e_travel["degraded"]) and bool(e_travel["strong"])
    assert h12.loc[("law_e", "wave_speed"), "degradation"] == pytest.approx(0.8)  # |-2 e|: 1.0 against 0.2
    assert not bool(h12.loc[("law_e", "throughput"), "degraded"])  # 0.05 for every law: no degradation
    c_travel = h12.loc[("law_c", "travel_time")]
    assert c_travel["degradation"] == pytest.approx(0.1) and bool(c_travel["degraded"]) and not bool(c_travel["strong"])
    verdicts = {network: h12.loc[(network, "throughput"), "h12_1"] for network in ("law_e", "law_f", "law_c")}
    assert verdicts == {"law_e": "confirmed", "law_f": "confirmed", "law_c": "open"}  # law_c: wave speed only
    assert bool(h12.loc[("law_f", "throughput"), "complete"]) and not bool(h12.loc[("law_e", "throughput"), "complete"])
    rows = pd.read_csv(tree["out"] / "verdicts.csv")
    overall = rows[(rows["hypothesis"] == "H12.1") & (rows["unit"] == "overall")].iloc[0]
    assert overall["verdict"] == "confirmed" and overall["corridors"] == "A only" and not overall["complete"]
    assert overall["basis"] == ("confirmed for >= 2 of 3 networks on every corridor: A: law_e confirmed, law_f "
                                "confirmed, law_c open")  # fmt: skip
    law_c = rows[(rows["hypothesis"] == "H12.1") & (rows["unit"] == "law_c")].iloc[0]
    assert "wave speed +0.200 [+0.200, +0.200] degraded by >= 0.15" in law_c["basis"]
    assert law_c["corridors"] == "A"


def test_verdict_rules():
    strong = {"degraded": True, "strong": True}
    weak, none, unknown = {"degraded": True, "strong": False}, {"degraded": False, "strong": False}, {"degraded": None}
    assert verdict_h12_1([strong, strong, unknown]) == "confirmed"
    assert verdict_h12_1([strong, weak, none]) == "open"
    assert verdict_h12_1([none, none, none]) == "refuted"
    assert verdict_h12_1([none, none, unknown]) == ""
    assert verdict_every(["confirmed", "confirmed"]) == "confirmed" and verdict_every(["refuted"]) == "refuted"
    assert verdict_every(["confirmed", "refuted"]) == "open" and verdict_every(["confirmed", ""]) == "" and not verdict_every([])
    cfg = CorridorTablesConfig.from_mapping({}, ".", ".", ".")
    assert verdict_correlation({"estimate": 0.7, "p_permutation": 0.01, "n": 10}, cfg) == "confirmed"
    assert verdict_correlation({"estimate": 0.7, "p_permutation": 0.01, "n": 9}, cfg) == "open"
    assert verdict_correlation({"estimate": 0.7, "p_permutation": 0.06, "n": 13}, cfg) == "open"
    assert verdict_correlation({"estimate": 0.29, "p_permutation": 0.5, "n": 13}, cfg) == "refuted"
    assert verdict_correlation({"estimate": float("nan")}, cfg) == ""
    assert corridor_text(["I-80"]) == "I-80 only" and corridor_text(["I-80", "US-101"]) == "I-80 and US-101"


def test_h12_2_and_h12_3(tree):
    micro = pd.read_csv(tree["out"] / "h12_2.csv").set_index(["experiment", "model"])
    candidate = micro.loc[("e4_stable_ft", "residual_idm")]
    assert candidate["role"] == "candidate" and candidate["runs"] == 4 and candidate["drivers"] == 6
    assert candidate["rmse_vs_reference"] == pytest.approx(3.6 / 4.0 - 1.0) and bool(candidate["superior"])
    assert candidate["rmse_vs_reference_pairs"] == 6 and candidate["rmse_runs"] == pytest.approx(3.6)
    assert micro.loc[("e3_reference", "idm"), "runs"] == 2  # the first seed of the IDM
    assert micro.loc[("e4_free_ft", "mlp"), "rmse_vs_reference"] == pytest.approx(0.1)
    assert not bool(micro.loc[("e4_free_ft", "mlp"), "superior"]) and micro.loc[("e4_free_ft", "mlp"), "role"] == "reference row"
    rows = pd.read_csv(tree["out"] / "verdicts.csv").set_index(["hypothesis", "unit"])
    h12_2 = rows.loc[("H12.2", "residual_idm_certified")]
    assert h12_2["verdict"] == "confirmed" and h12_2["corridors"] == "A only"
    assert h12_2["basis"].startswith("both parts hold; macro (TOST +-10 %): A: throughput +0.0 % [+0.0 %, +0.0 %] "
                                     "equivalent, travel time +0.0 % [+0.0 %, +0.0 %] equivalent, wave speed")  # fmt: skip
    assert h12_2["basis"].endswith("micro: RMSE vs idm -10.0 % [-10.0 %, -10.0 %] over 6 drivers, superior")
    assert "e4_free_ft mlp: RMSE vs IDM +10.0 %" in h12_2["information"]
    spearman = rows.loc[("H12.3", "Spearman")]
    assert spearman["verdict"] == "confirmed" and spearman["corridors"] == "A"  # r = 1, p < 0.05, 6 >= 5 laws
    assert not spearman["complete"]  # law_x and idm_heterogeneous have no macro error: 6 of 8 laws
    assert rows.loc[("H12.3", "overall"), "verdict"] == "confirmed"
    assert rows.loc[("H12.3", "Pearson (does not decide)"), "information"] == "information"
    assert ("H12.3", "Spearman, dynamic macro error (does not decide)") in rows.index
    text = (tree["out"] / "verdicts.md").read_text(encoding="utf-8")
    assert "| H12.1 | overall | A only | confirmed |" in text
    verdict_line = next(line for line in tree["lines"] if line.startswith("TABLE verdicts"))
    assert verdict_line.startswith("TABLE verdicts: H12.1 confirmed, H12.2 confirmed, H12.3 confirmed (A only)")


def test_e4_rmax_table(tree):
    rmax = pd.read_csv(tree["out"] / "e4_rmax.csv").set_index("r_max")
    certified = rmax.loc[0.3]
    assert (certified["runs_highd"], certified["a_priori_highd"], certified["runs_ngsim"]) == (4, 4, 4)
    assert certified["a_priori_ngsim"] == 4 and certified["unstable_eq_highd"] == pytest.approx(0.1)
    assert certified["rmse_highd"] == pytest.approx(2.5) and certified["rmse_ngsim"] == pytest.approx(3.6)
    assert certified["runs_corridor"] == 6 and certified["macro_error"] == pytest.approx(0.11)
    assert certified["collisions_per_1000_vkm"] == 0.0 and certified["corridor"] == "A"
    empty = rmax.loc[0.1]
    assert (empty["runs_highd"], empty["runs_ngsim"], empty["runs_corridor"]) == (0, 0, 0) and pd.isna(empty["macro_error"])
    missing = (tree["out"] / "missing.txt").read_text(encoding="utf-8")
    assert "[e4_rmax] e4_stable_r0.1/follownet_highd/residual_idm: run missing (all 4: folds 0, 1, seeds 0, 1)" in missing


def test_sensitivity_table(tree):
    sens = pd.read_csv(tree["out"] / "sensitivity.csv").set_index(["variant", "law"])
    base = sens.loc["baseline"]
    assert base.loc["idm_global", "runs"] == 2 and base.loc["law_e", "runs"] == 1  # a_p1/law_e/seed1: stale window
    assert base.loc["idm_global", "macro_error"] == pytest.approx(0.105)
    assert base["rank"].tolist() == [1, 2, 3] and base["ranking"].iloc[0] == "idm_global < law_c < law_e"
    assert base["order_holds"].astype(bool).all()
    v1 = sens.loc["v1"]
    assert v1["rank"].tolist() == [2, 1, 3] and not v1["order_holds"].astype(bool).any()
    assert v1.loc["law_e", "error_travel_time"] == pytest.approx(0.9)
    v2 = sens.loc["v2"]
    assert (v2["runs"] == 0).all() and v2["ranking"].isna().all() and v2["order_holds"].isna().all()
    missing = (tree["out"] / "missing.txt").read_text(encoding="utf-8")
    assert "[sensitivity] a_p1_v2: scenario not built, no runs" in missing


def test_missing_items_are_listed(tree):
    missing = (tree["out"] / "missing.txt").read_text(encoding="utf-8").splitlines()
    assert "[runs] a_p1/law_c/seed2/macro.json missing" in missing
    assert "[runs] a_p0/law_d/seed0/macro.json unreadable (JSONDecodeError)" in missing
    assert "[runs] A/law_x: run missing (all 6: a_p0, a_p1 x seeds 0, 1, 2)" in missing  # one line for the law
    assert "[laws] laws/law_x.json missing" in missing
    assert "[runs] a_p1/law_e/seed1: window 120-840 s, ground truth 180-840 s: left out, rerun the metrics" in missing
    assert "[runs] A/idm_heterogeneous: run missing (all 6: a_p0, a_p1 x seeds 0, 1, 2)" in missing
    assert any("idm_heterogeneous: band share not stable not defined" in line for line in missing)
    assert any(line.startswith("[tost] A: collisions_per_1000_vkm") for line in missing)
    assert [line.split(":")[0] for line in tree["lines"]] == [
        "TABLE laws", "TABLE components", "TABLE instability", "TABLE tost", "TABLE h12_1", "TABLE h12_2",
        "TABLE e4_rmax", "TABLE sensitivity", "TABLE verdicts", "TABLE asymmetry", "TABLE asymmetry_contrasts",
        "TABLE correlation_pooled", "TABLE power", "TABLE temporal", "TABLE correlation_clustered",
        "TABLE contacts_absolute", "TABLE h12_2_error", "TABLE ablation_pairs"]  # fmt: skip  # M8 and its review last


def test_a_second_corridor(tmp_path):
    """Corridor B enters with the macro.json of its ground truth; corridor C (configured, no ground truth) gets a
    line in missing.txt; every table has rows per corridor and the verdicts name the corridors used."""
    design = {**DESIGN, "scenarios": [*SCENARIOS, "b_p0", "c_p0"], "corridors": {"a": "A", "b": "B", "c": "C"},
              "laws": ["idm_global", "residual_idm_certified", "law_c", "law_e", "law_f"],
              "law_corridors": {"law_c": ["A"]}}  # fmt: skip
    root = tmp_path / "two"
    corridor = root / "runs" / "corridor"
    for law in ("idm_global", "residual_idm_certified", "law_e", "law_f"):  # corridor B: law_c does not run there
        for seed in SEEDS:
            write_json(corridor / "b_p0" / law / f"seed{seed}" / "macro.json", metrics(law, "b_p0", seed))
    write_json(corridor / "scenarios" / "b_p0" / "macro.json", metrics("idm_global", "b_p0", 0))
    out = build(root, design)
    laws = pd.read_csv(out["out"] / "laws.csv")
    per_law = laws[laws["scenario"] == "all"]
    assert per_law.groupby("corridor")["law"].apply(list).to_dict() == {
        "A": ["idm_global", "residual_idm_certified", "law_c", "law_e", "law_f"],
        "B": ["idm_global", "residual_idm_certified", "law_e", "law_f"]}  # fmt: skip
    assert per_law.set_index(["corridor", "law"]).loc[("B", "law_e"), "runs"] == 3
    assert laws[laws["law"] == "ground truth"]["scenario"].tolist() == ["a_p0", "a_p1", "b_p0"]
    missing = (out["out"] / "missing.txt").read_text(encoding="utf-8")
    assert "[runs] corridor C (c_p0): no ground truth with a macro.json, not in the tables" in missing
    tost = pd.read_csv(out["out"] / "tost.csv")
    assert tost.groupby("corridor")["pairs"].max().to_dict() == {"A": 6, "B": 3}
    corr = pd.read_csv(out["out"] / "instability_correlation.csv")
    assert set(corr["corridor"]) == {"A", "B"} and len(corr) == 32
    verdicts = pd.read_csv(out["out"] / "verdicts.csv")
    overall = verdicts[verdicts["unit"] == "overall"].set_index("hypothesis")
    assert set(overall["corridors"]) == {"A and B"} and not overall["complete"].astype(bool).any()  # C is missing
    h12_1 = verdicts[(verdicts["hypothesis"] == "H12.1") & (verdicts["unit"] != "overall")]
    assert h12_1["corridors"].tolist() == ["A", "A", "A", "B", "B"]  # law_c runs on A only
    assert overall.loc["H12.1", "verdict"] == "confirmed"  # law_e and law_f on both corridors
    assert overall.loc["H12.1", "basis"].endswith("A: law_e confirmed, law_f confirmed, law_c open; B: law_e confirmed, "
                                                  "law_f confirmed")  # fmt: skip
    assert "B: throughput" in verdicts.set_index("hypothesis").loc["H12.2", "basis"]


def test_tables_of_an_empty_tree(tmp_path):
    cfg = CorridorTablesConfig.from_mapping({"n_resamples": 20, "n_permutations": 50}, tmp_path / "corridor", tmp_path,
                                            tmp_path / "out")  # fmt: skip
    lines = make_corridor_tables(cfg)
    assert len(lines) == 18 and (tmp_path / "out" / "tost.md").exists() and (tmp_path / "out" / "verdicts.md").exists()
    for name in ("asymmetry", "asymmetry_contrasts", "correlation_pooled", "power", "temporal", "correlation_clustered",
                 "contacts_absolute", "h12_2_error", "ablation_pairs"):  # M8 and its review: next to out
        assert (tmp_path / "m8" / f"{name}.md").exists() and (tmp_path / "m8" / f"{name}.csv").exists(), name
    laws = pd.read_csv(tmp_path / "out" / "laws.csv")
    assert len(laws) == 14 + 4 + 3 and laws["macro_error"].isna().all()  # I-80: the laws of D97, D114, D111; 3 truths
    assert set(laws["corridor"]) == {"I-80"}  # US-101 has no ground truth
    verdicts = pd.read_csv(tmp_path / "out" / "verdicts.csv")
    assert verdicts["verdict"].isna().all() and set(verdicts[verdicts["unit"] == "overall"]["corridors"]) == {"I-80 only"}
    for name in ("h12_1", "h12_2", "e4_rmax", "sensitivity"):
        assert (tmp_path / "out" / f"{name}.csv").exists() and (tmp_path / "out" / f"{name}.md").exists()
    with pytest.raises(ValueError, match="unknown keys"):
        CorridorTablesConfig.from_mapping({"law": ["a"]}, tmp_path, tmp_path, tmp_path)


def test_helpers():
    assert correlation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0]), "spearman") == pytest.approx(1.0)
    assert math.isnan(correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), "pearson"))
    constant = correlation_ci([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], "spearman", n_resamples=50)
    assert math.isnan(constant["estimate"]) and constant["n_valid"] == 0 and constant["n"] == 4
    assert correlation_ci([1.0, np.nan], [2.0, 3.0], "pearson")["n"] == 1
    flat = flatten_macro({"throughput_vph": None, "waves": {"n_waves": 2}, "macro_error": {"value": 0.3}})
    assert math.isnan(flat["throughput_vph"]) and flat["n_waves"] == 2.0 and flat["macro_error"] == 0.3
    assert math.isnan(flat["error_fd"]) and math.isnan(flat["travel_time_median"])
    json.dumps(flat)  # plain floats
