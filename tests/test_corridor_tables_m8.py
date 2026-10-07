"""Corridor tables of M8 (cf_stability/eval/corridor_tables.py, cf_stability/eval/stats.py; docs/m8_contract.md,
sections 5, 7, 9 and 10; D118, D120-D122) on a hand-made tree: two corridors, the laws of the design with a
heterogeneous certified hybrid (closed-form instability of its cores), a residual amplitude, a penalised and a free
law, a temporal hold-out law on one scenario; asymmetry.json files with known values; the statistics of the pooled
correlation and of the power against direct computations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from cf_stability.eval.corridor_tables import (
    CorridorTablesConfig,
    make_corridor_tables,
    make_m8_corridor_tables,
    relative_error,
)
from cf_stability.eval.stats import (
    minimal_detectable,
    paired_t_power,
    pooled_sd,
    seeds_needed,
    stratified_permutation_p,
    stratified_ranks,
    stratified_spearman,
    stratified_spearman_ci,
)
from cf_stability.utils import write_json

SCENARIOS = ("a_p0", "a_p1", "b_p0")
SEEDS = (0, 1, 2, 3)
# law: (unstable among equilibria of its members, macro error on A, macro error on B, asymmetry index)
LAWS = {
    "idm_global": (0.0, 0.10, 0.12, 1.2),
    "residual_idm_certified": (0.05, 0.11, 0.10, 1.3),
    "law_free": (0.6, 0.40, 0.50, 0.4),
    "law_pen": (0.4, 0.30, 0.45, 0.5),
    "law_r": (0.2, 0.20, 0.20, 1.25),  # a residual amplitude (rmax_laws): in correlation_pooled only
}
TRUTH_INDEX = {"a_p0": 1.0, "a_p1": 0.9, "b_p0": 1.1}
CORES = {"v0": np.array([33.0, 30.0]), "T": np.array([1.5, 0.6]), "s0": np.array([2.0, 1.0]), "a": np.array([1.2, 0.25]),
         "b": np.array([1.5, 0.5])}  # fmt: skip
DESIGN = {
    "scenarios": list(SCENARIOS), "corridors": {"a": "A", "b": "B"}, "seeds": list(SEEDS), "n_resamples": 200,
    "n_permutations": 500, "laws": ["idm_global", "residual_idm_certified", "law_free", "law_pen", "het"],
    "rmax_laws": ["law_r"], "temporal_laws": ["law_t", "law_absent"], "optional_laws": ["het", "law_t", "law_absent"],
    "law_corridors": {"law_t": ["A"], "law_absent": ["A"]}, "law_scenarios": {"law_t": ["a_p1"], "law_absent": ["a_p1"]},
    "temporal_pairs": {"law_t": "law_free", "law_absent": "idm_global"}, "temporal_scenarios": ["a_p1"],
    "asymmetry_contrasts": [["law_pen", "law_free"], ["residual_idm_certified", "law_missing"]],
    "networks": ["law_free"], "correlation_min_laws": 3, "rmax": [], "sensitivity_variants": [],
    "sensitivity_scenario": "a_p0", "sensitivity_laws": ["idm_global"], "sensitivity_order": ["idm_global"],
    "micro_others": [], "run_folds": [0], "run_seeds": [0],
}  # fmt: skip


def macro(error: float, throughput: float, travel: float) -> dict:
    return {"window": [180.0, 840.0], "throughput_vph": throughput, "mean_speed": 6.0, "collisions_per_1000_vkm": 0.0,
            "travel_time": {"mean": travel, "values": [travel]}, "waves": {"wave_speed_xcorr": -5.0},
            "macro_error": {"components": {"throughput": 0.01, "travel_time": error}, "value": error, "n_components": 2},
            "macro_error_dynamic": {"value": 2 * error, "n_components": 1}, "config_hash": "m"}  # fmt: skip


def asymmetry(index: float, peak: float = 0.0078125, centroid: float = 0.02) -> dict:
    return {"window": [180.0, 840.0], "config_hash": "q", "config": {"threshold": 0.1, "sample_s": 2.0, "window_s": 128.0,
                                                                     "f_min": 0.002, "f_max": 0.05, "min_coverage": 0.5},
            "acceleration": {"asymmetry_index": index, "accelerating_share": 0.3, "decelerating_share": 0.4,
                             "mean_acceleration": 0.4, "mean_deceleration": 0.4 / index, "n_samples": 1000},
            "spectrum": {"peak_frequency": peak, "centroid": centroid, "band_rms": 1.0, "n_detectors": 5}}  # fmt: skip


def law_value(law: str, scenario: str, seed: int) -> tuple[float, float, float]:
    """(macro error, throughput, travel time) of a run: a fixed spread over the seeds."""
    unstable, on_a, on_b, _ = LAWS[law]
    base = on_a if scenario.startswith("a") else on_b
    spread = {"idm_global": 0.01, "residual_idm_certified": 0.02}.get(law, 0.005)
    return base + spread * (seed - 1.5), 7000.0 + 10.0 * seed + (50.0 if law == "residual_idm_certified" else 0.0), \
        70.0 + 0.5 * seed  # fmt: skip


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("m8") / "runs"
    corridor = root / "corridor"
    for scenario in SCENARIOS:
        write_json(corridor / "scenarios" / scenario / "macro.json", {**macro(0.0, 7000.0, 70.0), "macro_error": None})
        write_json(corridor / "scenarios" / scenario / "asymmetry.json", asymmetry(TRUTH_INDEX[scenario]))
        for law, (_, _, _, index) in LAWS.items():
            for seed in SEEDS:
                run = corridor / scenario / law / f"seed{seed}"
                write_json(run / "macro.json", macro(*law_value(law, scenario, seed)))
                write_json(run / "run.json", {"g_mean": 1.0, "dN_mean": 0.0})
                if not (law == "law_free" and scenario == "b_p0" and seed == 3):  # one file missing
                    write_json(run / "asymmetry.json", asymmetry(index + 0.01 * seed, centroid=0.02 + 0.001 * seed))
    for seed in SEEDS:  # the temporal hold-out law on a_p1 only: law_free + 0.05
        error, throughput, travel = law_value("law_free", "a_p1", seed)
        write_json(corridor / "a_p1" / "law_t" / f"seed{seed}" / "macro.json", macro(error + 0.05, throughput, travel))
        write_json(corridor / "a_p1" / "law_t" / f"seed{seed}" / "asymmetry.json", asymmetry(0.45))
    laws = corridor / "laws"
    for law, (unstable, *_) in LAWS.items():
        member = root / "members" / law / "driver_fold0_seed0"
        write_json(member / "stability.json", {"audit": {"summary": {"share_unstable_numerical": {"support": unstable},
                                                                     "band_numerical": {"support": {"stable": 0.5}}}}})  # fmt: skip
        write_json(laws / f"{law}.json", {"law": law, "kind": "models",
                                          "members": [{"run": f"members/{law}/driver_fold0_seed0"}]})  # fmt: skip
    write_json(laws / "law_t.json", {"law": "law_t", "kind": "models", "members": [{"run": "members/law_free/driver_fold0_seed0"}]})
    np.savez(laws / "het.npz", **CORES, index=np.array([0, 4]))
    write_json(laws / "het.json", {"law": "het", "kind": "residual_heterogeneous", "members": [{"run": "x"}],
                                   "table": "het.npz", "support_v": [0.0, 18.0], "window": 1})  # fmt: skip
    cfg = CorridorTablesConfig.from_mapping(DESIGN, corridor, root, root / "_tables" / "m5", root / "_tables" / "m8")
    lines = make_corridor_tables(cfg)
    return {"root": root, "cfg": cfg, "lines": lines, "m5": root / "_tables" / "m5", "m8": root / "_tables" / "m8"}


# ------------------------------------------------------------------------------------------- statistics


def test_pooled_sd_and_power():
    values = np.array([1.0, 2.0, 3.0, 10.0, 12.0])
    groups = np.array(["a", "a", "a", "b", "b"])
    expected = math.sqrt(((1 - 2) ** 2 + 0 + (3 - 2) ** 2 + (10 - 11) ** 2 + (12 - 11) ** 2) / (2 + 1))
    assert pooled_sd(values, groups) == pytest.approx(expected)
    assert math.isnan(pooled_sd([1.0, 2.0], ["a", "b"]))  # one value per group
    # power of the paired t-test against the noncentral t directly and against a simulation
    effect, n = 0.8, 15
    critical = sps.t.ppf(0.975, n - 1)
    direct = sps.nct.sf(critical, n - 1, effect * math.sqrt(n)) + sps.nct.cdf(-critical, n - 1, effect * math.sqrt(n))
    assert paired_t_power(effect, n, n - 1) == pytest.approx(direct)
    rng = np.random.default_rng(0)
    d = rng.normal(effect, 1.0, size=(20000, n))
    t = d.mean(axis=1) / (d.std(axis=1, ddof=1) / math.sqrt(n))
    assert paired_t_power(effect, n, n - 1) == pytest.approx(np.mean(np.abs(t) > critical), abs=0.01)
    # the smallest n reaching the power: the one below does not
    n_needed = seeds_needed(0.5)
    assert paired_t_power(0.5, n_needed, n_needed - 1) >= 0.8 > paired_t_power(0.5, n_needed - 1, n_needed - 2)
    assert n_needed == 34  # the classic number for d = 0.5, two-sided 5 %, 80 %
    per_stratum = seeds_needed(0.5, strata=3)
    assert paired_t_power(0.5, 3 * per_stratum, 3 * (per_stratum - 1)) >= 0.8 and per_stratum <= math.ceil(34 / 3) + 1
    assert seeds_needed(math.inf) == 2 and seeds_needed(0.0) is None and seeds_needed(math.nan) is None
    detectable = minimal_detectable(2.0, 10, 3)
    assert paired_t_power(detectable / 2.0, 30, 27) == pytest.approx(0.8, abs=1e-6)
    assert minimal_detectable(0.0, 10) == 0.0 and math.isnan(minimal_detectable(1.0, 1))


def test_stratified_spearman():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    y = np.array([5.0, 6.0, 8.0, 7.0, 0.1, 0.2, 0.3])
    strata = np.array(["a"] * 4 + ["b"] * 3)
    assert stratified_ranks(x, strata).tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875, 1 / 6, 0.5, 5 / 6])
    one = np.zeros(4, dtype=int)
    assert stratified_spearman(x[:4], y[:4], one) == pytest.approx(sps.spearmanr(x[:4], y[:4]).statistic)
    # within the strata the order agrees but across them it does not: the pooled Spearman ignores the levels
    assert sps.spearmanr(x, y).statistic < 0 < stratified_spearman(x, y, strata)
    assert stratified_spearman(np.r_[x[4:], x[4:]], np.r_[y[4:], y[4:]], ["a"] * 3 + ["b"] * 3) == pytest.approx(1.0)
    ci = stratified_spearman_ci(x, y, strata, n_resamples=200)
    assert ci["n"] == 7 and ci["n_strata"] == 2 and ci["low"] <= ci["estimate"] <= ci["high"] and ci["n_valid"] > 100
    p = stratified_permutation_p(np.r_[x, x], np.r_[x, x], ["a"] * 7 + ["b"] * 7, n_permutations=2000)
    assert p == pytest.approx(1 / 2001, abs=5e-4)  # 14 pairs in perfect order within both strata
    assert math.isnan(stratified_permutation_p([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0, 0, 0]))
    assert relative_error(1.2, 1.0) == pytest.approx(0.2) and math.isnan(relative_error(1.0, 0.0))


# ------------------------------------------------------------------------------------------------- tables


def test_optional_and_temporal_laws_in_the_m5_tables(tree):
    laws = pd.read_csv(tree["m5"] / "laws.csv")
    rows = laws[laws["scenario"] == "all"].set_index(["corridor", "law"])
    assert ("A", "law_t") in rows.index and ("B", "law_t") not in rows.index and ("A", "law_absent") not in rows.index
    assert rows.loc[("A", "law_t"), "runs"] == 4 and rows.loc[("A", "law_t"), "runs_expected"] == 4
    assert rows.loc[("A", "law_t"), "scenario_set"] == "a_p1" and rows.loc[("A", "law_free"), "scenario_set"] == "all"
    assert ("A", "het") in rows.index and rows.loc[("A", "het"), "runs"] == 0  # a law file, no run yet
    missing = (tree["m5"] / "missing.txt").read_text(encoding="utf-8")
    assert "[runs] law_absent: no law file and no runs yet (a law of M8): not in the tables" in missing
    instability = pd.read_csv(tree["m5"] / "instability.csv").set_index(["corridor", "law"])
    het = instability.loc[("A", "het")]
    assert het["kind"] == "residual_heterogeneous" and het["members"] == 2 and 0.0 < het["unstable_eq"] < 1.0
    assert bool(instability.loc[("A", "law_t"), "in_correlation"]) is False
    assert bool(instability.loc[("A", "law_free"), "in_correlation"]) is True
    corr = pd.read_csv(tree["m5"] / "instability_correlation.csv")
    pick = corr[(corr["corridor"] == "A") & (corr["error"] == "macro error") & (corr["method"] == "spearman")
                & (corr["laws"] == "all laws") & (corr["measure"] == "unstable among equilibria")]  # fmt: skip
    assert pick["n"].iloc[0] == 4  # the four laws of the design with runs: not het (no run), not law_t
    components = pd.read_csv(tree["m5"] / "components.csv").set_index(["corridor", "law"])
    assert components.loc[("A", "law_t"), "scenario_set"] == "a_p1"


def test_asymmetry_table(tree):
    table = pd.read_csv(tree["m8"] / "asymmetry.csv")
    rows = table[table["scenario"] == "all"].set_index(["corridor", "law"])
    free = rows.loc[("A", "law_free")]
    assert free["runs"] == 8 and free["asymmetry_index"] == pytest.approx(0.4 + 0.015)
    truth = np.array([TRUTH_INDEX[s] for s in ("a_p0", "a_p1") for _ in SEEDS])
    values = np.array([0.4 + 0.01 * seed for _ in ("a_p0", "a_p1") for seed in SEEDS])
    assert free["error_asymmetry_index"] == pytest.approx(np.mean((values - truth) / truth))
    assert free["error_peak_frequency"] == pytest.approx(0.0) and free["error_asymmetry_index_low"] < 0
    assert rows.loc[("B", "law_free"), "runs"] == 3  # one asymmetry.json missing
    assert rows.loc[("A", "law_t"), "runs"] == 4 and rows.loc[("A", "law_t"), "scenario_set"] == "a_p1"
    truths = table[table["law"] == "ground truth"].set_index(["corridor", "scenario"])
    assert truths.loc[("A", "a_p1"), "asymmetry_index"] == pytest.approx(0.9)
    assert truths.loc[("A", "mean"), "asymmetry_index"] == pytest.approx(0.95)
    md = (tree["m8"] / "asymmetry.md").read_text(encoding="utf-8")
    assert "threshold 0.1 m/s^2, detector series of 2 s, Welch segments of 128 s, band 0.002-0.05 Hz" in md
    missing = (tree["m8"] / "missing_corridor.txt").read_text(encoding="utf-8")
    assert "[asymmetry] b_p0/law_free/seed3/asymmetry.json missing" in missing
    contrasts = pd.read_csv(tree["m8"] / "asymmetry_contrasts.csv").set_index(["corridor", "law", "versus"])
    pen = contrasts.loc[("A", "law_pen", "law_free")]
    assert pen["pairs"] == 8 and pen["asymmetry_index_difference"] == pytest.approx(0.1)
    assert bool(pen["index_changed"]) and pen["index_closer"] == "closer"  # 0.5 is nearer to 0.9-1.0 than 0.4
    assert pen["closer_asymmetry_index_high"] < 0 and contrasts.loc[("B", "law_pen", "law_free"), "pairs"] == 3
    assert "[asymmetry_contrasts] A: residual_idm_certified or law_missing not in the tables, no contrast" in missing


def test_correlation_pooled(tree):
    table = pd.read_csv(tree["m8"] / "correlation_pooled.csv")
    main = table[(table["error"] == "macro error") & (table["laws"] == "all laws with runs (D122)")].set_index("scope")
    assert list(main.index) == ["A", "B", "pooled (corridor as stratum)"]
    assert main.loc["A", "n"] == 5 and main.loc["pooled (corridor as stratum)", "n"] == 10  # law_r included, het not
    errors_a = [LAWS[law][1] for law in ("idm_global", "residual_idm_certified", "law_free", "law_pen", "law_r")]
    unstable = [LAWS[law][0] for law in ("idm_global", "residual_idm_certified", "law_free", "law_pen", "law_r")]
    assert main.loc["A", "estimate"] == pytest.approx(sps.spearmanr(unstable, errors_a).statistic)
    errors_b = [LAWS[law][2] for law in ("idm_global", "residual_idm_certified", "law_free", "law_pen", "law_r")]
    pooled = stratified_spearman(unstable * 2, errors_a + errors_b, ["A"] * 5 + ["B"] * 5)
    assert main.loc["pooled (corridor as stratum)", "estimate"] == pytest.approx(pooled)
    assert "law_r" in main.loc["A", "law_list"] and "het" not in main.loc["A", "law_list"]
    design = table[(table["laws"] == "laws of H12.3 (D108)") & (table["error"] == "macro error")].set_index("scope")
    assert design.loc["A", "n"] == 4  # without the residual amplitude
    missing = (tree["m8"] / "missing_corridor.txt").read_text(encoding="utf-8")
    assert "[correlation_pooled] A/het: no run with a macro.json, not in the correlation" in missing
    values = pd.read_csv(tree["m8"] / "correlation_pooled_laws.csv").set_index(["corridor", "law"])
    assert len(values) == 10 and values.loc[("B", "law_r"), "unstable_eq"] == pytest.approx(0.2)
    assert values.loc[("A", "law_free"), "macro_error"] == pytest.approx(0.40)
    assert "# The laws of the pooled correlation (D122)" in (tree["m8"] / "correlation_pooled.md").read_text(encoding="utf-8")


def test_power(tree):
    table = pd.read_csv(tree["m8"] / "power.csv").set_index(["corridor", "metric"])
    row = table.loc[("A", "macro_error")]
    assert row["pairs"] == 8 and row["scenarios"] == 2 and row["seeds"] == 4
    seeds = np.array(SEEDS, dtype=float)
    assert row["sd_a"] == pytest.approx(np.std(0.01 * seeds, ddof=1))  # the same spread in every scenario
    assert row["sd_b"] == pytest.approx(np.std(0.02 * seeds, ddof=1))
    assert row["sd_difference"] == pytest.approx(np.std(0.01 * seeds, ddof=1))
    mean = 0.5 * (0.10 + 0.11)
    assert row["delta"] == pytest.approx(0.1 * mean) and row["effect"] == pytest.approx(0.1 * mean / row["sd_difference"])
    assert row["n_per_scenario"] == seeds_needed(row["effect"], 2) and row["n_one_scenario"] == seeds_needed(row["effect"])
    assert row["power_design"] == pytest.approx(paired_t_power(row["effect"], 8, 6))
    assert row["observed"] == pytest.approx(0.11 / 0.10 - 1.0)
    assert row["sentence"].startswith("On A, the seed-to-seed standard deviation of the macro error within a scenario")
    md = (tree["m8"] / "power.md").read_text(encoding="utf-8")
    assert "Sentences for the paper:" in md and md.count("- On ") == 6  # three metrics x two corridors
    assert table.loc[("A", "throughput_vph"), "sd_difference"] == pytest.approx(0.0)
    assert table.loc[("A", "throughput_vph"), "n_per_scenario"] == 2  # no spread of the difference: any 2 seeds


def test_temporal(tree):
    table = pd.read_csv(tree["m8"] / "temporal.csv").set_index("law")
    t = table.loc["law_t"]
    assert t["versus"] == "law_free" and t["runs"] == 4 and t["runs_versus"] == 4 and t["scenarios"] == "a_p1"
    assert t["macro_error_difference"] == pytest.approx(0.05) and t["outcome"] == "worse"
    assert t["macro_error_law"] == pytest.approx(0.45) and t["macro_error_versus"] == pytest.approx(0.40)
    assert t["macro_error_relative"] == pytest.approx(0.45 / 0.40 - 1.0) and t["macro_error_pairs"] == 4
    assert t["macro_error_dynamic_difference"] == pytest.approx(0.10)
    absent = table.loc["law_absent"]
    assert absent["runs"] == 0 and pd.isna(absent["macro_error_law"])
    missing = (tree["m8"] / "missing_corridor.txt").read_text(encoding="utf-8")
    assert "[temporal] law_absent: no runs yet (D120), no comparison" in missing
    assert any(line.startswith("TABLE temporal: law_t worse, law_absent n/a") for line in tree["lines"])


def test_m8_tables_alone_and_the_notes_files(tree, tmp_path):
    """make_m8_corridor_tables writes the five tables of M8 and the three of its review with missing_corridor.txt; the
    notes of M8 are not in the missing.txt of M5."""
    m5 = (tree["m5"] / "missing.txt").read_text(encoding="utf-8")
    assert "[asymmetry]" not in m5 and "[temporal]" not in m5
    cfg = CorridorTablesConfig.from_mapping(DESIGN, tree["root"] / "corridor", tree["root"], tmp_path / "m5",
                                            tmp_path / "m8_alone")  # fmt: skip
    lines = make_m8_corridor_tables(cfg)
    assert [line.split(":")[0] for line in lines] == ["TABLE asymmetry", "TABLE asymmetry_contrasts",
                                                      "TABLE correlation_pooled", "TABLE power", "TABLE temporal",
                                                      "TABLE correlation_clustered", "TABLE contacts_absolute",
                                                      "TABLE h12_2_error"]  # fmt: skip
    for name in ("asymmetry", "asymmetry_contrasts", "correlation_pooled", "power", "temporal", "correlation_clustered",
                 "contacts_absolute", "h12_2_error"):
        alone = pd.read_csv(tmp_path / "m8_alone" / f"{name}.csv")
        pd.testing.assert_frame_equal(alone, pd.read_csv(tree["m8"] / f"{name}.csv"), check_dtype=False)
    assert (tmp_path / "m8_alone" / "missing_corridor.txt").exists() and not (tmp_path / "m5").exists()
