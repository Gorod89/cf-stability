"""Corridor tables of the review of M8 (cf_stability/eval/corridor_tables.py, cf_stability/eval/stats.py) on a hand-made
tree: two corridors, five laws of the design in three families, a law of the ablation with a closed-form core and one
without inputs; the clustered inference of H12.3 against direct computations (expanded resamples, exact permutation
distribution), the absolute contact exposure and the error-based comparison of H12.2 against the values written into
the macro.json and run.json files."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from cf_stability.eval.corridor_tables import CorridorTablesConfig, make_corridor_tables
from cf_stability.eval.stats import (
    cluster_bootstrap_spearman,
    clustered_spearman,
    first_appearance_codes,
    linked_permutation_p,
    multiplicities,
    stratified_bootstrap_spearman,
    stratified_permutation_p,
    stratified_spearman,
    weighted_pearson,
    weighted_stratified_ranks,
)
from cf_stability.utils import write_json

SCENARIOS = ("a_p0", "a_p1", "b_p0")
SEEDS = (0, 1, 2, 3)
COMPONENTS = ("throughput", "mean_speed", "queue_discharge_flow", "fd", "wave_speed", "n_waves", "wave_amplitude",
              "travel_time")  # fmt: skip
DYNAMIC = ("fd", "wave_speed", "n_waves", "wave_amplitude")
# law: (unstable among equilibria of its members, error scale on A, on B)
LAWS = {
    "idm_global": (0.0, 1.0, 1.0),
    "residual_idm_certified": (0.0, 0.9, 0.8),
    "law_free": (0.6, 3.0, 4.0),
    "law_pen": (0.4, 2.5, 2.0),
    "law_r": (0.1, 1.5, 1.6),
}
FAMILIES = {"F1": ["idm_global", "residual_idm_certified", "abl_core"], "F2": ["law_free", "law_pen"], "F3": ["law_r"]}
CORE = {"v0": 35.2, "T": 0.98, "s0": 3.19, "a": 1.57, "b": 0.5}  # a margin core: string stable at every grid speed
DESIGN = {
    "scenarios": list(SCENARIOS), "corridors": {"a": "A", "b": "B"}, "seeds": list(SEEDS), "n_resamples": 200,
    "n_permutations": 2000, "laws": ["idm_global", "residual_idm_certified", "law_free", "law_pen"],
    "rmax_laws": ["law_r"], "temporal_laws": [], "ablation_laws": ["abl_core", "abl_absent"],
    "optional_laws": ["abl_core", "abl_absent"], "families": FAMILIES, "physics_families": ["F1"],
    "cluster_resamples": 400, "cluster_seed": 7, "networks": ["law_free"], "correlation_min_laws": 3,
    "rmax": [{"r_max": 0.3, "core": "certified", "highd": "e4_stable", "ngsim": "e4_stable_ft",
              "law": "residual_idm_certified"}],
    "rmax_ablation": [{"r_max": 0.0, "core": "core alone", "highd": None, "ngsim": None, "law": "abl_core"},
                      {"r_max": 0.3, "core": "margin, no certificate", "highd": "e4_x", "ngsim": "e4_x_ft",
                       "law": "abl_absent"}],
    "sensitivity_variants": [], "sensitivity_scenario": "a_p0", "sensitivity_laws": ["idm_global"],
    "sensitivity_order": ["idm_global"], "micro_others": [], "run_folds": [0], "run_seeds": [0],
    "asymmetry_contrasts": [],
}  # fmt: skip


def components(law: str, scenario: str, seed: int) -> dict[str, float]:
    """Signed errors of a run: scaled by the law, with chosen differences between the certified hybrid and the IDM."""
    _, on_a, on_b = LAWS.get(law, LAWS["idm_global"])
    scale = on_a if scenario.startswith("a") else on_b
    out = {"throughput": -0.02 * scale - 0.001 * seed, "mean_speed": 0.03, "queue_discharge_flow": -0.02 * scale,
           "fd": 0.1 * scale, "wave_speed": 0.2 * scale + 0.01 * seed, "n_waves": -0.5 * scale,
           "wave_amplitude": 0.05 * scale, "travel_time": 0.07 * scale + 0.002 * seed}  # fmt: skip
    if law == "residual_idm_certified":  # throughput closer by 0.01, wave speed further by 0.1, mean speed equal
        reference = components("idm_global", scenario, seed)
        out.update(throughput=-(abs(reference["throughput"]) - 0.01), mean_speed=reference["mean_speed"],
                   wave_speed=-(abs(reference["wave_speed"]) + 0.1))  # fmt: skip
    return out


def exposure(law: str, scenario: str, seed: int) -> tuple[dict, dict]:
    """(macro.json values, run.json values) of the exposure and the demand of a run."""
    on_a = scenario.startswith("a")
    distance = (600.0 if on_a else 1000.0) + 10.0 * seed
    episodes = 50.0 + seed if law == "law_free" else 0.0
    inserted = 0.98 if law == "law_free" else 1.0
    macro = {"vehicle_km": distance, "collision_episodes": episodes, "n_vehicles_window": 1500, "n_planned": 1350,
             "n_vehicles": 1800, "inserted_share": inserted, "mean_depart_delay_s": 5.0 if law == "law_free" else 0.5,
             "vehicles_in_contact_share": 0.1 if law == "law_free" else 0.0,
             "collisions_per_1000_vkm": 1000.0 * episodes / distance}  # fmt: skip
    run = {"n_planned": 1800, "n_inserted": 1800 - (20 if law == "law_free" else 0), "g_mean": 1.0, "dN_mean": 0.0,
           "mean_depart_delay_s": 4.0 if law == "law_free" else 0.4}  # fmt: skip
    return macro, run


def macro(law: str, scenario: str, seed: int) -> dict:
    errors = components(law, scenario, seed)
    values, _ = exposure(law, scenario, seed)
    return {"window": [180.0, 840.0], "throughput_vph": 7000.0, "travel_time": {"mean": 70.0},
            "waves": {"wave_speed_xcorr": -5.0},
            "macro_error": {"components": errors, "value": float(np.mean(np.abs(list(errors.values())))),
                            "n_components": 8},
            "macro_error_dynamic": {"value": float(np.mean([abs(errors[n]) for n in DYNAMIC])), "n_components": 4},
            "config_hash": "m", **values}  # fmt: skip


def write_training_run(runs, experiment: str, data: str, rmse: float) -> None:
    run = runs / experiment / data / "residual_idm" / "driver_fold0_seed0"
    write_json(run / "metrics.json", {"config_hash": experiment, "test": {"rmse_s_mean": rmse}})
    write_json(run / "stability.json", {"audit": {"summary": {"share_unstable_numerical": {"support": 0.0},
                                                              "band_numerical": {"support": {"stable": 1.0}}}}})  # fmt: skip
    pd.DataFrame({"event_id": [f"d{k}|e" for k in range(3)], "follower_id": [f"d{k}" for k in range(3)], "rmse_s": rmse,
                  "rmse_v": 0.5, "collided": False}).to_parquet(run / "test_events.parquet", index=False)  # fmt: skip
    write_json(run / "certificate.json", {"model": "residual_idm", "applicable": True, "a_priori": {"holds": True}})


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("review") / "runs"
    corridor = root / "corridor"
    for scenario in SCENARIOS:
        on_a = scenario.startswith("a")
        write_json(corridor / "scenarios" / scenario / "macro.json", {
            "window": [180.0, 840.0], "throughput_vph": 7000.0, "macro_error": None, "collision_episodes": 0.0,
            "vehicle_km": 620.0 if on_a else 1030.0, "n_vehicles_window": 1500, "n_planned": 1350, "n_vehicles": 1836,
        })  # fmt: skip
        for law in (*LAWS, "abl_core"):
            for seed in SEEDS:
                run = corridor / scenario / law / f"seed{seed}"
                write_json(run / "macro.json", macro(law, scenario, seed))
                write_json(run / "run.json", exposure(law, scenario, seed)[1])
    laws = corridor / "laws"
    for law, (unstable, *_) in LAWS.items():
        member = root / "members" / law / "driver_fold0_seed0"
        write_json(member / "stability.json", {"audit": {"summary": {"share_unstable_numerical": {"support": unstable},
                                                                     "band_numerical": {"support": {"stable": 0.5}}}}})  # fmt: skip
        write_json(laws / f"{law}.json", {"law": law, "kind": "models",
                                          "members": [{"run": f"members/{law}/driver_fold0_seed0"}]})  # fmt: skip
    write_json(laws / "abl_core.json", {"law": "abl_core", "kind": "idm", "device": "cpu", "members": [],
                                        "params": [CORE, CORE], "support_v": [0.0, 18.0], "window": 1})  # fmt: skip
    write_training_run(root, "e4_stable", "follownet_highd", 2.5)
    write_training_run(root, "e4_stable_ft", "ngsim_i80", 3.6)
    cfg = CorridorTablesConfig.from_mapping(DESIGN, corridor, root, root / "_tables" / "m5", root / "_tables" / "m8")
    lines = make_corridor_tables(cfg)
    return {"root": root, "cfg": cfg, "lines": lines, "m5": root / "_tables" / "m5", "m8": root / "_tables" / "m8"}


# ------------------------------------------------------------------------------------------- statistics


def test_weighted_ranks_are_those_of_the_expanded_samples():
    """The multiplicity form of the stratified Spearman equals the computation on the expanded sample, with ties,
    rows of weight 0 and a stratum absent from a sample."""
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=12), 1)
    x[:3] = 0.0  # ties
    y = x + rng.normal(scale=0.5, size=12)
    s = np.array(["a"] * 7 + ["b"] * 5)
    weights = rng.integers(0, 3, size=(300, 12)).astype(float)
    weights[0, 7:] = 0.0  # stratum b absent
    weights[1] = 0.0
    weights[1, :2] = 1.0  # two rows only: no correlation
    got = weighted_pearson(weighted_stratified_ranks(x, s, weights), weighted_stratified_ranks(y, s, weights), weights)
    for r in range(300):
        rows = np.repeat(np.arange(12), weights[r].astype(int))
        expected = stratified_spearman(x[rows], y[rows], s[rows])
        assert (math.isnan(got[r]) and math.isnan(expected)) or got[r] == pytest.approx(expected, abs=1e-12), r
    draws = np.array([[0, 2, 2], [1, 1, 1]])
    assert multiplicities(draws, 3).tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
    codes, names = first_appearance_codes(["b", "a", "b", "c"])
    assert codes.tolist() == [0, 1, 0, 2] and names == ["b", "a", "c"]


def test_cluster_bootstrap_against_a_loop_over_the_draws():
    """The interval is that of the loop over the same draws (clusters numbered in the order of first appearance, the
    rows of a cluster taken as often as it is drawn); the family bootstrap continues the stream of the law bootstrap."""
    rng = np.random.default_rng(1)
    laws = np.array([f"l{k}" for k in range(9)] + [f"l{k}" for k in range(8)])  # l8 on the first stratum only
    s = np.array([0] * 9 + [1] * 8)
    unstable = rng.uniform(size=9)
    x = unstable[[int(law[1:]) for law in laws]]  # one value per law on both strata
    y = x + rng.normal(scale=0.3, size=len(x))
    families = np.array(["f" + str(int(law[1:]) // 3) for law in laws])

    def loop(clusters: np.ndarray, generator: np.random.Generator, n: int = 300) -> tuple[float, float]:
        codes, names = first_appearance_codes(clusters)
        members = [np.flatnonzero(codes == k) for k in range(len(names))]
        draws = generator.integers(0, len(names), size=(n, len(names)))
        boot = np.array([stratified_spearman(x[r], y[r], s[r]) for r in (np.concatenate([members[k] for k in d])
                                                                          for d in draws)])  # fmt: skip
        return tuple(np.quantile(boot[np.isfinite(boot)], [0.025, 0.975]))

    generator = np.random.default_rng(42)
    expected_laws, expected_families = loop(laws, generator), loop(families, generator)
    generator = np.random.default_rng(42)
    by_law = cluster_bootstrap_spearman(x, y, s, laws, generator, 300)
    by_family = cluster_bootstrap_spearman(x, y, s, families, generator, 300)
    assert (by_law["low"], by_law["high"]) == pytest.approx(expected_laws, abs=1e-12)
    assert (by_family["low"], by_family["high"]) == pytest.approx(expected_families, abs=1e-12)
    assert by_law["n_clusters"] == 9 and by_family["n_clusters"] == 3 and by_law["n_valid"] == 300
    assert by_law["estimate"] == pytest.approx(stratified_spearman(x, y, s))
    combined = clustered_spearman(x, y, s, laws, families, n_resamples=300, n_permutations=500, seed=42)
    assert (combined["law_low"], combined["family_high"]) == pytest.approx((expected_laws[0], expected_families[1]))
    assert combined["n"] == 17 and combined["n_laws"] == 9 and combined["n_families"] == 3 and combined["n_strata"] == 2
    rows = stratified_bootstrap_spearman(x, y, s, np.random.default_rng(3), 300)
    assert rows["n_valid"] == 300 and rows["low"] < rows["estimate"] < rows["high"]


def test_linked_permutation_against_the_exact_distribution():
    """Five laws, one of them on one stratum only: the Monte Carlo p-value of the linked permutation against the exact
    one over the 120 assignments of the laws' values; within one stratum it is the permutation of D108/D122."""
    laws = np.array(["l0", "l1", "l2", "l3", "l4", "l0", "l1", "l2", "l3"])
    s = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    value = np.array([0.0, 0.2, 0.5, 0.7, 0.9])
    x = value[[int(law[1:]) for law in laws]]
    y = np.array([0.1, 0.3, 0.2, 0.6, 0.5, 0.2, 0.1, 0.5, 0.4])
    observed = abs(stratified_spearman(x, y, s))
    exact = np.mean([abs(stratified_spearman(np.array(p)[[int(law[1:]) for law in laws]], y, s)) >= observed - 1e-12
                     for p in itertools.permutations(value)])  # fmt: skip
    assert linked_permutation_p(x, y, s, laws, 20000, seed=2) == pytest.approx(exact, abs=0.01)
    one = s == 0
    assert linked_permutation_p(x[one], y[one], s[one], laws[one], 3000, seed=5) == \
        stratified_permutation_p(x[one], y[one], s[one], 3000, seed=5)  # fmt: skip
    assert math.isnan(linked_permutation_p(np.zeros(4), np.arange(4.0), np.zeros(4), np.arange(4), 100))


# ------------------------------------------------------------------------------------------------- tables


def test_correlation_clustered_table(tree):
    table = pd.read_csv(tree["m8"] / "correlation_clustered.csv", keep_default_na=False, na_values=[""])
    assert set(table["scope"]) == {"A", "B", "pooled (corridor as stratum)"}
    assert set(table["error"]) == {"macro error", "macro error (dynamic)"}
    pooled = table[(table["scope"] == "pooled (corridor as stratum)") & (table["error"] == "macro error")]
    assert pooled["subset"].tolist() == ["all laws with runs", "without F1", "without F2", "without F3",
                                         "learned laws only (without F1)"]  # fmt: skip
    every = pooled.iloc[0]
    assert every["n"] == 10 and every["n_laws"] == 5 and every["n_families"] == 3 and every["n_strata"] == 2
    assert "abl_core" not in every["law_list"]  # the ablation stays out of the correlations
    values = pd.read_csv(tree["m8"] / "correlation_pooled_laws.csv")
    family = {law: name for name, laws in FAMILIES.items() for law in laws}
    expected = clustered_spearman(values["unstable_eq"], values["macro_error"], values["corridor"], values["law"],
                                  values["law"].map(family), n_resamples=400, n_permutations=2000, seed=7)  # fmt: skip
    for key in ("estimate", "law_low", "law_high", "family_low", "family_high", "pairs_low", "pairs_high", "p_linked"):
        assert every[key] == pytest.approx(expected[key]), key
    pooled_d122 = pd.read_csv(tree["m8"] / "correlation_pooled.csv")
    d122 = pooled_d122[(pooled_d122["scope"] == "pooled (corridor as stratum)") & (pooled_d122["error"] == "macro error")
                       & (pooled_d122["laws"] == "all laws with runs (D122)")].iloc[0]  # fmt: skip
    assert every["estimate"] == pytest.approx(d122["estimate"])  # the same coefficient as correlation_pooled
    without = pooled[pooled["subset"] == "without F2"].iloc[0]
    rest = values[values["law"].map(family) != "F2"]
    assert without["n"] == 6 and without["left_out"] == "F2"
    assert without["estimate"] == pytest.approx(stratified_spearman(rest["unstable_eq"], rest["macro_error"],
                                                                    rest["corridor"]))  # fmt: skip
    learned = pooled[pooled["subset"].str.startswith("learned")].iloc[0]
    assert learned["n"] == 6 and learned["n_families"] == 2 and learned["family_list"] == "F2, F3"
    a = table[(table["scope"] == "A") & (table["error"] == "macro error") & (table["subset"] == "all laws with runs")]
    assert a.iloc[0]["n"] == 5 and a.iloc[0]["n_laws"] == 5  # per corridor the law clusters are the laws
    md = (tree["m8"] / "correlation_clustered.md").read_text(encoding="utf-8")
    assert "default_rng(7)" in md and "F1: idm_global, residual_idm_certified;" in md and "| law clusters |" in md
    assert any(line.startswith("TABLE correlation_clustered: pooled (corridor as stratum): macro error") for line in
               tree["lines"])  # fmt: skip


def test_contacts_absolute_table(tree):
    table = pd.read_csv(tree["m8"] / "contacts_absolute.csv").set_index(["corridor", "law"])
    free = table.loc[("A", "law_free")]
    seeds = np.array(SEEDS, dtype=float)
    assert free["runs"] == 8 and free["runs_with_contacts"] == 8
    assert free["collision_episodes"] == pytest.approx(np.mean(50.0 + seeds))
    assert free["vehicles_in_contact"] == pytest.approx(150.0) and free["vehicles_in_contact_share"] == pytest.approx(0.1)
    assert free["vehicle_km"] == pytest.approx(np.mean(600.0 + 10.0 * seeds))
    assert free["vehicle_km_ratio"] == pytest.approx(np.mean((600.0 + 10.0 * seeds) / 620.0))
    assert free["rate_pooled"] == pytest.approx(1000.0 * np.sum(50.0 + seeds) / np.sum(600.0 + 10.0 * seeds))
    assert free["collisions_per_1000_vkm"] == pytest.approx(np.mean(1000.0 * (50.0 + seeds) / (600.0 + 10.0 * seeds)))
    assert free["rate_pooled_low"] <= free["rate_pooled"] <= free["rate_pooled_high"]
    assert free["shortfall_window"] == pytest.approx(1350 * 0.02) and free["shortfall_run"] == pytest.approx(20.0)
    assert free["inserted_share"] == pytest.approx(0.98) and free["depart_delay_run"] == pytest.approx(4.0)
    assert free["depart_delay_window"] == pytest.approx(5.0) and free["n_planned_run"] == pytest.approx(1800.0)
    idm = table.loc[("B", "idm_global")]
    assert idm["runs"] == 4 and idm["collision_episodes"] == 0.0 and idm["runs_with_contacts"] == 0
    assert idm["rate_pooled"] == 0.0 and idm["shortfall_run"] == 0.0
    truth = table.loc[("A", "ground truth")]
    assert truth["vehicle_km"] == pytest.approx(620.0) and truth["n_vehicles_window"] == pytest.approx(1500.0)
    assert truth["n_planned_run"] == pytest.approx(1836.0) and pd.isna(truth["runs"])
    assert ("A", "abl_core") in table.index and ("A", "abl_absent") not in table.index
    md = (tree["m8"] / "contacts_absolute.md").read_text(encoding="utf-8")
    assert "| contact episodes per run |" in md and "episodes / 1000 veh-km (pooled)" in md


def test_h12_2_error_table(tree):
    table = pd.read_csv(tree["m8"] / "h12_2_error.csv").set_index(["corridor", "component"])
    a = table.loc["A"]
    assert a.index.tolist()[:3] == ["throughput", "travel_time", "wave_speed"]  # the macro triple first
    assert a["in_triple"].tolist()[:4] == [True, True, True, False]
    assert a.loc["throughput", "difference"] == pytest.approx(-0.01) and a.loc["throughput", "outcome"] == "smaller error"
    assert a.loc["wave_speed", "difference"] == pytest.approx(0.1) and a.loc["wave_speed", "outcome"] == "larger error"
    assert a.loc["mean_speed", "difference"] == 0.0 and a.loc["mean_speed", "outcome"] == "no difference"
    assert a.loc["mean_speed", "wilcoxon_p"] == 1.0 and a.loc["throughput", "pairs"] == 8
    reference = np.array([abs(components("idm_global", s, k)["throughput"]) for s in ("a_p0", "a_p1") for k in SEEDS])
    assert a.loc["throughput", "relative"] == pytest.approx((reference.mean() - 0.01) / reference.mean() - 1.0)
    assert a.loc["throughput", "wilcoxon_p_holm"] >= a.loc["throughput", "wilcoxon_p"]
    assert a.loc["macro_error", "kind"] == "summary" and pd.isna(a.loc["macro_error", "wilcoxon_p_holm"])
    assert set(table.index.get_level_values("corridor")) == {"A", "B"}
    md = (tree["m8"] / "h12_2_error.md").read_text(encoding="utf-8")
    assert "the pre-specified H12.2 used the TOST of the raw metrics" in md and "stays as reported" in md


def test_ablation_rows(tree):
    """An ablation law with a law file has rows in laws, components, instability (closed form, out of the correlation),
    contacts and e4_rmax (empty training cells, the counts of the other rows stay integers); one without inputs is left
    out with a note."""
    laws = pd.read_csv(tree["m5"] / "laws.csv")
    rows = laws[laws["scenario"] == "all"].set_index(["corridor", "law"])
    assert rows.loc[("A", "abl_core"), "runs"] == 8 and rows.loc[("B", "abl_core"), "runs"] == 4
    assert ("A", "abl_absent") not in rows.index
    components_table = pd.read_csv(tree["m5"] / "components.csv").set_index(["corridor", "law"])
    assert ("B", "abl_core") in components_table.index
    instability = pd.read_csv(tree["m5"] / "instability.csv").set_index(["corridor", "law"])
    core = instability.loc[("A", "abl_core")]
    assert core["kind"] == "idm" and core["members"] == 2 and core["unstable_eq"] == 0.0
    assert not bool(core["in_correlation"])
    corr = pd.read_csv(tree["m5"] / "instability_correlation.csv")
    assert corr[(corr["corridor"] == "A") & (corr["laws"] == "all laws")]["n"].max() == 4  # the design only
    rmax = pd.read_csv(tree["m5"] / "e4_rmax.csv")
    assert rmax["law"].tolist() == ["residual_idm_certified", "abl_core", "residual_idm_certified", "abl_core"]
    extra = rmax[rmax["law"] == "abl_core"].iloc[0]
    assert pd.isna(extra["runs_highd"]) and pd.isna(extra["rmse_ngsim"]) and extra["runs_corridor"] == 8
    text = (tree["m5"] / "e4_rmax.csv").read_text(encoding="utf-8").splitlines()
    regular = next(line for line in text[1:] if ",residual_idm_certified," in line).split(",")
    header = text[0].split(",")
    assert regular[header.index("runs_highd")] == "1" and regular[header.index("drivers_ngsim")] == "3"
    md = (tree["m5"] / "e4_rmax.md").read_text(encoding="utf-8")
    assert "Extra rows of the factorial ablation" in md and "abl_core (r_max 0, core alone" in md
    assert "abl_absent" not in md
    missing = (tree["m5"] / "missing.txt").read_text(encoding="utf-8")
    assert "[runs] abl_absent: no law file and no runs yet (a law of the review of M8): not in the tables" in missing
    assert "e4_x" not in missing  # no training notes for a row that is not in the table
    assert "Factorial ablation of the certified hybrid" in (tree["m5"] / "laws.md").read_text(encoding="utf-8")
