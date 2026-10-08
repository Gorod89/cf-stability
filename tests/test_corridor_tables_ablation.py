"""Table ablation_pairs of the review of M8 (cf_stability/eval/corridor_tables.py): the variants of the factorial
ablation of the certified hybrid against each other on a hand-made tree: two corridors, five variants with runs (one run
of D missing on I-80), B' without law file and runs; the pairing by scenario and seed, the signs of known differences,
the relative change without a reference, the intervals and the Wilcoxon p-values against direct computations, the rows
and the notes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from cf_stability.eval.corridor_tables import ABLATION_PAIRS, M9_TABLES, CorridorTablesConfig, make_corridor_tables
from cf_stability.eval.stats import bootstrap_ci, paired_comparison
from cf_stability.utils import REPO_ROOT, write_json

SCENARIOS = ("i80_p0", "i80_p1", "us101_p0")
I80 = ("i80_p0", "i80_p1")
SEEDS = (0, 1, 2, 3)
A, B, B_I80, C, D, E = ("idm_global", "idm_core_margin", "idm_margin_i80", "residual_idm_free_r0.3",
                        "residual_idm_margin_free_r0.3", "residual_idm_certified")  # fmt: skip
OFFSET = {A: 0.20, B: 0.20, C: 0.18, D: 0.22, E: 0.15}  # the laws with runs (not B'): macro error above a common part
MISSING = ("i80_p1", D, 2)  # the run of D that does not exist
DESIGN = {
    "scenarios": list(SCENARIOS), "seeds": list(SEEDS), "n_resamples": 200, "n_permutations": 200, "laws": [A, E],
    "rmax_laws": [C], "temporal_laws": [], "temporal_pairs": {}, "ablation_laws": [B, B_I80, D],
    "optional_laws": [B, B_I80, D], "networks": [], "rmax": [], "rmax_ablation": [], "power_metrics": ["macro_error"],
    "asymmetry_contrasts": [], "cluster_resamples": 50, "sensitivity_variants": [], "sensitivity_scenario": "i80_p0",
    "sensitivity_laws": [A], "sensitivity_order": [A], "micro_others": [], "run_folds": [0], "run_seeds": [0],
}  # fmt: skip  # ablation_pairs: the default (E-D, E-C, E-B, D-C, D-B, B-A, B'-A)


def error(law: str, scenario: str, seed: int) -> float:
    """Macro error of a run: a part of the scenario and the seed, the offset of the law, a seed term shared by D and E
    (E - D is -0.07 on every pair) and an alternating term of B (B - A averages to 0 over the seeds)."""
    value = 0.05 * SCENARIOS.index(scenario) + 0.01 * seed + OFFSET[law]
    if law in (D, E):
        value += 0.003 * seed**2
    if law == B:
        value += 0.02 if seed % 2 == 0 else -0.02
    return value


def collisions(law: str, scenario: str, seed: int) -> float:
    """Collisions per 1000 vehicle-km: the free core C more with the seed, D a few, the others none."""
    return {C: 2.0 + seed, D: 0.5}.get(law, 0.0)


def macro(law: str, scenario: str, seed: int) -> dict:
    value = error(law, scenario, seed)
    return {"window": [180.0, 840.0], "macro_error": {"value": value, "n_components": 8},
            "macro_error_dynamic": {"value": 1.5 * value, "n_components": 4},
            "collisions_per_1000_vkm": collisions(law, scenario, seed), "config_hash": "m"}  # fmt: skip


def common(candidate: str, reference: str, scenarios: tuple[str, ...], value) -> tuple[np.ndarray, np.ndarray]:
    """The values of the candidate and of the reference on their common runs (order: scenarios, then seeds)."""
    keys = [(s, k) for s in scenarios for k in SEEDS if MISSING not in ((s, candidate, k), (s, reference, k))]
    return np.array([value(candidate, s, k) for s, k in keys]), np.array([value(reference, s, k) for s, k in keys])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation") / "runs"
    corridor = root / "corridor"
    for scenario in SCENARIOS:
        write_json(corridor / "scenarios" / scenario / "macro.json", {"window": [180.0, 840.0], "macro_error": None})
        for law in OFFSET:
            for seed in SEEDS:
                if (scenario, law, seed) == MISSING:
                    continue
                run = corridor / scenario / law / f"seed{seed}"
                write_json(run / "macro.json", macro(law, scenario, seed))
                write_json(run / "run.json", {"g_mean": 1.0, "dN_mean": 0.0})
    cfg = CorridorTablesConfig.from_mapping(DESIGN, corridor, root, root / "_tables" / "m5", root / "_tables" / "m8")
    lines = make_corridor_tables(cfg)
    table = pd.read_csv(root / "_tables" / "m8" / "ablation_pairs.csv").set_index(["corridor", "pair", "metric"])
    return {"lines": lines, "table": table, "m5": root / "_tables" / "m5", "m8": root / "_tables" / "m8"}


def test_rows_and_notes(tree):
    """One row per corridor, pair and metric; the pair with a law that is not in the tables (B') has none and a note
    in missing_corridor.txt (not in missing.txt); the pairs count the runs of both laws."""
    assert "ablation_pairs" in M9_TABLES
    rows = tree["table"].reset_index()
    assert len(rows) == 2 * 6 * 3  # corridors x pairs with both laws x metrics
    assert rows["pair"].unique().tolist() == ["E - D", "E - C", "E - B", "D - C", "D - B", "B - A"]  # no B' - A
    assert rows["metric"].unique().tolist() == ["macro_error", "macro_error_dynamic", "collisions_per_1000_vkm"]
    assert rows.groupby("corridor").size().to_dict() == {"I-80": 18, "US-101": 18}
    i80 = rows[rows["corridor"] == "I-80"].set_index("pair")["pairs"]
    assert i80.loc[["E - D", "D - C", "D - B"]].eq(7).all() and i80.loc[["E - C", "E - B", "B - A"]].eq(8).all()
    assert rows.loc[rows["corridor"] == "US-101", "pairs"].eq(4).all()
    missing = (tree["m8"] / "missing_corridor.txt").read_text(encoding="utf-8").splitlines()
    for corridor in ("I-80", "US-101"):
        assert f"[ablation_pairs] {corridor}: idm_margin_i80 or idm_global not in the tables, no comparison" in missing
    assert "[ablation_pairs]" not in (tree["m5"] / "missing.txt").read_text(encoding="utf-8")


def test_pairing_and_signs_of_the_macro_error(tree):
    """E - D: the means over the common runs and the same difference on every pair (paired by scenario and seed, not by
    position); B - A: the interval of the alternating differences against the bootstrap of the configured settings."""
    table = tree["table"]
    row = table.loc[("I-80", "E - D", "macro_error")]
    cand, ref = common(E, D, I80, error)
    assert row["candidate"] == E and row["reference"] == D and row["pairs"] == len(cand) == 7
    assert row["mean_candidate"] == pytest.approx(cand.mean()) and row["mean_reference"] == pytest.approx(ref.mean())
    assert abs(row["mean_candidate"] - np.mean([error(E, s, k) for s in I80 for k in SEEDS])) > 1e-3  # not all runs
    for key in ("difference", "difference_low", "difference_high"):
        assert row[key] == pytest.approx(-0.07)
    expected = paired_comparison(ref, cand, n_resamples=200, level=0.95, seed=0)
    assert row["relative"] == pytest.approx(cand.mean() / ref.mean() - 1.0)
    assert (row["relative_low"], row["relative_high"]) == pytest.approx((expected["ci_low"], expected["ci_high"]))
    assert row["wilcoxon_p"] == pytest.approx(expected["p_value"]) and row["wilcoxon_p"] < 0.05
    assert row["outcome"] == "lower"
    assert table.loc[("I-80", "E - D", "macro_error_dynamic"), "difference"] == pytest.approx(1.5 * -0.07)
    assert table.loc[("I-80", "E - C", "macro_error"), "outcome"] == "lower"
    assert table.loc[("I-80", "E - B", "macro_error"), "outcome"] == "lower"
    assert table.loc[("I-80", "D - C", "macro_error"), "outcome"] == "higher"
    assert table.loc[("US-101", "D - C", "macro_error"), "difference"] == pytest.approx(0.04 + 0.003 * np.mean(
        np.square(SEEDS)))  # fmt: skip
    row = table.loc[("I-80", "B - A", "macro_error")]
    cand, ref = common(B, A, I80, error)
    ci = bootstrap_ci(cand - ref, None, np.mean, 200, 0.95, 0)
    assert (row["difference"], row["difference_low"], row["difference_high"]) == pytest.approx(
        (ci["estimate"], ci["low"], ci["high"]), abs=1e-12)  # fmt: skip
    assert row["difference_low"] < 0.0 < row["difference_high"] and row["outcome"] == "no difference"
    assert row["wilcoxon_p"] == pytest.approx(paired_comparison(ref, cand, n_resamples=200, seed=0)["p_value"])


def test_collisions_and_the_relative_change(tree):
    """No collisions on either side: no difference, p 1 and no relative change; a reference without collisions: no
    relative change; a candidate without collisions against a colliding reference: -100 %."""
    table = tree["table"]
    none = table.loc[("I-80", "B - A", "collisions_per_1000_vkm")]
    assert (none["difference"], none["difference_low"], none["difference_high"]) == (0.0, 0.0, 0.0)
    assert none["outcome"] == "no difference" and none["wilcoxon_p"] == 1.0 and pd.isna(none["relative"])
    free = table.loc[("I-80", "E - C", "collisions_per_1000_vkm")]
    assert free["difference"] == pytest.approx(-np.mean([2.0 + k for k in SEEDS])) and free["outcome"] == "lower"
    assert (free["relative"], free["relative_low"], free["relative_high"]) == pytest.approx((-1.0, -1.0, -1.0))
    margin = table.loc[("US-101", "D - B", "collisions_per_1000_vkm")]
    assert margin["difference"] == pytest.approx(0.5) and margin["outcome"] == "higher"
    assert pd.isna(margin["relative"]) and pd.isna(margin["relative_low"]) and margin["wilcoxon_p"] < 1.0


def test_markdown_and_printed_line(tree):
    md = (tree["m8"] / "ablation_pairs.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    assert sum(line.startswith("# ") for line in lines) == 1  # one title
    assert lines[0].startswith("# Factorial ablation of the certified hybrid: paired differences between its variants")
    for text in ("A idm_global:", "B idm_core_margin:", "C residual_idm_free_r0.3:", "D residual_idm_margin_free_r0.3:",
                 "E residual_idm_certified:"):  # fmt: skip
        assert text in md, text
    assert "B' idm_margin_i80" not in md  # no rows: not among the variants of the notes
    assert ("| corridor | pair | candidate | reference | metric | pairs | mean candidate | mean reference | "
            "difference | relative | p (Wilcoxon) | outcome |") in md  # fmt: skip
    assert "| I-80 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | macro error | 7 |" in md
    assert any(line.startswith("TABLE ablation_pairs: 36 rows, macro error lower: I-80 E - D, I-80 E - C, I-80 E - B, "
                               "US-101 E - D") for line in tree["lines"])  # fmt: skip


def test_configured_pairs():
    """The pairs of configs/corridor_metrics.yaml are the default of the module; a configured list becomes tuples."""
    design = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs" / "corridor_metrics.yaml"))["design"]
    assert tuple(tuple(pair) for pair in design["ablation_pairs"]) == ABLATION_PAIRS
    cfg = CorridorTablesConfig.from_mapping({"ablation_pairs": [[E, A]]}, "corridor", "runs", "out")
    assert cfg.ablation_pairs == ((E, A),)
