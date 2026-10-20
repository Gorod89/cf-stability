"""The windowed laws of M11 (D128: ``windowed_laws`` of cf_stability/eval/corridor_tables.py) on a hand-made tree: two
corridors, two laws of the design, a windowed law with runs and member audits and one without law file and runs. The
present law has rows in laws, components and instability (with the audits of its members) and a note line, it is not in
the instability correlation nor in correlation_pooled; the absent one is left out with a note."""

from __future__ import annotations

import pandas as pd
import pytest

from cf_stability.eval.corridor_tables import WINDOWED_LAWS, CorridorTablesConfig, make_corridor_tables
from cf_stability.utils import write_json

SCENARIOS = ("i80_p0", "i80_p1", "us101_p0")
SEEDS = (0, 1, 2)
DESIGN_LAWS = ("idm_global", "residual_idm_certified")
PRESENT, ABSENT = "gru_windowed", "lstm_windowed"
OFFSET = {"idm_global": 0.20, "residual_idm_certified": 0.15, PRESENT: 0.30}
DESIGN = {
    "scenarios": list(SCENARIOS), "seeds": list(SEEDS), "n_resamples": 100, "n_permutations": 100,
    "laws": list(DESIGN_LAWS), "rmax_laws": [], "temporal_laws": [], "temporal_pairs": {}, "ablation_laws": [],
    "optional_laws": [], "windowed_laws": [PRESENT, ABSENT], "networks": [], "rmax": [], "rmax_ablation": [],
    "power_metrics": ["macro_error"], "asymmetry_contrasts": [], "cluster_resamples": 50, "sensitivity_variants": [],
    "sensitivity_scenario": "i80_p0", "sensitivity_laws": ["idm_global"], "sensitivity_order": ["idm_global"],
    "micro_others": [], "run_folds": [0], "run_seeds": [0], "ablation_pairs": [],
}  # fmt: skip


def macro(law: str, scenario: str, seed: int) -> dict:
    value = 0.05 * SCENARIOS.index(scenario) + 0.01 * seed + OFFSET[law]
    return {"window": [180.0, 840.0], "macro_error": {"value": value, "n_components": 8},
            "macro_error_dynamic": {"value": 1.5 * value, "n_components": 4}, "collisions_per_1000_vkm": 0.0,
            "config_hash": "m"}  # fmt: skip


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("windowed") / "runs"
    corridor = root / "corridor"
    for scenario in SCENARIOS:
        write_json(corridor / "scenarios" / scenario / "macro.json", {"window": [180.0, 840.0], "macro_error": None})
        for law in OFFSET:
            for seed in SEEDS:
                run = corridor / scenario / law / f"seed{seed}"
                write_json(run / "macro.json", macro(law, scenario, seed))
                write_json(run / "run.json", {"g_mean": 1.0, "dN_mean": 0.0})
    audits = {PRESENT: ((0.8, 0.2), (0.6, 0.4)), "idm_global": ((0.1, 0.9),), "residual_idm_certified": ((0.0, 1.0),)}
    for law, shares in audits.items():  # member runs with audits (unstable among equilibria, stable band share)
        members = []
        for fold, (unstable, stable) in enumerate(shares):
            run = f"{law}_members/ngsim_i80/model/driver_fold{fold}_seed0"
            summary = {"share_unstable_numerical": {"support": unstable},
                       "band_numerical": {"support": {"stable": stable}}}  # fmt: skip
            write_json(root / run / "stability.json", {"audit": {"summary": summary}})
            members.append({"run": run, "model": f"{run}/model.pt"})
        write_json(corridor / "laws" / f"{law}.json", {"law": law, "kind": "models", "members": members})
    cfg = CorridorTablesConfig.from_mapping(DESIGN, corridor, root, root / "_tables" / "m5", root / "_tables" / "m8")
    lines = make_corridor_tables(cfg)
    return {"lines": lines, "m5": root / "_tables" / "m5", "m8": root / "_tables" / "m8"}


def test_rows_of_the_windowed_law_and_none_of_the_absent_one(tree):
    assert WINDOWED_LAWS == ("gru_windowed", "lstm_windowed", "perl_windowed")
    laws = pd.read_csv(tree["m5"] / "laws.csv")
    assert set(laws["law"]) - {"ground truth"} == {*DESIGN_LAWS, PRESENT}
    mine = laws[laws["law"] == PRESENT].set_index("corridor")
    assert sorted(mine.index) == ["I-80", "US-101"]
    assert mine.loc["I-80", "macro_error"] == pytest.approx(0.30 + 0.025 + 0.01)  # mean over 2 scenarios x 3 seeds
    components = pd.read_csv(tree["m5"] / "components.csv")
    assert PRESENT in set(components["law"]) and ABSENT not in set(components["law"])
    missing = (tree["m5"] / "missing.txt").read_text(encoding="utf-8")
    assert f"{ABSENT}: no law file and no runs yet (a law of M11): not in the tables" in missing
    md = (tree["m5"] / "laws.md").read_text(encoding="utf-8")
    assert "Windowed-gain penalty of M11 (D128; rows only, not in the correlations nor the verdicts of H12)" in md


def test_instability_rows_with_the_member_audits_and_no_correlation(tree):
    instability = pd.read_csv(tree["m5"] / "instability.csv")
    row = instability[(instability["law"] == PRESENT) & (instability["corridor"] == "I-80")].iloc[0]
    assert row["members"] == 2 and row["audited"] == 2 and not row["in_correlation"]
    assert row["unstable_eq"] == pytest.approx(0.7) and row["not_stable"] == pytest.approx(0.7)
    correlation = pd.read_csv(tree["m5"] / "instability_correlation.csv")
    assert set(correlation["n"]) == {len(DESIGN_LAWS)}  # the windowed law is not in the correlation
    pooled = pd.read_csv(tree["m8"] / "correlation_pooled.csv")
    assert PRESENT not in " ".join(map(str, pooled.to_numpy().ravel()))
