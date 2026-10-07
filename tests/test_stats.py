"""Statistics of the experiments (cf_stability/eval/stats.py): hand-computed values and known properties."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from cf_stability.eval.stats import bootstrap_ci, driver_table, holm, paired_comparison, tost_relative


def test_bootstrap_resamples_whole_groups():
    """Group a holds three zeros, group b one 12. Drawing two groups gives the means 0 (aa), 3 (ab, ba)
    and 12 (bb) with the probabilities 1/4, 1/2, 1/4: the 95 % interval is [0, 12]. Drawing four rows
    instead gives the mean 12 with probability 1/256 and 9 with 12/256: the upper end is 9."""
    values, groups = [0.0, 0.0, 0.0, 12.0], ["a", "a", "a", "b"]
    grouped = bootstrap_ci(values, groups, n_resamples=2000)
    assert (grouped["estimate"], grouped["low"], grouped["high"]) == (3.0, 0.0, 12.0)
    assert (grouped["n"], grouped["n_groups"], grouped["n_resamples"], grouped["level"]) == (4, 2, 2000, 0.95)
    rows = bootstrap_ci(values, n_resamples=2000)
    assert (rows["estimate"], rows["low"], rows["high"], rows["n_groups"]) == (3.0, 0.0, 9.0, 4)
    # the labels name the groups, their order and the order of the rows do not matter
    assert bootstrap_ci([12.0, 0.0, 0.0, 0.0], [7, 3, 3, 3], n_resamples=2000) == grouped


def test_bootstrap_details():
    assert bootstrap_ci([2.5] * 5)["low"] == bootstrap_ci([2.5] * 5)["high"] == 2.5
    with_nan = bootstrap_ci([1.0, np.nan, 3.0, np.inf, 5.0], groups=[1, 2, 3, 4, 5], statistic=np.median)
    assert with_nan["n"] == 3 and with_nan["estimate"] == 3.0 and with_nan["n_groups"] == 3
    empty = bootstrap_ci([np.nan])
    assert empty["n"] == 0 and math.isnan(empty["estimate"]) and math.isnan(empty["low"])
    rng = np.random.default_rng(1)
    x = rng.normal(10.0, 1.0, 50)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3) != bootstrap_ci(x, seed=4)  # deterministic per seed
    pairs = np.column_stack((x, 1.1 * x))  # rows of two columns and a statistic of both
    ratio = bootstrap_ci(pairs, statistic=lambda r: r[:, 1].mean() / r[:, 0].mean())
    assert ratio["low"] == pytest.approx(1.1) and ratio["high"] == pytest.approx(1.1)
    with pytest.raises(ValueError, match="group labels"):
        bootstrap_ci([1.0, 2.0], groups=[1])


def test_bootstrap_interval_covers_the_mean_of_normal_data():
    rng = np.random.default_rng(0)
    covered = [
        (ci := bootstrap_ci(rng.normal(5.0, 2.0, 40), n_resamples=400, seed=k))["low"] <= 5.0 <= ci["high"]
        for k in range(200)
    ]
    assert 0.88 <= np.mean(covered) <= 0.99  # nominal 0.95; the percentile interval is a little short for n = 40


def test_driver_table():
    run = lambda drivers, rmse, collided: pd.DataFrame(  # noqa: E731
        {"event_id": [f"e{k}" for k in range(len(drivers))], "follower_id": drivers, "site": "a", "n_scored": 100,
         "rmse_s": rmse, "rmse_v": [0.5] * len(drivers), "collided": collided}
    )  # fmt: skip
    seed0 = run(["d1", "d1", "d2"], [1.0, 3.0, 5.0], [False, True, False])
    seed1 = run(["d2", "d1", "d1"], [7.0, 2.0, 2.0], [False, False, False])
    other_fold = run(["d3"], [4.0], [True])
    table = driver_table([seed0, seed1, other_fold])
    assert list(table.index) == ["d1", "d2", "d3"] and table.index.name == "follower_id"
    assert table["rmse_s"].tolist() == [2.0, 6.0, 4.0]  # d1: (2 + 2) / 2, d2: (5 + 7) / 2
    assert table["collided"].tolist() == [0.25, 0.0, 1.0]  # d1: collision rate 1/2 with seed 0, 0 with seed 1
    assert table["n_events"].tolist() == [2.0, 1.0, 1.0] and table["n_runs"].tolist() == [2, 2, 1]
    assert table["site"].tolist() == ["a", "a", "a"] and table["rmse_v"].tolist() == [0.5, 0.5, 0.5]
    assert driver_table({"s0": seed0, "s1": seed1, "f": other_fold}).equals(table)
    assert driver_table([]).empty


def test_paired_comparison():
    out = paired_comparison([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 3.5, 4.5])
    assert (out["n"], out["mean_a"], out["mean_b"], out["mean_difference"]) == (4, 2.5, 3.25, 0.75)
    assert out["relative_change"] == pytest.approx(0.3) and out["ci_low"] <= 0.3 <= out["ci_high"]
    # Series are paired by their labels; pairs with a missing value are dropped
    a = pd.Series([1.0, 2.0, np.nan, 4.0, 9.0], index=["d1", "d2", "d3", "d4", "d5"])
    b = pd.Series([4.5, 2.5, 3.0, 1.5], index=["d4", "d2", "d3", "d1"])
    out = paired_comparison(a, b)
    assert (out["n"], out["mean_a"], out["mean_b"]) == (3, pytest.approx(7.0 / 3), pytest.approx(8.5 / 3))
    # the Wilcoxon test is that of SciPy on the differences
    rng = np.random.default_rng(2)
    x = rng.lognormal(1.0, 0.3, 40)
    y = x * rng.normal(1.05, 0.05, 40)
    out = paired_comparison(x, y, n_resamples=500)
    test = sps.wilcoxon(y - x)
    assert out["p_value"] == test.pvalue and out["wilcoxon_statistic"] == test.statistic and out["p_value"] < 0.01
    assert out["ci_low"] < y.mean() / x.mean() - 1.0 < out["ci_high"]
    with pytest.raises(ValueError, match="length"):
        paired_comparison([1.0, 2.0], [1.0])


def test_holm():
    """Textbook example: four hypotheses, level 0.05 rejects H4 and H1 (Holm 1979, as in Wikipedia)."""
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02], rtol=1e-12)
    assert [p <= 0.05 for p in adjusted] == [True, False, False, True]
    assert holm({"mlp": 0.5, "gru": 0.6}) == {"mlp": 1.0, "gru": 1.0}  # capped at 1
    series = holm(pd.Series([0.01, np.nan, 0.04], index=["a", "b", "c"], name="p"))
    assert series.index.tolist() == ["a", "b", "c"] and series.name == "p"
    np.testing.assert_allclose(series.to_numpy(), [0.02, np.nan, 0.04], rtol=1e-12)  # NaN does not count
    assert holm([]).size == 0


@pytest.mark.parametrize("test", ["t", "wilcoxon"])
def test_tost_relative(test):
    rng = np.random.default_rng(3)
    a = rng.lognormal(1.0, 0.3, 60)
    equal = tost_relative(a, a.copy(), test=test)
    assert equal["equivalent"] and equal["p_value"] < 1e-6 and equal["relative_difference"] == 0.0
    near = tost_relative(a, a * rng.normal(1.03, 0.02, 60), test=test)
    assert near["equivalent"] and 0.0 < near["relative_difference"] < 0.1
    far = tost_relative(a, 1.2 * a, test=test)
    assert not far["equivalent"] and far["p_upper"] > 0.99 and far["relative_difference"] == pytest.approx(0.2)
    if test == "t":  # the two one-sided tests by their definition
        b = a * rng.normal(1.0, 0.1, 60)
        out = tost_relative(a, b, margin=0.1)
        assert out["p_lower"] == sps.ttest_1samp(b - 0.9 * a, 0.0, alternative="greater").pvalue
        assert out["p_upper"] == sps.ttest_1samp(b - 1.1 * a, 0.0, alternative="less").pvalue
        assert out["p_value"] == max(out["p_lower"], out["p_upper"])
