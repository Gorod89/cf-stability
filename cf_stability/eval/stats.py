"""Statistics of the experiments (docs/m4_contract.md, section 2.3; D88).

Unit of resampling and of the paired tests: the driver for RMSE-type numbers (mean over the events
of a driver and over the seeds, :func:`driver_table`), the run (fold and seed) for numbers of a model
(stability shares, growth error, hysteresis area). Percentile bootstrap with 1000 resamples,
Wilcoxon signed-rank test, Holm's correction over the architectures of one hypothesis; the TOST of
:func:`tost_relative` tests equivalence within a relative margin (M5). Everything is deterministic
for a fixed ``seed``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def bootstrap_ci(
    values: ArrayLike, groups: ArrayLike | None = None, statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000, level: float = 0.95, seed: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Percentile bootstrap interval of ``statistic(values)`` over the groups.

    ``values``: ``[n]`` or ``[n, k]`` (rows, e.g. pairs); ``groups``: ``[n]`` labels of the resampling
    unit (drivers, runs), None when every row is its own unit. Every resample draws as many groups
    as there are, with replacement, and takes all rows of each drawn group; ``statistic`` gets the
    rows of a resample. Rows with a non-finite value are left out. ``{estimate, low, high, level, n,
    n_groups, n_resamples}``; NaN without rows.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"values must have one or two dimensions, got shape {x.shape}")
    labels = np.arange(len(x)) if groups is None else np.asarray(groups)
    if labels.shape != (len(x),):
        raise ValueError(f"{len(labels)} group labels for {len(x)} rows")
    finite = np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=1)
    x, labels = x[finite], labels[finite]
    out = {"estimate": np.nan, "low": np.nan, "high": np.nan, "level": level, "n": len(x), "n_groups": 0,
           "n_resamples": n_resamples}  # fmt: skip
    if len(x) == 0:
        return out
    _, inverse = np.unique(labels, return_inverse=True)  # groups in sorted order: independent of the row order
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")  # rows grouped
    sizes = np.bincount(inverse)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    rng = np.random.default_rng(seed)
    singletons = bool((sizes == 1).all())  # every row its own group: the draw indexes the rows directly
    boot = np.empty(n_resamples)
    for r in range(n_resamples):  # one draw at a time: the memory stays that of one resample
        draw = rng.integers(0, len(sizes), size=len(sizes))
        if singletons:
            rows = draw
        else:
            lengths = sizes[draw]
            ends = np.cumsum(lengths)
            rows = np.arange(ends[-1]) - np.repeat(ends - lengths, lengths) + np.repeat(starts[draw], lengths)
        boot[r] = statistic(x[order[rows]])
    alpha = 1.0 - level
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    out.update(estimate=float(statistic(x)), low=float(low), high=float(high), n_groups=len(sizes))
    return out


def driver_table(
    frames: Sequence[pd.DataFrame] | Mapping[Any, pd.DataFrame], driver: str = "follower_id"
) -> pd.DataFrame:
    """One row per driver from per-event frames of several runs (the ``test_events.parquet`` of all
    folds and seeds of one configuration).

    Numeric and boolean columns are averaged over the events of a driver within every run (a
    boolean gives a rate), then over the runs in which the driver appears (the seeds). Index: the
    driver; extra columns ``n_events`` (events per run, averaged) and ``n_runs``; ``site`` is kept.
    """
    per_run = []
    for frame in frames.values() if isinstance(frames, Mapping) else frames:
        numeric = [c for c in frame.select_dtypes(include=["number", "bool"]).columns if c != driver]
        grouped = frame.groupby(driver, sort=True)
        means = grouped[numeric].mean().astype(np.float64)
        means["n_events"] = grouped.size().astype(np.float64)
        if "site" in frame.columns:
            means["site"] = grouped["site"].first()
        per_run.append(means)
    if not per_run:
        return pd.DataFrame(columns=["n_events", "n_runs"]).rename_axis(driver)
    stacked = pd.concat(per_run)
    grouped = stacked.groupby(level=0, sort=True)
    table = grouped[[c for c in stacked.columns if c != "site"]].mean()
    table["n_runs"] = grouped.size()
    if "site" in stacked.columns:
        table.insert(0, "site", grouped["site"].first())
    return table.rename_axis(driver)


def _paired(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of ``a`` and ``b``: common index labels of two Series, else by position; non-finite pairs dropped."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        both = pd.concat({"a": a, "b": b}, axis=1, join="inner")
        x, y = both["a"].to_numpy(dtype=np.float64), both["b"].to_numpy(dtype=np.float64)
    else:
        x, y = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"paired samples differ in length: {len(x)} and {len(y)}")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _relative_change(rows: np.ndarray) -> float:
    return float(rows[:, 1].mean() / rows[:, 0].mean() - 1.0)


def paired_comparison(
    a: ArrayLike, b: ArrayLike, *, n_resamples: int = 1000, level: float = 0.95, seed: int = 0
) -> dict[str, Any]:
    """Paired comparison of ``b`` with the reference ``a`` over their common units (drivers or runs).

    ``a``, ``b``: Series indexed by the unit (the common labels are paired) or sequences of equal
    length (paired by position); pairs with a non-finite value are dropped. Relative change of the
    means ``mean(b) / mean(a) - 1`` with its percentile bootstrap interval over the units; Wilcoxon
    signed-rank test of ``b - a`` (``scipy.stats.wilcoxon`` with its defaults, two-sided).
    """
    x, y = _paired(a, b)
    pairs = np.column_stack((x, y))
    ci = bootstrap_ci(pairs, statistic=_relative_change, n_resamples=n_resamples, level=level, seed=seed)
    statistic = p_value = np.nan
    if len(x) and not np.any(y != x):
        statistic, p_value = 0.0, 1.0  # what SciPy returns for differences that are all zero, without its warning
    elif len(x):
        test = sps.wilcoxon(y - x)
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n": len(x),
        "mean_a": float(x.mean()) if len(x) else np.nan,
        "mean_b": float(y.mean()) if len(x) else np.nan,
        "mean_difference": float((y - x).mean()) if len(x) else np.nan,
        "relative_change": ci["estimate"],
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "level": level,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def holm(p_values: ArrayLike | Mapping[Any, float]) -> Any:
    """Holm's step-down adjusted p-values: the ``i``-th smallest of ``m`` becomes the running maximum of
    ``(m - j) p_(j)``, ``j <= i`` (0-based), capped at 1. Same order and kind as the input (sequence ->
    array, Series -> Series, mapping -> dict); NaN stays NaN and does not count in ``m``."""
    if isinstance(p_values, pd.Series):
        return pd.Series(holm(p_values.to_numpy(dtype=np.float64)), index=p_values.index, name=p_values.name)
    if isinstance(p_values, Mapping):
        keys = list(p_values)
        return dict(zip(keys, holm([p_values[k] for k in keys]).tolist()))
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    m = len(order)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return adjusted


def tost_relative(
    a: ArrayLike, b: ArrayLike, margin: float = 0.10, *, alpha: float = 0.05, test: str = "t"
) -> dict[str, Any]:
    """Two one-sided paired tests of the equivalence of the means of ``b`` and the reference ``a``
    within ``+-margin`` relative to ``mean(a)``.

    Null hypotheses ``mean(b) <= (1 - margin) mean(a)`` and ``mean(b) >= (1 + margin) mean(a)``, tested
    on the paired differences ``b - (1 - margin) a`` (alternative: greater than 0) and ``b - (1 + margin) a``
    (less than 0) with a one-sample t-test (``test="t"``) or the Wilcoxon signed-rank test
    (``test="wilcoxon"``). ``p_value`` is the larger one-sided p-value; ``equivalent`` when it is below
    ``alpha``. Pairing and dropping as in :func:`paired_comparison`.
    """
    if test not in ("t", "wilcoxon"):
        raise ValueError(f"test must be 't' or 'wilcoxon', got {test!r}")
    x, y = _paired(a, b)

    def one_sided(d: np.ndarray, alternative: str) -> float:
        if test == "t":
            return float(sps.ttest_1samp(d, 0.0, alternative=alternative).pvalue)
        return float(sps.wilcoxon(d, alternative=alternative).pvalue)

    p_lower = one_sided(y - (1.0 - margin) * x, "greater") if len(x) else np.nan
    p_upper = one_sided(y - (1.0 + margin) * x, "less") if len(x) else np.nan
    p_value = max(p_lower, p_upper) if len(x) else np.nan
    return {
        "n": len(x),
        "margin": margin,
        "relative_difference": float(y.mean() / x.mean() - 1.0) if len(x) else np.nan,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_value": p_value,
        "equivalent": bool(p_value < alpha),
        "test": test,
    }


# ----------------------------------------------------------------------------------- power (M8, D122)


def pooled_sd(values: ArrayLike, groups: ArrayLike) -> float:
    """Standard deviation within the groups, pooled: ``sqrt(sum (x - mean_group)^2 / sum (n_group - 1))`` over the
    finite values (groups with one value add nothing); NaN without two values in a group."""
    x, g = np.asarray(values, dtype=np.float64).reshape(-1), np.asarray(groups).reshape(-1)
    keep = np.isfinite(x)
    x, g = x[keep], g[keep]
    squares, dof = 0.0, 0
    for label in np.unique(g):
        part = x[g == label]
        squares += float(np.sum((part - part.mean()) ** 2))
        dof += len(part) - 1
    return math.sqrt(squares / dof) if dof > 0 else math.nan


def paired_t_power(effect: float, n_pairs: int, df: int, alpha: float = 0.05) -> float:
    """Power of the two-sided t-test of a mean difference of paired values: ``effect`` = true mean difference over the
    standard deviation of the differences, ``n_pairs`` differences, ``df`` degrees of freedom of the variance estimate
    (noncentral t). NaN for ``df < 1``; 1 for an infinite effect."""
    if df < 1 or n_pairs < 1 or math.isnan(effect):
        return math.nan
    if math.isinf(effect):
        return 1.0
    critical = sps.t.ppf(1.0 - alpha / 2.0, df)
    shift = abs(effect) * math.sqrt(n_pairs)
    return float(sps.nct.sf(critical, df, shift) + sps.nct.cdf(-critical, df, shift))


def seeds_needed(
    effect: float, strata: int = 1, alpha: float = 0.05, power: float = 0.8, n_max: int = 1_000_000
) -> int | None:
    """Smallest number ``n >= 2`` of pairs per stratum (e.g. seeds per scenario) for which the paired t-test over
    ``strata x n`` differences (``df = strata (n - 1)``: the stratum means removed) reaches ``power`` for the
    standardised ``effect``; None for a zero or undefined effect or beyond ``n_max``. The search starts at the
    sample size of the normal approximation, below which the t-test cannot reach the power."""
    if math.isnan(effect) or effect == 0.0 or strata < 1:
        return None
    if math.isinf(effect):
        return 2
    z = sps.norm.ppf(1.0 - alpha / 2.0) + sps.norm.ppf(power)
    n = max(2, int(math.floor((z / abs(effect)) ** 2 / strata)))
    while n <= n_max:
        if paired_t_power(effect, strata * n, strata * (n - 1), alpha) >= power:
            return n
        n += 1
    return None


def minimal_detectable(sd: float, n: int, strata: int = 1, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true mean difference that the paired t-test over ``strata x n`` differences with the standard
    deviation ``sd`` detects with ``power`` (bisection on the effect); NaN for an undefined ``sd`` or ``n < 2``."""
    if math.isnan(sd) or n < 2 or strata < 1:
        return math.nan
    if sd == 0.0:
        return 0.0

    def reached(effect: float) -> bool:
        return paired_t_power(effect, strata * n, strata * (n - 1), alpha) >= power

    low, high = 0.0, 1.0
    while not reached(high):
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if reached(mid) else (mid, high)
    return high * sd


# ------------------------------------------------------------------- stratified rank correlation (D122)


def stratified_ranks(values: ArrayLike, strata: ArrayLike) -> np.ndarray:
    """Ranks within every stratum scaled to (0, 1): ``(rank - 0.5) / n_stratum`` (average ranks for ties)."""
    x, s = np.asarray(values, dtype=np.float64).reshape(-1), np.asarray(strata).reshape(-1)
    out = np.empty(len(x))
    for label in np.unique(s):
        idx = np.flatnonzero(s == label)
        out[idx] = (sps.rankdata(x[idx]) - 0.5) / len(idx)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def stratified_spearman(x: ArrayLike, y: ArrayLike, strata: ArrayLike) -> float:
    """Spearman correlation pooled over strata: the ranks of ``x`` and of ``y`` within every stratum, scaled to (0, 1)
    (:func:`stratified_ranks`), then the Pearson correlation of the pooled ranks. With one stratum it is the Spearman
    correlation. Pairs with a non-finite value are left out; NaN with fewer than three pairs or a constant rank."""
    x, y, s = (np.asarray(a).reshape(-1) for a in (x, y, strata))
    x, y = x.astype(np.float64), y.astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, s = x[keep], y[keep], s[keep]
    return _pearson(stratified_ranks(x, s), stratified_ranks(y, s))


def stratified_spearman_ci(
    x: ArrayLike, y: ArrayLike, strata: ArrayLike, n_resamples: int = 1000, level: float = 0.95, seed: int = 0
) -> dict[str, Any]:
    """:func:`stratified_spearman` with its percentile bootstrap interval, resampling the units within every stratum
    (resamples with a constant variable have no correlation and are left out: ``n_valid``)."""
    x, y, s = (np.asarray(a).reshape(-1) for a in (x, y, strata))
    x, y = x.astype(np.float64), y.astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, s = x[keep], y[keep], s[keep]
    out = {"n": int(len(x)), "n_strata": int(len(np.unique(s))), "estimate": stratified_spearman(x, y, s),
           "low": math.nan, "high": math.nan, "n_valid": 0}  # fmt: skip
    if len(x) < 3:
        return out
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == label) for label in np.unique(s)]
    boot = []
    for _ in range(n_resamples):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        boot.append(stratified_spearman(x[idx], y[idx], s[idx]))
    valid = np.asarray(boot)[np.isfinite(boot)]
    out["n_valid"] = int(len(valid))
    if len(valid):
        alpha = 1.0 - level
        out["low"], out["high"] = (float(q) for q in np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0]))
    return out


def stratified_permutation_p(
    x: ArrayLike, y: ArrayLike, strata: ArrayLike, n_permutations: int = 10000, seed: int = 0
) -> float:
    """Two-sided permutation p-value of :func:`stratified_spearman`: ``x`` permuted within every stratum,
    ``(1 + #{|r_perm| >= |r|}) / (1 + n_permutations)``; NaN when the correlation is undefined."""
    x, y, s = (np.asarray(a).reshape(-1) for a in (x, y, strata))
    x, y = x.astype(np.float64), y.astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, s = x[keep], y[keep], s[keep]
    u, w = stratified_ranks(x, s), stratified_ranks(y, s)
    observed = _pearson(u, w)
    if math.isnan(observed) or n_permutations < 1:
        return math.nan
    rng = np.random.default_rng(seed)
    shuffled = np.tile(u, (n_permutations, 1))
    for label in np.unique(s):  # the within-stratum ranks of a permuted x are the permuted ranks
        idx = np.flatnonzero(s == label)
        shuffled[:, idx] = rng.permuted(np.tile(u[idx], (n_permutations, 1)), axis=1)
    r = (shuffled - u.mean()) @ (w - w.mean()) / (len(u) * u.std() * w.std())
    hits = int(np.sum(np.abs(r) >= abs(observed) - 1e-12))
    return (hits + 1) / (n_permutations + 1)
