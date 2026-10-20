"""Empirical leader-to-follower speed gain of human drivers against the gains of the laws (M11, D131, hypothesis H-N4).

    python scripts/analysis/empirical_gain.py            # runs/_tables/m11/empirical_gain{,_cells}.{csv,md} and the
                                                         # figure runs/_report/supplement/figures/empirical_gain.{png,pdf,txt}

Drivers: every event of the event set (``EventSet.from_parquet``) of at least ``--min-duration`` (40 s); per event the
H1 estimate of the transfer function from the leader's to the follower's speed by Welch's method
(``scipy.signal.csd`` and ``welch``: Hann segments of 40 s, 50 % overlap, linear detrending per segment, density
scaling), ``|G| = |P_xy| / P_xx`` at the Welch frequencies (multiples of 1/40 Hz) between 0.3 and 2 rad/s
(0.31-1.88 rad/s: 11 lines; lower frequencies are not estimable from events of this length: the lowest line with a
whole period inside a segment is 0.157 rad/s and a Hann window spreads it over its neighbours). The event's speed bin
is that of its mean follower speed (5 m/s wide: 0-5, 5-10, ...); per bin and frequency (a cell) the median and the
25-75 % range of the events' gains. Cells with fewer than ``--min-events`` events are left out.

Laws: the numerical gains of ``stability.json`` (the audit: 25 frequencies 0.02-2 rad/s at the grid speeds 5-30 m/s,
the equilibria the audit analysed, inside or outside the band; entries whose rollout was clipped, stopped or collided
are left out) of the E1 runs ``runs/e1/follownet_highd/<model>/driver_fold<k>_seed0``, folds 0-4, of the five
unconstrained networks (MLP, PIDL, GRU, LSTM, PERL), the IDM and k-NN; interpolated to every cell, the log gain
linearly in log frequency and in speed, at the median mean speed of the cell's events (no extrapolation: a cell below
5 m/s has no law value).

Statistic (D131): the excess log gain of a law = mean over the cells with both values of ``log |G_law| - log |G_emp,
median|`` (the law side: the mean over the folds of the log gain), with two cluster-bootstrap intervals (95 %
percentile): over the events (the empirical medians of resampled events, the law side fixed) and over the folds (the
folds resampled, the drivers fixed), and the two together (joint). Rule: confirmed when the excess is positive with an
interval excluding zero for at least four of the five networks and the IDM's excess is negative; refuted when fewer
than three networks exceed the drivers (excess > 0); otherwise open.

The FollowNet HighD events are 15-s windows (14.9 s, D19): none reaches 40 s, so the rule runs on NGSIM I-80 alone. As a
deviation, the HighD estimate with one Hann segment per event (14.9 s, frequencies of multiples of 1/14.9 Hz in
0.3-2 rad/s: 0.42, 0.84, 1.27, 1.69 rad/s) is reported beside it (``--highd-segment``), not under the rule.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from cf_stability.data.schema import DT, EventSet  # noqa: E402
from cf_stability.eval.tables import Column, Table, markdown  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

NETWORKS = ("mlp", "pidl", "gru", "lstm", "perl")
LAWS = (*NETWORKS, "idm", "knn")
FOLDS = (0, 1, 2, 3, 4)
OMEGA_RANGE = (0.3, 2.0)  # rad/s
BIN_WIDTH = 5.0  # m/s
FLAGS = ("clipped", "stopped", "collided")
SEED = 20261019
MEDIAN = "median of the events (D131)"
POOLED = "pooled H1 (sensitivity)"
COHERENT = "coherent cells only"  # the variants of both estimators on the cells with a pooled coherence >= MIN_COHERENCE
MIN_COHERENCE = 0.5
ESTIMATORS = (MEDIAN, POOLED, f"{MEDIAN}, {COHERENT}", f"{POOLED}, {COHERENT}")
Y_LIMITS = (0.05, 3.0)  # the gain axis of the figure


# ----------------------------------------------------------------------------------------------- drivers


def event_spectra(v_lead: np.ndarray, v: np.ndarray, nperseg: int, fs: float = 1.0 / DT, overlap: float = 0.5
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Angular frequencies (rad/s) and the Welch spectra ``P_xy`` (complex), ``P_xx``, ``P_yy`` of the leader's speed
    ``x`` and the follower's ``y``: Hann segments of ``nperseg`` samples with ``overlap``, linear detrending per
    segment, density scaling (``scipy.signal.csd``: ``P_xy`` = mean of ``conj(X) Y``)."""
    options = {"fs": fs, "window": "hann", "nperseg": nperseg, "noverlap": int(round(overlap * nperseg)),
               "detrend": "linear", "scaling": "density"}  # fmt: skip
    f, p_xy = signal.csd(v_lead, v, **options)
    _, p_xx = signal.welch(v_lead, **options)
    _, p_yy = signal.welch(v, **options)
    return 2.0 * np.pi * f, p_xy, p_xx, p_yy


def event_gain(v_lead: np.ndarray, v: np.ndarray, nperseg: int, fs: float = 1.0 / DT,
               overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Angular frequencies (rad/s) and ``|P_xy / P_xx|`` (H1) of the leader's speed ``x`` to the follower's ``y``
    (:func:`event_spectra`)."""
    omega, p_xy, p_xx, _ = event_spectra(v_lead, v, nperseg, fs, overlap)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.abs(p_xy) / p_xx
    return omega, gain


def driver_gains(events: EventSet, min_duration: float, segment_s: float, omega_range: Sequence[float] = OMEGA_RANGE,
                 spectra: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:  # fmt: skip
    """One row per event of at least ``min_duration`` (its id, follower, mean speed, speed bin, segments) and the gains
    ``[events, frequencies]`` at the Welch frequencies inside ``omega_range``; the frequencies (rad/s). ``spectra``
    (a dict) receives the events' ``p_xy``, ``p_xx``, ``p_yy`` at these frequencies for the pooled estimate."""
    nperseg = int(round(segment_s / DT))
    rows, gains, omega_kept = [], [], None
    parts: dict[str, list[np.ndarray]] = {"p_xy": [], "p_xx": [], "p_yy": []}
    for event in events:
        if event.duration < min_duration - 1e-9 or len(event) < nperseg:
            continue
        omega, p_xy, p_xx, p_yy = event_spectra(event.v_lead, event.v, nperseg)
        keep = (omega >= omega_range[0] - 1e-9) & (omega <= omega_range[1] + 1e-9)
        omega_kept = omega[keep]
        mean_speed = float(np.mean(event.v))
        rows.append({"event_id": event.event_id, "follower_id": event.follower_id, "duration": event.duration,
                     "mean_speed": mean_speed, "bin": int(math.floor(mean_speed / BIN_WIDTH)),
                     "segments": int((len(event) - nperseg) // (nperseg - int(round(0.5 * nperseg))) + 1)})  # fmt: skip
        with np.errstate(invalid="ignore", divide="ignore"):
            gains.append(np.abs(p_xy[keep]) / p_xx[keep])
        for key, value in (("p_xy", p_xy), ("p_xx", p_xx), ("p_yy", p_yy)):
            parts[key].append(value[keep])
    frame = pd.DataFrame(rows)
    if spectra is not None and gains:
        spectra.update({key: np.vstack(value) for key, value in parts.items()})
    return frame, (np.vstack(gains) if gains else np.zeros((0, 0))), (omega_kept if omega_kept is not None else np.zeros(0))


def cell_pooled(bins: np.ndarray, spectra: Mapping[str, np.ndarray], weights: np.ndarray, bin_ids: Sequence[int]
                ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The pooled H1 estimate per bin ``|sum w P_xy| / sum w P_xx`` (the cross-spectra of all segments of the bin's
    events averaged before the ratio, ``w`` = the event's segments: the follower's fluctuations that the leader does
    not cause average out) and the coherence ``|sum w P_xy|^2 / (sum w P_xx sum w P_yy)``, each ``[bins, freqs]``."""
    n_f = spectra["p_xx"].shape[1]
    gain, coherence = np.full((len(bin_ids), n_f), np.nan), np.full((len(bin_ids), n_f), np.nan)
    for k, b in enumerate(bin_ids):
        mine = bins == b
        if mine.any():
            w = weights[mine][:, None]
            xy = (w * spectra["p_xy"][mine]).sum(axis=0)
            xx, yy = (w * spectra["p_xx"][mine]).sum(axis=0), (w * spectra["p_yy"][mine]).sum(axis=0)
            gain[k], coherence[k] = np.abs(xy) / xx, np.abs(xy) ** 2 / (xx * yy)
    return gain, coherence


def cell_medians(bins: np.ndarray, gains: np.ndarray, bin_ids: Sequence[int]) -> np.ndarray:
    """``[bins, frequencies]`` median of the events' gains per bin (NaN for a bin without events)."""
    out = np.full((len(bin_ids), gains.shape[1]), np.nan)
    for k, b in enumerate(bin_ids):
        mine = gains[bins == b]
        if len(mine):
            out[k] = np.nanmedian(mine, axis=0)
    return out


# -------------------------------------------------------------------------------------------------- laws


def audit_gain_grid(audit: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(speeds [m], omega [n], gain [m, n])`` of a stored audit: the numerical gains of the equilibria it analysed,
    NaN where a rollout was clipped, stopped or collided and at speeds without equilibrium."""
    omega = np.asarray(audit["omega"], dtype=np.float64)
    speeds, rows = [], []
    for record in audit["equilibria"]:
        speeds.append(float(record["v"]))
        if record.get("status") in ("ok", "multiple", "outside") and record.get("gain"):
            g = np.asarray([np.nan if x is None else x for x in record["gain"]], dtype=np.float64)
            for flag in FLAGS:
                if record.get(flag):
                    g = np.where(np.asarray(record[flag], dtype=bool), np.nan, g)
            rows.append(g)
        else:
            rows.append(np.full(len(omega), np.nan))
    return np.asarray(speeds), omega, np.vstack(rows)


def interpolate_log_gain(speeds: np.ndarray, omega: np.ndarray, gain: np.ndarray, v: float,
                         targets: np.ndarray) -> np.ndarray:  # fmt: skip
    """``log |G|`` at the speed ``v`` and the frequencies ``targets``: linear in log frequency at the two grid speeds
    around ``v``, then linear in speed; NaN outside the grid or where a needed value is missing."""
    order = np.argsort(speeds)
    speeds, gain = speeds[order], gain[order]
    if not speeds[0] <= v <= speeds[-1]:
        return np.full(len(targets), np.nan)
    k = int(np.clip(np.searchsorted(speeds, v, side="right") - 1, 0, len(speeds) - 2))
    w = (v - speeds[k]) / (speeds[k + 1] - speeds[k])
    log_omega, log_targets = np.log(omega), np.log(targets)
    out = []
    for row in (k, k + 1):
        g = gain[row]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.interp(log_targets, log_omega, np.log(g))
        # a target between two frequencies needs both of them finite
        idx = np.clip(np.searchsorted(log_omega, log_targets, side="right") - 1, 0, len(omega) - 2)
        ok = np.isfinite(g[idx]) & np.isfinite(g[idx + 1]) & (g[idx] > 0) & (g[idx + 1] > 0)
        out.append(np.where(ok, values, np.nan))
    if w == 0.0:
        return out[0]
    if w == 1.0:
        return out[1]
    return (1.0 - w) * out[0] + w * out[1]


def law_log_gains(runs_root: Path, law: str, cell_speeds: np.ndarray, targets: np.ndarray, notes: list[str]
                  ) -> np.ndarray:  # fmt: skip
    """``[folds, bins, frequencies]`` log gains of a law of E1 at the cells."""
    out = np.full((len(FOLDS), len(cell_speeds), len(targets)), np.nan)
    for f, fold in enumerate(FOLDS):
        path = runs_root / "e1" / "follownet_highd" / law / f"driver_fold{fold}_seed0" / "stability.json"
        if not path.is_file():
            notes.append(f"{path.relative_to(runs_root.parent)} missing")
            continue
        speeds, omega, gain = audit_gain_grid(read_json(path)["audit"])
        for b, v in enumerate(cell_speeds):
            if np.isfinite(v):
                out[f, b] = interpolate_log_gain(speeds, omega, gain, float(v), targets)
    return out


# ------------------------------------------------------------------------------------------- statistics


def excess(law_log: np.ndarray, emp_median: np.ndarray, valid: np.ndarray) -> float:
    """Mean over the valid cells of the law's log gain (``[bins, freqs]``) minus the log of the empirical median."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = law_log - np.log(emp_median)
    keep = valid & np.isfinite(d)
    return float(d[keep].mean()) if keep.any() else math.nan


def resampled_medians(bins: np.ndarray, gains: np.ndarray, bin_ids: Sequence[int], n_resamples: int,
                      rng: np.random.Generator) -> list[np.ndarray]:  # fmt: skip
    """The drivers' cell medians of ``n_resamples`` bootstrap samples of the events (one set for every law)."""
    draws = rng.integers(0, len(bins), size=(n_resamples, len(bins)))
    return [cell_medians(bins[d], gains[d], bin_ids) for d in draws]


def bootstrap_excess(
    law_folds: np.ndarray, median: np.ndarray, medians_r: Sequence[np.ndarray], fold_draws: np.ndarray,
    valid: np.ndarray, level: float = 0.95,
) -> dict[str, float]:  # fmt: skip
    """Point estimate and the three percentile intervals of the excess log gain of one law (``law_folds [folds, bins,
    freqs]``): the events resampled (``medians_r``, the law side fixed), the folds resampled (``fold_draws [R, folds]``,
    the drivers fixed), both together (joint, resample r of each)."""
    alpha = 1.0 - level
    with warnings.catch_warnings():  # a cell without any fold value stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        fold_mean = np.nanmean(law_folds, axis=0)
        laws_r = [np.nanmean(law_folds[draw], axis=0) for draw in fold_draws]
    out = {"excess": excess(fold_mean, median, valid)}
    by_events, by_folds, joint = [], [], []
    for med_r, law_r in zip(medians_r, laws_r):
        by_events.append(excess(fold_mean, med_r, valid))
        by_folds.append(excess(law_r, median, valid))
        joint.append(excess(law_r, med_r, valid))
    for name, values in (("events", by_events), ("folds", by_folds), ("joint", joint)):
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        out[f"{name}_low"], out[f"{name}_high"] = ((float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
                                                   if len(v) else (math.nan, math.nan))  # fmt: skip
    return out


def rule(rows: Sequence[Mapping[str, Any]], interval: str) -> tuple[str, int, int]:
    """The branch of the rule of D131 with the intervals ``interval`` (events, folds, joint): ``(branch, networks
    with an excess > 0 whose interval excludes zero, networks with an excess > 0)``."""
    nets = [r for r in rows if r["law"] in NETWORKS]
    idm = next((r for r in rows if r["law"] == "idm"), None)
    above = sum(1 for r in nets if np.isfinite(r["excess"]) and r["excess"] > 0 and r[f"{interval}_low"] > 0)
    exceed = sum(1 for r in nets if np.isfinite(r["excess"]) and r["excess"] > 0)
    idm_negative = idm is not None and np.isfinite(idm["excess"]) and idm["excess"] < 0
    if above >= 4 and idm_negative:
        return "confirmed", above, exceed
    if exceed < 3:
        return "refuted", above, exceed
    return "open", above, exceed


# --------------------------------------------------------------------------------------------- analysis


def analyse(name: str, events: EventSet, runs_root: Path, min_duration: float, segment_s: float, min_events: int,
            n_resamples: int, seed: int, notes: list[str]) -> dict[str, Any]:  # fmt: skip
    """Drivers' cells, laws at the cells and the excess per law for one event set."""
    spectra: dict[str, np.ndarray] = {}
    frame, gains, omega = driver_gains(events, min_duration, segment_s, spectra=spectra)
    if not len(frame):
        notes.append(f"{name}: no event of at least {min_duration:g} s: no cells")
        return {"name": name, "events": frame, "cells": pd.DataFrame(), "rows": [], "omega": omega, "curves": {}}
    bins = frame["bin"].to_numpy()
    counts = frame.groupby("bin").size()
    bin_ids = sorted(int(b) for b, n in counts.items() if n >= min_events)
    dropped = sorted(int(b) for b, n in counts.items() if n < min_events)
    if dropped:
        notes.append(f"{name}: speed bins with fewer than {min_events} events left out: "
                     + ", ".join(f"{BIN_WIDTH * b:g}-{BIN_WIDTH * (b + 1):g} m/s ({counts[b]})" for b in dropped))  # fmt: skip
    cell_speeds = np.array([float(frame.loc[frame["bin"] == b, "mean_speed"].median()) for b in bin_ids])
    median = cell_medians(bins, gains, bin_ids)
    weights = frame["segments"].to_numpy(np.float64)
    pooled, coherence = cell_pooled(bins, spectra, weights, bin_ids)
    q25 = np.vstack([np.nanquantile(gains[bins == b], 0.25, axis=0) for b in bin_ids])
    q75 = np.vstack([np.nanquantile(gains[bins == b], 0.75, axis=0) for b in bin_ids])
    rng = np.random.default_rng(seed)
    rows, curves = [], {}
    cells = []
    laws_log = {law: law_log_gains(runs_root, law, cell_speeds, omega, notes) for law in LAWS}
    for k, b in enumerate(bin_ids):
        for j, w in enumerate(omega):
            cell = {"data": name, "bin": f"{BIN_WIDTH * b:g}-{BIN_WIDTH * (b + 1):g}", "speed": cell_speeds[k],
                    "omega": w, "events": int((bins == b).sum()), "median": median[k, j], "q25": q25[k, j],
                    "q75": q75[k, j], "pooled": pooled[k, j], "coherence": coherence[k, j]}  # fmt: skip
            for law in LAWS:
                cell[f"gain_{law}"] = float(np.exp(np.nanmean(laws_log[law][:, k, j]))) \
                    if np.isfinite(laws_log[law][:, k, j]).any() else math.nan  # fmt: skip
            cells.append(cell)
    draws = rng.integers(0, len(bins), size=(n_resamples, len(bins)))  # the same event resamples for every law
    medians_r = [cell_medians(bins[d], gains[d], bin_ids) for d in draws]
    pooled_r = [cell_pooled(bins[d], {key: value[d] for key, value in spectra.items()}, weights[d], bin_ids)[0]
                for d in draws]  # fmt: skip
    fold_draws = rng.integers(0, len(FOLDS), size=(n_resamples, len(FOLDS)))
    coherent = np.nan_to_num(coherence, nan=0.0) >= MIN_COHERENCE
    variants = ((MEDIAN, median, medians_r, None), (POOLED, pooled, pooled_r, None),
                (ESTIMATORS[2], median, medians_r, coherent), (ESTIMATORS[3], pooled, pooled_r, coherent))  # fmt: skip
    for estimator, empirical, resampled, subset in variants:
        for law in LAWS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                valid = np.isfinite(np.nanmean(laws_log[law], axis=0)) & np.isfinite(empirical)
            if subset is not None:
                valid = valid & subset
            stats = bootstrap_excess(laws_log[law], empirical, resampled, fold_draws, valid)
            rows.append({"data": name, "estimator": estimator, "law": law, "cells": int(valid.sum()),
                         "events": int(np.isin(bins, bin_ids).sum()),
                         "folds": int(np.isfinite(laws_log[law]).any(axis=(1, 2)).sum()), **stats})  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        law_curves = {law: np.exp(np.nanmean(laws_log[law], axis=0)) for law in LAWS}
    curves = {"bins": bin_ids, "speeds": cell_speeds, "median": median, "q25": q25, "q75": q75, "pooled": pooled,
              "laws": law_curves, "counts": counts}  # fmt: skip
    return {"name": name, "events": frame, "cells": pd.DataFrame(cells), "rows": rows, "omega": omega, "curves": curves}


def figure(results: Sequence[Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """Gain curves per speed bin: the drivers' median with the 25-75 % range (and the pooled H1 estimate) against the
    laws (mean over the folds of the log gain); one row of panels per event set. The gain axis is cut at
    ``Y_LIMITS`` (the pooled estimate falls below it where the coherence vanishes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from cf_stability.eval import figures

    figures.apply_style()
    usable = [r for r in results if r["curves"]]
    n_cols = max(len(r["curves"]["bins"]) for r in usable)
    fig, axes = plt.subplots(len(usable), n_cols, figsize=(figures.FULL_WIDTH, 1.75 * len(usable) + 0.7),
                             squeeze=False, sharey=True, sharex=True)  # fmt: skip
    rows = {0: "NGSIM I-80", 1: "HighD (15-s events)"}
    for row, result in enumerate(usable):
        c, omega = result["curves"], result["omega"]
        for k, b in enumerate(c["bins"]):
            ax = axes[row, k]
            ax.fill_between(omega, c["q25"][k], c["q75"][k], color=figures.PALETTE["grey"], alpha=0.35, lw=0)
            ax.plot(omega, c["median"][k], color=figures.PALETTE["black"], lw=2.0)
            ax.plot(omega, c["pooled"][k], color=figures.PALETTE["black"], lw=1.0, ls=":")
            for law in LAWS:
                g = c["laws"][law][k]
                if np.isfinite(g).any():
                    ax.plot(omega, g, color=figures.color(law), lw=1.0, ls="--" if law in ("idm", "knn") else "-")
            ax.axhline(1.0, color=figures.PALETTE["dark"], lw=0.5, ls="-", alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(*Y_LIMITS)
            label = f"{BIN_WIDTH * b:g}-{BIN_WIDTH * (b + 1):g} m/s" + "\n" + f"{int(c['counts'][b])} events"
            ax.text(0.03, 0.04, label, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=plt.rcParams["legend.fontsize"])  # fmt: skip
        for k in range(len(c["bins"]), n_cols):
            axes[row, k].set_visible(False)
        axes[row, 0].set_ylabel(rows.get(row, result["name"]) + "\n|G|")
    for ax in axes.flat:
        if ax.get_visible():
            figures._plain_log(ax.yaxis, *Y_LIMITS)
            figures._plain_log(ax.xaxis, 0.3, 2.0, max_ticks=3)
            ax.tick_params(labelbottom=True)
    for ax in axes[-1, :]:
        ax.set_xlabel("ω (rad/s)")
    handles = [Line2D([], [], color=figures.PALETTE["black"], lw=2, label="drivers, median"),
               Patch(color=figures.PALETTE["grey"], alpha=0.35, label="drivers, 25-75 %"),
               Line2D([], [], color=figures.PALETTE["black"], lw=1.0, ls=":", label="drivers, pooled H1")]  # fmt: skip
    handles += [Line2D([], [], color=figures.color(law), lw=1.0, ls="--" if law in ("idm", "knn") else "-",
                       label=figures.name(law)) for law in LAWS]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    paths = figures.save(fig, out_dir, "empirical_gain")
    return paths


def caption(results: Sequence[Mapping[str, Any]], min_events: int) -> str:
    parts = []
    for r in results:
        if r["curves"]:
            parts.append(f"{r['name']}: {int(r['curves']['counts'][r['curves']['bins']].sum())} events")
    return (
        "Empirical leader-to-follower speed gain of the drivers against the laws (D131). Drivers: per event the H1 "
        "estimate |P_xy| / P_xx of Welch's method (Hann segments, 50 % overlap, linear detrending); median (black) and "
        "25-75 % range (grey) over the events of a speed bin (the event's mean follower speed; bins with at least "
        f"{min_events} events, their number in the panel), at the Welch lines in 0.3-2 rad/s; dotted: the pooled H1 "
        "estimate of the bin (the spectra of all its segments averaged before the ratio; a sensitivity, not the "
        "statistic of D131). Laws: the numerical gains of the audits of the E1 runs (follownet_highd, folds 0-4, seed "
        "0), the log gain interpolated to the cell (the median speed of its events) and averaged over the folds; "
        "dashed: IDM and k-NN. NGSIM I-80: events of at least 40 s, 40-s segments; HighD (a deviation of D131): its "
        "15-s events, one segment each. " + "; ".join(parts) + "."
    )


def write(table: Table, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.frame.to_csv(out / f"{table.name}.csv", index=False)
    (out / f"{table.name}.md").write_text(markdown(table), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events-root", default="data/events")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--figures", default="runs/_report/supplement/figures")
    parser.add_argument("--min-duration", type=float, default=40.0)
    parser.add_argument("--segment", type=float, default=40.0)
    parser.add_argument("--highd-segment", type=float, default=14.9, help="the deviation for the 15-s HighD events")
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    events_root, runs_root = resolve_path(args.events_root), resolve_path(args.runs_root)
    notes: list[str] = []
    results = []
    highd = EventSet.from_parquet(events_root / "follownet_highd")
    i80 = EventSet.from_parquet(events_root / "ngsim_i80")
    longest = max((e.duration for e in highd), default=0.0)
    notes.append(f"follownet_highd: {len(highd)} events, the longest {longest:.1f} s: none of at least "
                 f"{args.min_duration:g} s, the rule runs on ngsim_i80 alone.")  # fmt: skip
    results.append(analyse("NGSIM I-80", i80, runs_root, args.min_duration, args.segment, args.min_events,
                           args.resamples, args.seed, notes))  # fmt: skip
    results.append(analyse("HighD (deviation: 14.9-s segments)", highd, runs_root, args.highd_segment,
                           args.highd_segment, args.min_events, args.resamples, args.seed + 1, notes))  # fmt: skip
    rows = []
    branches = []
    for result in results:
        for estimator in ESTIMATORS:
            mine = [r for r in result["rows"] if r["estimator"] == estimator]
            for interval in ("events", "folds", "joint"):
                branch, above, exceed = rule(mine, interval) if mine else ("n/a", 0, 0)
                branches.append({"data": result["name"], "estimator": estimator, "interval": interval,
                                 "branch": branch, "networks_above": above, "networks_exceeding": exceed})  # fmt: skip
        rows += result["rows"]
    frame = pd.DataFrame(rows)
    for interval in ("events", "folds", "joint"):
        frame[f"{interval}_excludes_zero"] = (frame[f"{interval}_low"] > 0) | (frame[f"{interval}_high"] < 0)
    branch_frame = pd.DataFrame(branches)
    out = resolve_path(args.out)
    branch_text = "; ".join(f"{b['data']}, {b['estimator']}, intervals over {b['interval']}: {b['branch']} "
                            f"({b['networks_above']} of 5 networks above the drivers with the interval excluding 0, "
                            f"{b['networks_exceeding']} with an excess > 0)" for b in branches)  # fmt: skip
    i80_events = results[0]["events"]
    write(Table("empirical_gain", "Empirical gain of the drivers against the laws (D131, H-N4)", [
        f"Drivers: events of at least {args.min_duration:g} s of the event set (NGSIM I-80: {len(i80_events)} of "
        f"{len(i80)}), per event the H1 estimate |P_xy| / P_xx of the leader-to-follower speed transfer by Welch's "
        f"method (Hann segments of {args.segment:g} s, 50 % overlap, linear detrending per segment), at the Welch "
        "lines in 0.3-2 rad/s; cells: 5-m/s bins of the event's mean follower speed x frequency, the median over the "
        f"events of a cell (bins with at least {args.min_events} events).",
        "Laws: the numerical gains of the audits (stability.json) of the E1 runs on follownet_highd (folds 0-4, seed "
        "0; entries of clipped, stopped or collided rollouts left out), the log gain interpolated linearly in log "
        "frequency and in speed to the cell (the median mean speed of its events), the mean over the folds.",
        "excess: mean over the cells with both values of log|G_law| - log|G_drivers, median| (positive: the law "
        f"amplifies more than the median driver). Intervals: 95 % percentile, {args.resamples} resamples each, of a "
        "bootstrap over the events (the drivers' medians recomputed, the law side fixed), over the folds (the drivers "
        f"fixed) and of both together (joint); numpy default_rng({args.seed}).",
        "Rule of D131: confirmed when the excess is positive with an interval excluding zero for at least four of the "
        "five networks (MLP, PIDL, GRU, LSTM, PERL) and the IDM's excess is negative; refuted when fewer than three "
        "networks exceed the drivers (excess > 0); otherwise open. " + branch_text + ".",
        "HighD: its FollowNet events are 15-s windows (14.9 s), none reaches 40 s: the rows of HighD use one Hann "
        "segment of the whole event (14.9 s; lines 0.42, 0.84, 1.27, 1.69 rad/s), a deviation of D131, not under the "
        "rule.",
        "estimator: median of the events (D131's statistic: the median over the events of a cell of the per-event H1 "
        "gain) and, as a sensitivity, the pooled H1 of the cell (|sum P_xy| / sum P_xx over all segments of its "
        "events, weighted by the segments of an event; with one or two segments per event the per-event ratio is "
        "nearly |Y| / |X| and the follower's fluctuations that the leader does not cause raise it, the pooled ratio "
        "averages them out, but the phases of different drivers partly cancel in the pooled cross-spectrum, which lowers "
        "it; empirical_gain_cells has the coherence of the pooled spectra); both estimators also on the coherent cells "
        f"only (pooled coherence >= {MIN_COHERENCE:g}).",
        *notes,
    ], frame, [
        Column("data", "drivers", "text"), Column("estimator", "estimator", "text"), Column("law", "law", "text"),
        Column("cells", "cells", "int"),
        Column("events", "events", "int"), Column("folds", "folds", "int"), Column("excess", "excess log gain", digits=3),
        Column("events_low", "events low", digits=3), Column("events_high", "events high", digits=3),
        Column("folds_low", "folds low", digits=3), Column("folds_high", "folds high", digits=3),
        Column("joint_low", "joint low", digits=3), Column("joint_high", "joint high", digits=3),
    ]), out)  # fmt: skip
    cells = pd.concat([r["cells"] for r in results if len(r["cells"])], ignore_index=True)
    write(Table("empirical_gain_cells", "Empirical gain per cell (D131)", [
        "Per event set, speed bin and Welch frequency: the events of the bin, the median speed of their means (the "
        "speed of the cell), the drivers' median gain with its 25-75 % range, and every law's gain at the cell (exp of "
        "the mean over the folds of the interpolated log gain).",
    ], cells, [
        Column("data", "drivers", "text"), Column("bin", "bin (m/s)", "text"), Column("speed", "speed", digits=1),
        Column("omega", "ω (rad/s)", digits=3), Column("events", "events", "int"), Column("median", "median", digits=3),
        Column("q25", "25 %", digits=3), Column("q75", "75 %", digits=3), Column("pooled", "pooled H1", digits=3),
        Column("coherence", "coherence", digits=3), *(Column(f"gain_{law}", law, digits=3) for law in LAWS),
    ]), out)  # fmt: skip
    branch_frame.to_csv(out / "empirical_gain_rule.csv", index=False)
    figure_dir = resolve_path(args.figures)
    paths = figure(results, figure_dir)
    (figure_dir / "empirical_gain.txt").write_text(caption(results, args.min_events) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['data'][:10]:<10} {r['estimator'][:6]:<6} {r['law']:<5} excess {r['excess']:+.3f}  events [{r['events_low']:+.3f}, "
              f"{r['events_high']:+.3f}]  folds [{r['folds_low']:+.3f}, {r['folds_high']:+.3f}]  joint "
              f"[{r['joint_low']:+.3f}, {r['joint_high']:+.3f}]  cells {r['cells']}")  # fmt: skip
    for b in branches:
        print(f"RULE {b['data']} {b['estimator']} ({b['interval']}): {b['branch']}  above {b['networks_above']}, exceeding "
              f"{b['networks_exceeding']}")  # fmt: skip
    print(f"empirical_gain: {time.perf_counter() - t_start:.0f} s -> {out / 'empirical_gain.md'}, "
          f"{', '.join(str(p) for p in paths)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
