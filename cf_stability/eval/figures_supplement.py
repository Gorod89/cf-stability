"""Supplementary figures and tables of M8 (docs/m8_contract.md, items 1, 2, 3 and 8; D115, D116, D123).

:func:`make_supplement` writes

* the tables ``lowfreq_expansion``, ``certificate_tightness`` and ``band_width`` to
  ``runs/_tables/m8/<name>.csv`` (full precision) and ``<name>.md`` (rounded, the definitions as notes);
* the figures ``lowfreq_expansion``, ``certificate_tightness``, ``band_width``, ``stability_map_<architecture>``,
  ``contours_<corridor>_p<k>`` and ``fd_<corridor>_p<k>`` to ``runs/_report/supplement/figures/<name>.png``
  (200 dpi) and ``<name>.pdf``, each with its caption in ``<name>.txt``;
* every note on a missing or excluded input to ``runs/_report/supplement/figures_supplement_notes.txt``.

Sources: the audits (``stability.json``, block ``audit``: per grid speed the status, the equilibrium, the band
edges, the partial derivatives of the memoryless view, the numerical gains and the rollout flags per frequency),
the certificates (``certificate.json``, block ``per_speed``), the event sets of ``data/events`` (read with
:meth:`EventSet.from_parquet`) and the corridor files of M5. The figures follow the style of
:mod:`cf_stability.eval.figures` (its palette, colours and helpers, no titles, labels with units). The corridor
panels are prepared exactly as the figures of M6 and the metrics of M5 prepare them; they are drawn on a grid of
``grid_columns`` columns because ``figures.speed_contours`` and ``figures.fundamental_diagrams`` fix two and three
columns, which would make a 20-panel figure 7 x 19 inches.

Nothing missing raises: a missing or excluded input becomes a note (printed, in the notes file and in the caption
of the figure or the notes of the table it concerns); a figure without any input is left out, a table without
rows is written with its header.
"""

from __future__ import annotations

import dataclasses
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # files only, as cf_stability.eval.figures

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cf_stability.eval import figures  # noqa: E402
from cf_stability.eval.figures import FULL_WIDTH, PALETTE, SHARES, WITH, WITHOUT  # noqa: E402
from cf_stability.eval.tables import Column, Table, experiment_name, write_table  # noqa: E402
from cf_stability.stability.analytic import transfer_continuous  # noqa: E402
from cf_stability.stability.equilibrium import V_GRID  # noqa: E402
from cf_stability.train.tensors import BandConfig, spacing_band  # noqa: E402
from cf_stability.utils import REPO_ROOT, read_json  # noqa: E402

OUTPUTS = ("lowfreq_expansion", "certificate_tightness", "band_width", "stability_maps", "corridor")
EQUILIBRIUM = ("ok", "multiple", "outside")  # statuses with an equilibrium (FOUND of cf_stability.stability.audit)
FLAGS = ("clipped", "stopped", "collided")  # rollout flags of the numerical response (cf_stability.stability.frequency)
PARTIALS = ("f_s", "f_dv", "f_v")
CERTIFICATE_KEYS = ("v", "a_priori_margin", "guaranteed_margin", "found", "s")  # per-speed entries used (D116)
ARMS = {"E1": "E1, no penalty", "E2": "E2, chosen weight"}
ARM_COLORS = {"E1": WITHOUT, "E2": WITH}
BOUNDS = {"a_priori": "a priori", "at_equilibria": "at the equilibria"}
STAGE_COLORS = {"before": PALETTE["blue"], "after": PALETTE["orange"]}
MAP_STYLE = {
    **{key: (colour, label) for key, colour, label in SHARES},
    "indifferent": (PALETTE["dark"], "indifferent law (nominal gap)"),
    "undefined": (PALETTE["black"], "flag undefined"),
}  # fmt: skip
RESIDUAL_LARGE = 0.1  # relative residual of the sinusoid fit above which a response is reported as not sinusoidal
NOTES_FILE = "figures_supplement_notes.txt"
CORRIDOR_NAMES = {"i80": "I-80", "us101": "US-101"}  # used when configs/corridor_metrics.yaml has no design.corridors


@dataclasses.dataclass(frozen=True)
class SupplementConfig:
    """Paths and choices of the supplement (defaults: docs/m8_contract.md; ``scripts/make_figures_supplement.py``)."""

    runs_root: Path
    out_dir: Path  # runs/_report/supplement: figures/ and the notes file
    tables_dir: Path  # runs/_tables/m8
    tables_m4: Path  # chosen_weights.json: the chosen E2 weight of every architecture (D85)
    tables_m5: Path  # laws.csv: the laws of every corridor
    corridor_root: Path
    events_root: Path
    configs_dir: Path = REPO_ROOT / "configs"  # corridor_metrics.yaml: corridor names and the macro settings
    outputs: tuple[str, ...] = OUTPUTS
    data: str = "follownet_highd"  # event set of E1 and E2
    fold: int = 0
    seed: int = 0
    expansion_architectures: tuple[str, ...] = ("idm", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl")
    e1_only: tuple[str, ...] = ("idm",)  # architectures without an E2 arm
    n_low: int = 3  # lowest audit frequencies of the expansion check
    certificate_model: str = "residual_idm"
    certificate_experiments: tuple[tuple[float, str, str], ...] = (  # r_max, before and after fine-tuning
        (0.1, "e4_stable_r0.1", "e4_stable_ft_r0.1"), (0.2, "e4_stable_r0.2", "e4_stable_ft_r0.2"),
        (0.3, "e4_stable", "e4_stable_ft"), (0.5, "e4_stable_r0.5", "e4_stable_ft_r0.5"),
    )  # fmt: skip
    certificate_data: tuple[str, str] = ("follownet_highd", "ngsim_i80")  # event sets before / after fine-tuning
    certificate_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    certificate_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    band_sets: tuple[str, ...] = ("follownet_highd", "ngsim_i80")
    min_driver_samples: int = 20  # near-steady samples of a driver at a speed (D123)
    few_drivers: int = 10  # band_width figure: open markers where fewer drivers qualify
    map_architectures: tuple[str, ...] = ("mlp", "pidl", "residual_idm", "gru", "lstm", "perl")
    corridors: tuple[str, ...] = ("i80", "us101")  # scenario prefixes
    periods: tuple[int, ...] = (0, 1, 2)
    corridor_seed: int = 0
    main_laws: tuple[str, ...] = ("idm_global", "residual_idm_certified", "mlp", "gru", "lstm")  # first after the truth
    max_panels: int = 20  # ground truth included
    grid_columns: int = 4
    dpi: int = 200
    font_size: float = 9.0

    @classmethod
    def under(cls, runs_root: Path, **changes: Any) -> SupplementConfig:
        """The layout of the repository below ``runs_root`` (events and configs of the repository)."""
        runs_root = Path(runs_root)
        paths = {
            "out_dir": runs_root / "_report" / "supplement", "tables_dir": runs_root / "_tables" / "m8",
            "tables_m4": runs_root / "_tables" / "m4", "tables_m5": runs_root / "_tables" / "m5",
            "corridor_root": runs_root / "corridor", "events_root": REPO_ROOT / "data" / "events",
        }  # fmt: skip
        return cls(runs_root=runs_root, **{**paths, **changes})


# ----------------------------------------------------------------------------------------------- helpers


def _float(value: Any) -> float:
    """Finite float, else NaN."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def chosen_experiment(chosen: Mapping[str, Any], arch: str, prefix: str = "e2") -> str | None:
    """Experiment of the chosen weight of ``arch`` (``chosen_weights.json``), as the report of M6 names it."""
    entry = chosen.get(arch) or {}
    weight = entry.get("override", entry.get("weight"))
    if weight is None or not entry.get("kind"):
        return None
    return experiment_name(prefix, entry["kind"], float(weight))


def law_color(law: str) -> str:
    """Colour of a law: that of figures.COLORS, else that of the longest law name of it that prefixes ``law``
    (``residual_idm_certified_r0.1`` -> ``residual_idm_certified``)."""
    if law in figures.COLORS:
        return figures.COLORS[law]
    prefixes = [key for key in figures.COLORS if law.startswith(key)]
    return figures.COLORS[max(prefixes, key=len)] if prefixes else PALETTE["grey"]


def _label(ax: Any, text: str, size: float | None = None) -> None:
    """Name of a panel above its upper left corner (figures._label, with a smaller font where panels are narrow)."""
    if size is None:
        figures._label(ax, text)
    else:
        ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=size)


def _grid_labels(fig: Figure, axes: np.ndarray, used: int, xlabel: str, ylabel: str) -> None:
    """Axis labels of a grid of panels with shared axes: one label per figure side, the x tick labels on the lowest
    used panel of every column (above an unused one as well)."""
    rows, cols = axes.shape
    for j in range(cols):
        filled = [i for i in range(rows) if i * cols + j < used]
        if filled:
            axes[filled[-1], j].xaxis.set_tick_params(labelbottom=True)
    fig.supxlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])


def _square(ax: Any, x: Sequence[float], y: Sequence[float], pad: float = 0.06) -> tuple[float, float]:
    """Equal limits of both axes around the finite values, square panel; returns the limits."""
    values = np.concatenate([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
    values = values[np.isfinite(values)]
    low, high = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
    span = max(high - low, 1e-3 * max(1.0, abs(high)))
    limits = (low - pad * span, high + pad * span)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_box_aspect(1)
    return limits


def _diagonal(ax: Any, limits: tuple[float, float]) -> None:
    ax.plot(limits, limits, color=PALETTE["dark"], lw=0.8, zorder=1)


# ------------------------------------------------------------------------- 1 low-frequency expansion (D115)


def expansion_gain(f_s: Any, f_dv: Any, f_v: Any, omega: Sequence[float]) -> np.ndarray:
    """``|G| = sqrt(max(0, 1 - omega^2 M / f_s^2))``, ``M = f_v^2 + 2 f_v f_dv - 2 f_s`` (D115): ``[n, n_omega]``."""
    f_s, f_dv, f_v = (np.asarray(x, dtype=np.float64).reshape(-1, 1) for x in (f_s, f_dv, f_v))
    margin = f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s
    with np.errstate(divide="ignore", invalid="ignore"):
        square = 1.0 - np.asarray(omega, dtype=np.float64) ** 2 * margin / f_s**2
    return np.sqrt(np.maximum(square, 0.0))


def exact_gain(f_s: Any, f_dv: Any, f_v: Any, omega: Sequence[float]) -> np.ndarray:
    """``|G(i omega)|`` of the continuous memoryless linearisation (``transfer_continuous``; equation (1) of
    docs/theory_notes.md): ``[n, n_omega]``."""
    tensors = [torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1)) for x in (f_s, f_dv, f_v)]
    return transfer_continuous(*tensors, torch.as_tensor(np.asarray(omega, dtype=np.float64))).abs().numpy()


def expansion_points(audit: Mapping[str, Any], n_low: int) -> tuple[pd.DataFrame, Counter]:
    """Per speed with an equilibrium (status ok, multiple or outside) and per frequency of the ``n_low`` lowest of
    an audit: the numerical gain, the expansion and (1) from the stored partials, the rollout flag of that
    frequency, the margin and the slow pole ``|f_s| / |f_dv + f_v|``. The counter holds the speeds left out, by
    reason; a missing field is named."""
    skipped: Counter = Counter()
    records = audit.get("equilibria") or []
    omega = [_float(w) for w in (audit.get("omega") or [])[:n_low]]
    if len(omega) < n_low or not all(np.isfinite(omega)):
        skipped[f"audit field 'omega' missing or with fewer than {n_low} frequencies"] += max(len(records), 1)
        return pd.DataFrame(), skipped
    if not records:
        skipped["audit field 'equilibria' missing or empty"] += 1
    keep = []
    for r in records:
        status = r.get("status")
        if status == "indifferent":
            skipped["indifferent law (every spacing an equilibrium, f_s = 0: no expansion)"] += 1
        if status not in EQUILIBRIUM:
            continue
        missing = [k for k in PARTIALS if not np.isfinite(_float(r.get(k)))]
        gains = r.get("gain") if isinstance(r.get("gain"), list) else []
        if len(gains) < n_low or not all(np.isfinite([_float(g) for g in gains[:n_low]])):
            missing.append("gain")
        missing += [k for k in FLAGS if not isinstance(r.get(k), list) or len(r[k]) < n_low]
        if missing:
            skipped[f"field {', '.join(missing)} missing"] += 1
        elif float(r["f_s"]) == 0.0:
            skipped["f_s = 0 (no expansion)"] += 1
        else:
            keep.append(r)
    if not keep:
        return pd.DataFrame(), skipped
    f_s, f_dv, f_v = (np.array([float(r[k]) for r in keep]) for k in PARTIALS)
    expansion, exact = expansion_gain(f_s, f_dv, f_v, omega), exact_gain(f_s, f_dv, f_v, omega)
    margin = f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s
    local = (f_s > 0.0) & (f_v + f_dv < 0.0)  # local stability of the memoryless view (criterion of analytic.py)
    with np.errstate(divide="ignore"):
        slow = np.abs(f_s) / np.abs(f_dv + f_v)
    rows = []
    for i, r in enumerate(keep):
        fit = r.get("residual") if isinstance(r.get("residual"), list) else []
        for k, w in enumerate(omega):
            rows.append({
                "v": float(r["v"]), "status": r["status"], "in_support": r.get("in_support"), "k": k, "omega": w,
                "f_s": f_s[i], "f_dv": f_dv[i], "f_v": f_v[i], "margin": margin[i], "slow_pole": slow[i],
                "local_stable": bool(local[i]), "fit_residual": _float(fit[k]) if k < len(fit) else math.nan,
                "flagged": any(bool(r[name][k]) for name in FLAGS), "gain_numerical": float(r["gain"][k]),
                "gain_expansion": float(expansion[i, k]), "gain_exact": float(exact[i, k]),
            })  # fmt: skip
    return pd.DataFrame(rows), skipped


def lowfreq_rows(points: pd.DataFrame) -> pd.DataFrame:
    """Per architecture, arm and frequency: counts, |numerical - expansion| (median, 90 % quantile), the share of
    speeds with the same sign of |G| - 1, the disagreements with -omega^2 < M < 0, |numerical - (1)| (median), the
    median slow pole, and the speeds without a linear stationary response (locally unstable in the memoryless view;
    fit residual above ``RESIDUAL_LARGE``); over the speeds whose rollout at that frequency is not flagged."""
    rows = []
    if not len(points):
        return pd.DataFrame(rows)
    for (arch, arm, _k), part in points.groupby(["architecture", "arm", "k"], sort=False):
        used = part[~part["flagged"].astype(bool)]
        omega = float(part["omega"].iloc[0])
        diff = (used["gain_numerical"] - used["gain_expansion"]).abs()
        exact = (used["gain_numerical"] - used["gain_exact"]).abs()
        agree = np.sign(used["gain_numerical"] - 1.0) == np.sign(used["gain_expansion"] - 1.0)
        blind = ~agree & (used["margin"] < 0.0) & (used["margin"] > -(omega**2))
        rows.append({
            "architecture": arch, "arm": arm, "experiment": part["experiment"].iloc[0], "omega": omega,
            "equilibria": len(part), "flagged": int(part["flagged"].astype(bool).sum()), "speeds": len(used),
            "abs_diff_median": diff.median() if len(used) else math.nan,
            "abs_diff_q90": diff.quantile(0.9) if len(used) else math.nan,
            "sign_agreement": float(agree.mean()) if len(used) else math.nan,
            "sign_disagree": int((~agree).sum()), "disagree_blind": int(blind.sum()),
            "abs_diff_exact_median": exact.median() if len(used) else math.nan,
            "slow_pole_median": used["slow_pole"].median() if len(used) else math.nan,
            "locally_unstable": int((~used["local_stable"].astype(bool)).sum()),
            "residual_large": int((used["fit_residual"] > RESIDUAL_LARGE).sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


LOWFREQ_COLUMNS = [  # "\|": a pipe inside a cell of a Markdown table
    Column("architecture", "architecture", "text"), Column("arm", "arm", "text"),
    Column("experiment", "experiment", "text"), Column("omega", "ω (rad/s)", "num", 4),
    Column("equilibria", "speeds with equilibrium", "int"), Column("flagged", "flagged (left out)", "int"),
    Column("speeds", "speeds", "int"), Column("abs_diff_median", r"\|num - exp\| median", "num", 4),
    Column("abs_diff_q90", r"\|num - exp\| q90", "num", 4),
    Column("sign_agreement", r"same sign of \|G\| - 1", "num", 3), Column("sign_disagree", "sign differs", "int"),
    Column("disagree_blind", "of which -ω² < M < 0", "int"),
    Column("abs_diff_exact_median", r"\|num - (1)\| median", "num", 4),
    Column("slow_pole_median", "slow pole median (rad/s)", "num", 4),
    Column("locally_unstable", "locally unstable", "int"), Column("residual_large", "fit residual > 0.1", "int"),
]  # fmt: skip


def lowfreq_expansion_figure(points: pd.DataFrame, architectures: Sequence[str]) -> Figure:
    """Numerical gain of the audit against the expansion at the lowest frequency, one panel per architecture, the
    points coloured by arm (open: not locally stable in the memoryless view), the diagonal and the lines |G| = 1."""
    first = points[(points["k"] == 0) & ~points["flagged"].astype(bool)] if len(points) else points
    present = [a for a in architectures if len(first) and (first["architecture"] == a).any()]
    if not present:
        raise ValueError("no audit with partial derivatives and gains")
    omega = float(first["omega"].iloc[0])
    cols = 4
    rows = math.ceil((len(present) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 1.95 * rows + 0.3), squeeze=False)
    for ax, arch in zip(axes.flat, present):
        part = first[first["architecture"] == arch]
        for arm, colour in ARM_COLORS.items():
            sel = part[part["arm"] == arm]
            stable = sel["local_stable"].astype(bool)
            ax.plot(sel.loc[stable, "gain_expansion"], sel.loc[stable, "gain_numerical"], "o", ms=3.2, color=colour,
                    alpha=0.75, mew=0, zorder=3)  # fmt: skip
            ax.plot(sel.loc[~stable, "gain_expansion"], sel.loc[~stable, "gain_numerical"], "o", ms=3.2, mfc="none",
                    mec=colour, mew=0.8, zorder=3)  # fmt: skip
        limits = _square(ax, part["gain_expansion"], part["gain_numerical"])
        _diagonal(ax, limits)
        for line in (ax.axhline, ax.axvline):
            line(1.0, color=PALETTE["grey"], ls=":", lw=0.8, zorder=0)
        figures._label(ax, figures.name(arch))
    fig.supxlabel(f"|G| of the expansion at {omega:.2g} rad/s", fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(f"numerical |G| at {omega:.2g} rad/s", fontsize=plt.rcParams["axes.labelsize"])
    legend_ax = axes.flat[len(present)]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", color=ARM_COLORS["E1"], label=ARMS["E1"]),
        Line2D([], [], ls="none", marker="o", color=ARM_COLORS["E2"], label=ARMS["E2"]),
        Line2D([], [], ls="none", marker="o", mfc="none", mec=PALETTE["dark"], label="locally unstable (open)"),
        Line2D([], [], color=PALETTE["dark"], lw=0.8, label="numerical = expansion"),
        Line2D([], [], color=PALETTE["grey"], ls=":", lw=0.8, label="|G| = 1"),
    ], loc="center")  # fmt: skip
    figures._hide_unused(axes, len(present) + 1)
    return fig


# ----------------------------------------------------------------------------- 2 certificate tightness (D116)


def computed_certificate_speeds(run: Path, speeds: Sequence[float], n_scan: int = 400) -> list[dict[str, Any]]:
    """Per-speed bounds of a ResidualIDM run computed from ``model.pt`` with the certificate code, as
    ``scripts/certificate.py`` computes them (a priori with the model's ``r_max``; at the equilibria anchored to the
    band of ``metrics.json``), for a ``certificate.json`` that does not store them."""
    from cf_stability.models import ResidualIDM, load_model
    from cf_stability.stability.certificate import certificate_a_priori, certificate_at_equilibria, to_json
    from cf_stability.stability.equilibrium import Band

    model = load_model(run / "model.pt")
    if not isinstance(model, ResidualIDM):
        raise ValueError(f"model.pt holds a {model.name} model, not a ResidualIDM")
    metrics = run / "metrics.json"
    context = read_json(metrics).get("context") if metrics.exists() else None
    prior = certificate_a_priori(model, speeds, n_scan, keep_r_max=True)
    post = certificate_at_equilibria(model, speeds, band=Band.from_context(context, model))
    return to_json([
        {"v": post["v"][k], "a_priori_margin": prior["guaranteed_margin"][k], "s": post["s"][k],
         "found": post["found"][k], "status": post["status"][k], "guaranteed_margin": post["guaranteed_margin"][k]}
        for k in range(len(post["v"]))
    ])  # fmt: skip


def certificate_points(per_speed: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> tuple[list[dict], Counter]:
    """Per grid speed of one run: the audited margin of the hybrid (``stability.json``, any equilibrium) against
    the a priori bound (every speed with an equilibrium) and against the bound at the equilibria (the anchored
    equilibria the certificate uses, at the spacing of the audit). The counter holds the speeds left out, by
    reason."""
    skipped: Counter = Counter()
    records = {round(float(r["v"]), 6): r for r in audit.get("equilibria") or [] if r.get("v") is not None}
    out = []
    for p in per_speed:
        r = records.get(round(_float(p.get("v")), 6))
        if r is None:
            skipped["speed of the certificate not in the audit"] += 1
            continue
        audited = _float(r.get("margin"))
        if r.get("status") not in EQUILIBRIUM or not np.isfinite(audited):
            continue  # no equilibrium at this speed: nothing to bound
        prior = _float(p.get("a_priori_margin"))
        if np.isfinite(prior):
            out.append({"v": float(r["v"]), "status": r["status"], "bound": "a_priori", "certified": prior,
                        "audited": audited})  # fmt: skip
        else:
            skipped["a priori bound not finite (no feasible spacing)"] += 1
        if not p.get("found"):
            continue  # the certificate at the equilibria uses the anchored equilibria only
        bound, s_cert, s_audit = _float(p.get("guaranteed_margin")), _float(p.get("s")), _float(r.get("s"))
        if not np.isfinite(bound):
            skipped["bound at the equilibria not finite"] += 1
        elif not abs(s_cert - s_audit) <= 1e-6 * max(1.0, abs(s_audit)):
            skipped["equilibrium of the certificate differs from that of the audit"] += 1
        else:
            out.append({"v": float(r["v"]), "status": r["status"], "bound": "at_equilibria", "certified": bound,
                        "audited": audited})  # fmt: skip
    return out, skipped


def certificate_rows(points: pd.DataFrame) -> pd.DataFrame:
    """Per r_max, stage and bound: runs, speeds, medians of the bound and of the margin, the slack (audited minus
    certified: median, 10 % quantile, minimum) and the negative slacks (count and share)."""
    rows = []
    if not len(points):
        return pd.DataFrame(rows)
    for (r_max, stage, bound), part in points.groupby(["r_max", "stage", "bound"], sort=False):
        slack = part["audited"] - part["certified"]
        rows.append({
            "r_max": r_max, "stage": stage, "bound": BOUNDS[bound], "experiment": part["experiment"].iloc[0],
            "data": part["data"].iloc[0], "runs": int(part["run"].nunique()), "speeds": len(part),
            "certified_median": part["certified"].median(), "audited_median": part["audited"].median(),
            "slack_median": slack.median(), "slack_q10": slack.quantile(0.1), "slack_min": slack.min(),
            "negative": int((slack < 0).sum()), "negative_share": float((slack < 0).mean()),
        })  # fmt: skip
    return pd.DataFrame(rows)


CERTIFICATE_COLUMNS = [
    Column("r_max", "r_max (m/s²)", "weight"), Column("stage", "stage", "text"), Column("bound", "bound", "text"),
    Column("experiment", "experiment", "text"), Column("data", "data", "text"), Column("runs", "runs", "int"),
    Column("speeds", "speeds", "int"), Column("certified_median", "bound median (s⁻²)", "num", 4),
    Column("audited_median", "audited M median (s⁻²)", "num", 4), Column("slack_median", "slack median", "num", 4),
    Column("slack_q10", "slack q10", "num", 4), Column("slack_min", "slack min", "num", 4),
    Column("negative", "negative slack", "int"), Column("negative_share", "share negative", "num", 3),
]  # fmt: skip


def certificate_tightness_figure(points: pd.DataFrame, r_values: Sequence[float]) -> Figure:
    """Audited margin against the certified lower bound, one panel per r_max (columns) and bound (rows), the
    points coloured by stage, the diagonal."""
    bounds = [b for b in BOUNDS if len(points) and (points["bound"] == b).any()]
    r_values = [r for r in r_values if len(points) and np.isclose(points["r_max"], r).any()]
    if not bounds or not r_values:
        raise ValueError("no certificate with per-speed bounds and no audit to compare")
    width = FULL_WIDTH / max(len(r_values), 2)  # square panels
    fig, axes = plt.subplots(len(bounds), len(r_values), figsize=(FULL_WIDTH, width * len(bounds) + 0.9),
                             squeeze=False, sharex=True, sharey=True)  # fmt: skip
    for i, bound in enumerate(bounds):
        for j, r_max in enumerate(r_values):
            ax = axes[i, j]
            part = points[(points["bound"] == bound) & np.isclose(points["r_max"], r_max)]
            for stage, colour in STAGE_COLORS.items():
                sel = part[part["stage"] == stage]
                ax.plot(sel["certified"], sel["audited"], "o", ms=2.0, color=colour, alpha=0.45, mew=0, zorder=3)
            _label(ax, f"r_max = {r_max:g} m/s²", size=8.0)
            ax.text(0.96, 0.04, f"{BOUNDS[bound]} bound" if bound == "a_priori" else f"bound {BOUNDS[bound]}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=PALETTE["dark"])  # fmt: skip
    limits = _square(axes[0, 0], points["certified"], points["audited"])
    for ax in axes.flat:
        ax.set_box_aspect(1)
        _diagonal(ax, limits)
    fig.supxlabel("certified lower bound of the margin (s⁻²)", fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel("audited margin M (s⁻²)", fontsize=plt.rcParams["axes.labelsize"])
    handles = [
        Line2D([], [], ls="none", marker="o", color=STAGE_COLORS["before"], label="before fine-tuning (HighD)"),
        Line2D([], [], ls="none", marker="o", color=STAGE_COLORS["after"], label="after fine-tuning (NGSIM I-80)"),
        Line2D([], [], color=PALETTE["dark"], lw=0.8, label="margin = bound"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside upper center", ncol=3)  # below the panels sits the x label
    return fig


# -------------------------------------------------------------------------- 3 band width (D123 part a)


def band_width_rows(events: Sequence[Any], data: str, cfg: BandConfig | None = None, min_driver_samples: int = 20,
                    speeds: Sequence[float] = V_GRID) -> pd.DataFrame:  # fmt: skip
    """Per grid speed: the spacing band of D78 (``spacing_band`` of the near-steady samples of all ``events``), its
    relative width ``(s_high - s_low) / s_median``, and the standard deviation (ddof 1) over the drivers
    (``follower_id``) with at least ``min_driver_samples`` near-steady samples at the speed of their median
    near-steady spacing, absolute and relative to ``s_median``. A speed without band (fewer than
    ``cfg.min_samples`` samples) has no band columns; one without two such drivers no dispersion."""
    cfg = cfg or BandConfig()
    if not events:
        return pd.DataFrame()
    s, v, v_lead, a = (np.concatenate([getattr(ev, name) for ev in events]) for name in ("s", "v", "v_lead", "a"))
    dv = v - v_lead  # as training_context passes it
    codes = {name: i for i, name in enumerate(dict.fromkeys(ev.follower_id for ev in events))}
    driver = np.concatenate([np.full(len(ev.s), codes[ev.follower_id]) for ev in events])
    band = spacing_band(s, dv, v, a, cfg, speeds)
    listed = {} if band is None else {float(x): k for k, x in enumerate(band["v"])}
    steady = (np.abs(dv) < cfg.dv_max) & (np.abs(a) < cfg.a_max)  # the near-steady samples of spacing_band
    rows = []
    for speed in speeds:
        near = steady & (np.abs(v - speed) < cfg.half_width)
        groups = pd.Series(s[near]).groupby(driver[near])
        counts = groups.size()
        medians = groups.median()[counts >= min_driver_samples]
        k = listed.get(float(speed))
        low, median, high = (math.nan,) * 3 if k is None else (band[key][k] for key in ("s_low", "s_median", "s_high"))
        std = float(medians.std(ddof=1)) if len(medians) >= 2 else math.nan
        rows.append({
            "data": data, "v": float(speed), "samples": int(near.sum()), "band": k is not None, "s_low": low,
            "s_median": median, "s_high": high, "relative_width": (high - low) / median if k is not None else math.nan,
            "drivers": int(len(medians)), "driver_std": std,
            "driver_std_relative": std / median if k is not None else math.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


BAND_COLUMNS = [
    Column("data", "data", "text"), Column("v", "v (m/s)", "num", 0), Column("samples", "near-steady samples", "int"),
    Column("s_low", "s 5 % (m)", "num", 2), Column("s_median", "s 50 % (m)", "num", 2),
    Column("s_high", "s 95 % (m)", "num", 2), Column("relative_width", "relative width", "num", 3),
    Column("drivers", "drivers (>= 20 samples)", "int"), Column("driver_std", "driver std (m)", "num", 2),
    Column("driver_std_relative", "driver std / s 50 %", "num", 3),
]  # fmt: skip


def band_width_figure(frame: pd.DataFrame, sets: Sequence[str], few_drivers: int) -> Figure:
    """Per event set: relative width of the band and the per-driver dispersion relative to the band median against
    the speed (open markers: fewer than ``few_drivers`` drivers)."""
    present = [d for d in sets if len(frame) and (frame["data"] == d).any()]
    if not present:
        raise ValueError("no event set")
    fig, axes = plt.subplots(1, len(present), figsize=(FULL_WIDTH, 2.6), squeeze=False, sharey=True)
    colour = PALETTE["vermillion"]
    for ax, data in zip(axes.flat, present):
        part = frame[frame["data"] == data].sort_values("v")
        width = part[part["band"].astype(bool)]
        ax.plot(width["v"], width["relative_width"], "o-", color=PALETTE["black"], ms=3.5, lw=1.0)
        spread = part[np.isfinite(part["driver_std_relative"].astype(float))]
        ax.plot(spread["v"], spread["driver_std_relative"], "--", color=colour, lw=1.0)
        many = spread["drivers"] >= few_drivers
        ax.plot(spread.loc[many, "v"], spread.loc[many, "driver_std_relative"], "s", color=colour, ms=3.5)
        ax.plot(spread.loc[~many, "v"], spread.loc[~many, "driver_std_relative"], "s", mfc="white", mec=colour, ms=3.5)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("speed (m/s)")
        figures._label(ax, data)
    axes[0, 0].set_ylabel("relative to the median spacing")
    handles = [
        Line2D([], [], color=PALETTE["black"], marker="o", ms=3.5, label="band width (s95 - s5) / s50"),
        Line2D([], [], color=colour, ls="--", marker="s", ms=3.5, label="std over drivers of their median / s50"),
        Line2D([], [], color=colour, ls="none", marker="s", mfc="white", ms=3.5,
               label=f"fewer than {few_drivers} drivers"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return fig


# ------------------------------------------------------------------------- 4 stability maps (D123 part b)


def map_category(record: Mapping[str, Any]) -> str:
    """Category of a speed on the map, as ``_band_category`` of the audit with the numerical flag: the status
    ``outside``, ``none`` or ``indifferent``, else ``stable`` / ``unstable`` / ``undefined``."""
    status = record.get("status")
    if status in ("outside", "none", "indifferent"):
        return str(status)
    flag = record.get("unstable")
    return "undefined" if flag is None else "unstable" if flag else "stable"


def stability_map_figure(panels: Sequence[Mapping[str, Any]]) -> Figure:
    """Equilibrium spacing against the speed per arm: the band of the data as a strip (its median dashed), every
    speed as a point coloured by its category (open: no band at that speed), ``none`` as a triangle at the top,
    the speeds outside the support shaded; log spacing axis shared by the panels."""
    if not any(p.get("records") for p in panels):
        raise ValueError("no stability.json")
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 2.9), squeeze=False, sharey=True, sharex=True)
    values = []
    for panel in panels:
        for r in panel.get("records") or []:
            values += [_float(r.get(k)) for k in ("s", "band_low", "band_high")]
    values = np.asarray([x for x in values if np.isfinite(x) and x > 0])
    low, high = (float(values.min()) / 1.3, float(values.max()) * 1.6) if len(values) else (1.0, 200.0)
    seen: dict[str, Any] = {}
    for ax, panel in zip(axes.flat, panels):
        records = panel.get("records") or []
        _label(ax, panel["label"])
        if not records:
            ax.text(0.5, 0.5, panel.get("empty", "no stability.json"), transform=ax.transAxes, ha="center", va="center")
            continue
        speeds = [float(r["v"]) for r in records]
        support = panel.get("support")
        if support:
            for a, b in ((min(speeds) - 0.5, float(support[0])), (float(support[1]), max(speeds) + 0.5)):
                if b > a:
                    ax.axvspan(a, b, color=PALETTE["grey"], alpha=0.15, lw=0, zorder=0)
                    support_patch = Patch(color=PALETTE["grey"], alpha=0.15, label="outside the speed support")
                    seen.setdefault("support", support_patch)
        edges = [(float(r["v"]), _float(r.get("band_low")), _float(r.get("band_high"))) for r in records]
        banded = sorted(e for e in edges if np.isfinite(e[1]) and np.isfinite(e[2]))
        if banded:
            v_band, s_low, s_high = (np.asarray(c) for c in zip(*banded))
            ax.fill_between(v_band, s_low, s_high, color=PALETTE["grey"], alpha=0.35, lw=0, zorder=1)
            seen.setdefault("band", Patch(color=PALETTE["grey"], alpha=0.35, label="band of the data (5-95 %)"))
        median = panel.get("median")
        if median is not None and len(median[0]):
            ax.plot(median[0], median[1], "--", color=PALETTE["dark"], lw=0.8, zorder=2)
            seen.setdefault("median", Line2D([], [], ls="--", color=PALETTE["dark"], lw=0.8, label="band median"))
        for r in records:
            category = map_category(r)
            colour, text = MAP_STYLE[category]
            if category == "none":
                ax.plot(float(r["v"]), 0.95, "v", color=colour, ms=5, transform=ax.get_xaxis_transform(), zorder=4,
                        clip_on=False)  # fmt: skip
            elif np.isfinite(_float(r.get("s"))):
                has_band = r.get("in_band") is not None
                ax.plot(float(r["v"]), float(r["s"]), "D" if category == "indifferent" else "o", ms=4.5, color=colour,
                        mfc=colour if has_band else "white", mew=1.0, zorder=4)  # fmt: skip
                if not has_band:
                    seen.setdefault("no band", Line2D([], [], ls="none", marker="o", mfc="white", color=PALETTE["dark"],
                                                      label="no band at this speed"))  # fmt: skip
            marker = "v" if category == "none" else "D" if category == "indifferent" else "o"
            seen.setdefault(category, Line2D([], [], ls="none", marker=marker, color=colour, label=text))
        ax.set_xlabel("speed (m/s)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(low, high)
    figures._plain_log(axes[0, 0].yaxis, low, high)
    axes[0, 0].set_ylabel("equilibrium spacing (m)")
    order = ("band", "median", "stable", "unstable", "outside", "none", "indifferent", "undefined", "no band",
             "support")  # fmt: skip
    handles = [seen[key] for key in order if key in seen]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)))
    return fig


# ------------------------------------------------------------------------- 5 corridor grids (D123 part b)


def _grid(n: int, cols: int, height: float, **kwargs: Any) -> tuple[Figure, np.ndarray]:
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    return plt.subplots(rows, cols, figsize=(FULL_WIDTH, height * rows + 0.5), squeeze=False, **kwargs)


def speed_contours_grid(fields: Sequence[Mapping[str, Any]], window: Sequence[float] | None, vmax: float,
                        cols: int = 4) -> Figure:  # fmt: skip
    """The panels of ``figures.speed_contours`` (x-t speed fields on one colour scale, cells without a vehicle
    grey, the analysis window dashed, the label inside the panel) on a grid of ``cols`` columns."""
    fields = [f for f in fields if f.get("speed") is not None]
    if not fields:
        raise ValueError("no trajectories")
    fig, axes = _grid(len(fields), cols, 1.4, sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#D9D9D9")
    image = None
    for ax, field in zip(axes.flat, fields):
        x_edges, t_edges = np.asarray(field["x_edges"]), np.asarray(field["t_edges"])
        image = ax.imshow(np.ma.masked_invalid(np.asarray(field["speed"], dtype=float)), origin="lower", aspect="auto",
                          extent=(t_edges[0], t_edges[-1], x_edges[0], x_edges[-1]), cmap=cmap, vmin=0.0, vmax=vmax,
                          interpolation="nearest")  # fmt: skip
        for t in window or ():
            ax.axvline(t, color="white", ls="--", lw=0.8)
        ax.text(0.03, 0.95, field["label"], transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=6,
                color="white", bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.2, "lw": 0},
                in_layout=False)  # a long law name must not widen the gaps of the grid  # fmt: skip
    _grid_labels(fig, axes, len(fields), "time from the start of the period (s)", "x (m)")
    figures._hide_unused(axes, len(fields))
    fig.colorbar(image, ax=axes, label="speed (m/s)", shrink=0.6)
    return fig


def fundamental_diagrams_grid(panels: Sequence[Mapping[str, Any]], truth_curve: tuple[Any, Any] | None,
                              bin_width: float, cols: int = 4) -> Figure:  # fmt: skip
    """The panels of ``figures.fundamental_diagrams`` (density-flow scatter of the Edie cells, lanes together, the
    binned mean flow of the ground truth on every panel) on a grid of ``cols`` columns."""
    panels = [p for p in panels if p.get("density") is not None and len(p["density"])]
    if not panels:
        raise ValueError("no trajectories")
    fig, axes = _grid(len(panels), cols, 1.45, sharex=True, sharey=True)
    for ax, panel in zip(axes.flat, panels):
        ax.scatter(panel["density"], panel["flow"], s=4, color=law_color(panel["key"]), alpha=0.6, lw=0)
        if truth_curve is not None and len(truth_curve[0]):
            centres = np.asarray(truth_curve[0], dtype=float) + 0.5 * bin_width
            ax.plot(centres, truth_curve[1], "-", color=PALETTE["black"], lw=1.1)
        ax.text(0.03, 0.95, panel["label"], transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=6,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 1.0, "lw": 0}, in_layout=False)  # fmt: skip
    _grid_labels(fig, axes, len(panels), "density, all lanes (veh/km)", "flow, all lanes (veh/h)")
    figures._hide_unused(axes, len(panels))
    curve = Line2D([], [], color=PALETTE["black"], lw=1.1, label="ground truth, mean flow per density bin")
    fig.legend(handles=[curve], loc="outside upper center")  # below the panels sits the x label
    return fig


# --------------------------------------------------------------------------------------------- the maker


class Supplement:
    """Builds the tables and figures of the supplement; every missing input becomes a note."""

    def __init__(self, cfg: SupplementConfig) -> None:
        self.cfg = cfg
        self.lines: list[str] = []
        self.notes: list[str] = []
        self.written: dict[str, list[str]] = {"figures": [], "tables": []}
        self.figure_dir = cfg.out_dir / "figures"
        self.chosen: dict[str, Any] = {}
        path = cfg.tables_m4 / "chosen_weights.json"
        try:
            self.chosen = read_json(path)
        except (OSError, ValueError) as exc:
            self.note("E2", f"{self.label(path)} {'missing' if isinstance(exc, FileNotFoundError) else 'unreadable'}: "
                            "no chosen weights, no E2 arm")  # fmt: skip

    # ------------------------------------------------------------------------------------ helpers
    def note(self, where: str, text: str) -> str:
        line = f"{where}: {text}"
        self.notes.append(line)
        self.lines.append(f"NOTE {line}")
        return line

    def label(self, path: Path) -> str:
        for root in (self.cfg.runs_root, REPO_ROOT):
            try:
                return Path(path).relative_to(root).as_posix()
            except ValueError:
                continue
        return Path(path).as_posix()

    def run_dir(self, experiment: str, data: str, model: str, fold: int, seed: int) -> Path:
        return self.cfg.runs_root / experiment / data / model / f"driver_fold{fold}_seed{seed}"

    def read(self, path: Path, where: str, notes: list[str] | None = None) -> dict[str, Any] | None:
        """A JSON file; None and a note when it is missing or unreadable."""
        try:
            return read_json(path)
        except FileNotFoundError:
            text = f"{self.label(path)} missing"
        except (OSError, ValueError) as exc:
            text = f"{self.label(path)} unreadable ({type(exc).__name__})"
        line = self.note(where, text)
        if notes is not None:
            notes.append(line)
        return None

    def figure(self, name: str, build: Callable[[], tuple[Figure, str]]) -> None:
        """Build one figure and its caption and write ``<name>.png|pdf|txt``; a failure becomes a note."""
        try:
            fig, caption = build()
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            self.note(f"figure {name}", f"not written: {exc or type(exc).__name__}")
            return
        except Exception as exc:  # an unexpected failure must not stop the other outputs
            tail = traceback.format_exc(limit=3).strip().splitlines()[-1]
            self.note(f"figure {name}", f"not written: {type(exc).__name__}: {exc} ({tail})")
            plt.close("all")
            return
        paths = figures.save(fig, self.figure_dir, name, self.cfg.dpi)
        (self.figure_dir / f"{name}.txt").write_text(caption.strip() + "\n", encoding="utf-8")
        self.written["figures"].append(name)
        self.lines.append(f"FIGURE {name}: {self.label(paths[0])} (+ .pdf, .txt)")

    def table(self, table: Table) -> None:
        if table.frame.empty and not len(table.frame.columns):  # no rows: the header of the Markdown columns
            table.frame = pd.DataFrame(columns=[c.key for c in table.columns])
        write_table(table, self.cfg.tables_dir)
        self.written["tables"].append(table.name)
        self.lines.append(f"TABLE {table.name}: {len(table.frame)} rows -> {self.label(self.cfg.tables_dir)}/"
                          f"{table.name}.csv|md")  # fmt: skip

    def guarded(self, name: str, step: Callable[[], None]) -> None:
        """Run one output group; an unexpected failure becomes a note and the other groups still run."""
        try:
            step()
        except Exception as exc:
            tail = traceback.format_exc(limit=3).strip().splitlines()[-1]
            self.note(name, f"failed: {type(exc).__name__}: {exc} ({tail})")
            plt.close("all")

    @staticmethod
    def with_notes(caption: str, notes: Sequence[str]) -> str:
        return caption if not notes else caption + "\n\nNotes: " + " ".join(f"{n.rstrip('.')}." for n in notes)

    # --------------------------------------------------------------------------- 1 expansion (D115)
    def lowfreq_expansion(self) -> None:
        c = self.cfg
        frames, notes = [], []
        for arch in c.expansion_architectures:
            arms = [("E1", "e1")]
            experiment = chosen_experiment(self.chosen, arch)
            if experiment:
                arms.append(("E2", experiment))
            elif arch not in c.e1_only:
                notes.append(self.note("lowfreq_expansion", f"{arch}: no chosen E2 weight, E1 only"))
            for arm, experiment in arms:
                run = self.run_dir(experiment, c.data, arch, c.fold, c.seed)
                payload = self.read(run / "stability.json", "lowfreq_expansion", notes)
                if payload is None:
                    continue
                audit, where = payload.get("audit"), f"{self.label(run)}/stability.json"
                if not isinstance(audit, Mapping):
                    notes.append(self.note("lowfreq_expansion", f"{where}: field 'audit' missing"))
                    continue
                points, skipped = expansion_points(audit, c.n_low)
                for reason, count in skipped.items():
                    notes.append(self.note("lowfreq_expansion", f"{where}: {count} speed(s) left out, {reason}"))
                if len(points):
                    frames.append(points.assign(architecture=arch, arm=arm, experiment=experiment))
        points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows = lowfreq_rows(points)
        omegas = sorted(points["omega"].unique()) if len(points) else []
        freq_text = ", ".join(f"{w:.4g}" for w in omegas) or f"the {c.n_low} lowest"
        first = points[points["k"] == 0] if len(points) else points
        flagged, n_first = (int(first["flagged"].astype(bool).sum()), len(first)) if len(points) else (0, 0)
        definitions = [
            f"Low-frequency expansion of the gain (D115): fold-{c.fold} runs (seed {c.seed}, {c.data}) of E1 and of "
            "the chosen E2 weight (D85, chosen_weights.json; " + ", ".join(c.e1_only) + ": E1 only). Per grid speed "
            "with an equilibrium (status ok, multiple or outside: inside or outside the band) the numerical gain of "
            f"the audit (stability.json, audit.equilibria[].gain) at the {c.n_low} lowest audit frequencies ({freq_text} "
            "rad/s) against the expansion |G| = sqrt(max(0, 1 - ω² M / f_s²)), M = f_v² + 2 f_v f_dv - 2 f_s, from the "
            "stored partial derivatives f_s, f_dv, f_v (the memoryless view: derivatives summed over the window, for "
            "GRU, LSTM and PERL; D66).",
            "Speeds whose rollout at that frequency was clipped, stopped or collided (audit flags) are left out "
            "(flagged): their gain is not a linear response. speeds = speeds with equilibrium - flagged.",
            "|num - exp|: absolute difference of the numerical gain and the expansion, median and 90 % quantile over "
            "the speeds; same sign of |G| - 1: share of the speeds where sign(numerical - 1) = sign(expansion - 1); "
            "of which -ω² < M < 0: sign differences where the exact memoryless gain (1) is below 1 while the "
            "expansion is above (the band 0 < -M < ω² of docs/theory_notes.md, section 2).",
            "|num - (1)|: median absolute difference of the numerical gain and the exact gain |G(iω)| of the "
            "continuous memoryless linearisation (equation (1) of docs/theory_notes.md, transfer_continuous): the "
            "part of |num - exp| that is not the truncation of the expansion (memory of the recurrent models, the "
            "discrete integration). slow pole: |f_s| / |f_dv + f_v| (rad/s); the expansion holds for ω well below it.",
            "Speeds without a linear stationary response, kept in the numbers above: locally unstable = the memoryless "
            "view is not locally stable (f_s <= 0 or f_v + f_dv >= 0); fit residual > 0.1 = the projection of the "
            "follower speed on sin, cos and 1 leaves more than 10 % of its norm (audit.equilibria[].residual).",
        ]  # fmt: skip
        self.table(Table("lowfreq_expansion", "Low-frequency expansion of the gain (D115)", definitions + notes,
                         rows, LOWFREQ_COLUMNS))  # fmt: skip
        caption = (
            f"Low-frequency expansion of the gain (D115). Fold-{c.fold} runs of E1 (no penalty) and of the chosen E2 "
            f"weight (D85; {', '.join(c.e1_only)}: E1 only) on {c.data}, seed {c.seed}: at every grid speed with an "
            "equilibrium, the numerical gain |G| of the audit (stability.json) at the lowest audit frequency "
            f"{omegas[0] if omegas else 0.02:.2g} rad/s (y) against the expansion |G| = sqrt(max(0, 1 - ω² M / f_s²)), "
            "M = f_v² + 2 f_v f_dv - 2 f_s, from the stored partial derivatives (memoryless view for GRU, LSTM and "
            "PERL) (x); open markers: equilibria that are not locally stable in the memoryless view (no stationary "
            "response). Solid line: numerical = expansion; dotted: |G| = 1 (points in the upper left and lower right "
            "quadrants have a different sign of |G| - 1). Speeds whose rollout at that frequency was clipped, stopped "
            f"or collided are left out ({flagged} of {n_first}). Numbers at the three lowest frequencies: table "
            "lowfreq_expansion."
        )  # fmt: skip
        self.figure("lowfreq_expansion", lambda: (lowfreq_expansion_figure(points, c.expansion_architectures),
                                                  self.with_notes(caption, notes)))  # fmt: skip

    # --------------------------------------------------------------------- 2 certificate tightness (D116)
    def certificate_tightness(self) -> None:
        c = self.cfg
        frames, notes, computed = [], [], []
        for r_max, before, after in c.certificate_experiments:
            for stage, experiment, data in (("before", before, c.certificate_data[0]),
                                            ("after", after, c.certificate_data[1])):  # fmt: skip
                missing = []
                for fold in c.certificate_folds:
                    for seed in c.certificate_seeds:
                        run = self.run_dir(experiment, data, c.certificate_model, fold, seed)
                        files = [run / name for name in ("certificate.json", "stability.json")]
                        absent = [p.name for p in files if not p.exists()]
                        if absent:
                            missing.append(f"fold{fold}_seed{seed} ({', '.join(absent)})")
                            continue
                        points = self.certificate_run(run, r_max, notes, computed)
                        if points is not None and len(points):
                            frames.append(points.assign(r_max=r_max, stage=stage, experiment=experiment, data=data,
                                                        run=f"{fold}/{seed}"))  # fmt: skip
                if missing:
                    expected = len(c.certificate_folds) * len(c.certificate_seeds)
                    text = f"{experiment}/{data}: {len(missing)} of {expected} runs without input: {', '.join(missing)}"
                    notes.append(self.note("certificate_tightness", text))
        if computed:
            notes.append(self.note("certificate_tightness", f"per-speed bounds not stored in certificate.json of "
                                   f"{len(computed)} run(s), computed from model.pt with cf_stability.stability."
                                   f"certificate: {', '.join(computed)}"))  # fmt: skip
        points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows = certificate_rows(points)
        n_negative = int((points["audited"] < points["certified"]).sum()) if len(points) else 0
        definitions = [
            "Tightness of the certificate of E4 (D116): every run of e4_stable, e4_stable_ft and the r_max variants "
            "e4_stable_r{0.1,0.2,0.5}, e4_stable_ft_r{0.1,0.2,0.5} (ResidualIDM, core with margin 0.2, certified "
            f"budget; folds {list(c.certificate_folds)} x seeds {list(c.certificate_seeds)}; before fine-tuning on "
            f"{c.certificate_data[0]}, after on {c.certificate_data[1]}).",
            "Per grid speed: audited = the margin M = f_v² + 2 f_v f_dv - 2 f_s of the hybrid at its equilibrium "
            "(analytic criterion of the audit, stability.json audit.equilibria[].margin); bound a priori = "
            "certificate.json per_speed[].a_priori_margin (minimum over every spacing that can be an equilibrium of a "
            "residual of amplitude r_max and over the box of the residual derivatives), compared at every speed with "
            "an equilibrium (inside or outside the band); bound at the equilibria = per_speed[].guaranteed_margin "
            "(core derivatives at the hybrid's equilibrium plus the worst residual derivatives), compared at the "
            "anchored equilibria the certificate uses (found), at the spacing of the audit.",
            "slack = audited - bound: median, 10 % quantile and minimum over the speeds of all runs; negative slack "
            "would violate the certificate (expected 0).",
        ]  # fmt: skip
        self.table(Table("certificate_tightness", "Tightness of the certificate of E4 (D116)", definitions + notes,
                         rows, CERTIFICATE_COLUMNS))  # fmt: skip
        r_values = [r for r, _, _ in c.certificate_experiments]
        caption = (
            "Tightness of the certificate of E4 (D116): per grid speed of every run of the certified hybrid "
            "(ResidualIDM with the margin core and the certified budget; e4_stable and e4_stable_r{0.1, 0.2, 0.5} on "
            "HighD before fine-tuning, e4_stable_ft and e4_stable_ft_r{0.1, 0.2, 0.5} on NGSIM I-80 after; 5 folds x 5 "
            "seeds each) the audited margin M of the hybrid (analytic criterion of the audit; y) against the certified "
            "lower bound of certificate.json (x). Columns: residual amplitude r_max; top row: the a priori bound "
            "(every speed with an equilibrium), bottom row: the bound at the anchored equilibria. Line: margin = "
            f"bound; a point below it would violate the certificate ({n_negative} of {len(points)} points). Table "
            "certificate_tightness: slack per r_max, stage and bound."
        )  # fmt: skip
        self.figure("certificate_tightness", lambda: (certificate_tightness_figure(points, r_values),
                                                      self.with_notes(caption, notes)))  # fmt: skip

    def certificate_run(self, run: Path, r_max: float, notes: list[str], computed: list[str]) -> pd.DataFrame | None:
        """The points of one run (``certificate_points``), the per-speed bounds computed from ``model.pt`` when the
        certificate file does not store them."""
        where = "certificate_tightness"
        cert = self.read(run / "certificate.json", where, notes)
        payload = self.read(run / "stability.json", where, notes)
        if cert is None or payload is None:
            return None
        label = self.label(run)
        if cert.get("error") or cert.get("applicable") is False:
            notes.append(self.note(where, f"{label}/certificate.json holds no certificate "
                                          f"({cert.get('error') or 'not applicable'})"))  # fmt: skip
            return None
        audit = payload.get("audit")
        if not isinstance(audit, Mapping):
            notes.append(self.note(where, f"{label}/stability.json: field 'audit' missing"))
            return None
        model = run / "model.pt"
        if model.exists():
            stale = [p.name for p in (run / "certificate.json", run / "stability.json")
                     if p.stat().st_mtime < model.stat().st_mtime]  # fmt: skip
            if stale:
                notes.append(self.note(where, f"{label}: {', '.join(stale)} older than model.pt, run left out"))
                return None
        stored = (cert.get("residual") or {}).get("r_max")
        if stored is not None and not math.isclose(float(stored), r_max, rel_tol=1e-9):
            notes.append(self.note(where, f"{label}: certificate.json has r_max {stored}, the design {r_max}"))
        per_speed = cert.get("per_speed")
        lacking = sorted({k for p in per_speed or [{}] for k in CERTIFICATE_KEYS if k not in p})
        if lacking:
            speeds = [float(r["v"]) for r in audit.get("equilibria") or [] if r.get("v") is not None] or list(V_GRID)
            n_scan = int(((cert.get("config") or {}).get("certificate") or {}).get("n_scan") or 400)
            try:
                per_speed = computed_certificate_speeds(run, speeds, n_scan)
            except Exception as exc:  # the run is left out, the others go on
                notes.append(self.note(where, f"{label}/certificate.json lacks per-speed field(s) {', '.join(lacking)} "
                                              f"and model.pt gives none ({type(exc).__name__}: {exc})"))  # fmt: skip
                return None
            computed.append(f"{label} (missing: {', '.join(lacking)})")
        points, skipped = certificate_points(per_speed, audit)
        for reason, count in skipped.items():
            notes.append(self.note(where, f"{label}: {count} speed(s) left out, {reason}"))
        return pd.DataFrame(points)

    # -------------------------------------------------------------------------- 3 band width (D123 a)
    def band_width(self) -> None:
        from cf_stability.data.schema import EventSet

        c = self.cfg
        frames, notes = [], []
        band_cfg = BandConfig()
        for data in c.band_sets:
            directory = c.events_root / data
            try:
                events = EventSet.from_parquet(directory).events
            except (OSError, ValueError, KeyError) as exc:
                notes.append(self.note("band_width", f"{self.label(directory)} not readable ({type(exc).__name__}: "
                                                     f"{str(exc).splitlines()[0] if str(exc) else ''})"))  # fmt: skip
                continue
            rows = band_width_rows(events, data, band_cfg, c.min_driver_samples)
            if not len(rows) or not rows["band"].any():
                notes.append(self.note("band_width", f"{data}: no grid speed with {band_cfg.min_samples} near-steady "
                                                     "samples (no band)"))  # fmt: skip
            frames.append(rows)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        definitions = [
            "Spacing band of the data (D78, D123 part a): per grid speed the 5 %, 50 % and 95 % quantiles of the "
            f"spacing of the near-steady samples of all events of the set (|dv| < {band_cfg.dv_max:g} m/s, |a| < "
            f"{band_cfg.a_max:g} m/s², speed within {band_cfg.half_width:g} m/s of the grid speed; "
            "cf_stability.train.tensors.spacing_band, dv = v - v_lead); a speed with fewer than "
            f"{band_cfg.min_samples} such samples has no band. relative width = (s95 - s5) / s50.",
            "Per-driver dispersion: the standard deviation (ddof 1) over the drivers with at least "
            f"{c.min_driver_samples} near-steady samples at the speed of the driver's median near-steady spacing "
            "there, in m and relative to "
            "s50 (none with fewer than two such drivers). Drivers are follower ids; follownet_highd has none, every "
            "event (15 s) is its own driver (D19).",
        ]  # fmt: skip
        self.table(Table("band_width", "Relative width of the spacing band and per-driver dispersion (D123)",
                         definitions + notes, frame, BAND_COLUMNS))  # fmt: skip
        caption = (
            "The price of a common equilibrium (D123 part a): per grid speed, the relative width (s95 - s5) / s50 of "
            "the spacing band of the data (D78: quantiles of the near-steady samples, |dv| < 0.5 m/s, |a| < 0.3 m/s², "
            "within 0.5 m/s of the speed; all events of the set, speeds with at least 200 samples) and the standard "
            f"deviation over the drivers with at least {c.min_driver_samples} near-steady samples at that speed of "
            "their median near-steady spacing, divided by s50 (open squares: fewer than "
            f"{c.few_drivers} drivers). Left: {c.band_sets[0]}" + (f", right: {c.band_sets[1]}" if len(c.band_sets) > 1
                                                                   else "") + ". follownet_highd has no driver ids: "
            "every 15-s event is its own driver (D19). Numbers: table band_width."
        )  # fmt: skip
        self.figure("band_width", lambda: (band_width_figure(frame, c.band_sets, c.few_drivers),
                                           self.with_notes(caption, notes)))  # fmt: skip

    # ---------------------------------------------------------------------------- 4 stability maps (D123 b)
    def stability_maps(self) -> None:
        for arch in self.cfg.map_architectures:
            self.figure(f"stability_map_{arch}", lambda arch=arch: self.stability_map(arch))

    def stability_map(self, arch: str) -> tuple[Figure, str]:
        c = self.cfg
        where, notes, panels = f"stability_map_{arch}", [], []
        arms = [("E1", "e1", "E1, no penalty")]
        experiment = chosen_experiment(self.chosen, arch)
        if experiment:
            entry = self.chosen.get(arch) or {}
            weight = entry.get("override", entry.get("weight"))
            arms.append(("E2", experiment, f"E2, {entry.get('kind')} penalty, weight {float(weight):g}"))
        else:
            notes.append(self.note(where, f"{arch}: no chosen E2 weight"))
        for _, experiment, text in arms:
            run = self.run_dir(experiment, c.data, arch, c.fold, c.seed)
            payload = self.read(run / "stability.json", where, notes)
            audit = (payload or {}).get("audit") or {}
            records = audit.get("equilibria") or []
            if payload is not None and not records:
                notes.append(self.note(where, f"{self.label(run)}/stability.json: no records in audit.equilibria"))
            metrics = run / "metrics.json"
            band = None
            if metrics.exists():
                try:
                    band = (read_json(metrics).get("context") or {}).get("band")
                except (OSError, ValueError):
                    band = None
            median = None if not band else (band.get("v") or [], band.get("s_median") or [])
            panels.append({"label": text, "records": records, "support": audit.get("support_v"), "median": median,
                           "empty": f"{experiment}: no stability.json"})  # fmt: skip
        counts = Counter(map_category(r) for p in panels for r in p["records"])
        summary = ", ".join(f"{k} {v}" for k, v in counts.items())
        caption = (
            f"Equilibria of the {figures.name(arch)} of fold {c.fold} ({c.data}, seed {c.seed}) on the speed grid: "
            + " and ".join(f"{text} ({experiment})" for _, experiment, text in arms) + ", one panel each. Grey strip: "
            "spacing band of the training data of the fold (5-95 % quantiles of the near-steady samples, D78; "
            "linear between the grid speeds), dashed: its median (metrics.json); points: the equilibrium of every "
            "grid speed (searched inside the band where the speed has one, D79), coloured by the status of the audit "
            "with the numerical rule (gain above 1.02): stable or unstable inside the band, outside the band (the "
            "first equilibrium of 1-200 m), none (no equilibrium: triangle at the top of the panel); open markers: "
            "speeds without band; shaded: speeds outside the 1-99 % speed range of the training data. Log spacing "
            f"axis. Speeds by status over both panels: {summary or 'none'}."
        )  # fmt: skip
        return stability_map_figure(panels), self.with_notes(caption, notes)

    # ------------------------------------------------------------------------------ 5 corridor (D123 b)
    def corridor_design(self) -> tuple[dict[str, str], dict[str, Any], list[str]]:
        """Corridor names (design.corridors), the macro settings of the metrics (``macro``) and notes."""
        notes: list[str] = []
        path = self.cfg.configs_dir / "corridor_metrics.yaml"
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            notes.append(self.note("corridor", f"{self.label(path)} not readable ({type(exc).__name__}): default "
                                               "corridor names and macro settings"))  # fmt: skip
            raw = {}
        names = {**CORRIDOR_NAMES, **((raw.get("design") or {}).get("corridors") or {})}
        return names, raw.get("macro") or {}, notes

    def corridor(self) -> None:
        c = self.cfg
        names, macro, notes = self.corridor_design()
        laws_table = None
        path = c.tables_m5 / "laws.csv"
        try:
            laws_table = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            notes.append(self.note("corridor", f"{self.label(path)} not readable ({type(exc).__name__}): the laws are "
                                               "the law directories of the scenarios"))  # fmt: skip
        for corridor in c.corridors:
            for period in c.periods:
                scenario = f"{corridor}_p{period}"
                try:
                    inputs, info, scenario_notes = self.corridor_inputs(names.get(corridor, corridor), scenario, macro,
                                                                        laws_table)  # fmt: skip
                except Exception as exc:  # the figures of this scenario are left out, the others stay
                    tail = traceback.format_exc(limit=3).strip().splitlines()[-1]
                    self.note(f"corridor {scenario}", f"inputs failed: {type(exc).__name__}: {exc} ({tail})")
                    continue
                scenario_notes = notes + scenario_notes
                self.figure(f"contours_{corridor}_p{period}",
                            lambda i=inputs, s=info, n=scenario_notes: self.contour_figure(i, s, n))  # fmt: skip
                self.figure(f"fd_{corridor}_p{period}",
                            lambda i=inputs, s=info, n=scenario_notes: self.fd_figure(i, s, n))  # fmt: skip

    def corridor_laws(self, name: str, scenario: str, laws_table: pd.DataFrame | None) -> tuple[list[str], list[str]]:
        """Laws of the scenario with a run of the figure seed: the main laws, then the laws of ``laws.csv`` for the
        corridor in its order, then other law directories with such a run (sorted); and the notes."""
        c = self.cfg
        root = c.corridor_root / scenario
        seed_dir = f"seed{c.corridor_seed}"
        listed: list[str] = []
        if laws_table is not None and "law" in laws_table:
            rows = laws_table if "corridor" not in laws_table else laws_table[laws_table["corridor"] == name]
            listed = [str(x) for x in dict.fromkeys(rows["law"].dropna()) if str(x) != "ground truth"]
        others = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")) if root.is_dir() \
            else []  # fmt: skip
        ordered = list(dict.fromkeys([*c.main_laws, *listed, *others]))
        present = [law for law in ordered if (root / law / seed_dir / "trajectories.npz").exists()]
        notes = []
        absent = [law for law in dict.fromkeys([*c.main_laws, *listed]) if law not in present]
        if absent:
            notes.append(self.note(f"corridor {scenario}", f"no {seed_dir} run of {', '.join(absent)}"))
        extra = [law for law in present if law not in listed and law not in c.main_laws]
        if extra and listed:
            notes.append(self.note(f"corridor {scenario}", f"laws with a run but not in laws.csv: {', '.join(extra)}"))
        if len(present) > c.max_panels - 1:
            notes.append(self.note(f"corridor {scenario}", f"{len(present)} laws, {c.max_panels - 1} panels: left out "
                                                           f"{', '.join(present[c.max_panels - 1:])}"))  # fmt: skip
            present = present[: c.max_panels - 1]
        return present, notes

    def corridor_inputs(
        self, name: str, scenario: str, macro: Mapping[str, Any], laws_table: pd.DataFrame | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        """The prepared trajectories of the ground truth and of the laws of corridor ``name`` (as ``corridor_metrics``
        prepares them), the geometry and the macro settings of the scenario, and the notes."""
        from cf_stability.corridor.macro import (
            Geometry, MacroConfig, geometry_from_scenario, load_npz, prepare_trajectories, scenario_macro_config,
        )  # fmt: skip

        c = self.cfg
        where = f"corridor {scenario}"
        notes: list[str] = []
        scenario_dir = c.corridor_root / "scenarios" / scenario
        scenario_json = self.read(scenario_dir / "scenario.json", where, notes) or {}
        if not scenario_json:
            notes.append(self.note(where, "default geometry and window"))
        geometry = geometry_from_scenario(scenario_json, Geometry())
        try:
            mcfg = scenario_macro_config(MacroConfig.from_mapping(macro), scenario_json)
        except (TypeError, ValueError) as exc:
            notes.append(self.note(where, f"macro settings not usable ({exc}): defaults"))
            mcfg = scenario_macro_config(MacroConfig(), scenario_json)
        laws, law_notes = self.corridor_laws(name, scenario, laws_table)
        notes += law_notes
        truth = ("ground truth", "ground truth", scenario_dir / "ground_truth.npz", scenario_dir / "vehicles_truth.npz")
        items = [truth]
        for law in laws:
            run = c.corridor_root / scenario / law / f"seed{c.corridor_seed}"
            items.append((law, law, run / "trajectories.npz", run / "vehicles.npz"))
        inputs = []
        for label, key, trajectories, vehicles in items:
            if not trajectories.exists():
                notes.append(self.note(where, f"{self.label(trajectories)} missing"))
                continue
            try:
                arrays = load_npz(trajectories)
                fleet = load_npz(vehicles) if vehicles.exists() else None
                prepared = prepare_trajectories(arrays, fleet, geometry, mcfg)
            except (OSError, ValueError, KeyError) as exc:
                notes.append(self.note(where, f"{self.label(trajectories)} unreadable ({type(exc).__name__})"))
                continue
            inputs.append({"label": label, "key": key, "prepared": prepared})
        info = {"scenario": scenario, "corridor": name, "period": scenario.rsplit("_p", 1)[-1], "geometry": geometry,
                "macro": mcfg}  # fmt: skip
        return inputs, info, notes

    def contour_figure(self, inputs: list[dict[str, Any]], info: Mapping[str, Any],
                       notes: list[str]) -> tuple[Figure, str]:  # fmt: skip
        """Speed fields as figure 5 of M6 prepares them (Edie cells of the wave field, raw), all panels."""
        from cf_stability.corridor.macro import edie_grid, interval_edges
        from cf_stability.corridor.waves import speed_field

        if not inputs:
            raise ValueError("no trajectories of the corridor")
        g, wc = info["geometry"], info["macro"].waves
        t_end = max(float(np.nanmax(i["prepared"]["t"])) for i in inputs if len(i["prepared"]["t"]))
        fields, empty_lanes = [], []
        for item in inputs:
            p = item["prepared"]
            grid = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                             interval_edges(g.x_in, g.x_out, wc.dx), interval_edges(0.0, t_end, wc.dt),
                             lanes=wc.lanes)  # fmt: skip
            speed = speed_field(grid["total"]["distance"], grid["total"]["time"], smooth=0)
            empty = [int(lane) for i, lane in enumerate(grid["lanes"]) if grid["per_lane"]["time"][i].sum() == 0]
            if empty:
                empty_lanes.append(f"{item['label']}: no vehicle in lane(s) {empty}")
            fields.append({"label": item["label"], "speed": speed, "x_edges": grid["x_edges"],
                           "t_edges": grid["t_edges"]})  # fmt: skip
        truth = next((f["speed"] for f in fields if f["label"] == "ground truth"), fields[0]["speed"])
        vmax = math.ceil(float(np.nanpercentile(truth, 99))) if np.isfinite(truth).any() else 30
        fig = speed_contours_grid(fields, g.window, vmax, self.cfg.grid_columns)
        laws = [f["label"] for f in fields if f["label"] != "ground truth"]
        has_truth = fields[0]["label"] == "ground truth"
        truth_text = "the ground truth (first panel) and " if has_truth else "no ground truth; "
        lanes = ", ".join(str(x) for x in wc.lanes)
        caption = (
            f"Speed fields of {info['corridor']}, period {info['period']} (scenario {info['scenario']}): x-t Edie "
            f"speed of {wc.dx:g} m x {wc.dt:g} s cells over the lanes {lanes} together (as the wave field of the "
            f"metrics, raw cells), one colour scale 0-{vmax} m/s (the 99 % quantile of the ground truth), cells "
            f"without a vehicle grey; dashed: the analysis window {g.window[0]:g}-{g.window[1]:g} s. Panels: "
            f"{truth_text}every law with a run of seed {self.cfg.corridor_seed} ({len(laws)}): {', '.join(laws)}."
        )  # fmt: skip
        return fig, self.with_notes(caption, [*notes, *empty_lanes])

    def fd_figure(self, inputs: list[dict[str, Any]], info: Mapping[str, Any], notes: list[str]) -> tuple[Figure, str]:
        """Fundamental diagrams as figure 6 of M6 prepares them (Edie cells of the metrics, all lanes), all panels."""
        from cf_stability.corridor.macro import edie_grid, fundamental_diagram, interval_edges, section_edges

        if not inputs:
            raise ValueError("no trajectories of the corridor")
        g, mcfg = info["geometry"], info["macro"]
        w0, w1 = (float(w) for w in g.window)
        panels, truth_curve = [], None
        for item in inputs:
            p = item["prepared"]
            grid = edie_grid({"vehicle": p["piece"], "t": p["t"], "x": p["x"], "lane": p["lane"]},
                             section_edges(g.x_in, g.x_out, mcfg.cell_dx),
                             interval_edges(w0, w1, mcfg.cell_dt))  # fmt: skip
            flow, density = grid["total"]["flow"].ravel() * 3600.0, grid["total"]["density"].ravel() * 1000.0
            keep = np.isfinite(flow) & np.isfinite(density) & (grid["total"]["time"].ravel() > 0)
            panels.append({"label": item["label"], "key": item["key"], "density": density[keep], "flow": flow[keep]})
            if item["key"] == "ground truth":
                fd = fundamental_diagram(flow[keep], density[keep], mcfg.fd_bin)
                truth_curve = (fd["density_bins"], fd["flow"])
        extra = [] if truth_curve is not None else ["no ground truth: no binned curve"]
        fig = fundamental_diagrams_grid(panels, truth_curve, mcfg.fd_bin, self.cfg.grid_columns)
        laws = [p["label"] for p in panels if p["key"] != "ground truth"]
        caption = (
            f"Fundamental diagrams of {info['corridor']}, period {info['period']} (scenario {info['scenario']}): flow "
            f"against density of the Edie cells ({mcfg.cell_dx:g} m x {mcfg.cell_dt:g} s, all lanes together, the "
            f"section {g.x_in:g}-{g.x_out:g} m and the analysis window {w0:g}-{w1:g} s, as the metrics of M5); black "
            f"line on every panel: the mean flow of the ground truth per density bin of {mcfg.fd_bin:g} veh/km (bin "
            f"centres). Panels: the ground truth first, then every law with a run of seed {self.cfg.corridor_seed} "
            f"({len(laws)}): {', '.join(laws)}."
        )  # fmt: skip
        return fig, self.with_notes(caption, [*notes, *extra])

    # ----------------------------------------------------------------------------------------- run
    def write_notes(self) -> Path:
        path = self.cfg.out_dir / NOTES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "Notes of scripts/make_figures_supplement.py: missing or excluded inputs (one per line)."
        path.write_text("\n".join([header, *(f"- {n}" for n in self.notes)]) + "\n", encoding="utf-8")
        return path


def make_supplement(cfg: SupplementConfig) -> list[str]:
    """Write the tables and figures of ``cfg.outputs``; returns the printed lines (one per output and note)."""
    unknown = set(cfg.outputs) - set(OUTPUTS)
    if unknown:
        raise ValueError(f"unknown outputs {sorted(unknown)} (known: {list(OUTPUTS)})")
    figures.apply_style(cfg.font_size)
    maker = Supplement(cfg)
    for name in OUTPUTS:
        if name in cfg.outputs:
            maker.guarded(name, getattr(maker, name))
    path = maker.write_notes()
    maker.lines.append(f"SUPPLEMENT: {len(maker.written['figures'])} figures, {len(maker.written['tables'])} tables, "
                       f"{len(maker.notes)} notes ({maker.label(path)})")  # fmt: skip
    return maker.lines
