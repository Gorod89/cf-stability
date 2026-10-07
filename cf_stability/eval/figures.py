"""Figures of the report (docs/m6_contract.md, section 3): matplotlib, one style, PNG (200 dpi) and PDF.

Every function takes the numbers it draws (gathered by ``cf_stability.eval.report`` from the tables and
the run files) and returns a figure; :func:`save` writes it in both formats. No figure carries a title:
panels are named by a bold label above their upper left corner. Colours: the Okabe-Ito palette
(colour-blind safe), one colour per model or law across all figures.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # files only, no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

for _logger in ("fontTools", "matplotlib"):  # the PDF backend logs every font subset at level INFO
    logging.getLogger(_logger).setLevel(logging.WARNING)

PALETTE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "dark": "#555555",
}  # fmt: skip
COLORS = {  # one colour per model (M4) and law (M5); Newell's yellow darkened to be read on white
    "idm": PALETTE["black"], "mlp": PALETTE["orange"], "pidl": PALETTE["sky"], "residual_idm": PALETTE["green"],
    "gru": PALETTE["blue"], "lstm": PALETTE["vermillion"], "perl": PALETTE["purple"], "knn": PALETTE["grey"],
    "persistence": PALETTE["dark"], "newell": "#B39B00", "ovm": "#8C6D31",
    "ground truth": PALETTE["black"], "idm_global": PALETTE["black"], "idm_heterogeneous": PALETTE["dark"],
    "residual_idm_certified": PALETTE["green"], "mlp_penalty": PALETTE["orange"], "gru_penalty": PALETTE["blue"],
    "lstm_penalty": PALETTE["vermillion"],
}  # fmt: skip
NAMES = {
    "idm": "IDM", "mlp": "MLP", "pidl": "PIDL", "residual_idm": "ResidualIDM", "gru": "GRU", "lstm": "LSTM",
    "perl": "PERL", "knn": "kNN", "persistence": "persistence", "newell": "Newell", "ovm": "OVM",
}  # fmt: skip
WITHOUT, WITH = PALETTE["blue"], PALETTE["vermillion"]  # without and with penalty
FULL_WIDTH = 7.0  # inches: the text width of a two-column page; panels are read at column width
OFFSETS = ((4, 3), (4, -3), (-4, 3), (-4, -3), (0, 7), (0, -7), (7, 0), (-7, 0))  # points: label positions tried


def name(model: str) -> str:
    return NAMES.get(model, model)


def color(key: str) -> str:
    return COLORS.get(key, PALETTE["grey"])


def apply_style(font_size: float = 9.0) -> None:
    """The style of every figure: readable at column width, colour-blind safe, no top and right spines."""
    plt.rcParams.update({
        "font.size": font_size, "axes.labelsize": font_size, "axes.titlesize": font_size,
        "legend.fontsize": font_size - 1.5, "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "axes.prop_cycle": cycler(color=[PALETTE[k] for k in ("blue", "vermillion", "green", "orange", "sky",
                                                              "purple", "black", "yellow")]),
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "figure.constrained_layout.use": True, "image.cmap": "viridis",
    })  # fmt: skip


def save(fig: Figure, out_dir: Path, stem: str, dpi: int = 200) -> list[Path]:
    """``<stem>.png`` (``dpi``) and ``<stem>.pdf`` in ``out_dir``; the figure is closed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    try:
        fig.savefig(paths[0], dpi=dpi)
        fig.savefig(paths[1])
    finally:
        plt.close(fig)
    return paths


def _label(ax: Any, text: str) -> None:
    """Name of a panel, above its upper left corner (the figures carry no titles)."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def _finite(*values: Any) -> bool:
    return all(v is not None and np.isfinite(float(v)) for v in values)


def _errors(frame: pd.DataFrame, key: str) -> np.ndarray | None:
    """Asymmetric error bars ``[value - low, high - value]`` of ``key``; None without both ends."""
    low, high = f"{key}_low", f"{key}_high"
    if low not in frame or high not in frame:
        return None
    lower = (frame[key] - frame[low]).clip(lower=0).fillna(0.0).to_numpy(float)
    upper = (frame[high] - frame[key]).clip(lower=0).fillna(0.0).to_numpy(float)
    return np.vstack((lower, upper))


def _hide_unused(axes: np.ndarray, used: int) -> None:
    for ax in axes.flat[used:]:
        ax.set_visible(False)


def _plain_log(axis: Any, low: float, high: float, max_ticks: int = 6) -> None:
    """Ticks of a log axis as plain numbers (0.5, 1, 2 rather than powers of ten): the densest of the steps
    1-2-5, 1-3 and 1 per decade that puts at most ``max_ticks`` ticks between ``low`` and ``high``."""
    subs: tuple[float, ...] = (1.0,)
    if 0 < low < high:
        decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        for candidate in ((1.0, 2.0, 5.0), (1.0, 3.0)):
            if sum(low <= s * 10.0**d <= high for d in decades for s in candidate) <= max_ticks:
                subs = candidate
                break
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _overlap(a: Bbox, b: Bbox) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(width, 0.0) * max(height, 0.0)


def _distance(box: Bbox, point: Sequence[float]) -> float:
    """Distance of a point from a box (0 inside)."""
    return math.hypot(max(box.x0 - point[0], 0.0, point[0] - box.x1), max(box.y0 - point[1], 0.0, point[1] - box.y1))


def place_labels(ax: Any, xs: Sequence[float], ys: Sequence[float], texts: Sequence[str],
                 extra: Sequence[tuple[float, float]] = (), fontsize: float = 7.0) -> None:  # fmt: skip
    """Annotate every point with its text where it covers the least: eight positions around the point at
    one, two and three and a half times the distance (the far ones with a leader line), scored by the area
    shared with the labels placed before, with the other points (and ``extra`` points) and outside the axes;
    points closer than a marker share one label (texts joined by commas). Call it last: the figure is drawn
    once to fix the limits and the layout before the boxes are measured."""
    if not len(texts):
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(np.column_stack([xs, ys]).astype(float))
    half = 3.0 * fig.dpi / 72.0  # pixels: half the side of the box kept free around a point
    groups: list[list[int]] = []  # points closer than a marker share one label ("0.1, 1")
    for i, point in enumerate(points):
        near = next((g for g in groups if np.abs(points[g[0]] - point).max() < half), None)
        if near is None:
            groups.append([i])
        else:
            near.append(i)
    if len(extra):
        points = np.vstack([points, ax.transData.transform(np.asarray(extra, dtype=float).reshape(-1, 2))])
    marks = [Bbox.from_extents(px - half, py - half, px + half, py + half) for px, py in points]
    frame = ax.get_window_extent(renderer)
    pad = 2.5 * fig.dpi / 72.0  # pixels: free margin around a label
    placed: list[Bbox] = []
    for group in groups:
        i, text = group[0], ", ".join(texts[j] for j in group)
        best: tuple[float, Any, Bbox] | None = None
        for scale in (1.0, 2.0, 3.5):
            for dx, dy in OFFSETS:
                arrow = {"arrowstyle": "-", "lw": 0.5, "color": PALETTE["dark"], "shrinkA": 0, "shrinkB": 2}
                ann = ax.annotate(text, (xs[i], ys[i]), xytext=(scale * dx, scale * dy), textcoords="offset points",
                                  ha="left" if dx > 0 else "right" if dx < 0 else "center",
                                  va="bottom" if dy > 0 else "top" if dy < 0 else "center", fontsize=fontsize,
                                  arrowprops=arrow if scale > 1 else None)  # fmt: skip
                ann.update_positions(renderer)
                box = Text.get_window_extent(ann, renderer).padded(pad)
                cost = sum(_overlap(box, b) for b in placed)
                cost += sum(_overlap(box, m) for j, m in enumerate(marks) if j not in group)
                cost += box.width * box.height - _overlap(box, frame)
                own = _distance(box, points[i])
                if any(_distance(box, p) < own + pad for j, p in enumerate(points) if j not in group):
                    cost += 0.5 * box.width * box.height  # nearer to another point than to its own: ambiguous
                if best is None or cost < best[0] - 1e-6:
                    if best is not None:
                        best[1].remove()
                    best = (cost, ann, box)
                else:
                    ann.remove()
                if best[0] <= 0.0:
                    break
            if best is not None and best[0] <= 0.0:
                break
        if best is not None:
            placed.append(best[2])


# ------------------------------------------------------------------------------------- 1 E1 scatter

SHARES = (("stable", PALETTE["green"], "stable"), ("unstable", PALETTE["vermillion"], "unstable"),
          ("outside", PALETTE["sky"], "outside the band"), ("none", PALETTE["grey"], "no equilibrium"))  # fmt: skip


def rmse_vs_instability(e1: pd.DataFrame, reference: str = "idm") -> Figure:
    """E1, one point per model: spacing RMSE (x) against the share of unstable equilibria among the
    equilibria found (y) with their intervals, the reference marked; and the band shares as stacked bars
    (the rest of a bar: speeds whose equilibria are indifferent, |G| = 1 at every frequency)."""
    points = e1[e1["rmse_s"].notna() & e1["unstable_eq"].notna()]
    bars = e1[e1[[s[0] for s in SHARES]].notna().any(axis=1)]
    if points.empty and bars.empty:
        raise ValueError("table e1 holds no RMSE and no shares")
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1), gridspec_kw={"width_ratios": [1.4, 1]})
    for _, row in points.iterrows():
        is_ref = row["model"] == reference
        part = points.loc[[row.name]]
        left.errorbar(row["rmse_s"], row["unstable_eq"], xerr=_errors(part, "rmse_s"),
                      yerr=_errors(part, "unstable_eq"), fmt="s" if is_ref else "o", ms=6 if is_ref else 4,
                      color=color(row["model"]), mfc="white" if is_ref else color(row["model"]), elinewidth=0.8,
                      capsize=2, zorder=3)  # fmt: skip
    left.set_xlabel("spacing RMSE of the test parts (m)")
    left.set_ylabel("unstable among the equilibria found")
    left.set_ylim(-0.05, 1.05)
    bottom = np.zeros(len(bars))
    for key, colour, label in SHARES:
        values = bars[key].fillna(0.0).to_numpy(float)
        right.bar(range(len(bars)), values, bottom=bottom, color=colour, label=label, width=0.75)
        bottom += values
    rest = np.clip(1.0 - bottom, 0.0, 1.0)
    if (rest > 0.01).any():
        right.bar(range(len(bars)), rest, bottom=bottom, color="white", edgecolor=PALETTE["dark"], hatch="////",
                  lw=0.5, width=0.75, label="indifferent")  # fmt: skip
    right.set_xticks(range(len(bars)), [name(m) for m in bars["model"]], rotation=60, ha="right")
    right.set_ylabel("share of the grid speeds in support")
    right.set_ylim(0, 1.0)
    right.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, columnspacing=0.8, handlelength=1.0,
                 borderaxespad=0.0)  # fmt: skip
    labels = [name(m) + (" (reference)" if m == reference else "") for m in points["model"]]
    place_labels(left, points["rmse_s"].to_numpy(float), points["unstable_eq"].to_numpy(float), labels)
    return fig


# ------------------------------------------------------------------------------------- 2 E2 trade-off


def e2_tradeoff(
    sweep: pd.DataFrame, e1_points: Mapping[str, tuple[float, float]], architectures: Sequence[str]
) -> Figure:
    """Per architecture the sweep: test RMSE (x) against the share stable inside the band (y), the weights
    connected and labelled, the point of E1 without penalty, the chosen weight ringed."""
    present = [a for a in architectures if (sweep["architecture"] == a).any() or a in e1_points]
    if not present:
        raise ValueError("no architecture in the sweep")
    cols = 3
    rows = math.ceil(len(present) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 2.4 * rows), squeeze=False)
    labels = []
    for ax, arch in zip(axes.flat, present):
        part = sweep[(sweep["architecture"] == arch) & sweep["test_rmse_s"].notna() & sweep["stable"].notna()]
        part = part.sort_values("weight")
        ax.plot(part["test_rmse_s"], part["stable"], "-o", color=color(arch), ms=4)
        chosen = part[part["chosen"].astype(str).str.lower().isin(["true", "1"])]
        ax.plot(chosen["test_rmse_s"], chosen["stable"], "o", ms=11, mfc="none", mec=PALETTE["black"], mew=1.0)
        extra = []
        if arch in e1_points and _finite(*e1_points[arch]):
            ax.plot(*e1_points[arch], "s", ms=6, mfc="white", mec=PALETTE["black"], mew=1.0)
            extra.append(tuple(float(v) for v in e1_points[arch]))
        ax.set_ylim(-0.05, 1.08)
        _label(ax, name(arch))
        labels.append((ax, part, extra))
    for ax in axes[:, 0]:
        ax.set_ylabel("stable inside the band")
    for ax in axes[-1, :]:
        ax.set_xlabel("test spacing RMSE (m)")
    _hide_unused(axes, len(present))
    ring = {"ls": "none", "mfc": "none", "mec": PALETTE["black"]}
    handles = [Line2D([], [], color=PALETTE["dark"], marker="o", ms=4, label="penalty weights (labelled)"),
               Line2D([], [], marker="o", ms=11, label="chosen weight", **ring),
               Line2D([], [], marker="s", ms=6, label="E1, no penalty", **{**ring, "mfc": "white"})]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    for ax, part, extra in labels:
        place_labels(ax, part["test_rmse_s"].to_numpy(float), part["stable"].to_numpy(float),
                     [f"{w:g}" for w in part["weight"]], extra)  # fmt: skip
    return fig


# ------------------------------------------------------------------------------------- 3 gain curves


def gain_curves(curves: Mapping[str, Mapping[str, Any]], threshold: float = 1.02) -> Figure:
    """|G(omega)| of the numerical frequency response: per architecture the curve of every speed in support
    (thin) and their median (thick), without (E1) and with penalty (chosen E2 weight)."""
    present = [a for a, c in curves.items() if c.get("without") or c.get("with")]
    if not present:
        raise ValueError("no stability.json with gains")
    cols = 4
    rows = math.ceil((len(present) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 2.1 * rows), squeeze=False, sharex=True)
    for ax, arch in zip(axes.flat, present):
        omega = np.asarray(curves[arch]["omega"], dtype=float)
        for key, colour in (("without", WITHOUT), ("with", WITH)):
            gains = [np.asarray(g, dtype=float) for g in curves[arch].get(key) or [] if len(g) == len(omega)]
            for g in gains:
                ax.plot(omega, g, color=colour, lw=0.4, alpha=0.35)
            if gains:
                ax.plot(omega, np.nanmedian(np.vstack(gains), axis=0), color=colour, lw=2.0)
        ax.axhline(threshold, color=PALETTE["dark"], ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax.xaxis, *ax.get_xlim(), max_ticks=4)
        _plain_log(ax.yaxis, *ax.get_ylim())
        _label(ax, name(arch))
    for ax in axes[:, 0]:
        ax.set_ylabel("|G(ω)|")
    for ax in axes[-1, :]:
        ax.set_xlabel("ω (rad/s)")
        ax.xaxis.set_tick_params(labelbottom=True)
    legend_ax = axes.flat[len(present)]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], color=WITHOUT, lw=2, label="E1, no penalty (median)"),
        Line2D([], [], color=WITH, lw=2, label="E2, chosen weight (median)"),
        Line2D([], [], color=PALETTE["grey"], lw=0.5, label="one speed in support"),
        Line2D([], [], color=PALETTE["dark"], ls="--", lw=0.8, label=f"|G| = {threshold:g}"),
    ], loc="center")  # fmt: skip
    _hide_unused(axes, len(present) + 1)
    return fig


# ------------------------------------------------------------------------------------- 4 growth curves


def growth_curves(panels: Sequence[Mapping[str, Any]]) -> Figure:
    """Platoon test: speed standard deviation per platoon position, the empirical curve of every profile
    against the model curves (mean over the folds) of the laws of the matching driving mode; a model curve
    that ends early (NaN: a platoon of a fold collided at the next position) ends in a cross."""
    panels = [p for p in panels if p.get("models") or p.get("empirical") is not None]
    if not panels:
        raise ValueError("no platoon.json")
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 2.3 * rows), squeeze=False)
    seen: dict[str, Line2D] = {}
    ends = False
    for ax, panel in zip(axes.flat, panels):
        for label, curve in (panel.get("models") or {}).items():
            values = np.asarray(curve["std"], dtype=float)
            keep = np.isfinite(values) & (values > 0)
            line = ax.plot(np.flatnonzero(keep), values[keep], color=color(curve["model"]), lw=1.0,
                           ls="--" if curve["penalised"] else "-")[0]  # fmt: skip
            seen.setdefault(label, Line2D([], [], color=line.get_color(), ls=line.get_linestyle(), label=label))
            if keep.any() and not keep[-1]:
                last = int(np.flatnonzero(keep)[-1])
                ax.plot(last, values[last], "x", color=line.get_color(), ms=5, mew=1.2, zorder=5)
                ends = True
        empirical = panel.get("empirical")
        if empirical is not None:
            values = np.asarray(empirical, dtype=float)
            keep = np.isfinite(values) & (values > 0)
            ax.plot(np.flatnonzero(keep), values[keep], "o-", color=PALETTE["black"], ms=3, lw=1.0, zorder=4)
            data = Line2D([], [], color=PALETTE["black"], marker="o", ms=3, label="data (OpenACC)")
            seen.setdefault("data (OpenACC)", data)
        if ax.has_data():
            ax.set_yscale("log")
            _plain_log(ax.yaxis, *ax.get_ylim())
        _label(ax, panel["label"])
    for ax in axes[:, 0]:
        ax.set_ylabel("speed std (m/s)")
    for ax in axes[-1, :]:
        ax.set_xlabel("platoon position")
    _hide_unused(axes, len(panels))
    handles = list(seen.values())
    if ends:
        handles.append(Line2D([], [], ls="none", marker="x", color=PALETTE["dark"], mew=1.2,
                              label="collision at the next position in a fold"))  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    return fig


# ------------------------------------------------------------------------------------ 5 speed contours


def speed_contours(fields: Sequence[Mapping[str, Any]], window: Sequence[float] | None, vmax: float) -> Figure:
    """x-t speed fields on one colour scale; cells without a vehicle grey; the analysis window marked."""
    fields = [f for f in fields if f.get("speed") is not None]
    if not fields:
        raise ValueError("no trajectories")
    cols = 2
    rows = math.ceil(len(fields) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 1.9 * rows), squeeze=False, sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis").with_extremes(bad="#D9D9D9")  # cells without a vehicle
    image = None
    for ax, field in zip(axes.flat, fields):
        x_edges, t_edges = np.asarray(field["x_edges"]), np.asarray(field["t_edges"])
        image = ax.imshow(np.ma.masked_invalid(np.asarray(field["speed"], dtype=float)), origin="lower", aspect="auto",
                          extent=(t_edges[0], t_edges[-1], x_edges[0], x_edges[-1]), cmap=cmap, vmin=0.0, vmax=vmax,
                          interpolation="nearest")  # fmt: skip
        if window is not None:
            for t in window:
                ax.axvline(t, color="white", ls="--", lw=0.9)
        ax.text(0.02, 0.95, field["label"], transform=ax.transAxes, ha="left", va="top", fontweight="bold",
                color="white", bbox={"facecolor": "black", "alpha": 0.4, "pad": 1.5, "lw": 0})  # fmt: skip
    for ax in axes[:, 0]:
        ax.set_ylabel("x (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("time from the start of the period (s)")
    _hide_unused(axes, len(fields))
    fig.colorbar(image, ax=axes, label="speed (m/s)", shrink=0.8)
    return fig


# --------------------------------------------------------------------------------- 6 fundamental diagrams


def fundamental_diagrams(
    panels: Sequence[Mapping[str, Any]], truth_curve: tuple[Sequence[float], Sequence[float]] | None, bin_width: float
) -> Figure:
    """Density-flow scatter of the Edie cells (lanes together) per panel, the binned mean flow of the
    ground truth (bin centres) on every panel."""
    panels = [p for p in panels if p.get("density") is not None and len(p["density"])]
    if not panels:
        raise ValueError("no trajectories")
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 2.3 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, panel in zip(axes.flat, panels):
        ax.scatter(panel["density"], panel["flow"], s=6, color=color(panel["key"]), alpha=0.6, lw=0)
        if truth_curve is not None and len(truth_curve[0]):
            centres = np.asarray(truth_curve[0], dtype=float) + 0.5 * bin_width
            ax.plot(centres, truth_curve[1], "-", color=PALETTE["black"], lw=1.3)
        _label(ax, panel["label"])
    for ax in axes[:, 0]:
        ax.set_ylabel("flow, all lanes (veh/h)")
    for ax in axes[-1, :]:
        ax.set_xlabel("density, all lanes (veh/km)")
        ax.xaxis.set_tick_params(labelbottom=True)
    _hide_unused(axes, len(panels))
    curve = Line2D([], [], color=PALETTE["black"], lw=1.3, label="ground truth, mean flow per density bin")
    fig.legend(handles=[curve], loc="outside lower center")
    return fig


# ------------------------------------------------------------------------------ 7 macro error vs instability


def macro_error_vs_instability(panels: Sequence[Mapping[str, Any]]) -> Figure:
    """Per law: share of unstable equilibria of its members (x) against a macro error with its interval (y),
    laws with collisions marked, the correlations over the laws in the legend above each panel."""
    if not any(len(p["frame"]) for p in panels):
        raise ValueError("no law with both numbers")
    # one panel per row: the labels of up to twenty laws need the width of the whole figure
    fig, axes = plt.subplots(len(panels), 1, figsize=(FULL_WIDTH, 3.6 * len(panels)), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        frame = panel["frame"]
        for _, row in frame.iterrows():
            collides = bool(row.get("collides") is True or str(row.get("collides")).lower() == "true")
            part = frame.loc[[row.name]]
            ax.errorbar(row["x"], row["y"], yerr=_errors(part, "y"), fmt="s" if collides else "o", ms=5,
                        color=color(row["law"]), mfc="white" if collides else color(row["law"]), elinewidth=0.8,
                        capsize=2, zorder=3)  # fmt: skip
        texts = list(panel.get("correlations") or []) or ["correlations: not available"]
        ax.legend(handles=[Line2D([], [], ls="none", label=text) for text in texts], loc="lower left",
                  bbox_to_anchor=(0.0, 1.01), handlelength=0, handletextpad=0, borderaxespad=0.0)  # fmt: skip
        ax.set_xlim(-0.05, 1.15)  # room for the labels of the laws near 1
        ax.set_xlabel("unstable among the equilibria of the members")
        ax.set_ylabel(panel["ylabel"])
        if not len(frame):
            ax.text(0.5, 0.5, panel.get("empty", "no data"), transform=ax.transAxes, ha="center", va="center")
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=PALETTE["dark"], label="no collisions"),
                        Line2D([], [], ls="none", marker="s", mfc="white", color=PALETTE["dark"], label="collisions")],
               loc="outside lower center", ncol=2)  # fmt: skip
    for ax, panel in zip(axes.flat, panels):
        frame = panel["frame"]
        if len(frame):
            place_labels(ax, frame["x"].to_numpy(float), frame["y"].to_numpy(float), [str(v) for v in frame["law"]],
                         fontsize=6.5)  # fmt: skip
    return fig

