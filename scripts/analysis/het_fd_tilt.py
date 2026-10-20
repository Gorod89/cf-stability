"""FD-matched member weights of the heterogeneous certified hybrid (docs/m11_contract.md, D132, hypothesis H-N5).

    python scripts/analysis/het_fd_tilt.py                     # fit on i80_p1, write the tilt and the table
    python scripts/analysis/het_fd_tilt.py --device cuda       # the equilibria on the GPU

The law ``residual_idm_certified_het`` (D118) draws one certified per-event core per vehicle uniformly. The law
``residual_idm_certified_het_fd`` draws the same cores with the weights ``w_i ~ exp(beta_T T_i + beta_s s0_i)``. This
script fits ``(beta_T, beta_s)`` to the fundamental diagram of the ground truth of a scenario (``i80_p1``):

* cores: the table of the uniform law (``runs/corridor/laws/<law>.npz``: the certified cores that pass the recheck,
  in the order the law draws them) and its five fold members (``<law>.json``: the residuals);
* equilibrium gap of a vehicle with core ``i`` and member ``m`` at the speed ``v``: the root of
  ``idm_i(s, 0, v) + r_m(s, 0, v) = 0`` (the closed-form IDM gap ``s = s*/sqrt(1 - (v/v0)^delta + r/a)``,
  ``s* = s0 + v T``, iterated over the residual of the member until it moves by less than 1e-7 m; the residual
  changes by at most ``B_s`` (5e-3 1/s) per metre, so the iteration contracts fast); ``s_i(v)`` = the mean over the
  members (the member of a vehicle is drawn uniformly and independently of its core);
* the mixture: mean spacing ``S(v) = sum_i w_i s_i(v) + L`` (``L``: the mean vehicle length of the scenario's
  demand), per-lane density ``1 / S(v)``, total density ``k = n_lanes / S(v)`` (``n_lanes``: the mean lane count of
  the cells of the fundamental diagram, 100 m long between ``x_in`` and ``x_out``: the diagram of ``macro.json`` sums
  the lanes), flow ``q = k v``; ``S`` rises with ``v``, so at every density bin of the truth (its centre, lower edge +
  half the bin width) ``v`` is found by linear interpolation on the speed grid and ``q_b = k_b v_b``;
* distance ``D(beta) = sum_b (q_b(beta) - q_truth_b)^2`` over the bins of the truth, reported as the RMSE over the
  bins divided by the mean flow of the truth's cells (the ``fd`` component of the macro error); fit: L-BFGS-B
  (scipy) from the uniform weights (``beta = 0``) within ``|beta| <= 5``; a grid of 41 x 41 betas is reported as a
  check of the landscape.

Writes ``data/calibration/ngsim_i80/het_fd_tilt.json`` (``beta_T``, ``beta_s``, the distances, the settings: the file
the law ``residual_idm_certified_het_fd`` reads, ``configs/corridor/laws.yaml``), ``runs/_tables/m11/het_fd_fit.{csv,
md}`` and the figure ``runs/_tables/m11/het_fd_fit.{png,pdf}`` (the uniform and the tilted mixture against the binned
truth; the curve of the cores alone, without the residual, as a dashed line).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cf_stability.eval.tables import Column, Table, write_table  # noqa: E402
from cf_stability.utils import config_hash, read_json, resolve_path, write_json  # noqa: E402

KEYS = ("v0", "T", "s0", "a", "b")
BOUND = 5.0  # |beta| <= 5 (D132)
SPEEDS = (0.0, 25.0, 0.1)  # m/s: the grid of the equilibria (the congested truth of I-80 lies below 11 m/s)
TOLERANCE = 1e-7  # m: the fixed point of the equilibrium gap
DELTA = 4.0


def equilibrium_gaps(
    params: np.ndarray, residuals: Sequence[Any] | None, speeds: np.ndarray, device: str = "cpu",
    chunk: int = 200_000, max_iter: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:  # fmt: skip
    """``s_i(v)`` ``[cores, speeds]``: the equilibrium gap of core ``i`` with the residual of every member of
    ``residuals`` (ResidualIDM models; their mean), or of the core alone (``residuals`` None), in float64."""
    p = torch.as_tensor(params, dtype=torch.float64, device=device)
    v = torch.as_tensor(speeds, dtype=torch.float64, device=device)
    v0, T, s0, a, _ = (p[:, j : j + 1] for j in range(5))
    s_star = s0 + v[None, :] * T
    free = 1.0 - (v[None, :] / v0) ** DELTA
    core = s_star / torch.sqrt(free)
    if not residuals:
        return core.cpu().numpy(), {"iterations": [0], "max_step": [0.0]}
    total = torch.zeros_like(core)
    info: dict[str, Any] = {"iterations": [], "max_step": []}
    v_flat = v[None, :].expand_as(core).reshape(-1)
    for model in residuals:
        m = model.to(device).double().eval()
        s = core.clone()
        for it in range(max_iter):
            flat = s.reshape(-1)
            r = torch.empty_like(flat)
            with torch.no_grad():
                for start in range(0, len(flat), chunk):
                    end = min(start + chunk, len(flat))
                    state = torch.stack((flat[start:end], torch.zeros_like(flat[start:end]), v_flat[start:end]), -1)
                    r[start:end] = m.residual(state[:, None, :])
            arg = free + r.reshape(s.shape) / a
            if not (arg > 0).all():
                raise ValueError("no equilibrium gap: 1 - (v/v0)^delta + r/a <= 0 on the speed grid")
            new = s_star / torch.sqrt(arg)
            step = float((new - s).abs().max())
            s = new
            if step < TOLERANCE:
                break
        info["iterations"].append(it + 1)
        info["max_step"].append(step)
        total += s
    return (total / len(residuals)).cpu().numpy(), info


def weights(params: np.ndarray, beta: Sequence[float]) -> np.ndarray:
    """``w_i ~ exp(beta_T T_i + beta_s s0_i)``, normalised (log-sum-exp)."""
    logit = float(beta[0]) * params[:, 1] + float(beta[1]) * params[:, 2]
    logit = logit - logit.max()
    w = np.exp(logit)
    return w / w.sum()


def mixture_flow(
    gaps: np.ndarray, w: np.ndarray, speeds: np.ndarray, length: float, lanes: float, density: np.ndarray
) -> np.ndarray:  # fmt: skip
    """Flow (veh/h) of the mixture at the total densities ``density`` (veh/km): ``v`` with ``lanes / S(v) = k``
    (``S`` = mean gap + ``length``, rising with ``v``), ``q = k v``; ``v`` clipped to the ends of the grid."""
    spacing = w @ gaps + length  # [speeds], m
    target = lanes * 1000.0 / density  # m per vehicle and lane at the density
    if np.any(np.diff(spacing) <= 0):
        raise ValueError("the mean spacing of the mixture does not rise with the speed on the grid")
    v = np.interp(target, spacing, speeds)  # clipped to the grid
    return density * v * 3.6


def fd_distance(q: np.ndarray, truth: np.ndarray, mean_flow: float) -> tuple[float, float]:
    """``sum (q - truth)^2`` (veh^2/h^2) and the RMSE over the bins / the mean flow of the truth."""
    squares = float(np.sum((q - truth) ** 2))
    return squares, math.sqrt(squares / len(truth)) / mean_flow


def lane_count(geometry: Mapping[str, Any], cell_dx: float) -> tuple[float, list[float]]:
    """The mean lane count of the cells of the diagram (cells of ``cell_dx`` between ``x_in`` and ``x_out``, aligned
    with the multiples of ``cell_dx``; the lanes of every edge weighted by its share of the cell)."""
    x_in, x_out = float(geometry["x_in"]), float(geometry["x_out"])
    inner = [cell_dx * k for k in range(math.floor(x_in / cell_dx) + 1, math.ceil(x_out / cell_dx))]
    edges = sorted({x_in, x_out, *[x for x in inner if x_in < x < x_out]})
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lanes = 0.0
        for edge in geometry["edges"]:
            overlap = min(hi, float(edge["x1"])) - max(lo, float(edge["x0"]))
            if overlap > 0:
                lanes += overlap * float(edge["lanes"])
        counts.append(lanes / (hi - lo))
    return float(np.mean(counts)), counts


def fit(objective, start: Sequence[float]) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    res = minimize(objective, np.asarray(start, dtype=np.float64), method="L-BFGS-B",
                   bounds=[(-BOUND, BOUND)] * 2, options={"maxiter": 500})  # fmt: skip
    return res.x, {"success": bool(res.success), "message": str(res.message), "nfev": int(res.nfev), "nit": int(res.nit)}


def figure(path: Path, density: np.ndarray, truth: np.ndarray, curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
           title: str) -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(density, truth, color="black", s=18, zorder=3, label="ground truth (bins)")
    styles = {"uniform": ("#1f77b4", "-"), "tilted": ("#d62728", "-"), "uniform, cores alone": ("#1f77b4", ":"),
              "tilted, cores alone": ("#d62728", ":")}  # fmt: skip
    for name, (k, q) in curves.items():
        color, line = styles.get(name, ("grey", "-"))
        ax.plot(k, q, color=color, linestyle=line, linewidth=1.6, label=name)
    ax.set_xlabel("density (veh/km, all lanes)")
    ax.set_ylabel("flow (veh/h, all lanes)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="i80_p1")
    parser.add_argument("--law", default="residual_idm_certified_het", help="the uniform law: its cores and members")
    parser.add_argument("--laws-root", default="runs/corridor/laws")
    parser.add_argument("--scenarios-root", default="runs/corridor/scenarios")
    parser.add_argument("--out", default="data/calibration/ngsim_i80/het_fd_tilt.json")
    parser.add_argument("--tables", default="runs/_tables/m11")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    from cf_stability.models import load_model

    law_path = resolve_path(args.laws_root) / f"{args.law}.json"
    law = read_json(law_path)
    with np.load(law_path.parent / law["table"]) as data:
        params = np.stack([np.asarray(data[k], dtype=np.float64) for k in KEYS], axis=1)
        index = np.asarray(data["index"], dtype=np.int64)
    members = [load_model(resolve_path(m["model"])) for m in law["members"]]
    scenario_dir = resolve_path(args.scenarios_root) / args.scenario
    truth_macro = read_json(scenario_dir / "macro.json")
    fd = truth_macro["fd"]
    width = float(fd["bin_width"])
    density = np.asarray(fd["density_bins"], dtype=np.float64) + 0.5 * width
    truth = np.asarray(fd["flow"], dtype=np.float64)
    mean_flow = float(fd["mean_flow"])
    with np.load(scenario_dir / "vehicles_truth.npz") as data:
        length = float(np.mean(data["length"]))
    geometry = read_json(scenario_dir / "scenario.json")["geometry"]
    lanes, per_cell = lane_count(geometry, float(truth_macro["config"]["cell_dx"]))

    speeds = np.arange(round(SPEEDS[0] / SPEEDS[2]), round(SPEEDS[1] / SPEEDS[2]) + 1) * SPEEDS[2]
    t_eq = time.perf_counter()
    gaps, info = equilibrium_gaps(params, members, speeds, args.device)
    core_gaps, _ = equilibrium_gaps(params, None, speeds)
    t_eq = time.perf_counter() - t_eq

    def objective(beta: np.ndarray, which: np.ndarray = gaps) -> float:
        return fd_distance(mixture_flow(which, weights(params, beta), speeds, length, lanes, density), truth,
                           mean_flow)[0]  # fmt: skip

    uniform = np.zeros(2)
    beta, opt = fit(objective, uniform)
    grid_values = np.linspace(-BOUND, BOUND, 41)
    landscape = np.array([[objective(np.array([bt, bs])) for bs in grid_values] for bt in grid_values])
    k_best = np.unravel_index(np.argmin(landscape), landscape.shape)
    grid_best = np.array([grid_values[k_best[0]], grid_values[k_best[1]]])
    beta_cores, opt_cores = fit(lambda b: objective(b, core_gaps), uniform)  # the cores alone, for information

    rows = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    k_curve = np.linspace(density.min() - 30.0, density.max() + 30.0, 200)
    for name, b, which in (("uniform", uniform, gaps), ("tilted", beta, gaps), ("grid best", grid_best, gaps),
                           ("uniform, cores alone", uniform, core_gaps), ("tilted, cores alone", beta, core_gaps),
                           ("tilted on the cores alone", beta_cores, core_gaps)):  # fmt: skip
        w = weights(params, b)
        q = mixture_flow(which, w, speeds, length, lanes, density)
        squares, rel = fd_distance(q, truth, mean_flow)
        effective = float(1.0 / np.sum(w**2))
        rows.append({
            "mixture": name, "residual": "members" if which is gaps else "none", "beta_T": float(b[0]),
            "beta_s": float(b[1]), "fd_sse": squares, "fd_rmse_rel": rel, "effective_cores": effective,
            "mean_T": float(w @ params[:, 1]), "mean_s0": float(w @ params[:, 2]),
            "mean_spacing_5": float(np.interp(5.0, speeds, w @ which)) + length,
            "mean_spacing_10": float(np.interp(10.0, speeds, w @ which)) + length,
        })  # fmt: skip
        if name in ("uniform", "tilted", "uniform, cores alone", "tilted, cores alone"):
            curves[name] = (k_curve, mixture_flow(which, w, speeds, length, lanes, k_curve))
    frame = pd.DataFrame(rows)
    before, after = rows[0], rows[1]
    settings = {
        "scenario": args.scenario, "law": args.law, "law_file": law_path.relative_to(resolve_path(".")).as_posix(),
        "n_cores": int(len(params)), "members": [m["run"] for m in law["members"]],
        "vehicle_length": length, "lanes": lanes, "lanes_per_cell": per_cell, "speeds": list(SPEEDS),
        "density": "bin centre (lower edge + bin_width / 2)", "bound": BOUND, "start": [0.0, 0.0],
        "delta": DELTA, "tolerance_m": TOLERANCE,
    }  # fmt: skip
    payload = {
        "kind": "het_fd_tilt", "beta_T": float(beta[0]), "beta_s": float(beta[1]),
        "fd_rmse_rel_uniform": before["fd_rmse_rel"], "fd_rmse_rel_tilted": after["fd_rmse_rel"],
        "fd_sse_uniform": before["fd_sse"], "fd_sse_tilted": after["fd_sse"],
        "effective_cores_tilted": after["effective_cores"], "optimizer": opt,
        "grid_best": {"beta_T": float(grid_best[0]), "beta_s": float(grid_best[1]),
                      "fd_rmse_rel": rows[2]["fd_rmse_rel"]},
        "cores_alone": {"beta_T": float(beta_cores[0]), "beta_s": float(beta_cores[1]),
                        "fd_rmse_rel": rows[5]["fd_rmse_rel"], "optimizer": opt_cores},
        "equilibria": {"iterations": info["iterations"], "max_step": info["max_step"], "seconds": t_eq},
        "truth": {"density_bins": fd["density_bins"], "flow": fd["flow"], "count": fd["count"],
                  "bin_width": width, "mean_flow": mean_flow},
        "flow_uniform": mixture_flow(gaps, weights(params, uniform), speeds, length, lanes, density).tolist(),
        "flow_tilted": mixture_flow(gaps, weights(params, beta), speeds, length, lanes, density).tolist(),
        "cores_index": index.tolist(), "settings": settings,
    }  # fmt: skip
    payload["key"] = config_hash({"settings": settings, "params": params.tolist()})
    write_json(resolve_path(args.out), payload)

    out = resolve_path(args.tables)
    notes = [
        f"Law {args.law} (D118): {len(params)} certified per-event cores of {law['source']['cores']} (those that pass "
        "the recheck), the residuals of its five fold members; weights w_i ~ exp(beta_T T_i + beta_s s0_i) (D132).",
        f"Truth: the binned fundamental diagram of {args.scenario} (macro.json fd: {len(truth)} bins of {width:g} "
        f"veh/km, all lanes; mean flow of the cells {mean_flow:.0f} veh/h). Mixture: mean spacing S(v) = sum_i w_i s_i(v) "
        f"+ L, L = {length:.3f} m (mean length of the demand), s_i(v) = mean over the members of the equilibrium gap of "
        f"core i with the member's residual (the cores alone: without residual); total density k = n_lanes / S(v), "
        f"n_lanes = {lanes:.3f} (mean lane count of the cells of the diagram: {', '.join(f'{c:.2f}' for c in per_cell)});"
        " flow q = k v at the bin centres.",
        "fd_rmse_rel: RMSE of the flow over the bins / mean flow of the truth (the fd component of the macro error); "
        "fd_sse: the sum of squares that is minimised (veh²/h²). Fit: L-BFGS-B from beta = 0 within |beta| <= 5; "
        "grid best: the smallest of a 41 x 41 grid on [-5, 5]² (a check); effective cores: 1 / sum w_i²; mean T, s0: "
        "weighted means; mean spacing at 5 and 10 m/s (m).",
    ]  # fmt: skip
    columns = [
        Column("mixture", "mixture", "text"), Column("residual", "residual", "text"),
        Column("beta_T", "β_T (1/s)", digits=3), Column("beta_s", "β_s (1/m)", digits=3),
        Column("fd_rmse_rel", "FD RMSE / mean flow", digits=4), Column("fd_sse", "FD SSE", digits=0),
        Column("effective_cores", "effective cores", digits=0), Column("mean_T", "mean T (s)", digits=3),
        Column("mean_s0", "mean s0 (m)", digits=3), Column("mean_spacing_5", "spacing at 5 m/s (m)", digits=2),
        Column("mean_spacing_10", "spacing at 10 m/s (m)", digits=2),
    ]  # fmt: skip
    write_table(Table("het_fd_fit", "FD-matched weights of the heterogeneous certified hybrid (D132)", notes, frame,
                      columns), out)  # fmt: skip
    title = (f"{args.scenario}: equilibrium flow of the mixture, uniform (FD error {before['fd_rmse_rel']:.3f})\n"
             f"and tilted (β_T = {beta[0]:.2f} 1/s, β_s = {beta[1]:.2f} 1/m; FD error {after['fd_rmse_rel']:.3f})")  # fmt: skip
    figure(out / "het_fd_fit", density, truth, curves, title)
    print(f"het_fd_tilt: beta_T {beta[0]:+.4f}  beta_s {beta[1]:+.4f}  FD RMSE/mean flow {before['fd_rmse_rel']:.4f} "
          f"(uniform) -> {after['fd_rmse_rel']:.4f} (tilted); grid best ({grid_best[0]:+.2f}, {grid_best[1]:+.2f}) "
          f"{rows[2]['fd_rmse_rel']:.4f}; cores alone: ({beta_cores[0]:+.3f}, {beta_cores[1]:+.3f}) "
          f"{rows[5]['fd_rmse_rel']:.4f}; equilibria {t_eq:.0f} s, iterations {info['iterations']}; "
          f"{time.perf_counter() - t_start:.0f} s -> {resolve_path(args.out)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
