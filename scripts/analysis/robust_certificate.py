"""Robust (non-equilibrium) certificate of the corridor laws and the residual budget allocation (M11, D130, A3).

    python scripts/analysis/robust_certificate.py              # runs/_tables/m11/robust_certificate{,_members}.{csv,md}
                                                               # and runs/_tables/m11/budget_allocation.{csv,md}
    python scripts/analysis/robust_certificate.py --workers 8 --states 20000

For every member of a law (the fold members of its law file) and region, the bounded-real certificate of
``cf_stability/stability/robust.py`` (common quadratic storage function, ``P = L L^T``, bisection on gamma, scipy,
every certificate verified in float64) of the follower's perturbation dynamics over the derivative box of the region:

* regions of states ``(s, 0, v)`` at the grid speeds 5-30 m/s: ``band`` = the spacing band of the member's training
  context (NGSIM I-80, the band of the audit, ``context["band"]``) at the grid speeds where it exists; ``I(v)`` = the
  feasible gaps of the certificate of the member's IDM core at its ``r_max`` (``certificate.feasible_ends`` clamped to
  [1, 200] m; for the laws without residual, ``r_max = 0``: the equilibrium spacing alone);
* boxes (the rule of D130): the IDM core's derivative ranges over the spacings of the region (exact, the ends of the
  interval) widened by the residual budget ``|dr/dx_j| <= B_j`` (``ResidualIDM.jacobian_bound``) for the hybrids; for
  the MLP and PIDL the ranges of their memoryless partials at 20 000 states drawn uniformly over the region (a
  necessary-condition check: the sampled range lies inside the true one; band only, they have no ``I(v)``);
* scope ``per speed``: one box per grid speed, the member's gamma = the largest over the speeds; ``whole``: one box
  over all speeds of the region together (trajectories that also change their speed);
* supplementary variant ``hull`` (not part of the rule): the polytope that holds the core's derivative curve over the
  spacings (``robust.idm_curve_polytope``: the derivatives move together with the spacing; chord and tangents), summed
  with the residual box, per speed and over the union of the speeds (its convex hull).

The law's gamma is the largest over its members (every vehicle gets one of them); ``infeasible`` when a member's LMI has
no verified certificate (a vertex that is not Hurwitz, or none up to gamma 1e4). Every vertex with ``f_s > 0`` has
``G(0) = 1``, so a gain below 1 cannot be certified: ``string stable`` is a verified certificate at ``gamma = 1 +
1e-3``. The table puts the macro errors of the laws on both corridors (``runs/_tables/m5/laws.csv``) beside them.

Budget allocation (D130): for the margin core of every member of E (``residual_idm_certified``) the budgets ``(B_s,
B_dv, B_v)`` that maximise their sum subject to the certificate (``gamma <= 1 + 1e-3``) on the band (coordinate ascent
with bisection, ``robust.allocate_budget``; box and hull, per speed and whole), compared with the uniform allocation of
``r_max`` 0.3 (the member's own bounds ``r_max lipschitz prod(norms) / (20, 2, 10)``); and, as a supplement, the same
allocation under the pointwise criterion of the a priori certificate on ``I(v)`` (``certificate.guaranteed_margin >= 0``
with the guaranteed signs at every spacing of the scan of ``_a_priori``, no common storage function).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.tables import Column, Table, markdown  # noqa: E402
from cf_stability.stability import robust  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

LAWS = (  # law, label, kind of the boxes
    ("residual_idm_certified_r0.1", "E, r_max 0.1", "core"),
    ("residual_idm_certified_r0.2", "E, r_max 0.2", "core"),
    ("residual_idm_certified", "E (r_max 0.3)", "core"),
    ("residual_idm_certified_r0.5", "E, r_max 0.5", "core"),
    ("residual_idm_free_r0.3", "C", "core"),
    ("residual_idm_margin_free_r0.3", "D", "core"),
    ("idm_core_margin", "B", "core"),
    ("idm_global", "IDM (A)", "core"),
    ("mlp", "MLP", "empirical"),
    ("pidl", "PIDL", "empirical"),
)
V_GRID = tuple(float(v) for v in range(5, 31))
N_TANGENTS = 8
SEED = 20261019


# ------------------------------------------------------------------------------------------------ members


def members_of(law: str, laws_root: Path) -> list[dict[str, Any]]:
    """The members of a law: ``{run, context, core, bounds, r_max, model}`` (``core``: the IDM parameters, ``bounds``:
    the residual's Jacobian bounds, ``model``: the network of an empirical law)."""
    from cf_stability.corridor.laws import IDM_KEYS
    from cf_stability.models import load_model
    from cf_stability.models.idm import IDM
    from cf_stability.models.residual_idm import ResidualIDM
    from cf_stability.stability.certificate import residual_bounds

    payload = read_json(laws_root / f"{law}.json")
    out = []
    if payload.get("members"):
        for member in payload["members"]:
            model = load_model(resolve_path(member["model"]))
            context = read_json(resolve_path(member["run"]) / "metrics.json")["context"]
            entry: dict[str, Any] = {"run": member["run"], "context": context, "core": None, "bounds": None,
                                     "r_max": 0.0, "model": None}  # fmt: skip
            if isinstance(model, ResidualIDM):
                entry.update(core=model.idm.params_dict(), bounds=residual_bounds(model).tolist(), r_max=model.r_max)
            elif isinstance(model, IDM):
                entry.update(core=model.params_dict(), bounds=[0.0, 0.0, 0.0])
            else:
                entry["model"] = model
            out.append(entry)
        return out
    runs = (payload.get("source") or {}).get("core_runs") or []  # idm_core_margin: the cores of these runs
    for params, run in zip(payload["params"], runs):
        out.append({"run": run, "context": read_json(resolve_path(run) / "metrics.json")["context"],
                    "core": {k: float(params[k]) for k in IDM_KEYS}, "bounds": [0.0, 0.0, 0.0], "r_max": 0.0,
                    "model": None})  # fmt: skip
    return out


def regions_of(member: Mapping[str, Any]) -> dict[str, dict[float, tuple[float, float]]]:
    """``band`` and, with a core, ``I(v)`` (empty regions left out)."""
    regions = {"band": robust.region_band(member["context"], V_GRID)}
    if member["core"] is not None:
        regions["I(v)"] = robust.region_feasible(member["core"], member["r_max"], V_GRID)
    return {name: region for name, region in regions.items() if region}


def tasks_of(law: str, label: str, k: int, member: Mapping[str, Any], n_states: int, seed: int) -> list[dict[str, Any]]:
    """The certificate problems of one member: per region, variant (box; hull with a core), scope (every speed and the
    whole region) the vertices of the polytope."""
    tasks = []
    for region_name, region in regions_of(member).items():
        if member["core"] is not None:
            boxes = robust.law_boxes(region, core=member["core"], bounds=member["bounds"])
        else:
            boxes = robust.law_boxes(region, model=member["model"], n_states=n_states, seed=seed + k)
        base = {"law": law, "label": label, "member": k, "run": member["run"], "region": region_name,
                "kind": boxes["kind"]}  # fmt: skip
        for v, box in boxes["per_speed"].items():
            tasks.append({**base, "variant": "box", "scope": "per speed", "v": v, "vertices": box.vertices(),
                          "box": box.as_dict(), "s_low": region[v][0], "s_high": region[v][1]})  # fmt: skip
        tasks.append({**base, "variant": "box", "scope": "whole", "v": math.nan, "vertices": boxes["whole"].vertices(),
                      "box": boxes["whole"].as_dict(), "s_low": math.nan, "s_high": math.nan})  # fmt: skip
        if member["core"] is not None:
            union = []
            for v in sorted(region):
                polygon = robust.idm_curve_polytope(member["core"], v, *region[v], n_tangents=N_TANGENTS)
                points = robust.hull_vertices(robust.widen(polygon, member["bounds"]))
                union.append(polygon)
                tasks.append({**base, "variant": "hull", "scope": "per speed", "v": v, "vertices": points,
                              "box": None, "s_low": region[v][0], "s_high": region[v][1]})  # fmt: skip
            whole = robust.hull_vertices(robust.widen(robust.hull_vertices(np.concatenate(union)), member["bounds"]))
            tasks.append({**base, "variant": "hull", "scope": "whole", "v": math.nan, "vertices": whole, "box": None,
                          "s_low": math.nan, "s_high": math.nan})  # fmt: skip
    return tasks


KEY = ("law", "member", "region", "variant", "scope", "v")  # one polytope
CHECKPOINT = "robust_certificate_polytopes.partial.csv"
FIELDS = (  # the columns of a polytope's result, in a fixed order (the checkpoint is appended to in chunks)
    "law", "label", "member", "run", "region", "kind", "variant", "scope", "v", "s_low", "s_high",
    *(f"box_{name}_{end}" for name in ("f_s", "f_dv", "f_v") for end in ("low", "high")),
    "status", "gamma", "lower_bound", "schur_gain", "hurwitz", "string_stable", "n_vertices", "max_eigenvalue",
    "reason", "seconds",
)


def task_key(row: Mapping[str, Any]) -> tuple:
    v = row["v"]
    return (str(row["law"]), int(row["member"]), str(row["region"]), str(row["variant"]), str(row["scope"]),
            None if v is None or not np.isfinite(float(v)) else round(float(v), 6))  # fmt: skip


def solve(task: Mapping[str, Any]) -> dict[str, Any]:
    """The certificate of one polytope (a worker of the pool); an exception is recorded as status ``error``."""
    t0 = time.perf_counter()
    out = {key: value for key, value in task.items() if key not in ("vertices", "box")}
    out.update({f"box_{key}": value for key, value in (task.get("box") or {}).items()})
    try:
        result = robust.robust_certificate(task["vertices"])
    except Exception as exc:  # noqa: BLE001 - one polytope must not stop the others
        out.update(status="error", gamma=None, lower_bound=None, hurwitz=None, string_stable=False,
                   n_vertices=int(len(task["vertices"])), max_eigenvalue=None, schur_gain=None,
                   reason=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)  # fmt: skip
        return out
    out.update(status=result["status"], gamma=result["gamma"], lower_bound=result["lower_bound"],
               hurwitz=result["hurwitz"], string_stable=result["string_stable"], n_vertices=result["n_vertices"],
               max_eigenvalue=result["max_eigenvalue"], schur_gain=result.get("schur_gain"),
               reason=result.get("reason"), seconds=time.perf_counter() - t0)  # fmt: skip
    return out


def run_tasks(tasks: Sequence[Mapping[str, Any]], out: Path, workers: int) -> pd.DataFrame:
    """Solve the polytopes in a pool of ``workers`` processes, appending every result to the checkpoint file
    ``out/CHECKPOINT``; the polytopes already in it are not solved again (a restart continues)."""
    path = out / CHECKPOINT
    done = pd.read_csv(path) if path.is_file() and path.stat().st_size > 0 else pd.DataFrame()
    known = {task_key(r) for r in done.to_dict("records")} if len(done) else set()
    todo = [t for t in tasks if task_key(t) not in known]
    print(f"POLYTOPES {len(tasks)}: {len(tasks) - len(todo)} in the checkpoint, {len(todo)} to solve, {workers} workers",
          flush=True)  # fmt: skip
    from concurrent.futures import as_completed

    buffer: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def flush() -> None:
        if buffer:
            frame = pd.DataFrame(buffer).reindex(columns=list(FIELDS))
            frame.to_csv(path, mode="a", index=False, header=not path.is_file() or path.stat().st_size == 0)
            buffer.clear()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(solve, task) for task in todo]
        for k, future in enumerate(as_completed(futures), 1):
            buffer.append(future.result())
            if len(buffer) >= 20:
                flush()
            if k % 100 == 0:
                print(f"  {k}/{len(todo)} polytopes, {time.perf_counter() - t0:.0f} s", flush=True)
        flush()
    frame = pd.read_csv(path)
    keys = {task_key(t) for t in tasks}
    frame = frame[[task_key(r) in keys for r in frame.to_dict("records")]].copy()
    for column in ("string_stable", "hurwitz"):  # booleans of the CSV (an error row may hold an empty cell)
        frame[column] = frame[column].map(lambda x: str(x) == "True")
    frame["_key"] = [task_key(r) for r in frame.to_dict("records")]
    frame = frame.drop_duplicates(subset="_key", keep="last").drop(columns="_key")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------------------- budget allocation


def pointwise_allocation(member: Mapping[str, Any], n_scan: int = 400, cycles: int = 6, step_rtol: float = 1e-3
                         ) -> dict[str, Any]:  # fmt: skip
    """Coordinate ascent of ``B_s + B_dv + B_v`` under the pointwise criterion of the a priori certificate on the
    member's ``I(v)``: at every spacing of the scan (``n_scan`` points of every speed's interval, as ``_a_priori``) the
    guaranteed margin of the core's derivatives widened by ``B`` is >= 0 and the signs ``f_s > 0``, ``f_v + f_dv < 0``
    are guaranteed. ``{feasible, bounds, total}``."""
    import torch

    from cf_stability.stability.certificate import _guarantees

    region = robust.region_feasible(member["core"], member["r_max"], V_GRID)
    s = np.concatenate([np.linspace(*region[v], n_scan) for v in sorted(region)])
    v = np.concatenate([np.full(n_scan, v) for v in sorted(region)])
    f = [torch.as_tensor(x) for x in robust.idm_partials(member["core"], s, v)]

    def ok(bounds: np.ndarray) -> bool:
        g = _guarantees(*f, torch.as_tensor(bounds, dtype=torch.float64).reshape(1, 3).expand(len(s), 3))
        return bool(g["holds"].all())

    bounds = np.zeros(3)
    if not ok(bounds):
        return {"feasible": False, "bounds": bounds.tolist(), "total": 0.0}
    for _ in range(cycles):
        grown = False
        for j in range(3):
            def test(value: float, j: int = j) -> bool:
                trial = bounds.copy()
                trial[j] = value
                return ok(trial)

            low, high = bounds[j], max(2.0 * bounds[j], 1e-4)
            while high < 100.0 and test(high):
                low, high = high, 2.0 * high
            while high - low > step_rtol * max(high, 1e-9):
                mid = 0.5 * (low + high)
                low, high = (mid, high) if test(mid) else (low, mid)
            grown = grown or low > bounds[j] * (1.0 + step_rtol) + 1e-12
            bounds[j] = low
        if not grown:
            break
    # order of the bounds of the certificate: (s, dv, v)
    return {"feasible": True, "bounds": bounds.tolist(), "total": float(bounds.sum()),
            "uniform_holds": ok(np.asarray(member["bounds"], dtype=np.float64))}  # fmt: skip


def allocation_task(job: Mapping[str, Any]) -> dict[str, Any]:
    """The budget allocation of one member, variant and scope on the band (a worker of the pool)."""
    t0 = time.perf_counter()
    member = job["member_info"]
    region = robust.region_band(member["context"], V_GRID)
    if job["variant"] == "box":
        cores = [robust.idm_box(member["core"], v, *region[v]) for v in sorted(region)]
    else:
        cores = [robust.idm_curve_polytope(member["core"], v, *region[v], n_tangents=N_TANGENTS) for v in sorted(region)]
    if job["scope"] == "whole":
        if job["variant"] == "box":
            cores = [robust.Box(*((min(getattr(b, n)[0] for b in cores), max(getattr(b, n)[1] for b in cores))
                                  for n in ("f_s", "f_dv", "f_v")))]  # fmt: skip
        else:
            cores = [robust.hull_vertices(np.concatenate(cores))]
    result = robust.allocate_budget(cores)
    uniform = all(robust.string_stable(robust.widen(c, member["bounds"]))[0] for c in cores)
    return {"member": job["member"], "run": member["run"], "variant": job["variant"], "scope": job["scope"],
            "criterion": "certificate gamma <= 1 + 1e-3 on the band", **result, "uniform_holds": uniform,
            "seconds": time.perf_counter() - t0}  # fmt: skip


# ------------------------------------------------------------------------------------------------- tables


def law_rows(results: pd.DataFrame, laws_csv: pd.DataFrame) -> pd.DataFrame:
    """Per law, region, variant and scope: the largest gamma over its members (and speeds), the status, the speeds and
    members certified, the worst member and speed, the macro errors of the law on both corridors."""
    rows = []
    known = laws_csv[laws_csv["law"] != "ground truth"]
    errors = {(r["law"], r["corridor"]): float(r["macro_error"]) for r in known.to_dict("records")}
    for (law, region, variant, scope), part in results.groupby(["law", "region", "variant", "scope"], sort=False):
        infeasible = part[part["status"] != "certified"]  # infeasible or error
        certified = part[part["status"] == "certified"]
        status = ("error" if (part["status"] == "error").any() else "infeasible" if len(infeasible) else "certified")
        gamma = float(certified["gamma"].max()) if len(infeasible) == 0 and len(certified) else math.nan
        worst = part.loc[part["gamma"].astype(float).idxmax()] if len(certified) and not len(infeasible) else None
        row = {"law": law, "label": part["label"].iloc[0], "region": region, "variant": variant, "scope": scope,
               "kind": part["kind"].iloc[0], "members": int(part["member"].nunique()), "polytopes": len(part),
               "status": status, "gamma": gamma, "polytopes_error": int((part["status"] == "error").sum()),
               "gamma_min": float(certified["gamma"].min()) if len(certified) else math.nan,
               "lower_bound": float(part["lower_bound"].max()),
               "string_stable": bool(len(infeasible) == 0 and part["string_stable"].astype(bool).all()),
               "polytopes_infeasible": int(len(infeasible)),
               "polytopes_string_stable": int(part["string_stable"].astype(bool).sum()),
               "worst_member": None if worst is None else int(worst["member"]),
               "worst_v": None if worst is None or not np.isfinite(worst["v"]) else float(worst["v"]),
               "speeds": int(part["v"].nunique()) if scope == "per speed" else None}  # fmt: skip
        if len(infeasible):
            row["worst_v"] = float(infeasible["v"].iloc[0]) if np.isfinite(infeasible["v"].iloc[0]) else None
            row["worst_member"] = int(infeasible["member"].iloc[0])
        for corridor in ("I-80", "US-101"):
            row[f"macro_error_{corridor}"] = float(errors.get((law, corridor), math.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def rule_text(laws: pd.DataFrame) -> list[str]:
    """The decision rule of D130 on the band (box: the rule; hull: the supplement), per scope."""
    lines = []
    for variant in ("box", "hull"):
        for scope in ("whole", "per speed"):
            def get(law: str) -> pd.Series | None:
                part = laws[(laws["law"] == law) & (laws["region"] == "band") & (laws["variant"] == variant)
                            & (laws["scope"] == scope)]  # fmt: skip
                return None if not len(part) else part.iloc[0]

            e, d = get("residual_idm_certified"), get("residual_idm_margin_free_r0.3")
            if e is None or d is None:
                continue
            e_ok = bool(e["string_stable"])
            d_fails = (d["status"] != "certified") or not bool(d["string_stable"])
            branch = "explains" if e_ok and d_fails else "does not explain"
            fmt = lambda r: "infeasible" if r["status"] != "certified" else f"{r['gamma']:.4f}"  # noqa: E731
            lines.append(f"band, {variant}, {scope}: E gamma {fmt(e)} (string stable: {e_ok}), D gamma {fmt(d)} "
                         f"-> {branch}{' (supplementary variant)' if variant == 'hull' else ''}")  # fmt: skip
    return lines


def write(table: Table, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.frame.to_csv(out / f"{table.name}.csv", index=False)
    (out / f"{table.name}.md").write_text(markdown(table), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corridor-root", default="runs/corridor")
    parser.add_argument("--tables", default="runs/_tables")
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--states", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--laws", nargs="*", default=None)
    parser.add_argument("--no-allocation", action="store_true")
    parser.add_argument("--from-polytopes", default=None, help="the tables only, from a robust_certificate_polytopes.csv "
                                                               "of an earlier run (no polytope is solved again)")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    laws_root = resolve_path(args.corridor_root) / "laws"
    chosen = [entry for entry in LAWS if not args.laws or entry[0] in args.laws]
    members: dict[str, list[dict[str, Any]]] = {}
    tasks: list[dict[str, Any]] = []
    for law, label, _ in chosen:
        members[law] = members_of(law, laws_root)
        for k, member in enumerate(members[law]):
            tasks += tasks_of(law, label, k, member, args.states, args.seed)
        print(f"LAW {law}: {len(members[law])} members, {sum(t['law'] == law for t in tasks)} polytopes", flush=True)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_polytopes:
        frame = pd.read_csv(resolve_path(args.from_polytopes))
        for column in ("string_stable", "hurwitz"):
            frame[column] = frame[column].map(lambda x: str(x) == "True")
        frame = frame[frame["law"].isin([law for law, _, _ in chosen])].reset_index(drop=True)
        print(f"POLYTOPES {len(frame)} read from {args.from_polytopes}", flush=True)
    else:
        frame = run_tasks(tasks, out, args.workers)
        frame.to_csv(out / "robust_certificate_polytopes.csv", index=False)
        if len(frame) == len(tasks):  # complete: the checkpoint is no longer needed
            (out / CHECKPOINT).unlink(missing_ok=True)
    errors = frame[frame["status"] == "error"]
    if len(errors):
        print(f"ERRORS in {len(errors)} polytopes, e.g. {errors['reason'].iloc[0]}", flush=True)
    laws_csv = pd.read_csv(resolve_path(args.tables) / "m5" / "laws.csv")
    per_law = law_rows(frame, laws_csv)
    lines = rule_text(per_law)
    for line in lines:
        print("RULE", line)
    per_member = (frame.groupby(["law", "label", "member", "run", "region", "variant", "scope"], sort=False)
                  .agg(status=("status", lambda s: "certified" if (s == "certified").all() else
                                         "error" if (s == "error").any() else "infeasible"),
                       gamma=("gamma", lambda g: float(np.max(g)) if g.notna().all() else math.nan),
                       lower_bound=("lower_bound", "max"), polytopes=("status", "size"),
                       string_stable=("string_stable", "all"), seconds=("seconds", "sum")).reset_index())  # fmt: skip
    common = [
        "Robust (non-equilibrium) certificate of D130 (cf_stability/stability/robust.py): the bounded-real LMI with one "
        "common quadratic storage function P (2 x 2) over the vertices of the polytope of the follower's perturbation "
        "dynamics e' = w - u, u' = f_s e + f_dv (u - w) + f_v u (output u; dv = v - v_lead, the convention of the project; "
        "the contract writes the f_dv term in the other sign convention), P = L L^T, gamma by bisection (relative "
        "tolerance 1e-4), at every gamma the largest vertex eigenvalue minimised with scipy (Nelder-Mead from several "
        "starts, then BFGS), every certificate verified in float64 (all vertex eigenvalues <= -1e-9, P positive "
        "definite). The bisection is bracketed from above by the first verified gain at or above the smallest "
        "Schur-complement gain over P (gamma^2 > max_i B_i^T P (-N_i)^-1 P B_i, N_i = A_i^T P + P A_i + C^T C, "
        "minimised the same way) and from below by the largest vertex H-infinity norm, raised to 0.999 times that Schur "
        "gain when no certificate exists there. infeasible: a vertex not Hurwitz (f_s <= 0 or f_dv + f_v >= 0), no P "
        "with every N_i negative definite (no common quadratic storage function), or no verified certificate up to "
        "gamma 1e4.",
        "Every vertex with f_s > 0 has G(0) = 1: the L2 gain of the polytope is at least 1 and gamma < 1 cannot be "
        "certified; string stable = a verified certificate at gamma = 1 + 1e-3 (gamma reaches 1 only for a single "
        "frozen system: a common storage function over different systems needs a little more, e.g. 1.005 for the two "
        "ends of the derivative curve of the margin core at 5 m/s on I(v)).",
        "Regions of states (s, 0, v) at the grid speeds 5-30 m/s: band = the spacing band of the member's training "
        "context (NGSIM I-80; the band of the audit) where it exists; I(v) = the feasible gaps of the certificate of the "
        "member's core at its r_max (r_max 0 for B and the IDM: the equilibrium spacing). box (the rule): the core's "
        "derivative ranges over the spacings (exact: the ends of the interval) widened by the residual's Jacobian "
        "bounds (r_max lipschitz prod(norms) / (20, 2, 10)); MLP and PIDL: the ranges of their partials at "
        f"{args.states} states drawn uniformly over the band (a necessary-condition check: the sampled range is inside "
        "the true one). hull (supplementary, not the rule): the polytope holding the core's derivative curve over the "
        f"spacings (chord and {N_TANGENTS + 1} tangents; the derivatives move together with s) summed with the residual "
        "box. per speed: one polytope per grid speed, gamma = the largest over the speeds; whole: one polytope over "
        "all speeds of the region (box: the union of the ranges; hull: the convex hull of the union).",
    ]
    table = Table("robust_certificate", "Robust (non-equilibrium) certificate of the laws (D130, A3)", [
        *common,
        "Per law: gamma = the largest over its members (and speeds); status infeasible when one polytope of a member has "
        "no certificate; lower bound: the largest vertex H-infinity norm; polytopes string stable: those with a "
        "certificate at 1 + 1e-3. Macro error: the law's mean over its runs on each corridor (runs/_tables/m5/laws.csv).",
        "Rule of D130 (band, box): explains when E has a string-stable certificate (gamma <= 1 + 1e-3, the reading of "
        "'gamma < 1' above) and D has none (gamma > 1 + 1e-3 or infeasible). " + "; ".join(lines) + ".",
    ], per_law, [
        Column("label", "law", "text"), Column("law", "law file", "text"), Column("region", "region", "text"),
        Column("variant", "polytope", "text"), Column("scope", "scope", "text"), Column("members", "members", "int"),
        Column("polytopes", "polytopes", "int"), Column("status", "status", "text"), Column("gamma", "gamma", digits=4),
        Column("lower_bound", "lower bound", digits=4), Column("string_stable", "string stable", "flag"),
        Column("polytopes_string_stable", "polytopes string stable", "int"), Column("worst_v", "worst speed", digits=0),
        Column("macro_error_I-80", "macro error I-80", digits=3), Column("macro_error_US-101", "macro error US-101",
                                                                          digits=3),
    ])  # fmt: skip
    write(table, out)
    write(Table("robust_certificate_members", "Robust certificate per member (D130)", [
        *common, "Per member, region, polytope and scope: gamma (the largest over the speeds for per speed).",
    ], per_member, [
        Column("label", "law", "text"), Column("member", "member", "int"), Column("run", "run", "text"),
        Column("region", "region", "text"), Column("variant", "polytope", "text"), Column("scope", "scope", "text"),
        Column("status", "status", "text"), Column("gamma", "gamma", digits=4),
        Column("lower_bound", "lower bound", digits=4), Column("polytopes", "polytopes", "int"),
    ]), out)  # fmt: skip
    if not args.no_allocation and "residual_idm_certified" in members:
        jobs = [{"member": k, "member_info": {key: m[key] for key in ("run", "context", "core", "bounds", "r_max")},
                 "variant": variant, "scope": scope}
                for k, m in enumerate(members["residual_idm_certified"]) for variant in ("box", "hull")
                for scope in ("whole", "per speed")]  # fmt: skip
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            allocation = list(pool.map(allocation_task, jobs))
        for k, m in enumerate(members["residual_idm_certified"]):
            point = pointwise_allocation(m)
            allocation.append({"member": k, "run": m["run"], "variant": "pointwise (a priori criterion)",
                               "scope": "I(v), every spacing", "criterion": "guaranteed margin >= 0 and signs at every "
                               "spacing of I(v) (no common storage function)", **point})  # fmt: skip
        alloc = pd.DataFrame(allocation)
        uniform = {k: m["bounds"] for k, m in enumerate(members["residual_idm_certified"])}
        alloc["uniform_B_s"] = [uniform[k][0] for k in alloc["member"]]
        alloc["uniform_B_dv"] = [uniform[k][1] for k in alloc["member"]]
        alloc["uniform_B_v"] = [uniform[k][2] for k in alloc["member"]]
        alloc["uniform_total"] = alloc[["uniform_B_s", "uniform_B_dv", "uniform_B_v"]].sum(axis=1)
        for j, name in enumerate(("B_s", "B_dv", "B_v")):
            alloc[name] = [b[j] for b in alloc["bounds"]]
        alloc["ratio_to_uniform"] = alloc["total"] / alloc["uniform_total"]
        write(Table("budget_allocation", "Residual budget allocation for the margin core of E (D130)", [
            "Budgets (B_s, B_dv, B_v) of the residual's Jacobian bounds |dr/d(s, dv, v)| that maximise B_s + B_dv + B_v "
            "for the IDM core of every fold member of residual_idm_certified (E), by coordinate ascent with bisection "
            "(cf_stability.stability.robust.allocate_budget), reported, not trained. Criterion of D130: a verified "
            "certificate at gamma = 1 + 1e-3 on the band of the member's training context, box or hull (as in "
            "robust_certificate), for every speed (per speed) or over the whole band; feasible false: the core alone "
            "has none, every budget is 0.",
            "Supplement: the same ascent under the pointwise criterion of the a priori certificate on I(v) (r_max 0.3; "
            "certificate.guaranteed_margin >= 0 and the guaranteed signs at 400 spacings of every speed's I(v), without "
            "a common storage function).",
            "uniform: the member's own bounds at r_max 0.3 (r_max lipschitz prod(layer norms) / (20, 2, 10), the "
            "physical scaler of ResidualIDM); uniform holds: the criterion holds with them; ratio: the total of the "
            "allocation over the uniform total.",
        ], alloc.drop(columns=["bounds"]), [
            Column("member", "member", "int"), Column("variant", "polytope", "text"), Column("scope", "scope", "text"),
            Column("feasible", "feasible", "flag"), Column("B_s", "B_s", digits=4), Column("B_dv", "B_dv", digits=4),
            Column("B_v", "B_v", digits=4), Column("total", "total", digits=4),
            Column("uniform_total", "uniform total", digits=4), Column("uniform_holds", "uniform holds", "flag"),
            Column("ratio_to_uniform", "ratio", digits=2),
        ]), out)  # fmt: skip
        print(f"TABLE budget_allocation: {len(alloc)} rows -> {out / 'budget_allocation.md'}")
    print(f"robust_certificate: {len(chosen)} laws, {len(tasks)} polytopes, {time.perf_counter() - t_start:.0f} s -> "
          f"{out / 'robust_certificate.md'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
