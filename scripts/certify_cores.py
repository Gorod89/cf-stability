"""A priori certificate of per-event IDM cores under the shared certified residual of E4 (D118).

    python scripts/certify_cores.py
    python scripts/certify_cores.py folds=[0,1,2,3,4]
    python scripts/certify_cores.py cores=data/calibration/ngsim_i80/idm_event_margin0.2_<key>.json

For every core of the margin-core file of ``calibrate_idm.py per_event_margin=<m>`` (``cores``; by
default the only ``data/calibration/<dataset>/idm_event_margin<margin>_<key>.json``) the a priori
certificate of D71/D86 is checked with the residual of the run ``residual`` of every fold in ``folds``:
the hybrid ``IDM(core) + r`` with ``|r| <= r_max`` and the Jacobian bounds of that residual is string
stable, with guaranteed signs, at every spacing that can be an equilibrium at every speed of the grid
(``certify_core`` = ``certificate_a_priori`` of a ResidualIDM with that core, without the budget). The
residual enters through ``r_max`` and its bounds only, so the residual of one run certifies any number of
cores. Writes ``<cores file stem>_certified.json`` next to the cores file (format in its ``settings``)
and prints one line per fold and the share of certified cores.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.models import ResidualIDM, load_model  # noqa: E402
from cf_stability.stability.certificate import certify_core, residual_of, to_json  # noqa: E402
from cf_stability.stability.equilibrium import V_GRID  # noqa: E402
from cf_stability.utils import (  # noqa: E402
    REPO_ROOT,
    config_hash,
    git_revision,
    read_json,
    resolve_path,
    to_plain,
    write_json,
)

SUFFIX = "_certified"
FORMAT = (
    "cores: one record per core of the margin-core file (cores_file), in its order: event_id, follower_id, "
    "params {v0, T, s0, a, b} (the IDM core), margin_min and margin_holds (copied from the margin-core file: "
    "the closed-form string-stability margin of the core on the speed grid of its calibration), certified (the "
    "a priori certificate holds at every speed of `speeds` with the residual of every fold in `folds`), "
    "guaranteed_margin_min (smallest guaranteed margin of the hybrid over the speeds and these folds; null when "
    "no spacing can be an equilibrium), folds {<fold>: {holds, guaranteed_margin_min, n_hold (speeds where it "
    "holds), v_worst (speed of the smallest guaranteed margin)}}. A law built from this file uses the cores "
    "with certified = true and pairs them only with the residuals of the runs in residual_runs (the folds in "
    "`folds`): a core is certified for these residuals and no other"
)


def cores_file(cfg: DictConfig) -> Path:
    """``cores``, or the only margin-core file of ``dataset`` and ``margin`` in the calibration root."""
    if cfg.cores is not None:
        path = resolve_path(cfg.cores)
        if not path.is_file():
            raise FileNotFoundError(f"cores={cfg.cores}: {path} does not exist")
        return path
    directory = resolve_path(cfg.paths.calibration_root) / cfg.dataset
    pattern = f"idm_event_margin{float(cfg.margin):g}_*.json"
    found = sorted(p for p in directory.glob(pattern) if not p.stem.endswith(SUFFIX))
    if len(found) != 1:
        names = ", ".join(p.name for p in found) or "none"
        raise FileNotFoundError(
            f"cores=null needs exactly one {directory / pattern} (found: {names}); set cores=<file>"
        )
    return found[0]


def shown(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it (as the file records it)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_residual(run: Path) -> ResidualIDM:
    model = load_model(run / "model.pt")
    if not isinstance(model, ResidualIDM):
        raise ValueError(f"{run}: the residual comes from a residual_idm run, this one holds a {model.name!r} model")
    return model


def certify_record(
    record: Mapping[str, Any], residuals: Mapping[int, Mapping[str, Any]], speeds: Sequence[float], n_scan: int
) -> dict[str, Any]:
    """The certificate of one core with the residual of every fold (``residuals``: ``residual_of`` per fold)."""
    folds: dict[str, dict[str, Any]] = {}
    for fold, residual in residuals.items():
        out = certify_core(record["params"], residual["r_max"], residual["bounds"], speeds, n_scan)
        margins = out["guaranteed_margin"]
        finite = torch.isfinite(margins)
        worst = int(torch.where(finite, margins, torch.inf).argmin()) if finite.any() else None
        folds[str(fold)] = {
            "holds": bool(out["holds"].all()),
            "guaranteed_margin_min": float(margins[worst]) if worst is not None else None,
            "n_hold": int(out["holds"].sum()),
            "v_worst": float(speeds[worst]) if worst is not None else None,
        }
    minima = [f["guaranteed_margin_min"] for f in folds.values() if f["guaranteed_margin_min"] is not None]
    return {
        "event_id": record["event_id"],
        "follower_id": record.get("follower_id"),
        "params": dict(record["params"]),
        "margin_min": record.get("margin_min"),
        "margin_holds": record.get("margin_holds"),
        "certified": all(f["holds"] for f in folds.values()),
        "guaranteed_margin_min": min(minima) if len(minima) == len(folds) else None,
        "folds": folds,
    }


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q05": None, "median": None}
    x = torch.tensor(values, dtype=torch.float64)
    return {"min": float(x.min()), "q05": float(torch.quantile(x, 0.05)), "median": float(torch.quantile(x, 0.5))}


@hydra.main(version_base="1.3", config_path="../configs", config_name="certify_cores")
def main(cfg: DictConfig) -> None:
    t_start = time.perf_counter()
    plain = to_plain(cfg)
    settings = plain["certificate"]
    speeds = tuple(float(v) for v in settings["speeds"]) if settings.get("speeds") is not None else V_GRID
    n_scan = int(settings["n_scan"])
    folds = [int(k) for k in plain["folds"]]
    if not folds or len(set(folds)) != len(folds):
        raise ValueError(f"folds: distinct fold numbers, got {plain['folds']}")
    path = cores_file(cfg)
    source = read_json(path)
    if source.get("kind") != "idm_event_margin" or not source.get("events"):
        raise ValueError(f"{path}: not a margin-core file of calibrate_idm.py per_event_margin=<m>, or no events")
    runs = {k: resolve_path(str(cfg.residual).format(fold=k)) for k in folds}
    models = {k: load_residual(run) for k, run in runs.items()}
    residuals = {k: residual_of(model) for k, model in models.items()}
    records = [certify_record(record, residuals, speeds, n_scan) for record in source["events"]]

    n = len(records)
    certified = [r for r in records if r["certified"]]
    per_fold = {}
    for k in folds:
        hold = sum(r["folds"][str(k)]["holds"] for r in records)
        per_fold[str(k)] = {"n_certified": hold, "share_certified": hold / n}
    margins = [r["guaranteed_margin_min"] for r in records if r["guaranteed_margin_min"] is not None]
    summary = {
        "n_cores": n,
        "n_certified": len(certified),
        "share_certified": len(certified) / n,
        "per_fold": per_fold,
        "n_margin_holds": sum(bool(r["margin_holds"]) for r in records),
        "n_certified_margin_holds": sum(bool(r["margin_holds"]) for r in certified),
        "guaranteed_margin_min": quantiles(margins),
    }
    out = resolve_path(cfg.output) if cfg.output is not None else path.with_name(f"{path.stem}{SUFFIX}.json")
    payload = {
        "kind": "idm_event_margin_certified",
        "cores_file": shown(path),
        "cores_key": source.get("key"),
        "cores_config_hash": source.get("config_hash"),
        "settings": {
            "folds": folds,
            "residual": str(cfg.residual),
            "residual_runs": {str(k): shown(run) for k, run in runs.items()},
            # the residual of every fold as the certificate sees it, and the core the run itself carries
            "residuals": {str(k): {**residuals[k], "core_params": models[k].idm.params_dict()} for k in folds},
            "speeds": list(speeds),
            "n_scan": n_scan,
            "stability_margin": source.get("settings", {}).get("stability_margin"),
            "certificate": (
                "a priori certificate of D71/D86 (cf_stability/stability/certificate.py: certify_core = "
                "certificate_a_priori of a ResidualIDM with the core and the residual of the run, as "
                "certificate.json['a_priori'] of E4): at every speed the hybrid is string stable with guaranteed "
                "signs at every spacing in [1, 200] m where |f_idm(s, 0, v)| <= r_max"
            ),
            "format": FORMAT,
        },
        "summary": summary,
        "cores": records,
        "config": plain,
        "config_hash": config_hash(settings),
        "git_revision": git_revision(),
    }
    payload["wall_time_s"] = time.perf_counter() - t_start
    write_json(out, to_json(payload))

    lines = [f"certify_cores  {shown(path)}  {n} cores  folds {folds}  speeds {speeds[0]:g}-{speeds[-1]:g} m/s"]
    for k in folds:
        r = residuals[k]
        lines.append(
            f"  fold {k}: certified {per_fold[str(k)]['n_certified']}/{n} ({per_fold[str(k)]['share_certified']:.1%})  "
            f"residual r_max {r['r_max']:g}, r_max*lipschitz {r['product']:.4g}, bounds "
            + "/".join(f"{b:.4g}" for b in r["bounds"].tolist()) + f"  ({shown(runs[k])})"
        )
    q = summary["guaranteed_margin_min"]
    lines.append(
        f"  certified for every fold: {len(certified)}/{n} ({summary['share_certified']:.1%}); margin holds "
        f"{summary['n_margin_holds']}/{n}; guaranteed margin min/q05/median "
        + "/".join("n/a" if q[key] is None else f"{q[key]:.3f}" for key in ("min", "q05", "median"))
    )
    lines.append(f"  written  : {shown(out)}  ({payload['wall_time_s']:.0f} s)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
