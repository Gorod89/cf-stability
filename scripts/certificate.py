"""Stability certificate of trained ResidualIDM runs (D71, D86; docs/m4_contract.md, section 3.3).

    python scripts/certificate.py run=runs/e4_stable/follownet_highd/residual_idm/driver_fold0_seed0
    python scripts/certificate.py experiment=e4_stable data=follownet_highd [force=true]

Writes ``certificate.json`` into every run directory. For a ResidualIDM: ``{run, model, applicable:
true, core: {params, margin_min, margin_max, n_equilibria, n_grid}, residual: {r_max, lipschitz,
product, bounds, layer_norms}, a_priori: {holds, n_hold, n_grid, guaranteed_margin_min,
admissible_budget}, at_equilibria: {holds, n_hold, n_equilibria, guaranteed_margin_min,
empirical_margin_min, band}, per_speed: [...], config, config_hash, git_revision, wall_time_s}``.
The core margins are the closed-form margins of the IDM core at its own equilibria (over the speeds
below ``v0``); the a priori certificate is taken at the model's ``r_max`` and ``admissible_budget``
is the largest ``r_max * lipschitz`` with that ``r_max`` (the budget of ``certificate.enforce``);
the a posteriori one uses the equilibria anchored to the spacing band of ``metrics.json["context"]``
when the context has a band. Any other model gets ``{run, model, applicable: false, config, ...}``.

One line per run. A failing run does not stop the others (``FAILED`` line, error in
``certificate.json``); the exit status is then 1. In experiment mode runs whose
``certificate.json`` is newer than ``model.pt`` are skipped unless ``force=true``.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch import Tensor  # noqa: E402

from cf_stability.models import ResidualIDM, load_model  # noqa: E402
from cf_stability.stability.certificate import (  # noqa: E402
    certificate_a_priori,
    certificate_at_equilibria,
    core_margin,
    empirical_margin,
    to_json,
)
from cf_stability.stability.equilibrium import V_GRID, Band  # noqa: E402
from cf_stability.utils import config_hash, git_revision, read_json, resolve_path, to_plain, write_json  # noqa: E402

OUTPUT = "certificate.json"


def is_current(run_dir: Path) -> bool:
    out = run_dir / OUTPUT
    return out.exists() and out.stat().st_mtime > (run_dir / "model.pt").stat().st_mtime


def run_directories(cfg: DictConfig) -> list[Path]:
    """``run``, or the runs of ``experiment`` / ``data`` with a checkpoint (without a newer output unless ``force``)."""
    if cfg.run is not None:
        if not resolve_path(cfg.run).is_dir():
            raise FileNotFoundError(f"run directory {resolve_path(cfg.run)} does not exist")
        return [resolve_path(cfg.run)]
    data = cfg.get("data")
    if cfg.experiment is None or data is None:
        raise ValueError("set run=<run directory> or experiment=<name> data=<event set>")
    root = resolve_path(cfg.paths.runs_root) / cfg.experiment / data.name
    runs = sorted(path.parent for path in root.glob("*/*/model.pt"))
    return [run for run in runs if cfg.force or not is_current(run)]


def _extreme(values: Tensor, mask: Tensor | None = None, largest: bool = False) -> float | None:
    """Smallest (largest) finite value of ``values[mask]``, None without any."""
    x = values if mask is None else values[mask]
    x = x[torch.isfinite(x)]
    if not len(x):
        return None
    return float(x.max() if largest else x.min())


def certify(
    model: ResidualIDM, context: Mapping[str, Any] | None, speeds: Sequence[float], n_scan: int
) -> dict[str, Any]:
    """The certificate entries of ``certificate.json`` for one ResidualIDM."""
    band = Band.from_context(context, model)
    core = core_margin(model, speeds)
    prior = certificate_a_priori(model, speeds, n_scan, keep_r_max=True)
    post = certificate_at_equilibria(model, speeds, band=band)
    empirical = empirical_margin(model, speeds, band=band)
    found = post["found"]
    per_speed = [  # to_json below: tensors to numbers, non-finite values to None
        {
            "v": post["v"][k],
            "core_margin": core[k],
            "a_priori_holds": prior["holds"][k],
            "a_priori_margin": prior["guaranteed_margin"][k],
            "a_priori_s_low": prior["s_low"][k],
            "a_priori_s_high": prior["s_high"][k],
            "a_priori_budget": prior["max_budget"][k],
            "s": post["s"][k],
            "status": post["status"][k],
            "found": found[k],
            "holds": post["holds"][k],
            "guaranteed_margin": post["guaranteed_margin"][k],
            "empirical_margin": empirical[k],
            "budget": post["max_budget"][k],
        }
        for k in range(len(post["v"]))
    ]
    return to_json({
        "core": {
            "params": model.idm.params_dict(),
            "margin_min": _extreme(core),
            "margin_max": _extreme(core, largest=True),
            "n_equilibria": int(torch.isfinite(core).sum()),
            "n_grid": len(core),
        },
        "residual": {
            "r_max": model.r_max,
            "lipschitz": model.lipschitz,
            "product": model.r_max * model.lipschitz,
            "bounds": prior["bounds"],
            "layer_norms": prior["layer_norms"],
        },
        "a_priori": {
            "holds": bool(prior["holds"].all()),
            "n_hold": prior["n_holds"],
            "n_grid": len(prior["v"]),
            "guaranteed_margin_min": _extreme(prior["guaranteed_margin"]),
            "admissible_budget": float(prior["max_budget"].min()),
        },
        "at_equilibria": {
            "holds": bool(post["holds"].all()),
            "n_hold": post["n_holds"],
            "n_equilibria": int(found.sum()),
            "guaranteed_margin_min": _extreme(post["guaranteed_margin"], found),
            "empirical_margin_min": _extreme(empirical, found),
            "band": band is not None,
        },
        "per_speed": per_speed,
    })


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def summary_line(label: str, c: Mapping[str, Any], wall: float) -> str:
    prior, post = c["a_priori"], c["at_equilibria"]
    equilibria = f"{post['n_equilibria']} eq" + (", band" if post["band"] else "")
    return (
        f"{label:<30} a priori {prior['n_hold']}/{prior['n_grid']} (min {_fmt(prior['guaranteed_margin_min'], '.3f')}, "
        f"admissible {prior['admissible_budget']:.3g})  at equilibria {post['n_hold']}/{prior['n_grid']} "
        f"({equilibria}; min {_fmt(post['guaranteed_margin_min'], '.3f')}, "
        f"empirical {_fmt(post['empirical_margin_min'], '.3f')})  r_max*lipschitz {c['residual']['product']:.3g}  "
        f"core margin {_fmt(c['core']['margin_min'], '.3f')}  {wall:.0f} s"
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="certificate")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    settings = plain["certificate"]
    speeds = tuple(float(v) for v in settings["speeds"]) if settings.get("speeds") is not None else V_GRID
    header = {"config": plain, "config_hash": config_hash(settings), "git_revision": git_revision()}
    failed = 0
    for run_dir in run_directories(cfg):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        t_start = time.perf_counter()
        payload: dict[str, Any] = {**header, "run": str(run_dir), "model": None}
        try:
            model = load_model(run_dir / "model.pt")
            payload["model"] = model.name
            payload["applicable"] = isinstance(model, ResidualIDM)
            if payload["applicable"]:
                metrics = run_dir / "metrics.json"
                context = read_json(metrics).get("context") if metrics.exists() else None
                payload.update(certify(model, context, speeds, int(settings["n_scan"])))
                line = summary_line(label, payload, time.perf_counter() - t_start)
            else:
                line = f"{label:<30} not applicable ({model.name})"
        except Exception as exc:  # a failing run must not stop the others
            failed += 1
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
            line = f"{label:<30} FAILED  {payload['error'].splitlines()[0]}"
        payload["wall_time_s"] = time.perf_counter() - t_start
        write_json(run_dir / OUTPUT, payload)
        print(line, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
