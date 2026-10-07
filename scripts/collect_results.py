"""One table of the results of runs (docs/m4_contract.md, section 2.3).

    python scripts/collect_results.py experiments=[e1,e2_jacobian_w0.1] name=e1_e2
    python scripts/collect_results.py 'experiments=[e2_*]' name=e2_sweep

Collects every run ``<runs_root>/<experiment>/<data>/<model>/<split>_fold<k>_seed<s>/`` of the
experiments (names or glob patterns) with ``cf_stability.eval.collect.collect_runs`` (training, audit,
platoon test, transfer, certificate; missing files give missing values) and writes
``<runs_root>/_tables/<name>.parquet`` and ``<name>.csv``. Prints one line.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.eval.collect import collect_runs  # noqa: E402
from cf_stability.utils import resolve_path, to_plain  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="collect_results")
def main(cfg: DictConfig) -> None:
    experiments = to_plain(cfg.experiments)
    experiments = [experiments] if isinstance(experiments, str) else [str(e) for e in experiments or []]
    if not experiments or not cfg.name:
        raise ValueError("set experiments=[<name or pattern>,...] name=<table>")
    runs_root = resolve_path(cfg.paths.runs_root)
    table = collect_runs(runs_root, experiments)
    out = runs_root / "_tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / f"{cfg.name}.parquet", index=False)
    table.to_csv(out / f"{cfg.name}.csv", index=False)
    n_experiments = table["experiment"].nunique() if len(table) else 0
    print(
        f"TABLE {cfg.name}: {len(table)} runs of {n_experiments} experiments, {table.shape[1]} columns -> "
        f"{out / cfg.name}.parquet, .csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
