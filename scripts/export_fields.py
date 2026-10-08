"""Edie fields of the corridor runs for the figures (``fields.npz``, :mod:`cf_stability.corridor.fields`).

    python scripts/export_fields.py                                  # every corridor run with trajectories.npz
    python scripts/export_fields.py --scenario i80_p1 --law idm_global --seed 0
    python scripts/export_fields.py --force                          # rewrite files that are up to date
    python scripts/export_fields.py --contour-seeds all              # the speed-field grid of every seed (larger)

Writes ``fields.npz`` next to ``macro.json`` of every selected run under ``--corridor-root`` that has
``trajectories.npz`` and ``run.json``: the Edie grids that the corridor figures draw (the speed contours and the
fundamental diagrams of ``scripts/make_report.py`` and ``scripts/make_figures_supplement.py``), with the cells, lanes
and preparation of ``configs/corridor_metrics.yaml`` (``macro``, with the ``metrics`` of the scenario over it) and,
for the scenario of the report, of ``configs/make_report.yaml``. The speed-field grid (0.1-0.2 MB) is written for the
seeds the figures draw (``--contour-seeds``, default 0), the cells of the diagrams (5 KB) for every run. A file that
holds the settings of now and is not older than its inputs (``trajectories.npz``, ``vehicles.npz``, ``run.json``,
``scenario.json``) is kept unless ``--force``. Prints one line per run (``OK``, ``KEPT``, ``SKIPPED``: no
trajectories, its ``fields.npz`` left as it is; ``FAILED``) and a summary with the count and the size of the files;
exits with 1 when a run failed. Reads the runs only; simulates nothing. ``scripts/run_corridor.py`` runs it for every
new run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_stability.corridor.fields import CONTOUR_SEEDS, FIELDS_FILE, update_fields  # noqa: E402
from cf_stability.corridor.macro import find_items  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402


def contour_seeds(values: Sequence[str]) -> tuple[int, ...] | None:
    """``all`` (None: every seed) or the seeds."""
    if [v.lower() for v in values] == ["all"]:
        return None
    return tuple(int(v) for v in values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corridor-root", default="runs/corridor", help="runs <scenario>/<law>/seed<k>")
    parser.add_argument("--scenarios-root", default=None, help="scenario.json files; default: <corridor-root>/scenarios")
    parser.add_argument("--configs-dir", default="configs", help="corridor_metrics.yaml and make_report.yaml")
    parser.add_argument("--scenario", default=None, help="only this scenario")
    parser.add_argument("--law", default=None, help="only this law")
    parser.add_argument("--seed", type=int, default=None, help="only this seed")
    parser.add_argument("--contour-seeds", nargs="+", default=[str(s) for s in CONTOUR_SEEDS],
                        help="seeds whose file holds the speed-field grid, or 'all' (default: the seed of the figures)")
    parser.add_argument("--force", action="store_true", help="rewrite files that are up to date")
    args = parser.parse_args(argv)
    root = resolve_path(args.corridor_root)
    scenarios_root = resolve_path(args.scenarios_root) if args.scenarios_root else None
    start = time.perf_counter()
    counts = {"OK": 0, "KEPT": 0, "SKIPPED": 0, "FAILED": 0}
    for line in update_fields(root, scenario=args.scenario, law=args.law, seed=args.seed, force=args.force,
                              contour_seeds=contour_seeds(args.contour_seeds), scenarios_root=scenarios_root,
                              configs_dir=resolve_path(args.configs_dir)):  # fmt: skip
        counts[line.split(" ", 1)[0]] += 1
        print(line, flush=True)
    files = [item.directory / FIELDS_FILE for item in find_items(root, args.scenario, args.law, args.seed)
             if not item.is_truth and (item.directory / FIELDS_FILE).exists()]  # fmt: skip
    size = sum(path.stat().st_size for path in files)
    print(f"FIELDS: written {counts['OK']}, kept {counts['KEPT']}, skipped {counts['SKIPPED']}, failed "
          f"{counts['FAILED']}; {len(files)} {FIELDS_FILE} files of the selection, {size / 2**20:.1f} MB "
          f"({time.perf_counter() - start:.0f} s)", flush=True)  # fmt: skip
    return 1 if counts["FAILED"] else 0


if __name__ == "__main__":
    sys.exit(main())
