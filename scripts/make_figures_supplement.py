"""Supplementary figures and tables of M8 (docs/m8_contract.md, items 1, 2, 3 and 8; D115, D116, D123).

    python scripts/make_figures_supplement.py                       # everything, from runs/ and data/events/
    python scripts/make_figures_supplement.py --only lowfreq_expansion certificate_tightness
    python scripts/make_figures_supplement.py --runs-root runs --dpi 200
    python scripts/make_figures_supplement.py --only corridor --corridor-source fields   # as the published files

Writes the tables ``lowfreq_expansion``, ``certificate_tightness`` and ``band_width`` to
``<runs_root>/_tables/m8/<name>.csv|md``, the figures ``lowfreq_expansion``, ``certificate_tightness``,
``band_width``, ``stability_map_<architecture>``, ``contours_<corridor>_p<k>`` and ``fd_<corridor>_p<k>`` to
``<runs_root>/_report/supplement/figures/<name>.png|pdf`` with the caption ``<name>.txt``, and the notes on missing
inputs to ``<runs_root>/_report/supplement/figures_supplement_notes.txt``
(:mod:`cf_stability.eval.figures_supplement`). The corridor panels take a run from its ``trajectories.npz``, else
from its ``fields.npz`` (``scripts/export_fields.py``; ``--corridor-source``). Missing inputs give notes, except that
with ``--strict`` (the default) a corridor panel without its input stops the script with an error naming the run and
the file (``--no-strict``: a note, the figure is drawn without the panel and marked incomplete). Prints one line per
output and per note; reads only, trains and simulates nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_stability.corridor.fields import SOURCES  # noqa: E402
from cf_stability.eval.figures_supplement import OUTPUTS, SupplementConfig, make_supplement  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402


def parse(argv: list[str] | None = None) -> SupplementConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", default="runs", help="run directories, the tables of M4/M5 and the corridor")
    parser.add_argument("--out-dir", default=None, help="default: <runs-root>/_report/supplement")
    parser.add_argument("--tables-dir", default=None, help="default: <runs-root>/_tables/m8")
    parser.add_argument("--tables-m4", default=None, help="chosen_weights.json; default: <runs-root>/_tables/m4")
    parser.add_argument("--tables-m5", default=None, help="laws.csv; default: <runs-root>/_tables/m5")
    parser.add_argument("--corridor-root", default=None, help="default: <runs-root>/corridor")
    parser.add_argument("--events-root", default="data/events", help="event sets of the band widths")
    parser.add_argument("--configs-dir", default="configs", help="corridor_metrics.yaml (corridors, macro settings)")
    parser.add_argument("--only", nargs="+", choices=OUTPUTS, default=list(OUTPUTS), help="output groups to write")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (every figure is also written as PDF)")
    parser.add_argument("--corridor-source", choices=SOURCES, default="auto",
                        help="corridor runs: auto = trajectories.npz where the run has it, else fields.npz; "
                             "trajectories or fields = that file only")  # fmt: skip
    parser.add_argument("--ablation-panels", action=argparse.BooleanOptionalAction, default=False,
                        help="draw the laws of design.ablation_laws of corridor_metrics.yaml in the corridor grids "
                             "(default: left out with a note; they are compared in their own tables)")  # fmt: skip
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="a corridor panel without its input stops the script (--no-strict: a note, the figure "
                             "without the panel)")  # fmt: skip
    args = parser.parse_args(argv)
    runs_root = resolve_path(args.runs_root)
    paths = {
        "out_dir": args.out_dir, "tables_dir": args.tables_dir, "tables_m4": args.tables_m4,
        "tables_m5": args.tables_m5, "corridor_root": args.corridor_root,
    }  # fmt: skip
    changes = {key: resolve_path(value) for key, value in paths.items() if value is not None}
    return SupplementConfig.under(
        runs_root, events_root=resolve_path(args.events_root), configs_dir=resolve_path(args.configs_dir),
        outputs=tuple(args.only), dpi=args.dpi, strict=args.strict, corridor_source=args.corridor_source,
        ablation_panels=args.ablation_panels, **changes,
    )  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    for line in make_supplement(parse(argv)):
        print(line, flush=True)


if __name__ == "__main__":
    main()
