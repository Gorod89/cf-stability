"""Paired corridor comparisons of the law of D135 (docs/m12_contract.md, agent B').

    python scripts/analysis/m12_pairs.py            # after scripts/corridor_metrics.py (the macro.json of the runs)

The table builder of the factorial ablation (``table_ablation_pairs`` of ``cf_stability/eval/corridor_tables.py``,
``ablation_pairs: [candidate, reference]``) with the design of ``configs/corridor_metrics.yaml`` and these pairs, as
``scripts/analysis/m11_pairs.py`` builds the tables of M11:

* ``tt2_pairs`` (D135, H-N2'): ``residual_idm_tt2`` (the two-timescale certified hybrid with the latent carried along the
  vehicle) against E (``residual_idm_certified``) and against ``residual_idm_tt`` (the law of D129, whose latent
  restarts with every window).

Per corridor, pair and metric (macro error, dynamic macro error, collisions per 1000 vehicle-km): the difference
candidate - reference paired by scenario and seed (the same seed: the same members, drawn alike) with its percentile
bootstrap interval, the relative change with its interval and the Wilcoxon p-value, as in ``ablation_pairs``. Writes
``runs/_tables/m12/tt2_pairs.{csv,md}``; no decision rule (D135 reports these with intervals and no verdict).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from cf_stability.eval.corridor_tables import CorridorTableMaker, CorridorTablesConfig  # noqa: E402
from cf_stability.eval.tables import Table, write_table  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402

TT2, TT, E = "residual_idm_tt2", "residual_idm_tt", "residual_idm_certified"
GROUPS = {
    "tt2_pairs": ("Two-timescale certified hybrid with a persistent latent (D135, H-N2'): paired differences against E "
                  "and against the law of D129", ((TT2, E), (TT2, TT))),
}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/corridor_metrics.yaml")
    parser.add_argument("--out", default="runs/_tables/m12")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    raw = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    paths = raw["paths"]
    corridor_root, runs_root = resolve_path(paths["corridor_root"]), resolve_path(paths["runs_root"])
    out = resolve_path(args.out)
    base = CorridorTablesConfig.from_mapping(raw.get("design") or {}, corridor_root, runs_root, out / "_unused",
                                             out / "_unused")  # fmt: skip
    for name, (title, pairs) in GROUPS.items():
        cfg = dataclasses.replace(base, ablation_pairs=tuple(pairs))
        maker = CorridorTableMaker(cfg)
        table = maker.table_ablation_pairs()
        frame = table.frame.copy()
        if len(frame):
            frame["pair"] = frame["candidate"] + " - " + frame["reference"]
        notes = [note for note in table.notes if not note.startswith("Factorial ablation")]
        notes.insert(0, f"{title}. Candidate and reference per corridor, paired by scenario and seed (unit run): "
                        "the builder of ablation_pairs (cf_stability/eval/corridor_tables.py) with these pairs; E = "
                        "residual_idm_certified (the certified hybrid, uniform fold members); residual_idm_tt = the "
                        "two-timescale law of D129 (M11, the latent restarted with every window of 30 states).")  # fmt: skip
        missing = [line for line in maker.missing if line.startswith("[ablation_pairs]")]
        if missing:
            notes.append(f"Missing: {'; '.join(line.split('] ', 1)[-1] for line in missing[:6])}.")
        write_table(Table(name, title, notes, frame, table.columns), out)
        print(f"TABLE {name}: {len(frame)} rows -> {out / (name + '.md')}")
        if len(frame):
            for row in frame[frame["metric"] == "macro_error"].to_dict("records"):
                print(f"  {row['corridor']:<7} {row['pair']:<45} macro error {row['mean_candidate']:.4f} vs "
                      f"{row['mean_reference']:.4f}: {row['difference']:+.4f} [{row['difference_low']:+.4f}, "
                      f"{row['difference_high']:+.4f}] ({row['pairs']} pairs, p {row.get('wilcoxon_p', float('nan')):.4f}, "
                      f"{row['outcome']})")  # fmt: skip
    print(f"m12_pairs: {time.perf_counter() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
