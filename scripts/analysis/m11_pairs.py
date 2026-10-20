"""Paired corridor comparisons of the laws of D129 and D132 (docs/m11_contract.md, agent B).

    python scripts/analysis/m11_pairs.py            # after scripts/corridor_metrics.py (the macro.json of the runs)

The table builder of the factorial ablation (``table_ablation_pairs`` of ``cf_stability/eval/corridor_tables.py``,
``ablation_pairs: [candidate, reference]``) with the design of ``configs/corridor_metrics.yaml`` and these pairs:

* ``het_fd_pairs`` (D132, H-N5): ``residual_idm_certified_het_fd`` (the cores drawn with the weights fitted to the
  fundamental diagram of i80_p1) against ``residual_idm_certified_het`` (the same cores, uniform) and against E
  (``residual_idm_certified``);
* ``tt_pairs`` (D129, H-N2): ``residual_idm_tt`` (the two-timescale certified hybrid) against E.

Per corridor, pair and metric (macro error, dynamic macro error, collisions per 1000 vehicle-km): the difference
candidate - reference paired by scenario and seed (the same seed: the same members for the laws whose members are drawn
alike) with its percentile bootstrap interval, the relative change with its interval and the Wilcoxon p-value, as in
``ablation_pairs``. Writes ``runs/_tables/m11/{het_fd_pairs,tt_pairs}.{csv,md}`` and prints the branch of the decision
rule of H-N5 (the tilted law below the uniform one on both sites with intervals excluding zero).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cf_stability.eval.corridor_tables import CorridorTableMaker, CorridorTablesConfig  # noqa: E402
from cf_stability.eval.tables import Table, write_table  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402

TT, HET, HET_FD, E = "residual_idm_tt", "residual_idm_certified_het", "residual_idm_certified_het_fd", "residual_idm_certified"
GROUPS = {
    "het_fd_pairs": ("FD-matched member weights of the heterogeneous certified hybrid (D132, H-N5): paired differences",
                     ((HET_FD, HET), (HET_FD, E))),
    "tt_pairs": ("Two-timescale certified hybrid (D129, H-N2): paired differences against E", ((TT, E),)),
}  # fmt: skip


def h_n5_branch(frame: pd.DataFrame) -> str:
    """The branch of the rule of H-N5 on the macro error of the tilted law against the uniform one, per corridor."""
    rows = frame[(frame["candidate"] == HET_FD) & (frame["reference"] == HET) & (frame["metric"] == "macro_error")]
    if not len(rows):
        return "no rows (runs missing)"
    below = [bool(r["difference_high"] < 0) for r in rows.to_dict("records") if math.isfinite(r["difference_high"])]
    sites = ", ".join(f"{r['corridor']}: {r['difference']:+.4f} [{r['difference_low']:+.4f}, {r['difference_high']:+.4f}]"
                      for r in rows.to_dict("records"))  # fmt: skip
    if len(below) == len(rows) and all(below) and len(rows) >= 2:
        return f"below on both sites with intervals excluding zero (confirmed branch): {sites}"
    return f"not below with an interval excluding zero on both sites (not the confirmed branch): {sites}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/corridor_metrics.yaml")
    parser.add_argument("--out", default="runs/_tables/m11")
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
                        "residual_idm_certified (the certified hybrid, uniform fold members).")  # fmt: skip
        missing = [line for line in maker.missing if line.startswith("[ablation_pairs]")]
        if missing:
            notes.append(f"Missing: {'; '.join(line.split('] ', 1)[-1] for line in missing[:6])}.")
        write_table(Table(name, title, notes, frame, table.columns), out)
        print(f"TABLE {name}: {len(frame)} rows -> {out / (name + '.md')}")
        for row in frame[frame["metric"] == "macro_error"].to_dict("records"):
            print(f"  {row['corridor']:<7} {row['pair']:<60} macro error {row['mean_candidate']:.4f} vs "
                  f"{row['mean_reference']:.4f}: {row['difference']:+.4f} [{row['difference_low']:+.4f}, "
                  f"{row['difference_high']:+.4f}] ({row['pairs']} pairs, p {row.get('wilcoxon_p', float('nan')):.4f}, "
                  f"{row['outcome']})")  # fmt: skip
        if name == "het_fd_pairs":
            print(f"  H-N5: {h_n5_branch(frame)}")
    print(f"m11_pairs: {time.perf_counter() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
