"""Paired corridor comparisons of the laws of D134 (docs/m12_contract.md, agent A'): the recurrent laws trained with the
windowed-gain penalty with the maximum over the frequencies.

    python scripts/analysis/wmax_pairs.py            # after scripts/corridor_metrics.py (the macro.json of the runs)

The table builder of the factorial ablation (``table_ablation_pairs`` of ``cf_stability/eval/corridor_tables.py``,
``ablation_pairs: [candidate, reference]``) with the design of ``configs/corridor_metrics.yaml`` and the pairs of
``wmax_pairs``: ``gru_wmax`` against ``gru_windowed`` (the mean over the frequencies of M11, D128) and against ``gru``
(no penalty); ``lstm_wmax`` against ``lstm_windowed`` and ``lstm``; ``perl_wmax`` against ``perl_windowed`` and ``perl``
(reported). Per corridor, pair and metric (macro error, dynamic macro error, collisions per 1000 vehicle-km): the
difference candidate - reference paired by scenario and seed (the same seed: the same draws of the fold members) with
its percentile bootstrap interval, the relative change with its interval and the Wilcoxon p-value, as in
``ablation_pairs``. Writes ``runs/_tables/m12/wmax_pairs.{csv,md}``. Descriptive, no verdict (the secondary numbers of
D134).
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

NAME = "wmax_pairs"
TITLE = ("Windowed-gain penalty with the maximum over the frequencies (D134, H-N3'): paired differences against the "
         "mean over the frequencies (M11) and against no penalty")  # fmt: skip
PAIRS = (
    ("gru_wmax", "gru_windowed"), ("gru_wmax", "gru"),
    ("lstm_wmax", "lstm_windowed"), ("lstm_wmax", "lstm"),
    ("perl_wmax", "perl_windowed"), ("perl_wmax", "perl"),  # reported
)  # fmt: skip


def make_table(raw: dict, out: Path, pairs: Sequence[tuple[str, str]] = PAIRS) -> tuple[Table, list[str]]:
    """The table of the pairs from the design of ``configs/corridor_metrics.yaml`` (``raw``) and the missing lines."""
    paths = raw["paths"]
    corridor_root, runs_root = resolve_path(paths["corridor_root"]), resolve_path(paths["runs_root"])
    base = CorridorTablesConfig.from_mapping(raw.get("design") or {}, corridor_root, runs_root, out / "_unused",
                                             out / "_unused")  # fmt: skip
    maker = CorridorTableMaker(dataclasses.replace(base, ablation_pairs=tuple(pairs)))
    table = maker.table_ablation_pairs()
    frame = table.frame.copy()
    if len(frame):
        frame["pair"] = frame["candidate"] + " - " + frame["reference"]
    notes = [note for note in table.notes if not note.startswith("Factorial ablation")]
    notes.insert(0, f"{TITLE}. Candidate and reference per corridor, paired by scenario and seed (unit run): the builder "
                    "of ablation_pairs (cf_stability/eval/corridor_tables.py) with these pairs. gru_wmax, lstm_wmax, "
                    "perl_wmax: the recurrent laws trained on follownet_highd with the penalty of D134 at the weight "
                    "chosen on its pilot and fine-tuned on ngsim_i80 with it (configs/queue/m12_ft_wmax.yaml); "
                    "gru_windowed, lstm_windowed, perl_windowed: the same with the mean over the frequencies (M11, "
                    "D128); gru, lstm, perl: no penalty (D97). PERL is reported. Descriptive, no verdict.")  # fmt: skip
    missing = [line.split("] ", 1)[-1] for line in maker.missing if line.startswith("[ablation_pairs]")]
    if missing:
        notes.append(f"Missing: {'; '.join(missing[:6])}.")
    return Table(NAME, TITLE, notes, frame, table.columns), missing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/corridor_metrics.yaml")
    parser.add_argument("--out", default="runs/_tables/m12")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    raw = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    out = resolve_path(args.out)
    table, missing = make_table(raw, out)
    write_table(table, out)
    frame = table.frame
    print(f"TABLE {NAME}: {len(frame)} rows, {len(missing)} missing -> {out / (NAME + '.md')}")
    for row in frame[frame["metric"] == "macro_error"].to_dict("records") if len(frame) else []:
        print(f"  {row['corridor']:<7} {row['pair']:<28} macro error {row['mean_candidate']:.4f} vs "
              f"{row['mean_reference']:.4f}: {row['difference']:+.4f} [{row['difference_low']:+.4f}, "
              f"{row['difference_high']:+.4f}] ({row['pairs']} pairs, p {row.get('wilcoxon_p', float('nan')):.4f}, "
              f"{row['outcome']})")  # fmt: skip
    print(f"wmax_pairs: {time.perf_counter() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
