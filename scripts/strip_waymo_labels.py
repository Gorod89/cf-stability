"""Replace the labels of the Waymo pairs in the per-event tables by ordinal numbers (D143).

    python scripts/strip_waymo_labels.py              # the transfer tables of the runs and the calibrations per event
    python scripts/strip_waymo_labels.py --dry-run    # report only

The tables are ``runs/**/transfer_waymo.parquet`` and ``data/calibration/waymo/idm_per_event*.parquet``.

The identifiers of the Waymo events are built from the segment number and the local vehicle number of the pairs of
Hu et al. (``cf_stability/data/waymo.py``). The per-event tables of this repository carry ordinal numbers in their
place (``cf_stability.data.waymo.ordinal_ids``): the position of the identifier in the sorted list of the identifiers
of the event set, read from the fold definition ``data/splits/waymo_driver.json``, which ``scripts/extract_events.py
data=waymo`` writes from the reader's own copy of the pairs. The rows keep their order and every other column its
values, and the numbers sort as the labels do, so the tables built from these files do not change.

A table that ``scripts/evaluate_transfer.py`` or ``scripts/calibrate_idm.py`` writes anew carries the labels again:
run this script on it before it is compared with the published table or published. A table that is already numbered
is left as it is. A table whose identifiers are not all labels of the event set (an identifier outside the set, or
labels and numbers mixed) is left as it is and named, and the exit status is then 1.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from cf_stability.data.waymo import DATASET, ordinal_ids  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

COLUMNS = ("event_id", "follower_id")
TRANSFER, CALIBRATION = f"transfer_{DATASET}.parquet", "idm_per_event*.parquet"


def tables(runs_root: Path, calibration_root: Path) -> list[Path]:
    """The per-event tables of the Waymo pairs: the transfer tables of the runs and the calibrations per event."""
    return [*sorted(runs_root.rglob(TRANSFER)), *sorted((calibration_root / DATASET).glob(CALIBRATION))]


def number_table(path: Path, events: dict[str, str], followers: dict[str, str], write: bool = True) -> str:
    """``numbered`` (nothing to do), ``rewritten`` (or to rewrite) or ``unknown`` (not all labels of the set)."""
    frame = pd.read_parquet(path)
    found = [set(frame[column]) for column in COLUMNS]
    if found[0] <= set(events.values()) and found[1] <= set(followers.values()):
        return "numbered"
    if not (found[0] <= set(events) and found[1] <= set(followers)):
        return "unknown"
    if write:
        for column, mapping in zip(COLUMNS, (events, followers)):
            numbers = frame[column].astype(object).map(mapping).astype(frame[column].dtype)
            if numbers.isna().any():  # a column type that cannot hold the numbers (a fixed set of categories)
                raise ValueError(f"{path}: column {column} of type {frame[column].dtype} cannot hold the numbers")
            frame[column] = numbers
        frame.to_parquet(path, index=False)
    return "rewritten"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--calibration-root", default="data/calibration")
    parser.add_argument("--split", default=f"data/splits/{DATASET}_driver.json", help="fold definition of the set")
    parser.add_argument("--dry-run", action="store_true", help="report the tables that would be rewritten")
    args = parser.parse_args(argv)
    split_path = resolve_path(args.split)
    if not split_path.is_file():
        print(f"{split_path} is missing: scripts/extract_events.py data={DATASET} writes it from a copy of the pairs")
        return 1
    split = read_json(split_path)
    events, followers = ordinal_ids(split["folds"], split["groups"])
    paths = tables(resolve_path(args.runs_root), resolve_path(args.calibration_root))
    counts: Counter = Counter()
    for path in paths:
        state = number_table(path, events, followers, write=not args.dry_run)
        counts[state] += 1
        if state == "unknown":
            print(f"identifiers that are not all labels of the event set, left as it is: {path}")
    print(
        f"{len(paths)} tables ({len(events)} events, {len(followers)} followers): {counts['rewritten']} "
        f"{'to rewrite' if args.dry_run else 'rewritten'}, {counts['numbered']} already numbered, "
        f"{counts['unknown']} left as they are"
    )
    return 1 if counts["unknown"] else 0


if __name__ == "__main__":
    sys.exit(main())
