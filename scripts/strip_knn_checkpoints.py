"""Rewrite the kNN checkpoints without their samples (D137).

    python scripts/strip_knn_checkpoints.py                 # every runs/*/*/knn/*/model.pt, verified against a re-fit
    python scripts/strip_knn_checkpoints.py --runs-root runs --no-verify

A checkpoint written before D137 holds the sub-sample of the training events of its run; the rewritten checkpoint
holds the scaler, the number of samples and the seed only. With verification (the default) the samples re-fitted
from the training events of the run must equal the stored ones bit for bit, else the file is left as it is; the
verification needs the event data of the run (``data/events``, ``data/splits``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_stability.eval.knn_refit import knn_runs, strip_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--no-verify", action="store_true", help="rewrite without re-fitting from the events")
    args = parser.parse_args(argv)
    runs = knn_runs(args.runs_root)
    if not runs:
        print(f"no kNN run under {args.runs_root}")
        return 0
    failed = 0
    for run in runs:
        try:
            info = strip_checkpoint(run, verify=not args.no_verify)
        except Exception as exc:  # noqa: BLE001 - every run is reported, the exit status says it
            failed += 1
            print(f"FAILED {run}: {exc}")
            continue
        state = "legacy" if info["legacy"] else "already stripped"
        check = "verified" if info.get("verified") else "not verified"
        print(f"{run}: {state}, {info['n_points_stored']} stored points, {check}, "
              f"{info['bytes_before'] // 1024} -> {info['bytes_after'] // 1024} KB")
    print(f"{len(runs) - failed} of {len(runs)} checkpoints rewritten")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
