"""Queue of experiment jobs in separate processes (docs/m4_contract.md, section 2.1; D82).

    python scripts/run_experiment.py configs/queue/e2_sweep.yaml [--workers N] [--dry-run] [--only <substring>]
                                     [--max-jobs N]

Runs the chain of steps (train, audit, platoon, transfer, certificate) of every job of the queue
file with up to ``workers`` jobs at once, skipping complete steps, so that the queue can be
restarted at any moment. Writes ``<runs_root>/<name>/_logs/<job>.log`` (output of the steps of a job)
and ``<runs_root>/<name>/_status.json`` (state of every job, rewritten after every job) and prints
one line per finished job and the totals. The format of the queue file is described in
``cf_stability/eval/experiment_queue.py``.

Steps added here (:func:`register_steps`): ``audit_q01_99`` and ``audit_q10_90``, the audit with the
spacing band of other quantiles (D119, ``BAND_VARIANTS`` of ``cf_stability/stability/audit.py``). They
run ``scripts/audit_stability.py`` like ``audit`` and are complete when their own output file
(``stability_q01_99.json``, ``stability_q10_90.json``) is; the queue file passes the quantiles with
``step_overrides`` (``configs/queue/audit_bands.yaml``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import MutableMapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_stability.eval import experiment_queue  # noqa: E402
from cf_stability.eval.experiment_queue import bind_children_to_this_process, main  # noqa: E402
from cf_stability.stability.audit import BAND_VARIANTS, band_output  # noqa: E402

EXTRA_STEPS: dict[str, tuple[str, str]] = {  # step -> (script relative to the repository root, output file)
    f"audit_{name}": ("scripts/audit_stability.py", band_output(quantiles)) for name, quantiles in BAND_VARIANTS.items()
}


def register_steps(steps: MutableMapping[str, tuple[str, str]] | None = None) -> None:
    """Add the steps of ``EXTRA_STEPS`` to the steps the queue knows (``experiment_queue.STEPS`` by default)."""
    steps = experiment_queue.STEPS if steps is None else steps
    for name, entry in EXTRA_STEPS.items():
        steps.setdefault(name, entry)


if __name__ == "__main__":
    register_steps()
    bind_children_to_this_process()  # Windows: the steps end with the queue, however it ends
    sys.exit(main())
