"""Report of the project (docs/m6_contract.md; section 7 of the specification).

    python scripts/make_report.py
    python scripts/make_report.py refresh=true     # first rerun make_tables.py and corridor_metrics.py tables=true
    python scripts/make_report.py corridor_source=fields   # figures 5 and 6 from fields.npz only (as published)

Reads the tables of M4, M5 and M8 (``runs/_tables/m4|m5|m8/*.csv``), the run files of the figures (the corridor
runs: ``trajectories.npz``, else ``fields.npz`` of ``scripts/export_fields.py``) and the figures of the supplement
(``runs/_report/supplement/figures/<name>.png|pdf`` with the caption ``<name>.txt``), and writes ``runs/_report/``:
``tables/<name>.csv|tex`` (every table of M4, M5 and M8, the verdicts, the existence arm of E2, the components and
the dynamic macro error), ``figures/<name>.png|pdf`` (the seven figures), ``report.md`` (sections 1-10 of the
contract and section 11, the supplement of M8) and ``manifest.json`` (run counts, hashes of the inputs, software
versions, date). Missing inputs give empty tables, missing or incomplete figures and notes in ``report.md`` (an
expected table or figure of M8 without its file: a note), except that with ``strict=true`` (the default) a panel of
the corridor figures without its input stops the report with an error naming the run and the file. Prints one line
per output group; ``FIGURES n of 7`` counts the complete figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.eval.report import ReportConfig, make_report  # noqa: E402
from cf_stability.utils import resolve_path, to_plain  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="make_report")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    paths = dict(plain.pop("paths"))
    runs_root = resolve_path(paths["runs_root"])
    defaults = {"out_dir": runs_root / "_report", "tables_m4": runs_root / "_tables" / "m4",
                "tables_m5": runs_root / "_tables" / "m5", "tables_m8": runs_root / "_tables" / "m8",
                "corridor_root": runs_root / "corridor"}  # fmt: skip
    resolved = {key: resolve_path(paths[key]) if paths.get(key) else defaults[key] for key in defaults}
    resolved["supplement_figures"] = (resolve_path(paths["supplement_figures"]) if paths.get("supplement_figures")
                                      else resolved["out_dir"] / "supplement" / "figures")  # fmt: skip
    resolved.update(runs_root=runs_root, docs_dir=resolve_path(paths["docs_dir"]),
                    configs_dir=resolve_path(paths["configs_dir"]))  # fmt: skip
    config = {**plain, "paths": {key: str(value) for key, value in resolved.items()}}
    for line in make_report(ReportConfig.from_mapping(plain, resolved), config):
        print(line, flush=True)


if __name__ == "__main__":
    main()
