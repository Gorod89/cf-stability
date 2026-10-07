"""Tables of the M4 report from the runs of E1-E5 (docs/m4_contract.md, section 4).

    python scripts/make_tables.py
    python scripts/make_tables.py tables=[e1,e2_sweep] paths.runs_root=runs

Reads the runs under ``paths.runs_root`` (only reads) and writes ``<out_dir>/<table>.md|csv`` for the
tables e1 (H1.1), e2_sweep (choice of the weight, D85), e2 (H1.2), e2_lowfreq (the combined penalty of
the recurrent models, H1.2 (combined), D110), e3 (transfer), e4 (H1.5) and e5 (H1.3 on the
collision-free prefix of the platoons, D109), ``verdicts.md|csv``, ``chosen_weights.json`` and
``missing.txt`` (every missing run, file or value); ``out_dir`` defaults to ``<runs_root>/_tables/m4``.
The table of M8 e2_monotone (the monotonicity terms only, D117) goes to ``paths.out_dir_m8``
(``<runs_root>/_tables/m8``), its notes to the same missing.txt.
Prints one line per table. The logic and the units are described in ``cf_stability/eval/tables.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.eval.tables import TablesConfig, make_tables  # noqa: E402
from cf_stability.utils import resolve_path, to_plain  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="make_tables")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    paths = plain.pop("paths")
    runs_root = resolve_path(paths["runs_root"])
    out_dir = resolve_path(paths["out_dir"]) if paths.get("out_dir") else runs_root / "_tables" / "m4"
    m8_dir = resolve_path(paths["out_dir_m8"]) if paths.get("out_dir_m8") else runs_root / "_tables" / "m8"
    for line in make_tables(TablesConfig.from_mapping(plain, runs_root, out_dir, m8_dir)):
        print(line, flush=True)


if __name__ == "__main__":
    main()
