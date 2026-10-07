"""Macroscopic metrics of the corridor runs and ground truths, and the tables of M5 (docs/m5_contract.md, 5-6).

    python scripts/corridor_metrics.py                                   # every ground truth and run
    python scripts/corridor_metrics.py scenario=i80_p1 law=idm_global seed=3
    python scripts/corridor_metrics.py tables=true                       # then runs/_tables/m5/*.md|csv and the
                                                                         # corridor tables of M8 in runs/_tables/m8/

Writes ``macro.json`` next to every selected ``trajectories.npz`` (runs) and ``ground_truth.npz`` (ground
truths, ``runs/corridor/scenarios/<scenario>/``); a file that is up to date is kept unless ``force=true``.
The settings ``macro`` apply to every scenario; a scenario of another geometry (US-101, D112) carries its
detectors and the lanes of its wave field in ``scenario.json["metrics"]``, which apply over them.
Prints one line per file (``OK``, ``KEPT``, ``SKIPPED``, ``FAILED``); a failing file does not stop the
others. With ``tables=true`` writes the tables ``laws``, ``components``, ``instability``, ``tost``, the
verdicts of H12 (``h12_1``, ``h12_2``, ``verdicts``), ``e4_rmax``, ``sensitivity`` and ``missing.txt``
(``cf_stability/eval/corridor_tables.py``), and the corridor tables of M8 to ``paths.out_dir_m8``
(``asymmetry``, ``asymmetry_contrasts``, ``correlation_pooled``, ``power``, ``temporal``, ``missing_corridor.txt``;
the ``asymmetry.json`` files come from ``scripts/corridor_asymmetry.py``) with those of the review of M8
(``correlation_clustered``: H12.3 with law and family clusters; ``contacts_absolute``: contact episodes with their
exposure; ``h12_2_error``: the certified hybrid against the IDM on the errors), and prints one line per table. The
metrics process never imports ``libsumo`` (docs/m5_contract.md, 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.corridor.macro import Geometry, MacroConfig, update_macro  # noqa: E402
from cf_stability.utils import resolve_path, to_plain  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="corridor_metrics")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    paths = plain["paths"]
    corridor_root = resolve_path(paths["corridor_root"])
    runs_root = resolve_path(paths["runs_root"])
    geometry, fallback = plain.get("geometry") or {}, Geometry()
    default = Geometry(float(geometry.get("x_in", fallback.x_in)), float(geometry.get("x_out", fallback.x_out)),
                       tuple(float(w) for w in geometry.get("window", fallback.window)))  # fmt: skip
    if plain.get("compute", True):
        seed = plain.get("seed")
        lines = update_macro(
            corridor_root, MacroConfig.from_mapping(plain["macro"]), default, scenario=plain.get("scenario"),
            law=plain.get("law"), seed=None if seed is None else int(seed), force=bool(plain.get("force")),
        )  # fmt: skip
        for line in lines:
            print(line, flush=True)
    if plain.get("tables"):
        from cf_stability.eval.corridor_tables import CorridorTablesConfig, make_corridor_tables

        out_dir = resolve_path(paths["out_dir"]) if paths.get("out_dir") else runs_root / "_tables" / "m5"
        m8_dir = resolve_path(paths["out_dir_m8"]) if paths.get("out_dir_m8") else runs_root / "_tables" / "m8"
        tables_cfg = CorridorTablesConfig.from_mapping(plain.get("design") or {}, corridor_root, runs_root, out_dir,
                                                       m8_dir)  # fmt: skip
        for line in make_corridor_tables(tables_cfg):
            print(line, flush=True)


if __name__ == "__main__":
    main()
