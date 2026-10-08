"""Acceleration asymmetry and oscillation spectrum of the corridor runs and ground truths (docs/m8_contract.md,
section 9; D121).

    python scripts/corridor_asymmetry.py                                 # every ground truth and run
    python scripts/corridor_asymmetry.py workers=6                       # in a pool of 6 processes
    python scripts/corridor_asymmetry.py scenario=i80_p1 law=idm_global seed=3
    python scripts/corridor_asymmetry.py compute=false tables=true       # the tables of M8 of the corridor

Writes ``asymmetry.json`` next to ``macro.json`` of every selected run (``trajectories.npz``) and ground truth
(``runs/corridor/scenarios/<scenario>/``, ``ground_truth.npz``): the acceleration asymmetry index, the shares of
time accelerating and decelerating, and the Welch spectrum of the speed series of the virtual detectors with its
peak frequency and centroid (``cf_stability/corridor/asymmetry.py``). Uses ``configs/corridor_metrics.yaml``: the
settings ``asymmetry``, the detectors and wave lanes of ``macro`` (with the ``metrics`` settings of a scenario over
them), the filters ``scenario``, ``law``, ``seed``, ``force`` and ``workers``. A file that holds the hash of its
settings and is newer than its inputs is kept unless ``force=true``. Prints one line per file (``OK``, ``KEPT``,
``SKIPPED``, ``FAILED``). With ``tables=true`` writes the corridor tables of M8 (asymmetry, asymmetry_contrasts,
correlation_pooled, power, temporal) and of its review (correlation_clustered, contacts_absolute, h12_2_error,
ablation_pairs) to ``runs/_tables/m8/`` as ``scripts/corridor_metrics.py tables=true`` does.
The process never imports ``libsumo`` (docs/m5_contract.md, section 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.corridor.asymmetry import AsymmetryConfig, update_asymmetry  # noqa: E402
from cf_stability.corridor.macro import Geometry, MacroConfig  # noqa: E402
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
        lines = update_asymmetry(
            corridor_root, MacroConfig.from_mapping(plain["macro"]), AsymmetryConfig.from_mapping(plain.get("asymmetry")),
            default, scenario=plain.get("scenario"), law=plain.get("law"), seed=None if seed is None else int(seed),
            force=bool(plain.get("force")), workers=int(plain.get("workers") or 1),
        )  # fmt: skip
        counts: dict[str, int] = {}
        for line in lines:
            counts[line.split(" ", 1)[0]] = counts.get(line.split(" ", 1)[0], 0) + 1
            print(line, flush=True)
        print("DONE " + ", ".join(f"{key} {value}" for key, value in sorted(counts.items())), flush=True)
    if plain.get("tables"):
        from cf_stability.eval.corridor_tables import CorridorTablesConfig, make_m8_corridor_tables

        out_dir = resolve_path(paths["out_dir"]) if paths.get("out_dir") else runs_root / "_tables" / "m5"
        m8_dir = resolve_path(paths["out_dir_m8"]) if paths.get("out_dir_m8") else runs_root / "_tables" / "m8"
        tables_cfg = CorridorTablesConfig.from_mapping(plain.get("design") or {}, corridor_root, runs_root, out_dir,
                                                       m8_dir)  # fmt: skip
        for line in make_m8_corridor_tables(tables_cfg):
            print(line, flush=True)


if __name__ == "__main__":
    main()
