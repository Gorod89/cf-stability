"""Extract the car-following events of one dataset and write its driver and site splits.

    python scripts/extract_events.py data=<name> [extraction.max_duration=60 ...]
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections import Counter
from pathlib import Path

if importlib.util.find_spec("cf_stability") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import DictConfig

from cf_stability.data import get_builder
from cf_stability.data.extraction import ExtractionConfig
from cf_stability.data.schema import EventSet
from cf_stability.data.splits import write_splits
from cf_stability.utils import config_hash, git_revision, resolve_path, to_plain

VIEW_KEYS = ("events", "splits", "filter")  # a data config with one of them is a view of another event set (D87)


def refuse_view(data_cfg: dict) -> None:
    """A view (``configs/data/openacc_acc.yaml``, ...) reads the events of another set; extracting it would
    write the unfiltered events under the view's name and replace its splits."""
    keys = [key for key in VIEW_KEYS if key in data_cfg]
    if keys:
        raise ValueError(
            f"data={data_cfg.get('name')} is a view of the event set {data_cfg.get('events', data_cfg.get('name'))!r} "
            f"(keys {keys}); extract that set instead, the view is applied by scripts/train.py"
        )


def without_keys(cfg, names: tuple[str, ...]):
    """Copy of a nested config without the given keys."""
    if isinstance(cfg, dict):
        return {k: without_keys(v, names) for k, v in cfg.items() if k not in names}
    if isinstance(cfg, list):
        return [without_keys(v, names) for v in cfg]
    return cfg


def extraction_config(cfg: dict) -> ExtractionConfig:
    """Extraction options; a non-null ``data.max_duration`` overrides ``extraction.max_duration``."""
    options = dict(cfg["extraction"])
    if cfg["data"].get("max_duration") is not None:
        options["max_duration"] = cfg["data"]["max_duration"]
    return ExtractionConfig.from_mapping(options)


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract_events")
def main(cfg: DictConfig) -> None:
    plain = to_plain(cfg)
    data_cfg = plain["data"]
    refuse_view(data_cfg)
    name = data_cfg["name"]
    extraction = extraction_config(plain)
    options = dataclasses.asdict(extraction)
    # the hash covers what determines the events, not where they are written or how many processes ran
    digest = config_hash({"data": without_keys(data_cfg, ("n_jobs",)), "extraction": options})

    stats: Counter = Counter()
    events = get_builder(data_cfg["loader"])(data_cfg, extraction, stats)
    if len(events) == 0:
        raise RuntimeError(f"no events extracted for {name}: {dict(stats)}")
    events.validate(dt=extraction.dt)

    manifest = {
        "config": plain,
        "extraction": options,
        "config_hash": digest,
        "git_revision": git_revision(),
        "stats": dict(sorted(stats.items())),  # counters plus loader-specific entries (floats, dicts)
        "events_per_site": dict(sorted(Counter(ev.site for ev in events).items())),
    }
    events_dir = resolve_path(plain["paths"]["events_root"]) / name
    splits_dir = resolve_path(plain["paths"]["splits_root"])
    if data_cfg["loader"] == "highd":
        # highD licence: extracted events must stay inside the project
        from cf_stability.data.highd import assert_inside_project

        assert_inside_project(events_dir)
        assert_inside_project(splits_dir)
    out_dir = events.to_parquet(events_dir, manifest=manifest)
    split_paths = write_splits(
        events.events_frame(),
        name,
        splits_dir,
        n_folds=plain["splits"]["n_folds"],
        seed=plain["splits"]["seed"],
        extra={"events_config_hash": digest},
    )
    print_summary(events, stats, out_dir, split_paths, digest)


def print_summary(events: EventSet, stats: Counter, out_dir: Path, split_paths: list[Path], digest: str) -> None:
    summary = events.summary()
    quantiles = np.quantile([ev.duration for ev in events], [0.0, 0.1, 0.5, 0.9, 1.0])
    print(
        f"events {summary['n_events']}  samples {summary['n_samples']}  "
        f"followers {summary['n_followers']}  sites {summary['n_sites']}"
    )
    print("duration [s]  min {:.1f}  p10 {:.1f}  median {:.1f}  p90 {:.1f}  max {:.1f}".format(*quantiles))
    print("stats  " + "  ".join(f"{key}={stats[key]}" for key in sorted(stats)))
    print(f"events -> {out_dir}")
    for path in split_paths:
        print(f"split  -> {path}")
    print(f"config hash {digest}")


if __name__ == "__main__":
    main()
