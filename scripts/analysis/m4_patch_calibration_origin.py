"""Adds n_events and events_config_hash to cached fold calibrations written before 2026-09-29.

A file is patched only when its name hash equals the key of the training events of its fold with
the settings stored in the file; other files (sub-samples, views) are left alone.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from cf_stability.data.splits import fold_ids, load_split  # noqa: E402
from cf_stability.utils import config_hash, read_json, write_json  # noqa: E402

for name in sys.argv[2:]:
    folder = REPO / "data" / "calibration" / name
    manifest = read_json(REPO / "data" / "events" / name / "manifest.json")
    for path in sorted(folder.glob("*_global_*_fold*_*.json")):
        m = re.fullmatch(r"(idm|newell|ovm)_global_(driver|site)_fold(\d+)_([0-9a-f]+)", path.stem)
        fit = read_json(path)
        if not m or fit.get("events_config_hash"):
            print(f"{name}/{path.name}: {'has the origin' if m else 'name not recognised'}")
            continue
        law, split, fold, key = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        train_ids = fold_ids(load_split(REPO / "data" / "splits" / f"{name}_{split}.json"), fold)[0]
        expected = config_hash({"law": law, "event_ids": sorted(train_ids), "calibration": fit["settings"]})
        if expected != key:
            print(f"{name}/{path.name}: key differs from the training events of the fold, left alone")
            continue
        origin = {"data": fit.get("data", name), "split": split, "fold": fold, "n_events": len(train_ids),
                  "events_config_hash": manifest.get("config_hash")}
        rest = {k: v for k, v in fit.items() if k not in origin}
        write_json(path, {**origin, **rest})
        print(f"{name}/{path.name}: origin added ({len(train_ids)} events, {manifest.get('config_hash')})")
