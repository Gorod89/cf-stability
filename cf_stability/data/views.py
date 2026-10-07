"""Views of an event set and the parts of a fold (D87; docs/m4_contract.md, 3.1; D120).

A data config may carry ``events`` (directory under the events root, default ``name``), ``splits``
(prefix of the split files, default the events directory) and ``filter`` with the keys

* ``mode``: the follower drives in this mode (``follower_id`` ends with ``:<mode>``; OpenACC, D87);
* ``exclude_files``: run files left out (``meta["file"]``, or ``<campaign folder>/<file>``; D87);
* ``follower_prefix``: ``follower_id`` starts with this text (e.g. ``ngsim_i80/i80/p0_``, the drivers of
  period 0 of I-80; D120).

The filter is applied to the three parts of the fold (training, validation, test). Used by
``scripts/train.py`` and, to rebuild the training part of a run from its stored config, by
``scripts/audit_stability.py`` (spacing band of other quantiles, D119).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_stability.data.schema import Event, EventSet
from cf_stability.data.splits import fold_ids, load_split
from cf_stability.utils import resolve_path

PARTS = ("train", "val", "test")
FILTER_KEYS = ("mode", "exclude_files", "follower_prefix")


def view_filter(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """``filter`` of a data config, checked; None without filter. ``follower_prefix`` (D120) is part of the
    rule only when it is set, so that the rule (and ``metrics.json["data_view"]``) of the views of D87 keeps
    its form."""
    raw = data.get("filter")
    if not raw:
        return None
    unknown = set(raw) - set(FILTER_KEYS)
    if unknown:
        raise ValueError(f"data={data['name']}: unknown filter keys {sorted(unknown)} (accepted: {list(FILTER_KEYS)})")
    rule = {"mode": raw.get("mode"), "exclude_files": [str(f) for f in raw.get("exclude_files") or ()]}
    if raw.get("follower_prefix"):
        rule["follower_prefix"] = str(raw["follower_prefix"])
    return rule


def keep_event(event: Event, rule: Mapping[str, Any] | None) -> bool:
    """Whether ``event`` belongs to a view: the follower drives in ``rule["mode"]`` (``follower_id`` ends
    with ``:<mode>``), its ``follower_id`` starts with ``rule["follower_prefix"]`` (when given) and its run
    file is not in ``rule["exclude_files"]`` (entries ``<file>`` match ``meta["file"]``, entries
    ``<campaign folder>/<file>`` match that pair: file names repeat across the OpenACC campaigns)."""
    if rule is None:
        return True
    if rule["mode"] is not None and not event.follower_id.endswith(f":{rule['mode']}"):
        return False
    prefix = rule.get("follower_prefix")
    if prefix and not event.follower_id.startswith(prefix):
        return False
    file = event.meta.get("file")
    if file is None or not rule["exclude_files"]:
        return True
    return not ({file, f"{event.meta.get('campaign')}/{file}"} & set(rule["exclude_files"]))


def load_parts(
    data: Mapping[str, Any], split: str, fold: int, events_root: str | Path, splits_root: str | Path
) -> tuple[dict[str, list[Event]], dict[str, Any]]:
    """Training, validation and test events of the fold, restricted to the view of the data config ``data``,
    and the description of the view (``metrics.json["data_view"]``). An empty part is an error."""
    events_dir = data.get("events") or data["name"]
    splits_prefix = data.get("splits") or events_dir
    rule = view_filter(data)
    event_set = EventSet.from_parquet(resolve_path(events_root) / events_dir)
    split_file = resolve_path(splits_root) / f"{splits_prefix}_{split}.json"
    parts, counts = {}, {}
    for name, ids in zip(PARTS, fold_ids(load_split(split_file), fold)):
        events = event_set.subset(ids).events
        parts[name] = [ev for ev in events if keep_event(ev, rule)]
        counts[name] = {"kept": len(parts[name]), "dropped": len(events) - len(parts[name])}
    empty = [name for name in PARTS if not parts[name]]
    if empty:
        raise ValueError(
            f"data={data['name']}: fold {fold} of the {split} split {splits_prefix}_{split}.json has no "
            f"{' / '.join(empty)} events (events {events_dir}, filter {rule}; kept/dropped per part {counts})"
        )
    return parts, {"events": events_dir, "splits": splits_prefix, "filter": rule, "parts": counts}


def subsample(events: Sequence[Event], n: int | None, seed: int) -> list[Event]:
    """At most ``n`` of ``events`` (seeded draw, order kept): ``max_train_events`` of ``scripts/train.py``."""
    if n is None or n >= len(events):
        return list(events)
    keep = np.random.default_rng(seed).choice(len(events), size=n, replace=False)
    return [events[i] for i in np.sort(keep)]


def run_training_events(config: Mapping[str, Any]) -> list[Event]:
    """The training events of a run, rebuilt from the config stored in its ``metrics.json`` (``config``): the
    view, the split, the fold and ``max_train_events`` as ``scripts/train.py`` used them."""
    paths = config["paths"]
    fold, roots = int(config["fold"]), (paths["events_root"], paths["splits_root"])
    parts, _ = load_parts(config["data"], config["split"], fold, *roots)
    return subsample(parts["train"], config.get("max_train_events"), int(config["seed"]))
