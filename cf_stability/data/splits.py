"""Group k-fold splits at driver and site level (docs/data_contract.md, section 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cf_stability.utils import read_json, write_json

KINDS = ("driver", "site")


def make_group_folds(
    groups: Sequence[str], weights: Sequence[float], n_folds: int = 5, seed: int = 0
) -> dict[str, int]:
    """Assign groups to folds balanced by weight.

    ``groups`` and ``weights`` are per item (e.g. per event); the weight of a group is the
    sum over its items. The unique groups (sorted) are shuffled with
    ``numpy.random.default_rng(seed)`` and dealt, in that order, to the currently lightest
    fold (lowest index on ties).
    """
    totals: dict[str, float] = {}
    for group, weight in zip(groups, weights, strict=True):
        totals[str(group)] = totals.get(str(group), 0.0) + float(weight)
    names = sorted(totals)
    if not 1 <= n_folds <= len(names):
        raise ValueError(f"need 1 <= n_folds <= number of groups, got {n_folds} folds for {len(names)} groups")
    load = np.zeros(n_folds)
    folds = {}
    for i in np.random.default_rng(seed).permutation(len(names)):
        k = int(np.argmin(load))
        folds[names[i]] = k
        load[k] += totals[names[i]]
    return dict(sorted(folds.items()))


def group_labels(events_frame: pd.DataFrame, kind: str) -> pd.Series:
    """Group of every event: ``follower_id`` (driver) or ``"<dataset>/<site>"`` (site)."""
    if kind == "driver":
        return events_frame["follower_id"].astype(str)
    if kind == "site":
        return events_frame["dataset"].astype(str) + "/" + events_frame["site"].astype(str)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def make_splits(events_frame: pd.DataFrame, kind: str, n_folds: int = 5, seed: int = 0) -> dict:
    """Split payload of one kind for an ``EventSet.events_frame()``.

    With fewer groups than folds the number of folds is reduced to the number of groups
    (leave-one-group-out); ``n_folds_requested`` and ``note`` record it.
    """
    groups = group_labels(events_frame, kind)
    n_groups = int(groups.nunique())
    if n_groups == 0:
        raise ValueError("no events to split")
    n_eff = min(int(n_folds), n_groups)
    group_fold = make_group_folds(groups.tolist(), events_frame["n_samples"].tolist(), n_eff, seed)
    split: dict[str, Any] = {
        "dataset": "+".join(sorted(events_frame["dataset"].astype(str).unique())),
        "kind": kind,
        "n_folds": n_eff,
        "seed": int(seed),
        "groups": group_fold,
        "folds": {str(e): group_fold[g] for e, g in sorted(zip(events_frame["event_id"], groups))},
    }
    if n_eff < n_folds:
        split["n_folds_requested"] = int(n_folds)
        if n_eff == 1:
            split["note"] = f"only one {kind} group: single fold, no training or validation data"
        else:
            split["note"] = (
                f"only {n_groups} {kind} groups for {n_folds} requested folds: "
                f"n_folds reduced to {n_groups} (leave-one-group-out)"
            )
    return split


def write_splits(
    events_frame: pd.DataFrame,
    dataset: str,
    out_dir: str | Path,
    n_folds: int = 5,
    seed: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write ``<dataset>_driver.json`` and ``<dataset>_site.json``; ``extra`` keys are added."""
    paths = []
    for kind in KINDS:
        payload = {**(extra or {}), **make_splits(events_frame, kind, n_folds, seed)}
        paths.append(write_json(Path(out_dir) / f"{dataset}_{kind}.json", payload))
    return paths


def load_split(path: str | Path) -> dict:
    return read_json(path)


def fold_ids(
    split: Mapping[str, Any], k: int, val_fraction: float = 0.2
) -> tuple[list[str], list[str], list[str]]:
    """``(train_ids, val_ids, test_ids)`` of fold ``k``: test = fold k, validation = fold k + 1.

    With two folds a whole validation fold would leave nothing to train on: the fold that is
    not the test fold is divided by follower, at least ``val_fraction`` of its events going to
    validation (followers shuffled with ``default_rng([seed, k])``). With a single fold there
    is no training or validation data (all events are test events).
    """
    n = int(split["n_folds"])
    if not 0 <= k < n:
        raise ValueError(f"fold {k} out of range for {n} folds")
    items = sorted(split["folds"].items())
    test = [event_id for event_id, fold in items if fold == k]
    rest = [(event_id, fold) for event_id, fold in items if fold != k]
    if n > 2:
        val_fold = (k + 1) % n
        return (
            [event_id for event_id, fold in rest if fold != val_fold],
            [event_id for event_id, fold in rest if fold == val_fold],
            test,
        )
    by_follower: dict[str, list[str]] = {}
    for event_id, _ in rest:
        by_follower.setdefault(event_id.split("|", 1)[0], []).append(event_id)
    followers = sorted(by_follower)
    val: list[str] = []
    rng = np.random.default_rng([int(split["seed"]), k])
    for i in rng.permutation(len(followers)):
        if len(val) >= val_fraction * len(rest) or len(followers) < 2:
            break
        val.extend(by_follower[followers[i]])
    chosen = set(val)
    return [event_id for event_id, _ in rest if event_id not in chosen], sorted(val), test
