import numpy as np
import pandas as pd
import pytest

from cf_stability.data.splits import fold_ids, group_labels, load_split, make_group_folds, make_splits, write_splits


def make_frame(n_followers: int = 40, n_sites: int = 6, seed: int = 0) -> pd.DataFrame:
    """events_frame-like table: 1-4 events per follower, 150-2000 samples each."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_followers):
        site = f"site{f % n_sites}"
        for k in range(int(rng.integers(1, 5))):
            rows.append(
                {
                    "event_id": f"ds/{site}/{f}|{f + 100}|{k}",
                    "dataset": "ds",
                    "site": site,
                    "follower_id": f"ds/{site}/{f}",
                    "n_samples": int(rng.integers(150, 2000)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kind", ["driver", "site"])
def test_each_event_in_one_fold_and_no_group_in_two(kind):
    frame = make_frame()
    split = make_splits(frame, kind)
    assert (split["dataset"], split["kind"], split["n_folds"], split["seed"]) == ("ds", kind, 5, 0)
    assert sorted(split["folds"]) == sorted(frame["event_id"])
    assert set(split["folds"].values()) == set(range(5))
    for event_id, group in zip(frame["event_id"], group_labels(frame, kind)):
        assert split["folds"][event_id] == split["groups"][group]
    assert "note" not in split and "n_folds_requested" not in split


def test_deterministic_for_a_seed():
    frame = make_frame()
    shuffled = frame.sample(frac=1.0, random_state=1)
    assert make_splits(frame, "driver", seed=0) == make_splits(shuffled, "driver", seed=0)
    assert make_splits(frame, "driver", seed=0)["groups"] != make_splits(frame, "driver", seed=1)["groups"]


def test_folds_balanced_in_samples():
    frame = make_frame()
    split = make_splits(frame, "driver")
    load = frame.groupby(frame["event_id"].map(split["folds"]))["n_samples"].sum()
    heaviest_group = frame.groupby("follower_id")["n_samples"].sum().max()
    assert load.max() - load.min() <= heaviest_group  # guarantee of the greedy assignment
    assert load.max() <= 1.25 * load.mean()


def test_fewer_groups_than_folds():
    frame = make_frame(n_followers=12, n_sites=3)
    split = make_splits(frame, "site")
    assert split["n_folds"] == 3 and split["n_folds_requested"] == 5 and "leave-one-group-out" in split["note"]
    assert sorted(split["groups"].values()) == [0, 1, 2]


def test_single_group_gives_one_fold(tmp_path):
    frame = make_frame(n_followers=4, n_sites=1)
    driver_path, site_path = write_splits(frame, "ds", tmp_path)
    site = load_split(site_path)
    assert site["n_folds"] == 1 and site["n_folds_requested"] == 5 and site["note"]
    train, val, test = fold_ids(site, 0)
    assert train == [] and val == [] and sorted(test) == sorted(frame["event_id"])
    assert load_split(driver_path)["n_folds"] == 4


@pytest.mark.parametrize("k", range(2))
def test_two_folds_take_validation_from_the_training_fold(k):
    frame = make_frame(n_followers=30, n_sites=2)
    split = make_splits(frame, "site")
    assert split["n_folds"] == 2
    train, val, test = fold_ids(split, k)
    assert sorted(train + val + test) == sorted(frame["event_id"])
    assert set(test) == {e for e, f in split["folds"].items() if f == k}
    assert train and 0.2 * (len(train) + len(val)) <= len(val) < 0.5 * (len(train) + len(val))
    followers = lambda ids: {e.split("|", 1)[0] for e in ids}  # noqa: E731
    assert not followers(train) & followers(val)
    assert fold_ids(split, k) == (train, val, test)


@pytest.mark.parametrize("k", range(5))
def test_fold_ids_partition(k):
    frame = make_frame()
    split = make_splits(frame, "driver")
    train, val, test = fold_ids(split, k)
    assert sorted(train + val + test) == sorted(frame["event_id"])
    assert len(set(train) | set(val) | set(test)) == len(frame)
    assert set(test) == {e for e, f in split["folds"].items() if f == k}
    assert set(val) == {e for e, f in split["folds"].items() if f == (k + 1) % 5}
    with pytest.raises(ValueError):
        fold_ids(split, 5)


def test_write_and_load(tmp_path):
    frame = make_frame()
    paths = write_splits(frame, "ds", tmp_path / "splits", n_folds=4, seed=3, extra={"events_config_hash": "abc"})
    assert [p.name for p in paths] == ["ds_driver.json", "ds_site.json"]
    for path, kind in zip(paths, ("driver", "site")):
        assert load_split(path) == {"events_config_hash": "abc", **make_splits(frame, kind, n_folds=4, seed=3)}


def test_make_group_folds_sums_item_weights():
    folds = make_group_folds(["a", "b", "a", "c"], [1, 5, 1, 3], n_folds=3, seed=0)
    assert sorted(folds) == ["a", "b", "c"] and sorted(folds.values()) == [0, 1, 2]
    with pytest.raises(ValueError):
        make_group_folds(["a", "a"], [1, 2], n_folds=2)
