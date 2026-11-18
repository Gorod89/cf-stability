"""Ordinal identifiers in the per-event tables of the Waymo pairs (D143): ``ordinal_ids`` keeps the order of the
labels and the automated vehicle; ``scripts/strip_waymo_labels.py`` rewrites the tables once, keeps their rows and
values, leaves the statistics over followers unchanged and refuses a table with an identifier outside the event set
or with labels and numbers mixed. The labels of this file are made up."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cf_stability.data.waymo import ordinal_ids
from cf_stability.eval.stats import driver_table
from cf_stability.utils import write_json

ROOT = Path(__file__).resolve().parents[1]
FOLLOWERS = [  # segment numbers outside the range of the file of the pairs (3 to 1000)
    "waymo/all/AV", "waymo/all/s90010_v13", "waymo/all/s9001_v51", "waymo/all/s9020_v7", "waymo/all/s9090_v40",
]  # fmt: skip
EVENTS = [
    "waymo/all/AV|s9001_v3|s9001|0", "waymo/all/AV|s9090_v2|s9090|0", "waymo/all/s90010_v13|s90010_v8|s90010|0",
    "waymo/all/s9001_v51|AV|s9001|0", "waymo/all/s9001_v51|AV|s9001|1", "waymo/all/s9020_v7|s9020_v1|s9020|0",
    "waymo/all/s9090_v40|s9090_v2|s9090|0",
]  # fmt: skip


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("strip_waymo_labels", ROOT / "scripts" / "strip_waymo_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, tree: Path, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    argv = ["--runs-root", str(tree / "runs"), "--calibration-root", str(tree / "calibration"),
            "--split", str(tree / "splits" / "waymo_driver.json"), *extra]  # fmt: skip
    with contextlib.redirect_stdout(out):
        code = module.main(argv)
    return code, out.getvalue()


def table(order: list[int], shift: float) -> pd.DataFrame:
    events = [EVENTS[i] for i in order]
    return pd.DataFrame({
        "event_id": events, "follower_id": [e.split("|", 1)[0] for e in events],
        "rmse_s": shift + np.asarray(order, dtype=float) ** 2, "collided": [i % 3 == 0 for i in order],
    })  # fmt: skip


def make_tree(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "a": tmp_path / "runs" / "e1" / "set" / "mlp" / "driver_fold0_seed0" / "transfer_waymo.parquet",
        "b": tmp_path / "runs" / "e1" / "set" / "mlp" / "driver_fold1_seed0" / "transfer_waymo.parquet",
        "other": tmp_path / "runs" / "e1" / "set" / "mlp" / "driver_fold0_seed0" / "transfer_ngsim_i80.parquet",
        "calibration": tmp_path / "calibration" / "waymo" / "idm_per_event.parquet",
    }
    content = {"a": ([3, 0, 6, 1, 4, 2, 5], 0.5), "b": ([3, 0, 6, 1, 4, 2, 5], 1.5), "other": ([0, 1], 0.0),
               "calibration": ([6, 5, 4, 3, 2, 1, 0], 2.0)}  # fmt: skip
    for name, (order, shift) in content.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        table(order, shift).to_parquet(paths[name], index=False)
    write_json(tmp_path / "splits" / "waymo_driver.json",
               {"groups": {f: i % 2 for i, f in enumerate(FOLLOWERS)}, "folds": {e: 0 for e in EVENTS}})  # fmt: skip
    return paths


def test_ordinal_ids_keep_the_order_and_the_automated_vehicle():
    events, followers = ordinal_ids(reversed(EVENTS), FOLLOWERS + FOLLOWERS)
    assert [events[e] for e in sorted(EVENTS)] == [f"waymo/all/e{k:04d}" for k in range(len(EVENTS))]
    numbers = [f"waymo/all/f{k:04d}" for k in range(1, 5)]
    assert [followers[f] for f in sorted(FOLLOWERS)] == ["waymo/all/AV", *numbers]
    assert sorted(followers.values()) == [followers[f] for f in sorted(FOLLOWERS)]
    assert not any("_v" in value or "|" in value for value in [*events.values(), *followers.values()])


def test_script_numbers_the_tables_once_and_keeps_the_statistics(script, tmp_path):
    paths = make_tree(tmp_path)
    before = {name: pd.read_parquet(path) for name, path in paths.items()}
    code, out = run(script, tmp_path, "--dry-run")
    assert code == 0 and "3 to rewrite, 0 already numbered" in out
    pd.testing.assert_frame_equal(pd.read_parquet(paths["a"]), before["a"])  # a dry run writes nothing

    code, out = run(script, tmp_path)
    assert code == 0 and "3 tables (7 events, 5 followers): 3 rewritten, 0 already numbered, 0 left" in out
    events, followers = ordinal_ids(EVENTS, FOLLOWERS)
    for name in ("a", "b", "calibration"):
        after = pd.read_parquet(paths[name])
        assert list(after["event_id"]) == [events[e] for e in before[name]["event_id"]]
        assert list(after["follower_id"]) == [followers[f] for f in before[name]["follower_id"]]
        assert after.dtypes.equals(before[name].dtypes)
        pd.testing.assert_frame_equal(after.drop(columns=["event_id", "follower_id"]),
                                      before[name].drop(columns=["event_id", "follower_id"]))  # fmt: skip
    pd.testing.assert_frame_equal(pd.read_parquet(paths["other"]), before["other"])  # another target set is not touched

    labelled = driver_table([before["a"], before["b"]])
    numbered = driver_table([pd.read_parquet(paths["a"]), pd.read_parquet(paths["b"])])
    assert list(numbered.index) == [followers[f] for f in labelled.index]
    np.testing.assert_array_equal(numbered.to_numpy(), labelled.to_numpy())

    stamp = paths["a"].stat().st_mtime_ns
    code, out = run(script, tmp_path)
    assert code == 0 and "0 rewritten, 3 already numbered" in out and paths["a"].stat().st_mtime_ns == stamp


def test_script_refuses_identifiers_outside_the_event_set(script, tmp_path):
    paths = make_tree(tmp_path)
    stray = table([0, 1], 0.0)
    stray.loc[1, "event_id"] = "waymo/all/s9500_v5|s9500_v1|s9500|0"
    stray.to_parquet(paths["b"], index=False)
    code, out = run(script, tmp_path)
    assert code == 1 and "not all labels of the event set, left as it is" in out and "2 rewritten" in out
    pd.testing.assert_frame_equal(pd.read_parquet(paths["b"]), stray)

    mixed = pd.read_parquet(paths["a"])  # numbered by the run above; one label put back
    mixed.loc[0, "event_id"] = EVENTS[3]
    mixed.to_parquet(paths["a"], index=False)
    code, out = run(script, tmp_path)
    assert code == 1 and "0 rewritten, 1 already numbered, 2 left as they are" in out
    pd.testing.assert_frame_equal(pd.read_parquet(paths["a"]), mixed)

    (tmp_path / "splits" / "waymo_driver.json").unlink()
    code, out = run(script, tmp_path)
    assert code == 1 and "is missing" in out and "extract_events.py data=waymo" in out
