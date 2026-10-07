import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cf_stability.data import get_builder, synthetic
from cf_stability.data.schema import EventSet
from cf_stability.data.splits import load_split
from cf_stability.utils import REPO_ROOT, read_json

SCRIPT = REPO_ROOT / "scripts" / "extract_events.py"


def run_extraction(out: Path, cwd: Path) -> str:
    overrides = [
        "data=synthetic",
        "data.n_platoons=2",
        "data.n_vehicles=4",
        "data.duration=120.0",
        f"paths.events_root='{(out / 'events').as_posix()}'",
        f"paths.splits_root='{(out / 'splits').as_posix()}'",
        f"hydra.run.dir='{(out / 'hydra').as_posix()}'",
    ]
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *overrides],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_registry():
    assert get_builder("synthetic") is synthetic.build_events


def test_extraction_refuses_a_view(tmp_path):
    """A view (configs/data/openacc_acc.yaml) must not overwrite the events and splits of the set it reads."""
    overrides = [
        "data=openacc_acc",
        f"paths.events_root='{(tmp_path / 'events').as_posix()}'",
        f"paths.splits_root='{(tmp_path / 'splits').as_posix()}'",
        f"hydra.run.dir='{(tmp_path / 'hydra').as_posix()}'",
    ]
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *overrides],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    assert proc.returncode != 0
    assert "data=openacc_acc is a view of the event set 'openacc'" in proc.stderr
    assert not (tmp_path / "events").exists() and not (tmp_path / "splits").exists()


def test_extract_events_script(tmp_path):
    stdout = run_extraction(tmp_path / "run1", cwd=tmp_path)
    events_dir = tmp_path / "run1" / "events" / "synthetic"
    for name in ("samples.parquet", "events.parquet", "manifest.json"):
        assert (events_dir / name).is_file()
    events = EventSet.from_parquet(events_dir)
    events.validate()
    assert len(events) == 6

    manifest = read_json(events_dir / "manifest.json")
    digest = manifest["config_hash"]
    assert len(digest) == 12 and f"config hash {digest}" in stdout
    assert manifest["summary"]["n_events"] == 6 and manifest["stats"]["events"] == 6
    assert manifest["events_per_site"] == {"platoon0": 3, "platoon1": 3}
    assert manifest["config"]["data"]["n_platoons"] == 2 and "git_revision" in manifest

    splits = {kind: load_split(tmp_path / "run1" / "splits" / f"synthetic_{kind}.json") for kind in ("driver", "site")}
    assert set(splits["driver"]["folds"]) == {ev.event_id for ev in events}
    assert splits["driver"]["n_folds"] == 5 and splits["driver"]["events_config_hash"] == digest
    assert splits["site"]["n_folds"] == 2 and splits["site"]["n_folds_requested"] == 5

    # same config from another working directory into other output paths
    run_extraction(tmp_path / "run2", cwd=REPO_ROOT)
    events_dir2 = tmp_path / "run2" / "events" / "synthetic"
    assert read_json(events_dir2 / "manifest.json")["config_hash"] == digest
    pd.testing.assert_frame_equal(
        pd.read_parquet(events_dir / "samples.parquet"), pd.read_parquet(events_dir2 / "samples.parquet")
    )
    for kind in ("driver", "site"):
        assert load_split(tmp_path / "run2" / "splits" / f"synthetic_{kind}.json") == splits[kind]
