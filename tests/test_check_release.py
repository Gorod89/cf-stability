"""scripts/check_release.py: the checkout of the repository passes; a checkout whose corridor runs have neither
fields.npz nor trajectories.npz fails with exit status 1 (the figure builders would stop on it); the status of one
fields.npz (format, run, grid settings, grids) on hand-made files."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check_release():
    path = ROOT / "scripts" / "check_release.py"
    spec = importlib.util.spec_from_file_location("check_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, argv: list[str] | None = None) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = module.main(argv or [])
    return code, out.getvalue()


def write_fields(path: Path, **over) -> None:
    header = {"format": np.int64(1), "settings_hash": np.str_("abc"), "scenario": np.str_("sc"), "law": np.str_("law"),
              "seed": np.int64(0), "grids": np.asarray(["contours", "fd"])}  # fmt: skip
    header.update(over)
    np.savez(path, **header)


def test_the_checkout_passes(check_release):
    code, out = run(check_release)
    assert code == 0, out
    assert " 0 FAIL" in out and "OK    fields.npz" in out
    assert "that holds the grids of the figures" in out


def test_corridor_runs_without_fields_or_trajectories_fail(check_release, monkeypatch):
    """The negative check of the audit: the files are not touched, the checker looks for other names."""
    monkeypatch.setattr(check_release, "FIELDS", "fields_absent.npz")
    monkeypatch.setattr(check_release, "TRAJECTORIES", "trajectories_absent.npz")
    code, out = run(check_release)
    assert code == 1
    assert "FAIL  fields_absent.npz" in out and "neither fields_absent.npz nor trajectories_absent.npz" in out
    assert "the checkout is incomplete for level 2" in out


def test_fields_status(check_release, tmp_path):
    path = tmp_path / "fields.npz"
    status = check_release.fields_status
    parts = ("sc", "law", 0)
    write_fields(path)
    assert status(path, 1, "abc", parts, True) is None
    assert status(path, 1, "abc", parts, False) is None
    write_fields(path, settings_hash=np.str_("other"))
    assert "stale" in status(path, 1, "abc", parts, True)
    write_fields(path, seed=np.int64(3))
    assert status(path, 1, "abc", parts, True) == "belongs to the run sc/law/seed3"
    write_fields(path, format=np.int64(2))
    assert status(path, 1, "abc", parts, True).startswith("format 2")
    write_fields(path, grids=np.asarray(["fd"]))
    assert status(path, 1, "abc", parts, True) == "holds no contours grid"  # the figure seed needs the speed field
    assert status(path, 1, "abc", parts, False) is None  # the other seeds need the diagram cells only
    write_fields(path, grids=np.asarray(["contours"]))
    assert status(path, 1, "abc", parts, False) == "holds no fd grid"
    path.write_bytes(b"not a zip")
    assert status(path, 1, "abc", parts, True).startswith("unreadable")
