"""SUMO in this environment (docs/m5_contract.md, section 0).

``libsumo`` 1.27.1 brings its own Arrow library: it cannot share a process with the ``pyarrow`` of
the environment, whichever is imported second fails to load. SUMO cannot open paths with non-ASCII
characters either, and the ``sumo`` package points ``SUMO_HOME`` to such a path (the repository lies
under ``C:\\Users\\Город``). Hence:

* a simulation process calls :func:`import_libsumo` before anything imports pandas or torch, and
  hands SUMO only paths from :func:`short_path`;
* the build process (pandas, pyarrow) runs ``netconvert`` as a child process (:func:`run_netconvert`)
  and never imports ``libsumo``.

This module imports nothing heavy itself, so both kinds of processes can use it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def short_path(path: str | Path) -> str:
    """8.3 form of ``path`` (Windows), ASCII only, for programs that cannot open non-ASCII paths.

    A path that does not exist yet keeps its last component: the short form of the existing parent
    plus the name (the names of this package are ASCII). Raises when no ASCII form exists, e.g. on a
    volume without short names.
    """
    path = Path(path).absolute()
    if os.name != "nt" or str(path).isascii():
        return str(path)
    import ctypes

    missing: list[str] = []
    base = path
    while not base.exists() and base.parent != base:
        missing.insert(0, base.name)
        base = base.parent
    buffer = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(base), buffer, len(buffer))
    out = str(Path(buffer.value if n else str(base), *missing))
    if not out.isascii():
        raise OSError(f"no ASCII (8.3) form of {path}: SUMO cannot open it; use a directory with an ASCII path")
    return out


def sumo_home() -> Path:
    """Installation directory of the ``eclipse-sumo`` package (``bin/``, ``data/``, ``tools/``)."""
    import sumo  # the eclipse-sumo package; it does not load libsumo

    return Path(getattr(sumo, "SUMO_HOME", Path(sumo.__file__).parent))


def binary(name: str) -> str:
    """ASCII path of the SUMO program ``name`` (``netconvert``, ``sumo``)."""
    suffix = ".exe" if os.name == "nt" else ""
    path = sumo_home() / "bin" / f"{name}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (package eclipse-sumo)")
    return short_path(path)


def import_libsumo() -> Any:
    """``libsumo`` with ``SUMO_HOME`` set to the ASCII form of the installation.

    Must run before pandas (which imports pyarrow when it can) and torch are imported; raises when
    pyarrow is loaded already, because libsumo would then fail with an obscure DLL error.
    """
    if "libsumo" in sys.modules:
        return sys.modules["libsumo"]
    if "pyarrow" in sys.modules:
        raise RuntimeError(
            "pyarrow is loaded in this process: libsumo cannot be imported after it (docs/m5_contract.md, "
            "section 0); import cf_stability.corridor.loop before pandas"
        )
    os.environ["SUMO_HOME"] = short_path(sumo_home())
    import libsumo

    return libsumo


def run_netconvert(
    nodes: Path, edges: Path, connections: Path, output: Path, options: Sequence[str] = (), timeout: float = 300.0
) -> subprocess.CompletedProcess:
    """``netconvert`` on plain XML files as a child process, with short paths; raises on failure."""
    command = [
        binary("netconvert"), "--node-files", short_path(nodes), "--edge-files", short_path(edges),
        "--connection-files", short_path(connections), "--output-file", short_path(output), *options,
    ]  # fmt: skip
    env = {**os.environ, "SUMO_HOME": short_path(sumo_home())}
    proc = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
        stdin=subprocess.DEVNULL,
    )  # fmt: skip
    if proc.returncode != 0 or not Path(output).exists():
        raise RuntimeError(f"netconvert failed (exit code {proc.returncode}): {(proc.stdout + proc.stderr).strip()}")
    return proc
