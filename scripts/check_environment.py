"""Check of the environment before the commands of README.md (section Setup).

    python scripts/check_environment.py                      # every check
    python scripts/check_environment.py --no-sumo --no-gpu   # without SUMO and the GPU (levels 1-2, CI)

Prints one line per check (OK, WARN, FAIL or SKIP) and a summary; exit status 1 when a check fails.

* Python 3.11 or 3.12 (pyproject.toml; the results were computed with 3.11.8).
* The packages torch, numpy, pandas, pyarrow, hydra-core, scipy and matplotlib: imported, with their versions
  against the pins of requirements.txt (another version is a warning); torch with CUDA and the name of the GPU.
* The paths of the repository, of the Python environment and of SUMO: ASCII, or on Windows an ASCII 8.3 short
  form, which cf_stability/corridor/sumo_env.py hands to SUMO (SUMO cannot open other paths). Without SUMO
  (--no-sumo) a path without an ASCII form is a warning only.
* SUMO: SUMO_HOME (the eclipse-sumo package, which the corridor code uses, and the environment variable when it
  is set), the programs sumo and netconvert under it (each started with --version), and `import libsumo` in a
  child process: libsumo and pyarrow cannot share a process (docs/m5_contract.md, section 0), and this process
  imports pyarrow.
* The free space of the drive of the repository (FAIL below --min-free-gb; a warning below the about 20 GB of a
  full reproduction).

Nothing is installed, trained or simulated.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [("torch", "torch"), ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
            ("hydra-core", "hydra"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]  # (distribution, module)
PYTHON = ((3, 11), (3, 13))  # supported: from the first up to the second, excluded (pyproject.toml)
FULL_GB = 20.0  # free space a full reproduction needs (README.md: data, events, runs with trajectories)
LIBSUMO_CHILD = (  # run in a child process: the parent has imported pyarrow
    "import os, sys\n"
    "os.environ['SUMO_HOME'] = sys.argv[1]\n"
    "import libsumo\n"
    "import importlib.metadata\n"
    "print(importlib.metadata.version('libsumo'))\n"
)


class Report:
    """The lines of the checks and their counts."""

    def __init__(self) -> None:
        self.counts = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}

    def line(self, status: str, name: str, detail: str) -> None:
        self.counts[status] += 1
        print(f"{status:<4}  {name:<22} {detail}", flush=True)


def pins() -> dict[str, str]:
    """Pinned versions of requirements.txt, by lower-case distribution name."""
    out = {}
    path = ROOT / "requirements.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#;]+)", line)
            if m:
                out[m.group(1).lower()] = m.group(2)
    return out


def public(version: str) -> str:
    """The version without its local label: '2.11.0+cu128' -> '2.11.0'."""
    return version.split("+", 1)[0]


def short_form(path: Path) -> str | None:
    """An ASCII form of path: itself, or on Windows its 8.3 short form (as cf_stability/corridor/sumo_env.py
    computes it); None when there is none."""
    text = str(path)
    if text.isascii():
        return text
    if os.name != "nt" or not path.exists():
        return None
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer))
    return buffer.value if n and buffer.value.isascii() else None


def check_path(rep: Report, name: str, path: Path, required: bool) -> str | None:
    form = short_form(path)
    if form == str(path):
        rep.line("OK", name, f"{path} (ASCII)")
    elif form is not None:
        rep.line("OK", name, f"{path} (non-ASCII; 8.3 form {form})")
    else:
        why = ("no ASCII 8.3 short form (8.3 names disabled on the volume, see `fsutil 8dot3name query`)"
               if os.name == "nt" else "not ASCII")
        rep.line("FAIL" if required else "WARN", name,
                 f"{path}: {why}; SUMO cannot open it: use a directory with an ASCII path")
    return form


def check_python(rep: Report) -> None:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if PYTHON[0] <= (v.major, v.minor) < PYTHON[1]:
        rep.line("OK", "python", text + ("" if (v.major, v.minor) == (3, 11) else "; the results: 3.11.8"))
    else:
        rep.line("FAIL", "python", f"{text}: needs 3.11 or 3.12 (pyproject.toml)")


def check_packages(rep: Report, gpu: bool) -> None:
    pinned = pins()
    for dist, module in PACKAGES:
        try:
            mod = importlib.import_module(module)
            version = importlib.metadata.version(dist)
        except Exception as exc:  # a package that does not import is a failure, whatever the reason
            rep.line("FAIL", dist, f"does not import: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
            continue
        pin = pinned.get(dist.lower())
        if pin is None or public(pin) == public(version):
            rep.line("OK", dist, version + (f" (pinned {pin})" if pin and pin != version else ""))
        else:
            rep.line("WARN", dist, f"{version}, requirements.txt pins {pin} (the results were computed with it)")
        if module == "torch":
            check_cuda(rep, mod, gpu)


def check_cuda(rep: Report, torch, gpu: bool) -> None:
    if not gpu:
        rep.line("SKIP", "torch CUDA", "--no-gpu (training, level 3, needs a CUDA GPU)")
        return
    if torch.cuda.is_available():
        names = ", ".join(torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count()))
        rep.line("OK", "torch CUDA", f"CUDA {torch.version.cuda}: {names}")
    else:
        rep.line("FAIL", "torch CUDA", f"no CUDA device (torch {torch.__version__}, CUDA build "
                                       f"{torch.version.cuda}); training needs one, levels 1-2 run with --no-gpu")


def sumo_home() -> tuple[Path | None, str]:
    """SUMO_HOME as the corridor code takes it (the eclipse-sumo package) and a note on the environment variable."""
    env = os.environ.get("SUMO_HOME")
    try:
        import sumo  # the eclipse-sumo package; it does not load libsumo

        home = Path(getattr(sumo, "SUMO_HOME", Path(sumo.__file__).parent))
    except ImportError:
        return (Path(env) if env else None), ("environment variable SUMO_HOME; package eclipse-sumo missing"
                                              if env else "")
    note = "package eclipse-sumo"
    if env:
        note += ("; the environment variable SUMO_HOME points to the same" if Path(env).resolve() == home.resolve()
                 else f"; the environment variable SUMO_HOME ({env}) is not used by the corridor code")
    return home, note


def check_sumo(rep: Report) -> Path | None:
    home, note = sumo_home()
    if home is None or not home.exists():
        rep.line("FAIL", "SUMO_HOME", "no SUMO: install eclipse-sumo==1.27.1 and libsumo==1.27.1 (requirements.txt)")
        return None
    rep.line("OK", "SUMO_HOME", f"{home} ({note})")
    form = check_path(rep, "SUMO path", home, required=True)
    if form is None:
        return None
    env = {**os.environ, "SUMO_HOME": form}
    for name in ("sumo", "netconvert"):
        exe = home / "bin" / (name + (".exe" if os.name == "nt" else ""))
        if not exe.exists():
            found = shutil.which(name)
            if not found:
                rep.line("FAIL", name, f"{exe} missing and not on PATH")
                continue
            exe = Path(found)
        try:
            proc = subprocess.run([short_form(exe) or str(exe), "--version"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=120, env=env, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rep.line("FAIL", name, f"{exe} does not start: {type(exc).__name__}: {exc}")
            continue
        first = next((x.strip() for x in (proc.stdout + proc.stderr).splitlines() if x.strip()), "")
        rep.line("OK" if proc.returncode == 0 else "FAIL", name, f"{first or 'no output'} (exit code {proc.returncode})")
    return home


def check_libsumo(rep: Report, home: Path) -> None:
    form = short_form(home) or str(home)
    try:
        proc = subprocess.run([sys.executable, "-c", LIBSUMO_CHILD, form], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300, cwd=str(ROOT), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        rep.line("FAIL", "import libsumo", "the child process did not finish within 300 s")
        return
    out = (proc.stdout + proc.stderr).strip().splitlines()
    if proc.returncode == 0:
        rep.line("OK", "import libsumo", f"libsumo {out[-1] if out else '?'} in a child process (not with pyarrow)")
    else:
        rep.line("FAIL", "import libsumo", f"exit code {proc.returncode}: {out[-1] if out else 'no output'}")


def check_disk(rep: Report, min_free_gb: float) -> None:
    free = shutil.disk_usage(ROOT).free / 1e9
    text = f"{free:.1f} GB free on the drive of the repository"
    if free < min_free_gb:
        rep.line("FAIL", "disk", f"{text}; at least {min_free_gb:g} GB needed (--min-free-gb)")
    elif free < FULL_GB:
        rep.line("WARN", "disk", f"{text}; a full reproduction (level 3) needs about {FULL_GB:g} GB")
    else:
        rep.line("OK", "disk", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-sumo", action="store_true", help="skip SUMO, its paths and libsumo (levels 1-2, CI)")
    parser.add_argument("--no-gpu", action="store_true", help="skip the CUDA check (levels 1-2, CI)")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="free disk space below which the check fails")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # paths with characters the console cannot print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    rep = Report()
    check_python(rep)
    check_packages(rep, gpu=not args.no_gpu)
    check_path(rep, "repository path", ROOT, required=not args.no_sumo)
    check_path(rep, "environment path", Path(sys.prefix), required=not args.no_sumo)
    if args.no_sumo:
        for name in ("SUMO_HOME", "sumo", "netconvert", "import libsumo"):
            rep.line("SKIP", name, "--no-sumo (the corridor runs of level 3 need SUMO)")
    else:
        home = check_sumo(rep)
        if home is not None:
            check_libsumo(rep, home)
        else:
            rep.line("SKIP", "import libsumo", "no usable SUMO_HOME")
    check_disk(rep, args.min_free_gb)
    c = rep.counts
    print(f"{sum(c.values())} checks: {c['OK']} OK, {c['WARN']} WARN, {c['FAIL']} FAIL, {c['SKIP']} SKIP"
          + (" -- environment not ready" if c["FAIL"] else ""))
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
