"""Small shared helpers: repository paths, config hashing, JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Keys added to a config after runs had been made with it, with their defaults (docs/m7_contract.md, 3.4).
# While such a key holds its default, `config_hash` leaves it out: the hashes of the earlier runs stay
# valid and the restartable queues do not train them again. Any other value is hashed as usual.
LATE_KEYS: dict[tuple[str, ...], Any] = {
    ("train", "penalty", "jacobian_weight"): 0.0,  # D110
    ("train", "penalty", "guard"): 0.0,  # D110
    ("per_event_margin",): None,  # D118, configs/calibrate_idm.yaml
}


def resolve_path(path: str | Path) -> Path:
    """Interpret a relative path as relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def to_plain(cfg: Any) -> Any:
    """Convert an OmegaConf object (or anything JSON-like) to plain containers."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return cfg


def json_default(value: Any) -> Any:
    """``default`` hook for json.dumps: numpy scalars and arrays keep their type, the rest becomes str."""
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _canonical(value: Any) -> Any:
    """Numbers as floats (60 and 60.0 are the same setting), containers recursively."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return _canonical(json_default(value))


def _drop_default(tree: Any, path: tuple[str, ...], default: Any) -> None:
    """Remove the key at ``path`` from the canonical ``tree`` (in place) when it holds ``default``
    (same canonical type and value: ``0`` and ``0.0`` are the default ``0.0``, ``false`` is not)."""
    for key in path[:-1]:
        if not isinstance(tree, dict) or key not in tree:
            return
        tree = tree[key]
    if isinstance(tree, dict) and path[-1] in tree:
        value = tree[path[-1]]
        if type(value) is type(default) and value == default:
            del tree[path[-1]]


def config_hash(cfg: Any, length: int = 12) -> str:
    """Stable hash of a config: sha256 of its canonical JSON form, without the keys of ``LATE_KEYS``
    that hold their defaults."""
    canonical = _canonical(to_plain(cfg))  # new containers: the keys can be dropped in place
    for path, default in LATE_KEYS.items():
        _drop_default(canonical, path, _canonical(default))
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def git_revision(root: str | Path | None = None) -> str | None:
    """Commit hash of the repository, None when there is no commit or no git.

    Read from the files of ``.git``, without a child process: a ``git`` that hangs at start-up
    cannot be waited out on Windows and blocks the run for good (docs/decisions.md, D77).
    """
    git_dir = Path(REPO_ROOT if root is None else root) / ".git"
    try:
        if git_dir.is_file():  # worktree: the file holds "gitdir: <path>"
            git_dir = (git_dir.parent / git_dir.read_text(encoding="utf-8").split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        roots, common = [git_dir], git_dir / "commondir"
        if common.is_file():  # worktree: the branches live in the main repository
            roots.append((git_dir / common.read_text(encoding="utf-8").strip()).resolve())
        for base in roots:
            if (base / ref).is_file():
                return (base / ref).read_text(encoding="utf-8").strip() or None
        for base in roots:
            if (base / "packed-refs").is_file():
                for line in (base / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref) and line[:1] not in "#^":
                        return line.split(" ", 1)[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    return None


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
