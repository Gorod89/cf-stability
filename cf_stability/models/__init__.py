"""Model zoo (docs/m2_contract.md, section 2): registry, factory and checkpoints."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

import torch

from cf_stability.models.base import CFModel, InputScaler, ModelContext
from cf_stability.models.idm import IDM
from cf_stability.models.knn import KNN
from cf_stability.models.mlp import MLP
from cf_stability.models.newell import Newell
from cf_stability.models.ovm import OVM
from cf_stability.models.perl import PERL
from cf_stability.models.persistence import Persistence
from cf_stability.models.pidl import PIDL
from cf_stability.models.recurrent import GRU, LSTM, Recurrent
from cf_stability.models.residual_idm import ResidualIDM

MODELS: dict[str, type[CFModel]] = {
    "idm": IDM,
    "newell": Newell,
    "ovm": OVM,
    "persistence": Persistence,
    "knn": KNN,
    "mlp": MLP,
    "gru": GRU,
    "lstm": LSTM,
    "pidl": PIDL,
    "perl": PERL,
    "residual_idm": ResidualIDM,
}


def build_model(model_cfg: Mapping[str, Any], context: ModelContext | None = None) -> CFModel:
    """Model ``model_cfg["name"]`` with the other keys as constructor arguments; unknown keys raise.

    ``train_overrides`` of a model config belongs to the training script and is ignored here.
    """
    cfg = dict(model_cfg)
    cfg.pop("train_overrides", None)
    name = cfg.pop("name", None)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    cls = MODELS[name]
    parameters = inspect.signature(cls).parameters
    accepted = set(parameters) - {"context"}
    unknown = set(cfg) - accepted
    if unknown:
        raise ValueError(f"unknown keys for model {name!r}: {sorted(unknown)} (accepted: {sorted(accepted)})")
    if "context" in parameters:
        cfg["context"] = context or ModelContext()
    return cls(**cfg)


def save_model(model: CFModel, path: str | Path) -> None:
    """Checkpoint ``{"name", "config", "state_dict"}``."""
    with open(path, "wb") as f:  # a file object avoids the C++ path handling (non-ASCII paths on Windows)
        torch.save({"name": model.name, "config": model.config(), "state_dict": model.state_dict()}, f)


def load_model(path: str | Path, map_location: str | torch.device | None = None) -> CFModel:
    """Model of a checkpoint written by :func:`save_model`, in eval mode, on ``map_location`` (CPU when ``None``).

    Eval mode matters for ``residual_idm``: in training mode every call runs a power iteration.
    """
    with open(path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=True)
    model = build_model({"name": checkpoint.get("name"), **checkpoint["config"]})
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model if map_location is None else model.to(map_location)


__all__ = [
    "MODELS",
    "CFModel",
    "GRU",
    "IDM",
    "InputScaler",
    "KNN",
    "LSTM",
    "MLP",
    "ModelContext",
    "Newell",
    "OVM",
    "PERL",
    "PIDL",
    "Persistence",
    "Recurrent",
    "ResidualIDM",
    "build_model",
    "load_model",
    "save_model",
]
