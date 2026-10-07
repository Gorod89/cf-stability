"""Dataset loaders.

Every loader module ``cf_stability.data.<loader>`` provides
``build_events(data_cfg, extraction, stats=None) -> EventSet``; modules are imported lazily.
"""

from __future__ import annotations

import importlib
from typing import Callable


def get_builder(loader_name: str) -> Callable:
    """``build_events`` of the module ``cf_stability.data.<loader_name>``."""
    return importlib.import_module(f"cf_stability.data.{loader_name}").build_events
