"""Public package API for scbiomarker."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import CellEncoder


def __getattr__(name: str) -> Any:
    if name in {"GraphCellEncoder", "TransformerConvCellEncoder"}:
        module = import_module("scbiomarker.CellEncoder")
        return getattr(module, name)
    raise AttributeError(f"module 'scbiomarker' has no attribute '{name}'")


__all__ = ["CellEncoder", "GraphCellEncoder", "TransformerConvCellEncoder"]
