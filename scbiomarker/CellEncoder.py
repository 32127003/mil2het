"""Public CellEncoder module wrapper with lazy import."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "modules.CellEncoder"
_impl_module: ModuleType | None = None


def _load_impl_module() -> ModuleType:
    global _impl_module
    if _impl_module is None:
        _impl_module = import_module(_IMPL_MODULE)
    return _impl_module


def __getattr__(name: str) -> Any:
    return getattr(_load_impl_module(), name)


def __dir__() -> list[str]:
    try:
        implementation_dir = set(dir(_load_impl_module()))
    except Exception:
        implementation_dir = set()
    return sorted(set(globals().keys()) | implementation_dir)


__all__ = [
    "GraphAttentionLayer",
    "TransformerConvLayer",
    "GraphCellEncoder",
    "TransformerConvCellEncoder",
    "scatter_softmax",
    "infer_batch_chunk_size",
]
