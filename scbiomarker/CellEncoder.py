"""Public CellEncoder module wrapper with lazy import.

The implementation depends on PyTorch.
`torch_scatter` is optional and used when available for faster segmented ops.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "modules.CellEncoder"
_impl_module: ModuleType | None = None


def _load_impl_module() -> ModuleType:
    global _impl_module
    if _impl_module is None:
        try:
            _impl_module = import_module(_IMPL_MODULE)
        except ModuleNotFoundError as error:
            missing_name = getattr(error, "name", "")
            if str(missing_name) == "torch":
                raise ModuleNotFoundError(
                    "Missing dependency 'torch' required by scbiomarker CellEncoder.\n"
                    "Install PyTorch for your system:\n"
                    "https://pytorch.org/get-started/locally/\n"
                ) from error
            raise
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
