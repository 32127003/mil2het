from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "modules.MultipleInstanceLearning"
_impl_module: ModuleType | None = None


def _load_impl_module() -> ModuleType:
    global _impl_module
    if _impl_module is None:
        try:
            _impl_module = import_module(_IMPL_MODULE)
        except ModuleNotFoundError as error:
            missing_name = str(getattr(error, "name", ""))
            if missing_name == "torch":
                raise ModuleNotFoundError(
                    "Missing dependency 'torch' required by mil2het MultipleInstanceLearning.\n"
                    "Install PyTorch for your system:\n"
                    "https://pytorch.org/get-started/locally/\n"
                ) from error
            if missing_name == "torch_scatter":
                raise ModuleNotFoundError(
                    "Missing optional dependency 'torch_scatter' required by mil2het MultipleInstanceLearning.\n"
                    "Install dependencies with matching PyTorch/CUDA versions:\n"
                    "1) Install CUDA-specific PyTorch for your system:\n"
                    "   https://pytorch.org/get-started/locally/\n"
                    "2) Install torch-scatter for the same torch/cuda pair:\n"
                    "   pip install --only-binary=torch-scatter torch-scatter -f "
                    "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html\n"
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
    "scatter_softmax_1d",
    "GatedAttentionMIL",
    "PatientMILAggregator",
]
