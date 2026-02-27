"""Public biomarker module wrapper with lazy import."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "modules.biomarker"
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
                    "Missing dependency 'torch' required by scbiomarker biomarker module.\n"
                    "Install PyTorch for your system:\n"
                    "https://pytorch.org/get-started/locally/\n"
                ) from error
            if missing_name == "torch_scatter":
                raise ModuleNotFoundError(
                    "Missing optional dependency 'torch_scatter' required by scbiomarker biomarker module.\n"
                    "Install dependencies with matching PyTorch/CUDA versions:\n"
                    "1) Install CUDA-specific PyTorch for your system:\n"
                    "   https://pytorch.org/get-started/locally/\n"
                    "2) Install torch-scatter for the same torch/cuda pair:\n"
                    "   pip install torch-scatter -f "
                    "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html\n"
                ) from error
            if missing_name == "scanpy":
                raise ModuleNotFoundError(
                    "Missing dependency 'scanpy' required by scbiomarker biomarker module.\n"
                    "Install with:\n"
                    "pip install scanpy==1.9.6\n"
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
    "BagDefinition",
    "BagCache",
    "PatientClassifier",
    "load_config",
    "load_config_like_train_from_run_snapshot",
    "resolve_device",
    "resolve_cell_encoder_spec",
    "load_prior_embeddings_by_view",
    "load_np_max_genes",
    "build_protein_embedding_matrix",
    "build_celltype_regulation_onehot",
    "build_global_deg_zscore_vector",
    "sample_patient_bag_indices",
    "compute_binary_metrics",
    "aggregate_patient_probabilities",
    "stable_spearman",
    "jaccard_index",
    "forward_bag_from_expression",
    "forward_bag_from_embeddings",
    "sample_rows_with_replacement",
    "parse_args",
    "main",
]
