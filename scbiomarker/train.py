from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "scripts.train"
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
                    "Missing dependency 'torch' required by scbiomarker train utilities.\n"
                    "Install PyTorch for your system:\n"
                    "https://pytorch.org/get-started/locally/\n"
                ) from error
            if missing_name == "torch_scatter":
                raise ModuleNotFoundError(
                    "Missing optional dependency 'torch_scatter' required by scbiomarker train utilities.\n"
                    "Install dependencies with matching PyTorch/CUDA versions:\n"
                    "1) Install CUDA-specific PyTorch for your system:\n"
                    "   https://pytorch.org/get-started/locally/\n"
                    "2) Install torch-scatter for the same torch/cuda pair:\n"
                    "   pip install torch-scatter -f "
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
    "autocast_cuda",
    "ensure_cublas_workspace_config",
    "configure_runtime_backends",
    "run_training_phase",
    "build_train_arg_parser",
    "build_train_config_from_cli_args",
    "load_train_config_from_cli",
    "main",
    "PatientEmbeddingEMAMemory",
    "PatientBagDataset",
    "PatientClassifier",
    "collate_patient_bags",
    "compute_binary_metrics",
    "aggregate_patient_probabilities",
    "resolve_cell_encoder_spec",
    "build_protein_embedding_matrix",
    "load_prior_embeddings_by_view",
    "build_celltype_regulation_onehot",
    "build_global_deg_zscore_vector",
    "build_graph_cache",
    "build_split_records_cache",
    "build_experiment_directory",
    "build_optimizer",
    "run_epoch",
]
