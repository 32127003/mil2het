from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "scripts.split_dataset"
_impl_module: ModuleType | None = None


def _load_impl_module() -> ModuleType:
    global _impl_module
    if _impl_module is None:
        try:
            _impl_module = import_module(_IMPL_MODULE)
        except ModuleNotFoundError as error:
            missing_name = str(getattr(error, "name", ""))
            if missing_name in {"scripts", _IMPL_MODULE}:
                raise
            raise ModuleNotFoundError(
                f"Missing dependency '{missing_name}' required by mil2het split_dataset utilities.\n"
                "Install the project dependencies for split generation before using "
                "mil2het.split_dataset.\n"
            ) from error
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
    "rebalance_empty_folds",
    "rebalance_fold_sizes",
    "fold_label_counters",
    "enforce_train_label_coverage",
    "patient_to_folds",
    "patient_folds_to_cell_folds",
    "select_valid_fold_indices",
    "patient_folds_to_sample_folds",
    "sample_folds_to_cell_folds",
    "patient_overlap_counts",
    "normalize_binary_label_values",
    "infer_dataset_name",
    "resolve_adata_path",
    "resolve_split_output_directory",
    "build_legacy_split_config",
    "build_and_save_folds",
    "run_split_generation",
    "build_split_config_from_cli_args",
    "build_split_arg_parser",
    "load_split_config_from_cli",
]
