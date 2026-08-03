from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "scripts.preselection"
_impl_module: ModuleType | None = None


def _load_impl_module() -> ModuleType:
    global _impl_module
    if _impl_module is None:
        try:
            _impl_module = import_module(_IMPL_MODULE)
        except ModuleNotFoundError as error:
            missing_name = str(getattr(error, "name", ""))
            if missing_name == _IMPL_MODULE or _IMPL_MODULE.startswith(f"{missing_name}."):
                raise
            if missing_name:
                raise ModuleNotFoundError(
                    f"Missing dependency '{missing_name}' required by mil2het preselection utilities.\n"
                    "Install the project environment dependencies before using preselection."
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
    "RWRConfig",
    "read_gene_list",
    "write_gene_list",
    "deduplicate_preserve_order",
    "load_ppi_node_set",
    "remap_var_names_to_best_symbol_source",
    "load_split_train_indices",
    "collect_deg_rows",
    "select_deg",
    "select_deg_union_by_celltype",
    "compute_global_deg_zscore",
    "run_rwr",
    "np_scores",
    "infer_dataset_name",
    "resolve_adata_path",
    "resolve_preselection_output_root",
    "discover_split_indices",
    "preselection",
    "run_split_preselection",
    "build_preselection_config_from_cli_args",
    "build_preselection_arg_parser",
    "load_preselection_config_from_cli",
]
