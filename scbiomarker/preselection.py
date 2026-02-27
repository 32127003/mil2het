from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_IMPL_MODULE = "scripts.preselection"
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
    "preselection",
]
