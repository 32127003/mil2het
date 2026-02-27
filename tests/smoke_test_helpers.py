from __future__ import annotations

from types import ModuleType
from typing import Sequence


def assert_module_endpoints(
    module: ModuleType,
    endpoint_names: Sequence[str],
    *,
    module_label: str,
) -> None:
    missing = [name for name in endpoint_names if not hasattr(module, name)]
    assert not missing, f"{module_label}: missing endpoints: {missing}"

    exported = set(getattr(module, "__all__", []))
    not_exported = [name for name in endpoint_names if name not in exported]
    assert not not_exported, f"{module_label}: endpoints missing from __all__: {not_exported}"

    none_values = [name for name in endpoint_names if getattr(module, name) is None]
    assert not none_values, f"{module_label}: endpoints resolved to None: {none_values}"


def print_success(message: str) -> None:
    print(f"[ok] {message}")
