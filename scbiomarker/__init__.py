from __future__ import annotations

from importlib import import_module
from typing import Any

from . import (
    CellEncoder,
    MultipleInstanceLearning,
    biomarker,
    config,
    pipeline,
    preselection,
    prior_interface_find,
    split_dataset,
    train,
)


def __getattr__(name: str) -> Any:
    lazy_exports = {
        "GraphCellEncoder": ("scbiomarker.CellEncoder", "GraphCellEncoder"),
        "TransformerConvCellEncoder": ("scbiomarker.CellEncoder", "TransformerConvCellEncoder"),
        "GatedAttentionMIL": ("scbiomarker.MultipleInstanceLearning", "GatedAttentionMIL"),
        "PatientMILAggregator": ("scbiomarker.MultipleInstanceLearning", "PatientMILAggregator"),
        "MultiViewPriorInterfaceFIND": ("scbiomarker.prior_interface_find", "MultiViewPriorInterfaceFIND"),
        "PipelineResult": ("scbiomarker.pipeline", "PipelineResult"),
        "run_pipeline": ("scbiomarker.pipeline", "run_pipeline"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'scbiomarker' has no attribute '{name}'")


__all__ = [
    "CellEncoder",
    "MultipleInstanceLearning",
    "config",
    "prior_interface_find",
    "biomarker",
    "pipeline",
    "split_dataset",
    "preselection",
    "train",
    "GraphCellEncoder",
    "TransformerConvCellEncoder",
    "GatedAttentionMIL",
    "PatientMILAggregator",
    "MultiViewPriorInterfaceFIND",
    "PipelineResult",
    "run_pipeline",
]
