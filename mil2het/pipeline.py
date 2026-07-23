from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from anndata import AnnData

from . import biomarker, config as workflow_config, preselection, split_dataset, train


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    for key, value in override.items():
        key_text = str(key)
        if (
            key_text in base
            and isinstance(base[key_text], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key_text], value)
            continue
        base[key_text] = copy.deepcopy(value)
    return base


def _normalize_embedding_views(embedding_views: Mapping[str, os.PathLike[str] | str] | None) -> dict[str, str]:
    if embedding_views is None:
        return {}

    normalized: dict[str, str] = {}
    for view_name, path_value in embedding_views.items():
        name_text = str(view_name).strip()
        if name_text == "":
            raise ValueError("embedding view names must be non-empty.")
        path_text = os.fspath(path_value).strip()
        if path_text == "":
            raise ValueError(f"embedding view '{name_text}' has an empty path.")
        normalized[name_text] = path_text
    return normalized


def _build_explicit_overrides(
    *,
    input_h5ad: os.PathLike[str] | str | None,
    patient_column: str | None,
    celltype_column: str | None,
    label_column: str | None,
    sample_column: str | None,
    treatment_column: str | None,
    ppi_path: os.PathLike[str] | str | None,
    embedding_views: Mapping[str, os.PathLike[str] | str] | None,
    output_root: os.PathLike[str] | str | None,
    run_dir: os.PathLike[str] | str | None,
    analysis_output_dir: os.PathLike[str] | str | None,
    num_folds: int | None,
    split_number: int | None,
    seed: int | None,
    epochs: int | None,
    lr: float | None,
    k: int | None,
    train_only: bool | None,
    analysis_only: bool | None,
    positive_labels: list[str] | None,
    negative_labels: list[str] | None,
    pathway_path: os.PathLike[str] | str | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    workflow_overrides: dict[str, Any] = {}
    column_overrides: dict[str, Any] = {}
    resource_overrides: dict[str, Any] = {}
    training_overrides: dict[str, Any] = {}
    label_overrides: dict[str, Any] = {}

    if input_h5ad is not None:
        workflow_overrides["input_h5ad"] = os.fspath(input_h5ad)
    if output_root is not None:
        workflow_overrides["output_root"] = os.fspath(output_root)
    if run_dir is not None:
        workflow_overrides["run_dir"] = os.fspath(run_dir)
    if analysis_output_dir is not None:
        workflow_overrides["analysis_output_dir"] = os.fspath(analysis_output_dir)
    if num_folds is not None:
        workflow_overrides["num_folds"] = int(num_folds)
    if split_number is not None:
        workflow_overrides["split_number"] = int(split_number)
    if seed is not None:
        workflow_overrides["seed"] = int(seed)
    if train_only is not None:
        workflow_overrides["train_only"] = bool(train_only)
    if analysis_only is not None:
        workflow_overrides["analysis_only"] = bool(analysis_only)

    if patient_column is not None:
        column_overrides["patient"] = str(patient_column)
    if celltype_column is not None:
        column_overrides["celltype"] = str(celltype_column)
    if label_column is not None:
        column_overrides["label"] = str(label_column)
    if sample_column is not None:
        column_overrides["sample"] = str(sample_column)
    if treatment_column is not None:
        column_overrides["treatment"] = str(treatment_column)

    if ppi_path is not None:
        resource_overrides["ppi_path"] = os.fspath(ppi_path)
    if embedding_views is not None:
        resource_overrides["embedding_views"] = _normalize_embedding_views(embedding_views)

    if epochs is not None:
        training_overrides["epochs"] = int(epochs)
    if lr is not None:
        training_overrides["lr"] = float(lr)
    if k is not None:
        training_overrides["k"] = int(k)

    if positive_labels is not None:
        label_overrides["positive"] = [str(value) for value in positive_labels]
    if negative_labels is not None:
        label_overrides["negative"] = [str(value) for value in negative_labels]

    if workflow_overrides:
        overrides["workflow"] = workflow_overrides
    if column_overrides:
        overrides["columns"] = column_overrides
    if resource_overrides:
        overrides["resources"] = resource_overrides
    if training_overrides:
        overrides["training"] = training_overrides
    if label_overrides:
        overrides["labels"] = label_overrides
    if pathway_path is not None:
        overrides["biomarker_pathway_gene_set_path"] = os.fspath(pathway_path)
        overrides["pathway_gene_set_path"] = os.fspath(pathway_path)
    return overrides


def _materialize_adata(adata: AnnData, output_root: str) -> str:
    target_dir = Path(output_root).resolve() / "inputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".h5ad",
        prefix="mil2het_input_",
        dir=str(target_dir),
        delete=False,
    ) as handle:
        materialized_path = handle.name
    adata.write_h5ad(materialized_path)
    return materialized_path


def _write_config_snapshot(config_dict: Mapping[str, Any], target_path: str) -> str:
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dict = workflow_config.serialize_workflow_config_snapshot(config_dict)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot_dict, handle, sort_keys=False)
    return str(target)


@dataclass(slots=True)
class PipelineResult:
    config: dict[str, Any]
    phases_completed: tuple[str, ...]
    input_h5ad: str
    output_root: str
    splits_directory: str
    preselection_output_root: str
    run_dir: str
    analysis_output_dir: str
    config_snapshot_path: str
    training_artifacts: dict[str, Any] | None
    analysis_artifacts: dict[str, Any] | None
    materialized_input_h5ad: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": copy.deepcopy(self.config),
            "phases_completed": list(self.phases_completed),
            "input_h5ad": self.input_h5ad,
            "output_root": self.output_root,
            "splits_directory": self.splits_directory,
            "preselection_output_root": self.preselection_output_root,
            "run_dir": self.run_dir,
            "analysis_output_dir": self.analysis_output_dir,
            "config_snapshot_path": self.config_snapshot_path,
            "training_artifacts": copy.deepcopy(self.training_artifacts),
            "analysis_artifacts": copy.deepcopy(self.analysis_artifacts),
            "materialized_input_h5ad": self.materialized_input_h5ad,
        }


def run_pipeline(
    config_path: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    *,
    input_h5ad: os.PathLike[str] | str | None = None,
    adata: AnnData | None = None,
    patient_column: str | None = None,
    celltype_column: str | None = None,
    label_column: str | None = None,
    sample_column: str | None = None,
    treatment_column: str | None = None,
    ppi_path: os.PathLike[str] | str | None = None,
    embedding_views: Mapping[str, os.PathLike[str] | str] | None = None,
    output_root: os.PathLike[str] | str | None = None,
    run_dir: os.PathLike[str] | str | None = None,
    analysis_output_dir: os.PathLike[str] | str | None = None,
    pathway_path: os.PathLike[str] | str | None = None,
    positive_labels: list[str] | None = None,
    negative_labels: list[str] | None = None,
    num_folds: int | None = None,
    split_number: int | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    k: int | None = None,
    train_only: bool | None = None,
    analysis_only: bool | None = None,
    device: Any = None,
    gpu_index: int | None = None,
) -> PipelineResult:
    if adata is not None and input_h5ad is not None:
        raise ValueError("Provide either adata or input_h5ad, not both.")

    merged_overrides = copy.deepcopy(dict(config_overrides or {}))
    explicit_overrides = _build_explicit_overrides(
        input_h5ad=input_h5ad,
        patient_column=patient_column,
        celltype_column=celltype_column,
        label_column=label_column,
        sample_column=sample_column,
        treatment_column=treatment_column,
        ppi_path=ppi_path,
        embedding_views=embedding_views,
        output_root=output_root,
        run_dir=run_dir,
        analysis_output_dir=analysis_output_dir,
        num_folds=num_folds,
        split_number=split_number,
        seed=seed,
        epochs=epochs,
        lr=lr,
        k=k,
        train_only=train_only,
        analysis_only=analysis_only,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        pathway_path=pathway_path,
    )
    _deep_merge(merged_overrides, explicit_overrides)

    materialized_input_h5ad: str | None = None
    if adata is not None:
        provisional_config = workflow_config.load_workflow_config_dict(
            config_path=config_path,
            overrides=merged_overrides,
        )
        if bool(provisional_config.get("analysis_only", False)):
            raise ValueError("analysis_only requires run_dir and does not accept adata input.")
        materialized_input_h5ad = _materialize_adata(adata, str(provisional_config["output_root"]))
        _deep_merge(
            merged_overrides,
            {
                "workflow": {
                    "input_h5ad": materialized_input_h5ad,
                }
            },
        )

    config_dict = workflow_config.load_workflow_config_dict(
        config_path=config_path,
        overrides=merged_overrides,
    )
    if bool(config_dict["analysis_only"]) and input_h5ad is not None:
        raise ValueError("analysis_only requires run_dir and does not accept input_h5ad.")

    config_namespace = workflow_config.dict_to_namespace(config_dict)

    phases_completed: list[str] = []
    split_output_dir = str(config_dict["splits_directory"])
    preselection_output_root = str(config_dict["preselection_output_root"])
    effective_run_dir = str(config_dict.get("run_dir", "") or "")
    effective_analysis_output_dir = str(config_dict["biomarker_output_dir"])
    config_snapshot_path = ""
    training_artifacts: dict[str, Any] | None = None
    analysis_artifacts: dict[str, Any] | None = None

    training_device = device
    if not bool(config_dict["analysis_only"]) and device is None:
        resolved_gpu_index = 0 if gpu_index is None else int(gpu_index)
        setattr(config_namespace, "gpu", resolved_gpu_index)
        config_dict["gpu"] = resolved_gpu_index
        if resolved_gpu_index < 0:
            training_device = "cpu"

    if not bool(config_dict["analysis_only"]):
        split_dataset.run_split_generation(config_namespace)
        phases_completed.append("split")

        preselection_output_root = str(preselection.run_split_preselection(config_namespace))
        config_dict["preselection_output_root"] = preselection_output_root
        setattr(config_namespace, "preselection_output_root", preselection_output_root)
        phases_completed.append("preselection")

        training_artifacts = dict(train.run_training_phase(config_namespace, device=training_device))
        effective_run_dir = str(training_artifacts["output_dir"])
        config_dict["run_dir"] = effective_run_dir
        config_dict["biomarker_run_dir"] = effective_run_dir
        phases_completed.append("training")

        config_snapshot_path = _write_config_snapshot(
            config_dict,
            os.path.join(effective_run_dir, "workflow_config.yaml"),
        )
    else:
        if config_path is not None and str(config_path).strip() != "":
            config_snapshot_path = str(Path(str(config_path)).resolve())
        else:
            run_snapshot_path = Path(str(effective_run_dir)).resolve() / "workflow_config.yaml"
            if run_snapshot_path.is_file():
                config_snapshot_path = str(run_snapshot_path)
            else:
                generated_config_dir = Path(str(config_dict["output_root"])).resolve() / "generated_configs"
                config_snapshot_path = _write_config_snapshot(
                    config_dict,
                    str(generated_config_dir / "analysis_workflow_config.yaml"),
                )

    if not bool(config_dict["train_only"]):
        analysis_artifacts = dict(
            biomarker.run_analysis_phase(
                effective_run_dir,
                config_path=config_snapshot_path,
                output_dir=effective_analysis_output_dir,
                pathway_path="" if pathway_path is None else os.fspath(pathway_path),
                gpu_index=gpu_index,
            )
        )
        effective_analysis_output_dir = str(analysis_artifacts["output_dir"])
        phases_completed.append("analysis")

    return PipelineResult(
        config=copy.deepcopy(config_dict),
        phases_completed=tuple(phases_completed),
        input_h5ad=str(config_dict["input_h5ad"]),
        output_root=str(config_dict["output_root"]),
        splits_directory=split_output_dir,
        preselection_output_root=preselection_output_root,
        run_dir=effective_run_dir,
        analysis_output_dir=effective_analysis_output_dir,
        config_snapshot_path=config_snapshot_path,
        training_artifacts=copy.deepcopy(training_artifacts),
        analysis_artifacts=copy.deepcopy(analysis_artifacts),
        materialized_input_h5ad=materialized_input_h5ad,
    )


__all__ = [
    "PipelineResult",
    "run_pipeline",
]
