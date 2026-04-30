from __future__ import annotations

import argparse
import copy
import os
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from configs.config import asthma_split_configuration, asthma_train_configuration

DEFAULT_CONFIG_RESOURCE = "default_config.yaml"

_DICT_REPLACE_KEYS = {"embedding_views"}
_SNAPSHOT_SECTION_KEYS = {"workflow", "columns", "labels", "resources", "training"}
_SNAPSHOT_RUNTIME_KEYS = {
    "dataset",
    "adata_path",
    "adata_directory",
    "splits_directory",
    "preselection_output_root",
    "experiment_root",
    "biomarker_run_dir",
    "biomarker_output_dir",
}
_LEGACY_SNAPSHOT_MARKERS = {
    "adata_path",
    "adata_directory",
    "splits_directory",
    "preselection_output_root",
    "experiment_root",
    "biomarker_run_dir",
    "biomarker_output_dir",
    "gene_embedding_views",
    "protein_embedding_paths",
}

_LEGACY_DEFAULTS = copy.deepcopy(asthma_train_configuration)
_LEGACY_DEFAULTS.update(
    {
        "seed": int(asthma_split_configuration["seed"]),
        "num_folds": int(asthma_split_configuration["num_folds"]),
        "num_valid_patients": int(asthma_split_configuration["num_valid_patients"]),
        "deg_max_p_value": float(asthma_split_configuration["deg_max_p_value"]),
        "deg_min_abs_logfc": float(asthma_split_configuration["deg_min_abs_logfc"]),
        "restart_prob": float(asthma_split_configuration["restart_prob"]),
        "convergence_threshold_l1": float(asthma_split_configuration["convergence_threshold_l1"]),
        "max_iterations": int(asthma_split_configuration["max_iterations"]),
        "directed": bool(asthma_split_configuration["directed"]),
        "dataset": "",
    }
)


def dict_to_namespace(config: Mapping[str, Any]) -> SimpleNamespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, str(key), copy.deepcopy(value))
    return namespace


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    for key, value in override.items():
        key_str = str(key)
        if (
            key_str not in _DICT_REPLACE_KEYS
            and key_str in base
            and isinstance(base[key_str], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key_str], value)
            continue
        base[key_str] = copy.deepcopy(value)
    return base


def _load_yaml_dict(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must have a dictionary root: {path}")
    return payload


def load_default_config_dict() -> dict[str, Any]:
    resource = files("mil2het").joinpath(DEFAULT_CONFIG_RESOURCE)
    return _load_yaml_dict(str(resource))


def load_yaml_config_dict(config_path: str) -> dict[str, Any]:
    suffix = Path(config_path).suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML config path, received: {config_path}")
    return _load_yaml_dict(config_path)


def _nested_get(config: Mapping[str, Any], *keys: str) -> tuple[bool, Any]:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _coalesce(config: Mapping[str, Any], paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        found, value = _nested_get(config, *path)
        if found and value is not None:
            return value
    return default


def _snapshot_value(
    config: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    for path in paths:
        found, value = _nested_get(config, *path)
        if not found or value is None:
            continue
        if isinstance(value, str):
            if value.strip() == "":
                continue
            return value
        if isinstance(value, Mapping) and len(value) == 0:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0:
            continue
        return value
    return default


def _normalize_embedding_views(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("embedding_views must be a mapping of view_name -> path.")

    normalized: dict[str, str] = {}
    for key, raw_path in value.items():
        key_text = str(key).strip()
        if key_text == "":
            raise ValueError("embedding view names must be non-empty.")
        path_text = str(raw_path).strip()
        if path_text == "":
            raise ValueError(f"embedding view '{key_text}' has an empty path.")
        normalized[key_text] = path_text
    return normalized


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a list of source names, not a mapping.")
    raw_values = [value] if isinstance(value, str) else list(value)
    normalized: list[str] = []
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if text != "":
            normalized.append(text)
    return normalized


def _infer_dataset_name(config: Mapping[str, Any], input_h5ad: str) -> str:
    dataset_value = str(_coalesce(config, (("dataset",),), "")).strip()
    if dataset_value != "":
        return dataset_value
    if input_h5ad == "":
        return ""
    stem = Path(input_h5ad).stem
    if stem.endswith("_data"):
        stem = stem[:-5]
    return stem


def _build_output_path(base_dir: str, leaf: str, explicit: str) -> str:
    explicit_text = str(explicit).strip()
    if explicit_text != "":
        return explicit_text
    return os.path.join(base_dir, leaf)


def _build_snapshot_sections(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "workflow": {
            "input_h5ad": str(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "input_h5ad"),
                        ("input_h5ad",),
                        ("adata_path",),
                    ),
                    "",
                )
            ).strip(),
            "output_root": str(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "output_root"),
                        ("output_root",),
                    ),
                    "",
                )
            ).strip(),
            "run_dir": str(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "run_dir"),
                        ("run_dir",),
                        ("biomarker_run_dir",),
                    ),
                    "",
                )
            ).strip(),
            "analysis_output_dir": str(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "analysis_output_dir"),
                        ("analysis_output_dir",),
                        ("biomarker_output_dir",),
                    ),
                    "",
                )
            ).strip(),
            "num_folds": int(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "num_folds"),
                        ("num_folds",),
                    ),
                    _LEGACY_DEFAULTS["num_folds"],
                )
            ),
            "split_number": int(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "split_number"),
                        ("split_number",),
                    ),
                    0,
                )
            ),
            "seed": int(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "seed"),
                        ("seed",),
                    ),
                    _LEGACY_DEFAULTS["seed"],
                )
            ),
            "train_only": bool(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "train_only"),
                        ("train_only",),
                        ("skip_analysis",),
                    ),
                    False,
                )
            ),
            "analysis_only": bool(
                _snapshot_value(
                    config,
                    (
                        ("workflow", "analysis_only"),
                        ("analysis_only",),
                    ),
                    False,
                )
            ),
        },
        "columns": {
            "patient": str(
                _snapshot_value(
                    config,
                    (
                        ("columns", "patient"),
                        ("patient_column",),
                    ),
                    "",
                )
            ).strip(),
            "celltype": str(
                _snapshot_value(
                    config,
                    (
                        ("columns", "celltype"),
                        ("celltype_column",),
                    ),
                    "",
                )
            ).strip(),
            "label": str(
                _snapshot_value(
                    config,
                    (
                        ("columns", "label"),
                        ("label_column",),
                    ),
                    "",
                )
            ).strip(),
            "sample": _snapshot_value(
                config,
                (
                    ("columns", "sample"),
                    ("sample_column",),
                ),
                None,
            ),
            "treatment": _snapshot_value(
                config,
                (
                    ("columns", "treatment"),
                    ("treatment_column",),
                ),
                None,
            ),
        },
        "labels": {
            "positive": [
                str(value)
                for value in _snapshot_value(
                    config,
                    (
                        ("labels", "positive"),
                        ("binary_positive_labels",),
                    ),
                    ["1"],
                )
            ],
            "negative": [
                str(value)
                for value in _snapshot_value(
                    config,
                    (
                        ("labels", "negative"),
                        ("binary_negative_labels",),
                    ),
                    ["0"],
                )
            ],
        },
        "resources": {
            "ppi_path": str(
                _snapshot_value(
                    config,
                    (
                        ("resources", "ppi_path"),
                        ("ppi_path",),
                    ),
                    "",
                )
            ).strip(),
            "embedding_views": _normalize_embedding_views(
                _snapshot_value(
                    config,
                    (
                        ("resources", "embedding_views"),
                        ("embedding_views",),
                        ("gene_embedding_views",),
                        ("protein_embedding_paths",),
                    ),
                    {},
                )
            ),
        },
        "training": {
            "epochs": int(
                _snapshot_value(
                    config,
                    (
                        ("training", "epochs"),
                        ("epochs",),
                    ),
                    _LEGACY_DEFAULTS["epochs"],
                )
            ),
            "lr": float(
                _snapshot_value(
                    config,
                    (
                        ("training", "lr"),
                        ("lr",),
                    ),
                    _LEGACY_DEFAULTS["lr"],
                )
            ),
            "k": int(
                _snapshot_value(
                    config,
                    (
                        ("training", "k"),
                        ("k",),
                    ),
                    _LEGACY_DEFAULTS["k"],
                )
            ),
        },
    }


def serialize_workflow_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(dict(config))
    _deep_merge(snapshot, _build_snapshot_sections(config))
    for key in _SNAPSHOT_RUNTIME_KEYS:
        if key in config:
            snapshot[str(key)] = copy.deepcopy(config[str(key)])
    return snapshot


def _looks_like_flat_workflow_snapshot(config: Mapping[str, Any]) -> bool:
    if any(key in config for key in _LEGACY_SNAPSHOT_MARKERS):
        return True
    return all(key not in config for key in _SNAPSHOT_SECTION_KEYS) and any(
        key in config
        for key in {
            "input_h5ad",
            "output_root",
            "run_dir",
            "patient_column",
            "celltype_column",
            "label_column",
            "ppi_path",
            "epochs",
            "lr",
            "k",
        }
    )


def upgrade_legacy_flat_workflow_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = copy.deepcopy(dict(config))
    _deep_merge(upgraded, _build_snapshot_sections(config))
    return upgraded


def normalize_workflow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(_LEGACY_DEFAULTS)
    legacy_top_level_overrides = {
        str(key): copy.deepcopy(value)
        for key, value in config.items()
        if str(key) not in _SNAPSHOT_SECTION_KEYS
    }
    _deep_merge(normalized, legacy_top_level_overrides)

    input_h5ad = str(
        _coalesce(
            config,
            (
                ("workflow", "input_h5ad"),
                ("input_h5ad",),
                ("adata_path",),
            ),
            "",
        )
    ).strip()
    output_root = str(
        _coalesce(
            config,
            (
                ("workflow", "output_root"),
                ("output_root",),
            ),
            "./outputs/mil2het",
        )
    ).strip()
    run_dir = str(
        _coalesce(
            config,
            (
                ("workflow", "run_dir"),
                ("run_dir",),
                ("biomarker_run_dir",),
            ),
            "",
        )
    ).strip()
    analysis_output_dir = str(
        _coalesce(
            config,
            (
                ("workflow", "analysis_output_dir"),
                ("analysis_output_dir",),
                ("biomarker_output_dir",),
            ),
            "",
        )
    ).strip()

    patient_column = str(
        _coalesce(
            config,
            (
                ("columns", "patient"),
                ("patient_column",),
            ),
            "",
        )
    ).strip()
    celltype_column = str(
        _coalesce(
            config,
            (
                ("columns", "celltype"),
                ("celltype_column",),
            ),
            "",
        )
    ).strip()
    label_column = str(
        _coalesce(
            config,
            (
                ("columns", "label"),
                ("label_column",),
            ),
            "",
        )
    ).strip()

    sample_value = _coalesce(
        config,
        (
            ("columns", "sample"),
            ("sample_column",),
        ),
        None,
    )
    sample_column = None if sample_value in {None, ""} else str(sample_value).strip()

    treatment_value = _coalesce(
        config,
        (
            ("columns", "treatment"),
            ("treatment_column",),
        ),
        None,
    )
    treatment_column = None if treatment_value in {None, ""} else str(treatment_value).strip()

    ppi_path = str(
        _coalesce(
            config,
            (
                ("resources", "ppi_path"),
                ("ppi_path",),
            ),
            "",
        )
    ).strip()
    embedding_views = _normalize_embedding_views(
        _coalesce(
            config,
            (
                ("resources", "embedding_views"),
                ("embedding_views",),
                ("gene_embedding_views",),
                ("protein_embedding_paths",),
            ),
            {},
        )
    )
    protein_embedding_sources = _normalize_string_list(
        _coalesce(
            config,
            (
                ("protein_embedding_sources",),
                ("prior_view_sources",),
            ),
            list(embedding_views.keys()),
        ),
        field_name="protein_embedding_sources",
    )
    prior_view_sources = _normalize_string_list(
        _coalesce(
            config,
            (
                ("prior_view_sources",),
                ("protein_embedding_sources",),
            ),
            list(embedding_views.keys()),
        ),
        field_name="prior_view_sources",
    )

    positive_labels = list(
        _coalesce(
            config,
            (
                ("labels", "positive"),
                ("binary_positive_labels",),
            ),
            ["1"],
        )
    )
    negative_labels = list(
        _coalesce(
            config,
            (
                ("labels", "negative"),
                ("binary_negative_labels",),
            ),
            ["0"],
        )
    )
    if len(positive_labels) == 0:
        raise ValueError("positive labels must be non-empty.")
    if len(negative_labels) == 0:
        raise ValueError("negative labels must be non-empty.")

    train_only = bool(
        _coalesce(
            config,
            (
                ("workflow", "train_only"),
                ("train_only",),
                ("skip_analysis",),
            ),
            False,
        )
    )
    analysis_only = bool(
        _coalesce(
            config,
            (
                ("workflow", "analysis_only"),
                ("analysis_only",),
            ),
            False,
        )
    )
    if train_only and analysis_only:
        raise ValueError("train_only and analysis_only cannot both be true.")
    if analysis_only and run_dir == "":
        raise ValueError("analysis_only requires run_dir.")

    dataset_name = _infer_dataset_name(config=config, input_h5ad=input_h5ad)

    normalized["input_h5ad"] = input_h5ad
    normalized["adata_path"] = input_h5ad
    normalized["output_root"] = output_root
    normalized["run_dir"] = run_dir
    normalized["analysis_output_dir"] = analysis_output_dir
    normalized["dataset"] = dataset_name
    normalized["patient_column"] = patient_column
    normalized["celltype_column"] = celltype_column
    normalized["label_column"] = label_column
    normalized["sample_column"] = sample_column
    normalized["treatment_column"] = treatment_column
    normalized["ppi_path"] = ppi_path
    normalized["embedding_views"] = embedding_views
    normalized["gene_embedding_views"] = copy.deepcopy(embedding_views)
    normalized["protein_embedding_paths"] = copy.deepcopy(embedding_views)
    normalized["protein_embedding_sources"] = (
        protein_embedding_sources if len(protein_embedding_sources) > 0 else list(embedding_views.keys())
    )
    normalized["prior_view_sources"] = (
        prior_view_sources if len(prior_view_sources) > 0 else list(embedding_views.keys())
    )
    normalized["binary_positive_labels"] = [str(value) for value in positive_labels]
    normalized["binary_negative_labels"] = [str(value) for value in negative_labels]
    normalized["binary_positive_label"] = str(positive_labels[0])
    normalized["binary_negative_label"] = str(negative_labels[0])
    normalized["train_only"] = train_only
    normalized["analysis_only"] = analysis_only
    normalized["skip_analysis"] = train_only
    normalized["seed"] = int(
        _coalesce(
            config,
            (
                ("workflow", "seed"),
                ("seed",),
            ),
            normalized["seed"],
        )
    )
    normalized["num_folds"] = int(
        _coalesce(
            config,
            (
                ("workflow", "num_folds"),
                ("num_folds",),
            ),
            normalized["num_folds"],
        )
    )
    normalized["split_number"] = int(
        _coalesce(
            config,
            (
                ("workflow", "split_number"),
                ("split_number",),
            ),
            0,
        )
    )
    normalized["epochs"] = int(
        _coalesce(
            config,
            (
                ("training", "epochs"),
                ("epochs",),
            ),
            normalized["epochs"],
        )
    )
    if not analysis_only and int(normalized["epochs"]) < 5:
        raise ValueError(
            "mil2het workflow requires epochs >= 5 when training is enabled, "
            f"received epochs={int(normalized['epochs'])}."
        )
    normalized["lr"] = float(
        _coalesce(
            config,
            (
                ("training", "lr"),
                ("lr",),
            ),
            normalized["lr"],
        )
    )
    normalized["k"] = int(
        _coalesce(
            config,
            (
                ("training", "k"),
                ("k",),
            ),
            normalized["k"],
        )
    )

    if input_h5ad != "":
        normalized["adata_directory"] = str(Path(input_h5ad).resolve().parent)

    normalized["splits_directory"] = _build_output_path(
        base_dir=output_root,
        leaf="splits",
        explicit=str(_coalesce(config, (("splits_directory",),), "")).strip(),
    )
    normalized["preselection_output_root"] = _build_output_path(
        base_dir=output_root,
        leaf="preselection",
        explicit=str(_coalesce(config, (("preselection_output_root",),), "")).strip(),
    )
    normalized["experiment_root"] = _build_output_path(
        base_dir=output_root,
        leaf="training",
        explicit=str(_coalesce(config, (("experiment_root",),), "")).strip(),
    )
    normalized["biomarker_run_dir"] = run_dir
    normalized["biomarker_output_dir"] = _build_output_path(
        base_dir=output_root,
        leaf="analysis",
        explicit=analysis_output_dir,
    )
    return normalized


def load_workflow_config_dict(
    config_path: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(load_default_config_dict())
    if config_path is not None and str(config_path).strip() != "":
        loaded = load_yaml_config_dict(str(config_path))
        if _looks_like_flat_workflow_snapshot(loaded):
            loaded = upgrade_legacy_flat_workflow_snapshot(loaded)
        _deep_merge(merged, loaded)
    if overrides:
        _deep_merge(merged, overrides)
    return normalize_workflow_config(merged)


def load_workflow_config_namespace(
    config_path: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    return dict_to_namespace(load_workflow_config_dict(config_path=config_path, overrides=overrides))


def _parse_embedding_arguments(values: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        text = str(value).strip()
        if "=" not in text:
            raise ValueError(
                "gene embedding overrides must use the form <view_name>=<path>."
            )
        view_name, path_value = text.split("=", 1)
        view_name = view_name.strip()
        path_value = path_value.strip()
        if view_name == "" or path_value == "":
            raise ValueError(
                "gene embedding overrides must use the form <view_name>=<path>."
            )
        if view_name in parsed and parsed[view_name] != path_value:
            raise ValueError(f"duplicate embedding override for view '{view_name}'.")
        parsed[view_name] = path_value
    return parsed


def build_workflow_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load mil2het workflow config.")
    parser.add_argument("--config", dest="config_path", default=None, help="External YAML config path.")
    parser.add_argument("--adata", dest="input_h5ad", default=None, help="Input .h5ad path.")
    parser.add_argument("--patient-column", "--patient", dest="patient_column", default=None)
    parser.add_argument(
        "--celltype-column",
        "--celltype",
        "--cell-type",
        "--cell_type",
        dest="celltype_column",
        default=None,
    )
    parser.add_argument("--label-column", "--label", dest="label_column", default=None)
    parser.add_argument("--sample-column", "--sample", dest="sample_column", default=None)
    parser.add_argument("--treatment-column", "--treatment", dest="treatment_column", default=None)
    parser.add_argument("--ppi", dest="ppi_path", default=None)
    parser.add_argument("--output-dir", "--output-root", dest="output_root", default=None)
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    parser.add_argument("--analysis-output-dir", dest="analysis_output_dir", default=None)
    parser.add_argument("--num-folds", dest="num_folds", type=int, default=None)
    parser.add_argument("--split-number", dest="split_number", type=int, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--lr", dest="lr", type=float, default=None)
    parser.add_argument(
        "--gene-embedding",
        "--add-gene-embedding",
        "--add_gene_embedding",
        dest="embedding_views",
        action="append",
        default=None,
        help="Repeated override of the form <view_name>=<path>.",
    )
    parser.add_argument("--train-only", dest="train_only", action="store_true", default=None)
    parser.add_argument("--train_only", dest="train_only", action="store_true", default=None)
    parser.add_argument(
        "--skip-analysis",
        "--skip_analysis",
        dest="train_only",
        action="store_true",
        default=None,
    )
    parser.add_argument("--analysis-only", dest="analysis_only", action="store_true", default=None)
    parser.add_argument("--analysis_only", dest="analysis_only", action="store_true", default=None)
    return parser


def workflow_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    workflow_overrides: dict[str, Any] = {}
    column_overrides: dict[str, Any] = {}
    resource_overrides: dict[str, Any] = {}
    training_overrides: dict[str, Any] = {}

    if args.input_h5ad is not None:
        workflow_overrides["input_h5ad"] = str(args.input_h5ad)
    if args.output_root is not None:
        workflow_overrides["output_root"] = str(args.output_root)
    if args.run_dir is not None:
        workflow_overrides["run_dir"] = str(args.run_dir)
    if args.analysis_output_dir is not None:
        workflow_overrides["analysis_output_dir"] = str(args.analysis_output_dir)
    if args.num_folds is not None:
        workflow_overrides["num_folds"] = int(args.num_folds)
    if args.split_number is not None:
        workflow_overrides["split_number"] = int(args.split_number)
    if args.seed is not None:
        workflow_overrides["seed"] = int(args.seed)
    if args.train_only is not None:
        workflow_overrides["train_only"] = bool(args.train_only)
    if args.analysis_only is not None:
        workflow_overrides["analysis_only"] = bool(args.analysis_only)

    if args.patient_column is not None:
        column_overrides["patient"] = str(args.patient_column)
    if args.celltype_column is not None:
        column_overrides["celltype"] = str(args.celltype_column)
    if args.label_column is not None:
        column_overrides["label"] = str(args.label_column)
    if args.sample_column is not None:
        column_overrides["sample"] = str(args.sample_column)
    if args.treatment_column is not None:
        column_overrides["treatment"] = str(args.treatment_column)

    if args.ppi_path is not None:
        resource_overrides["ppi_path"] = str(args.ppi_path)
    if args.embedding_views is not None:
        resource_overrides["embedding_views"] = _parse_embedding_arguments(args.embedding_views)

    if args.epochs is not None:
        training_overrides["epochs"] = int(args.epochs)
    if args.lr is not None:
        training_overrides["lr"] = float(args.lr)

    if workflow_overrides:
        overrides["workflow"] = workflow_overrides
    if column_overrides:
        overrides["columns"] = column_overrides
    if resource_overrides:
        overrides["resources"] = resource_overrides
    if training_overrides:
        overrides["training"] = training_overrides
    return overrides


def parse_workflow_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_workflow_arg_parser()
    args = parser.parse_args(argv)
    overrides = workflow_overrides_from_args(args)
    config_dict = load_workflow_config_dict(config_path=args.config_path, overrides=overrides)
    return argparse.Namespace(
        config_path=args.config_path,
        overrides=overrides,
        config_dict=config_dict,
        config=dict_to_namespace(config_dict),
    )


__all__ = [
    "DEFAULT_CONFIG_RESOURCE",
    "build_workflow_arg_parser",
    "dict_to_namespace",
    "load_default_config_dict",
    "load_workflow_config_dict",
    "load_workflow_config_namespace",
    "load_yaml_config_dict",
    "normalize_workflow_config",
    "parse_workflow_cli_args",
    "workflow_overrides_from_args",
]
