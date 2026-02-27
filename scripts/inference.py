"""Inference for checkpoints produced by `scripts/train.py`.

Examples:
  python scripts/inference.py \
    --run-dir experiment_asthma_find/train_runs/<run_name> \
    --checkpoint-type best \
    --gpu 0

  # If run_config.json is missing in run-dir, provide runtime fields explicitly.
  python scripts/inference.py \
    --run-dir experiment_asthma_find/train_runs/<run_name> \
    --dataset asthma --evaluation kfold --split-number 2 \
    --checkpoint-type best --gpu 0
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.train as train_mod


def _resolve_prior_apply_flags(config: SimpleNamespace) -> Tuple[bool, bool]:
    prior_absolute_enabled_raw = getattr(config, "prior_absolute_enabled", None)
    prior_relational_enabled_raw = getattr(config, "prior_relational_enabled", None)
    if prior_absolute_enabled_raw is None or prior_relational_enabled_raw is None:
        legacy_prior_strategy = getattr(config, "prior_strategy", None)
        if isinstance(legacy_prior_strategy, dict):
            if prior_absolute_enabled_raw is None:
                prior_absolute_enabled_raw = bool(legacy_prior_strategy.get("hidden_additive", True))
            if prior_relational_enabled_raw is None:
                prior_relational_enabled_raw = bool(legacy_prior_strategy.get("regularizer_multiview_knn", True))
        else:
            if prior_absolute_enabled_raw is None:
                prior_absolute_enabled_raw = True
            if prior_relational_enabled_raw is None:
                prior_relational_enabled_raw = True
    return bool(prior_absolute_enabled_raw), bool(prior_relational_enabled_raw)


def _resolve_device(gpu_index: int) -> torch.device:
    if int(gpu_index) < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but --gpu >= 0 was provided.")
    if int(gpu_index) >= int(torch.cuda.device_count()):
        raise ValueError(
            f"Invalid --gpu index {int(gpu_index)} for {int(torch.cuda.device_count())} visible CUDA devices."
        )
    device = torch.device(f"cuda:{int(gpu_index)}")
    torch.cuda.set_device(device)
    return device


def _first_existing_path(candidates: Sequence[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{label} not found. Tried: {list(candidates)}")


def _config_path_candidates(raw_path: str) -> List[str]:
    normalized = str(raw_path).strip()
    if normalized == "":
        return []
    if os.path.isabs(normalized):
        return [os.path.abspath(normalized)]
    candidates = [
        os.path.abspath(normalized),
        os.path.abspath(os.path.join(PROJECT_ROOT, normalized)),
        os.path.abspath(os.path.join(SCRIPT_DIR, normalized)),
    ]
    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _infer_dataset_from_run_dir(run_dir: str) -> Optional[str]:
    run_dir_lower = str(run_dir).lower()
    if "asthma_ext" in run_dir_lower:
        return "asthma_ext"
    if "asthma" in run_dir_lower:
        return "asthma"
    return None


def _infer_split_number_from_run_dir(run_dir: str) -> Optional[int]:
    basename = os.path.basename(os.path.abspath(run_dir))
    match = re.search(r"(?:^|[_-])split(?:_|-)?(\d+)(?:$|[_-])", basename)
    if match is None:
        return None
    return int(match.group(1))


def _infer_seed_from_run_dir(run_dir: str) -> Optional[int]:
    basename = os.path.basename(os.path.abspath(run_dir))
    match = re.search(r"(?:^|[_-])seed(?:_|-)?(\\d+)(?:$|[_-])", basename)
    if match is None:
        return None
    return int(match.group(1))


def _load_python_module_from_file(path: str) -> ModuleType:
    module_name = f"inference_cfg_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_snapshot_config_dict(module: ModuleType, dataset: Optional[str]) -> Dict[str, object]:
    required_keys = {
        "experiment_root",
        "splits_directory",
        "label_column",
        "patient_column",
        "celltype_column",
        "cells_per_bag",
        "bags_per_patient_per_epoch",
        "mil_pooling",
        "classifier_name",
    }

    scored: List[Tuple[int, str, Dict[str, object]]] = []
    for key, value in vars(module).items():
        if not isinstance(value, dict):
            continue
        score = int(sum(1 for required_key in required_keys if required_key in value))
        if score <= 0:
            continue
        scored.append((score, str(key), dict(value)))

    if len(scored) == 0:
        raise ValueError("No usable training config dict found in snapshot module.")

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    if dataset is not None:
        dataset_lower = str(dataset).lower().strip()
        preferred: List[Tuple[int, str, Dict[str, object]]] = []
        for item in scored:
            name_lower = item[1].lower()
            if dataset_lower == "asthma_ext" and "asthma_ext" in name_lower:
                preferred.append(item)
            elif dataset_lower == "asthma" and ("asthma_ext" not in name_lower and "asthma" in name_lower):
                preferred.append(item)
        if len(preferred) > 0:
            return preferred[0][2]

    return scored[0][2]


def _load_snapshot_config(run_dir: str, dataset_hint: Optional[str]) -> Tuple[Dict[str, object], str]:
    candidate_files = [
        os.path.join(run_dir, "asthma_config.py"),
        os.path.join(run_dir, "asthma_find_config.py"),
    ]
    candidate_files.extend(
        sorted(
            os.path.join(run_dir, file_name)
            for file_name in os.listdir(run_dir)
            if file_name.endswith("_config.py")
        )
    )

    unique_candidates: List[str] = []
    seen = set()
    for path in candidate_files:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)

    for path in unique_candidates:
        if not os.path.exists(path):
            continue
        try:
            module = _load_python_module_from_file(path)
            config_dict = _select_snapshot_config_dict(module=module, dataset=dataset_hint)
            return config_dict, path
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not load a snapshot config (*.py) from run-dir. "
        "Expected run_config.json or a copied config file."
    )


def _resolve_split_number(config: SimpleNamespace, override: Optional[int], run_dir: str) -> int:
    if override is not None:
        return int(override)

    current = getattr(config, "split_number", None)
    if current is not None:
        return int(current)

    inferred = _infer_split_number_from_run_dir(run_dir)
    if inferred is not None:
        return int(inferred)

    raise ValueError(
        "split_number could not be resolved. Pass --split-number explicitly or include it in run-dir name."
    )


def _resolve_evaluation(
    config: SimpleNamespace,
    override: Optional[str],
    *,
    dataset: str,
    split_number: int,
) -> str:
    if override is not None and str(override).strip() != "":
        return str(override).strip().lower()

    current = str(getattr(config, "evaluation", "")).strip().lower()
    if current in {"kfold", "lopo"}:
        return current

    split_dir_raw = str(getattr(config, "splits_directory", "")).strip()
    for split_dir in _config_path_candidates(split_dir_raw):
        candidates = [
            ("kfold", os.path.join(split_dir, "kfold", f"{dataset}_idx_{int(split_number)}.pkl")),
            ("lopo", os.path.join(split_dir, "lopo", f"{dataset}_idx_{int(split_number)}.pkl")),
            ("kfold", os.path.join(split_dir, f"{dataset}_idx_{int(split_number)}.pkl")),
        ]
        for evaluation_name, candidate_path in candidates:
            if os.path.exists(candidate_path):
                return str(evaluation_name)

    return "kfold"


def _resolve_adata_path(config: SimpleNamespace, dataset: str) -> str:
    candidates: List[str] = []

    adata_path_raw = str(getattr(config, "adata_path", "")).strip()
    candidates.extend(_config_path_candidates(adata_path_raw))

    adata_directory_raw = str(getattr(config, "adata_directory", "")).strip()
    for adata_directory in _config_path_candidates(adata_directory_raw):
        candidates.append(
            os.path.abspath(
                os.path.join(
                    adata_directory,
                    str(dataset),
                    f"{dataset}_data.h5ad",
                )
            )
        )

    return _first_existing_path(candidates, "adata path")


def _resolve_preselection_root(config: SimpleNamespace, dataset: str, evaluation: str, split_number: int) -> str:
    project_root = str(train_mod.PROJECT_ROOT)
    split_subdir = str(getattr(config, "split_preselection_subdir", "")).strip()
    candidates = [
        os.path.join(project_root, "data", dataset, "preselection", evaluation, f"split_{int(split_number)}"),
        os.path.join(project_root, "data", dataset, "preselection", evaluation, "fold_0", f"split_{int(split_number)}"),
        os.path.join(project_root, "data", dataset, "preselection", f"split_{int(split_number)}"),
    ]
    if split_subdir != "":
        candidates.append(
            os.path.join(project_root, "data", dataset, split_subdir, f"split_idx_{int(split_number)}")
        )

    return _first_existing_path(candidates, "preselection root")


def _resolve_split_path(config: SimpleNamespace, dataset: str, evaluation: str, split_number: int) -> str:
    candidates: List[str] = []
    for split_root in _config_path_candidates(str(config.splits_directory)):
        candidates.extend(
            [
                os.path.join(split_root, evaluation, f"{dataset}_idx_{int(split_number)}.pkl"),
                os.path.join(split_root, f"{dataset}_idx_{int(split_number)}.pkl"),
            ]
        )
    return _first_existing_path(candidates, "split file")


def _load_config_from_run(
    run_dir: str,
    *,
    dataset_override: Optional[str],
    evaluation_override: Optional[str],
    split_number_override: Optional[int],
) -> Tuple[SimpleNamespace, str]:
    run_config_path = os.path.join(run_dir, "run_config.json")

    config: SimpleNamespace
    provenance: str

    if os.path.exists(run_config_path):
        with open(run_config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config_dict = payload.get("config", None)
        if not isinstance(config_dict, dict):
            raise ValueError(f"Invalid config payload in {run_config_path}")
        config = train_mod.dict2namespace(dict(config_dict))
        provenance = f"run_config.json ({run_config_path})"
    else:
        dataset_hint = dataset_override if dataset_override is not None else _infer_dataset_from_run_dir(run_dir)
        snapshot_config_dict, snapshot_path = _load_snapshot_config(run_dir=run_dir, dataset_hint=dataset_hint)
        config = train_mod.dict2namespace(dict(snapshot_config_dict))
        provenance = f"snapshot config ({snapshot_path})"

    dataset = str(dataset_override).strip() if dataset_override is not None else str(getattr(config, "dataset", "")).strip()
    if dataset == "":
        inferred_dataset = _infer_dataset_from_run_dir(run_dir)
        if inferred_dataset is None:
            raise ValueError("dataset could not be resolved. Pass --dataset explicitly.")
        dataset = str(inferred_dataset)
    config.dataset = str(dataset)

    split_number = _resolve_split_number(config=config, override=split_number_override, run_dir=run_dir)
    config.split_number = int(split_number)

    evaluation = _resolve_evaluation(
        config=config,
        override=evaluation_override,
        dataset=str(dataset),
        split_number=int(split_number),
    )
    if evaluation not in {"kfold", "lopo"}:
        raise ValueError(f"Unsupported evaluation mode: {evaluation}")
    config.evaluation = str(evaluation)

    if not hasattr(config, "binary_positive_labels"):
        single_positive = getattr(config, "binary_positive_label", "1")
        config.binary_positive_labels = [str(single_positive)]
    if not hasattr(config, "binary_negative_labels"):
        single_negative = getattr(config, "binary_negative_label", "0")
        config.binary_negative_labels = [str(single_negative)]

    if not hasattr(config, "seed"):
        inferred_seed = _infer_seed_from_run_dir(run_dir)
        config.seed = int(inferred_seed) if inferred_seed is not None else 0
    if not hasattr(config, "deterministic_training"):
        config.deterministic_training = False
    if not hasattr(config, "deterministic_algorithms"):
        config.deterministic_algorithms = False
    if not hasattr(config, "deterministic_warn_only"):
        config.deterministic_warn_only = True

    config.adata_path = _resolve_adata_path(config=config, dataset=str(dataset))

    config.cell_encoder_name = train_mod.normalize_cell_encoder_name(str(config.cell_encoder_name))
    prior_absolute_enabled, prior_relational_enabled = _resolve_prior_apply_flags(config)
    config.prior_absolute_enabled = bool(prior_absolute_enabled)
    config.prior_relational_enabled = bool(prior_relational_enabled)

    return config, provenance


def _build_data_and_mappings(
    config: SimpleNamespace,
    *,
    preselection_root: str,
) -> Tuple[
    object,
    Dict[int, str],
    Dict[int, str],
    Dict[str, int],
    Dict[str, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    print(f"Loading adata: {config.adata_path}", flush=True)
    adata = sc.read_h5ad(str(config.adata_path))

    label_column = str(config.label_column)
    patient_column = str(config.patient_column)
    celltype_column = str(config.celltype_column)
    treatment_column = None
    if config.treatment_column is not None and str(config.treatment_column).strip() != "":
        treatment_column = str(config.treatment_column)

    required_columns = [label_column, patient_column, celltype_column]
    for column_name in required_columns:
        if column_name not in adata.obs.columns:
            raise ValueError(
                f"Missing required obs column '{column_name}'. "
                f"Available columns: {list(adata.obs.columns)}"
            )
    if treatment_column is not None and treatment_column not in adata.obs.columns:
        raise ValueError(
            f"Missing optional treatment obs column '{treatment_column}'. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    raw_labels = adata.obs[label_column].astype(str).tolist()
    mapped_labels, _ = train_mod.map_labels(
        label_values=raw_labels,
        binary_positive_labels=list(config.binary_positive_labels),
        binary_negative_labels=list(config.binary_negative_labels),
    )
    _label_mapping, label_indices = train_mod.create_category_mapping(mapped_labels)

    celltype_values = adata.obs[celltype_column].astype(str).tolist()
    celltype_mapping, celltype_indices = train_mod.create_category_mapping(celltype_values)

    if treatment_column is not None:
        treatment_values = adata.obs[treatment_column].astype(str).tolist()
    else:
        treatment_values = ["NA"] * adata.n_obs
    treatment_mapping, treatment_indices = train_mod.create_category_mapping(treatment_values)

    patient_values = adata.obs[patient_column].astype(str).tolist()
    patient_mapping, patient_indices = train_mod.create_category_mapping(patient_values)

    label_array = np.asarray(label_indices, dtype=np.int64)
    celltype_array = np.asarray(celltype_indices, dtype=np.int64)
    treatment_array = np.asarray(treatment_indices, dtype=np.int64)
    patient_array = np.asarray(patient_indices, dtype=np.int64)

    celltype_index_to_name = {int(index): str(name) for name, index in celltype_mapping.items()}
    patient_index_to_name = {int(index): str(name) for name, index in patient_mapping.items()}

    genes, _maximum_genes = train_mod.load_k_np_genes(
        str(config.dataset),
        int(config.k),
        preselection_root=preselection_root,
    )

    return (
        adata,
        celltype_index_to_name,
        patient_index_to_name,
        celltype_mapping,
        treatment_mapping,
        label_array,
        celltype_array,
        treatment_array,
        patient_array,
        genes,
    )


def _load_cell_encoder_state_compat(cell_encoder: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        cell_encoder.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as strict_error:
        allowed_missing = {"edge_index", "edge_index_knn", "edge_index_ppi"}
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key not in allowed_missing
        }
        incompatible = cell_encoder.load_state_dict(filtered_state, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
        disallowed_missing = sorted(key for key in missing_keys if key not in allowed_missing)
        disallowed_unexpected = sorted(unexpected_keys)
        if len(disallowed_missing) > 0 or len(disallowed_unexpected) > 0:
            raise RuntimeError(
                "Failed to load cell_encoder checkpoint with compatibility mode. "
                f"disallowed_missing={disallowed_missing}, disallowed_unexpected={disallowed_unexpected}"
            ) from strict_error
        ignored = sorted(key for key in missing_keys if key in allowed_missing)
        print(
            "[Checkpoint][warn] cell_encoder loaded with graph-buffer compatibility mode. "
            f"ignored={ignored}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference from saved FIND train run checkpoint")
    parser.add_argument("--run-dir", required=True, help="train run directory containing checkpoint files")
    parser.add_argument("--checkpoint-type", default="best", choices=["best", "last"])
    parser.add_argument("--split", default="test", choices=["val", "test", "both"])
    parser.add_argument("--gpu", type=int, default=0, help="cuda index; use -1 for CPU")
    parser.add_argument("--dataset", default=None, help="optional dataset override (e.g. asthma, asthma_ext)")
    parser.add_argument("--evaluation", default=None, choices=["kfold", "lopo"], help="optional evaluation override")
    parser.add_argument("--split-number", type=int, default=None, help="optional split index override")
    args = parser.parse_args()

    run_dir = os.path.abspath(str(args.run_dir))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    config, config_source = _load_config_from_run(
        run_dir=run_dir,
        dataset_override=args.dataset,
        evaluation_override=args.evaluation,
        split_number_override=args.split_number,
    )
    device = _resolve_device(int(args.gpu))

    preselection_root = _resolve_preselection_root(
        config=config,
        dataset=str(config.dataset),
        evaluation=str(config.evaluation),
        split_number=int(config.split_number),
    )
    split_path = _resolve_split_path(
        config=config,
        dataset=str(config.dataset),
        evaluation=str(config.evaluation),
        split_number=int(config.split_number),
    )

    train_mod.ensure_cublas_workspace_config()
    train_mod.configure_runtime_backends(
        device=device,
        deterministic_training=bool(config.deterministic_training),
    )
    train_mod.set_global_seed(
        int(config.seed),
        deterministic=bool(config.deterministic_training),
    )
    if bool(config.deterministic_algorithms):
        torch.use_deterministic_algorithms(
            True,
            warn_only=bool(config.deterministic_warn_only),
        )

    print(f"[Inference] run_dir={run_dir}", flush=True)
    print(f"[Inference] config_source={config_source}", flush=True)
    print(f"[Inference] checkpoint_type={args.checkpoint_type}", flush=True)
    print(
        f"[Inference] dataset={config.dataset} evaluation={config.evaluation} "
        f"split_number={int(config.split_number)}",
        flush=True,
    )
    print(f"[Inference] preselection_root={preselection_root}", flush=True)
    print(f"[Inference] split_path={split_path}", flush=True)
    print(f"[Inference] device={device}", flush=True)

    (
        adata,
        celltype_index_to_name,
        patient_index_to_name,
        celltype_mapping,
        treatment_mapping,
        label_array,
        celltype_array,
        treatment_array,
        patient_array,
        genes,
    ) = _build_data_and_mappings(config=config, preselection_root=preselection_root)

    gene_indices = train_mod.map_genes_to_adata(adata, genes)
    expression_matrix = adata[:, gene_indices].X

    encoder_spec = train_mod.resolve_cell_encoder_spec(config=config, num_nodes=int(len(genes)))
    config.cell_encoder_name = str(encoder_spec["name"])
    encoder_class = encoder_spec["encoder_class"]
    encoder_hidden_dimension = int(encoder_spec["hidden_dimension"])
    encoder_number_of_layers = int(encoder_spec["number_of_layers"])
    encoder_number_of_heads = int(encoder_spec["number_of_heads"])
    encoder_dropout_probability = float(encoder_spec["dropout_probability"])
    encoder_graph_readout = str(encoder_spec["graph_readout"])
    cell_embedding_dimension = int(encoder_spec["cell_embedding_dimension"])

    print(
        "[CellEncoder] "
        f"name={config.cell_encoder_name} hidden_dim={encoder_hidden_dimension} "
        f"layers={encoder_number_of_layers} heads={encoder_number_of_heads} "
        f"readout={encoder_graph_readout} cell_embedding_dim={cell_embedding_dimension}",
        flush=True,
    )

    prior_view_sources = [str(name).strip() for name in list(config.prior_view_sources) if str(name).strip()]
    if len(prior_view_sources) == 0:
        raise ValueError("prior_view_sources must be non-empty for FIND inference.")

    prior_embeddings_by_view = train_mod.load_prior_embeddings_by_view(
        genes=genes,
        view_names=prior_view_sources,
        config=config,
    )
    if len(prior_embeddings_by_view) == 0:
        raise ValueError("No prior view embeddings were loaded for FIND inference.")

    prior_absolute_enabled = bool(getattr(config, "prior_absolute_enabled", True))
    prior_relational_enabled = bool(getattr(config, "prior_relational_enabled", True))
    prior_strategy_label = (
        f"absolute={int(prior_absolute_enabled)},relational={int(prior_relational_enabled)}"
    )
    prior_injection_enabled = bool(prior_absolute_enabled or prior_relational_enabled)
    lambda_edge_bias_effective = (
        float(config.lambda_edge_bias) if bool(prior_relational_enabled) else 0.0
    )

    prior_view_dims = {
        str(view_name): int(view_tensor.shape[1])
        for view_name, view_tensor in prior_embeddings_by_view.items()
    }
    print(
        "[PriorInterface] "
        f"strategy={prior_strategy_label} "
        f"absolute={prior_absolute_enabled} relational={prior_relational_enabled} "
        f"views={prior_view_sources} dims={prior_view_dims} "
        f"prior_knn_k={int(config.prior_knn_k)} "
        f"lambda_edge_bias={float(lambda_edge_bias_effective):.6g}",
        flush=True,
    )

    ordered_celltypes = [
        name
        for name, _index in sorted(celltype_mapping.items(), key=lambda item: int(item[1]))
    ]
    edge_index, regulation_onehot, global_zscore, _graph_cache_hit, _graph_cache_meta = train_mod.build_graph_cache(
        config=config,
        genes=genes,
        celltype_names=ordered_celltypes,
        cache_path=os.devnull,
        preselection_root=preselection_root,
    )

    split_indices = train_mod.load_pickle(split_path)
    split_indices, split_index_resolution = train_mod.normalize_split_indices_to_current_adata(
        split_indices=split_indices,
        adata_obs_names=adata.obs_names,
        config=config,
    )
    if str(split_index_resolution.get("mode", "direct")) != "direct":
        print(
            "[SplitResolve] "
            f"mode={split_index_resolution.get('mode')} "
            f"reference={split_index_resolution.get('reference_adata_path', '')} "
            f"dropped_reference_cells={int(split_index_resolution.get('dropped_reference_cells', 0))} "
            f"coverage={float(split_index_resolution.get('coverage_ratio', 0.0)):.6f}",
            flush=True,
        )

    train_base_records, val_base_records, test_base_records, _split_cache_hit = train_mod.build_split_records_cache(
        cache_path=os.devnull,
        split_indices=split_indices,
        label_indices=label_array,
        patient_indices=patient_array,
        split_source_path=split_path,
    )

    val_records = train_mod.expand_patient_bag_records(val_base_records, int(config.val_bags_per_patient))
    test_records = train_mod.expand_patient_bag_records(test_base_records, int(config.test_bags_per_patient))

    if len(test_records) == 0:
        raise ValueError("Test records cannot be empty.")

    datasets: Dict[str, train_mod.PatientBagDataset] = {
        "test": train_mod.PatientBagDataset(
            expression_matrix=expression_matrix,
            bag_records=test_records,
            celltype_indices=celltype_array,
            treatment_indices=treatment_array,
        )
    }
    if len(val_records) > 0:
        datasets["val"] = train_mod.PatientBagDataset(
            expression_matrix=expression_matrix,
            bag_records=val_records,
            celltype_indices=celltype_array,
            treatment_indices=treatment_array,
        )

    dataloader_kwargs = train_mod.build_dataloader_kwargs(config, device)
    dataloaders = {
        split_name: DataLoader(dataset_obj, shuffle=False, **dataloader_kwargs)
        for split_name, dataset_obj in datasets.items()
    }

    pooling = str(config.mil_pooling).lower()
    if pooling == "attention":
        pooling = "flat_attention"

    mil_kwargs: Dict[str, object] = {}
    mil_signature = inspect.signature(train_mod.PatientMILAggregator.__init__)

    prior_view_tensors_device = {
        str(view_name): view_tensor.to(device)
        for view_name, view_tensor in prior_embeddings_by_view.items()
    }
    protein_embeddings = torch.zeros((len(genes), int(encoder_hidden_dimension)), dtype=torch.float32)

    cell_encoder = encoder_class(
        num_nodes=int(len(genes)),
        hidden_dimension=int(encoder_hidden_dimension),
        number_of_layers=int(encoder_number_of_layers),
        number_of_heads=int(encoder_number_of_heads),
        dropout_probability=float(encoder_dropout_probability),
        edge_index=edge_index.to(device),
        protein_prior_embeddings=protein_embeddings.to(device),
        global_zscore=global_zscore.to(device),
        regulation_onehot=regulation_onehot.to(device),
        num_celltypes=int(len(celltype_mapping)),
        num_treatments=int(len(treatment_mapping)),
        attention_logit_clamp=float(config.gnn_attention_logit_clamp),
        expression_feature_scale=float(config.gnn_expression_feature_scale),
        max_message_memory_mb=float(config.gnn_max_message_memory_mb),
        graph_readout=str(encoder_graph_readout),
        protein_prior_alpha=1.0,
        protein_prior_alpha_learnable=False,
        prior_injection_enabled=bool(prior_injection_enabled),
        prior_absolute_enabled=bool(prior_absolute_enabled),
        prior_relational_enabled=bool(prior_relational_enabled),
        protein_prior_base_embeddings=None,
        protein_prior_projection_hidden_dim=0,
        protein_prior_projection_dropout=0.0,
        protein_prior_projection_mode="find",
        protein_prior_view_embeddings=prior_view_tensors_device,
        prior_interface_num_heads=int(config.prior_interface_num_heads),
        prior_interface_dropout=float(config.prior_interface_dropout),
        prior_interface_ffn_hidden_dim=int(config.prior_interface_ffn_hidden_dim),
        prior_knn_k=int(config.prior_knn_k),
        prior_knn_symmetric=bool(config.prior_knn_symmetric),
        lambda_edge_bias=float(lambda_edge_bias_effective),
    ).to(device)

    mil_aggregator = train_mod.PatientMILAggregator(
        embedding_dimension=cell_embedding_dimension,
        num_celltypes=int(len(celltype_mapping)),
        pooling=pooling,
        attention_hidden_dimension=int(config.mil_attention_hidden_dim),
        **mil_kwargs,
    ).to(device)
    classifier = train_mod.PatientClassifier(cell_embedding_dimension, config).to(device)

    checkpoint_path = os.path.join(run_dir, f"{str(args.checkpoint_type)}_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    _load_cell_encoder_state_compat(cell_encoder, checkpoint_payload["cell_encoder"])
    mil_aggregator.load_state_dict(checkpoint_payload["mil_aggregator"], strict=True)
    classifier.load_state_dict(checkpoint_payload["classifier"], strict=True)
    checkpoint_epoch = int(checkpoint_payload.get("epoch", -1))

    print(f"[Checkpoint] loaded={checkpoint_path}", flush=True)
    print(f"[Checkpoint] epoch={checkpoint_epoch}", flush=True)

    pos_weight_cfg = config.pos_weight
    pos_weight_tensor = None
    if pos_weight_cfg is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight_cfg)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    sample_mode = str(config.sample_bags)
    top_n = int(config.mil_attention_top_n)
    patient_probability_reduction = str(config.patient_probability_reduction)

    seed_base = int(config.seed) * 1000003
    split_seed_anchor = {
        "train": seed_base + 11,
        "val": seed_base + 23,
        "test": seed_base + 37,
    }

    node_feature_beta_value = float(config.node_feature_beta)
    cell_encoder.set_node_feature_beta(float(node_feature_beta_value))

    regularizer_epoch_kwargs = {
        "node_feature_beta": float(node_feature_beta_value),
        "use_consistency_regularizer": bool(config.use_consistency_regularizer),
        "consistency_weight_lambda": float(config.consistency_weight_lambda),
        "consistency_num_bags_per_patient_per_step": int(config.consistency_num_bags_per_patient_per_step),
        "consistency_detach_second_branch": bool(config.consistency_detach_second_branch),
        "consistency_loss_type": str(config.consistency_loss_type),
        "gradient_clip_max_norm": float(config.gradient_clip_max_norm),
        "mixup_alpha": float(config.mixup_alpha),
        "lambda_sparse": float(config.lambda_sparse),
        "prior_strategy_value": str(prior_strategy_label),
        "prior_strategy_regularizer_multiview_knn": bool(prior_relational_enabled),
        "prior_strategy_hidden_additive": bool(prior_absolute_enabled),
    }

    requested_split = str(args.split).strip().lower()
    eval_splits: List[str]
    if requested_split == "both":
        eval_splits = ["val", "test"]
    elif requested_split in {"val", "test"}:
        eval_splits = [requested_split]
    else:
        raise ValueError(f"Unsupported --split: {args.split}")

    for split_name in eval_splits:
        if split_name not in dataloaders:
            if split_name == "val":
                print("[Inference] val split is empty or unavailable; skipping.", flush=True)
                continue
            raise RuntimeError(f"Requested split '{split_name}' is unavailable.")

        with torch.inference_mode():
            metrics, _bag_predictions, _patient_predictions, _top_cells = train_mod.run_epoch(
                split_name=split_name,
                epoch=checkpoint_epoch,
                dataloader=dataloaders[split_name],
                cell_encoder=cell_encoder,
                mil_aggregator=mil_aggregator,
                classifier=classifier,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=device,
                sample_mode=sample_mode,
                bag_size_cells=int(config.cells_per_bag),
                seed_anchor=int(split_seed_anchor[split_name]),
                patient_index_to_name=patient_index_to_name,
                celltype_index_to_name=celltype_index_to_name,
                step_log_path=os.devnull,
                collect_top_cells=False,
                top_n=top_n,
                aggregate_patient_metrics=True,
                patient_probability_reduction=patient_probability_reduction,
                patient_embedding_ema=None,
                patient_embedding_ema_blend=0.0,
                epoch_dependent_sampling=False,
                **regularizer_epoch_kwargs,
            )

        print(
            f"[{split_name.upper()}] "
            f"loss={float(metrics['loss']):.4f} "
            f"auprc={float(metrics['auprc']):.4f} "
            f"auroc={float(metrics['auroc']):.4f} "
            f"f1={float(metrics['f1']):.4f} "
            f"brier={float(metrics['brier']):.4f}",
            flush=True,
        )

    print("[Inference] done. This mode does not save output files.", flush=True)


if __name__ == "__main__":
    main()
