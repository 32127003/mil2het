"""Biomarker scoring pipeline based on pathway/celltype block permutation.

Implements:
- Step 1: cell-type-specific pathway importance and gene score S_t(g)
- Step 2: cell-type prioritization score w_t
- Final: S_final(g) = sum_t w_t * S_t(g)

Expected usage:
  python modules/biomarker.py --run_dir <train_run_dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy import sparse
from scipy.stats import spearmanr

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CellEncoder import GraphCellEncoder, TransformerConvCellEncoder
from MultipleInstanceLearning import PatientMILAggregator
from mil2het.config import dict_to_namespace, load_workflow_config_dict
from utils import *


@dataclass
class BagDefinition:
    bag_id: int
    patient_index: int
    patient_id: str
    patient_label: int
    bag_repeat_index: int
    cell_indices: np.ndarray


@dataclass
class BagCache:
    celltypes: np.ndarray
    treatments: np.ndarray
    expression: np.ndarray
    embeddings: np.ndarray
    baseline_prob: float


def _extract_config_dict_from_module(module: object, dataset_hint: Optional[str]) -> Dict[str, object]:
    config_dict = getattr(module, "config", None)
    if isinstance(config_dict, dict):
        return dict(config_dict)
    configuration_dict = getattr(module, "configuration", None)
    if isinstance(configuration_dict, dict):
        selected_dict = dict(configuration_dict)
        if "dataset" not in selected_dict and dataset_hint is not None:
            selected_dict["dataset"] = str(dataset_hint)
        return selected_dict

    # Fallback: allow train config modules (e.g., asthma_find_config.py) that expose
    # one or more '*_train_configuration' dicts instead of a single 'config' dict.
    train_config_names: List[str] = []
    for name in dir(module):
        if not name.endswith("_train_configuration"):
            continue
        value = getattr(module, name, None)
        if isinstance(value, dict):
            train_config_names.append(name)

    if not train_config_names:
        raise ValueError(
            "Config file must define a dict named 'config' or at least one '*_train_configuration' dict."
        )

    # Prefer dataset-matched train config when possible.
    preferred_name: Optional[str] = None
    if dataset_hint is not None:
        dataset_hint_norm = str(dataset_hint).strip().lower()
        if dataset_hint_norm == "asthma_ext" and "asthma_ext_train_configuration" in train_config_names:
            preferred_name = "asthma_ext_train_configuration"
        elif dataset_hint_norm == "asthma" and "asthma_train_configuration" in train_config_names:
            preferred_name = "asthma_train_configuration"

    if preferred_name is None:
        preferred_name = "asthma_train_configuration" if "asthma_train_configuration" in train_config_names else train_config_names[0]

    selected_dict = dict(getattr(module, preferred_name))
    if "dataset" not in selected_dict and dataset_hint is not None:
        selected_dict["dataset"] = str(dataset_hint)

    return selected_dict


def load_config(
    config_path: str,
    dataset_hint: Optional[str] = None,
) -> Tuple[SimpleNamespace, Dict[str, object]]:
    if str(config_path).lower().endswith((".yaml", ".yml")):
        config_dict = load_workflow_config_dict(config_path=config_path)
        if str(config_dict.get("dataset", "")).strip() == "" and dataset_hint is not None:
            config_dict["dataset"] = str(dataset_hint)
        return dict_to_namespace(config_dict), dict(config_dict)

    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    config_dict = _extract_config_dict_from_module(module, dataset_hint=dataset_hint)
    return SimpleNamespace(**config_dict), dict(config_dict)


def load_config_like_train_from_run_snapshot(
    config_path: str,
    dataset_hint: Optional[str] = None,
) -> Tuple[SimpleNamespace, Dict[str, object]]:
    """Load run snapshot config with the same dataset->dict pattern as train.py."""
    if str(config_path).lower().endswith((".yaml", ".yml")):
        config_dict = load_workflow_config_dict(config_path=config_path)
        if str(config_dict.get("dataset", "")).strip() == "" and dataset_hint is not None:
            config_dict["dataset"] = str(dataset_hint)
        return dict_to_namespace(config_dict), dict(config_dict)

    dataset_value = str(dataset_hint).strip().lower() if dataset_hint is not None else ""
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    train_key_by_dataset = {
        "asthma": "asthma_train_configuration",
        "asthma_relabeled": "asthma_relabeled_train_configuration",
        "asthma_ext": "asthma_ext_train_configuration",
    }
    selected_key = train_key_by_dataset.get(dataset_value, "")
    if selected_key != "":
        selected_dict = getattr(module, selected_key, None)
        if isinstance(selected_dict, dict):
            args_dict = {"dataset": str(dataset_value)}
            args_dict.update(dict(selected_dict))
            config = SimpleNamespace(**args_dict)
            # Keep parity with current train.py behavior for asthma_ext.
            if str(dataset_value) == "asthma_ext" and hasattr(config, "sample_column"):
                config.patient_column = getattr(config, "sample_column")
            return config, dict(args_dict)

    # Fallback for older snapshots (e.g., configuration/config dict).
    config_dict = _extract_config_dict_from_module(module, dataset_hint=dataset_hint)
    return SimpleNamespace(**config_dict), dict(config_dict)


def resolve_path(base_dir: str, path_value: str) -> str:
    text = str(path_value)
    if os.path.isabs(text):
        return text
    return os.path.abspath(os.path.join(base_dir, text))


def infer_dataset_hint_from_run_dir(run_dir: str) -> Optional[str]:
    run_dir_lower = str(run_dir).lower()
    if "asthma_ext" in run_dir_lower:
        return "asthma_ext"
    if "asthma_relabeled" in run_dir_lower:
        return "asthma_relabeled"
    if "asthma" in run_dir_lower:
        return "asthma"
    return None


def infer_split_number_from_run_dir(run_dir: str) -> Optional[int]:
    basename = os.path.basename(os.path.abspath(run_dir))
    match = re.search(r"(?:^|[_-])split(?:_|-)?(\d+)(?:$|[_-])", basename)
    if match is None:
        return None
    return int(match.group(1))


def locate_run_snapshot_config_path(run_dir: str, cli_config: str = "") -> str:
    cli_text = str(cli_config).strip()
    if cli_text != "":
        if os.path.isabs(cli_text):
            config_candidates = [os.path.abspath(cli_text)]
        else:
            config_candidates = [
                os.path.abspath(cli_text),
                os.path.abspath(os.path.join(run_dir, cli_text)),
            ]
        for config_candidate in config_candidates:
            if os.path.isfile(config_candidate):
                return config_candidate
        if len(config_candidates) == 1:
            raise FileNotFoundError(f"Config file not found: {config_candidates[0]}")
        raise FileNotFoundError(
            "Config file not found. "
            f"Tried current working directory path {config_candidates[0]} "
            f"and run_dir-relative path {config_candidates[1]}."
        )

    candidates = [
        os.path.join(run_dir, "workflow_config.yaml"),
        os.path.join(run_dir, "asthma_config.py"),
    ]
    candidates.extend(
        sorted(
            os.path.join(run_dir, file_name)
            for file_name in os.listdir(run_dir)
            if file_name.endswith("_config.py")
        )
    )
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized

    raise FileNotFoundError(
        "No config snapshot found in run_dir. "
        "Expected run_dir/workflow_config.yaml, run_dir/asthma_config.py, or another *_config.py."
    )


def build_resolution_base_dirs(config_path: str, run_dir: str) -> List[str]:
    candidates = [
        os.path.dirname(os.path.abspath(config_path)),
        os.path.abspath(str(run_dir)),
        os.path.join(PROJECT_ROOT, "configs"),
        PROJECT_ROOT,
    ]
    ordered: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def resolve_path_with_base_dirs(
    base_dirs: Sequence[str],
    path_value: str,
    *,
    prefer_existing: bool,
) -> str:
    text = str(path_value).strip()
    if text == "":
        return text
    if os.path.isabs(text):
        return os.path.abspath(text)

    candidates: List[str] = []
    for base_dir in base_dirs:
        candidates.append(os.path.abspath(os.path.join(str(base_dir), text)))

    if prefer_existing:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    if len(candidates) == 0:
        return os.path.abspath(text)
    return candidates[0]


def resolve_explicit_runtime_path(
    path_value: str,
    *,
    prefer_existing: bool,
    base_dirs: Sequence[str],
) -> str:
    text = str(path_value).strip()
    if text == "":
        return text
    if os.path.isabs(text):
        return os.path.abspath(text)

    cwd_candidate = os.path.abspath(text)
    if not prefer_existing or os.path.exists(cwd_candidate):
        return cwd_candidate

    return resolve_path_with_base_dirs(base_dirs, text, prefer_existing=prefer_existing)


def resolve_device(gpu_index: Optional[int], config: SimpleNamespace) -> torch.device:
    if gpu_index is not None:
        requested = int(gpu_index)
        if requested < 0:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --gpu >= 0 was provided.")
        if requested >= int(torch.cuda.device_count()):
            raise ValueError(
                f"Invalid --gpu index {requested} for {int(torch.cuda.device_count())} visible CUDA devices."
            )
        device = torch.device(f"cuda:{requested}")
        torch.cuda.set_device(device)
        return device

    if str(getattr(config, "device", "cpu")).lower() == "cuda" and torch.cuda.is_available():
        cuda_index = int(getattr(config, "cuda_device_index", 0))
        if cuda_index < 0 or cuda_index >= int(torch.cuda.device_count()):
            raise ValueError(
                f"Invalid cuda_device_index={cuda_index} for {int(torch.cuda.device_count())} visible CUDA devices."
            )
        device = torch.device(f"cuda:{cuda_index}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def load_run_config_payload(run_dir: str) -> Optional[Dict[str, object]]:
    run_config_path = os.path.join(str(run_dir), "run_config.json")
    if not os.path.isfile(run_config_path):
        return None
    try:
        with open(run_config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        raise RuntimeError(f"Failed to read run_config.json from {run_config_path}: {error}") from error


def merge_run_config_into_config(config: SimpleNamespace, run_config_payload: Dict[str, object]) -> None:
    # Prefer resolved paths from train_new (these are already absolute).
    resolved_paths = run_config_payload.get("resolved_paths", {})
    if isinstance(resolved_paths, dict):
        for key, value in resolved_paths.items():
            if value is None:
                continue
            value_text = str(value).strip()
            if value_text == "":
                continue
            setattr(config, key, value_text)

    # Fill in missing scalar config values (dataset, split_number, etc.) from train_new.
    payload_config = run_config_payload.get("config", {})
    if isinstance(payload_config, dict):
        for key, value in payload_config.items():
            # The run_dir is authoritative for these keys.
            if key in {"dataset", "split_number", "evaluation", "patient_states_to_run"}:
                setattr(config, key, value)
                continue
            if not hasattr(config, key):
                setattr(config, key, value)
                continue
            current = getattr(config, key)
            if current is None or (isinstance(current, str) and current.strip() == ""):
                setattr(config, key, value)



def ensure_binary_label_lists(config: SimpleNamespace) -> None:
    if not hasattr(config, "binary_positive_labels"):
        single = getattr(config, "binary_positive_label", None)
        if single is not None and str(single).strip() != "":
            setattr(config, "binary_positive_labels", [str(single)])

    if not hasattr(config, "binary_negative_labels"):
        single = getattr(config, "binary_negative_label", None)
        if single is not None and str(single).strip() != "":
            setattr(config, "binary_negative_labels", [str(single)])


def _infer_patient_state_for_adata(config: SimpleNamespace) -> str:
    for key in ["patient_state", "patient_states_to_run", "patient_states"]:
        value = getattr(config, key, None)
        if value is None:
            continue
        if isinstance(value, str):
            tokens = [t for t in re.split(r"[,\s]+", value) if t.strip() != ""]
            if tokens:
                return tokens[0]
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return str(value[0])
    return "baseline"


def resolve_adata_path(base_dirs: Sequence[str], config: SimpleNamespace) -> str:
    adata_path_raw = getattr(config, "adata_path", None)
    if adata_path_raw is not None and str(adata_path_raw).strip() != "":
        return resolve_path_with_base_dirs(base_dirs, str(adata_path_raw), prefer_existing=True)

    adata_directory_raw = getattr(config, "adata_directory", None)
    if adata_directory_raw is None or str(adata_directory_raw).strip() == "":
        raise ValueError("Either config.adata_path or config.adata_directory must be set to load the dataset.")
    adata_directory = resolve_path_with_base_dirs(base_dirs, str(adata_directory_raw), prefer_existing=True)

    dataset = str(getattr(config, "dataset", "")).strip()
    if dataset == "":
        raise ValueError("config.dataset must be set to resolve adata_path from adata_directory.")

    patient_state = _infer_patient_state_for_adata(config)

    candidates = [
        os.path.join(adata_directory, dataset, f"{dataset}_data.h5ad"),
        os.path.join(adata_directory, f"{dataset}_data.h5ad"),
        os.path.join(adata_directory, dataset, f"{dataset}_{patient_state}_data.h5ad"),
        os.path.join(adata_directory, f"{dataset}_{patient_state}_data.h5ad"),
        os.path.join(adata_directory, dataset, patient_state, f"{dataset}_{patient_state}_data.h5ad"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the h5ad file. Tried:\n  - " + "\n  - ".join(candidates)
    )


def resolve_obs_column_name(adata, configured_name: str, config_key: str, required: bool) -> Optional[str]:
    if configured_name is None or str(configured_name).strip() == "":
        if required:
            raise ValueError(f"Config key '{config_key}' must be set.")
        return None

    configured = str(configured_name)
    if configured in adata.obs.columns:
        return configured

    lowered = configured.lower()
    matches = [str(column) for column in adata.obs.columns if str(column).lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous obs column for {config_key}='{configured}': {matches}")
    if required:
        raise ValueError(
            f"Missing obs column for {config_key}='{configured}'. "
            f"Available: {list(adata.obs.columns)}"
        )
    return None


def resolve_qkv_projection_choice(config: SimpleNamespace) -> str:
    qkv_config = getattr(config, "QKV", None)
    raw_choice = None
    if isinstance(qkv_config, dict):
        raw_choice = qkv_config.get("choice", None)
    if raw_choice is None:
        raw_choice = getattr(config, "qkv_choice", getattr(config, "QKV_choice", "global"))

    choice = str(raw_choice).strip().lower()
    if choice in {
        "cell",
        "cellspecific",
        "cell-specific",
        "cell_specific",
        "celltypespecific",
        "celltype-specific",
        "celltype_specific",
    }:
        choice = "celltype_specific"
    if choice not in {"global", "celltype_specific"}:
        raise ValueError("QKV choice must be one of {'global','celltype_specific'}.")
    return choice


def normalize_cell_encoder_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"graph_gat", "gat", "graph_attention", "graphattention"}:
        return "graph_gat"
    if normalized in {"transformer_conv", "transformerconv", "tranformer_conv", "tranformerconv"}:
        return "transformer_conv"
    raise ValueError(
        "cell_encoder_name must be one of {'graph_gat','transformer_conv'} "
        f"(got {str(name)!r})."
    )


def resolve_cell_encoder_spec(config: SimpleNamespace, num_nodes: int) -> Dict[str, object]:
    encoder_name = normalize_cell_encoder_name(str(getattr(config, "cell_encoder_name", "graph_gat")))
    if int(num_nodes) <= 0:
        raise ValueError("num_nodes must be positive.")

    if encoder_name == "graph_gat":
        hidden_dimension = int(config.gnn_hidden_dim)
        number_of_layers = int(config.gnn_num_layers)
        number_of_heads = int(config.gnn_num_heads)
        dropout_probability = float(config.gnn_dropout)
        graph_readout = str(config.gnn_graph_readout)
        encoder_class = GraphCellEncoder
        cell_embedding_dimension = int(hidden_dimension)
    else:
        hidden_dimension = int(config.transformer_conv_feat_dim)
        number_of_layers = int(config.transformer_conv_num_layers)
        number_of_heads = int(config.transformer_conv_heads)
        dropout_probability = float(config.transformer_conv_dropout)
        graph_readout = str(config.transformer_conv_graph_readout)
        encoder_class = TransformerConvCellEncoder
        readout_normalized = str(graph_readout).strip().lower()
        if readout_normalized in {"flat", "concat"}:
            readout_normalized = "flatten"
        if readout_normalized == "flatten":
            cell_embedding_dimension = int(num_nodes) * int(hidden_dimension)
        else:
            cell_embedding_dimension = int(hidden_dimension)

    return {
        "name": str(encoder_name),
        "encoder_class": encoder_class,
        "hidden_dimension": int(hidden_dimension),
        "number_of_layers": int(number_of_layers),
        "number_of_heads": int(number_of_heads),
        "dropout_probability": float(dropout_probability),
        "graph_readout": str(graph_readout),
        "cell_embedding_dimension": int(cell_embedding_dimension),
    }


def resolve_prior_apply_flags(config: SimpleNamespace) -> Tuple[bool, bool]:
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


def load_prior_embeddings_by_view(
    genes: Sequence[str],
    view_names: Sequence[str],
    config: SimpleNamespace,
) -> Dict[str, torch.Tensor]:
    configured_paths = _normalized_configured_embedding_paths(config)

    loaded: Dict[str, torch.Tensor] = {}
    for view_name in [str(name).strip() for name in list(view_names) if str(name).strip()]:
        source_dict = load_protein_embedding_dict(view_name, embedding_paths=configured_paths)
        matrix, _ = _build_embedding_matrix_with_missing_fallback(
            genes=genes,
            embedding_dict=source_dict,
            source_name=str(view_name),
        )
        matrix = matrix.float().contiguous()
        loaded[str(view_name)] = matrix
    return loaded


def _build_embedding_matrix_with_missing_fallback(
    genes: Sequence[str],
    embedding_dict: Dict[str, np.ndarray],
    *,
    source_name: str,
) -> Tuple[torch.Tensor, List[str]]:
    missing = [str(gene) for gene in genes if str(gene) not in embedding_dict]
    if len(embedding_dict) == 0:
        raise ValueError(
            f"Protein embedding source '{source_name}' is empty; cannot infer embedding dimension."
        )
    if len(missing) > 0:
        raise ValueError(
            f"Protein embedding source '{source_name}' is missing {len(missing)} required genes "
            f"(e.g. {missing[:5]}). Align the embedding gene identifiers with the selected gene set."
        )
    return build_embedding_matrix(genes, embedding_dict), []


def normalize_embedding_dict(embedding_object) -> Dict[str, np.ndarray]:
    if not isinstance(embedding_object, dict):
        raise ValueError(f"Embedding object must be dict, got {type(embedding_object)}")
    normalized: Dict[str, np.ndarray] = {}
    for key, value in embedding_object.items():
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        normalized[str(key).upper()] = vector
    if len(normalized) == 0:
        raise ValueError("Embedding dictionary is empty.")
    return normalized


def resolve_preselection_root(config: SimpleNamespace) -> str:
    dataset_root = os.path.join(PROJECT_ROOT, "data", str(config.dataset))
    mode = getattr(config, "split_train_only_preselection", None)

    split_subdir = str(getattr(config, "split_preselection_subdir", "split_preselection"))
    split_index = int(getattr(config, "split_number", 0))
    explicit_global_root_candidates: List[str] = []
    explicit_split_root_candidates: List[str] = []

    for attr_name in ("preselection_root", "preselection_output_root"):
        explicit_root = str(getattr(config, attr_name, "")).strip()
        if explicit_root == "":
            continue
        explicit_global_root_candidates.append(explicit_root)
        explicit_split_root_candidates.extend(
            [
                os.path.join(explicit_root, f"split_{split_index}"),
                os.path.join(explicit_root, f"split_idx_{split_index}"),
                explicit_root,
            ]
        )

    legacy_split_root_candidates = [
        os.path.join(dataset_root, "preselection", f"split_{split_index}"),
        os.path.join(dataset_root, split_subdir, f"split_idx_{split_index}"),
        os.path.join(dataset_root, split_subdir, f"split_{split_index}"),
    ]
    split_root_candidates = explicit_split_root_candidates + legacy_split_root_candidates

    explicit_split_root = ""
    for candidate in explicit_split_root_candidates:
        if os.path.isdir(candidate):
            explicit_split_root = str(candidate)
            break

    legacy_split_root = ""
    for candidate in legacy_split_root_candidates:
        if os.path.isdir(candidate):
            legacy_split_root = str(candidate)
            break

    split_root = explicit_split_root or legacy_split_root

    if mode is False:
        for candidate in explicit_global_root_candidates:
            if os.path.isdir(candidate):
                return str(candidate)
        return explicit_global_root_candidates[0] if explicit_global_root_candidates else dataset_root

    if mode is True:
        if split_root == "":
            raise FileNotFoundError(
                "split-specific preselection directory not found: "
                f"{split_root_candidates} (set split_train_only_preselection=False to use global files)."
            )
        return split_root

    # Auto mode (default): prefer split-specific preselection when available.
    if split_root != "":
        return split_root
    return dataset_root


def _deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def _canonical_embedding_source_name(source_name: str) -> str:
    normalized = str(source_name).strip().lower()
    if normalized in {"esm3"}:
        return "ESM3"
    if normalized in {"genept", "gpt"}:
        return "GPT"
    if normalized in {"node2vec", "n2v"}:
        return "node2vec"
    return str(source_name).strip()


def _normalized_configured_embedding_paths(config: SimpleNamespace) -> Optional[Dict[str, str]]:
    for attribute_name in ("protein_embedding_paths", "gene_embedding_views", "embedding_views"):
        configured_paths_raw = getattr(config, attribute_name, None)
        if isinstance(configured_paths_raw, argparse.Namespace):
            configured_paths_raw = vars(configured_paths_raw)
        if not isinstance(configured_paths_raw, dict) or len(configured_paths_raw) <= 0:
            continue
        return {str(key): str(value) for key, value in configured_paths_raw.items()}
    return None


def _resolve_embedding_view_sources(config: SimpleNamespace) -> List[str]:
    configured_paths = _normalized_configured_embedding_paths(config)
    candidate_sources: List[str] = []

    for attribute_name in ("prior_view_sources", "protein_embedding_sources"):
        raw_values = getattr(config, attribute_name, [])
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        for raw_value in list(raw_values or []):
            text = str(raw_value).strip()
            if text != "":
                candidate_sources.append(text)

    if isinstance(configured_paths, dict):
        candidate_sources.extend(str(name) for name in configured_paths.keys())

    return _deduplicate_preserve_order(
        [_canonical_embedding_source_name(source_name) for source_name in candidate_sources]
    )


def _select_genes_with_embedding_coverage(
    ranked_genes: Sequence[str],
    *,
    target_k: int,
    required_sources: Sequence[str],
    embedding_paths: Optional[Dict[str, str]],
) -> List[str]:
    required_sources_clean = _deduplicate_preserve_order(
        [
            _canonical_embedding_source_name(str(source).strip())
            for source in required_sources
            if str(source).strip() != ""
        ]
    )
    if len(required_sources_clean) == 0:
        return [str(gene).upper() for gene in ranked_genes][:target_k]

    coverage_by_source: Dict[str, set] = {}
    for source_name in required_sources_clean:
        source_dict = load_protein_embedding_dict(source_name, embedding_paths=embedding_paths)
        coverage_by_source[source_name] = set(source_dict.keys())

    selected: List[str] = []
    skipped_by_source: Dict[str, int] = {source_name: 0 for source_name in required_sources_clean}
    scanned_count = 0
    for gene_name in [str(gene).upper() for gene in ranked_genes]:
        scanned_count += 1
        missing_sources = [
            source_name
            for source_name in required_sources_clean
            if gene_name not in coverage_by_source[source_name]
        ]
        if len(missing_sources) > 0:
            for source_name in missing_sources:
                skipped_by_source[source_name] += 1
            continue
        selected.append(gene_name)
        if len(selected) >= int(target_k):
            break

    if len(selected) < int(target_k):
        print(
            "[GeneSelect][warn] "
            f"could not reach target_k={int(target_k)} with required_sources={required_sources_clean}. "
            f"selected_k={len(selected)} scanned={scanned_count}/{len(ranked_genes)} "
            f"skipped_by_source={skipped_by_source}",
            flush=True,
        )
    else:
        print(
            "[GeneSelect] "
            f"selected_k={len(selected)} scanned={scanned_count}/{len(ranked_genes)} "
            f"required_sources={required_sources_clean} skipped_by_source={skipped_by_source}",
            flush=True,
        )

    return selected[: int(target_k)]


def load_np_max_genes(
    dataset: str,
    k: int,
    preselection_root: Optional[str] = None,
    required_embedding_sources: Optional[Sequence[str]] = None,
    embedding_paths: Optional[Dict[str, str]] = None,
) -> List[str]:
    base_root = str(preselection_root) if preselection_root is not None else os.path.join(PROJECT_ROOT, "data", str(dataset))
    np_candidates = [
        os.path.join(base_root, "NP", "NP_max.tsv"),
        os.path.join(base_root, "NP_genes", "NP_max.tsv"),
    ]
    np_path = ""
    for candidate in np_candidates:
        if os.path.isfile(candidate):
            np_path = str(candidate)
            break
    if np_path == "":
        raise FileNotFoundError(f"NP gene file not found. Tried: {np_candidates}")
    table = pd.read_csv(np_path, sep="\t")
    if "gene" not in table.columns:
        raise ValueError(f"Expected 'gene' column in {np_path}")
    genes = table["gene"].astype(str).str.upper().tolist()
    if len(genes) == 0:
        raise ValueError(f"No genes found in {np_path}")
    target_k = int(k)
    if target_k <= 0:
        return genes
    if len(genes) < target_k:
        return genes

    required_sources = list(required_embedding_sources or [])
    if len(required_sources) == 0:
        return genes[:target_k]
    selected = _select_genes_with_embedding_coverage(
        ranked_genes=genes,
        target_k=target_k,
        required_sources=required_sources,
        embedding_paths=embedding_paths,
    )
    if len(selected) == 0:
        raise ValueError(
            "No genes remained after embedding-coverage filtering. "
            f"required_sources={required_sources}"
        )
    return selected


def load_cell_encoder_state_dict_with_edge_buffer_fallback(
    cell_encoder: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    edge_buffer_keys = ["edge_index_ppi", "edge_index_knn", "edge_index"]
    current_state = cell_encoder.state_dict()
    for key in edge_buffer_keys:
        loaded_tensor = state_dict.get(key, None)
        current_tensor = current_state.get(key, None)
        if not isinstance(loaded_tensor, torch.Tensor) or not isinstance(current_tensor, torch.Tensor):
            continue
        if tuple(loaded_tensor.shape) == tuple(current_tensor.shape):
            continue
        print(
            "[Checkpoint][warn] "
            f"buffer '{key}' shape mismatch current={tuple(current_tensor.shape)} "
            f"checkpoint={tuple(loaded_tensor.shape)}; using checkpoint buffer.",
            flush=True,
        )
        if key in getattr(cell_encoder, "_buffers", {}):
            cell_encoder._buffers[key] = loaded_tensor.to(device=device, dtype=torch.long).contiguous()
    cell_encoder.load_state_dict(state_dict)


def resolve_deg_directory(dataset: str, deg_dir: Optional[str] = None) -> str:
    if deg_dir is not None and str(deg_dir).strip() != "":
        candidate = str(deg_dir)
        if os.path.isdir(candidate):
            return candidate
        parent_dir = os.path.dirname(candidate)
        if os.path.basename(candidate) == "DEG_genes":
            alt = os.path.join(parent_dir, "DEG")
            if os.path.isdir(alt):
                return alt
        if os.path.basename(candidate) == "DEG":
            alt = os.path.join(parent_dir, "DEG_genes")
            if os.path.isdir(alt):
                return alt

    dataset_root = os.path.join(PROJECT_ROOT, "data", str(dataset))
    for candidate in [os.path.join(dataset_root, "DEG"), os.path.join(dataset_root, "DEG_genes")]:
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(dataset_root, "DEG")


def build_protein_embedding_matrix(
    genes: Sequence[str],
    config: SimpleNamespace,
) -> Tuple[torch.Tensor, str, Optional[torch.Tensor]]:
    choice = str(config.protein_embedding_choice).strip().lower()
    if choice == "none":
        dim = int(getattr(config, "gnn_hidden_dim", 0))
        if dim <= 0:
            raise ValueError("gnn_hidden_dim must be set when protein_embedding_choice='none'.")
        return torch.zeros((len(genes), dim), dtype=torch.float32), "zeros", None

    if choice not in {"concat", "mean"}:
        raise ValueError(
            "protein_embedding_choice must be one of {'concat','mean','none'}. "
            f"Got: {str(getattr(config, 'protein_embedding_choice', choice))!r}"
        )

    configured_paths = _normalized_configured_embedding_paths(config)
    sources = _resolve_embedding_view_sources(config)
    if len(sources) == 0:
        raise ValueError(
            "At least one embedding view is required. "
            "Set gene_embedding_views/protein_embedding_paths or prior_view_sources."
        )
    source_matrices: List[torch.Tensor] = []
    for source_name in sources:
        source_dict = load_protein_embedding_dict(
            source_name,
            embedding_paths=configured_paths,
        )
        source_matrix, _ = _build_embedding_matrix_with_missing_fallback(
            genes=genes,
            embedding_dict=source_dict,
            source_name=str(source_name),
        )
        source_matrices.append(source_matrix)

    if choice == "concat":
        merged_matrix = torch.cat(source_matrices, dim=1).contiguous()
    else:
        source_dims = {int(matrix.shape[1]) for matrix in source_matrices}
        if len(source_dims) != 1:
            raise ValueError(
                "protein_embedding_choice='mean' requires equal source dimensions. "
                f"Got dims={sorted(source_dims)} for sources={sources}."
            )
        merged_matrix = torch.stack(source_matrices, dim=0).mean(dim=0).contiguous()

    gnn_hidden_dim = int(getattr(config, "gnn_hidden_dim", 0))
    if gnn_hidden_dim <= 0:
        raise ValueError(
            "gnn_hidden_dim must be set when protein_embedding_choice is 'concat' or 'mean'."
        )
    fixed_placeholder = torch.zeros((len(genes), gnn_hidden_dim), dtype=torch.float32)
    source_description = f"learnable_{choice}[{','.join(sources)}]"
    return fixed_placeholder, source_description, merged_matrix


def build_celltype_regulation_onehot(
    dataset: str,
    genes: Sequence[str],
    celltype_names: Sequence[str],
    deg_dir: Optional[str] = None,
) -> torch.Tensor:
    deg_dir = resolve_deg_directory(dataset=str(dataset), deg_dir=deg_dir)
    gene_keys = [str(gene).upper() for gene in genes]
    regulation = np.zeros((len(celltype_names), len(gene_keys), 3), dtype=np.float32)
    regulation[:, :, 2] = 1.0

    for celltype_index, celltype_name in enumerate(celltype_names):
        safe_name = sanitize_filename_component(str(celltype_name))
        metrics_path = os.path.join(deg_dir, f"DEG_metrics_{safe_name}_pass.tsv")
        if not os.path.isfile(metrics_path):
            continue
        table = pd.read_csv(metrics_path, sep="\t")
        if "gene" not in table.columns or "regulation" not in table.columns:
            raise ValueError(f"Expected columns 'gene' and 'regulation' in {metrics_path}")

        table["gene"] = table["gene"].astype(str).str.upper()
        reg_map = dict(zip(table["gene"], table["regulation"]))
        for gene_index, gene in enumerate(gene_keys):
            value = reg_map.get(gene, 0.0)
            try:
                value_float = float(value)
            except Exception:
                value_float = 0.0
            if value_float > 0:
                regulation[celltype_index, gene_index] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            elif value_float < 0:
                regulation[celltype_index, gene_index] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                regulation[celltype_index, gene_index] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return torch.tensor(regulation, dtype=torch.float32)


def build_global_deg_zscore_vector(
    dataset: str,
    genes: Sequence[str],
    deg_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, str, int]:
    deg_dir = resolve_deg_directory(dataset=str(dataset), deg_dir=deg_dir)
    zscore_path = os.path.join(deg_dir, "DEG_zscore_global.tsv")
    gene_keys = [str(gene).upper() for gene in genes]
    if not os.path.isfile(zscore_path):
        return torch.zeros((len(gene_keys),), dtype=torch.float32), zscore_path, int(len(gene_keys))

    table = pd.read_csv(zscore_path, sep="\t")
    if "gene" not in table.columns:
        raise ValueError(f"Expected column 'gene' in {zscore_path}")
    score_col = "zscore" if "zscore" in table.columns else ("score" if "score" in table.columns else None)
    if score_col is None:
        raise ValueError(f"Expected column 'zscore' (or 'score') in {zscore_path}")
    table["gene"] = table["gene"].astype(str).str.upper()
    score_map = dict(zip(table["gene"], pd.to_numeric(table[score_col], errors="coerce").fillna(0.0).astype(float)))

    values = []
    missing = 0
    for gene in gene_keys:
        if gene in score_map:
            values.append(float(score_map[gene]))
        else:
            values.append(0.0)
            missing += 1
    return torch.tensor(values, dtype=torch.float32), zscore_path, int(missing)


def get_dense_rows(matrix, row_indices: np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix[row_indices].toarray().astype(np.float32)
    return np.asarray(matrix[row_indices], dtype=np.float32)


def map_values_with_mapping(values: Sequence[str], mapping: Dict[str, int], name: str) -> np.ndarray:
    output = np.zeros((len(values),), dtype=np.int64)
    missing: List[str] = []
    for idx, value in enumerate(values):
        key = str(value)
        if key not in mapping:
            missing.append(key)
        else:
            output[idx] = int(mapping[key])
    if len(missing) > 0:
        uniq = sorted(set(missing))
        raise ValueError(f"{name}: {len(uniq)} unseen categories not in mapping. Examples: {uniq[:5]}")
    return output


def _sample_without_replacement(num_items: int, take: int, rng: np.random.Generator) -> np.ndarray:
    if num_items <= 0 or take <= 0:
        return np.zeros((0,), dtype=np.int64)
    if take >= num_items:
        return np.arange(num_items, dtype=np.int64)
    return np.sort(rng.choice(num_items, size=take, replace=False).astype(np.int64))


def _allocate_proportional_quotas(type_counts: Dict[int, int], target_size: int) -> Dict[int, int]:
    if target_size <= 0 or len(type_counts) == 0:
        return {key: 0 for key in type_counts}

    total = float(sum(int(v) for v in type_counts.values()))
    if total <= 0:
        return {key: 0 for key in type_counts}

    raw = {key: target_size * (float(count) / total) for key, count in type_counts.items()}
    quota = {key: min(int(np.floor(raw[key])), int(type_counts[key])) for key in type_counts}
    assigned = int(sum(quota.values()))
    remainder = max(0, int(target_size - assigned))

    if remainder > 0:
        keys = sorted(type_counts.keys(), key=lambda k: (-(raw[k] - np.floor(raw[k])), -int(type_counts[k]), int(k)))
        for key in keys:
            if remainder <= 0:
                break
            avail = int(type_counts[key]) - int(quota[key])
            if avail <= 0:
                continue
            take = min(avail, remainder)
            quota[key] += int(take)
            remainder -= int(take)
    return quota


def normalize_sample_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"uniform"}:
        return "random"
    if normalized not in {"random", "celltype", "proportional"}:
        raise ValueError("sample mode must be one of {'random','celltype','proportional','uniform'}.")
    return normalized


def sample_patient_bag_indices(
    patient_cell_indices: np.ndarray,
    patient_celltypes: np.ndarray,
    bag_size: int,
    sample_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    num_cells = int(patient_cell_indices.shape[0])
    if num_cells <= 0:
        return np.zeros((0,), dtype=np.int64)

    target_size = int(bag_size)
    if target_size <= 0:
        target_size = num_cells
    target_size = min(target_size, num_cells)
    if target_size <= 0:
        return np.zeros((0,), dtype=np.int64)

    mode = normalize_sample_mode(sample_mode)
    if mode == "random":
        local = _sample_without_replacement(num_cells, target_size, rng)
        return patient_cell_indices[local]

    unique_types = sorted({int(v) for v in patient_celltypes.tolist()})
    if len(unique_types) == 0:
        local = _sample_without_replacement(num_cells, target_size, rng)
        return patient_cell_indices[local]

    if mode == "celltype":
        picked = int(unique_types[int(rng.integers(0, len(unique_types)))])
        pos = np.where(patient_celltypes == picked)[0].astype(np.int64)
        local = _sample_without_replacement(int(pos.shape[0]), target_size, rng)
        return patient_cell_indices[pos[local]]

    type_to_pos: Dict[int, np.ndarray] = {}
    type_counts: Dict[int, int] = {}
    for t in unique_types:
        pos = np.where(patient_celltypes == int(t))[0].astype(np.int64)
        type_to_pos[int(t)] = pos
        type_counts[int(t)] = int(pos.shape[0])
    quotas = _allocate_proportional_quotas(type_counts, target_size)

    selected_parts: List[np.ndarray] = []
    selected_mask = np.zeros((num_cells,), dtype=bool)
    for t in unique_types:
        q = int(quotas.get(int(t), 0))
        if q <= 0:
            continue
        pos = type_to_pos[int(t)]
        local = _sample_without_replacement(int(pos.shape[0]), q, rng)
        picked = pos[local]
        selected_parts.append(picked)
        selected_mask[picked] = True

    selected = np.concatenate(selected_parts, axis=0) if selected_parts else np.zeros((0,), dtype=np.int64)
    if int(selected.shape[0]) < target_size:
        remaining = np.where(~selected_mask)[0].astype(np.int64)
        need = int(target_size - int(selected.shape[0]))
        if remaining.shape[0] > 0:
            extra = _sample_without_replacement(int(remaining.shape[0]), need, rng)
            selected = np.concatenate([selected, remaining[extra]], axis=0)
    selected = np.sort(np.unique(selected.astype(np.int64)))
    if int(selected.shape[0]) > target_size:
        local = _sample_without_replacement(int(selected.shape[0]), target_size, rng)
        selected = selected[local]
    return patient_cell_indices[selected]


def compute_binary_metrics(labels: Sequence[int], probabilities_pos: Sequence[float]) -> Dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    prob_np = np.asarray(probabilities_pos, dtype=np.float64)
    if labels_np.size == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "logloss": float("nan"),
        }
    pred = (prob_np >= 0.5).astype(np.int64)
    prob_mat = np.stack([1.0 - prob_np, prob_np], axis=1)
    precision, recall, f1 = precision_recall_f1_weighted(labels_np, pred)
    eps = 1e-12
    prob_safe = np.clip(prob_np, eps, 1.0 - eps)
    logloss = float(
        np.mean(
            -(labels_np.astype(np.float64) * np.log(prob_safe) + (1.0 - labels_np.astype(np.float64)) * np.log(1.0 - prob_safe))
        )
    )
    return {
        "accuracy": accuracy(labels_np, pred),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc_weighted(labels_np, prob_mat),
        "auprc": auprc_weighted(labels_np, prob_mat),
        "logloss": logloss,
    }


def select_metric(metrics: Dict[str, float], metric_name: str) -> float:
    key = str(metric_name).strip().lower()
    if key == "loss":
        key = "logloss"
    if key not in {"accuracy", "precision", "recall", "f1", "auroc", "auprc", "logloss"}:
        raise ValueError(
            "biomarker_metric must be one of "
            "{'accuracy','precision','recall','f1','auroc','auprc','logloss'}"
        )
    value = float(metrics.get(key, float("nan")))
    if np.isnan(value):
        return float("-inf")
    if key == "logloss":
        return -value
    return value


def stable_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if int(a.size) == 0 or int(b.size) == 0:
        return float("nan")
    if int(a.size) != int(b.size):
        raise ValueError("Spearman inputs must have the same length.")
    try:
        corr = spearmanr(a, b).correlation
    except Exception:
        return float("nan")
    return float(corr) if corr is not None else float("nan")


def jaccard_index(a: Sequence[object], b: Sequence[object]) -> float:
    set_a = set(str(v) for v in a)
    set_b = set(str(v) for v in b)
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return float(len(set_a & set_b)) / float(len(union))


def aggregate_patient_probabilities(
    bag_defs: Sequence[BagDefinition],
    bag_probabilities: Dict[int, float],
    reduction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(reduction).strip().lower()
    if mode not in {"mean", "median"}:
        raise ValueError("patient probability reduction must be one of {'mean','median'}.")

    grouped_prob: Dict[int, List[float]] = defaultdict(list)
    grouped_label: Dict[int, List[int]] = defaultdict(list)
    for bag in bag_defs:
        grouped_prob[int(bag.patient_index)].append(float(bag_probabilities[int(bag.bag_id)]))
        grouped_label[int(bag.patient_index)].append(int(bag.patient_label))

    patients = np.asarray(sorted(grouped_prob.keys()), dtype=np.int64)
    labels = np.zeros((patients.shape[0],), dtype=np.int64)
    probs = np.zeros((patients.shape[0],), dtype=np.float64)
    for idx, patient in enumerate(patients.tolist()):
        label_values = np.asarray(grouped_label[int(patient)], dtype=np.int64)
        values, counts = np.unique(label_values, return_counts=True)
        labels[idx] = int(values[np.argmax(counts)])

        prob_values = np.asarray(grouped_prob[int(patient)], dtype=np.float64)
        probs[idx] = float(np.median(prob_values)) if mode == "median" else float(np.mean(prob_values))
    return patients, labels, probs


class PatientClassifier(nn.Module):
    def __init__(self, input_dimension: int, config: SimpleNamespace) -> None:
        super().__init__()
        if str(config.classifier_name).lower() == "linear":
            self.layers = nn.Linear(int(input_dimension), 1)
        elif str(config.classifier_name).lower() == "mlp":
            self.layers = nn.Sequential(
                nn.Linear(int(input_dimension), int(config.classifier_hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(config.classifier_dropout)),
                nn.Linear(int(config.classifier_hidden_dim), 1),
            )
        else:
            raise ValueError("classifier_name must be one of {'linear','mlp'}.")

    def forward(self, patient_embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(patient_embeddings).view(-1)


def forward_bag_from_expression(
    cell_encoder: GraphCellEncoder,
    mil_aggregator: PatientMILAggregator,
    classifier: PatientClassifier,
    expression: np.ndarray,
    celltypes: np.ndarray,
    treatments: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    expr_t = torch.tensor(expression, dtype=torch.float32, device=device)
    ct_t = torch.tensor(celltypes, dtype=torch.long, device=device)
    tr_t = torch.tensor(treatments, dtype=torch.long, device=device)

    encoder_out = cell_encoder(
        expression_values=expr_t,
        celltype_index=ct_t,
        treatment_index=tr_t,
        return_node_outputs=False,
        return_attention=False,
    )
    cell_embeddings = encoder_out["cell_embeddings"]

    bag_index = torch.zeros((int(cell_embeddings.shape[0]),), dtype=torch.long, device=device)
    mil_out = mil_aggregator(
        cell_embeddings=cell_embeddings,
        celltype_index=ct_t,
        bag_index=bag_index,
        num_bags=1,
    )
    logits = classifier(mil_out["patient_embeddings"])  # [1]
    prob = torch.sigmoid(logits)[0].detach().cpu().item()
    return float(prob), cell_embeddings.detach().cpu().numpy().astype(np.float32)


def forward_bag_from_embeddings(
    mil_aggregator: PatientMILAggregator,
    classifier: PatientClassifier,
    embeddings: np.ndarray,
    celltypes: np.ndarray,
    device: torch.device,
) -> float:
    emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    ct_t = torch.tensor(celltypes, dtype=torch.long, device=device)
    bag_index = torch.zeros((int(emb_t.shape[0]),), dtype=torch.long, device=device)
    mil_out = mil_aggregator(
        cell_embeddings=emb_t,
        celltype_index=ct_t,
        bag_index=bag_index,
        num_bags=1,
    )
    logits = classifier(mil_out["patient_embeddings"])  # [1]
    prob = torch.sigmoid(logits)[0].detach().cpu().item()
    return float(prob)


def sample_rows_with_replacement(
    matrix: np.ndarray,
    target_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_rows <= 0:
        return np.zeros((0, matrix.shape[1]), dtype=matrix.dtype)
    if matrix.shape[0] <= 0:
        return np.zeros((target_rows, matrix.shape[1]), dtype=matrix.dtype)
    if matrix.shape[0] >= target_rows:
        local = rng.choice(matrix.shape[0], size=target_rows, replace=False)
    else:
        local = rng.choice(matrix.shape[0], size=target_rows, replace=True)
    return matrix[local]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute biomarker scores from trained scGOAT + MIL model.")
    parser.add_argument("--run_dir", required=True, help="Training run directory containing best checkpoint")
    parser.add_argument(
        "--config",
        default="",
        help="Optional config filename under run_dir (default: run_dir/asthma_config.py)",
    )
    parser.add_argument("--out_dir", default="", help="Output directory for biomarker tables")
    parser.add_argument("--pathway_path", default="", help="Pathway dict file (json/pkl/pt)")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA index for inference-time biomarker scoring. Use -1 for CPU.",
    )
    return parser.parse_args()


def run_analysis_phase(
    run_dir: str,
    *,
    config_path: str = "",
    output_dir: str = "",
    pathway_path: str = "",
    gpu_index: int | None = None,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir).strip())
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # Load the snapshot config copied during training from the selected run directory.
    run_config_payload = load_run_config_payload(run_dir)
    dataset_hint: Optional[str] = infer_dataset_hint_from_run_dir(run_dir)
    if isinstance(run_config_payload, dict):
        dataset_value = run_config_payload.get("config", {}).get("dataset", None)
        if dataset_value is not None and str(dataset_value).strip() != "":
            dataset_hint = str(dataset_value).strip()

    resolved_config_path = locate_run_snapshot_config_path(run_dir=run_dir, cli_config=str(config_path))
    config, config_dict = load_config_like_train_from_run_snapshot(
        resolved_config_path,
        dataset_hint=dataset_hint,
    )

    config_resolution_base_dirs = build_resolution_base_dirs(config_path=resolved_config_path, run_dir=run_dir)

    # If the run directory contains run_config.json (train_new output), use it to fill in
    # missing dataset/split info and to override stale relative paths. This does NOT
    # change biomarker scoring logic.
    if isinstance(run_config_payload, dict):
        merge_run_config_into_config(config, run_config_payload)
    if getattr(config, "split_number", None) is None or str(getattr(config, "split_number", "")).strip() == "":
        inferred_split_number = infer_split_number_from_run_dir(run_dir)
        if inferred_split_number is not None:
            setattr(config, "split_number", int(inferred_split_number))

    # Resolve core paths after merging run_config.json (if present).
    if hasattr(config, "experiment_root"):
        config.experiment_root = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            str(getattr(config, "experiment_root")),
            prefer_existing=False,
        )
    if hasattr(config, "ppi_path"):
        config.ppi_path = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            str(getattr(config, "ppi_path")),
            prefer_existing=True,
        )
    if hasattr(config, "splits_directory"):
        config.splits_directory = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            str(getattr(config, "splits_directory")),
            prefer_existing=True,
        )
    if hasattr(config, "adata_directory"):
        config.adata_directory = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            str(getattr(config, "adata_directory")),
            prefer_existing=True,
        )

    # Resolve adata path (supports either explicit adata_path or directory-based convention).
    config.adata_path = resolve_adata_path(config_resolution_base_dirs, config)

    # Ensure binary label lists exist (train_new/biomarker expect plural forms).
    ensure_binary_label_lists(config)

    output_dir_cli = str(output_dir).strip()
    output_dir_cfg = str(getattr(config, "biomarker_output_dir", "")).strip()
    output_dir_value = output_dir_cli or output_dir_cfg or os.path.join(run_dir, "biomarker")
    if output_dir_cli != "":
        output_dir = resolve_explicit_runtime_path(
            output_dir_cli,
            prefer_existing=False,
            base_dirs=config_resolution_base_dirs,
        )
    elif not os.path.isabs(output_dir_value):
        output_dir = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            output_dir_value,
            prefer_existing=False,
        )
    else:
        output_dir = os.path.abspath(output_dir_value)
    os.makedirs(output_dir, exist_ok=True)

    pathway_path_cli = str(pathway_path).strip()
    pathway_path_cfg = str(
        getattr(config, "biomarker_pathway_gene_set_path", getattr(config, "pathway_gene_set_path", ""))
    ).strip()
    pathway_path_value = pathway_path_cli or pathway_path_cfg
    if pathway_path_value == "":
        raise ValueError(
            "Pathway file must be provided via pathway_path or config.biomarker_pathway_gene_set_path"
        )
    if pathway_path_cli != "":
        pathway_path = resolve_explicit_runtime_path(
            pathway_path_cli,
            prefer_existing=True,
            base_dirs=config_resolution_base_dirs,
        )
    elif not os.path.isabs(pathway_path_value):
        pathway_path = resolve_path_with_base_dirs(
            config_resolution_base_dirs,
            pathway_path_value,
            prefer_existing=True,
        )
    else:
        pathway_path = os.path.abspath(pathway_path_value)

    device = resolve_device(gpu_index, config)
    print(f"[Runtime] device={device}", flush=True)

    seed = int(getattr(config, "biomarker_random_seed", getattr(config, "seed", 0)))
    set_global_seed(seed, deterministic=bool(getattr(config, "deterministic_training", False)))
    print(f"Loading adata: {config.adata_path}", flush=True)
    adata = sc.read_h5ad(str(config.adata_path))

    metadata_path = os.path.join(run_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    label_column = resolve_obs_column_name(adata, config.label_column, "label_column", required=True)
    patient_column = resolve_obs_column_name(adata, config.patient_column, "patient_column", required=True)
    celltype_column = resolve_obs_column_name(adata, config.celltype_column, "celltype_column", required=True)
    treatment_column = resolve_obs_column_name(
        adata,
        getattr(config, "treatment_column", None),
        "treatment_column",
        required=False,
    )

    raw_labels = adata.obs[label_column].astype(str).tolist()
    mapped_labels, _ = map_labels(
        label_values=raw_labels,
        binary_positive_labels=list(config.binary_positive_labels),
        binary_negative_labels=list(config.binary_negative_labels),
    )

    label_mapping_meta = metadata.get("label_mapping", None)
    if isinstance(label_mapping_meta, dict) and len(label_mapping_meta) > 0:
        label_mapping = {str(key): int(value) for key, value in label_mapping_meta.items()}
        label_indices = np.array([int(label_mapping[str(v)]) for v in mapped_labels], dtype=np.int64)
    else:
        label_mapping, label_indices = create_category_mapping(mapped_labels)

    celltype_mapping_meta = metadata.get("celltype_mapping", None)
    if isinstance(celltype_mapping_meta, dict) and len(celltype_mapping_meta) > 0:
        celltype_mapping = {str(key): int(value) for key, value in celltype_mapping_meta.items()}
    else:
        celltype_values_for_map = adata.obs[celltype_column].astype(str).tolist()
        celltype_mapping, _ = create_category_mapping(celltype_values_for_map)

    treatment_mapping_meta = metadata.get("treatment_mapping", None)
    if isinstance(treatment_mapping_meta, dict) and len(treatment_mapping_meta) > 0:
        treatment_mapping = {str(key): int(value) for key, value in treatment_mapping_meta.items()}
    else:
        if treatment_column is not None:
            treatment_values_for_map = adata.obs[treatment_column].astype(str).tolist()
        else:
            treatment_values_for_map = ["NA"] * adata.n_obs
        treatment_mapping, _ = create_category_mapping(treatment_values_for_map)

    celltype_values = adata.obs[celltype_column].astype(str).tolist()
    celltype_array = map_values_with_mapping(celltype_values, celltype_mapping, "celltype")

    if treatment_column is not None:
        treatment_values = adata.obs[treatment_column].astype(str).tolist()
    else:
        treatment_values = ["NA"] * adata.n_obs
    treatment_array = map_values_with_mapping(treatment_values, treatment_mapping, "treatment")

    patient_values = adata.obs[patient_column].astype(str).tolist()
    patient_mapping, patient_array = create_category_mapping(patient_values)
    patient_index_to_id = {int(index): str(name) for name, index in patient_mapping.items()}

    prior_view_sources = _resolve_embedding_view_sources(config)
    if len(prior_view_sources) == 0:
        raise ValueError(
            "At least one embedding view is required for model reconstruction. "
            "Set gene_embedding_views/protein_embedding_paths or prior_view_sources."
        )

    configured_embedding_paths = _normalized_configured_embedding_paths(config)

    preselection_root = resolve_preselection_root(config)
    print(
        f"[Preselection] split={int(config.split_number)} root={preselection_root}",
        flush=True,
    )
    genes = load_np_max_genes(
        str(config.dataset),
        int(config.k),
        preselection_root=preselection_root,
        required_embedding_sources=prior_view_sources,
        embedding_paths=configured_embedding_paths,
    )
    print(f"[GeneSelect] selected={len(genes)} genes (target_k={int(config.k)})", flush=True)
    gene_indices = map_genes_to_adata(adata, genes)
    expression_matrix = adata[:, gene_indices].X

    encoder_spec = resolve_cell_encoder_spec(config=config, num_nodes=int(len(genes)))
    config.cell_encoder_name = str(encoder_spec["name"])
    encoder_class = encoder_spec["encoder_class"]
    encoder_hidden_dimension = int(encoder_spec["hidden_dimension"])
    encoder_number_of_layers = int(encoder_spec["number_of_layers"])
    encoder_number_of_heads = int(encoder_spec["number_of_heads"])
    encoder_dropout_probability = float(encoder_spec["dropout_probability"])
    encoder_graph_readout = str(encoder_spec["graph_readout"])
    cell_embedding_dimension = int(encoder_spec["cell_embedding_dimension"])

    prior_absolute_enabled, prior_relational_enabled = resolve_prior_apply_flags(config)
    prior_injection_enabled = bool(prior_absolute_enabled or prior_relational_enabled)
    lambda_edge_bias_effective = (
        float(getattr(config, "lambda_edge_bias", 0.0))
        if bool(prior_relational_enabled)
        else 0.0
    )

    prior_embeddings_by_view = load_prior_embeddings_by_view(
        genes=genes,
        view_names=prior_view_sources,
        config=config,
    )
    if len(prior_embeddings_by_view) == 0:
        raise ValueError("No prior view embeddings were loaded for model reconstruction.")

    ordered_celltypes = [name for name, _ in sorted(celltype_mapping.items(), key=lambda item: int(item[1]))]
    deg_dir = resolve_deg_directory(
        dataset=str(config.dataset),
        deg_dir=os.path.join(str(preselection_root), "DEG"),
    )
    regulation_onehot = build_celltype_regulation_onehot(
        str(config.dataset),
        genes,
        ordered_celltypes,
        deg_dir=deg_dir,
    )
    global_zscore, zscore_source_path, zscore_missing = build_global_deg_zscore_vector(
        str(config.dataset),
        genes,
        deg_dir=deg_dir,
    )
    edge_index, _ = build_edge_index(genes, str(config.ppi_path), bool(config.gnn_self_loop))

    prior_view_tensors_device = {
        str(view_name): view_tensor.to(device)
        for view_name, view_tensor in prior_embeddings_by_view.items()
    }
    protein_embeddings = torch.zeros((len(genes), int(encoder_hidden_dimension)), dtype=torch.float32)
    protein_source = "find_multiview"
    protein_prior_base_embeddings = None
    qkv_projection_choice = "find"

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
        prior_interface_num_heads=int(getattr(config, "prior_interface_num_heads", 4)),
        prior_interface_dropout=float(getattr(config, "prior_interface_dropout", 0.1)),
        prior_interface_ffn_hidden_dim=int(getattr(config, "prior_interface_ffn_hidden_dim", 0)),
        prior_knn_k=int(getattr(config, "prior_knn_k", 16)),
        prior_knn_symmetric=bool(getattr(config, "prior_knn_symmetric", True)),
        lambda_edge_bias=float(lambda_edge_bias_effective),
    ).to(device)

    mil_aggregator = PatientMILAggregator(
        embedding_dimension=cell_embedding_dimension,
        num_celltypes=int(len(celltype_mapping)),
        pooling=config.mil_pooling,
        attention_hidden_dimension=int(getattr(config, "mil_attention_hidden_dim", config.classifier_hidden_dim)),
    ).to(device)

    classifier = PatientClassifier(cell_embedding_dimension, config).to(device)

    checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        load_cell_encoder_state_dict_with_edge_buffer_fallback(
            cell_encoder=cell_encoder,
            state_dict=payload["cell_encoder"],
            device=device,
        )
        mil_aggregator.load_state_dict(payload["mil_aggregator"])
        classifier.load_state_dict(payload["classifier"])
    else:
        model_path = os.path.join(run_dir, "best_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Neither best_checkpoint.pt nor best_model.pt found in run_dir: {run_dir}"
            )
        payload = torch.load(model_path, map_location=device)
        if isinstance(payload, dict) and "cell_encoder" in payload:
            load_cell_encoder_state_dict_with_edge_buffer_fallback(
                cell_encoder=cell_encoder,
                state_dict=payload["cell_encoder"],
                device=device,
            )
            mil_aggregator.load_state_dict(payload["mil_aggregator"])
            classifier.load_state_dict(payload["classifier"])
        else:
            raise ValueError("Unsupported best_model.pt format for biomarker pipeline.")

    cell_encoder.eval()
    mil_aggregator.eval()
    classifier.eval()

    split_name = str(getattr(config, "biomarker_split", "test")).strip().lower()
    split_map = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
    if split_name not in split_map:
        raise ValueError("biomarker_split must be one of {'train','val','test'}")

    split_path = os.path.join(str(config.splits_directory), f"{config.dataset}_idx_{int(config.split_number)}.pkl")
    split_indices = load_pickle(split_path)
    if not isinstance(split_indices, list) or len(split_indices) != 3:
        raise ValueError("Split file must be list [train_cells, val_cells, test_cells].")
    split_cells = np.asarray(split_indices[split_map[split_name]], dtype=np.int64)

    bag_size = int(getattr(config, "biomarker_cells_per_bag", getattr(config, "cells_per_bag", 64)))
    bags_per_patient = int(getattr(config, "biomarker_bags_per_patient", 1))
    sample_mode = str(getattr(config, "biomarker_sample_mode", getattr(config, "sample_bags", getattr(config, "bag_sampling", "proportional"))))
    metric_name = str(getattr(config, "biomarker_metric", "auprc"))
    patient_prob_reduction = str(getattr(config, "biomarker_patient_probability_reduction", "mean"))
    pathway_repeats = int(getattr(config, "biomarker_pathway_permutations", 10))
    celltype_repeats = int(getattr(config, "biomarker_celltype_permutations", 20))

    max_pathways = int(getattr(config, "biomarker_max_pathways", 0))
    min_pathway_size = int(getattr(config, "biomarker_min_pathway_size", 5))
    max_pathway_size = int(getattr(config, "biomarker_max_pathway_size", 5000))

    patient_to_cells: Dict[int, List[int]] = defaultdict(list)
    for cell_index in split_cells.tolist():
        patient_to_cells[int(patient_array[int(cell_index)])].append(int(cell_index))

    bag_defs: List[BagDefinition] = []
    for patient_index in sorted(patient_to_cells.keys()):
        patient_cells = np.asarray(sorted(patient_to_cells[patient_index]), dtype=np.int64)
        if int(patient_cells.shape[0]) <= 0:
            continue
        patient_labels = label_indices[patient_cells]
        values, counts = np.unique(patient_labels, return_counts=True)
        patient_label = int(values[np.argmax(counts)])

        patient_id = patient_index_to_id[int(patient_index)]
        patient_celltypes = celltype_array[patient_cells]

        for repeat_index in range(max(1, bags_per_patient)):
            rng = np.random.default_rng(seed + int(patient_index) * 1000003 + int(repeat_index) * 7919)
            sampled_cells = sample_patient_bag_indices(
                patient_cell_indices=patient_cells,
                patient_celltypes=patient_celltypes,
                bag_size=bag_size,
                sample_mode=sample_mode,
                rng=rng,
            )
            bag_defs.append(
                BagDefinition(
                    bag_id=len(bag_defs),
                    patient_index=int(patient_index),
                    patient_id=patient_id,
                    patient_label=patient_label,
                    bag_repeat_index=int(repeat_index),
                    cell_indices=np.asarray(sampled_cells, dtype=np.int64),
                )
            )

    if len(bag_defs) == 0:
        raise ValueError("No bags were built for biomarker evaluation.")

    bag_lookup: Dict[Tuple[int, int], int] = {}
    for bag in bag_defs:
        bag_lookup[(int(bag.patient_index), int(bag.bag_repeat_index))] = int(bag.bag_id)

    patient_ids_unique = sorted({int(b.patient_index) for b in bag_defs})
    celltype_ids_eval = sorted({int(celltype_array[int(c)]) for c in split_cells.tolist()})

    print(f"Evaluating split={split_name}: patients={len(patient_ids_unique)}, bags={len(bag_defs)}", flush=True)

    bag_cache: Dict[int, BagCache] = {}
    baseline_probs: Dict[int, float] = {}

    with torch.inference_mode():
        for bag in tqdm(bag_defs, desc="[Baseline] bags"):
            expr = get_dense_rows(expression_matrix, bag.cell_indices)
            local_celltypes = celltype_array[bag.cell_indices]
            local_treatments = treatment_array[bag.cell_indices]
            prob, embeddings = forward_bag_from_expression(
                cell_encoder=cell_encoder,
                mil_aggregator=mil_aggregator,
                classifier=classifier,
                expression=expr,
                celltypes=local_celltypes,
                treatments=local_treatments,
                device=device,
            )
            baseline_probs[int(bag.bag_id)] = float(prob)
            bag_cache[int(bag.bag_id)] = BagCache(
                celltypes=np.asarray(local_celltypes, dtype=np.int64),
                treatments=np.asarray(local_treatments, dtype=np.int64),
                expression=np.asarray(expr, dtype=np.float32),
                embeddings=np.asarray(embeddings, dtype=np.float32),
                baseline_prob=float(prob),
            )

    patient_idx_base, labels_base, probs_base = aggregate_patient_probabilities(
        bag_defs=bag_defs,
        bag_probabilities=baseline_probs,
        reduction=patient_prob_reduction,
    )
    baseline_metrics = compute_binary_metrics(labels_base, probs_base)
    baseline_metric_value = select_metric(baseline_metrics, metric_name)
    metric_key = str(metric_name).strip().lower()
    if metric_key == "loss":
        metric_key = "logloss"
    if metric_key == "logloss":
        baseline_display_value = float(baseline_metrics.get("logloss", float("nan")))
        print(
            f"Baseline patient-level logloss={baseline_display_value:.6f} "
            f"(patients={len(patient_idx_base)}, reduction={patient_prob_reduction})",
            flush=True,
        )
    else:
        baseline_display_value = float(baseline_metric_value)
        print(
            f"Baseline patient-level {metric_name}={baseline_metric_value:.6f} "
            f"(patients={len(patient_idx_base)}, reduction={patient_prob_reduction})",
            flush=True,
        )

    # Save baseline predictions early so partial runs still yield something useful.
    baseline_patient_rows = []
    for patient_index, label_value, prob_value in zip(
        patient_idx_base.tolist(),
        labels_base.tolist(),
        probs_base.tolist(),
    ):
        baseline_patient_rows.append(
            {
                "patient_index": int(patient_index),
                "patient_id": patient_index_to_id[int(patient_index)],
                "true_label": int(label_value),
                "pred_prob": float(prob_value),
                "pred_label": int(float(prob_value) >= 0.5),
            }
        )
    pd.DataFrame(baseline_patient_rows).to_csv(
        os.path.join(output_dir, "baseline_patient_predictions.tsv"),
        sep="\t",
        index=False,
    )

    stability_enabled = bool(getattr(config, "biomarker_stability_enabled", True))

    def normalize_topk_list(value, default: Sequence[int]) -> List[int]:
        if value is None:
            items = list(default)
        elif isinstance(value, (list, tuple)):
            items = [int(v) for v in value]
        else:
            items = [int(value)]
        normalized = sorted({int(v) for v in items if int(v) > 0})
        return normalized if normalized else list(default)

    stability_celltype_topks = normalize_topk_list(
        getattr(config, "biomarker_stability_top_celltype_ks", [3, 5, 8]),
        default=[3, 5, 8],
    )
    stability_gene_topks = normalize_topk_list(
        getattr(config, "biomarker_stability_top_gene_ks", [20, 50, 100]),
        default=[20, 50, 100],
    )
    stability_pathway_topks = normalize_topk_list(
        getattr(config, "biomarker_stability_top_pathway_ks", [20, 50]),
        default=[20, 50],
    )

    # Step 2: celltype prioritization delta at MIL input (run first to decide Step 1 scope).
    celltype_priority_rows: List[Dict[str, object]] = []
    celltype_delta_repeats_map: Dict[int, List[float]] = {}
    priority_partial_path = os.path.join(output_dir, "celltype_prioritization_partial.tsv")
    priority_repeats_path = os.path.join(output_dir, "celltype_prioritization_repeats.tsv")
    with open(priority_partial_path, "w", encoding="utf-8") as handle, open(
        priority_repeats_path, "w", encoding="utf-8"
    ) as repeats_handle:
        handle.write(
            "celltype_index\tcelltype_name\tdelta_cell_metric\t"
            "delta_cell_metric_std\tdelta_cell_metric_min\tdelta_cell_metric_max\tn_repeats\n"
        )
        repeats_handle.write(
            "celltype_index\tcelltype_name\trepeat\tmetric_pert\tdelta_cell_metric\n"
        )
        handle.flush()
        repeats_handle.flush()
        with torch.inference_mode():
            for celltype_id in tqdm(celltype_ids_eval, desc="[Step2] celltypes"):
                delta_repeats: List[float] = []
                for repeat in range(max(1, celltype_repeats)):
                    rng = np.random.default_rng(seed + int(celltype_id) * 51059 + int(repeat) * 7919)
                    perm = rng.permutation(len(patient_ids_unique))
                    donor_map = {
                        int(patient_ids_unique[i]): int(patient_ids_unique[int(perm[i])])
                        for i in range(len(patient_ids_unique))
                    }

                    pert_probs: Dict[int, float] = {}
                    for bag in bag_defs:
                        cache = bag_cache[int(bag.bag_id)]
                        local_celltypes = cache.celltypes
                        embeddings = np.asarray(cache.embeddings, dtype=np.float32).copy()

                        target_rows = np.where(local_celltypes == int(celltype_id))[0].astype(np.int64)
                        if target_rows.size > 0:
                            donor_patient = donor_map[int(bag.patient_index)]
                            donor_bag_id = bag_lookup[(int(donor_patient), int(bag.bag_repeat_index))]
                            donor_cache = bag_cache[int(donor_bag_id)]
                            donor_rows = np.where(donor_cache.celltypes == int(celltype_id))[0].astype(np.int64)
                            if donor_rows.size > 0:
                                donor_emb = donor_cache.embeddings[donor_rows]
                                sampled_emb = sample_rows_with_replacement(
                                    donor_emb,
                                    target_rows=int(target_rows.shape[0]),
                                    rng=rng,
                                )
                                embeddings[target_rows] = sampled_emb

                        prob = forward_bag_from_embeddings(
                            mil_aggregator=mil_aggregator,
                            classifier=classifier,
                            embeddings=embeddings,
                            celltypes=local_celltypes,
                            device=device,
                        )
                        pert_probs[int(bag.bag_id)] = float(prob)

                    _, labels_pert, probs_pert = aggregate_patient_probabilities(
                        bag_defs=bag_defs,
                        bag_probabilities=pert_probs,
                        reduction=patient_prob_reduction,
                    )
                    metrics_pert = compute_binary_metrics(labels_pert, probs_pert)
                    metric_pert_score = select_metric(metrics_pert, metric_name)
                    delta_value = float(baseline_metric_value - metric_pert_score)
                    delta_repeats.append(delta_value)

                    if metric_key == "logloss":
                        metric_pert_display = float(metrics_pert.get("logloss", float("nan")))
                    else:
                        metric_pert_display = float(metric_pert_score)
                    repeats_handle.write(
                        f"{int(celltype_id)}\t"
                        f"{ordered_celltypes[int(celltype_id)] if 0 <= int(celltype_id) < len(ordered_celltypes) else str(celltype_id)}\t"
                        f"{int(repeat)}\t{metric_pert_display}\t{delta_value}\n"
                    )
                    repeats_handle.flush()

                celltype_name = (
                    ordered_celltypes[int(celltype_id)]
                    if 0 <= int(celltype_id) < len(ordered_celltypes)
                    else str(celltype_id)
                )
                celltype_delta_repeats_map[int(celltype_id)] = list(delta_repeats)
                delta_mean = float(np.mean(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                delta_std = float(np.std(delta_repeats, ddof=1)) if len(delta_repeats) > 1 else 0.0
                delta_min = float(np.min(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                delta_max = float(np.max(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                row = {
                    "celltype_index": int(celltype_id),
                    "celltype_name": celltype_name,
                    "delta_cell_metric": delta_mean,
                    "delta_cell_metric_std": delta_std,
                    "delta_cell_metric_min": delta_min,
                    "delta_cell_metric_max": delta_max,
                    "n_repeats": int(len(delta_repeats)),
                }
                celltype_priority_rows.append(row)
                handle.write(
                    f"{int(celltype_id)}\t{celltype_name}\t{delta_mean}\t{delta_std}\t{delta_min}\t{delta_max}\t{int(len(delta_repeats))}\n"
                )
                handle.flush()

    priority_df = pd.DataFrame(celltype_priority_rows)
    positive = np.maximum(priority_df["delta_cell_metric"].to_numpy(dtype=np.float64), 0.0)
    denom = float(np.sum(positive))
    if denom <= 0.0:
        if int(priority_df.shape[0]) > 0:
            weights = np.ones((priority_df.shape[0],), dtype=np.float64) / float(priority_df.shape[0])
        else:
            weights = np.zeros((0,), dtype=np.float64)
    else:
        weights = positive / denom
    priority_df["weight"] = weights
    priority_df = priority_df.sort_values("weight", ascending=False).reset_index(drop=True)
    priority_df["rank"] = np.arange(1, int(priority_df.shape[0]) + 1)
    priority_df.to_csv(os.path.join(output_dir, "celltype_prioritization.tsv"), sep="\t", index=False)

    if stability_enabled and int(celltype_repeats) > 1 and int(priority_df.shape[0]) > 0:
        step2_stability_path = os.path.join(output_dir, "celltype_prioritization_stability.tsv")
        step2_rank_path = os.path.join(output_dir, "celltype_prioritization_rank_stability.tsv")
        mean_scores = priority_df["delta_cell_metric"].to_numpy(dtype=np.float64)
        celltype_order = priority_df["celltype_index"].to_numpy(dtype=np.int64)
        mean_topk = {
            int(k): celltype_order[np.argsort(-mean_scores)[: min(int(k), int(celltype_order.shape[0]))]].tolist()
            for k in stability_celltype_topks
        }

        ranks_by_celltype: Dict[int, List[int]] = {int(ct): [] for ct in celltype_order.tolist()}
        with open(step2_stability_path, "w", encoding="utf-8") as handle:
            header_cols = ["repeat", "spearman_delta_vs_mean"] + [
                f"jaccard_top{k}" for k in stability_celltype_topks
            ]
            handle.write("\t".join(header_cols) + "\n")
            handle.flush()
            for repeat in range(int(celltype_repeats)):
                repeat_scores = np.array(
                    [float(celltype_delta_repeats_map[int(ct)][int(repeat)]) for ct in celltype_order.tolist()],
                    dtype=np.float64,
                )
                spearman = stable_spearman(mean_scores, repeat_scores)
                repeat_top = celltype_order[np.argsort(-repeat_scores)]
                jac_values = []
                for k in stability_celltype_topks:
                    topk_repeat = repeat_top[: min(int(k), int(repeat_top.shape[0]))].tolist()
                    jac_values.append(jaccard_index(mean_topk[int(k)], topk_repeat))

                handle.write(
                    "\t".join(
                        [str(int(repeat)), f"{spearman}"] + [f"{v}" for v in jac_values]
                    )
                    + "\n"
                )
                handle.flush()

                repeat_rank = np.empty((int(repeat_top.shape[0]),), dtype=np.int64)
                for position, ct in enumerate(repeat_top.tolist(), start=1):
                    repeat_rank[int(np.where(celltype_order == int(ct))[0][0])] = int(position)
                for idx, ct in enumerate(celltype_order.tolist()):
                    ranks_by_celltype[int(ct)].append(int(repeat_rank[int(idx)]))

        with open(step2_rank_path, "w", encoding="utf-8") as handle:
            handle.write(
                "celltype_index\tcelltype_name\trank_mean\trank_std\t"
                + "\t".join([f"top{k}_freq" for k in stability_celltype_topks])
                + "\n"
            )
            handle.flush()
            for ct in celltype_order.tolist():
                ranks = np.asarray(ranks_by_celltype[int(ct)], dtype=np.float64)
                rank_mean = float(np.mean(ranks)) if ranks.size > 0 else float("nan")
                rank_std = float(np.std(ranks, ddof=1)) if ranks.size > 1 else 0.0
                freqs = []
                for k in stability_celltype_topks:
                    freqs.append(float(np.mean((ranks <= float(k)).astype(np.float64))) if ranks.size > 0 else float("nan"))
                ct_name = (
                    ordered_celltypes[int(ct)]
                    if 0 <= int(ct) < len(ordered_celltypes)
                    else str(int(ct))
                )
                handle.write(
                    "\t".join(
                        [str(int(ct)), ct_name, f"{rank_mean}", f"{rank_std}"] + [f"{v}" for v in freqs]
                    )
                    + "\n"
                )
                handle.flush()

    weight_map = {
        int(row["celltype_index"]): float(row["weight"]) for _, row in priority_df.iterrows()
    }

    step1_mode = str(getattr(config, "biomarker_step1_celltype_mode", "all")).strip().lower()
    step1_top_k = int(getattr(config, "biomarker_step1_top_k", 0))
    step1_cum_weight = float(getattr(config, "biomarker_step1_cum_weight", 0.0))
    step1_weight_threshold = float(getattr(config, "biomarker_step1_weight_threshold", 0.0))

    ranked_celltypes = [int(v) for v in priority_df["celltype_index"].astype(int).tolist()]
    ranked_weights = [float(v) for v in priority_df["weight"].astype(float).tolist()]
    step1_celltype_ids: List[int]
    if step1_mode in ("all", ""):
        step1_celltype_ids = ranked_celltypes
    elif step1_mode == "top_k":
        if step1_top_k <= 0:
            step1_celltype_ids = ranked_celltypes
        else:
            step1_celltype_ids = ranked_celltypes[: min(step1_top_k, len(ranked_celltypes))]
    elif step1_mode == "cum_weight":
        threshold = step1_cum_weight if step1_cum_weight > 0.0 else 1.0
        selected: List[int] = []
        total = 0.0
        for ct, w in zip(ranked_celltypes, ranked_weights):
            selected.append(int(ct))
            total += float(w)
            if total >= float(threshold):
                break
        step1_celltype_ids = selected if len(selected) > 0 else ranked_celltypes
    elif step1_mode == "weight_threshold":
        threshold = float(step1_weight_threshold)
        selected = [int(ct) for ct, w in zip(ranked_celltypes, ranked_weights) if float(w) >= threshold]
        step1_celltype_ids = selected if len(selected) > 0 else ranked_celltypes
    else:
        raise ValueError(
            "Unknown biomarker_step1_celltype_mode. "
            f"Got {step1_mode!r}, expected one of: all, top_k, cum_weight, weight_threshold."
        )

    selected_df = priority_df[priority_df["celltype_index"].isin(step1_celltype_ids)].copy()
    selected_df.to_csv(os.path.join(output_dir, "step1_selected_celltypes.tsv"), sep="\t", index=False)
    print(
        f"[Step1] Using {len(step1_celltype_ids)}/{len(celltype_ids_eval)} celltypes "
        f"(mode={step1_mode}).",
        flush=True,
    )

    pathway_names, pathway_index_map = load_pathway_gene_sets(pathway_path, genes)
    filtered_pathways: List[str] = []
    for pathway_name in pathway_names:
        size = int(pathway_index_map[pathway_name].numel())
        if size < min_pathway_size:
            continue
        if size > max_pathway_size:
            continue
        filtered_pathways.append(pathway_name)

    if max_pathways > 0 and len(filtered_pathways) > max_pathways:
        filtered_pathways = filtered_pathways[:max_pathways]

    if len(filtered_pathways) == 0:
        raise ValueError("No pathways available after filtering. Check pathway file and size thresholds.")

    print(
        f"Using {len(filtered_pathways)} pathways "
        f"(min_size={min_pathway_size}, max_size={max_pathway_size}).",
        flush=True,
    )

    # Fast packed bag tensors for batched MIL inference.
    num_bags_total = int(len(bag_defs))
    bag_offsets = np.zeros((num_bags_total,), dtype=np.int64)
    packed_embeddings: List[np.ndarray] = []
    packed_celltypes: List[np.ndarray] = []
    packed_bag_index: List[np.ndarray] = []
    cursor = 0
    for bag_id in range(num_bags_total):
        bag_offsets[bag_id] = int(cursor)
        cache = bag_cache[int(bag_id)]
        num_cells = int(cache.embeddings.shape[0])
        cursor += num_cells
        packed_embeddings.append(cache.embeddings)
        packed_celltypes.append(cache.celltypes)
        packed_bag_index.append(np.full((num_cells,), int(bag_id), dtype=np.int64))

    packed_embeddings_np = (
        np.concatenate(packed_embeddings, axis=0)
        if packed_embeddings
        else np.zeros((0, int(cell_embedding_dimension)), dtype=np.float32)
    )
    packed_celltypes_np = (
        np.concatenate(packed_celltypes, axis=0)
        if packed_celltypes
        else np.zeros((0,), dtype=np.int64)
    )
    packed_bag_index_np = (
        np.concatenate(packed_bag_index, axis=0)
        if packed_bag_index
        else np.zeros((0,), dtype=np.int64)
    )

    packed_embeddings_t = torch.tensor(packed_embeddings_np, dtype=torch.float32, device=device)
    packed_celltypes_t = torch.tensor(packed_celltypes_np, dtype=torch.long, device=device)
    packed_bag_index_t = torch.tensor(packed_bag_index_np, dtype=torch.long, device=device)
    working_embeddings_t = packed_embeddings_t.clone()

    encoder_chunk_size = int(getattr(config, "biomarker_cell_encoder_chunk_size", 128))
    if encoder_chunk_size <= 0:
        encoder_chunk_size = 128

    # Precompute per-bag positions for each evaluated celltype (small speedup in the inner loop).
    bag_rows_by_celltype: List[Dict[int, np.ndarray]] = []
    for bag_id in range(num_bags_total):
        cache = bag_cache[int(bag_id)]
        mapping: Dict[int, np.ndarray] = {}
        for ct in step1_celltype_ids:
            mapping[int(ct)] = np.where(cache.celltypes == int(ct))[0].astype(np.int64)
        bag_rows_by_celltype.append(mapping)

    # Precompute patient->bag indices for fast patient-level aggregation.
    patient_to_bags: Dict[int, List[int]] = defaultdict(list)
    for bag in bag_defs:
        patient_to_bags[int(bag.patient_index)].append(int(bag.bag_id))
    patient_bag_ids = [
        np.asarray(sorted(patient_to_bags[int(pid)]), dtype=np.int64) for pid in patient_idx_base.tolist()
    ]
    use_median_reduction = str(patient_prob_reduction).strip().lower() == "median"

    # Step 1 gene score S_t(g) setup.
    pathway_size_map = {
        str(pathway_name): int(pathway_index_map[str(pathway_name)].numel()) for pathway_name in filtered_pathways
    }
    gene_to_pathways: Dict[int, List[str]] = defaultdict(list)
    for pathway_name in filtered_pathways:
        idx = pathway_index_map[str(pathway_name)].detach().cpu().numpy().astype(np.int64)
        for gene_index in idx.tolist():
            gene_to_pathways[int(gene_index)].append(str(pathway_name))

    step1_delta_path = os.path.join(output_dir, "celltype_pathway_delta.tsv")
    step1_delta_repeats_path = os.path.join(output_dir, "celltype_pathway_delta_repeats.tsv")
    celltype_gene_path = os.path.join(output_dir, "celltype_gene_biomarker.tsv")
    gene_stability_path = os.path.join(output_dir, "gene_ranking_stability.tsv")
    pathway_stability_path = os.path.join(output_dir, "pathway_ranking_stability.tsv")

    final_scores = np.zeros((len(genes),), dtype=np.float64)
    step1_repeat_count = max(1, int(pathway_repeats))
    final_scores_by_repeat = [
        np.zeros((len(genes),), dtype=np.float64) for _ in range(int(step1_repeat_count))
    ]
    selected_weight_sum = 0.0

    with open(step1_delta_path, "w", encoding="utf-8") as delta_handle, open(
        step1_delta_repeats_path, "w", encoding="utf-8"
    ) as delta_repeats_handle, open(celltype_gene_path, "w", encoding="utf-8") as gene_handle, open(
        gene_stability_path, "w", encoding="utf-8"
    ) as gene_stability_handle, open(pathway_stability_path, "w", encoding="utf-8") as pathway_stability_handle:
        delta_handle.write(
            "celltype_index\tcelltype_name\tpathway\tpathway_size\t"
            "delta_gene_metric\tdelta_gene_metric_std\tdelta_gene_metric_min\t"
            "delta_gene_metric_max\tn_repeats\n"
        )
        delta_repeats_handle.write(
            "celltype_index\tcelltype_name\tpathway\trepeat\tmetric_pert\tdelta_gene_metric\n"
        )
        gene_handle.write("celltype_index\tcelltype_name\tgene\tS_t\trank\tnum_pathways\n")

        gene_stability_header = ["celltype_index", "celltype_name", "repeat", "spearman_score_vs_mean"] + [
            f"jaccard_top{k}" for k in stability_gene_topks
        ]
        gene_stability_handle.write("\t".join(gene_stability_header) + "\n")

        pathway_stability_header = ["celltype_index", "celltype_name", "repeat", "spearman_delta_vs_mean"] + [
            f"jaccard_top{k}" for k in stability_pathway_topks
        ]
        pathway_stability_handle.write("\t".join(pathway_stability_header) + "\n")

        delta_handle.flush()
        delta_repeats_handle.flush()
        gene_handle.flush()
        gene_stability_handle.flush()
        pathway_stability_handle.flush()

        with torch.inference_mode():
            for celltype_id in tqdm(step1_celltype_ids, desc="[Step1] celltypes"):
                celltype_name = (
                    ordered_celltypes[int(celltype_id)]
                    if 0 <= int(celltype_id) < len(ordered_celltypes)
                    else str(celltype_id)
                )

                delta_by_pathway: Dict[str, float] = {}
                delta_repeats_by_pathway: Dict[str, List[float]] = {}
                for pathway_name in tqdm(filtered_pathways, desc=f"[Step1] pathways ({celltype_name})", leave=False):
                    pathway_idx = pathway_index_map[pathway_name].detach().cpu().numpy().astype(np.int64)
                    delta_repeats: List[float] = []
                    metric_pert_displays: List[float] = []
                    for repeat in range(max(1, pathway_repeats)):
                        rng = np.random.default_rng(seed + int(celltype_id) * 17011 + int(repeat) * 31337)
                        perm = rng.permutation(len(patient_ids_unique))
                        donor_map = {
                            int(patient_ids_unique[i]): int(patient_ids_unique[int(perm[i])])
                            for i in range(len(patient_ids_unique))
                        }

                        working_embeddings_t.copy_(packed_embeddings_t)

                        update_expr_chunks: List[np.ndarray] = []
                        update_indices_chunks: List[np.ndarray] = []
                        update_treatment_chunks: List[np.ndarray] = []
                        for bag in bag_defs:
                            bag_id = int(bag.bag_id)
                            target_rows = bag_rows_by_celltype[bag_id][int(celltype_id)]
                            if target_rows.size <= 0:
                                continue

                            donor_patient = donor_map[int(bag.patient_index)]
                            donor_bag_id = bag_lookup[(int(donor_patient), int(bag.bag_repeat_index))]
                            donor_rows = bag_rows_by_celltype[int(donor_bag_id)][int(celltype_id)]
                            if donor_rows.size <= 0:
                                continue

                            # Recompute embeddings only for the affected (celltype, pathway) rows.
                            target_expr = bag_cache[int(bag_id)].expression[target_rows].copy()
                            donor_expr = bag_cache[int(donor_bag_id)].expression[donor_rows]
                            donor_block = donor_expr[:, pathway_idx]
                            sampled_block = sample_rows_with_replacement(
                                donor_block,
                                target_rows=int(target_rows.shape[0]),
                                rng=rng,
                            )
                            target_expr[:, pathway_idx] = sampled_block
                            update_expr_chunks.append(target_expr)
                            update_indices_chunks.append(bag_offsets[int(bag_id)] + target_rows)
                            update_treatment_chunks.append(
                                bag_cache[int(bag_id)].treatments[target_rows].astype(np.int64, copy=False)
                            )

                        if update_expr_chunks:
                            update_expr = np.concatenate(update_expr_chunks, axis=0)
                            update_indices = np.concatenate(update_indices_chunks, axis=0)
                            update_treatments = np.concatenate(update_treatment_chunks, axis=0)

                            for start in range(0, int(update_expr.shape[0]), int(encoder_chunk_size)):
                                end = min(int(update_expr.shape[0]), start + int(encoder_chunk_size))
                                expr_chunk = torch.tensor(update_expr[start:end], dtype=torch.float32, device=device)
                                ct_chunk = torch.full(
                                    (int(end - start),),
                                    int(celltype_id),
                                    dtype=torch.long,
                                    device=device,
                                )
                                tr_chunk = torch.tensor(update_treatments[start:end], dtype=torch.long, device=device)
                                encoder_out = cell_encoder(
                                    expression_values=expr_chunk,
                                    celltype_index=ct_chunk,
                                    treatment_index=tr_chunk,
                                    return_node_outputs=False,
                                    return_attention=False,
                                )
                                idx_chunk = torch.tensor(update_indices[start:end], dtype=torch.long, device=device)
                                working_embeddings_t[idx_chunk] = encoder_out["cell_embeddings"]

                        mil_out = mil_aggregator(
                            cell_embeddings=working_embeddings_t,
                            celltype_index=packed_celltypes_t,
                            bag_index=packed_bag_index_t,
                            num_bags=num_bags_total,
                        )
                        logits = classifier(mil_out["patient_embeddings"])
                        bag_probs = torch.sigmoid(logits).detach().cpu().numpy()

                        probs_pert = np.zeros((len(patient_idx_base),), dtype=np.float64)
                        for patient_offset, bag_ids in enumerate(patient_bag_ids):
                            values = bag_probs[bag_ids]
                            probs_pert[patient_offset] = (
                                float(np.median(values)) if use_median_reduction else float(np.mean(values))
                            )
                        metrics_pert = compute_binary_metrics(labels_base, probs_pert)
                        metric_pert_score = select_metric(metrics_pert, metric_name)
                        delta_value = float(baseline_metric_value - metric_pert_score)
                        delta_repeats.append(delta_value)
                        if metric_key == "logloss":
                            metric_pert_displays.append(float(metrics_pert.get("logloss", float("nan"))))
                        else:
                            metric_pert_displays.append(float(metric_pert_score))

                    delta_repeats_by_pathway[str(pathway_name)] = list(delta_repeats)

                    for repeat, (metric_display, delta_value) in enumerate(zip(metric_pert_displays, delta_repeats)):
                        delta_repeats_handle.write(
                            f"{int(celltype_id)}\t{celltype_name}\t{str(pathway_name)}\t"
                            f"{int(repeat)}\t{float(metric_display)}\t{float(delta_value)}\n"
                        )
                    delta_repeats_handle.flush()

                    delta_mean = float(np.mean(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                    delta_std = float(np.std(delta_repeats, ddof=1)) if len(delta_repeats) > 1 else 0.0
                    delta_min = float(np.min(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                    delta_max = float(np.max(delta_repeats)) if len(delta_repeats) > 0 else 0.0
                    delta_by_pathway[str(pathway_name)] = float(delta_mean)
                    delta_handle.write(
                        f"{int(celltype_id)}\t{celltype_name}\t{str(pathway_name)}\t"
                        f"{int(pathway_idx.shape[0])}\t{delta_mean}\t{delta_std}\t{delta_min}\t{delta_max}\t{int(len(delta_repeats))}\n"
                    )
                    delta_handle.flush()

                # Per-celltype gene score S_t(g) + persistence.
                score_values = np.zeros((len(genes),), dtype=np.float64)
                for gene_index in range(len(genes)):
                    pathways = gene_to_pathways.get(int(gene_index), [])
                    if len(pathways) == 0:
                        score_values[gene_index] = 0.0
                        continue
                    contributions = []
                    for pathway_name in pathways:
                        delta_value = float(delta_by_pathway.get(str(pathway_name), 0.0))
                        pathway_size = float(max(pathway_size_map.get(str(pathway_name), 1), 1))
                        contributions.append(delta_value / pathway_size)
                    score_values[gene_index] = float(np.mean(contributions))

                score_values_by_repeat: List[np.ndarray] = []
                if stability_enabled and int(step1_repeat_count) > 1:
                    mean_pathway_scores = np.asarray(
                        [float(delta_by_pathway.get(str(name), 0.0)) for name in filtered_pathways],
                        dtype=np.float64,
                    )
                    mean_pathway_topk = {
                        int(k): np.asarray(filtered_pathways, dtype=object)[
                            np.argsort(-mean_pathway_scores)[: min(int(k), int(mean_pathway_scores.shape[0]))]
                        ].tolist()
                        for k in stability_pathway_topks
                    }
                    mean_gene_topk = {
                        int(k): np.argsort(-score_values)[: min(int(k), int(score_values.shape[0]))].tolist()
                        for k in stability_gene_topks
                    }

                    for repeat in range(int(step1_repeat_count)):
                        repeat_pathway_scores = np.asarray(
                            [float(delta_repeats_by_pathway[str(name)][int(repeat)]) for name in filtered_pathways],
                            dtype=np.float64,
                        )
                        spearman_delta = stable_spearman(mean_pathway_scores, repeat_pathway_scores)
                        repeat_pathway_order = np.argsort(-repeat_pathway_scores)
                        repeat_pathway_names = np.asarray(filtered_pathways, dtype=object)[repeat_pathway_order].tolist()
                        pathway_jaccards = []
                        for k in stability_pathway_topks:
                            repeat_top = repeat_pathway_names[: min(int(k), len(repeat_pathway_names))]
                            pathway_jaccards.append(jaccard_index(mean_pathway_topk[int(k)], repeat_top))
                        pathway_stability_handle.write(
                            "\t".join(
                                [str(int(celltype_id)), str(celltype_name), str(int(repeat)), f"{spearman_delta}"]
                                + [f"{v}" for v in pathway_jaccards]
                            )
                            + "\n"
                        )
                        pathway_stability_handle.flush()

                        score_repeat = np.zeros((len(genes),), dtype=np.float64)
                        for gene_index in range(len(genes)):
                            pathways = gene_to_pathways.get(int(gene_index), [])
                            if len(pathways) == 0:
                                score_repeat[gene_index] = 0.0
                                continue
                            contributions = []
                            for pathway_name in pathways:
                                delta_value = float(delta_repeats_by_pathway[str(pathway_name)][int(repeat)])
                                pathway_size = float(max(pathway_size_map.get(str(pathway_name), 1), 1))
                                contributions.append(delta_value / pathway_size)
                            score_repeat[gene_index] = float(np.mean(contributions))
                        score_values_by_repeat.append(score_repeat)

                        spearman_gene = stable_spearman(score_values, score_repeat)
                        repeat_gene_order = np.argsort(-score_repeat)
                        gene_jaccards = []
                        for k in stability_gene_topks:
                            repeat_top = repeat_gene_order[: min(int(k), int(repeat_gene_order.shape[0]))].tolist()
                            gene_jaccards.append(jaccard_index(mean_gene_topk[int(k)], repeat_top))
                        gene_stability_handle.write(
                            "\t".join(
                                [str(int(celltype_id)), str(celltype_name), str(int(repeat)), f"{spearman_gene}"]
                                + [f"{v}" for v in gene_jaccards]
                            )
                            + "\n"
                        )
                        gene_stability_handle.flush()

                order = np.argsort(-score_values)
                rank = np.empty_like(order)
                rank[order] = np.arange(1, len(genes) + 1)

                per_celltype_rows: List[Dict[str, object]] = []
                for gene_index, gene_name in enumerate(genes):
                    pathways = gene_to_pathways.get(int(gene_index), [])
                    per_celltype_rows.append(
                        {
                            "celltype_index": int(celltype_id),
                            "celltype_name": celltype_name,
                            "gene": str(gene_name),
                            "S_t": float(score_values[gene_index]),
                            "rank": int(rank[gene_index]),
                            "num_pathways": int(len(pathways)),
                        }
                    )

                per_celltype_df = (
                    pd.DataFrame(per_celltype_rows)
                    .sort_values("S_t", ascending=False)
                    .reset_index(drop=True)
                )
                safe_celltype = sanitize_filename_component(celltype_name)
                per_celltype_df.to_csv(
                    os.path.join(output_dir, f"celltype_gene_biomarker_{safe_celltype}.tsv"),
                    sep="\t",
                    index=False,
                )
                per_celltype_df.to_csv(gene_handle, sep="\t", index=False, header=False)
                gene_handle.flush()

                w = float(weight_map.get(int(celltype_id), 0.0))
                if w != 0.0:
                    final_scores += w * score_values
                    selected_weight_sum += w
                    if stability_enabled and int(step1_repeat_count) > 1 and len(score_values_by_repeat) == int(step1_repeat_count):
                        for repeat in range(int(step1_repeat_count)):
                            final_scores_by_repeat[int(repeat)] += w * score_values_by_repeat[int(repeat)]

    final_order = np.argsort(-final_scores)
    final_rank = np.empty_like(final_order)
    final_rank[final_order] = np.arange(1, len(genes) + 1)

    final_rows = []
    for gene_index, gene in enumerate(genes):
        final_rows.append(
            {
                "gene": str(gene),
                "S_final": float(final_scores[gene_index]),
                "rank": int(final_rank[gene_index]),
            }
        )
    final_df = pd.DataFrame(final_rows).sort_values("S_final", ascending=False).reset_index(drop=True)
    final_df.to_csv(os.path.join(output_dir, "final_biomarker.tsv"), sep="\t", index=False)

    if stability_enabled and int(step1_repeat_count) > 1 and int(len(final_scores_by_repeat)) == int(step1_repeat_count):
        final_stability_path = os.path.join(output_dir, "final_biomarker_stability.tsv")
        mean_topk = {
            int(k): np.argsort(-final_scores)[: min(int(k), int(final_scores.shape[0]))].tolist()
            for k in stability_gene_topks
        }
        with open(final_stability_path, "w", encoding="utf-8") as handle:
            header_cols = ["repeat", "spearman_score_vs_mean"] + [f"jaccard_top{k}" for k in stability_gene_topks]
            handle.write("\t".join(header_cols) + "\n")
            handle.flush()
            for repeat in range(int(step1_repeat_count)):
                scores_repeat = np.asarray(final_scores_by_repeat[int(repeat)], dtype=np.float64)
                spearman = stable_spearman(final_scores, scores_repeat)
                repeat_order = np.argsort(-scores_repeat)
                jac_values = []
                for k in stability_gene_topks:
                    repeat_top = repeat_order[: min(int(k), int(repeat_order.shape[0]))].tolist()
                    jac_values.append(jaccard_index(mean_topk[int(k)], repeat_top))
                handle.write(
                    "\t".join([str(int(repeat)), f"{spearman}"] + [f"{v}" for v in jac_values]) + "\n"
                )
                handle.flush()

    metadata_out = {
        "config_path": resolved_config_path,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "dataset": str(config.dataset),
        "split": split_name,
        "split_number": int(config.split_number),
        "num_patients": int(len(patient_ids_unique)),
        "num_bags": int(len(bag_defs)),
        "bag_size": int(bag_size),
        "bags_per_patient": int(bags_per_patient),
        "sample_mode": sample_mode,
        "metric": metric_name,
        "patient_probability_reduction": patient_prob_reduction,
        "pathway_file": pathway_path,
        "num_pathways": int(len(filtered_pathways)),
        "pathway_permutations": int(pathway_repeats),
        "celltype_permutations": int(celltype_repeats),
        "stability_enabled": bool(stability_enabled),
        "stability_top_gene_ks": [int(v) for v in stability_gene_topks],
        "stability_top_pathway_ks": [int(v) for v in stability_pathway_topks],
        "stability_top_celltype_ks": [int(v) for v in stability_celltype_topks],
        "step1_celltype_mode": step1_mode,
        "step1_celltype_top_k": int(step1_top_k),
        "step1_celltype_cum_weight": float(step1_cum_weight),
        "step1_celltype_weight_threshold": float(step1_weight_threshold),
        "step1_celltypes_selected": [int(v) for v in step1_celltype_ids],
        "step1_celltypes_selected_count": int(len(step1_celltype_ids)),
        "step1_selected_weight_sum": float(selected_weight_sum),
        "num_genes": int(len(genes)),
        "protein_embedding_source": protein_source,
        "protein_embedding_global_projection_enabled": bool(protein_prior_base_embeddings is not None),
        "protein_embedding_base_dim": (
            int(protein_prior_base_embeddings.shape[1]) if protein_prior_base_embeddings is not None else 0
        ),
        "qkv_choice": str(qkv_projection_choice),
        "qkv_pretrain": bool(getattr(config, "QKV_pretrain", getattr(config, "qkv_pretrain", False))),
        "protein_prior_projection_hidden_dim": int(getattr(config, "protein_prior_projection_hidden_dim", 0)),
        "protein_prior_projection_dropout": float(getattr(config, "protein_prior_projection_dropout", 0.1)),
        "preselection_root": str(preselection_root),
        "deg_zscore_source_path": str(zscore_source_path),
        "deg_zscore_missing": int(zscore_missing),
        "baseline_metrics": {key: float(value) for key, value in baseline_metrics.items()},
        "baseline_metric_value": float(baseline_metric_value),
        "config": config_dict,
    }
    save_json(os.path.join(output_dir, "biomarker_metadata.json"), metadata_out)

    print("Biomarker computation finished.", flush=True)
    print(f"Saved to: {output_dir}", flush=True)
    print(
        "Files: baseline_patient_predictions.tsv, celltype_prioritization.tsv, "
        "celltype_prioritization_repeats.tsv, celltype_prioritization_stability.tsv, "
        "step1_selected_celltypes.tsv, celltype_pathway_delta.tsv, celltype_pathway_delta_repeats.tsv, "
        "celltype_gene_biomarker.tsv, gene_ranking_stability.tsv, pathway_ranking_stability.tsv, "
        "final_biomarker.tsv, final_biomarker_stability.tsv",
        flush=True,
    )

    return metadata_out


def main() -> None:
    args = parse_args()
    run_analysis_phase(
        args.run_dir,
        config_path=str(args.config),
        output_dir=str(args.out_dir),
        pathway_path=str(args.pathway_path),
        gpu_index=args.gpu,
    )


if __name__ == "__main__":
    main()
