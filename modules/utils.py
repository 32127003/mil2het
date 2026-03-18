"""Utility functions for Biomarker GAT training and biomarker extraction."""

from __future__ import annotations

import argparse
import json
import hashlib
import importlib.util
import os
import pickle
import random
from collections import Counter
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from scbiomarker.config import dict_to_namespace, load_workflow_config_dict

MODULE_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_ROOT, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


def _data_path(*parts: str) -> str:
    return os.path.join(DATA_ROOT, *parts)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds across python, numpy, and torch."""
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def sanitize_filename_component(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def min_max_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _sorted_unique(values: Iterable[str]) -> List[str]:
    unique_values = list({str(value) for value in values})
    numeric_values: Optional[List[float]] = []
    for value in unique_values:
        try:
            numeric_values.append(float(value))
        except ValueError:
            numeric_values = None
            break
    if numeric_values is not None:
        paired = sorted(zip(numeric_values, unique_values), key=lambda item: item[0])
        return [value for _, value in paired]
    return sorted(unique_values)


def create_category_mapping(values: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray]:
    categories = _sorted_unique(values)
    mapping = {category: index for index, category in enumerate(categories)}
    indices = np.array([mapping[str(value)] for value in values], dtype=np.int64)
    return mapping, indices


def map_labels(
    label_values: Sequence[str],
    binary_positive_labels: Optional[Sequence[str]] = None,
    binary_negative_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    normalized_labels = [str(value) for value in label_values]

    positive_set = {str(value) for value in (binary_positive_labels or [])}
    negative_set = {str(value) for value in (binary_negative_labels or [])}
    if len(positive_set) == 0 and len(negative_set) == 0:
        raise ValueError(
            "binary label mapping requires at least one of "
            "`binary_positive_labels` or `binary_negative_labels`."
        )

    intersection = positive_set & negative_set
    if len(intersection) > 0:
        raise ValueError(f"binary label sets overlap: {sorted(intersection)}")

    unique_original = sorted(set(normalized_labels))
    if len(positive_set) > 0 and len(negative_set) > 0:
        uncovered = [label for label in unique_original if label not in positive_set and label not in negative_set]
        if len(uncovered) > 0:
            raise ValueError(
                "binary label mapping leaves uncovered labels when both positive/negative sets are provided: "
                f"{uncovered}"
            )

    mapped_labels: List[str] = []
    for label in normalized_labels:
        if label in positive_set:
            mapped_labels.append("1")
            continue
        if label in negative_set:
            mapped_labels.append("0")
            continue
        if len(positive_set) > 0 and len(negative_set) == 0:
            mapped_labels.append("0")
            continue
        if len(negative_set) > 0 and len(positive_set) == 0:
            mapped_labels.append("1")
            continue
        raise ValueError(f"Unable to map label '{label}' in binary mode.")

    mapped_counter = Counter(mapped_labels)
    return mapped_labels, {
        "positive_labels": sorted(positive_set),
        "negative_labels": sorted(negative_set),
        "original_labels": unique_original,
        "mapped_counts": {key: int(value) for key, value in mapped_counter.items()},
    }


def load_pickle(path: str):
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def save_pickle(path: str, obj) -> None:
    with open(path, "wb") as file_handle:
        pickle.dump(obj, file_handle)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)


def save_csv(path: str, rows: List[Dict]) -> None:
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def load_preselection_genes(dataset: str, preselection: str, k: int) -> List[str]:
    genes_path = _data_path(dataset, f"{preselection}_genes", f"{preselection}_max.tsv")
    if not os.path.isfile(genes_path):
        raise FileNotFoundError(f"Preselection gene file not found: {genes_path}")
    gene_table = pd.read_csv(genes_path, sep="\t")
    if "gene" not in gene_table.columns:
        raise ValueError(f"Expected 'gene' column in {genes_path}")
    genes = gene_table["gene"].astype(str).str.upper().tolist()
    if len(genes) == 0:
        raise ValueError(f"No genes found in {genes_path}")
    target_k = int(k)
    if len(genes) < target_k:
        raise ValueError(f"Not enough genes in {genes_path} for k={target_k}: only {len(genes)}")
    selected = genes[:target_k]

    # Backward compatibility for legacy NP_{k}.tsv files that were saved with >k rows
    # or include genes without DEG regulation labels.
    if str(preselection).upper() == "NP":
        regulation_gene_set = _load_regulation_gene_set(dataset)
        if regulation_gene_set is not None:
            missing_in_selected = [gene for gene in selected if gene not in regulation_gene_set]
            if len(missing_in_selected) > 0 or len(genes) > target_k:
                filtered_ranked = [gene for gene in genes if gene in regulation_gene_set]
                if len(filtered_ranked) < target_k:
                    raise ValueError(
                        f"NP gene file cannot provide k={target_k} genes with regulation labels. "
                        f"Available={len(filtered_ranked)} from {genes_path}"
                    )
                return filtered_ranked[:target_k]

    return selected


def _load_regulation_gene_set(dataset: str) -> Optional[set]:
    merged_path = _data_path(dataset, "DEG_genes", "DEG_merged.tsv")
    if os.path.isfile(merged_path):
        table = pd.read_csv(merged_path, sep="\t")
        if table.shape[1] > 0:
            if "gene" in table.columns:
                genes = table["gene"]
            else:
                genes = table.iloc[:, 0]
            return set(genes.astype(str).str.upper().tolist())

    candidate_paths = [
        _data_path(dataset, "DEG_genes", "DEG_pass_max_metrics.tsv"),
        _data_path(dataset, "DEG_genes", "DEG_max_metrics.tsv"),
    ]
    for regulation_path in candidate_paths:
        if not os.path.isfile(regulation_path):
            continue
        regulation_table = pd.read_csv(regulation_path, sep="\t")
        if "gene" not in regulation_table.columns:
            continue
        return set(regulation_table["gene"].astype(str).str.upper().tolist())
    return None


def load_preselection_scores(dataset: str, preselection: str, k: int, genes: Sequence[str]) -> np.ndarray:
    if preselection in {"HVG", "NP"}:
        scores_path = _data_path(dataset, f"{preselection}_genes", f"{preselection}_max.tsv")
        score_column = "score"
    elif preselection == "DEG":
        scores_path = _data_path(dataset, "DEG_genes", f"DEG_{k}_metrics.tsv")
        score_column = "abs_logfc"
    else:
        raise ValueError(f"Unknown preselection mode: {preselection}")

    score_table = pd.read_csv(scores_path, sep="\t")
    if "gene" not in score_table.columns or score_column not in score_table.columns:
        raise ValueError(f"Expected columns 'gene' and '{score_column}' in {scores_path}")
    score_table["gene"] = score_table["gene"].astype(str).str.upper()
    score_map = dict(zip(score_table["gene"], score_table[score_column]))

    scores = []
    missing = []
    for gene in genes:
        if gene not in score_map:
            missing.append(gene)
        else:
            scores.append(float(score_map[gene]))
    if missing:
        raise ValueError(f"Missing preselection scores for {len(missing)} genes: {missing[:5]}")
    return min_max_scale(scores)


def load_regulation_values(dataset: str, genes: Sequence[str]) -> np.ndarray:
    candidate_paths = [
        _data_path(dataset, "DEG_genes", "DEG_pass_max_metrics.tsv"),
        _data_path(dataset, "DEG_genes", "DEG_max_metrics.tsv"),
    ]

    missing_best: Optional[List[str]] = None
    selected_path: Optional[str] = None
    for regulation_path in candidate_paths:
        if not os.path.isfile(regulation_path):
            continue
        regulation_table = pd.read_csv(regulation_path, sep="\t")
        if "gene" not in regulation_table.columns or "regulation" not in regulation_table.columns:
            raise ValueError(f"Expected columns 'gene' and 'regulation' in {regulation_path}")

        regulation_table["gene"] = regulation_table["gene"].astype(str).str.upper()
        regulation_map = dict(zip(regulation_table["gene"], regulation_table["regulation"]))

        regulation_values = []
        missing = []
        for gene in genes:
            if gene not in regulation_map:
                missing.append(gene)
            else:
                regulation_values.append(float(regulation_map[gene]))

        if len(missing) == 0:
            return np.asarray(regulation_values, dtype=np.float32)

        if missing_best is None or len(missing) < len(missing_best):
            missing_best = missing
            selected_path = regulation_path

    if selected_path is None:
        # Fall back to zeros if regulation metrics are not available.
        return np.zeros((len(genes),), dtype=np.float32)
    raise ValueError(
        f"Missing regulation entries for {len(missing_best)} genes in {selected_path}: {missing_best[:5]}"
    )


def load_celltype_regulation_values(
    dataset: str,
    genes: Sequence[str],
    celltype_names: Sequence[str],
) -> np.ndarray:
    gene_keys = [str(g).upper() for g in genes]
    regulation_matrix = np.zeros((len(celltype_names), len(genes)), dtype=np.float32)
    for idx, celltype in enumerate(celltype_names):
        safe_celltype = sanitize_filename_component(str(celltype))
        metrics_path = _data_path(dataset, "DEG_genes", f"DEG_metrics_{safe_celltype}_pass.tsv")
        if not os.path.isfile(metrics_path):
            continue
        table = pd.read_csv(metrics_path, sep="\t")
        if "gene" not in table.columns or "regulation" not in table.columns:
            raise ValueError(f"Expected columns 'gene' and 'regulation' in {metrics_path}")
        table["gene"] = table["gene"].astype(str).str.upper()
        regulation_map = dict(zip(table["gene"], table["regulation"]))
        for gene_idx, gene in enumerate(gene_keys):
            value = regulation_map.get(gene, 0.0)
            regulation_matrix[idx, gene_idx] = float(value)
    return regulation_matrix


def map_genes_to_adata(adata, genes: Sequence[str]) -> List[int]:
    adata_genes = [str(gene).upper() for gene in adata.var_names]
    gene_to_index = {gene: index for index, gene in enumerate(adata_genes)}
    indices = []
    missing = []
    for gene in genes:
        index = gene_to_index.get(gene)
        if index is None:
            missing.append(gene)
        else:
            indices.append(int(index))
    if missing:
        raise ValueError(f"Missing {len(missing)} genes in AnnData: {missing[:5]}")
    return indices


def build_edge_index(
    genes: Sequence[str], ppi_path: str, self_loop: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    ppi_table = pd.read_csv(ppi_path, sep="\t")
    if "protein1" not in ppi_table.columns or "protein2" not in ppi_table.columns:
        raise ValueError(f"Expected columns 'protein1' and 'protein2' in {ppi_path}")

    sources = ppi_table["protein1"].astype(str).str.upper()
    targets = ppi_table["protein2"].astype(str).str.upper()

    edges: List[Tuple[int, int]] = []
    for source, target in zip(sources, targets):
        if source in gene_to_index and target in gene_to_index:
            source_index = gene_to_index[source]
            target_index = gene_to_index[target]
            edges.append((source_index, target_index))
            edges.append((target_index, source_index))

    if self_loop:
        for index in range(len(genes)):
            edges.append((index, index))

    if len(edges) == 0:
        raise ValueError("No edges were retained from the PPI prior.")

    edge_index = np.array(edges, dtype=np.int64).T
    order = np.lexsort((edge_index[0], edge_index[1]))
    edge_index = edge_index[:, order]
    edge_tensor = torch.tensor(edge_index, dtype=torch.long)
    self_loop_mask = edge_tensor[0] == edge_tensor[1]
    return edge_tensor, self_loop_mask


DEFAULT_PROTEIN_EMBEDDING_PATHS = {
    "ESM3": _data_path("gene_embedding", "ESM3_embeddings.pkl"),
    "GPT": _data_path("gene_embedding", "GPT_embeddings.pkl"),
    "node2vec": _data_path("gene_embedding", "Node2vec_embeddings.pkl"),
}
DEFAULT_PREINTEGRATED_EMBEDDING_PATHS = {
    "concat": _data_path("gene_embedding", "concat_embeddings.pkl"),
    "concat_128": _data_path("gene_embedding", "concat_128_embeddings.pkl"),
    "mean": _data_path("gene_embedding", "mean_embeddings.pkl"),
    "simclr": _data_path("gene_embedding", "simclr_embeddings.pkl"),
}
NODE2VEC_FALLBACK_PATHS = [
    _data_path("gene_embedding", "gene_embedding.pkl"),
]


def _hash_gene_list(genes: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for gene in genes:
        hasher.update(str(gene).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _file_signature(path: str) -> Dict[str, object]:
    absolute_path = os.path.abspath(str(path))
    if not os.path.exists(absolute_path):
        return {"path": absolute_path, "missing": True}
    stat = os.stat(absolute_path)
    return {
        "path": absolute_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_protein_embedding_cache_path(config) -> str:
    explicit_path = str(getattr(config, "protein_embedding_cache_path", "")).strip()
    if explicit_path:
        return explicit_path
    dataset_name = str(getattr(config, "dataset", "dataset"))
    preselection_name = str(getattr(config, "preselection", "PRESELECTION"))
    k_value = str(getattr(config, "k", "k"))
    return os.path.join(
        _data_path("gene_embedding", "cache"),
        dataset_name,
        f"protein_embedding_{preselection_name}_{k_value}.pt",
    )


def _resolve_preintegrated_embedding_path(merge_mode: str) -> Optional[str]:
    mode = str(merge_mode).lower()
    return DEFAULT_PREINTEGRATED_EMBEDDING_PATHS.get(mode, None)


def _build_protein_embedding_cache_key(genes: Sequence[str], config) -> Dict[str, object]:
    embedding_paths = getattr(config, "protein_embedding_paths", None)
    source_list = list(getattr(config, "protein_embedding_sources", []) or [])
    protein_embedding_name = str(getattr(config, "protein_embedding", "GenePT"))
    merge_mode = str(getattr(config, "protein_embedding_merge", "mean")).lower()

    key: Dict[str, object] = {
        "format_version": 2,
        "num_genes": int(len(genes)),
        "genes_sha256": _hash_gene_list(genes),
        "protein_embedding": protein_embedding_name,
        "protein_embedding_sources": [str(source) for source in source_list],
        "protein_embedding_merge": merge_mode,
    }

    if source_list:
        preintegrated_path = _resolve_preintegrated_embedding_path(merge_mode)
        key["preintegrated_embedding_file"] = (
            _file_signature(preintegrated_path) if preintegrated_path is not None else None
        )
    else:
        if isinstance(embedding_paths, dict):
            key["protein_embedding_paths"] = {
                str(path_key): _file_signature(path_value)
                for path_key, path_value in sorted(embedding_paths.items(), key=lambda item: str(item[0]))
            }
        else:
            key["protein_embedding_paths"] = None

        canonical_name = _canonical_embedding_name(protein_embedding_name)
        resolved_path = _resolve_embedding_path(canonical_name, embedding_paths)
        key["resolved_source_signature"] = _file_signature(resolved_path)

    return key


def _load_protein_embedding_matrix_cache(
    cache_path: str,
    expected_key: Dict[str, object],
    expected_num_genes: int,
) -> Optional[torch.Tensor]:
    if not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key", None) != expected_key:
        return None
    matrix = payload.get("matrix", None)
    if not isinstance(matrix, torch.Tensor):
        return None
    if matrix.dim() != 2 or int(matrix.shape[0]) != int(expected_num_genes):
        return None
    return matrix.float().contiguous()
def _canonical_embedding_name(embedding_name: str) -> str:
    name = str(embedding_name).strip().lower()
    if name in {"esm3"}:
        return "ESM3"
    if name in {"genept", "gpt"}:
        return "GPT"
    if name in {"node2vec", "n2v"}:
        return "node2vec"
    raise ValueError(f"Unknown protein embedding type: {embedding_name}")


def _load_embedding_object(path: str):
    path_lower = path.lower()
    if path_lower.endswith((".pt", ".pth")):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    try:
        with open(path, "rb") as file_handle:
            return pickle.load(file_handle)
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _to_numpy_1d(vector) -> np.ndarray:
    if isinstance(vector, torch.Tensor):
        array = vector.detach().cpu().numpy()
    else:
        array = np.asarray(vector)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _normalize_embedding_dict(embedding_object) -> Dict[str, np.ndarray]:
    if not isinstance(embedding_object, dict):
        raise ValueError(f"Protein embedding file must contain a dict, got {type(embedding_object)}")

    embedding_dict: Dict[str, np.ndarray] = {}
    for gene, vector in embedding_object.items():
        gene_key = str(gene).upper()
        embedding_dict[gene_key] = _to_numpy_1d(vector)
    return embedding_dict


def _load_embedding_dict_from_path(path: str) -> Dict[str, np.ndarray]:
    embedding_object = _load_embedding_object(path)
    return _normalize_embedding_dict(embedding_object)


def _save_embedding_dict_to_path(
    path: str,
    genes: Sequence[str],
    embedding_matrix: torch.Tensor,
) -> None:
    matrix = embedding_matrix.detach().cpu().numpy().astype(np.float32)
    if int(matrix.shape[0]) != len(genes):
        raise ValueError(
            f"Embedding rows ({int(matrix.shape[0])}) do not match gene count ({len(genes)})."
        )
    embedding_dict = {str(gene).upper(): matrix[index] for index, gene in enumerate(genes)}
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as file_handle:
        pickle.dump(embedding_dict, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_embedding_path(embedding_name: str, embedding_paths: Optional[Dict[str, str]] = None) -> str:
    if isinstance(embedding_paths, dict):
        for key, value in embedding_paths.items():
            key_text = str(key).strip()
            if key_text == "":
                continue
            if key_text == str(embedding_name).strip():
                return str(value)
            try:
                if _canonical_embedding_name(key_text) == _canonical_embedding_name(embedding_name):
                    return str(value)
            except ValueError:
                if key_text.lower() == str(embedding_name).strip().lower():
                    return str(value)
    canonical_name = _canonical_embedding_name(embedding_name)
    return DEFAULT_PROTEIN_EMBEDDING_PATHS[canonical_name]


def load_protein_embedding_dict(
    embedding_name: str,
    embedding_path: Optional[str] = None,
    embedding_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    if embedding_path:
        resolved_path = str(embedding_path)
        canonical_name = str(embedding_name).strip()
    else:
        try:
            canonical_name = _canonical_embedding_name(embedding_name)
        except ValueError:
            canonical_name = str(embedding_name).strip()
        try:
            resolved_path = _resolve_embedding_path(canonical_name, embedding_paths)
        except ValueError as error:
            raise ValueError(
                "Unknown protein embedding source name. "
                "Provide a matching entry in protein_embedding_paths / embedding_views for arbitrary view names. "
                f"source={embedding_name!r}"
            ) from error

    candidate_paths = [resolved_path]
    if canonical_name == "node2vec":
        candidate_paths.extend([path for path in NODE2VEC_FALLBACK_PATHS if path not in candidate_paths])

    last_error: Optional[Exception] = None
    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            last_error = FileNotFoundError(f"Protein embedding file not found: {candidate_path}")
            continue
        try:
            embedding_object = _load_embedding_object(candidate_path)
            return _normalize_embedding_dict(embedding_object)
        except Exception as error:  # pragma: no cover - fallback path error propagation
            last_error = error
            continue

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Unable to load protein embedding file for {canonical_name}.")


def build_embedding_matrix(genes: Sequence[str], embedding_dict: Dict[str, np.ndarray]) -> torch.Tensor:
    missing = [gene for gene in genes if gene not in embedding_dict]
    if missing:
        raise ValueError(f"Missing protein embeddings for {len(missing)} genes: {missing[:5]}")

    vectors = [embedding_dict[gene] for gene in genes]
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1_weighted(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(precision), float(recall), float(f1)


def auroc_weighted(y_true: Sequence[int], y_proba: np.ndarray) -> float:
    try:
        y_true_array = np.asarray(y_true, dtype=np.int64)
        y_proba_array = np.asarray(y_proba, dtype=np.float64)
        unique_labels = np.unique(y_true_array)
        if unique_labels.shape[0] < 2:
            return float("nan")

        if y_proba_array.ndim == 1:
            # Binary probability for positive class.
            return float(roc_auc_score(y_true_array, y_proba_array))

        if y_proba_array.ndim == 2 and y_proba_array.shape[1] == 2:
            # Binary two-column probability matrix -> use positive column.
            return float(roc_auc_score(y_true_array, y_proba_array[:, 1]))

        return float(
            roc_auc_score(
                y_true_array,
                y_proba_array,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        return float("nan")


def auprc_weighted(y_true: Sequence[int], y_proba: np.ndarray) -> float:
    try:
        y_true_array = np.asarray(y_true, dtype=np.int64)
        y_proba_array = np.asarray(y_proba, dtype=np.float64)
        unique_labels = np.unique(y_true_array)
        if unique_labels.shape[0] < 2:
            return float("nan")

        if y_proba_array.ndim == 1:
            return float(average_precision_score(y_true_array, y_proba_array))

        if y_proba_array.ndim == 2 and y_proba_array.shape[1] == 2:
            return float(average_precision_score(y_true_array, y_proba_array[:, 1]))

        num_classes = y_proba_array.shape[1]
        one_hot = np.eye(num_classes)[y_true_array]
        return float(average_precision_score(one_hot, y_proba_array, average="weighted"))
    except ValueError:
        return float("nan")


def load_pathway_gene_sets(
    path: str,
    gene_names: Sequence[str],
) -> Tuple[List[str], Dict[str, torch.LongTensor]]:
    """Load pathway sets and return deterministic names + preselection-aligned gene indices."""
    if not path:
        raise ValueError("pathway_gene_set_path must be provided when pathway masking is enabled.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pathway gene set file not found: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file_handle:
            pathway_object = json.load(file_handle)
    elif path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as file_handle:
            pathway_object = pickle.load(file_handle)
    elif path.endswith(".pt"):
        pathway_object = torch.load(path, map_location="cpu")
    else:
        with open(path, "r", encoding="utf-8") as file_handle:
            pathway_object = json.load(file_handle)

    if not isinstance(pathway_object, dict):
        raise ValueError(f"Pathway gene set file must contain a dict, got {type(pathway_object)}")

    gene_to_index = {str(gene).upper(): index for index, gene in enumerate(gene_names)}
    ordered_gene_names = [str(gene).upper() for gene in gene_names]

    pathway_names: List[str] = []
    pathway_dict: Dict[str, torch.LongTensor] = {}
    for pathway_name in sorted(pathway_object.keys()):
        raw_genes = pathway_object[pathway_name]
        if raw_genes is None:
            continue
        gene_set = {str(gene).upper() for gene in raw_genes}
        indices = [gene_to_index[gene] for gene in ordered_gene_names if gene in gene_set]
        if len(indices) == 0:
            continue
        pathway_name_text = str(pathway_name)
        pathway_names.append(pathway_name_text)
        pathway_dict[pathway_name_text] = torch.tensor(indices, dtype=torch.long)

    return pathway_names, pathway_dict


def compute_group_mean_expression(
    adata,
    gene_names: Sequence[str],
    grouping_cols: Sequence[str],
) -> Tuple[List, List[int], np.ndarray]:
    """Compute mean expression per group for selected genes."""
    if isinstance(grouping_cols, str):
        grouping_cols = [grouping_cols]
    for column in grouping_cols:
        if column not in adata.obs.columns:
            raise ValueError(f"Grouping column not found in AnnData: {column}")

    group_values = adata.obs[grouping_cols].astype(str)
    if len(grouping_cols) == 1:
        group_keys_all = group_values.iloc[:, 0].tolist()
        group_keys = _sorted_unique(group_keys_all)
        group_key_to_index = {key: index for index, key in enumerate(group_keys)}
        group_indices = np.array([group_key_to_index[key] for key in group_keys_all], dtype=np.int64)
    else:
        group_keys_all = list(zip(*[group_values[col].tolist() for col in grouping_cols]))
        group_keys = sorted(set(group_keys_all))
        group_key_to_index = {key: index for index, key in enumerate(group_keys)}
        group_indices = np.array([group_key_to_index[key] for key in group_keys_all], dtype=np.int64)

    gene_indices = map_genes_to_adata(adata, gene_names)
    expression_matrix = adata[:, gene_indices].X

    num_groups = len(group_keys)
    num_genes = len(gene_names)
    mean_matrix = np.zeros((num_groups, num_genes), dtype=np.float32)
    group_counts: List[int] = []
    for group_index in range(num_groups):
        row_mask = group_indices == group_index
        count = int(row_mask.sum())
        group_counts.append(count)
        if count == 0:
            continue
        if sparse.issparse(expression_matrix):
            group_mean = expression_matrix[row_mask].mean(axis=0)
            mean_matrix[group_index] = np.asarray(group_mean).ravel()
        else:
            mean_matrix[group_index] = np.asarray(expression_matrix[row_mask]).mean(axis=0)

    return group_keys, group_counts, mean_matrix


def apply_pathway_mask(
    expression_matrix: torch.Tensor,
    gene_index_tensor: torch.LongTensor,
    baseline_mode: str,
    baseline_matrix: Optional[torch.Tensor],
    group_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """Replace masked genes with a chosen baseline."""
    if gene_index_tensor.numel() == 0:
        return expression_matrix

    gene_index_tensor = gene_index_tensor.to(expression_matrix.device)
    if baseline_mode == "celltype_mean":
        if baseline_matrix is None or group_index is None:
            raise ValueError("baseline_matrix and group_index required for celltype_mean masking.")
        baseline = baseline_matrix.to(expression_matrix.device)[group_index]
    elif baseline_mode == "batch_mean":
        baseline = expression_matrix.mean(dim=0, keepdim=True).expand_as(expression_matrix)
    elif baseline_mode == "zero":
        baseline = torch.zeros_like(expression_matrix)
    else:
        raise ValueError(f"Unknown baseline mode: {baseline_mode}")

    masked = expression_matrix.clone()
    masked[:, gene_index_tensor] = baseline[:, gene_index_tensor]
    return masked


def load_config_from_path(config_path: str) -> SimpleNamespace:
    if str(config_path).lower().endswith((".yaml", ".yml")):
        return dict_to_namespace(load_workflow_config_dict(config_path=config_path))

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_dict = getattr(module, "config", None)
    if not isinstance(config_dict, dict):
        legacy_configuration = getattr(module, "configuration", None)
        if isinstance(legacy_configuration, dict):
            config_dict = legacy_configuration
        else:
            raise ValueError("Config file must define a dictionary named `config`.")
    return dict2namespace(config_dict)


def resolve_obs_column_name(
    adata,
    configured_name: Optional[str],
    config_key: str,
    required: bool,
) -> Optional[str]:
    if configured_name is None or str(configured_name).strip() == "":
        if required:
            raise ValueError(f"Config key '{config_key}' must be set.")
        return None

    configured = str(configured_name)
    if configured in adata.obs.columns:
        return configured

    lowered = configured.lower()
    lower_to_actual: Dict[str, List[str]] = {}
    for column_name in adata.obs.columns:
        key = str(column_name).lower()
        lower_to_actual.setdefault(key, []).append(str(column_name))
    matches = lower_to_actual.get(lowered, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous obs column for config key '{config_key}'='{configured}': matches={matches}"
        )

    if required:
        raise ValueError(
            f"Missing obs column for config key '{config_key}'='{configured}'. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    return None


def logits_to_positive_probability(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim == 1:
        return 1.0 / (1.0 + np.exp(-array))
    if array.ndim == 2 and array.shape[1] == 1:
        values = array[:, 0]
        return 1.0 / (1.0 + np.exp(-values))
    if array.ndim == 2 and array.shape[1] == 2:
        max_values = np.max(array, axis=1, keepdims=True)
        exp_values = np.exp(array - max_values)
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probs[:, 1]
    raise ValueError("logits must have shape [N], [N,1], or [N,2].")


def compute_binary_metrics(y_true: Sequence[int], y_positive: Sequence[float]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pos_array = np.asarray(y_positive, dtype=np.float64)
    y_pred_array = (y_pos_array >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array, y_pred_array, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    try:
        if np.unique(y_true_array).shape[0] >= 2:
            metrics["auroc"] = float(roc_auc_score(y_true_array, y_pos_array))
            metrics["auprc"] = float(average_precision_score(y_true_array, y_pos_array))
    except ValueError:
        pass
    return metrics


def compute_binary_log_loss(
    y_true: Sequence[int],
    y_positive: Sequence[float],
    eps: float = 1e-7,
) -> float:
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pos_array = np.asarray(y_positive, dtype=np.float64)
    if y_true_array.shape[0] == 0:
        return float("nan")
    clipped = np.clip(y_pos_array, float(eps), 1.0 - float(eps))
    loss = -(y_true_array * np.log(clipped) + (1.0 - y_true_array) * np.log(1.0 - clipped))
    return float(np.mean(loss))
