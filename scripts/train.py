"""Patient-level FIND training for scGOAT (GraphCellEncoder + MIL).

- Graph nodes from NP_max genes, edges from PPI induced subgraph on those genes.
- End-to-end train: CellEncoder + MIL + classifier with BCEWithLogitsLoss.
- Split-level cache for graph + split patient records.
- Validation/test metrics are computed at patient level by aggregating repeated bag probabilities.
- Saves run config, step logs, epoch logs, best checkpoint, and MIL top-cell CSV.

Example
```bash
python scripts/train.py --dataset asthma --gpu 1 --split-number 0
python scripts/train.py --dataset asthma_ext --gpu 1 --split-number 0 
```
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from scipy import sparse
from torch import autocast as torch_autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

def autocast_cuda(enabled: bool):
    if not bool(enabled):
        return nullcontext()
    return torch_autocast(device_type="cuda", enabled=enabled)



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.CellEncoder import GraphCellEncoder, TransformerConvCellEncoder
from modules.MultipleInstanceLearning import PatientMILAggregator
from modules.utils import *
from configs.config import *


def ensure_cublas_workspace_config() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_runtime_backends(device: torch.device, deterministic_training: bool) -> None:
    if device.type != "cuda":
        return
    if deterministic_training:
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass



class PatientEmbeddingEMAMemory:
    """MIDAM-lite state: patient-level EMA over MIL patient embeddings."""

    def __init__(self, embedding_dimension: int, decay: float) -> None:
        self.embedding_dimension = int(embedding_dimension)
        self.decay = float(decay)
        if not (0.0 <= self.decay < 1.0):
            raise ValueError("patient_ema_decay must be in [0, 1).")
        self.table: Dict[int, torch.Tensor] = {}

    def lookup(
        self,
        patient_indices: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_items = int(patient_indices.shape[0])
        values = torch.zeros((num_items, self.embedding_dimension), device=device, dtype=dtype)
        mask = torch.zeros((num_items,), device=device, dtype=torch.bool)
        for row_index, patient_index in enumerate(patient_indices.tolist()):
            cached = self.table.get(int(patient_index))
            if cached is None:
                continue
            values[row_index] = cached.to(device=device, dtype=dtype)
            mask[row_index] = True
        return values, mask

    def update(self, patient_indices: np.ndarray, patient_embeddings: torch.Tensor) -> None:
        if int(patient_embeddings.shape[0]) == 0:
            return
        embeddings_cpu = patient_embeddings.detach().to(device="cpu", dtype=torch.float32)
        groups: Dict[int, List[int]] = defaultdict(list)
        for row_index, patient_index in enumerate(patient_indices.tolist()):
            groups[int(patient_index)].append(int(row_index))

        for patient_index, rows in groups.items():
            current_mean = embeddings_cpu[rows].mean(dim=0)
            previous = self.table.get(int(patient_index))
            if previous is None:
                self.table[int(patient_index)] = current_mean
            else:
                self.table[int(patient_index)] = self.decay * previous + (1.0 - self.decay) * current_mean

def _hash_gene_list(genes: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for gene in genes:
        hasher.update(str(gene).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _file_signature(path: str) -> Dict[str, object]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return {"path": abs_path, "missing": True}
    stat = os.stat(abs_path)
    return {
        "path": abs_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_cache(path: str, expected_key: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != expected_key:
        return None
    return payload


def get_dense_rows(matrix, row_indices: np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix[row_indices].toarray().astype(np.float32)
    return np.asarray(matrix[row_indices], dtype=np.float32)


def logits_to_probability_matrix(logits: torch.Tensor) -> np.ndarray:
    if logits.dim() == 1:
        positive = torch.sigmoid(logits)
    elif logits.dim() == 2 and logits.shape[1] == 1:
        positive = torch.sigmoid(logits[:, 0])
    else:
        raise ValueError("Expected logits shape [N] or [N,1] for binary BCE.")
    negative = 1.0 - positive
    return torch.stack([negative, positive], dim=1).detach().cpu().numpy()


def compute_binary_metrics(labels: Sequence[int], probabilities_pos: Sequence[float]) -> Dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    prob_pos_np = np.asarray(probabilities_pos, dtype=np.float64)
    prob_pos_np = np.nan_to_num(prob_pos_np, nan=0.5, posinf=1.0, neginf=0.0)
    prob_pos_np = np.clip(prob_pos_np, 0.0, 1.0)
    if labels_np.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.5,
            "auprc": 0.0,
            "brier": 0.0,
        }

    pred_np = (prob_pos_np >= 0.5).astype(np.int64)
    prob_mat = np.stack([1.0 - prob_pos_np, prob_pos_np], axis=1)
    brier = float(np.mean((prob_pos_np - labels_np.astype(np.float64)) ** 2))
    positive_rate = float(np.mean(labels_np.astype(np.float64)))

    precision, recall, f1 = precision_recall_f1_weighted(labels_np, pred_np)
    auroc_value = float(auroc_weighted(labels_np, prob_mat))
    auprc_value = float(auprc_weighted(labels_np, prob_mat))
    if np.isnan(auroc_value):
        auroc_value = 0.5
    if np.isnan(auprc_value):
        auprc_value = float(positive_rate)
    return {
        "accuracy": float(accuracy(labels_np, pred_np)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc_value),
        "auprc": float(auprc_value),
        "brier": brier,
    }


def metrics_to_json_ready(metrics: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            payload[str(key)] = float(value)
        elif isinstance(value, str):
            payload[str(key)] = str(value)
        else:
            try:
                payload[str(key)] = float(value)  # type: ignore[arg-type]
            except Exception:
                payload[str(key)] = str(value)
    return payload


def aggregate_patient_probabilities(
    patient_indices: Sequence[int],
    labels: Sequence[int],
    probabilities_pos: Sequence[float],
    reduction: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_np = np.asarray(patient_indices, dtype=np.int64)
    label_np = np.asarray(labels, dtype=np.int64)
    prob_np = np.asarray(probabilities_pos, dtype=np.float64)
    if patient_np.size != label_np.size or patient_np.size != prob_np.size:
        raise ValueError("patient_indices, labels, probabilities_pos must have same length.")
    if patient_np.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )

    mode = str(reduction).strip().lower()
    if mode not in {"mean", "median"}:
        raise ValueError("patient probability reduction must be one of {'mean','median'}.")

    grouped_prob: Dict[int, List[float]] = defaultdict(list)
    grouped_label: Dict[int, List[int]] = defaultdict(list)
    for patient_index, label_value, prob_value in zip(patient_np.tolist(), label_np.tolist(), prob_np.tolist()):
        grouped_prob[int(patient_index)].append(float(prob_value))
        grouped_label[int(patient_index)].append(int(label_value))

    unique_patients = np.asarray(sorted(grouped_prob.keys()), dtype=np.int64)
    aggregated_labels = np.zeros((unique_patients.shape[0],), dtype=np.int64)
    aggregated_prob = np.zeros((unique_patients.shape[0],), dtype=np.float64)
    for idx, patient_index in enumerate(unique_patients.tolist()):
        label_values = np.asarray(grouped_label[int(patient_index)], dtype=np.int64)
        values, counts = np.unique(label_values, return_counts=True)
        aggregated_labels[idx] = int(values[np.argmax(counts)])

        prob_values = np.asarray(grouped_prob[int(patient_index)], dtype=np.float64)
        if mode == "median":
            aggregated_prob[idx] = float(np.median(prob_values))
        else:
            aggregated_prob[idx] = float(np.mean(prob_values))
    return unique_patients, aggregated_labels, aggregated_prob


def metric_improved(current_value: float, best_value: float, mode: str) -> bool:
    if np.isnan(current_value):
        return False
    if np.isnan(best_value):
        return True
    if str(mode).lower() == "min":
        return current_value < best_value
    return current_value > best_value


def resolve_model_selection_metric(
    metric_name: str,
    default_metric_key: str,
    default_mode: str,
) -> Tuple[str, str]:
    normalized_name = str(metric_name).strip().lower()
    if normalized_name.startswith("val_"):
        normalized_name = normalized_name[4:]

    max_metric_aliases = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auroc": "auroc",
        "auprc": "auprc",
    }
    min_metric_aliases = {
        "loss": "loss",
        "bce": "loss",
        "nll": "loss",
        "task_loss": "task_loss",
        "total_loss": "total_loss",
        "brier": "brier",
        "brier_score": "brier",
    }

    if normalized_name in max_metric_aliases:
        return str(max_metric_aliases[normalized_name]), "max"
    if normalized_name in min_metric_aliases:
        return str(min_metric_aliases[normalized_name]), "min"
    return str(default_metric_key), str(default_mode).strip().lower()


def get_validation_metric_value(metrics: Dict[str, float], metric_key: str) -> float:
    key = str(metric_key).strip().lower()
    if key == "task_loss":
        return float(metrics.get("task_loss", metrics.get("loss", float("nan"))))
    if key == "total_loss":
        return float(metrics.get("total_loss", metrics.get("loss", float("nan"))))
    return float(metrics.get(key, float("nan")))


def is_within_primary_epsilon_band(
    metric_value: float,
    best_primary_value: float,
    mode: str,
    epsilon: float,
) -> bool:
    if np.isnan(metric_value) or np.isnan(best_primary_value):
        return False
    eps = max(0.0, float(epsilon))
    if str(mode).lower() == "min":
        return float(metric_value) <= float(best_primary_value) + eps
    return float(metric_value) >= float(best_primary_value) - eps


def linear_warmup_value(
    epoch: int,
    start_value: float,
    end_value: float,
    num_steps: int,
) -> float:
    """Linear schedule per epoch (1-based epoch index)."""
    steps = int(num_steps)
    if steps <= 1:
        return float(end_value)
    t = (float(max(1, int(epoch))) - 1.0) / float(steps - 1)
    t = float(np.clip(t, 0.0, 1.0))
    return float(start_value) + (float(end_value) - float(start_value)) * t


def append_csv_row(path: str, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def optimizer_trainable_parameters(optimizer: Optional[torch.optim.Optimizer]) -> List[torch.nn.Parameter]:
    if optimizer is None:
        return []
    parameters: List[torch.nn.Parameter] = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            if parameter is None:
                continue
            if not isinstance(parameter, torch.nn.Parameter):
                continue
            if not bool(parameter.requires_grad):
                continue
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            parameters.append(parameter)
    return parameters


def build_patient_bag_records(
    cell_indices: Sequence[int],
    label_indices: np.ndarray,
    patient_indices: np.ndarray,
) -> List[Dict[str, object]]:
    patient_to_cells: Dict[int, List[int]] = defaultdict(list)
    for cell_index in cell_indices:
        patient_to_cells[int(patient_indices[int(cell_index)])].append(int(cell_index))

    records: List[Dict[str, object]] = []
    for patient_index in sorted(patient_to_cells.keys()):
        bag_cells = np.asarray(sorted(patient_to_cells[patient_index]), dtype=np.int64)
        bag_labels = label_indices[bag_cells]
        values, counts = np.unique(bag_labels, return_counts=True)
        patient_label = int(values[np.argmax(counts)])
        records.append(
            {
                "patient_index": int(patient_index),
                "patient_label": int(patient_label),
                "cell_indices": bag_cells,
                "bag_repeat_index": 0,
            }
        )
    return records


def expand_patient_bag_records(
    base_records: Sequence[Dict[str, object]],
    bags_per_patient: int,
) -> List[Dict[str, object]]:
    repeat_count = max(1, int(bags_per_patient))
    if repeat_count == 1:
        return [dict(record) for record in base_records]

    expanded: List[Dict[str, object]] = []
    for record in base_records:
        for repeat_index in range(repeat_count):
            copied = dict(record)
            copied["bag_repeat_index"] = int(repeat_index)
            expanded.append(copied)
    return expanded


class PatientBagDataset(Dataset):
    def __init__(
        self,
        expression_matrix,
        bag_records: Sequence[Dict[str, object]],
        celltype_indices: np.ndarray,
        treatment_indices: np.ndarray,
    ) -> None:
        self.expression_matrix = expression_matrix
        self.bag_records = list(bag_records)
        self.celltype_indices = np.asarray(celltype_indices, dtype=np.int64)
        self.treatment_indices = np.asarray(treatment_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.bag_records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.bag_records[index]
        cell_indices = np.asarray(record["cell_indices"], dtype=np.int64)
        expression = get_dense_rows(self.expression_matrix, cell_indices)
        celltype = self.celltype_indices[cell_indices]
        treatment = self.treatment_indices[cell_indices]
        return {
            "expression": torch.tensor(expression, dtype=torch.float32),
            "celltype_index": torch.tensor(celltype, dtype=torch.long),
            "treatment_index": torch.tensor(treatment, dtype=torch.long),
            "patient_label": torch.tensor(int(record["patient_label"]), dtype=torch.long),
            "patient_index": torch.tensor(int(record["patient_index"]), dtype=torch.long),
            "bag_repeat_index": torch.tensor(int(record.get("bag_repeat_index", 0)), dtype=torch.long),
            "cell_indices": torch.tensor(cell_indices, dtype=torch.long),
        }


def collate_patient_bags(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    expression_list: List[torch.Tensor] = []
    celltype_list: List[torch.Tensor] = []
    treatment_list: List[torch.Tensor] = []
    bag_index_list: List[torch.Tensor] = []
    cell_indices_list: List[torch.Tensor] = []
    patient_labels: List[torch.Tensor] = []
    patient_indices: List[torch.Tensor] = []
    bag_repeat_indices: List[torch.Tensor] = []

    for bag_id, item in enumerate(batch):
        num_cells = int(item["expression"].shape[0])
        expression_list.append(item["expression"])
        celltype_list.append(item["celltype_index"])
        treatment_list.append(item["treatment_index"])
        bag_index_list.append(torch.full((num_cells,), bag_id, dtype=torch.long))
        cell_indices_list.append(item["cell_indices"])
        patient_labels.append(item["patient_label"])
        patient_indices.append(item["patient_index"])
        bag_repeat_indices.append(item["bag_repeat_index"])

    return {
        "expression": torch.cat(expression_list, dim=0),
        "celltype_index": torch.cat(celltype_list, dim=0),
        "treatment_index": torch.cat(treatment_list, dim=0),
        "bag_index": torch.cat(bag_index_list, dim=0),
        "num_bags": torch.tensor(len(batch), dtype=torch.long),
        "patient_labels": torch.stack(patient_labels, dim=0),
        "patient_indices": torch.stack(patient_indices, dim=0),
        "bag_repeat_index": torch.stack(bag_repeat_indices, dim=0),
        "cell_indices": torch.cat(cell_indices_list, dim=0),
    }


def bag_cell_counts_from_index(bag_index: torch.Tensor, num_bags: int) -> List[int]:
    counts = torch.bincount(bag_index.cpu(), minlength=int(num_bags))
    return [int(value) for value in counts.tolist()]


def _sample_without_replacement(num_items: int, take: int, rng: np.random.Generator) -> np.ndarray:
    if num_items <= 0 or take <= 0:
        return np.zeros((0,), dtype=np.int64)
    if take >= num_items:
        return np.arange(num_items, dtype=np.int64)
    return np.sort(rng.choice(num_items, size=take, replace=False).astype(np.int64))


def _allocate_proportional_quotas(type_counts: Dict[int, int], target_size: int) -> Dict[int, int]:
    if target_size <= 0 or len(type_counts) == 0:
        return {key: 0 for key in type_counts.keys()}

    total = float(sum(int(value) for value in type_counts.values()))
    if total <= 0:
        return {key: 0 for key in type_counts.keys()}

    raw = {key: target_size * (float(count) / total) for key, count in type_counts.items()}
    quota = {key: min(int(np.floor(raw[key])), int(type_counts[key])) for key in type_counts.keys()}
    assigned = int(sum(quota.values()))
    remainder = max(0, int(target_size - assigned))

    if remainder > 0:
        fractional_keys = sorted(
            type_counts.keys(),
            key=lambda key: (-(raw[key] - np.floor(raw[key])), -int(type_counts[key]), int(key)),
        )
        for key in fractional_keys:
            if remainder <= 0:
                break
            available = int(type_counts[key]) - int(quota[key])
            if available <= 0:
                continue
            take = min(available, remainder)
            quota[key] += int(take)
            remainder -= int(take)

    if remainder > 0:
        by_available = sorted(
            type_counts.keys(),
            key=lambda key: (-(int(type_counts[key]) - int(quota[key])), int(key)),
        )
        for key in by_available:
            if remainder <= 0:
                break
            available = int(type_counts[key]) - int(quota[key])
            if available <= 0:
                continue
            take = min(available, remainder)
            quota[key] += int(take)
            remainder -= int(take)

    return quota


def normalize_sample_bags(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"uniform"}:
        return "random"
    if normalized not in {"random", "celltype", "proportional"}:
        raise ValueError("sample mode must be one of {'random','celltype','proportional','uniform'}.")
    return normalized


def sample_patient_bag_indices(
    bag_celltype: torch.Tensor,
    bag_size_cells: int,
    sample_mode: str,
    deterministic_seed: int,
) -> torch.Tensor:
    num_cells = int(bag_celltype.shape[0])
    if num_cells <= 0:
        return torch.zeros((0,), dtype=torch.long)

    target_size = int(bag_size_cells)
    if target_size <= 0:
        target_size = num_cells
    target_size = min(target_size, num_cells)
    if target_size <= 0:
        return torch.zeros((0,), dtype=torch.long)

    rng = np.random.default_rng(int(deterministic_seed))
    mode = normalize_sample_bags(sample_mode)

    if mode == "random":
        selected = _sample_without_replacement(num_cells, target_size, rng)
        return torch.tensor(selected, dtype=torch.long)

    celltype_np = bag_celltype.detach().cpu().numpy().astype(np.int64)
    unique_types = sorted({int(value) for value in celltype_np.tolist()})
    if len(unique_types) == 0:
        selected = _sample_without_replacement(num_cells, target_size, rng)
        return torch.tensor(selected, dtype=torch.long)

    if mode == "celltype":
        picked_type = int(unique_types[int(rng.integers(0, len(unique_types)))])
        type_positions = np.where(celltype_np == picked_type)[0].astype(np.int64)
        local = _sample_without_replacement(int(type_positions.shape[0]), target_size, rng)
        selected = type_positions[local]
        return torch.tensor(np.sort(selected), dtype=torch.long)

    type_to_positions: Dict[int, np.ndarray] = {}
    type_counts: Dict[int, int] = {}
    for type_id in unique_types:
        positions = np.where(celltype_np == int(type_id))[0].astype(np.int64)
        type_to_positions[int(type_id)] = positions
        type_counts[int(type_id)] = int(positions.shape[0])

    quotas = _allocate_proportional_quotas(type_counts=type_counts, target_size=target_size)

    selected_parts: List[np.ndarray] = []
    selected_mask = np.zeros((num_cells,), dtype=bool)
    for type_id in unique_types:
        quota = int(quotas.get(int(type_id), 0))
        if quota <= 0:
            continue
        positions = type_to_positions[int(type_id)]
        local = _sample_without_replacement(int(positions.shape[0]), quota, rng)
        picked = positions[local]
        selected_parts.append(picked)
        selected_mask[picked] = True

    selected = np.concatenate(selected_parts, axis=0) if selected_parts else np.zeros((0,), dtype=np.int64)
    if int(selected.shape[0]) < target_size:
        remaining = np.where(~selected_mask)[0].astype(np.int64)
        need = int(target_size - int(selected.shape[0]))
        if remaining.shape[0] > 0:
            extra_local = _sample_without_replacement(int(remaining.shape[0]), need, rng)
            selected = np.concatenate([selected, remaining[extra_local]], axis=0)

    selected = np.sort(np.unique(selected.astype(np.int64)))
    if int(selected.shape[0]) > target_size:
        local = _sample_without_replacement(int(selected.shape[0]), target_size, rng)
        selected = selected[local]
    return torch.tensor(np.sort(selected), dtype=torch.long)


def sample_patient_bag_inputs(
    bag_expression: torch.Tensor,
    bag_celltype: torch.Tensor,
    bag_treatment: torch.Tensor,
    bag_size_cells: int,
    sample_mode: str,
    deterministic_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_index = sample_patient_bag_indices(
        bag_celltype=bag_celltype,
        bag_size_cells=bag_size_cells,
        sample_mode=sample_mode,
        deterministic_seed=deterministic_seed,
    )
    if int(selected_index.numel()) <= 0:
        full_index = torch.arange(int(bag_expression.shape[0]), dtype=torch.long)
        return bag_expression, bag_celltype, bag_treatment, full_index
    return (
        bag_expression[selected_index],
        bag_celltype[selected_index],
        bag_treatment[selected_index],
        selected_index,
    )


def sample_collated_batch(
    batch: Dict[str, torch.Tensor],
    bag_size_cells: int,
    sample_mode: str,
    seed_base: int,
) -> Dict[str, torch.Tensor]:
    expression = batch["expression"]
    celltype_idx = batch["celltype_index"]
    treatment_idx = batch["treatment_index"]
    cell_indices = batch["cell_indices"]

    num_bags = int(batch["num_bags"].item())
    bag_counts = bag_cell_counts_from_index(batch["bag_index"], num_bags)

    sampled_expression: List[torch.Tensor] = []
    sampled_celltype: List[torch.Tensor] = []
    sampled_treatment: List[torch.Tensor] = []
    sampled_cell_indices: List[torch.Tensor] = []
    sampled_bag_index: List[torch.Tensor] = []

    start = 0
    for bag_id, bag_count in enumerate(bag_counts):
        if bag_count <= 0:
            continue
        end = start + int(bag_count)
        bag_expression = expression[start:end]
        bag_celltype = celltype_idx[start:end]
        bag_treatment = treatment_idx[start:end]
        bag_cell_indices = cell_indices[start:end]

        patient_index = int(batch["patient_indices"][bag_id].item())
        bag_repeat_index = int(batch["bag_repeat_index"][bag_id].item())
        deterministic_seed = (
            int(seed_base)
            + int(patient_index) * 1000003
            + int(bag_repeat_index) * 7919
        )

        bag_expression, bag_celltype, bag_treatment, local_index = sample_patient_bag_inputs(
            bag_expression=bag_expression,
            bag_celltype=bag_celltype,
            bag_treatment=bag_treatment,
            bag_size_cells=bag_size_cells,
            sample_mode=sample_mode,
            deterministic_seed=deterministic_seed,
        )
        sampled_global_indices = bag_cell_indices[local_index]

        sampled_expression.append(bag_expression)
        sampled_celltype.append(bag_celltype)
        sampled_treatment.append(bag_treatment)
        sampled_cell_indices.append(sampled_global_indices)
        sampled_bag_index.append(torch.full((int(bag_expression.shape[0]),), bag_id, dtype=torch.long))
        start = end

    if start != int(expression.shape[0]):
        raise RuntimeError("Sampled batch construction failed due to cell count mismatch.")

    return {
        "expression": torch.cat(sampled_expression, dim=0),
        "celltype_index": torch.cat(sampled_celltype, dim=0),
        "treatment_index": torch.cat(sampled_treatment, dim=0),
        "cell_indices": torch.cat(sampled_cell_indices, dim=0),
        "bag_index": torch.cat(sampled_bag_index, dim=0),
        "num_bags": torch.tensor(num_bags, dtype=torch.long),
        "patient_labels": batch["patient_labels"],
        "patient_indices": batch["patient_indices"],
        "bag_repeat_index": batch["bag_repeat_index"],
    }


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


def check_embedding_coverage(
    ranked_genes: Sequence[str],
    *,
    target_k: int,
    required_sources: Sequence[str],
    embedding_paths: Optional[Dict[str, str]],
    force_fill_unmatched: bool = False,  # <- 필요하면 True로
) -> List[str]:
    ranked = [str(gene).upper() for gene in ranked_genes]
    target_k = int(target_k)
    if target_k <= 0 or len(ranked) == 0:
        return []

    required_sources_clean = _deduplicate_preserve_order(
        [
            _canonical_embedding_source_name(str(source).strip())
            for source in required_sources
            if str(source).strip() != ""
        ]
    )

    # required_sources가 없으면 그냥 상위 k
    if len(required_sources_clean) == 0:
        if len(ranked) < target_k:
            print(
                f"[warn] ranked genes has {len(ranked)} genes (< k={target_k}); using all available genes.",
                flush=True,
            )
            return ranked
        return ranked[:target_k]

    # 소스별 coverage set 로드
    coverage_by_source: Dict[str, set] = {}
    for source_name in required_sources_clean:
        source_dict = load_protein_embedding_dict(source_name, embedding_paths=embedding_paths)
        coverage_by_source[source_name] = set(source_dict.keys())

    selected: List[str] = []
    skipped_by_source: Dict[str, int] = {source_name: 0 for source_name in required_sources_clean}
    unmatched_candidates: List[str] = []

    scanned_count = 0
    for gene_name in ranked:
        scanned_count += 1
        missing_sources = [
            source_name
            for source_name in required_sources_clean
            if gene_name not in coverage_by_source[source_name]
        ]
        if len(missing_sources) > 0:
            for source_name in missing_sources:
                skipped_by_source[source_name] += 1
            unmatched_candidates.append(gene_name)
            continue

        selected.append(gene_name)
        if len(selected) >= target_k:
            break

    # 정말로 k개를 "강제로" 맞추고 싶으면(coverage가 부족해도),
    # coverage 없는 후보를 ranked 순서대로 채워 넣는 fallback
    forced_added = 0
    if force_fill_unmatched and len(selected) < target_k:
        for gene_name in unmatched_candidates:
            if len(selected) >= target_k:
                break
            # 중복 방지 (ranked에 중복이 있을 수 있으니)
            if gene_name in selected:
                continue
            selected.append(gene_name)
            forced_added += 1

    if len(selected) < target_k:
        print(
            "[GeneSelect][warn] "
            f"could not reach target_k={target_k} with required_sources={required_sources_clean}. "
            f"selected_k={len(selected)} scanned={scanned_count}/{len(ranked)} "
            f"forced_added={forced_added} skipped_by_source={skipped_by_source}",
            flush=True,
        )
    else:
        print(
            "[GeneSelect] "
            f"selected_k={len(selected)} scanned={scanned_count}/{len(ranked)} "
            f"required_sources={required_sources_clean} "
            f"forced_added={forced_added} skipped_by_source={skipped_by_source}",
            flush=True,
        )

    return selected[:target_k]

def load_k_np_genes(
    dataset: str,
    k: int,
    preselection_root: Optional[str] = None,
    required_embedding_sources: Optional[Sequence[str]] = None,
    embedding_paths: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[str]]:
    base_root = str(preselection_root) if preselection_root is not None else os.path.join(PROJECT_ROOT, "data", str(dataset))
    np_path = os.path.join(base_root, "NP", "NP_max.tsv")
    if not os.path.isfile(np_path):
        raise FileNotFoundError(f"NP gene file not found: {np_path}")

    table = pd.read_csv(np_path, sep="\t")
    genes = table["gene"].astype(str).str.upper().tolist()

    target_k = int(k)
    if target_k <= 0:
        return [], genes

    if len(genes) < target_k:
        print(
            f"[warn] NP_max has {len(genes)} genes (< k={target_k}); using all available genes.",
            flush=True,
        )
        return genes, genes

    required_sources = [str(name).strip() for name in list(required_embedding_sources or []) if str(name).strip()]
    if len(required_sources) == 0:
        return genes[:target_k], genes

    selected = check_embedding_coverage(
        ranked_genes=genes,
        target_k=target_k,
        required_sources=required_sources,
        embedding_paths=embedding_paths,
        force_fill_unmatched=False,
    )
    if len(selected) == 0:
        raise ValueError(
            "No genes remained after embedding-coverage filtering. "
            f"required_sources={required_sources}"
        )

    return selected, genes


def resolve_training_preselection_root(config: SimpleNamespace) -> str:
    split_number = int(getattr(config, "split_number", 0))
    preselection_output_root = str(
        getattr(config, "preselection_output_root", "") or ""
    ).strip()
    if preselection_output_root == "":
        raise ValueError("config.preselection_output_root must be set.")

    split_root = os.path.join(preselection_output_root, f"split_{split_number}")
    if os.path.isdir(os.path.join(split_root, "NP")) or os.path.isdir(
        os.path.join(split_root, "DEG")
    ):
        return split_root

    raise FileNotFoundError(
        "split-specific preselection directory not found. "
        f"Expected: {split_root}. Run modules/preselection.py first."
    )

def _embedding_source_path_for_logging(
    source_name: str,
    configured_paths: Optional[Dict[str, str]],
) -> Optional[str]:
    source = str(source_name)
    if isinstance(configured_paths, dict) and source in configured_paths:
        return os.path.abspath(str(configured_paths[source]))
    if source in DEFAULT_PROTEIN_EMBEDDING_PATHS:
        return os.path.abspath(str(DEFAULT_PROTEIN_EMBEDDING_PATHS[source]))
    if source in DEFAULT_PREINTEGRATED_EMBEDDING_PATHS:
        return os.path.abspath(str(DEFAULT_PREINTEGRATED_EMBEDDING_PATHS[source]))
    return None


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


def _canonical_embedding_source_name(source_name: str) -> str:
    normalized = str(source_name).strip().lower()
    if normalized in {"esm3"}:
        return "ESM3"
    if normalized in {"genept", "gpt"}:
        return "GPT"
    if normalized in {"node2vec", "n2v"}:
        return "node2vec"
    return str(source_name).strip()


def _resolve_embedding_source_path_for_cache(
    source_name: str,
    configured_paths: Optional[Dict[str, str]],
) -> Optional[str]:
    lowercase_name = source_name.strip().lower()
    if isinstance(configured_paths, dict):
        for key, value in configured_paths.items():
            if key.strip().lower() == lowercase_name:
                return os.path.abspath(str(value))

    default_path = _embedding_source_path_for_logging(lowercase_name, configured_paths=None)
    if default_path is not None and os.path.exists(default_path):
        return default_path
    if lowercase_name == "node2vec":
        for fallback_path in NODE2VEC_FALLBACK_PATHS:
            resolved = os.path.abspath(str(fallback_path))
            if os.path.exists(resolved):
                return resolved
    return default_path


def _smoothness_prior_knn_cache_path(
    *,
    view_names: Sequence[str],
    config: SimpleNamespace,
) -> str:
    explicit_path = str(getattr(config, "smoothness_prior_knn_cache_path", "")).strip()
    if explicit_path:
        return os.path.abspath(explicit_path)

    dataset_name = sanitize_filename_component(str(getattr(config, "dataset", "dataset")))
    split_subdir = sanitize_filename_component(
        str(getattr(config, "split_preselection_subdir", "split_preselection"))
    )
    split_number_raw = config.split_number
    try:
        split_number = int(split_number_raw) if split_number_raw is not None else None
    except Exception:
        split_number = None
    split_token = f"split_idx_{split_number}" if split_number is not None else "split_idx_unknown"

    views_serialized = ",".join([str(name).strip() for name in list(view_names)])
    views_hash = hashlib.sha1(views_serialized.encode("utf-8")).hexdigest()[:12]
    k_value = int(getattr(config, "k", 0))
    topk = int(getattr(config, "smoothness_prior_knn_k", 0))
    filename = f"smoothness_prior_knn_views-{views_hash}_k{k_value}_top{topk}.pt"

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "gene_embedding",
            "cache",
            dataset_name,
            split_subdir,
            split_token,
            filename,
        )
    )


def _build_smoothness_prior_knn_cache_key(
    *,
    genes: Sequence[str],
    view_names: Sequence[str],
    config: SimpleNamespace,
) -> Dict[str, object]:
    configured_paths = _normalized_configured_embedding_paths(config)
    source_file_signatures: List[Dict[str, object]] = []
    for view_name in [str(name).strip() for name in list(view_names)]:
        resolved_path = _resolve_embedding_source_path_for_cache(
            source_name=view_name,
            configured_paths=configured_paths,
        )
        if resolved_path is None:
            signature = {"view": str(view_name), "file": {"unresolved": True}}
        else:
            signature = {"view": str(view_name), "file": _file_signature(resolved_path)}
        source_file_signatures.append(signature)

    split_number_raw = config.split_number
    try:
        split_number_value: object = int(split_number_raw) if split_number_raw is not None else None
    except Exception:
        split_number_value = str(split_number_raw)

    return {
        "format_version": 1,
        "dataset": str(config.dataset),
        "split_number": split_number_value,
        "split_preselection_subdir": str(config.split_preselection_subdir),
        "num_genes": int(len(genes)),
        "genes_sha256": _hash_gene_list(genes),
        "smoothness_prior_knn_k": int(config.smoothness_prior_knn_k),
        "smoothness_similarity_eps": float(config.smoothness_similarity_eps),
        "views": [str(name).strip() for name in list(view_names)],
        "view_embedding_files": source_file_signatures,
    }


def _merged_embedding_cache_path(
    *,
    merge_choice: str,
    config: SimpleNamespace,
) -> str:
    explicit_path = str(config.protein_embedding_merged_cache_path).strip()
    if explicit_path:
        return os.path.abspath(explicit_path)

    dataset_name = sanitize_filename_component(str(getattr(config, "dataset", "dataset")))
    split_subdir = sanitize_filename_component(
        str(getattr(config, "split_preselection_subdir", "split_preselection"))
    )
    split_number_raw = getattr(config, "split_number", None)
    try:
        split_number = int(split_number_raw) if split_number_raw is not None else None
    except Exception:
        split_number = None
    split_token = f"split_idx_{split_number}" if split_number is not None else "split_idx_unknown"
    k_value = int(getattr(config, "k", 0))
    filename = f"{str(merge_choice)}_k{k_value}.pkl"

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "gene_embedding",
            "cache",
            dataset_name,
            split_subdir,
            split_token,
            filename,
        )
    )


def _normalize_embedding_object(embedding_object) -> Dict[str, np.ndarray]:
    if not isinstance(embedding_object, dict):
        raise ValueError(f"Embedding object must be a dict, got {type(embedding_object)}")
    normalized: Dict[str, np.ndarray] = {}
    for gene_name, vector in embedding_object.items():
        if isinstance(vector, torch.Tensor):
            vector_array = vector.detach().cpu().numpy()
        else:
            vector_array = np.asarray(vector)
        normalized[str(gene_name).upper()] = np.asarray(vector_array, dtype=np.float32).reshape(-1)
    return normalized


def normalize_source_embedding_matrix(matrix: torch.Tensor, norm_mode: str) -> torch.Tensor:
    mode = str(norm_mode).strip().lower()
    if mode == "none":
        return matrix.float().contiguous()
    if mode == "layernorm":
        matrix_float = matrix.float()
        return torch_functional.layer_norm(matrix_float, normalized_shape=(int(matrix_float.shape[1]),)).contiguous()
    if mode == "l2":
        return torch_functional.normalize(matrix.float(), p=2.0, dim=1, eps=1e-12).contiguous()
    raise ValueError("norm_mode must be one of {'layernorm','l2','none'}.")


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



def resolve_cell_encoder_spec(
    config: SimpleNamespace,
    num_nodes: int,
) -> Dict[str, object]:
    encoder_name = normalize_cell_encoder_name(str(config.cell_encoder_name))
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

    if int(hidden_dimension) <= 0:
        raise ValueError("cell encoder hidden dimension must be positive.")
    if int(number_of_layers) <= 0:
        raise ValueError("cell encoder number_of_layers must be positive.")
    if int(number_of_heads) <= 0:
        raise ValueError("cell encoder number_of_heads must be positive.")
    if float(dropout_probability) < 0.0:
        raise ValueError("cell encoder dropout must be >= 0.")
    if int(cell_embedding_dimension) <= 0:
        raise ValueError("cell_embedding_dimension must be positive.")

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


def build_protein_embedding_matrix(
    genes: Sequence[str],
    config: SimpleNamespace,
    hidden_dimension: int,
) -> Tuple[torch.Tensor, str, Optional[torch.Tensor], List[str]]:
    choice = str(config.protein_embedding_choice).strip().lower()
    target_hidden_dimension = int(hidden_dimension)
    if target_hidden_dimension <= 0:
        raise ValueError("hidden_dimension must be positive for protein embedding construction.")
    if choice == "none":
        return torch.zeros((len(genes), target_hidden_dimension), dtype=torch.float32), "zeros", None, []

    if choice not in {"concat", "mean"}:
        raise ValueError(
            "protein_embedding_choice must be one of {'concat','mean','none'}. "
            f"Got: {str(config.protein_embedding_choice)!r}"
        )

    configured_paths = _normalized_configured_embedding_paths(config)

    sources = _resolve_embedding_view_sources(config)
    if len(sources) == 0:
        raise ValueError(
            "At least one embedding view is required. "
            "Set gene_embedding_views/protein_embedding_paths or prior_view_sources."
        )
    source_norm_mode = "l2"
    output_norm_mode = "layernorm"
    cache_path = _merged_embedding_cache_path(
        merge_choice=choice,
        config=config,
    )
    cache_meta_path = f"{cache_path}.meta.json"
    expected_cache_meta: Dict[str, object] = {
        "format_version": 1,
        "merge_choice": str(choice),
        "sources": [str(source) for source in sources],
        "source_norm_mode": str(source_norm_mode),
        "output_norm_mode": str(output_norm_mode),
        "num_genes": int(len(genes)),
        "genes_sha256": _hash_gene_list(genes),
    }
    source_paths: List[str] = [cache_path]
    merged_matrix: Optional[torch.Tensor] = None

    if os.path.exists(cache_path):
        try:
            cache_meta_ok = False
            if os.path.exists(cache_meta_path):
                try:
                    with open(cache_meta_path, "r", encoding="utf-8") as file_handle:
                        loaded_meta = json.load(file_handle)
                    cache_meta_ok = bool(loaded_meta == expected_cache_meta)
                except Exception:
                    cache_meta_ok = False
            if not cache_meta_ok:
                raise ValueError("cache metadata mismatch")
            cache_object = load_pickle(cache_path)
            cache_dict = _normalize_embedding_object(cache_object)
            merged_matrix, _ = _build_embedding_matrix_with_missing_fallback(
                genes=genes,
                embedding_dict=cache_dict,
                source_name=f"merged-cache:{choice}",
            )
            merged_matrix = merged_matrix.contiguous()
            merged_matrix = normalize_source_embedding_matrix(merged_matrix, output_norm_mode)
            print(
                f"[ProteinEmbedding] merged-cache hit (source_norm={source_norm_mode}, output_norm={output_norm_mode}): {cache_path}",
                flush=True,
            )
        except Exception as error:
            print(
                f"[ProteinEmbedding] merged-cache invalid -> rebuild ({cache_path}): {error}",
                flush=True,
            )
            merged_matrix = None

    if merged_matrix is None:
        source_matrices: List[torch.Tensor] = []
        raw_source_paths: List[str] = []
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
            source_matrix = normalize_source_embedding_matrix(source_matrix, source_norm_mode)
            source_matrices.append(source_matrix)
            source_path = _embedding_source_path_for_logging(source_name, configured_paths)
            if source_path is not None:
                raw_source_paths.append(source_path)

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
        merged_matrix = normalize_source_embedding_matrix(merged_matrix, output_norm_mode)

        merged_numpy = merged_matrix.detach().cpu().numpy().astype(np.float32)
        merged_dict = {
            str(gene_name).upper(): merged_numpy[row_index]
            for row_index, gene_name in enumerate(genes)
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        save_pickle(cache_path, merged_dict)
        save_json(cache_meta_path, expected_cache_meta)
        print(
            f"[ProteinEmbedding] merged-cache miss: built from sources "
            f"(source_norm={source_norm_mode}, output_norm={output_norm_mode}) "
            f"and saved to {cache_path}",
            flush=True,
        )
        source_paths = [cache_path] + raw_source_paths

    fixed_placeholder = torch.zeros((len(genes), target_hidden_dimension), dtype=torch.float32)
    source_description = (
        f"learnable_{choice}_srcnorm-{source_norm_mode}_outnorm-{output_norm_mode}[{','.join(sources)}]"
    )
    return fixed_placeholder, source_description, merged_matrix, sorted(set(source_paths))


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
            source_name=f"smoothness:{str(view_name)}",
        )
        matrix = matrix.float().contiguous()
        loaded[str(view_name)] = matrix
    return loaded


def compute_prior_knn_graph_multiview(
    prior_embeddings_by_view: Dict[str, torch.Tensor],
    config: SimpleNamespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(prior_embeddings_by_view) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    k = int(config.smoothness_prior_knn_k)
    eps = float(config.smoothness_similarity_eps)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    similarities: Dict[str, torch.Tensor] = {}
    edge_union = set()
    num_genes = int(next(iter(prior_embeddings_by_view.values())).shape[0])
    topk = min(int(k), max(0, int(num_genes) - 1))
    if topk <= 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    for view_name, embedding in prior_embeddings_by_view.items():
        matrix = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if matrix.dim() != 2 or int(matrix.shape[0]) != int(num_genes):
            raise ValueError(f"Invalid prior embedding shape for view '{view_name}': {tuple(matrix.shape)}")
        normalized = torch_functional.normalize(matrix, p=2.0, dim=1, eps=max(eps, 1e-12))
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-float("inf"))
        knn_index = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=False).indices
        for source_idx in range(int(num_genes)):
            for destination_idx in knn_index[source_idx].tolist():
                edge_union.add((int(source_idx), int(destination_idx)))
        similarities[str(view_name)] = similarity

    if len(edge_union) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    edges_sorted = sorted(edge_union)
    source = torch.tensor([edge[0] for edge in edges_sorted], dtype=torch.long)
    destination = torch.tensor([edge[1] for edge in edges_sorted], dtype=torch.long)
    edge_index_prior = torch.stack([source, destination], dim=0).contiguous()

    num_views = float(len(prior_embeddings_by_view))
    edge_weights = torch.zeros((len(edges_sorted),), dtype=torch.float32)
    for edge_pos, (source_idx, destination_idx) in enumerate(edges_sorted):
        score_sum = 0.0
        for similarity in similarities.values():
            value = float(similarity[int(source_idx), int(destination_idx)].item())
            score_sum += max(0.0, value)
        edge_weights[edge_pos] = float(score_sum / max(num_views, 1.0))
    return edge_index_prior, edge_weights.contiguous()


def load_or_build_prior_knn_graph_multiview(
    *,
    genes: Sequence[str],
    view_names: Sequence[str],
    config: SimpleNamespace,
) -> Tuple[torch.Tensor, torch.Tensor, bool, str]:
    views_clean = [str(name).strip() for name in list(view_names) if str(name).strip()]
    if len(views_clean) == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            False,
            "",
        )

    cache_enabled = bool(getattr(config, "smoothness_prior_knn_cache_enabled", True))
    cache_path = _smoothness_prior_knn_cache_path(
        view_names=views_clean,
        config=config,
    )
    cache_key = _build_smoothness_prior_knn_cache_key(
        genes=genes,
        view_names=views_clean,
        config=config,
    )

    if cache_enabled:
        payload = _load_cache(cache_path, cache_key)
        if payload is not None:
            edge_index_cached = payload.get("edge_index_prior", None)
            edge_weight_cached = payload.get("edge_weight_prior", None)
            if isinstance(edge_index_cached, torch.Tensor) and isinstance(edge_weight_cached, torch.Tensor):
                return (
                    edge_index_cached.long().contiguous(),
                    edge_weight_cached.float().contiguous(),
                    True,
                    cache_path,
                )

    prior_embeddings_by_view = load_prior_embeddings_by_view(
        genes=genes,
        view_names=views_clean,
        config=config,
    )
    edge_index_prior, edge_weight_prior = compute_prior_knn_graph_multiview(
        prior_embeddings_by_view=prior_embeddings_by_view,
        config=config,
    )

    if cache_enabled:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                "cache_key": cache_key,
                "edge_index_prior": edge_index_prior.cpu().contiguous(),
                "edge_weight_prior": edge_weight_prior.cpu().contiguous(),
            },
            cache_path,
        )

    return edge_index_prior.long().contiguous(), edge_weight_prior.float().contiguous(), False, cache_path


def select_node_hidden_for_smoothness(
    encoder_outputs: Dict[str, torch.Tensor],
    config: SimpleNamespace,
) -> torch.Tensor:
    layer_selector = config.smoothness_apply_layer
    if isinstance(layer_selector, str) and str(layer_selector).strip().lower() == "last":
        return encoder_outputs["node_hidden"]
    node_hidden_layers_raw = encoder_outputs.get("node_hidden_layers", ())
    if not isinstance(node_hidden_layers_raw, (tuple, list)) or len(node_hidden_layers_raw) == 0:
        return encoder_outputs["node_hidden"]

    layer_index = int(layer_selector)
    num_layers = int(len(node_hidden_layers_raw))
    if layer_index < 0:
        layer_index += num_layers
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(
            f"smoothness_apply_layer index out of range: {layer_index} (num_layers={num_layers})"
        )
    return node_hidden_layers_raw[layer_index]


def compute_smoothness_loss_node_hidden(
    node_hidden: torch.Tensor,
    edge_index_prior: torch.Tensor,
    edge_weight_prior: torch.Tensor,
    config: SimpleNamespace,
) -> torch.Tensor:
    if node_hidden.dim() == 2:
        node_hidden = node_hidden.unsqueeze(0)
    if node_hidden.dim() != 3:
        raise ValueError("node_hidden must have shape [num_cells, num_nodes, hidden_dim] or [num_nodes, hidden_dim].")
    if edge_index_prior.dim() != 2 or int(edge_index_prior.shape[0]) != 2:
        raise ValueError("edge_index_prior must have shape [2, num_edges].")
    if edge_weight_prior.dim() != 1:
        raise ValueError("edge_weight_prior must have shape [num_edges].")

    num_edges = int(edge_index_prior.shape[1])
    if num_edges <= 0:
        return node_hidden.new_zeros(())

    source = edge_index_prior[0].to(device=node_hidden.device, dtype=torch.long)
    destination = edge_index_prior[1].to(device=node_hidden.device, dtype=torch.long)
    weights = edge_weight_prior.to(device=node_hidden.device, dtype=node_hidden.dtype)

    diff = node_hidden[:, source, :] - node_hidden[:, destination, :]
    squared_distance = (diff * diff).sum(dim=-1)
    weighted = squared_distance * weights.view(1, -1)

    per_cell = weighted.sum(dim=1)
    if bool(config.smoothness_normalize_by_num_edges):
        per_cell = per_cell / float(max(num_edges, 1))
    return per_cell.mean()


def compute_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    config: SimpleNamespace,
) -> torch.Tensor:
    if logits_a.shape != logits_b.shape:
        raise ValueError("logits_a and logits_b must have the same shape.")
    mode = str(config.consistency_loss_type).strip().lower()
    if mode == "mse_logit":
        lhs = logits_a
        rhs = logits_b
    elif mode == "mse_prob":
        lhs = torch.sigmoid(logits_a)
        rhs = torch.sigmoid(logits_b)
    else:
        raise ValueError("consistency_loss_type must be one of {'mse_prob','mse_logit'}.")
    if bool(config.consistency_detach_second_branch):
        rhs = rhs.detach()
    return torch.mean((lhs - rhs) ** 2)


def _majority_label(values: np.ndarray, default_value: int = 0) -> int:
    arr = np.asarray(values, dtype=np.int64).ravel()
    if int(arr.size) <= 0:
        return int(default_value)
    uniq, counts = np.unique(arr, return_counts=True)
    return int(uniq[int(np.argmax(counts))])



def protein_embedding_source_paths_for_logging(
    config: SimpleNamespace,
    primary_source_path: str,
) -> List[str]:
    choice = str(config.protein_embedding_choice).strip().lower()
    if choice == "none":
        return []

    configured_paths = _normalized_configured_embedding_paths(config)
    sources = _resolve_embedding_view_sources(config)
    resolved_paths: List[str] = []
    for source_name in sources:
        path = _embedding_source_path_for_logging(str(source_name), configured_paths)
        if path is not None:
            resolved_paths.append(path)

    if len(resolved_paths) == 0:
        if os.path.exists(str(primary_source_path)):
            return [os.path.abspath(str(primary_source_path))]
        return []
    return sorted(set(resolved_paths))


def build_celltype_regulation_onehot(
    dataset: str,
    genes: Sequence[str],
    celltype_names: Sequence[str],
    deg_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str]]:
    if deg_dir is None:
        deg_dir = os.path.join(PROJECT_ROOT, "data", str(dataset), "DEG")
    gene_keys = [str(gene).upper() for gene in genes]
    regulation = np.zeros((len(celltype_names), len(gene_keys), 3), dtype=np.float32)
    regulation[:, :, 2] = 1.0  # default = none

    used_metric_paths: List[str] = []
    for celltype_index, celltype_name in enumerate(celltype_names):
        safe_celltype = sanitize_filename_component(str(celltype_name))
        metrics_path = os.path.join(deg_dir, f"DEG_metrics_{safe_celltype}_pass.tsv")
        if not os.path.isfile(metrics_path):
            continue
        used_metric_paths.append(metrics_path)
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

    return torch.tensor(regulation, dtype=torch.float32), used_metric_paths


def build_global_deg_zscore_vector(
    dataset: str,
    genes: Sequence[str],
    deg_dir: Optional[str] = None,
    zscore_standardize: bool = True,
    zscore_clip_min: Optional[float] = -2.0,
    zscore_clip_max: Optional[float] = 2.0,
) -> Tuple[torch.Tensor, str, int]:
    """Load global DEG z-scores (scanpy `scores`) and align to the preselected gene list."""
    if deg_dir is None:
        deg_dir = os.path.join(PROJECT_ROOT, "data", str(dataset), "DEG")
    zscore_path = os.path.join(deg_dir, "DEG_zscore_global.tsv")
    gene_keys = [str(gene).upper() for gene in genes]
    if not os.path.isfile(zscore_path):
        raise FileNotFoundError(
            "Global DEG z-score file not found. "
            f"Expected: {zscore_path}. Run modules/preselection.py first."
        )

    table = pd.read_csv(zscore_path, sep="\t")
    if "gene" not in table.columns:
        raise ValueError(f"Expected column 'gene' in {zscore_path}")
    score_col = "zscore" if "zscore" in table.columns else ("score" if "score" in table.columns else None)
    if score_col is None:
        raise ValueError(f"Expected column 'zscore' (or 'score') in {zscore_path}")
    table["gene"] = table["gene"].astype(str).str.upper()
    score_map = dict(zip(table["gene"], pd.to_numeric(table[score_col], errors="coerce").fillna(0.0).astype(float)))

    values: List[float] = []
    present_mask: List[bool] = []
    missing = 0
    for gene in gene_keys:
        if gene in score_map:
            values.append(float(score_map[gene]))
            present_mask.append(True)
        else:
            values.append(0.0)
            present_mask.append(False)
            missing += 1
    value_array = np.asarray(values, dtype=np.float32)
    present_array = np.asarray(present_mask, dtype=bool)

    if bool(zscore_standardize):
        present_values = value_array[present_array]
        if present_values.size > 1:
            mean_value = float(present_values.mean())
            std_value = float(present_values.std(ddof=0))
            if std_value > 1e-8:
                value_array[present_array] = (present_values - mean_value) / std_value
            else:
                value_array[present_array] = 0.0
        elif present_values.size == 1:
            value_array[present_array] = 0.0
        value_array[~present_array] = 0.0

    if zscore_clip_min is not None or zscore_clip_max is not None:
        lower = -np.inf if zscore_clip_min is None else float(zscore_clip_min)
        upper = np.inf if zscore_clip_max is None else float(zscore_clip_max)
        if lower > upper:
            raise ValueError(f"Invalid zscore clip range: min={lower} > max={upper}")
        value_array = np.clip(value_array, lower, upper)

    return torch.tensor(value_array, dtype=torch.float32), zscore_path, int(missing)


def build_graph_cache(
    config: SimpleNamespace,
    genes: Sequence[str],
    celltype_names: Sequence[str],
    cache_path: str,
    preselection_root: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Dict[str, object]]:
    ppi_path = str(config.ppi_path)
    dataset = str(config.dataset)
    base_root = str(preselection_root) if preselection_root is not None else os.path.join(PROJECT_ROOT, "data", dataset)
    deg_dir = os.path.join(base_root, "DEG")
    zscore_path = os.path.join(deg_dir, "DEG_zscore_global.tsv")
    zscore_standardize = bool(config.zscore_standardize)
    zscore_clip_min = config.zscore_clip_min
    zscore_clip_max = config.zscore_clip_max
    split_number = int(config.split_number)
    regulation_metric_paths = [
        os.path.join(deg_dir, f"DEG_metrics_{sanitize_filename_component(str(celltype_name))}_pass.tsv")
        for celltype_name in celltype_names
    ]
    regulation_metric_signatures = [
        {
            "celltype": str(celltype_name),
            "file": _file_signature(path),
        }
        for celltype_name, path in zip(celltype_names, regulation_metric_paths)
    ]

    expected_key = {
        "format_version": 4,
        "dataset": dataset,
        "split_number": split_number,
        "num_genes": int(len(genes)),
        "genes_sha256": _hash_gene_list(genes),
        "celltypes": list(celltype_names),
        "preselection_root_path": os.path.abspath(base_root),
        "ppi_file": _file_signature(ppi_path),
        "deg_dir": _file_signature(deg_dir),
        "preselection_root": _file_signature(base_root),
        "deg_global_zscore": _file_signature(zscore_path),
        "deg_celltype_regulation_files": regulation_metric_signatures,
        "self_loop": bool(config.gnn_self_loop),
        "zscore_standardize": bool(zscore_standardize),
        "zscore_clip_min": zscore_clip_min,
        "zscore_clip_max": zscore_clip_max,
    }

    payload = _load_cache(cache_path, expected_key)
    if payload is not None:
        edge_index = payload.get("edge_index", None)
        regulation_onehot = payload.get("regulation_onehot", None)
        global_zscore = payload.get("global_zscore", None)
        if (
            isinstance(edge_index, torch.Tensor)
            and isinstance(regulation_onehot, torch.Tensor)
            and isinstance(global_zscore, torch.Tensor)
        ):
            return (
                edge_index.long().contiguous(),
                regulation_onehot.float().contiguous(),
                global_zscore.float().contiguous(),
                True,
                {
                    "regulation_metric_files": payload.get("regulation_metric_files", []),
                    "zscore_source_path": payload.get("zscore_source_path", zscore_path),
                    "zscore_missing": int(payload.get("zscore_missing", -1)),
                    "zscore_standardize": bool(payload.get("zscore_standardize", zscore_standardize)),
                    "zscore_clip_min": payload.get("zscore_clip_min", zscore_clip_min),
                    "zscore_clip_max": payload.get("zscore_clip_max", zscore_clip_max),
                },
            )

    edge_index, _ = build_edge_index(genes, ppi_path, bool(config.gnn_self_loop))
    regulation_onehot, metric_files = build_celltype_regulation_onehot(dataset, genes, celltype_names, deg_dir=deg_dir)
    global_zscore, zscore_source_path, zscore_missing = build_global_deg_zscore_vector(
        dataset,
        genes,
        deg_dir=deg_dir,
        zscore_standardize=zscore_standardize,
        zscore_clip_min=zscore_clip_min,
        zscore_clip_max=zscore_clip_max,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "cache_key": expected_key,
            "edge_index": edge_index.cpu().contiguous(),
            "regulation_onehot": regulation_onehot.cpu().contiguous(),
            "global_zscore": global_zscore.cpu().contiguous(),
            "regulation_metric_files": list(metric_files),
            "zscore_source_path": str(zscore_source_path),
            "zscore_missing": int(zscore_missing),
            "zscore_standardize": bool(zscore_standardize),
            "zscore_clip_min": zscore_clip_min,
            "zscore_clip_max": zscore_clip_max,
        },
        cache_path,
    )
    return (
        edge_index.long().contiguous(),
        regulation_onehot.float().contiguous(),
        global_zscore.float().contiguous(),
        False,
        {
            "regulation_metric_files": metric_files,
            "zscore_source_path": str(zscore_source_path),
            "zscore_missing": int(zscore_missing),
            "zscore_standardize": bool(zscore_standardize),
            "zscore_clip_min": zscore_clip_min,
            "zscore_clip_max": zscore_clip_max,
        },
    )


def build_split_records_cache(
    cache_path: str,
    split_indices: Sequence[Sequence[int]],
    label_indices: np.ndarray,
    patient_indices: np.ndarray,
    split_source_path: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], bool]:
    expected_key = {
        "format_version": 1,
        "split_source": _file_signature(split_source_path),
        "label_sha256": hashlib.sha256(np.asarray(label_indices, dtype=np.int64).tobytes()).hexdigest(),
        "patient_sha256": hashlib.sha256(np.asarray(patient_indices, dtype=np.int64).tobytes()).hexdigest(),
    }
    payload = _load_cache(cache_path, expected_key)
    if payload is not None:
        train_records = payload.get("train_records", None)
        val_records = payload.get("val_records", None)
        test_records = payload.get("test_records", None)
        if isinstance(train_records, list) and isinstance(val_records, list) and isinstance(test_records, list):
            return train_records, val_records, test_records, True

    train_records = build_patient_bag_records(split_indices[0], label_indices, patient_indices)
    val_records = build_patient_bag_records(split_indices[1], label_indices, patient_indices)
    test_records = build_patient_bag_records(split_indices[2], label_indices, patient_indices)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "cache_key": expected_key,
            "train_records": train_records,
            "val_records": val_records,
            "test_records": test_records,
        },
        cache_path,
    )
    return train_records, val_records, test_records, False


def _coerce_split_index_arrays(split_indices: Sequence[Sequence[int]]) -> List[np.ndarray]:
    if not isinstance(split_indices, list) or len(split_indices) != 3:
        raise ValueError("Split file must be list [train_cells, val_cells, test_cells].")
    arrays: List[np.ndarray] = []
    for split_name, raw_values in zip(["train", "val", "test"], split_indices):
        index_array = np.asarray(raw_values, dtype=np.int64).reshape(-1)
        if int(index_array.size) > 0 and int(index_array.min()) < 0:
            raise ValueError(f"Split '{split_name}' includes negative indices.")
        arrays.append(index_array)
    return arrays


def _candidate_reference_adata_paths(config: SimpleNamespace) -> List[str]:
    candidates: List[str] = []
    configured = str(getattr(config, "split_index_reference_adata_path", "")).strip()
    if configured != "":
        candidates.append(os.path.abspath(configured))

    current_adata_path = os.path.abspath(str(config.adata_directory))
    if "_no_doublets" in current_adata_path:
        candidates.append(current_adata_path.replace("_no_doublets", ""))

    deduplicated: List[str] = []
    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated


def normalize_split_indices_to_current_adata(
    split_indices: Sequence[Sequence[int]],
    adata_obs_names: Sequence[object],
    config: SimpleNamespace,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    split_arrays = _coerce_split_index_arrays(split_indices)
    current_obs_names = np.asarray([str(value) for value in adata_obs_names], dtype=object)
    current_n_obs = int(current_obs_names.shape[0])
    split_max_index = int(max((int(array.max()) for array in split_arrays if int(array.size) > 0), default=-1))

    if split_max_index < current_n_obs:
        return split_arrays, {
            "mode": "direct",
            "split_max_index": int(split_max_index),
            "current_n_obs": int(current_n_obs),
            "reference_adata_path": "",
            "dropped_reference_cells": 0,
            "mapped_cells": int(sum(int(array.size) for array in split_arrays)),
        }

    current_name_to_index = {name: int(index) for index, name in enumerate(current_obs_names.tolist())}
    reference_paths = _candidate_reference_adata_paths(config)
    last_error: Optional[Exception] = None

    for reference_path in reference_paths:
        if not os.path.exists(reference_path):
            continue
        reference_adata = None
        try:
            reference_adata = sc.read_h5ad(reference_path, backed="r")
            reference_n_obs = int(reference_adata.n_obs)
            if split_max_index >= reference_n_obs:
                continue

            reference_obs_names = np.asarray(reference_adata.obs_names.astype(str), dtype=object)
            remapped_arrays: List[np.ndarray] = []
            dropped_reference_cells = 0

            for source_array in split_arrays:
                source_names = reference_obs_names[source_array]
                mapped_indices: List[int] = []
                for source_name in source_names.tolist():
                    mapped_index = current_name_to_index.get(str(source_name))
                    if mapped_index is None:
                        dropped_reference_cells += 1
                        continue
                    mapped_indices.append(int(mapped_index))
                remapped_arrays.append(np.asarray(mapped_indices, dtype=np.int64))

            merged = (
                np.concatenate(remapped_arrays, axis=0)
                if any(int(array.size) > 0 for array in remapped_arrays)
                else np.zeros((0,), dtype=np.int64)
            )
            unique_merged = np.unique(merged)
            if int(unique_merged.size) != int(merged.size):
                raise ValueError("Remapped split indices contain duplicate cells across train/val/test.")

            return remapped_arrays, {
                "mode": "obs_name_remap",
                "split_max_index": int(split_max_index),
                "current_n_obs": int(current_n_obs),
                "reference_adata_path": str(reference_path),
                "reference_n_obs": int(reference_n_obs),
                "dropped_reference_cells": int(dropped_reference_cells),
                "mapped_cells": int(merged.size),
                "coverage_ratio": float(unique_merged.size / max(current_n_obs, 1)),
            }
        except Exception as error:
            last_error = error
        finally:
            if reference_adata is not None:
                try:
                    reference_adata.file.close()
                except Exception:
                    pass

    reference_hint = ", ".join(reference_paths) if len(reference_paths) > 0 else "(none)"
    detail = f" Last remap error: {last_error}" if last_error is not None else ""
    raise IndexError(
        "Split indices are out of bounds for current adata. "
        f"split_max_index={split_max_index}, current_n_obs={current_n_obs}. "
        "Provide split files built for this adata, or configure split_index_reference_adata_path "
        f"to an adata used to generate the split indices. Candidates tried: {reference_hint}.{detail}"
    )


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


def build_experiment_directory(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{timestamp}_enc-{config.cell_encoder_name}_emb-{config.protein_embedding_choice}_"
        f"pool-{config.mil_pooling}_clf-{config.classifier_name}_"
        f"bag{config.cells_per_bag}_bpp{config.bags_per_patient_per_epoch}_"
        f"lr{config.lr}_seed{config.seed}_split{config.split_number}"
    )

    output_dir = os.path.join(str(config.experiment_root), "train_runs", run_name)
    cache_dir = os.path.join(str(config.experiment_root), "cache", f"split_idx_{config.split_number}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    return {
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "run_config_path": os.path.join(output_dir, "run_config.json"),
        "metadata_path": os.path.join(output_dir, "metadata.json"),
        "history_path": os.path.join(output_dir, "history.csv"),
        "train_step_log_path": os.path.join(output_dir, "train_step_metrics.csv"),
        "val_step_log_path": os.path.join(output_dir, "val_step_metrics.csv"),
        "test_step_log_path": os.path.join(output_dir, "test_step_metrics.csv"),
        "best_model_path": os.path.join(output_dir, "best_model.pt"),
        "best_checkpoint_path": os.path.join(output_dir, "best_checkpoint.pt"),
        "last_checkpoint_path": os.path.join(output_dir, "last_checkpoint.pt"),
        "epoch5_checkpoint_path": os.path.join(output_dir, "epoch5_checkpoint.pt"),
        "final_metrics_path": os.path.join(output_dir, "final_metrics.json"),
        "patient_predictions_val_path": os.path.join(output_dir, "patient_predictions_val.csv"),
        "patient_predictions_test_path": os.path.join(output_dir, "patient_predictions_test.csv"),
    }


def _write_run_recovery_artifacts(
    artifacts: Dict[str, str],
    run_config_payload: Dict[str, object],
    metadata_payload: Dict[str, object],
) -> None:
    save_json(str(artifacts["run_config_path"]), run_config_payload)
    save_json(str(artifacts["metadata_path"]), metadata_payload)


def _absolute_path_text(path_value: object) -> str:
    text = str(path_value).strip()
    if text == "":
        return ""
    return str(Path(text).expanduser().resolve())


def build_dataloader_kwargs(config: SimpleNamespace, device: torch.device) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "pin_memory": bool(config.dataloader_pin_memory) and device.type == "cuda",
        "collate_fn": collate_patient_bags,
    }
    if int(config.num_workers) > 0:
        kwargs["persistent_workers"] = bool(config.dataloader_persistent_workers)
        kwargs["prefetch_factor"] = int(config.dataloader_prefetch_factor)
    return kwargs


def run_epoch(
    split_name: str,
    epoch: int,
    dataloader: DataLoader,
    cell_encoder: GraphCellEncoder,
    mil_aggregator: PatientMILAggregator,
    classifier: PatientClassifier,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    sample_mode: str,
    bag_size_cells: int,
    seed_anchor: int,
    patient_index_to_name: Dict[int, str],
    celltype_index_to_name: Dict[int, str],
    step_log_path: str,
    collect_top_cells: bool,
    top_n: int,
    aggregate_patient_metrics: bool,
    patient_probability_reduction: str,
    patient_embedding_ema: Optional[PatientEmbeddingEMAMemory] = None,
    patient_embedding_ema_blend: float = 0.0,
    epoch_dependent_sampling: bool = True,
    node_feature_beta: float = 1.0,
    use_consistency_regularizer: bool = False,
    consistency_weight_lambda: float = 0.0,
    consistency_num_bags_per_patient_per_step: int = 1,
    consistency_detach_second_branch: bool = False,
    consistency_loss_type: str = "mse_prob",
    gradient_clip_max_norm: float = 1.0,
    mixup_alpha: float = 0.2,
    lambda_sparse: float = 1e-4,
    prior_strategy_value: str = "existing_injection",
    prior_strategy_regularizer_multiview_knn: bool = False,
    prior_strategy_hidden_additive: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    is_train = optimizer is not None
    cell_encoder.train(mode=is_train)
    mil_aggregator.train(mode=is_train)
    classifier.train(mode=is_train)

    non_blocking_transfer = device.type == "cuda"
    # Use AMP only for training.
    # Validation/test run in FP32 to avoid excessive probability quantization.
    use_amp = bool(device.type == "cuda" and is_train)

    if bool(use_consistency_regularizer) and int(consistency_num_bags_per_patient_per_step) != 2:
        raise ValueError("consistency_num_bags_per_patient_per_step must be 2 when consistency regularizer is enabled.")

    consistency_active = bool(is_train and bool(use_consistency_regularizer))
    effective_lambda_cons = float(consistency_weight_lambda) if consistency_active else 0.0
    effective_lambda_sparse = float(lambda_sparse) if is_train else 0.0
    mixup_alpha_value = max(0.0, float(mixup_alpha)) if is_train else 0.0
    gradient_clip_max_norm_value = float(gradient_clip_max_norm)
    if gradient_clip_max_norm_value < 0.0:
        raise ValueError("gradient_clip_max_norm must be >= 0.")
    trainable_parameters = (
        optimizer_trainable_parameters(optimizer)
        if bool(is_train and gradient_clip_max_norm_value > 0.0)
        else []
    )

    total_loss_weighted = 0.0
    total_task_loss_weighted = 0.0
    total_cons_loss_weighted = 0.0
    total_sparse_loss_weighted = 0.0
    total_mixup_loss_weighted = 0.0
    total_loss_bce_weighted = 0.0
    total_node_score_mean_weighted = 0.0
    total_node_score_abs_mean_weighted = 0.0
    total_bags = 0

    all_labels: List[int] = []
    all_prob_pos: List[float] = []
    all_patient_indices: List[int] = []

    bag_prediction_rows: List[Dict[str, object]] = []
    patient_prediction_rows: List[Dict[str, object]] = []
    top_cells_rows: List[Dict[str, object]] = []

    tqdm_total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else len(dataloader)
    progress_bar = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    progress = tqdm(
        total=max(1, int(tqdm_total)),
        desc=f"{split_name.capitalize()} Epoch{epoch}",
        unit="bag",
        dynamic_ncols=True,
        leave=False,
        bar_format=progress_bar,
    )

    seed_base_primary = (
        int(seed_anchor) + int(epoch) * 10007
        if bool(epoch_dependent_sampling)
        else int(seed_anchor)
    )

    def forward_branch(
        sampled_batch: Dict[str, torch.Tensor],
        apply_ema: bool,
    ) -> Dict[str, object]:
        patient_indices_local = sampled_batch["patient_indices"].detach().cpu().numpy().astype(np.int64)
        bag_repeat_local = sampled_batch["bag_repeat_index"].detach().cpu().numpy().astype(np.int64)
        expression_local = sampled_batch["expression"].to(device, non_blocking=non_blocking_transfer)
        celltype_local = sampled_batch["celltype_index"].to(device, non_blocking=non_blocking_transfer)
        treatment_local = sampled_batch["treatment_index"].to(device, non_blocking=non_blocking_transfer)
        bag_index_local = sampled_batch["bag_index"].to(device, non_blocking=non_blocking_transfer)
        labels_local = sampled_batch["patient_labels"].to(device, non_blocking=non_blocking_transfer).float()

        encoder_outputs_local = cell_encoder(
            expression_values=expression_local,
            celltype_index=celltype_local,
            treatment_index=treatment_local,
            return_node_outputs=False,
            return_attention=False,
        )
        mil_outputs_local = mil_aggregator(
            cell_embeddings=encoder_outputs_local["cell_embeddings"],
            celltype_index=celltype_local,
            bag_index=bag_index_local,
            num_bags=int(labels_local.shape[0]),
        )
        patient_embeddings_local = mil_outputs_local["patient_embeddings"]
        if (
            apply_ema
            and patient_embedding_ema is not None
            and float(patient_embedding_ema_blend) > 0.0
        ):
            ema_values, ema_mask = patient_embedding_ema.lookup(
                patient_indices=patient_indices_local,
                device=patient_embeddings_local.device,
                dtype=patient_embeddings_local.dtype,
            )
            if bool(ema_mask.any()):
                blend = float(patient_embedding_ema_blend)
                patient_embeddings_local = torch.where(
                    ema_mask.view(-1, 1),
                    (1.0 - blend) * patient_embeddings_local + blend * ema_values,
                    patient_embeddings_local,
                )

        pos_weight = getattr(criterion, "pos_weight", None)

        def bce_logits(local_logits: torch.Tensor, reduction: str) -> torch.Tensor:
            return torch_functional.binary_cross_entropy_with_logits(
                local_logits,
                labels_local,
                pos_weight=pos_weight,
                reduction=reduction,
            )

        logits_local = classifier(patient_embeddings_local)
        base_bce_local = bce_logits(logits_local, reduction="mean")
        mixup_local = torch.zeros_like(base_bce_local)

        if mixup_alpha_value > 0.0 and int(patient_embeddings_local.shape[0]) > 1:
            # Use a scalar beta sample to avoid half-precision distribution issues under autocast.
            lam = torch.tensor(
                float(np.random.beta(float(mixup_alpha_value), float(mixup_alpha_value))),
                device=patient_embeddings_local.device,
                dtype=patient_embeddings_local.dtype,
            )
            permutation = torch.randperm(
                int(patient_embeddings_local.shape[0]),
                device=patient_embeddings_local.device,
            )
            mixed_embeddings = lam * patient_embeddings_local + (1.0 - lam) * patient_embeddings_local[permutation]
            mixed_labels = lam * labels_local + (1.0 - lam) * labels_local[permutation]
            mixed_logits = classifier(mixed_embeddings)
            mixup_local = torch_functional.binary_cross_entropy_with_logits(
                mixed_logits,
                mixed_labels,
                pos_weight=pos_weight,
                reduction="mean",
            )

        node_scores_local = encoder_outputs_local.get("node_scores", None)
        if node_scores_local is None:
            raise RuntimeError("Cell encoder outputs must include 'node_scores' for sparse regularization.")
        sparse_local = torch.mean(torch.abs(node_scores_local))
        node_score_mean_local = node_scores_local.mean()
        node_score_abs_mean_local = torch.abs(node_scores_local).mean()
        task_loss_local = base_bce_local + mixup_local

        return {
            "patient_indices": patient_indices_local,
            "bag_repeat_index": bag_repeat_local,
            "labels": labels_local,
            "logits": logits_local,
            "encoder_outputs": encoder_outputs_local,
            "mil_outputs": mil_outputs_local,
            "task_loss": task_loss_local,
            "loss_bce": base_bce_local,
            "loss_mixup": mixup_local,
            "sparse_loss": sparse_local,
            "node_score_mean": node_score_mean_local,
            "node_score_abs_mean": node_score_abs_mean_local,
            "sampled": sampled_batch,
        }

    for step_index, batch in enumerate(dataloader, start=1):
        sampled = sample_collated_batch(
            batch=batch,
            bag_size_cells=bag_size_cells,
            sample_mode=sample_mode,
            seed_base=int(seed_base_primary),
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast_cuda(use_amp):
            branch_a = forward_branch(sampled_batch=sampled, apply_ema=is_train)
            if consistency_active:
                sampled_second = sample_collated_batch(
                    batch=batch,
                    bag_size_cells=bag_size_cells,
                    sample_mode=sample_mode,
                    seed_base=int(seed_base_primary + 982451653 + 31 * int(step_index)),
                )
                branch_b = forward_branch(sampled_batch=sampled_second, apply_ema=is_train)
            else:
                branch_b = None

            if branch_b is None:
                task_loss = branch_a["task_loss"]
                loss_bce_component = branch_a["loss_bce"]
                loss_mixup_component = branch_a["loss_mixup"]
                sparse_loss_component = branch_a["sparse_loss"]
                consistency_loss_component = torch.zeros_like(task_loss)
                node_score_mean_component = branch_a["node_score_mean"]
                node_score_abs_mean_component = branch_a["node_score_abs_mean"]
                logits = branch_a["logits"]
                labels = branch_a["labels"]
            else:
                task_loss = 0.5 * (branch_a["task_loss"] + branch_b["task_loss"])
                loss_bce_component = 0.5 * (branch_a["loss_bce"] + branch_b["loss_bce"])
                loss_mixup_component = 0.5 * (branch_a["loss_mixup"] + branch_b["loss_mixup"])
                sparse_loss_component = 0.5 * (branch_a["sparse_loss"] + branch_b["sparse_loss"])
                node_score_mean_component = 0.5 * (branch_a["node_score_mean"] + branch_b["node_score_mean"])
                node_score_abs_mean_component = 0.5 * (
                    branch_a["node_score_abs_mean"] + branch_b["node_score_abs_mean"]
                )
                consistency_loss_component = compute_consistency_loss(
                    logits_a=branch_a["logits"],
                    logits_b=branch_b["logits"],
                    config=SimpleNamespace(
                        consistency_loss_type=str(consistency_loss_type),
                        consistency_detach_second_branch=bool(consistency_detach_second_branch),
                    ),
                )
                logits = branch_a["logits"]
                labels = branch_a["labels"]

            loss = (
                task_loss
                + float(effective_lambda_sparse) * sparse_loss_component
                + float(effective_lambda_cons) * consistency_loss_component
            )

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if gradient_clip_max_norm_value > 0.0 and len(trainable_parameters) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=float(gradient_clip_max_norm_value),
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_max_norm_value > 0.0 and len(trainable_parameters) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=float(gradient_clip_max_norm_value),
                    )
                optimizer.step()

        if is_train and patient_embedding_ema is not None:
            patient_embedding_ema.update(
                patient_indices=branch_a["patient_indices"],
                patient_embeddings=branch_a["mil_outputs"]["patient_embeddings"],
            )

        probabilities = logits_to_probability_matrix(logits)
        prob_pos = probabilities[:, 1]
        label_np = labels.detach().cpu().numpy().astype(np.int64)

        batch_metrics = compute_binary_metrics(label_np, prob_pos)
        batch_loss = float(loss.detach().item())
        batch_task_loss = float(task_loss.detach().item())
        batch_loss_bce = float(loss_bce_component.detach().item())
        batch_loss_mixup = float(loss_mixup_component.detach().item())
        batch_sparse_loss = float(sparse_loss_component.detach().item())
        batch_consistency_loss = float(consistency_loss_component.detach().item())
        batch_node_score_mean = float(node_score_mean_component.detach().item())
        batch_node_score_abs_mean = float(node_score_abs_mean_component.detach().item())

        total_loss_weighted += batch_loss * int(label_np.shape[0])
        total_task_loss_weighted += batch_task_loss * int(label_np.shape[0])
        total_cons_loss_weighted += batch_consistency_loss * int(label_np.shape[0])
        total_sparse_loss_weighted += batch_sparse_loss * int(label_np.shape[0])
        total_mixup_loss_weighted += batch_loss_mixup * int(label_np.shape[0])
        total_loss_bce_weighted += batch_loss_bce * int(label_np.shape[0])
        total_node_score_mean_weighted += batch_node_score_mean * int(label_np.shape[0])
        total_node_score_abs_mean_weighted += batch_node_score_abs_mean * int(label_np.shape[0])
        total_bags += int(label_np.shape[0])
        all_labels.extend(label_np.tolist())
        all_prob_pos.extend(prob_pos.tolist())

        sampled_primary = branch_a["sampled"]
        sampled_cell_indices = sampled_primary["cell_indices"].detach().cpu().numpy().astype(np.int64)
        sampled_celltypes = sampled_primary["celltype_index"].detach().cpu().numpy().astype(np.int64)
        sampled_bag_counts = bag_cell_counts_from_index(
            sampled_primary["bag_index"], int(sampled_primary["num_bags"].item())
        )
        if int(len(sampled_bag_counts)) != int(label_np.shape[0]):
            raise RuntimeError(
                "Sampled bag count mismatch: "
                f"num_bags={len(sampled_bag_counts)} labels={int(label_np.shape[0])}"
            )
        sampled_bag_cell_counts: List[int] = []
        sampled_bag_cell_hashes: List[str] = []
        sampled_cursor = 0
        for bag_cell_count in sampled_bag_counts:
            next_cursor = sampled_cursor + int(bag_cell_count)
            bag_cells = np.sort(sampled_cell_indices[sampled_cursor:next_cursor].astype(np.int64, copy=False))
            sampled_bag_cell_counts.append(int(bag_cells.shape[0]))
            sampled_bag_cell_hashes.append(hashlib.sha1(bag_cells.tobytes()).hexdigest()[:16])
            sampled_cursor = next_cursor

        patient_indices_batch = branch_a["patient_indices"]
        bag_repeat_batch = branch_a["bag_repeat_index"]
        for bag_local_index in range(int(label_np.shape[0])):
            patient_index = int(patient_indices_batch[bag_local_index])
            all_patient_indices.append(patient_index)
            bag_prediction_rows.append(
                {
                    "split": str(split_name),
                    "patient_index": patient_index,
                    "patient_id": patient_index_to_name.get(patient_index, str(patient_index)),
                    "bag_repeat_index": int(bag_repeat_batch[bag_local_index]),
                    "true_label": int(label_np[bag_local_index]),
                    "pred_prob": float(prob_pos[bag_local_index]),
                    "pred_label": int(prob_pos[bag_local_index] >= 0.5),
                    "bag_cell_count": int(sampled_bag_cell_counts[bag_local_index]),
                    "bag_cell_hash": str(sampled_bag_cell_hashes[bag_local_index]),
                }
            )

        if collect_top_cells:
            attention = branch_a["mil_outputs"]["gamma_attention"].detach().cpu()
            bag_counts = sampled_bag_counts

            cursor = 0
            for bag_local_index, bag_cell_count in enumerate(bag_counts):
                if bag_cell_count <= 0:
                    continue
                next_cursor = cursor + int(bag_cell_count)
                bag_attention = attention[cursor:next_cursor]
                bag_cell_idx = sampled_cell_indices[cursor:next_cursor]
                bag_celltype = sampled_celltypes[cursor:next_cursor]

                k = max(1, min(int(top_n), int(bag_attention.numel())))
                top_values, top_positions = torch.topk(bag_attention, k=k, largest=True, sorted=True)

                patient_index = int(patient_indices_batch[bag_local_index])
                patient_id = patient_index_to_name.get(patient_index, str(patient_index))
                true_label = int(label_np[bag_local_index])
                pred_prob = float(prob_pos[bag_local_index])
                bag_repeat_index = int(bag_repeat_batch[bag_local_index])

                for rank, (local_pos, weight) in enumerate(zip(top_positions.tolist(), top_values.tolist()), start=1):
                    cell_index = int(bag_cell_idx[int(local_pos)])
                    celltype_value = int(bag_celltype[int(local_pos)])
                    top_cells_rows.append(
                        {
                            "split": str(split_name),
                            "patient_index": patient_index,
                            "patient_id": patient_id,
                            "bag_repeat_index": bag_repeat_index,
                            "true_label": true_label,
                            "pred_prob": pred_prob,
                            "rank": int(rank),
                            "cell_index": cell_index,
                            "celltype_index": celltype_value,
                            "celltype_name": celltype_index_to_name.get(celltype_value, str(celltype_value)),
                            "attention_weight": float(weight),
                        }
                    )
                cursor = next_cursor

        if aggregate_patient_metrics:
            running_patient_ids, running_labels, running_prob = aggregate_patient_probabilities(
                patient_indices=all_patient_indices,
                labels=all_labels,
                probabilities_pos=all_prob_pos,
                reduction=patient_probability_reduction,
            )
            running_metrics = compute_binary_metrics(running_labels, running_prob)
            running_units = int(running_patient_ids.shape[0])
        else:
            running_metrics = compute_binary_metrics(all_labels, all_prob_pos)
            running_units = int(len(all_labels))
        running_loss = float(total_loss_weighted / max(total_bags, 1))
        running_task_loss = float(total_task_loss_weighted / max(total_bags, 1))
        running_cons_loss = float(total_cons_loss_weighted / max(total_bags, 1))
        running_sparse_loss = float(total_sparse_loss_weighted / max(total_bags, 1))
        running_mixup_loss = float(total_mixup_loss_weighted / max(total_bags, 1))
        running_loss_bce = float(total_loss_bce_weighted / max(total_bags, 1))
        running_node_score_mean = float(total_node_score_mean_weighted / max(total_bags, 1))
        running_node_score_abs_mean = float(total_node_score_abs_mean_weighted / max(total_bags, 1))

        step_row = {
            "split": str(split_name),
            "epoch": int(epoch),
            "step": int(step_index),
            "num_bags": int(label_np.shape[0]),
            "loss": float(batch_loss),
            "task_loss": float(batch_task_loss),
            "smooth_loss": 0.0,
            "sparse_loss": float(batch_sparse_loss),
            "cons_loss": float(batch_consistency_loss),
            "total_loss": float(batch_loss),
            "loss_bce": float(batch_loss_bce),
            "loss_mixup": float(batch_loss_mixup),
            "effective_lambda_smooth": 0.0,
            "effective_lambda_sparse": float(effective_lambda_sparse),
            "effective_lambda_cons": float(effective_lambda_cons),
            "prior_strategy": str(prior_strategy_value),
            "prior_strategy_regularizer_multiview_knn": bool(prior_strategy_regularizer_multiview_knn),
            "prior_strategy_hidden_additive": bool(prior_strategy_hidden_additive),
            "node_feature_beta": float(node_feature_beta),
            "mixup_alpha": float(mixup_alpha_value),
            "lambda_sparse": float(effective_lambda_sparse),
            "lambda_edge_bias": float(getattr(cell_encoder, "lambda_edge_bias", 0.0)),
            "node_score_mean": float(batch_node_score_mean),
            "node_score_abs_mean": float(batch_node_score_abs_mean),
            "auprc": float(batch_metrics["auprc"]),
            "auroc": float(batch_metrics["auroc"]),
            "f1": float(batch_metrics["f1"]),
            "brier": float(batch_metrics["brier"]),
            "running_units": int(running_units),
            "running_loss": float(running_loss),
            "running_task_loss": float(running_task_loss),
            "running_smooth_loss": 0.0,
            "running_sparse_loss": float(running_sparse_loss),
            "running_cons_loss": float(running_cons_loss),
            "running_total_loss": float(running_loss),
            "running_loss_bce": float(running_loss_bce),
            "running_loss_mixup": float(running_mixup_loss),
            "running_node_score_mean": float(running_node_score_mean),
            "running_node_score_abs_mean": float(running_node_score_abs_mean),
            "running_auprc": float(running_metrics["auprc"]),
            "running_auroc": float(running_metrics["auroc"]),
            "running_f1": float(running_metrics["f1"]),
            "running_brier": float(running_metrics["brier"]),
        }
        append_csv_row(
            path=step_log_path,
            fieldnames=[
                "split",
                "epoch",
                "step",
                "num_bags",
                "loss",
                "task_loss",
                "smooth_loss",
                "sparse_loss",
                "cons_loss",
                "total_loss",
                "loss_bce",
                "loss_mixup",
                "effective_lambda_smooth",
                "effective_lambda_sparse",
                "effective_lambda_cons",
                "prior_strategy",
                "prior_strategy_regularizer_multiview_knn",
                "prior_strategy_hidden_additive",
                "node_feature_beta",
                "mixup_alpha",
                "lambda_sparse",
                "lambda_edge_bias",
                "node_score_mean",
                "node_score_abs_mean",
                "auprc",
                "auroc",
                "f1",
                "brier",
                "running_units",
                "running_loss",
                "running_task_loss",
                "running_smooth_loss",
                "running_sparse_loss",
                "running_cons_loss",
                "running_total_loss",
                "running_loss_bce",
                "running_loss_mixup",
                "running_node_score_mean",
                "running_node_score_abs_mean",
                "running_auprc",
                "running_auroc",
                "running_f1",
                "running_brier",
            ],
            row=step_row,
        )

        progress.update(int(label_np.shape[0]))
        progress.set_postfix(
            {
                "loss": f"{running_loss:.4f}",
                "task": f"{running_task_loss:.4f}",
                "sparse": f"{running_sparse_loss:.4f}",
                "cons": f"{running_cons_loss:.4f}",
                "bce": f"{running_loss_bce:.4f}",
                "mix": f"{running_mixup_loss:.4f}",
                "auprc": f"{running_metrics['auprc']:.4f}",
                "auroc": f"{running_metrics['auroc']:.4f}",
                "f1": f"{running_metrics['f1']:.4f}",
                "brier": f"{running_metrics['brier']:.4f}",
            }
        )

    progress.close()

    if aggregate_patient_metrics:
        patient_ids, patient_labels, patient_prob = aggregate_patient_probabilities(
            patient_indices=all_patient_indices,
            labels=all_labels,
            probabilities_pos=all_prob_pos,
            reduction=patient_probability_reduction,
        )
        epoch_metrics = compute_binary_metrics(patient_labels, patient_prob)
        epoch_metrics["num_patients"] = float(patient_ids.shape[0])
        epoch_metrics["num_bags"] = float(len(all_labels))

        for patient_index, label_value, prob_value in zip(
            patient_ids.tolist(),
            patient_labels.tolist(),
            patient_prob.tolist(),
        ):
            patient_prediction_rows.append(
                {
                    "split": str(split_name),
                    "patient_index": int(patient_index),
                    "patient_id": patient_index_to_name.get(int(patient_index), str(int(patient_index))),
                    "true_label": int(label_value),
                    "pred_prob": float(prob_value),
                    "pred_label": int(float(prob_value) >= 0.5),
                }
            )
    else:
        epoch_metrics = compute_binary_metrics(all_labels, all_prob_pos)

    epoch_metrics["loss"] = float(total_loss_weighted / max(total_bags, 1))
    epoch_metrics["task_loss"] = float(total_task_loss_weighted / max(total_bags, 1))
    epoch_metrics["smooth_loss"] = 0.0
    epoch_metrics["sparse_loss"] = float(total_sparse_loss_weighted / max(total_bags, 1))
    epoch_metrics["cons_loss"] = float(total_cons_loss_weighted / max(total_bags, 1))
    epoch_metrics["total_loss"] = float(total_loss_weighted / max(total_bags, 1))
    epoch_metrics["loss_bce"] = float(total_loss_bce_weighted / max(total_bags, 1))
    epoch_metrics["loss_mixup"] = float(total_mixup_loss_weighted / max(total_bags, 1))
    epoch_metrics["effective_lambda_smooth"] = 0.0
    epoch_metrics["effective_lambda_sparse"] = float(effective_lambda_sparse)
    epoch_metrics["effective_lambda_cons"] = float(effective_lambda_cons)
    epoch_metrics["prior_strategy"] = str(prior_strategy_value)
    epoch_metrics["prior_strategy_regularizer_multiview_knn"] = bool(prior_strategy_regularizer_multiview_knn)
    epoch_metrics["prior_strategy_hidden_additive"] = bool(prior_strategy_hidden_additive)
    epoch_metrics["node_feature_beta"] = float(node_feature_beta)
    epoch_metrics["mixup_alpha"] = float(mixup_alpha_value)
    epoch_metrics["lambda_sparse"] = float(effective_lambda_sparse)
    epoch_metrics["lambda_edge_bias"] = float(getattr(cell_encoder, "lambda_edge_bias", 0.0))
    epoch_metrics["node_score_mean"] = float(total_node_score_mean_weighted / max(total_bags, 1))
    epoch_metrics["node_score_abs_mean"] = float(total_node_score_abs_mean_weighted / max(total_bags, 1))
    return epoch_metrics, bag_prediction_rows, patient_prediction_rows, top_cells_rows


def build_optimizer(
    config: SimpleNamespace,
    cell_encoder: GraphCellEncoder,
    mil_aggregator: PatientMILAggregator,
    classifier: PatientClassifier,
) -> torch.optim.Optimizer:
    params = list(cell_encoder.parameters()) + list(mil_aggregator.parameters()) + list(classifier.parameters())
    if str(config.optimizer).lower() == "adam":
        return torch.optim.Adam(params, lr=float(config.lr), weight_decay=float(config.weight_decay))
    if str(config.optimizer).lower() == "adamw":
        return torch.optim.AdamW(params, lr=float(config.lr), weight_decay=float(config.weight_decay))
    raise ValueError("optimizer must be one of {'adam','adamw'}.")




def run_training_phase(config, *, device=None):
    ensure_cublas_workspace_config()

    seed = int(getattr(config, "seed", 42))
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training script.")
        print(f"{config.gpu}: datatype={type(config.gpu)} value={config.gpu}")
        if int(config.gpu) < 0 or int(config.gpu) >= int(torch.cuda.device_count()):
            raise ValueError(
                f"gpu index {config.gpu} not in {int(torch.cuda.device_count())} visible CUDA devices."
            )
        device = torch.device(f"cuda:{config.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)
        print(f"{device}: using provided device override", flush=True)

    print(config)
    configure_runtime_backends(device=device, deterministic_training=bool(config.deterministic_training))
    set_global_seed(seed, deterministic=bool(config.deterministic_training))
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=bool(config.deterministic_warn_only))
    artifacts = build_experiment_directory(config)
    config_module_file = os.path.join(PROJECT_ROOT, "configs", f"config.py")
    shutil.copy2(config_module_file, os.path.join(artifacts["output_dir"], os.path.basename(config_module_file)))
    

    adata_path = str(getattr(config, "adata_path", "") or getattr(config, "input_h5ad", "") or "").strip()
    if adata_path == "":
        adata_path = str(config.adata_directory) + f"/{config.dataset}" + f"/{config.dataset}_data.h5ad"
    print(f"Loading adata: {adata_path}", flush=True)
    adata = sc.read_h5ad(str(adata_path))

    if config.treatment_column is not None and str(config.treatment_column).strip() != "":
        treatment_column = str(config.treatment_column)
    else: 
        treatment_column = None

    raw_labels = adata.obs[config.label_column].astype(str).tolist()
    mapped_labels, label_mapping_details = map_labels(
        label_values=raw_labels,
        binary_positive_labels=list(config.binary_positive_labels),
        binary_negative_labels=list(config.binary_negative_labels),
    )
    label_mapping, label_indices = create_category_mapping(mapped_labels)

    celltype_values = adata.obs[config.celltype_column].astype(str).tolist()
    celltype_mapping, celltype_indices = create_category_mapping(celltype_values)
    if treatment_column:
        treatment_values = adata.obs[treatment_column].astype(str).tolist()
    else:
        treatment_values = ["NA"] * adata.n_obs
    treatment_mapping, treatment_indices = create_category_mapping(treatment_values)
    patient_values = adata.obs[config.patient_column].astype(str).tolist()
    patient_mapping, patient_indices = create_category_mapping(patient_values)

    label_array = np.asarray(label_indices, dtype=np.int64)
    celltype_array = np.asarray(celltype_indices, dtype=np.int64)
    treatment_array = np.asarray(treatment_indices, dtype=np.int64)
    patient_array = np.asarray(patient_indices, dtype=np.int64)

    celltype_index_to_name = {int(index): str(name) for name, index in celltype_mapping.items()}
    patient_index_to_name = {int(index): str(name) for name, index in patient_mapping.items()}

    print("loading preselection data...", flush=True)
    preselection_root = resolve_training_preselection_root(config)
    setattr(config, "run_dir", str(artifacts["output_dir"]))

    output_root_value = str(getattr(config, "output_root", "") or "").strip()
    if output_root_value == "":
        output_root_value = str(Path(str(config.experiment_root)).resolve().parent)
    run_config_payload = {
        "timestamp": datetime.now().isoformat(),
        "resolved_paths": {
            "adata_path": _absolute_path_text(adata_path),
            "adata_directory": _absolute_path_text(config.adata_directory),
            "output_root": _absolute_path_text(output_root_value),
            "experiment_root": _absolute_path_text(config.experiment_root),
            "splits_directory": _absolute_path_text(config.splits_directory),
            "preselection_output_root": _absolute_path_text(getattr(config, "preselection_output_root", "") or ""),
            "preselection_split_root": _absolute_path_text(preselection_root),
            "ppi_path": _absolute_path_text(config.ppi_path),
            "run_dir": _absolute_path_text(artifacts["output_dir"]),
        },
        "config": dict(config.__dict__),
    }
    metadata_payload = {
        "label_mapping": {str(key): int(value) for key, value in label_mapping.items()},
        "celltype_mapping": {str(key): int(value) for key, value in celltype_mapping.items()},
        "treatment_mapping": {str(key): int(value) for key, value in treatment_mapping.items()},
    }
    _write_run_recovery_artifacts(
        artifacts=artifacts,
        run_config_payload=run_config_payload,
        metadata_payload=metadata_payload,
    )


    np_source_path = os.path.join(preselection_root, "NP", "NP_max.tsv")
    deg_source_dir = os.path.join(preselection_root, "DEG")
    deg_zscore_source_path = os.path.join(deg_source_dir, "DEG_zscore_global.tsv")
    zscore_standardize = bool(config.zscore_standardize)
    zscore_clip_min = config.zscore_clip_min
    zscore_clip_max = config.zscore_clip_max
    print("[Preselection] " f"split={int(config.split_number)} "f"NP={np_source_path} "f"DEG_dir={deg_source_dir} "
    f"DEG_zscore={deg_zscore_source_path} "f"zscore_standardize={zscore_standardize} "f"zscore_clip=[{zscore_clip_min},{zscore_clip_max}]", flush=True,)

    required_embedding_sources = _resolve_embedding_view_sources(config)
    if len(required_embedding_sources) == 0:
        raise ValueError(
            "At least one embedding view is required for FIND prior interface. "
            "Set gene_embedding_views/protein_embedding_paths or prior_view_sources."
        )
    configured_embedding_paths = _normalized_configured_embedding_paths(config)
    genes, maximum_genes = load_k_np_genes(
        config.dataset,
        config.k,
        preselection_root=preselection_root,
        required_embedding_sources=required_embedding_sources,
        embedding_paths=configured_embedding_paths,
    )
    print(f"Loaded preselected {len(genes)} genes from gene space (maximum {len(maximum_genes)})", flush=True)
    print(f"Loaded {len(genes)} genes after embedding coverage check", flush=True)
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
    print(
        "[CellEncoder] "
        f"name={config.cell_encoder_name} "
        f"hidden_dim={encoder_hidden_dimension} "
        f"layers={encoder_number_of_layers} heads={encoder_number_of_heads} "
        f"readout={encoder_graph_readout} cell_embedding_dim={cell_embedding_dimension}",
        flush=True,
    )

    prior_view_sources = list(required_embedding_sources)
    prior_embeddings_by_view = load_prior_embeddings_by_view(
        genes=genes,
        view_names=prior_view_sources,
        config=config,
    )
    if len(prior_embeddings_by_view) == 0:
        raise ValueError("No prior view embeddings were loaded for FIND prior interface.")
    protein_embedding_source_paths = protein_embedding_source_paths_for_logging(
        config=config,
        primary_source_path="",
    )
    if len(protein_embedding_source_paths) == 0:
        protein_embedding_source_paths = [
            _embedding_source_path_for_logging(source_name, _normalized_configured_embedding_paths(config))
            for source_name in prior_view_sources
        ]
        protein_embedding_source_paths = [path for path in protein_embedding_source_paths if path is not None]

    protein_embedding_source = "find_multiview"
    protein_prior_base_embeddings = None
    protein_embeddings = torch.zeros((len(genes), int(encoder_hidden_dimension)), dtype=torch.float32)
    qkv_projection_choice = "find"
    qkv_pretrain_enabled = False
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
    prior_absolute_enabled = bool(prior_absolute_enabled_raw)
    prior_relational_enabled = bool(prior_relational_enabled_raw)
    config.prior_absolute_enabled = bool(prior_absolute_enabled)
    config.prior_relational_enabled = bool(prior_relational_enabled)
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
        f"absolute={bool(prior_absolute_enabled)} "
        f"relational={bool(prior_relational_enabled)} "
        f"views={prior_view_sources} dims={prior_view_dims} "
        f"prior_knn_k={int(config.prior_knn_k)} "
        f"lambda_edge_bias={float(lambda_edge_bias_effective):.6g}",
        flush=True,
    )
    print(
        "[Regularization] "
        f"mixup_alpha={float(config.mixup_alpha):.6g} "
        f"lambda_sparse={float(config.lambda_sparse):.6g}",
        flush=True,
    )

    if bool(config.use_consistency_regularizer) and int(config.consistency_num_bags_per_patient_per_step) != 2:
        raise ValueError("consistency_num_bags_per_patient_per_step must be 2 when use_consistency_regularizer=True.")
    consistency_loss_type = str(config.consistency_loss_type).strip().lower()
    if consistency_loss_type not in {"mse_prob", "mse_logit"}:
        raise ValueError("consistency_loss_type must be one of {'mse_prob','mse_logit'}.")

    graph_cache_path = os.path.join(artifacts["cache_dir"], "graph_cache.pt")
    ordered_celltypes = [
        name
        for name, _index in sorted(celltype_mapping.items(), key=lambda item: int(item[1]))
    ]
    edge_index, regulation_onehot, global_zscore, graph_cache_hit, graph_cache_meta = build_graph_cache(
        config=config,
        genes=genes,
        celltype_names=ordered_celltypes,
        cache_path=graph_cache_path,
        preselection_root=preselection_root,
    )

    split_path = os.path.join(
        str(config.splits_directory),
        f"{config.dataset}_idx_{int(config.split_number)}.pkl",
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_indices = load_pickle(split_path)
    split_indices, split_index_resolution = normalize_split_indices_to_current_adata(
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

    records_cache_path = os.path.join(artifacts["cache_dir"], "split_records_cache.pt")
    train_base_records, val_base_records, test_base_records, split_cache_hit = build_split_records_cache(
        cache_path=records_cache_path,
        split_indices=split_indices,
        label_indices=label_array,
        patient_indices=patient_array,
        split_source_path=split_path,
    )

    train_records = expand_patient_bag_records(train_base_records, int(config.bags_per_patient_per_epoch))

    val_records = expand_patient_bag_records(
        val_base_records,
        int(config.val_bags_per_patient),
    )
    test_records = expand_patient_bag_records(
        test_base_records,
        int(config.test_bags_per_patient),
    )

    if len(train_records) == 0 or len(test_records) == 0 or len(val_records) == 0:
        raise ValueError(
            "Train/test/val records cannot be empty."
        )

    train_dataset = PatientBagDataset(
        expression_matrix=expression_matrix,
        bag_records=train_records,
        celltype_indices=celltype_array,
        treatment_indices=treatment_array,
    )
    val_dataset: Optional[PatientBagDataset] = None

    val_dataset = PatientBagDataset(
        expression_matrix=expression_matrix,
        bag_records=val_records,
        celltype_indices=celltype_array,
        treatment_indices=treatment_array,
    )

    test_dataset = PatientBagDataset(
        expression_matrix=expression_matrix,
        bag_records=test_records,
        celltype_indices=celltype_array,
        treatment_indices=treatment_array,
    )
    dataloader_kwargs = build_dataloader_kwargs(config, device)
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    prior_view_tensors_device = {
        str(view_name): view_tensor.to(device)
        for view_name, view_tensor in prior_embeddings_by_view.items()
    }

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

    mil_aggregator = PatientMILAggregator(
        embedding_dimension=cell_embedding_dimension,
        num_celltypes=int(len(celltype_mapping)),
        pooling=config.mil_pooling,
        attention_hidden_dimension=int(config.mil_attention_hidden_dim),
    ).to(device)

    classifier = PatientClassifier(cell_embedding_dimension, config).to(device)

    optimizer = build_optimizer(config, cell_encoder, mil_aggregator, classifier)
    scaler: Optional[GradScaler] = GradScaler(enabled=(device.type == "cuda")) if device.type == "cuda" else None

    pos_weight_cfg = config.pos_weight
    pos_weight_tensor = None
    if pos_weight_cfg is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight_cfg)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if pos_weight_cfg is None:
        print("Loss setup: BCEWithLogitsLoss(pos_weight=None)", flush=True)
    else:
        print(f"Loss setup: BCEWithLogitsLoss(pos_weight={float(pos_weight_cfg):.6f})", flush=True)

    with open(artifacts["history_path"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "val_loss",
                "val_task_loss",
                "val_smooth_loss",
                "val_sparse_loss",
                "val_mixup_loss",
                "val_cons_loss",
                "val_total_loss",
                "effective_lambda_smooth",
                "effective_lambda_sparse",
                "effective_lambda_cons",
                "prior_strategy",
                "prior_strategy_regularizer_multiview_knn",
                "prior_strategy_hidden_additive",
                "node_feature_beta",
                "mixup_alpha",
                "lambda_sparse",
                "lambda_edge_bias",
                "val_node_score_mean",
                "val_node_score_abs_mean",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_auprc",
                "val_auroc",
                "val_f1",
                "val_brier",
                "val_num_patients",
                "val_num_bags",
                "selection_primary_metric",
                "selection_secondary_metric",
                "selection_primary_epsilon",
                "selection_primary_value",
                "selection_secondary_value",
                "selection_primary_threshold",
                "selection_is_candidate",
                "best_primary_value",
                "best_secondary_value",
                "best_tradeoff_epoch",
                "best_metric",
            ],
        )
        writer.writeheader()

    selection_primary_key, selection_primary_mode = resolve_model_selection_metric(
        metric_name=str(config.model_selection_primary_metric),
        default_metric_key="auprc",
        default_mode="max",
    )
    selection_secondary_key, selection_secondary_mode = resolve_model_selection_metric(
        metric_name=str(config.model_selection_secondary_metric),
        default_metric_key="brier",
        default_mode="min",
    )
    selection_primary_epsilon = max(0.0, float(config.model_selection_primary_epsilon))

    if selection_primary_key not in {"auprc", "auroc"}:
        print(
            "[ModelSelection] warning: primary metric is not AU(PR)C/AUROC. "
            f"primary={selection_primary_key} mode={selection_primary_mode}",
            flush=True,
        )

    print(
        "[ModelSelection] "
        f"primary={selection_primary_key}({selection_primary_mode}) "
        f"secondary={selection_secondary_key}({selection_secondary_mode}) "
        f"epsilon={selection_primary_epsilon:.4f}",
        flush=True,
    )

    best_primary_value = float("nan")
    best_primary_threshold = float("nan")
    best_secondary_value = float("nan")
    best_score = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0

    sample_mode = str(config.sample_bags)
    top_n = int(config.mil_attention_top_n)
    patient_probability_reduction = str(config.patient_probability_reduction)
    use_patient_ema_pooling = bool(config.use_patient_ema_pooling)
    patient_ema_decay = float(config.patient_ema_decay)
    patient_ema_blend = float(config.patient_ema_blend)
    patient_embedding_ema: Optional[PatientEmbeddingEMAMemory] = None
    if use_patient_ema_pooling:
        patient_embedding_ema = PatientEmbeddingEMAMemory(
            embedding_dimension=cell_embedding_dimension,
            decay=float(patient_ema_decay),
        )
        print(
            f"MIDAM-lite: enabled (decay={patient_ema_decay:.4f}, blend={patient_ema_blend:.4f})",
            flush=True,
        )
    else:
        print("MIDAM-lite: disabled", flush=True)

    train_seed_anchor = int(seed) * 1000003 + 11
    val_seed_anchor = int(seed) * 1000003 + 23
    test_seed_anchor = int(seed) * 1000003 + 37

    node_feature_beta_value = float(config.node_feature_beta)
    cell_encoder.set_node_feature_beta(float(node_feature_beta_value))
    print(f"[NodeFeatureBeta] fixed beta={node_feature_beta_value:.4f}", flush=True)
    gradient_clip_max_norm = float(config.gradient_clip_max_norm)
    if gradient_clip_max_norm > 0.0:
        print(f"[GradientClipping] max_norm={gradient_clip_max_norm:.4f}", flush=True)
    else:
        print("[GradientClipping] disabled (gradient_clip_max_norm<=0).", flush=True)
    regularizer_epoch_kwargs = {
        "node_feature_beta": float(node_feature_beta_value),
        "use_consistency_regularizer": bool(config.use_consistency_regularizer),
        "consistency_weight_lambda": float(config.consistency_weight_lambda),
        "consistency_num_bags_per_patient_per_step": int(config.consistency_num_bags_per_patient_per_step),
        "consistency_detach_second_branch": bool(config.consistency_detach_second_branch),
        "consistency_loss_type": str(consistency_loss_type),
        "gradient_clip_max_norm": float(gradient_clip_max_norm),
        "mixup_alpha": float(config.mixup_alpha),
        "lambda_sparse": float(config.lambda_sparse),
        "prior_strategy_value": str(prior_strategy_label),
        "prior_strategy_regularizer_multiview_knn": bool(prior_relational_enabled),
        "prior_strategy_hidden_additive": bool(prior_absolute_enabled),
    }

    def save_selected_checkpoint(
        *,
        epoch_value: int,
        selected_primary_value: float,
        selected_secondary_value: float,
        selected_primary_threshold: float,
        selected_primary_best: float,
    ) -> None:
        torch.save(
            {
                "epoch": int(epoch_value),
                "cell_encoder": cell_encoder.state_dict(),
                "mil_aggregator": mil_aggregator.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": float(selected_primary_value),
                "best_primary_value": float(selected_primary_best),
                "best_primary_threshold": float(selected_primary_threshold),
                "best_secondary_value": float(selected_secondary_value),
            },
            artifacts["best_checkpoint_path"],
        )
        torch.save(
            {
                "cell_encoder": cell_encoder.state_dict(),
                "mil_aggregator": mil_aggregator.state_dict(),
                "classifier": classifier.state_dict(),
            },
            artifacts["best_model_path"],
        )

    assert val_loader is not None

    with torch.inference_mode():
        val_metrics_epoch0, _, _, _ = run_epoch(
            split_name="val",
            epoch=0,
            dataloader=val_loader,
            cell_encoder=cell_encoder,
            mil_aggregator=mil_aggregator,
            classifier=classifier,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            sample_mode=sample_mode,
            bag_size_cells=int(config.cells_per_bag),
            seed_anchor=val_seed_anchor,
            patient_index_to_name=patient_index_to_name,
            celltype_index_to_name=celltype_index_to_name,
            step_log_path=artifacts["val_step_log_path"],
            collect_top_cells=False,
            top_n=top_n,
            aggregate_patient_metrics=True,
            patient_probability_reduction=patient_probability_reduction,
            patient_embedding_ema=None,
            patient_embedding_ema_blend=0.0,
            epoch_dependent_sampling=False,
            **regularizer_epoch_kwargs,
        )

    primary_epoch0 = get_validation_metric_value(val_metrics_epoch0, selection_primary_key)
    secondary_epoch0 = get_validation_metric_value(val_metrics_epoch0, selection_secondary_key)

    if metric_improved(primary_epoch0, best_primary_value, selection_primary_mode):
        best_primary_value = float(primary_epoch0)

    if str(selection_primary_mode).lower() == "min":
        best_primary_threshold = float(best_primary_value) + float(selection_primary_epsilon)
    else:
        best_primary_threshold = float(best_primary_value) - float(selection_primary_epsilon)

    candidate_epoch0 = is_within_primary_epsilon_band(
        metric_value=primary_epoch0,
        best_primary_value=best_primary_value,
        mode=selection_primary_mode,
        epsilon=selection_primary_epsilon,
    )

    if candidate_epoch0 and np.isfinite(secondary_epoch0):
        best_score = float(primary_epoch0)
        best_secondary_value = float(secondary_epoch0)
        best_epoch = 0
        epochs_without_improvement = 0
        save_selected_checkpoint(
            epoch_value=0,
            selected_primary_value=float(best_score),
            selected_secondary_value=float(best_secondary_value),
            selected_primary_threshold=float(best_primary_threshold),
            selected_primary_best=float(best_primary_value),
        )

    append_csv_row(
        path=artifacts["history_path"],
        fieldnames=[
            "epoch",
            "val_loss",
            "val_task_loss",
            "val_smooth_loss",
            "val_sparse_loss",
            "val_mixup_loss",
            "val_cons_loss",
            "val_total_loss",
            "effective_lambda_smooth",
            "effective_lambda_sparse",
            "effective_lambda_cons",
            "prior_strategy",
            "prior_strategy_regularizer_multiview_knn",
            "prior_strategy_hidden_additive",
            "node_feature_beta",
            "mixup_alpha",
            "lambda_sparse",
            "lambda_edge_bias",
            "val_node_score_mean",
            "val_node_score_abs_mean",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_auprc",
            "val_auroc",
            "val_f1",
            "val_brier",
            "val_num_patients",
            "val_num_bags",
            "selection_primary_metric",
            "selection_secondary_metric",
            "selection_primary_epsilon",
            "selection_primary_value",
            "selection_secondary_value",
            "selection_primary_threshold",
            "selection_is_candidate",
            "best_primary_value",
            "best_secondary_value",
            "best_tradeoff_epoch",
            "best_metric",
        ],
        row={
            "epoch": 0,
            "val_loss": float(val_metrics_epoch0["loss"]),
            "val_task_loss": float(val_metrics_epoch0.get("task_loss", val_metrics_epoch0["loss"])),
            "val_smooth_loss": float(val_metrics_epoch0.get("smooth_loss", 0.0)),
            "val_sparse_loss": float(val_metrics_epoch0.get("sparse_loss", 0.0)),
            "val_mixup_loss": float(val_metrics_epoch0.get("loss_mixup", 0.0)),
            "val_cons_loss": float(val_metrics_epoch0.get("cons_loss", 0.0)),
            "val_total_loss": float(val_metrics_epoch0.get("total_loss", val_metrics_epoch0["loss"])),
            "effective_lambda_smooth": float(val_metrics_epoch0.get("effective_lambda_smooth", 0.0)),
            "effective_lambda_sparse": float(val_metrics_epoch0.get("effective_lambda_sparse", 0.0)),
            "effective_lambda_cons": float(val_metrics_epoch0.get("effective_lambda_cons", 0.0)),
            "prior_strategy": str(val_metrics_epoch0.get("prior_strategy", str(prior_strategy_label))),
            "prior_strategy_regularizer_multiview_knn": bool(
                val_metrics_epoch0.get(
                    "prior_strategy_regularizer_multiview_knn",
                    prior_relational_enabled,
                )
            ),
            "prior_strategy_hidden_additive": bool(
                val_metrics_epoch0.get(
                    "prior_strategy_hidden_additive",
                    prior_absolute_enabled,
                )
            ),
            "node_feature_beta": float(val_metrics_epoch0.get("node_feature_beta", node_feature_beta_value)),
            "mixup_alpha": float(val_metrics_epoch0.get("mixup_alpha", config.mixup_alpha)),
            "lambda_sparse": float(val_metrics_epoch0.get("lambda_sparse", config.lambda_sparse)),
            "lambda_edge_bias": float(lambda_edge_bias_effective),
            "val_node_score_mean": float(val_metrics_epoch0.get("node_score_mean", 0.0)),
            "val_node_score_abs_mean": float(val_metrics_epoch0.get("node_score_abs_mean", 0.0)),
            "val_accuracy": float(val_metrics_epoch0["accuracy"]),
            "val_precision": float(val_metrics_epoch0["precision"]),
            "val_recall": float(val_metrics_epoch0["recall"]),
            "val_auprc": float(val_metrics_epoch0["auprc"]),
            "val_auroc": float(val_metrics_epoch0["auroc"]),
            "val_f1": float(val_metrics_epoch0["f1"]),
            "val_brier": float(val_metrics_epoch0["brier"]),
            "val_num_patients": float(val_metrics_epoch0.get("num_patients", float("nan"))),
            "val_num_bags": float(val_metrics_epoch0.get("num_bags", float("nan"))),
            "selection_primary_metric": str(selection_primary_key),
            "selection_secondary_metric": str(selection_secondary_key),
            "selection_primary_epsilon": float(selection_primary_epsilon),
            "selection_primary_value": float(primary_epoch0),
            "selection_secondary_value": float(secondary_epoch0),
            "selection_primary_threshold": float(best_primary_threshold),
            "selection_is_candidate": bool(candidate_epoch0),
            "best_primary_value": float(best_primary_value),
            "best_secondary_value": float(best_secondary_value),
            "best_tradeoff_epoch": int(best_epoch),
            "best_metric": float(best_score),
        },
    )

    print(
        f"[Epoch 000] (pre-train) "
        f"val_total={val_metrics_epoch0.get('total_loss', val_metrics_epoch0['loss']):.4f} "
        f"(task={val_metrics_epoch0.get('task_loss', val_metrics_epoch0['loss']):.4f}, "
        f"sparse={val_metrics_epoch0.get('sparse_loss', 0.0):.4f}, "
        f"cons={val_metrics_epoch0.get('cons_loss', 0.0):.4f}, "
        f"bce={val_metrics_epoch0['loss_bce']:.4f}, "
        f"mix={val_metrics_epoch0.get('loss_mixup', 0.0):.4f}) "
        f"node_score_mean={val_metrics_epoch0.get('node_score_mean', 0.0):.4f} "
        f"node_score_abs_mean={val_metrics_epoch0.get('node_score_abs_mean', 0.0):.4f} "
        f"AUPRC={val_metrics_epoch0['auprc']:.4f} AUROC={val_metrics_epoch0['auroc']:.4f} "
        f"F1={val_metrics_epoch0['f1']:.4f} Brier={val_metrics_epoch0['brier']:.4f} "
        f"sel_primary={primary_epoch0:.4f} sel_secondary={secondary_epoch0:.4f} "
        f"best_tradeoff_epoch={best_epoch:03d} best_primary={best_primary_value:.4f} "
        f"best_secondary={best_secondary_value:.4f}",
        flush=True,
    )

    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()

        cell_encoder.set_node_feature_beta(float(node_feature_beta_value))

        train_metrics, _, _, _ = run_epoch(
            split_name="train",
            epoch=epoch,
            dataloader=train_loader,
            cell_encoder=cell_encoder,
            mil_aggregator=mil_aggregator,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            sample_mode=sample_mode,
            bag_size_cells=int(config.cells_per_bag),
            seed_anchor=train_seed_anchor,
            patient_index_to_name=patient_index_to_name,
            celltype_index_to_name=celltype_index_to_name,
            step_log_path=artifacts["train_step_log_path"],
            collect_top_cells=False,
            top_n=top_n,
            aggregate_patient_metrics=False,
            patient_probability_reduction=patient_probability_reduction,
            patient_embedding_ema=patient_embedding_ema,
            patient_embedding_ema_blend=patient_ema_blend,
            epoch_dependent_sampling=True,
            **regularizer_epoch_kwargs,
        )

        with torch.inference_mode():
            val_metrics, _, val_patient_predictions_epoch, _ = run_epoch(
                split_name="val",
                epoch=epoch,
                dataloader=val_loader,
                cell_encoder=cell_encoder,
                mil_aggregator=mil_aggregator,
                classifier=classifier,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=device,
                sample_mode=sample_mode,
                bag_size_cells=int(config.cells_per_bag),
                seed_anchor=val_seed_anchor,
                patient_index_to_name=patient_index_to_name,
                celltype_index_to_name=celltype_index_to_name,
                step_log_path=artifacts["val_step_log_path"],
                collect_top_cells=False,
                top_n=top_n,
                aggregate_patient_metrics=True,
                patient_probability_reduction=patient_probability_reduction,
                patient_embedding_ema=None,
                patient_embedding_ema_blend=0.0,
                epoch_dependent_sampling=False,
                **regularizer_epoch_kwargs,
            )
        save_csv(
            os.path.join(artifacts["output_dir"], f"patient_predictions_val_epoch_{int(epoch):03d}.csv"),
            val_patient_predictions_epoch,
        )

        primary_value = get_validation_metric_value(val_metrics, selection_primary_key)
        secondary_value = get_validation_metric_value(val_metrics, selection_secondary_key)

        if metric_improved(primary_value, best_primary_value, selection_primary_mode):
            best_primary_value = float(primary_value)

        if str(selection_primary_mode).lower() == "min":
            best_primary_threshold = float(best_primary_value) + float(selection_primary_epsilon)
        else:
            best_primary_threshold = float(best_primary_value) - float(selection_primary_epsilon)

        candidate_in_band = is_within_primary_epsilon_band(
            metric_value=primary_value,
            best_primary_value=best_primary_value,
            mode=selection_primary_mode,
            epsilon=selection_primary_epsilon,
        )
        selected_in_band = is_within_primary_epsilon_band(
            metric_value=best_score,
            best_primary_value=best_primary_value,
            mode=selection_primary_mode,
            epsilon=selection_primary_epsilon,
        )

        update_tradeoff_checkpoint = False
        if candidate_in_band and np.isfinite(secondary_value):
            if np.isnan(best_secondary_value) or not bool(selected_in_band):
                update_tradeoff_checkpoint = True
            elif metric_improved(secondary_value, best_secondary_value, selection_secondary_mode):
                update_tradeoff_checkpoint = True
            elif np.isfinite(best_secondary_value) and np.isclose(
                float(secondary_value),
                float(best_secondary_value),
                rtol=0.0,
                atol=1e-12,
            ):
                if metric_improved(primary_value, best_score, selection_primary_mode):
                    update_tradeoff_checkpoint = True

        if update_tradeoff_checkpoint:
            best_score = float(primary_value)
            best_secondary_value = float(secondary_value)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            save_selected_checkpoint(
                epoch_value=int(epoch),
                selected_primary_value=float(best_score),
                selected_secondary_value=float(best_secondary_value),
                selected_primary_threshold=float(best_primary_threshold),
                selected_primary_best=float(best_primary_value),
            )
        else:
            epochs_without_improvement += 1

        last_checkpoint_payload = {
            "epoch": int(epoch),
            "cell_encoder": cell_encoder.state_dict(),
            "mil_aggregator": mil_aggregator.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": float(best_score),
            "best_primary_value": float(best_primary_value),
            "best_primary_threshold": float(best_primary_threshold),
            "best_secondary_value": float(best_secondary_value),
            "best_epoch": int(best_epoch),
        }
        torch.save(last_checkpoint_payload, artifacts["last_checkpoint_path"])
        if int(epoch) == 5:
            torch.save(last_checkpoint_payload, artifacts["epoch5_checkpoint_path"])

        append_csv_row(
            path=artifacts["history_path"],
            fieldnames=[
                "epoch",
                "val_loss",
                "val_task_loss",
                "val_smooth_loss",
                "val_sparse_loss",
                "val_mixup_loss",
                "val_cons_loss",
                "val_total_loss",
                "effective_lambda_smooth",
                "effective_lambda_sparse",
                "effective_lambda_cons",
                "prior_strategy",
                "prior_strategy_regularizer_multiview_knn",
                "prior_strategy_hidden_additive",
                "node_feature_beta",
                "mixup_alpha",
                "lambda_sparse",
                "lambda_edge_bias",
                "val_node_score_mean",
                "val_node_score_abs_mean",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_auprc",
                "val_auroc",
                "val_f1",
                "val_brier",
                "val_num_patients",
                "val_num_bags",
                "selection_primary_metric",
                "selection_secondary_metric",
                "selection_primary_epsilon",
                "selection_primary_value",
                "selection_secondary_value",
                "selection_primary_threshold",
                "selection_is_candidate",
                "best_primary_value",
                "best_secondary_value",
                "best_tradeoff_epoch",
                "best_metric",
            ],
            row={
                "epoch": int(epoch),
                "val_loss": float(val_metrics["loss"]),
                "val_task_loss": float(val_metrics.get("task_loss", val_metrics["loss"])),
                "val_smooth_loss": float(val_metrics.get("smooth_loss", 0.0)),
                "val_sparse_loss": float(val_metrics.get("sparse_loss", 0.0)),
                "val_mixup_loss": float(val_metrics.get("loss_mixup", 0.0)),
                "val_cons_loss": float(val_metrics.get("cons_loss", 0.0)),
                "val_total_loss": float(val_metrics.get("total_loss", val_metrics["loss"])),
                "effective_lambda_smooth": float(val_metrics.get("effective_lambda_smooth", 0.0)),
                "effective_lambda_sparse": float(val_metrics.get("effective_lambda_sparse", 0.0)),
                "effective_lambda_cons": float(val_metrics.get("effective_lambda_cons", 0.0)),
                "prior_strategy": str(val_metrics.get("prior_strategy", str(prior_strategy_label))),
                "prior_strategy_regularizer_multiview_knn": bool(
                    val_metrics.get(
                        "prior_strategy_regularizer_multiview_knn",
                        prior_relational_enabled,
                    )
                ),
                "prior_strategy_hidden_additive": bool(
                    val_metrics.get(
                        "prior_strategy_hidden_additive",
                        prior_absolute_enabled,
                    )
                ),
                "node_feature_beta": float(val_metrics.get("node_feature_beta", node_feature_beta_value)),
                "mixup_alpha": float(val_metrics.get("mixup_alpha", config.mixup_alpha)),
                "lambda_sparse": float(val_metrics.get("lambda_sparse", config.lambda_sparse)),
                "lambda_edge_bias": float(lambda_edge_bias_effective),
                "val_node_score_mean": float(val_metrics.get("node_score_mean", 0.0)),
                "val_node_score_abs_mean": float(val_metrics.get("node_score_abs_mean", 0.0)),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_auprc": float(val_metrics["auprc"]),
                "val_auroc": float(val_metrics["auroc"]),
                "val_f1": float(val_metrics["f1"]),
                "val_brier": float(val_metrics["brier"]),
                "val_num_patients": float(val_metrics.get("num_patients", float("nan"))),
                "val_num_bags": float(val_metrics.get("num_bags", float("nan"))),
                "selection_primary_metric": str(selection_primary_key),
                "selection_secondary_metric": str(selection_secondary_key),
                "selection_primary_epsilon": float(selection_primary_epsilon),
                "selection_primary_value": float(primary_value),
                "selection_secondary_value": float(secondary_value),
                "selection_primary_threshold": float(best_primary_threshold),
                "selection_is_candidate": bool(candidate_in_band),
                "best_primary_value": float(best_primary_value),
                "best_secondary_value": float(best_secondary_value),
                "best_tradeoff_epoch": int(best_epoch),
                "best_metric": float(best_score),
            },
        )

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_metrics.get('total_loss', train_metrics['loss']):.4f} "
            f"(task={train_metrics.get('task_loss', train_metrics['loss']):.4f}, "
            f"sparse={train_metrics.get('sparse_loss', 0.0):.4f}, "
            f"cons={train_metrics.get('cons_loss', 0.0):.4f}, "
            f"bce={train_metrics['loss_bce']:.4f}, "
            f"mix={train_metrics.get('loss_mixup', 0.0):.4f}) "
            f"val_total={val_metrics.get('total_loss', val_metrics['loss']):.4f} "
            f"(task={val_metrics.get('task_loss', val_metrics['loss']):.4f}, "
            f"sparse={val_metrics.get('sparse_loss', 0.0):.4f}, "
            f"cons={val_metrics.get('cons_loss', 0.0):.4f}, "
            f"bce={val_metrics['loss_bce']:.4f}, "
            f"mix={val_metrics.get('loss_mixup', 0.0):.4f}) "
            f"node_score_mean={val_metrics.get('node_score_mean', 0.0):.4f} "
            f"node_score_abs_mean={val_metrics.get('node_score_abs_mean', 0.0):.4f} "
            f"AUPRC={val_metrics['auprc']:.4f} AUROC={val_metrics['auroc']:.4f} "
            f"F1={val_metrics['f1']:.4f} Brier={val_metrics['brier']:.4f} "
            f"sel_primary={primary_value:.4f} sel_secondary={secondary_value:.4f} "
            f"best_tradeoff_epoch={best_epoch:03d} best_primary={best_primary_value:.4f} "
            f"best_secondary={best_secondary_value:.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= int(config.early_stopping_patience):
            print(
                "Early stopping at epoch "
                f"{epoch} (no Pareto-tradeoff improvement for {epochs_without_improvement} epochs).",
                flush=True,
            )
            break

    eval_epoch = int(best_epoch)
    eval_checkpoint_path = artifacts["best_checkpoint_path"]
    if int(best_epoch) < 5:
        eval_epoch = 5
        eval_checkpoint_path = artifacts["epoch5_checkpoint_path"]
        if not os.path.exists(eval_checkpoint_path):
            raise RuntimeError(
                "best_epoch<5 but epoch5 checkpoint was not saved. "
                "Ensure training reaches at least epoch 5."
            )
        print(
            f"[Checkpoint] best_epoch={int(best_epoch)} < 5; using epoch {int(eval_epoch)} checkpoint for final evaluation.",
            flush=True,
        )
    elif not os.path.exists(eval_checkpoint_path):
        raise RuntimeError("Best checkpoint was not saved.")

    best_payload = torch.load(eval_checkpoint_path, map_location=device)
    cell_encoder.load_state_dict(best_payload["cell_encoder"])
    mil_aggregator.load_state_dict(best_payload["mil_aggregator"])
    classifier.load_state_dict(best_payload["classifier"])

    with torch.inference_mode():
        val_metrics, val_bag_predictions, val_patient_predictions, val_top_cells = run_epoch(
            split_name="val",
            epoch=int(eval_epoch),
            dataloader=val_loader,
            cell_encoder=cell_encoder,
            mil_aggregator=mil_aggregator,
            classifier=classifier,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            sample_mode=sample_mode,
            bag_size_cells=int(config.cells_per_bag),
            seed_anchor=val_seed_anchor,
            patient_index_to_name=patient_index_to_name,
            celltype_index_to_name=celltype_index_to_name,
            step_log_path=artifacts["val_step_log_path"],
            collect_top_cells=True,
            top_n=top_n,
            aggregate_patient_metrics=True,
            patient_probability_reduction=patient_probability_reduction,
            patient_embedding_ema=None,
            patient_embedding_ema_blend=0.0,
            epoch_dependent_sampling=False,
            **regularizer_epoch_kwargs,
        )

        test_metrics, test_bag_predictions, test_patient_predictions, test_top_cells = run_epoch(
            split_name="test",
            epoch=int(eval_epoch),
            dataloader=test_loader,
            cell_encoder=cell_encoder,
            mil_aggregator=mil_aggregator,
            classifier=classifier,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            sample_mode=sample_mode,
            bag_size_cells=int(config.cells_per_bag),
            seed_anchor=test_seed_anchor,
            patient_index_to_name=patient_index_to_name,
            celltype_index_to_name=celltype_index_to_name,
            step_log_path=artifacts["test_step_log_path"],
            collect_top_cells=True,
            top_n=top_n,
            aggregate_patient_metrics=True,
            patient_probability_reduction=patient_probability_reduction,
            patient_embedding_ema=None,
            patient_embedding_ema_blend=0.0,
            epoch_dependent_sampling=False,
            **regularizer_epoch_kwargs,
        )

    save_csv(os.path.join(artifacts["output_dir"], "bag_predictions_val.csv"), val_bag_predictions)
    save_csv(os.path.join(artifacts["output_dir"], "bag_predictions_test.csv"), test_bag_predictions)
    save_csv(os.path.join(artifacts["output_dir"], "patient_predictions_val.csv"), val_patient_predictions)
    save_csv(os.path.join(artifacts["output_dir"], "patient_predictions_test.csv"), test_patient_predictions)
    save_csv(os.path.join(artifacts["output_dir"], "mil_top_cells_val.csv"), val_top_cells)
    save_csv(os.path.join(artifacts["output_dir"], "mil_top_cells_test.csv"), test_top_cells)

    save_json(
        os.path.join(artifacts["output_dir"], "final_metrics.json"),
        {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "model_selection": {
                "primary_metric": str(selection_primary_key),
                "primary_mode": str(selection_primary_mode),
                "primary_epsilon": float(selection_primary_epsilon),
                "best_primary_value": float(best_primary_value),
                "primary_threshold": float(best_primary_threshold),
                "secondary_metric": str(selection_secondary_key),
                "secondary_mode": str(selection_secondary_mode),
                "best_secondary_value": float(best_secondary_value),
                "best_tradeoff_epoch": int(best_epoch),
            },
            "early_stopping_metric_legacy": str(config.early_stopping_metric),
            "early_stopping_patience": int(config.early_stopping_patience),
            "val": metrics_to_json_ready(val_metrics),
            "test": metrics_to_json_ready(test_metrics),
        },
    )

    print("Training finished.", flush=True)
    print(f"Output directory: {artifacts['output_dir']}", flush=True)
    print(
        "Saved: run_config.json, metadata.json, history.csv, *_step_metrics.csv, "
        "bag_predictions_{val,test}.csv, patient_predictions_{val,test}.csv, "
        "mil_top_cells_{val,test}.csv, best_model.pt",
        flush=True,
    )

    return artifacts


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train scGOAT + MIL patient classifier")
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. asthma")
    parser.add_argument("--seed", required=False, type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--gpu", required=True, type=int, help="cuda device index")
    parser.add_argument("--split-number", required=True, type=int, help="split index")
    return parser


def build_train_config_from_cli_args(args):
    args_dict = args.__dict__

    if args.dataset == "asthma":
        args_dict.update(asthma_train_configuration)
    elif args.dataset == "asthma_ext":
        args_dict.update(asthma_ext_train_configuration)
    elif args.dataset == "vitiligo":
        args_dict.update(vitiligo_train_configuration)
    elif args.dataset == "covid":
        args_dict.update(covid_train_configuration)
    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}, custom split configuration is required for new datasets."
        )
    return dict2namespace(args_dict)


def load_train_config_from_cli(argv: Optional[Sequence[str]] = None):
    parser = build_train_arg_parser()
    args = parser.parse_args(argv)
    return build_train_config_from_cli_args(args)


def main() -> None:
    config = load_train_config_from_cli()
    run_training_phase(config)


if __name__ == "__main__":
    main()
