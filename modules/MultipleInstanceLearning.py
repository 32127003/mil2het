"""Multiple-instance learning pooling with optional recursive patient refinement."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max

try:
    from .recursion import RecursivePatientRefiner
except ImportError:  # pragma: no cover
    from recursion import RecursivePatientRefiner


def scatter_softmax_1d(logits: torch.Tensor, segment_index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.dim() != 1:
        raise ValueError("logits must have shape [num_items].")
    if segment_index.dim() != 1:
        raise ValueError("segment_index must have shape [num_items].")
    if int(logits.shape[0]) != int(segment_index.shape[0]):
        raise ValueError("logits and segment_index must have same length.")
    if int(num_segments) <= 0:
        raise ValueError("num_segments must be positive.")

    max_per_segment, _ = scatter_max(logits, segment_index, dim=0, dim_size=int(num_segments))
    max_per_item = max_per_segment[segment_index]
    exp_values = torch.exp(logits - max_per_item)
    denom = scatter_add(exp_values, segment_index, dim=0, dim_size=int(num_segments))
    return exp_values / (denom[segment_index] + 1e-9)


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(int(input_dimension), int(hidden_dimension), bias=True)
        self.gate_projection = nn.Linear(int(input_dimension), int(hidden_dimension), bias=True)
        self.score_projection = nn.Linear(int(hidden_dimension), 1, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        segment_index: torch.Tensor,
        num_segments: int,
    ) -> Dict[str, torch.Tensor]:
        if embeddings.dim() != 2:
            raise ValueError("embeddings must have shape [num_instances, dim].")
        gated = torch.tanh(self.value_projection(embeddings)) * torch.sigmoid(self.gate_projection(embeddings))
        logits = self.score_projection(gated).squeeze(-1)
        attention = scatter_softmax_1d(logits, segment_index, int(num_segments))
        pooled = scatter_add(embeddings * attention.unsqueeze(-1), segment_index, dim=0, dim_size=int(num_segments))
        return {"pooled": pooled, "attention": attention, "logits": logits}


class PatientMILAggregator(nn.Module):
    """Aggregate cell embeddings to patient embeddings, optionally with recursive refinement."""

    def __init__(
        self,
        embedding_dimension: int,
        num_celltypes: int,
        pooling: str,
        attention_hidden_dimension: int,
        recursive_steps: int = 1,
        connector_hidden_dimension: int = 0,
        connector_dropout: float = 0.1,
        recursion_dropout: float = 0.1,
        recursion_celltype_aware: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dimension = int(embedding_dimension)
        self.num_celltypes = int(num_celltypes)
        print(pooling)
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be one of 'attention', 'mean'.")
        self.pooling = pooling

        hidden = int(attention_hidden_dimension)
        self.cell_pool = GatedAttentionMIL(self.embedding_dimension, hidden)
        self.celltype_pool = GatedAttentionMIL(self.embedding_dimension, hidden)
        self.recursive_refiner = RecursivePatientRefiner(
            embedding_dimension=self.embedding_dimension,
            num_celltypes=self.num_celltypes,
            attention_hidden_dimension=hidden,
            recursive_steps=int(recursive_steps),
            connector_hidden_dimension=int(connector_hidden_dimension),
            connector_dropout=float(connector_dropout),
            recursion_dropout=float(recursion_dropout),
            recursion_celltype_aware=bool(recursion_celltype_aware),
        )

    def _base_pool(
        self,
        cell_embeddings: torch.Tensor,
        celltype_index: torch.Tensor,
        bag_index: torch.Tensor,
        num_bags_int: int,
    ) -> Dict[str, torch.Tensor]:
        type_segment_index = bag_index * self.num_celltypes + celltype_index
        num_type_segments = num_bags_int * self.num_celltypes

        if self.pooling == "attention":
            flat_outputs = self.cell_pool(cell_embeddings, bag_index, num_bags_int)
            patient_embeddings = flat_outputs["pooled"]
            gamma = flat_outputs["attention"]

            celltype_attention_full = scatter_add(gamma, type_segment_index, dim=0, dim_size=num_type_segments)
            denominator = celltype_attention_full[type_segment_index].clamp_min(1e-9)
            alpha = gamma / denominator

            active_type_mask = celltype_attention_full > 0
            active_type_segment_index = torch.nonzero(active_type_mask, as_tuple=False).view(-1)
            beta = celltype_attention_full[active_type_segment_index]
            beta_patient_index = torch.div(active_type_segment_index, self.num_celltypes, rounding_mode="floor")
            beta_celltype_index = active_type_segment_index.remainder(self.num_celltypes)

        elif self.pooling == "mean":
            bag_counts = scatter_add(
                torch.ones_like(bag_index, dtype=cell_embeddings.dtype),
                bag_index,
                dim=0,
                dim_size=num_bags_int,
            ).clamp_min(1.0)
            patient_embeddings = scatter_add(cell_embeddings, bag_index, dim=0, dim_size=num_bags_int) / bag_counts.unsqueeze(-1)
            gamma = 1.0 / bag_counts[bag_index]

            celltype_attention_full = scatter_add(gamma, type_segment_index, dim=0, dim_size=num_type_segments)
            denominator = celltype_attention_full[type_segment_index].clamp_min(1e-9)
            alpha = gamma / denominator

            active_type_mask = celltype_attention_full > 0
            active_type_segment_index = torch.nonzero(active_type_mask, as_tuple=False).view(-1)
            beta = celltype_attention_full[active_type_segment_index]
            beta_patient_index = torch.div(active_type_segment_index, self.num_celltypes, rounding_mode="floor")
            beta_celltype_index = active_type_segment_index.remainder(self.num_celltypes)

        else:
            type_counts = scatter_add(
                torch.ones_like(type_segment_index, dtype=cell_embeddings.dtype),
                type_segment_index,
                dim=0,
                dim_size=num_type_segments,
            )
            within_type_outputs = self.cell_pool(cell_embeddings, type_segment_index, num_type_segments)
            celltype_embeddings_full = within_type_outputs["pooled"]
            alpha = within_type_outputs["attention"]

            active_type_mask = type_counts > 0
            active_type_segment_index = torch.nonzero(active_type_mask, as_tuple=False).view(-1)
            active_patient_index = torch.div(active_type_segment_index, self.num_celltypes, rounding_mode="floor")
            active_celltype_index = active_type_segment_index.remainder(self.num_celltypes)
            active_celltype_embeddings = celltype_embeddings_full[active_type_segment_index]

            patient_outputs = self.celltype_pool(active_celltype_embeddings, active_patient_index, num_bags_int)
            patient_embeddings = patient_outputs["pooled"]
            beta = patient_outputs["attention"]
            beta_patient_index = active_patient_index
            beta_celltype_index = active_celltype_index

            beta_full = torch.zeros((num_type_segments,), dtype=beta.dtype, device=beta.device)
            beta_full[active_type_segment_index] = beta
            gamma = alpha * beta_full[type_segment_index]

        return {
            "patient_embeddings": patient_embeddings,
            "gamma_attention": gamma,
            "alpha_attention": alpha,
            "beta_attention": beta,
            "beta_patient_index": beta_patient_index,
            "beta_celltype_index": beta_celltype_index,
        }

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        celltype_index: torch.Tensor,
        bag_index: torch.Tensor,
        num_bags: int,
    ) -> Dict[str, torch.Tensor]:
        if cell_embeddings.dim() != 2:
            raise ValueError("cell_embeddings must have shape [num_cells, dim].")
        if celltype_index.dim() != 1 or bag_index.dim() != 1:
            raise ValueError("celltype_index and bag_index must have shape [num_cells].")
        if int(cell_embeddings.shape[0]) != int(celltype_index.shape[0]) or int(cell_embeddings.shape[0]) != int(bag_index.shape[0]):
            raise ValueError("cell_embeddings/celltype_index/bag_index lengths must match.")

        num_bags_int = int(num_bags)
        if num_bags_int <= 0:
            raise ValueError("num_bags must be positive.")

        base_outputs = self._base_pool(
            cell_embeddings=cell_embeddings,
            celltype_index=celltype_index,
            bag_index=bag_index,
            num_bags_int=num_bags_int,
        )
        recursion_outputs = self.recursive_refiner(
            anchor_embeddings=base_outputs["patient_embeddings"],
            base_gamma_attention=base_outputs["gamma_attention"],
            cell_embeddings=cell_embeddings,
            celltype_index=celltype_index,
            bag_index=bag_index,
            num_bags_int=num_bags_int,
        )
        base_outputs.update(recursion_outputs)
        return base_outputs
