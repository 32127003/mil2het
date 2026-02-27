"""Recursive refinement blocks for patient-level MIL aggregation."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch_scatter import scatter_add, scatter_max


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


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones((self.dimension,), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or int(x.shape[1]) != int(self.dimension):
            raise ValueError("RMSNorm input must have shape [batch, dimension].")
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight.view(1, -1)


class ScaledResidualConnector(nn.Module):
    """Connector used between recursive steps."""

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        self.embedding_dimension = int(embedding_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.norm = RMSNorm(self.embedding_dimension)
        self.scale = nn.Parameter(torch.zeros((self.embedding_dimension,), dtype=torch.float32))
        self.up_projection = nn.Linear(self.embedding_dimension, self.hidden_dimension, bias=True)
        self.down_projection = nn.Linear(self.hidden_dimension, self.embedding_dimension, bias=True)
        nn.init.zeros_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        self.dropout = nn.Dropout(float(dropout_probability))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        residual = normalized * self.scale.view(1, -1)
        mlp = self.down_projection(
            self.dropout(torch_functional.gelu(self.up_projection(normalized)))
        )
        return residual + mlp


class RecursivePatientRefiner(nn.Module):
    """Refine patient embeddings from anchor MIL output with recursive cross-attention."""

    def __init__(
        self,
        embedding_dimension: int,
        num_celltypes: int,
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
        self.recursive_steps = max(1, int(recursive_steps))
        self.recursion_celltype_aware = bool(recursion_celltype_aware)
        self.recursion_dropout = nn.Dropout(float(recursion_dropout))

        self.rec_query_projection: Optional[nn.Linear] = None
        self.rec_key_projection: Optional[nn.Linear] = None
        self.rec_score_projection: Optional[nn.Linear] = None
        self.rec_value_projection: Optional[nn.Linear] = None
        self.rec_update_projection: Optional[nn.Linear] = None
        self.rec_celltype_embedding: Optional[nn.Embedding] = None
        self.rec_celltype_key_projection: Optional[nn.Linear] = None
        self.rec_celltype_value_projection: Optional[nn.Linear] = None
        self.connector: Optional[ScaledResidualConnector] = None

        # Recursive refinement is ablated when recursive_steps <= 1.
        # Avoid allocating large projection matrices in this case.
        if int(self.recursive_steps) <= 1:
            return

        hidden = int(attention_hidden_dimension)
        self.rec_query_projection = nn.Linear(self.embedding_dimension, hidden, bias=True)
        self.rec_key_projection = nn.Linear(self.embedding_dimension, hidden, bias=True)
        self.rec_score_projection = nn.Linear(hidden, 1, bias=False)
        self.rec_value_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension, bias=True)
        self.rec_update_projection = nn.Linear(self.embedding_dimension * 2, self.embedding_dimension, bias=True)

        if self.recursion_celltype_aware:
            self.rec_celltype_embedding = nn.Embedding(self.num_celltypes, self.embedding_dimension)
            self.rec_celltype_key_projection = nn.Linear(self.embedding_dimension, hidden, bias=False)
            self.rec_celltype_value_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension, bias=False)
        else:
            self.rec_celltype_embedding = None
            self.rec_celltype_key_projection = None
            self.rec_celltype_value_projection = None

        connector_hidden = int(connector_hidden_dimension)
        if connector_hidden <= 0:
            connector_hidden = self.embedding_dimension * 2
        self.connector = ScaledResidualConnector(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=connector_hidden,
            dropout_probability=float(connector_dropout),
        )

    def _recursive_step(
        self,
        patient_input: torch.Tensor,
        cell_embeddings: torch.Tensor,
        celltype_index: torch.Tensor,
        bag_index: torch.Tensor,
        num_bags_int: int,
    ) -> Dict[str, torch.Tensor]:
        if (
            self.rec_query_projection is None
            or self.rec_key_projection is None
            or self.rec_score_projection is None
            or self.rec_value_projection is None
            or self.rec_update_projection is None
        ):
            raise RuntimeError("Recursive step called but recursive projections are not initialized.")

        query = self.rec_query_projection(patient_input)[bag_index]
        key = self.rec_key_projection(cell_embeddings)
        value = self.rec_value_projection(cell_embeddings)

        if self.recursion_celltype_aware and self.rec_celltype_embedding is not None:
            safe_celltype = torch.clamp(celltype_index, min=0, max=self.num_celltypes - 1)
            celltype_embedding = self.rec_celltype_embedding(safe_celltype)
            assert self.rec_celltype_key_projection is not None
            assert self.rec_celltype_value_projection is not None
            key = key + self.rec_celltype_key_projection(celltype_embedding)
            value = value + self.rec_celltype_value_projection(celltype_embedding)

        logits = self.rec_score_projection(torch.tanh(query + key)).squeeze(-1)
        attention = scatter_softmax_1d(logits, bag_index, num_bags_int)
        message = scatter_add(
            value * attention.unsqueeze(-1),
            bag_index,
            dim=0,
            dim_size=num_bags_int,
        )
        update = self.rec_update_projection(torch.cat([patient_input, message], dim=-1))
        patient_backbone = patient_input + self.recursion_dropout(update)
        return {"patient_backbone": patient_backbone, "gamma_attention": attention}

    def forward(
        self,
        *,
        anchor_embeddings: torch.Tensor,
        base_gamma_attention: torch.Tensor,
        cell_embeddings: torch.Tensor,
        celltype_index: torch.Tensor,
        bag_index: torch.Tensor,
        num_bags_int: int,
    ) -> Dict[str, object]:
        step_embeddings: List[torch.Tensor] = [anchor_embeddings]
        step_inputs: List[torch.Tensor] = [anchor_embeddings]
        step_gamma: List[torch.Tensor] = [base_gamma_attention]
        patient_embeddings = anchor_embeddings
        gamma_attention = base_gamma_attention

        if int(self.recursive_steps) > 1:
            current_input = anchor_embeddings
            step_embeddings = []
            step_inputs = [anchor_embeddings]
            step_gamma = []
            for _ in range(int(self.recursive_steps)):
                recursive_outputs = self._recursive_step(
                    patient_input=current_input,
                    cell_embeddings=cell_embeddings,
                    celltype_index=celltype_index,
                    bag_index=bag_index,
                    num_bags_int=num_bags_int,
                )
                patient_backbone = recursive_outputs["patient_backbone"]
                step_embeddings.append(patient_backbone)
                step_gamma.append(recursive_outputs["gamma_attention"])

                assert self.connector is not None
                delta = self.connector(patient_backbone)
                current_input = anchor_embeddings + delta
                step_inputs.append(current_input)

            patient_embeddings = step_embeddings[-1]
            gamma_attention = step_gamma[-1]

        return {
            "patient_embeddings": patient_embeddings,
            "gamma_attention": gamma_attention,
            "patient_embeddings_anchor": anchor_embeddings,
            "patient_embeddings_steps": tuple(step_embeddings),
            "patient_embeddings_inputs": tuple(step_inputs),
            "gamma_attention_steps": tuple(step_gamma),
            "recursive_steps": int(len(step_embeddings)),
        }
