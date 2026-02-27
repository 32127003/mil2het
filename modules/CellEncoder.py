"""Graph cell encoders with FIND-style prior interface.

Per-cell node features are built from:
- expression (1)
- global DEG z-score (1)
- celltype-specific regulation onehot (3)

Prior usage:
- Absolute prior from FIND-style prompt/query interface is injected only via FiLM.
- Relational prior from FIND-style prompt/query interface is injected only as sparse edge bias.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch_scatter import scatter_add, scatter_max

try:
    from .prior_interface_find import MultiViewPriorInterfaceFIND
except ImportError:  # pragma: no cover
    from prior_interface_find import MultiViewPriorInterfaceFIND


def scatter_softmax(logits: torch.Tensor, destination_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Segment-softmax over edges grouped by destination node.

    logits: [B, E, H]
    destination_index: [E]
    returns: [B, E, H]
    """
    if logits.dim() != 3:
        raise ValueError("Expected logits with shape [batch, edges, heads].")
    batch_size, num_edges, num_heads = logits.shape

    logits = logits.permute(0, 2, 1).contiguous()  # [B, H, E]
    index = destination_index.view(1, 1, num_edges).expand(batch_size, num_heads, num_edges)

    max_values, _ = scatter_max(logits, index, dim=2, dim_size=num_nodes)
    max_per_edge = max_values.gather(2, index)
    exp_values = torch.exp(logits - max_per_edge)
    denom = scatter_add(exp_values, index, dim=2, dim_size=num_nodes)
    denom_per_edge = denom.gather(2, index)
    attention = exp_values / (denom_per_edge + 1e-9)
    attention = attention.to(dtype=logits.dtype)
    return attention.permute(0, 2, 1).contiguous()


def infer_batch_chunk_size(
    batch_size: int,
    num_edges: int,
    feature_dimension: int,
    dtype: torch.dtype,
    max_message_memory_mb: float,
) -> int:
    if int(batch_size) <= 1:
        return int(batch_size)
    if int(num_edges) <= 0 or int(feature_dimension) <= 0:
        return int(batch_size)
    bytes_per_element = int(torch.empty((), dtype=dtype).element_size())
    per_sample_bytes = int(num_edges) * int(feature_dimension) * bytes_per_element
    budget_bytes = max(1, int(float(max_message_memory_mb) * 1024 * 1024))
    chunk_size = budget_bytes // max(1, per_sample_bytes)
    chunk_size = max(1, int(chunk_size))
    return min(int(batch_size), int(chunk_size))


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        number_of_heads: int,
        dropout_probability: float,
        attention_logit_clamp: float,
        max_message_memory_mb: float,
    ) -> None:
        super().__init__()
        self.output_dimension = int(output_dimension)
        self.number_of_heads = int(number_of_heads)
        self.attention_logit_clamp = float(attention_logit_clamp)
        self.max_message_memory_mb = float(max_message_memory_mb)

        self.linear_projection = nn.Linear(
            int(input_dimension),
            self.number_of_heads * self.output_dimension,
            bias=False,
        )
        self.attention_source = nn.Parameter(torch.randn(self.number_of_heads, self.output_dimension) * 0.1)
        self.attention_destination = nn.Parameter(torch.randn(self.number_of_heads, self.output_dimension) * 0.1)
        self.dropout = nn.Dropout(float(dropout_probability))

    def forward(
        self,
        node_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if node_hidden.dim() != 3:
            raise ValueError("node_hidden must have shape [batch, num_nodes, dim].")
        batch_size, num_nodes, _ = node_hidden.shape
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        source_index = edge_index[0]
        destination_index = edge_index[1]

        if edge_bias is not None:
            if edge_bias.dim() != 1 or int(edge_bias.shape[0]) != int(source_index.numel()):
                raise ValueError("edge_bias must have shape [num_edges].")
            edge_bias = edge_bias.to(device=node_hidden.device, dtype=node_hidden.dtype)

        projected = self.linear_projection(node_hidden)
        projected = projected.view(batch_size, num_nodes, self.number_of_heads, self.output_dimension)

        source_node_logits = (
            projected * self.attention_source.view(1, 1, self.number_of_heads, self.output_dimension)
        ).sum(dim=-1)
        destination_node_logits = (
            projected * self.attention_destination.view(1, 1, self.number_of_heads, self.output_dimension)
        ).sum(dim=-1)

        chunk_size = infer_batch_chunk_size(
            batch_size=batch_size,
            num_edges=int(source_index.numel()),
            feature_dimension=int(self.number_of_heads * self.output_dimension),
            dtype=projected.dtype,
            max_message_memory_mb=self.max_message_memory_mb,
        )

        aggregated_chunks = []
        attention_chunks = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            projected_chunk = projected[start:end]

            attention_logits = (
                source_node_logits[start:end, source_index, :]
                + destination_node_logits[start:end, destination_index, :]
            )
            attention_logits = torch_functional.leaky_relu(attention_logits, negative_slope=0.2)
            if edge_bias is not None:
                attention_logits = attention_logits + edge_bias.view(1, -1, 1)
            attention_logits = torch.clamp(
                attention_logits,
                -self.attention_logit_clamp,
                self.attention_logit_clamp,
            )
            attention_weights = scatter_softmax(attention_logits, destination_index, num_nodes)
            attention_weights = self.dropout(attention_weights)

            aggregated_chunk = projected_chunk.new_zeros((end - start, num_nodes, self.output_dimension))
            for head_index in range(self.number_of_heads):
                projected_source_head = projected_chunk[:, source_index, head_index, :]  # [b, E, F]
                weighted = projected_source_head * attention_weights[:, :, head_index].unsqueeze(-1)
                if weighted.dtype != aggregated_chunk.dtype:
                    weighted = weighted.to(dtype=aggregated_chunk.dtype)
                aggregated_chunk.index_add_(1, destination_index, weighted)
            aggregated_chunk = aggregated_chunk / float(self.number_of_heads)

            aggregated_chunks.append(aggregated_chunk)
            if return_attention:
                attention_chunks.append(attention_weights)

        aggregated = (
            torch.cat(aggregated_chunks, dim=0)
            if aggregated_chunks
            else projected.new_zeros((batch_size, num_nodes, self.output_dimension))
        )
        outputs: Dict[str, torch.Tensor] = {"node_hidden": aggregated}
        if return_attention:
            outputs["attention"] = (
                torch.cat(attention_chunks, dim=0)
                if attention_chunks
                else projected.new_zeros((batch_size, int(source_index.numel()), self.number_of_heads))
            )
        return outputs


class TransformerConvLayer(nn.Module):
    """Lightweight Transformer-style message passing over a fixed gene graph."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        number_of_heads: int,
        dropout_probability: float,
        attention_logit_clamp: float,
        max_message_memory_mb: float,
    ) -> None:
        super().__init__()
        self.output_dimension = int(output_dimension)
        self.number_of_heads = int(number_of_heads)
        self.attention_logit_clamp = float(attention_logit_clamp)
        self.max_message_memory_mb = float(max_message_memory_mb)

        self.query_projection = nn.Linear(
            int(input_dimension),
            self.number_of_heads * self.output_dimension,
            bias=False,
        )
        self.key_projection = nn.Linear(
            int(input_dimension),
            self.number_of_heads * self.output_dimension,
            bias=False,
        )
        self.value_projection = nn.Linear(
            int(input_dimension),
            self.number_of_heads * self.output_dimension,
            bias=False,
        )
        self.dropout = nn.Dropout(float(dropout_probability))
        self.scaling = float(1.0 / math.sqrt(max(1, int(self.output_dimension))))

    def forward(
        self,
        node_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if node_hidden.dim() != 3:
            raise ValueError("node_hidden must have shape [batch, num_nodes, dim].")
        batch_size, num_nodes, _ = node_hidden.shape
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        source_index = edge_index[0]
        destination_index = edge_index[1]

        if edge_bias is not None:
            if edge_bias.dim() != 1 or int(edge_bias.shape[0]) != int(source_index.numel()):
                raise ValueError("edge_bias must have shape [num_edges].")
            edge_bias = edge_bias.to(device=node_hidden.device, dtype=node_hidden.dtype)

        query = self.query_projection(node_hidden).view(
            batch_size,
            num_nodes,
            self.number_of_heads,
            self.output_dimension,
        )
        key = self.key_projection(node_hidden).view(
            batch_size,
            num_nodes,
            self.number_of_heads,
            self.output_dimension,
        )
        value = self.value_projection(node_hidden).view(
            batch_size,
            num_nodes,
            self.number_of_heads,
            self.output_dimension,
        )

        chunk_size = infer_batch_chunk_size(
            batch_size=batch_size,
            num_edges=int(source_index.numel()),
            feature_dimension=int(self.number_of_heads * self.output_dimension),
            dtype=query.dtype,
            max_message_memory_mb=self.max_message_memory_mb,
        )

        aggregated_chunks = []
        attention_chunks = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            query_chunk = query[start:end]
            key_chunk = key[start:end]
            value_chunk = value[start:end]

            logits = (
                query_chunk[:, destination_index, :, :] * key_chunk[:, source_index, :, :]
            ).sum(dim=-1) * self.scaling
            if edge_bias is not None:
                logits = logits + edge_bias.view(1, -1, 1)
            logits = torch.clamp(logits, -self.attention_logit_clamp, self.attention_logit_clamp)
            attention_weights = scatter_softmax(logits, destination_index, num_nodes)
            attention_weights = self.dropout(attention_weights)

            aggregated_chunk = value_chunk.new_zeros((end - start, num_nodes, self.output_dimension))
            for head_index in range(self.number_of_heads):
                value_source_head = value_chunk[:, source_index, head_index, :]  # [b, E, F]
                weighted = value_source_head * attention_weights[:, :, head_index].unsqueeze(-1)
                if weighted.dtype != aggregated_chunk.dtype:
                    weighted = weighted.to(dtype=aggregated_chunk.dtype)
                aggregated_chunk.index_add_(1, destination_index, weighted)
            aggregated_chunk = aggregated_chunk / float(self.number_of_heads)

            aggregated_chunks.append(aggregated_chunk)
            if return_attention:
                attention_chunks.append(attention_weights)

        aggregated = (
            torch.cat(aggregated_chunks, dim=0)
            if aggregated_chunks
            else value.new_zeros((batch_size, num_nodes, self.output_dimension))
        )
        outputs: Dict[str, torch.Tensor] = {"node_hidden": aggregated}
        if return_attention:
            outputs["attention"] = (
                torch.cat(attention_chunks, dim=0)
                if attention_chunks
                else value.new_zeros((batch_size, int(source_index.numel()), self.number_of_heads))
            )
        return outputs


class NodeAttentionReadout(nn.Module):
    def __init__(self, hidden_dimension: int, dropout_probability: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(int(hidden_dimension), int(hidden_dimension)),
            nn.Tanh(),
            nn.Dropout(float(dropout_probability)),
            nn.Linear(int(hidden_dimension), 1, bias=False),
        )

    def forward(self, node_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        if node_hidden.dim() != 3:
            raise ValueError("node_hidden must have shape [batch, num_nodes, dim].")
        logits = self.score(node_hidden).squeeze(-1)  # [B, N]
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(node_hidden * weights.unsqueeze(-1), dim=1)
        return {"pooled": pooled, "weights": weights, "logits": logits}


class NodeScoreHead(nn.Module):
    """Per-node biomarker score head used for sparse regularization."""

    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(hidden_dimension), 1, bias=True)

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        if node_hidden.dim() != 3:
            raise ValueError("node_hidden must have shape [batch, num_nodes, dim].")
        logits = self.linear(node_hidden).squeeze(-1)
        return torch.sigmoid(logits)


class GraphCellEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dimension: int,
        number_of_layers: int,
        number_of_heads: int,
        dropout_probability: float,
        edge_index: torch.Tensor,
        protein_prior_embeddings: torch.Tensor,
        global_zscore: torch.Tensor,
        regulation_onehot: torch.Tensor,
        num_celltypes: int,
        num_treatments: int,
        attention_logit_clamp: float,
        expression_feature_scale: float,
        max_message_memory_mb: float,
        graph_readout: str = "mean",
        protein_prior_alpha: float = 1.0,
        protein_prior_alpha_learnable: bool = False,
        node_feature_beta: float = 1.0,
        prior_injection_enabled: bool = True,
        prior_absolute_enabled: bool = True,
        prior_relational_enabled: bool = True,
        protein_prior_base_embeddings: Optional[torch.Tensor] = None,
        protein_prior_projection_hidden_dim: int = 0,
        protein_prior_projection_dropout: float = 0.1,
        protein_prior_projection_mode: str = "global",
        protein_prior_view_embeddings: Optional[Union[Dict[str, torch.Tensor], Sequence[torch.Tensor]]] = None,
        prior_interface_num_heads: int = 4,
        prior_interface_dropout: float = 0.1,
        prior_interface_ffn_hidden_dim: int = 0,
        prior_knn_k: int = 16,
        prior_knn_symmetric: bool = True,
        lambda_edge_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dimension = int(hidden_dimension)
        self.num_celltypes = int(num_celltypes)
        self.num_treatments = max(0, int(num_treatments))
        self.expression_feature_scale = float(expression_feature_scale)
        self.prior_injection_enabled = bool(prior_injection_enabled)
        self.prior_absolute_enabled = bool(prior_absolute_enabled) and bool(self.prior_injection_enabled)
        self.prior_relational_enabled = bool(prior_relational_enabled) and bool(self.prior_injection_enabled)
        self.graph_readout = str(graph_readout).lower()
        if self.graph_readout not in {"mean", "attention"}:
            raise ValueError("graph_readout must be one of {'mean','attention'}.")

        # Compatibility fields retained for external callers/checkpoint expectations.
        if bool(protein_prior_alpha_learnable):
            self.protein_prior_alpha = nn.Parameter(torch.tensor(float(protein_prior_alpha), dtype=torch.float32))
        else:
            self.register_buffer(
                "protein_prior_alpha",
                torch.tensor(float(protein_prior_alpha), dtype=torch.float32),
            )
        self.protein_prior_projection = None
        self.protein_prior_projection_mode = str(protein_prior_projection_mode)
        if protein_prior_base_embeddings is not None:
            self.register_buffer(
                "protein_prior_base_embeddings",
                protein_prior_base_embeddings.detach().float().contiguous(),
            )
        else:
            self.protein_prior_base_embeddings = None
        del protein_prior_projection_hidden_dim, protein_prior_projection_dropout

        if global_zscore.dim() != 1 or int(global_zscore.shape[0]) != int(self.num_nodes):
            raise ValueError("global_zscore must have shape [num_nodes].")
        self.register_buffer("global_zscore", global_zscore.float().contiguous())

        if regulation_onehot.dim() != 3:
            raise ValueError("regulation_onehot must have shape [num_celltypes, num_nodes, 3].")
        if (
            int(regulation_onehot.shape[0]) != int(self.num_celltypes)
            or int(regulation_onehot.shape[1]) != int(self.num_nodes)
            or int(regulation_onehot.shape[2]) != 3
        ):
            raise ValueError("regulation_onehot must have shape [num_celltypes, num_nodes, 3].")
        self.register_buffer("regulation_onehot", regulation_onehot.float().contiguous())

        node_feature_dim = 1 + 1 + 3
        self.node_feature_proj = nn.Sequential(
            nn.Linear(int(node_feature_dim), int(self.hidden_dimension), bias=True),
            nn.LayerNorm(int(self.hidden_dimension)),
        )
        self.register_buffer(
            "node_feature_beta",
            torch.tensor(float(node_feature_beta), dtype=torch.float32),
            persistent=False,
        )

        prior_views = self._prepare_prior_views(
            protein_prior_embeddings=protein_prior_embeddings,
            protein_prior_base_embeddings=protein_prior_base_embeddings,
            protein_prior_view_embeddings=protein_prior_view_embeddings,
        )
        self.prior_interface = MultiViewPriorInterfaceFIND(
            frozen_embeddings_by_view=prior_views,
            absolute_query_input_dimension=int(node_feature_dim),
            hidden_dimension=int(self.hidden_dimension),
            number_of_heads=int(prior_interface_num_heads),
            dropout_probability=float(prior_interface_dropout),
            feedforward_hidden_dimension=int(prior_interface_ffn_hidden_dim),
        )

        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        base_edge_index = edge_index.long().contiguous()
        prior_knn_edge_index = self.prior_interface.build_sparse_knn_edge_index(
            k=int(prior_knn_k),
            symmetric=bool(prior_knn_symmetric),
        ).to(device=base_edge_index.device)
        union_edge_index = self._merge_edge_indices(base_edge_index, prior_knn_edge_index)
        self.register_buffer("edge_index_ppi", base_edge_index)
        self.register_buffer("edge_index_knn", prior_knn_edge_index.long().contiguous())
        self.register_buffer("edge_index", union_edge_index.long().contiguous())

        self.lambda_edge_bias = (
            float(lambda_edge_bias) if bool(self.prior_relational_enabled) else 0.0
        )
        self.dropout = nn.Dropout(float(dropout_probability))

        layer_count = int(number_of_layers)
        self.gnn_layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    input_dimension=int(self.hidden_dimension),
                    output_dimension=int(self.hidden_dimension),
                    number_of_heads=int(number_of_heads),
                    dropout_probability=float(dropout_probability),
                    attention_logit_clamp=float(attention_logit_clamp),
                    max_message_memory_mb=float(max_message_memory_mb),
                )
                for _ in range(layer_count)
            ]
        )
        self.film_layer_norms = nn.ModuleList(
            [nn.LayerNorm(int(self.hidden_dimension)) for _ in range(layer_count)]
        )
        self.film_scale_layers = nn.ModuleList(
            [nn.Linear(int(self.hidden_dimension), int(self.hidden_dimension), bias=True) for _ in range(layer_count)]
        )
        self.film_shift_layers = nn.ModuleList(
            [nn.Linear(int(self.hidden_dimension), int(self.hidden_dimension), bias=True) for _ in range(layer_count)]
        )
        for scale_layer, shift_layer in zip(self.film_scale_layers, self.film_shift_layers):
            nn.init.zeros_(scale_layer.weight)
            nn.init.ones_(scale_layer.bias)
            nn.init.zeros_(shift_layer.weight)
            nn.init.zeros_(shift_layer.bias)

        self.node_score_head = NodeScoreHead(int(self.hidden_dimension))

        self.readout: Optional[nn.Module]
        if self.graph_readout == "attention":
            self.readout = NodeAttentionReadout(int(self.hidden_dimension), float(dropout_probability))
        else:
            self.readout = None

    @staticmethod
    def _merge_edge_indices(edge_index_a: torch.Tensor, edge_index_b: torch.Tensor) -> torch.Tensor:
        if int(edge_index_a.shape[0]) != 2 or int(edge_index_b.shape[0]) != 2:
            raise ValueError("edge indices must have shape [2, num_edges].")
        if int(edge_index_b.shape[1]) == 0:
            return edge_index_a.long().contiguous()

        num_nodes = int(max(int(edge_index_a.max().item()), int(edge_index_b.max().item())) + 1)
        combined = torch.cat([edge_index_a.long(), edge_index_b.long()], dim=1)
        edge_code = combined[0] * num_nodes + combined[1]
        unique_code = torch.unique(edge_code, sorted=True)
        source = torch.div(unique_code, num_nodes, rounding_mode="floor")
        destination = unique_code % num_nodes
        return torch.stack([source, destination], dim=0).long().contiguous()

    def _prepare_prior_views(
        self,
        *,
        protein_prior_embeddings: torch.Tensor,
        protein_prior_base_embeddings: Optional[torch.Tensor],
        protein_prior_view_embeddings: Optional[Union[Dict[str, torch.Tensor], Sequence[torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        if protein_prior_view_embeddings is None:
            if protein_prior_base_embeddings is not None:
                prior_views: Dict[str, torch.Tensor] = {
                    "base": protein_prior_base_embeddings.detach().float().contiguous()
                }
            else:
                prior_views = {"default": protein_prior_embeddings.detach().float().contiguous()}
            return self._validate_prior_views(prior_views)

        if isinstance(protein_prior_view_embeddings, dict):
            prior_views = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in protein_prior_view_embeddings.items()
            }
            return self._validate_prior_views(prior_views)

        if isinstance(protein_prior_view_embeddings, (list, tuple)):
            prior_views = {
                f"view_{index}": tensor.detach().float().contiguous()
                for index, tensor in enumerate(protein_prior_view_embeddings)
            }
            return self._validate_prior_views(prior_views)

        raise ValueError("protein_prior_view_embeddings must be None, dict, list, or tuple.")

    def _validate_prior_views(self, prior_views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(prior_views) <= 0:
            raise ValueError("At least one prior view must be provided.")
        for view_name, view_tensor in prior_views.items():
            if view_tensor.dim() != 2:
                raise ValueError(f"Prior view '{view_name}' must have shape [num_nodes, dim].")
            if int(view_tensor.shape[0]) != int(self.num_nodes):
                raise ValueError(
                    f"Prior view '{view_name}' has invalid num_nodes={int(view_tensor.shape[0])}; expected {int(self.num_nodes)}."
                )
            if int(view_tensor.shape[1]) <= 0:
                raise ValueError(f"Prior view '{view_name}' must have non-zero feature dimension.")
        return prior_views

    def _compute_edge_bias(self, relational_prior: torch.Tensor, target_dtype: torch.dtype) -> Optional[torch.Tensor]:
        if float(self.lambda_edge_bias) == 0.0:
            return None
        if relational_prior.dim() != 2 or int(relational_prior.shape[0]) != int(self.num_nodes):
            raise ValueError("relational_prior must have shape [num_nodes, hidden_dim].")

        normalized = torch_functional.normalize(
            relational_prior,
            p=2.0,
            dim=1,
            eps=1e-8,
        )
        source = self.edge_index[0]
        destination = self.edge_index[1]
        cosine_similarity = (normalized[source] * normalized[destination]).sum(dim=-1)
        edge_bias = float(self.lambda_edge_bias) * cosine_similarity
        return edge_bias.to(dtype=target_dtype)

    @torch.no_grad()
    def set_node_feature_beta(self, value: float) -> None:
        self.node_feature_beta.fill_(float(value))

    def forward(
        self,
        *,
        expression_values: torch.Tensor,
        celltype_index: torch.Tensor,
        treatment_index: Optional[torch.Tensor] = None,
        prior_distribution_table: Optional[torch.Tensor] = None,
        prior_group_index: Optional[torch.Tensor] = None,
        tau: float = 0.0,
        eps: float = 1e-8,
        return_node_outputs: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del treatment_index, prior_distribution_table, prior_group_index, tau, eps

        if expression_values.dim() != 2 or int(expression_values.shape[1]) != int(self.num_nodes):
            raise ValueError("expression_values must have shape [num_cells, num_nodes].")
        if celltype_index.dim() != 1 or int(celltype_index.shape[0]) != int(expression_values.shape[0]):
            raise ValueError("celltype_index must have shape [num_cells].")

        batch_size = int(expression_values.shape[0])

        expr = expression_values.float() * float(self.expression_feature_scale)
        expr = expr.view(batch_size, self.num_nodes, 1)

        z = self.global_zscore.view(1, self.num_nodes, 1).expand(batch_size, -1, -1)
        reg = self.regulation_onehot[celltype_index]  # [B, N, 3]

        node_features = torch.cat([expr, z, reg], dim=-1)
        hidden = self.node_feature_proj(node_features)
        beta_value = self.node_feature_beta.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden * beta_value

        if bool(self.prior_absolute_enabled) or bool(self.prior_relational_enabled):
            prompt_tokens = self.prior_interface.build_prompt_tokens(
                node_indices_or_gene_ids=None,
                device=hidden.device,
            )
        else:
            prompt_tokens = None

        if bool(self.prior_absolute_enabled):
            absolute_prior = self.prior_interface.forward_absolute(
                node_features=node_features,
                node_indices_or_gene_ids=None,
                prompt_tokens=prompt_tokens,
            ).to(device=hidden.device, dtype=hidden.dtype)
        else:
            absolute_prior = hidden.new_zeros((batch_size, self.num_nodes, self.hidden_dimension))

        if bool(self.prior_relational_enabled):
            relational_prior = self.prior_interface.forward_relational(
                node_indices_or_gene_ids=None,
                prompt_tokens=prompt_tokens,
            ).to(device=hidden.device, dtype=hidden.dtype)
            edge_bias = self._compute_edge_bias(relational_prior, target_dtype=hidden.dtype)
        else:
            edge_bias = None

        hidden = self.dropout(hidden)

        first_layer_attention = None
        node_hidden_layers = []
        for layer_index, layer in enumerate(self.gnn_layers):
            need_layer_attention = bool(return_attention and layer_index == 0)
            layer_outputs = layer(
                hidden,
                self.edge_index,
                return_attention=need_layer_attention,
                edge_bias=edge_bias,
            )
            hidden = layer_outputs["node_hidden"]

            film_scale = self.film_scale_layers[layer_index](absolute_prior)
            film_shift = self.film_shift_layers[layer_index](absolute_prior)
            hidden = film_scale * self.film_layer_norms[layer_index](hidden) + film_shift

            hidden = torch_functional.elu(hidden)
            hidden = self.dropout(hidden)
            node_hidden_layers.append(hidden)
            if need_layer_attention and "attention" in layer_outputs:
                first_layer_attention = layer_outputs["attention"].mean(dim=-1)  # [B, E]

        node_scores = self.node_score_head(hidden)

        if self.graph_readout == "mean":
            cell_embeddings = hidden.mean(dim=1)
            pooling_weights = None
        else:
            assert self.readout is not None
            readout_out = self.readout(hidden)
            cell_embeddings = readout_out["pooled"]
            pooling_weights = readout_out["weights"]

        outputs: Dict[str, torch.Tensor] = {
            "cell_embeddings": cell_embeddings,
            "node_scores": node_scores,
        }
        if return_node_outputs:
            outputs["node_hidden"] = hidden
            outputs["node_hidden_layers"] = tuple(node_hidden_layers)
        if return_attention and first_layer_attention is not None:
            outputs["first_layer_attention"] = first_layer_attention
        if return_attention and pooling_weights is not None:
            outputs["pooling_weights"] = pooling_weights
        return outputs


class TransformerConvCellEncoder(GraphCellEncoder):
    """TransformerConv-style cell encoder with FIND-style prior injection."""

    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dimension: int,
        number_of_layers: int,
        number_of_heads: int,
        dropout_probability: float,
        edge_index: torch.Tensor,
        protein_prior_embeddings: torch.Tensor,
        global_zscore: torch.Tensor,
        regulation_onehot: torch.Tensor,
        num_celltypes: int,
        num_treatments: int,
        attention_logit_clamp: float,
        expression_feature_scale: float,
        max_message_memory_mb: float,
        graph_readout: str = "flatten",
        protein_prior_alpha: float = 1.0,
        protein_prior_alpha_learnable: bool = False,
        node_feature_beta: float = 1.0,
        prior_injection_enabled: bool = True,
        prior_absolute_enabled: bool = True,
        prior_relational_enabled: bool = True,
        protein_prior_base_embeddings: Optional[torch.Tensor] = None,
        protein_prior_projection_hidden_dim: int = 0,
        protein_prior_projection_dropout: float = 0.1,
        protein_prior_projection_mode: str = "global",
        protein_prior_view_embeddings: Optional[Union[Dict[str, torch.Tensor], Sequence[torch.Tensor]]] = None,
        prior_interface_num_heads: int = 4,
        prior_interface_dropout: float = 0.1,
        prior_interface_ffn_hidden_dim: int = 0,
        prior_knn_k: int = 16,
        prior_knn_symmetric: bool = True,
        lambda_edge_bias: float = 0.0,
    ) -> None:
        super().__init__(
            num_nodes=num_nodes,
            hidden_dimension=hidden_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            dropout_probability=dropout_probability,
            edge_index=edge_index,
            protein_prior_embeddings=protein_prior_embeddings,
            global_zscore=global_zscore,
            regulation_onehot=regulation_onehot,
            num_celltypes=num_celltypes,
            num_treatments=num_treatments,
            attention_logit_clamp=attention_logit_clamp,
            expression_feature_scale=expression_feature_scale,
            max_message_memory_mb=max_message_memory_mb,
            graph_readout="mean",
            protein_prior_alpha=protein_prior_alpha,
            protein_prior_alpha_learnable=protein_prior_alpha_learnable,
            node_feature_beta=node_feature_beta,
            prior_injection_enabled=prior_injection_enabled,
            prior_absolute_enabled=prior_absolute_enabled,
            prior_relational_enabled=prior_relational_enabled,
            protein_prior_base_embeddings=protein_prior_base_embeddings,
            protein_prior_projection_hidden_dim=protein_prior_projection_hidden_dim,
            protein_prior_projection_dropout=protein_prior_projection_dropout,
            protein_prior_projection_mode=protein_prior_projection_mode,
            protein_prior_view_embeddings=protein_prior_view_embeddings,
            prior_interface_num_heads=prior_interface_num_heads,
            prior_interface_dropout=prior_interface_dropout,
            prior_interface_ffn_hidden_dim=prior_interface_ffn_hidden_dim,
            prior_knn_k=prior_knn_k,
            prior_knn_symmetric=prior_knn_symmetric,
            lambda_edge_bias=lambda_edge_bias,
        )

        readout_mode = str(graph_readout).strip().lower()
        if readout_mode in {"flat", "concat"}:
            readout_mode = "flatten"
        if readout_mode not in {"flatten", "mean", "attention"}:
            raise ValueError("graph_readout must be one of {'flatten','mean','attention'}.")
        self.transformer_graph_readout = readout_mode
        self.transformer_readout: Optional[nn.Module]
        if self.transformer_graph_readout == "attention":
            self.transformer_readout = NodeAttentionReadout(
                int(self.hidden_dimension),
                float(dropout_probability),
            )
        else:
            self.transformer_readout = None

        self.gnn_layers = nn.ModuleList(
            [
                TransformerConvLayer(
                    input_dimension=int(self.hidden_dimension),
                    output_dimension=int(self.hidden_dimension),
                    number_of_heads=int(number_of_heads),
                    dropout_probability=float(dropout_probability),
                    attention_logit_clamp=float(attention_logit_clamp),
                    max_message_memory_mb=float(max_message_memory_mb),
                )
                for _ in range(int(number_of_layers))
            ]
        )

    def forward(
        self,
        *,
        expression_values: torch.Tensor,
        celltype_index: torch.Tensor,
        treatment_index: Optional[torch.Tensor] = None,
        prior_distribution_table: Optional[torch.Tensor] = None,
        prior_group_index: Optional[torch.Tensor] = None,
        tau: float = 0.0,
        eps: float = 1e-8,
        return_node_outputs: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        base_outputs = super().forward(
            expression_values=expression_values,
            celltype_index=celltype_index,
            treatment_index=treatment_index,
            prior_distribution_table=prior_distribution_table,
            prior_group_index=prior_group_index,
            tau=tau,
            eps=eps,
            return_node_outputs=True,
            return_attention=return_attention,
        )
        node_hidden = base_outputs["node_hidden"]
        batch_size = int(node_hidden.shape[0])

        pooling_weights = None
        if self.transformer_graph_readout == "flatten":
            cell_embeddings = node_hidden.reshape(
                batch_size,
                int(self.num_nodes) * int(self.hidden_dimension),
            )
        elif self.transformer_graph_readout == "mean":
            cell_embeddings = node_hidden.mean(dim=1)
        else:
            assert self.transformer_readout is not None
            readout_out = self.transformer_readout(node_hidden)
            cell_embeddings = readout_out["pooled"]
            pooling_weights = readout_out["weights"]

        outputs: Dict[str, torch.Tensor] = {
            "cell_embeddings": cell_embeddings,
            "node_scores": base_outputs["node_scores"],
        }
        if return_node_outputs:
            outputs["node_hidden"] = node_hidden
            if "node_hidden_layers" in base_outputs:
                outputs["node_hidden_layers"] = base_outputs["node_hidden_layers"]
        if return_attention and "first_layer_attention" in base_outputs:
            outputs["first_layer_attention"] = base_outputs["first_layer_attention"]
        if return_attention and pooling_weights is not None:
            outputs["pooling_weights"] = pooling_weights
        return outputs
