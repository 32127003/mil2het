"""multiview prior modulator for gene nodes.

This module adapts frozen multiview protein embeddings into two priors:
- absolute prior: per-cell/per-node representation for FiLM modulation
- relational prior: cell-invariant per-node representation for edge bias
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

try:
    from torch.nn.attention import SDPBackend as _SDPBackend
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel
except Exception:
    _SDPBackend = None
    _sdpa_kernel = None


class _CrossAttentionAdapter(nn.Module):
    """Single-layer cross-attention followed by a small FFN."""

    def __init__(
        self,
        hidden_dimension: int,
        number_of_heads: int,
        dropout_probability: float,
        feedforward_hidden_dimension: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dimension)
        heads = int(number_of_heads)
        if heads <= 0:
            raise ValueError("number_of_heads must be positive.")
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dimension ({hidden_dim}) must be divisible by number_of_heads ({heads})."
            )
        self.number_of_heads = heads
        # SDPA kernels can fail when flattened batch becomes too large.
        # Keep chunk size below a conservative per-head launch threshold.
        max_batches_by_head = max(1, (65535 // int(self.number_of_heads)) - 1)
        self.max_attention_batch = int(min(32768, max_batches_by_head))

        # Keep native MHA path for speed; use a guarded SDP kernel context to avoid
        # flaky flash-kernel launches on large flattened batches.
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=int(number_of_heads),
            dropout=float(dropout_probability),
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(float(dropout_probability))
        self.attention_norm = nn.LayerNorm(hidden_dim)

        ffn_hidden = int(feedforward_hidden_dimension)
        if ffn_hidden <= 0:
            ffn_hidden = int(hidden_dim * 2)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden, bias=True),
            nn.SiLU(),
            nn.Dropout(float(dropout_probability)),
            nn.Linear(ffn_hidden, hidden_dim, bias=True),
        )
        self.feedforward_dropout = nn.Dropout(float(dropout_probability))
        self.feedforward_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _native_sdp_context(device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        # Prefer new API to avoid deprecated warning spam that breaks tqdm rendering.
        if _sdpa_kernel is not None and _SDPBackend is not None:
            try:
                return _sdpa_kernel(backends=[_SDPBackend.MATH])
            except TypeError:
                # Compatibility for older early variants of torch.nn.attention.sdpa_kernel.
                return _sdpa_kernel([_SDPBackend.MATH])
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "sdp_kernel"):
            # Use math-only SDP kernel for stable launches on flattened large batches.
            return cuda_backend.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        # torch<2.0 has no SDP kernel context API.
        return nullcontext()

    @staticmethod
    def _autocast_off_context(device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _run_native_attention_chunked(
        self,
        query: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(query.shape[0])
        if batch_size <= int(self.max_attention_batch):
            with self._native_sdp_context(query.device):
                output, _ = self.attention(
                    query=query,
                    key=prompt_tokens,
                    value=prompt_tokens,
                    need_weights=False,
                )
            return output

        outputs = []
        for start in range(0, batch_size, int(self.max_attention_batch)):
            end = min(batch_size, start + int(self.max_attention_batch))
            with self._native_sdp_context(query.device):
                chunk_output, _ = self.attention(
                    query=query[start:end],
                    key=prompt_tokens[start:end],
                    value=prompt_tokens[start:end],
                    need_weights=False,
                )
            outputs.append(chunk_output)
        return torch.cat(outputs, dim=0)

    def forward(self, query: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3 or prompt_tokens.dim() != 3:
            raise ValueError("query and prompt_tokens must have shape [batch, sequence, hidden_dim].")
        if int(query.shape[0]) != int(prompt_tokens.shape[0]):
            raise ValueError("query and prompt_tokens must have the same batch size.")
        if int(query.shape[2]) != int(prompt_tokens.shape[2]):
            raise ValueError("query and prompt_tokens must have the same hidden dimension.")
        output_dtype = query.dtype
        query_fp32 = query.float()
        prompt_tokens_fp32 = prompt_tokens.float()

        with self._autocast_off_context(query.device):
            attention_output = self._run_native_attention_chunked(query_fp32, prompt_tokens_fp32)
            query_fp32 = self.attention_norm(query_fp32 + self.attention_dropout(attention_output))
            feedforward_output = self.feedforward(query_fp32)
            query_fp32 = self.feedforward_norm(query_fp32 + self.feedforward_dropout(feedforward_output))
        return query_fp32.to(dtype=output_dtype)

    def forward_with_shared_prompts(
        self,
        query: torch.Tensor,
        prompt_tokens_shared: torch.Tensor,
    ) -> torch.Tensor:
        if query.dim() != 3 or prompt_tokens_shared.dim() != 3:
            raise ValueError("query must be [B,N,D] and prompt_tokens_shared must be [N,K,D].")
        if int(query.shape[1]) != int(prompt_tokens_shared.shape[0]):
            raise ValueError("query and prompt_tokens_shared must agree on num_nodes.")
        if int(query.shape[2]) != int(prompt_tokens_shared.shape[2]):
            raise ValueError("query and prompt_tokens_shared must agree on hidden dimension.")

        output_dtype = query.dtype
        query_fp32 = query.float()
        prompt_tokens_shared_fp32 = prompt_tokens_shared.float()

        batch_size = int(query.shape[0])
        number_of_nodes = int(query.shape[1])
        hidden_dimension = int(query.shape[2])
        prompt_length = int(prompt_tokens_shared.shape[1])

        node_chunk_size = max(1, int(self.max_attention_batch) // max(1, batch_size))
        output_chunks = []
        with self._autocast_off_context(query.device):
            for start in range(0, number_of_nodes, node_chunk_size):
                end = min(number_of_nodes, start + node_chunk_size)
                chunk_nodes = int(end - start)
                query_flat = query_fp32[:, start:end, :].reshape(batch_size * chunk_nodes, 1, hidden_dimension)
                prompt_chunk = prompt_tokens_shared_fp32[start:end, :, :]
                prompt_flat = (
                    prompt_chunk
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1, -1)
                    .reshape(batch_size * chunk_nodes, prompt_length, hidden_dimension)
                )
                output_flat = self._run_native_attention_chunked(query_flat, prompt_flat)
                output_chunks.append(output_flat.reshape(batch_size, chunk_nodes, hidden_dimension))
            attention_output = torch.cat(output_chunks, dim=1)

            query_fp32 = self.attention_norm(query_fp32 + self.attention_dropout(attention_output))
            feedforward_output = self.feedforward(query_fp32)
            query_fp32 = self.feedforward_norm(query_fp32 + self.feedforward_dropout(feedforward_output))
        return query_fp32.to(dtype=output_dtype)


class MultiViewPriorInterfaceFIND(nn.Module):
    """Prompt/query interface over frozen multiview protein embeddings."""

    def __init__(
        self,
        *,
        frozen_embeddings_by_view: Union[Dict[str, torch.Tensor], Sequence[torch.Tensor]],
        absolute_query_input_dimension: int,
        hidden_dimension: int,
        number_of_heads: int = 4,
        dropout_probability: float = 0.1,
        feedforward_hidden_dimension: int = 0,
    ) -> None:
        super().__init__()

        if isinstance(frozen_embeddings_by_view, dict):
            ordered_items = [(str(name), tensor) for name, tensor in frozen_embeddings_by_view.items()]
        elif isinstance(frozen_embeddings_by_view, (list, tuple)):
            ordered_items = [(f"view_{index}", tensor) for index, tensor in enumerate(frozen_embeddings_by_view)]
        else:
            raise ValueError("frozen_embeddings_by_view must be a dict or a sequence of tensors.")
        if len(ordered_items) == 0:
            raise ValueError("frozen_embeddings_by_view must contain at least one view.")

        self.view_names = [name for name, _ in ordered_items]
        self.number_of_views = int(len(ordered_items))
        self.hidden_dimension = int(hidden_dimension)

        number_of_nodes: Optional[int] = None
        projection_layers = []
        projection_norms = []

        for view_index, (view_name, view_tensor) in enumerate(ordered_items):
            if not isinstance(view_tensor, torch.Tensor):
                raise ValueError(f"View '{view_name}' must be a torch.Tensor.")
            if view_tensor.dim() != 2:
                raise ValueError(f"View '{view_name}' must have shape [num_nodes, dim].")
            if number_of_nodes is None:
                number_of_nodes = int(view_tensor.shape[0])
            elif int(view_tensor.shape[0]) != int(number_of_nodes):
                raise ValueError("All views must have the same number of nodes.")

            self.register_buffer(
                f"frozen_view_embedding_{view_index}",
                view_tensor.detach().float().contiguous(),
            )
            projection_layers.append(
                nn.Linear(int(view_tensor.shape[1]), int(self.hidden_dimension), bias=True)
            )
            projection_norms.append(nn.LayerNorm(int(self.hidden_dimension)))

        assert number_of_nodes is not None
        self.number_of_nodes = int(number_of_nodes)
        self.prompt_projection_layers = nn.ModuleList(projection_layers)
        self.prompt_projection_norms = nn.ModuleList(projection_norms)

        self.absolute_query_projection = nn.Linear(
            int(absolute_query_input_dimension),
            int(self.hidden_dimension),
            bias=True,
        )
        self.absolute_adapter = _CrossAttentionAdapter(
            hidden_dimension=int(self.hidden_dimension),
            number_of_heads=int(number_of_heads),
            dropout_probability=float(dropout_probability),
            feedforward_hidden_dimension=int(feedforward_hidden_dimension),
        )

        self.relational_query = nn.Parameter(torch.empty(int(self.hidden_dimension)))
        nn.init.normal_(self.relational_query, mean=0.0, std=0.02)
        self.relational_adapter = _CrossAttentionAdapter(
            hidden_dimension=int(self.hidden_dimension),
            number_of_heads=int(number_of_heads),
            dropout_probability=float(dropout_probability),
            feedforward_hidden_dimension=int(feedforward_hidden_dimension),
        )
        self._eval_prompt_cache: Optional[torch.Tensor] = None

    def _clear_eval_cache(self) -> None:
        self._eval_prompt_cache = None

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(mode):
            self._clear_eval_cache()
        return self

    def _resolve_index_tensor(
        self,
        node_indices_or_gene_ids,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if node_indices_or_gene_ids is None:
            return torch.arange(self.number_of_nodes, device=device, dtype=torch.long)
        if isinstance(node_indices_or_gene_ids, torch.Tensor):
            index_tensor = node_indices_or_gene_ids.to(device=device, dtype=torch.long)
        else:
            index_tensor = torch.as_tensor(node_indices_or_gene_ids, device=device)
            if index_tensor.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError("node_indices_or_gene_ids must contain integer node indices.")
            index_tensor = index_tensor.long()
        return index_tensor

    def _frozen_view_tensor(self, view_index: int) -> torch.Tensor:
        return getattr(self, f"frozen_view_embedding_{int(view_index)}")

    def _build_prompt_tokens(self, flat_node_indices: torch.Tensor) -> torch.Tensor:
        if flat_node_indices.dim() != 1:
            raise ValueError("flat_node_indices must be a 1D tensor.")

        prompt_tokens_by_view = []
        for view_index in range(self.number_of_views):
            frozen_embeddings = self._frozen_view_tensor(view_index)
            view_embeddings = frozen_embeddings[flat_node_indices]
            projected = self.prompt_projection_layers[view_index](view_embeddings)
            projected = self.prompt_projection_norms[view_index](projected)
            prompt_tokens_by_view.append(projected)
        prompt_tokens = torch.stack(prompt_tokens_by_view, dim=1)
        return prompt_tokens

    def build_prompt_tokens(
        self,
        node_indices_or_gene_ids=None,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        resolve_device = self.relational_query.device if device is None else device
        if node_indices_or_gene_ids is None and not self.training:
            cached = self._eval_prompt_cache
            if cached is not None and cached.device == resolve_device:
                return cached

        node_indices = self._resolve_index_tensor(
            node_indices_or_gene_ids,
            device=resolve_device,
        )
        if node_indices.dim() != 1:
            raise ValueError("build_prompt_tokens expects 1D node indices.")
        prompt_tokens = self._build_prompt_tokens(node_indices)
        if node_indices_or_gene_ids is None and not self.training:
            self._eval_prompt_cache = prompt_tokens.detach()
        return prompt_tokens

    def forward_absolute(
        self,
        node_features: torch.Tensor,
        node_indices_or_gene_ids=None,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.dim() not in {2, 3}:
            raise ValueError("node_features must have shape [num_nodes, dim] or [batch, num_nodes, dim].")

        squeeze_batch = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            squeeze_batch = True

        batch_size, number_of_nodes, _ = node_features.shape
        node_indices = self._resolve_index_tensor(
            node_indices_or_gene_ids,
            device=node_features.device,
        )

        if node_indices.dim() == 1:
            if int(node_indices.numel()) != int(number_of_nodes):
                raise ValueError(
                    "1D node_indices_or_gene_ids must match num_nodes in node_features."
                )
            if prompt_tokens is None:
                prompt_tokens_shared = self._build_prompt_tokens(node_indices)
            else:
                prompt_tokens_shared = prompt_tokens
            if prompt_tokens_shared.dim() != 3:
                raise ValueError("prompt_tokens for 1D node indices must have shape [num_nodes, num_views, hidden_dim].")
            if int(prompt_tokens_shared.shape[0]) != int(number_of_nodes):
                raise ValueError("prompt_tokens num_nodes dimension does not match node_features.")
            if int(prompt_tokens_shared.shape[1]) != int(self.number_of_views):
                raise ValueError("prompt_tokens num_views dimension is invalid.")
            if int(prompt_tokens_shared.shape[2]) != int(self.hidden_dimension):
                raise ValueError("prompt_tokens hidden dimension is invalid.")
            prompt_tokens_shared = prompt_tokens_shared.to(device=node_features.device, dtype=node_features.dtype)
        elif node_indices.dim() == 2:
            if int(node_indices.shape[0]) != int(batch_size) or int(node_indices.shape[1]) != int(number_of_nodes):
                raise ValueError(
                    "2D node_indices_or_gene_ids must have shape [batch, num_nodes]."
                )
            if prompt_tokens is None:
                prompt_tokens_flat = self._build_prompt_tokens(node_indices.reshape(-1))
            else:
                prompt_tokens_flat = prompt_tokens
            if prompt_tokens_flat.dim() != 3:
                raise ValueError("prompt_tokens for 2D node indices must have shape [batch*num_nodes, num_views, hidden_dim].")
            prompt_tokens_flat = prompt_tokens_flat.to(device=node_features.device, dtype=node_features.dtype)
        else:
            raise ValueError("node_indices_or_gene_ids must be 1D or 2D when provided.")

        initial_query = self.absolute_query_projection(node_features)
        if node_indices.dim() == 1:
            fused_query = self.absolute_adapter.forward_with_shared_prompts(
                initial_query,
                prompt_tokens_shared,
            )
        else:
            initial_query = initial_query.reshape(batch_size * number_of_nodes, 1, self.hidden_dimension)
            fused_query = self.absolute_adapter(initial_query, prompt_tokens_flat)
            fused_query = fused_query.reshape(batch_size, number_of_nodes, self.hidden_dimension)

        if squeeze_batch:
            fused_query = fused_query.squeeze(0)
        return fused_query

    def forward_relational(
        self,
        node_indices_or_gene_ids=None,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_indices = self._resolve_index_tensor(
            node_indices_or_gene_ids,
            device=self.relational_query.device,
        )
        if node_indices.dim() != 1:
            raise ValueError("forward_relational expects a 1D node index tensor.")

        if prompt_tokens is None:
            prompt_tokens_local = self._build_prompt_tokens(node_indices)
        else:
            prompt_tokens_local = prompt_tokens
        if prompt_tokens_local.dim() != 3:
            raise ValueError("prompt_tokens must have shape [num_nodes, num_views, hidden_dim].")
        prompt_tokens_local = prompt_tokens_local.to(
            device=self.relational_query.device,
            dtype=self.relational_query.dtype,
        )
        relational_query = self.relational_query.view(1, 1, self.hidden_dimension).expand(
            int(prompt_tokens_local.shape[0]),
            1,
            self.hidden_dimension,
        )
        fused_relational = self.relational_adapter(relational_query, prompt_tokens_local)
        return fused_relational.squeeze(1)

    @torch.no_grad()
    def build_sparse_knn_edge_index(
        self,
        *,
        k: int,
        symmetric: bool = True,
        similarity_eps: float = 1e-8,
    ) -> torch.Tensor:
        number_of_nodes = int(self.number_of_nodes)
        top_k = min(max(0, int(k)), max(0, number_of_nodes - 1))
        if top_k <= 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self._frozen_view_tensor(0).device)

        normalized_views = []
        for view_index in range(self.number_of_views):
            view_tensor = self._frozen_view_tensor(view_index)
            normalized_views.append(
                torch_functional.normalize(
                    view_tensor,
                    p=2.0,
                    dim=1,
                    eps=max(float(similarity_eps), 1e-12),
                )
            )
        merged = torch.cat(normalized_views, dim=1)
        merged = torch_functional.normalize(
            merged,
            p=2.0,
            dim=1,
            eps=max(float(similarity_eps), 1e-12),
        )

        similarity = merged @ merged.transpose(0, 1)
        similarity.fill_diagonal_(-float("inf"))
        neighbor_index = torch.topk(similarity, k=top_k, dim=1, largest=True, sorted=False).indices

        edge_device = neighbor_index.device
        source = (
            torch.arange(number_of_nodes, dtype=torch.long, device=edge_device)
            .unsqueeze(1)
            .expand(number_of_nodes, top_k)
            .reshape(-1)
        )
        destination = neighbor_index.reshape(-1).to(dtype=torch.long, device=edge_device)
        edge_index = torch.stack([source, destination], dim=0)

        if bool(symmetric):
            reverse = torch.stack([destination, source], dim=0)
            edge_index = torch.cat([edge_index, reverse], dim=1)

        edge_code = edge_index[0] * number_of_nodes + edge_index[1]
        unique_code = torch.unique(edge_code, sorted=True)
        unique_source = torch.div(unique_code, number_of_nodes, rounding_mode="floor")
        unique_destination = unique_code % number_of_nodes
        return torch.stack([unique_source, unique_destination], dim=0).long().contiguous()
