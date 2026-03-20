"""Precompute multimodal protein embeddings.

This script creates:
- data/gene_embedding/concat_embeddings.pkl
- data/gene_embedding/mean_embeddings.pkl
- data/gene_embedding/simclr_embeddings.pkl
- data/gene_embedding/concat_128_embeddings.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from utils import (
    DEFAULT_PREINTEGRATED_EMBEDDING_PATHS,
    DEFAULT_PROTEIN_EMBEDDING_PATHS,
    _save_embedding_dict_to_path,
    build_embedding_matrix,
    load_protein_embedding_dict,
    tqdm,
)


class QKV(nn.Module):
    """MLP projector used for offline embedding alignment/projection."""

    def __init__(self, config, input_dim: int):
        super().__init__()
        self.config = config
        assert (input_dim + config.n_embd) % 2 == 0, "(input_dim + config.n_embd) % 2 == 0"
            
        self.fc1 = nn.Linear(input_dim, (input_dim + config.n_embd) // 2)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear((input_dim + config.n_embd) // 2, int(config.n_embd))
        self.dropout = nn.Dropout(float(config.dropout_qkv))
        self.ln = nn.LayerNorm((input_dim + config.n_embd) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.activation(x)

        x = self.ln(x)

        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def _apply_group_norm_if_enabled(matrix: torch.Tensor, enabled: bool, num_groups: int) -> torch.Tensor:
    if not enabled:
        return matrix
    num_groups = int(num_groups)
    if num_groups <= 0:
        return matrix
    feature_dim = int(matrix.shape[1])
    if feature_dim % num_groups != 0:
        return matrix
    x = matrix.view(int(matrix.shape[0]), num_groups, feature_dim // num_groups)
    x = x - x.mean(dim=2, keepdim=True)
    x = x / x.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
    return x.view(int(matrix.shape[0]), feature_dim)


def _nt_xent_pair_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    z_i = torch_functional.normalize(z_i, dim=1)
    z_j = torch_functional.normalize(z_j, dim=1)
    logits = torch.matmul(z_i, z_j.t()) / float(max(temperature, 1e-8))
    labels = torch.arange(int(z_i.shape[0]), device=z_i.device)
    return torch_functional.cross_entropy(logits, labels)


def _run_simclr_alignment(
    embeddings: Dict[str, torch.Tensor],
    config: SimpleNamespace,
) -> torch.Tensor:
    if len(embeddings) == 0:
        raise ValueError("embeddings must be non-empty")
    first = next(iter(embeddings.values()))
    num_genes = int(first.shape[0])
    feature_dim = int(first.shape[1])
    for name, tensor in embeddings.items():
        if int(tensor.shape[0]) != num_genes:
            raise ValueError(f"Embedding rows mismatch for {name}")

    projection_dim = int(getattr(config, "protein_embedding_simclr_projection_dim", 256))
    temperature = float(getattr(config, "protein_embedding_simclr_temperature", 0.1))
    learning_rate = float(getattr(config, "protein_embedding_simclr_lr", 1e-3))
    epochs = int(getattr(config, "protein_embedding_simclr_epochs", 40))
    batch_size = int(getattr(config, "protein_embedding_simclr_batch_size", 256))
    show_progress = bool(getattr(config, "protein_embedding_progress_bar", False))
    progress_prefix = str(getattr(config, "protein_embedding_progress_prefix", "simclr"))

    requested_device = str(getattr(config, "protein_embedding_simclr_device", "cpu")).lower()
    runtime_device = _resolve_runtime_device(requested_device)
    device = torch.device(runtime_device)

    feature_bank = {name: tensor.to(device) for name, tensor in embeddings.items()}
    modality_names = list(feature_bank.keys())

    scale_params = nn.ParameterDict({name: nn.Parameter(torch.ones(feature_dim, device=device)) for name in modality_names})
    bias_params = nn.ParameterDict({name: nn.Parameter(torch.zeros(feature_dim, device=device)) for name in modality_names})
    projection_heads = nn.ModuleDict({name: nn.Linear(feature_dim, projection_dim, bias=False).to(device) for name in modality_names})

    parameter_list = list(scale_params.parameters()) + list(bias_params.parameters())
    for module in projection_heads.values():
        parameter_list.extend(module.parameters())
    optimizer = torch.optim.Adam(parameter_list, lr=learning_rate)

    epoch_iterator = range(max(epochs, 0))
    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc=f"{progress_prefix} epochs", leave=False)
    for _ in epoch_iterator:
        permutation = torch.randperm(num_genes, device=device)
        for start in range(0, num_genes, max(batch_size, 1)):
            batch_index = permutation[start : start + max(batch_size, 1)]
            if batch_index.numel() < 2:
                continue

            projected_views: Dict[str, torch.Tensor] = {}
            for name in modality_names:
                aligned = feature_bank[name][batch_index] * scale_params[name] + bias_params[name]
                projected = projection_heads[name](aligned)
                projected_views[name] = torch_functional.normalize(projected, dim=1)

            loss_terms = []
            for i in range(len(modality_names)):
                for j in range(i + 1, len(modality_names)):
                    name_i = modality_names[i]
                    name_j = modality_names[j]
                    loss_terms.append(_nt_xent_pair_loss(projected_views[name_i], projected_views[name_j], temperature))
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    aligned_views = []
    with torch.no_grad():
        for name in modality_names:
            aligned_views.append(feature_bank[name] * scale_params[name] + bias_params[name])
    fused = torch.stack(aligned_views, dim=0).mean(dim=0).cpu()
    if bool(getattr(config, "protein_embedding_simclr_l2_normalize", True)):
        fused = torch_functional.normalize(fused, p=2, dim=1)
    return fused


def _fit_qkv_projection(
    input_matrix: torch.Tensor,
    reference_matrix: torch.Tensor,
    config: SimpleNamespace,
) -> torch.Tensor:
    qkv_dropout = float(getattr(config, "protein_embedding_qkv_dropout", 0.1))
    qkv_learning_rate = float(getattr(config, "protein_embedding_qkv_lr", 1e-3))
    qkv_epochs = int(getattr(config, "protein_embedding_qkv_epochs", 80))
    qkv_batch_size = int(getattr(config, "protein_embedding_qkv_batch_size", 256))
    target_dim = int(getattr(config, "protein_embedding_target_dim", 3072))
    qkv_hidden_dim = int(getattr(config, "protein_embedding_qkv_hidden_dim", 0))
    qkv_device_text = str(getattr(config, "protein_embedding_qkv_device", "auto")).lower()
    show_progress = bool(getattr(config, "protein_embedding_progress_bar", False))
    progress_prefix = str(getattr(config, "protein_embedding_progress_prefix", "qkv"))

    runtime_device = _resolve_runtime_device(qkv_device_text)
    qkv_device = torch.device(runtime_device)

    qkv_config = SimpleNamespace(
        n_embd=target_dim,
        dropout_qkv=qkv_dropout,
        hidden_dim=qkv_hidden_dim if qkv_hidden_dim > 0 else None,
    )
    qkv_module = QKV(qkv_config, input_dim=int(input_matrix.shape[1])).to(qkv_device)
    optimizer = torch.optim.Adam(qkv_module.parameters(), lr=qkv_learning_rate)
    mse_loss = nn.MSELoss()

    input_matrix = input_matrix.to(qkv_device)
    reference_matrix = reference_matrix.to(qkv_device)
    num_genes = int(input_matrix.shape[0])
    epoch_iterator = range(max(qkv_epochs, 0))
    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc=f"{progress_prefix} epochs", leave=False)
    for _ in epoch_iterator:
        permutation = torch.randperm(num_genes, device=qkv_device)
        for start in range(0, num_genes, max(qkv_batch_size, 1)):
            batch_index = permutation[start : start + max(qkv_batch_size, 1)]
            if batch_index.numel() == 0:
                continue
            prediction = qkv_module(input_matrix[batch_index])
            target = reference_matrix[batch_index]
            loss = mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    qkv_module.eval()
    with torch.no_grad():
        projected = qkv_module(input_matrix)
    return projected.cpu()


def _project_esm_to_target_dim(
    esm_matrix: torch.Tensor,
    reference_matrix: torch.Tensor,
    config: SimpleNamespace,
) -> torch.Tensor:
    target_dim = int(getattr(config, "protein_embedding_target_dim", 3072))
    if int(esm_matrix.shape[1]) == target_dim:
        return esm_matrix
    return _fit_qkv_projection(
        input_matrix=esm_matrix,
        reference_matrix=reference_matrix,
        config=config,
    )


def _build_multimodal_embedding_matrix(
    genes: Sequence[str],
    sources: Sequence[str],
    config: SimpleNamespace,
) -> torch.Tensor:
    embedding_paths = getattr(config, "protein_embedding_paths", None)
    target_dim = int(getattr(config, "protein_embedding_target_dim", 3072))
    merge_mode = str(getattr(config, "protein_embedding_merge", "mean")).lower()
    use_group_norm = bool(getattr(config, "protein_embedding_groupnorm_enabled", True))
    group_norm_groups = int(getattr(config, "protein_embedding_groupnorm_groups", 32))

    source_matrices: Dict[str, torch.Tensor] = {}
    for source in sources:
        source_name = str(source)
        source_dict = load_protein_embedding_dict(source_name, embedding_paths=embedding_paths)
        source_matrix = build_embedding_matrix(genes, source_dict)
        source_matrices[source_name] = source_matrix

    if "GPT" not in source_matrices or "node2vec" not in source_matrices or "ESM3" not in source_matrices:
        raise ValueError("Multimodal protein embedding requires sources including GPT, node2vec, and ESM3.")

    gpt_matrix = source_matrices["GPT"]
    node2vec_matrix = source_matrices["node2vec"]
    esm_matrix = source_matrices["ESM3"]

    if int(gpt_matrix.shape[1]) != target_dim:
        raise ValueError(f"GPT embedding dim must be {target_dim}, got {int(gpt_matrix.shape[1])}.")
    if int(node2vec_matrix.shape[1]) != target_dim:
        raise ValueError(f"node2vec embedding dim must be {target_dim}, got {int(node2vec_matrix.shape[1])}.")

    gpt_matrix = _apply_group_norm_if_enabled(gpt_matrix, use_group_norm, group_norm_groups)
    node2vec_matrix = _apply_group_norm_if_enabled(node2vec_matrix, use_group_norm, group_norm_groups)

    reference_matrix = 0.5 * (gpt_matrix + node2vec_matrix)
    esm_projected = _project_esm_to_target_dim(esm_matrix, reference_matrix, config)
    esm_projected = torch_functional.layer_norm(esm_projected, normalized_shape=(int(esm_projected.shape[1]),))

    if merge_mode == "concat":
        concatenated_matrix = torch.cat([gpt_matrix, node2vec_matrix, esm_projected], dim=1)
        concat_reference = (gpt_matrix + node2vec_matrix + esm_projected) / 3.0
        return _fit_qkv_projection(
            input_matrix=concatenated_matrix,
            reference_matrix=concat_reference,
            config=config,
        )
    if merge_mode == "mean":
        stacked = torch.stack([gpt_matrix, node2vec_matrix, esm_projected], dim=0)
        return stacked.mean(dim=0)
    if merge_mode == "simclr":
        simclr_matrix = _run_simclr_alignment(
            {
                "GPT": gpt_matrix,
                "node2vec": node2vec_matrix,
                "ESM3": esm_projected,
            },
            config,
        )
        return simclr_matrix
    raise ValueError(f"Unknown protein_embedding_merge mode: {merge_mode}")


def build_concat_128_embedding(
    source_path: str,
    output_path: str,
    target_dim: int,
    device: str,
    l2_normalize: bool = True,
) -> Tuple[List[str], torch.Tensor]:
    """Build concat_128 embedding by PCA-projecting an existing integrated embedding file."""
    with open(source_path, "rb") as file_handle:
        embedding_object = pickle.load(file_handle)
    if not isinstance(embedding_object, dict):
        raise ValueError(f"Expected dict at {source_path}, got {type(embedding_object)}")

    genes = sorted({str(gene).upper() for gene in embedding_object.keys()})
    matrix = torch.stack(
        [torch.tensor(embedding_object[gene], dtype=torch.float32).view(-1) for gene in genes],
        dim=0,
    )
    if int(matrix.shape[1]) < int(target_dim):
        raise ValueError(
            f"target_dim={int(target_dim)} is larger than source dim={int(matrix.shape[1])}."
        )
    if int(matrix.shape[1]) == int(target_dim):
        projected = matrix
    else:
        runtime_device = _resolve_runtime_device(device)
        x = matrix.to(device=runtime_device, dtype=torch.float32)
        x = x - x.mean(dim=0, keepdim=True)
        # Uses randomized low-rank approximation.
        _u, _s, v = torch.pca_lowrank(x, q=int(target_dim), center=False)
        projected = (x @ v[:, : int(target_dim)]).detach().cpu()

    projected = projected.float()
    if l2_normalize:
        projected = torch_functional.normalize(projected, p=2, dim=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_embedding_dict_to_path(output_path, genes, projected)
    return genes, projected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrate protein embeddings offline.")
    parser.add_argument("--gpt_path", default=DEFAULT_PROTEIN_EMBEDDING_PATHS["GPT"])
    parser.add_argument("--node2vec_path", default=DEFAULT_PROTEIN_EMBEDDING_PATHS["node2vec"])
    parser.add_argument("--esm3_path", default=DEFAULT_PROTEIN_EMBEDDING_PATHS["ESM3"])
    parser.add_argument("--target_dim", type=int, default=3072)
    parser.add_argument("--disable_groupnorm", action="store_true")
    parser.add_argument("--groupnorm_groups", type=int, default=32)
    parser.add_argument("--qkv_dropout", type=float, default=0.1)
    parser.add_argument("--qkv_lr", type=float, default=1e-3)
    parser.add_argument("--qkv_epochs", type=int, default=20)
    parser.add_argument("--qkv_batch_size", type=int, default=2048)
    parser.add_argument("--qkv_hidden_dim", type=int, default=3072)
    parser.add_argument("--qkv_device", type=str, default="cuda")
    parser.add_argument("--simclr_projection_dim", type=int, default=256)
    parser.add_argument("--simclr_temperature", type=float, default=0.1)
    parser.add_argument("--simclr_lr", type=float, default=1e-3)
    parser.add_argument("--simclr_epochs", type=int, default=40)
    parser.add_argument("--simclr_batch_size", type=int, default=256)
    parser.add_argument("--simclr_device", type=str, default="cuda")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["concat", "mean", "simclr", "concat_128"],
        choices=["concat", "mean", "simclr", "concat_128"],
        help="Which integration modes to build.",
    )
    parser.add_argument(
        "--concat_128_source_path",
        type=str,
        default=str(DEFAULT_PREINTEGRATED_EMBEDDING_PATHS["concat"]),
        help="Source integrated embedding dict (3072-d) to PCA-project to 128 dims.",
    )
    parser.add_argument(
        "--concat_128_output_path",
        type=str,
        default=os.path.join(
            os.path.dirname(str(DEFAULT_PREINTEGRATED_EMBEDDING_PATHS["concat"])),
            "concat_128_embeddings.pkl",
        ),
        help="Output path for concat_128 embedding dict (128-d).",
    )
    parser.add_argument("--concat_128_target_dim", type=int, default=128)
    parser.add_argument(
        "--concat_128_device",
        type=str,
        default="auto",
        help="Device for PCA (auto/cpu/cuda[:idx]).",
    )
    return parser


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mode_output_path(mode: str) -> str:
    default_path = DEFAULT_PREINTEGRATED_EMBEDDING_PATHS[mode]
    return os.path.abspath(os.path.join(os.path.dirname(__file__), default_path))


def _load_common_genes(embedding_paths: Dict[str, str]) -> List[str]:
    gpt_dict = load_protein_embedding_dict("GPT", embedding_path=embedding_paths["GPT"])
    node2vec_dict = load_protein_embedding_dict("node2vec", embedding_path=embedding_paths["node2vec"])
    esm3_dict = load_protein_embedding_dict("ESM3", embedding_path=embedding_paths["ESM3"])
    common_genes = sorted(set(gpt_dict.keys()) & set(node2vec_dict.keys()) & set(esm3_dict.keys()))
    if not common_genes:
        raise ValueError("No common genes across GPT, node2vec, and ESM3 embeddings.")
    return common_genes


def _build_mode_config(args: argparse.Namespace, embedding_paths: Dict[str, str], mode: str) -> SimpleNamespace:
    output_path = _mode_output_path(mode)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return SimpleNamespace(
        protein_embedding_paths=embedding_paths,
        protein_embedding_merge=mode,
        protein_embedding_progress_bar=True,
        protein_embedding_progress_prefix=mode,
        protein_embedding_target_dim=int(args.target_dim),
        protein_embedding_groupnorm_enabled=not bool(args.disable_groupnorm),
        protein_embedding_groupnorm_groups=int(args.groupnorm_groups),
        protein_embedding_qkv_dropout=float(args.qkv_dropout),
        protein_embedding_qkv_lr=float(args.qkv_lr),
        protein_embedding_qkv_epochs=int(args.qkv_epochs),
        protein_embedding_qkv_batch_size=int(args.qkv_batch_size),
        protein_embedding_qkv_hidden_dim=int(args.qkv_hidden_dim),
        protein_embedding_qkv_device=str(args.qkv_device),
        protein_embedding_simclr_projection_dim=int(args.simclr_projection_dim),
        protein_embedding_simclr_temperature=float(args.simclr_temperature),
        protein_embedding_simclr_lr=float(args.simclr_lr),
        protein_embedding_simclr_epochs=int(args.simclr_epochs),
        protein_embedding_simclr_batch_size=int(args.simclr_batch_size),
        protein_embedding_simclr_device=str(args.simclr_device),
        protein_embedding_simclr_l2_normalize=True,
    )


def _resolve_runtime_device(requested_device: str) -> str:
    requested_text = str(requested_device).lower()
    if requested_text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_text.startswith("cuda"):
        return requested_text if torch.cuda.is_available() else "cpu"
    return "cpu"


def _compute_pca_ratio_summary(
    simclr_matrix: torch.Tensor,
    max_components: int = 128,
    max_rows: int = 8192,
) -> Dict[str, object]:
    row_count, embedding_dim = int(simclr_matrix.shape[0]), int(simclr_matrix.shape[1])
    if row_count < 3:
        return {
            "pca_rows_used": row_count,
            "pca_components_used": 0,
            "pca_explained_variance_top50": [],
            "pca_pc1_ratio": float("nan"),
            "pca_cumulative_top10": float("nan"),
            "pca_cumulative_top50": float("nan"),
            "effective_rank_approx_top128": float("nan"),
        }

    pca_rows = min(max_rows, row_count)
    generator = torch.Generator(device=simclr_matrix.device)
    generator.manual_seed(1)
    selected_indices = torch.randperm(row_count, generator=generator, device=simclr_matrix.device)[:pca_rows]
    pca_matrix = simclr_matrix[selected_indices].float()
    pca_matrix = pca_matrix - pca_matrix.mean(dim=0, keepdim=True)

    max_rank = min(pca_rows - 1, embedding_dim)
    component_count = min(max_components, max_rank)
    if component_count <= 0:
        return {
            "pca_rows_used": pca_rows,
            "pca_components_used": 0,
            "pca_explained_variance_top50": [],
            "pca_pc1_ratio": float("nan"),
            "pca_cumulative_top10": float("nan"),
            "pca_cumulative_top50": float("nan"),
            "effective_rank_approx_top128": float("nan"),
        }

    _, singular_values, _ = torch.pca_lowrank(pca_matrix, q=component_count, center=False)
    explained_variance = (singular_values ** 2) / float(max(pca_rows - 1, 1))
    total_variance = pca_matrix.var(dim=0, unbiased=True).sum().clamp_min(1e-12)
    explained_ratio = (explained_variance / total_variance).detach().cpu()

    top50 = explained_ratio[: min(50, explained_ratio.numel())]
    cumulative_top10 = float(explained_ratio[: min(10, explained_ratio.numel())].sum().item())
    cumulative_top50 = float(top50.sum().item())
    pc1_ratio = float(top50[0].item()) if top50.numel() > 0 else float("nan")

    explained_sum = float(explained_ratio.sum().item())
    residual_ratio = max(0.0, 1.0 - explained_sum)
    probability_bins = explained_ratio
    if residual_ratio > 0.0:
        probability_bins = torch.cat([probability_bins, torch.tensor([residual_ratio], dtype=probability_bins.dtype)])
    probability_bins = probability_bins / probability_bins.sum().clamp_min(1e-12)
    entropy = -(probability_bins * torch.log(probability_bins + 1e-12)).sum()
    effective_rank = float(torch.exp(entropy).item())

    return {
        "pca_rows_used": int(pca_rows),
        "pca_components_used": int(component_count),
        "pca_explained_variance_top50": [float(value) for value in top50.tolist()],
        "pca_pc1_ratio": float(pc1_ratio),
        "pca_cumulative_top10": float(cumulative_top10),
        "pca_cumulative_top50": float(cumulative_top50),
        "effective_rank_approx_top128": float(effective_rank),
    }


def run_simclr_sanity_check(
    simclr_matrix: torch.Tensor,
    gene_count: int,
    output_path: str,
) -> Dict[str, object]:
    if simclr_matrix.dim() != 2:
        raise ValueError(f"SIMCLR matrix must be 2D, got shape={tuple(simclr_matrix.shape)}")
    if int(simclr_matrix.shape[0]) != int(gene_count):
        raise ValueError(
            "SIMCLR matrix row count mismatch: "
            f"rows={int(simclr_matrix.shape[0])}, genes={int(gene_count)}"
        )
    if not torch.isfinite(simclr_matrix).all():
        raise ValueError("SIMCLR matrix contains NaN/Inf values.")

    row_norms = torch.linalg.norm(simclr_matrix, dim=1)
    zero_norm_count = int((row_norms <= 1e-12).sum().item())
    if zero_norm_count > 0:
        raise ValueError(f"SIMCLR matrix has {zero_norm_count} zero-norm rows.")
    row_norm_mean_value = float(row_norms.mean().item())
    row_norm_std_value = float(row_norms.std(unbiased=False).item())
    if abs(row_norm_mean_value - 1.0) > 0.05 or row_norm_std_value > 0.05:
        raise ValueError(
            "SIMCLR output is expected to be L2-normalized per row. "
            f"Observed row_norm_mean={row_norm_mean_value:.6f}, row_norm_std={row_norm_std_value:.6f}."
        )

    per_dimension_std = simclr_matrix.std(dim=0, unbiased=False)
    dead_dimension_count = int((per_dimension_std <= 1e-12).sum().item())
    if dead_dimension_count > max(4, int(0.05 * simclr_matrix.shape[1])):
        raise ValueError(
            "SIMCLR matrix appears collapsed across many dimensions: "
            f"dead_dims={dead_dimension_count}/{int(simclr_matrix.shape[1])}"
        )

    sample_size = min(2048, int(simclr_matrix.shape[0]))
    generator = torch.Generator(device=simclr_matrix.device)
    generator.manual_seed(0)
    sampled_indices = torch.randperm(
        int(simclr_matrix.shape[0]), generator=generator, device=simclr_matrix.device
    )[:sample_size]
    sampled_rows = torch_functional.normalize(simclr_matrix[sampled_indices], p=2, dim=1)
    cosine_matrix = sampled_rows @ sampled_rows.T
    upper_mask = torch.triu(torch.ones_like(cosine_matrix, dtype=torch.bool), diagonal=1)
    pairwise_cosine_values = cosine_matrix[upper_mask]
    mean_pairwise_cosine = float(pairwise_cosine_values.mean().item()) if pairwise_cosine_values.numel() > 0 else 0.0
    if mean_pairwise_cosine > 0.95:
        raise ValueError(
            "SIMCLR embedding collapse detected: "
            f"mean_pairwise_cosine={mean_pairwise_cosine:.4f} > 0.95"
        )

    quantiles = {}
    if pairwise_cosine_values.numel() > 0:
        quantile_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=pairwise_cosine_values.device)
        quantile_values = torch.quantile(pairwise_cosine_values, quantile_levels).detach().cpu().tolist()
        quantiles = {
            "q01": float(quantile_values[0]),
            "q05": float(quantile_values[1]),
            "q50": float(quantile_values[2]),
            "q95": float(quantile_values[3]),
            "q99": float(quantile_values[4]),
        }
    else:
        quantiles = {"q01": float("nan"), "q05": float("nan"), "q50": float("nan"), "q95": float("nan"), "q99": float("nan")}

    k_nearest = min(10, max(sample_size - 1, 1))
    cosine_without_self = cosine_matrix.clone()
    diagonal_index = torch.arange(sample_size, device=cosine_without_self.device)
    cosine_without_self[diagonal_index, diagonal_index] = -1.0
    nearest_values, _ = torch.topk(cosine_without_self, k=k_nearest, dim=1, largest=True, sorted=False)
    nearest_row_mean = nearest_values.mean(dim=1)
    nearest_summary = {
        "k": int(k_nearest),
        "mean": float(nearest_values.mean().item()),
        "std": float(nearest_values.std(unbiased=False).item()),
        "row_mean_q05": float(torch.quantile(nearest_row_mean, 0.05).item()),
        "row_mean_q50": float(torch.quantile(nearest_row_mean, 0.50).item()),
        "row_mean_q95": float(torch.quantile(nearest_row_mean, 0.95).item()),
    }

    pca_summary = _compute_pca_ratio_summary(simclr_matrix=simclr_matrix, max_components=128, max_rows=8192)

    sanity_report = {
        "num_genes": int(simclr_matrix.shape[0]),
        "embedding_dim": int(simclr_matrix.shape[1]),
        "row_norm_mean": float(row_norm_mean_value),
        "row_norm_std": float(row_norm_std_value),
        "row_norm_min": float(row_norms.min().item()),
        "row_norm_max": float(row_norms.max().item()),
        "dead_dimension_count": int(dead_dimension_count),
        "mean_pairwise_cosine_sampled": float(mean_pairwise_cosine),
        "pairwise_cosine_quantiles_sampled": quantiles,
        "topk_nearest_neighbor_cosine": nearest_summary,
        **pca_summary,
    }

    sanity_path = output_path.replace(".pkl", "_sanity.json")
    with open(sanity_path, "w", encoding="utf-8") as file_handle:
        json.dump(sanity_report, file_handle, indent=2)
    print(f"[integrate] simclr sanity saved: {sanity_path}")
    return sanity_report


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = _project_root()

    selected_modes = list(dict.fromkeys(args.modes))

    embedding_paths = {
        "GPT": os.path.abspath(os.path.join(os.path.dirname(__file__), args.gpt_path)),
        "node2vec": os.path.abspath(os.path.join(os.path.dirname(__file__), args.node2vec_path)),
        "ESM3": os.path.abspath(os.path.join(os.path.dirname(__file__), args.esm3_path)),
    }

    print(f"[integrate] project_root={project_root}")
    print(f"[integrate] selected modes: {selected_modes}")
    common_genes: List[str] = []
    if any(mode != "concat_128" for mode in selected_modes):
        print(f"[integrate] loading source embeddings from: {embedding_paths}")
        print(
            "[integrate] qkv settings: "
            f"epochs={int(args.qkv_epochs)} batch_size={int(args.qkv_batch_size)} "
            f"hidden_dim={int(args.qkv_hidden_dim)} device={str(args.qkv_device)}"
        )
        resolved_qkv_device = _resolve_runtime_device(str(args.qkv_device))
        resolved_simclr_device = _resolve_runtime_device(str(args.simclr_device))
        print(
            "[integrate] runtime devices: "
            f"qkv={resolved_qkv_device} (requested={str(args.qkv_device)}), "
            f"simclr={resolved_simclr_device} (requested={str(args.simclr_device)})"
        )
        common_genes = _load_common_genes(embedding_paths)
        print(f"[integrate] common genes: {len(common_genes)}")

    metadata = {
        "num_genes": int(len(common_genes)) if common_genes else None,
        "target_dim": int(args.target_dim),
        "embedding_paths": embedding_paths,
        "modes": {},
    }

    for mode in tqdm(selected_modes, desc="integration modes"):
        print(f"[integrate] building mode={mode}")
        if mode == "concat_128":
            source_path = os.path.abspath(str(args.concat_128_source_path))
            output_path = os.path.abspath(str(args.concat_128_output_path))
            target_dim = int(args.concat_128_target_dim)
            device = str(args.concat_128_device)

            genes_128, matrix_128 = build_concat_128_embedding(
                source_path=source_path,
                output_path=output_path,
                target_dim=target_dim,
                device=device,
                l2_normalize=True,
            )
            if metadata.get("num_genes") is None:
                metadata["num_genes"] = int(len(genes_128))
            metadata["modes"][mode] = {
                "output_path": output_path,
                "shape": [int(matrix_128.shape[0]), int(matrix_128.shape[1])],
                "source_path": source_path,
                "device": str(device),
            }
            print(f"[integrate] saved mode={mode} -> {output_path} shape={tuple(matrix_128.shape)}")
            continue

        mode_config = _build_mode_config(args, embedding_paths, mode)
        matrix = _build_multimodal_embedding_matrix(common_genes, ["GPT", "node2vec", "ESM3"], mode_config)

        output_path = _mode_output_path(mode)
        _save_embedding_dict_to_path(output_path, common_genes, matrix)

        metadata["modes"][mode] = {
            "output_path": output_path,
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        }
        if mode == "simclr":
            metadata["modes"][mode]["sanity"] = run_simclr_sanity_check(
                simclr_matrix=matrix,
                gene_count=len(common_genes),
                output_path=output_path,
            )
        print(f"[integrate] saved mode={mode} -> {output_path} shape={tuple(matrix.shape)}")

    metadata_path = os.path.join(project_root, "data", "gene_embedding", "integrated_embeddings_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)
    print(f"[integrate] metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
