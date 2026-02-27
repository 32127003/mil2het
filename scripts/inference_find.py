"""Run inference for FIND checkpoints saved by scripts/train_find.py.

Example:
  python scripts/inference_find.py \
    --run-dir experiment_asthma_find/train_runs/<run_name> \
    --checkpoint-type best \
    --gpu 0
"""

import argparse
import inspect
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

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


def _load_config_from_run(run_dir: str) -> SimpleNamespace:
    run_config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(run_config_path):
        raise FileNotFoundError(f"run_config.json not found: {run_config_path}")
    with open(run_config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config_dict = payload.get("config", None)
    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid config payload in {run_config_path}")

    config = train_mod.dict2namespace(dict(config_dict))
    train_mod.apply_training_defaults(config)
    config.cell_encoder_name = train_mod.normalize_cell_encoder_name(str(config.cell_encoder_name))
    prior_absolute_enabled, prior_relational_enabled = _resolve_prior_apply_flags(config)
    config.prior_absolute_enabled = bool(prior_absolute_enabled)
    config.prior_relational_enabled = bool(prior_relational_enabled)

    module_variant = str(getattr(config, "module_variant", payload.get("module_variant", ""))).strip()
    if module_variant not in {"", "modules"}:
        raise ValueError(
            f"inference_find.py expects FIND runs (module_variant='modules'), got '{module_variant}'."
        )
    return config


def _build_data_and_mappings(
    config: SimpleNamespace,
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
    list[str],
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
    mapped_labels, _ = train_mod.apply_label_mapping(
        label_values=raw_labels,
        mode="binary",
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

    preselection_root = os.path.join(
        train_mod.PROJECT_ROOT,
        "data",
        str(config.dataset),
        str(config.split_preselection_subdir),
        f"split_idx_{int(config.split_number)}",
    )
    if not os.path.isdir(preselection_root):
        raise FileNotFoundError(
            "split-specific preselection directory not found: "
            f"{preselection_root}. Run modules/preselection.py first."
        )

    ranked_genes = train_mod.load_np_max_genes(
        str(config.dataset),
        0,
        preselection_root=preselection_root,
    )
    required_sources = [str(name) for name in list(config.prior_view_sources)]
    genes = train_mod.select_top_k_genes_with_embedding_coverage(
        ranked_genes=ranked_genes,
        target_k=int(config.k),
        required_sources=required_sources,
        configured_paths=train_mod._normalized_configured_embedding_paths(config),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference from saved FIND train run checkpoint")
    parser.add_argument("--run-dir", required=True, help="train run directory containing checkpoint files")
    parser.add_argument("--checkpoint-type", default="best", choices=["best", "last"])
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "both"],
        help="Deprecated: inference now always runs on test split only.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="cuda index; use -1 for CPU")
    parser.add_argument(
        "--output-tag",
        default="",
        help="Deprecated: inference no longer saves output files.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(str(args.run_dir))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    config = _load_config_from_run(run_dir)
    device = _resolve_device(int(args.gpu))

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
    print(f"[Inference] checkpoint_type={args.checkpoint_type}", flush=True)
    if str(args.split) != "test":
        print(
            f"[Inference] requested split={args.split} ignored; running test split only.",
            flush=True,
        )
    if str(args.output_tag).strip() != "":
        print(
            "[Inference] output-tag is ignored; inference does not save output files.",
            flush=True,
        )
    print("[Inference] split=test", flush=True)
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
    ) = _build_data_and_mappings(config)

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

    prior_view_dims = {
        str(view_name): int(view_tensor.shape[1])
        for view_name, view_tensor in prior_embeddings_by_view.items()
    }
    prior_absolute_enabled = bool(getattr(config, "prior_absolute_enabled", True))
    prior_relational_enabled = bool(getattr(config, "prior_relational_enabled", True))
    lambda_edge_bias_effective = (
        float(config.lambda_edge_bias) if bool(prior_relational_enabled) else 0.0
    )
    print(
        "[PriorInterface] "
        f"views={prior_view_sources} dims={prior_view_dims} "
        f"absolute={prior_absolute_enabled} relational={prior_relational_enabled} "
        f"prior_knn_k={int(config.prior_knn_k)} "
        f"lambda_edge_bias={float(lambda_edge_bias_effective):.6g}",
        flush=True,
    )

    graph_cache_path = os.devnull
    ordered_celltypes = [
        name
        for name, _index in sorted(celltype_mapping.items(), key=lambda item: int(item[1]))
    ]
    edge_index, regulation_onehot, global_zscore, _graph_cache_hit, _graph_cache_meta = train_mod.build_graph_cache(
        config=config,
        genes=genes,
        celltype_names=ordered_celltypes,
        cache_path=graph_cache_path,
        preselection_root=os.path.join(
            train_mod.PROJECT_ROOT,
            "data",
            str(config.dataset),
            str(config.split_preselection_subdir),
            f"split_idx_{int(config.split_number)}",
        ),
    )

    split_path = os.path.join(
        str(config.splits_directory),
        f"{config.dataset}_idx_{int(config.split_number)}.pkl",
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_indices = train_mod.load_pickle(split_path)

    _train_base_records, _val_base_records, test_base_records, _split_cache_hit = train_mod.build_split_records_cache(
        cache_path=os.devnull,
        split_indices=split_indices,
        label_indices=label_array,
        patient_indices=patient_array,
        split_source_path=split_path,
    )

    test_records = train_mod.expand_patient_bag_records(test_base_records, int(config.test_bags_per_patient))
    if len(test_records) == 0:
        raise ValueError("Test records cannot be empty.")

    test_dataset = train_mod.PatientBagDataset(
        expression_matrix=expression_matrix,
        bag_records=test_records,
        celltype_indices=celltype_array,
        treatment_indices=treatment_array,
    )

    dataloader_kwargs = train_mod.build_dataloader_kwargs(config, device)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    pooling = str(config.mil_pooling).lower()
    if pooling == "attention":
        pooling = "flat_attention"

    mil_kwargs: Dict[str, object] = {}
    mil_signature = inspect.signature(train_mod.PatientMILAggregator.__init__)
    if "recursive_steps" in mil_signature.parameters:
        mil_kwargs["recursive_steps"] = int(config.recursive_steps)
    if "connector_hidden_dimension" in mil_signature.parameters:
        mil_kwargs["connector_hidden_dimension"] = int(config.recursive_connector_hidden_dim)
    if "connector_dropout" in mil_signature.parameters:
        mil_kwargs["connector_dropout"] = float(config.recursive_connector_dropout)
    if "recursion_dropout" in mil_signature.parameters:
        mil_kwargs["recursion_dropout"] = float(config.recursive_dropout)
    if "recursion_celltype_aware" in mil_signature.parameters:
        mil_kwargs["recursion_celltype_aware"] = bool(config.recursive_celltype_aware)

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
        prior_injection_enabled=bool(prior_absolute_enabled or prior_relational_enabled),
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
    cell_encoder.load_state_dict(checkpoint_payload["cell_encoder"], strict=True)
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

    test_seed_anchor = int(config.seed) * 1000003 + 37

    node_feature_beta_value = float(config.node_feature_beta)
    cell_encoder.set_node_feature_beta(float(node_feature_beta_value))

    recursive_stepwise_loss_weight = float(config.recursive_stepwise_loss_weight)
    recursive_monotonic_loss_weight = float(config.recursive_monotonic_loss_weight)
    recursive_monotonic_beta = float(config.recursive_monotonic_beta)

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
    }

    with torch.inference_mode():
        test_metrics, _test_bag_predictions, _test_patient_predictions, _test_top_cells = train_mod.run_epoch(
            split_name="test",
            epoch=checkpoint_epoch,
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
            step_log_path=os.devnull,
            collect_top_cells=True,
            top_n=top_n,
            aggregate_patient_metrics=True,
            patient_probability_reduction=patient_probability_reduction,
            patient_embedding_ema=None,
            patient_embedding_ema_blend=0.0,
            epoch_dependent_sampling=False,
            recursive_stepwise_loss_weight=recursive_stepwise_loss_weight,
            recursive_monotonic_loss_weight=recursive_monotonic_loss_weight,
            recursive_monotonic_beta=recursive_monotonic_beta,
            **regularizer_epoch_kwargs,
        )

    print("[Inference] no output files are saved in this mode.", flush=True)
    print(
        "[TEST] "
        f"loss={test_metrics['loss']:.4f} "
        f"auprc={test_metrics['auprc']:.4f} "
        f"auroc={test_metrics['auroc']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"brier={test_metrics['brier']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
