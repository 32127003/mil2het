config = {

    # Paths (relative)
    "experiment_root": "../experiment_asthma",
    "adata_path": "../data/asthma/asthma_baseline_data.h5ad",
    "ppi_path": "../data/ppi_network.tsv",
    "splits_directory": "../data/splits/asthma/baseline/kfold/",

    # Dataset & prediction setup
    "dataset": "asthma",
    "prediction_unit": "patients",  # choices: ["patients", "cells"]

    # Metadata columns
    "label_column": "Response2",
    "patient_column": "PRISM_ID",
    "celltype_column": "CellType_minor",
    "treatment_column": "Treatment",

    # Binary mapping
    "binary_positive_labels": ["1"],
    "binary_negative_labels": ["0"],

    # Gene / protein setup
    "preselection": "NP",  # choices: ["NP", "DEG"]
    "k": 2000,
    "protein_embedding_choice": "concat",  # choices: ["concat", "mean", "none"]
    # "concat"/"mean" -> learnable QKV prior projection, "none" -> zero prior (QKV off).
    "protein_embedding_sources": ["GPT", "node2vec", "ESM3"],
    "split_preselection_subdir": "split_preselection",

    # Cell encoder selection
    "cell_encoder_name": "graph_gat",  # choices: ["graph_gcn","graph_gat","expression","expression_weighted_sum","deepsets","set_transformer"]
    "cell_embedding_dim": 128,
    "expression_projection_dim": 128,
    "expression_top_k_genes": 2000,
    "expression_normalize_weights": True,

    # GNN params
    "gnn_num_layers": 2,
    "gnn_hidden_dim": 128,
    "gnn_num_heads": 4,
    "gnn_dropout": 0.1,
    "gnn_attention_logit_clamp": 10.0,
    "gnn_expression_feature_scale": 2.0,
    "gnn_max_message_memory_mb": 256,
    "gnn_self_loop": True,
    "gnn_graph_readout": "mean",  # choices: ["mean", "attention"]
    "protein_prior_alpha": 1.0,
    "protein_prior_alpha_learnable": False,

    # Scale for proj(cell_features) in h0 = beta*proj(cell_features) + alpha*protein_prior.
    "node_feature_beta_start": 0.1,
    "node_feature_beta_end": 1.0,
    "node_feature_beta_steps": 10,

    # MIL / classifier (must match checkpoint)
    "cells_per_bag": 32,
    "bags_per_patient_per_epoch": 16,
    "mil_pooling": "attention",  # choices: ["mean", "attention", "flat_attention", "hierarchical_attention"]
    "classifier_name": "mlp",  # choices: ["mlp", "linear"]
    "classifier_hidden_dim": 256,
    "classifier_dropout": 0.1,

    # Recursive MIL (must match checkpoint when using modules_recursive)
    "recursive_steps": 2,
    "recursive_celltype_aware": True,
    "recursive_connector_hidden_dim": 256,
    "recursive_connector_dropout": 0.1,
    "recursive_dropout": 0.1,

    # ---------------------------------------------------------------------
    # Biomarker evaluation knobs (consumed by modules_recursive/biomarker.py)
    # ---------------------------------------------------------------------
    # Provide `biomarker_run_dir` here or pass `--run_dir` when running biomarker.py.
    "biomarker_run_dir": "",
    # Optional output directory. If empty, defaults to `<run_dir>/biomarker`.
    "biomarker_output_dir": "",
    # Evaluation split: {"train","val","test"}.
    "biomarker_split": "test",

    # Bag sampling for biomarker evaluation (can differ from training).
    "biomarker_cells_per_bag": 32,
    "biomarker_bags_per_patient": 32,
    "biomarker_sample_mode": "proportional",  # {"random","celltype","proportional"}
    "biomarker_patient_probability_reduction": "mean",  # {"mean","median"}

    # Metric used for delta computation.
    "biomarker_metric": "logloss",  # {"accuracy","precision","recall","f1","auroc","auprc","logloss"}

    # Stability diagnostics (no extra model inference; computed from permutation repeats).
    "biomarker_stability_enabled": True,
    "biomarker_stability_top_gene_ks": [20, 50, 100],
    "biomarker_stability_top_pathway_ks": [20, 50],
    "biomarker_stability_top_celltype_ks": [3, 5, 8],

    # Pathway gene sets (Step 1).
    "biomarker_pathway_gene_set_path": "../data/pathway/reactome_pathway.pkl",
    "biomarker_min_pathway_size": 10,
    "biomarker_max_pathway_size": 50,
    "biomarker_max_pathways": 0,  # 0 => use all after filtering

    # Number of random donor permutations to estimate deltas.
    "biomarker_pathway_permutations": 4,
    "biomarker_celltype_permutations": 8,

    # Step order / compute reduction:
    # - Step2 runs first (cell-type prioritization)
    # - Step1 (pathway/gene) can be restricted to top cell types for speed.
    "biomarker_step1_celltype_mode": "top_k",  # {"all","top_k","cum_weight","weight_threshold"}
    "biomarker_step1_top_k": 8,  # used when mode="top_k" (<=0 => all)
    "biomarker_step1_cum_weight": 0.9,  # used when mode="cum_weight"
    "biomarker_step1_weight_threshold": 0.05,  # used when mode="weight_threshold"

    # Seed for biomarker sampling/permutations.
    "biomarker_random_seed": 42,    
    
    # Misc / reproducibility
    "deterministic_training": False,  # used by biomarker.py seed setup
    "seed": 42,
    "split_number": 4,

    # Device
    "device": "cuda",  # choices: ["cuda", "cpu"]
    "cuda_device_index": 5,

}
