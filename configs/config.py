asthma_split_configuration = {

    # Directories & Paths
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/asthma/",

    # Task
    "label_column": "Response2",
    "patient_column": "PRISM_ID",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "CellType_minor",

    # DEG parameters
    "deg_max_p_value": 0.05,
    "deg_min_abs_logfc": 1.0,

    # Network Propagation parameters
    "restart_prob": 0.1,
    "convergence_threshold_l1": 1e-6,
    "max_iterations": 1000,
    "directed": False,



    "seed": 42,
    "num_folds": 5,
    "num_valid_patients": 0,
}

asthma_ext_split_configuration = {
    
    # Directories & Paths
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/asthma_ext/",

    # Task
    "label_column": "label",
    "patient_column": "id",
    "sample_column": "Channel",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "cluster",

    # DEG parameters
    "deg_max_p_value": 0.05,
    "deg_min_abs_logfc": 1.0,

    # Network Propagation parameters
    "restart_prob": 0.1,
    "convergence_threshold_l1": 1e-6,
    "max_iterations": 1000,
    "directed": False,


    "seed": 42,
    "num_folds": 4,
    "num_valid_patients": 0,
}

vitiligo_split_configuration = {
    
    # Directories & Paths
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/vitiligo/",

    # Task
    "label_column": "label",
    "patient_column": "Patient",
    "sample_column": "ID",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "new_anno",

    # DEG parameters
    "deg_max_p_value": 0.05,
    "deg_min_abs_logfc": 1.0,

    # Network Propagation parameters
    "restart_prob": 0.1,
    "convergence_threshold_l1": 1e-6,
    "max_iterations": 1000,
    "directed": False,


    "seed": 42,
    "num_folds": 4,
    "num_valid_patients": 2,
}

covid_split_configuration = {
    
    # Directories & Paths
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/covid/",

    # Task
    "label_column": "label",
    "patient_column": "donor_id",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "Annotation",

    # DEG parameters
    "deg_max_p_value": 0.05,
    "deg_min_abs_logfc": 1.0,

    # Network Propagation parameters
    "restart_prob": 0.1,
    "convergence_threshold_l1": 1e-6,
    "max_iterations": 1000,
    "directed": False,


    "seed": 42,
    "num_folds": 5,

}


asthma_train_configuration = {
    # Directories & Paths
    "experiment_root": "./experiment_asthma_find/",
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/asthma/",
    
    
    # Binary mapping (plural forms used by train_new/biomarker)
    "binary_positive_labels": ["1"],
    "binary_negative_labels": ["0"],
    

    # Task
    "label_column": "Response2",
    "patient_column": "PRISM_ID",
    "celltype_column": "CellType_minor",
    "treatment_column": "Treatment",
    "binary_positive_label": "1",
    "binary_negative_label": "0",

    # Gene/protein setup
    "k": 2000,
    "protein_embedding_choice": "concat",
    "protein_embedding_sources": ["GPT", "node2vec", "ESM3"],
    "protein_embedding_paths": {},
    "protein_embedding_merged_cache_path": "",
    "prior_view_sources": ["GPT", "node2vec", "ESM3"],

    # FIND prior interface / injection
    # True/True => fully use absolute(FiLM) + relational(edge bias) priors.
    "prior_absolute_enabled": True,
    "prior_relational_enabled": True,
    "prior_interface_num_heads": 4,
    "prior_interface_dropout": 0.1,
    "prior_interface_ffn_hidden_dim": 64,
    "prior_knn_k": 0,  # 16   
    "prior_knn_symmetric": True,
    "lambda_edge_bias": 0.1,

    # Cell encoder selector
    "cell_encoder_name": "transformer_conv",

    # Cell encoder (common)
    "gnn_attention_logit_clamp": 10.0,
    "gnn_expression_feature_scale": 2.0,
    "gnn_max_message_memory_mb": 256,
    "gnn_self_loop": True,
    "node_feature_beta": 1.0,

    # Cell encoder (graph_gat only)
    "gnn_num_layers": 1,
    "gnn_hidden_dim": 128,
    "gnn_num_heads": 2,
    "gnn_dropout": 0.2,
    "gnn_graph_readout": "attention",

    # Cell encoder (transformer_conv only)
    "transformer_conv_feat_dim": 32,
    "transformer_conv_num_layers": 1,
    "transformer_conv_heads": 2,
    "transformer_conv_dropout": 0.2,
    "transformer_conv_graph_readout": "mean",

    "zscore_standardize": True,
    "zscore_clip_min": -2.0,
    "zscore_clip_max": 2.0,

    # Node-score sparse regularizer + mixup
    "mixup_alpha": 0.2,
    "lambda_sparse": 1e-4,

    # Patient bag-resampling consistency regularizer
    "use_consistency_regularizer": False,
    "consistency_weight_lambda": 0.01,
    "consistency_num_bags_per_patient_per_step": 2,
    "consistency_loss_type": "mse_prob",

    "consistency_detach_second_branch": False,

    # MIL/classifier
    "cells_per_bag": 32,
    "bags_per_patient_per_epoch": 16,
    "val_bags_per_patient": 16,
    "test_bags_per_patient": 64,
    "mil_pooling": "attention",
    "mil_attention_hidden_dim": 128,
    "mil_attention_top_n": 20,
    "sample_bags": "proportional",
    "patient_probability_reduction": "mean",
    "classifier_name": "mlp",
    "classifier_hidden_dim": 128,
    "classifier_dropout": 0.1,


    # Biomarker extraction (post-hoc)
    "biomarker_run_dir": "",
    "biomarker_output_dir": "",
    "biomarker_split": "test",
    "biomarker_cells_per_bag": 32,
    "biomarker_bags_per_patient": 32,
    "biomarker_sample_mode": "proportional",
    "biomarker_patient_probability_reduction": "mean",
    "biomarker_metric": "logloss",
    "biomarker_pathway_gene_set_path": "../data/pathway/reactome_pathway.pkl",
    "biomarker_min_pathway_size": 10,
    "biomarker_max_pathway_size": 50,
    "biomarker_max_pathways": 0,
    "biomarker_pathway_permutations": 4,
    "biomarker_celltype_permutations": 8,
    "biomarker_step1_celltype_mode": "top_k",
    "biomarker_step1_top_k": 8,
    "biomarker_step1_cum_weight": 0.9,
    "biomarker_step1_weight_threshold": 0.05,
    "biomarker_random_seed": 42,
    "biomarker_stability_enabled": True,
    "biomarker_stability_top_gene_ks": [20, 50, 100],
    "biomarker_stability_top_pathway_ks": [20, 50],
    "biomarker_stability_top_celltype_ks": [3, 5, 8],
    


    # Training
    "optimizer": "adamw",
    "pos_weight": None, 
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "gradient_clip_max_norm": 1.0,
    "use_patient_ema_pooling": False,
    "patient_ema_decay": 0.95,
    "patient_ema_blend": 0.2,
    "epochs": 200,
    "model_selection_primary_metric": "loss",
    "model_selection_secondary_metric": "AUROC",
    "model_selection_primary_epsilon": 0.05,
    "early_stopping_metric": "val_loss",
    "early_stopping_patience": 20,
    "batch_size": 16,
    "num_workers": 0,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": False,
    "dataloader_prefetch_factor": 2,
    "deterministic_training": False,
    "deterministic_algorithms": False,
    "deterministic_warn_only": True,
}


asthma_ext_train_configuration = {

    # Directories & Paths    
    "experiment_root": "./experiment_asthma_ext_/",
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/asthma_ext/",

    # Binary mapping (plural forms used by train_new/biomarker)
    "binary_positive_labels": ["1"],
    "binary_negative_labels": ["0"],
    

    # Task
    "label_column": "label",
    "patient_column": "Channel",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "cluster",
    "treatment_column": None,


    # Gene/protein setup
    "k": 2000,
    "protein_embedding_choice": "concat",
    "protein_embedding_paths": {},
    "protein_embedding_merged_cache_path": "",
    "prior_view_sources": ["GPT", "node2vec", "ESM3"],

    # FIND prior interface / injection
    # True/True => fully use absolute(FiLM) + relational(edge bias) priors.
    "prior_absolute_enabled": True,
    "prior_relational_enabled": False,
    "prior_interface_num_heads": 4,
    "prior_interface_dropout": 0.1,
    "prior_interface_ffn_hidden_dim": 64,
    "prior_knn_k": 16,
    "prior_knn_symmetric": True,
    "lambda_edge_bias": 0.1,

    # Cell encoder selector
    "cell_encoder_name": "transformer_conv",

    # Cell encoder (common)
    "gnn_attention_logit_clamp": 10.0,
    "gnn_expression_feature_scale": 2.0,
    "gnn_max_message_memory_mb": 256,
    "gnn_self_loop": True,
    "node_feature_beta": 1.0,

    # Cell encoder (graph_gat only)
    "gnn_num_layers": 1,
    "gnn_hidden_dim": 128,
    "gnn_num_heads": 2,
    "gnn_dropout": 0.2,
    "gnn_graph_readout": "attention",

    # Cell encoder (transformer_conv only)
    "transformer_conv_feat_dim": 32,
    "transformer_conv_num_layers": 1,
    "transformer_conv_heads": 2,
    "transformer_conv_dropout": 0.2,
    "transformer_conv_graph_readout": "mean",

    "zscore_standardize": True,
    "zscore_clip_min": -2.0,
    "zscore_clip_max": 2.0,

    # Node-score sparse regularizer + mixup
    "mixup_alpha": 0.1,
    "lambda_sparse": 1e-4,

    # Patient bag-resampling consistency regularizer
    "use_consistency_regularizer": False,
    "consistency_weight_lambda": 0.01,
    "consistency_num_bags_per_patient_per_step": 2,
    "consistency_loss_type": "mse_prob",
    "consistency_detach_second_branch": False,

    # MIL/classifier
    "cells_per_bag": 32,
    "bags_per_patient_per_epoch": 16,
    "val_bags_per_patient": 16,
    "test_bags_per_patient": 64,
    "mil_pooling": "attention",
    "mil_attention_hidden_dim": 256,
    "mil_attention_top_n": 20,
    "sample_bags": "proportional",
    "patient_probability_reduction": "mean",
    "classifier_name": "mlp",
    "classifier_hidden_dim": 256,
    "classifier_dropout": 0.1,

    # Biomarker extraction 
    "biomarker_run_dir": "",
    "biomarker_output_dir": "",
    "biomarker_split": "test",
    "biomarker_cells_per_bag": 32,
    "biomarker_bags_per_patient": 32,
    "biomarker_sample_mode": "proportional",
    "biomarker_patient_probability_reduction": "mean",
    "biomarker_metric": "logloss",
    "biomarker_pathway_gene_set_path": "../data/pathway/reactome_pathway.pkl",
    "biomarker_min_pathway_size": 10,
    "biomarker_max_pathway_size": 50,
    "biomarker_max_pathways": 0,
    "biomarker_pathway_permutations": 4,
    "biomarker_celltype_permutations": 8,
    "biomarker_step1_celltype_mode": "top_k",
    "biomarker_step1_top_k": 8,
    "biomarker_step1_cum_weight": 0.9,
    "biomarker_step1_weight_threshold": 0.05,
    "biomarker_random_seed": 42,
    "biomarker_stability_enabled": True,
    "biomarker_stability_top_gene_ks": [20, 50, 100],
    "biomarker_stability_top_pathway_ks": [20, 50],
    "biomarker_stability_top_celltype_ks": [3, 5, 8],
    


    # Training
    "optimizer": "adamw",
    "pos_weight": None,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "gradient_clip_max_norm": 1.0,
    "use_patient_ema_pooling": False,
    "patient_ema_decay": 0.95,
    "patient_ema_blend": 0.2,
    "epochs": 200,
    "model_selection_primary_metric": "loss",
    "model_selection_secondary_metric": "AUROC",
    "model_selection_primary_epsilon": 0.05,
    "early_stopping_metric": "val_loss",
    "early_stopping_patience": 20,
    "batch_size": 16,
    "num_workers": 0,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": False,
    "dataloader_prefetch_factor": 2,
    "deterministic_training": False,
    "deterministic_algorithms": False,
    "deterministic_warn_only": True,
}

vitiligo_train_configuration = {

    # Directories & Paths    
    "experiment_root": "./experiment_vitiligo/",
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/vitiligo/",

    # Binary mapping (plural forms used by train/biomarker)
    "binary_positive_labels": ["1"],
    "binary_negative_labels": ["0"],

    # Task
    "label_column": "label",
    "patient_column": "ID",
    "binary_positive_label": "1",
    "binary_negative_label": "0",
    "celltype_column": "new_anno",
    "treatment_column": None,


    # Gene/protein setup
    "k": 2000,
    "protein_embedding_choice": "concat",
    "protein_embedding_paths": {},
    "protein_embedding_merged_cache_path": "",
    "prior_view_sources": ["GPT", "node2vec", "ESM3"],

    # FIND prior interface / injection
    # True/True => fully use absolute(FiLM) + relational(edge bias) priors.
    "prior_absolute_enabled": True,
    "prior_relational_enabled": True,
    "prior_interface_num_heads": 4,
    "prior_interface_dropout": 0.1,
    "prior_interface_ffn_hidden_dim": 64,
    "prior_knn_k": 16,
    "prior_knn_symmetric": True,
    "lambda_edge_bias": 0.1,

    # Cell encoder selector
    "cell_encoder_name": "transformer_conv",

    # Cell encoder (common)
    "gnn_attention_logit_clamp": 10.0,
    "gnn_expression_feature_scale": 2.0,
    "gnn_max_message_memory_mb": 256,
    "gnn_self_loop": True,
    "node_feature_beta": 1.0,

    # Cell encoder (graph_gat only)
    "gnn_num_layers": 1,
    "gnn_hidden_dim": 128,
    "gnn_num_heads": 2,
    "gnn_dropout": 0.2,
    "gnn_graph_readout": "attention",

    # Cell encoder (transformer_conv only)
    "transformer_conv_feat_dim": 16,
    "transformer_conv_num_layers": 1,
    "transformer_conv_heads": 2,
    "transformer_conv_dropout": 0.2,
    "transformer_conv_graph_readout": "mean",

    "zscore_standardize": True,
    "zscore_clip_min": -2.0,
    "zscore_clip_max": 2.0,

    # Node-score sparse regularizer + mixup
    "mixup_alpha": 0.2,
    "lambda_sparse": 1e-4,

    # Patient bag-resampling consistency regularizer
    "use_consistency_regularizer": False,
    "consistency_weight_lambda": 0.01,
    "consistency_num_bags_per_patient_per_step": 2,
    "consistency_loss_type": "mse_prob",
    "consistency_detach_second_branch": False,

    # MIL/classifier
    "cells_per_bag": 32,
    "bags_per_patient_per_epoch": 16,
    "val_bags_per_patient": 16,
    "test_bags_per_patient": 64,
    "mil_pooling": "attention",
    "mil_attention_hidden_dim": 128,
    "mil_attention_top_n": 20,
    "sample_bags": "proportional",
    "patient_probability_reduction": "mean",
    "classifier_name": "mlp",
    "classifier_hidden_dim": 128,
    "classifier_dropout": 0.1,

    # Biomarker extraction 
    "biomarker_run_dir": "",
    "biomarker_output_dir": "",
    "biomarker_split": "test",
    "biomarker_cells_per_bag": 32,
    "biomarker_bags_per_patient": 32,
    "biomarker_sample_mode": "proportional",
    "biomarker_patient_probability_reduction": "mean",
    "biomarker_metric": "logloss",
    "biomarker_pathway_gene_set_path": "../data/pathway/reactome_pathway.pkl",
    "biomarker_min_pathway_size": 10,
    "biomarker_max_pathway_size": 50,
    "biomarker_max_pathways": 0,
    "biomarker_pathway_permutations": 4,
    "biomarker_celltype_permutations": 8,
    "biomarker_step1_celltype_mode": "top_k",
    "biomarker_step1_top_k": 8,
    "biomarker_step1_cum_weight": 0.9,
    "biomarker_step1_weight_threshold": 0.05,
    "biomarker_random_seed": 42,
    "biomarker_stability_enabled": True,
    "biomarker_stability_top_gene_ks": [20, 50, 100],
    "biomarker_stability_top_pathway_ks": [20, 50],
    "biomarker_stability_top_celltype_ks": [3, 5, 8],
    


    # Training
    "optimizer": "adamw",
    "pos_weight": None,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "gradient_clip_max_norm": 1.0,
    "use_patient_ema_pooling": False,
    "patient_ema_decay": 0.95,
    "patient_ema_blend": 0.2,
    "epochs": 200,
    "model_selection_primary_metric": "loss",
    "model_selection_secondary_metric": "AUROC",
    "model_selection_primary_epsilon": 0.05,
    "early_stopping_metric": "val_loss",
    "early_stopping_patience": 20,
    "batch_size": 16,
    "num_workers": 0,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": False,
    "dataloader_prefetch_factor": 2,
    "deterministic_training": False,
    "deterministic_algorithms": False,
    "deterministic_warn_only": True,
}


covid_train_configuration = {
    # Directories & Paths
    "experiment_root": "./experiment_covid/",
    "adata_directory": "./data",
    "ppi_path": "./data/ppi_network.tsv",
    "splits_directory": "./data/splits/covid/",
    
    
    # Binary mapping (plural forms used by train_new/biomarker)
    "binary_positive_labels": ["1"],
    "binary_negative_labels": ["0"],
    

    # Task
    "label_column": "label",
    "patient_column": "donor_id",
    "celltype_column": "Annotation",
    "treatment_column": None,
    "binary_positive_label": "1",
    "binary_negative_label": "0",

    # Gene/protein setup
    "k": 2000,
    "protein_embedding_choice": "concat",
    "protein_embedding_sources": ["GPT", "node2vec", "ESM3"],
    "protein_embedding_paths": {},
    "protein_embedding_merged_cache_path": "",
    "prior_view_sources": ["GPT", "node2vec", "ESM3"],

    # FIND prior interface / injection
    # True/True => fully use absolute(FiLM) + relational(edge bias) priors.
    "prior_absolute_enabled": True,
    "prior_relational_enabled": True,
    "prior_interface_num_heads": 4,
    "prior_interface_dropout": 0.1,
    "prior_interface_ffn_hidden_dim": 64,
    "prior_knn_k": 0,  # 16   
    "prior_knn_symmetric": True,
    "lambda_edge_bias": 0.1,

    # Cell encoder selector
    "cell_encoder_name": "transformer_conv",

    # Cell encoder (common)
    "gnn_attention_logit_clamp": 10.0,
    "gnn_expression_feature_scale": 2.0,
    "gnn_max_message_memory_mb": 256,
    "gnn_self_loop": True,
    "node_feature_beta": 1.0,

    # Cell encoder (graph_gat only)
    "gnn_num_layers": 1,
    "gnn_hidden_dim": 128,
    "gnn_num_heads": 2,
    "gnn_dropout": 0.2,
    "gnn_graph_readout": "attention",

    # Cell encoder (transformer_conv only)
    "transformer_conv_feat_dim": 16,
    "transformer_conv_num_layers": 1,
    "transformer_conv_heads": 2,
    "transformer_conv_dropout": 0.2,
    "transformer_conv_graph_readout": "mean",

    "zscore_standardize": True,
    "zscore_clip_min": -2.0,
    "zscore_clip_max": 2.0,

    # Node-score sparse regularizer + mixup
    "mixup_alpha": 0.2,
    "lambda_sparse": 1e-4,

    # Patient bag-resampling consistency regularizer
    "use_consistency_regularizer": False,
    "consistency_weight_lambda": 0.01,
    "consistency_num_bags_per_patient_per_step": 2,
    "consistency_loss_type": "mse_prob",
    
    "consistency_detach_second_branch": False,

    # MIL/classifier
    "cells_per_bag": 32,
    "bags_per_patient_per_epoch": 16,
    "val_bags_per_patient": 16,
    "test_bags_per_patient": 64,
    "mil_pooling": "attention",
    "mil_attention_hidden_dim": 128,
    "mil_attention_top_n": 20,
    "sample_bags": "proportional",
    "patient_probability_reduction": "mean",
    "classifier_name": "mlp",
    "classifier_hidden_dim": 128,
    "classifier_dropout": 0.1,


    # Biomarker extraction (post-hoc)
    "biomarker_run_dir": "",
    "biomarker_output_dir": "",
    "biomarker_split": "test",
    "biomarker_cells_per_bag": 32,
    "biomarker_bags_per_patient": 32,
    "biomarker_sample_mode": "proportional",
    "biomarker_patient_probability_reduction": "mean",
    "biomarker_metric": "logloss",
    "biomarker_pathway_gene_set_path": "../data/pathway/reactome_pathway.pkl",
    "biomarker_min_pathway_size": 10,
    "biomarker_max_pathway_size": 50,
    "biomarker_max_pathways": 0,
    "biomarker_pathway_permutations": 4,
    "biomarker_celltype_permutations": 8,
    "biomarker_step1_celltype_mode": "top_k",
    "biomarker_step1_top_k": 8,
    "biomarker_step1_cum_weight": 0.9,
    "biomarker_step1_weight_threshold": 0.05,
    "biomarker_random_seed": 42,
    "biomarker_stability_enabled": True,
    "biomarker_stability_top_gene_ks": [20, 50, 100],
    "biomarker_stability_top_pathway_ks": [20, 50],
    "biomarker_stability_top_celltype_ks": [3, 5, 8],
    


    # Training
    "optimizer": "adamw",
    "pos_weight": None, 
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "gradient_clip_max_norm": 1.0,
    "use_patient_ema_pooling": False,
    "patient_ema_decay": 0.95,
    "patient_ema_blend": 0.2,
    "epochs": 200,
    "model_selection_primary_metric": "loss",
    "model_selection_secondary_metric": "AUROC",
    "model_selection_primary_epsilon": 0.05,
    "early_stopping_metric": "val_loss",
    "early_stopping_patience": 20,
    "batch_size": 16,
    "num_workers": 0,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": False,
    "dataloader_prefetch_factor": 2,
    "deterministic_training": False,
    "deterministic_algorithms": False,
    "deterministic_warn_only": True,
}