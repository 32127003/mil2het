from __future__ import annotations

import tempfile
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from mil2het import CellEncoder, MultipleInstanceLearning, train

from smoke_test_helpers import assert_module_endpoints, print_success

TRAIN_ENDPOINTS = [
    "autocast_cuda",
    "ensure_cublas_workspace_config",
    "configure_runtime_backends",
    "PatientEmbeddingEMAMemory",
    "PatientBagDataset",
    "PatientClassifier",
    "collate_patient_bags",
    "compute_binary_metrics",
    "aggregate_patient_probabilities",
    "resolve_cell_encoder_spec",
    "build_protein_embedding_matrix",
    "load_prior_embeddings_by_view",
    "build_celltype_regulation_onehot",
    "build_global_deg_zscore_vector",
    "build_graph_cache",
    "build_split_records_cache",
    "build_experiment_directory",
    "build_optimizer",
    "run_epoch",
    "run_training_phase",
    "build_train_arg_parser",
    "build_train_config_from_cli_args",
    "load_train_config_from_cli",
]


def ring_edge_index(num_nodes: int) -> torch.Tensor:
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = (src + 1) % num_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def build_regulation_onehot(num_celltypes: int, num_nodes: int) -> torch.Tensor:
    regulation = torch.zeros(num_celltypes, num_nodes, 3)
    labels = torch.randint(0, 3, (num_celltypes, num_nodes))
    regulation.scatter_(2, labels.unsqueeze(-1), 1.0)
    return regulation


def test_train_endpoints_exist() -> None:
    assert_module_endpoints(
        train,
        TRAIN_ENDPOINTS,
        module_label="mil2het.train",
    )


def test_train_runtime_and_dataset_helpers() -> None:
    with train.autocast_cuda(enabled=False):
        value = torch.tensor([1.0]) + torch.tensor([2.0])
    assert value.item() == 3.0

    train.ensure_cublas_workspace_config()
    train.configure_runtime_backends(device=torch.device("cpu"), deterministic_training=False)

    memory = train.PatientEmbeddingEMAMemory(embedding_dimension=4, decay=0.9)
    patient_indices = np.array([0, 1, 0], dtype=np.int64)
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    memory.update(patient_indices=patient_indices, patient_embeddings=embeddings)
    looked_up, mask = memory.lookup(patient_indices=np.array([0, 1, 2], dtype=np.int64), device=torch.device("cpu"), dtype=torch.float32)
    assert looked_up.shape == (3, 4)
    assert mask.tolist() == [True, True, False]

    bag_records = [
        {"patient_index": 0, "patient_label": 1, "cell_indices": np.array([0, 1], dtype=np.int64), "bag_repeat_index": 0},
        {"patient_index": 1, "patient_label": 0, "cell_indices": np.array([2, 3], dtype=np.int64), "bag_repeat_index": 0},
    ]
    expression = np.random.randn(4, 5).astype(np.float32)
    celltypes = np.array([0, 1, 0, 1], dtype=np.int64)
    treatments = np.zeros((4,), dtype=np.int64)

    dataset = train.PatientBagDataset(
        expression_matrix=expression,
        bag_records=bag_records,
        celltype_indices=celltypes,
        treatment_indices=treatments,
    )
    sample0 = dataset[0]
    sample1 = dataset[1]
    batch = train.collate_patient_bags([sample0, sample1])
    assert batch["expression"].shape[0] == 4
    assert int(batch["num_bags"].item()) == 2

    metrics = train.compute_binary_metrics(labels=[0, 1, 1, 0], probabilities_pos=[0.1, 0.8, 0.9, 0.2])
    assert "auroc" in metrics

    patients, labels, probs = train.aggregate_patient_probabilities(
        patient_indices=[0, 0, 1, 1],
        labels=[1, 1, 0, 0],
        probabilities_pos=[0.8, 0.7, 0.1, 0.2],
        reduction="mean",
    )
    assert patients.tolist() == [0, 1]
    assert labels.tolist() == [1, 0]
    assert probs.shape == (2,)


def test_train_model_spec_embeddings_and_optimizer() -> None:
    spec = train.resolve_cell_encoder_spec(
        config=SimpleNamespace(
            cell_encoder_name="graph_gat",
            gnn_hidden_dim=8,
            gnn_num_layers=2,
            gnn_num_heads=2,
            gnn_dropout=0.1,
            gnn_graph_readout="mean",
            transformer_conv_feat_dim=8,
            transformer_conv_num_layers=2,
            transformer_conv_heads=2,
            transformer_conv_dropout=0.1,
            transformer_conv_graph_readout="mean",
        ),
        num_nodes=10,
    )
    assert spec["name"] == "graph_gat"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        custom_embedding_path = temp_path / "llm_view.pkl"
        with custom_embedding_path.open("wb") as file_handle:
            pickle.dump(
                {
                    "G1": np.array([1.0, 0.0], dtype=np.float32),
                    "G2": np.array([0.0, 1.0], dtype=np.float32),
                    "G3": np.array([1.0, 1.0], dtype=np.float32),
                },
                file_handle,
            )

        custom_config = SimpleNamespace(
            protein_embedding_choice="concat",
            prior_view_sources=["LLM_view"],
            protein_embedding_paths={"LLM_view": str(custom_embedding_path)},
            protein_embedding_merged_cache_path=str(temp_path / "custom_cache.pkl"),
            dataset="toy",
            split_preselection_subdir="split_preselection",
            split_number=0,
            k=2,
        )
        custom_by_view = train.load_prior_embeddings_by_view(
            genes=["G1", "G2"],
            view_names=["LLM_view"],
            config=custom_config,
        )
        assert set(custom_by_view.keys()) == {"LLM_view"}
        assert custom_by_view["LLM_view"].shape == (2, 2)

        missing_embedding_path = temp_path / "missing_view.pkl"
        with missing_embedding_path.open("wb") as file_handle:
            pickle.dump(
                {
                    "G1": np.array([1.0, 0.0], dtype=np.float32),
                },
                file_handle,
            )
        missing_config = SimpleNamespace(
            protein_embedding_choice="concat",
            prior_view_sources=["Missing_view"],
            protein_embedding_paths={"Missing_view": str(missing_embedding_path)},
            protein_embedding_merged_cache_path=str(temp_path / "missing_cache.pkl"),
            dataset="toy",
            split_preselection_subdir="split_preselection",
            split_number=0,
            k=2,
        )
        try:
            train.load_prior_embeddings_by_view(
                genes=["G1", "G2"],
                view_names=["Missing_view"],
                config=missing_config,
            )
        except ValueError as error:
            assert "missing 1" in str(error)
        else:
            raise AssertionError("expected missing-gene embedding load to fail loudly")

        impl_module = train._load_impl_module()
        original_loader = impl_module.load_protein_embedding_dict

        def fake_load_protein_embedding_dict(source_name: str, embedding_paths=None):
            del source_name, embedding_paths
            return {
                "G1": np.array([1.0, 0.0], dtype=np.float32),
                "G2": np.array([0.0, 1.0], dtype=np.float32),
                "G3": np.array([1.0, 1.0], dtype=np.float32),
            }

        impl_module.load_protein_embedding_dict = fake_load_protein_embedding_dict
        try:
            embedding_config = SimpleNamespace(
                protein_embedding_choice="concat",
                gene_embedding_views={
                    "custom_llm": str(temp_path / "custom_llm.pt"),
                    "ppi_prior": str(temp_path / "ppi_prior.pt"),
                },
                protein_embedding_merged_cache_path=str(temp_path / "merged_cache.pkl"),
                dataset="toy",
                split_preselection_subdir="split_preselection",
                split_number=0,
                k=2,
            )

            fixed, description, merged, source_paths = train.build_protein_embedding_matrix(
                genes=["G1", "G2"],
                config=embedding_config,
                hidden_dimension=8,
            )
            assert fixed.shape == (2, 8)
            assert merged is not None and merged.shape == (2, 4)
            assert "concat" in description
            assert len(source_paths) >= 1

            by_view = train.load_prior_embeddings_by_view(
                genes=["G1", "G2"],
                view_names=["custom_llm", "ppi_prior"],
                config=embedding_config,
            )
            assert set(by_view.keys()) == {"custom_llm", "ppi_prior"}
            assert by_view["custom_llm"].shape == (2, 2)

            try:
                train.load_prior_embeddings_by_view(
                    genes=["G1", "G4"],
                    view_names=["custom_llm"],
                    config=embedding_config,
                )
            except ValueError as error:
                assert "missing 1 required genes" in str(error)
            else:
                raise AssertionError("expected missing-gene failure for strict embedding loading")
        finally:
            impl_module.load_protein_embedding_dict = original_loader

        deg_dir = temp_path / "DEG"
        deg_dir.mkdir(parents=True)
        pd.DataFrame({"gene": ["G1", "G2"], "regulation": [1, -1]}).to_csv(
            deg_dir / "DEG_metrics_Tcell_pass.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame({"gene": ["G1", "G2"], "zscore": [2.0, -1.0]}).to_csv(
            deg_dir / "DEG_zscore_global.tsv",
            sep="\t",
            index=False,
        )

        regulation, metric_paths = train.build_celltype_regulation_onehot(
            dataset="toy",
            genes=["G1", "G2"],
            celltype_names=["Tcell"],
            deg_dir=str(deg_dir),
        )
        assert regulation.shape == (1, 2, 3)
        assert len(metric_paths) == 1

        zscore_vector, zscore_path, missing = train.build_global_deg_zscore_vector(
            dataset="toy",
            genes=["G1", "G2", "G4"],
            deg_dir=str(deg_dir),
        )
        assert zscore_vector.shape == (3,)
        assert zscore_path.endswith("DEG_zscore_global.tsv")
        assert missing == 1

        artifacts = train.build_experiment_directory(
            SimpleNamespace(
                experiment_root=str(temp_path),
                cell_encoder_name="graph_gat",
                protein_embedding_choice="concat",
                mil_pooling="attention",
                classifier_name="linear",
                cells_per_bag=64,
                bags_per_patient_per_epoch=2,
                lr=1e-3,
                seed=42,
                split_number=0,
            )
        )
        assert Path(artifacts["output_dir"]).is_dir()
        assert Path(artifacts["cache_dir"]).is_dir()

    num_nodes = 8
    num_celltypes = 3
    cell_encoder = CellEncoder.GraphCellEncoder(
        num_nodes=num_nodes,
        hidden_dimension=8,
        number_of_layers=1,
        number_of_heads=2,
        dropout_probability=0.0,
        edge_index=ring_edge_index(num_nodes),
        protein_prior_embeddings=torch.randn(num_nodes, 4),
        global_zscore=torch.randn(num_nodes),
        regulation_onehot=build_regulation_onehot(num_celltypes, num_nodes),
        num_celltypes=num_celltypes,
        num_treatments=0,
        attention_logit_clamp=5.0,
        expression_feature_scale=1.0,
        max_message_memory_mb=64,
        graph_readout="mean",
        prior_injection_enabled=False,
    )
    mil_aggregator = MultipleInstanceLearning.PatientMILAggregator(
        embedding_dimension=8,
        num_celltypes=num_celltypes,
        pooling="mean",
        attention_hidden_dimension=4,
        recursive_steps=1,
    )
    classifier = train.PatientClassifier(
        input_dimension=8,
        config=SimpleNamespace(classifier_name="linear", classifier_hidden_dim=8, classifier_dropout=0.1),
    )

    optimizer = train.build_optimizer(
        config=SimpleNamespace(optimizer="adam", lr=1e-3, weight_decay=1e-4),
        cell_encoder=cell_encoder,
        mil_aggregator=mil_aggregator,
        classifier=classifier,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)

    assert callable(train.build_graph_cache)
    assert callable(train.build_split_records_cache)
    assert callable(train.run_epoch)


def test_train_run_training_phase_direct_call() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        adata = sc.AnnData(X=np.random.randn(6, 4).astype(np.float32))
        adata.var_names = pd.Index(["G1", "G2", "G3", "G4"])
        adata.obs["patient_id"] = [f"p{i // 2}" for i in range(6)]
        adata.obs["label"] = ["1", "1", "0", "0", "1", "0"]
        adata.obs["celltype"] = ["T", "B", "T", "B", "T", "B"]
        adata_path = temp_path / "toy_data.h5ad"
        adata.write_h5ad(adata_path)

        splits_dir = temp_path / "splits"
        splits_dir.mkdir(parents=True)
        with (splits_dir / "toy_idx_0.pkl").open("wb") as file_handle:
            pickle.dump([[0, 1], [2, 3], [4, 5]], file_handle)

        preselection_output_root = temp_path / "preselection"
        preselection_root = preselection_output_root / "split_0"
        (preselection_root / "DEG").mkdir(parents=True)
        (preselection_root / "NP").mkdir(parents=True)

        config = train.build_train_config_from_cli_args(
            SimpleNamespace(dataset="asthma", seed=0, gpu=0, split_number=0)
        )
        config.dataset = "toy"
        config.seed = 0
        config.gpu = 0
        config.split_number = 0
        config.adata_path = str(adata_path)
        config.adata_directory = str(temp_path)
        config.splits_directory = str(splits_dir)
        config.preselection_root = ""
        config.preselection_output_root = str(preselection_output_root)
        config.experiment_root = str(temp_path / "experiment")
        config.patient_column = "patient_id"
        config.label_column = "label"
        config.celltype_column = "celltype"
        config.treatment_column = None
        config.binary_positive_labels = ["1"]
        config.binary_negative_labels = ["0"]
        config.epochs = 5
        config.early_stopping_patience = 10
        config.deterministic_training = False
        config.deterministic_algorithms = False
        config.deterministic_warn_only = False
        config.prior_view_sources = ["custom_view"]
        config.gene_embedding_views = {"custom_view": str(temp_path / "custom_view.pkl")}
        config.protein_embedding_paths = dict(config.gene_embedding_views)

        impl_module = train._load_impl_module()
        patched_names = [
            "ensure_cublas_workspace_config",
            "configure_runtime_backends",
            "build_experiment_directory",
            "load_k_np_genes",
            "build_graph_cache",
            "load_prior_embeddings_by_view",
            "resolve_cell_encoder_spec",
            "build_split_records_cache",
            "run_epoch",
            "PatientMILAggregator",
            "PatientClassifier",
        ]
        originals = {name: getattr(impl_module, name) for name in patched_names}

        class DummyEncoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                del args, kwargs
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def set_node_feature_beta(self, value: float) -> None:
                self.node_feature_beta = float(value)

        class DummyStack(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                del args, kwargs
                self.weight = torch.nn.Parameter(torch.zeros(1))

        observed_preselection_roots: list[str] = []

        def fake_build_experiment_directory(cfg):
            del cfg
            output_dir = temp_path / "run"
            cache_dir = output_dir / "cache"
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            return {
                "output_dir": str(output_dir),
                "cache_dir": str(cache_dir),
                "history_path": str(output_dir / "history.csv"),
                "train_step_log_path": str(output_dir / "train_step_metrics.csv"),
                "val_step_log_path": str(output_dir / "val_step_metrics.csv"),
                "test_step_log_path": str(output_dir / "test_step_metrics.csv"),
                "best_model_path": str(output_dir / "best_model.pt"),
                "best_checkpoint_path": str(output_dir / "best_checkpoint.pt"),
                "last_checkpoint_path": str(output_dir / "last_checkpoint.pt"),
                "epoch5_checkpoint_path": str(output_dir / "epoch5_checkpoint.pt"),
            }

        def fake_load_k_np_genes(*args, **kwargs):
            del args
            observed_preselection_roots.append(str(kwargs.get("preselection_root", "")))
            return ["G1", "G2"], ["G1", "G2"]

        def fake_build_graph_cache(*args, **kwargs):
            del args
            observed_preselection_roots.append(str(kwargs.get("preselection_root", "")))
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((1, 2, 3), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.float32),
                False,
                {},
            )

        def fake_load_prior_embeddings_by_view(*args, **kwargs):
            del args, kwargs
            return {"custom_view": torch.zeros((2, 2), dtype=torch.float32)}

        def fake_resolve_cell_encoder_spec(*args, **kwargs):
            del args, kwargs
            return {
                "name": "dummy",
                "encoder_class": DummyEncoder,
                "hidden_dimension": 8,
                "number_of_layers": 1,
                "number_of_heads": 1,
                "dropout_probability": 0.0,
                "graph_readout": "mean",
                "cell_embedding_dimension": 8,
            }

        def fake_build_split_records_cache(*args, **kwargs):
            del args, kwargs
            record = {"patient_index": 0, "patient_label": 1, "cell_indices": np.array([0, 1]), "bag_repeat_index": 0}
            return [record], [record], [record], False

        def fake_run_epoch(*args, **kwargs):
            epoch = int(kwargs.get("epoch", 0))
            split_name = str(kwargs.get("split_name", "train"))
            del args
            metric_base = 0.5 + 0.05 * float(epoch)
            metrics = {
                "loss": metric_base,
                "task_loss": metric_base - 0.05,
                "smooth_loss": 0.0,
                "sparse_loss": 0.0,
                "cons_loss": 0.0,
                "total_loss": metric_base,
                "loss_bce": metric_base - 0.1,
                "loss_mixup": 0.0,
                "effective_lambda_smooth": 0.0,
                "effective_lambda_sparse": 0.0,
                "effective_lambda_cons": 0.0,
                "prior_strategy": "absolute=1,relational=1",
                "prior_strategy_regularizer_multiview_knn": True,
                "prior_strategy_hidden_additive": True,
                "node_feature_beta": 1.0,
                "mixup_alpha": 0.0,
                "lambda_sparse": 0.0,
                "lambda_edge_bias": 0.0,
                "node_score_mean": 0.0,
                "node_score_abs_mean": 0.0,
                "accuracy": metric_base,
                "precision": metric_base,
                "recall": metric_base,
                "auprc": metric_base,
                "auroc": metric_base,
                "f1": metric_base,
                "brier": 1.0 - metric_base,
                "num_patients": 1.0,
                "num_bags": 1.0,
            }
            rows = [{"epoch": epoch, "split": split_name, "metric": metric_base}]
            return metrics, rows, rows, rows

        try:
            impl_module.build_experiment_directory = fake_build_experiment_directory
            impl_module.load_k_np_genes = fake_load_k_np_genes
            impl_module.build_graph_cache = fake_build_graph_cache
            impl_module.load_prior_embeddings_by_view = fake_load_prior_embeddings_by_view
            impl_module.resolve_cell_encoder_spec = fake_resolve_cell_encoder_spec
            impl_module.build_split_records_cache = fake_build_split_records_cache
            impl_module.run_epoch = fake_run_epoch
            impl_module.PatientMILAggregator = DummyStack
            impl_module.PatientClassifier = DummyStack
            impl_module.ensure_cublas_workspace_config = lambda: None
            impl_module.configure_runtime_backends = lambda **kwargs: None

            artifacts = train.run_training_phase(config, device=torch.device("cpu"))
        finally:
            for name, original in originals.items():
                setattr(impl_module, name, original)

        assert Path(artifacts["output_dir"]).is_dir()
        assert Path(artifacts["best_checkpoint_path"]).is_file()
        assert observed_preselection_roots
        assert set(observed_preselection_roots) == {str(preselection_root)}


def main() -> None:
    test_train_endpoints_exist()
    test_train_runtime_and_dataset_helpers()
    test_train_model_spec_embeddings_and_optimizer()
    test_train_run_training_phase_direct_call()
    print_success("train endpoints")


if __name__ == "__main__":
    main()
