from __future__ import annotations

import tempfile
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scbiomarker import CellEncoder, MultipleInstanceLearning, train

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
        module_label="scbiomarker.train",
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


def main() -> None:
    test_train_endpoints_exist()
    test_train_runtime_and_dataset_helpers()
    test_train_model_spec_embeddings_and_optimizer()
    print_success("train endpoints")


if __name__ == "__main__":
    main()
