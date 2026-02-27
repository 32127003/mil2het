from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scbiomarker import CellEncoder, MultipleInstanceLearning, biomarker

from smoke_test_helpers import assert_module_endpoints, print_success

BIOMARKER_ENDPOINTS = [
    "BagDefinition",
    "BagCache",
    "PatientClassifier",
    "load_config",
    "load_config_like_train_from_run_snapshot",
    "resolve_device",
    "resolve_cell_encoder_spec",
    "load_prior_embeddings_by_view",
    "load_np_max_genes",
    "build_protein_embedding_matrix",
    "build_celltype_regulation_onehot",
    "build_global_deg_zscore_vector",
    "sample_patient_bag_indices",
    "compute_binary_metrics",
    "aggregate_patient_probabilities",
    "stable_spearman",
    "jaccard_index",
    "forward_bag_from_expression",
    "forward_bag_from_embeddings",
    "sample_rows_with_replacement",
    "parse_args",
    "main",
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


def test_biomarker_endpoints_exist() -> None:
    assert_module_endpoints(
        biomarker,
        BIOMARKER_ENDPOINTS,
        module_label="scbiomarker.biomarker",
    )


def test_biomarker_config_and_path_helpers() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        config_file = temp_path / "toy_config.py"
        config_file.write_text("config = {'dataset': 'toy', 'gnn_hidden_dim': 8}\n", encoding="utf-8")
        config_ns, config_dict = biomarker.load_config(str(config_file), dataset_hint="toy")
        assert config_ns.dataset == "toy"
        assert config_dict["gnn_hidden_dim"] == 8

        train_snapshot = temp_path / "train_snapshot.py"
        train_snapshot.write_text(
            "asthma_train_configuration = {'cell_encoder_name': 'graph_gat'}\n",
            encoding="utf-8",
        )
        snapshot_ns, snapshot_dict = biomarker.load_config_like_train_from_run_snapshot(
            str(train_snapshot),
            dataset_hint="asthma",
        )
        assert snapshot_ns.dataset == "asthma"
        assert snapshot_dict["cell_encoder_name"] == "graph_gat"

        run_dir = temp_path / "run_dir"
        run_dir.mkdir(parents=True)
        snapshot_copy = run_dir / "asthma_config.py"
        snapshot_copy.write_text("config = {'dataset': 'asthma'}\n", encoding="utf-8")

        located = biomarker.locate_run_snapshot_config_path(str(run_dir))
        assert os.path.samefile(located, snapshot_copy)

        base_dirs = biomarker.build_resolution_base_dirs(str(snapshot_copy), str(run_dir))
        resolved = biomarker.resolve_path_with_base_dirs(base_dirs, "asthma_config.py", prefer_existing=True)
        assert os.path.samefile(resolved, snapshot_copy)

    dataset_hint = biomarker.infer_dataset_hint_from_run_dir("/tmp/asthma_ext_split_0")
    assert dataset_hint == "asthma_ext"
    split_number = biomarker.infer_split_number_from_run_dir("/tmp/train_split_5_run")
    assert split_number == 5

    device = biomarker.resolve_device(gpu_index=-1, config=SimpleNamespace(device="cpu", cuda_device_index=0))
    assert device.type == "cpu"


def test_biomarker_feature_builders_and_metrics() -> None:
    graph_spec = biomarker.resolve_cell_encoder_spec(
        SimpleNamespace(
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
        num_nodes=6,
    )
    assert graph_spec["name"] == "graph_gat"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        np_dir = temp_path / "NP"
        np_dir.mkdir(parents=True)
        pd.DataFrame({"gene": ["G1", "G2", "G3"]}).to_csv(np_dir / "NP_max.tsv", sep="\t", index=False)
        np_genes = biomarker.load_np_max_genes(dataset="toy", k=2, preselection_root=str(temp_path))
        assert np_genes == ["G1", "G2"]

        deg_dir = temp_path / "DEG"
        deg_dir.mkdir(parents=True)
        pd.DataFrame({"gene": ["G1", "G2"], "regulation": [1, -1]}).to_csv(
            deg_dir / "DEG_metrics_Tcell_pass.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame({"gene": ["G1", "G2"], "zscore": [2.5, -1.2]}).to_csv(
            deg_dir / "DEG_zscore_global.tsv",
            sep="\t",
            index=False,
        )

        regulation = biomarker.build_celltype_regulation_onehot(
            dataset="toy",
            genes=["G1", "G2"],
            celltype_names=["Tcell"],
            deg_dir=str(deg_dir),
        )
        assert regulation.shape == (1, 2, 3)

        zscore_vector, zscore_path, missing = biomarker.build_global_deg_zscore_vector(
            dataset="toy",
            genes=["G1", "G2", "G4"],
            deg_dir=str(deg_dir),
        )
        assert zscore_vector.shape == (3,)
        assert os.path.basename(zscore_path) == "DEG_zscore_global.tsv"
        assert missing == 1

    impl_module = biomarker._load_impl_module()
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
            protein_embedding_paths=None,
            protein_embedding_choice="concat",
            protein_embedding_sources=["GPT", "ESM3"],
            gnn_hidden_dim=4,
        )

        by_view = biomarker.load_prior_embeddings_by_view(
            genes=["G1", "G2"],
            view_names=["GPT", "ESM3"],
            config=embedding_config,
        )
        assert set(by_view.keys()) == {"GPT", "ESM3"}
        assert by_view["GPT"].shape == (2, 2)

        fixed, description, merged = biomarker.build_protein_embedding_matrix(
            genes=["G1", "G2"],
            config=embedding_config,
        )
        assert fixed.shape == (2, 4)
        assert "concat" in description
        assert merged is not None and merged.shape == (2, 4)
    finally:
        impl_module.load_protein_embedding_dict = original_loader

    patient_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    patient_celltypes = np.array([0, 1, 0, 1, 2, 2], dtype=np.int64)
    sampled = biomarker.sample_patient_bag_indices(
        patient_cell_indices=patient_indices,
        patient_celltypes=patient_celltypes,
        bag_size=4,
        sample_mode="proportional",
        rng=np.random.default_rng(0),
    )
    assert sampled.ndim == 1

    metrics = biomarker.compute_binary_metrics(labels=[0, 1, 1, 0], probabilities_pos=[0.2, 0.8, 0.7, 0.1])
    assert "auroc" in metrics

    bag_defs = [
        biomarker.BagDefinition(0, 10, "P10", 1, 0, np.array([0, 1], dtype=np.int64)),
        biomarker.BagDefinition(1, 10, "P10", 1, 1, np.array([2, 3], dtype=np.int64)),
        biomarker.BagDefinition(2, 11, "P11", 0, 0, np.array([4, 5], dtype=np.int64)),
    ]
    bag_probs = {0: 0.9, 1: 0.7, 2: 0.2}
    patient_ids, labels, probs = biomarker.aggregate_patient_probabilities(
        bag_defs=bag_defs,
        bag_probabilities=bag_probs,
        reduction="mean",
    )
    assert patient_ids.tolist() == [10, 11]
    assert labels.tolist() == [1, 0]
    assert probs.shape == (2,)

    corr = biomarker.stable_spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    assert -1.0 <= corr <= 1.0
    assert biomarker.jaccard_index(["a", "b"], ["b", "c"]) == 1.0 / 3.0

    sampled_rows = biomarker.sample_rows_with_replacement(
        matrix=np.arange(12, dtype=np.float32).reshape(6, 2),
        target_rows=4,
        rng=np.random.default_rng(0),
    )
    assert sampled_rows.shape == (4, 2)


def test_biomarker_forward_helpers_and_cli() -> None:
    torch.manual_seed(0)

    num_nodes = 8
    num_cells = 5
    num_celltypes = 3

    edge_index = ring_edge_index(num_nodes)
    encoder = CellEncoder.GraphCellEncoder(
        num_nodes=num_nodes,
        hidden_dimension=8,
        number_of_layers=1,
        number_of_heads=2,
        dropout_probability=0.0,
        edge_index=edge_index,
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
    mil = MultipleInstanceLearning.PatientMILAggregator(
        embedding_dimension=8,
        num_celltypes=num_celltypes,
        pooling="mean",
        attention_hidden_dimension=4,
        recursive_steps=1,
    )
    classifier = biomarker.PatientClassifier(
        input_dimension=8,
        config=SimpleNamespace(
            classifier_name="linear",
            classifier_hidden_dim=8,
            classifier_dropout=0.1,
        ),
    )

    expression = np.random.randn(num_cells, num_nodes).astype(np.float32)
    celltypes = np.random.randint(0, num_celltypes, size=num_cells, dtype=np.int64)
    treatments = np.zeros((num_cells,), dtype=np.int64)

    prob_from_expression, cell_embeddings = biomarker.forward_bag_from_expression(
        cell_encoder=encoder,
        mil_aggregator=mil,
        classifier=classifier,
        expression=expression,
        celltypes=celltypes,
        treatments=treatments,
        device=torch.device("cpu"),
    )
    assert 0.0 <= prob_from_expression <= 1.0
    assert cell_embeddings.shape == (num_cells, 8)

    prob_from_embeddings = biomarker.forward_bag_from_embeddings(
        mil_aggregator=mil,
        classifier=classifier,
        embeddings=cell_embeddings,
        celltypes=celltypes,
        device=torch.device("cpu"),
    )
    assert 0.0 <= prob_from_embeddings <= 1.0

    original_argv = sys.argv
    try:
        sys.argv = ["smoke", "--run_dir", "dummy_run"]
        args = biomarker.parse_args()
        assert args.run_dir == "dummy_run"
    finally:
        sys.argv = original_argv

    assert callable(biomarker.main)


def main() -> None:
    test_biomarker_endpoints_exist()
    test_biomarker_config_and_path_helpers()
    test_biomarker_feature_builders_and_metrics()
    test_biomarker_forward_helpers_and_cli()
    print_success("biomarker endpoints")


if __name__ == "__main__":
    main()
