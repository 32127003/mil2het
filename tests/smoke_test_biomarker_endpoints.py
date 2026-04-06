from __future__ import annotations

import os
import pickle
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from mil2het import CellEncoder, MultipleInstanceLearning, biomarker

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
    "run_analysis_phase",
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


class ToyAdataView:
    def __init__(self, matrix: np.ndarray) -> None:
        self.X = matrix


class ToyAdata:
    def __init__(self, expression: np.ndarray, obs: pd.DataFrame, var_names: list[str]) -> None:
        self._expression = np.asarray(expression, dtype=np.float32)
        self.obs = obs
        self.var_names = list(var_names)
        self.n_obs = int(self._expression.shape[0])

    def __getitem__(self, item):
        row_sel, col_sel = item
        return ToyAdataView(self._expression[row_sel][:, col_sel])


def test_biomarker_endpoints_exist() -> None:
    assert_module_endpoints(
        biomarker,
        BIOMARKER_ENDPOINTS,
        module_label="mil2het.biomarker",
    )


def test_biomarker_config_and_path_helpers() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        config_file = temp_path / "toy_config.py"
        config_file.write_text("config = {'dataset': 'toy', 'gnn_hidden_dim': 8}\n", encoding="utf-8")
        config_ns, config_dict = biomarker.load_config(str(config_file), dataset_hint="toy")
        assert config_ns.dataset == "toy"
        assert config_dict["gnn_hidden_dim"] == 8

        yaml_config = temp_path / "toy_config.yml"
        yaml_config.write_text(
            "\n".join(
                [
                    "workflow:",
                    "  input_h5ad: /tmp/toy_data.h5ad",
                    "  analysis_only: true",
                    "  run_dir: /tmp/run_dir",
                    "columns:",
                    "  patient: patient_id",
                    "  celltype: celltype",
                    "  label: label",
                    "resources:",
                    "  ppi_path: /tmp/ppi.tsv",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        yaml_ns, yaml_dict = biomarker.load_config(str(yaml_config), dataset_hint="toy")
        assert yaml_ns.analysis_only is True
        assert yaml_ns.run_dir == "/tmp/run_dir"
        assert yaml_dict["patient_column"] == "patient_id"

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
        workflow_snapshot = run_dir / "workflow_config.yaml"
        workflow_snapshot.write_text("workflow:\n  input_h5ad: /tmp/toy_data.h5ad\n", encoding="utf-8")
        snapshot_copy = run_dir / "asthma_config.py"
        snapshot_copy.write_text("config = {'dataset': 'asthma'}\n", encoding="utf-8")

        located = biomarker.locate_run_snapshot_config_path(str(run_dir))
        assert os.path.samefile(located, workflow_snapshot)

        cwd_relative_dir = temp_path / "cwd_relative"
        cwd_relative_dir.mkdir(parents=True)
        cwd_relative_snapshot = cwd_relative_dir / "workflow_config.yaml"
        cwd_relative_snapshot.write_text("workflow:\n  input_h5ad: /tmp/cwd_relative.h5ad\n", encoding="utf-8")
        original_cwd = os.getcwd()
        try:
            os.chdir(str(temp_path))
            cwd_relative_located = biomarker.locate_run_snapshot_config_path(
                str(run_dir),
                cli_config=os.path.join("cwd_relative", "workflow_config.yaml"),
            )
        finally:
            os.chdir(original_cwd)
        assert os.path.samefile(cwd_relative_located, cwd_relative_snapshot)

        base_dirs = biomarker.build_resolution_base_dirs(str(snapshot_copy), str(run_dir))
        resolved = biomarker.resolve_path_with_base_dirs(base_dirs, "asthma_config.py", prefer_existing=True)
        assert os.path.samefile(resolved, snapshot_copy)

        run_relative_snapshot = run_dir / "explicit_workflow.yaml"
        run_relative_snapshot.write_text("workflow:\n  input_h5ad: /tmp/run_relative.h5ad\n", encoding="utf-8")
        run_relative_located = biomarker.locate_run_snapshot_config_path(
            str(run_dir),
            cli_config="explicit_workflow.yaml",
        )
        assert os.path.samefile(run_relative_located, run_relative_snapshot)

        workflow_snapshot.unlink()
        legacy_located = biomarker.locate_run_snapshot_config_path(str(run_dir))
        assert os.path.samefile(legacy_located, snapshot_copy)

    dataset_hint = biomarker.infer_dataset_hint_from_run_dir("/tmp/asthma_ext_split_0")
    assert dataset_hint == "asthma_ext"
    split_number = biomarker.infer_split_number_from_run_dir("/tmp/train_split_5_run")
    assert split_number == 5

    device = biomarker.resolve_device(gpu_index=-1, config=SimpleNamespace(device="cpu", cuda_device_index=0))
    assert device.type == "cpu"

    ordered_sources = biomarker._resolve_embedding_view_sources(
        SimpleNamespace(
            protein_embedding_paths={
                "esm3": "/tmp/esm3.pt",
                "node2vec": "/tmp/node2vec.pt",
            },
            prior_view_sources=["node2vec", "esm3"],
            protein_embedding_sources=["node2vec", "esm3"],
        )
    )
    assert ordered_sources == ["node2vec", "ESM3"]


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
            protein_embedding_paths={"LLM_view": str(custom_embedding_path)},
            protein_embedding_choice="concat",
            protein_embedding_sources=["LLM_view"],
            gnn_hidden_dim=4,
        )
        custom_by_view = biomarker.load_prior_embeddings_by_view(
            genes=["G1", "G2"],
            view_names=["LLM_view"],
            config=custom_config,
        )
        assert set(custom_by_view.keys()) == {"LLM_view"}
        assert custom_by_view["LLM_view"].shape == (2, 2)

        missing_embedding_path = temp_path / "missing_view.pkl"
        with missing_embedding_path.open("wb") as file_handle:
            pickle.dump({"G1": np.array([1.0, 0.0], dtype=np.float32)}, file_handle)
        missing_config = SimpleNamespace(
            protein_embedding_paths={"Missing_view": str(missing_embedding_path)},
            protein_embedding_choice="concat",
            protein_embedding_sources=["Missing_view"],
            gnn_hidden_dim=4,
        )
        try:
            biomarker.load_prior_embeddings_by_view(
                genes=["G1", "G2"],
                view_names=["Missing_view"],
                config=missing_config,
            )
        except ValueError as error:
            assert "missing 1" in str(error)
        else:
            raise AssertionError("expected missing-gene biomarker load to fail loudly")

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
            gene_embedding_views={
                "view_alpha": "/tmp/view_alpha.pt",
                "view_beta": "/tmp/view_beta.pt",
            },
            protein_embedding_choice="concat",
            gnn_hidden_dim=4,
        )

        by_view = biomarker.load_prior_embeddings_by_view(
            genes=["G1", "G2"],
            view_names=["view_alpha", "view_beta"],
            config=embedding_config,
        )
        assert set(by_view.keys()) == {"view_alpha", "view_beta"}
        assert by_view["view_alpha"].shape == (2, 2)

        fixed, description, merged = biomarker.build_protein_embedding_matrix(
            genes=["G1", "G2"],
            config=embedding_config,
        )
        assert fixed.shape == (2, 4)
        assert "concat" in description
        assert merged is not None and merged.shape == (2, 4)

        try:
            biomarker.load_prior_embeddings_by_view(
                genes=["G1", "G4"],
                view_names=["view_alpha"],
                config=embedding_config,
            )
        except ValueError as error:
            assert "missing 1 required genes" in str(error)
        else:
            raise AssertionError("expected missing-gene failure for strict embedding loading")
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


def test_biomarker_run_analysis_phase_smoke() -> None:
    impl_module = biomarker._load_impl_module()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        run_dir = temp_path / "run_dir"
        run_dir.mkdir(parents=True)
        preselection_root = temp_path / "preselection"
        np_dir = preselection_root / "NP"
        deg_dir = preselection_root / "DEG"
        splits_dir = temp_path / "splits"
        np_dir.mkdir(parents=True)
        deg_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)

        genes = ["G1", "G2"]
        pd.DataFrame({"gene": genes}).to_csv(np_dir / "NP_max.tsv", sep="\t", index=False)
        for celltype_name in ["Bcell", "Tcell"]:
            pd.DataFrame({"gene": genes, "regulation": [1, -1]}).to_csv(
                deg_dir / f"DEG_metrics_{celltype_name}_pass.tsv",
                sep="\t",
                index=False,
            )
        pd.DataFrame({"gene": genes, "zscore": [1.5, -0.5]}).to_csv(
            deg_dir / "DEG_zscore_global.tsv",
            sep="\t",
            index=False,
        )

        pathway_path = temp_path / "toy_pathways.json"
        with pathway_path.open("w", encoding="utf-8") as handle:
            json.dump({"pathway_a": genes}, handle)

        ppi_path = temp_path / "toy_ppi.tsv"
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")

        embedding_path = temp_path / "toy_view.pkl"
        with embedding_path.open("wb") as handle:
            pickle.dump(
                {
                    "G1": np.array([1.0, 0.0], dtype=np.float32),
                    "G2": np.array([0.0, 1.0], dtype=np.float32),
                },
                handle,
            )

        output_dir = temp_path / "analysis_output"
        config_path = run_dir / "asthma_config.py"
        config_payload = {
            "dataset": "toy",
            "split_number": 0,
            "seed": 0,
            "biomarker_random_seed": 0,
            "deterministic_training": False,
            "adata_path": str(temp_path / "toy_data.h5ad"),
            "label_column": "label",
            "patient_column": "patient",
            "celltype_column": "celltype",
            "binary_positive_labels": ["pos"],
            "binary_negative_labels": ["neg"],
            "biomarker_output_dir": str(output_dir),
            "biomarker_pathway_gene_set_path": str(pathway_path),
            "splits_directory": str(splits_dir),
            "ppi_path": str(ppi_path),
            "k": 2,
            "gnn_self_loop": True,
            "cell_encoder_name": "graph_gat",
            "gnn_hidden_dim": 4,
            "gnn_num_layers": 1,
            "gnn_num_heads": 1,
            "gnn_dropout": 0.0,
            "gnn_graph_readout": "mean",
            "gnn_attention_logit_clamp": 5.0,
            "gnn_expression_feature_scale": 1.0,
            "gnn_max_message_memory_mb": 64.0,
            "prior_absolute_enabled": False,
            "prior_relational_enabled": False,
            "prior_interface_num_heads": 1,
            "prior_interface_dropout": 0.0,
            "prior_interface_ffn_hidden_dim": 0,
            "prior_knn_k": 1,
            "prior_knn_symmetric": True,
            "gene_embedding_views": {"toy_view": str(embedding_path)},
            "mil_pooling": "mean",
            "mil_attention_hidden_dim": 4,
            "classifier_name": "linear",
            "classifier_hidden_dim": 4,
            "classifier_dropout": 0.0,
            "biomarker_split": "test",
            "biomarker_bags_per_patient": 1,
            "biomarker_cells_per_bag": 2,
            "biomarker_sample_mode": "proportional",
            "biomarker_metric": "auprc",
            "biomarker_patient_probability_reduction": "mean",
            "biomarker_pathway_permutations": 1,
            "biomarker_celltype_permutations": 1,
            "biomarker_stability_enabled": False,
            "biomarker_min_pathway_size": 1,
            "biomarker_max_pathway_size": 10,
            "biomarker_max_pathways": 1,
            "biomarker_step1_celltype_mode": "all",
            "biomarker_step1_top_k": 1,
            "biomarker_step1_cum_weight": 1.0,
            "biomarker_step1_weight_threshold": 0.0,
            "split_train_only_preselection": False,
        }
        config_path.write_text("config = " + repr(config_payload) + "\n", encoding="utf-8")

        config_ns, _ = biomarker.load_config_like_train_from_run_snapshot(str(config_path), dataset_hint="toy")
        config_ns.adata_path = str(temp_path / "toy_data.h5ad")

        edge_index, _ = impl_module.build_edge_index(genes, str(ppi_path), True)
        regulation_onehot = impl_module.build_celltype_regulation_onehot(
            "toy",
            genes,
            ["Bcell", "Tcell"],
            deg_dir=str(deg_dir),
        )
        global_zscore, _, _ = impl_module.build_global_deg_zscore_vector("toy", genes, deg_dir=str(deg_dir))
        protein_prior_embeddings = torch.zeros((len(genes), int(config_ns.gnn_hidden_dim)), dtype=torch.float32)

        cell_encoder = impl_module.GraphCellEncoder(
            num_nodes=len(genes),
            hidden_dimension=int(config_ns.gnn_hidden_dim),
            number_of_layers=int(config_ns.gnn_num_layers),
            number_of_heads=int(config_ns.gnn_num_heads),
            dropout_probability=float(config_ns.gnn_dropout),
            edge_index=edge_index,
            protein_prior_embeddings=protein_prior_embeddings,
            global_zscore=global_zscore,
            regulation_onehot=regulation_onehot,
            num_celltypes=2,
            num_treatments=1,
            attention_logit_clamp=float(config_ns.gnn_attention_logit_clamp),
            expression_feature_scale=float(config_ns.gnn_expression_feature_scale),
            max_message_memory_mb=float(config_ns.gnn_max_message_memory_mb),
            graph_readout=str(config_ns.gnn_graph_readout),
            protein_prior_alpha=1.0,
            protein_prior_alpha_learnable=False,
            prior_injection_enabled=False,
            prior_absolute_enabled=False,
            prior_relational_enabled=False,
            protein_prior_base_embeddings=None,
            protein_prior_projection_hidden_dim=0,
            protein_prior_projection_dropout=0.0,
            protein_prior_projection_mode="find",
            protein_prior_view_embeddings={
                "toy_view": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            },
            prior_interface_num_heads=int(config_ns.prior_interface_num_heads),
            prior_interface_dropout=float(config_ns.prior_interface_dropout),
            prior_interface_ffn_hidden_dim=int(config_ns.prior_interface_ffn_hidden_dim),
            prior_knn_k=int(config_ns.prior_knn_k),
            prior_knn_symmetric=bool(config_ns.prior_knn_symmetric),
            lambda_edge_bias=0.0,
        )
        mil_aggregator = impl_module.PatientMILAggregator(
            embedding_dimension=int(config_ns.gnn_hidden_dim),
            num_celltypes=2,
            pooling=str(config_ns.mil_pooling),
            attention_hidden_dimension=int(config_ns.mil_attention_hidden_dim),
        )
        classifier = impl_module.PatientClassifier(
            input_dimension=int(config_ns.gnn_hidden_dim),
            config=SimpleNamespace(
                classifier_name="linear",
                classifier_hidden_dim=int(config_ns.classifier_hidden_dim),
                classifier_dropout=float(config_ns.classifier_dropout),
            ),
        )
        torch.save(
            {
                "cell_encoder": cell_encoder.state_dict(),
                "mil_aggregator": mil_aggregator.state_dict(),
                "classifier": classifier.state_dict(),
            },
            run_dir / "best_checkpoint.pt",
        )

        split_payload = [[], [], [0, 1, 2, 3]]
        with (splits_dir / "toy_idx_0.pkl").open("wb") as handle:
            pickle.dump(split_payload, handle)

        toy_obs = pd.DataFrame(
            {
                "label": ["neg", "neg", "pos", "pos"],
                "patient": ["P0", "P0", "P1", "P1"],
                "celltype": ["Bcell", "Tcell", "Bcell", "Tcell"],
            }
        )
        toy_expression = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.2],
                [0.2, 0.5],
            ],
            dtype=np.float32,
        )
        toy_adata = ToyAdata(toy_expression, toy_obs, genes)

        original_read_h5ad = impl_module.sc.read_h5ad
        original_resolve_preselection_root = impl_module.resolve_preselection_root
        impl_module.sc.read_h5ad = lambda path: toy_adata
        impl_module.resolve_preselection_root = lambda config: str(preselection_root)
        try:
            result = biomarker.run_analysis_phase(
                str(run_dir),
                config_path=str(config_path),
                output_dir=str(output_dir),
                pathway_path=str(pathway_path),
                gpu_index=-1,
            )
        finally:
            impl_module.sc.read_h5ad = original_read_h5ad
            impl_module.resolve_preselection_root = original_resolve_preselection_root

        assert result["output_dir"] == str(output_dir)
        assert result["dataset"] == "toy"
        assert result["num_patients"] == 2
        assert (output_dir / "final_biomarker.tsv").is_file()
        assert (output_dir / "biomarker_metadata.json").is_file()


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
    test_biomarker_run_analysis_phase_smoke()
    test_biomarker_forward_helpers_and_cli()
    print_success("biomarker endpoints")


if __name__ == "__main__":
    main()
