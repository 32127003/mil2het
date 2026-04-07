from __future__ import annotations

import tempfile
from pathlib import Path

from mil2het import config

from smoke_test_helpers import assert_module_endpoints, print_success

CONFIG_ENDPOINTS = [
    "DEFAULT_CONFIG_RESOURCE",
    "build_workflow_arg_parser",
    "dict_to_namespace",
    "load_default_config_dict",
    "load_workflow_config_dict",
    "load_workflow_config_namespace",
    "load_yaml_config_dict",
    "normalize_workflow_config",
    "parse_workflow_cli_args",
    "workflow_overrides_from_args",
]


def test_config_endpoints_exist() -> None:
    assert_module_endpoints(
        config,
        CONFIG_ENDPOINTS,
        module_label="mil2het.config",
    )


def test_config_loading_and_precedence() -> None:
    default_config = config.load_default_config_dict()
    assert "workflow" in default_config
    assert "columns" in default_config

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        custom_yaml = temp_path / "custom.yml"
        custom_yaml.write_text(
            "\n".join(
                [
                    "workflow:",
                    "  input_h5ad: /tmp/custom_data.h5ad",
                    "  output_root: /tmp/out",
                    "columns:",
                    "  patient: patient_id",
                    "  celltype: celltype",
                    "  label: response",
                    "resources:",
                    "  ppi_path: /tmp/ppi.tsv",
                    "  embedding_views:",
                    "    ESM3: /tmp/esm3.pt",
                    "training:",
                    "  epochs: 12",
                    "  lr: 0.005",
                    "biomarker_pathway_gene_set_path: /tmp/toy_pathways.json",
                    "biomarker_min_pathway_size: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = config.load_workflow_config_dict(config_path=str(custom_yaml))
        assert loaded["input_h5ad"] == "/tmp/custom_data.h5ad"
        assert loaded["patient_column"] == "patient_id"
        assert loaded["ppi_path"] == "/tmp/ppi.tsv"
        assert loaded["epochs"] == 12
        assert loaded["lr"] == 0.005
        assert loaded["protein_embedding_paths"] == {"ESM3": "/tmp/esm3.pt"}
        assert loaded["prior_view_sources"] == ["ESM3"]
        assert loaded["biomarker_pathway_gene_set_path"] == "/tmp/toy_pathways.json"
        assert loaded["biomarker_min_pathway_size"] == 2

        parsed = config.parse_workflow_cli_args(
            [
                "--config",
                str(custom_yaml),
                "--epochs",
                "20",
                "--lr",
                "0.02",
                "--skip-analysis",
                "--gene-embedding",
                "LLM=/tmp/llm.pt",
            ]
        )
        assert parsed.config.epochs == 20
        assert parsed.config.lr == 0.02
        assert parsed.config.train_only is True
        assert parsed.config.analysis_only is False
        assert parsed.config.protein_embedding_paths == {"LLM": "/tmp/llm.pt"}
        assert parsed.config.prior_view_sources == ["LLM"]
        assert parsed.config.biomarker_pathway_gene_set_path == "/tmp/toy_pathways.json"
        assert parsed.config.biomarker_min_pathway_size == 2

        namespace = config.load_workflow_config_namespace(
            config_path=str(custom_yaml),
            overrides={"workflow": {"train_only": True}},
        )
        assert namespace.train_only is True
        assert namespace.skip_analysis is True


def test_config_rejects_training_epochs_below_five() -> None:
    try:
        config.load_workflow_config_dict(
            overrides={
                "workflow": {"input_h5ad": "/tmp/toy.h5ad"},
                "training": {"epochs": 4},
            }
        )
    except ValueError as error:
        assert "epochs >= 5" in str(error)
    else:
        raise AssertionError("expected epochs<5 to fail loudly when training is enabled")

    analysis_only = config.load_workflow_config_dict(
        overrides={
            "workflow": {
                "analysis_only": True,
                "run_dir": "/tmp/existing_run",
            },
            "training": {"epochs": 1},
        }
    )
    assert analysis_only["analysis_only"] is True
    assert analysis_only["epochs"] == 1


def test_legacy_flat_snapshot_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        flat_snapshot = temp_path / "workflow_config.yaml"
        flat_snapshot.write_text(
            "\n".join(
                [
                    "input_h5ad: /tmp/toy_data.h5ad",
                    "adata_path: /tmp/toy_data.h5ad",
                    "output_root: /tmp/out",
                    "run_dir: /tmp/out/training/train_runs/toy_run",
                    "analysis_output_dir: /tmp/out/analysis",
                    "patient_column: patient_id",
                    "celltype_column: celltype",
                    "label_column: label",
                    "ppi_path: /tmp/ppi.tsv",
                    "embedding_views:",
                    "  toy: /tmp/toy.pkl",
                    "binary_positive_labels:",
                    "  - 'yes'",
                    "binary_negative_labels:",
                    "  - 'no'",
                    "epochs: 12",
                    "lr: 0.01",
                    "k: 64",
                    "dataset: toy",
                    "adata_directory: /tmp",
                    "splits_directory: /tmp/out/splits",
                    "preselection_output_root: /tmp/out/preselection",
                    "experiment_root: /tmp/out/training",
                    "biomarker_run_dir: /tmp/out/training/train_runs/toy_run",
                    "biomarker_output_dir: /tmp/out/analysis",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = config.load_workflow_config_dict(config_path=str(flat_snapshot))
        assert loaded["input_h5ad"] == "/tmp/toy_data.h5ad"
        assert loaded["adata_path"] == "/tmp/toy_data.h5ad"
        assert loaded["output_root"] == "/tmp/out"
        assert loaded["run_dir"] == "/tmp/out/training/train_runs/toy_run"
        assert loaded["analysis_output_dir"] == "/tmp/out/analysis"
        assert loaded["patient_column"] == "patient_id"
        assert loaded["celltype_column"] == "celltype"
        assert loaded["label_column"] == "label"
        assert loaded["ppi_path"] == "/tmp/ppi.tsv"
        assert loaded["embedding_views"] == {"toy": "/tmp/toy.pkl"}
        assert loaded["binary_positive_labels"] == ["yes"]
        assert loaded["binary_negative_labels"] == ["no"]
        assert loaded["epochs"] == 12
        assert loaded["lr"] == 0.01
        assert loaded["k"] == 64
        assert loaded["dataset"] == "toy"
        assert loaded["splits_directory"] == "/tmp/out/splits"
        assert loaded["preselection_output_root"] == "/tmp/out/preselection"
        assert loaded["experiment_root"] == "/tmp/out/training"
        assert loaded["biomarker_run_dir"] == "/tmp/out/training/train_runs/toy_run"
        assert loaded["biomarker_output_dir"] == "/tmp/out/analysis"


def test_config_preserves_explicit_embedding_source_order() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        custom_yaml = temp_path / "ordered.yml"
        custom_yaml.write_text(
            "\n".join(
                [
                    "workflow:",
                    "  input_h5ad: /tmp/custom_data.h5ad",
                    "columns:",
                    "  patient: patient_id",
                    "  celltype: celltype",
                    "  label: response",
                    "resources:",
                    "  embedding_views:",
                    "    esm3: /tmp/esm3.pt",
                    "    node2vec: /tmp/node2vec.pt",
                    "prior_view_sources:",
                    "  - node2vec",
                    "  - esm3",
                    "protein_embedding_sources:",
                    "  - node2vec",
                    "  - esm3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        loaded = config.load_workflow_config_dict(config_path=str(custom_yaml))
        assert list(loaded["embedding_views"].keys()) == ["esm3", "node2vec"]
        assert loaded["prior_view_sources"] == ["node2vec", "esm3"]
        assert loaded["protein_embedding_sources"] == ["node2vec", "esm3"]


def test_config_preserves_top_level_biomarker_pathway_aliases() -> None:
    loaded = config.load_workflow_config_dict(
        overrides={
            "workflow": {
                "input_h5ad": "/tmp/custom_data.h5ad",
            },
            "columns": {
                "patient": "patient_id",
                "celltype": "celltype",
                "label": "label",
            },
            "pathway_gene_set_path": "/tmp/pathways.json",
            "biomarker_pathway_gene_set_path": "/tmp/biomarker_pathways.json",
            "biomarker_min_pathway_size": 3,
        }
    )
    assert loaded["pathway_gene_set_path"] == "/tmp/pathways.json"
    assert loaded["biomarker_pathway_gene_set_path"] == "/tmp/biomarker_pathways.json"
    assert loaded["biomarker_min_pathway_size"] == 3


def main() -> None:
    test_config_endpoints_exist()
    test_config_loading_and_precedence()
    test_config_rejects_training_epochs_below_five()
    test_legacy_flat_snapshot_round_trip()
    test_config_preserves_explicit_embedding_source_order()
    test_config_preserves_top_level_biomarker_pathway_aliases()
    print_success("config endpoints")


if __name__ == "__main__":
    main()
