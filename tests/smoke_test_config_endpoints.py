from __future__ import annotations

import tempfile
from pathlib import Path

from scbiomarker import config

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
        module_label="scbiomarker.config",
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

        namespace = config.load_workflow_config_namespace(
            config_path=str(custom_yaml),
            overrides={"workflow": {"train_only": True}},
        )
        assert namespace.train_only is True
        assert namespace.skip_analysis is True


def main() -> None:
    test_config_endpoints_exist()
    test_config_loading_and_precedence()
    print_success("config endpoints")


if __name__ == "__main__":
    main()
