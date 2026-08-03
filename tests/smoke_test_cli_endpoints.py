from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from mil2het import cli
from mil2het.pipeline import PipelineResult

from smoke_test_helpers import assert_module_endpoints, print_success

CLI_ENDPOINTS = [
    "build_cli_arg_parser",
    "main",
    "run_cli",
]


def test_cli_endpoints_exist() -> None:
    assert_module_endpoints(
        cli,
        CLI_ENDPOINTS,
        module_label="mil2het.cli",
    )


def build_fake_result(temp_path: Path) -> PipelineResult:
    run_dir = temp_path / "outputs" / "training" / "train_runs" / "toy_run"
    return PipelineResult(
        config={"output_root": str(temp_path / "outputs")},
        phases_completed=("split", "preselection", "training"),
        input_h5ad=str(temp_path / "toy_data.h5ad"),
        output_root=str(temp_path / "outputs"),
        splits_directory=str(temp_path / "outputs" / "splits"),
        preselection_output_root=str(temp_path / "outputs" / "preselection"),
        run_dir=str(run_dir),
        analysis_output_dir=str(temp_path / "outputs" / "analysis"),
        config_snapshot_path=str(temp_path / "outputs" / "training" / "train_runs" / "toy_run" / "workflow_config.yaml"),
        training_artifacts={
            "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
            "final_metrics_path": str(run_dir / "final_metrics.json"),
            "patient_predictions_val_path": str(run_dir / "patient_predictions_val.csv"),
            "patient_predictions_test_path": str(run_dir / "patient_predictions_test.csv"),
        },
        analysis_artifacts=None,
        latest_run_path=str(temp_path / "outputs" / "latest_run.json"),
    )


def test_cli_help_and_override_wiring() -> None:
    help_text = cli.build_cli_arg_parser().format_help()
    normalized_help_text = " ".join(help_text.split())
    assert "mil2het" in help_text
    assert "--config" in help_text
    assert "--gpu" in help_text
    assert "--analysis-only" in help_text
    assert "--add-gene-embedding" in help_text
    assert "--patient" in help_text
    assert "--cell-type" in help_text
    assert "--k" in help_text
    assert "Use -1 to force CPU execution." in normalized_help_text

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "toy_config.yaml"
        config_path.write_text("workflow:\n  output_root: /tmp/from-config\n", encoding="utf-8")

        captured_kwargs: dict[str, object] = {}
        original_run_pipeline = cli.run_pipeline

        def fake_run_pipeline(*args, **kwargs):
            captured_kwargs["args"] = args
            captured_kwargs["kwargs"] = kwargs
            return build_fake_result(temp_path)

        cli.run_pipeline = fake_run_pipeline
        try:
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                result = cli.run_cli(
                    [
                        "toy_data.h5ad",
                        "--config",
                        str(config_path),
                        "--patient",
                        "patient_id",
                        "--cell_type",
                        "celltype",
                        "--label",
                        "label",
                        "--ppi",
                        "/tmp/ppi.tsv",
                        "--add_gene_embedding",
                        "llm=/tmp/llm.pkl",
                        "--epochs",
                        "20",
                        "--lr",
                        "0.02",
                        "--k",
                        "64",
                        "--skip_analysis",
                        "--gpu",
                        "-1",
                    ]
                )
        finally:
            cli.run_pipeline = original_run_pipeline

        assert result.phases_completed == ("split", "preselection", "training")
        kwargs = captured_kwargs["kwargs"]
        assert kwargs["config_path"] == str(config_path)
        assert kwargs["gpu_index"] == -1
        overrides = kwargs["config_overrides"]
        assert overrides["workflow"]["input_h5ad"] == "toy_data.h5ad"
        assert overrides["workflow"]["train_only"] is True
        assert overrides["columns"]["patient"] == "patient_id"
        assert overrides["columns"]["celltype"] == "celltype"
        assert overrides["columns"]["label"] == "label"
        assert overrides["resources"]["ppi_path"] == "/tmp/ppi.tsv"
        assert overrides["resources"]["embedding_views"]["llm"] == "/tmp/llm.pkl"
        assert overrides["training"]["epochs"] == 20
        assert overrides["training"]["lr"] == 0.02
        assert overrides["training"]["k"] == 64

        stdout_text = stdout_buffer.getvalue()
        assert "phases=split,preselection,training" in stdout_text
        assert "best_checkpoint_path=" in stdout_text
        assert "final_metrics_path=" in stdout_text
        assert "patient_predictions_val_path=" in stdout_text
        assert "patient_predictions_test_path=" in stdout_text
        assert "latest_run_path=" in stdout_text


def test_cli_analysis_only_wiring() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        captured_kwargs: dict[str, object] = {}
        original_run_pipeline = cli.run_pipeline

        def fake_run_pipeline(*args, **kwargs):
            captured_kwargs["args"] = args
            captured_kwargs["kwargs"] = kwargs
            return build_fake_result(temp_path)

        cli.run_pipeline = fake_run_pipeline
        try:
            result = cli.run_cli(
                [
                    "--analysis-only",
                    "--run-dir",
                    "/tmp/run_dir",
                    "--analysis-output-dir",
                    "/tmp/analysis_dir",
                    "--pathway-path",
                    "/tmp/pathways.json",
                    "--gpu",
                    "-1",
                ]
            )
        finally:
            cli.run_pipeline = original_run_pipeline

        assert result.phases_completed == ("split", "preselection", "training")
        kwargs = captured_kwargs["kwargs"]
        assert kwargs["pathway_path"] == "/tmp/pathways.json"
        assert kwargs["gpu_index"] == -1
        overrides = kwargs["config_overrides"]
        assert overrides["workflow"]["analysis_only"] is True
        assert overrides["workflow"]["run_dir"] == "/tmp/run_dir"
        assert overrides["workflow"]["analysis_output_dir"] == "/tmp/analysis_dir"


def test_cli_defaults_training_gpu_to_zero() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "toy_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "workflow:",
                    "  output_root: /tmp/out",
                    "columns:",
                    "  patient: patient_id",
                    "  celltype: celltype",
                    "  label: label",
                    "resources:",
                    "  ppi_path: /tmp/ppi.tsv",
                    "  embedding_views:",
                    "    llm: /tmp/llm.pkl",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        captured_kwargs: dict[str, object] = {}
        original_run_pipeline = cli.run_pipeline

        def fake_run_pipeline(*args, **kwargs):
            captured_kwargs["args"] = args
            captured_kwargs["kwargs"] = kwargs
            return build_fake_result(temp_path)

        cli.run_pipeline = fake_run_pipeline
        try:
            result = cli.run_cli(
                [
                    "toy_data.h5ad",
                    "--config",
                    str(config_path),
                ]
            )
        finally:
            cli.run_pipeline = original_run_pipeline

        assert result.phases_completed == ("split", "preselection", "training")
        kwargs = captured_kwargs["kwargs"]
        assert kwargs["gpu_index"] == 0


def test_cli_main_returns_success_exit_code() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        original_run_cli = cli.run_cli

        def fake_run_cli(argv=None):
            del argv
            return build_fake_result(temp_path)

        cli.run_cli = fake_run_cli
        try:
            exit_code = cli.main([])
        finally:
            cli.run_cli = original_run_cli

        assert exit_code == 0


def main() -> None:
    test_cli_endpoints_exist()
    test_cli_help_and_override_wiring()
    test_cli_analysis_only_wiring()
    test_cli_defaults_training_gpu_to_zero()
    test_cli_main_returns_success_exit_code()
    print_success("cli endpoints")


if __name__ == "__main__":
    main()
