from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from mil2het import cli, pipeline, run_pipeline

from smoke_test_helpers import print_success


def build_toy_adata() -> sc.AnnData:
    adata = sc.AnnData(X=np.random.randn(6, 4).astype(np.float32))
    adata.var_names = pd.Index(["G1", "G2", "G3", "G4"])
    adata.obs["patient_id"] = [f"P{i // 2}" for i in range(6)]
    adata.obs["celltype"] = ["T", "B", "T", "B", "T", "B"]
    adata.obs["label"] = ["1", "1", "0", "0", "1", "0"]
    return adata


def build_example_files(temp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    adata_path = temp_path / "single_cell_data.h5ad"
    build_toy_adata().write_h5ad(adata_path)

    ppi_path = temp_path / "toy_ppi.tsv"
    ppi_path.write_text("protein1\tprotein2\nG1\tG2\n", encoding="utf-8")

    embedding_paths = {
        "llm_view": str(temp_path / "llm_view.pkl"),
        "ppi_view": str(temp_path / "ppi_view.pkl"),
    }
    for path_value in embedding_paths.values():
        Path(path_value).write_bytes(b"placeholder")
    return adata_path, ppi_path, embedding_paths


def patch_train_only_pipeline(temp_path: Path):
    call_order: list[str] = []
    original_split = pipeline.split_dataset.run_split_generation
    original_preselection = pipeline.preselection.run_split_preselection
    original_train = pipeline.train.run_training_phase

    def fake_split(config) -> None:
        call_order.append("split")
        splits_dir = Path(str(config.splits_directory))
        splits_dir.mkdir(parents=True, exist_ok=True)
        (splits_dir / "toy_idx_0.pkl").write_bytes(b"split")

    def fake_preselection(config) -> str:
        call_order.append("preselection")
        output_root = Path(str(config.preselection_output_root))
        (output_root / "split_0" / "NP").mkdir(parents=True, exist_ok=True)
        (output_root / "split_0" / "DEG").mkdir(parents=True, exist_ok=True)
        return str(output_root)

    def fake_train(config, *, device=None):
        del device
        call_order.append("training")
        run_dir = Path(str(config.experiment_root)) / "train_runs" / "todo_step1_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_checkpoint.pt").write_bytes(b"checkpoint")
        return {
            "output_dir": str(run_dir),
            "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
        }

    pipeline.split_dataset.run_split_generation = fake_split
    pipeline.preselection.run_split_preselection = fake_preselection
    pipeline.train.run_training_phase = fake_train

    def restore() -> None:
        pipeline.split_dataset.run_split_generation = original_split
        pipeline.preselection.run_split_preselection = original_preselection
        pipeline.train.run_training_phase = original_train

    return call_order, restore


def test_todo_cli_train_only_example() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        adata_path, ppi_path, embedding_paths = build_example_files(temp_path)
        call_order, restore = patch_train_only_pipeline(temp_path)
        try:
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                result = cli.run_cli(
                    [
                        str(adata_path),
                        "--patient",
                        "patient_id",
                        "--cell_type",
                        "celltype",
                        "--label",
                        "label",
                        "--ppi",
                        str(ppi_path),
                        "--add_gene_embedding",
                        f"llm_view={embedding_paths['llm_view']}",
                        "--add_gene_embedding",
                        f"ppi_view={embedding_paths['ppi_view']}",
                        "--train_only",
                    ]
                )
        finally:
            restore()

        assert result.phases_completed == ("split", "preselection", "training")
        assert call_order == ["split", "preselection", "training"]
        stdout_text = stdout_buffer.getvalue()
        assert "phases=split,preselection,training" in stdout_text
        assert result.input_h5ad == str(adata_path)


def test_todo_import_train_only_example() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _, ppi_path, embedding_paths = build_example_files(temp_path)
        call_order, restore = patch_train_only_pipeline(temp_path)
        try:
            result = run_pipeline(
                adata=build_toy_adata(),
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=str(ppi_path),
                embedding_views=embedding_paths,
                output_root=str(temp_path / "import_outputs"),
                train_only=True,
                device="cpu",
            )
        finally:
            restore()

        assert result.phases_completed == ("split", "preselection", "training")
        assert result.analysis_artifacts is None
        assert result.materialized_input_h5ad is not None
        assert Path(str(result.materialized_input_h5ad)).is_file()
        assert call_order == ["split", "preselection", "training"]


def main() -> None:
    test_todo_cli_train_only_example()
    test_todo_import_train_only_example()
    print_success("todo step-1 examples")


if __name__ == "__main__":
    main()
