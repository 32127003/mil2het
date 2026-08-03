from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import yaml

from mil2het import pipeline, run_pipeline

from smoke_test_helpers import assert_module_endpoints, print_success

PIPELINE_ENDPOINTS = [
    "PipelineResult",
    "run_pipeline",
]


def test_pipeline_endpoints_exist() -> None:
    assert_module_endpoints(
        pipeline,
        PIPELINE_ENDPOINTS,
        module_label="mil2het.pipeline",
    )
    assert callable(run_pipeline)


def test_run_pipeline_rejects_missing_input_before_workflow_phases() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with pytest.raises(ValueError, match=r"workflow\.input_h5ad.*not found"):
            pipeline.run_pipeline(
                input_h5ad=temp_path / "missing.h5ad",
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=temp_path / "ppi.tsv",
                embedding_views={"esm": temp_path / "esm.pkl"},
                output_root=temp_path / "outputs",
                train_only=True,
                device="cpu",
            )


def build_toy_adata() -> sc.AnnData:
    adata = sc.AnnData(X=np.random.randn(6, 4).astype(np.float32))
    adata.var_names = pd.Index(["G1", "G2", "G3", "G4"])
    adata.obs["patient_id"] = [f"P{i // 2}" for i in range(6)]
    adata.obs["celltype"] = ["T", "B", "T", "B", "T", "B"]
    adata.obs["label"] = ["1", "1", "0", "0", "1", "0"]
    return adata


def test_run_pipeline_rejects_missing_required_adata_column() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ppi_path = temp_path / "ppi.tsv"
        embedding_path = temp_path / "esm.pkl"

        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        embedding_path.write_bytes(b"placeholder")

        cases = [
            ("patient_id", "columns.patient"),
            ("celltype", "columns.celltype"),
            ("label", "columns.label"),
        ]
        for column_name, field_name in cases:
            input_h5ad = temp_path / f"cohort-without-{column_name}.h5ad"
            adata = build_toy_adata()
            del adata.obs[column_name]
            adata.write_h5ad(input_h5ad)
            escaped_field_name = field_name.replace(".", r"\.")

            with pytest.raises(
                ValueError,
                match=rf"{escaped_field_name}.*{column_name}.*adata\.obs",
            ):
                pipeline.run_pipeline(
                    input_h5ad=input_h5ad,
                    patient_column="patient_id",
                    celltype_column="celltype",
                    label_column="label",
                    ppi_path=ppi_path,
                    embedding_views={"esm": embedding_path},
                    output_root=temp_path / "outputs",
                    num_folds=3,
                    train_only=True,
                    device="cpu",
                )


def test_run_pipeline_rejects_missing_training_resource() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_h5ad = temp_path / "cohort.h5ad"
        embedding_path = temp_path / "esm.pkl"

        build_toy_adata().write_h5ad(input_h5ad)
        embedding_path.write_bytes(b"placeholder")

        with pytest.raises(ValueError, match=r"resources\.ppi_path.*missing-ppi\.tsv"):
            pipeline.run_pipeline(
                input_h5ad=input_h5ad,
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=temp_path / "missing-ppi.tsv",
                embedding_views={"esm": embedding_path},
                output_root=temp_path / "outputs",
                num_folds=3,
                train_only=True,
                device="cpu",
            )

        ppi_path = temp_path / "ppi.tsv"
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"resources\.embedding_views\.esm.*missing-esm\.pkl"):
            pipeline.run_pipeline(
                input_h5ad=input_h5ad,
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=ppi_path,
                embedding_views={"esm": temp_path / "missing-esm.pkl"},
                output_root=temp_path / "outputs",
                num_folds=3,
                train_only=True,
                device="cpu",
            )

        with pytest.raises(ValueError, match=r"resources\.embedding_views.*at least one"):
            pipeline.run_pipeline(
                input_h5ad=input_h5ad,
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=ppi_path,
                embedding_views={},
                output_root=temp_path / "outputs",
                num_folds=3,
                train_only=True,
                device="cpu",
            )


def test_run_pipeline_rejects_unwritable_output_root_before_workflow_phases() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_h5ad = temp_path / "cohort.h5ad"
        ppi_path = temp_path / "ppi.tsv"
        embedding_path = temp_path / "esm.pkl"
        output_root = temp_path / "not-a-directory"

        build_toy_adata().write_h5ad(input_h5ad)
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        embedding_path.write_bytes(b"placeholder")
        output_root.write_text("occupied", encoding="utf-8")

        with pytest.raises(ValueError, match=r"workflow\.output_root.*not-a-directory"):
            pipeline.run_pipeline(
                input_h5ad=input_h5ad,
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=ppi_path,
                embedding_views={"esm": embedding_path},
                output_root=output_root,
                num_folds=3,
                train_only=True,
                device="cpu",
            )


def test_run_pipeline_rejects_invalid_training_split_selection() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_h5ad = temp_path / "cohort.h5ad"
        ppi_path = temp_path / "ppi.tsv"
        embedding_path = temp_path / "esm.pkl"

        build_toy_adata().write_h5ad(input_h5ad)
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        embedding_path.write_bytes(b"placeholder")

        for num_folds, split_number in [(2, 0), (3, -1), (3, 3)]:
            with pytest.raises(ValueError, match=r"workflow\.(num_folds|split_number)"):
                pipeline.run_pipeline(
                    input_h5ad=input_h5ad,
                    patient_column="patient_id",
                    celltype_column="celltype",
                    label_column="label",
                    ppi_path=ppi_path,
                    embedding_views={"esm": embedding_path},
                    output_root=temp_path / "outputs",
                    num_folds=num_folds,
                    split_number=split_number,
                    train_only=True,
                    device="cpu",
                )


def test_run_pipeline_full_train_only_and_analysis_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_root = temp_path / "outputs"
        ppi_path = temp_path / "toy_ppi.tsv"
        node2vec_path = temp_path / "node2vec.pkl"
        esm3_path = temp_path / "esm3.pkl"
        pathway_path = temp_path / "toy_pathways.json"
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        node2vec_path.write_bytes(b"placeholder")
        esm3_path.write_bytes(b"placeholder")
        pathway_path.write_text('{"toy_pathway": ["G1", "G2"]}\n', encoding="utf-8")

        call_order: list[str] = []
        split_calls: list[object] = []
        preselection_calls: list[object] = []
        train_calls: list[object] = []
        analysis_calls: list[tuple[str, str, str, str, int | None]] = []

        original_split = pipeline.split_dataset.run_split_generation
        original_preselection = pipeline.preselection.run_split_preselection
        original_train = pipeline.train.run_training_phase
        original_analysis = pipeline.biomarker.run_analysis_phase

        def fake_split(config) -> None:
            split_calls.append(config)
            call_order.append("split")
            Path(str(config.splits_directory)).mkdir(parents=True, exist_ok=True)
            (Path(str(config.splits_directory)) / "toy_idx_0.pkl").write_bytes(b"split")

        def fake_preselection(config) -> str:
            preselection_calls.append(config)
            call_order.append("preselection")
            out_dir = Path(str(config.preselection_output_root))
            (out_dir / "split_0" / "NP").mkdir(parents=True, exist_ok=True)
            (out_dir / "split_0" / "DEG").mkdir(parents=True, exist_ok=True)
            return str(out_dir)

        def fake_train(config, *, device=None):
            del device
            train_calls.append(config)
            call_order.append("training")
            run_dir = Path(str(config.experiment_root)) / "train_runs" / "toy_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "best_checkpoint.pt").write_bytes(b"checkpoint")
            (run_dir / "final_metrics.json").write_text('{"test": {}}\n', encoding="utf-8")
            (run_dir / "patient_predictions_val.csv").write_text(
                "patient_id,pred_prob\nP1,0.5\n",
                encoding="utf-8",
            )
            (run_dir / "patient_predictions_test.csv").write_text(
                "patient_id,pred_prob\nP2,0.7\n",
                encoding="utf-8",
            )
            return {
                "output_dir": str(run_dir),
                "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
                "final_metrics_path": str(run_dir / "final_metrics.json"),
                "patient_predictions_val_path": str(run_dir / "patient_predictions_val.csv"),
                "patient_predictions_test_path": str(run_dir / "patient_predictions_test.csv"),
            }

        def fake_analysis(run_dir: str, *, config_path: str = "", output_dir: str = "", pathway_path: str = "", gpu_index=None):
            analysis_calls.append((run_dir, config_path, output_dir, pathway_path, gpu_index))
            call_order.append("analysis")
            analysis_dir = Path(output_dir)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / "final_biomarker.tsv").write_text("gene\tscore\nG1\t1.0\n", encoding="utf-8")
            return {
                "run_dir": run_dir,
                "output_dir": str(analysis_dir),
                "dataset": "toy",
            }

        pipeline.split_dataset.run_split_generation = fake_split
        pipeline.preselection.run_split_preselection = fake_preselection
        pipeline.train.run_training_phase = fake_train
        pipeline.biomarker.run_analysis_phase = fake_analysis
        try:
            full_result = pipeline.run_pipeline(
                adata=build_toy_adata(),
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=str(ppi_path),
                embedding_views={"node2vec": str(node2vec_path), "esm3": str(esm3_path)},
                output_root=str(output_root),
                pathway_path=str(pathway_path),
                split_number=0,
                num_folds=3,
                seed=0,
                epochs=5,
                lr=0.01,
                k=2,
                train_only=False,
                analysis_only=False,
                device="cpu",
                gpu_index=-1,
            )
            assert full_result.phases_completed == ("split", "preselection", "training", "analysis")
            assert full_result.materialized_input_h5ad is not None
            assert Path(str(full_result.materialized_input_h5ad)).is_file()
            assert Path(full_result.config_snapshot_path).is_file()
            assert full_result.analysis_output_dir == str(output_root / "analysis")
            assert analysis_calls[0][4] == -1
            assert call_order == ["split", "preselection", "training", "analysis"]
            assert split_calls[0].input_h5ad.endswith(".h5ad")
            snapshot_payload = yaml.safe_load(Path(full_result.config_snapshot_path).read_text(encoding="utf-8"))
            assert snapshot_payload["workflow"]["input_h5ad"].endswith(".h5ad")
            assert snapshot_payload["workflow"]["output_root"] == str(output_root)
            assert snapshot_payload["columns"]["patient"] == "patient_id"
            assert list(snapshot_payload["resources"]["embedding_views"].keys()) == ["node2vec", "esm3"]
            assert snapshot_payload["resources"]["embedding_views"] == {
                "node2vec": str(node2vec_path),
                "esm3": str(esm3_path),
            }
            assert snapshot_payload["prior_view_sources"] == ["node2vec", "esm3"]
            assert snapshot_payload["training"]["epochs"] == 5
            assert snapshot_payload["run_dir"] == str(full_result.run_dir)
            latest_run_path = output_root / "latest_run.json"
            assert full_result.latest_run_path == str(latest_run_path.resolve())
            latest_run_payload = json.loads(latest_run_path.read_text(encoding="utf-8"))
            assert latest_run_payload["schema_version"] == 1
            assert latest_run_payload["status"] == "training_complete"
            assert latest_run_payload["run_dir"] == str(Path(full_result.run_dir).resolve())
            assert latest_run_payload["best_checkpoint_path"].endswith("/best_checkpoint.pt")
            assert latest_run_payload["final_metrics_path"].endswith("/final_metrics.json")
            assert latest_run_payload["patient_predictions_val_path"].endswith(
                "/patient_predictions_val.csv"
            )
            assert latest_run_payload["patient_predictions_test_path"].endswith(
                "/patient_predictions_test.csv"
            )
            assert latest_run_payload["config_snapshot_path"] == str(
                Path(full_result.config_snapshot_path).resolve()
            )
            assert latest_run_payload["relative_paths"]["run_dir"] == (
                "training/train_runs/toy_run"
            )
            assert not list(output_root.glob(".latest_run.json.*"))

            call_order.clear()
            analysis_calls.clear()
            train_only_result = pipeline.run_pipeline(
                adata=build_toy_adata(),
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=str(ppi_path),
                embedding_views={"node2vec": str(node2vec_path), "esm3": str(esm3_path)},
                output_root=str(output_root / "train_only"),
                pathway_path=str(pathway_path),
                train_only=True,
                device="cpu",
            )
            assert train_only_result.phases_completed == ("split", "preselection", "training")
            assert train_only_result.analysis_artifacts is None
            assert analysis_calls == []
            assert call_order == ["split", "preselection", "training"]

            call_order.clear()
            analysis_only_result = pipeline.run_pipeline(
                output_root=str(output_root / "analysis_only"),
                run_dir=str(full_result.run_dir),
                analysis_output_dir=str(output_root / "analysis_only" / "analysis"),
                pathway_path=str(pathway_path),
                embedding_views={"node2vec": str(node2vec_path), "esm3": str(esm3_path)},
                ppi_path=str(ppi_path),
                analysis_only=True,
                gpu_index=-1,
            )
            assert analysis_only_result.phases_completed == ("analysis",)
            assert analysis_only_result.training_artifacts is None
            assert analysis_only_result.run_dir == full_result.run_dir
            assert Path(analysis_only_result.config_snapshot_path).is_file()
            assert call_order == ["analysis"]
            assert len(split_calls) == 2
            assert len(preselection_calls) == 2
            assert len(train_calls) == 2
            assert len(analysis_calls) == 1
            assert analysis_calls[0][0] == full_result.run_dir
            assert analysis_calls[0][1] == full_result.config_snapshot_path
        finally:
            pipeline.split_dataset.run_split_generation = original_split
            pipeline.preselection.run_split_preselection = original_preselection
            pipeline.train.run_training_phase = original_train
            pipeline.biomarker.run_analysis_phase = original_analysis


def test_run_pipeline_config_file_preserves_biomarker_settings() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_root = temp_path / "outputs"
        input_h5ad = temp_path / "toy_data.h5ad"
        build_toy_adata().write_h5ad(input_h5ad)

        ppi_path = temp_path / "toy_ppi.tsv"
        node2vec_path = temp_path / "node2vec.pkl"
        esm3_path = temp_path / "esm3.pkl"
        pathway_path = temp_path / "toy_pathways.json"
        config_path = temp_path / "workflow_config.yaml"
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        node2vec_path.write_bytes(b"placeholder")
        esm3_path.write_bytes(b"placeholder")
        pathway_path.write_text('{"toy_pathway": ["G1", "G2"]}\n', encoding="utf-8")
        config_path.write_text(
            "\n".join(
                [
                    "workflow:",
                    f"  input_h5ad: {input_h5ad}",
                    f"  output_root: {output_root}",
                    "  num_folds: 3",
                    "columns:",
                    "  patient: patient_id",
                    "  celltype: celltype",
                    "  label: label",
                    "resources:",
                    f"  ppi_path: {ppi_path}",
                    "  embedding_views:",
                    f"    node2vec: {node2vec_path}",
                    f"    esm3: {esm3_path}",
                    "training:",
                    "  epochs: 5",
                    "  k: 2",
                    f"biomarker_pathway_gene_set_path: {pathway_path}",
                    "biomarker_min_pathway_size: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        call_order: list[str] = []
        analysis_calls: list[tuple[str, str, str, str, int | None]] = []
        original_split = pipeline.split_dataset.run_split_generation
        original_preselection = pipeline.preselection.run_split_preselection
        original_train = pipeline.train.run_training_phase
        original_analysis = pipeline.biomarker.run_analysis_phase

        def fake_split(config) -> None:
            call_order.append("split")
            splits_dir = Path(str(config.splits_directory))
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "toy_idx_0.pkl").write_bytes(b"split")

        def fake_preselection(config) -> str:
            call_order.append("preselection")
            out_dir = Path(str(config.preselection_output_root))
            (out_dir / "split_0" / "NP").mkdir(parents=True, exist_ok=True)
            (out_dir / "split_0" / "DEG").mkdir(parents=True, exist_ok=True)
            return str(out_dir)

        def fake_train(config, *, device=None):
            del device
            call_order.append("training")
            run_dir = Path(str(config.experiment_root)) / "train_runs" / "toy_config_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "best_checkpoint.pt").write_bytes(b"checkpoint")
            return {
                "output_dir": str(run_dir),
                "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
            }

        def fake_analysis(run_dir: str, *, config_path: str = "", output_dir: str = "", pathway_path: str = "", gpu_index=None):
            analysis_calls.append((run_dir, config_path, output_dir, pathway_path, gpu_index))
            call_order.append("analysis")
            snapshot_payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            assert snapshot_payload["biomarker_pathway_gene_set_path"] == str(pathway_path_on_disk)
            assert snapshot_payload["biomarker_min_pathway_size"] == 2
            assert pathway_path == ""
            analysis_dir = Path(output_dir)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            return {
                "run_dir": run_dir,
                "output_dir": str(analysis_dir),
                "dataset": "toy",
            }

        pathway_path_on_disk = pathway_path.resolve()

        pipeline.split_dataset.run_split_generation = fake_split
        pipeline.preselection.run_split_preselection = fake_preselection
        pipeline.train.run_training_phase = fake_train
        pipeline.biomarker.run_analysis_phase = fake_analysis
        try:
            result = pipeline.run_pipeline(
                config_path=str(config_path),
                device="cpu",
                gpu_index=-1,
            )
            assert result.phases_completed == ("split", "preselection", "training", "analysis")
            assert call_order == ["split", "preselection", "training", "analysis"]
            assert len(analysis_calls) == 1
            snapshot_payload = yaml.safe_load(Path(result.config_snapshot_path).read_text(encoding="utf-8"))
            assert snapshot_payload["biomarker_pathway_gene_set_path"] == str(pathway_path_on_disk)
            assert snapshot_payload["biomarker_min_pathway_size"] == 2
        finally:
            pipeline.split_dataset.run_split_generation = original_split
            pipeline.preselection.run_split_preselection = original_preselection
            pipeline.train.run_training_phase = original_train
            pipeline.biomarker.run_analysis_phase = original_analysis


def main() -> None:
    test_pipeline_endpoints_exist()
    test_run_pipeline_full_train_only_and_analysis_only()
    test_run_pipeline_config_file_preserves_biomarker_settings()
    print_success("pipeline endpoints")


if __name__ == "__main__":
    main()
