from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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


def build_toy_adata() -> sc.AnnData:
    adata = sc.AnnData(X=np.random.randn(6, 4).astype(np.float32))
    adata.var_names = pd.Index(["G1", "G2", "G3", "G4"])
    adata.obs["patient_id"] = [f"P{i // 2}" for i in range(6)]
    adata.obs["celltype"] = ["T", "B", "T", "B", "T", "B"]
    adata.obs["label"] = ["1", "1", "0", "0", "1", "0"]
    return adata


def test_run_pipeline_full_train_only_and_analysis_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_root = temp_path / "outputs"
        ppi_path = temp_path / "toy_ppi.tsv"
        embedding_path = temp_path / "toy_view.pkl"
        pathway_path = temp_path / "toy_pathways.json"
        ppi_path.write_text("protein1\tprotein2\n", encoding="utf-8")
        embedding_path.write_bytes(b"placeholder")
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
            return {
                "output_dir": str(run_dir),
                "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
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
                embedding_views={"toy_view": str(embedding_path)},
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
            assert snapshot_payload["resources"]["embedding_views"] == {"toy_view": str(embedding_path)}
            assert snapshot_payload["training"]["epochs"] == 5
            assert snapshot_payload["run_dir"] == str(full_result.run_dir)

            call_order.clear()
            analysis_calls.clear()
            train_only_result = pipeline.run_pipeline(
                adata=build_toy_adata(),
                patient_column="patient_id",
                celltype_column="celltype",
                label_column="label",
                ppi_path=str(ppi_path),
                embedding_views={"toy_view": str(embedding_path)},
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
                embedding_views={"toy_view": str(embedding_path)},
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


def main() -> None:
    test_pipeline_endpoints_exist()
    test_run_pipeline_full_train_only_and_analysis_only()
    print_success("pipeline endpoints")


if __name__ == "__main__":
    main()
