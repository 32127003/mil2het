from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scanpy as sc

from scbiomarker import split_dataset

from smoke_test_helpers import assert_module_endpoints, print_success

SPLIT_DATASET_ENDPOINTS = [
    "rebalance_empty_folds",
    "rebalance_fold_sizes",
    "fold_label_counters",
    "enforce_train_label_coverage",
    "patient_to_folds",
    "patient_folds_to_cell_folds",
    "select_valid_fold_indices",
    "patient_folds_to_sample_folds",
    "sample_folds_to_cell_folds",
    "patient_overlap_counts",
    "normalize_binary_label_values",
    "infer_dataset_name",
    "resolve_adata_path",
    "resolve_split_output_directory",
    "build_legacy_split_config",
    "build_and_save_folds",
    "run_split_generation",
    "build_split_config_from_cli_args",
]


def test_split_dataset_endpoints_exist() -> None:
    assert_module_endpoints(
        split_dataset,
        SPLIT_DATASET_ENDPOINTS,
        module_label="scbiomarker.split_dataset",
    )


def test_split_dataset_utilities() -> None:
    rng = np.random.default_rng(0)

    patient_folds = [["p0", "p1"], [], ["p2", "p3", "p4"]]
    split_dataset.rebalance_empty_folds(patient_folds, minimum_fold_size=1)
    assert all(len(fold) >= 1 for fold in patient_folds)

    split_dataset.rebalance_fold_sizes(patient_folds, random_generator=rng)
    sizes = [len(fold) for fold in patient_folds]
    assert max(sizes) - min(sizes) <= 1

    patient_to_label = {f"p{i}": str(i % 2) for i in range(6)}
    folds = split_dataset.patient_to_folds(
        patient_to_label=patient_to_label,
        num_folds=3,
        random_generator=np.random.default_rng(1),
    )
    assert len(folds) == 3
    assert all(len(fold) >= 1 for fold in folds)

    counters = split_dataset.fold_label_counters(folds, patient_to_label)
    assert len(counters) == 3

    split_dataset.enforce_train_label_coverage(
        patient_folds=folds,
        patient_to_label=patient_to_label,
        num_folds=3,
    )

    all_indices = np.arange(12, dtype=np.int64)
    patient_values = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"])
    sample_values = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5"])

    cell_folds = split_dataset.patient_folds_to_cell_folds(
        all_indices=all_indices,
        patient_values=patient_values,
        patient_folds=folds,
    )
    assert len(cell_folds) == 3

    valid_fold_indices = split_dataset.select_valid_fold_indices(
        patient_folds=folds,
        num_folds=3,
        target_valid_patients=1,
    )
    assert len(valid_fold_indices) == 3

    sample_folds = split_dataset.patient_folds_to_sample_folds(
        patient_values=patient_values,
        sample_values=sample_values,
        patient_folds=folds,
    )
    assert len(sample_folds) == 3

    sample_cell_folds = split_dataset.sample_folds_to_cell_folds(
        all_indices=all_indices,
        patient_values=patient_values,
        sample_values=sample_values,
        sample_folds=sample_folds,
    )
    assert len(sample_cell_folds) == 3

    train_indices = sorted(cell_folds[0])
    valid_indices = sorted(cell_folds[1])
    test_indices = sorted(cell_folds[2])
    overlap_counts = split_dataset.patient_overlap_counts(
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        patient_values=patient_values,
    )
    assert overlap_counts == (0, 0, 0)


def test_split_dataset_build_and_save_folds() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        adata_root = temp_path / "adata"
        adata_root.mkdir(parents=True)

        num_cells = 18
        num_features = 4
        x = np.random.randn(num_cells, num_features).astype(np.float32)

        adata = sc.AnnData(X=x)
        patient_ids = [f"p{i // 3}" for i in range(num_cells)]
        labels = [str((i // 3) % 2) for i in range(num_cells)]
        adata.obs["patient_id"] = patient_ids
        adata.obs["label"] = labels
        adata.obs["sample_id"] = [f"s{i // 3}" for i in range(num_cells)]

        adata_path = adata_root / "toy_data.h5ad"
        adata.write_h5ad(adata_path)

        out_dir = temp_path / "splits"
        config = SimpleNamespace(
            adata_path=str(adata_path),
            dataset="",
            patient_column="patient_id",
            label_column="label",
            output_dir=str(out_dir),
            sample_column="sample_id",
            seed=0,
            num_folds=3,
            num_valid_patients=0,
            binary_positive_labels=["1"],
            binary_negative_labels=["0"],
        )

        assert split_dataset.infer_dataset_name(config) == "toy"
        assert split_dataset.resolve_adata_path(config) == str(adata_path)
        assert split_dataset.resolve_split_output_directory(config) == str(out_dir)
        split_dataset.run_split_generation(config)

        expected_files = [
            out_dir / "toy_idx_0.pkl",
            out_dir / "toy_idx_1.pkl",
            out_dir / "toy_idx_2.pkl",
            out_dir / "split_description.txt",
        ]
        for expected in expected_files:
            assert expected.is_file(), f"missing split artifact: {expected}"


def main() -> None:
    test_split_dataset_endpoints_exist()
    test_split_dataset_utilities()
    test_split_dataset_build_and_save_folds()
    print_success("split_dataset endpoints")


if __name__ == "__main__":
    main()
