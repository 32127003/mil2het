"""

usage example:
python scripts/split_dataset.py --dataset asthma
python scripts/split_dataset.py --dataset asthma_ext


"""

import argparse
import sys
import os
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mil2het.config import parse_workflow_cli_args
from modules.utils import *
from configs.config import *


def rebalance_empty_folds(
    patient_folds: List[List[str]],
    minimum_fold_size: int,
) -> None:
    for fold_index in range(len(patient_folds)):
        while len(patient_folds[fold_index]) < minimum_fold_size:
            donor_index = int(np.argmax([len(values) for values in patient_folds]))
            if donor_index == fold_index:
                break
            if len(patient_folds[donor_index]) <= minimum_fold_size:
                break
            patient_folds[fold_index].append(patient_folds[donor_index].pop())


def rebalance_fold_sizes(
    patient_folds: List[List[str]],
    random_generator: np.random.Generator,
) -> None:
    # Balance patient counts so fold size gap is at most 1.
    # deterministic given the same input and random generator state
    max_iterations = 10000
    for _ in range(max_iterations):
        fold_sizes = [len(values) for values in patient_folds]
        largest_index = int(np.argmax(fold_sizes))
        smallest_index = int(np.argmin(fold_sizes))
        largest_size = int(fold_sizes[largest_index])
        smallest_size = int(fold_sizes[smallest_index])
        if largest_size - smallest_size <= 1:
            return
        if largest_size <= 1:
            return
        move_position = int(random_generator.integers(0, largest_size))
        moved_patient = patient_folds[largest_index].pop(move_position)
        patient_folds[smallest_index].append(moved_patient)


def fold_label_counters(
    patient_folds: Sequence[Sequence[str]],
    patient_to_label: Dict[str, str],
) -> List[Counter]:
    counters: List[Counter] = []
    for fold_patients in patient_folds:
        counter = Counter()
        for patient_id in fold_patients:
            counter[str(patient_to_label[str(patient_id)])] += 1
        counters.append(counter)
    return counters


def enforce_train_label_coverage(
    patient_folds: List[List[str]],
    patient_to_label: Dict[str, str],
    num_folds: int,
) -> None:
    # Try to keep every label present in train split for each fold-based split.
    # Train is defined as all folds except (test=fold_index, valid=fold_index+1).
    # For labels with only one patient this cannot be guaranteed for every split.
    if int(num_folds) < 3:
        return
  
    label_totals = Counter(str(label_value) for label_value in patient_to_label.values())
    enforce_labels = [label for label, count in sorted(label_totals.items()) if int(count) >= 2]
    if len(enforce_labels) == 0:
        return

    max_passes = max(1, num_folds * len(enforce_labels) * 8)
    for _ in range(max_passes):
        changed = False
        fold_label_counts = fold_label_counters(patient_folds, patient_to_label)
        for split_index in range(num_folds):
            excluded_folds = {split_index, (split_index + 1) % num_folds}
            train_folds = [index for index in range(num_folds) if index not in excluded_folds]
            if len(train_folds) == 0:
                continue

            for label_value in enforce_labels:
                train_label_count = int(
                    sum(int(fold_label_counts[index].get(label_value, 0)) for index in train_folds)
                )
                if train_label_count > 0:
                    continue

                donor_candidates = []
                for donor_index in sorted(excluded_folds):
                    donor_label_count = int(fold_label_counts[donor_index].get(label_value, 0))
                    if donor_label_count <= 0:
                        continue
                    donor_size = int(len(patient_folds[donor_index]))
                    if donor_size <= 1:
                        continue
                    donor_candidates.append((donor_size, donor_label_count, donor_index))
                if len(donor_candidates) == 0:
                    continue

                donor_candidates = sorted(donor_candidates, key=lambda item: (-item[0], -item[1], item[2]))
                donor_index = int(donor_candidates[0][2])
                receiver_index = min(train_folds, key=lambda index: (len(patient_folds[index]), index))
                movable = sorted(
                    patient_id
                    for patient_id in patient_folds[donor_index]
                    if str(patient_to_label[str(patient_id)]) == label_value
                )
                if len(movable) == 0:
                    continue

                moved_patient = movable[0]
                patient_folds[donor_index].remove(moved_patient)
                patient_folds[receiver_index].append(moved_patient)
                changed = True
                break
            if changed:
                break
        if not changed:
            return

def patient_to_folds(
    patient_to_label: Dict[str, str],
    num_folds: int,
    random_generator: np.random.Generator,
) -> List[List[str]]:
    if len(patient_to_label) < num_folds:
        raise ValueError(f"Need at least {num_folds} patients for {num_folds}-fold splitting, but {len(patient_to_label)} patients found.")

    label_to_patients: Dict[str, List[str]] = {}
    for pateint_id, label_value in patient_to_label.items():
        label_to_patients.setdefault(str(label_value), []).append(str(pateint_id))
    
    patient_folds: List[List[str]] = [[] for _ in range(num_folds)]
    for label_value in sorted(label_to_patients.keys()):
        patients = sorted(label_to_patients[label_value])
        permutation = random_generator.permutation(len(patients)).tolist()
        shuffled_patients = [patients[i] for i in permutation]
        start_offset = int(random_generator.integers(0, len(patients)))
        for local_rnk, patient_id in enumerate(shuffled_patients):
            fold_index = (start_offset + local_rnk) % num_folds
            patient_folds[fold_index].append(patient_id)

    rebalance_empty_folds(patient_folds=patient_folds, minimum_fold_size=1)
    rebalance_fold_sizes(patient_folds=patient_folds, random_generator=random_generator)
    enforce_train_label_coverage(
        patient_folds=patient_folds,
        patient_to_label=patient_to_label,
        num_folds=num_folds,
    )
    if any(len(values) == 0 for values in patient_folds):
        raise RuntimeError("Failed to assign at least one patient to every fold.")
    return [sorted(values) for values in patient_folds]
    


def patient_folds_to_cell_folds(
    all_indices: np.ndarray,
    patient_values: np.ndarray,
    patient_folds: List[List[str]],
) -> List[List[int]]:
    patient_to_cells: Dict[str, List[int]] = {}
    for cell_index, patient_id in zip(all_indices.tolist(), patient_values.tolist()):
        patient_to_cells.setdefault(str(patient_id), []).append(int(cell_index))

    cell_folds: List[List[int]] = [[] for _ in range(len(patient_folds))]
    for fold_index, fold_patients in enumerate(patient_folds):
        fold_cells: List[int] = []
        for patient_id in fold_patients:
            fold_cells.extend(patient_to_cells.get(patient_id, []))
        cell_folds[fold_index] = sorted(fold_cells)
    return cell_folds


def select_valid_fold_indices(
    patient_folds: Sequence[Sequence[str]],
    num_folds: int,
    target_valid_patients: int,
) -> List[int]:
    if int(num_folds) <= 1:
        raise ValueError(f"num_folds must be >= 2 to choose a validation fold. Received {num_folds}.")
    if int(target_valid_patients) <= 0:
        return [int((fold_index + 1) % num_folds) for fold_index in range(num_folds)]

    fold_sizes = [int(len(values)) for values in patient_folds]
    valid_usage = [0 for _ in range(num_folds)]
    selected_valid_folds: List[int] = []

    for test_fold in range(num_folds):
        candidates = [fold_index for fold_index in range(num_folds) if int(fold_index) != int(test_fold)]
        selected_fold = min(
            candidates,
            key=lambda fold_index: (
                abs(int(fold_sizes[fold_index]) - int(target_valid_patients)),
                int(fold_sizes[fold_index]) > int(target_valid_patients),
                int(fold_sizes[fold_index]),
                int(valid_usage[fold_index]),
                int(fold_index),
            ),
        )
        selected_valid_folds.append(int(selected_fold))
        valid_usage[selected_fold] += 1

    return selected_valid_folds


def patient_folds_to_sample_folds(
    patient_values: np.ndarray,
    sample_values: np.ndarray,
    patient_folds: List[List[str]],
) -> List[List[Tuple[str, str]]]:
    patient_to_samples: Dict[str, set] = {}
    for patient_id, sample_id in zip(patient_values.tolist(), sample_values.tolist()):
        patient_key = str(patient_id)
        sample_key = str(sample_id)
        patient_to_samples.setdefault(patient_key, set()).add(sample_key)

    sample_folds: List[List[Tuple[str, str]]] = [[] for _ in range(len(patient_folds))]
    for fold_index, fold_patients in enumerate(patient_folds):
        fold_samples: List[Tuple[str, str]] = []
        for patient_id in fold_patients:
            patient_key = str(patient_id)
            for sample_key in sorted(patient_to_samples.get(patient_key, set())):
                fold_samples.append((patient_key, sample_key))
        sample_folds[fold_index] = sorted(fold_samples, key=lambda item: (item[0], item[1]))
    return sample_folds


def sample_folds_to_cell_folds(
    all_indices: np.ndarray,
    patient_values: np.ndarray,
    sample_values: np.ndarray,
    sample_folds: List[List[Tuple[str, str]]],
) -> List[List[int]]:
    patient_sample_to_cells: Dict[Tuple[str, str], List[int]] = {}
    for cell_index, patient_id, sample_id in zip(
        all_indices.tolist(),
        patient_values.tolist(),
        sample_values.tolist(),
    ):
        patient_key = str(patient_id)
        sample_key = str(sample_id)
        patient_sample_to_cells.setdefault((patient_key, sample_key), []).append(int(cell_index))

    cell_folds: List[List[int]] = [[] for _ in range(len(sample_folds))]
    for fold_index, fold_samples in enumerate(sample_folds):
        fold_cells: List[int] = []
        for patient_key, sample_key in fold_samples:
            fold_cells.extend(patient_sample_to_cells.get((str(patient_key), str(sample_key)), []))
        cell_folds[fold_index] = sorted(fold_cells)
    return cell_folds

def patient_overlap_counts(
    train_indices: Sequence[int],
    valid_indices: Sequence[int],
    test_indices: Sequence[int],
    patient_values: np.ndarray,
) -> Tuple[int, int, int]:
    train_patients = {str(patient_values[int(index)]) for index in train_indices}
    valid_patients = {str(patient_values[int(index)]) for index in valid_indices}
    test_patients = {str(patient_values[int(index)]) for index in test_indices}
    return (
        int(len(train_patients & valid_patients)),
        int(len(train_patients & test_patients)),
        int(len(valid_patients & test_patients)),
    )


def normalize_binary_label_values(raw_values: Sequence[str] | str) -> List[str]:
    if isinstance(raw_values, str):
        return [str(raw_values)]
    return [str(value) for value in list(raw_values)]


def infer_dataset_name(config) -> str:
    dataset_value = str(getattr(config, "dataset", "") or "").strip()
    if dataset_value != "":
        return dataset_value

    adata_path = str(
        getattr(config, "adata_path", "") or getattr(config, "input_h5ad", "") or ""
    ).strip()
    if adata_path == "":
        raise ValueError("Either config.dataset or config.adata_path/input_h5ad must be set.")

    stem = Path(adata_path).stem
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    if stem == "":
        raise ValueError(f"Could not infer dataset name from adata path: {adata_path}")
    return stem


def resolve_adata_path(config) -> str:
    adata_path = str(
        getattr(config, "adata_path", "") or getattr(config, "input_h5ad", "") or ""
    ).strip()
    if adata_path != "":
        return adata_path

    dataset_name = infer_dataset_name(config)
    adata_directory = str(getattr(config, "adata_directory", "") or "").strip()
    if adata_directory == "":
        raise ValueError("Either config.adata_path/input_h5ad or config.adata_directory must be set.")
    return os.path.join(adata_directory, dataset_name, f"{dataset_name}_data.h5ad")


def resolve_split_output_directory(config) -> str:
    output_directory = str(
        getattr(config, "output_dir", "") or getattr(config, "splits_directory", "") or ""
    ).strip()
    if output_directory == "":
        raise ValueError("Either config.output_dir or config.splits_directory must be set.")
    return output_directory


def build_legacy_split_config(dataset_name: str):
    dataset_key = str(dataset_name).strip()
    if dataset_key == "asthma":
        config_dict = dict(asthma_split_configuration)
    elif dataset_key == "asthma_ext":
        config_dict = dict(asthma_ext_split_configuration)
    elif dataset_key == "vitiligo":
        config_dict = dict(vitiligo_split_configuration)
    elif dataset_key == "covid":
        config_dict = dict(covid_split_configuration)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_key}, custom split configuration is required for new datasets."
        )

    config_dict["dataset"] = dataset_key
    return dict2namespace(config_dict)


def build_and_save_folds(
    config,
    random_seed: int,
    num_folds: int,
    out_dir: str,
    binary_positive_labels: Sequence[str] | str,
    binary_negative_labels: Sequence[str] | str,
) -> None:
    if int(num_folds) < 3:
        raise ValueError(
            f"kfold split requires num_folds >= 3 because valid/test folds are separate. Received num_folds={num_folds}."
        )

    set_global_seed(random_seed)
    random_generator = np.random.default_rng(random_seed)

    dataset_name = infer_dataset_name(config)
    adata_path = resolve_adata_path(config)
    adata = sc.read_h5ad(adata_path)
    all_indices = np.arange(adata.n_obs, dtype=np.int64)
    patient_values = adata.obs[config.patient_column].astype(str).to_numpy()
    sample_values: Optional[np.ndarray] = None
    sample_column_value = getattr(config, "sample_column", None)
    sample_column_name = str(sample_column_value).strip() if sample_column_value is not None else ""
    sample_grouping_enabled = sample_column_name != ""
    if sample_grouping_enabled:
        if sample_column_name not in adata.obs.columns:
            raise ValueError(
                f"Missing sample column '{sample_column_name}' in adata.obs."
            )
        sample_values = adata.obs[sample_column_name].astype(str).to_numpy()
    unique_patients = sorted(set(patient_values))

    label_values_raw = adata.obs[config.label_column].astype(str).to_numpy()
    label_values_mapped, label_mapping_details = map_labels(
        label_values=label_values_raw,
        binary_positive_labels=normalize_binary_label_values(binary_positive_labels),
        binary_negative_labels=normalize_binary_label_values(binary_negative_labels),
    )
    label_values = np.asarray(label_values_mapped, dtype=object)
    patient_to_labels = {}
    for patient_id, label_value in zip(patient_values.tolist(), label_values.tolist()):
        patient_to_labels.setdefault(str(patient_id), []).append(str(label_value))

    patient_to_label = {}
    patient_to_label_conflicts = {}

    for patient_id, labels in patient_to_labels.items():
        unique_labels = sorted(set(labels))
        if len(unique_labels) != 1:
            patient_to_label_conflicts[patient_id] = unique_labels
            counts = Counter(labels)
            patient_to_label[patient_id] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        else:
            patient_to_label[patient_id] = unique_labels[0]
    print(f"Patient to label mapping: {patient_to_label}")    
    print(f"Patients with label conflicts: {patient_to_label_conflicts}")
    
    patient_folds = patient_to_folds(
        patient_to_label=patient_to_label,
        num_folds=num_folds,
        random_generator=random_generator,
    )
    enforce_422_for_asthma_ext = (
        dataset_name == "asthma_ext"
        and sample_grouping_enabled
        and int(num_folds) == 4
        and int(len(unique_patients)) == 8
    )
    if enforce_422_for_asthma_ext:
        rebalance_fold_sizes(patient_folds=patient_folds, random_generator=random_generator)
        patient_folds = [sorted(values) for values in patient_folds]
        if any(int(len(values)) != 2 for values in patient_folds):
            raise RuntimeError(
                "Failed to enforce patient-fold sizes for asthma_ext 8-patient 4-fold setup. "
                f"Observed fold patient counts={[len(values) for values in patient_folds]}"
            )

    if sample_grouping_enabled:
        if sample_values is None:
            raise RuntimeError("Internal error: sample_values must be prepared when sample grouping is enabled.")
        sample_folds = patient_folds_to_sample_folds(
            patient_values=patient_values,
            sample_values=sample_values,
            patient_folds=patient_folds,
        )
        unique_patient_samples = sorted(
            {
                (str(patient_id), str(sample_id))
                for patient_id, sample_id in zip(patient_values.tolist(), sample_values.tolist())
            }
        )
        assigned_patient_samples = [sample_key for fold_values in sample_folds for sample_key in fold_values]
        if int(len(assigned_patient_samples)) != int(len(set(assigned_patient_samples))):
            raise RuntimeError("Duplicate patient-sample assignment detected across folds.")
        if set(assigned_patient_samples) != set(unique_patient_samples):
            missing_keys = sorted(set(unique_patient_samples) - set(assigned_patient_samples))
            extra_keys = sorted(set(assigned_patient_samples) - set(unique_patient_samples))
            raise RuntimeError(
                "Patient-sample assignment mismatch in folds. "
                f"missing={missing_keys[:5]} extra={extra_keys[:5]}"
            )
        print(
            f"sample grouping: total patient-samples={len(unique_patient_samples)} "
            f"sample_fold_sizes={[len(values) for values in sample_folds]}",
            flush=True,
        )
        sample_group_total = int(len(unique_patient_samples))
        sample_group_fold_sizes = [int(len(values)) for values in sample_folds]
        base_folds = sample_folds_to_cell_folds(
            all_indices=all_indices,
            patient_values=patient_values,
            sample_values=sample_values,
            sample_folds=sample_folds,
        )
    else:
        sample_group_total = None
        sample_group_fold_sizes = None
        base_folds = patient_folds_to_cell_folds(
            all_indices=all_indices,
            patient_values=patient_values,
            patient_folds=patient_folds,
        )

    target_valid_patients = int(getattr(config, "num_valid_patients", 0))
    valid_folds = select_valid_fold_indices(
        patient_folds=patient_folds,
        num_folds=int(num_folds),
        target_valid_patients=target_valid_patients,
    )
    if int(target_valid_patients) > 0:
        print(
            f"validation fold strategy: target_valid_patients={target_valid_patients} "
            f"| patient_fold_sizes={[len(values) for values in patient_folds]} "
            f"| valid_fold_indices={valid_folds}",
            flush=True,
        )

    assigned = np.concatenate([np.asarray(fold,dtype=np.int64) for fold in base_folds], axis=0)
    print(f"Assigned {len(assigned)} out of {adata.n_obs} cells to folds based on patients."
          f" Unassigned cells: {adata.n_obs - len(assigned)}.")
    os.makedirs(out_dir, exist_ok=True)
    # save and print general description about the splits in txt file
    split_description_path = os.path.join(out_dir, "split_description.txt")
    sample_info_lines = ""
    if sample_group_total is not None:
        sample_info_lines = (
            f"Total patient-samples={sample_group_total}\n"
            f"Sample fold sizes={sample_group_fold_sizes}\n"
        )
    with open(split_description_path, "w") as f:
        f.write(
            f"Dataset={dataset_name}\n"
            f"Total cells={adata.n_obs}\n"
            f"Total patients={len(unique_patients)}\n"
            f"{sample_info_lines}"
            f"Fold sizes={[len(fold) for fold in base_folds]}\n"
            f"Patient to label mapping: {patient_to_label}\n"
            f"Patients with label conflicts: {patient_to_label_conflicts}\n"
            f"details: {label_mapping_details}\n"
        )
    print(
        f"Dataset={dataset_name}"
        f"{adata.n_obs} cells"
        f"fold_sizes={[len(fold) for fold in base_folds]}"
        f"details: {label_mapping_details}"
    )
    for fold_index in range(num_folds):
        test_fold = fold_index
        valid_fold = int(valid_folds[int(fold_index)])
        train_folds = [idx for idx in range(num_folds) if idx not in (test_fold, valid_fold)]

        train_indices = np.sort(
            np.concatenate([np.asarray(base_folds[idx], dtype=np.int64) for idx in train_folds], axis=0)
        ).astype(int).tolist()
        valid_indices = np.sort(np.asarray(base_folds[valid_fold], dtype=np.int64)).astype(int).tolist()
        test_indices = np.sort(np.asarray(base_folds[test_fold], dtype=np.int64)).astype(int).tolist()

        overlap_counts = patient_overlap_counts(
            train_indices=train_indices,
            valid_indices=valid_indices,
            test_indices=test_indices,
            patient_values=patient_values,
        )
        if any(count > 0 for count in overlap_counts):
            raise RuntimeError(
                "Patient-disjoint split creation failed: "
                f"train-valid={overlap_counts[0]}, train-test={overlap_counts[1]}, "
                f"valid-test={overlap_counts[2]}"
            )

        split_label_summary = ""

        train_patients = {str(patient_values[int(index)]) for index in train_indices}
        valid_patients = {str(patient_values[int(index)]) for index in valid_indices}
        test_patients = {str(patient_values[int(index)]) for index in test_indices}
        if enforce_422_for_asthma_ext:
            if not (
                int(len(train_patients)) == 4
                and int(len(valid_patients)) == 2
                and int(len(test_patients)) == 2
            ):
                raise RuntimeError(
                    "asthma_ext 8-patient 4-fold split must be 4/2/2 (train/valid/test). "
                    f"Observed {len(train_patients)}/{len(valid_patients)}/{len(test_patients)} "
                    f"at split {int(fold_index)}."
                )
        train_label_counter = Counter(patient_to_label[patient_id] for patient_id in train_patients)
        valid_label_counter = Counter(patient_to_label[patient_id] for patient_id in valid_patients)
        test_label_counter = Counter(patient_to_label[patient_id] for patient_id in test_patients)
        total_label_counter = train_label_counter + valid_label_counter + test_label_counter
        all_labels = sorted(set(total_label_counter.keys()))
        enforce_labels = [label for label in all_labels if int(total_label_counter.get(label, 0)) >= 2]
        missing_train = [label for label in enforce_labels if int(train_label_counter.get(label, 0)) == 0]
        split_label_summary = (
            " | patient_labels="
            f"train{dict(sorted(train_label_counter.items()))},"
            f"valid{dict(sorted(valid_label_counter.items()))},"
            f"test{dict(sorted(test_label_counter.items()))}"
        )
        if len(missing_train) > 0:
            split_label_summary += f" | warning_missing_train_labels={missing_train}"

        output_path = os.path.join(out_dir, f"{dataset_name}_idx_{fold_index}.pkl")
        save_pickle(output_path, [train_indices, valid_indices, test_indices])

        print(
            f"Saved {output_path} | train/valid/test={len(train_indices)}/{len(valid_indices)}/{len(test_indices)} "
            f"| patient_overlap(train-valid/train-test/valid-test)="
            f"{overlap_counts[0]}/{overlap_counts[1]}/{overlap_counts[2]}"
            f"{split_label_summary}"
        )


def run_split_generation(config) -> None:
    out_dir = resolve_split_output_directory(config)
    os.makedirs(out_dir, exist_ok=True)
    build_and_save_folds(
        config=config,
        random_seed=int(getattr(config, "seed", 42)),
        num_folds=int(getattr(config, "num_folds", 5)),
        out_dir=out_dir,
        binary_positive_labels=getattr(config, "binary_positive_labels", getattr(config, "binary_positive_label", ["1"])),
        binary_negative_labels=getattr(config, "binary_negative_labels", getattr(config, "binary_negative_label", ["0"])),
    )


def build_split_config_from_cli_args(args):
    if args.dataset is not None and args.adata_path is None and args.patient_column is None and args.label_column is None:
        return build_legacy_split_config(args.dataset)

    if args.adata_path is None:
        raise ValueError("--adata-path is required when not using a legacy --dataset preset.")
    if args.patient_column is None:
        raise ValueError("--patient-column is required when not using a legacy --dataset preset.")
    if args.label_column is None:
        raise ValueError("--label-column is required when not using a legacy --dataset preset.")
    if args.output_dir is None:
        raise ValueError("--output-dir is required when not using a legacy --dataset preset.")

    dataset_name = str(args.dataset_name or args.dataset or "").strip()
    if dataset_name == "":
        stem = Path(str(args.adata_path)).stem
        dataset_name = stem[: -len("_data")] if stem.endswith("_data") else stem

    config_dict = {
        "adata_path": str(args.adata_path),
        "dataset": str(dataset_name),
        "patient_column": str(args.patient_column),
        "label_column": str(args.label_column),
        "sample_column": args.sample_column,
        "output_dir": str(args.output_dir),
        "splits_directory": str(args.output_dir),
        "seed": int(args.seed if args.seed is not None else 42),
        "num_folds": int(args.num_folds if args.num_folds is not None else 5),
        "num_valid_patients": int(args.num_valid_patients if args.num_valid_patients is not None else 0),
        "binary_positive_labels": list(args.binary_positive_labels or ["1"]),
        "binary_negative_labels": list(args.binary_negative_labels or ["0"]),
    }
    return dict2namespace(config_dict)


def build_split_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create 5-fold dataset splits.")
    parser.add_argument("--dataset", type=str, required=False, help="Legacy dataset preset name.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Explicit dataset/output prefix.")
    parser.add_argument("--adata-path", type=str, default=None, help="Explicit .h5ad path.")
    parser.add_argument("--patient-column", type=str, default=None, help="Patient column in adata.obs.")
    parser.add_argument("--label-column", type=str, default=None, help="Label column in adata.obs.")
    parser.add_argument("--sample-column", type=str, default=None, help="Optional sample column in adata.obs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for split artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num-folds", type=int, default=None, help="Number of folds to create.")
    parser.add_argument("--num-valid-patients", type=int, default=None, help="Target validation-patient count.")
    parser.add_argument("--positive-label", dest="binary_positive_labels", action="append", default=None)
    parser.add_argument("--negative-label", dest="binary_negative_labels", action="append", default=None)
    return parser


def load_split_config_from_cli(argv: Optional[Sequence[str]] = None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv_list or "--adata" in argv_list:
        config = parse_workflow_cli_args(argv_list).config
        if str(getattr(config, "output_dir", "") or "").strip() == "":
            config.output_dir = str(getattr(config, "splits_directory", "") or "")
        return config

    parser = build_split_arg_parser()
    args = parser.parse_args(argv_list)
    return build_split_config_from_cli_args(args)


if __name__ == "__main__":
    config = load_split_config_from_cli()
    print(f"Configuration: {config}")
    run_split_generation(config)
