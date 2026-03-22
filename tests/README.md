# mil2het tests

Canonical in-repo pytest suite for the `mil2het` package.

## Local test setup

Create or reuse a development environment, then install the package plus pytest:

```bash
conda create -n scbiomarker-test python=3.10 -y
conda run -n scbiomarker-test python -m pip install --upgrade pip setuptools wheel pytest
conda run -n scbiomarker-test python -m pip install .
```

Torch-based tests still need manual installation of `torch` and `torch-scatter`:

```bash
# Install CPU-only torch for a chosen version.
conda run -n scbiomarker-test python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1

# Install matching CPU torch-scatter wheel.
conda run -n scbiomarker-test python -m pip install --no-build-isolation \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

## Run pytest

Run the full in-repo suite:

```bash
conda run -n scbiomarker-test pytest tests -q
```

Run the CPU-only suite used by CI:

```bash
conda run -n scbiomarker-test pytest tests --cpu-only -q
```

Run a single module test file:

```bash
conda run -n scbiomarker-test pytest tests/smoke_test_train_endpoints.py --cpu-only -q
```

For convenience, the legacy smoke runner now dispatches to pytest:

```bash
conda run -n scbiomarker-test python tests/run_all_endpoint_smoke_tests.py --cpu-only -q
```

## CPU-only vs GPU-marked tests

- `--cpu-only` skips tests marked with `@pytest.mark.gpu`.
- Tests that only verify CLI wiring or run torch on CPU are not considered GPU tests.
- The current suite is intended to be CPU-capable by default; the `gpu` marker exists so future CUDA-only tests can be filtered cleanly in CI.

## CI scope

- GitHub Actions treats `tests/` as the canonical package test suite.
- CI validates the built wheel, installs CPU-only `torch` plus matching CPU `torch-scatter`, and runs `pytest tests --cpu-only -q`.
- The separate workspace-level `test_biomarker` harness is not part of repo CI and remains a manual integration harness.

## Top-level exposed modules and aliases

```python
from mil2het import (
    CellEncoder,
    MultipleInstanceLearning,
    prior_interface_find,
    biomarker,
    split_dataset,
    preselection,
    train,
    GraphCellEncoder,
    TransformerConvCellEncoder,
    GatedAttentionMIL,
    PatientMILAggregator,
    MultiViewPriorInterfaceFIND,
)
```

## Exposed endpoints by module

### `mil2het.CellEncoder`

- `GraphAttentionLayer`
- `TransformerConvLayer`
- `GraphCellEncoder`
- `TransformerConvCellEncoder`
- `scatter_softmax`
- `infer_batch_chunk_size`

### `mil2het.MultipleInstanceLearning`

- `scatter_softmax_1d`
- `GatedAttentionMIL`
- `PatientMILAggregator`

### `mil2het.prior_interface_find`

- `MultiViewPriorInterfaceFIND`

### `mil2het.biomarker`

- `BagDefinition`
- `BagCache`
- `PatientClassifier`
- `load_config`
- `load_config_like_train_from_run_snapshot`
- `resolve_device`
- `resolve_cell_encoder_spec`
- `load_prior_embeddings_by_view`
- `load_np_max_genes`
- `build_protein_embedding_matrix`
- `build_celltype_regulation_onehot`
- `build_global_deg_zscore_vector`
- `sample_patient_bag_indices`
- `compute_binary_metrics`
- `aggregate_patient_probabilities`
- `stable_spearman`
- `jaccard_index`
- `forward_bag_from_expression`
- `forward_bag_from_embeddings`
- `sample_rows_with_replacement`
- `parse_args`
- `main`

### `mil2het.split_dataset`

- `rebalance_empty_folds`
- `rebalance_fold_sizes`
- `fold_label_counters`
- `enforce_train_label_coverage`
- `patient_to_folds`
- `patient_folds_to_cell_folds`
- `select_valid_fold_indices`
- `patient_folds_to_sample_folds`
- `sample_folds_to_cell_folds`
- `patient_overlap_counts`
- `build_and_save_folds`

### `mil2het.preselection`

- `RWRConfig`
- `read_gene_list`
- `write_gene_list`
- `deduplicate_preserve_order`
- `load_ppi_node_set`
- `remap_var_names_to_best_symbol_source`
- `load_split_train_indices`
- `collect_deg_rows`
- `select_deg`
- `select_deg_union_by_celltype`
- `compute_global_deg_zscore`
- `run_rwr`
- `np_scores`
- `preselection`

### `mil2het.train`

- `autocast_cuda`
- `ensure_cublas_workspace_config`
- `configure_runtime_backends`
- `PatientEmbeddingEMAMemory`
- `PatientBagDataset`
- `PatientClassifier`
- `collate_patient_bags`
- `compute_binary_metrics`
- `aggregate_patient_probabilities`
- `resolve_cell_encoder_spec`
- `build_protein_embedding_matrix`
- `load_prior_embeddings_by_view`
- `build_celltype_regulation_onehot`
- `build_global_deg_zscore_vector`
- `build_graph_cache`
- `build_split_records_cache`
- `build_experiment_directory`
- `build_optimizer`
- `run_epoch`
