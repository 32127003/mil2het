# test_biomarker

Smoke-test project for the `scbiomarker` pip module.

## Refresh install

Run from this workspace:

```bash
conda create -n test python=3.10
conda activate test
pip uninstall -y scbiomarker
pip install git+ssh://git@github.com/32127003/scbiomarker.git@py_module
```

## Dependency notes

- `CellEncoder` requires `torch`.
- `MultipleInstanceLearning`, `biomarker`, and `train` require `torch_scatter` in addition to `torch`.
- `split_dataset`, `preselection`, `biomarker`, and `train` require `scanpy`.
- Wrapper modules intentionally raise installation guidance errors when these dependencies are missing.

### Manual install required for `torch` and `torch_scatter`

`scbiomarker` does not fully auto-install GPU stack dependencies. Install these manually in the `test` env:

```bash
# Install PyTorch first (choose the correct command for your system):
# https://pytorch.org/get-started/locally/

# Install torch_scatter with matching torch/cuda build
conda run -n test pip install --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# Example (if torch==2.10.0+cu128)
conda run -n test pip install --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# Verify
conda run -n test python -c "import torch, torch_scatter; print(torch.__version__, torch.version.cuda, torch_scatter.__version__)"
```

## Run endpoint smoke tests

Run all endpoint smoke tests:

```bash
conda run -n test python test_biomarker/run_all_endpoint_smoke_tests.py
```

Run module-specific endpoint smoke tests:

```bash
conda run -n test python test_biomarker/smoke_test_cell_encoder_endpoints.py
conda run -n test python test_biomarker/smoke_test_multiple_instance_learning_endpoints.py
conda run -n test python test_biomarker/smoke_test_prior_interface_find_endpoints.py
conda run -n test python test_biomarker/smoke_test_biomarker_endpoints.py
conda run -n test python test_biomarker/smoke_test_split_dataset_endpoints.py
conda run -n test python test_biomarker/smoke_test_preselection_endpoints.py
conda run -n test python test_biomarker/smoke_test_train_endpoints.py
```

Run the TODO-facing step-1 example smoke script:

```bash
conda run -n test python test_biomarker/smoke_test_todo_step1_examples.py
```

This script covers both public step-1 entry styles:

- CLI example:
  `scbiomarker single_cell_data.h5ad --patient patient_id --cell_type celltype --label label --ppi /path/to/ppi.tsv --add_gene_embedding llm_view=/path/to/llm.pkl --add_gene_embedding ppi_view=/path/to/ppi.pkl --train_only`
- Python import example:
  `from scbiomarker import run_pipeline`
  then call `run_pipeline(..., train_only=True)` with in-memory `AnnData` input.

Run the shell wrapper for the installed pip CLI:

```bash
bash test_biomarker/test_pip_cli_step1.sh
```

This checks that the installed `scbiomarker` command exists in the `test` env, that `scbiomarker --help` works, and then runs the TODO step-1 smoke example from the installed package environment.

## Top-level exposed modules and aliases

```python
from scbiomarker import (
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

### `scbiomarker.CellEncoder`

- `GraphAttentionLayer`
- `TransformerConvLayer`
- `GraphCellEncoder`
- `TransformerConvCellEncoder`
- `scatter_softmax`
- `infer_batch_chunk_size`

### `scbiomarker.MultipleInstanceLearning`

- `scatter_softmax_1d`
- `GatedAttentionMIL`
- `PatientMILAggregator`

### `scbiomarker.prior_interface_find`

- `MultiViewPriorInterfaceFIND`

### `scbiomarker.biomarker`

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

### `scbiomarker.split_dataset`

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

### `scbiomarker.preselection`

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

### `scbiomarker.train`

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
