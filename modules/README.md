# modules_copy

Current pipeline modules for scGOAT refactor.

## Core
- `CellEncoder.py`
  - GAT-only cell encoder.
  - Node features: `expression + global_zscore + regulation_onehot + celltype_onehot`.
  - Protein prior injection: additive to node latent (`protein_prior_embeddings`).
- `MultipleInstanceLearning.py`
  - MIL aggregators for patient-level pooling.
- `biomarker.py`
  - Step1/Step2 permutation scoring and final biomarker output.
- `analysis.py`
  - Cross-fold stability and optional DEG overlap evaluation.
- `integrate_protein_embeddings.py`
  - Offline integration/projection, including `concat_128_embeddings.pkl`.
- `preselection.py`
  - DEG/NP preselection (always split-train-only generation).

## Training
- `scripts/train2.py`
- `configs/asthma_train_config2.py`

## Preselection example
```bash
cd /data2/project/bin_jip/Biomarker
python modules_copy/preselection.py \
  --dataset asthma \
  --h5ad_path ./data/asthma/asthma_baseline_filtered.h5ad \
  --ppi_network ./data/ppi_network.tsv \
  --groupby Response2 \
  --groups 1,0 \
  --celltype_column CellType_minor \
  --splits_directory ./data/splits/asthma \
  --split_indices 0,1,2,3,4
```

## Train example
```bash
cd /data2/project/bin_jip/Biomarker
python scripts/train2.py --config configs/asthma_train_config2.py
```

## Biomarker example
```bash
cd /data2/project/bin_jip/Biomarker
python modules_copy/biomarker.py \
  --config configs/asthma_train_config2.py \
  --run_dir <train_run_dir>
```
