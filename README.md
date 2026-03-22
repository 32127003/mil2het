# scbiomarker

[![CI](https://github.com/user/scbiomarker/actions/workflows/ci.yml/badge.svg)](https://github.com/user/scbiomarker/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%20%7C%202.5%20%7C%202.7-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python toolkit for single-cell biomarker discovery using multiple instance learning and multi-view gene embeddings.

## Overview

`scbiomarker` implements a two-phase workflow for identifying biomarkers from single-cell RNA sequencing data:

1. **Training Phase**: Learns cell-level representations using graph neural networks and aggregates patient-level predictions via multiple instance learning (MIL)
2. **Analysis Phase**: Identifies candidate biomarkers through network propagation and multi-view prior integration

The module supports multi-view gene embeddings (e.g., protein language model embeddings, PPI-derived embeddings) to incorporate prior biological knowledge into the biomarker discovery process.

## Installation

### From PyPI (recommended)

```bash
pip install scbiomarker
```

### With encoder support

For GPU-accelerated cell encoding and training:

```bash
pip install "scbiomarker[encoder]"
pip install torch  # install PyTorch for your system
pip install torch-scatter  # install matching your PyTorch version
```

### From source

```bash
git clone https://github.com/user/scbiomarker.git
cd scbiomarker
pip install -e ".[encoder]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 1.12 (optional, for encoder/training)
- torch-scatter >= 2.1 (optional, for encoder/training)

## Quick Start

### Command Line Interface

```bash
# Basic usage with required columns
scbiomarker data.h5ad \
  --patient patient_id \
  --celltype cell_type \
  --label condition \
  --ppi pathway_network.tsv

# With multi-view gene embeddings
scbiomarker data.h5ad \
  --patient patient_id \
  --celltype cell_type \
  --label condition \
  --ppi pathway_network.tsv \
  --gene-embedding esm=./embeddings/esm_embeddings.pt \
  --gene-embedding ppi=./embeddings/ppi_embeddings.pt

# Training only (skip analysis)
scbiomarker data.h5ad \
  --patient patient_id \
  --celltype cell_type \
  --label condition \
  --train-only

# Analysis only (requires pretrained checkpoint)
scbiomarker --run-dir ./outputs/training_run --analysis-only
```

### Python API

```python
from scbiomarker import run_pipeline, PipelineResult

# Run full pipeline
result: PipelineResult = run_pipeline(
    input_h5ad="data.h5ad",
    patient_column="patient_id",
    celltype_column="cell_type",
    label_column="condition",
    ppi_path="pathway_network.tsv",
    embedding_views={
        "esm": "./embeddings/esm_embeddings.pt",
        "ppi": "./embeddings/ppi_embeddings.pt",
    },
    epochs=200,
    lr=0.0003,
)

# Access outputs
print(f"Training run: {result.run_dir}")
print(f"Analysis output: {result.analysis_output_dir}")
print(f"Phases completed: {result.phases_completed}")
```

### Using AnnData directly

```python
import scanpy as sc
from scbiomarker import run_pipeline

adata = sc.read_h5ad("data.h5ad")

result = run_pipeline(
    adata=adata,
    patient_column="patient_id",
    celltype_column="cell_type",
    label_column="condition",
    ppi_path="pathway_network.tsv",
)
```

### Configuration

Use YAML configuration files for reproducible workflows:

```yaml
# config.yaml
workflow:
  input_h5ad: "data.h5ad"
  output_root: "./outputs/scbiomarker"
  num_folds: 5
  seed: 42
  train_only: false

columns:
  patient: "patient_id"
  celltype: "cell_type"
  label: "condition"

labels:
  positive: ["disease", "case"]
  negative: ["control", "healthy"]

resources:
  ppi_path: "pathway_network.tsv"
  embedding_views:
    esm: "./embeddings/esm_embeddings.pt"
    ppi: "./embeddings/ppi_embeddings.pt"

training:
  epochs: 200
  lr: 0.0003
  k: 2000  # number of preselected genes
```

```bash
scbiomarker --config config.yaml
```

## Key Components

### Cell Encoders

Graph neural network encoders for learning cell representations:

```python
from scbiomarker import GraphCellEncoder, TransformerConvCellEncoder

# Graph attention-based encoder
encoder = GraphCellEncoder(
    input_dim=embedding_dim,
    hidden_dim=256,
    num_layers=3,
)

# Transformer-based encoder
encoder = TransformerConvCellEncoder(
    input_dim=embedding_dim,
    hidden_dim=256,
    heads=4,
)
```

### Multiple Instance Learning

Patient-level aggregation using gated attention MIL:

```python
from scbiomarker import GatedAttentionMIL, PatientMILAggregator

mil = GatedAttentionMIL(
    input_dim=256,
    hidden_dim=128,
    num_classes=2,
)
```

### Multi-View Prior Integration

```python
from scbiomarker import MultiViewPriorInterfaceFIND

prior = MultiViewPriorInterfaceFIND(
    embedding_views={"esm": emb_esm, "ppi": emb_ppi},
    hidden_dim=256,
)
```

## Pipeline Outputs

| Directory | Contents |
|-----------|----------|
| `splits/` | Cross-validation fold splits |
| `preselection/` | Preselected gene sets per fold |
| `training/` | Model checkpoints and training logs |
| `analysis/` | Biomarker candidates and pathway enrichment results |

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[encoder]" pytest

# Run CPU-only tests
pytest tests --cpu-only

# Run all tests (requires GPU)
pytest tests
```

### Building

```bash
pip install build
python -m build
```

## Citation

If you use scbiomarker in your research, please cite:

```bibtex
@inproceedings{ju2026scbiomarker,
  title={scbiomarker: Multi-View Biomarker Discovery from Single-Cell Data},
  author={Ju, et al.},
  booktitle={ECCB},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.