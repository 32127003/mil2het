# mil2het

[![CI](https://github.com/32127003/mil2het/actions/workflows/ci.yml/badge.svg)](https://github.com/32127003/mil2het/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%20%E2%80%93%202.11-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python toolkit for patient phenotype prediction and single-cell biomarker
discovery using multiple instance learning and multi-view gene embeddings.

## Overview

`mil2het` runs a four-stage workflow on single-cell RNA sequencing data:

1. **Split**: Creates patient-aware training, validation, and test partitions.
2. **Preselection**: Selects genes using differential expression and network propagation.
3. **Training**: Learns cell-level representations and aggregates them into
   patient-level binary phenotype predictions with multiple instance learning.
4. **Analysis (optional)**: Reuses a trained run to prioritize biomarkers.

The primary Docker workflow stops after training and writes validation and test
predictions. Biomarker analysis remains available as an optional follow-on.

## Installation

### From source (recommended)

```bash
git clone https://github.com/32127003/mil2het.git
cd mil2het
uv sync
```

The checked-in uv environment is constrained to PyTorch 2.11 and installs the
matching CUDA 13.0 `torch-scatter` wheel for development.

### From PyPI

```bash
pip install mil2het
```

`torch-scatter` is optional at installation time. It is required when using the
MIL, training, and biomarker features. Install the wheel that exactly matches
your PyTorch and CUDA versions, then install the `encoder` extra. For example:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install --only-binary=torch-scatter \
  -f https://data.pyg.org/whl/torch-2.11.0+cu130.html \
  "mil2het[encoder]"
```

Using `--only-binary=torch-scatter` makes an unsupported combination fail
clearly instead of attempting an unreliable source build. See the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for other supported PyTorch and CUDA combinations.

### Requirements

- Python >= 3.10
- PyTorch >= 2.3, < 3
- torch-scatter >= 2.1, < 3 (optional; required for MIL/training/biomarker features)

## Docker

The Docker image is a batch command, not a server. Each invocation runs one
`mil2het` workflow, writes its artifacts to a mounted output directory, and
then exits. It does not expose a network port.

Build the production image:

```bash
docker build -t mil2het:0.1.0 .
docker run --rm mil2het:0.1.0 --help
```

Prepare a local runtime configuration from the checked-in template:

```bash
cp inputs/config.example.yaml inputs/config.yaml
mkdir -p outputs
```

Build the test target to run the CPU-only test suite inside the build
environment:

```bash
docker build --target test -t mil2het:test .
```

The test target runs `pytest -q --cpu-only` during the build. A successful
build confirms that the CPU test suite passed for the source copied into that
image.

Inputs and resources should be mounted read-only under `/inputs`. Outputs must
be mounted read-write under `/outputs`. The container runs as UID/GID 1000, so
the host output directory must be writable by that user.

Run phenotype training on a GPU:

```bash
docker run --rm \
  --gpus all \
  --shm-size=8g \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/outputs \
  mil2het:0.1.0 \
  --config /inputs/config.yaml \
  --gpu 0
```

GPU execution requires both an NVIDIA driver and NVIDIA Container Toolkit
configured for Docker. `nvidia-smi` working on the host is not sufficient by
itself. After configuring the toolkit, verify the container sees the GPU:

```bash
docker run --rm --gpus all \
  --entrypoint /opt/venv/bin/python \
  mil2het:0.1.0 \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Run the same workflow on CPU by omitting `--gpus all` and selecting GPU index
`-1`:

```bash
docker run --rm \
  --shm-size=8g \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/outputs \
  mil2het:0.1.0 \
  --config /inputs/config.yaml \
  --gpu -1
```

The command is a terminating batch job: it does not expose a port or remain
running. This release trains and evaluates a requested binary phenotype on
held-out patients from the supplied labelled cohort. Applying a saved
checkpoint to a separate, unlabelled cohort is not yet a supported workflow.

Large `.h5ad` inputs, embeddings, checkpoints, and generated artifacts are
runtime mounts and are not copied into the image.

### Docker outputs

With the example `output_root: /outputs/mil2het`, a successful run produces:

```text
outputs/mil2het/
├── latest_run.json
├── splits/
├── preselection/
│   └── split_0/
└── training/
    └── train_runs/
        └── <timestamped-run>/
            ├── best_checkpoint.pt
            ├── final_metrics.json
            ├── patient_predictions_val.csv
            ├── patient_predictions_test.csv
            └── workflow_config.yaml
```

`patient_predictions_test.csv` contains `split`, `patient_index`,
`patient_id`, `true_label`, `pred_prob`, and `pred_label`. `pred_prob` is the
probability of the configured positive phenotype; `true_label` and
`pred_label` use encoded values `1` and `0`. `final_metrics.json` records
held-out validation and test metrics, and `best_checkpoint.pt` is the selected
model checkpoint.

`latest_run.json` is the stable locator for automation. Its top-level paths are
absolute paths inside the container (for example `/outputs/...`). Host-side
tools should resolve its `relative_paths` entries against the mounted host
root, `outputs/mil2het/`.

### Exporting and transferring the image

Export the exact built image and create a checksum:

```bash
docker save mil2het:0.1.0 | gzip > mil2het-0.1.0-linux-amd64.tar.gz
sha256sum mil2het-0.1.0-linux-amd64.tar.gz \
  > mil2het-0.1.0-linux-amd64.tar.gz.sha256
```

Send both files. The recipient can verify and load them with:

```bash
sha256sum -c mil2het-0.1.0-linux-amd64.tar.gz.sha256
gunzip -c mil2het-0.1.0-linux-amd64.tar.gz | docker load
docker run --rm mil2het:0.1.0 --help
```

The cohort and biological resources are runtime inputs and must be transferred
separately when the recipient does not already have them.

## Quick Start

### Command Line Interface

```bash
# Train and evaluate a binary phenotype
mil2het data.h5ad \
  --patient patient_id \
  --celltype cell_type \
  --label condition \
  --ppi pathway_network.tsv \
  --gene-embedding esm=./embeddings/esm_embeddings.pt \
  --train-only

# With multi-view gene embeddings
mil2het data.h5ad \
  --patient patient_id \
  --celltype cell_type \
  --label condition \
  --ppi pathway_network.tsv \
  --gene-embedding esm=./embeddings/esm_embeddings.pt \
  --gene-embedding ppi=./embeddings/ppi_embeddings.pt \
  --train-only

# Optional biomarker analysis of an existing training run
mil2het \
  --run-dir ./outputs/training_run \
  --pathway-path ./data/pathways.json \
  --analysis-only
```

### Python API

```python
from mil2het import run_pipeline, PipelineResult

# Train and evaluate a binary phenotype
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
    train_only=True,
)

# Access outputs
print(f"Training run: {result.run_dir}")
print(f"Latest-run manifest: {result.latest_run_path}")
print(f"Phases completed: {result.phases_completed}")
```

### Using AnnData directly

```python
import scanpy as sc
from mil2het import run_pipeline

adata = sc.read_h5ad("data.h5ad")

result = run_pipeline(
    adata=adata,
    patient_column="patient_id",
    celltype_column="cell_type",
    label_column="condition",
    ppi_path="pathway_network.tsv",
    embedding_views={
        "esm": "./embeddings/esm_embeddings.pt",
    },
    train_only=True,
)
```

### Configuration

Use YAML configuration files for reproducible workflows:

```yaml
# config.yaml
workflow:
  input_h5ad: "data.h5ad"
  output_root: "./outputs/mil2het"
  num_folds: 5
  split_number: 0
  seed: 42
  train_only: true

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
mil2het --config config.yaml
```

## Key Components

### Cell Encoders

Graph neural network encoders for learning cell representations:

```python
from mil2het import GraphCellEncoder, TransformerConvCellEncoder

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
from mil2het import GatedAttentionMIL, PatientMILAggregator

mil = GatedAttentionMIL(
    input_dim=256,
    hidden_dim=128,
    num_classes=2,
)
```

### Multi-View Prior Integration

```python
from mil2het import MultiViewPriorInterfaceFIND

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
| `analysis/` | Optional biomarker candidates and pathway enrichment results |

## Development

### Running Tests

```bash
# Install development dependencies
uv sync

# Run CPU-only tests
uv run pytest tests --cpu-only

# Run all tests (requires GPU)
uv run pytest tests
```

### Building

```bash
pip install build
python -m build
```

### Documentation

Build the Sphinx documentation locally with the docs dependency group:

```bash
uv sync --group docs
env -u VIRTUAL_ENV uv run sphinx-build -E -b html -W --keep-going docs docs/_build/html
```

The generated HTML is written to `docs/_build/html/` and is intentionally not
tracked by Git.

### Read the Docs

Hosted documentation is configured by `.readthedocs.yaml` at the repository
root. Read the Docs should import this repository, build the branch that
contains the config file, install dependencies with `uv sync --group docs`, and
use `docs/conf.py` as the Sphinx configuration.

The hosted build intentionally does not install the CUDA/PyTorch stack from
`requirements.txt`. `docs/conf.py` mocks optional Torch imports for autodoc so
the API reference can build on Read the Docs' CPU build image. If model autodoc
is expanded to require real Torch introspection, update the docs dependency
strategy deliberately rather than adding the full training stack by default.

## Citation

If you use mil2het in your research, please cite:

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
