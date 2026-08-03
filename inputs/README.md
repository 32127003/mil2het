# Docker phenotype-training inputs

Place the labelled cohort and configured training resources in this directory:

```text
inputs/
├── cohort.h5ad
├── config.yaml
├── ppi.tsv
├── GPT_embeddings.pkl
├── Node2vec_embeddings.pkl
└── ESM3_embeddings.pkl
```

`cohort.h5ad` must contain the configured patient, cell-type, and phenotype
columns in `adata.obs`. The positive label values are encoded as `1` and the
negative label values as `0`. Gene symbols are read from `adata.var_names`.

`ppi.tsv` must be tab-separated and contain `protein1` and `protein2` columns.

At least one embedding view is required. Each configured embedding file must
contain a serialized mapping from gene symbol to a one-dimensional embedding
vector. Pickle and PyTorch-serialized mappings are supported.

Copy `config.example.yaml` to the ignored local file `config.yaml`, then edit
the column names, positive and negative label values, and resource paths to
match the cohort. The example paths are container paths under `/inputs`.

The primary Docker workflow trains the binary phenotype model and produces
held-out validation and test predictions. Pathway resources are not required.
Biomarker analysis is an optional follow-on; when enabled, its pathway mapping
may be JSON, pickle, or PyTorch serialized.

The container runs as UID/GID 1000, so the host `outputs/` directory must be
writable by that user.

From the repository root, run:

```bash
cp inputs/config.example.yaml inputs/config.yaml
mkdir -p outputs

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
configured for Docker. For CPU execution, remove `--gpus all` and use
`--gpu -1`:

```bash
docker run --rm \
  --shm-size=8g \
  --mount type=bind,src="$PWD/inputs",dst=/inputs,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/outputs \
  mil2het:0.1.0 \
  --config /inputs/config.yaml \
  --gpu -1
```

On success, inspect `outputs/mil2het/latest_run.json`. It points to the
timestamped run directory, checkpoint, metrics, effective configuration, and
patient prediction files. Top-level paths in the manifest are absolute paths
inside the container. Host-side tools should resolve entries under
`relative_paths` against `outputs/mil2het/`.
