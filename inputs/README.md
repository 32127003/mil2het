# Docker inputs

Place the runtime resources for a MIL2Het run in this directory:

```text
inputs/
├── cohort.h5ad
├── config.example.yaml
├── ppi.tsv
├── GPT_embeddings.pkl
├── Node2vec_embeddings.pkl
├── ESM3_embeddings.pkl
└── reactome.json
```

`cohort.h5ad` must contain the configured patient, cell-type, and phenotype
columns in `adata.obs`. Gene symbols are read from `adata.var_names`.

`ppi.tsv` must be tab-separated and contain `protein1` and `protein2` columns.

Each embedding file must contain a serialized mapping from gene symbol to a
one-dimensional embedding vector. Pickle and PyTorch-serialized mappings are
supported.

`reactome.json` must map pathway names to lists of gene symbols:

```json
{
  "INTERFERON_SIGNALING": ["STAT1", "STAT2", "IRF9"],
  "ANTIGEN_PRESENTATION": ["HLA-DQA1", "HLA-DRB5", "B2M"]
}
```

Copy `config.example.yaml` to the ignored local file `config.yaml`, then edit
its column names and positive/negative labels to match the cohort. Paths
already use their locations inside the container.

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

For CPU execution, remove `--gpus all` and use `--gpu -1`.
