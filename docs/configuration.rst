Configuration
=============

The package config contains workflow paths, column names, label values,
resource paths, and training hyperparameters. For phenotype training, start
from ``inputs/config.example.yaml`` and use the following shape:

.. code-block:: yaml

   workflow:
     input_h5ad: "/inputs/cohort.h5ad"
     output_root: "/outputs/mil2het"
     num_folds: 5
     split_number: 0
     seed: 42
     train_only: true

   columns:
     patient: "patient_id"
     celltype: "cell_type"
     label: "phenotype"

   labels:
     positive: ["1"]
     negative: ["0"]

   resources:
     ppi_path: "/inputs/ppi.tsv"
     embedding_views:
       ESM3: "/inputs/ESM3_embeddings.pkl"

   training:
     epochs: 200
     lr: 0.0003
     k: 2000

Config Loading
--------------

:func:`mil2het.config.load_workflow_config_dict` merges the default config,
an optional user YAML file, and explicit overrides. Explicit Python arguments
or CLI flags have the highest precedence.

``embedding_views`` is a mapping of arbitrary view name to embedding file path.
The same normalized mapping feeds training and biomarker analysis so gene-view
alignment is checked consistently.

Runtime Fields
--------------

``analysis_only`` requires ``run_dir`` and does not accept a new AnnData input.
``train_only`` is the recommended phenotype-training mode. It runs split,
preselection, training, and held-out validation/test prediction, records a run
snapshot, and skips optional biomarker outputs.
``output_root`` controls generated workflow directories. ``pathway_path`` is
only needed when optional biomarker analysis is enabled.

The training preflight requires readable ``input_h5ad``, ``ppi_path``, and
embedding files; the configured patient, cell-type, and phenotype columns in
``adata.obs``; nonempty, disjoint positive and negative label sets; a valid
split index; and a writable ``output_root``.
