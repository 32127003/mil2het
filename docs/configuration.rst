Configuration
=============

The default package config is ``mil2het/default_config.yaml``. It contains
workflow paths, column names, label values, resource paths, and training
hyperparameters.

.. code-block:: yaml

   workflow:
     input_h5ad: ""
     output_root: "./outputs/mil2het"
     run_dir: ""
     analysis_output_dir: ""
     num_folds: 5
     split_number: 0
     seed: 42
     train_only: false
     analysis_only: false

   columns:
     patient: ""
     celltype: ""
     label: ""
     sample: null
     treatment: null

   labels:
     positive: ["1"]
     negative: ["0"]

   resources:
     ppi_path: ""
     embedding_views: {}

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
``train_only`` records a run snapshot but skips biomarker outputs.
``output_root`` controls generated workflow directories. ``pathway_path`` is
stored as both the public pathway override and the legacy pathway config key so
analysis internals receive the expected value.
