Output Artifacts
================

The exact artifacts depend on phase selection, but
:class:`mil2het.pipeline.PipelineResult` always reports the main locations:

``splits_directory``
   Patient-aware split files used by downstream phases.

``preselection_output_root``
   DEG and network propagation outputs organized for the selected split.

``run_dir``
   Training outputs such as checkpoints, logs, patient predictions, and the
   ``workflow_config.yaml`` snapshot.

``analysis_output_dir``
   Biomarker analysis outputs generated from the selected run directory.

``config_snapshot_path``
   The effective workflow config used for training or analysis replay.

``latest_run_path``
   The stable ``latest_run.json`` manifest written under ``output_root`` after
   successful training.

``training_artifacts`` and ``analysis_artifacts`` contain additional
phase-specific paths when those phases run.

Phenotype Training Run
----------------------

A timestamped training run contains:

``best_checkpoint.pt``
   The selected model checkpoint.

``final_metrics.json``
   Final validation and test metrics.

``patient_predictions_val.csv`` and ``patient_predictions_test.csv``
   Patient-level held-out predictions. Columns are ``split``,
   ``patient_index``, ``patient_id``, ``true_label``, ``pred_prob``, and
   ``pred_label``. ``pred_prob`` is the probability of the configured positive
   phenotype, and labels are encoded as ``1`` and ``0``.

``workflow_config.yaml``
   The effective configuration snapshot.

With ``output_root=/outputs/mil2het`` in Docker, the layout is:

.. code-block:: text

   /outputs/mil2het/
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

The manifest's top-level paths are absolute container paths. When reading the
manifest on the host, resolve entries in ``relative_paths`` against the host
directory mounted as ``output_root``.
