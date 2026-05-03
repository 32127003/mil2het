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

``training_artifacts`` and ``analysis_artifacts`` contain additional
phase-specific paths when those phases run.
