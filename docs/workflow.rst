Workflow Overview
=================

``mil2het`` orchestrates four workflow phases:

1. ``split`` creates patient-aware train/validation/test fold artifacts.
2. ``preselection`` computes split-aware DEG and network propagation gene
   candidates.
3. ``training`` fits the MIL model and writes checkpoints, logs, and prediction
   artifacts into a run directory.
4. ``analysis`` reuses a run directory and configuration snapshot to produce
   biomarker outputs.

The top-level API returns :class:`mil2het.pipeline.PipelineResult`, which
records the effective configuration, completed phase names, major artifact
directories, the configuration snapshot path, and any phase-specific artifact
metadata.

Phase Control
-------------

``train_only`` skips biomarker analysis after training. It still runs split,
preselection, and training, then writes the training run snapshot used by later
analysis.

``analysis_only`` skips split, preselection, and training. It requires
``run_dir`` and uses either the provided config path, an existing
``workflow_config.yaml`` inside the run directory, or a generated analysis
snapshot under ``output_root``.

``run_dir`` identifies the training run used by analysis. In full or
train-only runs it is filled from the training artifacts. In analysis-only runs
it must point at a compatible existing run.

``output_root`` is the root for generated workflow outputs such as splits,
preselection products, materialized AnnData inputs, generated configs, and
default analysis output locations.

``pathway_path`` is passed to the biomarker analysis phase. Use the CLI
``--pathway-path`` flag or the Python ``pathway_path`` keyword when the
analysis phase should use a pathway gene-set file.
