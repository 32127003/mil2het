Workflow Overview
=================

``mil2het`` orchestrates four workflow phases:

1. ``split`` creates patient-aware train/validation/test fold artifacts.
2. ``preselection`` computes split-aware DEG and network propagation gene
   candidates.
3. ``training`` fits the MIL model and writes checkpoints, logs, and prediction
   artifacts into a run directory.
4. Optional ``analysis`` reuses a run directory and configuration snapshot to
   produce biomarker outputs.

The top-level API returns :class:`mil2het.pipeline.PipelineResult`, which
records the effective configuration, completed phase names, major artifact
directories, the configuration snapshot path, and any phase-specific artifact
metadata.

Phase Control
-------------

``train_only`` skips biomarker analysis after training. It still runs split,
preselection, training, validation, and test prediction, then writes the
training run snapshot and ``latest_run.json`` manifest. This is the recommended
mode for the Docker phenotype-training workflow.

``analysis_only`` skips split, preselection, and training. It requires
``run_dir`` and uses either the provided config path, an existing
``workflow_config.yaml`` inside the run directory, or a generated analysis
snapshot under ``output_root``.

``run_dir`` identifies the training run used by analysis. In full or
train-only runs it is filled from the training artifacts. In analysis-only runs
it must point at a compatible existing run.

``output_root`` is the root for generated workflow outputs such as splits,
preselection products, materialized AnnData inputs, generated configs, and
default analysis output locations. Timestamped training directories remain
authoritative; ``latest_run.json`` provides a stable locator for the latest
successful training run.

``pathway_path`` is passed to the biomarker analysis phase. Use the CLI
``--pathway-path`` flag or the Python ``pathway_path`` keyword when the
analysis phase should use a pathway gene-set file.

The Docker image executes this workflow as a terminating batch command. It
does not expose a server endpoint. The supported phenotype workflow evaluates
held-out patients from the labelled input cohort; separate unlabelled-cohort
inference is outside this release.
