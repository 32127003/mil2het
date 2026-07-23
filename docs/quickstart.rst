Quickstart
==========

The highest-level Python entry point is :func:`mil2het.pipeline.run_pipeline`.
Provide a YAML config, direct keyword arguments, or both. Direct keyword
arguments override matching YAML fields. The primary workflow trains and
evaluates one binary patient phenotype on held-out patients from a labelled
cohort.

.. code-block:: python

   from mil2het import run_pipeline

   result = run_pipeline(
       input_h5ad="data/cohort.h5ad",
       patient_column="patient_id",
       celltype_column="cell_type",
       label_column="case_control",
       ppi_path="data/ppi.tsv",
       embedding_views={
           "GPT": "data/GPT_embeddings.pkl",
           "node2vec": "data/Node2vec_embeddings.pkl",
           "ESM3": "data/ESM3_embeddings.pkl",
       },
       output_root="outputs/example",
       epochs=20,
       lr=0.0003,
       train_only=True,
   )

   print(result.phases_completed)
   print(result.run_dir)
   print(result.latest_run_path)

The timestamped ``run_dir`` contains ``best_checkpoint.pt``,
``final_metrics.json``, and the validation and test patient prediction CSVs.
``latest_run_path`` points to the stable manifest under ``output_root``.

Biomarker analysis is an optional follow-on. Use ``analysis_only=True`` with
``run_dir`` to replay it from an existing training run:

.. code-block:: python

   result = run_pipeline(
       config_path="workflow.yaml",
       analysis_only=True,
       run_dir="outputs/example/train_runs/run_001",
       pathway_path="data/pathways.json",
       gpu_index=-1,
   )

``gpu_index=-1`` forces CPU execution for training or analysis. Leaving
``gpu_index`` unset uses GPU index ``0`` for training-capable runs.

This release evaluates held-out patients from the supplied labelled cohort. It
does not expose a separate command for applying a checkpoint to a new,
unlabelled cohort.
