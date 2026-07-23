Quickstart
==========

The highest-level Python entry point is :func:`mil2het.pipeline.run_pipeline`.
Provide a YAML config, direct keyword arguments, or both. Direct keyword
arguments override matching YAML fields.

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
       pathway_path="data/pathways.json",
       output_root="outputs/example",
       epochs=20,
       lr=0.0003,
   )

   print(result.phases_completed)
   print(result.run_dir)
   print(result.analysis_output_dir)

Use ``train_only=True`` to stop after checkpoint generation, or
``analysis_only=True`` with ``run_dir`` to replay biomarker analysis from an
existing training run.

.. code-block:: python

   result = run_pipeline(
       config_path="workflow.yaml",
       analysis_only=True,
       run_dir="outputs/example/train_runs/run_001",
       pathway_path="data/pathways.json",
       gpu_index=-1,
   )

``gpu_index=-1`` forces CPU execution for analysis. Leaving ``gpu_index`` unset
uses GPU index ``0`` for training-capable runs.
