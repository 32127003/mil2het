Command Line
============

The ``mil2het`` command wraps :func:`mil2het.pipeline.run_pipeline`.

.. code-block:: console

   mil2het data/cohort.h5ad \
     --patient-column patient_id \
     --celltype-column cell_type \
     --label-column case_control \
     --ppi data/ppi.tsv \
     --gene-embedding GPT=data/GPT_embeddings.pkl \
     --gene-embedding node2vec=data/Node2vec_embeddings.pkl \
     --gene-embedding ESM3=data/ESM3_embeddings.pkl \
     --pathway-path data/pathways.json \
     --output-dir outputs/example \
     --epochs 20

Common Flags
------------

``--config`` loads an external YAML config. Direct flags override matching YAML
fields.

``--adata`` or the positional ``adata_input`` argument supplies the input
``.h5ad`` path. Use one form, not both.

``--train-only`` and ``--skip-analysis`` stop after training. ``--analysis-only``
starts from an existing ``--run-dir``.

``--gpu -1`` forces CPU execution for training and biomarker analysis. Any
non-negative value selects a GPU index for phases that use PyTorch.

``--pathway-path`` provides the biomarker analysis pathway gene-set file.
``--output-dir`` sets ``output_root``; ``--run-dir`` points to an existing
training run for analysis replay.
