Analysis-Only Replay
====================

Analysis replay uses a completed training run as input. Provide
``analysis_only=True`` in Python or ``--analysis-only`` in the CLI, and pass
``run_dir``/``--run-dir``.

.. code-block:: console

   mil2het --analysis-only \
     --run-dir outputs/example/train_runs/run_001 \
     --analysis-output-dir outputs/example/analysis_replay \
     --pathway-path data/pathways.json \
     --gpu -1

The replay path does not accept a new ``.h5ad`` input. It uses a provided config
path when available, otherwise it looks for ``workflow_config.yaml`` in the run
directory. If no run snapshot exists, the pipeline writes an analysis snapshot
under ``output_root``.

Use ``--gpu -1`` when replay should run on CPU. This is useful for validation
hosts without CUDA or when comparing deterministic CPU analysis behavior.
