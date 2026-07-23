mil2het
=======

``mil2het`` is an installable single-cell biomarker modeling toolkit. It wraps
the repository's split, preselection, training, and biomarker analysis phases
behind a Python package API and the ``mil2het`` command-line interface.

Start with :func:`mil2het.pipeline.run_pipeline` for end-to-end execution. Use
the phase wrapper modules only when a custom workflow needs direct access to
split, preselection, training, or analysis internals.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   workflow
   configuration
   cli
   outputs
   analysis_replay

.. toctree::
   :maxdepth: 2
   :caption: API

   api/index
