API Reference
=============

API Stability Tiers
-------------------

Public user API
   Import from ``mil2het`` for high-level workflow execution and model class
   access. The primary stable entry point is ``mil2het.run_pipeline``.

Advanced phase API
   Import from :mod:`mil2het.pipeline`, :mod:`mil2het.config`,
   :mod:`mil2het.cli`, :mod:`mil2het.split_dataset`,
   :mod:`mil2het.preselection`, :mod:`mil2het.train`, and
   :mod:`mil2het.biomarker` when building custom orchestration around a single
   workflow phase.

Implementation API
   Modules under ``scripts`` and ``modules`` are legacy implementation surfaces.
   They remain reachable through wrapper modules where listed in
   ``__all__``, but callers should prefer the package wrappers.

.. toctree::
   :maxdepth: 2

   facade
   workflow
   phases
   models
   internals
