Package Facade
==============

Use :mod:`mil2het` for stable package-level imports.

.. automodule:: mil2het
   :members: __all__

Primary Exports
---------------

.. list-table::
   :header-rows: 1

   * - Facade name
     - Canonical implementation
   * - ``run_pipeline``
     - :func:`mil2het.pipeline.run_pipeline`
   * - ``PipelineResult``
     - :class:`mil2het.pipeline.PipelineResult`
   * - ``GraphCellEncoder``
     - ``mil2het.CellEncoder.GraphCellEncoder``
   * - ``TransformerConvCellEncoder``
     - ``mil2het.CellEncoder.TransformerConvCellEncoder``
   * - ``GatedAttentionMIL``
     - ``mil2het.MultipleInstanceLearning.GatedAttentionMIL``
   * - ``PatientMILAggregator``
     - ``mil2het.MultipleInstanceLearning.PatientMILAggregator``
   * - ``MultiViewPriorInterfaceFIND``
     - ``mil2het.prior_interface_find.MultiViewPriorInterfaceFIND``

The model classes are lazily imported from wrapper modules so importing
``mil2het`` itself does not immediately require optional PyTorch dependencies.
