Implementation Notes
====================

The package wrappers delegate to legacy modules in ``scripts`` and ``modules``.
Those implementation modules contain the detailed model, training,
preprocessing, and analysis logic, but they are not the preferred import
surface for new callers.

Documented wrapper modules intentionally list only names exported through their
``__all__`` values. Private helpers and broad legacy wildcard imports are kept
out of the public reference unless they become explicit wrapper exports.
