Installation
============

Install the package from this repository when developing or validating the
wrapper-facing API:

.. code-block:: console

   uv sync --group dev
   uv run python -c "import mil2het; print(mil2het.__all__)"

The core package dependencies cover configuration, AnnData loading, Scanpy,
NumPy, pandas, SciPy, and scikit-learn. Training and biomarker execution also
require PyTorch and, for the MIL model utilities, ``torch-scatter`` built for
the same PyTorch/CUDA combination.

The package exposes the console command ``mil2het`` through
``mil2het.cli:main``.

Documentation Build
-------------------

Build the documentation from the repository root:

.. code-block:: console

   uv sync --group docs
   env -u VIRTUAL_ENV uv run sphinx-build -b html docs docs/_build/html
   env -u VIRTUAL_ENV uv run sphinx-build -b html -W --keep-going docs docs/_build/html

Generated HTML under ``docs/_build`` is a local artifact and should not be
committed.

Read the Docs
-------------

Read the Docs hosting is configured by ``.readthedocs.yaml`` in the repository
root. The hosted build uses Python 3.12, installs the ``docs`` dependency group
with native ``uv`` support, and builds Sphinx from ``docs/conf.py`` with
warnings treated as failures.

The hosted docs build does not install the CUDA-oriented training stack from
``requirements.txt``. Optional Torch imports are mocked in ``docs/conf.py`` so
autodoc can render the package wrappers on the Read the Docs CPU build image.
