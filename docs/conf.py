"""Sphinx configuration for the local mil2het package documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

project = "mil2het"
copyright = "2026, Jeonguk Choi"
author = "Jeonguk Choi"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.utils",
    "torch.utils.data",
    "torch_scatter",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True
nitpick_ignore = [
    ("py:class", "AnnData"),
    ("py:class", "PathLike"),
    ("py:class", "anndata._core.anndata.AnnData"),
]
nitpick_ignore_regex = [
    ("py:class", r"torch(\..*)?"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}

html_theme = "shibuya"
html_static_path = []
html_title = "mil2het documentation"
