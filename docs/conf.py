"""Sphinx configuration for causal-automata (Read the Docs)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable without an editable install (local builds).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "causal-automata"
author = "Masakazu Yamagiwa"
copyright = f"{datetime.now(timezone.utc).year}, {author}"
release = "0.2.0"
version = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# Autodoc can import jax_core when the jax extra is installed (RTD / local docs).
autodoc_mock_imports: list[str] = []
