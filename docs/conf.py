# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import os
import subprocess
from importlib.metadata import version as get_version
from pathlib import Path

project = "convolve-rs"
copyright = "2026, Alec Thomson"
author = "Alec Thomson"


def _resolve_version() -> str:
    """Work out a meaningful version for the docs.

    The in-repo version is a ``0.0.0`` placeholder; the release workflow
    injects the real version from the git tag (see Cargo.toml). When the
    installed package still reports the placeholder, fall back to Read the
    Docs build metadata (tag builds) or ``git describe`` (branch/local
    builds).
    """
    installed = get_version("convolve-rs")
    if installed != "0.0.0":
        return installed
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
        tag = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "")
        if tag:
            return tag.removeprefix("v")
    try:
        described = subprocess.run(  # noqa: S603
            ["git", "describe", "--tags", "--always", "--dirty"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return installed
    return f"{described} (dev)" if described else installed


version = release = _resolve_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # markdown pages + executed notebooks (bundles myst_parser)
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "autoapi.extension",
    # Runs the `convolvers` CLI at build time so the help text in cli.md is
    # generated from the binary, never hand-copied. Needs a Rust toolchain.
    "sphinxcontrib.programoutput",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# Notebooks are re-executed on every build so the examples in the docs are
# guaranteed to run against the current code; a failing cell fails the build.
nb_execution_mode = "force"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# The public Python API lives in two places: the pure-Python wrapper
# (convolve_rs/__init__.py) and the pyo3-stub-gen generated stub for the
# compiled extension (convolve_rs/_convolve_rs/__init__.pyi). autoapi parses
# both statically — listing *.pyi first makes it win when both exist.
autoapi_type = "python"
autoapi_dirs = ["../convolve_rs"]
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_member_order = "groupwise"
autoapi_keep_files = False
autoapi_root = "autoapi"
autoapi_add_toctree_entry = True

# Napoleon settings (docstrings are Google style, generated from the Rust
# `///` comments for the native module)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/alecthomson/convolve-rs",
    "source_branch": "main",
    "source_directory": "docs/",
}
