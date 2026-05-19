from __future__ import annotations
import os
from pathlib import Path
import sys

"""Sphinx configuration for dapper documentation."""

# Ensure Sphinx can find src/ before installation
sys.path.insert(0, os.path.abspath("../src"))


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
SRC_DIR = REPO_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

project = "dapper"
author = "dapper contributors"

try:
    import dapper  # noqa: F401

    release = getattr(dapper, "__version__", "")
except Exception:
    release = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

nb_execution_mode = "off"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Mock optional deps so docs build in minimal environments.
autodoc_mock_imports = [
    "ee",
    "geemap",
    "contextily",
    "geopandas",
    "shapely",
    "rasterio",
    "pyproj",
    "rioxarray",
    "netCDF4",
    "fastparquet",
    "intake",
    "intake_esm",
    "gcsfs",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "dapper_custom.css",
]

python_use_unqualified_type_names = False  # for handling mutiple imports of same name


def _generate_surface_var_docs() -> None:
    """Generate docs/_generated/surface_variables_tables.rst."""
    script = DOCS_DIR / "_scripts" / "gen_surface_params_table.py"
    if script.exists():
        # run via same interpreter to ensure imports resolve
        import subprocess

        subprocess.check_call([sys.executable, str(script)])


def setup(app):
    app.connect("builder-inited", lambda app: _generate_surface_var_docs())
