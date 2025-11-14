# -- Path setup --------------------------------------------------------------
from pathlib import Path
import os, sys

# Import your package from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project info ------------------------------------------------------------
project = "Dapper"
author = "Dapper contributors"
copyright = "exists"

# -- General config ----------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
]

# Don’t execute notebooks in docs builds
nb_execution_mode = "off"

# Autosummary: generate API pages automatically
autosummary_generate = True

# Google/Numpy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST: add anchors to headings so {ref} links resolve
myst_heading_anchors = 3

# Mock heavy/optional deps so CI can import your package
autodoc_mock_imports = [
    "numpy","pandas","xarray","numexpr","netCDF4","scipy","dask",
    "geopandas","shapely","rasterio","pyproj","ee"
]
# Can fix these with code changes
autodoc_mock_imports += [
    "fastparquet",
    "intake", "intake_esm", "xarray",
]

# Quiet noisy MyST cross-ref warnings from notebooks (optional)
suppress_warnings = ["myst.xref_missing"]


templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # toss scratch/tutorials we don’t want in nav
    "tutorials/met_data/deleteme*.ipynb",
]

# Intersphinx (optional)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
