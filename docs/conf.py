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


# ------------------------------------------------------------------
# Auto-generate surface variable tables from SURFACE_VAR_SPECS
# ------------------------------------------------------------------

def _generate_surface_var_docs():
    """
    Write docs/_generated/surface_variables_tables.rst from
    dapper.surf.surface_var_specs.SURFACE_VAR_SPECS.

    Groups variables by required_level into separate tables:
      * Required variables
      * Conditional variables
      * Optional variables
    """
    # Import here so docs build doesn't fail if dapper isn't importable
    from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS

    out_dir = Path(__file__).parent / "_generated"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_variables_tables.rst"

    def _write_group(title, items, f):
        if not items:
            return

        f.write(f"{title}\n")
        f.write(f"{'-' * len(title)}\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n")
        f.write("   :widths: 20 20 60\n\n")

        f.write("   * - **Variable**\n")
        f.write("     - **Dimensions**\n")
        f.write("     - **Description**\n\n")

        for name in sorted(items):
            spec = items[name]
            dims = spec.get("dims", "")
            doc = spec.get("doc", "").replace("\n", " ")
            # Optional extra prose about when it is required
            req_attr = spec.get("attrs", {}).get("requirement", "")
            if req_attr:
                doc = f"{doc} (Requirement: {req_attr})"

            f.write(f"   * - ``{name}``\n")
            f.write(f"     - ``{dims}``\n")
            f.write(f"     - {doc}\n\n")

    # Partition by required_level
    required = {
        k: v
        for k, v in SURFACE_VAR_SPECS.items()
        if v.get("required_level", "").lower() == "required"
    }
    conditional = {
        k: v
        for k, v in SURFACE_VAR_SPECS.items()
        if v.get("required_level", "").lower() == "conditional"
    }
    optional = {
        k: v
        for k, v in SURFACE_VAR_SPECS.items()
        if v.get("required_level", "").lower()
        not in ("required", "conditional")
    }

    with out_path.open("w", encoding="utf-8") as f:
        _write_group("Required variables", required, f)
        f.write("\n")
        _write_group("Conditional variables", conditional, f)
        f.write("\n")
        _write_group("Optional variables", optional, f)

# Run the generator when Sphinx imports this conf.py
_generate_surface_var_docs()
