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

    Produces:
      - One "All variables" table with Required-level as a column.
      - One table per context (feature/module), discovered automatically
        from the 'contexts' lists; no manual RST needed.
    """
    from pathlib import Path
    from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS

    out_dir = Path(__file__).parent / "_generated"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_variables_tables.rst"

    def _row_for(name, spec):
        dims = spec.get("dims", "")
        units = spec.get("units", "")
        req_level = spec.get("required_level", "")
        ctxs = spec.get("contexts", []) or []
        ctx_str = ", ".join(ctxs)
        doc = spec.get("doc", "").replace("\n", " ")
        req_attr = spec.get("attrs", {}).get("requirement", "")
        if req_attr:
            doc = f"{doc} (Requirement: {req_attr})"
        return dims, units, req_level, ctx_str, doc

    def _write_table(f, title, items):
        if not items:
            return

        f.write(f"{title}\n")
        f.write(f"{'-' * len(title)}\n\n")

        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n")
        f.write("   :widths: 16 16 10 14 14 30\n\n")

        f.write("   * - **Variable**\n")
        f.write("     - **Dimensions**\n")
        f.write("     - **Units**\n")
        f.write("     - **Required level**\n")
        f.write("     - **Contexts**\n")
        f.write("     - **Description**\n\n")

        for name in sorted(items):
            spec = items[name]
            dims, units, req_level, ctx_str, doc = _row_for(name, spec)
            f.write(f"   * - ``{name}``\n")
            f.write(f"     - ``{dims}``\n")
            f.write(f"     - ``{units}``\n")
            f.write(f"     - ``{req_level}``\n")
            f.write(f"     - {ctx_str}\n")
            f.write(f"     - {doc}\n\n")

    # Get a stable view of all specs
    specs = dict(SURFACE_VAR_SPECS)

    # Collect all context tags that actually appear
    all_contexts = []
    for spec in specs.values():
        for c in spec.get("contexts", []) or []:
            all_contexts.append(c)
    unique_contexts = sorted(set(all_contexts))

    with out_path.open("w", encoding="utf-8") as f:
        # 1) Master table with everything
        _write_table(f, "All surface variables", specs)
        f.write("\n")

        # 2) Per-context tables (variables can appear in multiple contexts)
        if unique_contexts:
            f.write("Variables by context\n")
            f.write("--------------------\n\n")
            for ctx in unique_contexts:
                subset = {
                    name: spec
                    for name, spec in specs.items()
                    if ctx in (spec.get("contexts", []) or [])
                }
                if not subset:
                    continue
                title = f"Context: {ctx}"
                _write_table(f, title, subset)
                f.write("\n")

# Run the generator whenever Sphinx imports this conf.py
_generate_surface_var_docs()
