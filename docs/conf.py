# docs/conf.py
import os
import sys
from datetime import date

# Make src/ importable (src layout)
sys.path.insert(0, os.path.abspath("../src"))

project = "dapper"
author = "NGEE Arctic"
copyright = f"{date.today().year}, {author}"

html_theme = "furo"
templates_path = ["_templates"]
exclude_patterns = [
    "tutorials/met_data/deleteme*.ipynb",
    "tutorials/olmt/README.md",
    "usage.rst",                  
    "api/modules.rst",            
]
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",  # <-- keep
]

# Don’t execute notebooks during doc build (CI-friendly)
nb_execution_mode = "off"

autosummary_generate = True
autosummary_imported_members = False   # don’t document re-exports
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

suppress_warnings = ["myst.header"]
