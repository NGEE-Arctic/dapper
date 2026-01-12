# Building the docs locally

From the repo root:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt

# macOS/Linux
make -C docs html

# Windows (cmd)
cd docs
make.bat html
```

Outputs land in `docs/_build/html/`.

Notes:
- Notebook execution is disabled in the doc build (`myst_nb` reads existing outputs).
- The surface-variable tables are generated at build time from the in-package registry.
