from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any


def apply_append_attrs(ds: Any, append_attrs: dict | None):
    """Update xarray Dataset global attrs in a NetCDF-safe way."""
    if not append_attrs:
        return ds

    for k, v in append_attrs.items():
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        ds.attrs[str(k)] = v
    return ds
