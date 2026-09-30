"""NetCDF global-attribute helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
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


def utc_timestamp() -> str:
    """Current UTC time as a naive ISO-8601 string with a trailing ``Z``."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"


def merge_global_attrs(
    attrs: Mapping[str, Any],
    *,
    dapper_attrs: Mapping[str, Any] | None = None,
    append_attrs: Mapping[str, Any] | None = None,
    add_created_utc: bool = True,
) -> dict[str, Any]:
    """Merge global attrs with precedence ``append_attrs`` > ``attrs`` > dapper defaults.

    ``dapper_attrs`` and ``dapper_created_utc`` only fill keys that are missing.
    """
    merged = dict(attrs)
    for k, v in (dapper_attrs or {}).items():
        merged.setdefault(k, v)
    if add_created_utc:
        merged.setdefault("dapper_created_utc", utc_timestamp())
    if append_attrs:
        merged.update(append_attrs)
    return merged
