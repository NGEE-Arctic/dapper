from __future__ import annotations

from typing import Literal

import numpy as np

LonWrap = Literal["auto", "0_360", "-180_180"]


def infer_lon_wrap(lon_vals: np.ndarray) -> Literal["0_360", "-180_180"]:
    """Infer whether longitudes are stored as [0, 360) or [-180, 180)."""
    finite = lon_vals[np.isfinite(lon_vals)]
    if finite.size == 0:
        return "-180_180"
    frac_0360 = np.mean((finite >= 0) & (finite < 360))
    return "0_360" if frac_0360 > 0.9 else "-180_180"


def normalize_lon(lon: float, wrap: Literal["0_360", "-180_180"]) -> float:
    """Normalize a longitude to the requested wrapping convention."""
    if wrap == "0_360":
        return lon % 360 if lon >= 0 else (lon + 360) % 360
    # [-180, 180)
    ln = ((lon + 180) % 360) - 180
    # avoid exact -180 which sometimes breaks nearest-neighbor logic
    return (180 - 1e-6) if ln == -180 else ln


def normalize_lons(lon_vals: np.ndarray, wrap: Literal["0_360", "-180_180"]) -> np.ndarray:
    """Vectorized longitude normalization."""
    lon_vals = np.asarray(lon_vals, dtype=float)
    out = lon_vals.copy()
    # preserve NaNs
    m = np.isfinite(out)
    out[m] = np.array([normalize_lon(float(v), wrap) for v in out[m]], dtype=float)
    return out
