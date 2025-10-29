
# dapper/surf/sample.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import xarray as xr

from dapper.utils.pathing import SURFDATA_HALFDEGREE_TOP

LatLonDimNames = Tuple[Optional[str], Optional[str]]

def _detect_latlon_dim_names(ds: xr.Dataset) -> LatLonDimNames:
    lat_candidates = ("lsmlat","lat","latitude","y")
    lon_candidates = ("lsmlon","lon","longitude","x")
    lat = next((d for d in ds.dims if d in lat_candidates), None)
    lon = next((d for d in ds.dims if d in lon_candidates), None)
    return lat, lon

def _get_latlon_vectors(ds: xr.Dataset, lat_dim: str, lon_dim: str) -> Tuple[np.ndarray, np.ndarray]:
    lat_vec = lon_vec = None
    if "LATIXY" in ds and set(ds["LATIXY"].dims) == {lat_dim, lon_dim}:
        lat_vec = np.asarray(ds["LATIXY"].isel({lon_dim: 0}).values, dtype=np.float64).ravel()
    if "LONGXY" in ds and set(ds["LONGXY"].dims) == {lat_dim, lon_dim}:
        lon_vec = np.asarray(ds["LONGXY"].isel({lat_dim: 0}).values, dtype=np.float64).ravel()
    if lat_vec is not None and lon_vec is not None:
        return lat_vec, lon_vec
    # Fallback: assume regular 0.5° global
    nlat, nlon = ds.sizes[lat_dim], ds.sizes[lon_dim]
    lat_vec = np.linspace(-90 + 0.25, 90 - 0.25, nlat, dtype=np.float64) if nlat > 1 else np.array([0.0])
    lon_vec = np.linspace(-180 + 0.25, 180 - 0.25, nlon, dtype=np.float64) if nlon > 1 else np.array([0.0])
    return lat_vec, lon_vec

def _normalize_lon_to_array(lon: float, lon_vec: np.ndarray) -> float:
    finite = lon_vec[np.isfinite(lon_vec)]
    if finite.size == 0:
        return lon
    frac_0360 = np.mean((finite >= 0) & (finite < 360))
    if frac_0360 > 0.9:
        return lon % 360 if lon >= 0 else (lon + 360) % 360
    ln = ((lon + 180) % 360) - 180
    return (180 - 1e-6) if ln == -180 else ln

def _slice_spatial(da: xr.DataArray, lat_dim: str, lon_dim: str, i: int, j: int) -> xr.DataArray:
    return da.isel({lat_dim: i, lon_dim: j})

def sample_point_values(
    nc_in: str | Path,
    lat: float,
    lon: float,
    *,
    decode_times: bool = True,
    chunks: Optional[Dict[str, int]] = None,
    include: Optional[set[str]] = None,
    exclude: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Point-sample all variables in an ELM surface NetCDF at (lat, lon) by nearest grid cell.
    Returns a dict structured for writing later (no file IO here).

    Returns
    -------
    {
      "__meta__": {
         "lat_in": float, "lon_in": float,
         "lat_dim": "lsmlat", "lon_dim": "lsmlon",
         "i": int, "j": int, "lat_on_grid": float, "lon_on_grid": float,
         "global_attrs": {...}
      },
      "__coords__": { "time": np.ndarray, "natpft": np.ndarray, "nlevsoi": np.ndarray, ... },
      "VARNAME": {
         "dims": ["time","natpft", ...],        # non-spatial dims only, in file order
         "orig_dims": ["time","lsmlat","lsmlon"],# original file dims (for writer)
         "data": np.ndarray,                     # shape matches 'dims'
         "attrs": {...},                         # variable attributes
         "dtype": "float32" | "int16" | ...
      },
      ...
    }
    """
    ds = xr.open_dataset(nc_in, decode_times=decode_times, chunks=chunks or {})
    lat_dim, lon_dim = _detect_latlon_dim_names(ds)
    if not lat_dim or not lon_dim:
        raise ValueError("Could not detect spatial dims (need lsmlat/lsmlon or lat/lon).")

    lat_vec, lon_vec = _get_latlon_vectors(ds, lat_dim, lon_dim)
    lon_norm = _normalize_lon_to_array(lon, lon_vec)
    i = int(np.abs(lat_vec - lat).argmin())
    j = int(np.abs(lon_vec - lon_norm).argmin())

    # prepare result
    out: Dict[str, Any] = {
        "__meta__": {
            "lat_in": float(lat), "lon_in": float(lon),
            "lat_dim": lat_dim, "lon_dim": lon_dim,
            "i": int(i), "j": int(j),
            "lat_on_grid": float(lat_vec[i]), "lon_on_grid": float(lon_vec[j]),
            "global_attrs": dict(ds.attrs),
        },
        "__coords__": {},
    }

    # capture 1-D coords for common non-spatial dims
    for dim in ds.dims:
        if dim not in (lat_dim, lon_dim) and dim in ds.coords:
            try:
                out["__coords__"][dim] = np.asarray(ds.coords[dim].values)
            except Exception:
                pass
        # If not in coords, but a known small-dim, still capture index vector
        if dim in ("time","natpft","lsmpft","nlevsoi","nlevslp","numurbl","numrad","nlevurb","nglcec","nglcecp1") and dim not in out["__coords__"]:
            out["__coords__"][dim] = np.arange(ds.sizes[dim])

    names = list(ds.data_vars)
    if include: names = [n for n in names if n in include]
    if exclude: names = [n for n in names if n not in exclude]

    for name in names:
        da = ds[name]
        orig_dims = tuple(da.dims)
        # If the var has both spatial dims, slice to the nearest cell
        if (lat_dim in orig_dims) and (lon_dim in orig_dims):
            da_pt = _slice_spatial(da, lat_dim, lon_dim, i, j).squeeze(drop=True)
            # non-spatial dims in file order:
            dims_no_spatial = [d for d in orig_dims if d not in (lat_dim, lon_dim)]
            data = np.asarray(da_pt.values)
            # Ensure 0-D arrays become shape ( ) not scalars; writer will handle reshaping
            out[name] = {
                "dims": dims_no_spatial,
                "orig_dims": list(orig_dims),
                "data": data,
                "attrs": dict(da.attrs or {}),
                "dtype": str(da.dtype),
            }
        else:
            # Keep scalars or non-spatial arrays as-is
            data = np.asarray(da.values)
            out[name] = {
                "dims": list(orig_dims),
                "orig_dims": list(orig_dims),
                "data": data,
                "attrs": dict(da.attrs or {}),
                "dtype": str(da.dtype),
            }

    return out
