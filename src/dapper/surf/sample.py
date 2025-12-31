# dapper/surf/sample.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import xarray as xr

LatLonDimNames = Tuple[Optional[str], Optional[str]]


def _detect_latlon_dim_names(ds: xr.Dataset) -> LatLonDimNames:
    lat_candidates = ("lsmlat", "lat", "latitude", "y")
    lon_candidates = ("lsmlon", "lon", "longitude", "x")
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

    # Fallback: assume regular global grid with cell centers (best-effort)
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


def _capture_small_dim_coords(ds: xr.Dataset, lat_dim: str, lon_dim: str) -> Dict[str, np.ndarray]:
    coords: Dict[str, np.ndarray] = {}
    for dim in ds.dims:
        if dim in (lat_dim, lon_dim):
            continue
        if dim in ds.coords:
            try:
                coords[dim] = np.asarray(ds.coords[dim].values)
            except Exception:
                pass
        if dim in (
            "time", "natpft", "lsmpft", "nlevsoi", "nlevslp", "numurbl",
            "numrad", "nlevurb", "nglcec", "nglcecp1"
        ) and dim not in coords:
            coords[dim] = np.arange(ds.sizes[dim])
    return coords


class SurfacePointSampler:
    """
    Efficient point sampler for an ELM surface dataset: opens NetCDF once, samples many points.
    """

    def __init__(
        self,
        nc_in: str | Path,
        *,
        decode_times: bool = True,
        chunks: Optional[Dict[str, int]] = None,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
    ) -> None:
        self.ds = xr.open_dataset(nc_in, decode_times=decode_times, chunks=chunks or {})
        self.lat_dim, self.lon_dim = _detect_latlon_dim_names(self.ds)
        if not self.lat_dim or not self.lon_dim:
            raise ValueError("Could not detect spatial dims (need lsmlat/lsmlon or lat/lon).")

        self.lat_vec, self.lon_vec = _get_latlon_vectors(self.ds, self.lat_dim, self.lon_dim)
        self.coords_src = _capture_small_dim_coords(self.ds, self.lat_dim, self.lon_dim)

        names = list(self.ds.data_vars)
        if include:
            names = [n for n in names if n in include]
        if exclude:
            names = [n for n in names if n not in exclude]
        self.names = names

        self.global_attrs = dict(self.ds.attrs)

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:
            pass

    def __enter__(self) -> "SurfacePointSampler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def sample(self, lat: float, lon: float) -> Dict[str, Any]:
        lon_norm = _normalize_lon_to_array(lon, self.lon_vec)
        i = int(np.abs(self.lat_vec - lat).argmin())
        j = int(np.abs(self.lon_vec - lon_norm).argmin())

        out: Dict[str, Any] = {
            "__meta__": {
                "lat_in": float(lat),
                "lon_in": float(lon),
                "lat_dim": self.lat_dim,
                "lon_dim": self.lon_dim,
                "i": int(i),
                "j": int(j),
                "lat_on_grid": float(self.lat_vec[i]),
                "lon_on_grid": float(self.lon_vec[j]),
                "global_attrs": dict(self.global_attrs),
            },
            "__coords__": dict(self.coords_src),
        }

        for name in self.names:
            da = self.ds[name]
            orig_dims = tuple(da.dims)

            if (self.lat_dim in orig_dims) and (self.lon_dim in orig_dims):
                da_pt = _slice_spatial(da, self.lat_dim, self.lon_dim, i, j).squeeze(drop=True)
                dims_no_spatial = [d for d in orig_dims if d not in (self.lat_dim, self.lon_dim)]
                data = np.asarray(da_pt.values)
                out[name] = {
                    "dims": dims_no_spatial,
                    "orig_dims": list(orig_dims),
                    "data": data,
                    "attrs": dict(da.attrs or {}),
                    "dtype": str(da.dtype),
                }
            else:
                data = np.asarray(da.values)
                out[name] = {
                    "dims": list(orig_dims),
                    "orig_dims": list(orig_dims),
                    "data": data,
                    "attrs": dict(da.attrs or {}),
                    "dtype": str(da.dtype),
                }

        return out


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
    Backwards-friendly convenience wrapper: opens, samples one point, closes.
    """
    with SurfacePointSampler(
        nc_in=nc_in,
        decode_times=decode_times,
        chunks=chunks,
        include=include,
        exclude=exclude,
    ) as sampler:
        return sampler.sample(lat=lat, lon=lon)
