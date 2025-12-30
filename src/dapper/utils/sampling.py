from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

LonWrap = Literal["auto", "0_360", "-180_180"]
SampleMethod = Literal["nearest"]


@dataclass(frozen=True)
class LatLonSpec:
    """Minimal spec for mapping (lat, lon) -> (i, j) in a gridded dataset."""
    lat_var: str
    lon_var: str
    lat_dim: str
    lon_dim: str
    lon_wrap: Literal["0_360", "-180_180"]
    lat_1d: np.ndarray
    lon_1d: np.ndarray


def ensure_weight(
    df: pd.DataFrame,
    *,
    weight_col: str = "weight",
    default_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Ensure df has a weight column. Returns a COPY if it needs to add the column.
    """
    if weight_col in df.columns:
        return df
    out = df.copy()
    out[weight_col] = float(default_weight)
    return out


def infer_lat_lon_vars(ds: xr.Dataset) -> tuple[str, str]:
    """
    Prefer ELM-style 2D vars (LATIXY/LONGXY), else fall back to common names.
    """
    lat_candidates = ["LATIXY", "lat", "LAT", "latitude", "nav_lat"]
    lon_candidates = ["LONGXY", "lon", "LON", "longitude", "nav_lon"]

    lat_var = next((v for v in lat_candidates if v in ds.variables), None)
    lon_var = next((v for v in lon_candidates if v in ds.variables), None)

    if lat_var is None or lon_var is None:
        raise ValueError(
            f"Could not infer lat/lon variables. Found lat={lat_var}, lon={lon_var}. "
            f"Available vars include: {list(ds.variables)[:50]}..."
        )
    return lat_var, lon_var


def _infer_lon_wrap(lon_vals: np.ndarray) -> Literal["0_360", "-180_180"]:
    lon_vals = np.asarray(lon_vals)
    finite = lon_vals[np.isfinite(lon_vals)]
    if finite.size == 0:
        return "-180_180"
    mn = float(finite.min())
    mx = float(finite.max())
    # typical ELM files store degrees_east in [0, 360)
    if mn >= 0.0 and mx > 180.0:
        return "0_360"
    return "-180_180"


def normalize_lon(lon: float, wrap: Literal["0_360", "-180_180"]) -> float:
    lon = float(lon)
    if wrap == "0_360":
        lon = lon % 360.0
        if lon < 0:
            lon += 360.0
    else:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon

def _to_0_360(lon_da: xr.DataArray) -> xr.DataArray:
    return ((lon_da % 360.0) + 360.0) % 360.0

def _lon_distance_deg(lon_vec: np.ndarray, lon0: float) -> np.ndarray:
    """
    Minimal angular distance on a circle (degrees), assuming lon_vec and lon0
    are in the same wrap convention.
    """
    delta = (lon_vec - lon0 + 180.0) % 360.0 - 180.0
    return np.abs(delta)


def infer_latlon_spec(
    ds: xr.Dataset,
    *,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
) -> LatLonSpec:
    """
    Build a LatLonSpec for fast nearest-neighbor lookup.

    Assumptions (fine for your landuse/surf ELM-style files):
      - LATIXY/LONGXY exist as 2D (lat_dim, lon_dim), OR
      - lat/lon exist as 1D vectors.
    """
    if lat_var is None or lon_var is None:
        lat_var2, lon_var2 = infer_lat_lon_vars(ds)
        lat_var = lat_var or lat_var2
        lon_var = lon_var or lon_var2

    lat_da = ds[lat_var]
    lon_da = ds[lon_var]

    if lat_da.ndim == 2 and lon_da.ndim == 2:
        if lat_da.dims != (lat_dim, lon_dim):
            # still allow it, but you must pass dims explicitly if different
            lat_dim = lat_da.dims[0]
            lon_dim = lat_da.dims[1]
        lat_1d = lat_da.isel({lon_dim: 0}).values
        lon_1d = lon_da.isel({lat_dim: 0}).values
    elif lat_da.ndim == 1 and lon_da.ndim == 1:
        lat_dim = lat_da.dims[0]
        lon_dim = lon_da.dims[0]
        lat_1d = lat_da.values
        lon_1d = lon_da.values
    else:
        raise NotImplementedError(
            f"Unsupported lat/lon shapes: {lat_var}{lat_da.shape}, {lon_var}{lon_da.shape}"
        )

    wrap = _infer_lon_wrap(lon_1d) if lon_wrap == "auto" else lon_wrap  # type: ignore[assignment]
    if wrap not in ("0_360", "-180_180"):
        raise ValueError(f"lon_wrap must resolve to '0_360' or '-180_180', got {wrap}")

    return LatLonSpec(
        lat_var=lat_var,
        lon_var=lon_var,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lon_wrap=wrap,
        lat_1d=np.asarray(lat_1d),
        lon_1d=np.asarray(lon_1d),
    )


def nearest_ij(spec: LatLonSpec, lat: float, lon: float) -> tuple[int, int]:
    """
    Nearest-neighbor (i, j) on a regular lat/lon grid (lat_1d, lon_1d).
    """
    lon_n = normalize_lon(lon, spec.lon_wrap)
    i = int(np.nanargmin(np.abs(spec.lat_1d - float(lat))))
    j = int(np.nanargmin(_lon_distance_deg(spec.lon_1d, lon_n)))
    return i, j


def _reorder_like_source(var_da: xr.DataArray, src_dims: Sequence[str], lat_dim: str, lon_dim: str) -> xr.DataArray:
    """
    After sampling/concat, force dimension order to match source
    (with spatial dims in the same relative position).
    """
    wanted = [d for d in src_dims if d not in (lat_dim, lon_dim)] + [lat_dim, lon_dim]
    wanted = [d for d in wanted if d in var_da.dims]
    for d in var_da.dims:
        if d not in wanted:
            wanted.append(d)
    return var_da.transpose(*wanted)


def sample_gridded_dataset_points(
    ds: xr.Dataset,
    points: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
    method: SampleMethod = "nearest",
    vars_include: Sequence[str] | None = None,
    vars_drop: Sequence[str] | None = None,
) -> xr.Dataset:
    """
    Sample all spatial vars (those containing BOTH lat_dim and lon_dim) at the
    provided point locations.

    Output convention:
      - lat_dim has length N (number of points)
      - lon_dim has length 1
      - no coordinate variables are created for lat_dim/lon_dim (matches your Toolik file)
    """
    if method != "nearest":
        raise NotImplementedError("Only method='nearest' is implemented right now.")

    spec = infer_latlon_spec(
        ds,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )
    lat_dim = spec.lat_dim
    lon_dim = spec.lon_dim

    data_vars = list(ds.data_vars)
    if vars_include is not None:
        keep = set(vars_include)
        data_vars = [v for v in data_vars if v in keep]
    if vars_drop is not None:
        drop = set(vars_drop)
        data_vars = [v for v in data_vars if v not in drop]

    spatial_vars = [v for v in data_vars if (lat_dim in ds[v].dims and lon_dim in ds[v].dims)]
    non_spatial_vars = [v for v in data_vars if v not in spatial_vars]

    # Build spatial-only sampled datasets and concat along lat_dim.
    sampled_slices: list[xr.Dataset] = []
    for _, row in points.iterrows():
        i, j = nearest_ij(spec, float(row[lat_col]), float(row[lon_col]))
        # Remove both spatial dims, then add lon_dim back as length-1 WITHOUT coords.
        sel = ds[spatial_vars].isel({lat_dim: i, lon_dim: j}).expand_dims({lon_dim: 1})
        sampled_slices.append(sel)

    out_spatial = xr.concat(sampled_slices, dim=lat_dim, create_index_for_new_dim=False)

    # Force per-variable dimension ordering to match the source dataset.
    for v in spatial_vars:
        out_spatial[v] = _reorder_like_source(out_spatial[v], ds[v].dims, lat_dim, lon_dim)

    # Merge in non-spatial vars once (YEAR, time, scalar strings, etc.)
    out = xr.merge([out_spatial, ds[non_spatial_vars]])

    # Keep coords from ds that matter (time, natpft, etc.) — but do NOT introduce lsmlat/lsmlon coords.
    # (By construction we avoided them.)
    return out


def write_netcdf(
    ds: xr.Dataset,
    out_path: str | Path,
    *,
    compress: bool = True,
    complevel: int = 4,
) -> Path:
    out_path = Path(out_path)

    encoding = {}
    if compress:
        for v in ds.data_vars:
            # don't try to compress strings/object arrays
            if ds[v].dtype.kind in {"U", "S", "O"}:
                continue
            encoding[v] = {"zlib": True, "complevel": int(complevel)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path

def points_to_nearest_cells(
    ds: xr.Dataset,
    points: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    weight_col: str = "weight",
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
) -> pd.DataFrame:
    """
    Return a dataframe with (i_lat, i_lon) and the chosen cell center for each point.
    Keeps the original weight column.
    """
    points = ensure_weight(points, weight_col=weight_col, default_weight=1.0)

    spec = infer_latlon_spec(
        ds,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )

    rows = []
    for idx, row in points.iterrows():
        lat0 = float(row[lat_col])
        lon0 = float(row[lon_col])
        lon_n = normalize_lon(lon0, spec.lon_wrap)
        i, j = nearest_ij(spec, lat0, lon0)

        lat_cell = float(ds[spec.lat_var].isel({spec.lat_dim: i, spec.lon_dim: j}).values)
        lon_cell = float(ds[spec.lon_var].isel({spec.lat_dim: i, spec.lon_dim: j}).values)

        rows.append(
            {
                "index": idx,
                lat_col: lat0,
                lon_col: lon0,
                "lon_normalized": lon_n,
                "i_lat": i,
                "i_lon": j,
                "lat_cell": lat_cell,
                "lon_cell": lon_cell,
                weight_col: float(row[weight_col]),
            }
        )

    return pd.DataFrame(rows).set_index("index")
