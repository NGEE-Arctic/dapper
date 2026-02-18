# src/dapper/geo/zonal.py
"""Zonal (area-weighted) sampling utilities."""

from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree

from dapper.geo import sampling

LonWrap = Literal["auto", "0_360", "-180_180"]
TieBreak = Literal["smallest", "largest", "first"]

MAX_ZONAL_CELLS = 2_000_000

# ----------------------------- geometry helpers -----------------------------

def normalize_geometry_lon(geom, wrap: Literal["0_360", "-180_180"]):
    """
    Apply the same lon wrap convention as sampling.normalize_lon to *all* coords.
    This is the simplest way to make target polygons comparable to source grid.
    """
    def _f(x, y, z=None):
        x2 = sampling.normalize_lon(x, wrap)
        return (x2, y) if z is None else (x2, y, z)
    return transform(_f, geom)


def laea_crs_for_targets(targets_wgs84: gpd.GeoDataFrame) -> str:
    """
    One equal-area CRS for the whole domain (LAEA centered on centroid).
    Returns PROJ string usable by GeoPandas.
    """
    c = targets_wgs84.unary_union.representative_point()
    lon0, lat0 = float(c.x), float(c.y)
    return f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"


def _bounds_1d(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    mid = 0.5 * (vec[:-1] + vec[1:])
    b = np.empty(len(vec) + 1, dtype=float)
    b[1:-1] = mid
    b[0] = vec[0] - (mid[0] - vec[0])
    b[-1] = vec[-1] + (vec[-1] - mid[-1])
    return b


@dataclass(frozen=True)
class RectilinearGrid:
    """Lightweight description of a rectilinear lat/lon grid in a consistent lon wrap."""
    
    lat_dim: str
    lon_dim: str
    lon_wrap: Literal["0_360", "-180_180"]
    lat_bnds: np.ndarray  # (nlat+1,)
    lon_bnds: np.ndarray  # (nlon+1,)

    @property
    def nlat(self) -> int:
        """Number of latitude cells."""
        
        return len(self.lat_bnds) - 1

    @property
    def nlon(self) -> int:
        """Number of longitude cells."""
        
        return len(self.lon_bnds) - 1


def infer_rectilinear_grid(
    ds: xr.Dataset,
    *,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
) -> RectilinearGrid:
    """Infer a rectilinear grid specification (bounds and lon wrap) from a Dataset.

    Preference order:
      1) For rectilinear grids, use 1D coordinate vectors on (lat_dim, lon_dim) if present,
         unless the caller explicitly provides lat_var/lon_var.
      2) Otherwise fall back to sampling.infer_latlon_spec (e.g., LATIXY/LONGXY or provided vars).
    """

    # Prefer 1D coords on the dims for rectilinear grids.
    # This avoids accidentally using derived 2D vars (LATIXY/LONGXY) that might be overridden.
    if lat_var is None and lon_var is None and (lat_dim in ds.coords) and (lon_dim in ds.coords):
        lat_da = ds.coords[lat_dim]
        lon_da = ds.coords[lon_dim]
        if lat_da.ndim == 1 and lon_da.ndim == 1:
            lat_1d = np.asarray(lat_da.values, dtype=float)
            lon_1d_raw = np.asarray(lon_da.values, dtype=float)

            wrap = sampling.infer_lon_wrap(lon_1d_raw) if lon_wrap == "auto" else lon_wrap  # type: ignore[assignment]
            if wrap not in ("0_360", "-180_180"):
                raise ValueError(f"lon_wrap must resolve to '0_360' or '-180_180', got {wrap}")

            # sampling.normalize_lon is scalar; vectorize here.
            lon_1d = np.asarray([sampling.normalize_lon(float(v), wrap) for v in lon_1d_raw], dtype=float)

            return RectilinearGrid(
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                lon_wrap=wrap,
                lat_bnds=_bounds_1d(lat_1d),
                lon_bnds=_bounds_1d(lon_1d),
            )

    # Fallback: infer from explicit lat/lon variables (e.g., LATIXY/LONGXY).
    spec = sampling.infer_latlon_spec(
        ds,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )
    lat_bnds = _bounds_1d(spec.lat_1d)
    lon_bnds = _bounds_1d(spec.lon_1d)
    return RectilinearGrid(
        lat_dim=spec.lat_dim,
        lon_dim=spec.lon_dim,
        lon_wrap=spec.lon_wrap,
        lat_bnds=lat_bnds,
        lon_bnds=lon_bnds,
    )


def _candidate_ij_for_bounds(grid: RectilinearGrid, bounds_lonlat) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute i/j ranges for a bbox in *the same lon wrap as the grid*.
    This avoids building polygons for the full globe.
    """
    minx, miny, maxx, maxy = map(float, bounds_lonlat)

    i0 = max(int(np.searchsorted(grid.lat_bnds, miny, side="right") - 1), 0)
    i1 = min(int(np.searchsorted(grid.lat_bnds, maxy, side="left")), grid.nlat)
    j0 = max(int(np.searchsorted(grid.lon_bnds, minx, side="right") - 1), 0)
    j1 = min(int(np.searchsorted(grid.lon_bnds, maxx, side="left")), grid.nlon)

    ii = np.arange(i0, i1, dtype=int)
    jj = np.arange(j0, j1, dtype=int)
    return ii, jj


@dataclass(frozen=True)
class ZonalWeights:
    """Zonal intersection weights grouped by feature id.

    - ``by_gid[gid]`` is a pandas DataFrame with columns:
      ``i_lat``, ``i_lon``, ``intersect_area_m2``, ``weight``.
    - ``weight`` is normalized to sum to 1 for that ``gid``.
    """
    by_gid: dict[str, pd.DataFrame]
    lon_wrap: Literal["0_360", "-180_180"]
    equal_area_crs: str


def intersect_weights_rectilinear(
    ds: xr.Dataset,
    targets: gpd.GeoDataFrame,   # EPSG:4326, must have gid + geometry
    *,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
    min_frac: float = 0.0,
) -> ZonalWeights:
    """Compute area-weighted intersections between target polygons and a rectilinear grid."""
    
    grid = infer_rectilinear_grid(
        ds, lat_dim=lat_dim, lon_dim=lon_dim,
        lat_var=lat_var, lon_var=lon_var, lon_wrap=lon_wrap
    )

    # Normalize targets to grid lon wrap
    t = targets.copy()
    t["geometry"] = t.geometry.apply(lambda g: normalize_geometry_lon(g, grid.lon_wrap))

    ea = laea_crs_for_targets(t)
    t_ea = t.to_crs(ea)

    # Build a single candidate subset based on union bbox
    ii, jj = _candidate_ij_for_bounds(grid, t.total_bounds)
    n_cand = int(len(ii)) * int(len(jj))
    if n_cand > MAX_ZONAL_CELLS:
        raise ValueError(
            f"Zonal guardrail: bbox candidate grid is {n_cand:,} cells (> {MAX_ZONAL_CELLS:,}). "
            "This likely indicates a huge polygon or lon-wrap/bounds issue. "
            "Consider reducing target extent, increasing min_frac, or sampling a coarser dataset."
        )
    if n_cand > (MAX_ZONAL_CELLS // 2):
        warnings.warn(
            f"Zonal warning: bbox candidate grid is {n_cand:,} cells; this may be slow.",
            RuntimeWarning,
        )

    polys = []
    ij = []
    for i in ii:
        for j in jj:
            p = box(grid.lon_bnds[j], grid.lat_bnds[i], grid.lon_bnds[j + 1], grid.lat_bnds[i + 1])
            polys.append(p)
            ij.append((i, j))

    if not polys:
        raise ValueError("No candidate source cells found for target bounds.")

    src_gs = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs(ea)
    src_polys = list(src_gs.values)
    tree = STRtree(src_polys)

    by_gid: dict[str, pd.DataFrame] = {}

    for row in t_ea.itertuples(index=False):
        gid = str(getattr(row, "gid"))
        geom = getattr(row, "geometry")

        cand_idx = tree.query(geom)  # indices into src_polys
        if len(cand_idx) > MAX_ZONAL_CELLS:
            raise ValueError(
                f"Zonal guardrail: gid={gid!r} has {len(cand_idx):,} candidate cells (> {MAX_ZONAL_CELLS:,}). "
                "Reduce target extent or use a coarser grid."
            )
        rows = []
        for k in cand_idx:
            inter = geom.intersection(src_polys[k])
            if inter.is_empty:
                continue
            a = float(inter.area)
            if a <= 0.0:
                continue
            i, j = ij[int(k)]
            rows.append((i, j, a))

        if not rows:
            raise ValueError(f"Target gid={gid!r} intersects 0 source cells.")

        if len(rows) > MAX_ZONAL_CELLS:
            raise ValueError(
                f"Zonal guardrail: gid={gid!r} intersects {len(rows):,} cells (> {MAX_ZONAL_CELLS:,}). "
                "Reduce target extent or use a coarser grid."
            )

        df = pd.DataFrame(rows, columns=["i_lat", "i_lon", "intersect_area_m2"])
        total = float(df["intersect_area_m2"].sum())
        df["weight"] = df["intersect_area_m2"] / total

        if min_frac > 0:
            # min_frac is relative to total intersect area for that gid (not cell area)
            df = df[df["weight"] >= float(min_frac)].copy()
            df["weight"] = df["intersect_area_m2"] / float(df["intersect_area_m2"].sum())

        by_gid[gid] = df.reset_index(drop=True)

    return ZonalWeights(by_gid=by_gid, lon_wrap=grid.lon_wrap, equal_area_crs=ea)


# ----------------------------- reducers -----------------------------

def _reduce_da(da_sel: xr.DataArray, w: xr.DataArray, agg: str) -> xr.DataArray:
    if agg == "wmean":
        return (da_sel * w).sum("cell") / w.sum("cell")
    if agg == "area_sum":
        # w here should be raw area, not normalized weights
        return (da_sel * w).sum("cell")
    if agg == "max":
        return da_sel.max("cell")
    if agg == "min":
        return da_sel.min("cell")
    if agg == "wmode":
        # Simple weighted mode (works for small category counts).
        vals = da_sel.values
        ww = w.values
        # da_sel is vectorized selection => shape (..., cell). We assume only 'cell' varies.
        # Convert to 1D over cell for the mode; for multi-dim (time, pft, etc.) caller should loop.
        raise NotImplementedError("wmode reducer needs a per-slice implementation (see notes).")
    raise ValueError(f"Unknown agg={agg!r}")


def sample_gridded_dataset_polygons(
    ds: xr.Dataset,
    targets: gpd.GeoDataFrame,
    *,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = None,
    lon_var: str | None = None,
    lon_wrap: LonWrap = "auto",
    vars_include: Sequence[str] | None = None,
    vars_drop: Sequence[str] | None = None,
    agg_policy: dict[str, str] | None = None,
    default_float: str = "wmean",
    default_int: str = "wmode",
    weights: "ZonalWeights | None" = None,   # NEW
) -> xr.Dataset:
    """
    Zonal-sample spatial vars (those with BOTH lat_dim and lon_dim) onto target polygons.

    Output convention matches sample_gridded_dataset_points:
      - lat_dim has length N (number of targets, in targets row order)
      - lon_dim has length 1
      - no coordinate variables are created for lat_dim/lon_dim
    """
    agg_policy = dict(agg_policy or {})

    zw = weights if weights is not None else intersect_weights_rectilinear(
        ds, targets,
        lat_dim=lat_dim, lon_dim=lon_dim,
        lat_var=lat_var, lon_var=lon_var,
        lon_wrap=lon_wrap,
    )

    data_vars = list(ds.data_vars)
    if vars_include is not None:
        keep = set(vars_include)
        data_vars = [v for v in data_vars if v in keep]
    if vars_drop is not None:
        drop = set(vars_drop)
        data_vars = [v for v in data_vars if v not in drop]

    # Determine dims (in case infer_latlon_spec changed them)
    # BUT: don't require 2D lat/lon vars for rectilinear grids where dims/coords already exist.
    if (lat_dim in ds.dims) and (lon_dim in ds.dims):
        # keep caller-provided dims
        pass
    else:
        spec = sampling.infer_latlon_spec(
            ds,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            lat_var=lat_var,
            lon_var=lon_var,
            lon_wrap=lon_wrap,
        )
        lat_dim = spec.lat_dim
        lon_dim = spec.lon_dim

    spatial_vars = [v for v in data_vars if (lat_dim in ds[v].dims and lon_dim in ds[v].dims)]
    non_spatial_vars = [v for v in data_vars if v not in spatial_vars]

    sampled_slices: list[xr.Dataset] = []

    # loop in target order
    for gid in targets["gid"].astype(str).tolist():
        wdf = zw.by_gid[gid]
        if len(wdf) > MAX_ZONAL_CELLS:
            raise ValueError(
                f"Zonal guardrail: gid={gid!r} has {len(wdf):,} weighted intersections (> {MAX_ZONAL_CELLS:,}). "
                "Reduce target extent or use a coarser grid."
            )
        ij = wdf[["i_lat", "i_lon"]].to_numpy(dtype=int)

        # Vectorized cell selection (pointwise i/j pairs)
        i_idx = xr.DataArray(ij[:, 0], dims="cell")
        j_idx = xr.DataArray(ij[:, 1], dims="cell")

        # Normalized weights for wmean / wmode
        w_norm = xr.DataArray(wdf["weight"].to_numpy(dtype=float), dims="cell")
        # Raw areas for area_sum
        w_area = xr.DataArray(wdf["intersect_area_m2"].to_numpy(dtype=float), dims="cell")

        out_vars = {}
        for v in spatial_vars:
            da = ds[v]

            # Choose agg
            agg = agg_policy.get(v)
            if agg is None:
                agg = default_int if da.dtype.kind in {"i", "u", "b"} else default_float
            if v == "AREA" and agg is None:
                agg = "area_sum"

            da_sel = da.isel({lat_dim: i_idx, lon_dim: j_idx})  # -> dims replace lat/lon with "cell"

            # Reduce
            if agg == "area_sum":
                da_red = (da_sel * w_area).sum("cell")
            elif agg == "wmean":
                da_red = (da_sel * w_norm).sum("cell") / w_norm.sum("cell")
            elif agg == "max":
                da_red = da_sel.max("cell")
            elif agg == "min":
                da_red = da_sel.min("cell")
            elif agg == "wmode":
                da_red = reduce_wmode(da_sel, w_norm, cell_dim="cell", tie_break="smallest")
            elif agg == "wmean_threshold":
                m = (da_sel * w_norm).sum("cell") / w_norm.sum("cell")
                da_red = (m >= 0.5).astype(np.int32)
            else:
                raise ValueError(f"Unknown agg={agg!r} for var={v}")

            # Match existing convention: lon_dim length 1, no lat_dim yet (concat will create it)
            da_red = da_red.expand_dims({lon_dim: 1})
            out_vars[v] = da_red

        sel_ds = xr.Dataset(out_vars)
        # Match per-variable dim ordering like point sampler does
        for v in spatial_vars:
            sel_ds[v] = sampling._reorder_like_source(sel_ds[v], ds[v].dims, lat_dim, lon_dim)

        sampled_slices.append(sel_ds)

    out_spatial = xr.concat(sampled_slices, dim=lat_dim, create_index_for_new_dim=False)
    out = xr.merge([out_spatial, ds[non_spatial_vars]])

    out.attrs["dapper_sampling_method"] = "zonal"
    out.attrs["dapper_sampling_lon_wrap_native"] = zw.lon_wrap
    out.attrs["dapper_sampling_equal_area_crs"] = zw.equal_area_crs
    return out

def _weighted_mode_1d(values: np.ndarray, weights: np.ndarray, *, tie_break: TieBreak) -> object:
    """
    Weighted mode of a 1D array, ignoring NaNs (if float).
    Returns a scalar of the same "kind" as values.
    """
    v = values
    w = weights

    if np.issubdtype(v.dtype, np.floating):
        mask = ~np.isnan(v)
        v = v[mask]
        w = w[mask]

    if v.size == 0:
        raise ValueError("weighted mode: all values were NaN/missing for this slice")

    uniq, inv = np.unique(v, return_inverse=True)
    wsum = np.zeros(len(uniq), dtype=float)
    np.add.at(wsum, inv, w)

    maxw = wsum.max()
    tied = np.flatnonzero(wsum == maxw)

    if tied.size == 1:
        return uniq[tied[0]]

    tied_vals = uniq[tied]

    if tie_break == "smallest":
        return tied_vals.min()
    if tie_break == "largest":
        return tied_vals.max()

    # tie_break == "first": pick the earliest occurrence in original order
    tied_set = set(tied_vals.tolist())
    for val in v:
        if val in tied_set:
            return val

    # Should never happen
    return tied_vals[0]


def reduce_wmode(
    da_sel: xr.DataArray,
    w: xr.DataArray,
    *,
    cell_dim: str = "cell",
    tie_break: TieBreak = "smallest",
) -> xr.DataArray:
    """
    Reduce da_sel over `cell_dim` using an area-weighted mode.

    da_sel: DataArray with dimension `cell_dim` and any number of other dims.
    w: 1D weights over `cell_dim` (normalized or not; only relative weights matter).
    """
    if cell_dim not in da_sel.dims:
        raise ValueError(f"reduce_wmode: {cell_dim!r} not in da_sel.dims={da_sel.dims}")

    # Align weights to da_sel's cell index (should already match)
    if w.dims != (cell_dim,):
        w = w.rename({w.dims[0]: cell_dim}) if len(w.dims) == 1 else w

    # Move cell dim to last for efficient reshaping
    lead_dims = [d for d in da_sel.dims if d != cell_dim]
    da_t = da_sel.transpose(*lead_dims, cell_dim)

    arr = np.asarray(da_t.values)
    ww = np.asarray(w.values, dtype=float)

    if arr.shape[-1] != ww.shape[0]:
        raise ValueError(f"reduce_wmode: cell axis mismatch {arr.shape[-1]} vs {ww.shape[0]}")

    ncell = arr.shape[-1]
    flat = arr.reshape(-1, ncell)

    out = np.empty(flat.shape[0], dtype=arr.dtype if arr.dtype.kind in {"i", "u", "b"} else float)

    for r in range(flat.shape[0]):
        out[r] = _weighted_mode_1d(flat[r], ww, tie_break=tie_break)

    out = out.reshape(arr.shape[:-1])

    # Build output DataArray (all dims except cell_dim)
    out_da = xr.DataArray(out, dims=lead_dims)

    # Keep coords for retained dims (drop cell coords)
    for d in lead_dims:
        if d in da_t.coords:
            out_da = out_da.assign_coords({d: da_t.coords[d]})

    return out_da
