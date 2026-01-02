
"""
CMIP6 utilities (Pangeo / intake-esm)

Design goals:
- Separate *search/listing* of available datasets from *sampling*.
- Cache the catalog so repeated searches don't re-parse the JSON.
- Support fast "dataset-first" sampling: open each remote zarr once, then compute means for many AOIs.

Typical workflow:
    col = open_cmip6_catalog()
    df_all = search_cmip6(params, col=col)
    df_use = dedupe_latest(df_all)
    df_use = filter_complete(df_use, required_vars=params["variables"])
    df_use = df_use[df_use["source_id"].isin([...])]  # optional hard filter
    out = sample_bbox_means_for_aois(df_use, aois={...}, out_csv=...)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

# Optional dependency (only used if you call bounds_from_geojson)
try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore

from dapper.elm import utils as eu

DEFAULT_CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


# -----------------------------------------------------------------------------
# Catalog open/search
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def open_cmip6_catalog(url: str = DEFAULT_CATALOG_URL):
    """
    Open (and cache) the Pangeo CMIP6 intake-esm catalog.

    Important:
      - intake-esm registers its plugin into `intake`, so you open via
        `intake.open_esm_datastore(...)`.
    """
    import intake_esm  # noqa: F401  (plugin registration)
    import intake

    return intake.open_esm_datastore(url)


def _normalize_params(params: dict) -> dict:
    params = dict(params)  # shallow copy
    if params.get("variables") == "elm":
        params["variables"] = eu.elm_data_dicts()["cmip_req_vars"]
    return params


def search_cmip6(params: dict, col=None) -> pd.DataFrame:
    """
    Search the CMIP6 catalog and return the raw matches dataframe.

    params keys (all optional):
        experiment: list[str] -> experiment_id
        table: str|list[str]  -> table_id
        variables: list[str]  -> variable_id
        ensemble: str|list[str] -> member_id
        models: list[str] -> source_id
        grid: str|list[str] -> grid_label

    Returns:
        pd.DataFrame of the intake-esm matches (metadata only).
    """
    params = _normalize_params(params)

    param_mapping = {
        "experiment": "experiment_id",
        "table": "table_id",
        "variables": "variable_id",
        "ensemble": "member_id",
        "models": "source_id",
        "grid": "grid_label",
    }

    search_args = {
        intake_key: params[key]
        for key, intake_key in param_mapping.items()
        if key in params and params[key] is not None
    }

    col = col or open_cmip6_catalog()
    matches = col.search(**search_args)
    return matches.df.copy()


def summarize_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: quick summary of what you matched.
    Returns a small table you can print/log.
    """
    cols = [c for c in ["experiment_id", "table_id", "variable_id", "member_id", "grid_label"] if c in df.columns]
    if not cols:
        return pd.DataFrame({"rows": [len(df)]})
    return df.groupby(cols, dropna=False).size().rename("n").reset_index().sort_values("n", ascending=False)


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by keeping the latest 'version' (if present) for each dataset key.

    In the Pangeo CMIP6 catalog, duplicates often exist for the same
    (model, experiment, member, table, grid, variable) with different versions.
    """
    key_cols = [c for c in ["source_id", "experiment_id", "member_id", "table_id", "grid_label", "variable_id"] if c in df.columns]
    if not key_cols:
        return df.copy()

    if "version" in df.columns:
        out = df.sort_values("version").drop_duplicates(subset=key_cols, keep="last").copy()
    else:
        # Fallback: just keep the last instance
        out = df.drop_duplicates(subset=key_cols, keep="last").copy()

    return out.reset_index(drop=True)


def filter_complete(
    df: pd.DataFrame,
    required_vars: Sequence[str],
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Keep only dataset groups that contain *all* required variables.

    By default, completeness is enforced per:
      (source_id, experiment_id, member_id, table_id, grid_label)

    This prevents "mixing" variables from different grids/members/experiments.
    """
    required = set(required_vars)
    if not required:
        return df.copy()

    if group_cols is None:
        group_cols = ["source_id", "experiment_id", "member_id", "table_id", "grid_label"]

    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        # Worst-case fallback: enforce completeness per model only.
        group_cols = ["source_id"] if "source_id" in df.columns else []

    if not group_cols:
        return df.copy()

    def _has_all(g: pd.DataFrame) -> bool:
        vars_here = set(g["variable_id"].values) if "variable_id" in g.columns else set()
        return required.issubset(vars_here)

    return df.groupby(group_cols, dropna=False).filter(_has_all).reset_index(drop=True)


def find_available_data(params: dict, col=None) -> pd.DataFrame:
    """
    Backwards-compatible wrapper around the new search/filter approach.

    NOTE:
      - This is now *fast* because open_cmip6_catalog is cached.
      - It enforces completeness across all requested variables.
    """
    params = _normalize_params(params)
    df = search_cmip6(params, col=col)
    df = dedupe_latest(df)

    if "variables" in params and isinstance(params["variables"], (list, tuple)):
        df = filter_complete(df, required_vars=params["variables"])

    return df


# -----------------------------------------------------------------------------
# AOI helpers
# -----------------------------------------------------------------------------

def bounds_from_geojson(path: Union[str, Path]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Read a polygon GeoJSON/shapefile and return:
        lat_bounds = (lat_min, lat_max)
        lon_bounds = (lon_min, lon_max)

    Requires geopandas.
    """
    if gpd is None:
        raise ImportError("geopandas is required for bounds_from_geojson()")

    gdf = gpd.read_file(str(path)).to_crs("EPSG:4326")
    geom = gdf.geometry.unary_union
    lon_min, lat_min, lon_max, lat_max = geom.bounds
    return (lat_min, lat_max), (lon_min, lon_max)


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def _wrap_lon_like(ds: xr.Dataset, lon_min: float, lon_max: float) -> Tuple[float, float]:
    """
    Match bbox lon convention to dataset convention (rough but effective).
    If dataset uses 0..360 and bbox uses negatives, wrap bbox to 0..360.
    """
    if "lon" not in ds.coords:
        return lon_min, lon_max
    lon = ds["lon"]
    try:
        lonmax = float(lon.max())
    except Exception:
        return lon_min, lon_max
    if lonmax > 180 and lon_min < 0:
        return lon_min % 360, lon_max % 360
    return lon_min, lon_max


def _subset_bbox(ds: xr.Dataset, var: str, lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]) -> xr.DataArray:
    """
    Subset a variable to a bounding box.

    Supports:
      - regular 1D lat/lon
      - curvilinear 2D lat/lon (masking)
    """
    da = ds[var]

    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = lon_bounds
    lon_min, lon_max = _wrap_lon_like(ds, lon_min, lon_max)

    lat = ds["lat"]
    lon = ds["lon"]

    # 1D grid: fast slicing
    if lat.ndim == 1 and lon.ndim == 1:
        # handle decreasing latitude
        lat_slice = slice(lat_min, lat_max) if float(lat[0]) < float(lat[-1]) else slice(lat_max, lat_min)

        if lon_min <= lon_max:
            return da.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
        # dateline-crossing bbox
        a = da.sel(lat=lat_slice, lon=slice(lon_min, float(lon.max())))
        b = da.sel(lat=lat_slice, lon=slice(float(lon.min()), lon_max))
        return xr.concat([a, b], dim="lon")

    # Curvilinear: mask (slower but general)
    mask = (lat >= lat_min) & (lat <= lat_max)
    if lon_min <= lon_max:
        mask = mask & (lon >= lon_min) & (lon <= lon_max)
    else:
        mask = mask & ((lon >= lon_min) | (lon <= lon_max))
    return da.where(mask)


def _spatial_mean(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """
    Compute an (optionally cos(lat) weighted) mean over horizontal dims.
    """
    # Prefer horizontal dims implied by lat/lon coordinates
    spatial_dims: List[str] = []
    if "lat" in ds.coords:
        spatial_dims.extend([d for d in ds["lat"].dims if d in da.dims])
    if "lon" in ds.coords:
        spatial_dims.extend([d for d in ds["lon"].dims if d in da.dims])
    spatial_dims = sorted(set(spatial_dims))

    if not spatial_dims:
        # fallback: everything except time
        spatial_dims = [d for d in da.dims if d != "time"]

    if not spatial_dims:
        return da  # nothing to average

    if "lat" in ds.coords:
        w = np.cos(np.deg2rad(ds["lat"]))
        return da.weighted(w).mean(dim=spatial_dims, skipna=True)

    return da.mean(dim=spatial_dims, skipna=True)


def _maybe_time_subset(da: xr.DataArray, time_min: Optional[str], time_max: Optional[str]) -> xr.DataArray:
    if time_min is None and time_max is None:
        return da
    if "time" not in da.dims:
        return da
    t0 = time_min if time_min is not None else None
    t1 = time_max if time_max is not None else None
    return da.sel(time=slice(t0, t1))


# -----------------------------------------------------------------------------
# Crash-resilient chunked output helpers
# -----------------------------------------------------------------------------

def _slugify(s: object, max_len: int = 120) -> str:
    """Make a filesystem-safe token."""
    import re
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s))
    out = out.strip("-")
    if len(out) > max_len:
        out = out[:max_len]
    return out or "na"


def _dataset_key_from_row(row) -> str:
    """Key for a single CMIP zarr store (row of the intake-esm dataframe)."""
    parts = [
        _slugify(getattr(row, "source_id", "na")),
        _slugify(getattr(row, "experiment_id", "na")),
        _slugify(getattr(row, "member_id", "na")),
        _slugify(getattr(row, "table_id", "na")),
        _slugify(getattr(row, "grid_label", "na")),
        _slugify(getattr(row, "variable_id", "na")),
    ]
    # Add a short hash of zstore for safety (versions / duplicates)
    z = getattr(row, "zstore", "")
    try:
        import hashlib
        h = hashlib.sha1(str(z).encode("utf-8")).hexdigest()[:10]
        parts.append(h)
    except Exception:
        pass
    return "__".join(parts)


def _atomic_write(df: pd.DataFrame, out_path: Path, fmt: str = "parquet") -> Path:
    """Write df atomically (tmp -> rename). Returns final path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    if fmt.lower() == "parquet":
        try:
            df.to_parquet(tmp, index=False)
        except Exception as e:
            # Parquet engine may be missing; fall back to CSV in the same directory.
            csv_fallback = out_path.with_suffix(".csv")
            tmp_csv = csv_fallback.with_suffix(".csv.tmp")
            df.to_csv(tmp_csv, index=False)
            tmp_csv.replace(csv_fallback)
            return csv_fallback
    elif fmt.lower() == "csv":
        df.to_csv(tmp, index=False)
    else:
        raise ValueError(f"Unsupported fmt={fmt!r}. Use 'parquet' or 'csv'.")

    tmp.replace(out_path)
    return out_path


def _read_chunks(out_dir: Path) -> pd.DataFrame:
    """Read all chunk files from out_dir (parquet and/or csv)."""
    parts: List[pd.DataFrame] = []
    for p in sorted(out_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(p))
    for p in sorted(out_dir.glob("*.csv")):
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _log_failure(fail_log: Optional[Path], dataset_key: str, err: Exception) -> None:
    if fail_log is None:
        return
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    import traceback
    msg = f"{dataset_key}\t{type(err).__name__}: {err}\n"
    tb = traceback.format_exc()
    with fail_log.open("a", encoding="utf-8") as f:
        f.write(msg)
        f.write(tb)
        f.write("\n")


def sample_bbox_means_for_aois(
    df: pd.DataFrame,
    aois: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    out_csv: Optional[Union[str, Path]] = None,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    show_progress: bool = True,
    *,
    out_dir: Optional[Union[str, Path]] = None,
    chunk_format: str = "parquet",
    resume: bool = True,
    return_df: bool = True,
    fail_log: Optional[Union[str, Path]] = None,
    retries: int = 3,
    retry_backoff: float = 1.0,
) -> pd.DataFrame:
    """
    Dataset-first sampler (pattern you wanted):
      - Loop over datasets (rows of df), open each zarr once
      - For that dataset, compute bbox-mean time series for each AOI

    Crash-resilient mode (recommended):
      - Set out_dir=... to write ONE chunk file per dataset row as you go.
      - If resume=True, already-written chunks are skipped on rerun.
      - If return_df=True, the function returns the concatenation of *all* chunks in out_dir
        (including chunks from prior runs).

    Notes:
      - "dataset" here is a single (model, experiment, member, table, grid, variable) zarr store.
      - Parquet append to a single file is intentionally avoided; parquet is much happier as a directory of files.

    Args:
        df: output of search_cmip6/find_available_data (ideally deduped + filtered).
        aois: dict mapping aoi_id -> ((lat_min, lat_max), (lon_min, lon_max)).
        out_csv: optional final combined CSV to write (built from in-memory output or by reading chunks).
        time_min/time_max: optional ISO-like strings for time slicing.
        show_progress: use tqdm if available.

        out_dir: directory to write chunk outputs (recommended for long runs).
        chunk_format: 'parquet' (default) or 'csv'. If parquet engine is missing, it falls back to csv automatically.
        resume: if True and out_dir provided, skip dataset rows that already have chunk output.
        return_df: if True, return concatenated dataframe (reads chunks if out_dir provided).
        fail_log: optional path to a log file for failures; failures are logged and the run continues.
        retries: number of attempts per dataset row on transient errors.
        retry_backoff: base backoff seconds (exponential) between retries.

    Returns:
        Combined long-format dataframe, unless return_df=False (then returns empty dataframe).
    """
    required_cols = {"zstore", "variable_id", "source_id", "experiment_id", "member_id"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    out_dir_path = Path(out_dir) if out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    fail_log_path = Path(fail_log) if fail_log is not None else None

    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    # Optional progress bar
    it = df.itertuples(index=False)
    tqdm_obj = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            tqdm_obj = tqdm
            it = tqdm(it, total=len(df), desc="Sampling CMIP6 datasets")
        except Exception:
            it = df.itertuples(index=False)

    out_rows: List[pd.DataFrame] = []

    for row in it:
        dataset_key = _dataset_key_from_row(row)

        # Resume logic: skip if chunk already exists
        if out_dir_path is not None and resume:
            p_parq = out_dir_path / f"{dataset_key}.parquet"
            p_csv = out_dir_path / f"{dataset_key}.csv"
            if p_parq.exists() or p_csv.exists():
                continue

        var = row.variable_id
        model = row.source_id
        expid = row.experiment_id
        member = row.member_id
        table = getattr(row, "table_id", None)
        grid = getattr(row, "grid_label", None)

        def _compute_one() -> pd.DataFrame:
            ds = xr.open_zarr(
                fsspec.get_mapper(row.zstore, token="anon", access="read_only"),
                consolidated=True,
                decode_times=time_coder,
            )

            if var not in ds:
                # create an empty-but-valid chunk so resume works
                cols = ["time", "value", "aoi_id", "variable", "units", "model", "experiment", "member"]
                if table is not None:
                    cols.append("table")
                if grid is not None:
                    cols.append("grid")
                return pd.DataFrame(columns=cols)

            units_in = ds[var].attrs.get("units", "")
            # pr: kg m-2 s-1 is numerically equal to mm s-1 water equivalent
            units_out = (
                "mm s-1"
                if (var == "pr" and "kg" in units_in and "m-2" in units_in and "s-1" in units_in)
                else units_in
            )

            aoi_dfs: List[pd.DataFrame] = []

            for aoi_id, (lat_bounds, lon_bounds) in aois.items():
                da_sub = _subset_bbox(ds, var, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
                da_sub = _maybe_time_subset(da_sub, time_min=time_min, time_max=time_max)
                ts = _spatial_mean(da_sub, ds)

                # Force compute here so each AOI contributes real values (remote IO happens here).
                ts = ts.load()

                df_ts = ts.to_dataframe(name="value").reset_index()

                # stringify cftime safely
                if "time" in df_ts.columns and not np.issubdtype(df_ts["time"].dtype, np.datetime64):
                    df_ts["time"] = df_ts["time"].astype(str)

                df_ts["aoi_id"] = aoi_id
                df_ts["variable"] = var
                df_ts["units"] = units_out
                df_ts["model"] = model
                df_ts["experiment"] = expid
                df_ts["member"] = member
                if table is not None:
                    df_ts["table"] = table
                if grid is not None:
                    df_ts["grid"] = grid
                df_ts["lat_min"], df_ts["lat_max"] = min(lat_bounds), max(lat_bounds)
                df_ts["lon_min"], df_ts["lon_max"] = lon_bounds[0], lon_bounds[1]

                aoi_dfs.append(df_ts)

            return pd.concat(aoi_dfs, ignore_index=True) if aoi_dfs else pd.DataFrame()

        # Retry wrapper (handles transient HTTP / chunk fetch failures)
        last_err: Optional[Exception] = None
        df_chunk: Optional[pd.DataFrame] = None
        for attempt in range(max(1, retries)):
            try:
                df_chunk = _compute_one()
                last_err = None
                break
            except Exception as e:
                last_err = e
                # exponential backoff
                import time as _time
                _time.sleep(retry_backoff * (2 ** attempt))

        if last_err is not None:
            _log_failure(fail_log_path, dataset_key, last_err)
            continue

        if df_chunk is None:
            continue

        # Write chunk immediately (crash-safe)
        if out_dir_path is not None:
            ext = ".parquet" if chunk_format.lower() == "parquet" else ".csv"
            chunk_path = out_dir_path / f"{dataset_key}{ext}"
            _atomic_write(df_chunk, chunk_path, fmt=chunk_format)

        # Optionally collect in-memory
        if return_df and out_dir_path is None:
            out_rows.append(df_chunk)

    # Build final output
    if return_df:
        if out_dir_path is not None:
            out = _read_chunks(out_dir_path)
        else:
            out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    else:
        out = pd.DataFrame()

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out

    required_cols = ["zstore", "variable_id", "source_id", "experiment_id", "member_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    # Optional progress bar
    it = df.itertuples(index=False)
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            it = tqdm(it, total=len(df), desc="Sampling CMIP6 datasets")
        except Exception:
            it = df.itertuples(index=False)

    out_rows: List[pd.DataFrame] = []

    for row in it:
        var = row.variable_id
        model = row.source_id
        expid = row.experiment_id
        member = row.member_id
        table = getattr(row, "table_id", None)
        grid = getattr(row, "grid_label", None)

        ds = xr.open_zarr(
            fsspec.get_mapper(row.zstore, token="anon", access="read_only"),
            consolidated=True,
            decode_times=time_coder,
        )

        if var not in ds:
            continue

        units_in = ds[var].attrs.get("units", "")
        # pr: kg m-2 s-1 is numerically equal to mm s-1 water equivalent
        units_out = "mm s-1" if (var == "pr" and "kg" in units_in and "m-2" in units_in and "s-1" in units_in) else units_in

        for aoi_id, (lat_bounds, lon_bounds) in aois.items():
            da_sub = _subset_bbox(ds, var, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
            da_sub = _maybe_time_subset(da_sub, time_min=time_min, time_max=time_max)
            ts = _spatial_mean(da_sub, ds)

            # Force compute here so each AOI contributes real values.
            # (This is where remote IO happens.)
            ts = ts.load()

            df_ts = ts.to_dataframe(name="value").reset_index()

            # stringify cftime safely
            if "time" in df_ts.columns and not np.issubdtype(df_ts["time"].dtype, np.datetime64):
                df_ts["time"] = df_ts["time"].astype(str)

            df_ts["aoi_id"] = aoi_id
            df_ts["variable"] = var
            df_ts["units"] = units_out
            df_ts["model"] = model
            df_ts["experiment"] = expid
            df_ts["member"] = member
            if table is not None:
                df_ts["table"] = table
            if grid is not None:
                df_ts["grid"] = grid
            df_ts["lat_min"], df_ts["lat_max"] = min(lat_bounds), max(lat_bounds)
            df_ts["lon_min"], df_ts["lon_max"] = lon_bounds[0], lon_bounds[1]

            out_rows.append(df_ts)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out


# -----------------------------------------------------------------------------
# Legacy / local-file helpers (kept, but trimmed)
# -----------------------------------------------------------------------------

def download_pangeo(
    df: pd.DataFrame,
    dir_out: Union[str, Path],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    lon_bounds: Optional[Tuple[float, float]] = None,
    polygon_path: Optional[Union[str, Path]] = None,
):
    """
    Download CMIP6 data from Pangeo to NetCDF, with optional spatial subsetting.

    Note: only one of (lat/lon), (lat_bounds/lon_bounds), or (polygon_path) should be provided.
    """
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    for _, row in df.iterrows():
        filename = f"{row.variable_id}_{row.source_id}_{row.experiment_id}_{row.member_id}.nc"
        try:
            ds = xr.open_zarr(
                fsspec.get_mapper(row.zstore, token="anon", access="read_only"),
                consolidated=True,
                decode_times=time_coder,
            )

            # Normalize longitude bbox if dataset uses 0–360
            _lon = lon
            _lon_bounds = lon_bounds
            if "lon" in ds.coords:
                try:
                    if float(ds.lon.max()) > 180:
                        if _lon is not None and _lon < 0:
                            _lon = _lon % 360
                        if _lon_bounds is not None:
                            _lon_bounds = tuple(l % 360 for l in _lon_bounds)
                except Exception:
                    pass

            if lat is not None and _lon is not None:
                ds = ds.sel(lat=lat, lon=_lon, method="nearest")
            elif lat_bounds is not None and _lon_bounds is not None:
                ds = ds.sel(lat=slice(lat_bounds[0], lat_bounds[1]), lon=slice(_lon_bounds[0], _lon_bounds[1]))
            elif polygon_path is not None:
                if gpd is None:
                    raise ImportError("geopandas is required for polygon masking in download_pangeo()")
                gdf = gpd.read_file(str(polygon_path)).to_crs("EPSG:4326")
                poly = gdf.geometry.unary_union

                minx, miny, maxx, maxy = poly.bounds
                ds = ds.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))

                lon2d, lat2d = np.meshgrid(ds.lon, ds.lat)
                points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()))
                mask = np.array([poly.contains(pt) for pt in points]).reshape(lat2d.shape)
                ds = ds.where(mask)

            ds.to_netcdf(dir_out / filename)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")


def extract_vars_from_files(files: Iterable[Union[str, Path]], start_date: str, end_date: str, path_out: Union[str, Path]):
    """
    Robust CMIP6 NetCDF merger for multiple calendars — using CFDatetimeCoder.
    This is slow but robust.
    """
    from tqdm import tqdm

    all_dfs: List[pd.DataFrame] = []
    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    for file in tqdm(list(files), desc="Processing"):
        try:
            ds = xr.open_dataset(str(file), decode_times=time_coder)

            varnames = [v for v in ds.data_vars if {"time", "lat", "lon"}.intersection(ds[v].dims)]
            for var in varnames:
                arr = ds[var]
                time = ds["time"].values

                if isinstance(time[0], np.datetime64):
                    times = pd.to_datetime(time)
                    mask = (times >= start_date) & (times <= end_date)
                else:
                    times = time
                    mask = np.array([(t >= cftime_date(start_date, t)) and (t <= cftime_date(end_date, t)) for t in time])

                values = arr.values[mask]
                filtered_times = np.array(times)[mask]

                lon = ds["lon"].values.item() if ds["lon"].size == 1 else ds["lon"].values
                lat = ds["lat"].values.item() if ds["lat"].size == 1 else ds["lat"].values

                parts = Path(file).stem.split("_")
                model = parts[1] if len(parts) > 1 else ""
                ssp = parts[2] if len(parts) > 2 else ""

                df1 = pd.DataFrame(
                    {
                        "date": filtered_times,
                        "lon": lon,
                        "lat": lat,
                        "value": values,
                        "var": var,
                        "model": model,
                        "ssp": ssp,
                    }
                )
                all_dfs.append(df1)

        except Exception as e:
            print(f"Failed: {file} — {e}")

    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    if all_dfs:
        out_df = pd.concat(all_dfs, ignore_index=True)
        if not np.issubdtype(out_df["date"].dtype, np.datetime64):
            out_df["date"] = out_df["date"].astype(str)
        out_df.to_csv(path_out, index=False)
        print(f"Saved to {path_out}")
    else:
        print("No valid data extracted.")


def cftime_date(string_date: str, sample_cftime):
    """
    Convert YYYY-MM-DD to same cftime type as sample_cftime.
    """
    import cftime

    y, m, d = map(int, string_date.split("-"))
    if isinstance(sample_cftime, cftime.DatetimeNoLeap):
        return cftime.DatetimeNoLeap(y, m, d)
    if isinstance(sample_cftime, cftime.Datetime360Day):
        return cftime.Datetime360Day(y, m, min(d, 30))
    return cftime.DatetimeProlepticGregorian(y, m, d)
