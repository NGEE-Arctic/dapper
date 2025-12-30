from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import numpy as np
import xarray as xr

from dapper.utils import sampling

LonWrap = Literal["auto", "0_360", "-180_180"]


def sample_landuse_timeseries(
    src_path: str | Path,
    df_loc: pd.DataFrame,
    out_path: str | Path,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    weight_col: str = "weight",
    default_weight: float = 1.0,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = "LATIXY",
    lon_var: str | None = "LONGXY",
    lon_wrap: LonWrap = "auto",
    decode_times: bool = False,
    chunks: dict | None = None,
    compress: bool = True,
    complevel: int = 4,
    vars_include: Sequence[str] | None = None,
    vars_drop: Sequence[str] | None = None,
    output_lon_wrap: LonWrap | None = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Sample a global landuse-timeseries file at each location in df_loc and write
    a point/domain landuse.timeseries file.

    Adds provenance variables to the output file:
      sample_input_lat/lon, sample_input_lon_normalized,
      sample_i_lat/i_lon, sample_lat_cell, sample_lon_cell_native,
      sample_lon_cell_output, sample_weight

    And provenance attrs:
      dapper_sampling_method, dapper_sampling_lon_wrap_native, dapper_sampling_source
    """
    def _lon_to_0_360(v):
        return float(((v % 360.0) + 360.0) % 360.0)

    def _lon_to_m180_180(v):
        return float(((v + 180.0) % 360.0) - 180.0)

    def _convert_lon_scalar(v: float, wrap: LonWrap | None) -> float:
        if wrap == "0_360":
            return _lon_to_0_360(v)
        if wrap == "-180_180":
            return _lon_to_m180_180(v)
        return float(v)

    df_loc = sampling.ensure_weight(df_loc, weight_col=weight_col, default_weight=default_weight)

    src_path = Path(src_path)
    ds = xr.open_dataset(src_path, decode_times=decode_times, chunks=chunks)

    # Infer native lon convention once (for provenance attrs)
    spec_native = sampling.infer_latlon_spec(
        ds,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )
    native_lon_wrap = spec_native.lon_wrap  # "0_360" or "-180_180"

    # Compute nearest-cell mapping (indices + cell centers)
    df_cells = sampling.points_to_nearest_cells(
        ds,
        df_loc,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )

    # Sample the dataset
    out = sampling.sample_gridded_dataset_points(
        ds,
        df_loc,
        lat_col=lat_col,
        lon_col=lon_col,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
        method="nearest",
        vars_include=vars_include,
        vars_drop=vars_drop,
    )

    # Optional: standardize LONGXY in the OUTPUT file
    if output_lon_wrap == "0_360" and "LONGXY" in out:
        out["LONGXY"] = sampling._to_0_360(out["LONGXY"])
    elif output_lon_wrap == "-180_180" and "LONGXY" in out:
        out["LONGXY"] = ((out["LONGXY"] + 180.0) % 360.0) - 180.0

    # -------------------- provenance variables (dims: lsmlat, lsmlon) --------------------
    # Ensure df_cells is aligned to df_loc row order (important if index not 0..N-1)
    df_cells_aligned = df_cells.reindex(df_loc.index)

    n = len(df_cells_aligned)
    # Output convention from sampler: lat_dim has length N; lon_dim has length 1
    # So provenance vars are (N, 1).
    def _var_2d(name: str, values, dtype=None):
        arr = np.asarray(values)
        if dtype is not None:
            arr = arr.astype(dtype)
        arr2 = arr.reshape((n, 1))
        out[name] = xr.DataArray(arr2, dims=(lat_dim, lon_dim))

    _var_2d("sample_input_lat", df_cells_aligned[lat_col].to_numpy(dtype="float64"))
    _var_2d("sample_input_lon", df_cells_aligned[lon_col].to_numpy(dtype="float64"))
    _var_2d("sample_input_lon_normalized", df_cells_aligned["lon_normalized"].to_numpy(dtype="float64"))
    _var_2d("sample_i_lat", df_cells_aligned["i_lat"].to_numpy(dtype="int32"), dtype="int32")
    _var_2d("sample_i_lon", df_cells_aligned["i_lon"].to_numpy(dtype="int32"), dtype="int32")
    _var_2d("sample_lat_cell", df_cells_aligned["lat_cell"].to_numpy(dtype="float64"))
    _var_2d("sample_lon_cell_native", df_cells_aligned["lon_cell"].to_numpy(dtype="float64"))

    # If you changed LONGXY in the output, also store an "output convention" lon cell
    lon_cell_out = np.array(
        [_convert_lon_scalar(v, output_lon_wrap) for v in df_cells_aligned["lon_cell"].to_numpy(dtype="float64")],
        dtype="float64",
    )
    _var_2d("sample_lon_cell_output", lon_cell_out)

    _var_2d("sample_weight", df_cells_aligned[weight_col].to_numpy(dtype="float64"))

    # Add light attrs for provenance vars
    out["sample_input_lat"].attrs.update({"long_name": "input latitude used for sampling", "units": "degrees_north"})
    out["sample_input_lon"].attrs.update({"long_name": "input longitude used for sampling (raw)", "units": "degrees_east"})
    out["sample_input_lon_normalized"].attrs.update({"long_name": "input longitude normalized to native grid convention", "units": "degrees_east"})
    out["sample_i_lat"].attrs.update({"long_name": f"selected {lat_dim} index in source grid", "units": "index"})
    out["sample_i_lon"].attrs.update({"long_name": f"selected {lon_dim} index in source grid", "units": "index"})
    out["sample_lat_cell"].attrs.update({"long_name": "latitude of selected source grid cell center", "units": "degrees_north"})
    out["sample_lon_cell_native"].attrs.update({"long_name": "longitude of selected source grid cell center (native convention)", "units": "degrees_east"})
    out["sample_lon_cell_output"].attrs.update({"long_name": "longitude of selected source grid cell center (output convention)", "units": "degrees_east"})
    out["sample_weight"].attrs.update({"long_name": "domain weight for this sampled cell", "units": "1"})

    # Global attrs (provenance breadcrumbs)
    out.attrs["dapper_sampling_method"] = "nearest"
    out.attrs["dapper_sampling_lon_wrap_native"] = native_lon_wrap
    out.attrs["dapper_sampling_source"] = str(src_path)
    out.attrs["dapper_sampling_output_lon_wrap"] = str(output_lon_wrap) if output_lon_wrap is not None else "native"

    out_path = sampling.write_netcdf(out, out_path, compress=compress, complevel=complevel)
    return out_path, df_cells
