from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

from dapper.utils import sampling
from dapper.domains.domain import Domain

LonWrap = Literal["auto", "0_360", "-180_180"]

def sample_landuse_timeseries(
    src_path: str | Path,
    df_loc: pd.DataFrame,
    out_path: str | Path,
    *,
    gid_col: str = "gid",
    lat_col: str = "lat",
    lon_col: str = "lon",
    weight_col: str = "weight",
    default_weight: float = 1.0,
    lat_dim: str = "lsmlat",
    lon_dim: str = "lsmlon",
    lat_var: str | None = "LATIXY",
    lon_var: str | None = "LONGXY",
    lon_wrap: sampling.LonWrap = "auto",
    output_lon_wrap: sampling.LonWrap | None = None,
    decode_times: bool = False,
    chunks: dict | None = None,
    compress: bool = True,
    complevel: int = 4,
    vars_include: Sequence[str] | None = None,
    vars_drop: Sequence[str] | None = None,
    sampling_method: Literal["nearest", "zonal"] = "nearest",
    targets: "gpd.GeoDataFrame | None" = None,
    agg_policy: dict[str, str] | None = None,
    write_zonal_mapping: bool = True,
) -> tuple[Path, pd.DataFrame]:
    """
    Sample a landuse time series dataset either by nearest-point sampling (existing behavior)
    or by zonal intersection of target polygons with the source grid.

    Returns (out_path, df_summary)
      - nearest: df_summary is df_loc aligned to sampled cells (includes i_lat/i_lon if available)
      - zonal  : df_summary includes sample_ncells and sample_area_total_m2 per gid
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import xarray as xr

    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_loc = df_loc.copy()
    if gid_col not in df_loc.columns:
        raise KeyError(f"df_loc must include column {gid_col!r}")

    if weight_col not in df_loc.columns:
        df_loc[weight_col] = float(default_weight)
    else:
        df_loc[weight_col] = df_loc[weight_col].fillna(float(default_weight))

    df_loc[gid_col] = df_loc[gid_col].astype(str)

    ds_src = xr.open_dataset(src_path, decode_times=decode_times, chunks=chunks)

    def _write_nc(ds_out: xr.Dataset, path: Path) -> None:
        if not compress:
            ds_out.to_netcdf(path, mode="w")
            return

        encoding = {}
        for v in ds_out.data_vars:
            encoding[v] = {"zlib": True, "complevel": int(complevel)}
        ds_out.to_netcdf(path, mode="w", encoding=encoding)

    if sampling_method == "nearest":
        df_cells = sampling.points_to_nearest_cells(
            ds_src,
            df_loc,
            lat_col=lat_col,
            lon_col=lon_col,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            lat_var=lat_var,
            lon_var=lon_var,
            lon_wrap=lon_wrap,
        )

        ds_out = sampling.sample_gridded_dataset_points(
            ds_src,
            df_cells,
            lat_col=lat_col,
            lon_col=lon_col,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            lat_var=lat_var,
            lon_var=lon_var,
            lon_wrap=lon_wrap,
            vars_include=vars_include,
            vars_drop=vars_drop,
            method="nearest",
        )

        if output_lon_wrap is not None and "LONGXY" in ds_out:
            lon_vals = ds_out["LONGXY"].values.reshape(-1)
            lon_vals2 = np.array([sampling.normalize_lon(float(v), output_lon_wrap) for v in lon_vals], dtype=float)
            spec = sampling.infer_latlon_spec(ds_out, lon_wrap=lon_wrap)
            ds_out["LONGXY"] = xr.DataArray(
                lon_vals2.reshape((ds_out.sizes[spec.lat_dim], ds_out.sizes[spec.lon_dim])),
                dims=(spec.lat_dim, spec.lon_dim),
                attrs=dict(ds_out["LONGXY"].attrs),
            )
            ds_out.attrs["output_lon_wrap"] = str(output_lon_wrap)

        ds_out.attrs["dapper_sampling_method"] = "nearest"
        _write_nc(ds_out, out_path)
        return out_path, df_cells

    if sampling_method != "zonal":
        raise ValueError(f"Unknown sampling_method={sampling_method!r}")

    import geopandas as gpd
    from dapper.utils import zonal

    if targets is None:
        raise ValueError("sampling_method='zonal' requires targets=GeoDataFrame with columns ['gid','geometry'].")

    if gid_col not in targets.columns or "geometry" not in targets.columns:
        raise KeyError(f"targets must include columns {gid_col!r} and 'geometry'")

    tgt = targets[[gid_col, "geometry"]].copy()
    tgt[gid_col] = tgt[gid_col].astype(str)
    if tgt.crs is None:
        tgt = tgt.set_crs("EPSG:4326")
    else:
        tgt = tgt.to_crs("EPSG:4326")

    zw = zonal.intersect_weights_rectilinear(
        ds_src,
        tgt.rename(columns={gid_col: "gid"}),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
    )

    ap = dict(agg_policy or {})
    include_set = set(vars_include) if vars_include is not None else None
    drop_set = set(vars_drop) if vars_drop is not None else set()

    for v in ds_src.data_vars:
        if include_set is not None and v not in include_set:
            continue
        if v in drop_set:
            continue
        if v in ap:
            continue

        # Default for mask-like variables (typically boolean-ish 0/1)
        if v == "mask" or "MASK" in v:
            ap[v] = "wmean_threshold"
            continue

        kind = ds_src[v].dtype.kind
        ap[v] = "wmode" if kind in {"i", "u", "b"} else "wmean"

    out = zonal.sample_gridded_dataset_polygons(
        ds_src,
        tgt.rename(columns={gid_col: "gid"}),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        lat_var=lat_var,
        lon_var=lon_var,
        lon_wrap=lon_wrap,
        vars_include=vars_include,
        vars_drop=vars_drop,
        agg_policy=ap,
        weights=zw,
    )

    order = tgt[gid_col].astype(str).tolist()
    df0 = df_loc.set_index(gid_col).loc[order].reset_index()

    spec = sampling.infer_latlon_spec(ds_src, lon_wrap=lon_wrap)
    od_lat_dim, od_lon_dim = spec.lat_dim, spec.lon_dim
    n = len(order)

    if od_lat_dim not in out.dims or od_lon_dim not in out.dims:
        out = out.expand_dims({od_lat_dim: np.arange(n, dtype=np.int32), od_lon_dim: np.arange(1, dtype=np.int32)})

    if "LATIXY" in out:
        out["LATIXY"] = xr.DataArray(
            df0[lat_col].to_numpy(dtype=np.float32).reshape((n, 1)),
            dims=(od_lat_dim, od_lon_dim),
            attrs=dict(out["LATIXY"].attrs),
        )
    if "LONGXY" in out:
        lon_vals = df0[lon_col].to_numpy(dtype=np.float64)
        if output_lon_wrap is not None:
            lon_vals = np.array([sampling.normalize_lon(float(v), output_lon_wrap) for v in lon_vals], dtype=np.float64)
            out.attrs["output_lon_wrap"] = str(output_lon_wrap)

        out["LONGXY"] = xr.DataArray(
            lon_vals.astype(np.float32).reshape((n, 1)),
            dims=(od_lat_dim, od_lon_dim),
            attrs=dict(out["LONGXY"].attrs),
        )

    ncells = np.array([len(zw.by_gid[str(g)]) for g in order], dtype=np.int32)
    area_m2 = np.array([zw.by_gid[str(g)]["intersect_area_m2"].sum() for g in order], dtype=np.float64)

    out["sample_ncells"] = xr.DataArray(ncells.reshape((n, 1)), dims=(od_lat_dim, od_lon_dim))
    out["sample_area_total_m2"] = xr.DataArray(area_m2.reshape((n, 1)).astype(np.float32), dims=(od_lat_dim, od_lon_dim))

    out.attrs["dapper_sampling_method"] = "zonal"
    out.attrs["dapper_sampling_equal_area_crs"] = zw.equal_area_crs
    out.attrs["dapper_sampling_lon_wrap_native"] = zw.lon_wrap

    if write_zonal_mapping:
        rows = []
        for g in order:
            dfw = zw.by_gid[str(g)].copy()
            dfw.insert(0, gid_col, str(g))
            rows.append(dfw)
        dfw_all = pd.concat(rows, ignore_index=True)
        csv_path = out_path.with_suffix(out_path.suffix + ".zonal_weights.csv")
        dfw_all.to_csv(csv_path, index=False)

    _write_nc(out, out_path)

    df_summary = pd.DataFrame({gid_col: order, "sample_ncells": ncells, "sample_area_total_m2": area_m2})
    return out_path, df_summary


def export_landuse_timeseries(
    domain: Domain,
    *,
    src_path: str | Path,
    out_dir: str | Path,
    filename: str = "landuse_timeseries.nc",
    overwrite: bool = False,
    append_attrs=None,
    **kwargs,
) -> dict[str, Path]:
    """
    Export landuse timeseries NetCDF(s) for a Domain.

    Output layout:
      - domain.mode='cellset': <out_dir>/landuse_timeseries.nc
      - domain.mode='sites'  : <out_dir>/<gid>/landuse_timeseries.nc

    Returns:
      dict[run_id, output_path]

    Notes:
        This is mainly useful for 'sites' mode (and is still supported for cellset).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for run_id, run_dom in domain.iter_runs():
        run_id = str(run_id)
        df_loc = run_dom.to_df_loc()

        run_out_dir = (out_dir / run_id) if domain.mode == "sites" else out_dir
        run_out_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_out_dir / filename

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists (overwrite=False).")

        if kwargs.get("sampling_method", "nearest") == "zonal":
            kwargs.setdefault("targets", run_dom.cells[["gid", "geometry"]].copy())

        path_written, _df_cells = sample_landuse_timeseries(
            src_path=src_path,
            df_loc=df_loc,
            out_path=out_path,
            **kwargs,
        )
        outputs[run_id] = Path(path_written)

    return outputs
