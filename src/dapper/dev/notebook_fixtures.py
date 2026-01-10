from pathlib import Path
import numpy as np
import xarray as xr

def _to_ds_lon(lon, lon_sample):
    """Convert lon bounds to dataset convention if dataset uses 0..360."""
    lon_sample = float(np.nanmean(lon_sample))
    if lon_sample > 180:  # dataset likely 0..360
        return (lon + 360) % 360
    return lon

def crop_to_bbox(in_nc, out_nc, lat_rng, lon_rng, buffer=0.01, time_sel=None, drop_vars=None):
    lat0, lat1 = lat_rng
    lon0, lon1 = lon_rng
    lat0, lat1 = (min(lat0, lat1) - buffer, max(lat0, lat1) + buffer)
    lon0, lon1 = (min(lon0, lon1) - buffer, max(lon0, lon1) + buffer)

    ds = xr.open_dataset(in_nc, decode_times=False)

    # --- Case A: 2D LATIXY/LONGXY ---
    if "LATIXY" in ds and "LONGXY" in ds and len(ds["LATIXY"].dims) == 2:
        lat2d = ds["LATIXY"].values
        lon2d = ds["LONGXY"].values
        lon0c = _to_ds_lon(lon0, lon2d)
        lon1c = _to_ds_lon(lon1, lon2d)

        mask = (lat2d >= lat0) & (lat2d <= lat1) & (lon2d >= lon0c) & (lon2d <= lon1c)
        if not np.any(mask):
            raise ValueError("No grid cells found in bbox; check lon convention / bounds.")

        i, j = np.where(mask)
        d0, d1 = ds["LATIXY"].dims
        ds2 = ds.isel({d0: slice(i.min(), i.max() + 1), d1: slice(j.min(), j.max() + 1)})

    # --- Case B: 1D lat/lon coords ---
    elif ("lat" in ds.coords and "lon" in ds.coords and ds["lat"].ndim == 1 and ds["lon"].ndim == 1):
        lon0c = _to_ds_lon(lon0, ds["lon"].values)
        lon1c = _to_ds_lon(lon1, ds["lon"].values)
        ds2 = ds.sel(lat=slice(lat0, lat1), lon=slice(lon0c, lon1c))

    # --- Case C: unstructured gridcell with 1D LATIXY/LONGXY ---
    elif "LATIXY" in ds and "LONGXY" in ds and ds["LATIXY"].ndim == 1:
        lat1d = ds["LATIXY"].values
        lon1d = ds["LONGXY"].values
        lon0c = _to_ds_lon(lon0, lon1d)
        lon1c = _to_ds_lon(lon1, lon1d)

        mask = (lat1d >= lat0) & (lat1d <= lat1) & (lon1d >= lon0c) & (lon1d <= lon1c)
        if not np.any(mask):
            raise ValueError("No grid cells found in bbox; check lon convention / bounds.")
        dim = ds["LATIXY"].dims[0]
        idx = np.where(mask)[0]
        ds2 = ds.isel({dim: slice(idx.min(), idx.max() + 1)})

    else:
        raise ValueError("Could not identify lat/lon coordinate structure in this file.")

    # Optional time subset (handy for landuse)
    if time_sel is not None and "time" in ds2.dims:
        ds2 = ds2.sel(time=time_sel)

    # Optional variable slimming
    if drop_vars:
        ds2 = ds2.drop_vars([v for v in drop_vars if v in ds2])

    # Write with compression
    enc = {v: {"zlib": True, "complevel": 4} for v in ds2.data_vars}
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds2.to_netcdf(out_nc, encoding=enc)
    return out_nc

# Example bbox for your 3 cells
LAT_RNG = (69., 70)
LON_RNG = (-152., -149)

# Surface: just crop
crop_to_bbox(r"X:\Research\NGEE Arctic\3. Surface Files\data\surfdata_0.5x0.5_simyr1850_c200609_with_TOP.nc", 
             r"X:\Research\NGEE Arctic\dapper\docs\data\end-to-end\surf_psuedoglobal.nc", LAT_RNG, LON_RNG)

# Landuse: crop + maybe trim time to reduce size
crop_to_bbox(r"X:\Research\NGEE Arctic\3. Surface Files\surfdata_map\landuse.timeseries_0.125x0.125_hist_simyr1850-2015_c191004.nc", 
             r"X:\Research\NGEE Arctic\dapper\docs\data\end-to-end\landuse_psuedoglobal.nc",
             LAT_RNG, LON_RNG)
            #  time_sel=slice(0, 12*10))  # first 10 years only
