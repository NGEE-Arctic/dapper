# Functions either in development or used for one-off data stuff. Created by JPS.
# Note that there may be imports here that aren't included in environment.yml file.

import ee
import geopandas as gpd

from dapper.utils import utils
from dapper.met import era5land as e5l
from dapper.met import era5land as gridded

ee.Initialize(project='ee-jonschwenk')


# Load Kuparuk gage watershed polygon; convert to ee object
kup = gpd.read_file(utils._DATA_DIR / 'Kuparuk_gageshed.shp')
kup = kup.buffer(0) # A hack to "fix" invalid geometries--not needed here but just demonstrating
kup_pgon = ee.Geometry.Polygon(list(kup.geometry.values[0].exterior.coords))

# Point to the ERA5-Land hourly polygon grid
e5lh_grid = ee.FeatureCollection('projects/ee-jonschwenk/assets/E3SM/e5lh_grid')

# Intersect and return the e5lh ids
intersecting_e5l_grids = e5lh_grid.filter(ee.Filter.intersects('.geo', kup_pgon))
# intersecting_e5l_grids = e5lh_grid.filterBounds(kup_pgon)
pids = intersecting_e5l_grids.aggregate_array('pids').getInfo()

# Now we select the E5LH grid cells we want to sample
sample_these_cells = e5lh_grid.filter(ee.Filter.inList('pids', pids))

params = {
    'start_date' : '1950-01-01', # YYYY-MM-DD
    'end_date' : '1957-01-01', # YYYY-MM-DD
    'geometries' : sample_these_cells, # Dictionary of {'name' : (lat, lon)} for all points to sample
    'geometry_id_field' : 'pids', # The e5lh_grid Asset calls this field "pids", so we must specify that
    'gee_bands' : 'elm', # Select ELM-required bands
    'gee_years_per_task' : 1, # Optional parameter; default is 5. For lots of points, you may want to reduce this for smaller GEE Tasks (but more of them)
    'gee_scale' : 'native',
    'gdrive_folder' : 'ngee_test_cells', # Which folder to store on your GDrive; will be created if not exists
    'job_name' : 'cell_test',
}
df_loc = e5l.sample_e5lh(params, skip_tasks=True)

csv_directory = r'X:\Research\NGEE Arctic\dapper\data\celltesting'
write_directory = r'X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted3'
append_attrs= {'note' : 'testing for bugs'}
exp = gridded.e5lh_to_elm_class(csv_directory, write_directory, df_loc, append_attrs=append_attrs)
exp.run(output_mode='site_dirs', pack_scope='global')
# gridded.e5lh_to_elm_gridded(csv_directory, write_directory, df_loc, append_attrs=append_attrs)

# import xarray as xr
# path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\clmforc.GSWP3.c2011.0.5x0.5.Prec.1901-01.nc"
# path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\era5\Daymet_ERA5.1km_QBOT_1980-2023_z01.nc"
# path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\GSWP3_FLDS_1901-2014_z14.nc"
# pathf = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\GSWP3_daymet4_FLDS_1980-2014_z01_gridded.nc"
# pathm = r"X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted\ERA5_QBOT_1950-1956_z1.nc"
# dsm = xr.open_dataset(pathm)
# dsf = xr.open_dataset(pathf)


import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Helper: nearest site by lat/lon (lon input can be -180..180 or 0..360) ---
def nearest_site_index(ds, lat0, lon0):
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Dataset lacks 'lat'/'lon' coords on dimension 'n'.")
    lat_arr = ds["lat"].values
    lon_arr = ds["lon"].values  # expected 0–360
    lon0_0360 = lon0 % 360.0

    # handle wrap-around near 0/360
    dlon = np.abs(lon_arr - lon0_0360)
    dlon = np.minimum(dlon, 360.0 - dlon)
    dlat = np.abs(lat_arr - lat0)

    w = np.cos(np.deg2rad(np.clip(lat0, -89.9, 89.9)))
    dist = np.hypot(dlat, dlon * max(w, 0.1))
    return int(np.argmin(dist))

def ts_at(ds, var, lat, lon):
    i = nearest_site_index(ds, lat, lon)
    ts = ds[var].transpose("time", "n").isel(n=i)
    plat = float(ds["lat"].values[i])
    plon0360 = float(ds["lon"].values[i])
    plon180 = ((plon0360 + 180) % 360) - 180
    return ts, i, plat, plon180

def coerce_time_to_numpy(ds, time_name="time"):
    """Convert cftime time coord to numpy.datetime64[ns] for plotting."""
    t = ds[time_name].values
    if isinstance(t[0], np.datetime64):
        return ds  # already numpy datetimes
    # Build ISO timestamps from cftime and cast to datetime64[ns]
    iso = []
    for tt in t:
        # cftime often has hour/min/sec; guard if not present
        hh = getattr(tt, "hour", 0)
        mm = getattr(tt, "minute", 0)
        ss = getattr(tt, "second", 0)
        iso.append(f"{tt.year:04d}-{tt.month:02d}-{tt.day:02d}T{hh:02d}:{mm:02d}:{ss:02d}")
    t_np = np.array(iso, dtype="datetime64[ns]")
    return ds.assign_coords({time_name: t_np})


path = r"X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted2\ERA5_PRECTmms_1950-1956.nc"
var  = "PRECTmms"   # change if needed

# --- Open & normalize dims/coords ---
ds = xr.open_dataset(path, decode_cf=True, engine="netcdf4")

# Use DTIME if present; rename to 'time' for convenience
time_dim = "DTIME" if "DTIME" in ds.dims else "time"
if time_dim != "time":
    ds = ds.rename({time_dim: "time"})
    try:
        import nc_time_axis  # noqa: F401
    except Exception:
        ds = coerce_time_to_numpy(ds, "time")


# Promote LATIXY/LONGXY to coords on 'n' (if present)
if "LATIXY" in ds and "LONGXY" in ds:
    ds = ds.assign_coords(
        lat=("n", ds["LATIXY"].values),
        lon=("n", ds["LONGXY"].values),
    )

# DataArray of interest, ensured shape (time, n)
da = ds[var]
if set(da.dims) != {"time", "n"}:
    # transpose if needed
    if "time" in da.dims and "n" in da.dims:
        da = da.transpose("time", "n")
    else:
        raise ValueError(f"{var} dims {da.dims} do not include both 'time' and 'n'")

nt, n = da.shape
print("dims:", dict(ds.dims))
print(f"sites (n): {n}   timesteps (time): {nt}")

# --- Basic sanity: how many sites have any data? ---
non_empty_mask = (~da.isnull()).any("time")
n_nonempty = int(non_empty_mask.sum().item())
print("non-empty sites:", n_nonempty, "/", n)

# compression sanity (on-disk int16 packed)
raw_bytes = nt * n * 2  # int16 bytes
print("raw GiB (int16):", raw_bytes/1024**3, "  file size GiB:", os.path.getsize(path)/1024**3)



# --- 1) Plot time series at your requested points (nearest sites) ---
points = [
    (68.0, -150.0),
    (67.5, -149.0),
    (69.0, -152.0),
]

plt.figure(figsize=(10,4))
labels = []
for plat, plon in points:
    ts, i, plat_s, plon180_s = ts_at(ds, var, plat, plon)
    # If the chosen site is empty, skip plotting (line would be empty)
    if np.isfinite(ts.values).any():
        ts.plot.line(add_legend=False)
        labels.append(f"site#{i} ({plat_s:.3f}, {plon180_s:.3f})")
    else:
        labels.append(f"site#{i} (no data)")
plt.legend(labels, loc="upper right")
plt.title(f"{var} time series at nearest sites")
plt.tight_layout()
plt.show()

# --- 2) Also plot a few known non-empty sites so you definitely see lines ---
if n_nonempty > 0:
    idx_nonempty = np.where(non_empty_mask.values)[0]
    take = idx_nonempty[:min(3, len(idx_nonempty))]
    plt.figure(figsize=(10,4))
    labels2 = []
    da_tn = da  # (time, n)
    for i in take:
        ts = da_tn.isel(n=i)
        ts.plot.line(add_legend=False)
        plat = float(ds["lat"].values[i]) if "lat" in ds.coords else np.nan
        plon0360 = float(ds["lon"].values[i]) if "lon" in ds.coords else np.nan
        plon180 = ((plon0360 + 180) % 360) - 180 if np.isfinite(plon0360) else np.nan
        labels2.append(f"site#{i} ({plat:.3f}, {plon180:.3f})")
    plt.legend(labels2, loc="upper right")
    plt.title(f"{var} time series at non-empty sites")
    plt.tight_layout()
    plt.show()
else:
    print("No non-empty sites found for plotting.")

# --- 3) “Spatial” snapshot: scatter the sites at selected times ---
times_to_show = [
    ds["time"].values[0],
    ds["time"].values[len(ds["time"])//2],
    ds["time"].values[-1],
]
if "lat" in ds.coords and "lon" in ds.coords:
    lats = ds["lat"].values
    lons0360 = ds["lon"].values
    lons180 = ((lons0360 + 180) % 360) - 180
    for t in times_to_show:
        vals = da.sel(time=t).values  # shape (n,)
        mask = np.isfinite(vals)
        plt.figure(figsize=(7,5))
        sc = plt.scatter(lons180[mask], lats[mask], c=vals[mask], s=20)
        plt.colorbar(sc, label=var)
        tlabel = np.datetime_as_string(t, unit="h") if np.issubdtype(ds["time"].dtype, np.datetime64) else str(t)
        plt.title(f"{var} @ {tlabel} (site scatter)")
        plt.xlabel("Longitude (°E, -180..180)")
        plt.ylabel("Latitude (°N)")
        plt.tight_layout()
        plt.show()
else:
    print("Skipping scatter maps: dataset lacks 'lat'/'lon' coords.")

# --- 4) Quick value diagnostics ---
print(f"{var} min/mean/max (decoded floats):",
      float(da.min().values),
      float(da.mean().values),
      float(da.max().values))

# --- 5) Confirm on-disk packing (int16 + attrs) ---
raw = xr.open_dataset(path, decode_cf=False, engine="netcdf4")
print("On-disk dtype (decode_cf=False):", raw[var].dtype)
print("On-disk attrs:",
      {k: raw[var].attrs.get(k) for k in ("scale_factor","add_offset","_FillValue")})
raw.close()



