import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from dapper.utils import zonal
from dapper.landuse.landuse import sample_landuse_timeseries

# ---- inputs you set ----
GLOBAL_LANDUSE = r"X:\Research\NGEE Arctic\3. Surface Files\surfdata_map\landuse.timeseries_0.125x0.125_hist_simyr1850-2015_c191004.nc"
OUT_NEAREST    = r"X:\Research\NGEE Arctic\3. Surface Files\outlanduse_nearest_test.nc"
OUT_ZONAL      = r"X:\Research\NGEE Arctic\3. Surface Files\outlanduse_zonal_test.nc"

# Either use your real Domain...
# df_loc  = domain.to_df_loc()                       # must include gid, lat, lon
# targets = domain.cells[["gid","geometry"]].copy()  # polygons, EPSG:4326

# ...or quick “toy” single polygon around Toolik for a smoke test:
gid = "cell_001"
lon0, lat0 = -149.6, 68.63
halfwidth_deg = 2.0  # make big enough to overlap many source cells

targets = gpd.GeoDataFrame(
    {"gid": [gid], "geometry": [box(lon0-halfwidth_deg, lat0-halfwidth_deg, lon0+halfwidth_deg, lat0+halfwidth_deg)]},
    crs="EPSG:4326",
)
rp = targets.geometry.iloc[0].representative_point()
df_loc = pd.DataFrame({"gid": [gid], "lon": [rp.x], "lat": [rp.y], "weight": [1.0]})

# ---- variables to test ----
var_mask = "PFTDATA_MASK"
var_2d   = "PCT_LAKE"
var_4d   = "PCT_NAT_PFT"

# Include a couple metadata vars to ensure they survive (optional)
vars_include = {var_mask, var_2d, var_4d, "LATIXY", "LONGXY", "YEAR", "input_pftdata_filename"}

# -------------------------
# 1) nearest run (control)
# -------------------------
p_near, df_near = sample_landuse_timeseries(
    src_path=GLOBAL_LANDUSE,
    df_loc=df_loc,
    out_path=OUT_NEAREST,
    vars_include=vars_include,
    sampling_method="nearest",
    lon_wrap="auto",
    compress=True,
)

# -------------------------
# 2) zonal run (the thing we care about)
# -------------------------
p_zonal, df_sum = sample_landuse_timeseries(
    src_path=GLOBAL_LANDUSE,
    df_loc=df_loc,
    out_path=OUT_ZONAL,
    vars_include=vars_include,
    sampling_method="zonal",
    targets=targets,
    lon_wrap="auto",
    agg_policy={var_mask: "wmean_threshold"},  # critical test: mask behavior
    write_zonal_mapping=True,
    compress=True,
)

print("nearest:", p_near)
print("zonal  :", p_zonal)
print(df_sum)

# -------------------------
# 3) read outputs and compare a few expectations
# -------------------------
near = xr.open_dataset(p_near)
zon  = xr.open_dataset(p_zonal)

# infer dims from output (usually lsmlat/lsmlon)
lat_dim = [d for d in zon.dims if d.lower().endswith("lat")][0]
lon_dim = [d for d in zon.dims if d.lower().endswith("lon")][0]

def at00(ds, v):
    da = ds[v]
    # handle 2D and 4D cases (pick first time/natpft for quick compare)
    slicer = {}
    for d in da.dims:
        if d == lat_dim: slicer[d] = 0
        elif d == lon_dim: slicer[d] = 0
        elif d == "time": slicer[d] = 0
        elif d == "natpft": slicer[d] = 0
        elif d == "numurbl": slicer[d] = 0
    return da.isel(slicer).item()

print("\n--- value spot checks (first indices) ---")
print("nearest", var_2d, at00(near, var_2d))
print("zonal  ", var_2d, at00(zon,  var_2d))

print("nearest", var_mask, at00(near, var_mask))
print("zonal  ", var_mask, at00(zon,  var_mask))

print("nearest", var_4d, at00(near, var_4d))
print("zonal  ", var_4d, at00(zon,  var_4d))

# LAT/LON should equal df_loc representative point (for zonal we overwrite LATIXY/LONGXY from df_loc)
lat_out = zon["LATIXY"].isel({lat_dim: 0, lon_dim: 0}).item()
lon_out = zon["LONGXY"].isel({lat_dim: 0, lon_dim: 0}).item()
print("\n--- derived coords check ---")
print("LATIXY out vs df_loc:", lat_out, df_loc["lat"].iloc[0])
print("LONGXY out vs df_loc:", lon_out, df_loc["lon"].iloc[0])

# Non-spatial vars should survive (if included)
print("\n--- non-spatial vars present? ---")
print("YEAR in zonal:", "YEAR" in zon.data_vars, "dims:", zon["YEAR"].dims if "YEAR" in zon else None)
print("input_pftdata_filename in zonal:", "input_pftdata_filename" in zon.data_vars)

# -------------------------
# 4) prove mask aggregation is doing what we think (weight-by-category)
# -------------------------
ds_src = xr.open_dataset(GLOBAL_LANDUSE)
zw = zonal.intersect_weights_rectilinear(ds_src, targets)

wdf = zw.by_gid[gid]
i_idx = xr.DataArray(wdf["i_lat"].to_numpy(int), dims="cell")
j_idx = xr.DataArray(wdf["i_lon"].to_numpy(int), dims="cell")
w = xr.DataArray(wdf["weight"].to_numpy(float), dims="cell")

# infer source dims
from dapper.utils import sampling
spec = sampling.infer_latlon_spec(ds_src, lon_wrap="auto")
da_sel = ds_src[var_mask].isel({spec.lat_dim: i_idx, spec.lon_dim: j_idx}).values

vals = np.asarray(da_sel).reshape(-1)
weights = w.values

sum0 = float(weights[vals == 0].sum()) if np.any(vals == 0) else 0.0
sum1 = float(weights[vals == 1].sum()) if np.any(vals == 1) else 0.0

print("\n--- mask weight breakdown ---")
print("sum weights where mask==0:", sum0)
print("sum weights where mask==1:", sum1)
print("zonal output mask (threshold @0.5):", at00(zon, var_mask))

# -------------------------
# 5) provenance checks: mapping csv + area sums
# -------------------------
csv_path = str(p_zonal) + ".zonal_weights.csv"
dfw = pd.read_csv(csv_path)
print("\n--- weights csv sanity ---")
print("weights csv:", csv_path)
print("rows:", len(dfw), "sum intersect area (m2):", dfw["intersect_area_m2"].sum())

ncells_out = int(zon["sample_ncells"].isel({lat_dim: 0, lon_dim: 0}).item())
area_out   = float(zon["sample_area_total_m2"].isel({lat_dim: 0, lon_dim: 0}).item())
print("sample_ncells out:", ncells_out, " expected:", len(wdf))
print("sample_area_total_m2 out:", area_out, " expected:", float(wdf["intersect_area_m2"].sum()))


print(ds_src["PCT_LAKE"].attrs)
print("min/max src PCT_LAKE:", float(ds_src["PCT_LAKE"].min()), float(ds_src["PCT_LAKE"].max()))
print("min/max zon PCT_LAKE:", float(zon["PCT_LAKE"].min()), float(zon["PCT_LAKE"].max()))

s_near = near["PCT_NAT_PFT"].isel({lat_dim:0, lon_dim:0, "time":0}).sum("natpft").item()
s_zon  = zon["PCT_NAT_PFT"].isel({lat_dim:0, lon_dim:0, "time":0}).sum("natpft").item()
print("sum natpft near:", s_near)
print("sum natpft zon :", s_zon)


import numpy as np
import xarray as xr
from dapper.utils import sampling
from dapper.utils import zonal

var = "PCT_LAKE"

ds_src = xr.open_dataset(GLOBAL_LANDUSE)
zw = zonal.intersect_weights_rectilinear(ds_src, targets)

wdf = zw.by_gid[gid]
i_idx = xr.DataArray(wdf["i_lat"].to_numpy(int), dims="cell")
j_idx = xr.DataArray(wdf["i_lon"].to_numpy(int), dims="cell")
w = wdf["weight"].to_numpy(float)

spec = sampling.infer_latlon_spec(ds_src, lon_wrap="auto")
vals = ds_src[var].isel({spec.lat_dim: i_idx, spec.lon_dim: j_idx}).values.reshape(-1)

manual = float(np.sum(vals * w) / np.sum(w))
print("manual wmean:", manual)

zonal_val = float(zon[var].isel({lat_dim:0, lon_dim:0}).item())
print("zonal:", zonal_val)
