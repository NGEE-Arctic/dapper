import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from dapper.utils import sampling
from dapper.utils import zonal
from dapper.dev.pathing import SURFDATA_HALFDEGREE_TOP

# -------------------------
# USER INPUTS
# -------------------------
GLOBAL_SURF_PATH = SURFDATA_HALFDEGREE_TOP
gid = "cell_001"

# pick a place likely to have heterogeneity; change as needed
lon0, lat0 = -149.6, 68.63   # Toolik-ish
halfwidth_deg = 2.0          # polygon spans 4° x 4°


# -------------------------
# LOAD DATASET
# -------------------------
ds = xr.open_dataset(GLOBAL_SURF_PATH)

# Choose a categorical-ish int variable
int_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.integer) or ds[v].dtype == np.bool_]
preferred = [v for v in ["SOIL_ORDER", "SOIL_COLOR", "URBAN_REGION_ID", "GLC_MEC", "GLC_ICESHEET", "PFTDATA_MASK"] if v in int_vars]

if preferred:
    cat_var = preferred[0]
else:
    if not int_vars:
        raise RuntimeError("No integer/bool variables found in this dataset.")
    cat_var = int_vars[0]

print(f"Using categorical candidate var: {cat_var}  dtype={ds[cat_var].dtype}")


# -------------------------
# BUILD A TARGET POLYGON (your user cell geometry)
# -------------------------
targets = gpd.GeoDataFrame(
    {"gid": [gid],
     "geometry": [box(lon0 - halfwidth_deg, lat0 - halfwidth_deg, lon0 + halfwidth_deg, lat0 + halfwidth_deg)]},
    crs="EPSG:4326",
)

# -------------------------
# INTERSECTIONS + WEIGHTS
# -------------------------
zw = zonal.intersect_weights_rectilinear(ds, targets, lon_wrap="auto")  # uses your inferred grid + equal-area CRS
wdf = zw.by_gid[gid]
print(f"Intersected {len(wdf)} source cells")
print(wdf.head())

# Infer actual dims in ds (lsmlat/lsmlon or whatever)
spec = sampling.infer_latlon_spec(ds, lon_wrap="auto")
lat_dim, lon_dim = spec.lat_dim, spec.lon_dim

# Select the contributing source cells for this var
i_idx = xr.DataArray(wdf["i_lat"].to_numpy(dtype=int), dims="cell")
j_idx = xr.DataArray(wdf["i_lon"].to_numpy(dtype=int), dims="cell")
w = xr.DataArray(wdf["weight"].to_numpy(dtype=float), dims="cell")

da_sel = ds[cat_var].isel({lat_dim: i_idx, lon_dim: j_idx})  # dims include "cell" now
# Some vars have extra dims (e.g., natpft); for this demo we’ll just squeeze all non-cell dims if possible
da_sel_squeezed = da_sel
for d in list(da_sel_squeezed.dims):
    if d != "cell" and da_sel_squeezed.sizes.get(d, 1) == 1:
        da_sel_squeezed = da_sel_squeezed.isel({d: 0})

if da_sel_squeezed.dims != ("cell",):
    print(f"NOTE: {cat_var} has extra dims {da_sel.dims}. "
          f"This demo will take the first index of each extra dim for the manual check.")
    # force a single slice for manual inspection
    slicer = {d: 0 for d in da_sel.dims if d != "cell"}
    da_sel_squeezed = da_sel.isel(slicer)

vals = da_sel_squeezed.values
uniq = np.unique(vals[~np.isnan(vals)] if np.issubdtype(vals.dtype, np.floating) else vals)

# Print weight by category
contrib = {}
for u in uniq:
    mask = (vals == u)
    contrib[int(u) if np.issubdtype(vals.dtype, np.integer) else float(u)] = float(w.values[mask].sum())

print("\nWeight by category (top 15):")
for k, v in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:15]:
    print(f"  {k}: {v:.4f}")


# -------------------------
# MANUAL WMODE CHECK
# -------------------------
manual_mode = zonal.reduce_wmode(xr.DataArray(vals, dims="cell"), w, tie_break="smallest").item()
print(f"\nManual wmode({cat_var}) = {manual_mode}")


# -------------------------
# END-TO-END ZONAL SAMPLER CHECK
# -------------------------
out = zonal.sample_gridded_dataset_polygons(
    ds,
    targets,
    vars_include=[cat_var],
    agg_policy={cat_var: "wmode"},
    lon_wrap="auto",
)

zonal_mode = out[cat_var].isel({lat_dim: 0, lon_dim: 0}).item()
print(f"Zonal sampler wmode({cat_var}) = {zonal_mode}")

if zonal_mode != manual_mode:
    print("WARNING: zonal sampler result != manual result. Inspect selection/weights/dims.")
else:
    print("OK: zonal sampler matches manual wmode.")


# -------------------------
# OPTIONAL: compare to nearest-point sampling at polygon centroid
# -------------------------
centroid = targets.geometry.iloc[0].representative_point()
pts = pd.DataFrame({"gid": [gid], "lat": [centroid.y], "lon": [centroid.x]})
nearest = sampling.sample_gridded_dataset_points(ds, pts, vars_include=[cat_var], lon_wrap="auto")
nearest_val = nearest[cat_var].isel({lat_dim: 0, lon_dim: 0}).item()
print(f"Nearest-point {cat_var} at centroid = {nearest_val}")
