import os
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

# TODO - allow custom path to global surface file
# TODO - this (section 8) should be better integrated (auto-computed when topounits are made)
# TODO - Area-weighted landuse
# TODO - Re-use domain everywhere throughout the workflow?
# ADD metadata to the landuse file (name of the originally-sampled file)
# Potentially add metadata to the surface file
# Allow users to add metadata to all the files with a custom dictionary (already can for met file)


# ---- Required inputs ----
GEE_PROJECT = "ee-jonschwenk"     # <-- change
SITE_ID     = "upper_kuparuk"

WATERSHED_VECTOR = r"X:\Research\NGEE Arctic\dapper\docs\data\upper_kup_watershed\upper_kuparak_watershed.gpkg"   # or any geopandas-readable polygon file
WATERSHED_LAYER  = None                           # set if needed for gpkg

MET_START_DATE = "1950-01-01"
MET_END_DATE   = "1955-01-01"    # or "2200-01-01" for “all available”

PATH_ELM_OUT = Path(r"docs/data/end-to-end/single-site") / SITE_ID
PATH_ELM_OUT.mkdir(parents=True, exist_ok=True)

PATH_MET     = PATH_ELM_OUT / "met" # directory
PATH_TOPO    = PATH_ELM_OUT / "topounits" # directory
PATH_DOMAIN    = PATH_ELM_OUT / f"domain.lnd.{SITE_ID}.nc"
PATH_SURF    = PATH_ELM_OUT / f"surfdata_{SITE_ID}_with_TOP.nc"
PATH_LANDUSE = PATH_ELM_OUT /  f"landuse.timeseries_{SITE_ID}.nc"

for p in [PATH_MET, PATH_TOPO]:
    p.mkdir(parents=True, exist_ok=True)

print("PATH_ELM_OUT:", PATH_ELM_OUT.resolve())

# 2. Load polygon (watershed) geometry
gdf_ws = gpd.read_file(WATERSHED_VECTOR)

# Keep/force single feature (union if multiple)
gdf_ws = gdf_ws.to_crs("EPSG:4326")
gdf_ws = gpd.GeoDataFrame({"gid":[SITE_ID], "geometry":[gdf_ws.geometry.values[0]]}, crs="EPSG:4326") # simplify the GeoDataFrame
print(gdf_ws)

# 3. Build a Domain file
from dapper.domains.aoi import AOI

aoi = AOI(name=SITE_ID, gdf=gdf_ws)

# Domain is a 1x1 point domain using representative point of the polygon
domain = aoi.to_domain_points()   # your method

# Confirm lon/lat
print(domain.gdf)

# Write the ELM domain file 
domain.write_elm_domain(PATH_DOMAIN)

print("Wrote:", domain_nc)


# 4. Sample ERA5 met data
from dapper.utils import gee_utils as gu
ee.Initialize(project=GEE_PROJECT)

params = {
    "start_date": MET_START_DATE,
    "end_date": MET_END_DATE,
    "geometries": gdf_ws,               # polygon(s) (area-average)
    "geometry_id_field": "gid",
    "gee_bands": "elm",                 # convenient shortcut for ELM-required raw bands :contentReference[oaicite:1]{index=1}
    "gee_years_per_task": 1,            # tune up later
    "gee_scale": "native",
    "gdrive_folder": f"dapper_{SITE_ID}_era5",
    "job_name": f"{SITE_ID}_wshed",
}

df_loc = gu.sample_e5lh(params, skip_tasks=False)  # submits tasks

# 5. WAIT FOR GEE TO FINISH
# Download files to local machine
PATH_GEE_RAW_ERA5 = PATH_MET / "raw_csv"

# 6. 
from dapper.config.metsources import era5
from dapper.met.adapters.era5 import ERA5Adapter
from dapper.met.exporter import Exporter

# Boot up ERA5 adapter to convert raw ERA5 to ELM
adapter = ERA5Adapter(
    df_loc=df_loc,
    csv_directory=PATH_GEE_RAW_ERA5,
    config=era5,          # provides variable mappings/requireds
)

# Export
exporter = Exporter(
    adapter=adapter,
    out_dir=PATH_MET / "elm_forcing",
    domain_name=SITE_ID,
)

# This should write ELM coupler-bypass formatted netcdfs + zone_mappings, etc.
exporter.export()
print("Met outputs in:", (PATH_MET / "elm_forcing"))

# 7. Make topounits
from dapper.topounit.topomake import make_topounits

# Convert local watershed geometry to an EE Feature
ee_geom = gu.parse_geometry_object(gdf_ws.geometry.iloc[0], name=SITE_ID)
feature = ee.Feature(ee_geom)

# Example: elev quartiles + N/S aspect => 8 bins
aspects = [(270, 90, "N"), (90.01, 269.99, "S")]

topos = make_topounits(
    feature=feature,
    sources=["elev", "aspect"],
    binning={
        "elev":   {"strategy": "percentiles", "n_bins": 4, "label_prefix": "ELEV"},
        "aspect": {"strategy": "fixed", "ranges": aspects, "label_prefix": "ASP"},
    },
    combine="cartesian",
    min_patch_pixels=9,
    target_scale=90,          # keep big AOIs reasonable
    export_scale="native",
    return_as="gdf",
    verbose=True
)

topos_path = PATH_TOPO / f"{SITE_ID}_topounits.gpkg"
topos.to_file(topos_path, driver="GPKG")
print("Wrote:", topos_path)
topos.head()

# 8. 
# Compute weights (fraction of watershed area)
topos_m = topos.to_crs("EPSG:6933")  # equal-area
topos["area_m2"] = topos_m.geometry.area
topos["weight"] = topos["area_m2"] / topos["area_m2"].sum()

# Keep a clean params df for surf insertion
topo_params = topos[["band_name", "weight"]].copy()
topo_params

# 9. Build surface file
from dapper.surf.sfile import SurfaceFile

# Use the domain representative point as the surface sample location
lon = float(domain.gdf.lon.iloc[0])
lat = float(domain.gdf.lat.iloc[0])

sf = SurfaceFile.from_halfdegree_point(lat=lat, lon=lon)  # global half-degree sample 

# Add topounit dimension + weights
sf.add_params_from_df(
    dim_name="topounit",
    df=topo_params,
    id_col="band_name",
)

# Optional: validate (at least registry sanity)
sf.validate(strict=False)  
sf.to_netcdf(PATH_SURF, overwrite=True)
print("Wrote:", surf_nc)


# 10. Build landuse file
from dapper.utils.pathing import LANDUSE_EIGHTH_DEGREE
from dapper.utils.sampling import sample_landuse_timeseries

# Build df_loc-style table for landuse sampling (point at domain rep point + weight=1)
df_loc_landuse = pd.DataFrame([{
    "gid": SITE_ID,
    "lat": float(domain.gdf.lat.iloc[0]),
    "lon": float(domain.gdf.lon.iloc[0]),
    "weight": 1.0
}])


sample_landuse_timeseries(
    nc_in=LANDUSE_EIGHTH_DEGREE,
    df_loc=df_loc_landuse,
    out_path=PATH_LANDUSE,
    lon_wrap="infer",     # your code already handled 0..360 vs -180..180
)

print("Wrote:", landuse_nc)

# Quick sanity check
ds_lu = xr.open_dataset(landuse_nc, decode_times=False)
if "PCT_NAT_PFT" in ds_lu:
    err = float(abs(ds_lu["PCT_NAT_PFT"].sum("natpft") - 100).max())
    print("Max |sum(PCT_NAT_PFT)-100|:", err)
