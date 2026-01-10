import os
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from dapper import Domain
from dapper import ERA5Adapter


## Pathing
# All paths are with respect to dapper repo root
DAPPER_ROOT = Path(r"X:\Research\NGEE Arctic\dapper")
OUT_ROOT = DAPPER_ROOT / 'docs' / 'tutorials' / 'end-to-end' / 'outputs'
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_SITES  = OUT_ROOT / "sites_mode"
# OUT_CELLSET = OUT_ROOT / "cellset_mode"

# Where the (pre-sampled) GEE CSV shards live.
# You can commit tiny example shards to the repo, or just place them here locally.
RAW_GEE_CSVS_DIR = Path(DAPPER_ROOT / "docs" / "data" / "end-to-end" / "gee_csvs").resolve()
RAW_GEE_CSVS_DIR.mkdir(parents=True, exist_ok=True)

# In order to create surface and landuse files, we sample from global files. dapper provides these
# "pseudo-global" files that have been cropped from true global files to cover the areas of interest
# in this notebook. The actual global files are way too big for a GitHub repo. If you have access
# to global files (you should), you can use those paths here instead of these.
SURF_GLOBAL_NC = DAPPER_ROOT / 'docs' / 'data' / 'end-to-end' / 'surf_pseudoglobal.nc'
LANDUSE_GLOBAL_NC = DAPPER_ROOT / 'docs' / 'data' / 'end-to-end' / 'landuse_pseudoglobal.nc'

print("OUT_ROOT:", OUT_ROOT)
print("RAW_GEE_CSVS_DIR:", RAW_GEE_CSVS_DIR)
print("SURF_GLOBAL_NC:", SURF_GLOBAL_NC)
print("LANDUSE_GLOBAL_NC:", LANDUSE_GLOBAL_NC)

## Build our geometries
# important to note that input MUST have a unique 'gid' column
lat_min, lat_max = 69.25, 69.75
lon_edges = [-152.25, -151.75, -151.25, -150.75]  # 3 cells

rows = []
for i in range(3):
    gid = f"cell_{i+1:02d}"
    lon_min, lon_max = lon_edges[i], lon_edges[i+1]
    geom = box(lon_min, lat_min, lon_max, lat_max)
    rows.append({"gid": gid, "geometry": geom})

gdf_cells = gpd.GeoDataFrame(rows, crs="EPSG:4326")
gdf_cells

# Would be better to plot this with a map behind it
# ax = gdf_cells.boundary.plot(figsize=(6, 4))
# gdf_cells.apply(lambda r: ax.text(r.geometry.representative_point().x, r.geometry.representative_point().y, r.gid), axis=1)
# ax.set_title("Three 0.5° cells near (69.5, -151.5)")
# ax.set_xlabel("lon")
# ax.set_ylabel("lat")

# Keep polygons in Domain.cells for BOTH modes
domain = Domain.from_provided(
    gdf_cells,
    name="three_cells",
    mode="sites",
    cell_kind="as_provided",
)


## Met sampling
GEE_PROJECT = os.environ.get("GEE_PROJECT", "ee-jonschwenk")     # <-- change me
GDRIVE_FOLDER = os.environ.get("GDRIVE_FOLDER", "dapper_e2e_3cells") # <-- change me

# Small time range for an example (adjust as desired)
START_DATE = "2020-01-01"
END_DATE   = "2022-01-01"   # keep small for a demo

import ee
ee.Initialize(project=GEE_PROJECT)

from dapper import sample_e5lh
params = dict(
    start_date=START_DATE,
    end_date=END_DATE,
    geometries=domain,        # can also pass domain_sites; geometries are the same here
    geometry_id_field="gid",
    gee_bands="elm",                  # "elm" or "all" or explicit list
    gdrive_folder=GDRIVE_FOLDER,
    job_name="e2e_3cells_era5",
    gee_scale="native",
    gee_years_per_task=1,
)

# Starts export tasks unless skip_tasks=True
gee_sampled_domain = sample_e5lh(params)

# Export the met data SITES (to test--can i just change the method in Domain to 'cells' and re-export?)
from dapper import ERA5Adapter
adapter = ERA5Adapter()
met_sites = domain.export_met(
    src_path=RAW_GEE_CSVS_DIR,
    adapter=adapter,
    out_dir=OUT_SITES,
    # keep it simple for end-to-end:
    calendar="noleap",
    dtime_resolution_hrs=1,
    dformat="BYPASS",
    overwrite=False,
    append_attrs={"dapper_example": "e2e_3cells_sites"},
)

## Export Domain file
domain.export_domain(out_dir=OUT_SITES, overwrite=True)

# sites: nearest (point)
surf_sites = domain.export_surface(
    src_path=SURF_GLOBAL_NC,
    out_dir=OUT_SITES,
    filename="surfdata.nc",
    overwrite=False,
    sampling_method="nearest",
    append_attrs={"dapper_example": "e2e_3cells_sites", "sampling": "nearest"},
)

## Landuse - zonal sampling
lu_sites = domain.export_landuse(
    src_path=LANDUSE_GLOBAL_NC,
    out_dir=OUT_SITES,
    filename="landuse_timeseries.nc",
    overwrite=True,
    sampling_method="zonal",
    append_attrs={"dapper_example": "e2e_3cells_sites", "sampling": "zonal"},
)

