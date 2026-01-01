import os
from pathlib import Path
from importlib import reload

import ee
import geopandas as gpd
import pandas as pd
import xarray as xr

# TODO - allow custom path to global surface file
# TODO - this (section 8) should be better integrated (auto-computed when topounits are made)
# TODO - Area-weighted landuse


# ---- Required inputs ----
GEE_PROJECT = "ee-jonschwenk"     # <-- change
SITE_ID     = "upper_kuparuk"

WATERSHED_VECTOR = r"X:\Research\NGEE Arctic\dapper\docs\data\upper_kup_watershed\upper_kuparak_watershed.gpkg"   # or any geopandas-readable polygon file
WATERSHED_LAYER  = None                           # set if needed for gpkg

MET_START_DATE = "1950-01-01"
MET_END_DATE   = "1955-01-01"    # or "2200-01-01" for “all available”

PATH_ELM_OUT = Path(r"docs/data/end-to-end/single-site") / SITE_ID

# 1. Build Domain
# Controls all the other functions in dapper
from dapper.domains.domain import Domain
import geopandas as gpd

gdf_ws = gpd.read_file(WATERSHED_VECTOR).to_crs("EPSG:4326")
gdf_ws = gpd.GeoDataFrame({"gid":[SITE_ID], "geometry":[gdf_ws.geometry.values[0]]}, crs="EPSG:4326")

domain = Domain.from_provided(
    gdf_ws,
    name=SITE_ID,
    mode="site",        # 1 row is fine either way, but be explicit
    cell_kind="site_points",  # representative point for sampling surf/landuse
    path_out=PATH_ELM_OUT
)

# 1) Export ELM domain file
domain_nc = domain.export_domain(domain.path_domain_nc("domain.nc"))
print("Wrote domain:", domain_nc)

# 2. Make Topounits
from dapper.topounit.topomake import make_topounits
from dapper.utils import gee_utils as gu
import ee

ee.Initialize(project=GEE_PROJECT)
ee_geom = gu.parse_geometry_object(domain.support.geometry.iloc[0], name=SITE_ID)
feature = ee.Feature(ee_geom)

binning = {
    "elev": {"strategy": "percentiles", "n_bins": 4},
    "aspect": {"strategy": "fixed"},  # uses default N/S ranges in your topomake
}
domain = domain.make_topounits(
    binning=binning,
    target_scale=90,
    verbose=True,
)

# 2a. Met data sampling
import ee
ee.Initialize(project=GEE_PROJECT)

from dapper.utils.gee_utils import sample_e5lh

GDRIVE_FOLDER = "dapper_met_exports"  # Google Drive folder name

params = {
    "job_name": f"era5l_{SITE_ID}",
    "geometries": domain,          # pass Domain directly (no Feature/FC needed)
    "geometry_id_field": "gid",
    "start_date": MET_START_DATE,
    "end_date": MET_END_DATE,
    "gee_bands": 'elm',
    "gee_scale": 10000,          # native ERA5-Land is ~9 km; 10 km is fine for polygons
    "gee_years_per_task": 1,     # smaller chunks for reliability
    "gdrive_folder": GDRIVE_FOLDER,
}

# Set skip_tasks=False to actually submit tasks
_ = sample_e5lh(params, domain_name=f"met_{SITE_ID}", skip_tasks=True)

print("Met sampling configured. If you want to submit tasks, set skip_tasks=False above.")

# 2b. Met data exporting
from dapper.met.exporter import Exporter
from dapper.met.adapters.era5 import ERA5Adapter

if LOCAL_CSV_DIR.exists():
    exp = Exporter(
        adapter=ERA5Adapter(),
        csv_directory=LOCAL_CSV_DIR,
        write_directory=domain.met_dir,
        domain=domain,
        dtime_resolution_hrs=1,
        calendar="NO_LEAP",
        append_attrs={"source": "ERA5-Land via GEE + dapper"},
    )

    # "elm-sites" writes a subdir per gid under MET/
    # For a single site this is fine; it also writes a per-site zone_mappings.txt
    exp.run(output_mode="elm-sites")

    print("Met export complete:", domain.met_dir)
else:
    print(f"Skipping met export because LOCAL_CSV_DIR does not exist: {LOCAL_CSV_DIR}")


# 3. Surface File
# --- Surface export (1 file because 1 row / 1 site) ---
surf_dir = OUT_DIR / "surf"
paths = domain.export_surface(
    out_dir=surf_dir,
    filename="surfdata.nc",
    overwrite=True,
    attach_topounits=True,
    nc_in='path_to_global_surface_file'
)

