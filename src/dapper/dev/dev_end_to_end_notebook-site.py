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
# Change "output_root" to "path_for_write" or something less mysterious to user
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
    output_root=PATH_ELM_OUT
)


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

# 3. Surface File
# --- Surface export (1 file because 1 row / 1 site) ---
surf_dir = OUT_DIR / "surf"
paths = domain.export_surface(
    out_dir=surf_dir,
    filename_template="surfdata_{run_id}.nc",
    overwrite=True,
    attach_topounits=True,
)

