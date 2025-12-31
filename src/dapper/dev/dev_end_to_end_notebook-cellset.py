# --- 0) Imports / config ---
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

import ee

from shapely.ops import unary_union
from shapely.geometry import box
from shapely.affinity import translate

from dapper.domains.domain import Domain  # or from dapper.domains.domain import Domain


# USER: set these
GEE_PROJECT = "ee-jonschwenk"
WATERSHED_VECTOR = r"X:\Research\NGEE Arctic\dapper\docs\data\upper_kup_watershed\upper_kuparak_watershed.gpkg"   # or shapefile
OUT_DIR = Path(r"X:\Research\NGEE Arctic\3. Surface Files\out\cellset_2cell")
RUN_NAME = "watershed_2cell"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- 1) Load watershed geometry (ensure EPSG:4326) ---
gdf_ws = gpd.read_file(WATERSHED_VECTOR).to_crs("EPSG:4326")
geom_ws = unary_union(gdf_ws.geometry.values)  # handles MultiPolygon etc.

# Optional: clean invalid geometry
try:
    geom_ws = geom_ws.buffer(0)
except Exception:
    pass


# --- 2) Split into 2 "cells" (left/right halves) ---
def split_watershed_into_two_cells(geom):
    minx, miny, maxx, maxy = geom.bounds
    midx = 0.5 * (minx + maxx)

    left = geom.intersection(box(minx, miny, midx, maxy))
    right = geom.intersection(box(midx, miny, maxx, maxy))

    # Fallback if split fails (e.g., weird skinny polygon)
    if left.is_empty or right.is_empty:
        rp = geom.representative_point()
        dx = (maxx - minx) * 0.15 if (maxx - minx) > 0 else 0.05
        buf = (maxx - minx) * 0.10 if (maxx - minx) > 0 else 0.05
        g1 = translate(rp, xoff=-dx).buffer(buf).intersection(geom)
        g2 = translate(rp, xoff=+dx).buffer(buf).intersection(geom)
        left, right = g1, g2

    if left.is_empty or right.is_empty:
        raise RuntimeError("Could not create two non-empty cell geometries from watershed.")

    return left, right
cell_geom_1, cell_geom_2 = split_watershed_into_two_cells(geom_ws)

gdf_cells = gpd.GeoDataFrame(
    {
        "gid": [f"{RUN_NAME}_c01", f"{RUN_NAME}_c02"],
        "geometry": [cell_geom_1, cell_geom_2],
    },
    crs="EPSG:4326",
)

assert gdf_cells["gid"].is_unique
print(gdf_cells[["gid"]])


# --- 3) Build a 2-cell Domain (cellset) ---
# This assumes your Domain API supports:
#   - from_provided(gdf, name=..., mode="cellset", cell_kind="site_points")
# and that domain.support keeps the polygons while domain.cells provides representative points.
domain = Domain.from_provided(
    gdf_cells,
    name=RUN_NAME,
    mode="cellset",
    cell_kind="site_points",
    output_root=PATH_ELM_OUT
)

print("Domain gids:", domain.to_df_loc()["gid"].tolist())


# --- 4) Compute topounits per cell (Domain wrapper) ---
ee.Initialize(project=GEE_PROJECT)

binning = {
    "elev": {"strategy": "percentiles", "n_bins": 4},
    "aspect": {"strategy": "fixed"},  # uses your built-in default N/S ranges
}

domain = domain.make_topounits(
    binning=binning,
    target_scale=90,
    verbose=True,
    allow_slow_ncells=10,  # guardrail; 2 cells is fine
)

# Quick sanity on results
topos = domain.topounits
print("Topounits rows:", len(topos))
print(topos.groupby("gid").size())

# Ensure weights exist; if not, compute them per gid (defensive)
if "TopounitPctOfCell" not in topos.columns:
    topos = topos.copy()
    # approximate area in an equal-area CRS for weights
    topos_m = topos.to_crs("EPSG:6933")
    topos["TopounitArea_m2"] = topos_m.geometry.area
    topos["TopounitPctOfCell"] = (
        topos.groupby("gid")["TopounitArea_m2"].transform(lambda x: 100.0 * x / x.sum())
    )
    domain = domain.with_topounits(topos, id_col=domain.topounits_id_col, gid_col=domain.topounits_gid_col, dim_name=domain.topounits_dim_name)

# Verify sum-to-100 per gid
chk = domain.topounits.groupby("gid")["TopounitPctOfCell"].sum()
print("Sum TopounitPctOfCell by gid:")
print(chk)

# 4. Surface File
surf_dir = OUT_DIR / "surf"
paths = domain.export_surface(
    out_dir=surf_dir,
    filename_template="surfdata_{run_id}.nc",  # run_id will be RUN_NAME in cellset
    overwrite=True,
    attach_topounits=True,
)
