# uv run --no-sync python with_topounits.py
import ee
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from dapper import Domain
from dapper import ERA5Adapter
from dapper import sample_e5lh

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 100)

ee.Initialize(project="jstagee")

feature = ee.Feature(
    ee.FeatureCollection("projects/ee-jonschwenk/assets/E3SM/Kuparuk_gageshed").first()
)

aoi_gdf = gpd.GeoDataFrame.from_features(
    [
        {
            "type": "Feature",
            "properties": {"legend_label": "AOI"},
            "geometry": feature.geometry().getInfo(),
        }
    ],
    crs="EPSG:4326",
)
domain = Domain.from_provided(
    aoi_gdf,
    name="kuparuk",
    mode="sites",
    cell_kind="as_provided",
)

domain = domain.make_topounits(
    sources=["hand"],
    binning={"hand": {"strategy": "percentiles", "n_bins": 6, "label_prefix": "HAND"}},
    target_scale=90,
    verbose=True,
)
print(domain.has_topounits())
print(domain.topounits)

GEE_PROJECT = "jstagee"
GDRIVE_FOLDER = "with_topounits"
START_DATE = "2020-01-01"
END_DATE = "2022-01-01"  # keep small for a demo - can put 2100-01-01 to get the latest available data

sampling_params = dict(
    start_date=START_DATE,
    end_date=END_DATE,
    geometries=domain,
    geometry_id_field="gid",
    gee_bands="elm",
    gdrive_folder=GDRIVE_FOLDER,
    job_name="with_topounits",
    gee_scale="native",
    gee_years_per_task=1,
)

# Starts export tasks unless skip_tasks=True
dom_sample = sample_e5lh(
    sampling_params, skip_tasks=False
)  # change skip_tasks to False in order to actually start the tasks on GEE
breakpoint()
# stage shards in gee_shards folder

met_sites = domain.export_met(
    src_path=Path("gee_shards"),
    adapter=ERA5Adapter(),
    out_dir=Path("."),
    calendar="noleap",
    dtime_resolution_hrs=1,  # can change if you want. I think even 0.5 works if you want interpolated 30-minute data, but I wouldn't recommend it.
    dformat="BYPASS",
    overwrite=True,
)
domain.export_domain(out_dir=Path("."), overwrite=True)

surf_sites = domain.export_surface(
    src_path=Path("../../data/end-to-end/surf_pseudoglobal.nc"),
    out_dir=Path("."),
    filename="surfdata.nc",
    overwrite=True,
    sampling_method="nearest",
    append_attrs={"sampling": "nearest"},
)
print("sites surface:", surf_sites)

# Let's look at the dimensions
# (note that the surface file contains topounit(s))
ds = xr.open_dataset(surf_sites["cell_0000"], decode_cf=False)
print("Dimensions:")
[print(f"  {dim}: {n}") for dim, n in ds.sizes.items()]
