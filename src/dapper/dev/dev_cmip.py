# Example for downloding CMIP (Atmosphere and LAI vars) over polygons of interest
from pathlib import Path
import os
import pandas as pd

from dapper.met import cmip_utils as cu

path_out = r'X:\Research\NGEE Arctic\10. CMIP for Kirsten\sampled'

# CMIP6 experiment_ids for requested SSPs
experiments = ["historical", "ssp126", "ssp245", "ssp585"]

# Climate vars (Amon)
clim_vars = ["ps", "pr", "rsds", "huss", "tas", "uas", "vas"]

# Sampling params
params = dict(
    experiment=["historical", "ssp126", "ssp245", "ssp585"],
    table="Amon",
    variables=["ps","pr","rsds","huss","tas","uas","vas"],
    ensemble="r1i1p1f1",
)

# Leaf area index
params = dict(
    experiment=["historical", "ssp126", "ssp245", "ssp585"],
    table="Lmon",
    variables=["lai"],
    ensemble="r1i1p1f1",
)

col = cu.open_cmip6_catalog()

df_all = cu.search_cmip6(params, col=col)
df_use = cu.filter_complete(cu.dedupe_latest(df_all), required_vars=params["variables"])

aois = {}
for gj in Path(r"X:\Research\NGEE Arctic\10. CMIP for Kirsten\Study Site Bounding Boxes").glob("*.geojson"):
    site = gj.stem[12:]
    if site not in ['Abisko', 'TVC', 'TL27', 'TL47', 'Toolik_Lake']:
        continue
    aois[site] = cu.bounds_from_geojson(gj)  # -> (lat_bounds, lon_bounds)


out = cu.sample_bbox_means_for_aois(
    df_use,
    aois=aois,  # {"aoi1": ((latmin, latmax),(lonmin, lonmax)), ...}
    out_dir=path_out,           # <-- resume-safe progress
    fail_log=os.path.join(path_out, "pathcmip_failures.log"),    # <-- keep going on errors
    retries=5,
    chunk_format="parquet",
    return_df=False,                         # <-- avoids re-reading everything at end
)


chunks = Path(path_out)
df_all = pd.concat((pd.read_parquet(p) for p in chunks.glob("*.parquet")), ignore_index=True)
df_all.to_csv(os.path.join(path_out, "cmip_all.csv"), index=False)
