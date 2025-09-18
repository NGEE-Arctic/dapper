
import ee
import pandas as pd
import geopandas as gpd
from importlib import reload

from dapper.utils import utils
from dapper.utils import gee_utils as gu
from dapper.met.adapters.era5 import ERA5Adapter
from dapper.met.exporter import Exporter

# ee.Initialize(project='ee-jonschwenk')

# # Load Kuparuk gage watershed polygon; convert to ee object
# kup = gpd.read_file(utils._DATA_DIR / 'Kuparuk_gageshed.shp')
# kup = kup.buffer(0) # A hack to "fix" invalid geometries--not needed here but just demonstrating
# kup_pgon = ee.Geometry.Polygon(list(kup.geometry.values[0].exterior.coords))

# # Point to the ERA5-Land hourly polygon grid
# e5lh_grid = ee.FeatureCollection('projects/ee-jonschwenk/assets/E3SM/e5lh_grid')

# # Intersect and return the e5lh ids
# intersecting_e5l_grids = e5lh_grid.filter(ee.Filter.intersects('.geo', kup_pgon))
# # intersecting_e5l_grids = e5lh_grid.filterBounds(kup_pgon)
# pids = intersecting_e5l_grids.aggregate_array('pids').getInfo()

# # Now we select the E5LH grid cells we want to sample
# sample_these_cells = e5lh_grid.filter(ee.Filter.inList('pids', pids))

# params = {
#     'start_date' : '1950-01-01', # YYYY-MM-DD
#     'end_date' : '1957-01-01', # YYYY-MM-DD
#     'geometries' : sample_these_cells, # Dictionary of {'name' : (lat, lon)} for all points to sample
#     'geometry_id_field' : 'pids', # The e5lh_grid Asset calls this field "pids", so we must specify that
#     'gee_bands' : 'elm', # Select ELM-required bands
#     'gee_years_per_task' : 1, # Optional parameter; default is 5. For lots of points, you may want to reduce this for smaller GEE Tasks (but more of them)
#     'gee_scale' : 'native',
#     'gdrive_folder' : 'ngee_test_cells', # Which folder to store on your GDrive; will be created if not exists
#     'job_name' : 'cell_test',
# }
# df_loc = gu.sample_e5lh(params, skip_tasks=True)
import pickle
# # df_loc.to_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle2.pkl')
df_loc = pd.read_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle2.pkl')



csv_dir = r'X:\Research\NGEE Arctic\dapper\data\celltesting'
write_dir = r'X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted3'

adapter = ERA5Adapter()

exp = Exporter(
    adapter=adapter,
    csv_directory=csv_dir,
    write_directory=write_dir,
    df_loc=df_loc,
    id_col=None,                 # we assume 'gid' in both shards and df_loc
    calendar='noleap',
    dtime_resolution_hrs=1,
    dtime_units='days',
    nzones=1,
    dformat="BYPASS",
    force_half_hour_for_hourly=True,
    append_attrs={"note":"smoke-test run"},  # optional globals
    chunks=None,               # let writers pick auto-chunking
    include_vars={"PRECTmms"}, # keep it small for the smoke test
    exclude_vars=None
)

exp.run(output_mode='sites_file', pack_scope='global')
print("Smoke test done. Check files in:", write_dir)


# Reduced smoke test of per-site export
df_loc_small = df_loc.iloc[:3].copy()  # or pick specific gids
exp = Exporter(
    adapter=ERA5Adapter(),
    csv_directory=r"X:\Research\NGEE Arctic\dapper\data\celltesting",
    write_directory=r"X:\Research\NGEE Arctic\dapper\data\celltesting\out_site_dirs",
    df_loc=df_loc_small,
    calendar='noleap',
    dtime_resolution_hrs=1,
    dtime_units='days',
    dformat="BYPASS",
    append_attrs={"note":"site_dirs smoke test"},
    include_vars={"PRECTmms"},   # a couple vars to keep it quick
)

exp.run(output_mode='gridded') 
