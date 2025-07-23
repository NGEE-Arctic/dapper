# Functions either in development or used for one-off data stuff. Created by JPS.
# Note that there may be imports here that aren't included in environment.yml file.

from dapper import e5l_utils as eu
import ee
ee.Initialize(project='ee-jonschwenk')
from shapely import Polygon
import geopandas as gpd


from dapper import utils
from dapper import e5l_utils as eu
from pathlib import Path

# Load Kuparuk gage watershed polygon; convert to ee object
kup = gpd.read_file(utils._DATA_DIR / 'Kuparuk_gageshed.shp')
kup = kup.buffer(0) # A hack to "fix" invalid geometries--not needed here but just demonstrating
kup_pgon = ee.Geometry.Polygon(list(kup.geometry.values[0].exterior.coords))

# Point to the ERA5-Land hourly polygon grid
e5lh_grid = ee.FeatureCollection('projects/ee-jonschwenk/assets/E3SM/e5lh_grid')

# Intersect and return the e5lh ids
intersecting_e5l_grids = e5lh_grid.filter(ee.Filter.intersects('.geo', kup_pgon))
# intersecting_e5l_grids = e5lh_grid.filterBounds(kup_pgon)
pids = intersecting_e5l_grids.aggregate_array('pids').getInfo()

# Now we select the E5LH grid cells we want to sample
sample_these_cells = e5lh_grid.filter(ee.Filter.inList('pids', pids))

params = {
    'start_date' : '1950-01-01', # YYYY-MM-DD
    'end_date' : '1957-01-01', # YYYY-MM-DD
    'geometries' : sample_these_cells, # Dictionary of {'name' : (lat, lon)} for all points to sample
    'geometry_id_field' : 'pids', # The e5lh_grid Asset calls this field "pids", so we must specify that
    'gee_bands' : 'elm', # Select ELM-required bands
    'gee_years_per_task' : 1, # Optional parameter; default is 5. For lots of points, you may want to reduce this for smaller GEE Tasks (but more of them)
    'gee_scale' : 'native',
    'gdrive_folder' : 'ngee_test_cells', # Which folder to store on your GDrive; will be created if not exists
    'job_name' : 'cell_test',
}
eu.sample_e5lh(params)

# Create coordinate DataFrame
df_loc = pd.DataFrame({'pids' : pids,
          'lat' : [float(p.split(',')[0]) for p in pids],
          'lon' : [float(p.split(',')[1]) for p in pids]})

csv_directory = r'X:\Research\NGEE Arctic\dapper\data\celltesting'
write_directory = r'X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted'
remove_leap = True
id_col = 'pids'
compress=True
compress_level=4
dformat='BYPASS'

import dapper.e5l_utils as eu
import importlib
importlib.reload(eu)
eu.e5lh_to_elm_gridded(csv_directory, write_directory, df_loc, remove_leap, id_col, nzones=1, compress_level=9)

import xarray as xr
path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\clmforc.GSWP3.c2011.0.5x0.5.Prec.1901-01.nc"
path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\era5\Daymet_ERA5.1km_QBOT_1980-2023_z01.nc"
path = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\GSWP3_FLDS_1901-2014_z14.nc"
pathf = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\gridded\GSWP3_daymet4_FLDS_1980-2014_z01_gridded.nc"
pathm = r"X:\Research\NGEE Arctic\dapper\data\celltesting\elm_formatted\ERA5_QBOT_1950-1956_z1.nc"
dsm = xr.open_dataset(pathm)
dsf = xr.open_dataset(pathf)









points = {'abisko' : (68.35, 18.78333),
        'tvc' : (68.742, -133.499),
        'toolik' : (68.62758, -149.59429),
        'chars' :  (69.1300, -105.0415),
        'qhi' : (69.5795, -139.0762),
        'sam' : (72.22, 126.3),
        'sjb' : (78.92163, 11.83109)}

# Create 1x1 degree box geometries
geometries = []
pids = []

for pid, (lat, lon) in points.items():
    # Create a 1x1 degree box centered on (lat, lon)
    half_size = 0.5  # Half-degree to make a 1x1 box
    box = Polygon([
        (lon - half_size, lat - half_size),  # Bottom-left
        (lon + half_size, lat - half_size),  # Bottom-right
        (lon + half_size, lat + half_size),  # Top-right
        (lon - half_size, lat + half_size),  # Top-left
        (lon - half_size, lat - half_size)   # Closing point
    ])
    geometries.append(box)
    pids.append(pid)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame({'pid': pids, 'geometry': geometries}, crs="EPSG:4326")


df_loc = pd.DataFrame({'pid' : points.keys(),
                       'lat' : [points[p][0] for p in points],
                       'lon' : [points[p][1] for p in points]}) # sorry, this is awkward to do but it will make things scalable as this repo develops


params = {
    'start_date' : '1950-01-01', # YYYY-MM-DD
    'end_date' : '1957-01-01', # YYYY-MM-DD
    'points' : points, # Dictionary of {'name' : (lat, lon)} for all points to sample
    'gee_bands' : 'elm', # Select ELM-required bands
    'gee_years_per_task' : 2, # Optional parameter; default is 5. For lots of points, you may want to reduce this for smaller GEE Tasks (but more of them)
    'gdrive_folder' : 'ngee_testing', # Which folder to store on your GDrive; will be created if not exists
    'file_name' : 'multipoint_test_batching' 
}
eu.sample_e5lh_at_points_multijob(params)







# from dapper import e5l_utils as eu
# import ee
# ee.Initialize(project='ee-jonschwenk')

# points = {'abisko' : (68.35, 18.78333),
#         'tvc' : (68.742, -133.499),
#         'toolik' : (68.62758, -149.59429),
#         'chars' :  (69.1300, -105.0415),
#         'qhi' : (69.5795, -139.0762),
#         'sam' : (72.22, 126.3),
#         'sjb' : (78.92163, 11.83109)}

# params = {
#     'start_date' : '1950-01-01', # YYYY-MM-DD
#     'end_date' : '1950-02-01', # YYYY-MM-DD
#     'points' : points, # Dictionary of {'name' : (lat, lon)} for all points to sample
#     'gee_bands' : [ 'temperature_2m', # These bands must match the gee_ic ImageCollection
#                     'u_component_of_wind_10m',
#                 ],
#     'gdrive_folder' : 'ngee_testing', # Which folder to store on your GDrive; will be created if not exists
#     'file_name' : 'multipoint_test' 
# }
# eu.sample_e5lh_at_points(params)

# The estimated file size for the new dataset with 100 unique sites, 40 variables, and an hourly date range from 1950-01-01 to 2025-01-01 
# is approximately 40.8 GB. 
# 1 GB for 7 sites (points), full time range

import pandas as pd
from pathlib import Path
import dapper.e5l_utils as eu
from dapper.utils import _ROOT_DIR

# Build our points dictionary (same as in 1a notebook)
points = {'abisko' : (68.35, 18.78333),
        'tvc' : (68.742, -133.499),
        'toolik' : (68.62758, -149.59429),
        'chars' :  (69.1300, -105.0415),
        'qhi' : (69.5795, -139.0762),
        'sam' : (72.22, 126.3),
        'sjb' : (78.92163, 11.83109)}

# Load the raw ERA5-Land hourly data we exported in notebook 1a.
path_e5lh = _ROOT_DIR / 'notebooks' / 'notebook_data' / 'ngee_test_era5_timeseries.csv'
df = pd.read_csv(path_e5lh)
df = eu.e5lh_to_elm_preprocess(df, remove_leap=True, verbose=True) # remove_leap is True by default, just showing it here FYI. verbose is False by default.
eu.validate_met_vars(df)


path = r"X:\Research\NGEE Arctic\1. Baseline PanArctic\data\e5l_7_sites.csv"
df = pd.read_csv(path)
df = eu.e5lh_to_elm_preprocess(df)
eu.validate_met_vars(df)

points = {'abisko' : (68.35, 18.78333),
        'tvc' : (68.742, -133.499),
        'toolik' : (68.62758, -149.59429),
        'chars' :  (69.1300, -105.0415),
        'qhi' : (69.5795, -139.0762),
        'sam' : (72.22, 126.3),
        'sjb' : (78.92163, 11.83109)}
df_loc = pd.DataFrame({'pid' : points.keys(),
                       'lat' : [points[p][0] for p in points],
                       'lon' : [points[p][1] for p in points]})

dir_out = Path(r'X:\Research\NGEE Arctic\1. Baseline PanArctic\data\temp_make_netcdfs')
dformat='BYPASS'




import os
import xarray as xr
path_samples = r'X:\Research\NGEE Arctic\dapper\data\tl'
files = os.listdir(path_samples)

ds = xr.open_dataset(os.path.join(path_samples, files[0]))

metdata={}
#outvars   - met variables used as ELM inputs
#invars    - corresponding variables to be read from input file
#conv_add  - offset for converting units (e.g. C to K)
#conv_mult - multiplier for converting units (e.g. hPa to Pa, PAR to FSDS)
#valid_min - minimum acceptable value for this variable (set as NaN outside range)
#valid_max - maximum acceptable value for this variable (set as NaN outside range)







# # Spin up a run
# import ee
# import pandas as pd
# from dapper import e5l_utils as eu
# from dapper import utils
# from pathlib import Path

# # Make sure to Initialize with the correct project name (do not use mine--it won't work for you)
# ee.Initialize(project='ee-jonschwenk')

# # Build our points dictionary
# points = {'abisko' : (68.35, 18.78333),
#         'tvc' : (68.742, -133.499),
#         'toolik' : (68.62758, -149.59429),
#         'chars' :  (69.1300, -105.0415),
#         'qhi' : (69.5795, -139.0762),
#         'sam' : (72.22, 126.3),
#         'sjb' : (78.92163, 11.83109)}

# bands = ['temperature_2m',
#         'u_component_of_wind_10m',
#         'v_component_of_wind_10m',
#         'surface_pressure',
#         'dewpoint_temperature_2m',
#         'total_precipitation_hourly',
#         'surface_solar_radiation_downwards_hourly',
#         'surface_thermal_radiation_downwards_hourly',
#         'snow_depth',
#         'snow_depth_water_equivalent',
#         'snow_density',
#         'snow_cover',
#         'snowfall_hourly']


# params = {
#     "start_date": "1950-01-01", # 1950-01-01 is the earliest possible; for speed we just sample a couple years here
#     "end_date": "2100-01-01", # If your end date is longer than what's available, it will just truncate at the last available date. Here I've used the year 2100 to ensure we download all data.
#     "gee_bands": bands,
#     "points": points,  
#     "gdrive_folder": "NGEE_exports",  # Google Drive folder name - will be created if it doesn't exist
#     "filename": "e5l_7_sites"  # Output CSV file name
# }

# # Send the job to GEE!
# eu.sample_e5lh_at_points(params)



# def _export_e5lh_images(params):
#     # Doesn't currently work; not sure if I want to finish it because point sampling might make more sense
#     """Exports ERA5-Land Hourly images over a requested domain.
#     params is a dictionary with the following defined:
#         name : str - name the region/sampling
#         start_date : str - YYYY-MM-DD format
#         end_date : str - YYYY-MM-DD format
#         geometry : shapely.Polygon - defines the region of interest
#         gee_ic : str - should be 'ECMWF/ERA5_LAND/HOURLY'
#         gee_bands : list of str - e.g. ['temperature_2m', 'u_component_of_wind_10m']. These bands must be in the ERA5-Land hourly GEE dataset.
#         gee_output_gdrive_folder : str - folder name to export files
#         gee_batch_nyears : int - number of years to batch the downloading
#         gee_scale : str or int - scale in meters exported pixels should be. Use 'native' to select the native ERA5-Land Hourly resolution of 0.1 degree.
#         out_directory : str - directory to store stuff on your local machine  
#     """

#     # load imagecollection
#     ic = ee.ImageCollection(params['gee_ic'])
#     # filter by datei
#     ic = ic.filterDate(params['start_date'], params['end_date'])
#     # filter by geometry
#     gee_geometry = parse_geometry_object(params['geometry'], params['name'])
#     ic = ic.filterBounds(gee_geometry)
#     # select bands (variables from ERA5-Land hourly)
#     ic = ic.select(params['gee_bands'])

#     # Start jobs on GEE
#     max_date = ic.aggregate_max('system:time_start')
#     last_image = ic.filter(ee.Filter.eq('system:time_start', max_date)).first()
#     last_date = ee.Date(max_date).format("YYYY-MM-dd").getInfo()
#     total_images = (datetime.strptime(last_date, '%Y-%m-%d') - datetime.strptime(params['start_date'], '%Y-%m-%d')).days * 24

#     if params['gee_scale'] == 'native':
#         scale = 11132
#     else:
#         scale = params['gee_scale']

#     geemap.ee_export_image_collection(ic,
#                                       out_dir = params['out_directory'],
#                                       region = ee.Geometry.Polygon(list(params['geometry'].exterior.coords)),
#                                       scale=scale,
#                                       crs='EPSG:4326')
