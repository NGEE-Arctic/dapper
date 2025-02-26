# Functions either in development or used for one-off data stuff. Created by JPS.
# Note that there may be imports here that aren't included in environment.yml file.
# Define the ROOT and DATA directories for ease of use later




# from ngeegee import e5l_utils as eu
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
import ngeegee.e5l_utils as eu
from ngeegee.utils import _ROOT_DIR

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
dformat='CPL_BYPASS'




import os
import xarray as xr
path_samples = r'X:\Research\NGEE Arctic\NGEEGEE\data\tl'
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
# from ngeegee import e5l_utils as eu
# from ngeegee import utils
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

