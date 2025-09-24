import ee
import json
import pandas as pd
import geopandas as gpd
from shapely import Point
import dapper.met.era5land as e5l
from pathlib import Path
from shapely.geometry import Point
from importlib import reload

# Make sure to Initialize with the correct project name (do not use mine--it won't work for you)
ee.Initialize(project="ee-jonschwenk")

# Build the sampling geodataframe
json1 = r"X:\Research\NGEE Arctic\4. Using Dapper\preplab_hack\core_sites.json"
json2 = r"X:\Research\NGEE Arctic\4. Using Dapper\preplab_hack\eval_sites.json"
rows = []
for j in [json1, json2]:
    with open(j) as f:
        data = json.load(f)

    # Extract names and points
    for key, val in data.items():
        name = val["name"]
        lat = val["point"]["lat"]
        lon = val["point"]["lon"]
        rows.append({"name": name, "geometry": Point(lon, lat)})

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
gdf.rename({"name": "gid"}, inplace=True, axis=1)

# Establish the sampling parameters
params = {
    "start_date": "1950-01-01",  # 1950-01-01 is the earliest possible; for speed we just sample a couple years here
    "end_date": "2100-01-01",  # If your end date is longer than what's available, it will just truncate at the last available date. Here I've used the year 2100 to ensure we download all data.
    "geometries": gdf,  # Either a GeoDataFrame with Point or Polygon geometries, or a string that is the path to a preloaded GEE asset. Either way, must have a 'gid' column.
    "gee_bands": "elm",  # You can also specify 'all' to get all bands/variables, or provide a list of variables (e.g. ['temperature_2m', 'total_evaporation_hourly', 'soil_temperature_level_1'])
    "gee_years_per_task": 10, # 10 is reasonable for O(10) points to sample over the full time period; reduce this number if you're using big polygons or many geometries
    "gee_scale": "native",  # Can also choose a number in meters. For ERA5-Land hourly data, it does no good to specify anything < 11000 as that's the native scale of the data
    "job_name": "p4_allsites",  # Output CSV file name
    "gdrive_folder": "p4_allsites",  # Google Drive folder name - will be created if it doesn't exist
}

# # Send the Tasks to GEE! This takes a little while as some time metadata is fetched using getInfo() for GEE.
df_loc = e5l.sample_e5lh(params, skip_tasks=True) # skip_tasks=True can be used to re-generate df_loc without actually sending tasks to GEE.

# import pickle
# df_loc.to_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle.pkl')
df_loc = pd.read_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle.pkl')


# Wait and download to local
csv_directory = Path(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs')
# write_directory = (csv_directory / "elm_formatted_30min")  # where I want my ELM-formatted netCDFs to go
write_directory = (csv_directory / "elm_formatted_3hr")  # where I want my ELM-formatted netCDFs to go

e5l.e5lh_to_elm(csv_directory, write_directory, df_loc, dtime_resolution_hrs=3)


import netCDF4
var = 'PRECTmms'
path = r"X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\elm_formatted_3hr\kfc\ERA5_{}_1950-2024_z01.nc".format(var)
with netCDF4.Dataset(path) as ds:
    datavals = ds.variables[var][:]
    dtime = ds.variables['DTIME'][:]
    dtime_units = ds.variables['DTIME'].getncattr('units')
    print(f"DTIME units: {dtime_units}")
    lat = ds.variables['LATIXY'][:]    
    lon = ds.variables['LONGXY'][:]
    vari = ds.variables[var]

    print("scale_factor:", getattr(vari, "scale_factor", None))
    print("add_offset:", getattr(vari, "add_offset", None))


    # Get them all in a dictionary
    global_attr_names = ds.ncattrs()
    global_attrs = {name: ds.getncattr(name) for name in global_attr_names}
    print(f"Global attribute values: {global_attrs}")

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
ref_date = datetime(1950, 1, 1)
date_vals = np.array([ref_date + timedelta(days=float(day)) for day in dtime])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(date_vals, datavals.data[:][0])
plt.ylabel(var)
plt.grid(True)
plt.tight_layout()
plt.show()


