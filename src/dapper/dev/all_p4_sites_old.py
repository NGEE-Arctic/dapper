import ee
import pandas as pd
import geopandas as gpd
from shapely import Point
import dapper.met.era5land as e5l
from pathlib import Path
from shapely.geometry import Point
from importlib import reload

# # Make sure to Initialize with the correct project name (do not use mine--it won't work for you)
# ee.Initialize(project="ee-jonschwenk")

# data = [
#     [1, "Abisko", "Sweden", 68.35, 18.783333],
#     [2, "Trail Valley Creek", "Canada", 68.742, -133.499],
#     [3, "CHARS", "Canada", 69.1300, -105.0415],
#     [4, "Toolik Lake", "USA", 68.62758, -149.59429],
#     [5, "Qikiqtaruk-Herschel Island", "Canada", 69.5795, -139.0762],
#     [6, "Samoylov Island", "Russia", 72.22, 126.3],
#     [7, "SJ-Blv Bayelva", "Norway", 78.92163, 11.83109],
#     [8, "Imnaviat Creek (Toolik sub-site)", "USA", 68.56066, -149.34047],
#     [9, "Upper Kuparuk (Toolik sub-site)", "USA", 68.60802, -149.30503],
#     [10, "Teller", "USA", 64.735, -165.953],
#     [11, "Kougarok MM 64", "USA", 65.162, -164.834],
#     [12, "Council", "USA", 64.849, -163.707],
#     [13, "Utqiagvik_BEO", "USA", 71.280, -156.605],
# ]
# df = pd.DataFrame(data, columns=["cat", "sitename", "COUNTRY", "lat", "lon"])
# geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
# gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
# gdf.rename({"sitename": "gid"}, inplace=True, axis=1)

# params = {
#     "start_date": "1950-01-01",  # 1950-01-01 is the earliest possible; for speed we just sample a couple years here
#     "end_date": "2100-01-01",  # If your end date is longer than what's available, it will just truncate at the last available date. Here I've used the year 2100 to ensure we download all data.
#     "geometries": gdf,  # Either a GeoDataFrame with Point or Polygon geometries, or a string that is the path to a preloaded GEE asset. Either way, must have a 'gid' column.
#     "gee_bands": "elm",  # You can also specify 'all' to get all bands/variables, or provide a list of variables (e.g. ['temperature_2m', 'total_evaporation_hourly', 'soil_temperature_level_1'])
#     "gee_years_per_task": 10,
#     "gee_scale": "native",  # Can also choose a number in meters. For ERA5-Land hourly data, it does no good to specify anything < 11000 as that's the native scale of the data
#     "job_name": "all_sites_rerun",  # Output CSV file name
#     "gdrive_folder": "all_sites_rerun",  # Google Drive folder name - will be created if it doesn't exist
# }

# # Send the Tasks to GEE! This takes a little while as some time metadata is fetched using getInfo() for GEE.
# df_loc = e5l.sample_e5lh(params, skip_tasks=True)

# import pickle
# df_loc.to_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle.pkl')
df_loc = pd.read_pickle(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\temp_pickle.pkl')


# Wait and download to local
csv_directory = Path(r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs')
write_directory = (csv_directory / "elm_formatted")  # where I want my ELM-formatted netCDFs to go

e5l.e5lh_to_elm(csv_directory, write_directory, df_loc, dtime_resolution_hrs=1)


# import netCDF4
# path = r"X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\elm_formatted\Imnaviat_Creek\ERA5_PRECTmms_1950-2024_z01.nc"
# with netCDF4.Dataset(path) as ds:
#     temperature = ds.variables['PRECTmms'][:]
#     dtime = ds.variables['DTIME'][:]
#     dtime_units = ds.variables['DTIME'].getncattr('units')
#     print(f"DTIME units: {dtime_units}")
#     lat = ds.variables['LATIXY'][:]    
#     lon = ds.variables['LONGXY'][:]
#     var = ds.variables["PRECTmms"]

#     print("scale_factor:", getattr(var, "scale_factor", None))
#     print("add_offset:", getattr(var, "add_offset", None))


#     # Get them all in a dictionary
#     global_attr_names = ds.ncattrs()
#     global_attrs = {name: ds.getncattr(name) for name in global_attr_names}
#     print(f"Global attribute values: {global_attrs}")

# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import numpy as np
# ref_date = datetime(1950, 1, 1)
# date_vals = np.array([ref_date + timedelta(days=float(day)) for day in dtime])

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(date_vals, temperature)
# plt.ylabel("PRECTmms")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
