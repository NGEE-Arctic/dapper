# Functions either in development or used for one-off data stuff. Created by JPS.
# Note that there may be imports here that aren't included in environment.yml file.
# Define the ROOT and DATA directories for ease of use later

def download_example_era5landhourly_image(path_out_nc='era5_land_20240101_1200.zip'):
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': ['2m_temperature'],  # Example variable
            'year': '2024',
            'month': '01',
            'day': '01',
            'time': '12:00',
            'data_format ': 'zip'
        },
        path_out_nc)


def export_e5lh_grid(path_e5lh_file=r'notebooks/data/e5lh_grid_sample.grib', path_out=r'notebooks/data'):
    
    # Make a shapefile of the grid using the example image
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    import numpy as np
    import xarray as xr
    import os

    # Load the NetCDF file
    ds = xr.open_dataset(path_e5lh_file, engine='cfgrib')

    # Extract latitude, longitude, and temperature variable (t2m)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    t2m = ds["t2m"].values  # Take first time step

    # Determine grid resolution
    lat_res = abs(lat[1] - lat[0])
    lon_res = abs(lon[1] - lon[0])

    # List to store polygons
    polygons, points, pids = [], [], []

    # Iterate over grid cells
    for i in range(len(lat)):
        if lat[i] < 55:  # Skip latitudes below 55°N
            continue
        
        for j in range(len(lon)):
            if np.isnan(t2m[i, j]):  # Skip ocean/no-data cells
                continue

            # Define grid cell corners
            lon_min, lon_max = lon[j] - lon_res / 2, lon[j] + lon_res / 2
            lat_min, lat_max = lat[i] - lat_res / 2, lat[i] + lat_res / 2

            # Create a polygon for the valid grid cell
            polygon = Polygon([
                (lon_min, lat_min),
                (lon_max, lat_min),
                (lon_max, lat_max),
                (lon_min, lat_max),
                (lon_min, lat_min)  # Close the polygon
            ])

            polygons.append(polygon)
            points.append(Point(lon[j], lat[i]))
            pids.append(f"{lat[i]:.2f},{lon[j]:.2f}")
            print(lat[i], lon[j])

    # Create a GeoDataFrame
    gdf_poly = gpd.GeoDataFrame(geometry=polygons, data={'pids' : pids}, crs="EPSG:4326")  # WGS84 CRS
    gdf_poly.to_file(os.path.join(path_out, 'e5lh_grid.shp'))

    gdf_points = gpd.GeoDataFrame(geometry=points, data={'pids' : pids}, crs="EPSG:4326")  # WGS84 CRS
    gdf_points.to_file(os.path.join(path_out, 'e5lh_centerpoints.shp'))

    return


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

def e5lh_to_elm_preprocess(df):
    """
    Unit conversions and renaming for "raw" ERA5-Land (hourly) data.

    df : (pandas.DataFrame) - the dataframe containing the raw GEE-exported csv.
    """
    from ngeegee.e5l_utils import compute_humidities

    # Start with the time dimension
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Clip to first available Jan 01 year and last available Dec. 31 year.
    df["year"] = df["date"].dt.year
    df["month_day"] = df["date"].dt.month * 100 + df["date"].dt.day  # Converts to integer format (e.g., 101 for Jan 1)
    # Group by year and check if both January 1 and December 31 exist
    valid_years = df.groupby("year")["month_day"].agg(lambda x: {101, 1231}.issubset(set(x)))
    # Get the first and last valid years
    valid_years = valid_years[valid_years].index
    if not valid_years.empty:
        start_year, end_year = valid_years[0], valid_years[-1]
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        df.drop(columns=['year', 'month_day'], inplace=True)
    else:
        print("There is not a full year's worth of data. Using the full dataset.")  

    # Remove leap days
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    # Add days since Jan 01 of first year
    # df['DTIME'] = np.arange(0, len(df), 1) 

    # # Reformat date column to string
    # df['date'] = df['date'].dt.strftime('%Y-%m-%dT00:00:00')

    # Convert units, compute indirect variables (humidities)
    
    # Compute wind magnitude
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m'].values**2+df['v_component_of_wind_10m'].values**2)

    # Compute wind direction; 0 is true North, 90 is east, etc.
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        wind_dir = np.degrees(np.arctan2(df['u_component_of_wind_10m'].values,df['v_component_of_wind_10m'].values))
        wind_dir[np.where(wind_dir>=180)] = wind_dir[np.where(wind_dir>=180)] - 180
        wind_dir[np.where(wind_dir<180)] = wind_dir[np.where(wind_dir<180)] + 180
        df['wind_direction'] = wind_dir

    # Precipitation - convert from meters/hour to mm/second
    if 'total_precipitation_hourly' in df.columns:
        df['total_precipitation_hourly'] = df['total_precipitation_hourly'].values / 3.6
        df['total_precipitation_hourly'].values[df['total_precipitation_hourly']<0] = 0 # negatives to 0

    # Temperature - convert to Celcius
    if 'temperature_2m' in df.columns:
        df['temperature_2m'] = df['temperature_2m'].values - 273.16

    # Solar rad downwards - convert from J/hr/m2 to W/m2
    if 'surface_solar_radiation_downwards_hourly' in df.columns:
        df['surface_solar_radiation_downwards_hourly'] = df['surface_solar_radiation_downwards_hourly'].values / 3600

    # Thermal rad downwards - convert from J/hr/m2 to W/m2
    if 'surface_thermal_radiation_downwards_hourly' in df.columns:
        df['surface_thermal_radiation_downwards_hourly'] = df['surface_thermal_radiation_downwards_hourly'].values / 3600

    # Compute humidities 
    if 'temperature_2m' in df.columns and 'dewpoint_temperature_2m' in df.columns and 'surface_pressure' in df.columns:
        df['relative_humidity'], df['specific_humidity'] = compute_humidities(df['temperature_2m'].values, 
                           df['dewpoint_temperature_2m'].values,
                           df['surface_pressure'].values)

    # Add validation (max/min of each variable)
    # outvars  = ['TBOT','RH','WIND','PSRF','FSDS',    'PRECTmms']
    # valid_min= [180.00,   0,     0,  8e4,         0,      0]
    # valid_max= [350.00,100.,    80,1.5e5,      2500,      15]

    return df

import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime
from ngeegee import utils
from shapely.geometry import Polygon
import os
import xarray as xr
import numpy as np
from datetime import datetime
from pathlib import Path
from ngeegee.utils import elm_data_dicts as edd

path = r"X:\Research\NGEE Arctic\1. Baseline PanArctic\data\e5l_7_sites.csv"
df = pd.read_csv(path)
df = e5lh_to_elm_preprocess(df)

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
def export_for_elm(df, df_loc, dir_out, dformat='CPL_BYPASS'):
    """
    Export in ELM-ready foramts.
    df has all the data. Sorted by date already.
    df_loc has a list of points (pids) and their locations (lat, lon).
    dformat must be CPL_BYPASS for now.
    """
    # except for 'site', other type of cpl_bypass requires zone_mapping.txt file

    # Grab some metadata dictionaries
    md = edd()

    if dformat not in ['DATM_MODE', 'CPL_BYPASS']:
        raise KeyError('You provided an unsupported dformat value. Currently only DATM_MODE and CPL_BYPASS are available.')
    elif dformat == 'DATM_MODE':
        print('DATM_MODE is not yet available. Exiting.')
        return

    zval = 13 # just making this up for now


    # Split into individual location (based on 'pid') dfs
    dfs = {k : group for k, group in df.groupby('pid')}

    for this_site in dfs:
        
        # Do for each site
        this_df = dfs[this_site]    
        start_year = str(pd.to_datetime(this_df['date'].values[0]).year)
        end_year = str(pd.to_datetime(this_df['date'].values[-1]).year)

        if dformat == 'CPL_BYPASS':
            do_vars = [v for v in md['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
        elif dformat == 'DATM_MODE':
            do_vars = [v for v in md['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
        
        # Create site directory if doesn't exist
        this_out_dir = dir_out / this_site
        if os.path.isdir(this_out_dir) is False:
            os.mkdir(this_out_dir)

            # Create and save netcdf for each variable
            for elm_var in do_vars:
                era5_var = next((k for k, v in md['namemapper'].items() if v == elm_var), None) # Column name in this_df

                if era5_var not in this_df.columns:
                    raise KeyError('A required variable was not found in the input dataframe: {}'.format(era5_var))
                
                # Create dataset
                ds = xr.Dataset(
                                    coords={"DTIME": ("DTIME", this_df['date'])},  # Set DTIME as a coordinate
                                    data_vars={
                                        "LONGXY": (("n",), np.array([df_loc['lon'].values[df_loc['pid']==this_site][0]], dtype=np.float32)),  # Example data
                                        "LATIXY": (("n",), np.array([df_loc['lat'].values[df_loc['pid']==this_site][0]], dtype=np.float32)),
                                        era5_var: (("n", "DTIME"), this_df[era5_var].values.reshape(1, -1))  # Example random data
                                    },
                                    attrs={"history": "Created using xarray",
                                           'units' : md['units'][elm_var],
                                           'description' : md['descriptions'][elm_var],
                                           'calendar' : 'noleap',
                                           'created_on' : datetime.today().strftime('%Y-%m-%d')}
                                )

                # Save file
                filename = 'ERA5_' + elm_var + '_' + start_year + '-' + end_year + '_z' + str(zval) + '.nc'
                this_out_file = this_out_dir / filename

                if os.path.isfile(this_out_file):
                    os.remove(this_out_file)

                ds.to_netcdf(this_out_file)




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

#Note - FLDS not included here (calculated)
outvars  = ['TBOT','RH','WIND','PSRF','FSDS',    'PRECTmms']
invars   = ['TA',  'RH','WS',  'PA', 'PPFD_OUT', 'H2O']   #matching header of input file
conv_add = [273.15,   0,     0,    0,         0,      0]
conv_mult= [     1,   1,     1, 1000,1./(0.48*4.6),   1]
valid_min= [180.00,   0,     0,  8e4,         0,      0]
valid_max= [350.00,100.,    80,1.5e5,      2500,      15]

dstemp = xr.open_dataset(r"X:\Research\NGEE Arctic\NGEEGEE\data\fengming_data\Daymet_ERA5.1km_FLDS_1980-2023_z01.nc")



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

