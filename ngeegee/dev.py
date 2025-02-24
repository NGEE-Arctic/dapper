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
df = pd.read_csv(r'X:\Research\NGEE Arctic\NGEEGEE\notebooks\notebook_data\ngee_test_era5_timeseries.csv')

def e5lh_to_elm_units(df):
    """
    Unit conversions and renaming for "raw" ERA5-Land (hourly) data.

    df : (pandas.DataFrame) - the dataframe containing the raw GEE-exported csv.
    """
    import numpy as np

    nc = nc.rename({'x': 'LON','y': 'LAT'})
    
    temp = nc.band_data[:,0,:,:].rename('temperature_2m')
    temp.attrs['units'] = 'Kelvin'
    temp.attrs['description'] = 'temperature from ERA5-land hourly'

    # Select the uwind band and add names and attributes
    uwind = nc.band_data[:,1,:,:].rename('u_component_wind_10m')
    uwind.attrs['units'] = 'm/s'
    uwind.attrs['description'] = 'Eastern component of 10m wind from ERA5-land hourly'
    
    # Select the vwind band and add names and attributes
    vwind = nc.band_data[:,2,:,:].rename('v_component_wind_10m')
    vwind.attrs['units'] = 'm/s'
    vwind.attrs['description'] = 'Northern component of 10m wind from ERA5-land hourly'
    
    # Select the surface pressure band and add names and attributes
    sp = nc.band_data[:,3,:,:].rename('surface_pressure')
    sp.attrs['units'] = 'Pa'
    sp.attrs['description'] = 'Surface pressure (force per unit area) from ERA5-land hourly'
    # Select the dewpoint temperature band and add names and attributes
    dpt = nc.band_data[:,4,:,:].rename('dewpoint_temperature_2m')
    dpt.attrs['units'] = 'Kelvin'
    dpt.attrs['description'] = 'Dewpoint temperature at 2m from ERA5-land hourly'
    # Select the precipitation band and add names and attributes
    prec = nc.band_data[:,5,:,:].rename('total_precipitation_hourly')
    prec.attrs['units'] = 'meters'
    prec.attrs['description'] = 'Total precipitation hourly (disaggregated) from ERA5-land hourly'
    
    # Select the solar radiation downwards band and add names and attributes
    ssrd = nc.band_data[:,6,:,:].rename('surface_solar_radiation_downwards_hourly')
    ssrd.attrs['units'] = 'Joules/m2'
    ssrd.attrs['description'] = 'Surface solar radiation downwards (disaggregated) to hourly from ERA5-land hourly'
    
    # Select the thermal radiation downwards band and add names and attributes
    strd = nc.band_data[:,7,:,:].rename('surface_thermal_radiation_downwards_hourly')
    strd.attrs['units'] = 'Joules/m2'
    strd.attrs['description'] = 'Surface thermal radiation downwards (disaggregated) to hourly from ERA5-land hourly'

    
    # Compute relative humidity 
    if 'temperature_2m' in df.columns and 'dewpoint_temperature_2m' in df.columns:
        df['rel_hum'] = rh(df['temperature_2m'].values, df['dewpoint_temperature_2m'].values)
    
    # Compute wind magnitude
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m'].values**2+df['v_component_of_wind_10m'].values**2)

    # Compute wind direction; 0 is true North, 90 is east, etc.
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        wind_dir = np.degrees(np.arctan2(df['u_component_of_wind_10m'].values,df['v_component_of_wind_10m']))
        wind_dir[np.where(wind_dir)>=180] = wind_dir[np.where(wind_dir)>=180] - 180
        wind_dir[np.where(wind_dir)<180] = wind_dir[np.where(wind_dir)<180] + 180
        df['wind_direction'] = wind_dir

    # Convert from (m) to (mm)
    Prtmp = Prtmp*1000

    #Convert T to C
    Ttmp=Ttmp-273.16

        for z in range(len(Prtmp)): 

            fid.write('{:5.0f}\t'.format(int(year[j]))+'{:5.0f}\t'.format(int(month[j]))+
                        '{:3.0f}\t'.format(int(day[j]))+'{:6.3f}\t'.format(int(hour[j]))+
                        '{:9.0f}\t'.format(int(ID[z]))+'{:12.1f}\t'.format(X[z])+
                        '{:12.1f}\t'.format(Y[z])+'{:8.1f}\t'.format(elev[z])+
                        '{:9.2f}\t'.format(Ttmp[z])+'{:9.2f}\t'.format(RHtmp[z])+
                        '{:9.2f}\t'.format(SPDtmp[z])+'{:9.2f}\t'.format(DIRtmp[z])+
                        '{:9.3f}\n'.format(Prtmp[z]))



    return


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

