# Functions that were only used once or have limited/no reusability.

def compute_elm_met_stats():
    """
    This function takes a set of ELM met input files and computes the mean, min, and max,
    then stores these statistics as a .csv. These statistics are used to validate
    unit conversions. 
    
    The sample files were taken from Trail Valley Creek from this repo:
    https://github.com/fmyuan/pt-e3sm-inputdata/tree/master/atm/datm7/Daymet_ERA5/BYPASS_TVC

    Validation should only consider order of magnitude, or other broad metrics, as TVC
    is obviously not representative of all potential sites.
    """
    import os
    import xarray as xr
    from dapper import utils
    import pandas as pd

    path_sample_files = r'X:\Research\NGEE Arctic\dapper\data\fengming_data'
    files = os.listdir(path_sample_files)
    stats = {}
    for f in files:
        var = f.split('_')[2]
        ds = xr.open_dataset(os.path.join(path_sample_files, f))
        data = ds[var].data
        dmean = data.mean()
        dmin = np.percentile(data, 1)
        dmax = np.percentile(data, 99)
        stats[var] = {'mean' : dmean,
                    'min' : dmin,
                    'max' : dmax}
    sdf = pd.DataFrame(stats)  
    sdf.to_csv(utils._DATA_DIR / 'elm_met_var_stats.csv')  


def download_example_era5landhourly_image(path_out_nc='era5_land_20240101_1200.zip'):
    """
    I needed an ERA5-Land Hourly image in order to know the gridcell boundaries.
    """
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
    """
    Makes a shapefile of the grid of ERA5-Land Hourly (0.1 degree grid)
    for Arctic locations. Exports both gridcell polygons and centerpoints.
    """

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
