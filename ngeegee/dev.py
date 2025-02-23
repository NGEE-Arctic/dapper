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


