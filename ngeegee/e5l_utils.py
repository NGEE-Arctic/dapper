import ee
import geemap
import pandas as pd
from datetime import datetime
from shapely.geometry import Polygon

# ee.Initialize(project='ee-jonschwenk')

# ## Generate a clipping polygon; you can also load in a shapefile and pull out the geometry
# # Define central point
# lat, lon = 68.62758, -149.59429

# # Define size of bounding box
# lon_width = 0.05 # This is the total width
# lat_height = 0.05 # This is the total height

# # Define the four corners of the polygon (lon, lat order for Shapely)
# polygon = Polygon([
#     (lon - lon_width/2, lat - lat_height/2),  # Bottom-left
#     (lon + lon_width/2, lat - lat_height/2),  # Bottom-right
#     (lon + lon_width/2, lat + lat_height/2),  # Top-right
#     (lon - lon_width/2, lat + lat_height/2),  # Top-left
#     (lon - lon_width/2, lat - lat_height/2)   # Closing the polygon
# ])

# # 37 minutes for 25 years, 2 bands


params = {
    'name' : 'test', # Name the region/sampling
    'start_date' : '1950-01-01', # YYYY-MM-DD
    'end_date' : '1950-02-01', # YYYY-MM-DD
    'geometry' : polygon, # This must be EPSG:4326; we do not check this so get it right
    'gee_ic' : 'ECMWF/ERA5_LAND/HOURLY', # Path to GEE Asset to sample; here we choose ERA5-Land Hourly
    'gee_bands' : [ 'temperature_2m', # These bands must match the gee_ic ImageCollection
                    'u_component_of_wind_10m',
                    'v_component_of_wind_10m',
                    'surface_pressure',
                    'dewpoint_temperature_2m',
                    'total_precipitation_hourly',
                    'surface_solar_radiation_downwards_hourly',
                    'surface_thermal_radiation_downwards_hourly',
                    'snow_depth',
                    'snow_depth_water_equivalent',
                    'snow_density',
                    'snow_cover',
                    'snowfall_hourly'],
    'gee_output_gdrive_folder' : 'test', # Which folder to store on your GDrive; will be created if not exists
    'gee_batch_nyears' : 5, # Number of years to batch GEE exports
    'gee_scale' : 'native', # scale is in meters; 'native' will use the native resolution 
    'out_directory' : r'X:\Research\NGEE Arctic\NGEEGEE\temp_data' 
}

def parse_geometry_object(geom, name): # Function to translate gdf geometries to ee geometries
    
    if type(geom) is str: # GEE Asset
        ret = geom
    elif type(geom) in [Polygon]:
        eegeom = ee.Geometry.Polygon(list(geom.exterior.coords))
        eefeature = ee.Feature(eegeom, {'name': name})
        ret = ee.FeatureCollection(eefeature)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")
    
    return ret

def export_e5lh_images(params):
    """Exports ERA5-Land Hourly images over a requested domain.
    params is a dictionary with the following defined:
        name : str - name the region/sampling
        start_date : str - YYYY-MM-DD format
        end_date : str - YYYY-MM-DD format
        geometry : shapely.Polygon - defines the region of interest
        gee_ic : str - should be 'ECMWF/ERA5_LAND/HOURLY'
        gee_bands : list of str - e.g. ['temperature_2m', 'u_component_of_wind_10m']. These bands must be in the ERA5-Land hourly GEE dataset.
        gee_output_gdrive_folder : str - folder name to export files
        gee_batch_nyears : int - number of years to batch the downloading
        gee_scale : str or int - scale in meters exported pixels should be. Use 'native' to select the native ERA5-Land Hourly resolution of 0.1 degree.
        out_directory : str - directory to store stuff on your local machine  
    """

    # load imagecollection
    ic = ee.ImageCollection(params['gee_ic'])
    # filter by datei
    ic = ic.filterDate(params['start_date'], params['end_date'])
    # filter by geometry
    gee_geometry = parse_geometry_object(params['geometry'], params['name'])
    ic = ic.filterBounds(gee_geometry)
    # select bands (variables from ERA5-Land hourly)
    ic = ic.select(params['gee_bands'])

    # Start jobs on GEE
    max_date = ic.aggregate_max('system:time_start')
    last_image = ic.filter(ee.Filter.eq('system:time_start', max_date)).first()
    last_date = ee.Date(max_date).format("YYYY-MM-dd").getInfo()
    total_images = (datetime.strptime(last_date, '%Y-%m-%d') - datetime.strptime(params['start_date'], '%Y-%m-%d')).days * 24

    if params['gee_scale'] == 'native':
        scale = 11132
    else:
        scale = params['gee_scale']

    geemap.ee_export_image_collection(ic,
                                      out_dir = params['out_directory'],
                                      region = ee.Geometry.Polygon(list(params['geometry'].exterior.coords)),
                                      scale=scale,
                                      crs='EPSG:4326')



def sample_e5lh_at_point(params):
    """
    Exports ERA5-Land hourly time-series data for a single point to Google Drive.

    Args:
        params (dict): Dictionary with required parameters:
            - 'gee_ic': GEE ImageCollection ID (e.g., "ECMWF/ERA5_LAND/HOURLY")
            - 'start_date': Start date (YYYY-MM-DD)
            - 'end_date': End date (YYYY-MM-DD)
            - 'gee_bands': List of bands to extract (e.g., ['temperature_2m', 'total_precipitation'])
            - 'gee_scale': Scale in meters (e.g., 1000) or 'native' for default scale
            - 'point': Tuple of (longitude, latitude)
            - 'drive_folder': Name of the Google Drive folder for export
            - 'file_name': Name of the output CSV file

    Returns:
        str: Status message confirming the export process has started.
    """

    # Load ImageCollection and filter by date
    ic = ee.ImageCollection(params['gee_ic']).filterDate(params['start_date'], params['end_date'])

    # Create a single-point geometry
    point = ee.Geometry.Point(params['point'])

    # Filter ImageCollection to the given point
    ic = ic.filterBounds(point).select(params['gee_bands'])

    # Convert images to a FeatureCollection with date as a property
    def image_to_feature(image):
        date = ee.Date(image.get('system:time_start')).format("YYYY-MM-dd HH:mm")
        values = image.reduceRegion(ee.Reducer.first(), point, scale=11132)
        feature = ee.Feature(None, values).set("date", date)
        return feature

    feature_collection = ic.map(image_to_feature)

    # Export to Google Drive as CSV
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=params['filename'],
        folder=params['gdrive_folder'],
        fileFormat="CSV"
    )
    task.start()

    return f"Export task started: {params['filename']} (Check Google Drive or Task Status in the Javascript Editor)"
