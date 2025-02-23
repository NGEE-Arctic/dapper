import ee
import geemap
import pandas as pd
from datetime import datetime
from ngeegee import utils
from shapely.geometry import Polygon

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

def _export_e5lh_images(params):
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

def validate_bands(bandlist):
    """
    Ensures that the requested bands are available and errors if not.
    """
    available_bands = set(e5lh_bands()['band_name'].tolist())
    not_in = [b for b in bandlist if b not in available_bands]
    if len(not_in) > 0:
        raise NameError("You requested the following bands which are not in ERA5-Land Hourly (perhaps check spelling?): {}. For a list of available bands, run eu.e5lh_bands()['band_name'].".format(not_in))
    
    return

def sample_e5lh_at_point(params):
    """
    Exports ERA5-Land hourly time-series data for a single point to Google Drive.
    
    Input is params, a dictionary with the following keys:
        start_date (str) : YYYY-MM-DD format
        end_date (str) : YYYY-MM-DD format
        point (tuple) : (longitude, latitude)
        gee_bands (str OR list of str) : either 'all' to select all available bands, or a list of bands e.g. ['temperature_2m', 'u_component_of_wind_10m']. These bands must be in the ERA5-Land hourly GEE dataset.
        gdrive_folder (str) : name of Google Drive folder to export results into 
        file_name (str) : name of the .csv file to export (without extension)
    """
    # Validate the requested bands - will error if there's a problem
    if params['gee_bands'] == 'all':
        params['gee_bands'] = e5lh_bands()['band_name'].tolist()
    else:
        validate_bands(params['gee_bands'])

    params['gee_ic'] = "ECMWF/ERA5_LAND/HOURLY" # Specify ERA5-Land Hourly imageCollection

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
        
        # Remove unwanted properties
        feature = feature.select(params['gee_bands'] + ["date"])  # Keeps only selected bands + date

        return feature

    feature_collection = ic.map(image_to_feature)

    # Export to Google Drive as CSV
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=params['filename'],
        folder=params['gdrive_folder'],
        fileFormat="CSV",
        selectors=params['gee_bands'] + ['date']
    )
    task.start()

    return f"Export task started: {params['filename']} (Check Google Drive or Task Status in the Javascript Editor for completion.)"


def sample_e5lh_at_points(params):
    """
    Exports ERA5-Land hourly time-series data for multiple points to Google Drive.
    
    Input is params, a dictionary with the following keys:
        start_date (str) : YYYY-MM-DD format
        end_date (str) : YYYY-MM-DD format
        points (list of dict) : List of dictionaries, each with 'lon', 'lat', and 'pid' keys
        gee_bands (str OR list of str) : 'all' to select all available bands, or a list of specific bands
        gdrive_folder (str) : Google Drive folder name for export
        file_name (str) : Name of the exported CSV file (without extension)
    """
    # Validate the requested bands
    if params['gee_bands'] == 'all':
        params['gee_bands'] = e5lh_bands()['band_name'].tolist()
    else:
        validate_bands(params['gee_bands'])

    params['gee_ic'] = "ECMWF/ERA5_LAND/HOURLY"  # ERA5-Land Hourly imageCollection

    # Load ImageCollection and filter by date
    ic = ee.ImageCollection(params['gee_ic']).filterDate(params['start_date'], params['end_date'])

    # Convert list of points into a FeatureCollection
    features = []
    for pointname in params['points']:
        geom = ee.Geometry.Point([params['points'][pointname][1], params['points'][pointname][0]])
        feature = ee.Feature(geom, {'pid': pointname})
        features.append(feature)
    point_fc = ee.FeatureCollection(features)

    # Function to extract values for each image and associate with the points
    def image_to_features(image):
        date = ee.Date(image.get('system:time_start')).format("YYYY-MM-dd HH:mm")
        values = image.reduceRegions(collection=point_fc, reducer=ee.Reducer.first(), scale=11132)
        return values.map(lambda f: f.set("date", date))
    
    # Map function over the ImageCollection
    feature_collection = ic.map(image_to_features).flatten()
    
    # Export to Google Drive as CSV
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=params['filename'],
        folder=params['gdrive_folder'],
        fileFormat="CSV",
        selectors=['pid', 'date'] + params['gee_bands']
    )
    task.start()

    return f"Export task started: {params['filename']} (Check Google Drive or Task Status in the Javascript Editor for completion.)"


def e5lh_bands():
    return pd.read_csv(utils._DATA_DIR / 'e5lh_band_metadata.csv')
    