import ee
import geemap
import pandas as pd
import numpy as np
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
    # Doesn't currently work; not sure if I want to finish it because point sampling might make more sense
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
    

def split_into_dfs(path_csv):
    """
    Splits a GEE-exported csv (from sample_e5lh_at_points) into a dictionary of dataframes
    based on the unique values in the 'pid' column.
    """
    df = pd.read_csv(path_csv)
    return {k : group for k, group in df.groupby('pid')}


def compute_humidities(temp, dewpoint_temp, surf_pressure):
    """
    Ported by JPS from code written by Ryan Crumley.
    temp - (np.array) - array of air temperature values (temperature_2m)
    dewpoint_temp : (np.array) - array of dewpoint temperature values (dewpoint_temperature_2m); must be same length as temp

    Returns:
        RH - relative humidity (%)
        Q - specific humidity (kg/kg)
    """
    # Convert Dewpoint Temp and Temp to RH using Clausius-Clapeyron
    # The following is taken from Margulis 2017 Textbook, Introduction to Hydrology 
    # from pages 49 & 50.
    # More info can be found at: https://margulis-group.github.io/teaching/
    
    # Define some constants
    esat_not = 611 # Constant (Pa)
    rw = 461.52 # Gas constant for moist air (J/kg)
    rd = 287.053 # Gas constant for dry air (J/kg)
    lv = 2453000 # Latent heat of vaporization (J/kg)
    ls = 2838000 # Latent heat of sublimation (J/kg)
    tnot = 273.15 # Temp constant (K)
    
    # Saturated Vapor Pressure (using Temperature)
    # NOTE: if temp is above 0(C) or 273.15(K) then use the latent heat of vaporization
    # and if temp is below 0(C) or 273.15(K) then use the latent heat of sublimation
    eSAT = np.where(temp>=273.15,
                esat_not*np.exp((lv/rw)*((1/tnot) - (1/temp))),
                esat_not*np.exp((ls/rw)*((1/tnot) - (1/temp))))

    # Actual Vapor Pressure (using Dewpoint Temperature)
    e = np.where(temp<=273.15,
            esat_not*np.exp((lv/rw)*((1/tnot) - (1/dewpoint_temp))),
            esat_not*np.exp((ls/rw)*((1/tnot) - (1/dewpoint_temp))))
    
    # Finally, calculate Relative Humidity using the ratio of the vapor pressures at 
    # certain temperatures.
    RH = (e/eSAT)*100

    # Mixing ratio - check units of surf_pressure
    w = (e*rd)/(rw*(surf_pressure-e))

    # Specific Humidity (kg/kg)
    Q = (w/(w+1))

    return RH, Q
