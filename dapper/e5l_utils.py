import ee
import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
from math import floor
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon
from fastparquet import write

from dapper import utils
from dapper import metadata as md


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


def validate_bands(bandlist):
    """
    Ensures that the requested bands are available and errors if not.
    """
    available_bands = set(md.e5lh_bands()['band_name'].tolist())
    not_in = [b for b in bandlist if b not in available_bands]
    if len(not_in) > 0:
        raise NameError("You requested the following bands which are not in ERA5-Land Hourly (perhaps check spelling?): {}. For a list of available bands, run md.e5lh_bands()['band_name'].".format(not_in))
    
    return


def sample_e5lh(params):
    """
    Exports ERA5-Land hourly time-series data for multiple polygons to Google Drive in N-year chunks.

    Input is params, a dictionary with the following keys:
        start_date (str) : YYYY-MM-DD format
        end_date (str) : YYYY-MM-DD format
        geometries (list of dict OR ee.FeatureCollection) : geopandas.GeoDataFrame with 'geometry' and 'pid' columns,  
                                                            OR a pre-loaded GEE FeatureCollection.
        geometry_id_field (str) : the ID field associated with the geometries; is 'gid' by default
        gee_bands (str OR list of str) : 'all' for all bands, 'elm' for ELM-required bands, or a list of specific bands.
        gdrive_folder (str) : Google Drive folder name for export.
        file_name (str) : Base name of the exported CSV file (without extension).
    """
    
    # Populate and validate requested bands
    if params['gee_bands'] == 'all':
        params['gee_bands'] = md.e5lh_bands()['band_name'].tolist()
    elif params['gee_bands'] == 'elm':
        params['gee_bands'] = md.elm_data_dicts()['elm_required_bands']
    else:
        validate_bands(params['gee_bands'])

    # Handle scale
    if params['gee_scale'] == 'native':
        scale = 11132 # Native ERA5-Land hourly scale in meters
    elif params['gee_scale'] < 11132:
        scale = 11132
    else:
        scale = params['gee_scale']

    # Prepare for batching
    if 'gee_years_per_task' not in params:
        params['gee_years_per_task'] = 5

    # Set the imageCollection
    params['gee_ic'] = "ECMWF/ERA5_LAND/HOURLY"
    ic = ee.ImageCollection(params['gee_ic'])

    # Convert start and end dates
    start_date = datetime.strptime(params['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(params['end_date'], "%Y-%m-%d")

    # Find latest available date in the image collection
    max_timestamp = ic.aggregate_max("system:time_start").getInfo()
    max_date = datetime.fromtimestamp(max_timestamp / 1000)

    # Determine number of batches
    batches = utils.determine_gee_batches(start_date, end_date, max_date, years_per_task=params['gee_years_per_task'])

    # Default to 'gid' if no field provided
    if 'geometry_id_field' not in params:
        params['geometry_id_field'] = 'gid'

    # Convert geometries to GEE FeatureCollection (supports dict input OR pre-loaded FeatureCollection)
    if isinstance(params['geometries'], str):
        geometries_fc = ee.FeatureCollection(params['geometries'])  # Directly use pre-loaded GEE asset
    elif isinstance(params['geometries'], ee.FeatureCollection):
        geometries_fc = ee.FeatureCollection(params['geometries']) # re-casting; should already be correct type but this fixes weird errors
    elif isinstance(params['geometries'], gpd.GeoDataFrame):
        gdf_reduced = params['geometries'].copy()
        gdf_reduced = gdf_reduced[[params['geometry_id_field'], 'geometry']]
        geojson_str = gdf_reduced.to_json()    
        geometries_fc = ee.FeatureCollection(json.loads(geojson_str))

    # Function to extract spatially averaged values over each polygon
    def image_to_features(image):
        date = ee.Date(image.get('system:time_start')).format("YYYY-MM-dd HH:mm")

        # Reduce regions (spatial average for each polygon)
        values = image.reduceRegions(
            collection=geometries_fc, 
            reducer=ee.Reducer.mean(),  # Compute spatial mean over polygon
            scale=scale,  # ERA5 spatial resolution ~11.1km
        )

        return values.map(lambda f: f.set("date", date))  # Attach date to results

    # Fire off the Tasks
    for batch_id, bdf in batches.iterrows():
        
        # Filter this Task by date range
        ic_filtered = ic.filterDate(
            bdf['task_start'].strftime("%Y-%m-%d"), 
            bdf['task_end'].strftime("%Y-%m-%d")
        )

        # Apply function over the ImageCollection
        feature_collection = ic_filtered.map(image_to_features).flatten()

        # Create a unique filename for each chunk
        file_suffix = f"{bdf['task_start'].strftime('%Y-%m-%d')}_{bdf['task_end'].strftime('%Y-%m-%d')}"
        export_filename = f"{params['job_name']}_{file_suffix}"

        # Export to Google Drive as CSV
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=export_filename,
            folder=params['gdrive_folder'],
            fileFormat="CSV",
            selectors=[params['geometry_id_field'], 'date'] + params['gee_bands']
        )
        task.start()

        print(f"Export task submitted: {export_filename}")

    return "All export tasks started. Check Google Drive or Task Status in the Javascript Editor for completion."



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
    # Populate and/or validate the requested bands
    if params['gee_bands'] == 'all':
        params['gee_bands'] = md.e5lh_bands()['band_name'].tolist()
    elif params['gee_bands'] == 'elm_required':
        params['gee_bands'] = md.elm_data_dicts()['elm_required_bands']
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


def e5lh_to_elm_unit_conversions(df):
    """
    Converts ERA5-Land hourly bands to units expected by ELM.

    This is not a comprehensive function for all E5LH variables; 
    only ELM variables are handled here.
    """
    # Compute wind magnitude (speed) and direction
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m'].values**2+df['v_component_of_wind_10m'].values**2)
        wind_dir = np.degrees(np.arctan2(df['u_component_of_wind_10m'].values,df['v_component_of_wind_10m'].values))
        wind_dir[np.where(wind_dir>=180)] = wind_dir[np.where(wind_dir>=180)] - 180
        wind_dir[np.where(wind_dir<180)] = wind_dir[np.where(wind_dir<180)] + 180
        df['wind_direction'] = wind_dir

    # Precipitation - convert from meters/hour to mm/second
    if 'total_precipitation_hourly' in df.columns:
        df['total_precipitation_hourly'] = df['total_precipitation_hourly'].values / 3.6

    # Solar rad downwards - convert from J/hr/m2 to W/m2
    if 'surface_solar_radiation_downwards_hourly' in df.columns:
        df['surface_solar_radiation_downwards_hourly'] = df['surface_solar_radiation_downwards_hourly'].values / 3600

    # Thermal rad downwards - convert from J/hr/m2 to W/m2
    if 'surface_thermal_radiation_downwards_hourly' in df.columns:
        df['surface_thermal_radiation_downwards_hourly'] = df['surface_thermal_radiation_downwards_hourly'].values / 3600

    return df


def sample_e5lh_at_points_batch(params):
    """
    Exports ERA5-Land hourly time-series data for multiple points to Google Drive in N year chunks.

    Input is params, a dictionary with the following keys:
        start_date (str) : YYYY-MM-DD format
        end_date (str) : YYYY-MM-DD format
        points (list of dict) : List of dictionaries, each with 'lon', 'lat', and 'pid' keys
        gee_bands (str OR list of str) : 'all' to select all available bands, 'elm' to select ELM-required, or a list of specific bands
        gdrive_folder (str) : Google Drive folder name for export
        file_name (str) : Base name of the exported CSV file (without extension)
    """
    # Populate and/or validate the requested bands
    if params['gee_bands'] == 'all':
        params['gee_bands'] = md.e5lh_bands()['band_name'].tolist()
    elif params['gee_bands'] == 'elm':
        params['gee_bands'] = md.elm_data_dicts()['elm_required_bands']
    else:
        validate_bands(params['gee_bands'])

    # Prepare for batching
    if 'gee_years_per_task' not in params:
        params['gee_years_per_task'] = 5

    # Set the imageCollection
    params['gee_ic'] = "ECMWF/ERA5_LAND/HOURLY"  # ERA5-Land Hourly imageCollection
    ic = ee.ImageCollection(params['gee_ic'])

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(params['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(params['end_date'], "%Y-%m-%d")

    # Find the latest available image in the imageCollection
    max_timestamp = ic.aggregate_max("system:time_start").getInfo()
    max_date = datetime.fromtimestamp(max_timestamp / 1000)

    # Approximate number of images to determine number of batches
    batches = utils.determine_gee_batches(start_date, end_date, max_date, years_per_task=params['gee_years_per_task'])

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

    # Fire off the Tasks
    for batch_id, bdf in batches.iterrows():
        
        # Filter this Task by date range
        ic = ee.ImageCollection(params['gee_ic']).filterDate(
            bdf['task_start'].strftime("%Y-%m-%d"), 
            bdf['task_end'].strftime("%Y-%m-%d")
        )

        # Map function over the ImageCollection
        feature_collection = ic.map(image_to_features).flatten()

        # Create a unique filename for each chunk
        file_suffix = f"{bdf['task_start'].strftime('%Y-%m-%d')}_{bdf['task_end'].strftime('%Y-%m-%d')}"
        export_filename = f"{params['file_name']}_{file_suffix}"

        # Export to Google Drive as CSV
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=export_filename,
            folder=params['gdrive_folder'],
            fileFormat="CSV",
            selectors=['pid', 'date'] + params['gee_bands']
        )
        task.start()

        print(f"Export task submitted: {export_filename}")

    return "All export tasks started. Check Google Drive or Task Status in the Javascript Editor for completion."


def _preprocess_e5hl_to_elm_file(file_path, start_year, end_year, remove_leap):
    """
    Unit conversions, computing indirect variables, and removing negative 
    values for "raw" ERA5-Land (hourly) data. Generalized to handle 
    GEE batching (multiple Tasks instead of just one).

    df : (pandas.DataFrame) - the dataframe containing the raw GEE-exported csv.
    remove_leap : (bool) - True if you want to know how many negative values were replaced for each variable
    verbose : (bool) - True if you want information about what's being corrected
    """
    df = pd.read_csv(file_path)

    # Start with the time dimension
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Clip time so that only full years exist
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]

    # Remove leap days
    if remove_leap is True:
        df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    # Convert units   
    df = e5lh_to_elm_unit_conversions(df)

    # Compute indirect variables (humidities)
    if all(col in df.columns for col in ['temperature_2m', 'dewpoint_temperature_2m', 'surface_pressure']):
        df['relative_humidity'], df['specific_humidity'] = compute_humidities(df['temperature_2m'].values, 
                                                                              df['dewpoint_temperature_2m'].values,
                                                                              df['surface_pressure'].values)
    else:
        print('Missing the required variables to compute humidities.')

    # Enforce non-negativeness for variables for which that is physically impossible
    nonnegs = md.elm_data_dicts()['nonneg']
    for c in df.columns:
        if c in nonnegs:
            negs = df[c]<0
            if sum(negs) > 0:
                df[c].values[negs] = 0
                # if verbose:
                #     pct_neg = sum(negs) / len(df) * 100
                #     print(f"{pct_neg:.2f}% of the values in {c} were negative and reset to 0.")

    return df


def e5hl_to_elm(csv_directory, write_directory, df_loc, remove_leap=True, id_col=None):
    """
    Batched version.

    Need to test that it works for a single csv.

    Next is handling polygon(s)
    """
    if type(csv_directory) is str:
        csv_directory = Path(csv_directory)
    if type(write_directory) is str:
        write_directory = Path(write_directory)


    # Determine our date range to make sure we provide only complete years of data
    files = [f for f in os.listdir(csv_directory) if os.path.splitext(f)[1] == '.csv']
    dates = [pd.read_csv(csv_directory / file, usecols=["date"]) for  file in files]
    dates = pd.concat(dates, ignore_index=True)
    dates['date'] = pd.to_datetime(dates['date'])
    dates.sort_values(by='date', inplace=True)

    # Create temporary folder for storing intermediate results
    temp_path = csv_directory / 'dapper_temp'
    utils.make_directory(temp_path, delete_all_contents=True)

    # Clip to first available Jan 01 year and last available Dec. 31 year.
    dates['year'] = dates['date'].dt.year
    dates['month_day'] = dates['date'].dt.month * 100 + dates['date'].dt.day  # Converts to integer format (e.g., 101 for Jan 1)
    # Group by year and check if both January 1 and December 31 exist
    valid_years = dates.groupby("year")["month_day"].agg(lambda x: {101, 1231}.issubset(set(x)))
    # Get the first and last valid years
    valid_years = valid_years[valid_years].index
    if not valid_years.empty:
        start_year, end_year = valid_years[0], valid_years[-1]
    else:
        start_year, end_year = dates['year'].values[0], dates['year'].values[0]
        print("There is not a full year's worth of data. Using the full dataset.")


    # Preprocess each file, save to intermediate parquet file
    for i, f in enumerate(files):
        file_path = csv_directory / f
        ppdf = _preprocess_e5hl_to_elm_file(file_path, start_year, end_year, remove_leap)

        # Infer id field name if not specified
        # Just uses the shortest field name that has "id" in it
        if i == 0:
            if id_col is None:
                poss_id = [c for c in ppdf.columns if 'id' in c]
                if len(poss_id_lens) == 0:
                    raise NameError("Could not infer id column. Specify it with 'id_col' kwarg when calling e5hl_to_elm().")
                else:
                    poss_id_lens = [len(pi) for pi in poss_id]
                    id_col = poss_id[poss_id_lens.index(min(poss_id_lens))]
                    print(f"Inferred '{id_col}' as id column. If this is not correct, re-run this function and specify 'id_col' kwarg.")
            df_loc.rename(columns={id_col : 'pid'}, inplace=True) # Set id_col to something consistent
        ppdf.rename(columns={id_col : 'pid'}, inplace=True)

        # Split by location and save to parquet
        ppdfg = ppdf.groupby(by='pid')
        for pid, this_df in ppdfg:
            parquet_path = temp_path / f"{pid}.parquet" 
            if os.path.isfile(parquet_path):
                write(parquet_path, this_df, append=True)  
            else:
                write(parquet_path, this_df)  

    # Export ELM netCDFs for each variable for each site parquet file
    utils.make_directory(write_directory, delete_all_contents=True)

    parquet_files = temp_path.glob('*.parquet')
    for pf in parquet_files:
        site = pf.stem
        site_write_directory = write_directory / site
        this_df = pd.read_parquet(pf)
        coords = df_loc[df_loc[id_col]==site]
        export_for_elm_site(this_df, coords['lon'].values[0], coords['lat'].values[0], site_write_directory)

    # Remove temporary files
    utils.remove_directory_contents(temp_path, remove_directory=True)

    return


def export_for_elm_site(df, lon, lat, elm_write_dir, zval=1, dformat='CPL_BYPASS', compress=True, compress_level=4):
    """
    Export in ELM-ready foramts.
    df has all the data. Sorted by date already.
    df_loc has a list of points (pids) and their locations (lat, lon).
    zval is the height in meters of the observations - defaults to 1.
    dformat must be CPL_BYPASS for now.
    """
    # except for 'site', other type of cpl_bypass requires zone_mapping.txt file

    # Grab some metadata dictionaries
    mdd = md.elm_data_dicts()

    if dformat not in ['DATM_MODE', 'CPL_BYPASS']:
        raise KeyError('You provided an unsupported dformat value. Currently only DATM_MODE and CPL_BYPASS are available.')
    elif dformat == 'DATM_MODE':
        print('DATM_MODE is not yet available. Exiting.')
        return
    
    # Make sure directory exists and is empty for each location
    utils.make_directory(elm_write_dir, delete_all_contents=True)

    start_year = str(pd.to_datetime(df['date'].values[0]).year)
    end_year = str(pd.to_datetime(df['date'].values[-1]).year)

    if dformat == 'CPL_BYPASS':
        do_vars = [v for v in mdd['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
    elif dformat == 'DATM_MODE':
        do_vars = [v for v in mdd['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
    
    # Create and save netcdf for each variable
    for elm_var in do_vars:
        era5_var = next((k for k, v in mdd['namemapper'].items() if v == elm_var), None) # Column name in df

        if era5_var not in df.columns:
            raise KeyError('A required variable was not found in the input dataframe: {}'.format(era5_var))
        
        # Packing params (like compression but not quite)
        add_offset, scale_factor = utils.elm_var_compression_params(elm_var)

        # Create dataset
        ds = xr.Dataset(
                            coords={"DTIME": ("DTIME", df['date'])},  # Set DTIME as a coordinate
                            data_vars={
                                "LONGXY": (("n",), np.array([lon], dtype=np.float32)),  
                                "LATIXY": (("n",), np.array([lat], dtype=np.float32)),
                                elm_var: (("n", "DTIME"), df[era5_var].values.reshape(1, -1))  
                            },
                            attrs={"history": "Created using xarray",
                                    'units' : mdd['units'][elm_var],
                                    'description' : mdd['descriptions'][elm_var],
                                    'calendar' : 'noleap',
                                    'created_on' : datetime.today().strftime('%Y-%m-%d'),
                                    'add_offset' : add_offset,
                                    'scale_factor' : scale_factor}
                        )

        # Save file
        filename = 'ERA5_' + elm_var + '_' + start_year + '-' + end_year + '_z' + str(zval) + '.nc'
        this_out_file = elm_write_dir / filename
        ds.to_netcdf(this_out_file, 
                     encoding = { elm_var: {
                                    "dtype": "int16",
                                    "scale_factor": scale_factor,
                                    "add_offset": add_offset,
                                    "_FillValue": -32767
                                        }
                                },
                     zlib=compress,
                     complevel=compress_level
                     )

    return


def export_for_elm_gridded(df, lon, lat, elm_write_dir, zval=1, dformat='CPL_BYPASS', compress=True, compress_level=4):
    """
    Export in ELM-ready foramts.
    df has all the data. Sorted by date already.
    df_loc has a list of points (pids) and their locations (lat, lon).
    zval is the height in meters of the observations - defaults to 1.
    dformat must be CPL_BYPASS for now.
    """
    # except for 'site', other type of cpl_bypass requires zone_mapping.txt file

    # Grab some metadata dictionaries
    mdd = md.elm_data_dicts()

    if dformat not in ['DATM_MODE', 'CPL_BYPASS']:
        raise KeyError('You provided an unsupported dformat value. Currently only DATM_MODE and CPL_BYPASS are available.')
    elif dformat == 'DATM_MODE':
        print('DATM_MODE is not yet available. Exiting.')
        return
    
    # Make sure directory exists and is empty for each location
    utils.make_directory(elm_write_dir, delete_all_contents=True)

    start_year = str(pd.to_datetime(df['date'].values[0]).year)
    end_year = str(pd.to_datetime(df['date'].values[-1]).year)

    if dformat == 'CPL_BYPASS':
        do_vars = [v for v in mdd['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
    elif dformat == 'DATM_MODE':
        do_vars = [v for v in mdd['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
    
    # Create and save netcdf for each variable
    for elm_var in do_vars:
        era5_var = next((k for k, v in mdd['namemapper'].items() if v == elm_var), None) # Column name in df

        if era5_var not in df.columns:
            raise KeyError('A required variable was not found in the input dataframe: {}'.format(era5_var))
        
        # Packing params (like compression but not quite)
        add_offset, scale_factor = utils.elm_var_compression_params(elm_var)

        # Create dataset
        ds = xr.Dataset(
                            coords={"DTIME": ("DTIME", df['date'])},  # Set DTIME as a coordinate
                            data_vars={
                                "LONGXY": (("n",), np.array([lon], dtype=np.float32)),  
                                "LATIXY": (("n",), np.array([lat], dtype=np.float32)),
                                elm_var: (("n", "DTIME"), df[era5_var].values.reshape(1, -1))  
                            },
                            attrs={"history": "Created using xarray",
                                    'units' : mdd['units'][elm_var],
                                    'description' : mdd['descriptions'][elm_var],
                                    'calendar' : 'noleap',
                                    'created_on' : datetime.today().strftime('%Y-%m-%d'),
                                    'add_offset' : add_offset,
                                    'scale_factor' : scale_factor}
                        )

        # Save file
        filename = 'ERA5_' + elm_var + '_' + start_year + '-' + end_year + '_z' + str(zval) + '.nc'
        this_out_file = elm_write_dir / filename
        ds.to_netcdf(this_out_file, 
                     encoding = { elm_var: {
                                    "dtype": "int16",
                                    "scale_factor": scale_factor,
                                    "add_offset": add_offset,
                                    "_FillValue": -32767
                                        }
                                },
                     zlib=compress,
                     complevel=compress_level
                     )

    return


def _preprocess_e5hl_to_elm_file_grid(df, start_year, end_year, remove_leap, dformat):
    """
    Unit conversions, computing indirect variables, and removing negative 
    values for "raw" ERA5-Land (hourly) data. Generalized to handle 
    GEE batching (multiple Tasks instead of just one).

    df : (pandas.DataFrame) - the dataframe containing the raw GEE-exported csv.
    remove_leap : (bool) - True if you want to know how many negative values were replaced for each variable
    verbose : (bool) - True if you want information about what's being corrected
    """
    # Start with the time dimension
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Clip time so that only full years exist
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]

    # Remove leap days
    if remove_leap is True:
        df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    # Convert units   
    df = e5lh_to_elm_unit_conversions(df)

    # Compute indirect variables (humidities)
    if all(col in df.columns for col in ['temperature_2m', 'dewpoint_temperature_2m', 'surface_pressure']):
        df['relative_humidity'], df['specific_humidity'] = compute_humidities(df['temperature_2m'].values, 
                                                                              df['dewpoint_temperature_2m'].values,
                                                                              df['surface_pressure'].values)
    else:
        print('Missing the required variables to compute humidities.')

    # Enforce non-negativeness for variables for which that is physically impossible
    nonnegs = md.elm_data_dicts()['nonneg']
    for c in df.columns:
        if c in nonnegs:
            negs = df[c]<0
            if sum(negs) > 0:
                df[c].values[negs] = 0

    # Rename columns
    mdd = md.elm_data_dicts()
    if dformat == 'CPL_BYPASS':
        do_vars = [v for v in mdd['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
    elif dformat == 'DATM_MODE':
        do_vars = [v for v in mdd['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
    renamer = {k: v for k, v in mdd['short_names'].items() if v in do_vars}
    renamer.update({'date' : 'time', 'lon' : 'LONGXY', 'lat' : 'LATIXY'})
    df.rename(columns=renamer, inplace=True)

    # Drop unnecessary columns
    do_vars.extend(['LONGXY', 'LATIXY', 'time', 'gid', 'zone'])
    df = df[do_vars]

    return df


def e5hl_to_elm_gridded(csv_directory, write_directory, df_loc, remove_leap=True, id_col=None, nzones=1, dformat='CPL_BYPASS', compress=True, compress_level=4):
    """
    Batched version for grids.

    compress_level - higher will compress more but take longer to write

    """
    if dformat not in ['DATM_MODE', 'CPL_BYPASS']:
        raise KeyError('You provided an unsupported dformat value. Currently only DATM_MODE and CPL_BYPASS are available.')
    elif dformat == 'DATM_MODE':
        print('DATM_MODE is not yet available. Exiting.')
        return
    
    if type(csv_directory) is str:
        csv_directory = Path(csv_directory)
    if type(write_directory) is str:
        write_directory = Path(write_directory)

    mdd = md.elm_data_dicts()
    
    # Determine our date range to make sure we provide only complete years of data
    files = [f for f in os.listdir(csv_directory) if os.path.splitext(f)[1] == '.csv']
    dates = [pd.read_csv(csv_directory / file, usecols=["date"]) for  file in files]
    dates = pd.concat(dates, ignore_index=True)
    dates['date'] = pd.to_datetime(dates['date'])
    dates.sort_values(by='date', inplace=True)

    # Clip to first available Jan 01 year and last available Dec. 31 year.
    dates['year'] = dates['date'].dt.year
    dates['month_day'] = dates['date'].dt.month * 100 + dates['date'].dt.day  # Converts to integer format (e.g., 101 for Jan 1)
    # Group by year and check if both January 1 and December 31 exist
    valid_years = dates.groupby("year")["month_day"].agg(lambda x: {101, 1231}.issubset(set(x)))
    # Get the first and last valid years
    valid_years = valid_years[valid_years].index
    if not valid_years.empty:
        start_year, end_year = valid_years[0], valid_years[-1]
    else:
        start_year, end_year = dates['year'].values[0], dates['year'].values[0]
        print("There is not a full year's worth of data. Using the full dataset.")

    # Create temporary folder for storing intermediate results
    utils.make_directory(write_directory, delete_all_contents=True)

    # Rename id field for consistency to 'gid'
    if id_col is None:
        id_col = utils.infer_id_field(df_loc)
    df_loc.rename(columns={id_col:'gid'}, inplace=True)

    # Prepare the netCDF grid
    df_loc = df_loc.sort_values(by=['lat', 'lon']).reset_index(drop=True)
    df_loc['gid'] = df_loc['gid'].astype(str)

    # Account for zones if not already provided in df_loc
    if 'zone' not in df_loc.columns:
        df_loc['zone'] = np.tile(np.arange(1, nzones + 1), (len(df_loc) // nzones) + 1)[:len(df_loc)]
    unique_zones = list(set(df_loc['zone']))

    # Save each file to netCDF
    for i, f in enumerate(files):
        print(f"Processing file {i+1} of {len(files)}: {f}")
        file_path = csv_directory / f
        this_df = pd.read_csv(file_path)

        # Rename id columns for consistency to 'gid'
        if i == 0:
            if id_col is None:
                id_col = utils.infer_id_field(ppdf.columns)
        this_df.rename(columns={id_col:'gid'}, inplace=True)

        # Add lat/lon data to preprocessed dataframe and sort
        this_df = this_df.merge(df_loc[['gid', 'lat', 'lon', 'zone']], on='gid', how='inner')

        ppdf = _preprocess_e5hl_to_elm_file_grid(this_df, start_year, end_year, remove_leap, dformat)
        ppdf = ppdf.sort_values(['time', 'LATIXY', 'LONGXY']).reset_index(drop=True)
         
        for elm_var in ppdf.columns:
            if elm_var in ['gid', 'time', 'LONGXY', 'LATIXY', 'zone']:
                continue
            for zone in unique_zones:

                filename = 'ERA5_' + elm_var + '_' + str(start_year) + '-' + str(end_year) + '_z' + str(zone) + '.nc'
                write_path = write_directory / filename

                # Initialize netCDF file
                if i == 0:
                    this_df_loc = df_loc[df_loc['zone']==zone]
                    utils.create_netcdf(this_df_loc, elm_var, write_path, dformat, compress, compress_level)                

                # Select required vars and zone
                save_df = ppdf[['time', 'LONGXY', 'LATIXY', 'gid', 'zone', elm_var]]
                save_df = save_df[save_df['zone']==zone]
                # Write to netCDF
                utils.append_netcdf(save_df, elm_var, write_path, dformat, compress, compress_level)

    # Generate zone_mappings file
    zm_write_path = write_directory / 'zone_mappings.txt'
    zms = df_loc[['lon', 'lat', 'zone']]
    zms['id'] = np.arange(1, len(zms)+1) # This might not be right, might need to repeat by zone?
    zms.to_csv(zm_write_path, index=False, header=False, sep='\t')

    return



def e5lh_to_elm_preprocess(df, remove_leap=True, verbose=False):
    """
    Unit conversions, computing indirect variables, and removing negative 
    values for "raw" ERA5-Land (hourly) data.

    df : (pandas.DataFrame) - the dataframe containing the raw GEE-exported csv.
    remove_leap : (bool) - True if you want to know how many negative values were replaced for each variable
    verbose : (bool) - True if you want information about what's being corrected
    """

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
        df = df.drop(columns=['year', 'month_day'])
    else:
        print("There is not a full year's worth of data. Using the full dataset.")  

    # Remove leap days
    if remove_leap is True:
        df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    # Convert units, compute indirect variables (humidities)   
    df = e5lh_to_elm_unit_conversions(df)

    if all(col in df.columns for col in ['temperature_2m', 'dewpoint_temperature_2m', 'surface_pressure']):
        df['relative_humidity'], df['specific_humidity'] = compute_humidities(df['temperature_2m'].values, 
                           df['dewpoint_temperature_2m'].values,
                           df['surface_pressure'].values)

    # Enforce non-negativeness for variables for which that is physically impossible
    nonnegs = md.elm_data_dicts()['nonneg']
    for c in df.columns:
        if c in nonnegs:
            negs = df[c]<0
            if sum(negs) > 0:
                df[c].values[negs] = 0
                if verbose:
                    pct_neg = sum(negs) / len(df) * 100
                    print(f"{pct_neg:.2f}% of the values in {c} were negative and reset to 0.")    

    return df


def export_for_elm(df, df_loc, dir_out, zval=1, dformat='CPL_BYPASS'):
    """
    For a single dataframe (one CSV exported from GEE) only! New 'batch' version is available.

    Export in ELM-ready foramts.
    df has all the data. Sorted by date already.
    df_loc has a list of points (pids) and their locations (lat, lon).
    zval is the height in meters of the observations - defaults to 1.
    dformat must be CPL_BYPASS for now.
    """
    # except for 'site', other type of cpl_bypass requires zone_mapping.txt file

    # Grab some metadata dictionaries
    mdd = md.elm_data_dicts()

    if dformat not in ['DATM_MODE', 'CPL_BYPASS']:
        raise KeyError('You provided an unsupported dformat value. Currently only DATM_MODE and CPL_BYPASS are available.')
    elif dformat == 'DATM_MODE':
        print('DATM_MODE is not yet available. Exiting.')
        return
    
    if os.path.isdir(dir_out) is False:
        os.mkdir(dir_out)

    # Split into individual location (based on 'pid') dfs
    dfs = {k : group for k, group in df.groupby('pid')}

    for site in dfs:
        
        # Do for each site
        this_df = dfs[site]    
        start_year = str(pd.to_datetime(this_df['date'].values[0]).year)
        end_year = str(pd.to_datetime(this_df['date'].values[-1]).year)

        if dformat == 'CPL_BYPASS':
            do_vars = [v for v in mdd['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
        elif dformat == 'DATM_MODE':
            do_vars = [v for v in mdd['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
        
        # Create site directory if doesn't exist
        this_out_dir = dir_out / site
        if os.path.isdir(this_out_dir) is False:
            os.mkdir(this_out_dir)

            # Create and save netcdf for each variable
            for elm_var in do_vars:
                era5_var = next((k for k, v in mdd['namemapper'].items() if v == elm_var), None) # Column name in this_df

                if era5_var not in this_df.columns:
                    raise KeyError('A required variable was not found in the input dataframe: {}'.format(era5_var))
                
                # Create dataset
                ds = xr.Dataset(
                                    coords={"DTIME": ("DTIME", this_df['date'])},  # Set DTIME as a coordinate
                                    data_vars={
                                        "LONGXY": (("n",), np.array([df_loc['lon'].values[df_loc['pid']==site][0]], dtype=np.float32)),  # Example data
                                        "LATIXY": (("n",), np.array([df_loc['lat'].values[df_loc['pid']==site][0]], dtype=np.float32)),
                                        era5_var: (("n", "DTIME"), this_df[era5_var].values.reshape(1, -1))  # Example random data
                                    },
                                    attrs={"history": "Created using xarray",
                                           'units' : mdd['units'][elm_var],
                                           'description' : mdd['descriptions'][elm_var],
                                           'calendar' : 'noleap',
                                           'created_on' : datetime.today().strftime('%Y-%m-%d')}
                                )

                # Save file
                filename = 'ERA5_' + elm_var + '_' + start_year + '-' + end_year + '_z' + str(zval) + '.nc'
                this_out_file = this_out_dir / filename

                if os.path.isfile(this_out_file):
                    os.remove(this_out_file)

                ds.to_netcdf(this_out_file)
