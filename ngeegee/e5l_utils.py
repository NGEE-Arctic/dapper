import ee
import os
import geemap
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from ngeegee import utils
from shapely.geometry import Polygon
from ngeegee import metadata as md

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
    available_bands = set(md.e5lh_bands()['band_name'].tolist())
    not_in = [b for b in bandlist if b not in available_bands]
    if len(not_in) > 0:
        raise NameError("You requested the following bands which are not in ERA5-Land Hourly (perhaps check spelling?): {}. For a list of available bands, run md.e5lh_bands()['band_name'].".format(not_in))
    
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


def validate_met_vars(df):
    """
    Uses pre-computed statistics to ensure that the unit conversions resulted in
    distributions for each variable that make sense.
    """
    # Load pre-computed variable statistics
    path_stats = utils._DATA_DIR / 'elm_met_var_stats.csv'
    sdf = pd.read_csv(path_stats, index_col=0)

    # Determine which variables can/can't be validated
    namemap = md.elm_data_dicts()['namemapper']
    nostats = []
    for c in df.columns:
        if c in ['pid', 'date']:
                continue
        if c in namemap:
                if namemap[c] in sdf.columns:
                    continue
                else:
                    nostats.append(c)
        else:
                nostats.append(c)
    check_vars = set(df.columns) - set(nostats) - set(['pid', 'date'])

    # Perform the validation of data ranges and orders of magnitude
    for v in check_vars:
        dmean, dmin, dmax = df[v].mean(), df[v].min(), df[v].max()

        this_stats = sdf[namemap[v]]
        vmean, vmin, vmax = this_stats['mean'], this_stats['min'], this_stats['max']

        # Check order of magnitude
        oom_dif = np.log10(vmean) - np.log10(dmean)
        if abs(oom_dif) > 0.5:
            print('HIGH CONCERN: {} is {} orders of magnitude different mean than the reference variable {}.'.format(v, f"{oom_dif:.1f}", namemap[v]))

        # Check range
        frac_beyond_range = np.sum(np.logical_or(df[v].values>vmax, df[v].values<vmin))/ len(df)
        if frac_beyond_range > 0.1: # More than 10% raise concern
             print("LOW CONCERN: {}% of the values in {} are beyond the range of the reference variable {}.".format(int(frac_beyond_range*100), v, namemap[v]))

        # OLMT provided the following code as well: see https://github.com/dmricciuto/OLMT/blob/ca01781f4925e4aad32cc697c2d09eb94eddd920/metdata_tools/site/data_to_elmbypass.py#L30
        # Use the OLMT ranges as an additional check
        olmt_vars = ['TBOT','RH','WIND','PSRF','FSDS','PRECTmms']
        olmt_mins = [180.00,   0,     0,  8e4,         0,      0]
        olmt_maxs = [350.00,100.,    80,1.5e5,      2500,      15]
        if namemap[v] in olmt_vars:
             if dmax > olmt_maxs[olmt_vars.index(namemap[v])] or  dmin < olmt_mins[olmt_vars.index(namemap[v])]:
                  print('MED CONCERN: the max and/or min values in {} exceed the expected range provided by OLMT (variable name {}).'.format(v, namemap[v]))

    if len(nostats) > 0:
         print('No reference statistics were available for the following variables, so their ranges were not validated: {}'.format(nostats))

    # Perform validation of negative values
    nonneg_bands = md.elm_data_dicts()['nonneg']
    for c in df.columns:
         if c in nonneg_bands:
            negs = df[c]<0
            if sum(negs) > 0:
                 print({'Negative values detected in variable {}'.format(c)})

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

    for this_site in dfs:
        
        # Do for each site
        this_df = dfs[this_site]    
        start_year = str(pd.to_datetime(this_df['date'].values[0]).year)
        end_year = str(pd.to_datetime(this_df['date'].values[-1]).year)

        if dformat == 'CPL_BYPASS':
            do_vars = [v for v in mdd['req_vars']['cbypass'] if v not in ['LONGXY', 'LATIXY', 'time']]
        elif dformat == 'DATM_MODE':
            do_vars = [v for v in mdd['req_vars']['datm'] if v not in ['LONGXY', 'LATIXY', 'time']]
        
        # Create site directory if doesn't exist
        this_out_dir = dir_out / this_site
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
                                        "LONGXY": (("n",), np.array([df_loc['lon'].values[df_loc['pid']==this_site][0]], dtype=np.float32)),  # Example data
                                        "LATIXY": (("n",), np.array([df_loc['lat'].values[df_loc['pid']==this_site][0]], dtype=np.float32)),
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
