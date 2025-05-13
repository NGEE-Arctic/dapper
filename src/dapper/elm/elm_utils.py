import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime

from dapper.elm import elm_utils as eutils
from dapper.utils import _DATA_DIR

def create_met_netcdf(df_loc, elm_var, write_path, dformat='BYPASS', compress=False, compress_level=0):
    if os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' already exists.")
        return

    if dformat == 'BYPASS':
        mdd = eutils.elm_data_dicts()
        add_offset, scale_factor = elm_var_compression_params(elm_var)

        # Ensure df_loc is sorted by LATIXY, LONGXY for consistent ordering
        df_loc = df_loc.sort_values(['lat', 'lon']).reset_index(drop=True)

        with nc.Dataset(write_path, mode='w', format='NETCDF4') as ds:
            ds.createDimension('n', len(df_loc))
            ds.createDimension('DTIME', None)

            # Define coordinate variables with correct names and types
            lat = ds.createVariable('LATIXY', 'f4', ('n',))
            lon = ds.createVariable('LONGXY', 'f4', ('n',))
            gid = ds.createVariable('gid', str, ('n',))
            
            # Define DTIME as datetime64[ns]
            dtime = ds.createVariable('DTIME', 'f8', ('DTIME',))
            dtime.setncattr('units', 'seconds since 1970-01-01 00:00:00')
            dtime.setncattr('calendar', 'standard')

            # Define data variable with (DTIME, n) order
            var = ds.createVariable(elm_var, 'f4', ('DTIME', 'n'), zlib=compress, complevel=compress_level, fill_value=-32767)

            # Populate lat, lon, and gid in sorted order
            lat[:] = df_loc['lat'].values
            lon[:] = df_loc['lon'].values
            gid[:] = df_loc['gid'].values

            # Set attributes for coordinates
            lat.setncattr('units', 'degrees_north')
            lon.setncattr('units', 'degrees_east')

            # Set attributes for data variable
            var.setncattr('add_offset', add_offset)
            var.setncattr('scale_factor', scale_factor)
            var.setncattr('units', mdd['units'][elm_var])
            var.setncattr('description', mdd['descriptions'][elm_var])

            ds.setncattr('history', "Created using netCDF4")
            ds.setncattr('calendar', 'noleap')
            ds.setncattr('created_on', datetime.today().strftime('%Y-%m-%d'))

    return


def append_met_netcdf(this_df, elm_var, write_path, dformat='BYPASS', compress=False, compress_level=0):
    if not os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' does not exist and cannot be appended.")
        return

    if dformat == 'BYPASS':
        with nc.Dataset(write_path, mode='a') as ds:
            if elm_var not in ds.variables:
                print(f"{elm_var} is missing in {write_path}. Cannot append data.")
                return

            add_offset = ds.variables[elm_var].getncattr('add_offset')
            scale_factor = ds.variables[elm_var].getncattr('scale_factor')

            # Convert times to datetime64[ns] and then to seconds since 1970-01-01
            times = pd.to_datetime(this_df['time']).values.astype('datetime64[ns]')
            unique_times = np.unique(times)
            int64_times = (unique_times.astype('datetime64[ns]').astype('int64') // 10**9).astype('f8')

            # Get existing size of DTIME and GID dimensions
            current_time_size = ds.dimensions['DTIME'].size
            current_gid_size = ds.dimensions['n'].size

            # Append new times directly in one go as float64 (seconds since 1970-01-01)
            ds['DTIME'][current_time_size:current_time_size + len(int64_times)] = int64_times

            # Sort this_df by LATIXY, LONGXY to match NetCDF order
            this_df = this_df.sort_values(['LATIXY', 'LONGXY']).reset_index(drop=True)

            # Pack data to float32 and apply offset/scale
            packed_data_column = np.round((this_df[elm_var].values - add_offset) / scale_factor).astype(np.float32)

            # Reshape data: (DTIME, n) order
            num_times = len(unique_times)
            reshaped_data = packed_data_column.reshape(num_times, current_gid_size)

            # Append reshaped data as float32 in (DTIME, n) order
            ds[elm_var][current_time_size:current_time_size + num_times, :] = reshaped_data

            # Force writing data to disk
            ds.sync()

        # print(f"Successfully appended data for {elm_var} to {write_path}.")

def validate_met_vars(df):
    """
    Uses pre-computed statistics to ensure that the unit conversions resulted in
    distributions for each variable that make sense.
    """
    # Load pre-computed variable statistics
    path_stats = _DATA_DIR / 'elm_met_var_stats.csv'
    sdf = pd.read_csv(path_stats, index_col=0)

    # Determine which variables can/can't be validated
    namemap = elm_data_dicts()['e5namemap']
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
    nonneg_bands = elm_data_dicts()['nonneg']
    for c in df.columns:
         if c in nonneg_bands:
            negs = df[c]<0
            if sum(negs) > 0:
                 print({'Negative values detected in variable {}'.format(c)})

    return


def elm_var_compression_params(elm_var):
    """
    Computes the offset and scale for BYPASS compression.
    See https://github.com/fmyuan/elm-pf-tools/blob/a581c104f7a20238e144daa8418e334d5d966e55/pytools/metdata_merge.py#L1309C1-L1325C1
    """
    
    ranges = {'PRECTmms' : [-0.04, 0.04],
              'FSDS' : [-20.0, 2000.0],
              'TBOT' : [175.0, 350.0],
              'RH' : [0.0, 100.0],
              'QBOT' : [0.0, 0.10],
              'FLDS' : [0.0, 1000.0],
              'PSRF' : [20000.0, 120000.0],
              'WIND' : [-1.0, 100.0]
    }

    add_offset = (ranges[elm_var][1]+ranges[elm_var][0])/2.0
    scale_factor = (ranges[elm_var][1]-ranges[elm_var][0])*1.1/(2**15)

    return add_offset, scale_factor


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


def elm_data_dicts():
    """
    Defines some dictionaries for ELM-expected variables.
    """
    # Required bands/vars are the minimum needed to generate a full suite of ELM data
    e5_required_bands = ['temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
                          'surface_solar_radiation_downwards_hourly', 'surface_thermal_radiation_downwards_hourly',
                          'total_precipitation_hourly', 'surface_pressure', 'dewpoint_temperature_2m']
    
    cmip_required_vars = ['sfcWind', 'rsds', 'rlds', 'huss', 'pr', 'tas', 'hur', 'ps'] #  dewpoint temperature 'tdps' is derivable so not included
        
    elm_required_vars = {'datm' : ['LONGXY','LATIXY','time', 'ZBOT','TBOT', 'PRECTmms', 'RH', 'FSDS', 'FLDS', 'PSRF', 'WIND'],
                      'cbypass' : ['LONGXY','LATIXY','time', 'TBOT', 'PRECTmms', 'QBOT', 'FSDS', 'FLDS', 'PSRF', 'WIND']}

    # Name mappings to ELM
    cmip_to_elm_short_name = {  'uas' : 'UWIND',
                                'vas' : 'VWIND',
                                'sfcWind' : 'WIND',
                                'rsds' : 'FSDS',
                                'rlds' : 'FLDS',
                                'huss' : 'QBOT',
                                'pr' : 'PRECTmms',
                                'ps' : 'PSRF',
                                'tas' : 'TBOT',
                                'tdps' : 'DTBOT',
                                'hur' : 'RH'}
    
    e5_to_elm_short_name = {  'u_component_of_wind_10m' : 'UWIND',
                                'v_component_of_wind_10m' : 'VWIND',
                                'wind_speed' : 'WIND',
                                'surface_solar_radiation_downwards_hourly' : 'FSDS',
                                'surface_thermal_radiation_downwards_hourly' : 'FLDS',
                                'specific_humidity' : 'QBOT',
                                'total_precipitation_hourly' : 'PRECTmms',
                                'surface_pressure' : 'PSRF',
                                'temperature_2m' : 'TBOT',
                                'dewpoint_temperature_2m' : 'DTBOT',
                                'relative_humidity' : 'RH'}

    # Output units
    units = {'TBOT' : 'K',
            'DTBOT' : 'unsure',
            'RH' : '%',
            'WIND' : 'm/s',
            'FSDS' : 'W/m2',
            'FLDS' : 'W/m2',
            'PSRF' : 'Pa',
            'PRECTmms' : 'mm/s', # equivalent to kg/m2/s
            'QBOT' : 'kg/kg',
            'ZBOT' : 'm',
            'UWIND' : 'm/s',
            'VWIND' : 'm/s'}

    # For scaling to make "short" netcdf. 
    # Taken from https://github.com/fmyuan/elm-pf-tools/blob/db70b67a28969154748f53e2446559ada323a136/pytools/metdata_processing/elm_metdata_write.py#L347C1-L366C1
    ranges = {'PRECTmms'  : [-0.04, 0.04],
            'FSDS'      : [-20, 2000],
            'TBOT'      : [175, 350],
            'RH'        : [0, 100],
            'QBOT'      : [0, 0.1],
            'FLDS'      : [0, 1000],
            'PSRF'      : [20000, 120000],
            'WIND'      : [-1, 100]}

    # Short descriptions
    short_descriptions = {
        'TBOT' : 'temperature at the lowest atm level (TBOT)',
        'DTBOT' : 'dewpoint temperature [era5 direct]',
        'RH' : 'relative humidity at the lowest atm level (RH)',
        'WIND' : 'wind magnitude at the lowest atm level (WIND)',
        'FSDS' : 'incident solar (FSDS)',
        'FLDS' : 'incident longwave (FLDS)',
        'PSRF' : 'pressure at the lowest atm level (PSRF)',
        'PRECTmms' : 'precipitation (PRECTmms)',
        'QBOT' : 'specific humidity at the lowest atm level (QBOT)',
        'ZBOT' : 'observational height (ZBOT)',
        'UWIND' : 'u component of wind velocity',
        'VWIND' : 'v component of wind velocity'}
    
    # Variables that cannot physically be negative
    non_negative_bands = [
            'surface_solar_radiation_downwards_hourly',
            'surface_thermal_radiation_downwards_hourly',
            'surface_thermal_radiation_upwards_hourly',
            'surface_net_solar_radiation_hourly',
            'surface_net_thermal_radiation_hourly',
            'surface_latent_heat_flux_hourly',
            'surface_sensible_heat_flux_hourly',
            'total_precipitation_hourly',
            'snowfall_hourly',
            'snowmelt_hourly',
            'runoff_hourly',
            'evaporation_hourly',
            'volumetric_soil_water_layer_1',
            'volumetric_soil_water_layer_2',
            'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4',
            'snow_depth',
            'skin_temperature',
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'surface_pressure',
            'relative_humidity',
            'specific_humidity'
        ]

    
    return {'e5namemap' : e5_to_elm_short_name,
            'cmipnamemap' : cmip_to_elm_short_name,
            'units' : units,
            'ranges' : ranges,
            'descriptions' : short_descriptions,
            'cmip_req_vars' : cmip_required_vars, 
            'elm_req_vars' : elm_required_vars,
            'nonneg' : non_negative_bands,
            'elm_required_bands' : e5_required_bands,
            'short_names' : e5_to_elm_short_name
            }


def gen_zone_mappings(df_loc, site=False):
    """
    Creates a dataframe of zone mappings.
    
    If site=False:
        Returns a DataFrame with columns ['lon', 'lat', 'zone', 'id'].
    If site=True:
        Returns a dictionary: {gid: single-row DataFrame}.
    """

    # Base mapping
    zone_mapping = df_loc[['lon', 'lat', 'zone']].copy()
    zone_mapping['lon'] = zone_mapping['lon'] % 360  # ELM uses 0–360 longitudes
    zone_mapping['id'] = np.arange(1, len(zone_mapping) + 1)
    zone_mapping['zone'] = zone_mapping['zone'].astype(int).astype(str).str.zfill(2)

    if site is True:
        # Override ID and zone to just "01"
        zone_mapping['id'] = 1
        zone_mapping['zone'] = '01'
        
        # Export a dictionary of single-row DataFrames
        zone_mapping_site = {
            gid: zone_mapping.iloc[[i]] for i, gid in enumerate(df_loc['gid'].values)
        }
        return zone_mapping_site

    return zone_mapping
