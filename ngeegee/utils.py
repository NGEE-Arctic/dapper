# Generic functions JPS
import os
import shutil
from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ngeegee.metadata as md

# Pathing for convenience
import ngeegee
_ROOT_DIR = Path(next(iter(ngeegee.__path__))).parent
_DATA_DIR = _ROOT_DIR / "data"

def determine_gee_batches(start_date, end_date, max_date, years_per_task=5, verbose=True):
    """
    Calculates how to batch tasks for splitting bigger GEE jobs.
    Currently assumes ERA5-Land hourly (i.e. hourly data with a known date range).
    
    Returns a DataFrame where each row defines the start and end time for each
    Task in a batch.
    """
    # Generate a DataFrame with start and end dates for each GEE task
    this_date = start_date
    break_dates = [this_date]
    end_date = min(max_date, end_date)
    while this_date < end_date:
        break_dates.append(break_dates[-1] + relativedelta(years=years_per_task))
        this_date = break_dates[-1]
    # Replace the last date with the maximum possible
    break_dates[-1] = end_date

    # Create DataFrame
    df = pd.DataFrame({'task_start' : break_dates[:-1], 
                       'task_end' : break_dates[1:]})

    if verbose:
        if len(df) == 1:
            print(f'Your request will be executed as {len(df)} Task in Google Earth Engine.')
        else:
            print(f'Your request will be executed as {len(df)} Tasks in Google Earth Engine.')

    return df


def make_directory(path, delete_all_contents=False):

    if os.path.isdir(path) is False:
        os.mkdir(path)
    elif delete_all_contents:
        remove_directory_contents(path)
    return


def remove_directory_contents(path, remove_directory=False):
    if any(path.glob("*")):  # Check if directory contains any files
        for item in path.glob("*"):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete folder and its contents 

    if remove_directory:
         path.rmdir()

def validate_met_vars(df):
    """
    Uses pre-computed statistics to ensure that the unit conversions resulted in
    distributions for each variable that make sense.
    """
    # Load pre-computed variable statistics
    path_stats = _DATA_DIR / 'elm_met_var_stats.csv'
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


def elm_var_compression_params(elm_var):
    """
    Computes the offset and scale for CPL_BYPASS compression.
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


def infer_id_field(columns):
    """
    Tries to discern the id field from a list of columns.
    Used when id_col is not specified.
    """
    poss_id = [c for c in columns if 'id' in c]
    if len(poss_id) == 0:
        raise NameError("Could not infer id column. Specify it with 'id_col' kwarg when calling e5hl_to_elm().")
    else:
        poss_id_lens = [len(pi) for pi in poss_id]
        id_col = poss_id[poss_id_lens.index(min(poss_id_lens))]
        print(f"Inferred '{id_col}' as id column. If this is not correct, re-run this function and specify 'id_col' kwarg.")

    return id_col


def create_netcdf(df_loc, elm_var, write_path, dformat='CPL_BYPASS', compress=False, compress_level=0):
    if os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' already exists.")
        return

    if dformat == 'CPL_BYPASS':
        mdd = md.elm_data_dicts()
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


def append_netcdf(this_df, elm_var, write_path, dformat='CPL_BYPASS', compress=False, compress_level=0):
    if not os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' does not exist and cannot be appended.")
        return

    if dformat == 'CPL_BYPASS':
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

            # Efficiently reshape data: (DTIME, n) order
            num_times = len(unique_times)
            reshaped_data = packed_data_column.reshape(num_times, current_gid_size)

            # Append reshaped data as float32 in (DTIME, n) order
            ds[elm_var][current_time_size:current_time_size + num_times, :] = reshaped_data

            # Force writing data to disk
            ds.sync()

        print(f"Successfully appended data for {elm_var} to {write_path}.")

