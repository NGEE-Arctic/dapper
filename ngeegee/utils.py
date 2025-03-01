# Generic functions JPS
import os
import shutil
from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np
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
