# Generic functions JPS
import os
import shutil
from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ngeegee

# Pathing for convenience
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


def remove_directory_contents(path):
    if any(path.glob("*")):  # Check if directory contains any files
        for item in path.glob("*"):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete folder and its contents 
