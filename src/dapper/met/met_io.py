import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime

from dapper.utils import utils
from dapper.met import elm_utils as eu

def initialize_met_netcdf(df_loc, elm_var, dtime_vals, dtime_units, write_path, 
                          add_offset=None, scale_factor=None, calendar='noleap', compress_level=0, dformat='BYPASS'):
    """
    Creates a preallocated NetCDF file using netCDF4 with provided DTIME values and units.
    """
    fillvalue = -32767

    if os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' already exists.")
        return

    if dformat == 'BYPASS':
        mdd = eu.elm_data_dicts()
        if add_offset is None or scale_factor is None:
            add_offset, scale_factor = eu.elm_var_packing_params(elm_var)

        df_loc = df_loc.sort_values(['lat', 'lon']).reset_index(drop=True)

        try:
            with nc.Dataset(write_path, mode='w', format='NETCDF4') as ds:
                compress = compress_level > 0

                ds.createDimension('n', len(df_loc))
                ds.createDimension('DTIME', len(dtime_vals))

                lat = ds.createVariable('LATIXY', 'f4', ('n',))
                lon = ds.createVariable('LONGXY', 'f4', ('n',))
                lat[:] = df_loc['lat'].values
                lon[:] = df_loc['lon_0-360'].values
                lat.units = 'degrees_north'
                lon.units = 'degrees_east'

                if len(df_loc) > 1:
                    gid = ds.createVariable('gid', str, ('n',))
                    gid[:] = df_loc['gid'].values

                dtime = ds.createVariable('DTIME', 'f8', ('DTIME',), zlib=compress, complevel=compress_level, fill_value=fillvalue)
                dtime[:] = dtime_vals
                dtime.units = dtime_units
                dtime.calendar = calendar
                dtime.long_name = 'observation_time'

                var = ds.createVariable(elm_var, 'i2', ('DTIME', 'n'), zlib=compress, complevel=compress_level, fill_value=fillvalue)
                var.add_offset = add_offset
                var.scale_factor = scale_factor
                var.units = mdd['units'][elm_var]
                var.description = mdd['descriptions'][elm_var]
                var.long_name = next((k for k, v in mdd['e5namemap'].items() if v == elm_var), None)
                var.mode = 'time-dependent'

                ds.history = "Created using netCDF4 with dapper"
                ds.calendar = calendar
                ds.created_on = datetime.today().strftime('%Y-%m-%d')
                ds.dapper_commit_hash = utils.get_git_commit_hash()
                ds.sampled_geometry = "\n".join(df_loc['sampled_geometry'].astype(str).tolist())
                ds.method = df_loc['method'].values[0]

        except Exception as e:
            print(f"Error creating NetCDF: {e}")


def append_met_netcdf(this_df, elm_var, write_path, dtime_vals, start_idx, dformat='BYPASS'):
    """
    Appends *unpacked* physical data to preallocated NetCDF at the given start index.
    Lets netCDF4 handle packing using the variable's scale_factor and add_offset.
    """
    if not os.path.exists(write_path):
        print(f"NetCDF file '{write_path}' does not exist and cannot be appended.")
        return

    if dformat == 'BYPASS':
        with nc.Dataset(write_path, mode='a') as ds:
            if elm_var not in ds.variables:
                raise KeyError(f"{elm_var} is missing in {write_path}. Cannot append data.")

            # Make sure netCDF4 auto-scaling is ON (default)
            var = ds.variables[elm_var]
            var.set_auto_scale(True)  # This line is defensive; it's True by default

            # Validate DTIME match
            this_df['time'] = pd.to_datetime(this_df['time'])
            this_df = this_df.sort_values(['time', 'LATIXY', 'LONGXY']).reset_index(drop=True)
            unique_times = this_df['time'].drop_duplicates().sort_values().to_numpy()

            num_times = len(unique_times)
            num_sites = ds.dimensions['n'].size
            end_idx = start_idx + num_times

            expected = dtime_vals[start_idx:end_idx]
            actual = ds.variables['DTIME'][start_idx:end_idx]
            if not np.allclose(expected, actual, atol=1e-6):
                raise ValueError("DTIME mismatch between expected and existing NetCDF values.")

            # Write physical floats — netCDF4 handles packing.
            reshaped = this_df[elm_var].values.reshape(num_times, num_sites)
            var[start_idx:end_idx, :] = reshaped

            ds.sync()


def create_dtime(df, calendar='standard', dtime_units='days', dtime_resolution_hrs=1):
    """
    Computes DTIME values and the corresponding DTIME attribute string from a dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'time' column (datetime64)
        calendar (str): Calendar type ('standard' or 'noleap')
        dtime_units (str): 'days' or 'hours'
        dtime_resolution_hrs (int): Desired time resolution in hours

    Returns:
        dtime_vals (np.ndarray): Array of DTIME values (float)
        dtime_attr (str): DTIME attribute string (e.g., "days since 2001-01-01 00:00:00")
        unique_times (np.ndarray): Array of unique datetime64 timestamps
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if calendar.lower() == "noleap":
        df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Round time to match resolution, if needed
    if dtime_resolution_hrs > 1:
        df = df.set_index("time")
        df = df.resample(f"{dtime_resolution_hrs}h").mean(numeric_only=True).dropna().reset_index()

    unique_times = df["time"].drop_duplicates().sort_values().to_numpy()
    ref_date = unique_times[0]

    if dtime_units == "days":
        dtime_vals = (unique_times - ref_date) / np.timedelta64(1, "D")
    elif dtime_units == "hours":
        dtime_vals = (unique_times - ref_date) / np.timedelta64(1, "h")
    else:
        raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")

    dtime_attr = f"{dtime_units} since {pd.Timestamp(ref_date).strftime('%Y-%m-%d %H:%M:%S')}"
    return dtime_vals.astype("float64"), dtime_attr, unique_times


def get_start_end_years(csv_filepaths, calendar='standard'):
    """
    Reads multiple CSVs, extracts and filters dates to full years only (Jan 1 to Dec 31),
    and returns the earliest and latest full year.

    Parameters:
        csv_filepaths (list): List of paths to CSVs containing a 'date' column.
        calendar (str): Calendar type ('standard' or 'noleap').

    Returns:
        (int, int): Start and end years
    """
    # Read and merge dates
    dates = [pd.read_csv(file, usecols=["date"]) for file in csv_filepaths]
    dates = pd.concat(dates, ignore_index=True)
    dates["date"] = pd.to_datetime(dates["date"])
    dates.sort_values(by="date", inplace=True)

    # Remove leap days if using noleap calendar
    if calendar.lower() == "noleap":
        dates = dates[~((dates["date"].dt.month == 2) & (dates["date"].dt.day == 29))]

    # Identify full years
    dates["year"] = dates["date"].dt.year
    dates["month_day"] = dates["date"].dt.month * 100 + dates["date"].dt.day
    valid_years = dates.groupby("year")["month_day"].agg(lambda x: {101, 1231}.issubset(set(x)))
    valid_years = valid_years[valid_years].index

    if not valid_years.empty:
        return valid_years[0], valid_years[-1]
    else:
        return dates["date"].dt.year.min(), dates["date"].dt.year.max()



# def e5lh_to_elm_gridded(
#     csv_directory,
#     write_directory,
#     df_loc,
#     remove_leap=True,
#     id_col=None,
#     nzones=1,
#     dformat="BYPASS",
#     compress=True,
#     compress_level=4,
# ):
#     """
#     Batched version for grids.

#     compress_level - higher will compress more but take longer to write

#     """
#     if dformat not in ["DATM_MODE", "BYPASS"]:
#         raise KeyError(
#             "You provided an unsupported dformat value. Currently only DATM_MODE and BYPASS are available."
#         )
#     elif dformat == "DATM_MODE":
#         print("DATM_MODE is not yet available. Exiting.")
#         return

#     if type(csv_directory) is str:
#         csv_directory = Path(csv_directory)
#     if type(write_directory) is str:
#         write_directory = Path(write_directory)

#     mdd = eu.elm_data_dicts()

#     # ELM/E3SM operate on a longitudinal range of 0-360

#     # Determine our date range to make sure we provide only complete years of data
#     files = [f for f in os.listdir(csv_directory) if os.path.splitext(f)[1] == ".csv"]
#     dates = [pd.read_csv(csv_directory / file, usecols=["date"]) for file in files]
#     dates = pd.concat(dates, ignore_index=True)
#     dates["date"] = pd.to_datetime(dates["date"])
#     dates.sort_values(by="date", inplace=True)

#     # Clip to first available Jan 01 year and last available Dec. 31 year.
#     dates["year"] = dates["date"].dt.year
#     dates["month_day"] = dates["date"].dt.month * 100 + dates["date"].dt.day # Converts to integer format (e.g., 101 for Jan 1)
#     # Group by year and check if both January 1 and December 31 exist
#     valid_years = dates.groupby("year")["month_day"].agg(
#         lambda x: {101, 1231}.issubset(set(x))
#     )
#     # Get the first and last valid years
#     valid_years = valid_years[valid_years].index
#     if not valid_years.empty:
#         start_year, end_year = valid_years[0], valid_years[-1]
#     else:
#         start_year, end_year = dates["year"].values[0], dates["year"].values[0]
#         print("There is not a full year's worth of data. Using the full dataset.")

#     # Create temporary folder for storing intermediate results
#     utils.make_directory(write_directory, delete_all_contents=True)

#     # Rename id field for consistency to 'gid'
#     if id_col is None:
#         id_col = utils.infer_id_field(df_loc)
#     df_loc.rename(columns={id_col: "gid"}, inplace=True)

#     # Prepare the netCDF grid
#     df_loc = df_loc.sort_values(by=["lat", "lon"]).reset_index(drop=True)
#     df_loc["gid"] = df_loc["gid"].astype(str)

#     # Account for zones if not already provided in df_loc
#     if "zone" not in df_loc.columns:
#         df_loc["zone"] = np.tile(np.arange(1, nzones + 1), (len(df_loc) // nzones) + 1)[
#             : len(df_loc)
#         ]
#     unique_zones = list(set(df_loc["zone"]))

#     # Save each file to netCDF
#     for i, f in enumerate(files):
#         print(f"Processing file {i+1} of {len(files)}: {f}")
#         file_path = csv_directory / f
#         this_df = pd.read_csv(file_path)

#         # Rename id columns for consistency to 'gid'
#         if i == 0:
#             if id_col is None:
#                 id_col = utils.infer_id_field(ppdf.columns)
#         this_df.rename(columns={id_col: "gid"}, inplace=True)

#         # Add lat/lon data to preprocessed dataframe and sort
#         this_df = this_df.merge(df_loc[["gid", "lat", "lon", "zone"]], on="gid", how="inner")

#         ppdf = _preprocess_e5lh_to_elm_file_grid(this_df, start_year, end_year, remove_leap, dformat)
#         ppdf = ppdf.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

#         for elm_var in ppdf.columns:
#             if elm_var in ["gid", "time", "LONGXY", "LATIXY", "zone"]:
#                 continue
#             for zone in unique_zones:

#                 filename = "ERA5_" + elm_var + "_" + str(start_year) + "-" + str(end_year) + "_z" + str(zone).zfill(2) + ".nc"
#                 write_path = write_directory / filename

#                 # Initialize netCDF file - NEED TO EDIT AFTER MODIFYING CREATE FUNCTION
#                 if i == 0:
#                     this_df_loc = df_loc[df_loc["zone"] == zone]
#                     eu.create_met_netcdf(
#                         this_df_loc,
#                         elm_var,
#                         write_path,
#                         dformat,
#                         compress,
#                         compress_level,
#                         attrs={"sampling_method": this_df_loc["method"].values[0]},
#                     )

#                 # Select required vars and zone
#                 save_df = ppdf[["time", "LONGXY", "LATIXY", "gid", "zone", elm_var]]
#                 save_df = save_df[save_df["zone"] == zone]
#                 # Write to netCDF
#                 eu.append_met_netcdf(
#                     save_df, elm_var, write_path, dformat, compress, compress_level
#                 )

#     # Generate zone_mappings file
#     zms = eu.gen_zone_mappings(df_loc)
#     zm_write_path = write_directory / "zone_mappings.txt"
#     zms.to_csv(zm_write_path, index=False, header=False, sep="\t")

#     return

