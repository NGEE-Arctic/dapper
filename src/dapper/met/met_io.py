import os
import warnings
import numpy as np
import pandas as pd
# import netCDF4 as nc
from pathlib import Path
from netCDF4 import Dataset
# from datetime import datetime

# from dapper.utils import utils
# from dapper.utils import elm_utils as eu

# def initialize_met_netcdf(df_loc, elm_var, dtime_vals, dtime_units, write_path, 
#                           add_offset=None, scale_factor=None, calendar='noleap', compress_level=0, dformat='BYPASS'):
#     """
#     Creates a preallocated NetCDF file using netCDF4 with provided DTIME values and units.
#     """
#     fillvalue = -32767

#     if os.path.exists(write_path):
#         print(f"NetCDF file '{write_path}' already exists.")
#         return

#     if dformat == 'BYPASS':
#         mdd = eu.elm_data_dicts()
#         if add_offset is None or scale_factor is None:
#             add_offset, scale_factor = eu.elm_var_packing_params(elm_var)

#         if scale_factor > 0: # testing if scale factor makes a difference. The reference "good" files had negative scale factors...
#             scale_factor *= -1

#         df_loc = df_loc.sort_values(['lat', 'lon']).reset_index(drop=True)

#         try:
#             with nc.Dataset(write_path, mode='w', format='NETCDF4') as ds:
#                 compress = compress_level > 0

#                 ds.createDimension('n', len(df_loc))
#                 ds.createDimension('DTIME', len(dtime_vals))

#                 lat = ds.createVariable('LATIXY', 'f4', ('n',))
#                 lon = ds.createVariable('LONGXY', 'f4', ('n',))
#                 lat[:] = df_loc['lat'].values
#                 lon[:] = df_loc['lon_0-360'].values
#                 lat.units = 'degrees_north'
#                 lon.units = 'degrees_east'

#                 if len(df_loc) > 1:
#                     gid = ds.createVariable('gid', str, ('n',))
#                     gid[:] = df_loc['gid'].values

#                 dtime = ds.createVariable('DTIME', 'f8', ('DTIME',), zlib=compress, complevel=compress_level, fill_value=fillvalue)
#                 dtime[:] = dtime_vals
#                 dtime.units = dtime_units
#                 dtime.calendar = calendar
#                 dtime.long_name = 'observation_time'

#                 var = ds.createVariable(elm_var, 'i2', ('n', 'DTIME'), zlib=compress, complevel=compress_level, fill_value=fillvalue) # testing dimensionality
#                 # var = ds.createVariable(elm_var, 'i2', ('DTIME', 'n'), zlib=compress, complevel=compress_level, fill_value=fillvalue)
#                 var.add_offset = add_offset
#                 var.scale_factor = scale_factor
#                 var.units = mdd['units'][elm_var]
#                 var.description = mdd['descriptions'][elm_var]
#                 var.long_name = next((k for k, v in mdd['e5namemap'].items() if v == elm_var), None)
#                 # var.mode = 'time-dependent'

#                 ds.history = "Created using netCDF4 with dapper"
#                 ds.calendar = calendar
#                 ds.created_on = datetime.today().strftime('%Y-%m-%d')
#                 ds.dapper_commit_hash = utils.get_git_commit_hash()
#                 ds.sampled_geometry = "\n".join(df_loc['sampled_geometry'].astype(str).tolist())
#                 ds.method = df_loc['method'].values[0]

#         except Exception as e:
#             print(f"Error creating NetCDF: {e}")


# def append_met_netcdf(this_df, elm_var, write_path, dtime_vals, start_idx, dformat='BYPASS'):
#     """
#     Appends *unpacked* physical data to preallocated NetCDF at the given start index.
#     Lets netCDF4 handle packing using the variable's scale_factor and add_offset.
#     """
#     if not os.path.exists(write_path):
#         print(f"NetCDF file '{write_path}' does not exist and cannot be appended.")
#         return

#     if dformat == 'BYPASS':
#         with nc.Dataset(write_path, mode='a') as ds:
#             if elm_var not in ds.variables:
#                 raise KeyError(f"{elm_var} is missing in {write_path}. Cannot append data.")

#             # Make sure netCDF4 auto-scaling is ON (default)
#             var = ds.variables[elm_var]
#             var.set_auto_scale(True)  # This line is defensive; it's True by default

#             # Validate DTIME match
#             this_df['time'] = pd.to_datetime(this_df['time'])
#             this_df = this_df.sort_values(['time', 'LATIXY', 'LONGXY']).reset_index(drop=True)
#             unique_times = this_df['time'].drop_duplicates().sort_values().to_numpy()

#             num_times = len(unique_times)
#             num_sites = ds.dimensions['n'].size
#             end_idx = start_idx + num_times

#             expected = dtime_vals[start_idx:end_idx]
#             actual = ds.variables['DTIME'][start_idx:end_idx]
#             if not np.allclose(expected, actual, atol=1e-6):
#                 raise ValueError("DTIME mismatch between expected and existing NetCDF values.")

#             # Write physical floats — netCDF4 handles packing.
#             # reshaped = this_df[elm_var].values.reshape(num_times, num_sites)
#             # var[start_idx:end_idx, :] = reshaped

#             reshaped = this_df[elm_var].values.reshape(num_times, num_sites).T # testing reshaping
#             var[:, start_idx:end_idx] = reshaped

#             ds.sync()


def create_dtime(
    df,
    calendar='standard',
    dtime_units='days',
    dtime_resolution_hrs=1,
    force_half_hour_for_hourly=True
):
    """
    Computes DTIME values and the corresponding DTIME attribute string from a dataframe.
    Optionally upsamples from hourly to 30-minute data to avoid the ELM hourly-index bug.

    Guarantees:
        - df_out['time'] is sorted ascending, no duplicates
        - dtime_vals aligns exactly with df_out['time']
        - Missing values for both state and rate variables are filled in both directions

    Parameters:
        df (pd.DataFrame): DataFrame with a 'time' column (datetime64)
        calendar (str): Calendar type ('standard' or 'noleap')
        dtime_units (str): 'days' or 'hours'
        dtime_resolution_hrs (int): Desired time resolution in hours
        force_half_hour_for_hourly (bool): If True and dtime_resolution_hrs==1, 
                                           switch to a 30-min grid and upsample.

    Returns:
        dtime_vals (np.ndarray)
        dtime_attr (str)
        df_out (pd.DataFrame)
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if calendar.lower() == "noleap":
        df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Variable categorization
    linear_vars = ['TBOT', 'DTBOT', 'RH', 'QBOT', 'PSRF', 'ZBOT',
                   'UWIND', 'VWIND', 'WIND']
    ffill_vars = ['FSDS', 'FLDS', 'PRECTmms']
    accum_vars = []  # per-interval totals go here if needed

    # Build target time axis
    if dtime_resolution_hrs == 1 and force_half_hour_for_hourly:
        print("1-hr export requested, but export will be 30-minute due to E3SM bug.")
        t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
        target_times = pd.date_range(t0, t1, freq="30min", inclusive="both").to_numpy()
    elif dtime_resolution_hrs > 1:
        df = df.set_index("time")
        df = df.resample(f"{dtime_resolution_hrs}h").mean(numeric_only=True).dropna().reset_index()
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()
    else:
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()

    ref_date = target_times[0]

    # Compute DTIME values
    if dtime_units == "days":
        dtime_vals = (target_times - ref_date) / np.timedelta64(1, "D")
    elif dtime_units == "hours":
        dtime_vals = (target_times - ref_date) / np.timedelta64(1, "h")
    else:
        raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")

    dtime_attr = f"{dtime_units} since {pd.Timestamp(ref_date).strftime('%Y-%m-%d %H:%M:%S')}"

    # Upsample/align data to target_times
    df = df.set_index("time").sort_index()
    target_index = pd.DatetimeIndex(target_times, name='time')
    df_out = pd.DataFrame(index=target_index)

    # 1) Interpolate state variables (fill both directions)
    cols = [c for c in linear_vars if c in df.columns]
    if cols:
        df_out[cols] = (
            df[cols]
            .reindex(target_index)
            .interpolate(method='time', limit_direction='both')
            .ffill().bfill()
        )

    # 2) Forward-fill rates/fluxes (fill both directions)
    cols = [c for c in ffill_vars if c in df.columns]
    if cols:
        df_out[cols] = (
            df[cols]
            .reindex(target_index)
            .ffill().bfill()
        )

    # 3) Split hourly accumulations
    for v in accum_vars:
        if v in df.columns:
            s = df[v]
            half = s / 2.0
            half_early = half.copy()
            half_early.index = half_early.index - pd.Timedelta(minutes=30)
            split_series = (
                half_early.reindex(target_index, fill_value=0.0) +
                half.reindex(target_index, fill_value=0.0)
            )
            df_out[v] = split_series

    # 4) Carry along any other columns (metadata, coords) — fill both directions
    other_cols = [c for c in df.columns if c not in (linear_vars + ffill_vars + accum_vars)]
    if other_cols:
        df_out[other_cols] = (
            df[other_cols]
            .reindex(target_index)
            .ffill().bfill()
        )

    # Final guarantee: time column exactly matches target_times
    df_out.index.name = 'time'
    df_out = df_out.reset_index().sort_values('time').drop_duplicates(subset='time', keep='first')
    assert np.array_equal(df_out['time'].to_numpy(), target_times), \
        "df_out['time'] does not match generated target_times"

    return dtime_vals.astype("float64"), dtime_attr, df_out


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


def _compute_auto_chunks(dims,
                         dim_lengths,
                         dtype,
                         dtime_units,
                         dtime_vals,
                         target_mb=1.5,
                         days_per_chunk=28.0,
                         write_pattern='by_site'):
    """
    Compute chunk sizes aligned to `dims`.

    Heuristics:
      - ('n','DTIME') with write_pattern='by_site'  -> (1, t_chunk)
      - ('DTIME','lat','lon') with 'by_cell'       -> (t_chunk, 1, 1)
      - Otherwise, aim for ~target_mb chunk size by shrinking t_chunk.

    Parameters
    ----------
    dims : tuple[str,...]
    dim_lengths : dict[str,int]
    dtype : str or np.dtype
    dtime_units : str ('days ...' or 'hours ...')
    dtime_vals : array-like numeric DTIME
    target_mb : float
    days_per_chunk : float
    write_pattern : 'by_site' | 'by_cell' | 'by_time'
    """
    # dtype size in bytes
    if isinstance(dtype, str):
        try:
            dtype_size = np.dtype(dtype).itemsize
        except TypeError:
            # common shorthands
            dtype_size = 2 if dtype in ("i2", "int16", "short") else 4
    else:
        dtype_size = np.dtype(dtype).itemsize

    # infer steps per day from DTIME
    dtime_vals = np.asarray(dtime_vals, dtype=float)
    nt = int(len(dtime_vals))
    if nt > 1:
        dt_raw = float(np.median(np.diff(dtime_vals)))
    else:
        dt_raw = 1.0

    units_lower = str(dtime_units).lower()
    if "day" in units_lower:
        dt_hours = dt_raw * 24.0
    elif "hour" in units_lower:
        dt_hours = dt_raw
    else:
        dt_hours = dt_raw  # assume already hours

    steps_per_day = 24.0 / dt_hours if (np.isfinite(dt_hours) and dt_hours > 0) else 24.0
    t_chunk = int(max(1, min(nt, round(days_per_chunk * steps_per_day))))

    # default chunks: 1 for non-time dims if writing by that index, else full
    chunks = []
    target_bytes = int(target_mb * 1024 * 1024)

    if dims == ('n', 'DTIME'):
        n = int(dim_lengths['n'])
        if write_pattern == 'by_site':
            n_chunk = 1
        else:
            n_chunk = min(n, 512)
        cur_bytes = n_chunk * t_chunk * dtype_size
        if cur_bytes > target_bytes and t_chunk > 1:
            shrink = int(np.ceil(cur_bytes / target_bytes))
            t_chunk = max(1, t_chunk // max(1, shrink))
        chunks = [int(n_chunk), int(t_chunk)]

    elif dims == ('DTIME', 'lat', 'lon'):
        # write by cell -> keep lat,lon = 1 so we don't rewrite neighboring cells' chunks
        lat_len = int(dim_lengths['lat'])
        lon_len = int(dim_lengths['lon'])
        if write_pattern == 'by_cell':
            lat_chunk = 1
            lon_chunk = 1
        else:
            # fallback: small tiles
            lat_chunk = min(lat_len, 8)
            lon_chunk = min(lon_len, 8)
        cur_bytes = t_chunk * lat_chunk * lon_chunk * dtype_size
        if cur_bytes > target_bytes and t_chunk > 1:
            shrink = int(np.ceil(cur_bytes / target_bytes))
            t_chunk = max(1, t_chunk // max(1, shrink))
        chunks = [int(t_chunk), int(lat_chunk), int(lon_chunk)]

    else:
        # generic: set 1 for any non-time dim if write_pattern hints so
        for d in dims:
            if d.upper() == 'DTIME' or d == 'time':
                chunks.append(int(t_chunk))
            else:
                chunks.append(1)
        # shrink if necessary
        cur = int(np.prod(chunks)) * dtype_size
        if cur > target_bytes and t_chunk > 1:
            shrink = int(np.ceil(cur / target_bytes))
            chunks = [c if (i != dims.index('DTIME')) else max(1, c // max(1, shrink))
                      for i, c in enumerate(chunks)]

    return tuple(int(c) for c in chunks)


def initialize_met_netcdf(path_nc,
                          var_name,
                          dims,
                          dim_lengths,
                          dtime_name,
                          dtime_vals,
                          dtime_units,
                          calendar,
                          coord_specs,
                          add_offset,
                          scale_factor,
                          dtype="i2",
                          fill_value=32767,
                          chunks=None,
                          write_pattern='by_site',
                          append_attrs=None,
                          nc_format="NETCDF4_CLASSIC"):
    """
    Generic initializer for all three layouts.

    Parameters
    ----------
    path_nc : Path-like
    var_name : str
    dims : tuple[str,...]               e.g., ('n','DTIME') or ('DTIME','lat','lon')
    dim_lengths : dict[str,int]
    dtime_name : str                    e.g., 'DTIME'
    dtime_vals : array-like (numeric)
    dtime_units : str                   CF units string
    calendar : str
    coord_specs : list of dicts         [{'name','dtype','dims','data','attrs':{...}}, ...]
    add_offset, scale_factor : float    packing params
    dtype : str or np.dtype             on-disk dtype (e.g., 'i2')
    fill_value : int/float              must match dtype
    chunks : tuple[int,...] or None     auto-chunk if None
    write_pattern : 'by_site' | 'by_cell' | 'by_time'
    append_attrs : dict                 extra global attributes
    nc_format : str                     e.g., "NETCDF4_CLASSIC"
    """
    path_nc = Path(path_nc)
    path_nc.parent.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = _compute_auto_chunks(dims=dims,
                                      dim_lengths=dim_lengths,
                                      dtype=dtype,
                                      dtime_units=dtime_units,
                                      dtime_vals=dtime_vals,
                                      write_pattern=write_pattern)

    ds = Dataset(path_nc, "w", format=nc_format)

    # Dimensions
    for d in dims:
        if d not in ds.dimensions:
            ds.createDimension(d, int(dim_lengths[d]))

    # Time coordinate (numeric DTIME)
    vtime = ds.createVariable(dtime_name, "f8", (dtime_name,))
    vtime[:] = np.asarray(dtime_vals, dtype="float64")
    vtime.setncattr("units", str(dtime_units))
    vtime.setncattr("calendar", str(calendar))
    vtime.setncattr("long_name", "time")

    # Coordinate variables
    for spec in coord_specs or []:
        v = ds.createVariable(spec["name"], spec["dtype"], spec["dims"])
        v[:] = np.asarray(spec["data"])
        for ak, av in (spec.get("attrs") or {}).items():
            v.setncattr(str(ak), av)

    # Data variable
    create_kwargs = dict(zlib=True, shuffle=True, complevel=1, fill_value=fill_value, chunksizes=chunks)
    vdata = ds.createVariable(var_name, dtype, dims, **create_kwargs)
    vdata.setncattr("add_offset", float(add_offset))
    vdata.setncattr("scale_factor", float(scale_factor))
    # keep both for downstream compatibility
    vdata.setncattr("missing_value", np.array(fill_value, dtype=np.int16 if str(dtype) in ("i2","int16","short") else type(fill_value)))

    # Globals
    ds.setncattr("Conventions", "CF-1.8")
    ds.setncattr("calendar", str(calendar))
    if append_attrs:
        for k, val in append_attrs.items():
            ds.setncattr(str(k), val)

    ds.close()


def append_met_netcdf(path_nc,
                      var_name,
                      data,
                      indexers):
    """
    Shape-agnostic appender. Assigns `data` (float array) into the variable
    using `indexers` dict, e.g.:

      sites_file/site_dirs:  indexers = {"n": isite, "DTIME": slice(0, nt)}
      gridded:               indexers = {"DTIME": slice(0, nt), "lat": iy, "lon": ix}

    Notes:
      - Variable must have on-disk integer dtype with 'scale_factor'/'add_offset'
        so netCDF4 will pack floats on assignment.
      - Any NaNs in `data` will be written as the variable's _FillValue.
    """
    arr = np.asarray(data, dtype="float64")

    with Dataset(path_nc, "r+") as ds:
        v = ds.variables[var_name]

        # Build index tuple in variable's dim order
        key = []
        for d in v.dimensions:
            if d not in indexers:
                raise KeyError(f"append_met_netcdf: indexer for dim '{d}' not provided.")
            k = indexers[d]
            if isinstance(k, tuple):
                # e.g., (start, stop) -> slice
                key.append(slice(k[0], k[1]))
            else:
                key.append(k)
        key = tuple(key)

        # Optional: bounds check for time window when slice is used
        # (netCDF4 will also raise if out of range)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            v[key] = arr















# ####################################################################
# ####################################################################
# ####################################################################
# ####################################################################


# def _compute_auto_chunks_sites(dtime_vals, dtime_units, n, dtype,
#                                target_mb=1.5, max_n_chunk=512, days_per_chunk=28.0,
#                                write_by_site=True):
#     """
#     Compute (n_chunk, t_chunk) for NetCDF chunking.

#     - If write_by_site=True (your current pattern: one site/row at a time),
#       force n_chunk=1 to avoid repeatedly rewriting the same compressed chunk.
#     - Otherwise, n_chunk=min(n, max_n_chunk).
#     - t_chunk ~ days_per_chunk worth of steps (derived from dtime_vals + units),
#       then shrunk to keep chunk size near target_mb.
#     """
#     nt = int(len(dtime_vals))

#     # infer nominal dt from DTIME values and units
#     if nt > 1:
#         dt_raw = float(np.median(np.diff(np.asarray(dtime_vals, dtype=float))))
#     else:
#         dt_raw = 1.0

#     units_lower = str(dtime_units).lower()
#     if "day" in units_lower:
#         dt_hours = dt_raw * 24.0
#     elif "hour" in units_lower:
#         dt_hours = dt_raw
#     else:
#         # unknown units → assume already in hours
#         dt_hours = dt_raw

#     if (not np.isfinite(dt_hours)) or dt_hours <= 0:
#         steps_per_day = 24.0
#     else:
#         steps_per_day = max(1.0, 24.0 / dt_hours)

#     # target ~28 days per chunk along time
#     t_chunk = int(max(1, min(nt, round(days_per_chunk * steps_per_day))))

#     # site chunking pattern
#     if write_by_site:
#         n_chunk = 1
#     else:
#         n_chunk = int(max(1, min(int(n), int(max_n_chunk))))

#     # dtype size (handle common strings)
#     if isinstance(dtype, str):
#         if dtype in ("i2", "int16", "short"):
#             dtype_size = 2
#         elif dtype in ("i4", "int32", "int"):
#             dtype_size = 4
#         elif dtype in ("f4", "float32"):
#             dtype_size = 4
#         elif dtype in ("f8", "float64"):
#             dtype_size = 8
#         else:
#             dtype_size = np.dtype(dtype).itemsize
#     else:
#         dtype_size = np.dtype(dtype).itemsize

#     # keep chunk size near target_mb by shrinking t_chunk if needed
#     target_bytes = int(target_mb * 1024 * 1024)
#     cur_bytes = n_chunk * t_chunk * dtype_size
#     if cur_bytes > target_bytes and t_chunk > 1:
#         shrink = int(np.ceil(cur_bytes / target_bytes))
#         t_chunk = max(1, t_chunk // max(1, shrink))

#     return (int(n_chunk), int(t_chunk))

# def initialize_met_netcdf_sites(
#     path_nc,
#     var_name,
#     dtime_vals,        # numeric DTIME from create_dtime(...)
#     dtime_units,       # e.g., "days since 1950-01-01 00:00:00"
#     lats,              # 1D (n,)
#     lons0360,          # 1D (n,), 0–360
#     add_offset,
#     scale_factor,
#     calendar="noleap",
#     dtype="i2",        # packed int16 on disk
#     fill_value=32767,  # int16 fill
#     chunks=None,       # if None, auto-chunk with write_by_site=True
#     append_attrs=None,
#     write_by_site=True # NEW: set True when assigning row-wise (isite) writes
# ):
#     """
#     Initialize a multi-site NetCDF with dims ('n','DTIME') and:
#       - DTIME(DTIME) numeric CF time using provided units/calendar
#       - LATIXY(n), LONGXY(n) as site coordinate variables
#       - var_name(n,DTIME) stored as packed int16 with scale_factor/add_offset

#     Chunking:
#       - If chunks is None, uses _compute_auto_chunks_sites(..., write_by_site=True)
#         so chunks=(1, t_chunk) by default, matching row-wise write pattern.
#     """
#     path_nc = Path(path_nc)
#     path_nc.parent.mkdir(parents=True, exist_ok=True)

#     n  = int(len(lats))
#     nt = int(len(dtime_vals))

#     # Auto-chunk if not provided
#     if chunks is None:
#         chunks = _compute_auto_chunks_sites(dtime_vals, dtime_units, n, dtype,
#                                             write_by_site=write_by_site)

#     ds = Dataset(path_nc, "w", format="NETCDF4_CLASSIC")

#     # Dimensions
#     ds.createDimension("n", n)
#     ds.createDimension("DTIME", nt)

#     # Time coordinate
#     vtime = ds.createVariable("DTIME", "f8", ("DTIME",))
#     vtime[:] = np.asarray(dtime_vals, dtype="float64")
#     vtime.setncattr("units", str(dtime_units))
#     vtime.setncattr("calendar", str(calendar))

#     # Site coordinates
#     vlat = ds.createVariable("LATIXY", "f4", ("n",))
#     vlon = ds.createVariable("LONGXY", "f4", ("n",))
#     vlat[:] = np.asarray(lats, dtype="float32")
#     vlon[:] = np.asarray(lons0360, dtype="float32")
#     vlat.setncattr("long_name", "latitude")
#     vlat.setncattr("units", "degrees_north")
#     vlon.setncattr("long_name", "longitude")
#     vlon.setncattr("units", "degrees_east")
#     vlon.setncattr("note", "0–360 convention")

#     # Data variable (packed int16; netCDF4 packs on assignment from floats)
#     create_kwargs = dict(zlib=True, shuffle=True, complevel=1, fill_value=fill_value)
#     if chunks is not None:
#         create_kwargs["chunksizes"] = tuple(chunks)

#     v = ds.createVariable(var_name, dtype, ("n", "DTIME"), **create_kwargs)
#     v.setncattr("add_offset", float(add_offset))
#     v.setncattr("scale_factor", float(scale_factor))
#     v.setncattr("missing_value", np.int16(fill_value) if dtype in ("i2", np.int16) else fill_value)

#     # Global attributes
#     ds.setncattr("Conventions", "CF-1.8")
#     ds.setncattr("calendar", str(calendar))
#     if append_attrs:
#         for k, val in append_attrs.items():
#             ds.setncattr(str(k), val)

#     ds.close()


# def append_met_netcdf_sites(
#     path_nc,
#     var_name,
#     vals_time_1d,   # float array aligned to global DTIME (physical units)
#     isite,          # row index in 'n' dimension
#     add_offset,     # kept for consistency; packing handled by NetCDF variable attrs
#     scale_factor,   # kept for consistency; packing handled by NetCDF variable attrs
#     fill_value=32767,
#     t0=0,
# ):
#     """Append one site's time series (floats) to a packed int16 NetCDF var with attrs.
#     The NetCDF variable should have dtype='i2' and attrs 'scale_factor'/'add_offset'
#     set at creation; netCDF4 will pack on assignment.
#     """

#     arr = np.asarray(vals_time_1d, dtype="float64")
#     n_time = arr.shape[0]

#     # basic sanity on write window
#     with Dataset(path_nc, "r+") as ds:
#         v = ds.variables[var_name]
#         if v.dimensions != ("n", "DTIME"):
#             raise ValueError(f"{var_name} has dims {v.dimensions}, expected ('n','DTIME').")
#         nt_var = v.shape[1]
#         if t0 < 0 or (t0 + n_time) > nt_var:
#             raise IndexError(f"Write window [{t0}:{t0+n_time}] exceeds DTIME length {nt_var}.")

#         # assign floats; netCDF4 packs using the variable's attrs
#         v[isite, t0:t0 + n_time] = arr
