# dapper/met/met_io.py
"""
Temporal helpers used by Exporter and adapters.
NetCDF I/O is handled in dapper.met.writers. This module is intentionally small.
"""

import numpy as np
import pandas as pd


def create_dtime(
    df,
    calendar: str = "standard",
    dtime_units: str = "days",
    dtime_resolution_hrs: int = 1,
    force_half_hour_for_hourly: bool = True,
):
    """
    Construct a numeric DTIME axis (CF-style) and align/upsample the data onto it.

    Given a DataFrame that contains a ``time`` column, this function:
      1) sorts and optionally removes Feb 29 (for a no-leap calendar),
      2) builds a target time grid (optionally upsampling 1-hourly data to 30-minute),
      3) reindexes the input to that grid with appropriate fill/interpolation rules per
         variable type, and
      4) returns the numeric DTIME values, a CF-like units string, and the aligned DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. **Must** include a ``time`` column (tz-naive or tz-aware datetimes are
        accepted, but see *Notes* for caveats). Other columns are treated as variables.
    calendar : {"standard", "noleap"}, default "standard"
        Calendar semantics for the output axis. If ``"noleap"``, rows on Feb-29 are removed
        prior to alignment.
    dtime_units : {"days", "hours"}, default "days"
        Units used for the numeric DTIME coordinate and its attribute string. The numeric
        coordinate is measured from the first target timestamp (see *Returns*).
    dtime_resolution_hrs : int, default 1
        Desired time-step of the **target** grid, in hours.
        - If ``== 1`` and ``force_half_hour_for_hourly`` is True, the function will **upsample**
          to a 30-minute grid to work around an E3SM/ELM hourly-index quirk.
        - If ``> 1``, the input is resampled with ``mean(numeric_only=True)`` to this cadence.
        - Otherwise (e.g., original already 30-minute), the function uses the input times.
    force_half_hour_for_hourly : bool, default True
        When ``True`` and ``dtime_resolution_hrs == 1``, generate a 30-minute target axis
        and upsample/interpolate to it (splitting true accumulations is supported but
        **disabled by default**; see ``accum_vars`` in code).

    Returns
    -------
    dtime_vals : numpy.ndarray of float64, shape (nt,)
        Numeric coordinate values on the target grid. If ``dtime_units == "days"``, values
        are in days relative to the first target timestamp (inclusive); if ``"hours"``,
        values are in hours.
    dtime_attr : str
        CF-style units attribute for DTIME, e.g. ``"days since 1950-01-01 00:00:00"``.
        The reference time equals the first element of the target grid.
    df_out : pandas.DataFrame
        The input variables aligned to the target grid. Contains a ``time`` column that
        **exactly** matches the target timestamps, followed by aligned variables. Alignment
        behavior:
          - *State-like* variables (``TBOT``, ``DTBOT``, ``RH``, ``QBOT``, ``PSRF``, ``ZBOT``,
            ``UWIND``, ``VWIND``, ``WIND``) are time-interpolated and filled both directions.
          - *Flux/rate-like* variables (``FSDS``, ``FLDS``, ``PRECTmms``) are forward-filled
            (then back-filled at the start).
          - *True accumulations* (none enabled by default) can be split over sub-steps
            (see code block under ``accum_vars``).
          - Any *other* columns present in ``df`` are carried through with forward/back fill
            **only when they survive the optional resample step** (see *Notes*).

    Notes
    -----
    - **Empty input**: the function assumes ``df`` has at least one row. If there is a chance
      of an empty DataFrame, validate before calling.
    - **Resampling (``dtime_resolution_hrs > 1``)**: the code uses
      ``df.resample(...).mean(numeric_only=True).dropna()``. This means **non-numeric
      columns (e.g., metadata like ``gid``, ``LONGXY``, ``LATIXY``) will be dropped** during
      resampling and therefore will *not* be present in ``df_out``. If you need those, join
      them back post hoc from a separate table keyed by site, or resample first and then
      merge metadata.
    - **Hourly→30-min upsample**: when ``dtime_resolution_hrs == 1`` and
      ``force_half_hour_for_hourly`` is True, an evenly spaced 30-minute axis is produced from
      min..max of the input times. State variables are time-interpolated; flux variables
      are forward-filled. If you have *true accumulations* (e.g., per-hour totals), enable
      and populate ``accum_vars`` and verify the split logic matches your convention.
    - **Timezones**: Pandas timezone-aware datetimes are tolerated, but the arithmetic for
      ``dtime_vals`` assumes consistent tz handling. For portability, prefer tz-naive UTC
      timestamps on input.
    - **Strict alignment check**: the function asserts that ``df_out['time']`` equals the
      generated target timestamps. This will raise if duplicates remain or if reindexing
      produced an unexpected index—helpful for catching data irregularities early.

    Examples
    --------
    >>> dvals, dattr, dout = create_dtime(df, calendar="noleap",
    ...                                   dtime_units="days",
    ...                                   dtime_resolution_hrs=1,
    ...                                   force_half_hour_for_hourly=True)

    Potential Gotchas
    -----------------
    - If your input cadence is *already* 30-minute and you set ``dtime_resolution_hrs == 1``
      with ``force_half_hour_for_hourly=True``, the function will still construct a 30-minute
      grid (no change), but ensure your categories (linear/ffill/accum) make sense for those variables.
    - For irregular or sparse time series, ``interpolate(method='time')`` can extrapolate at
      the ends due to the subsequent ``ffill().bfill()``. If that is undesirable, remove the
      end fills and handle edge segments explicitly before exporting.
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if calendar.lower() == "noleap":
        df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Variable categories (ELM-ish)
    linear_vars = ['TBOT', 'DTBOT', 'RH', 'QBOT', 'PSRF', 'ZBOT', 'UWIND', 'VWIND', 'WIND']
    ffill_vars  = ['FSDS', 'FLDS', 'PRECTmms']
    accum_vars  = []  # put true accumulations here if needed

    # Target time axis
    if dtime_resolution_hrs == 1 and force_half_hour_for_hourly:
        t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
        target_times = pd.date_range(t0, t1, freq="30min", inclusive="both").to_numpy()
    elif dtime_resolution_hrs > 1:
        df = df.set_index("time")
        df = df.resample(f"{dtime_resolution_hrs}h").mean(numeric_only=True).dropna().reset_index()
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()
    else:
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()

    ref_date = target_times[0]

    # Numeric DTIME
    if dtime_units == "days":
        dtime_vals = (target_times - ref_date) / np.timedelta64(1, "D")
    elif dtime_units == "hours":
        dtime_vals = (target_times - ref_date) / np.timedelta64(1, "h")
    else:
        raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")

    dtime_attr = f"{dtime_units} since {pd.Timestamp(ref_date).strftime('%Y-%m-%d %H:%M:%S')}"

    # Reindex to target axis
    df = df.set_index("time").sort_index()
    target_index = pd.DatetimeIndex(target_times, name="time")
    df_out = pd.DataFrame(index=target_index)

    # (1) Interpolate state vars
    cols = [c for c in linear_vars if c in df.columns]
    if cols:
        df_out[cols] = (
            df[cols].reindex(target_index)
                    .interpolate(method="time", limit_direction="both")
                    .ffill().bfill()
        )

    # (2) Forward-fill rates/fluxes
    cols = [c for c in ffill_vars if c in df.columns]
    if cols:
        df_out[cols] = df[cols].reindex(target_index).ffill().bfill()

    # (3) Split true accumulations (none configured by default)
    for v in accum_vars:
        if v in df.columns:
            s = df[v]
            half = s / 2.0
            half_early = half.copy()
            half_early.index = half_early.index - pd.Timedelta(minutes=30)
            split_series = (
                half_early.reindex(target_index, fill_value=0.0)
                + half.reindex(target_index, fill_value=0.0)
            )
            df_out[v] = split_series

    # (4) Carry through other columns (meta), fill both ways
    other_cols = [c for c in df.columns if c not in (linear_vars + ffill_vars + accum_vars)]
    if other_cols:
        df_out[other_cols] = df[other_cols].reindex(target_index).ffill().bfill()

    df_out.index.name = "time"
    df_out = (
        df_out.reset_index()
              .sort_values("time")
              .drop_duplicates(subset="time", keep="first")
    )
    # Ensure perfect match
    assert np.array_equal(df_out["time"].to_numpy(), target_times), \
        "df_out['time'] does not match generated target_times"

    return dtime_vals.astype("float64"), dtime_attr, df_out


def get_start_end_years(csv_filepaths, calendar: str = "standard"):
    """
    Inspect CSVs (must contain a 'date' column) and return earliest/latest
    full years present. If no full years, return min/max year in data.
    """
    dates = [pd.read_csv(file, usecols=["date"]) for file in csv_filepaths]
    dates = pd.concat(dates, ignore_index=True)
    dates["date"] = pd.to_datetime(dates["date"])
    dates.sort_values(by="date", inplace=True)

    if calendar.lower() == "noleap":
        dates = dates[~((dates["date"].dt.month == 2) & (dates["date"].dt.day == 29))]

    dates["year"] = dates["date"].dt.year
    dates["month_day"] = dates["date"].dt.month * 100 + dates["date"].dt.day

    full = dates.groupby("year")["month_day"].agg(lambda x: {101, 1231}.issubset(set(x)))
    full_years = full[full].index

    if len(full_years) > 0:
        return int(full_years[0]), int(full_years[-1])
    return int(dates["date"].dt.year.min()), int(dates["date"].dt.year.max())
