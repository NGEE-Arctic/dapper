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
    Build DTIME axis + align/upsample the dataframe to that axis.

    Returns
    -------
    dtime_vals : np.ndarray[float64]
    dtime_attr : str  (CF-style "days since YYYY-MM-DD HH:MM:SS")
    df_out     : pd.DataFrame  (has a 'time' column matching the DTIME axis)
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
