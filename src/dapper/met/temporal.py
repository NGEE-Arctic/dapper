# dapper/met/temporal.py
"""
Temporal helpers used by Exporter and adapters.
NetCDF I/O is handled in dapper.met.writers. This module is intentionally small.
"""

import numpy as np
import pandas as pd


_CUMDAYS_NONLEAP = np.asarray(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64
)


def _drop_feb29(times):
    """Drop Feb 29 timestamps from a 1D array-like of datetimes."""
    tidx = pd.DatetimeIndex(times)
    mask = ~((tidx.month == 2) & (tidx.day == 29))
    return tidx[mask].to_numpy()


def _noleap_offset(dtime_units: str, target_times, ref_date):
    """Compute numeric offsets in a 365-day (noleap) calendar."""

    tidx = pd.DatetimeIndex(target_times)
    ref = pd.Timestamp(ref_date)

    # Days since 0001-01-01 in a 365-day calendar (non-leap month lengths)
    y = tidx.year.astype(np.int64)
    m = tidx.month.astype(np.int64)
    d = tidx.day.astype(np.int64)
    day0 = 365 * (y - 1) + _CUMDAYS_NONLEAP[m - 1] + (d - 1)

    ref_day0 = 365 * (ref.year - 1) + _CUMDAYS_NONLEAP[ref.month - 1] + (ref.day - 1)

    # Fractional day from time-of-day
    frac = (
        tidx.hour.astype(np.float64) * 3600.0
        + tidx.minute.astype(np.float64) * 60.0
        + tidx.second.astype(np.float64)
        + tidx.microsecond.astype(np.float64) / 1.0e6
        + tidx.nanosecond.astype(np.float64) / 1.0e9
    ) / 86400.0
    ref_frac = (
        ref.hour * 3600.0
        + ref.minute * 60.0
        + ref.second
        + ref.microsecond / 1.0e6
        + ref.nanosecond / 1.0e9
    ) / 86400.0

    days = (day0.astype(np.float64) + frac) - (float(ref_day0) + float(ref_frac))

    if dtime_units == "days":
        return days
    if dtime_units == "hours":
        return days * 24.0
    raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")


def create_dtime(
    df,
    calendar: str = "standard",
    dtime_units: str = "days",
    dtime_resolution_hrs: float = 1.0,
):
    """
    Construct a numeric DTIME axis and align data onto it at an arbitrary cadence.
    Accepts fractional hours, e.g., 0.5 (30 min), 0.3 (18 min), 1.5 (90 min).
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")
    if dtime_resolution_hrs <= 0:
        raise ValueError("dtime_resolution_hrs must be > 0.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if calendar.lower() == "noleap":
        df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Variable categories (ELM-ish)
    linear_vars = ['TBOT', 'DTBOT', 'RH', 'QBOT', 'PSRF', 'ZBOT', 'UWIND', 'VWIND', 'WIND']
    ffill_vars  = ['FSDS', 'FLDS', 'PRECTmms']
    accum_vars  = []  # put true accumulations here if needed

    # --- derive target step in minutes (rounded to nearest minute) ---
    step_minutes = int(round(float(dtime_resolution_hrs) * 60.0))
    if step_minutes < 1:
        step_minutes = 1  # minimum of 1 minute

    # --- infer native cadence (median minute delta) ---
    if len(df) >= 2:
        diffs_min = (df["time"].sort_values().diff().dropna()
                     / np.timedelta64(1, "m")).to_numpy()
        native_step_minutes = int(round(np.median(diffs_min))) if diffs_min.size else step_minutes
        if native_step_minutes < 1:
            native_step_minutes = 1
    else:
        native_step_minutes = step_minutes  # trivial series

    # --- build target grid ---
    if step_minutes > native_step_minutes:
        # Downsample (coarser): resample to the target cadence
        df = df.set_index("time")
        rule = f"{step_minutes}min"
        df = df.resample(rule).mean(numeric_only=True).dropna().reset_index()
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()
    elif step_minutes == native_step_minutes:
        # Keep native timestamps (no-op)
        target_times = df["time"].drop_duplicates().sort_values().to_numpy()
    else:
        # Upsample (finer): construct evenly spaced grid and align/interpolate
        t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
        rule = f"{step_minutes}min"
        target_times = pd.date_range(t0, t1, freq=rule, inclusive="both").to_numpy()

    # In noleap calendars we must *also* ensure the target grid contains no Feb 29.
    if calendar.lower() == "noleap":
        target_times = _drop_feb29(target_times)

    ref_date = target_times[0]

    # Numeric DTIME
    if calendar.lower() == "noleap":
        # IMPORTANT: DTIME must be computed in the declared calendar.
        # Using real (Gregorian) timedeltas in leap years will shift Mar 1 → Mar 2
        # and push the end of year into the next year when interpreted as 'noleap'.
        dtime_vals = _noleap_offset(dtime_units, target_times, ref_date)
    else:
        if dtime_units == "days":
            dtime_vals = (target_times - ref_date) / np.timedelta64(1, "D")
        elif dtime_units == "hours":
            dtime_vals = (target_times - ref_date) / np.timedelta64(1, "h")
        else:
            raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")

    dtime_attr = f"{dtime_units} since {pd.Timestamp(ref_date).strftime('%Y-%m-%d %H:%M:%S')}"

    # Align to target axis with your existing rules
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

    # (3) True accumulations (none by default)
    for v in accum_vars:
        if v in df.columns:
            df_out[v] = df[v].reindex(target_index).ffill().bfill()

    # (4) Carry through other columns (meta), fill both ways
    other_cols = [c for c in df.columns if c not in (linear_vars + ffill_vars + accum_vars)]
    if other_cols:
        df_out[other_cols] = df[other_cols].reindex(target_index).ffill().bfill()

    df_out.index.name = "time"
    df_out = (df_out.reset_index()
                    .sort_values("time")
                    .drop_duplicates(subset="time", keep="first"))

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
