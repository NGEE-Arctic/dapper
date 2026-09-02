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

_NOLEAP_CALENDAR_ALIASES = {
    "noleap",
    "no_leap",
    "365_day",
    "365day",
}


def normalize_calendar(calendar: str = "standard") -> str:
    """Return a CF-compatible calendar name used consistently by dapper."""
    cal = "standard" if calendar is None else str(calendar).strip().lower()
    cal = cal.replace("-", "_").replace(" ", "_")
    if not cal:
        return "standard"
    if cal in _NOLEAP_CALENDAR_ALIASES:
        return "noleap"
    return cal


def is_noleap_calendar(calendar: str = "standard") -> bool:
    """Return True when *calendar* is any supported no-leap calendar alias."""
    return normalize_calendar(calendar) == "noleap"


def _drop_feb29(times):
    """Drop Feb 29 timestamps from a 1D array-like of datetimes."""
    tidx = pd.DatetimeIndex(times)
    mask = ~((tidx.month == 2) & (tidx.day == 29))
    return tidx[mask].to_numpy()


def _noleap_offset(dtime_units: str, target_times, ref_date):
    """Compute numeric offsets in a 365-day (noleap) calendar."""
    dtime_units = str(dtime_units).strip().lower()

    tidx = pd.DatetimeIndex(target_times)
    ref = pd.Timestamp(ref_date)

    # Days since 0001-01-01 in a 365-day calendar (non-leap month lengths)
    y = tidx.year.astype(np.int64)
    m = tidx.month.astype(np.int64)
    d = tidx.day.astype(np.int64)
    day0 = 365 * (y - 1) + _CUMDAYS_NONLEAP[m - 1] + (d - 1)

    ref_day0 = 365 * (ref.year - 1) + _CUMDAYS_NONLEAP[ref.month - 1] + (ref.day - 1)

    # Use hours as the intermediate unit. This avoids tiny drift that made
    # hourly "days since ..." values decode a few microseconds off.
    seconds = (
        np.asarray(tidx.hour, dtype=np.float64) * 3600.0
        + np.asarray(tidx.minute, dtype=np.float64) * 60.0
        + np.asarray(tidx.second, dtype=np.float64)
        + np.asarray(tidx.microsecond, dtype=np.float64) / 1.0e6
        + np.asarray(tidx.nanosecond, dtype=np.float64) / 1.0e9
    )
    ref_seconds = (
        ref.hour * 3600.0
        + ref.minute * 60.0
        + ref.second
        + ref.microsecond / 1.0e6
        + ref.nanosecond / 1.0e9
    )

    hours = (day0 - ref_day0).astype(np.float64) * 24.0
    hours = hours + (seconds - ref_seconds) / 3600.0

    if dtime_units == "days":
        return hours / 24.0
    if dtime_units == "hours":
        return hours
    raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")


def _numeric_dtime(target_times, calendar: str, dtime_units: str):
    """Return numeric DTIME values and their CF units attribute."""
    ref_date = target_times[0]
    if is_noleap_calendar(calendar):
        values = _noleap_offset(dtime_units, target_times, ref_date)
    elif dtime_units == "days":
        values = (target_times - ref_date) / np.timedelta64(1, "D")
    elif dtime_units == "hours":
        values = (target_times - ref_date) / np.timedelta64(1, "h")
    else:
        raise ValueError("Unsupported dtime_units: choose 'days' or 'hours'")

    units = f"{dtime_units} since {pd.Timestamp(ref_date).strftime('%Y-%m-%d %H:%M:%S')}"
    return np.asarray(values, dtype="float64"), units


def _create_interval_aligned_dtime(
    df,
    *,
    calendar: str,
    dtime_units: str,
    dtime_resolution_hrs: float,
    interval_end_vars,
    source_interval_hrs: float,
    target_start=None,
):
    """Align end-labeled interval fields and instantaneous states to one axis."""
    source = df.copy()
    source["time"] = pd.to_datetime(source["time"])
    source = source.sort_values("time").drop_duplicates(subset="time", keep="last")
    if source.empty:
        raise ValueError("No timestamps are available for temporal alignment.")

    target_minutes = max(1, int(round(float(dtime_resolution_hrs) * 60.0)))
    source_minutes = max(1, int(round(float(source_interval_hrs) * 60.0)))
    target_step = pd.Timedelta(minutes=target_minutes)
    source_step = pd.Timedelta(minutes=source_minutes)

    interval_end_vars = tuple(dict.fromkeys(interval_end_vars or ()))
    interval_cols = [name for name in interval_end_vars if name in source.columns]

    # A target value at t describes [t, t + target_step), so source coverage
    # through boundary T supports target timestamps only through T-target_step.
    if interval_cols:
        valid_interval_rows = source[interval_cols].notna().all(axis=1)
        if not valid_interval_rows.any():
            raise ValueError("End-labeled interval variables contain no complete source rows.")
        source_end = source.loc[valid_interval_rows, "time"].max()
    else:
        source_end = source["time"].max()

    start = pd.Timestamp(target_start) if target_start is not None else source["time"].min()
    end = pd.Timestamp(source_end) - target_step
    if end < start:
        raise ValueError(
            "Source data do not include enough lookahead to construct one target interval."
        )

    target_times = pd.date_range(
        start,
        end,
        freq=f"{target_minutes}min",
        inclusive="both",
    ).to_numpy()
    if is_noleap_calendar(calendar):
        target_times = _drop_feb29(target_times)
    if len(target_times) == 0:
        raise ValueError("No timestamps remain after applying calendar filtering.")

    target_index = pd.DatetimeIndex(target_times, name="time")
    state_source = source.set_index("time").sort_index()
    if is_noleap_calendar(calendar):
        state_source = state_source[
            ~((state_source.index.month == 2) & (state_source.index.day == 29))
        ]

    linear_vars = [
        "TBOT", "DTBOT", "RH", "QBOT", "PSRF", "ZBOT",
        "UWIND", "VWIND", "WIND",
    ]
    df_out = pd.DataFrame(index=target_index)

    state_cols = [name for name in linear_vars if name in state_source.columns]
    if state_cols:
        states = state_source[state_cols]
        if target_minutes > source_minutes:
            states = states.resample(
                f"{target_minutes}min", origin=start, label="left", closed="left"
            ).mean()
        df_out[state_cols] = (
            states.reindex(target_index)
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )

    if interval_cols:
        intervals = source.set_index("time")[interval_cols].sort_index()
        intervals.index = intervals.index - source_step
        if is_noleap_calendar(calendar):
            intervals = intervals[
                ~((intervals.index.month == 2) & (intervals.index.day == 29))
            ]
        intervals = intervals[~intervals.index.duplicated(keep="last")]

        if target_minutes > source_minutes:
            intervals = intervals.resample(
                f"{target_minutes}min", origin=start, label="left", closed="left"
            ).mean()
            aligned_intervals = intervals.reindex(target_index)
        elif target_minutes < source_minutes:
            aligned_intervals = intervals.reindex(target_index).ffill()
        else:
            aligned_intervals = intervals.reindex(target_index)
        df_out[interval_cols] = aligned_intervals

    other_cols = [
        name
        for name in state_source.columns
        if name not in set(linear_vars).union(interval_end_vars)
    ]
    if other_cols:
        df_out[other_cols] = state_source[other_cols].reindex(target_index).ffill().bfill()

    df_out = df_out.reset_index()
    dtime_vals, dtime_attr = _numeric_dtime(
        target_times, calendar=calendar, dtime_units=dtime_units
    )
    return dtime_vals, dtime_attr, df_out


def create_dtime(
    df,
    calendar: str = "standard",
    dtime_units: str = "days",
    dtime_resolution_hrs: float = 1.0,
    *,
    interval_end_vars=None,
    source_interval_hrs: float | None = None,
    target_start=None,
):
    """
    Construct a numeric DTIME axis and align data onto it at an arbitrary cadence.
    Accepts fractional hours, e.g., 0.5 (30 min), 0.3 (18 min), 1.5 (90 min).

    ``interval_end_vars`` identifies source fields whose timestamps label the
    end of an averaging/accumulation interval. These fields are relabeled to
    interval starts before calendar filtering or temporal resampling.
    """
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")
    if dtime_resolution_hrs <= 0:
        raise ValueError("dtime_resolution_hrs must be > 0.")

    calendar = normalize_calendar(calendar)
    dtime_units = str(dtime_units).strip().lower()

    if interval_end_vars:
        if source_interval_hrs is None or source_interval_hrs <= 0:
            raise ValueError("source_interval_hrs must be > 0 for end-labeled intervals.")
        return _create_interval_aligned_dtime(
            df,
            calendar=calendar,
            dtime_units=dtime_units,
            dtime_resolution_hrs=dtime_resolution_hrs,
            interval_end_vars=interval_end_vars,
            source_interval_hrs=source_interval_hrs,
            target_start=target_start,
        )

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if is_noleap_calendar(calendar):
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
        df = df.resample(rule).mean(numeric_only=True).dropna(how="all").reset_index()
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
    if is_noleap_calendar(calendar):
        target_times = _drop_feb29(target_times)
    if len(target_times) == 0:
        raise ValueError("No timestamps remain after applying calendar filtering.")

    ref_date = target_times[0]

    # Numeric DTIME
    if is_noleap_calendar(calendar):
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


def start_end_years_from_dates(
    date_values,
    calendar: str = "standard",
    clip_to_full_years: bool = True,
):
    """
    Return the start/end years represented by a sequence of datetimes.

    When ``clip_to_full_years`` is True, use the earliest/latest years that
    contain both Jan 1 and Dec 31. If no full years are present, fall back to
    the min/max years in the data.
    """
    dates = pd.DataFrame(
        {"date": pd.to_datetime(pd.Series(date_values), errors="coerce")}
    ).dropna(subset=["date"])
    if dates.empty:
        raise ValueError("No valid dates found while inferring start/end years.")

    dates.sort_values(by="date", inplace=True)

    if is_noleap_calendar(calendar):
        dates = dates[~((dates["date"].dt.month == 2) & (dates["date"].dt.day == 29))]
        if dates.empty:
            raise ValueError("No dates remain after applying noleap calendar filtering.")

    dates["year"] = dates["date"].dt.year
    dates["month_day"] = dates["date"].dt.month * 100 + dates["date"].dt.day

    if clip_to_full_years:
        full = dates.groupby("year")["month_day"].agg(
            lambda x: {101, 1231}.issubset(set(x))
        )
        full_years = full[full].index

        if len(full_years) > 0:
            return int(full_years[0]), int(full_years[-1])

    return int(dates["date"].dt.year.min()), int(dates["date"].dt.year.max())


def get_start_end_years(
    csv_filepaths,
    calendar: str = "standard",
    clip_to_full_years: bool = True,
):
    """
    Inspect CSVs (must contain a 'date' column) and return earliest/latest
    years present. By default, clip to full years when possible. If no full
    years are present, or ``clip_to_full_years`` is False, return the min/max
    year in data.
    """
    dates = [pd.read_csv(file, usecols=["date"]) for file in csv_filepaths]
    dates = pd.concat(dates, ignore_index=True)
    return start_end_years_from_dates(
        dates["date"],
        calendar=calendar,
        clip_to_full_years=clip_to_full_years,
    )
