import numpy as np
import pandas as pd
import pytest
from netCDF4 import num2date

from dapper.met.adapters.era5 import ERA5Adapter
from dapper.met.temporal import create_dtime, get_start_end_years, normalize_calendar


def _met_df(times):
    return pd.DataFrame(
        {
            "time": times,
            "TBOT": np.arange(len(times), dtype=float),
            "gid": "g1",
            "LATIXY": 70.0,
            "LONGXY": 250.0,
            "zone": 1,
        }
    )


def _decode_tuples(dtime_vals, dtime_units, calendar):
    decoded = num2date(
        dtime_vals,
        units=dtime_units,
        calendar=normalize_calendar(calendar),
        only_use_cftime_datetimes=True,
    )
    return [(d.year, d.month, d.day, d.hour) for d in decoded]


def _write_dates_csv(path, start, end, freq="1D"):
    pd.DataFrame({"date": pd.date_range(start, end, freq=freq)}).to_csv(
        path,
        index=False,
    )
    return path


def test_get_start_end_years_clips_partial_boundary_years_by_default(tmp_path):
    csv_path = _write_dates_csv(tmp_path / "era5.csv", "2024-01-01", "2025-06-13")

    assert get_start_end_years([csv_path], calendar="noleap") == (2024, 2024)


def test_get_start_end_years_can_keep_partial_boundary_years(tmp_path):
    csv_path = _write_dates_csv(tmp_path / "era5.csv", "2024-01-01", "2025-06-13")

    assert get_start_end_years(
        [csv_path],
        calendar="noleap",
        clip_to_full_years=False,
    ) == (2024, 2025)


def test_get_start_end_years_falls_back_when_no_full_year_exists(tmp_path):
    csv_path = _write_dates_csv(tmp_path / "era5.csv", "2025-01-19", "2025-06-13")

    assert get_start_end_years([csv_path], calendar="noleap") == (2025, 2025)


def test_era5_adapter_discover_files_respects_clip_to_full_years(tmp_path):
    _write_dates_csv(tmp_path / "era5.csv", "2024-01-01", "2025-06-13")
    adapter = ERA5Adapter()

    _, start_year, end_year = adapter.discover_files(tmp_path, calendar="noleap")
    assert (start_year, end_year) == (2024, 2024)

    _, start_year, end_year = adapter.discover_files(
        tmp_path,
        calendar="noleap",
        clip_to_full_years=False,
    )
    assert (start_year, end_year) == (2024, 2025)


def test_noleap_dtime_does_not_skip_march_1_after_leap_day():
    times = pd.date_range(
        "1952-02-28 20:00",
        "1952-03-01 06:00",
        freq="1h",
        inclusive="both",
    )

    dtime_vals, dtime_units, aligned = create_dtime(
        _met_df(times),
        calendar="noleap",
        dtime_units="days",
        dtime_resolution_hrs=1,
    )

    assert not ((aligned["time"].dt.month == 2) & (aligned["time"].dt.day == 29)).any()

    decoded_after = [
        t
        for t in _decode_tuples(dtime_vals, dtime_units, "noleap")
        if t > (1952, 2, 28, 20)
    ]
    assert decoded_after[:10] == [
        (1952, 2, 28, 21),
        (1952, 2, 28, 22),
        (1952, 2, 28, 23),
        (1952, 3, 1, 0),
        (1952, 3, 1, 1),
        (1952, 3, 1, 2),
        (1952, 3, 1, 3),
        (1952, 3, 1, 4),
        (1952, 3, 1, 5),
        (1952, 3, 1, 6),
    ]


def test_noleap_dtime_does_not_accumulate_extra_days_by_2025():
    times = pd.date_range("1950-01-01", "2025-01-19", freq="1D", inclusive="both")

    dtime_vals, dtime_units, aligned = create_dtime(
        _met_df(times),
        calendar="noleap",
        dtime_units="days",
        dtime_resolution_hrs=24,
    )

    assert len(aligned) == 75 * 365 + 19
    assert _decode_tuples(dtime_vals, dtime_units, "noleap")[-1] == (2025, 1, 19, 0)


@pytest.mark.parametrize("calendar", ["noleap", "NO_LEAP", "no_leap", "365_day", "365-day"])
def test_noleap_calendar_aliases_share_noleap_dtime_behavior(calendar):
    times = pd.date_range(
        "2020-02-28 22:00",
        "2020-03-01 02:00",
        freq="1h",
        inclusive="both",
    )

    dtime_vals, dtime_units, aligned = create_dtime(
        _met_df(times),
        calendar=calendar,
        dtime_units="days",
        dtime_resolution_hrs=1,
    )

    assert normalize_calendar(calendar) == "noleap"
    assert aligned["time"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2020-02-28 22:00",
        "2020-02-28 23:00",
        "2020-03-01 00:00",
        "2020-03-01 01:00",
        "2020-03-01 02:00",
    ]
    assert _decode_tuples(dtime_vals, dtime_units, calendar) == [
        (2020, 2, 28, 22),
        (2020, 2, 28, 23),
        (2020, 3, 1, 0),
        (2020, 3, 1, 1),
        (2020, 3, 1, 2),
    ]


def test_downsample_keeps_rows_when_some_variables_are_all_nan():
    times = pd.date_range("2020-01-01", "2020-01-01 06:00", freq="1h", inclusive="both")
    df = _met_df(times)
    df["TBOT"] = np.nan
    df["FSDS"] = np.arange(len(df), dtype=float)

    dtime_vals, dtime_units, aligned = create_dtime(
        df,
        calendar="noleap",
        dtime_units="days",
        dtime_resolution_hrs=3,
    )

    assert len(dtime_vals) == 3
    assert aligned["time"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2020-01-01 00:00",
        "2020-01-01 03:00",
        "2020-01-01 06:00",
    ]
    assert aligned["FSDS"].notna().all()
