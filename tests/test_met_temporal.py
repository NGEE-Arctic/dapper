import numpy as np
import pandas as pd
import pytest
from netCDF4 import num2date

from dapper.met.temporal import create_dtime, normalize_calendar


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
