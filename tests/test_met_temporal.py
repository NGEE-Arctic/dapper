import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset, num2date
from shapely.geometry import Point

from dapper.domains.domain import Domain
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


def _interval_met_df(times):
    df = _met_df(times)
    values = np.arange(1, len(times) + 1, dtype=float)
    df["FSDS"] = values * 10.0
    df["FLDS"] = values * 100.0
    df["PRECTmms"] = values / 1000.0
    return df


def _create_era5_dtime(df, **kwargs):
    return create_dtime(
        df,
        calendar=kwargs.pop("calendar", "noleap"),
        dtime_units="hours",
        dtime_resolution_hrs=kwargs.pop("dtime_resolution_hrs", 1),
        interval_end_vars=("FSDS", "FLDS", "PRECTmms"),
        source_interval_hrs=1,
        **kwargs,
    )


def _raw_era5_df(times, gid="g1"):
    n = len(times)
    return pd.DataFrame(
        {
            "date": times,
            "gid": gid,
            "lat": 70.0,
            "lon": -150.0,
            "zone": 1,
            "temperature_2m": 270.0 + np.arange(n),
            "dewpoint_temperature_2m": 268.0 + np.arange(n),
            "surface_pressure": 100000.0,
            "u_component_of_wind_10m": 3.0,
            "v_component_of_wind_10m": 4.0,
            "surface_solar_radiation_downwards_hourly": 360000.0 + 3600.0 * np.arange(n),
            "surface_thermal_radiation_downwards_hourly": 1080000.0 + 3600.0 * np.arange(n),
            "total_precipitation_hourly": 0.00036 + 0.000036 * np.arange(n),
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


def test_get_start_end_years_raises_when_no_full_year_exists(tmp_path):
    csv_path = _write_dates_csv(tmp_path / "era5.csv", "2025-01-19", "2025-06-13")

    with pytest.raises(ValueError, match="no complete calendar year"):
        get_start_end_years([csv_path], calendar="noleap")

    assert get_start_end_years(
        [csv_path],
        calendar="noleap",
        clip_to_full_years=False,
    ) == (2025, 2025)


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


def test_era5_interval_fields_are_relabelled_from_interval_end_to_start():
    times = pd.date_range("1950-01-01 01:00", periods=4, freq="1h")

    dtime_vals, dtime_units, aligned = _create_era5_dtime(
        _interval_met_df(times),
        target_start="1950-01-01 00:00",
    )

    assert _decode_tuples(dtime_vals, dtime_units, "noleap") == [
        (1950, 1, 1, 0),
        (1950, 1, 1, 1),
        (1950, 1, 1, 2),
        (1950, 1, 1, 3),
    ]
    assert aligned["FSDS"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert aligned["FLDS"].tolist() == [100.0, 200.0, 300.0, 400.0]
    assert aligned["PRECTmms"].tolist() == [0.001, 0.002, 0.003, 0.004]
    assert aligned["TBOT"].tolist() == [0.0, 0.0, 1.0, 2.0]


def test_era5_normal_start_uses_same_time_states_and_next_hour_intervals():
    times = pd.date_range("2021-01-01 00:00", periods=4, freq="1h")

    _, _, aligned = _create_era5_dtime(_interval_met_df(times))

    assert aligned["time"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2021-01-01 00:00",
        "2021-01-01 01:00",
        "2021-01-01 02:00",
    ]
    assert aligned["TBOT"].tolist() == [0.0, 1.0, 2.0]
    assert aligned["FSDS"].tolist() == [20.0, 30.0, 40.0]


def test_era5_noleap_alignment_keeps_february_28_final_interval():
    times = pd.date_range("2020-02-28 22:00", "2020-03-01 02:00", freq="1h")
    df = _interval_met_df(times)
    source_values = dict(zip(df["time"], df["FSDS"]))

    _, _, aligned = _create_era5_dtime(df)

    assert aligned["time"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2020-02-28 22:00",
        "2020-02-28 23:00",
        "2020-03-01 00:00",
        "2020-03-01 01:00",
    ]
    assert aligned.loc[1, "FSDS"] == source_values[pd.Timestamp("2020-02-29 00:00")]
    assert aligned.loc[2, "FSDS"] == source_values[pd.Timestamp("2020-03-01 01:00")]


def test_era5_interval_alignment_resamples_to_three_hours():
    times = pd.date_range("2021-01-01 00:00", "2021-01-02 00:00", freq="1h")

    _, _, aligned = _create_era5_dtime(
        _interval_met_df(times),
        dtime_resolution_hrs=3,
    )

    assert aligned["time"].iloc[0] == pd.Timestamp("2021-01-01 00:00")
    assert aligned["time"].iloc[-1] == pd.Timestamp("2021-01-01 21:00")
    assert aligned["FSDS"].iloc[0] == pytest.approx(np.mean([20.0, 30.0, 40.0]))
    assert aligned["FSDS"].iloc[-1] == pytest.approx(np.mean([230.0, 240.0, 250.0]))


def test_era5_interval_alignment_upsamples_through_last_supported_half_hour():
    times = pd.date_range("2021-01-01 00:00", "2021-01-02 00:00", freq="1h")

    _, _, aligned = _create_era5_dtime(
        _interval_met_df(times),
        dtime_resolution_hrs=0.5,
    )

    assert len(aligned) == 48
    assert aligned["time"].iloc[-1] == pd.Timestamp("2021-01-01 23:30")
    assert aligned["FSDS"].iloc[-2:].tolist() == [250.0, 250.0]


def test_era5_noleap_full_year_uses_next_january_midnight_lookahead():
    times = pd.date_range("2020-01-01 00:00", "2021-01-01 00:00", freq="1h")

    dtime_vals, dtime_units, aligned = _create_era5_dtime(_interval_met_df(times))

    assert len(aligned) == 365 * 24
    assert _decode_tuples(dtime_vals, dtime_units, "noleap")[0] == (2020, 1, 1, 0)
    assert _decode_tuples(dtime_vals, dtime_units, "noleap")[-1] == (
        2020,
        12,
        31,
        23,
    )
    assert aligned["FSDS"].iloc[-1] == times.size * 10.0


def test_era5_adapter_retains_noleap_boundary_and_next_year_lookahead():
    times = pd.to_datetime(
        [
            "2020-02-29 00:00",
            "2020-12-31 23:00",
            "2021-01-01 00:00",
            "2021-01-01 01:00",
        ]
    )

    processed = ERA5Adapter().preprocess_shard(
        _raw_era5_df(times),
        start_year=2020,
        end_year=2020,
        calendar="noleap",
        dformat="BYPASS",
    )

    assert processed["time"].tolist() == list(times[:3])


def test_era5_adapter_requests_midnight_only_at_gee_collection_boundary():
    adapter = ERA5Adapter()
    boundary_df = pd.DataFrame({"time": ["1950-01-01 01:00"]})
    normal_df = pd.DataFrame({"time": ["2021-01-01 00:00"]})

    boundary = adapter.temporal_options(
        boundary_df, start_year=1950, end_year=1950, calendar="noleap"
    )
    normal = adapter.temporal_options(
        normal_df, start_year=2021, end_year=2021, calendar="noleap"
    )

    assert boundary["target_start"] == pd.Timestamp("1950-01-01 00:00")
    assert "target_start" not in normal


def test_era5_exporter_writes_midnight_axis_shifted_fluxes_and_metadata(tmp_path):
    source_dir = tmp_path / "csv"
    source_dir.mkdir()
    times = pd.date_range("1950-01-01 01:00", periods=5, freq="1h")
    raw = _raw_era5_df(times)
    raw.drop(columns=["lat", "lon", "zone"]).to_csv(source_dir / "era5.csv", index=False)

    domain = Domain.from_geometry(
        Point(-150.0, 70.0),
        gid="g1",
        name="era5-smoke",
        mode="sites",
        path_out=tmp_path,
    )
    output_root = tmp_path / "output"
    domain.export_met(
        source_dir,
        adapter=ERA5Adapter(),
        out_dir=output_root,
        calendar="noleap",
        dtime_units="hours",
        dtime_resolution_hrs=1,
        clip_to_full_years=False,
        overwrite=True,
    )

    tbot_path = output_root / "g1" / "MET" / "TBOT.nc"
    fsds_path = output_root / "g1" / "MET" / "FSDS.nc"
    prect_path = output_root / "g1" / "MET" / "PRECTmms.nc"
    assert tbot_path.exists() and fsds_path.exists() and prect_path.exists()

    with Dataset(tbot_path) as ds:
        decoded = num2date(
            ds["DTIME"][:],
            ds["DTIME"].units,
            calendar=ds["DTIME"].calendar,
            only_use_cftime_datetimes=True,
        )
        assert (decoded[0].year, decoded[0].month, decoded[0].day, decoded[0].hour) == (
            1950,
            1,
            1,
            0,
        )
        assert len(decoded) == 5
        assert ds.initial_state_fill.startswith("1950-01-01 00:00")
        assert ds.interval_start_variables == "FSDS, FLDS, PRECTmms"
        assert np.isfinite(ds["TBOT"][:]).all()

    with Dataset(fsds_path) as ds:
        assert ds["FSDS"][0, 0] == pytest.approx(100.0, abs=0.1)
        assert ds["FSDS"].cell_methods == "time: mean"
        assert ds["FSDS"].time_representation == "interval_start"
        assert ds.forcing_time_convention == (
            "FSDS, FLDS, PRECTmms represent [DTIME, DTIME + timestep)"
        )

    with Dataset(prect_path) as ds:
        assert ds["PRECTmms"][0, 0] == pytest.approx(0.0001, abs=1e-6)
