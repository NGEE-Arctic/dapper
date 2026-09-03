from datetime import datetime, timezone

from dapper.integrations.earthengine.gee_utils import (
    _era5_source_end_exclusive,
    _parse_era5_datetime,
    determine_gee_batches,
)


def _timestamp_ms(value):
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def test_era5_source_end_adds_one_hour_lookahead():
    output_end = datetime(2025, 1, 1)
    latest = datetime(2025, 6, 1, 23)

    assert _era5_source_end_exclusive(output_end, _timestamp_ms(latest)) == datetime(
        2025, 1, 1, 1
    )


def test_era5_source_end_is_capped_after_latest_available_image():
    output_end = datetime(2100, 1, 1)
    latest = datetime(2025, 6, 1, 23)

    assert _era5_source_end_exclusive(output_end, _timestamp_ms(latest)) == datetime(
        2025, 6, 2, 0
    )


def test_gee_batches_preserve_hourly_final_boundary():
    batches = determine_gee_batches(
        datetime(2020, 1, 1),
        datetime(2022, 1, 1),
        datetime(2022, 1, 1),
        years_per_task=1,
        verbose=False,
    )
    batches.loc[batches.index[-1], "task_end"] = datetime(2022, 1, 1, 1)

    assert len(batches) == 2
    assert batches.iloc[0]["task_start"] == datetime(2020, 1, 1)
    assert batches.iloc[-1]["task_end"] == datetime(2022, 1, 1, 1)


def test_gee_date_parser_accepts_iso_hours_and_resolves_latest():
    latest = datetime(2026, 8, 28, 23)

    assert _parse_era5_datetime("2025-01-01T12:00:00") == datetime(
        2025, 1, 1, 12
    )
    assert _parse_era5_datetime(
        "latest",
        latest_timestamp_ms=_timestamp_ms(latest),
        allow_latest=True,
    ) == latest
