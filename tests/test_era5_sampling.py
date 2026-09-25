from __future__ import annotations

import json
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box
import xarray as xr

from dapper import Domain, ERA5Adapter
from dapper.integrations import era5 as era5_sampling
from dapper.integrations.era5 import (
    era5_land_grid_cells,
    plan_era5_land_sampling,
    sample_era5_land,
)


def _domain(geometries, gids=None):
    if not isinstance(geometries, list):
        geometries = [geometries]
    gids = gids or [f"site_{index}" for index in range(len(geometries))]
    support = gpd.GeoDataFrame(
        {"gid": gids, "geometry": geometries},
        crs="EPSG:4326",
    )
    return Domain.from_provided(support, name="test", mode="sites")


def test_era5_land_grid_cells_are_area_weighted():
    geometry = box(-149.64, 68.56, -149.50, 68.64)

    cells = era5_land_grid_cells(geometry)

    assert list(zip(cells["longitude"], cells["latitude"])) == [
        (-149.6, 68.6),
        (-149.5, 68.6),
    ]
    assert cells["weight"].sum() == pytest.approx(1.0)
    assert cells.iloc[0]["weight"] > cells.iloc[1]["weight"]


def test_auto_plan_prefers_arco_for_a_small_long_record_polygon():
    domain = _domain(box(-149.64, 68.56, -149.50, 68.64))

    plan = plan_era5_land_sampling(
        domain,
        "1950-01-02",
        "2026-01-01",
    )

    assert plan.backend == "arco"
    assert plan.grid_cell_count == 2
    assert plan.estimated_seconds < 300


def test_auto_plan_uses_gee_for_large_polygon():
    large = _domain(box(-150, 68, -148, 69))

    assert plan_era5_land_sampling(large, "2000-01-01", "2001-01-01").backend == "gee"


def test_auto_plan_does_not_require_gee_for_missing_arco_first_day():
    first_day = _domain(Point(-149.6, 68.6))

    plan = plan_era5_land_sampling(first_day, "1950-01-01", "1951-01-01")
    assert plan.backend == "arco"
    assert "1950-01-02" in plan.reason

    explicit = plan_era5_land_sampling(
        first_day,
        "1950-01-01",
        "1951-01-01",
        backend="arco",
    )
    assert explicit.backend == "arco"
    assert "will be omitted" in explicit.reason

    with pytest.raises(ValueError, match="begins in 1950"):
        plan_era5_land_sampling(first_day, "1949-01-01", "1951-01-01")


def test_auto_plan_accounts_for_per_location_request_time():
    points = [Point(-150 + index * 0.1, 68) for index in range(11)]
    domain = _domain(points)

    plan = plan_era5_land_sampling(domain, "2025-01-01", "2025-01-02")

    assert plan.grid_cell_count == 11
    assert plan.estimated_seconds >= 330
    assert plan.backend == "gee"


def test_gee_sampling_preserves_model_coordinates_and_exports_reference(
    monkeypatch, tmp_path
):
    domain = _domain(box(-149.64, 68.56, -149.50, 68.64))
    model_cells = domain.cells.copy()
    model_cells["zone"] = 3
    model_cells["weight"] = 0.75
    domain = domain.copy(cells=model_cells)
    model_cell = domain.cells.iloc[0]
    sampling_point = Point(model_cell["lon"] + 0.02, model_cell["lat"] + 0.01)
    sampling_domain = _domain(sampling_point)
    sampling_cells = sampling_domain.cells.copy()
    sampling_cells["method"] = "sampled at provided coordinate"
    sampling_cells["sampled_geometry"] = sampling_point.wkt
    sampling_cells["zone"] = 1
    sampling_cells["weight"] = 1.0
    sampling_domain = sampling_domain.copy(cells=sampling_cells)
    captured = {}

    def fake_sample_e5lh(params, domain_name, skip_tasks):
        captured.update(params)
        captured["domain_name"] = domain_name
        captured["skip_tasks"] = skip_tasks
        return sampling_domain

    monkeypatch.setattr(
        "dapper.integrations.earthengine.gee_utils.sample_e5lh",
        fake_sample_e5lh,
    )
    sampled = sample_era5_land(
        domain,
        "2025-01-01",
        "latest",
        backend="gee",
        gdrive_folder="test-folder",
        skip_tasks=True,
    )

    assert captured["geometries"].geometry.iloc[0].equals(
        domain.support_for(step="met").geometry.iloc[0]
    )
    assert captured["end_date"] == "latest"
    assert captured["gdrive_folder"] == "test-folder"
    assert captured["skip_tasks"] is True
    assert sampled.cells.iloc[0]["sampling_backend"] == "gee"
    assert sampled.cells.iloc[0]["lon"] == pytest.approx(model_cell["lon"])
    assert sampled.cells.iloc[0]["lat"] == pytest.approx(model_cell["lat"])
    assert sampled.cells.iloc[0]["geometry"].equals(model_cell["geometry"])
    assert sampled.cells.iloc[0]["zone"] == 3
    assert sampled.cells.iloc[0]["weight"] == pytest.approx(0.75)
    assert sampled.cells.iloc[0]["sampling_reference_lon"] == pytest.approx(
        sampling_point.x
    )
    assert sampled.cells.iloc[0]["sampling_reference_lat"] == pytest.approx(
        sampling_point.y
    )

    arco = sample_era5_land(
        domain,
        "2025-01-01",
        "latest",
        backend="arco",
        skip_tasks=True,
    )
    assert sampled.to_df_loc().iloc[0]["lon"] == pytest.approx(
        arco.to_df_loc().iloc[0]["lon"]
    )
    assert sampled.to_df_loc().iloc[0]["lat"] == pytest.approx(
        arco.to_df_loc().iloc[0]["lat"]
    )
    assert sampled.to_df_loc().iloc[0]["sampling_reference_lon"] == pytest.approx(
        sampling_point.x
    )

    csv_dir = tmp_path / "gee_csv"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "gid": ["site_0"] * 3,
            "date": pd.date_range("2025-01-01", periods=3, freq="h"),
            "temperature_2m": [270.0, 271.0, 272.0],
            "dewpoint_temperature_2m": [268.0, 269.0, 270.0],
            "surface_pressure": [90000.0, 90010.0, 90020.0],
            "u_component_of_wind_10m": [1.0, 1.0, 1.0],
            "v_component_of_wind_10m": [2.0, 2.0, 2.0],
            "surface_solar_radiation_downwards_hourly": [0.0, 3600.0, 7200.0],
            "surface_thermal_radiation_downwards_hourly": [720000.0] * 3,
            "total_precipitation_hourly": [0.0, 0.001, 0.002],
        }
    ).to_csv(csv_dir / "gee.csv", index=False)
    output_dir = tmp_path / "elm"
    sampled.export_met(
        src_path=csv_dir,
        adapter=ERA5Adapter(),
        out_dir=output_dir,
        overwrite=True,
        calendar="noleap",
        dtime_resolution_hrs=1,
        dtime_units="days",
        dformat="BYPASS",
        clip_to_full_years=False,
    )

    mapping = (output_dir / "site_0" / "MET" / "zone_mappings.txt").read_text()
    mapping_lon, mapping_lat, _, _ = mapping.strip().split("\t")
    assert float(mapping_lon) == pytest.approx(model_cell["lon"] % 360)
    assert float(mapping_lat) == pytest.approx(model_cell["lat"])
    with xr.open_dataset(
        output_dir / "site_0" / "MET" / "ERA5_TBOT_2025-2025_z01.nc"
    ) as output:
        assert float(output["LONGXY"].values[0]) == pytest.approx(
            model_cell["lon"] % 360
        )
        assert float(output["LATIXY"].values[0]) == pytest.approx(model_cell["lat"])
        assert output.attrs["sampling_reference_lon"] == pytest.approx(
            sampling_point.x
        )
        assert output.attrs["sampling_reference_lat"] == pytest.approx(
            sampling_point.y
        )

    arco_output_dir = tmp_path / "elm_arco"
    arco.export_met(
        src_path=csv_dir,
        adapter=ERA5Adapter(),
        out_dir=arco_output_dir,
        overwrite=True,
        calendar="noleap",
        dtime_resolution_hrs=1,
        dtime_units="days",
        dformat="BYPASS",
        clip_to_full_years=False,
    )
    arco_mapping = (
        arco_output_dir / "site_0" / "MET" / "zone_mappings.txt"
    ).read_text()
    assert arco_mapping == mapping


class _FakeCDSClient:
    def __init__(self):
        self.requests = []

    def retrieve(self, dataset, request, target):
        self.requests.append((dataset, request))
        request_start = request["date"][0].split("/", 1)[0]
        times = pd.date_range(request_start, periods=3, freq="h")
        coords = {
            "valid_time": times,
            "latitude": [0.0],
            "longitude": [0.0, 0.1],
        }
        shape = (len(times), 1, 2)

        def values(left, right):
            return np.broadcast_to(np.array([left, right]), shape).copy()

        groups = {
            "temperature.nc": xr.Dataset(
                {
                    "t2m": (("valid_time", "latitude", "longitude"), values(274, 276)),
                    "d2m": (("valid_time", "latitude", "longitude"), values(270, 272)),
                },
                coords=coords,
            ),
            "pressure.nc": xr.Dataset(
                {
                    "sp": (("valid_time", "latitude", "longitude"), values(90000, 90200)),
                    "tp": (("valid_time", "latitude", "longitude"), values(0.001, 0.003)),
                },
                coords=coords,
            ),
            "wind.nc": xr.Dataset(
                {
                    "u10": (("valid_time", "latitude", "longitude"), values(1, 3)),
                    "v10": (("valid_time", "latitude", "longitude"), values(2, 4)),
                },
                coords=coords,
            ),
            "radiation.nc": xr.Dataset(
                {
                    "ssrd": (("valid_time", "latitude", "longitude"), values(3600, 7200)),
                    "strd": (("valid_time", "latitude", "longitude"), values(720000, 724000)),
                },
                coords=coords,
            ),
        }
        with zipfile.ZipFile(target, "w") as archive:
            for filename, dataset_value in groups.items():
                archive.writestr(filename, dataset_value.to_netcdf())


def test_arco_sampling_writes_adapter_ready_csv_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        era5_sampling,
        "_arco_available_range",
        lambda variables: (
            pd.Timestamp("1950-01-02 00:00:00"),
            pd.Timestamp("2026-01-01 23:00:00"),
        ),
    )
    client = _FakeCDSClient()
    domain = _domain(box(-0.04, -0.04, 0.14, 0.04), gids=["tfs"])
    csv_dir = tmp_path / "arco"

    sampled = sample_era5_land(
        domain,
        "2020-01-01",
        "2020-01-01 02:00:00",
        backend="arco",
        output_dir=csv_dir,
        cds_client=client,
    )

    csv_path = csv_dir / "era5_land_arco_tfs.csv"
    frame = pd.read_csv(csv_path)
    assert len(frame) == 3
    assert frame["temperature_2m"].to_numpy() == pytest.approx([275, 275, 275])
    assert client.requests[0][0] == "reanalysis-era5-land-timeseries"
    assert client.requests[0][1]["area"] == pytest.approx([0.001, -0.001, -0.001, 0.101])
    assert sampled.cells.iloc[0]["sampling_backend"] == "arco"
    assert sampled.cells.iloc[0]["sampling_grid_cell_count"] == 2
    assert (csv_dir / "era5_land_sampling_manifest.json").exists()

    output_dir = tmp_path / "elm"
    sampled.export_met(
        src_path=csv_dir,
        adapter=ERA5Adapter(),
        out_dir=output_dir,
        overwrite=True,
        calendar="noleap",
        dtime_resolution_hrs=1,
        dtime_units="days",
        dformat="BYPASS",
        clip_to_full_years=False,
    )
    met_path = output_dir / "tfs" / "MET" / "ERA5_TBOT_2020-2020_z01.nc"
    with xr.open_dataset(met_path) as output:
        times = output["DTIME"].values
        assert len(times) == 2
        assert (times[0].year, times[0].month, times[0].day, times[0].hour) == (
            2020,
            1,
            1,
            0,
        )
        assert (times[-1].year, times[-1].month, times[-1].day, times[-1].hour) == (
            2020,
            1,
            1,
            1,
        )
        assert output.attrs["sampling_backend"] == "arco"
        assert output.attrs["sampling_grid_cell_count"] == 2
        assert output.attrs["sampling_requested_start"] == "2020-01-01 00:00:00"
        assert output.attrs["sampling_output_end"] == "2020-01-01 01:00:00"
        assert "area-weighted mean" in output.attrs["sampling_method"]


def test_arco_sampling_warns_and_clamps_missing_first_day(tmp_path, monkeypatch):
    monkeypatch.setattr(
        era5_sampling,
        "_arco_available_range",
        lambda variables: (
            pd.Timestamp("1950-01-02 00:00:00"),
            pd.Timestamp("2026-01-01 23:00:00"),
        ),
    )
    client = _FakeCDSClient()
    csv_dir = tmp_path / "arco"

    with pytest.warns(UserWarning, match="does not provide data for 1950-01-01"):
        sampled = sample_era5_land(
            _domain(Point(0, 0), gids=["first_day"]),
            "1950-01-01",
            "1950-01-02 02:00:00",
            backend="arco",
            output_dir=csv_dir,
            cds_client=client,
        )

    assert client.requests[0][1]["date"] == ["1950-01-02/1950-01-02"]
    frame = pd.read_csv(csv_dir / "era5_land_arco_first_day.csv")
    assert frame["date"].iloc[0].startswith("1950-01-02")
    assert len(frame) == 3
    assert sampled.cells.iloc[0]["sampling_requested_start"].startswith("1950-01-01")
    assert sampled.cells.iloc[0]["sampling_start"].startswith("1950-01-02")

    manifest = json.loads(
        (csv_dir / "era5_land_sampling_manifest.json").read_text()
    )
    feature = manifest["features"][0]
    assert feature["sampling_requested_start"].startswith("1950-01-01")
    assert feature["sampling_start"].startswith("1950-01-02")


def test_arco_dry_run_warns_and_records_clamped_start():
    with pytest.warns(UserWarning, match="clip_to_full_years=True"):
        sampled = sample_era5_land(
            _domain(Point(0, 0), gids=["first_day"]),
            "1950-01-01",
            "1951-01-01",
            backend="arco",
            skip_tasks=True,
        )

    assert sampled.cells.iloc[0]["sampling_requested_start"].startswith("1950-01-01")
    assert sampled.cells.iloc[0]["sampling_start"].startswith("1950-01-02")


@pytest.mark.skipif(
    os.environ.get("DAPPER_RUN_CDS_INTEGRATION") != "1",
    reason="set DAPPER_RUN_CDS_INTEGRATION=1 to contact ECMWF CDS",
)
def test_live_arco_point_download(tmp_path):
    sampled = sample_era5_land(
        _domain(Point(-149.6, 68.6), gids=["toolik"]),
        "2025-01-01",
        "2025-01-01 02:00:00",
        backend="arco",
        output_dir=tmp_path,
    )

    frame = pd.read_csv(tmp_path / "era5_land_arco_toolik.csv")
    assert len(frame) == 3
    assert not frame[list(era5_sampling._RAW_TO_CDS)].isna().any().any()
    assert sampled.cells.iloc[0]["sampling_grid_cell_count"] == 1
