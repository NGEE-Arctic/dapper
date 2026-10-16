"""Characterization tests for dapper.geo.sampling helpers (T-9).

These pin current behavior ahead of refactoring.
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dapper.geo import sampling


@pytest.fixture
def grid_ds():
    lat = np.array([-0.5, 0.5, 1.5])
    lon = np.array([358.5, 359.5, 0.5, 1.5])
    return xr.Dataset(
        {
            "TOPO": (("lsmlat", "lsmlon"), np.arange(12.0).reshape(3, 4)),
            "NAME": ((), "x"),
            "LATIXY": (("lsmlat", "lsmlon"), np.repeat(lat[:, None], 4, 1)),
            "LONGXY": (("lsmlat", "lsmlon"), np.repeat(lon[None, :], 3, 0)),
        },
        attrs={"src": "a"},
    )


@pytest.fixture
def points():
    return pd.DataFrame({"lat": [0.4, 1.4], "lon": [-0.6, 1.6]})


def test_infer_lat_lon_vars(grid_ds):
    assert sampling.infer_lat_lon_vars(grid_ds) == ("LATIXY", "LONGXY")
    with pytest.raises(ValueError, match="Could not infer lat/lon variables"):
        sampling.infer_lat_lon_vars(xr.Dataset({"a": 1}))


def test_infer_grid_metadata(grid_ds):
    assert sampling.infer_grid_metadata(grid_ds) == {
        "dapper_source_grid_lat_dim": "lsmlat",
        "dapper_source_grid_lon_dim": "lsmlon",
        "dapper_source_grid_lat_var": "LATIXY",
        "dapper_source_grid_lon_var": "LONGXY",
        "dapper_source_grid_lon_wrap": "0_360",
        "dapper_source_grid_nlat": 3,
        "dapper_source_grid_nlon": 4,
        "dapper_source_grid_dlat_deg": 1.0,
        "dapper_source_grid_dlon_deg": 1.0,
    }


def test_points_to_nearest_cells(grid_ds, points):
    out = sampling.points_to_nearest_cells(grid_ds, points)
    assert out.index.name == "index"
    assert out.columns.tolist() == [
        "lat",
        "lon",
        "lon_normalized",
        "i_lat",
        "i_lon",
        "lat_cell",
        "lon_cell",
        "weight",
    ]
    assert out["i_lat"].tolist() == [1, 2]
    assert out["i_lon"].tolist() == [1, 3]
    np.testing.assert_allclose(out["lon_normalized"], [359.4, 1.6])
    np.testing.assert_allclose(out["lon_cell"], [359.5, 1.5])
    np.testing.assert_allclose(out["weight"], [1.0, 1.0])

    weighted = sampling.points_to_nearest_cells(
        grid_ds, points.assign(weight=[2.0, 3.0]), lon_wrap="0_360"
    )
    np.testing.assert_allclose(weighted["weight"], [2.0, 3.0])


def test_sample_gridded_dataset_points(grid_ds, points):
    out = sampling.sample_gridded_dataset_points(grid_ds, points)
    assert out["TOPO"].dims == ("lsmlat", "lsmlon")
    np.testing.assert_allclose(out["TOPO"].values[:, 0], [5.0, 11.0])
    np.testing.assert_allclose(out["LONGXY"].values[:, 0], [359.5, 1.5])
    assert out["NAME"].item() == "x"
    assert "lsmlat" not in out.coords
    assert out.attrs == {"src": "a"}


def test_ensure_weight(points):
    added = sampling.ensure_weight(points, default_weight=4)
    assert added is not points
    assert added["weight"].tolist() == [4.0, 4.0]
    with_weight = points.assign(weight=1)
    assert sampling.ensure_weight(with_weight) is with_weight


def test_write_netcdf_attrs_and_encoding(grid_ds, tmp_path):
    path = sampling.write_netcdf(
        grid_ds,
        tmp_path / "sub" / "o.nc",
        dapper_attrs={"src": "d", "k": 1},
        append_attrs={"u": 2},
    )
    assert path == tmp_path / "sub" / "o.nc"
    with xr.open_dataset(path) as ds:
        attrs = dict(ds.attrs)
        assert ds["TOPO"].encoding["zlib"] is True
        assert ds["TOPO"].encoding["complevel"] == 4
    assert list(attrs) == ["src", "k", "dapper_created_utc", "u"]
    assert (attrs["src"], attrs["k"], attrs["u"]) == ("a", 1, 2)
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", attrs["dapper_created_utc"]
    )

    path2 = sampling.write_netcdf(
        grid_ds, tmp_path / "o2.nc", compress=False, add_created_utc=False
    )
    with xr.open_dataset(path2) as ds:
        assert dict(ds.attrs) == {"src": "a"}
        assert ds["TOPO"].encoding.get("zlib") is False
