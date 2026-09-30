"""Characterization tests for landuse time-series sampling (T-2).

These pin current behavior ahead of refactoring. Assertions marked
``PINS BUG Bn`` document known-incorrect behavior (see REFACTOR_NOTES.md)
and should be updated when that bug is fixed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from dapper import Domain
from dapper.landuse.landuse import export_landuse_timeseries, sample_landuse_timeseries


@pytest.fixture
def landuse_nc(tmp_path):
    lat = np.array([67.25, 67.75, 68.25, 68.75])
    lon = np.array([209.25, 209.75, 210.25, 210.75])
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    pft = np.zeros((2, 3, 4, 4))
    for t in range(2):
        for k in range(3):
            pft[t, k] = (k + 1) * 10 + t + np.arange(16).reshape(4, 4)
    ds = xr.Dataset(
        {
            "LATIXY": (("lsmlat", "lsmlon"), lat2d),
            "LONGXY": (("lsmlat", "lsmlon"), lon2d),
            "PCT_NAT_PFT": (("time", "natpft", "lsmlat", "lsmlon"), pft),
            "PCT_CROP": (
                ("time", "lsmlat", "lsmlon"),
                np.arange(32, dtype=float).reshape(2, 4, 4),
            ),
            "LANDMASK": (("lsmlat", "lsmlon"), np.ones((4, 4), dtype=np.int32)),
            "YEAR": (("time",), np.array([2000, 2001], dtype=np.int32)),
        }
    )
    path = tmp_path / "landuse.nc"
    ds.to_netcdf(path)
    return path


def _targets():
    return gpd.GeoDataFrame(
        {"gid": ["z1", "z2"]},
        geometry=[box(-150.5, 67.5, -150.0, 68.0), box(-149.5, 68.0, -149.0, 69.0)],
        crs="EPSG:4326",
    )


def test_nearest_sampling(landuse_nc, tmp_path):
    df = pd.DataFrame(
        {"gid": ["p1", "p2"], "lat": [67.3, 68.7], "lon": [-150.7, -149.2]}
    )
    out, cells = sample_landuse_timeseries(landuse_nc, df, tmp_path / "near.nc")

    assert out == tmp_path / "near.nc"
    assert cells["i_lat"].tolist() == [0, 3]
    assert cells["i_lon"].tolist() == [0, 3]
    np.testing.assert_allclose(cells["lon_normalized"], [209.3, 210.8])
    np.testing.assert_allclose(cells["weight"], [1.0, 1.0])

    with xr.open_dataset(out) as ds:
        assert ds["PCT_NAT_PFT"].dims == ("time", "natpft", "lsmlat", "lsmlon")
        np.testing.assert_allclose(
            ds["PCT_NAT_PFT"].values[:, :, :, 0],
            [[[10, 25], [20, 35], [30, 45]], [[11, 26], [21, 36], [31, 46]]],
        )
        np.testing.assert_allclose(ds["PCT_CROP"].values[..., 0], [[0, 15], [16, 31]])
        np.testing.assert_allclose(ds["LONGXY"].values[:, 0], [209.25, 210.75])
        assert ds["YEAR"].values.tolist() == [2000, 2001]
        assert ds.attrs == {"dapper_sampling_method": "nearest"}


def test_nearest_sampling_output_lon_wrap_and_attrs(landuse_nc, tmp_path):
    df = pd.DataFrame(
        {"gid": ["p1", "p2"], "lat": [67.3, 68.7], "lon": [-150.7, -149.2]}
    )
    out, _ = sample_landuse_timeseries(
        landuse_nc,
        df,
        tmp_path / "near.nc",
        output_lon_wrap="-180_180",
        append_attrs={"note": "x"},
        compress=False,
    )
    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["LONGXY"].values[:, 0], [-150.75, -149.25])
        assert ds.attrs["output_lon_wrap"] == "-180_180"
        assert ds.attrs["note"] == "x"


def test_zonal_sampling(landuse_nc, tmp_path):
    df = pd.DataFrame({"gid": ["z1", "z2"]})
    out, summary = sample_landuse_timeseries(
        landuse_nc, df, tmp_path / "zon.nc", sampling_method="zonal", targets=_targets()
    )

    assert summary["gid"].tolist() == ["z1", "z2"]
    assert summary["sample_ncells"].tolist() == [1, 4]
    np.testing.assert_allclose(
        summary["sample_area_total_m2"], [1.178661e09, 2.281917e09], rtol=1e-5
    )

    weights = pd.read_csv(tmp_path / "zon.nc.zonal_weights.csv")
    assert weights.columns.tolist() == [
        "gid",
        "i_lat",
        "i_lon",
        "intersect_area_m2",
        "weight",
    ]
    assert weights["gid"].tolist() == ["z1", "z2", "z2", "z2", "z2"]
    np.testing.assert_allclose(
        weights["weight"], [1.0, 0.000011, 0.505507, 0.000011, 0.494470], atol=1e-6
    )

    with xr.open_dataset(out) as ds:
        # PINS BUG B18: zonal output places lsmlat first instead of last-but-one.
        assert ds["PCT_NAT_PFT"].dims == ("lsmlat", "time", "natpft", "lsmlon")
        assert ds["PCT_CROP"].dims == ("lsmlat", "time", "lsmlon")
        np.testing.assert_allclose(
            ds["PCT_NAT_PFT"].values[..., 0],
            [
                [[20.0, 100 / 3, 140 / 3], [20.51282051, 100 / 3, 46.15384615]],
                [
                    [23.22555511, 100 / 3, 43.44111155],
                    [23.52303604, 100 / 3, 43.14363063],
                ],
            ],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            ds["PCT_CROP"].values[..., 0], [[5.0, 21.0], [12.97790336, 28.97790336]]
        )
        np.testing.assert_allclose(ds["LATIXY"].values[:, 0], [67.75, 68.5])
        np.testing.assert_allclose(ds["LONGXY"].values[:, 0], [-150.25, -149.25])
        assert ds["LANDMASK"].values[:, 0].tolist() == [1, 1]
        assert ds["sample_ncells"].values[:, 0].tolist() == [1, 4]
        assert ds.attrs["dapper_sampling_method"] == "zonal"
        assert ds.attrs["dapper_sampling_lon_wrap_native"] == "0_360"
        assert ds.attrs["dapper_sampling_equal_area_crs"].startswith(
            "+proj=laea +lat_0=67.75 +lon_0=209.75"
        )


def test_zonal_sampling_requires_targets(landuse_nc, tmp_path):
    df = pd.DataFrame({"gid": ["z1"]})
    with pytest.raises(ValueError, match="requires targets"):
        sample_landuse_timeseries(
            landuse_nc, df, tmp_path / "x.nc", sampling_method="zonal"
        )
    with pytest.raises(ValueError, match="Unknown sampling_method"):
        sample_landuse_timeseries(
            landuse_nc,
            df.assign(lat=1.0, lon=1.0),
            tmp_path / "y.nc",
            sampling_method="bad",
        )


def test_export_landuse_cellset(landuse_nc, tmp_path):
    dom = Domain.from_provided(_targets(), name="cs", mode="cellset")
    out = export_landuse_timeseries(dom, src_path=landuse_nc, out_dir=tmp_path / "exp")
    assert out == {"cs": tmp_path / "exp" / "landuse_timeseries.nc"}
    with xr.open_dataset(out["cs"]) as ds:
        np.testing.assert_allclose(ds["PCT_CROP"].values.ravel(), [5, 11, 21, 27])
    with pytest.raises(FileExistsError):
        export_landuse_timeseries(dom, src_path=landuse_nc, out_dir=tmp_path / "exp")


def test_export_landuse_sites_zonal(landuse_nc, tmp_path):
    dom = Domain.from_provided(
        _targets(), name="grp", mode="sites", cell_kind="as_provided"
    )
    out = export_landuse_timeseries(
        dom, src_path=landuse_nc, out_dir=tmp_path / "exp", sampling_method="zonal"
    )
    assert out == {
        "z1": tmp_path / "exp" / "z1" / "landuse_timeseries.nc",
        "z2": tmp_path / "exp" / "z2" / "landuse_timeseries.nc",
    }
    with xr.open_dataset(out["z1"]) as ds:
        np.testing.assert_allclose(ds["PCT_CROP"].values.ravel(), [5.0, 21.0])
    with xr.open_dataset(out["z2"]) as ds:
        # PINS BUG B5: the second site reuses the first site's zonal target.
        np.testing.assert_allclose(ds["PCT_CROP"].values.ravel(), [5.0, 21.0])
        np.testing.assert_allclose(ds["LATIXY"].values.ravel(), [67.75])
