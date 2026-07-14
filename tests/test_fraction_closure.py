import numpy as np
import xarray as xr

from dapper.surf.sfile import write_surface_nc
from dapper.surf.fraction_closure import normalize_fraction_closure


def test_write_surface_nc_enforces_fraction_closure(tmp_path):
    ds = xr.Dataset(
        coords={
            "natpft": np.arange(3, dtype=np.int32),
            "cft": np.arange(2, dtype=np.int32),
            "numurbl": np.arange(2, dtype=np.int32),
            "lsmlat": np.arange(2, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    ds["PCT_NATVEG"] = xr.DataArray(np.array([[40.0], [60.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_NAT_PFT"] = xr.DataArray(
        np.array(
            [
                [[10.0], [20.0]],
                [[20.0], [20.0]],
                [[9.5], [19.2]],
            ]
        ),
        dims=("natpft", "lsmlat", "lsmlon"),
    )

    ds["PCT_CROP"] = xr.DataArray(np.array([[30.0], [20.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_CFT"] = xr.DataArray(
        np.array(
            [
                [[10.0], [10.0]],
                [[18.0], [9.0]],
            ]
        ),
        dims=("cft", "lsmlat", "lsmlon"),
    )

    ds["PCT_WETLAND"] = xr.DataArray(np.array([[5.0], [5.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_LAKE"] = xr.DataArray(np.array([[20.0], [10.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_GLACIER"] = xr.DataArray(np.array([[5.0], [5.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_URBAN"] = xr.DataArray(
        np.array(
            [
                [[0.5], [0.6]],
                [[0.5], [0.4]],
            ]
        ),
        dims=("numurbl", "lsmlat", "lsmlon"),
    )

    ds["FSURF"] = xr.DataArray(np.array([[0.2], [0.8]]), dims=("lsmlat", "lsmlon"))
    ds["FGRD"] = xr.DataArray(np.array([[0.6], [0.1]]), dims=("lsmlat", "lsmlon"))

    out_path = tmp_path / "surf_closure.nc"
    write_surface_nc(ds, str(out_path))

    out = xr.open_dataset(out_path)

    pft_sum = out["PCT_NAT_PFT"].sum(dim="natpft")
    np.testing.assert_allclose(pft_sum.values, 100.0, atol=1e-5, rtol=0.0)

    cft_sum = out["PCT_CFT"].sum(dim="cft")
    np.testing.assert_allclose(cft_sum.values, out["PCT_CROP"].values, atol=1e-5, rtol=0.0)

    urban_sum = out["PCT_URBAN"].sum(dim="numurbl")
    landunit_total = (
        out["PCT_NATVEG"]
        + out["PCT_CROP"]
        + out["PCT_WETLAND"]
        + out["PCT_LAKE"]
        + out["PCT_GLACIER"]
        + urban_sum
    )
    np.testing.assert_allclose(landunit_total.values, 100.0, atol=1e-5, rtol=0.0)

    irrigation_total = out["FSURF"] + out["FGRD"]
    np.testing.assert_allclose(irrigation_total.values, 1.0, atol=1e-6, rtol=0.0)

    out.close()


def test_landuse_normalizer_enforces_fraction_closure():
    ds = xr.Dataset(
        coords={
            "natpft": np.arange(2, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(2, dtype=np.int32),
        }
    )

    ds["PCT_NATVEG"] = xr.DataArray(np.array([[50.0, 60.0]]), dims=("lsmlat", "lsmlon"))
    ds["PCT_NAT_PFT"] = xr.DataArray(
        np.array(
            [
                [[20.0, 30.0]],
                [[29.0, 31.0]],
            ]
        ),
        dims=("natpft", "lsmlat", "lsmlon"),
    )

    fixed = normalize_fraction_closure(ds)
    pft_sum = fixed["PCT_NAT_PFT"].sum(dim="natpft")
    np.testing.assert_allclose(pft_sum.values, 100.0, atol=1e-10, rtol=0.0)


def test_write_surface_nc_natural_patch_weights_near_exact_one(tmp_path):
    ds = xr.Dataset(
        coords={
            "natpft": np.arange(17, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    # Build a distribution that typically accumulates roundoff when stored as float32.
    vals = np.array([5.0] * 16 + [20.0], dtype=np.float64).reshape(17, 1, 1)
    vals[3, 0, 0] = 4.9999997
    vals[7, 0, 0] = 5.0000003
    ds["PCT_NAT_PFT"] = xr.DataArray(vals, dims=("natpft", "lsmlat", "lsmlon"))

    out_path = tmp_path / "surf_nat_patch_precision.nc"
    write_surface_nc(ds, str(out_path))
    out = xr.open_dataset(out_path)

    pft_sum_pct = float(out["PCT_NAT_PFT"].sum(dim="natpft").isel(lsmlat=0, lsmlon=0).item())
    wt_nat_patch_sum = pft_sum_pct / 100.0

    assert abs(wt_nat_patch_sum - 1.0) <= 1e-12
    out.close()


def test_write_surface_nc_landunit_weights_near_exact_one(tmp_path):
    ds = xr.Dataset(
        coords={
            "numurbl": np.arange(2, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    ds["PCT_NATVEG"] = xr.DataArray(np.array([[65.432198765]], dtype=np.float64), dims=("lsmlat", "lsmlon"))
    ds["PCT_CROP"] = xr.DataArray(np.array([[12.345678901]], dtype=np.float64), dims=("lsmlat", "lsmlon"))
    ds["PCT_WETLAND"] = xr.DataArray(np.array([[4.210987654]], dtype=np.float64), dims=("lsmlat", "lsmlon"))
    ds["PCT_LAKE"] = xr.DataArray(np.array([[8.765432109]], dtype=np.float64), dims=("lsmlat", "lsmlon"))
    ds["PCT_GLACIER"] = xr.DataArray(np.array([[7.777777777]], dtype=np.float64), dims=("lsmlat", "lsmlon"))
    ds["PCT_URBAN"] = xr.DataArray(
        np.array([[[0.700000001]], [[0.768000002]]], dtype=np.float64),
        dims=("numurbl", "lsmlat", "lsmlon"),
    )

    out_path = tmp_path / "surf_lunit_precision.nc"
    write_surface_nc(ds, str(out_path))
    out = xr.open_dataset(out_path)

    urban_sum = out["PCT_URBAN"].sum(dim="numurbl")
    pct_lunit = (
        out["PCT_NATVEG"]
        + out["PCT_CROP"]
        + out["PCT_WETLAND"]
        + out["PCT_LAKE"]
        + out["PCT_GLACIER"]
        + urban_sum
    )
    wt_lunit_sum = float(pct_lunit.isel(lsmlat=0, lsmlon=0).item()) / 100.0

    assert abs(wt_lunit_sum - 1.0) <= 1e-12
    out.close()


def test_write_surface_nc_topounit_pft_closure_exact(tmp_path):
    ds = xr.Dataset(
        coords={
            "topounit": np.arange(2, dtype=np.int32),
            "natpft": np.arange(4, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    # Two topounits with tiny partition drift in opposite directions.
    vals = np.array(
        [
            [[[25.0]], [[25.0]]],
            [[[25.0]], [[25.0]]],
            [[[25.0]], [[25.0]]],
            [[[24.9999998]], [[25.0000002]]],
        ],
        dtype=np.float64,
    )
    ds["PCT_NAT_PFT"] = xr.DataArray(vals, dims=("natpft", "topounit", "lsmlat", "lsmlon"))

    out_path = tmp_path / "surf_topounit_natpft_precision.nc"
    write_surface_nc(ds, str(out_path))

    out = xr.open_dataset(out_path)
    pft_sum = out["PCT_NAT_PFT"].sum(dim="natpft")
    np.testing.assert_allclose(pft_sum.values, 100.0, atol=1e-12, rtol=0.0)
    out.close()


def test_topounit_landunit_closure_exact():
    ds = xr.Dataset(
        coords={
            "topounit": np.arange(2, dtype=np.int32),
            "numurbl": np.arange(2, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    ds["PCT_NATVEG"] = xr.DataArray(np.array([[[50.0]], [[40.0]]]), dims=("topounit", "lsmlat", "lsmlon"))
    ds["PCT_CROP"] = xr.DataArray(np.array([[[20.0]], [[30.0]]]), dims=("topounit", "lsmlat", "lsmlon"))
    ds["PCT_WETLAND"] = xr.DataArray(np.array([[[10.0]], [[10.0]]]), dims=("topounit", "lsmlat", "lsmlon"))
    ds["PCT_LAKE"] = xr.DataArray(np.array([[[9.0]], [[8.0]]]), dims=("topounit", "lsmlat", "lsmlon"))
    ds["PCT_GLACIER"] = xr.DataArray(np.array([[[9.0]], [[10.0]]]), dims=("topounit", "lsmlat", "lsmlon"))
    ds["PCT_URBAN"] = xr.DataArray(
        np.array(
            [
                [[[1.0]], [[1.1]]],
                [[[1.0]], [[0.9]]],
            ],
            dtype=np.float64,
        ),
        dims=("numurbl", "topounit", "lsmlat", "lsmlon"),
    )

    fixed = normalize_fraction_closure(ds)
    urban_sum = fixed["PCT_URBAN"].sum(dim="numurbl")
    total = (
        fixed["PCT_NATVEG"]
        + fixed["PCT_CROP"]
        + fixed["PCT_WETLAND"]
        + fixed["PCT_LAKE"]
        + fixed["PCT_GLACIER"]
        + urban_sum
    )
    np.testing.assert_allclose(total.values, 100.0, atol=1e-12, rtol=0.0)


def test_topounit_weight_aliases_close_to_100():
    ds = xr.Dataset(
        coords={
            "topounit": np.arange(3, dtype=np.int32),
            "lsmlat": np.arange(1, dtype=np.int32),
            "lsmlon": np.arange(1, dtype=np.int32),
        }
    )

    ds["PCT_TOPUNIT"] = xr.DataArray(
        np.array([[[40.0]], [[30.0]], [[29.999999]]], dtype=np.float64),
        dims=("topounit", "lsmlat", "lsmlon"),
    )

    fixed = normalize_fraction_closure(ds)
    tot = fixed["PCT_TOPUNIT"].sum(dim="topounit")
    np.testing.assert_allclose(tot.values, 100.0, atol=1e-12, rtol=0.0)
