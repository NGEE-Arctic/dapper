"""Characterization tests for surface-file customization and editing (T-8).

These pin current behavior ahead of refactoring. Assertions marked
``Regression for Bn`` cover bugs fixed after the refactor (see
REFACTOR_NOTES.md).
"""

import re
import warnings

import numpy as np
import pytest
import xarray as xr

from dapper.surf.sfile import CustomizeError, SurfaceFile, customize_surface


@pytest.fixture
def surf_nc(tmp_path):
    ds = xr.Dataset(
        {
            "PCT_SAND": (
                ("nlevsoi", "lsmlat", "lsmlon"),
                np.full((3, 1, 1), 40.0, dtype=np.float32),
                {"units": "percent"},
            ),
            "SLOPE": (("lsmlat", "lsmlon"), np.array([[2.0]]), {"units": "degrees"}),
            "URBAN_REGION_ID": (
                ("lsmlat", "lsmlon"),
                np.array([[3]], dtype=np.int32),
                {"units": "unitless"},
            ),
        }
    )
    path = tmp_path / "in.nc"
    ds.to_netcdf(path)
    return path


def _read(path, var):
    with xr.open_dataset(path) as ds:
        da = ds[var].load()
    return da


def test_overwrite_scalar_default_output_path(surf_nc):
    out, report = customize_surface(surf_nc, {"PCT_SAND": 55.0})
    assert out == str(surf_nc.with_name("in_custom.nc"))
    assert report is None
    da = _read(out, "PCT_SAND")
    assert da.values.ravel().tolist() == [55.0, 55.0, 55.0]
    assert str(da.dtype) == "float32"
    assert da.attrs == {"units": "percent"}


def test_overwrite_with_1d_array_and_bad_shape(surf_nc, tmp_path):
    out, _ = customize_surface(
        surf_nc, {"PCT_SAND": np.array([1.0, 2.0, 3.0])}, tmp_path / "o.nc"
    )
    assert _read(out, "PCT_SAND").values.ravel().tolist() == [1.0, 2.0, 3.0]

    with pytest.raises(CustomizeError, match=re.escape("ndarray shape (2,)")):
        customize_surface(
            surf_nc, {"PCT_SAND": np.array([1.0, 2.0])}, tmp_path / "b.nc"
        )


def test_units_policy(surf_nc, tmp_path):
    ds = xr.open_dataset(surf_nc).load()
    ds["SLOPE"].attrs["units"] = "deg"
    mismatched = tmp_path / "mismatch.nc"
    ds.to_netcdf(mismatched)

    with pytest.raises(CustomizeError, match="units mismatch for SLOPE"):
        customize_surface(mismatched, {"SLOPE": 3.0}, tmp_path / "a.nc")

    # Regression for B15: "warn" warns and proceeds; "ignore" proceeds silently.
    with pytest.warns(UserWarning, match="units mismatch for SLOPE"):
        out, _ = customize_surface(
            mismatched, {"SLOPE": 3.0}, tmp_path / "b.nc", units_policy="warn"
        )
    da = _read(out, "SLOPE")
    assert da.values.ravel().tolist() == [3.0]
    assert da.attrs == {"units": "deg"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        customize_surface(
            mismatched, {"SLOPE": 4.0}, tmp_path / "c.nc", units_policy="ignore"
        )


def test_integer_variable_overwrite_keeps_dtype(surf_nc, tmp_path):
    # Regression for B15: existing int vars keep their dtype instead of failing
    # against the registry's default float32.
    out, _ = customize_surface(surf_nc, {"URBAN_REGION_ID": 5}, tmp_path / "o.nc")
    da = _read(out, "URBAN_REGION_ID")
    assert da.values.ravel().tolist() == [5]
    assert str(da.dtype) == "int32"

    with pytest.raises(CustomizeError, match="int/float switch"):
        customize_surface(
            surf_nc,
            {"URBAN_REGION_ID": {"value": 5, "dtype": "float32"}},
            tmp_path / "p.nc",
        )


def test_add_variables(surf_nc, tmp_path):
    with pytest.raises(CustomizeError, match="missing required dim for 'FMAX'"):
        customize_surface(surf_nc, {"FMAX": 0.3}, tmp_path / "a.nc")
    with pytest.raises(CustomizeError, match="unknown variable 'NOTREG'"):
        customize_surface(surf_nc, {"NOTREG": 1.0}, tmp_path / "b.nc")
    with pytest.raises(CustomizeError, match="set allow_add=True"):
        customize_surface(surf_nc, {"FMAX": 0.3}, tmp_path / "c.nc", allow_add=False)

    spec = {
        "value": 2.0,
        "dims": ["lsmlat", "lsmlon"],
        "dtype": "float64",
        "units": "m",
    }
    out, _ = customize_surface(
        surf_nc, {"NOTREG": spec}, tmp_path / "d.nc", strict_registry=False
    )
    da = _read(out, "NOTREG")
    assert da.values.ravel().tolist() == [2.0]
    assert str(da.dtype) == "float64"
    assert da.attrs == {"units": "m"}

    with pytest.raises(CustomizeError, match="requires spec dict"):
        customize_surface(
            surf_nc, {"NOTREG": 2.0}, tmp_path / "e.nc", strict_registry=False
        )


def test_resize_dim(surf_nc):
    sf = SurfaceFile.from_netcdf(surf_nc)
    with pytest.raises(KeyError, match="nope"):
        sf.resize_dim("nope", 3)
    sf.resize_dim("nlevsoi", 3)  # no-op
    assert sf.ds.sizes["nlevsoi"] == 3

    # Regression for B19: resizing a dim used by a data variable works.
    sf.resize_dim("nlevsoi", 5)
    np.testing.assert_array_equal(
        sf.ds["PCT_SAND"].values.ravel(), [40.0, 40.0, 40.0, np.nan, np.nan]
    )
    assert sf.ds["PCT_SAND"].dtype == np.float32
    assert sf.ds["PCT_SAND"].attrs == {"units": "percent"}
    assert list(sf.ds.data_vars) == ["PCT_SAND", "SLOPE", "URBAN_REGION_ID"]
    assert sf.ds["nlevsoi"].values.tolist() == [0, 1, 2, 3, 4]

    sf.resize_dim("nlevsoi", 2)
    assert sf.ds["PCT_SAND"].values.ravel().tolist() == [40.0, 40.0]
    assert sf.ds.sizes["nlevsoi"] == 2


def test_small_editing_helpers(surf_nc):
    sf = SurfaceFile.from_netcdf(surf_nc)
    sf.set_scalar("nlev", 2)
    sf.set_global_attrs(a=1)
    sf.set_global_attrs()
    sf.drop_params(["SLOPE", "not_present"])
    sf.drop_params("URBAN_REGION_ID")
    assert set(sf.ds.data_vars) == {"PCT_SAND", "nlev"}
    assert sf.ds.attrs == {"a": 1}
    assert sf.basic_registry_check() == {"known": {"PCT_SAND"}, "unknown": {"nlev"}}


def test_to_netcdf_attr_merge_and_overwrite(surf_nc, tmp_path):
    sf = SurfaceFile.from_netcdf(surf_nc)
    sf.set_global_attrs(a=1)

    path = sf.to_netcdf(
        tmp_path / "w.nc",
        encoding={"PCT_SAND": {"zlib": True}},
        dapper_attrs={"a": 2, "b": 3},
        append_attrs={"c": 4},
    )
    assert path == str(tmp_path / "w.nc")
    with xr.open_dataset(path) as ds:
        attrs = dict(ds.attrs)
    assert list(attrs) == ["a", "b", "dapper_created_utc", "c"]
    assert (attrs["a"], attrs["b"], attrs["c"]) == (1, 3, 4)
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", attrs["dapper_created_utc"]
    )

    with pytest.raises(FileExistsError):
        sf.to_netcdf(tmp_path / "w.nc")

    path2 = sf.to_netcdf(tmp_path / "w2.nc", add_created_utc=False)
    with xr.open_dataset(path2) as ds:
        assert dict(ds.attrs) == {"a": 1}
        assert str(ds["PCT_SAND"].dtype) == "float32"
