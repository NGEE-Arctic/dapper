import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pytest
from shapely.geometry import box

from dapper.surf.sfile import SurfaceFile
import dapper.surf.surface_var_specs as svs
from dapper.utils import zonal


class FakeDomain:
    """
    Minimal Domain stub for SurfaceFile.from_domain zonal path.
    """
    mode = "cellset"

    def __init__(self, cells_gdf: gpd.GeoDataFrame):
        self.cells = cells_gdf

    def ensure_cells_lon_lat(self):
        # add representative lon/lat if missing
        if "lon" not in self.cells.columns or "lat" not in self.cells.columns:
            reps = self.cells.geometry.representative_point()
            self.cells = self.cells.copy()
            self.cells["lon"] = reps.x.values
            self.cells["lat"] = reps.y.values
        return self

    def to_df_loc(self):
        self.ensure_cells_lon_lat()
        df = pd.DataFrame(
            {
                "gid": self.cells["gid"].astype(str).values,
                "lon": self.cells["lon"].astype(float).values,
                "lat": self.cells["lat"].astype(float).values,
                "weight": np.ones(len(self.cells), dtype=float),
            }
        )
        return df

    def has_topounits(self):
        return False


def _write_ds(tmp_path, ds: xr.Dataset, name: str) -> str:
    path = tmp_path / name
    ds.to_netcdf(path)
    return str(path)


def _make_small_rect_grid_ds(
    *,
    lon_centers,
    lat_centers,
    floatvar_vals,
    catvar_vals,
    mask_vals,
    area_units="m2",
) -> xr.Dataset:
    lat_dim = "lsmlat"
    lon_dim = "lsmlon"

    ds = xr.Dataset(
        coords={
            lat_dim: np.asarray(lat_centers, dtype=float),
            lon_dim: np.asarray(lon_centers, dtype=float),
        }
    )

    ds["FLOATVAR"] = xr.DataArray(np.asarray(floatvar_vals, dtype=float), dims=(lat_dim, lon_dim))
    ds["CATVAR"] = xr.DataArray(np.asarray(catvar_vals, dtype=np.int32), dims=(lat_dim, lon_dim))
    ds["TEST_MASK"] = xr.DataArray(np.asarray(mask_vals, dtype=np.int32), dims=(lat_dim, lon_dim))

    # Put "wrong" values in derived vars in source file; zonal surface path must override
    ds["LATIXY"] = xr.DataArray(np.full((len(lat_centers), len(lon_centers)), 999.0), dims=(lat_dim, lon_dim))
    ds["LONGXY"] = xr.DataArray(np.full((len(lat_centers), len(lon_centers)), 999.0), dims=(lat_dim, lon_dim))
    ds["AREA"] = xr.DataArray(np.full((len(lat_centers), len(lon_centers)), 999.0), dims=(lat_dim, lon_dim))
    ds["AREA"].attrs["units"] = area_units

    return ds


@pytest.fixture
def patch_surface_registry(monkeypatch):
    """
    Patch SURFACE_VAR_SPECS so our synthetic vars have explicit aggregation policies,
    and derived vars are marked derived.
    """
    # Keep originals and restore after
    original = dict(svs.SURFACE_VAR_SPECS)

    # Ensure derived vars exist and are marked derived (they should already be, but be explicit)
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "LATIXY", {"agg": "derived"})
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "LONGXY", {"agg": "derived"})
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "AREA", {"agg": "derived"})

    # Our synthetic test vars: force behaviors from registry (this is what we’re validating)
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "FLOATVAR", {"agg": "max"})
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "CATVAR", {"agg": "wmode"})
    monkeypatch.setitem(svs.SURFACE_VAR_SPECS, "TEST_MASK", {"agg": "wmean_threshold"})

    yield

    # restore registry
    svs.SURFACE_VAR_SPECS.clear()
    svs.SURFACE_VAR_SPECS.update(original)


def test_surface_zonal_registry_agg_and_derived_vars(tmp_path, patch_surface_registry):
    """
    Confirms:
      (1) registry agg is applied (FLOATVAR uses max, not mean)
      (2) derived vars LATIXY/LONGXY/AREA are injected from Domain (not sampled from source)
    """
    # 2x2 grid with lon cell edges at [0,1] and [1,2]
    lon_centers = [0.5, 1.5]
    lat_centers = [0.5, 1.5]

    # Only first lat row matters (polygon intersects y in [0,1])
    float_vals = [
        [0.0, 10.0],
        [123.0, 456.0],
    ]
    cat_vals = [
        [1, 2],
        [9, 9],
    ]
    mask_vals = [
        [0, 1],
        [1, 1],
    ]

    ds = _make_small_rect_grid_ds(
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        floatvar_vals=float_vals,
        catvar_vals=cat_vals,
        mask_vals=mask_vals,
        area_units="m2",
    )
    src_path = _write_ds(tmp_path, ds, "surf_src_small.nc")

    # Polygon overlaps first row; overlaps lon cell1 fully and lon cell2 partially (0.6 of width)
    # -> weights approx: cell1 1.0, cell2 0.6  => normalized: 0.625 / 0.375
    poly = box(0.0, 0.0, 1.6, 1.0)
    cells = gpd.GeoDataFrame({"gid": ["g1"], "geometry": [poly]}, crs="EPSG:4326")
    dom = FakeDomain(cells).ensure_cells_lon_lat()

    sf = SurfaceFile.from_domain(
        dom,
        src_path=src_path,
        sampling_method="zonal",
        lon_wrap="auto",
        include={"FLOATVAR", "CATVAR", "TEST_MASK", "LATIXY", "LONGXY", "AREA"},
    )
    out = sf.ds

    # Output is (lsmlat=Ntargets, lsmlon=1)
    lat_dim = "lsmlat"
    lon_dim = "lsmlon"

    # (1) FLOATVAR uses registry max over contributing cells: max(0,10)=10 (not mean ~3.75)
    float_out = float(out["FLOATVAR"].isel({lat_dim: 0, lon_dim: 0}).item())
    assert float_out == 10.0

    # CATVAR wmode with weights favoring cell1: mode should be 1
    cat_out = int(out["CATVAR"].isel({lat_dim: 0, lon_dim: 0}).item())
    assert cat_out == 1

    # TEST_MASK wmean_threshold: wmean = 0*0.625 + 1*0.375 = 0.375 -> threshold 0.5 => 0
    mask_out = int(out["TEST_MASK"].isel({lat_dim: 0, lon_dim: 0}).item())
    assert mask_out == 0

    # (2) Derived vars must NOT equal the 999 source values; must reflect domain rep point and polygon area
    latxy = float(out["LATIXY"].isel({lat_dim: 0, lon_dim: 0}).item())
    longxy = float(out["LONGXY"].isel({lat_dim: 0, lon_dim: 0}).item())
    assert abs(latxy - dom.cells["lat"].iloc[0]) < 1e-4
    assert abs(longxy - dom.cells["lon"].iloc[0]) < 1e-4

    # AREA derived from polygon area in equal-area CRS (units m2 in this test)
    ea = zonal.laea_crs_for_targets(cells[["gid", "geometry"]])
    expected_area_m2 = float(cells.to_crs(ea).geometry.area.iloc[0])
    area_out = float(out["AREA"].isel({lat_dim: 0, lon_dim: 0}).item())
    assert abs(area_out - expected_area_m2) / expected_area_m2 < 1e-6


def test_surface_zonal_lon_wrap_auto_handles_negative_lon_polygon(tmp_path, patch_surface_registry):
    """
    Confirms (3): lon wrapping is passed through and works when:
      - dataset lon centers are 0..360
      - polygon longitudes are -180..180
    """
    lon_centers = [350.5, 351.5]  # implies cell edges [350,351] and [351,352]
    lat_centers = [0.5, 1.5]

    # Make FLOATVAR distinct so we know we intersect the right cell
    float_vals = [
        [42.0, 0.0],
        [999.0, 999.0],
    ]
    cat_vals = [
        [1, 2],
        [9, 9],
    ]
    mask_vals = [
        [1, 1],
        [1, 1],
    ]

    ds = _make_small_rect_grid_ds(
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        floatvar_vals=float_vals,
        catvar_vals=cat_vals,
        mask_vals=mask_vals,
        area_units="m2",
    )
    src_path = _write_ds(tmp_path, ds, "surf_src_0360.nc")

    # Polygon specified with negative lon; after wrap it should overlap lon ~350..350.9 (first cell only)
    poly = box(-10.9, 0.0, -9.1, 1.0)
    cells = gpd.GeoDataFrame({"gid": ["g1"], "geometry": [poly]}, crs="EPSG:4326")
    dom = FakeDomain(cells).ensure_cells_lon_lat()

    sf = SurfaceFile.from_domain(
        dom,
        src_path=src_path,
        sampling_method="zonal",
        lon_wrap="auto",
        include={"FLOATVAR", "LATIXY", "LONGXY", "AREA"},
    )
    out = sf.ds

    lat_dim = "lsmlat"
    lon_dim = "lsmlon"
    float_out = float(out["FLOATVAR"].isel({lat_dim: 0, lon_dim: 0}).item())

    # Should successfully intersect and produce something close to 42 (not error / not 0 from wrong side)
    assert float_out > 40.0


def test_zonal_weights_reuse_does_not_recompute(tmp_path):
    """
    Sanity check that sample_gridded_dataset_polygons(weights=...) truly reuses weights
    (this is the foundation for surface caching).
    """
    lon_centers = [0.5, 1.5]
    lat_centers = [0.5, 1.5]

    ds = xr.Dataset(
        coords={"lsmlat": np.asarray(lat_centers, float), "lsmlon": np.asarray(lon_centers, float)}
    )
    ds["X"] = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lsmlat", "lsmlon"))

    poly = box(0.0, 0.0, 2.0, 1.0)
    targets = gpd.GeoDataFrame({"gid": ["g1"], "geometry": [poly]}, crs="EPSG:4326")

    zw = zonal.intersect_weights_rectilinear(ds, targets)

    # If this gets called, the test should fail
    def _boom(*args, **kwargs):
        raise RuntimeError("intersect_weights_rectilinear should NOT be called when weights are provided")

    import dapper.utils.zonal as zonal_mod
    zonal_mod.intersect_weights_rectilinear = _boom  # blunt monkeypatch without pytest fixture

    out = zonal.sample_gridded_dataset_polygons(
        ds,
        targets,
        vars_include=["X"],
        agg_policy={"X": "wmean"},
        weights=zw,
    )
    assert "X" in out
