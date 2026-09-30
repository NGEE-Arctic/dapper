# Test change proposals

Status (2026-09-30):
- **T-1, T-2, T-7, T-8 and T-9 are approved and added** in commit `d38811c`, as `tests/test_characterize_*.py`.
- P-1, T-3, T-4, T-5 and T-6 are still pending.
- Existing test files are unchanged.

## Changes to existing tests

### P-1 `tests/test_cmip_utils.py::test_intake` depends on the network
- **Current**: opens the live Pangeo catalog URL over HTTP. It passes only with internet access and doesn't exercise any dapper code (it tests `intake`/`intake_esm` directly).
- **Proposed change**: gate it the same way as the live CDS test (`skipif` unless `DAPPER_RUN_NETWORK_TESTS=1`), or register a `network` marker. Alternatively, replace it with a test of `dapper.met.cmip_utils.open_cmip6_catalog` against a tiny local ESM catalog JSON.
- **Why**: offline and CI-sandbox runs give a false failure, and the test gives no coverage of `cmip_utils`, which is at 0%.

## New characterization tests (proposed; not added)

Purpose: pin current behavior of weakly covered code **before** refactoring it (plan steps 8b, 10, 12, 14, 16). They assert today's output, including known-buggy behavior, so the refactor can't change it silently. Each test that pins a logged bug says so, so the pin can be updated when that bug is fixed.
All tests would be offline, use `tmp_path`, and build tiny synthetic inputs.

| ID | Target (current coverage) | What the test pins | Unblocks plan step |
|---|---|---|---|
| T-1 | `domains/elm_domain.to_elm_domain_dataset` (0%) and `Domain.export_domain`, `from_elm_domain`, `from_file` (uncovered) | For 2 box cells: shapes/dims `(nj, ni, nv)`, `xc/yc`, the xv/yv corner order (pins **B4**), area in sr, frac/mask, global attrs, sites vs cellset output paths, and a `from_file` round trip via a GeoPackage in `tmp_path`. | 10 |
| T-2 | `landuse.sample_landuse_timeseries` / `export_landuse_timeseries` (0%) | A synthetic 4×4 `lsmlat/lsmlon` landuse file with `PCT_NAT_PFT(time,natpft,…)`. Nearest path: sampled values and `LONGXY` wrap. Zonal path: weights CSV contents, `sample_ncells`, `sample_area_total_m2`, fraction closure. Sites-mode zonal with 2 gids (pins **B5**). | 8b |
| T-3 | `met/adapters/fluxnet` (10%) | A small HH FLUXNET CSV (TIMESTAMP_START/END, TA_F, VPD_F, PA_F, WS_F, SW_IN_F, LW_IN_F, P_F, with -9999 gaps): `discover_files` year range and resolution, `preprocess_shard` columns, unit conversions (C→K, kPa→Pa, mm/step→mm/s), RH-from-VPD fill, QBOT, coalescing `UserWarning`, and the error on an all-NaN required var. | 14 |
| T-4 | `surf/validate.SurfaceValidator` (0%) and `SurfaceFile.validate` | Report rows (check IDs, severity, passed) for a minimal valid 1×1 file and one with known violations (PCT out of range, missing LATIXY, time≠12). | 8b, 12 |
| T-5 | `met/validation.make_quicklooks` (0%) | Run on the sites and cellset outputs produced by the existing ERA5 exporter fixture pattern: PNG files are created per gid, and mode detection works. Uses the matplotlib Agg backend. | 16 |
| T-6 | `topounit/topomake` pure helpers (0%) | `build_bins_for_source` with `strategy="fixed"` (numeric and aspect), `_combine_cartesian` / `_combine_hierarchical` with fake mask objects (a stub with `.And()`), and `max_topounits` truncation. Checks band names and meta dicts. No GEE calls. | 15 |
| T-7 | `elm/utils` numerics (40%) | `elm_var_packing_params` with preset ranges and with data, including NaN (pins **B3**); `compute_humidities` reference values (pins **B2**); `compute_specific_humidity_from_rh`. | 17 |
| T-8 | `surf/sfile.customize_surface`, `SurfaceFile.resize_dim`, `to_netcdf(encoding=...)` (uncovered) | Overwriting a float var, adding a registry var, the non-strict add path, the units-mismatch error, the int-var error (pins **B15**), resize truncate/pad, and attr merge on `to_netcdf`. | 6, 12 |
| T-9 | `geo/sampling.write_netcdf`, `infer_grid_metadata`, `points_to_nearest_cells` (uncovered) | Attr precedence (`append_attrs` > source > `dapper_attrs`), the `dapper_created_utc` format (`YYYY-MM-DDTHH:MM:SS.ffffffZ`), compression encoding skips strings, and grid metadata keys. | 6, 8a |

Recommendation: approve at least **T-1, T-2, T-7, T-8, T-9**. These guard the medium-risk steps that I would otherwise have to pause on. T-3, T-5 and T-6 matter only if you want steps 14–16 done.

## Regression tests for remaining bug fixes (proposed; not added)

These bugs (see `REFACTOR_NOTES.md`) have no test covering them yet, so each fix waits for its regression test to be approved. All tests would be offline; the Earth Engine ones use a small fake `ee` object patched into the module.

| ID | Bug | Test | Fix it guards |
|---|---|---|---|
| R-1 | B1 | Run a sites-mode ERA5 export (same synthetic CSV pattern as `test_met_temporal`) and assert the `TBOT` variable has `long_name == "temperature at the lowest atm level (TBOT)"`. | Read the `"descriptions"` key in `Exporter.__init__`. |
| R-2 | B6 | `sample_image_over_polygons(gdf, image, geometry_id_field="gid")` with `parse_geometry_objects`, `ensure_pixel_centers_within_geometries` and `try_to_download_featurecollection` monkeypatched. Assert the result keeps `gid` and gains the sampled column. | Only drop `gid` after the merge when `geometry_id_field != "gid"`. |
| R-3 | B7 | `validate_bands(["b1"], gee_ic="X/Y")` with a fake `ee.ImageCollection` whose first image reports `["b1", "b2"]`. It should pass; `["zz"]` should raise `NameError`. | Read `available_bands` from `gee_ic`. |
| R-4 | B8 | `_compute_equalwidth_edges` with a fake image whose `reduceRegion(...).getInfo()` returns `{"v_min": 0, "v_max": 10}`. Expect `[0, 2.5, 5, 7.5, 10]` for `n_bins=4`. | `getInfo()` the stats before reading the keys. |
| R-5 | B9 | `prepare_for_plot` on a 1°×1° box at 68–69°N gives `area_km2` ≈ 4,640 km² (±1%). EPSG:3857 currently gives about 33,000 km². | Compute the area in a data-centered LAEA CRS. |
| R-6 | B12 | `_detect_mode` on a directory holding one NetCDF whose data var has non-standard dims but a global `domain_mode="sites"` returns `"sites"`. | Check `domain_mode` (then legacy `export_mode`). |
| R-7 | B14 | `Exporter._resolve_pack_scope("sites", "global")` raises `ValueError`, while `None`, `"per-site"`, `"per_site"`, `"site"` and `"local"` return `"per-site"`. **This changes behavior**: bad values currently pass silently. | Validate `pack_scope` in sites mode. |
| R-8 | B16 | After `SurfaceValidator().validate(path)`, reopening `path` with `netCDF4.Dataset(path, "w")` succeeds (HDF5 refuses to reopen a file that is still open). | `with xr.open_dataset(...)` in `validate`. |
