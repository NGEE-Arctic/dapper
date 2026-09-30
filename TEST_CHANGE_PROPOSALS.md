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
