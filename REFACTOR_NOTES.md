# dapper refactor notes

Baseline commit: `8b43332c37dc30d122749fb85a2399517163f009`. Branch: `refactor/cleanup`.
Baseline numbers are in `REFACTOR_PLAN.md` §2.

## Bugs found

Found during the survey (B1–B17) and while writing characterization tests (B18–B19).
"Verified" means I reproduced it against the baseline code. "By inspection" means I confirmed it by reading the code.

### Status

| Status | Bugs | Notes |
|---|---|---|
| **Fixed** (separate commits, each with a regression test) | B2 `a9b84af`, B3 `4e7a21f`, B4 `6f1c10b`, B5 `643a699`, B15 `fc4e7a9`, B18 `5c7f515`, B19 `bcb81ba` | The pins in the approved characterization tests became the regression tests. |
| **Fixed by an approved breaking change** | B13 `9ed4a0a` | `Domain.make_topounits` now raises. |
| **Gone with deleted code** | B10 (`wind_direction`, step 4), B11 (`validate_met_vars`, §5.1) | – |
| **Awaiting regression-test approval** | B1, B6, B7, B8, B9, B12, B14, B16 | See R-1…R-8 in `TEST_CHANGE_PROPOSALS.md`. B14's fix changes behavior (it rejects invalid `pack_scope`). |
| **Design question, left as-is** | B17 | – |

Line numbers in the table below refer to the baseline commit.

| # | Location | Symptom | Suggested fix | Evidence |
|---|---|---|---|---|
| B1 | `met/exporter.py:224` | `self._elm_desc = elm_data_dicts().get("short_descriptions", {})`, but the dict's key is `"descriptions"`. So `_elm_desc` is always `{}` and **no MET variable ever gets a `long_name` attribute**. | Use the `"descriptions"` key, or read from a constant in `schemas.elm`. | Verified: `'short_descriptions' in elm_data_dicts()` → `False` |
| B2 | `elm/utils.py:115-122` (`compute_humidities`) | The saturation vapor pressure uses latent heat of vaporization for T ≥ 273.15 K and sublimation below. The actual vapor pressure uses the **opposite** branches (vaporization for T ≤ 273.15). Result: RH > 100% whenever Td ≈ T, and a biased RH/QBOT in general for the ERA5 path. | Use the same `temp >= 273.15` condition for both, or compute both from Td with a consistent phase choice. | Verified: `compute_humidities(T, T, 1e5)` → RH = 119.4% at 290 K, 116.7% at 260 K |
| B3 | `elm/utils.py:359-362` via `met/exporter.py:636-639` | In sites mode `pack_params` gets the raw series. `data.min()`/`.max()` propagate NaN, so `scale_factor` = NaN, and the exporter's fallback then packs with `scale_factor=1.0`. For small-magnitude vars like QBOT (~1e-3) or PRECTmms, int16 packing with scale 1.0 destroys the data. It triggers for any site series that still contains a NaN. The docstring also claims "robust quantiles" but the code uses plain min/max. | Use `np.nanmin`/`np.nanmax` (as the cellset global path already does). | Verified: `elm_var_packing_params("QBOT", np.array([1e-3, nan, 3e-3]))` → `(nan, nan)` |
| B4 | `domains/elm_domain.py:76-77` | Vertex arrays are written as `[minx,maxx,minx,maxx]` / `[miny,miny,maxy,maxy]` (a Z order), not counter-clockwise (ll, lr, ur, ul) as ELM/CIME domain files expect. `Domain.from_elm_domain` (`domain.py:303-304`) builds `Polygon(zip(xv, yv))` from those vertices, so reading back a dapper-written `domain.nc` gives self-intersecting, zero-area cell polygons. | Write `[minx,maxx,maxx,minx]` / `[miny,miny,maxy,maxy]`. | Verified: `Polygon([(0,0),(1,0),(0,1),(1,1)])` → `is_valid=False, area=0` |
| B5 | `landuse/landuse.py:320-321` (`export_landuse_timeseries`) | `kwargs.setdefault("targets", run_dom.cells[...])` mutates the shared `kwargs` dict inside the per-run loop. In sites mode with `sampling_method="zonal"`, **every site after the first reuses the first site's polygon** as its zonal target. | Build a per-iteration dict: `run_kwargs = {**kwargs}; run_kwargs.setdefault(...)`. | By inspection |
| B6 | `integrations/earthengine/gee_utils.py:904-909` (`sample_image_over_polygons`) | When `geometry_id_field == "gid"`, `merge(left_on="gid", right_on="gid")` yields a single `gid` column and the `.drop(columns=["gid"])` then **removes the id column from the result**. | Drop `gid` only when `geometry_id_field != "gid"`, or rename before the merge. | Verified with pandas: merged columns → `['x', 'v']` |
| B7 | `integrations/earthengine/gee_utils.py:209-216` (`validate_bands`) | For any `gee_ic` other than ERA5-Land, the else branch assigns `band_names` but then reads `available_bands`, which raises `UnboundLocalError`. It also queries a hardcoded ERA5 collection instead of `gee_ic`. It can't be reached from `sample_e5lh` today. | Assign `available_bands` from `ee.ImageCollection(gee_ic)`. | By inspection |
| B8 | `topounit/topomake.py:273-277` (`_compute_equalwidth_edges`) | `stats.get(...)` returns an `ee.ComputedObject`, never `None`, and `float(ComputedObject)` raises. So `strategy="equalwidth"` can't work. | `stats.getInfo()` first, then read the keys. | By inspection (needs GEE to run) |
| B9 | `topounit/topoplot.py:95,112` (`prepare_for_plot`) | `area_km2` is computed in EPSG:3857 (Web Mercator), which inflates area by about 1/cos²(lat). At Arctic latitudes (~68°N) that is about a 7× overestimate. | Use an equal-area CRS (e.g., LAEA centered on the data, as `geo/zonal.py` does) or geodesic area. | By inspection |
| B10 | `met/adapters/era5.py:221-224` | The wind-direction formula applies `-180` then `+180` sequentially, so values that were exactly 180° end back at 180 instead of 0/360. The column is dropped before output, so this has no effect today. | Use `(np.degrees(np.arctan2(u, v)) + 180) % 360`, or delete (see plan step 4). | By inspection |
| B11 | `elm/utils.py:84` | `print({'Negative values detected ...'})` prints a **set literal**, not the message. | Remove the braces. | By inspection |
| B12 | `met/validation.py:255` | Mode auto-detection reads a global attr `export_mode`, but the exporter writes `domain_mode` (`exporter.py:750`). Detection only works through the dimension heuristics fallback. | Read `domain_mode` (and keep `export_mode` for old files). | By inspection |
| B13 | `domains/domain.py:802-820` (`Domain.make_topounits`) | Catches `RuntimeError`, prints it, and returns `None`, despite the `-> Domain` annotation. Callers doing `dom = dom.make_topounits(...)` silently lose their Domain. | Let it raise (breaking change §5.3). | By inspection; mypy flags it |
| B14 | `met/exporter.py:706-722` (`_resolve_pack_scope`) | For sites mode, **any** `pack_scope` string, including `"global"`, silently becomes `"per-site"`. | Raise on unsupported values for sites mode. | By inspection |
| B15 | `surf/sfile.py:447-456` (`customize_surface`) | `units_policy="warn"` behaves exactly like `"ignore"` and never warns. Also, `REGISTRY` dtypes are always `"float32"`: `schema.pdef` is called without `dtype` and the specs carry no dtype. So overwriting an existing **integer** variable (e.g., `URBAN_REGION_ID`) through `customize_surface` raises `CustomizeError("int/float switch")`. | Emit `warnings.warn` for `"warn"`; carry dtypes into the specs, or skip the int/float check when the registry dtype is only a default. | By inspection |
| B16 | `surf/validate.py:87` | `SurfaceValidator.validate` opens the dataset and never closes it. On Windows this leaves a file lock and blocks overwriting the file right after validating it (e.g., in `SurfaceFile.export(validate=True)` loops). | Use `with xr.open_dataset(...) as ds:`. | By inspection |
| B18 | `geo/zonal.py:413-420` (`sample_gridded_dataset_polygons`) | Per-variable dims are reordered on each per-target slice **before** `xr.concat(..., dim=lat_dim)`. That concat prepends the new `lsmlat` dim, so zonal outputs come out as `(lsmlat, time, natpft, lsmlon)` instead of ELM's `(..., lsmlat, lsmlon)`. This affects zonal landuse and zonal surface sampling. | Reorder after the concat, as `sample_gridded_dataset_points` does. | Verified (pinned in T-2) |
| B19 | `surf/sfile.py:1210` (`SurfaceFile.resize_dim`) | `assign_coords` with the new-length coordinate runs **before** the data variables are resized. So resizing any dim that a data variable uses always raises `ValueError: conflicting sizes`, and the method never works for its intended purpose. | Build the resized variables first, then assign the coordinate (or `ds.isel`/`ds.pad`). | Verified (pinned in T-8) |
| B17 | `met/adapters/fluxnet.py:161` vs ERA5 | FLUXNET filters to `start_year..end_year` with no lookahead. ERA5 keeps the next Jan-1 00:00 for interval alignment. FLUXNET has no `INTERVAL_END_VARS`, so this is consistent today. Flagged only because TIMESTAMP_END-labelled fluxes are interval-end values too, so the same alignment question applies. | Decide whether FLUXNET fluxes should be relabelled to interval start like ERA5. | By inspection (design question) |

## Per-module change log

All changes are on `refactor/cleanup`. Refactor, breaking-change and bug-fix commits are kept separate.

**Repo-wide**
- ruff/mypy config added. `ruff format` applied (AST-identical; the commit is listed in `.git-blame-ignore-revs`).
- Imports sorted and moved to the top. Annotations modernized.
- 27 placeholder module docstrings replaced, and stale or conversational comments removed.

**Per module**
- **domains/domain**:
  - `SAMPLING_PROVENANCE_COLUMNS` is shared with the exporter.
  - Redundant CRS, `Path` and `str` conversions removed from `from_file`/`from_geometry`.
  - Return types added.
  - `make_topounits` raises instead of returning `None` (§5.3).
- **domains/elm_domain**: counter-clockwise cell corners (B4).
- **met/exporter**:
  - Removed the never-called `_pass1_to_parquet_raw` and `_zone_mappings_path`, unused private params, an unreachable mode check, and redundant pack-scope branches.
  - `domain.mode` is read directly; `filename_prefix` is initialized in `__init__`.
  - Import aliases renamed (`utils` → `fs`, `dt` → `temporal`).
  - Type hints added; the class docstring is rewritten to match the real API.
- **met/temporal**:
  - `is_feb29()` replaces six copies of the Feb-29 mask.
  - `create_dtime` reuses `_numeric_dtime`; one `_LINEAR_VARS`/`_FFILL_VARS` definition.
  - Old vs new outputs verified identical across 292 input combinations.
- **met/writers**: unreachable dtype alias table and no-op chunk branches removed (verified identical across 225 combinations).
- **met/adapters/era5**:
  - Discarded `wind_direction` computation removed.
  - Docstring now matches the real preprocessing.
  - `id_column_for_csv` deprecated (§5.7).
- **met/adapters/fluxnet**: uses `temporal.is_feb29`.
- **met/validation**:
  - Raw-mode quicklooks removed (§5.1).
  - No backend switch at import (§5.4).
  - Impossible netCDF4 `None` check removed.
- **met/cmip_utils**:
  - 77 unreachable lines removed.
  - `extract_vars_from_files`, `cftime_date` and `summarize_search` removed (§5.1).
- **elm/utils**:
  - `validate_met_vars` and `gen_zone_mappings` removed (§5.1), so the module no longer imports `Domain`.
  - `elm_data_dicts` builds its identical entries from the canonical constants.
  - B2 and B3 fixed.
- **geo/lonwrap, geo/zonal, geo/sampling, landuse**:
  - One `LonWrap` definition; `normalize_lons` replaces the hand-rolled loops.
  - `_median_step` variants merged; dead `_reduce_da` and unreachable guards removed.
  - B5 and B18 fixed.
- **io/attrs**:
  - New `merge_global_attrs` / `utc_timestamp` replace three copies of the attribute merge and the deprecated `utcnow`.
  - `apply_append_attrs` removed. `io/provenance` deleted and `io.fs.make_directory` removed (§5.1).
- **surf/sfile**:
  - Shared attribute merge; dead locals and the unused `_latlon_dim_names` removed; redundant checks simplified.
  - `build_surface_dataset*` removed (§5.1).
  - `engine=` deprecated (§5.7).
  - B15 and B19 fixed.
- **surf/schema**:
  - Module docstring moved to the top.
  - Unused export-policy, registration and validation helpers removed (§5.1).
- **surf/validate**: docstring corrected (V-105, plus V-108 and V-109).
- **integrations/earthengine/gee_utils**:
  - Unused `_ROOT_DIR`/`_DATA_DIR` removed.
  - `split_into_dfs`, `infer_id_field` and `featurecollection_to_df_loc` removed (§5.1).
  - Docstrings fixed.
- **`__init__`**: `__version__` comes from the package metadata (§5.5).
- **pyproject / environment.yml**:
  - Dependencies match the actual imports; `notebooks` and `plot` extras added (§5.6).
  - ruff and mypy config added.

## Remaining smells deliberately left alone

- **Broad `except` blocks** whose narrowing could change behavior: `BaseAdapter.pack_params`, the `cmip_utils` retry loop and parquet fallback, `met/validation` mode detection, `gee_utils.try_to_download_featurecollection`, and the lazy `ee` import proxies. Kept to preserve behavior.
- **`print` for progress and warnings** (about 60 calls): converting to logging was declined (decision 3).
- **Plan steps 13–16**, paused because they have almost no tests:
  - 13: share the Earth Engine proxy between `gee_utils` and `topomake`
  - 14: dedupe the fluxnet timestamp parsing
  - 15: dedupe the topomake combiner metadata
  - 16: dedupe the quicklook plotting
  - T-3, T-5 and T-6 would unblock them.
- **Step 18 (splitting `sfile.py` and `gee_utils.py`)**: skipped because the question wasn't answered.
- **Constants that disagree** (QBOT packing range 0.1 vs 0.04, DTBOT units `"unsure"`): preserved.
- **Other unanswered questions, current behavior kept**:
  - Two topounit-attach paths (§6 Q7).
  - No fraction closure on the nearest-landuse path (§6 Q8).
- **`surf/sample.SurfacePointSampler`**: a legacy sampler that parallels `geo.sampling`, kept because `surf.sample` is in the documented module list.
- **Remaining mypy errors** (88), mostly xarray `Hashable` vs `str` and `None`-initialized Exporter attributes. Fixing the latter means restructuring `Exporter.run` state.
- **One remaining ruff finding** (F841 in `gee_utils.validate_bands`): it is bug B7 and goes away with that fix.
- **`dev/` and `elmtest/`**: kept (decision 2).

## Before / after

| Metric | Before (`8b43332`) | After (`refactor/cleanup`) |
|---|---|---|
| Package lines (src/dapper excl. dev), both formatted with ruff at 88 cols | 15,049 | 13,665 (−9.2%) |
| Package lines as committed (baseline unformatted) | 13,741 | 13,665 |
| Python statements (AST, excl. dev) | 5,767 | 5,213 (−9.6%) |
| `ruff check` src with the project config (E4/E7/E9/F/I/UP/B) | 412 | 1 (B7) |
| `ruff check` src, ruff 0.16 defaults (`--isolated`, excl. dev) | 434 | 71 |
| `ruff check` src (E4,E7,E9,F, excl. dev) | 70 | 1 (B7) |
| `ruff format --check` src files needing reformat | 39 / 48 | 0 / 47 |
| mypy errors (excl. dev) | 100 in 17 files | 88 in 16 files |
| Coverage (statements) | 36% (1,963 / 5,456) | 49% (2,451 / 4,974) |
| Tests | 47 passed, 1 skipped | 82 passed, 1 skipped (+35 approved characterization/regression tests) |

`git diff 8b43332 -- tests/` shows only the five **added** approved files (`tests/test_characterize_*.py`). All six original test files are byte-identical to the baseline.
