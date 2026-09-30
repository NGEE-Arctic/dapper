# dapper refactor plan

Status: **Phase 2 done up to the pause points** (branch `refactor/cleanup`). Steps 0–12 and 17 are done; 8b turned out to need no change. The §5 breaking changes are done, except §5.2, which was declined. Pinned bugs are fixed. Still waiting on you: steps 13–16, and the regression tests R-1…R-8 for the remaining bugs. See `REFACTOR_NOTES.md`.

Decisions (answers to §6 and the follow-up questions):
1. The plan is approved. All work stays on `refactor/cleanup`.
2. `dev/` and `elmtest/` stay as they are for now. They remain excluded from lint and type checks.
3. The print → logging change (§5.2) is **skipped**.
4. Ruff line length is **88**.
5. The breaking changes in §5 are approved, except §5.2 (see item 3).
6. Characterization tests T-1, T-2, T-7, T-8 and T-9 are approved. P-1 and T-3/T-4/T-5/T-6 are not approved yet.
7. Bug fixes are wanted. Each one goes in its own commit, after the refactor.

Not answered, so the defaults apply:
- §6 Q5: no module splits, so step 18 is skipped.
- §6 Q6: constant values are preserved as they are.
- §6 Q7 and Q8: current behavior is kept.

Steps 13–16 still wait for your go-ahead, because the tests that would cover them were not approved.

- Baseline commit: `8b43332c37dc30d122749fb85a2399517163f009` (`main`)
- Companion files:
  - `REFACTOR_NOTES.md` has the bugs found during the survey (logged, not fixed).
  - `TEST_CHANGE_PROPOSALS.md` has proposed test changes and new characterization tests. None are applied yet.

---

## 1. Package map

`src/dapper` has about 13.7k lines of package code (dev/ excluded) plus 2.0k lines of scratch scripts in `dev/`.
Only the top-level lazy API in `dapper/__init__.py` and the module-level `__all__` lists count as public API.
Those `__all__` lists live in `domains`, `surf`, `met.adapters`, `integrations.era5`, `schemas.elm` and `config.metsources.era5`.
There are no CLI entry points.

| Module | LOC | Role | Imports from dapper |
|---|---:|---|---|
| `__init__` | 75 | Lazy re-export of `Domain`, `ERA5Adapter`, `ERA5SamplingPlan`, `Exporter`, `plan_era5_land_sampling`, `sample_era5_land`, `sample_e5lh` | (lazy) |
| `domains.domain` | 1057 | `Domain` dataclass (provided/support/cells views, sites vs. cellset, output paths, `export_*` facades) | `geo.constants`; lazily `elm_domain`, `surf.sfile`, `landuse`, `met.exporter`, `topounit.topomake` |
| `domains.elm_domain` | 126 | Builds the ELM `domain.nc` Dataset from `Domain.cells` | (type-only) `Domain` |
| `met.exporter` | 933 | `Exporter`: CSV shards → per-gid parquet → packed int16 ELM MET NetCDF (sites / cellset) | `domains`, `elm.utils`, `io.fs`, `met.temporal`, `met.writers`, `schemas.elm`, `geo.constants` |
| `met.temporal` | 427 | DTIME axis construction, noleap handling, interval alignment, year-range inference | – |
| `met.writers` | 295 | netCDF4 low-level create/append and auto-chunking | `met.temporal` |
| `met.adapters.base` | 114 | `BaseAdapter` ABC (discover / normalize / preprocess / pack) | `elm.utils` |
| `met.adapters.era5` | 240 | ERA5-Land → ELM adapter | `config.metsources.era5`, `elm.utils`, `met.temporal`, `schemas.elm` |
| `met.adapters.fluxnet` | 495 | FLUXNET → ELM adapter | `elm.utils`, `met.temporal`, `schemas.elm` |
| `met.cmip_utils` | 806 | Pangeo CMIP6 search, bbox sampling, and download helpers | `elm.utils` |
| `met.validation` | 562 | Post-export PNG quicklooks | `met.temporal` |
| `elm.utils` | 371 | Humidity math, packing params, a grab-bag `elm_data_dicts()`, legacy validators | `domains.domain` (module level) |
| `schemas.elm` | 82 | Canonical ELM MET units, ranges and required vars | – |
| `config.metsources.era5` | 73 | ERA5 band lists and raw→ELM name map | – |
| `integrations.era5` | 1039 | `sample_era5_land`: ARCO/GEE backend planning plus the ARCO download path | `config`, `domains`, `gee_utils` (lazy) |
| `integrations.earthengine.gee_utils` | 917 | GEE geometry conversion, `sample_e5lh`, task/export helpers, polygon sampling | `config`, `domains` |
| `geo.lonwrap` | 38 | Longitude wrap inference and normalization | – |
| `geo.sampling` | 394 | Nearest-cell gridded sampling, grid metadata, `write_netcdf` | `geo.lonwrap` |
| `geo.zonal` | 519 | Area-weighted polygon/grid intersection and reducers | `geo.sampling` |
| `geo.constants` | 5 | `LATLON_DECIMALS` | – |
| `surf.sfile` | 1340 | `SurfaceFile` class plus build/write/customize helpers | `domains`, `geo.*`, `surf.*` |
| `surf.surface_var_specs` | 1058 | Surface-variable spec table (mostly data) | – |
| `surf.schema` | 369 | `REGISTRY`/`SCHEMA` built from specs, plus unused export policies | `surf.surface_var_specs` |
| `surf.validate` | 384 | `SurfaceValidator` (surface NetCDF checks → DataFrame) | `surf.schema` |
| `surf.fraction_closure` | 208 | Forces PCT_* partitions to close to 100 (or 1) | `surf.surface_var_specs` |
| `surf.sample` | 185 | Legacy dict-based point sampler (`SurfacePointSampler`) | `geo.lonwrap` |
| `landuse.landuse` | 332 | Landuse time-series sampling (nearest / zonal) and export | `domains`, `geo.*`, `surf.fraction_closure` |
| `topounit.topomake` | 976 | GEE-based topounit generation | `domains`, `gee_utils` |
| `topounit.topoplot` | 186 | Topounit plotting (contextily / folium) | – |
| `io.fs`, `io.attrs`, `io.provenance` | 91 | Small helpers (only `fs.remove_directory_contents` is used) | – |
| `elmtest/` | 0 (.py) | Empty package holding a design .md and a .sh script | – |
| `dev/` | 1967 | Scratch scripts: no `__init__`, not shipped, hardcoded `X:\...` paths, imports of modules that no longer exist | – |

Dependency notes:
- The layering is mostly clean: `geo`, `schemas` and `config` are leaves, and `Domain` imports its heavy consumers lazily.
- `elm.utils` imports `Domain` at module level, used by one legacy `isinstance` check. That means every MET adapter import pulls in geopandas and `domains`.
- `gee_utils` and `topomake` each have their own copy of the Earth Engine lazy-import proxy.

## 2. Baseline

Environment: a throwaway uv venv (not committed) with `pip install -e . pytest pytest-cov ruff mypy`.
Versions: Python 3.12.13, numpy 2.5.3, pandas 3.0.6, xarray 2026.9.0, geopandas 1.2.0, shapely 2.1.2, netCDF4 1.7.4.

### Tests
`pytest --cov=dapper`: **47 passed, 1 skipped, 86 warnings (80 s)**.
- Skipped: `test_era5_sampling.py::test_live_arco_point_download`, which needs `DAPPER_RUN_CDS_INTEGRATION=1`.
- `test_cmip_utils.py::test_intake` needs network access to the Pangeo catalog. It does not exercise dapper code.
- Warnings include `datetime.utcnow()` deprecation (exporter, sfile), `unary_union` deprecation (zonal), and a numpy shape-setting deprecation (writers).

### Coverage (statements, total 36%)

| Module | Cover | | Module | Cover |
|---|---:|---|---|---:|
| domains/domain | 46% | | surf/sfile | 31% |
| domains/elm_domain | **0%** | | surf/schema | 47% |
| met/exporter | 57% | | surf/validate | **0%** |
| met/temporal | 84% | | surf/fraction_closure | 88% |
| met/writers | 79% | | surf/sample | 21% |
| met/adapters/base | 60% | | surf/surface_var_specs | 81% |
| met/adapters/era5 | 93% | | landuse/landuse | **0%** |
| met/adapters/fluxnet | **10%** | | topounit/topomake | **0%** |
| met/cmip_utils | **0%** | | topounit/topoplot | **0%** |
| met/validation | **0%** | | io/fs | 36% |
| elm/utils | 40% | | io/attrs, io/provenance | **0%** |
| integrations/era5 | 81% | | geo/zonal | 78% |
| integrations/earthengine/gee_utils | 17% | | geo/sampling | 32% |
| schemas/elm | 81% | | geo/lonwrap | 62% |

### Lint / format / types
There is no ruff, mypy or pyright config in the repo.
- **`ruff check`** (ruff 0.16.9 defaults) on `src/`:
  - Everything: 552 errors, 421 auto-fixable.
  - Excluding `dev/`: 434 errors.
  - Top rules: UP006 111, UP045 104, I001 66, F401 44, BLE001 35, UP035 29, UP007 23, B023 17, F811 13, F841 9, S110 8, F821 7.
  - Classic `E4,E7,E9,F` subset, excluding dev/: 70 (E402 23, F401 15, F811 9, F841 9, E702 7, E701 4, E741 2, F541 1).
  - All 7 F821 (undefined name) hits are in `dev/`.
  - `tests/` has 21 findings. Tests are read-only, so they stay.
- **`ruff format --check`**: 58 of 68 src files would be reformatted (39 of 48 excluding dev/), plus 6 test files, which stay untouched.
- **`mypy src/dapper --ignore-missing-imports --exclude dev/`**: **100 errors in 17 files**.
  - By file: sfile 22, exporter 21, era5 9, validate 8, schema 8, surface_var_specs 7, zonal 7, …
  - Mostly `Hashable` vs `str` from xarray, `None`-initialized attributes, and untyped spec dicts.

## 3. Code smells (file:line)

### 3.1 Duplicated or near-duplicate helpers
- **Global-attr merge** (dapper_attrs setdefault → created_utc → append_attrs update) is copied three times: `geo/sampling.py:321-336`, `surf/sfile.py:190-204`, `surf/sfile.py:1324-1338`.
  - `io/attrs.py:10` is a fourth variant and is unused.
  - `domains/elm_domain.py:117-124` is a fifth.
- **Feb-29 masking** appears 6 times: `met/temporal.py:42,159-162,187-190,263,386`, `met/adapters/fluxnet.py:163`.
- **Numeric DTIME computation**: `temporal.create_dtime` (`temporal.py:307-323`) re-implements `_numeric_dtime` (`temporal.py:88-101`). The `linear_vars` list is duplicated at `temporal.py:164,266`.
- **Longitude normalization, vectorized by hand** 4 times: `geo/zonal.py:114`, `landuse/landuse.py:120,245`, plus an unused `geo/lonwrap.py:31` `normalize_lons`.
  - `LonWrap` Literal is defined 3 times: `geo/lonwrap.py:9`, `geo/zonal.py:22`, `landuse/landuse.py:17`.
- **Median grid step**: `_median_step` and `_median_step_lon` (`geo/sampling.py:155-184`) are near copies.
- **lat/lon dim detection**: `surf/sfile.py:226`, `surf/sample.py:16`, and the candidate tuples in `surf/validate.py:64-65`.
- **Earth Engine lazy-import proxy**: `integrations/earthengine/gee_utils.py:4-29` and `topounit/topomake.py:3-26`.
- **FLUXNET timestamp parsing**: `met/adapters/fluxnet.py:81-104` and `:133-156`. Candidate column tables at `:283-292` and `:360-367`.
- **Topounit metadata building**: `_combine_cartesian` and `_combine_hierarchical` (`topounit/topomake.py:378-391` vs `:413-422`). The default aspect ranges are repeated at `:297` and `:603`.
- **Exporter CSV gid inference and merge**: `met/exporter.py:792-819` vs `:862-892`.
- **Sampling provenance column list**: `domains/domain.py:660-678` and `met/exporter.py:455-469`.
- **ELM constants spread over three places**:
  - `elm/utils.py:165-284` (`elm_data_dicts`) duplicates `schemas/elm.py` (units, ranges, required vars) and `config/metsources/era5.py` (raw→ELM map, required bands).
  - `elm/utils.py:344-353` repeats the ranges again, with QBOT 0.1 vs 0.04.
  - `met/validation.py:20-46` repeats units and raw vars.
  - Some values disagree (see Open questions).
- **Plotting block** copied 3 times: `met/validation.py:337-360`, `:407-432`, `:531-557`.
- **Two topounit-attach paths**: `SurfaceFile._attach_topounits_from_domain` (`sfile.py:801`) and `add_topounits_from_domain` (`sfile.py:1022`). They give different outputs.

### 3.2 Dead code
- `met/cmip_utils.py:583-659`: 77 unreachable lines after `return out`.
- `met/exporter.py:851-909`: `_pass1_to_parquet_raw` is never called.
  - `years_span` and `filename_template` params of `_write_elm_combined` and `_write_elm_sites` are unused (`:488,588`).
  - The `mode == "sites"` check in `_write_elm_combined` (`:498`) can't be reached.
  - Unused local `lon` (`:612`).
  - `_resolve_pack_scope` has redundant branches (`:718-722`); every value maps to `"per-site"`.
- `geo/zonal.py:277-294`: `_reduce_da` is unused. `:387` can't be reached. The per-gid `MAX_ZONAL_CELLS` guards (`:236,255,363`) can't trigger after the global check at `:202`.
- `met/temporal.py:268,344-347`: `accum_vars` is always empty.
- `surf/sfile.py`: unused `SP` import (`:12`), `_is_int_dtype` and `dtype_str` (`:55,69`), `coords` (`:313,485`), `lat_dim, lon_dim` (`:430`).
  - The `engine` param of `customize_surface` is accepted but unused.
- `integrations/earthengine/gee_utils.py:46-49`: `_ROOT_DIR` and `_DATA_DIR` are unused. The shapely re-imports at `:37-40` shadow the module-level ones at `:61-66`.
- `met/adapters/era5.py:220-224`: `wind_direction` is computed and then dropped by the final column selection.
- Symbols unused anywhere (src, tests, docs, tutorials):
  - `gee_utils.split_into_dfs`, `infer_id_field` (points to a nonexistent `e5lh_to_elm`), `featurecollection_to_df_loc`
  - `elm.utils.gen_zone_mappings`, `validate_met_vars` (needs `docs/data`, which isn't shipped)
  - `io.attrs.apply_append_attrs`, `io.fs.make_directory`, `io.provenance.get_git_commit_hash`
  - `surf.schema.register_many`, `expand_registry` (its `as_json` arg is ignored), `propose_export_policy`, `EXPORT_POLICIES`, `validate_against_schema` (duplicates `SurfaceValidator`)
  - `surf.sfile.build_surface_dataset_cellset`, `met.cmip_utils.extract_vars_from_files`, `cftime_date`, `summarize_search`
  - These have public names, so deleting them goes to "Proposed breaking changes".
- `met/validation.py:178-187,236-245`: the `raw-site-parquet` and `raw-site-csv` modes read layouts that nothing produces any more; the only producer was the dead `_pass1_to_parquet_raw`. `:190` checks `Dataset is None`, which can never be true.
- Commented-out code: `domains/domain.py:759-760`, `met/adapters/fluxnet.py:446-447`.

### 3.3 Over-defensive code
- **Broad `except Exception`** (35× BLE001, 8× try-except-pass). Worst cases:
  - `met/adapters/base.py:110`: any error from the packer silently becomes `(min, 1.0)`.
  - `met/cmip_utils.py:346`: parquet failure silently falls back to CSV.
  - `:544`: retries every exception, including programming errors.
  - `gee_utils.py:762`: download failure becomes `None`.
  - `topomake.py:50,914`, `met/validation.py:88,267,269,465`, `surf/sample.py:61,107`, `writers.py:33`.
- **Exceptions turned into return values**:
  - `Domain.make_topounits` (`domain.py:818-820`) catches `RuntimeError`, prints it, and returns `None`, even though it is annotated `-> Domain`.
  - `make_topounits` (`topomake.py:736-744`) returns `None` after falling back to a Drive export.
- **Redundant checks**:
  - `getattr(self.domain, "mode", None)` when `domain` is always a `Domain` (`exporter.py:253,334,340,478,498`).
  - `getattr(x, "topounits", None) is not None and x.topounits is not None` (`sfile.py:896,1052`).
  - `getattr(domain, "support")` / `"gdf"` fallbacks (`gee_utils.py:147-153`, `topomake.py:580-584,888-891`).
  - `Domain.from_file` re-does the CRS normalization that `_ensure_geodf` already does (`domain.py:251-254`).
  - `Path(path_out)` is converted twice (`domain.py:230,261`).
  - `elm_data_dicts() or {}` (`exporter.py:224`).
  - `v.set_auto_scale(True)  # defensive` (`writers.py:294`).
  - `_dtype_nbytes` has an alias table that `np.dtype` already handles (`writers.py:30-43`).
- **Function-local re-imports** of already-imported modules: `landuse.py:55-58,136`, `sfile.py:425,945,1048-1050`, `domain.py:735`, `gee_utils.py:61-66`.

### 3.4 Single-implementation or forwarding abstractions
- `Domain._to_elm_domain_dataset` (`domain.py:856`) only forwards to `elm_domain.to_elm_domain_dataset`. The lazy import is justified, but the wrapper could be inlined into `export_domain`.
- `Domain.from_gdf` and `Domain.gdf` are aliases. They are used in docs, so they stay.
- `Exporter._run_dir_for_gid`, `_met_dir_for_gid` and `_zone_mappings_path` (`exporter.py:332-347`) are thin layers. `_zone_mappings_path` is unused.
- `surf.sample.SurfacePointSampler` plus `build_surface_dataset*` form a legacy pipeline parallel to `geo.sampling`. Nothing internal uses it, but `surf.sample` is in the documented module list.
- `BaseAdapter` has two real implementations, so it is justified.

### 3.5 Wrong, stale or noise comments and docstrings
- `Exporter` class docstring (`exporter.py:65-131`) documents parameters that don't exist: `csv_directory`, `df_loc`, `id_col`. It also says zone mappings go "at the root".
- `ERA5Adapter` docstring step 1 claims Feb-29 is removed in preprocess, but the code deliberately keeps it (`era5.py:35,117`). It also documents `normalize_locations` / `id_column_for_csv` as adapter responsibilities.
- `customize_surface` documents a `validate_units` param that doesn't exist. The `units_policy="warn"` option doesn't warn (`sfile.py:408,456`).
- `sample_e5lh` docstring mentions `dapper.domains.aoi.AOI` and `Domain.to_geometries()`, neither of which exists (`gee_utils.py:511-512`).
- `surf/schema.py:10-68`: the real module docstring is a string placed after the imports, and it references `dapper.surf.write`, which doesn't exist. The header comment says `# elm_surface_registry.py`.
- `SurfaceValidator` docstring says V-105 compares against `PCT_NATVEG`, but the code compares against 100. V-108 and V-109 are missing from the docstring (`validate.py:49`).
- `_geod_area_m2` comment says "fallback: 0" but the code returns NaN (`topomake.py:915`).
- Wrong filename headers: `config/metsources/era5.py:1` (`sources/`), `topoplot.py:1` (`topounits/plotting.py`).
- Placeholder module docstrings like `"""dapper module: x.y."""` in 27 files.
- Conversational or noise comments:
  - `domain.py:203` ("you said you don't care")
  - `zonal.py:311` (`# NEW`), `topoplot.py:124` (`# <-- NEW`)
  - `gee_utils.py:1` ("Generic functions JPS")
  - many "your …" phrasings (`writers.py:60`, `sampling.py:88,250`, `era5.py:200`)
- Comments that just restate the code are common across the package, for example `# Create DataFrame` and `# stable site order`.

### 3.6 Naming, magic numbers, inconsistencies
- The same concept goes by different names: `df_loc` / `df_loc_norm` / `points` / `targets`, `src_path` / `nc_in` / `path_nc`, and `out_dir` / `group_dir` / `write_directory`.
- `import dapper.met.temporal as dt` shadows the usual meaning of `dt`. `from dapper.io import fs as utils` is a misleading alias (`exporter.py:37-38`).
- Magic numbers:
  - `11132` (ERA5 m scale, `gee_utils.py:590-592`), `32767` / `"i2"` (packing), `1e-12`, `-9.96921e36` (`sfile.py:213,215`)
  - `0.25` half-degree fallback grid (`surf/sample.py:35-36`), `1800.0` (fluxnet), `611.2/17.67/243.5` (repeated in `fluxnet.py:419` and `elm/utils.py:149`)
- A literal tab character is used as a separator (`exporter.py:626`); elsewhere the code writes `"\t"`.
- `vars` shadows a builtin (`met/validation.py:110,288,374,445`). `propose_export_policy(..., ParDef=...)` shadows the class name.
- `__version__ = "0.1.0"` (`__init__.py:20`) but the pyproject version is `1.0`.

### 3.7 `print` used as logging, global side effects, hardcoded paths
- 63 `print` calls used for progress or warnings: met/validation (13), topomake (12), gee_utils (10), exporter (9), elm/utils (5), integrations/era5 (4), cmip_utils (4), fluxnet (3), sfile (2), domain (1).
- `met/validation.py:13-14` calls `matplotlib.use("Agg")` at import time, which changes the backend for the whole process.
- `elm/utils.py:13-14` hardcodes `<repo>/docs/data`, which doesn't exist in an installed package.
- `dev/pathing.py` hardcodes personal `X:\Research\...` paths.

### 3.8 Long functions and multi-job modules
- `surf/sfile.py` (1340 lines) does five jobs: dict-based builders, writer, `customize_surface`, zonal policy, and the `SurfaceFile` class. `SurfaceFile.from_domain` alone is about 120 lines.
- `gee_utils.py` (917) mixes geometry conversion, date helpers, task management, `sample_e5lh` (about 230 lines with its docstring) and polygon sampling.
- Other long functions: `Exporter.run` (~100), `_write_elm_sites` (~90), `temporal.create_dtime` (~145), `topomake.make_topounits` (~210), `landuse.sample_landuse_timeseries` (~260), `cmip_utils.sample_bbox_means_for_aois` (~195).

### 3.9 Missing or inaccurate type hints
- Most public functions in `exporter`, `gee_utils`, `topomake`, `elm/utils`, `cmip_utils` and `met/validation` have no annotations.
- `Domain.make_topounits -> Domain` can return `None`. `Domain.topounits_for_gid` has no return type.
- Most of the 100 mypy errors come from untyped `SURFACE_VAR_SPECS` (`dict[str, object]`) and from Exporter attributes initialized to `None` without `Optional`.

### 3.10 Dependencies (pyproject)
- **Declared but not imported anywhere in the package:** `pip`, `jupyter`, `jupytext`, `geemap`, `rioxarray`.
  - `xyzservices` is only used indirectly through contextily.
  - `gcsfs` is needed at runtime for `gs://` zarr stores, so keep it.
- **Imported but not declared** (currently arrive transitively): `shapely`, `pyproj`, `fsspec`, `python-dateutil`, `intake`.
  - `folium` is an optional import in `topoplot.plot_interactive`; it is in `environment.yml` but not in `pyproject`.
- `environment.yml` and `pyproject` have drifted apart (for example, folium and the pandas>=3 pin).

## 4. Proposed refactoring sequence (lowest → highest risk)

Rules for every step:
- One focused commit per step.
- Run the full test suite, `ruff check`, `ruff format --check` and `mypy` after each step.
- The pass/skip count must stay at 47 passed / 1 skipped.
- Nothing under `tests/` changes.
- Work happens on a branch `refactor/cleanup`, never on `main`.

"Coverage" means how well the touched lines are exercised by the existing tests.

| # | Step | Files | Coverage of touched code | Risk |
|---|---|---|---|---|
| 0 | **Tooling config only.** Add `[tool.ruff]` (target py311, `extend-exclude = ["src/dapper/dev", "tests"]`, rule set per Q3) and `[tool.mypy]` (py311, `ignore_missing_imports`, exclude dev) to `pyproject.toml`. No pytest config is touched. | pyproject.toml | n/a | none |
| 1 | **`ruff format` of `src/` only** (formatting-only commit; `git diff -w` reviewed). | all src | n/a (ruff guarantees AST-equivalence) | low |
| 2 | **Import hygiene**: sort imports (I001), remove unused imports (F401/F811), move the misplaced imports in `exporter.py:37-43` to the top. Re-exports that are listed in `__all__` stay. | ~30 files | import-time only; every module is imported by tests or by `import` checks | low |
| 3 | **Typing syntax modernization** (UP006/UP007/UP035/UP045/UP037): `Optional[X]` → `X \| None`, `Dict` → `dict`. Annotations only. | ~20 files | n/a | low |
| 4 | **Remove unreachable and private dead code** (§3.2): the cmip_utils post-return block, `_pass1_to_parquet_raw`, unused private params, `_reduce_da`, unreachable branches and guards, unused locals, `accum_vars`, the dead `wind_direction` computation, commented-out code, unused module globals. Public-named functions are **not** removed (see §5). | cmip_utils, exporter, zonal, temporal, sfile, gee_utils, era5 adapter, domain, fluxnet | The removed code is unreachable or unused. Its surrounding code is at 57–93%, except cmip_utils (0%, but the block can't run) | low |
| 5 | **Fix comments and docstrings** (§3.5): correct the wrong docstrings, drop noise comments, replace the placeholder module docstrings with one accurate line each, move the `schema.py` docstring to the top. | many | no runtime effect | low |
| 6 | **Deprecation-safe `utcnow`**: replace `datetime.utcnow().isoformat() + "Z"` with `datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"`. The output string is identical. Done while extracting the shared attr-merge helper (§3.1): one `_merge_global_attrs` in a small module, used by `geo.sampling.write_netcdf`, `sfile.write_surface_nc` and `SurfaceFile.to_netcdf`. | new `io/attrs.py` (reuse the file), sampling, sfile, exporter | `write_surface_nc` is well covered (fraction-closure tests); `to_netcdf` and `write_netcdf` are not | low–med |
| 7 | **temporal consolidation**: `_drop_feb29_mask` helper, `create_dtime` reuses `_numeric_dtime`, one `_LINEAR_VARS` constant; fluxnet uses the shared Feb-29 helper. | met/temporal, fluxnet (1 line) | temporal 84% with targeted noleap and alignment tests | med |
| 8 | **geo consolidation**: a single `LonWrap`; zonal and landuse use `lonwrap.normalize_lons` (same NaN semantics, checked); merge `_median_step*`; one lat/lon-dim detection helper shared by `sfile`, `surf.sample` and `validate` (candidate tuples kept identical). | geo/*, landuse, surf/sfile, surf/sample, surf/validate | zonal 78%, sfile 31%, **landuse 0%, validate 0%, sample 21%** | **high** for the landuse/validate/sample parts. Split into 8a (geo + sfile, med) and 8b (landuse/validate/sample, **high**, pause) |
| 9 | **Exporter cleanup**: shared provenance-column constant (Domain + Exporter); collapse `_resolve_pack_scope` to equivalent logic; drop redundant `getattr(domain, "mode")`; rename the `utils`/`dt` aliases; type the `None`-initialized attributes; add public type hints. | met/exporter, domains/domain | exporter 57% (sites and cellset paths both hit by `test_met_temporal`/`test_era5_sampling`) | med |
| 10 | **Domain cleanup**: remove the double CRS and `Path` conversions, the redundant local import, and the commented code; add type hints (`topounits_for_gid`, `iter_runs`). The `make_topounits` swallow stays as-is (it's a bug; see NOTES). | domains/domain | 46%; `from_file` and `from_elm_domain` are **uncovered** | med (**high** for `from_file`/`from_elm_domain`) |
| 11 | **Over-defensive cleanup with provably equivalent narrowing**, only where the raised-exception set is fully known: `_dtype_nbytes` (→ `except TypeError`), `surf/sample.close`, `topoplot._fmt_num`. Broad excepts whose narrowing could change behavior (`BaseAdapter.pack_params`, cmip retries, validation mode detection) are **left as they are** and listed in NOTES. | writers, sample, topoplot | writers 79%; sample and topoplot low | low–med |
| 12 | **sfile internal cleanup**: remove redundant local imports and unused locals, simplify the `topounits` checks, add type hints. No module split unless Q5 says yes. | surf/sfile | 31% overall; `add_topounits_from_domain` and `from_domain(zonal)` are covered, `customize_surface`, `resize_dim` and `validate` are not | **high** for the uncovered methods; pause |
| 13 | **Earth Engine proxy dedupe**: move the lazy-`ee` proxy to `integrations/earthengine/_ee.py`, used by `gee_utils` and `topomake`. The module attribute `gee_utils.ee` stays. | gee_utils, topomake | 17% / 0%; import-time behavior only | **high** (pause) |
| 14 | **fluxnet**: extract `_parse_fluxnet_timestamps` and the candidate-column table. | met/adapters/fluxnet | **10%** | **high** (pause; wants characterization tests T-3) |
| 15 | **topomake**: shared `_append_bin_meta` for the two combiners and an aspect-ranges constant. | topounit/topomake | **0%**, needs GEE | **high** (pause; wants mocked tests T-6) |
| 16 | **met/validation**: shared `_plot_panel_grid` helper; `vars` → `variables` (private functions only). | met/validation | **0%** | **high** (pause; wants T-5) |
| 17 | **ELM constants**: make `elm_data_dicts()` build its entries from `schemas.elm` / `config.metsources.era5` **only where values are identical**; values that differ stay as literals and are flagged. | elm/utils | 40% | med |
| 18 | **Optional module splits** (only if Q5 is approved): `sfile.py` → `surf/build.py`, `surf/write.py`, `surf/customize.py`, and `sfile.py` (class + re-exports). `gee_utils` → geometry / tasks / sampling, keeping re-exports **and** keeping `sample_e5lh` resolvable as `dapper.integrations.earthengine.gee_utils.sample_e5lh`, which a test monkeypatches. | surf/*, gee_utils | as above | **high** |

Steps 0–7 and 9 can go in sequence without check-ins.
**Before steps 8b, 10 (uncovered parts), 12, 13, 14, 15, 16 and 18 I will stop and check with you**, as the task asked.
For 14–16 I recommend approving the characterization tests in `TEST_CHANGE_PROPOSALS.md` first.

Constraints found in the tests that the refactor must respect:
- `integrations/era5._sample_gee` must keep looking up `gee_utils.sample_e5lh` **at call time**.
- `integrations.era5._arco_available_range` must stay a module-level name that is resolved at call time. Tests monkeypatch both.
- `test_zonal_surface_core` mutates `surface_var_specs.SURFACE_VAR_SPECS` in place. Code must keep reading that dict live, not a snapshot copy.

Expected end state: about 13.7k → roughly 12.9k package LOC, most of it from dead code (≈350 lines) and dedupe. `ruff check` on src (excl. dev) goes from 434 to 0 under the agreed rule set. Mypy goes from 100 down to "no new errors, fewer overall"; I'm not promising zero. Coverage stays the same or goes up slightly because dead code is removed.

## 5. Proposed breaking changes (not done without approval)

1. **Delete unused public-named helpers** (none are in `__all__` or top-level, but they aren't underscored). There are no references in src, tests, docs or tutorials:
   - `gee_utils.split_into_dfs`, `infer_id_field`, `featurecollection_to_df_loc`
   - `elm.utils.gen_zone_mappings`, `validate_met_vars`
   - `io.attrs.apply_append_attrs`, `io.fs.make_directory`, `io.provenance` (whole module)
   - `surf.schema.register_many`, `expand_registry`, `propose_export_policy`, `EXPORT_POLICIES`, `validate_against_schema`
   - `met.cmip_utils.extract_vars_from_files`, `cftime_date`, `summarize_search`
   - `surf.sfile.build_surface_dataset`, `build_surface_dataset_cellset` (`sfile` is a documented module)
   - `met/validation` raw-mode quicklooks
   - Alternative: keep them as thin shims that emit a `DeprecationWarning` for one release.
2. **Replace `print` with `logging`** (module-level `logger = logging.getLogger(__name__)`).
   - This changes what users see in notebooks: INFO is hidden unless logging is configured.
   - Proposal: progress messages → `logger.info`, warning-like prints → `logger.warning` (still visible by default) or `warnings.warn` where a `UserWarning` pattern already exists.
3. **`Domain.make_topounits` should raise instead of printing and returning `None`.** It also contradicts its own `-> Domain` annotation.
4. **Stop calling `matplotlib.use("Agg")` at import** of `dapper.met.validation`; switch backend only inside `make_quicklooks`, or not at all.
5. **Single-source `__version__`** from `importlib.metadata.version("dapper-elm")`. That makes it `"1.0"` instead of `"0.1.0"`. The Dockerfile prints it.
6. **Dependencies**: drop `pip`, `jupyter`, `jupytext`, `geemap`, `rioxarray` from runtime deps (move them to a `notebooks` extra or the dev group if wanted). Add `shapely`, `pyproj`, `fsspec`, `python-dateutil`, `intake`, plus an optional `plot` extra with `folium`. Sync `environment.yml`.
7. **Deprecate unused parameters** such as `customize_surface(engine=...)` and `ERA5Adapter.id_column_for_csv`. The latter is on a top-level class and is referenced in the docs, so it would be deprecated, not removed.
8. **Make `elm.utils` stop importing `Domain` at module level.** This only changes import cost, not behavior, but it touches a (documented-ish) module's import graph. It would only be needed if item 1 removes `gen_zone_mappings`.

## 6. Open questions

1. **`src/dapper/dev/`** (2k lines of scratch scripts: hardcoded personal paths, imports of modules that no longer exist, 7 undefined names). Delete, or move to `scripts/dev/` outside the package? It isn't shipped either way (no `__init__.py`). The same question applies to `src/dapper/elmtest/` (an empty package holding a .md and a .sh).
2. **print → logging**: approve §5.2? If not, I leave the prints alone. The only remaining style target it affects is the logging one.
3. **Ruff rule set and line length.** Proposal:
   - `select = ["E4","E7","E9","F","I","UP","B"]` with `line-length = 100`. Most existing lines fit; the ruff default of 88 would re-wrap much more code.
   - BLE / S / RUF / PERF would come in a later pass, if at all.
   - `tests/` excluded from lint and format because tests are read-only.
   - Is 100 OK, or do you prefer 88?
4. **Mypy level**: plain `mypy` with `ignore_missing_imports`, and the goal "no new errors plus fix errors in touched code". OK, or do you want a zero-error target (needs `TypedDict` for the specs and more)?
5. **Split `sfile.py` and `gee_utils.py`** (step 18)? Readability gain vs. churn and blame noise.
6. **Constants that disagree.** Which values are intended?
   - The QBOT packing range is `[0, 0.1]` in `elm_var_packing_params` but `[0, 0.04]` in `schemas.ELM_RANGES` and `elm_data_dicts()['ranges']`. A comment there says it was changed to 0.04 to avoid warnings.
   - DTBOT units are `"unsure"` vs `"K"`.
   - I'll preserve the current behavior regardless.
7. **Two topounit-attach paths.** `SurfaceFile.from_domain(attach_topounits=True)` attaches metadata-style 1D params. `SurfaceFile.export()` calls `add_topounits_from_domain` (TopounitFracArea plus PFT expansion). Is that divergence intended?
8. **Landuse closure.** The `nearest` path in `sample_landuse_timeseries` skips `normalize_fraction_closure`, but the zonal path applies it. Intended?
9. **Bugs** (see `REFACTOR_NOTES.md`): after the refactor, do you want fix commits for any of them? They would be separate commits, each with a test proposal first.
