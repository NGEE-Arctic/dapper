# dapper refactor notes

Baseline commit: `8b43332c37dc30d122749fb85a2399517163f009`.
The before/after numbers and per-module change log get filled in at wrap-up. The baseline numbers are in `REFACTOR_PLAN.md` §2.

## Bugs found (logged, NOT fixed)

These were found during the Phase 1 survey. Current behavior is left as-is unless you approve a fix, and any fix goes in its own commit, separate from the refactor.
"Verified" means I reproduced it in a throwaway script against the baseline code. "By inspection" means I confirmed it by reading the code.

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
| B17 | `met/adapters/fluxnet.py:161` vs ERA5 | FLUXNET filters to `start_year..end_year` with no lookahead. ERA5 keeps the next Jan-1 00:00 for interval alignment. FLUXNET has no `INTERVAL_END_VARS`, so this is consistent today. Flagged only because TIMESTAMP_END-labelled fluxes are interval-end values too, so the same alignment question applies. | Decide whether FLUXNET fluxes should be relabelled to interval start like ERA5. | By inspection (design question) |

## Remaining smells deliberately left alone

To be filled in at wrap-up. Candidates so far:
- Broad `except` blocks whose narrowing could change behavior (`BaseAdapter.pack_params`, `cmip_utils` retry loop, `met/validation` mode detection, `gee_utils.try_to_download_featurecollection`). Kept to preserve behavior.

## Per-module change log

To be filled in during Phase 2.

## Before / after

| Metric | Before | After |
|---|---|---|
| Package LOC (src/dapper excl. dev) | 13,741 | – |
| `ruff check` src (0.16 defaults, excl. dev) | 434 | – |
| `ruff check` src (E4,E7,E9,F, excl. dev) | 70 | – |
| `ruff format --check` src files needing reformat (excl. dev) | 39 / 48 | – |
| mypy errors (excl. dev) | 100 in 17 files | – |
| Coverage | 36% | – |
| Tests | 47 passed, 1 skipped | – |
