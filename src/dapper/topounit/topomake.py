import ee
import geopandas as gpd

from dapper.domain import Domain
from dapper.utils import gee_utils as gu


# ----------------------------
# Utilities
# ----------------------------

def _require_single_band_image(img, sid):
    """
    Ensure 'img' behaves like an ee.Image with exactly one band.
    We do not try to guess the band; users should select/mosaic beforehand.
    """
    if not hasattr(img, 'bandNames') or not hasattr(img, 'select'):
        raise TypeError(
            f"binning['{sid}']['image'] must be an ee.Image (single band). "
            f"Pass an ee.Image you've already prepared with .mosaic()/.select()."
        )
    try:
        band_names = img.bandNames().getInfo()  # tiny request; returns a small list
    except Exception:
        # If the server call fails, fall back to trying to select(0) and hope it errors if invalid
        band_names = None

    if band_names is None:
        # try select(0) to trip a clear error message if multiple bands
        _ = img.select(0)
        # don't rename here; we rename after clipping
        return img

    if len(band_names) != 1:
        raise ValueError(
            f"Custom image for source '{sid}' must have exactly one band. "
            f"Please do something like: myimg = ee.Image(path_or_ic).mosaic().select('band')."
        )
    return img

# -----------------------------------
# Multi-scale HAND auto-selection
# -----------------------------------

def choose_hand_image(desired_scale=None, hand_edges=None, verbose=False):
    """
    Auto-select among available HAND products based on a desired scale (m)
    and/or fixed edges. If edges extend >100 m, prefer *_1000 variants.
    Returns (hand_image_single_band, native_scale_m).
    """
    # Candidates
    hand30_100  = ee.ImageCollection("users/gena/global-hand/hand-100").mosaic()      # up to 100 m
    hand30_1000 = ee.Image("users/gena/GlobalHAND/30m/hand-1000")                    # up to 1000 m
    hand90_1000 = ee.Image("users/gena/GlobalHAND/90m-global/hand-1000")             # up to 1000 m

    cand = [
        {"name": "hand30_1000", "img": hand30_1000, "scale": gu._nominal_scale_m(hand30_1000), "maxval": 1000},
        {"name": "hand90_1000", "img": hand90_1000, "scale": gu._nominal_scale_m(hand90_1000), "maxval": 1000},
        {"name": "hand30_100",  "img": hand30_100,  "scale": gu._nominal_scale_m(hand30_100),  "maxval": 100},
    ]

    # If user-provided edges exceed 100 m, prefer *_1000 products
    if hand_edges is not None and len(hand_edges) >= 2 and max(hand_edges) > 100:
        preferred = [c for c in cand if c["maxval"] >= 1000]
    else:
        preferred = cand

    if desired_scale is None:
        # Prefer 90 m *_1000 as a sensible default
        pick = next((c for c in preferred if c["name"] == "hand90_1000"), preferred[0])
        if verbose:
            print(f"[HAND] Auto-selected {pick['name']} @ ~{pick['scale']} m (no desired scale).")
        return pick["img"].select(0).rename("v"), pick["scale"]

    # Choose not-finer-than desired_scale; otherwise fallback to coarsest
    candidates = sorted(preferred, key=lambda c: c["scale"])
    not_finer = [c for c in candidates if c["scale"] >= desired_scale]
    pick = sorted(not_finer, key=lambda c: c["scale"])[0] if not_finer else candidates[-1]
    if verbose:
        print(f"[HAND] Desired ~{desired_scale:.1f} m → selected {pick['name']} @ ~{pick['scale']} m.")
    return pick["img"].select(0).rename("v"), pick["scale"]


# ----------------------------------------------------
# Sources: elevation, HAND, aspect (as a source)
# ----------------------------------------------------

def build_source(source_id, feature, desired_scale_hint, binning_spec, dem_source='arcticdem', verbose=False):
    """
    Returns (single-band ee.Image renamed to 'v', native_scale_m, meta dict).

    Built-in source_ids:
      - 'elev'   : ArcticDEM elevation (meters)
      - 'hand'   : multi-scale HAND auto-selection (meters)
      - 'aspect' : Aspect (degrees) derived from DEM
      - 'cti'    : Compound Topographic Index (dimensionless), mosaicked & scaled

    Custom image:
      If binning[source_id] contains key 'image', that value MUST be a single-band ee.Image
      (already mosaicked/selected/masked as needed). We validate single-band and use it.
      If the image advertises a degree-based/undefined native scale (~111,319 m), we coerce
      the native scale to desired_scale_hint (if provided) or 90 m to avoid over-clamping.
    """
    def _coerce_native_scale_m(img, fallback_m=90.0):
        """Return a reasonable native scale in meters; fix ~1° (≈111 km) projections."""
        try:
            s = float(gu._nominal_scale_m(img))
        except Exception:
            s = float(fallback_m)
        # If scale looks like degrees/undefined (> ~5 km), trust the hint or fallback
        if s > 5000.0:
            return float(desired_scale_hint) if (desired_scale_hint is not None) else float(fallback_m)
        return s

    region = gu._geom_from_any(feature)

    # --- 0) Generic custom image (takes precedence if provided) ---
    if isinstance(binning_spec, dict) and ('image' in binning_spec):
        img = _require_single_band_image(binning_spec['image'], source_id)
        img = img.clip(region).select(0).rename('v')
        scale = _coerce_native_scale_m(img, fallback_m=90.0)
        meta = {
            "units": binning_spec.get("units", ""),                 # cosmetic
            "name":  binning_spec.get("name", str(source_id).upper())
        }
        if verbose:
            print(f"[SRC {source_id}] custom image, native_scale≈{scale:.1f} m")
        return img, scale, meta

    # --- 1) Elevation (ArcticDEM) ---
    if source_id == 'elev':
        if dem_source != 'arcticdem':
            raise KeyError(f"DEM source '{dem_source}' not supported.")
        img = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").select('elevation').rename('v').clip(region)
        scale = _coerce_native_scale_m(img, fallback_m=2.0)
        if verbose:
            print(f"[SRC elev] native_scale≈{scale:.1f} m")
        return img, scale, {"units": "m", "name": "Elevation"}

    # --- 2) HAND (auto-pick scale/product) ---
    if source_id == 'hand':
        hand_edges = binning_spec.get('edges') if binning_spec.get('strategy') == 'fixed' else None
        img, scale_native = choose_hand_image(desired_scale=desired_scale_hint, hand_edges=hand_edges, verbose=verbose)
        img = img.updateMask(img.mask()).rename('v').clip(region)
        scale = _coerce_native_scale_m(img, fallback_m=scale_native)
        if verbose:
            print(f"[SRC hand] native_scale≈{scale:.1f} m")
        return img, scale, {"units": "m", "name": "HAND"}

    # --- 3) Aspect (from DEM) ---
    if source_id == 'aspect':
        if dem_source != 'arcticdem':
            raise KeyError(f"DEM source '{dem_source}' not supported for aspect.")
        dem = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").select('elevation').clip(region)
        aspect = ee.Terrain.aspect(dem).rename('v')
        scale = _coerce_native_scale_m(dem, fallback_m=2.0)  # use DEM’s native scale
        if verbose:
            print(f"[SRC aspect] derived from DEM, native_scale≈{scale:.1f} m")
        return aspect, scale, {"units": "deg", "name": "Aspect", "from_dem": True}

    # --- 4) CTI (flow index) — unitless with scale factor 1e8 ---
    if source_id == 'cti':
        img = (ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti")
               .mosaic().select(0).toFloat().divide(1e8)   # apply 1e8 scale factor
               .rename('v').clip(region))
        scale = _coerce_native_scale_m(img, fallback_m=90.0)  # many tiles advertise ~1°; coerce to 90 m
        if verbose:
            print(f"[SRC cti] scaled by 1e8, native_scale≈{scale:.1f} m")
        return img, scale, {"units": "", "name": "CTI"}       # unitless

    # Unknown source id
    raise ValueError(f"Unknown source_id: {source_id}")


# --------------------------------
# Binning helpers per source
# --------------------------------

def _clone_meta(acc_meta):
    """Deep-copy the nested bits so each branch mutates its own dicts."""
    return {
        **acc_meta,
        'source_ids': list(acc_meta.get('source_ids', [])),
        'labels':     {**acc_meta.get('labels', {})},
        'bin_bounds': {**acc_meta.get('bin_bounds', {})},
        'bin_method': {**acc_meta.get('bin_method', {})},
    }


def _compute_percentile_edges(image, region, n_bins, analysis_scale, max_samples=200_000, band_name='v'):
    """
    Return monotonically increasing bin edges (list length n_bins+1) using
    sampling-based empirical quantiles inside 'region'.
    """
    samples = image.sample(
        region=region,
        scale=analysis_scale,
        geometries=False,
        dropNulls=True,
        numPixels=max_samples
    )
    arr = ee.List(samples.aggregate_array(band_name))
    size = ee.Number(arr.size())
    sorted_arr = arr.sort()

    def _edges_from_samples():
        # indices 0..n_bins -> values
        def idx_to_val(k):
            # floor((k/n_bins) * size)
            idx = ee.Number(k).multiply(size).divide(n_bins).int()
            # clamp to size-1 (in case of k==n_bins)
            idx = idx.min(size.subtract(1))
            return sorted_arr.get(idx)
        return ee.List.sequence(0, n_bins).map(idx_to_val)

    def _edges_from_minmax():
        stats = image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=analysis_scale,
            bestEffort=True,
            maxPixels=1e13
        )
        vmin = ee.Number(stats.get(f"{band_name}_min"))
        vmax = ee.Number(stats.get(f"{band_name}_max"))
        step = vmax.subtract(vmin).divide(n_bins)
        return ee.List.sequence(vmin, vmax, step)

    quant_edges = ee.Algorithms.If(size.gt(0), _edges_from_samples(), _edges_from_minmax())
    edges = ee.List(quant_edges).getInfo()  # client-side list

    # Ensure strict monotonicity (collapse duplicates with tiny epsilon)
    cleaned = [edges[0]]
    for e in edges[1:]:
        if e <= cleaned[-1]:
            e = float(cleaned[-1]) + 1e-9
        cleaned.append(float(e))
    return cleaned

def _compute_equalwidth_edges(image, region, n_bins, analysis_scale, band_name='v'):
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=analysis_scale,
        bestEffort=True,
        maxPixels=1e13
    )
    vmin = stats.get(f"{band_name}_min")
    vmax = stats.get(f"{band_name}_max")
    if (vmin is None) or (vmax is None):
        return [0.0, 1.0]
    vmin = float(vmin)
    vmax = float(vmax)
    if vmin == vmax:
        return [vmin, vmin + 1e-9]
    step = (vmax - vmin) / n_bins
    return [vmin + i * step for i in range(n_bins + 1)]

def build_bins_for_source(source_id, image, region, binning_spec, analysis_scale, aspect_ranges_default=None):
    """
    Produces a list of bin definitions for a source.
    Numeric sources (elev, hand): [{'id', 'low', 'high', 'label', 'method', 'units'}]
    Aspect (circular): [{'id','start','end','wrap','label','method','units'}]
    """
    strategy = binning_spec.get('strategy')
    label_prefix = binning_spec.get('label_prefix', source_id.upper())
    bins = []

    if source_id == 'aspect':
        if strategy != 'fixed':
            raise ValueError("Aspect currently supports 'fixed' ranges only.")
        ranges = binning_spec.get('ranges') or aspect_ranges_default or [(270, 90, 'N'), (90.01, 269.99, 'S')]
        for i, (start, end, name) in enumerate(ranges, start=1):
            start = float(start); end = float(end)
            wrap = start > end
            bins.append({
                'id': i,
                'label': f"{label_prefix}_{name}",
                'start': start,
                'end': end,
                'wrap': wrap,
                'method': 'fixed',
                'units': 'deg'
            })
        return bins

    # Numeric sources: elev, hand
    if strategy == 'percentiles':
        n_bins = int(binning_spec.get('n_bins', 5))
        edges = _compute_percentile_edges(image, region, n_bins, analysis_scale, band_name='v')
    elif strategy == 'equalwidth':
        n_bins = int(binning_spec.get('n_bins', 5))
        edges = _compute_equalwidth_edges(image, region, n_bins, analysis_scale, band_name='v')
    elif strategy == 'fixed':
        edges = binning_spec.get('edges', None)
        if edges is None or len(edges) < 2:
            raise ValueError("Fixed-edge binning requires 'edges' with at least two values.")
        edges = [float(x) for x in edges]
        n_bins = len(edges) - 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        if high <= low:
            continue  # skip degenerate
        bins.append({
            'id': i + 1,
            'label': f"{label_prefix}_{low:.3f}-{high:.3f}",
            'low': low,
            'high': high,
            'method': strategy,
            'units': 'm'
        })
    return bins


# ----------------------------------------
# Mask builders & combination logic
# ----------------------------------------

def _numeric_bin_mask(image, low, high):
    return image.gte(low).And(image.lt(high)).selfMask()

def _aspect_bin_mask(aspect_img, start, end, wrap):
    return (aspect_img.gte(start).Or(aspect_img.lt(end)) if wrap
            else aspect_img.gte(start).And(aspect_img.lt(end))).selfMask()

def _apply_min_patch(mask_img, min_patch_pixels):
    if min_patch_pixels is None or min_patch_pixels <= 1:
        return mask_img
    cpp = mask_img.connectedPixelCount(100, True)
    return mask_img.updateMask(cpp.gte(min_patch_pixels))

def _combine_cartesian(bin_masks_by_source, max_topounits=None, min_patch_pixels=None):
    items = list(bin_masks_by_source.items())
    if not items:
        return []
    combined = []

    def _recurse(idx, acc_meta, acc_mask, acc_code):
        nonlocal combined
        if max_topounits is not None and len(combined) >= max_topounits:
            return
        if idx == len(items):
            band_name = "topounit_" + "__".join(acc_code)
            mask = _apply_min_patch(acc_mask, min_patch_pixels)
            combined.append({'band_name': band_name, 'mask': mask, 'meta': acc_meta})
            return

        sid, binlist = items[idx]
        for entry in binlist:
            meta = _clone_meta(acc_meta)  # <-- deep-clone nested dicts/lists

            # append this source's info
            meta.setdefault('source_ids', []).append(sid)
            meta.setdefault('labels', {})[sid] = entry['def']['label']
            if 'low' in entry['def']:
                meta.setdefault('bin_bounds', {})[sid] = {'low': entry['def']['low'], 'high': entry['def']['high']}
            else:
                meta.setdefault('bin_bounds', {})[sid] = {
                    'start': entry['def']['start'], 'end': entry['def']['end'], 'wrap': entry['def']['wrap']
                }
            meta.setdefault('bin_method', {})[sid] = entry['def']['method']

            codepiece = f"{sid}{entry['def']['id']}"
            mask = entry['mask'] if acc_mask is None else acc_mask.And(entry['mask'])
            _recurse(idx + 1, meta, mask, acc_code + [codepiece])

    _recurse(0, {'topounit_schema': 'cartesian'}, None, [])
    return combined

def _combine_hierarchical(order, bin_masks_by_source, max_topounits=None, min_patch_pixels=None):
    if not order:
        return []
    combined = []

    def _recurse(idx, parent_mask, parent_meta, parent_code):
        nonlocal combined
        if max_topounits is not None and len(combined) >= max_topounits:
            return

        sid = order[idx]
        entries = bin_masks_by_source[sid]
        for entry in entries:
            this_mask = entry['mask'] if parent_mask is None else parent_mask.And(entry['mask'])

            meta = _clone_meta(parent_meta)  # <-- deep-clone nested dicts/lists
            meta.setdefault('source_ids', []).append(sid)
            meta.setdefault('labels', {})[sid] = entry['def']['label']
            if 'low' in entry['def']:
                meta.setdefault('bin_bounds', {})[sid] = {'low': entry['def']['low'], 'high': entry['def']['high']}
            else:
                meta.setdefault('bin_bounds', {})[sid] = {
                    'start': entry['def']['start'], 'end': entry['def']['end'], 'wrap': entry['def']['wrap']
                }
            meta.setdefault('bin_method', {})[sid] = entry['def']['method']

            codepiece = f"{sid}{entry['def']['id']}"
            if idx == len(order) - 1:
                band_name = "topounit_" + "__".join(parent_code + [codepiece])
                mask = _apply_min_patch(this_mask, min_patch_pixels)
                combined.append({'band_name': band_name, 'mask': mask, 'meta': {**meta, 'topounit_schema': 'hierarchical'}})
                if max_topounits is not None and len(combined) >= max_topounits:
                    return
            else:
                _recurse(idx + 1, this_mask, meta, parent_code + [codepiece])
                if max_topounits is not None and len(combined) >= max_topounits:
                    return

    _recurse(0, None, {}, [])
    return combined

# -----------------------------------------
# Main API 
# -----------------------------------------
def make_topounits(
    feature,
    sources,                      # e.g., ['elev'] or ['elev','hand'] or ['elev','aspect']
    binning,                      # dict: { 'elev': {...}, 'hand': {...}, 'aspect': {...} }
    combine='cartesian',          # 'cartesian' or 'hierarchical'
    combine_order=None,           # required if combine='hierarchical', e.g., ['elev','hand']
    max_topounits=256,
    dem_source='arcticdem',
    return_as='gdf',              # 'gdf' or 'asset'
    export_scale='native',
    asset_name='topounits',
    asset_ftype='GeoJSON',
    min_patch_pixels=None,        # e.g., 9 to drop tiny patches
    target_pixels_per_topounit=500,
    target_scale=None,            # if provided, analysis_scale = max(target_scale, coarsest_native)
    verbose=False
):
    """
    Generate “topounits” (topographic units) by binning one or more raster sources
    (elevation, HAND, aspect, CTI, or a user-supplied image) and combining the
    resulting classes into polygons.

    This function **does not** globally reproject rasters. Instead it:
    1) chooses an **analysis scale** from the AOI area and planned bin count,
    2) clamps that scale to be **no finer than the coarsest source’s native scale**,
    3) runs all sampling, statistics, and vectorization with `scale=analysis_scale`
    while letting Earth Engine handle per-pixel alignment internally.

    The output is **one feature per bin** (i.e., the union geometry of all patches
    that belong to that bin), with rich metadata describing how each bin was formed.

    Args:
        feature (ee.Feature | ee.FeatureCollection | ee.Geometry):
            The area of interest (AOI). If a FeatureCollection is passed, its
            `.geometry()` (union) is used as the AOI. Geodesic area is used to
            pick an analysis scale that avoids oversampling large polygons.

        sources (list[str]):
            Source identifiers to bin and (optionally) combine. Supported built-ins:
            - 'elev'   : ArcticDEM elevation (meters).
            - 'hand'   : Height Above Nearest Drainage; automatically selects an
                        appropriate HAND product (30m/90m; 0–100 or 0–1000 range).
            - 'aspect' : Aspect (degrees) derived from the DEM.
            - 'cti'    : Compound Topographic Index (dimensionless), mosaicked.
            Custom source via user-supplied image is also supported: pick any unique
            key (e.g., 'myvar') and provide `binning['myvar']['image'] = ee.Image(...)`
            (single-band, already mosaicked/selected). See `binning` below.

        binning (dict[str, dict]):
            Per-source binning specification. For each `sid in sources`, provide a
            dictionary describing *how* to make bins for that source. Common keys:

            For **numeric** sources ('elev', 'hand', 'cti', or a custom image):
            - 'strategy' (str, required): one of
                    'percentiles'   # equal-area bins over the AOI (empirical quantiles)
                    'equalwidth'    # equal-width bins between min/max in the AOI
                    'fixed'         # user-specified edges (monotonic list of floats)
            - if 'percentiles' or 'equalwidth':
                    'n_bins' (int): number of bins (default 5)
            - if 'fixed':
                    'edges' (list[float]): bin breakpoints; length = n_bins + 1
                    Example: [0, 1, 2, 5, 10, 100, 1000]  → 6 bins
            - Optional cosmetics:
                    'label_prefix' (str): short prefix used in labels (e.g., 'ELEV', 'HAND')
                    'units' (str): only for *custom* images; used in plotting labels
                    'name' (str): only for *custom* images; human-readable source name
            - For **custom image** only:
                    'image' (ee.Image, single-band): pre-prepared raster. You must
                    handle `.mosaic()` (if coming from an ImageCollection) and `.select(...)`
                    yourself **before** passing it in. The function validates it is single-band.

            For **aspect** (circular data):
            - 'strategy' must be 'fixed'.
            - 'ranges' (list[tuple]): list of (start_deg, end_deg, label) ranges; wrap
                across 360→0 is supported by letting start > end.
                Example (N/S):
                    [(270, 90, 'N'), (90.01, 269.99, 'S')]
                Example (4 winds):
                    [(315, 45, 'N'), (45, 135, 'E'), (135, 225, 'S'), (225, 315, 'W')]
            - Optional: 'label_prefix' (e.g., 'ASP').

        combine (str, default 'cartesian'):
            How to combine per-source bins into final topounits:
            - 'cartesian'    : full cross-product of bins across sources
                                (e.g., 3 elev × 2 aspect = 6 units).
            - 'hierarchical' : apply bins in a priority order; each level further
                                subdivides only where the parent has pixels. Useful
                                when the cartesian product would explode.

        combine_order (list[str] | None, default None):
            Required when `combine='hierarchical'`. The ordered list of `sources`
            to apply (e.g., ['elev', 'hand'] means “bin elevation, then within each
            elevation bin, subdivision by HAND”). Ignored for 'cartesian'.

        max_topounits (int, default 256):
            Safety cap on the number of combined bins emitted. For 'cartesian', if the
            theoretical product exceeds this, only the first `max_topounits` are materialized.
            For 'hierarchical', recursion stops once the cap is reached.

        dem_source (str, default 'arcticdem'):
            DEM used for 'elev' and 'aspect'. Currently only 'arcticdem' is supported.

        return_as (str, default 'gdf'):
            What to return:
            - 'gdf'   : a `geopandas.GeoDataFrame` with one row per **bin** (unioned
                        geometry of all patches in that bin). If direct download via
                        `FeatureCollection.getInfo()` fails, the function triggers an
                        export to Drive and returns `None`.
            - 'asset' : trigger export to Google Drive using `gu.export_fc(...)` and
                        return `None`.

        export_scale (str | float, default 'native'):
            Pixel scale (meters) used for vectorization (`reduceToVectors`). If 'native',
            uses the chosen `analysis_scale` (see below). You may pass a numeric meter
            value to override.

        asset_name (str, default 'topounits'):
            Name used when exporting (folder is 'topotest' via `gu.export_fc`).

        asset_ftype (str, default 'GeoJSON'):
            Export file type passed through to `gu.export_fc`. (E.g., 'GeoJSON', 'SHP'.)

        min_patch_pixels (int | None, default None):
            If set, tiny slivers are dropped before vectorization by requiring each
            connected component to have at least this many pixels (in the analysis
            scale grid). Example: 9 retains components ≥ 3×3 pixels.

        target_pixels_per_topounit (int, default 500):
            Controls how coarse the **analysis scale** will be when it is auto-selected
            from AOI area. Roughly, the AOI is divided into `planned_bins × target_pixels_per_topounit`
            pixels; the square root of that per-bin area yields the scale in meters.
            Larger values → coarser analysis.

        target_scale (float | None, default None):
            If set (meters), directly influences the **analysis scale**; the final
            scale becomes `max(target_scale, coarsest_native_scale_across_sources)`.
            Use this to keep large AOIs manageable (e.g., 90 or 120 m).

        verbose (bool, default False):
            Print diagnostics, including the chosen `analysis_scale`.

    How sources are built:
        - 'elev'   : ArcticDEM V4 2 m mosaic, band 'elevation'.
        - 'hand'   : Chooses among 30m/100m and 30m/1000m or 90m/1000m HAND products.
                    If you use fixed edges with values >100, a *_1000 product is preferred.
                    If `target_scale` is set, we prefer a product that is not finer than it.
        - 'aspect' : Computed from the DEM (degrees 0–360).
        - 'cti'    : `projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti` (mosaicked).
        - custom   : If `binning[sid]['image']` is present, it must be a **single-band ee.Image**.
                    You must `.mosaic()` and `.select(...)` yourself beforehand. We only validate
                    it is single-band and rename that band to 'v'.

    Scale selection (important):
        - Let A be AOI area (m²) and B be the **planned** number of bins
        (product of per-source bin counts; for aspect, number of ranges).
        - If `target_scale` is None:
            analysis_scale ≈ sqrt( A / (B × target_pixels_per_topounit) )
        Then: analysis_scale = max(analysis_scale, coarsest_native_source_scale).
        - If `target_scale` is provided:
            analysis_scale = max(target_scale, coarsest_native_source_scale).
        - This scale is used in `sample`, `reduceRegion`, and `reduceToVectors`.

    Output:
        If `return_as='gdf'`, a GeoDataFrame with (at minimum) these columns:
        - geometry                  : union geometry for the bin.
        - band_name                 : stable ID like 'topounit_elev1__aspect2'.
        - schema                    : 'cartesian' or 'hierarchical'.
        - source_ids (list[str])    : sources used for this bin (e.g., ['elev','aspect']).
        - labels (dict)             : per-source human labels (e.g., {'elev': 'ELEV_0-100', 'aspect': 'ASP_N'}).
        - bin_bounds (dict)         : per-source numeric bounds or angular ranges.
        - bin_method (dict)         : per-source method (e.g., 'percentiles', 'fixed').
        - analysis_scale_m (float)  : the final analysis scale used for the run.
        - planned_counts (dict)     : planned #bins per source (before combination).
        - combine, max_topounits, target_pixels_per_topounit, target_scale.

    Raises:
        ValueError:
            - Unknown `combine` strategy.
            - `combine='hierarchical'` without `combine_order`.
            - Unsupported binning strategy for a source.
            - Fixed-edge binning without ≥ 2 edges.
            - Aspect given a non-'fixed' strategy.
            - Custom image provided with multiple bands.
        KeyError:
            - Unsupported `dem_source` for 'elev'/'aspect'.

    Notes:
        • One row per **bin** (union of all patches for that bin). If you prefer one
        feature per *patch*, vectorization logic would need to keep all polygons
        from `reduceToVectors` instead of the union geometry.
        • For large AOIs, use `target_scale` and/or `min_patch_pixels` to balance
        performance and polygon cleanliness.
        • HAND auto-selection is heuristic and aims to avoid oversampling; you can
        still override the scale via `target_scale`.

    Examples (not-exhaustive):
        # Elevation percentiles (equal-area):
        gdf = make_topounits(
            feature=feature,
            sources=['elev'],
            binning={'elev': {'strategy': 'percentiles', 'n_bins': 5, 'label_prefix': 'ELEV'}},
            combine='cartesian',
            target_scale=90,
            return_as='gdf'
        )

        # HAND with fixed hydrologic thresholds (meters):
        gdf = make_topounits(
            feature=feature,
            sources=['hand'],
            binning={'hand': {'strategy': 'fixed', 'edges': [0,1,2,5,10,100,1000], 'label_prefix': 'HAND'}},
            combine='cartesian',
            return_as='gdf'
        )

        # Elevation × Aspect (N/S), Cartesian:
        aspects = [(270, 90, 'N'), (90.01, 269.99, 'S')]
        gdf = make_topounits(
            feature=feature,
            sources=['elev','aspect'],
            binning={
                'elev':   {'strategy': 'percentiles', 'n_bins': 4, 'label_prefix': 'ELEV'},
                'aspect': {'strategy': 'fixed', 'ranges': aspects, 'label_prefix': 'ASP'}
            },
            combine='cartesian',
            max_topounits=20,
            target_scale=90,
            return_as='gdf'
        )

        # Custom user image (single-band, already prepared):
        my_img = ee.Image('users/me/mystack').mosaic().select('band_of_interest')
        gdf = make_topounits(
            feature=feature,
            sources=['myvar'],
            binning={'myvar': {'image': my_img, 'strategy': 'equalwidth', 'n_bins': 6, 'label_prefix': 'MYVAR'}},
            combine='cartesian',
            return_as='gdf'
        )
    """
    # --- Normalize AOI: Domain or raw feature/geometry ---
    #
    # If a Domain is passed, use the union of its cell geometries as the AOI.
    # Otherwise, keep the existing behavior (let gee_utils._geom_from_any handle it).
    if isinstance(feature, Domain):
        if feature.gdf.empty:
            raise ValueError("Domain has no geometries; cannot build topounits.")
        aoi_geom = feature.gdf.unary_union  # shapely geometry in EPSG:4326
        feature_for_gee = aoi_geom
    else:
        feature_for_gee = feature

    region = gu._geom_from_any(feature_for_gee)

    # 1) Planned total bins (product across sources)
    def _planned_bins_for_source(sid):
        spec = binning[sid]
        if sid == 'aspect' and spec.get('strategy') == 'fixed':
            rngs = spec.get('ranges') or [(270, 90, 'N'), (90.01, 269.99, 'S')]
            return len(rngs)
        st = spec.get('strategy')
        if st in ('percentiles', 'equalwidth'):
            return int(spec.get('n_bins', 5))
        if st == 'fixed':
            edges = spec.get('edges', [])
            return max(0, len(edges) - 1)
        raise ValueError(f"Unknown/unsupported strategy for planned bin count: {st}")

    planned_counts = {sid: _planned_bins_for_source(sid) for sid in sources}
    total_planned = 1
    for v in planned_counts.values():
        total_planned *= max(1, v)

    # 2) Build sources (no reproject), record native scales
    source_images = {}
    native_scales = {}
    source_meta = {}
    desired_scale_hint = target_scale  # may be None

    for sid in sources:
        img, nat_scale, meta = build_source(
            sid,
            feature=feature_for_gee,   # <-- changed from 'feature'
            desired_scale_hint=desired_scale_hint,
            binning_spec=binning[sid],
            dem_source=dem_source,
            verbose=verbose
        )
        source_images[sid] = img  # single band 'v', already clipped
        native_scales[sid] = float(nat_scale)
        source_meta[sid] = meta

    # 3) Choose analysis scale from AOI area or user target; clamp to coarsest native
    coarsest_native = max(native_scales.values())
    if target_scale is not None:
        analysis_scale = max(float(target_scale), coarsest_native)
    else:
        aoi_area = region.area()  # m^2
        required_pixel_area = aoi_area.divide(total_planned * target_pixels_per_topounit)
        analysis_scale = float(required_pixel_area.sqrt().getInfo())
        analysis_scale = max(analysis_scale, coarsest_native)

    if verbose:
        print(f"[Scale] planned bins={total_planned}, coarsest_native={coarsest_native:.1f} m, "
              f"analysis_scale={analysis_scale:.1f} m")

    # 4) Build per-source bins and masks (no reprojection)
    bin_defs_by_source = {}
    bin_masks_by_source = {}

    for sid in sources:
        img = source_images[sid]
        # Build bin definitions
        bin_defs = build_bins_for_source(
            sid,
            img,
            region=region,
            binning_spec=binning[sid],
            analysis_scale=analysis_scale,
            aspect_ranges_default=binning.get('aspect', {}).get('ranges') if 'aspect' in binning else None
        )
        bin_defs_by_source[sid] = bin_defs

        # Build masks per bin
        entries = []
        if sid == 'aspect':
            for b in bin_defs:
                mask = _aspect_bin_mask(img, b['start'], b['end'], b['wrap'])
                entries.append({'def': b, 'mask': mask})
        else:
            for b in bin_defs:
                mask = _numeric_bin_mask(img, b['low'], b['high'])
                entries.append({'def': b, 'mask': mask})
        bin_masks_by_source[sid] = entries

    # 5) Combine masks
    if combine == 'cartesian':
        combined_entries = _combine_cartesian(
            bin_masks_by_source,
            max_topounits=max_topounits,
            min_patch_pixels=min_patch_pixels
        )
        if len(combined_entries) == 0:
            raise RuntimeError("No combined topounit masks produced (cartesian).")
    elif combine == 'hierarchical':
        if not combine_order:
            raise ValueError("combine='hierarchical' requires combine_order (e.g., ['elev','hand']).")
        combined_entries = _combine_hierarchical(
            combine_order,
            bin_masks_by_source,
            max_topounits=max_topounits,
            min_patch_pixels=min_patch_pixels
        )
        if len(combined_entries) == 0:
            raise RuntimeError("No combined topounit masks produced (hierarchical).")
    else:
        raise ValueError(f"Unknown combine strategy: {combine}")

    # 6) Export scale
    if export_scale == 'native':
        export_scale = analysis_scale

    # 7) Attach reproducibility props
    extra_props = {
        'analysis_scale_m': analysis_scale,
        'sources': sources,
        'planned_counts': planned_counts,
        'combine': combine,
        'max_topounits': max_topounits,
        'target_pixels_per_topounit': target_pixels_per_topounit,
        'target_scale': target_scale
    }

    # 8) Vectorize to polygons
    polygons_fc = gu.masks_to_featurecollection(
        combined_entries,
        region=region,
        export_scale=export_scale,
        extra_image_props=extra_props
    )

    # 9) Return or export
    if return_as == 'gdf':
        gdf = gu.try_to_download_featurecollection(polygons_fc, verbose=verbose)
        if gdf is None:
            print("Could not return as GeoDataFrame; exporting to Google Drive. Check Tasks in your GEE browser.")
            gu.export_fc(polygons_fc, f'{asset_name}', asset_ftype, folder='topotest', verbose=True)
            return None
        return gdf
    else:
        gu.export_fc(polygons_fc, f'{asset_name}', asset_ftype, folder='topotest', verbose=True)
        return None

