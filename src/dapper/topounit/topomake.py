"""Topographic unit generation utilities."""

try:
    import ee  # type: ignore
except Exception:  # pragma: no cover
    ee = None  # type: ignore

def _require_ee_global():
    global ee
    if ee is None:
        try:
            import ee as _ee  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Google Earth Engine (earthengine-api) is required for topounit creation. "
                "Install it (pip install earthengine-api) and authenticate (earthengine authenticate)."
            ) from e
        ee = _ee
    return ee

if ee is None:  # pragma: no cover
    class _EEProxy:
        def __getattr__(self, name):
            return getattr(_require_ee_global(), name)

    ee = _EEProxy()  # type: ignore

import math

from dapper.domains.domain import Domain
from dapper.integrations.earthengine import gee_utils as gu


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
    """Auto-select among available HAND products based on a desired scale (m)
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
# Topounit characteristics sampling 
# -----------------------------------------
def _attach_topounit_terrain_stats(
    polygons_fc,
    feature_for_gee,
    dem_source,
    analysis_scale,
    verbose=False,
):
    """
    Attach per-topounit terrain properties needed for topounit-enabled surface files:

        - TopounitAveElv  : mean elevation    [m]
        - TopounitSlope   : mean slope        [deg]
        - TopounitAspect  : mean aspect       [deg, 0–360, cw from north]

    Parameters
    ----------
    polygons_fc : ee.FeatureCollection
        One feature per topounit (unioned geometry).
    feature_for_gee :
        Same AOI object given to make_topounits (ee.Feature/Geometry/etc.).
    dem_source : str
        DEM source used by 'elev' (currently 'arcticdem').
    analysis_scale : float
        Scale in meters at which to compute zonal statistics.
    verbose : bool
        If True, prints a short status message.
    """
    # Reuse the same DEM source as the 'elev' topounit source.
    dem_img, _native_scale, _meta = build_source(
        source_id='elev',
        feature=feature_for_gee,
        desired_scale_hint=analysis_scale,
        binning_spec={},      # not used for 'elev'
        dem_source=dem_source,
        verbose=verbose,
    )

    # Elevation band (meters)
    dem = dem_img.rename('elev')

    # Slope in degrees
    slope_deg = ee.Terrain.slope(dem).rename('slope_deg')

    # Aspect in degrees (0–360, clockwise from north)
    aspect_deg = ee.Terrain.aspect(dem).rename('aspect_deg')

    # For a proper mean aspect, use circular mean of sin/cos
    aspect_rad = aspect_deg.multiply(math.pi / 180.0)
    aspect_sin = aspect_rad.sin().rename('aspect_sin')
    aspect_cos = aspect_rad.cos().rename('aspect_cos')

    # Multi-band image for one-shot reduceRegion
    stats_image = (
        dem
        .addBands(slope_deg)
        .addBands(aspect_deg)
        .addBands(aspect_sin)
        .addBands(aspect_cos)
    )

    def _add_stats(feat):
        geom = feat.geometry()

        stats = stats_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=analysis_scale,
            maxPixels=1e13,
            tileScale=2,
        )

        elev_mean      = stats.get('elev')         # m
        slope_mean_deg = stats.get('slope_deg')    # deg
        mean_sin       = stats.get('aspect_sin')
        mean_cos       = stats.get('aspect_cos')

        # Circular mean of aspect (degrees 0–360)
        aspect_mean_rad = ee.Number(mean_sin).atan2(ee.Number(mean_cos))
        aspect_mean_deg = aspect_mean_rad.multiply(180.0 / math.pi).mod(360.0)

        return feat.set({
            'TopounitAveElv': elev_mean,
            'TopounitSlope':  slope_mean_deg,
            'TopounitAspect': aspect_mean_deg,
        })

    fc_with_stats = polygons_fc.map(_add_stats)

    if verbose:
        print("[topounits] Attached terrain stats: TopounitAveElv, TopounitSlope, TopounitAspect.")

    return fc_with_stats

# -----------------------------------------
# Main API 
# -----------------------------------------
def make_topounits(
    aoi,
    sources=None,                 # if None, inferred from binning keys (insertion order)
    binning=None,                 # dict: { sid: {...}, ... }
    combine='cartesian',
    combine_order=None,
    max_topounits=256,
    dem_source='arcticdem',
    return_as='gdf',
    export_scale='native',
    asset_name='topounits',
    asset_ftype='GeoJSON',
    min_patch_pixels=None,
    target_pixels_per_topounit=500,
    target_scale=None,
    verbose=False
):
    """
    SINGLE-AOI topounit builder.

    aoi can be:
      - Domain (must have exactly 1 geometry in domain.support / domain.gdf)
      - shapely geometry
      - ee.Geometry / ee.Feature / ee.FeatureCollection
      - str (GEE FeatureCollection asset id)

    If sources is None, sources = list(binning.keys()) (dict insertion order).
    """
    if binning is None or not isinstance(binning, dict) or len(binning) == 0:
        raise ValueError("make_topounits requires a non-empty 'binning' dict.")

    # Infer sources from binning (keeps dict insertion order)
    if sources is None:
        sources = list(binning.keys())

    # Validate sources
    missing = [sid for sid in sources if sid not in binning]
    if missing:
        raise KeyError(f"Sources missing from binning: {missing}. Provide binning['sid'] for each source.")

    # Coerce AOI input to a single shapely/ee object, then to ee.Geometry
    if isinstance(aoi, Domain):
        gdf = getattr(aoi, "support", None)
        if gdf is None:
            gdf = getattr(aoi, "gdf", None)  # legacy fallback
        if gdf is None or gdf.empty:
            raise ValueError("Domain has no geometries; cannot build topounits.")

        if len(gdf) != 1:
            raise ValueError(
                "make_topounits(aoi=Domain) expects a single-geometry Domain. "
                "Use make_topounits_for_domain(domain, ...) for multi-geometry Domains."
            )

        aoi_obj = gdf.geometry.iloc[0]  # shapely geom
    else:
        aoi_obj = aoi

    # IMPORTANT: always use ee.Geometry for region-based EE ops (sample/reduceRegion/reduceToVectors)
    region = gu.parse_geometry_object(aoi_obj, name="aoi")

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

    # 2) Build sources, record native scales
    source_images = {}
    native_scales = {}
    source_meta = {}
    desired_scale_hint = target_scale

    # Pass region (ee.Geometry) downstream; build_source uses gu._geom_from_any(feature)
    for sid in sources:
        img, nat_scale, meta = build_source(
            sid,
            feature=region,
            desired_scale_hint=desired_scale_hint,
            binning_spec=binning[sid],
            dem_source=dem_source,
            verbose=verbose
        )
        source_images[sid] = img
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

    # 4) Build per-source bins and masks
    bin_defs_by_source = {}
    bin_masks_by_source = {}

    for sid in sources:
        img = source_images[sid]
        bin_defs = build_bins_for_source(
            sid,
            img,
            region=region,
            binning_spec=binning[sid],
            analysis_scale=analysis_scale,
            aspect_ranges_default=binning.get('aspect', {}).get('ranges') if 'aspect' in binning else None
        )
        bin_defs_by_source[sid] = bin_defs

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

    # Attach terrain stats (uses DEM; region is fine as "feature")
    polygons_fc = _attach_topounit_terrain_stats(
        polygons_fc=polygons_fc,
        feature_for_gee=region,
        dem_source=dem_source,
        analysis_scale=analysis_scale,
        verbose=verbose,
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


def add_topounit_image_samples(
    topounits_gdf,
    sampling_specs,
    default_scale=None,
    ensure_pixel_centers=True,
    verbose=False,
):
    """Attach additional sampled properties from arbitrary GEE images to each
    topounit.

    Parameters
    ----------
    topounits_gdf : GeoDataFrame
        Output from make_topounits(return_as='gdf'). Must contain 'band_name'.
    sampling_specs : list[dict]
        Each dict describes one sampled variable with keys:
            - 'image'   : ee.Image or image ID string
            - 'band'    : optional band name (if omitted, image must be single-band)
            - 'reducer' : str {'mean','min','max','std'} or ee.Reducer
            - 'out_name': name of the column to add
            - 'scale'   : optional scale [m]; overrides default_scale
    default_scale : float, optional
        Default scale [m] for specs that do not define 'scale'. If None, uses
        topounits_gdf['analysis_scale_m'].iloc[0] if present; otherwise the
        nominal scale of each image.
    ensure_pixel_centers : bool, default True
        If True, guard against polygons with no pixel centers.
    verbose : bool, default False
        Print which variables were attached.

    Returns
    -------
    GeoDataFrame
        Copy of topounits_gdf with new columns added.
    """
    if not sampling_specs:
        return topounits_gdf

    gdf = topounits_gdf.copy()

    # Default scale: prefer analysis_scale_m if no explicit default
    if default_scale is None and "analysis_scale_m" in gdf.columns:
        default_scale = float(gdf["analysis_scale_m"].iloc[0])

    attached_names = []

    for spec in sampling_specs:
        if "image" not in spec:
            raise KeyError("Each sampling spec must include an 'image' key.")

        img_spec = spec["image"]
        if isinstance(img_spec, ee.Image):
            img = img_spec
        elif isinstance(img_spec, str):
            img = ee.Image(img_spec)
        else:
            raise TypeError(
                f"sampling_specs['image'] must be an ee.Image or image ID string; got {type(img_spec)}"
            )

        band = spec.get("band")
        reducer = spec.get("reducer", "mean")
        out_name = spec.get("out_name")
        scale = spec.get("scale", default_scale)

        # If still None, fall back to image's native scale
        if scale is None:
            scale = gu._nominal_scale_m(img)

        gdf = gu.sample_image_over_polygons(
            gdf,
            image=img,
            geometry_id_field="band_name",
            band=band,
            reducer=reducer,
            out_name=out_name,
            scale=scale,
            ensure_pixel_centers=ensure_pixel_centers,
            verbose=verbose,
        )
        # sample_image_over_polygons returns the final out_name it used
        if out_name is None:
            # If user didn't set out_name, recompute default name
            if isinstance(reducer, str):
                key = reducer.lower()
            else:
                key = "custom"
            if band is None:
                # This is rare (single-band images), but be robust
                band_names = img.bandNames().getInfo()
                band_used = band_names[0]
            else:
                band_used = band
            out_name = f"{band_used}_{key}"

        attached_names.append(out_name)

    if verbose:
        print(
            "[topounits] Attached sampled properties: "
            + ", ".join(attached_names)
        )

    return gdf


def make_topounits_for_domain(
    domain: Domain,
    *,
    sources,
    binning,
    combine="cartesian",
    combine_order=None,
    max_topounits=256,
    dem_source="arcticdem",
    export_scale="native",
    min_patch_pixels=None,
    target_pixels_per_topounit=500,
    target_scale=None,
    verbose=False,
    allow_slow_ncells: int = 25,
):
    """
    Compute topounits per-domain-geometry and attach them to the Domain.

    - For domain.mode == 'sites': this computes per-run (each run is 1 geom)
      and returns a Domain with a concatenated topounits table; iter_runs()
      can later subset by gid.
    - For domain.mode == 'cellset': loops over each cell geometry and computes
      topounits for each gid separately (can be slow for large N).

    Returns
    -------
    Domain
        domain with domain.topounits populated (expects Domain.with_topounits exists).
    """
    import pandas as pd
    import geopandas as gpd
    from pyproj import Geod

    # Choose which geometry view to use for topounits (support is the right one)
    gdf = getattr(domain, "support", None)
    if gdf is None:
        gdf = getattr(domain, "gdf", None)
    if gdf is None or gdf.empty:
        raise ValueError("Domain has no geometries; cannot compute topounits.")

    if "gid" not in gdf.columns:
        raise KeyError("Domain geometry table must include a 'gid' column.")

    gdf = gdf.copy()
    gdf["gid"] = gdf["gid"].astype(str)

    ncells = len(gdf)
    if ncells > allow_slow_ncells and not verbose:
        raise ValueError(
            f"make_topounits_for_domain would run {ncells} separate GEE topounit builds. "
            f"Set verbose=True to proceed intentionally, or raise allow_slow_ncells."
        )

    geod = Geod(ellps="WGS84")

    def _geod_area_m2(geom) -> float:
        # returns signed area; take abs
        try:
            a, _p = geod.geometry_area_perimeter(geom)
            return float(abs(a))
        except Exception:
            # fallback: 0 rather than crashing; user will see pct=nan
            return float("nan")

    parts = []
    for row in gdf.itertuples(index=False):
        gid = str(getattr(row, "gid"))
        geom = getattr(row, "geometry")

        if verbose:
            print(f"[topounits] gid={gid}: building topounits...")

        tu = make_topounits(
            aoi=geom,
            sources=sources,
            binning=binning,
            combine=combine,
            combine_order=combine_order,
            max_topounits=max_topounits,
            dem_source=dem_source,
            return_as="gdf",
            export_scale=export_scale,
            min_patch_pixels=min_patch_pixels,
            target_pixels_per_topounit=target_pixels_per_topounit,
            target_scale=target_scale,
            verbose=verbose,
        )
        
        if tu is None:
            raise RuntimeError(
                f"make_topounits returned None for gid={gid} "
                "(likely fell back to Drive export). Use a smaller AOI/scale or run in an env where FC download works."
            )

        tu = tu.copy()
        tu["gid"] = gid

        # Stable ids across multi-cell: prefix band_name with gid
        if "band_name" not in tu.columns:
            raise KeyError("Topounits output must include 'band_name'.")
        tu["topounit_id"] = tu["gid"].astype(str) + "__" + tu["band_name"].astype(str)

        # Compute area fraction within the parent geometry (percent)
        cell_area = _geod_area_m2(geom)
        if not (cell_area and cell_area > 0):
            tu["TopounitArea_m2"] = float("nan")
            tu["TopounitPctOfCell"] = float("nan")
        else:
            # Intersection is defensive; should already be inside geom
            inter = tu.geometry.intersection(geom)
            tu["TopounitArea_m2"] = inter.apply(_geod_area_m2)
            tu["TopounitPctOfCell"] = 100.0 * (tu["TopounitArea_m2"] / cell_area)

        parts.append(tu)

    all_topounits = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=gdf.crs)

    # Attach to Domain (expects you implemented Domain.with_topounits)
    return domain.with_topounits(
        all_topounits,
        id_col="topounit_id",
        gid_col="gid",
        dim_name="topounit",
    )