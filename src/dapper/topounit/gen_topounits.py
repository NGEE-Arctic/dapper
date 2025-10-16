import ee
import geopandas as gpd
from dapper.utils import gee_utils as gu

# ----------------------------
# Utilities
# ----------------------------

def try_to_download_featurecollection(fc, verbose=True):
    """Attempt to load FeatureCollection as a GeoDataFrame; else return None."""
    try:
        fc_geojson = fc.getInfo()  # May raise EEException on large/complex geoms
        gdf = gpd.GeoDataFrame.from_features(fc_geojson['features'])
        gdf.set_crs(epsg=4326, inplace=True)
        if verbose:
            print("Success! FeatureCollection loaded as GeoDataFrame.")
        return gdf
    except Exception as e:
        if verbose:
            print("Direct download failed. Reason:", e)
        return None

def _geom_from_any(x):
    """Return ee.Geometry from ee.Feature, ee.FeatureCollection, or ee.Geometry."""
    if isinstance(x, ee.Feature):
        return x.geometry()
    if isinstance(x, ee.FeatureCollection):
        return x.geometry()
    return x  # assume ee.Geometry

def _nominal_scale_m(image):
    """Return nominal scale in meters for an ee.Image."""
    return float(image.projection().nominalScale().getInfo())


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
        {"name": "hand30_1000", "img": hand30_1000, "scale": _nominal_scale_m(hand30_1000), "maxval": 1000},
        {"name": "hand90_1000", "img": hand90_1000, "scale": _nominal_scale_m(hand90_1000), "maxval": 1000},
        {"name": "hand30_100",  "img": hand30_100,  "scale": _nominal_scale_m(hand30_100),  "maxval": 100},
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
    source_id in {'elev','hand','aspect'}.
    """
    region = _geom_from_any(feature)

    if source_id == 'elev':
        if dem_source == 'arcticdem':
            img = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").select('elevation').rename('v').clip(region)
        else:
            raise KeyError(f"DEM source '{dem_source}' not supported.")
        scale = _nominal_scale_m(img)
        return img, scale, {"units": "m", "name": "Elevation"}

    if source_id == 'hand':
        hand_edges = None
        if binning_spec.get('strategy') == 'fixed':
            hand_edges = binning_spec.get('edges')
        img, scale = choose_hand_image(desired_scale=desired_scale_hint, hand_edges=hand_edges, verbose=verbose)
        # Keep 0 as valid (channel), only mask nodata by existing mask
        img = img.updateMask(img.mask()).clip(region)
        return img, scale, {"units": "m", "name": "HAND"}

    if source_id == 'aspect':
        # Compute aspect from DEM (same DEM source); keep band name 'v'
        if dem_source != 'arcticdem':
            raise KeyError(f"DEM source '{dem_source}' not supported for aspect.")
        dem = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").select('elevation').clip(region)
        aspect = ee.Terrain.aspect(dem).rename('v')
        scale = _nominal_scale_m(dem)  # native DEM scale
        return aspect, scale, {"units": "deg", "name": "Aspect", "from_dem": True}

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
# Vectorization & export
# -----------------------------------------

def masks_to_featurecollection(mask_entries, region, export_scale, extra_image_props=None):
    """
    mask_entries: list of {'band_name','mask','meta'}
    Returns ee.FeatureCollection with metadata as properties.
    One feature per band (union of all polygons for that band).
    """
    features = []
    for entry in mask_entries:
        vectors = entry['mask'].reduceToVectors(
            geometry=region,
            scale=export_scale,
            geometryType='polygon',
            eightConnected=False,
            bestEffort=True,
            maxPixels=1e13
        )
        geom = vectors.geometry()  # union geometry of all parts; may be empty
        # Skip empty geometries (optional)
        feature = ee.Feature(geom, {
            'band_name': entry['band_name'],
            'schema': entry['meta'].get('topounit_schema'),
            'source_ids': entry['meta'].get('source_ids'),
            'labels': entry['meta'].get('labels'),
            'bin_bounds': entry['meta'].get('bin_bounds'),
            'bin_method': entry['meta'].get('bin_method'),
        })
        if extra_image_props:
            feature = feature.setMulti(extra_image_props)
        features.append(feature)
    return ee.FeatureCollection(features)


# -----------------------------------------
# Main flexible API (v2)
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
    Flexible topounit generator with multi-source + multi-strategy binning.
    No global image reprojection; GEE handles alignment implicitly.
    """
    region = _geom_from_any(feature)

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
            feature=feature,
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
    polygons_fc = masks_to_featurecollection(
        combined_entries,
        region=region,
        export_scale=export_scale,
        extra_image_props=extra_props
    )

    # 9) Return or export
    if return_as == 'gdf':
        gdf = try_to_download_featurecollection(polygons_fc, verbose=verbose)
        if gdf is None:
            print("Could not return as GeoDataFrame; exporting to Google Drive. Check Tasks in your GEE browser.")
            gu.export_fc(polygons_fc, f'{asset_name}', asset_ftype, folder='topotest', verbose=True)
            return None
        return gdf
    else:
        gu.export_fc(polygons_fc, f'{asset_name}', asset_ftype, folder='topotest', verbose=True)
        return None
