import ee
import geopandas as gpd
import json

def compute_sampling_scale(feature, n, target_pixels_per_topounit=500):
    """
    Computes an approximate scale (in meters) for sampling ArcticDEM,
    aiming for n_bands * target_pixels_per_band total pixels.

    Parameters:
    - feature: ee.Feature or ee.Geometry
    - n: int, number of topounits
    - target_pixels_per_band: int, desired number of pixels per band

    Returns:
    - ee.Number: scale in meters
    """
    total_target_pixels = n * target_pixels_per_topounit

    # Ensure geometry
    geom = feature.geometry() if isinstance(feature, ee.Feature) else feature

    # Get area in square meters
    area_m2 = geom.area()

    # Required pixel size in m²
    required_pixel_area = area_m2.divide(total_target_pixels)

    # Scale is the square root of pixel area
    scale = required_pixel_area.sqrt()

    return scale.getInfo()


def make_topounits(feature, n, method='epercentiles', dem_source='arcticdem', ftype='gdf', export_scale='native'):
    """
    Creates topounits based on a provided set of geometries (polygons or multipolygons)
    already cast as an ee.Feature.
    
    n represents the number of topounits wanted (in elevation percentiles or elev x aspect).
    method: 'epercentiles' for elevation-only, or 'elevaspect' for elevation + aspect.
    dem_source: currently only 'arcticdem' is supported.
    ftype: 'geodataframe' or 'imagecollection' (not fully implemented here).
    export_scale: can be 'native' or numeric value.
    """

    # Load DEM
    if dem_source == 'arcticdem':
        dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
        scale_binning = compute_sampling_scale(feature, n)
        scale_binning = max(scale_binning, 5)  # enforce max resolution to avoid memory issues
    else:
        raise KeyError('Your selection for dem in Topounit generation is not yet supported.')

    dem_clipped = dem.clip(feature.geometry())
    aspect_img = ee.Terrain.aspect(dem_clipped)

    # Sample elevation values
    elev_samples = dem_clipped.sample(region=feature.geometry(), scale=scale_binning, geometries=False)
    elevations = elev_samples.aggregate_array('elevation')
    elevations_list = ee.List(elevations)
    count = elevations_list.size()
    percentiles = [i * (100 / n) for i in range(1, n)]
    thresholds = []

    for p in percentiles:
        index = ee.Number(p).multiply(count).divide(100).int()
        thresholds.append(elevations_list.sort().get(index))

    thresholds = ee.List(thresholds).getInfo()
    thresholds.insert(0, 0)
    thresholds.append(100000)

    ### --- Masking Methods --- ###

    def make_epercentiles_masks(dem, thresholds):
        def make_band(i):
            i = int(i)
            low = thresholds[i]
            high = thresholds[i + 1]
            topounit_id = i + 1
            mask = dem.gte(low).And(dem.lt(high)).selfMask()
            return mask.rename(f"topounit_{topounit_id}")

        indices = range(len(thresholds) - 1)
        bands = [make_band(i) for i in indices]
        return add_epercentiles_metadata(ee.ImageCollection.fromImages(bands).toBands(), thresholds)

    def make_elevaspect_masks(dem, elev_thresholds, aspect_classes):
        bands = []
        bin_id = 1

        for i in range(len(elev_thresholds) - 1):
            low = elev_thresholds[i]
            high = elev_thresholds[i + 1]
            elev_mask = dem.gte(low).And(dem.lt(high)).selfMask()

            for aspect_code, aspect_mask in aspect_classes:
                combined_mask = elev_mask.And(aspect_mask)
                band = combined_mask.rename(f"topounit_{bin_id}")
                bands.append(band)
                bin_id += 1

        return add_elevaspect_metadata(ee.ImageCollection.fromImages(bands).toBands(), elev_thresholds, aspect_classes)

    def add_epercentiles_metadata(image, thresholds):
        band_info = [
            {'topounit_id': i + 1, 'min_elev': thresholds[i], 'max_elev': thresholds[i + 1]}
            for i in range(len(thresholds) - 1)
        ]
        return image.set({'topounit_metadata': band_info})

    def add_elevaspect_metadata(image, elev_thresholds, aspect_classes):
        metadata = []
        bin_id = 1

        for i in range(len(elev_thresholds) - 1):
            low = elev_thresholds[i]
            high = elev_thresholds[i + 1]

            for aspect_code, _ in aspect_classes:
                metadata.append({
                    'topounit_id': bin_id,
                    'min_elev': float(low),
                    'max_elev': float(high),
                    'aspect': aspect_code
                })
                bin_id += 1

        return image.set({'topounit_metadata': metadata})

    # Select method
    if method == 'epercentiles':
        mask = make_epercentiles_masks(dem_clipped, thresholds)

    elif method == 'elevaspect':
        north_mask = aspect_img.lte(180).selfMask()
        south_mask = aspect_img.gt(180).selfMask()
        aspect_classes = [('N', north_mask), ('S', south_mask)]
        mask = make_elevaspect_masks(dem_clipped, thresholds, aspect_classes)

    else:
        raise ValueError(f"Unsupported method: {method}")

    ### --- Vectorize --- ###

    def masks_to_featurecollection(mask_image, region, scale):
        band_names = mask_image.bandNames().getInfo()
        metadata_list = mask_image.get('topounit_metadata').getInfo()
        features = []

        for band_name in band_names:
            band = mask_image.select(band_name)
            vectors = band.reduceToVectors(
                geometry=region,
                scale=scale,
                geometryType='polygon',
                eightConnected=False,
                bestEffort=True,
                maxPixels=1e13
            )
            merged_geom = vectors.geometry()
            topounit_id = int(band_name.split('_')[-1])
            meta = next(m for m in metadata_list if m['topounit_id'] == topounit_id)
            feature = ee.Feature(merged_geom, meta)
            features.append(feature)

        return ee.FeatureCollection(features)

    if export_scale == 'native':
        if dem == 'arcticdem':
            export_scale = 2
        else:
            export_scale = 100
            
    polygons_fc = masks_to_featurecollection(
        mask_image=mask,
        region=feature.geometry(),
        scale=export_scale
    )

    ### --- Export --- ###
    def export_fc(fc, desc=None, fformat=None, folder=None, prefix=None):
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=desc,
            fileFormat=fformat,
            folder=folder,
            fileNamePrefix=prefix)
        task.start()
        return
    
    if ftype == 'gdf':
        gdf = try_to_download_featurecollection(polygons_fc)
        if gdf is None:
            print("🔁 Falling back to Google Drive export...")
            export_fc(polygons_fc, 
                      desc=f'topounit_{method}_export', 
                      fformat='GeoJSON',
                      folder='topotest',
                      prefix=f'topounits_{method}')
            return None
        return gdf    
    return


def try_to_download_featurecollection(fc):
    try:
        fc_geojson = fc.getInfo()  # May raise EEException
        gdf = gpd.GeoDataFrame.from_features(fc_geojson['features'])
        gdf.set_crs(epsg=4326, inplace=True)
 
        print("Success! FeatureCollection loaded as GeoDataFrame.")
        return gdf

    except Exception as e:
        print("⚠️ Direct download failed. Reason:", e)
        
        return None  # or raise if you want downstream logic to handle this


from dapper.gee import gee_utils as gutils
ee.Initialize(project='ee-jonschwenk')
f = gutils.parse_geometry_objects('projects/ee-jonschwenk/assets/E3SM/Kuparuk_gageshed')
feature = ee.Feature(f.first())

gdf = make_topounits(feature, 5, method='elevaspect', dem_source='arcticdem', ftype='gdf', export_scale=200)
gdf.to_file(r'X:\Research\NGEE Arctic\6. Topounits\elevaspect3.json', driver='GeoJSON')
