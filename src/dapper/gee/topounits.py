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


def make_topounits(feature, n, method='epercentiles', dem='arcticdem', ret='geodataframe'):
    """
    Creates topounits based on a provided set of geometries (polygons or multipolygons)
    already cast as an ee.Feature. 
    n represents the number of topounits wanted.
    method refers to how topounits should be computed; right now only epercentiles is available.
    Currently only ArcticDEM coverage although other DEMs are available for use.
    Can return an imageCollection ('imagecollection') of masks or a GeoDataFrame ('geodataframe') of vectorized polygons outlining the Topounits.
    """

    if dem == 'arcticdem':
        dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
        scale_binning = compute_sampling_scale(feature, n) # use 5 meter scale, adjustable based on size of geometry though
        scale_binning = max(scale_thresholding, 5)
    else:
        raise KeyError('Your selection for dem in Topounit generation is not yet supported.')
    
    # Clip the DEM to the area of interest
    dem_clipped = dem.clip(feature.geometry())
    
    # Sample elevation values within the polygon
    samples = dem_clipped.sample(region=feature.geometry(), scale=scale_binning, geometries=False)
    
    # Get the list of elevation values
    elevations = samples.aggregate_array('elevation')
        
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

    ## Masking
    # Creating an image of masks; each band represents a different topounit
    def make_masks(dem, thresholds):
        def make_band(i):
            i = int(i)  # Cast ee.Number to Python int
            low = thresholds[i]
            high = thresholds[i + 1]
            topounit_id = i + 1  # IDs start at 1
            mask = dem.gte(low).And(dem.lt(high)).selfMask()
            return mask.rename(f"topounit_{topounit_id}")

        indices = range(len(thresholds) - 1)
        bands = [make_band(i) for i in indices]
        return ee.ImageCollection.fromImages(bands).toBands()

    # To make sure topounit id's are attached to each band
    def add_metadata(image, thresholds):
        band_info = [
            {'topounit_id': i + 1, 'min_elev': thresholds[i], 'max_elev': thresholds[i + 1]}
            for i in range(len(thresholds) - 1)
        ]
        return image.set({'topounit_metadata': band_info})

    mask = make_masks(dem, thresholds)
    mask = add_metadata(mask, thresholds)

    ## Polygonizing
    def masks_to_featurecollection(mask_image, region, scale):
        band_names = mask_image.bandNames().getInfo()  # e.g., ['topounit_1', 'topounit_2', ...]

        features = []

        for band_name in band_names:
            band = mask_image.select(band_name)

            # Convert to vectors
            vectors = band.reduceToVectors(
                geometry=region,
                scale=scale,
                geometryType='polygon',
                eightConnected=False,
                bestEffort=True,
                maxPixels=1e13
            )

            # Merge all geometries into one (multi)polygon
            merged_geom = vectors.geometry()

            # Extract topounit_id from band name, e.g., 'topounit_1' → 1
            topounit_id = int(band_name.split('_')[-1])

            # Create single feature with topounit_id
            feature = ee.Feature(merged_geom, {'topounit_id': topounit_id})
            features.append(feature)

        return ee.FeatureCollection(features)

    polygons_fc = masks_to_featurecollection(
        mask_image=mask,
        region=feature.geometry(),
        scale=100
    )


    ## Export Topounits
    task = ee.batch.Export.table.toDrive(
        collection=polygons_fc,
        description='topounit_test',
        fileFormat='GeoJSON',  # or SHP
        folder='topotest',
        fileNamePrefix='topounits'
    )
    task.start()




    # Creating an imagecollection of masks
    def make_bin_mask(dem, low, high, id_tu):
        mask = dem.gte(low).And(dem.lt(high)).selfMask()
        return mask.set('id_topounit', id_tu)

    # thresholds = list of elevation edges, e.g., [100, 120, 140, 160]
    # dem = your elevation image
    masks = [
        make_bin_mask(dem, thresholds[i], thresholds[i+1], i + 1)
        for i in range(len(thresholds) - 1)
    ]

    mask_collection = ee.ImageCollection(masks)
    # print(mask_collection.aggregate_array('id_topounit').getInfo())

    # Generate masks for each percentile band
    masks = []
    lower_bound = -float('inf')
    for i, percentile in enumerate(percentiles):
        upper_bound = thresholds[f'elevation_p{int(percentile)}']
        mask = dem_clipped.gte(lower_bound).And(dem_clipped.lt(upper_bound))
        masks.append(mask)
        lower_bound = upper_bound
    # Add the last band for the remaining elevations
    mask = dem_clipped.gte(lower_bound)
    masks.append(mask)
    
    return masks

from dapper.gee import gee_utils as gutils
ee.Initialize(project='ee-jonschwenk')
f = gutils.parse_geometry_objects('projects/ee-jonschwenk/assets/E3SM/Kuparuk_gageshed')
feature = ee.Feature(f.first())
