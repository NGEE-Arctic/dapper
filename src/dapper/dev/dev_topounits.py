# Dev script to test implementations that mess with topounits

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

# dapper utilities
from dapper.utils import gee_utils as gu
from dapper.utils.utils import _DATA_DIR
from dapper.topounit import topomake 
import dapper.topounit.topoplot as tplt 

# Initialize EE (use your own GCP project)
ee.Initialize(project='ee-jonschwenk')

# Load Kuparuk gage watershed polygon; convert to ee object
kup = gpd.read_file(_DATA_DIR / 'kup_watershed' / 'Kuparuk_gageshed.shp')
kup = kup.buffer(0) # A hack to "fix" invalid geometries--not needed here but just demonstrating
kup_pgon = ee.Geometry.Polygon(list(kup.geometry.values[0].exterior.coords))

# # Load shared asset (FeatureCollection → first feature → Feature)
feature = ee.Feature(ee.FeatureCollection('projects/ee-jonschwenk/assets/E3SM/Kuparuk_gageshed').first())

topos = topomake.make_topounits(
    feature=feature,
    sources=['elev'],
    binning={'elev': {'strategy': 'percentiles', 'n_bins': 5, 'label_prefix': 'ELEV'}}, # 'label_prefix' is for plotting only
    return_as='gdf',
    export_scale='native',
    target_scale=90,          # keep large AOIs reasonable
    verbose=True
)

soil_specs = [
    {
        "image": "projects/soilgrids-isric/nitrogen_mean",
        "band": "nitrogen_0-5cm_mean",     # depth band
        "reducer": "mean",
        "out_name": "SoilN_0_5cm_mean",
        # "scale": 250,                    # optional; defaults as described above
    },
    {
        "image": "projects/soilgrids-isric/nitrogen_mean",
        "band": "nitrogen_0-5cm_mean",
        "reducer": "std",
        "out_name": "SoilN_0_5cm_std",
    },
]

topos = topomake.add_topounit_image_samples(
    topos,
    sampling_specs=soil_specs,
    # default_scale=250,   # if you want to override the topounit analysis scale
    verbose=True,
)
