# Generic functions JPS
import ee
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, shape
from dateutil.relativedelta import relativedelta

from dapper.met import era5land as e5l

# Pathing for convenience
import dapper
_ROOT_DIR = Path(next(iter(dapper.__path__))).parent
_DATA_DIR = _ROOT_DIR / "data"


def parse_geometry_object(geom, name):
    """
    Translates gdf geometries to ee geometries.
    """

    if type(geom) is str:  # GEE Asset
        ret = geom
    elif type(geom) in [Polygon]:
        eegeom = ee.Geometry.Polygon(list(geom.exterior.coords))
        eefeature = ee.Feature(eegeom, {"name": name})
        ret = ee.FeatureCollection(eefeature)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    return ret


def parse_geometry_objects(geom, geometry_id_field=None):
    """
    Translates gdf geometries to ee geometries.
    If geom is a string, it's interpreted as a path to an available GEE asset.
    If geom is a GeoDataFrame, the geometries for each are interpreted.
    geometry_id_field is the column that contains the unique identifier for each geometry/row in the GeoDataFrame.
    Returns a FeatureCollection, even if a single feature is present.
    """
    # Convert geometries to GEE FeatureCollection (supports dict input OR pre-loaded FeatureCollection)
    if isinstance(geom, str):
        geometries_fc = ee.FeatureCollection(geom)  # Directly use pre-loaded GEE asset
    elif isinstance(geom, ee.FeatureCollection):
        geometries_fc = ee.FeatureCollection(
            geom
        )  # re-casting; should already be correct type but this fixes weird errors
    elif isinstance(geom, gpd.GeoDataFrame):
        gdf_reduced = geom.copy()
        if geometry_id_field is None:
            raise KeyError(
                "No geometry id field was provided, but it is required. Ensure your GeoDataFrame has a unique identifier column."
            )
        geom_field = gdf_reduced.geometry.name
        gdf_reduced = gdf_reduced[[geometry_id_field, geom_field]]
        geojson_str = gdf_reduced.to_json()
        geometries_fc = ee.FeatureCollection(json.loads(geojson_str))

    return geometries_fc


def validate_bands(bandlist, gee_ic="ECMWF/ERA5_LAND/HOURLY"):
    """
    Ensures that the requested bands are available and errors if not.
    """
    if gee_ic == "ECMWF/ERA5_LAND/HOURLY":
        available_bands = set(e5l.e5lh_bands()["band_name"].tolist())
    else:
        collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        sample_image = collection.first()
        band_names = set(sample_image.bandNames().getInfo())

    not_in = [b for b in bandlist if b not in available_bands]
    if len(not_in) > 0:
        raise NameError(
            "You requested the following bands which are not in ERA5-Land Hourly (perhaps check spelling?): {}. For a list of available bands, run md.e5lh_bands()['band_name'].".format(
                not_in
            )
        )

    return


def determine_gee_batches(start_date, end_date, max_date, years_per_task=5, verbose=True):
    """
    Calculates how to batch tasks for splitting bigger GEE jobs.
    Currently assumes ERA5-Land hourly (i.e. hourly data with a known date range).

    Returns a DataFrame where each row defines the start and end time for each
    Task in a batch.
    """
    # Generate a DataFrame with start and end dates for each GEE task
    this_date = start_date
    break_dates = [this_date]
    end_date = min(max_date, end_date)
    while this_date < end_date:
        break_dates.append(break_dates[-1] + relativedelta(years=years_per_task))
        this_date = break_dates[-1]
    # Replace the last date with the maximum possible
    break_dates[-1] = end_date

    # Create DataFrame
    df = pd.DataFrame({"task_start": break_dates[:-1], "task_end": break_dates[1:]})

    if verbose:
        if len(df) == 1:
            print(f"Your request will be executed as one Task in Google Earth Engine.")
        else:
            print(f"Your request will be executed as {len(df)} Tasks in Google Earth Engine.")

    return df


def split_into_dfs(path_csv):
    """
    Splits a GEE-exported csv (from sample_e5lh_at_points) into a dictionary of dataframes
    based on the unique values in the 'pid' column.
    """
    df = pd.read_csv(path_csv)
    return {k: group for k, group in df.groupby("pid")}


def infer_id_field(columns):
    """
    Tries to discern the id field from a list of columns.
    Used when id_col is not specified.
    """
    poss_id = [c for c in columns if "id" in c]
    if len(poss_id) == 0:
        raise NameError(
            "Could not infer id column. Specify it with 'id_col' kwarg when calling e5lh_to_elm()."
        )
    else:
        poss_id_lens = [len(pi) for pi in poss_id]
        id_col = poss_id[poss_id_lens.index(min(poss_id_lens))]
        print(
            f"Inferred '{id_col}' as id column. If this is not correct, re-run this function and specify 'id_col' kwarg."
        )

    return id_col


def kill_all_tasks(verbose=True):
    tasks = ee.data.listOperations()
    for task in tasks:
        task_id = task["name"]
        state = task.get("metadata", {}).get("state", "")
        if state in ["PENDING", "RUNNING"]:
            ee.data.cancelOperation(task_id)
            if verbose:
                print(f"Cancelled task: {task_id}")


def ensure_pixel_centers_within_geometries(fc, sample_img, scale):
    """
    This function takes a featureCollection and ensures that each feature in the collection
    samples valid data for an underlying image (that ideally should be representative of
    an imageCollection). Point geometries are guaranteed to sample from any image regardless
    of resolution, so if a polygon or multipolygon in the featureCollection doesn't contain
    any pixel centers, it is replaced by its centroid as a Point geometry.
    """

    # Function to process each feature
    def check_pixels_and_maybe_centroid(feature):
        geom = feature.geometry()
        geom_type = geom.type()

        # Only act on polygons or multipolygons
        is_poly = ee.Algorithms.If(
            ee.List(["Polygon", "MultiPolygon"]).contains(geom_type), True, False
        )

        def process_polygon():
            # Reduce region to count valid pixels inside the geometry
            count = (
                sample_img.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=geom,
                    scale=scale,
                    maxPixels=1e9,
                )
                .values()
                .get(0)
            )

            # If no pixels (count == 0), replace geometry with centroid
            return ee.Algorithms.If(
                ee.Number(count).gt(0), feature, feature.setGeometry(geom.centroid())
            )

        return ee.Feature(ee.Algorithms.If(is_poly, process_polygon(), feature))

    fc_ensured = fc.map(check_pixels_and_maybe_centroid)
    return fc_ensured


def export_fc(
    fc, filename, fileformat, folder="dapper_exports", prefix=None, verbose=False
):
    """
    Export a FeatureCollection to Google Drive using Earth Engine's table export.

    Parameters:
    - fc: ee.FeatureCollection
        The feature collection to export.
    - filename: str
        The export task description and also used as the file name (if prefix is not provided).
    - fileformat: str
        File format for the export. Must be one of:
            - 'CSV'
            - 'GeoJSON'
            - 'KML'
            - 'KMZ'
    - folder: str, optional
        Google Drive folder to export to. Defaults to 'dapper_exports'.
    - prefix: str, optional
        File name prefix for the exported file. Defaults to the filename if not provided.
    - verbose: bool, optional
        If True, prints export destination information.

    Returns:
    - None
    """

    if prefix is None:
        prefix = filename

    if verbose:
        print(f'{filename} will be exported to folder "{folder}" in your Google Drive.')

    ee.batch.Export.table.toDrive(
        collection=fc,
        description=filename,
        fileFormat=fileformat,
        folder=folder,
        fileNamePrefix=prefix,
    ).start()


def featurecollection_to_df_loc(fc):
    """
    Converts an ee.FeatureCollection object to a GeoDataFrame
    and includes WKT representations of the sampled geometry.
    """
    geojson = fc.getInfo()

    rows = []
    for feature in geojson["features"]:
        gid = feature["properties"].get("gid", None)
        geom = shape(feature["geometry"])
        geom_type = geom.geom_type

        if geom_type == "Point":
            lon, lat = geom.x, geom.y
            method = "sampled at provided coordinate"
        elif geom_type == "Polygon":
            centroid = geom.centroid
            lat, lon = centroid.y, centroid.x
            method = "sampled across provided polygon"
        elif geom_type == "MultiPolygon":
            centroid = geom.centroid
            lat, lon = centroid.y, centroid.x
            method = "sampled across provided multipolygon"
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

        rows.append({
            "gid": gid,
            "lat": lat,
            "lon": lon,
            "method": method,
            "sampled_geometry": geom.wkt  # WKT representation
        })

    df_loc = pd.DataFrame(rows)
    gdf_loc = gpd.GeoDataFrame(
        df_loc,
        geometry=gpd.points_from_xy(df_loc.lon, df_loc.lat),
        crs="EPSG:4326"
    )

    return gdf_loc
