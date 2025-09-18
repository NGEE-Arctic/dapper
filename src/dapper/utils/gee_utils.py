# Generic functions JPS
import ee
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon, shape
from dateutil.relativedelta import relativedelta

from dapper.config.metsources import era5

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


def validate_bands(bandlist, gee_ic):
    """
    Ensures that the requested bands are available and errors if not.
    """
    if gee_ic == "ECMWF/ERA5_LAND/HOURLY":
        available_bands = set(era5.ALL_BANDS)
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


def infer_id_field(columns, verbose=False):
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
        if verbose:
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
    Ensures each feature in `fc` will sample valid data from `sample_img` at the given `scale`.
    For polygons/multipolygons with zero pixel centers inside, replaces geometry with its centroid.
    Properties are preserved.
    """
    band = sample_img.bandNames().get(0)

    def check_pixels_and_maybe_centroid(feature):
        geom = feature.geometry()
        geom_type = geom.type()
        is_poly = ee.List(["Polygon", "MultiPolygon"]).contains(geom_type)

        def process_polygon():
            d = sample_img.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geom,
                scale=scale,
                maxPixels=1e9,
                tileScale=2,  # helps with large/complex polygons
            )
            count = ee.Number(ee.Dictionary(d).get(band, 0))
            return ee.Algorithms.If(
                count.gt(0),
                feature,
                feature.setGeometry(geom.centroid(1, sample_img.projection()))
            )

        # Return feature unchanged if not polygon/multipolygon
        return ee.Algorithms.If(is_poly, process_polygon(), feature)

    return fc.map(check_pixels_and_maybe_centroid)


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


def sample_e5lh(params, skip_tasks=False):
    """
    Submit Google Earth Engine (GEE) export tasks for ERA5-Land Hourly time series
    over many geometries (polygons or points), chunked by year range, and return a
    locations table built from the submitted geometries.

    This function prepares a GEE `ImageCollection` of
    ``"ECMWF/ERA5_LAND/HOURLY"``; validates/expands requested bands; ensures each
    geometry samples at least one native pixel center (falling back to points when
    needed); partitions the requested date range into N-year batches; and (unless
    ``skip_tasks=True``) starts one Drive export task per batch. It always returns a
    small DataFrame derived from the final FeatureCollection (e.g., to persist `gid`
    and coordinates locally).

    Parameters
    ----------
    params : dict
        Configuration dictionary. Expected keys (case-sensitive):

        - **start_date** : str  
          Start date in ``"YYYY-MM-DD"``.

        - **end_date** : str  
          End date in ``"YYYY-MM-DD"``.

        - **geometries** : Union[str, ee.FeatureCollection, geopandas.GeoDataFrame]  
          One of:
          * **str**: GEE asset ID for a FeatureCollection (e.g., ``"users/me/my_fc"``).  
          * **ee.FeatureCollection**: a pre-constructed collection.  
          * **GeoDataFrame**: must contain the geometry column and an ID column (see
            ``geometry_id_field``).

        - **geometry_id_field** : str, optional  
          Name of the ID column in the provided geometries. Defaults to ``"gid"``.
          The value is copied into a property named ``"gid"`` on each feature.

        - **gee_bands** : Union[str, list[str]]  
          Which ERA5-Land bands to export. One of:
          * ``"all"`` – use all available bands (from ``era5.ALL_BANDS``).
          * ``"elm"`` – only bands required to derive ELM variables
            (from ``era5.REQUIRED_RAW_COLUMNS``).
          * list of band names – must be valid for the collection; validated via
            ``validate_bands(..., gee_ic="ECMWF/ERA5_LAND/HOURLY")``.

        - **gdrive_folder** : str  
          Google Drive folder name where CSV chunks are written.

        - **job_name** : str  
          Base name used to build per-batch export descriptions / filenames.

        - **gee_scale** : Union[str, int, float]  
          Sampling scale in meters. If ``"native"`` or a value < 11132, the native
          ERA5-Land scale of **11132 m** is used. Otherwise, the numeric value is used.

        - **gee_years_per_task** : int, optional  
          Years per export batch. Defaults to ``5`` if not provided.

        The function also sets ``params["gee_ic"] = "ECMWF/ERA5_LAND/HOURLY"`` internally.

    skip_tasks : bool, default False
        If ``True``, do everything except starting the GEE export tasks. Useful for
        dry-runs to validate band names, date partitioning, and geometry handling.

    Returns
    -------
    pandas.DataFrame
        A small locations table derived from the final FeatureCollection, as returned
        by ``featurecollection_to_df_loc(geometries_fc)``. Typically contains at least
        a ``"gid"`` column and may include coordinates/metadata depending on your
        helper’s implementation.

    Notes
    -----
    - **Authentication**: Call ``ee.Initialize()`` (and sign in) before using this function.
    - **Pixel-center alignment**: ERA5-Land Hourly is sampled at ~11.1 km native
      resolution. Polygons that fail to include a pixel center can yield empty
      samples; this function calls
      ``ensure_pixel_centers_within_geometries(...)`` using a representative image
      to guard against that (potentially converting polygons to points).
    - **Batching**: Date batching is computed by
      ``determine_gee_batches(start_date, end_date, max_date, years_per_task, ...)``.
      Each batch becomes a single Drive export of all requested bands and all features.
    - **Selectors**: CSVs include ``["gid", "date"] + params["gee_bands"]``.
    - **Time stamps**: Dates are produced from ``system:time_start`` and formatted as
      ``"YYYY-MM-dd HH:mm"`` in UTC.

    Raises
    ------
    KeyError
        If required keys are missing from ``params`` (e.g., ``gee_bands``,
        ``gee_scale``, ``geometries``, ``start_date``, ``end_date``, ``gdrive_folder``,
        or ``job_name``).
    ValueError
        If dates are malformed or ``geometries`` is an unsupported type.
    TypeError
        If ``gee_scale`` is not ``"native"`` and not numeric.
    ee.EEException
        Propagated Earth Engine errors (e.g., authentication, export quota).

    Examples
    --------
    >>> params = {
    ...     "start_date": "1950-01-01",
    ...     "end_date": "1951-12-31",
    ...     "geometries": "users/me/my_sites_fc",
    ...     "geometry_id_field": "gid",
    ...     "gee_bands": "elm",
    ...     "gee_scale": "native",
    ...     "gee_years_per_task": 5,
    ...     "gdrive_folder": "era5_exports",
    ...     "job_name": "era5l_sites"
    ... }
    >>> df_loc = sample_e5lh(params)   # starts Drive export tasks
    >>> df_loc.head()
    """

    # Populate and validate requested bands
    if params["gee_bands"] == "all":
        params["gee_bands"] = era5.ALL_BANDS
    elif params["gee_bands"] == "elm":
        params["gee_bands"] = era5.REQUIRED_RAW_COLUMNS
    else:
        validate_bands(params["gee_bands"], gee_ic="ECMWF/ERA5_LAND/HOURLY")

    # Handle scale
    if params["gee_scale"] == "native":
        scale = 11132  # Native ERA5-Land hourly scale in meters
    elif params["gee_scale"] < 11132:
        scale = 11132
    else:
        scale = params["gee_scale"]

    # Prepare for batching
    if "gee_years_per_task" not in params:
        params["gee_years_per_task"] = 5

    # Set the imageCollection
    params["gee_ic"] = "ECMWF/ERA5_LAND/HOURLY"
    ic = ee.ImageCollection(params["gee_ic"])

    # Convert start and end dates
    start_date = datetime.strptime(params["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(params["end_date"], "%Y-%m-%d")

    # Find latest available date in the image collection
    max_timestamp = ic.aggregate_max("system:time_start").getInfo()
    max_date = datetime.fromtimestamp(max_timestamp / 1000)

    # Determine number of batches
    batches = determine_gee_batches(start_date, end_date, max_date, years_per_task=params["gee_years_per_task"], verbose=not skip_tasks)

    # Default to 'gid' if no field provided
    if "geometry_id_field" not in params:
        params["geometry_id_field"] = "gid"

    # Convert geometries to GEE FeatureCollection (supports dict input OR pre-loaded FeatureCollection)
    if isinstance(params["geometries"], str):
        geometries_fc = ee.FeatureCollection(
            params["geometries"]
        )  # Directly use pre-loaded GEE asset
    elif isinstance(params["geometries"], ee.FeatureCollection):
        geometries_fc = ee.FeatureCollection(
            params["geometries"]
        )  # re-casting; should already be correct type but this fixes weird errors
    elif isinstance(params["geometries"], gpd.GeoDataFrame):
        gdf_reduced = params["geometries"].copy()
        gdf_reduced = gdf_reduced[[params["geometry_id_field"], "geometry"]]
        gdf_reduced = gdf_reduced.rename(columns={params["geometry_id_field"]: "gid"})
        geojson_str = gdf_reduced.to_json()
        geometries_fc = ee.FeatureCollection(json.loads(geojson_str))

    # If the provided polygons do not overlap a pixel center of the native image (ERA5L) resolution,
    # no data will be sampled. Here, we ensure that at least one pixel center is included.
    # If not, we convert the polygon to a point, as points do return data even if they're not
    # perfectly aligned with pixel centers.
    # Use a single ERA5 image
    sample_img = (
        ic.filterDate("2020-01-01T00:00", "2020-01-01T01:00")
        .first()
        .select("temperature_2m")
    )
    
    # make sure every feature has 'gid' set from the chosen id_field
    id_field = params.get("geometry_id_field", "gid")
    def _ensure_gid(f):
        return ee.Feature(f).set("gid", f.get(id_field))
    
    geometries_fc = ensure_pixel_centers_within_geometries(geometries_fc, sample_img, scale)
    geometries_fc = geometries_fc.map(_ensure_gid)

    # Function to extract spatially averaged values over each feature (polygon or point)
    def image_to_features(image):
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm")

        # Reduce regions (spatial average for each feature)
        values = image.reduceRegions(
            collection=geometries_fc,
            reducer=ee.Reducer.mean(),  # Compute spatial mean over feature
            scale=scale,
        )

        return values.map(lambda f: f.set("date", date))  # Attach date to results

    df_loc = featurecollection_to_df_loc(geometries_fc)

    # Fire off the Tasks
    if skip_tasks is False:
        for batch_id, bdf in batches.iterrows():

            # Filter this Task by date range
            ic_filtered = ic.filterDate(
                bdf["task_start"].strftime("%Y-%m-%d"), bdf["task_end"].strftime("%Y-%m-%d")
            )

            # Compute averages for each feature
            feature_collection = ic_filtered.map(image_to_features).flatten()

            # Create a unique filename for each chunk
            file_suffix = f"{bdf['task_start'].strftime('%Y-%m-%d')}_{bdf['task_end'].strftime('%Y-%m-%d')}"
            export_filename = f"{params['job_name']}_{file_suffix}"

            # Export to Google Drive as CSV
            selectors = ["gid", "date"] + params["gee_bands"]
            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=export_filename,
                folder=params["gdrive_folder"],
                fileFormat="CSV",
                selectors=selectors,
            )
            task.start()

            print(f"GEE Export task submitted: {export_filename}")
        print("All export tasks started. Check Google Drive or Task Status in the Javascript Editor for completion.")

    return df_loc
