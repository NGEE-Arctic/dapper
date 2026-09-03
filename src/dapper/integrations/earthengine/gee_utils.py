# Generic functions JPS
"""Google Earth Engine helpers and sampling utilities."""

try:
    import ee  # type: ignore
except Exception:  # pragma: no cover
    ee = None  # type: ignore

def _require_ee_global():
    """Import Earth Engine lazily and bind it to the module global 'ee'."""
    global ee
    if ee is None:
        try:
            import ee as _ee  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Google Earth Engine (earthengine-api) is required for this functionality. "
                "Install it (pip install earthengine-api) and authenticate (earthengine authenticate)."
            ) from e
        ee = _ee
    return ee

# If Earth Engine is not importable, expose a proxy that raises a clear error on use.
if ee is None:  # pragma: no cover
    class _EEProxy:
        def __getattr__(self, name):
            return getattr(_require_ee_global(), name)

    ee = _EEProxy()  # type: ignore


import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from shapely.ops import unary_union
from shapely.geometry import Polygon, shape
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection

from dapper.domains.domain import Domain
from dapper.config.metsources import era5


# Pathing for convenience
import dapper
_ROOT_DIR = Path(next(iter(dapper.__path__))).parent
_DATA_DIR = _ROOT_DIR / "data"


def parse_geometry_object(geom, name=None):
    """
    Convert a single geometry-like input into an ee.Geometry.

    Supported:
      - str: treated as a GEE asset id; returns its union geometry
      - shapely: Point / Polygon / MultiPolygon / LineString / MultiLineString / GeometryCollection
      - ee.Geometry / ee.Feature / ee.FeatureCollection
    """
    from shapely.geometry import (
        Point, Polygon, MultiPolygon,
        LineString, MultiLineString,
        GeometryCollection,
    )
    from shapely.ops import unary_union

    def _ring_coords(ls):
        return [[float(x), float(y)] for x, y in ls.coords]

    def _poly_coords(poly: Polygon):
        # EE expects [outer, hole1, hole2, ...]
        coords = [_ring_coords(poly.exterior)]
        for interior in poly.interiors:
            coords.append(_ring_coords(interior))
        return coords

    def _to_ee_geometry(g):
        if g is None:
            raise TypeError("Geometry is None")

        if isinstance(g, GeometryCollection):
            g = unary_union(g)

        if isinstance(g, Point):
            return ee.Geometry.Point([float(g.x), float(g.y)])

        if isinstance(g, Polygon):
            return ee.Geometry.Polygon(_poly_coords(g))

        if isinstance(g, MultiPolygon):
            polys = [_poly_coords(p) for p in g.geoms]
            return ee.Geometry.MultiPolygon(polys)

        if isinstance(g, LineString):
            return ee.Geometry.LineString(_ring_coords(g))

        if isinstance(g, MultiLineString):
            lines = [_ring_coords(ln) for ln in g.geoms]
            return ee.Geometry.MultiLineString(lines)

        # last resort: union then retry once
        g2 = unary_union(g)
        if g2 is not g:
            return _to_ee_geometry(g2)

        raise TypeError(f"Unsupported geometry type: {type(g)}")

    # asset id -> union geometry
    if isinstance(geom, str):
        return ee.FeatureCollection(geom).geometry()

    # pass-through EE types
    if isinstance(geom, ee.Geometry):
        return geom
    if isinstance(geom, ee.Feature):
        return geom.geometry()
    if isinstance(geom, ee.FeatureCollection):
        return ee.FeatureCollection(geom).geometry()

    # shapely
    if isinstance(geom, (Point, Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection)):
        return _to_ee_geometry(geom)

    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def parse_geometry_objects(geom, geometry_id_field=None):
    """
    Translate geometry containers to an ee.FeatureCollection.

    Accepted inputs:
      - Domain: uses domain.support (preferred) or legacy domain.gdf
      - str: interpreted as a GEE asset id
      - ee.FeatureCollection: returned (re-cast) as FeatureCollection
      - GeoDataFrame: requires geometry_id_field; converts rows to features

    Returns an ee.FeatureCollection (even if a single feature is present).

    Notes:
      - This function intentionally does NOT depend on AOI.
      - This function does NOT attempt to “fix” individual shapely geometries
        (e.g., MultiPolygon). GeoDataFrame -> GeoJSON -> EE handles that.
    """
    # Domain -> GeoDataFrame (['gid', 'geometry'])
    if isinstance(geom, Domain):
        gdf = getattr(geom, "support", None)
        if gdf is None:
            gdf = getattr(geom, "gdf", None)  # legacy fallback
        if gdf is None:
            raise AttributeError(
                "Domain has no 'support' (or legacy 'gdf') GeoDataFrame to extract geometries from."
            )
        return parse_geometry_objects(gdf, geometry_id_field=geometry_id_field or "gid")

    # String = GEE asset ID
    if isinstance(geom, str):
        return ee.FeatureCollection(geom)

    # Already a FeatureCollection (re-cast to avoid odd type issues)
    if isinstance(geom, ee.FeatureCollection):
        return ee.FeatureCollection(geom)

    # GeoDataFrame -> FeatureCollection
    if isinstance(geom, gpd.GeoDataFrame):
        if geometry_id_field is None:
            raise KeyError(
                "geometry_id_field is required for GeoDataFrame inputs. "
                "Provide the column that uniquely identifies each row/geometry."
            )
        if geometry_id_field not in geom.columns:
            raise KeyError(
                f"geometry_id_field={geometry_id_field!r} not found in GeoDataFrame columns: {list(geom.columns)}"
            )

        gdf_reduced = geom.copy()
        geom_field = gdf_reduced.geometry.name
        if geom_field is None or geom_field not in gdf_reduced.columns:
            raise KeyError("GeoDataFrame has no active geometry column.")

        # keep only id + geometry
        gdf_reduced = gdf_reduced[[geometry_id_field, geom_field]].copy()

        # force string IDs (preserve leading zeros)
        gdf_reduced[geometry_id_field] = gdf_reduced[geometry_id_field].astype(str).str.strip()

        # standardize to 'gid' for EE properties
        gdf_reduced = gdf_reduced.rename(columns={geometry_id_field: "gid"})

        # ensure CRS for stable lon/lat interpretation
        if gdf_reduced.crs is None:
            gdf_reduced = gdf_reduced.set_crs(epsg=4326)
        else:
            gdf_reduced = gdf_reduced.to_crs(epsg=4326)

        geojson_str = gdf_reduced.to_json()
        return ee.FeatureCollection(json.loads(geojson_str))

    raise TypeError(
        f"Unsupported geometries type: {type(geom)}; "
        "expected str, ee.FeatureCollection, GeoDataFrame, or Domain."
    )


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


def _era5_source_end_exclusive(output_end_exclusive, latest_timestamp_ms):
    """Return a GEE end boundary with one hourly-forcing lookahead image."""
    latest_image_time = datetime.fromtimestamp(
        latest_timestamp_ms / 1000,
        tz=timezone.utc,
    ).replace(tzinfo=None)
    return min(
        output_end_exclusive + timedelta(hours=1),
        latest_image_time + timedelta(hours=1),
    )


def _parse_era5_datetime(value, *, latest_timestamp_ms=None, allow_latest=False):
    """Parse a date/hour boundary and optionally resolve ``latest`` from GEE."""
    text = str(value).strip()
    if text.lower() == "latest":
        if not allow_latest or latest_timestamp_ms is None:
            raise ValueError("'latest' is only supported for end_date.")
        return datetime.fromtimestamp(
            latest_timestamp_ms / 1000,
            tz=timezone.utc,
        ).replace(tzinfo=None)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid date boundary {value!r}; use an ISO date or datetime."
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _gee_boundary_label(value):
    if value.hour == value.minute == value.second == 0:
        return value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d_%H%M")


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
    """Cancel all Earth Engine tasks visible to the current account."""
    
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


def featurecollection_to_domain(
    fc,
    name="gee",
    domain_nc=None,
    *,
    mode: str = "sites",
):
    """
    Converts an ee.FeatureCollection object to a Domain with one row per feature
    and representative lon/lat.

    - Points         → lon/lat from the point.
    - Polygons       → lon/lat from a representative interior point.
    - MultiPolygons  → same as polygons.

    The Domain's underlying GeoDataFrame has:
      - 'gid'   : copied from feature properties
      - 'lon'   : representative longitude (EPSG:4326)
      - 'lat'   : representative latitude (EPSG:4326)
      - 'method': how sampling was interpreted
      - 'sampled_geometry': WKT of the original geometry
      - 'geometry': Point at (lon, lat)
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
        elif geom_type in ("Polygon", "MultiPolygon"):
            rp = geom.representative_point()
            lon, lat = rp.x, rp.y
            method = f"sampled across provided {geom_type.lower()}"
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

        rows.append(
            {
                "gid": gid,
                "lon": lon,
                "lat": lat,
                "method": method,
                "sampled_geometry": geom.wkt,
            }
        )

    df_loc = pd.DataFrame(rows)
    gdf_loc = gpd.GeoDataFrame(
        df_loc,
        geometry=gpd.points_from_xy(df_loc.lon, df_loc.lat),
        crs="EPSG:4326",
    )
    gdf_loc["gid"] = gdf_loc["gid"].astype(str).str.strip()

    # A FeatureCollection is (almost always) multiple features; make that explicit.
    # Use sites-mode: one run per feature.
    return Domain.from_gdf(gdf_loc, name=name, mode=mode, domain_nc=domain_nc)


def featurecollection_to_df_loc(fc, name="gee"):
    """
    Legacy wrapper: convert a FeatureCollection to a df_loc-style GeoDataFrame.

    Prefer `featurecollection_to_domain(fc).cells` in new code.
    """
    dom = featurecollection_to_domain(fc, name=name)
    return dom.cells

def sample_e5lh(params, domain_name=None, skip_tasks=False):
    """
    Submit Google Earth Engine (GEE) export tasks for ERA5-Land Hourly time series.

    This prepares the ERA5-Land Hourly ImageCollection (``"ECMWF/ERA5_LAND/HOURLY"``),
    validates bands, ensures each geometry samples at least one pixel center (falling
    back to points when needed), batches the requested date range into N-year chunks,
    and (unless ``skip_tasks=True``) starts one Drive export task per batch.

    Parameters
    ----------
    params : dict
        Configuration dictionary. Expected keys (case-sensitive):

        - **start_date** (str): Inclusive output start date or datetime.
        - **end_date** (str): Exclusive output end date/datetime, or ``"latest"``.
        - **geometries**: One of the following:

          * **str**: GEE asset ID for a FeatureCollection (e.g., ``"users/me/my_fc"``).
          * **ee.FeatureCollection**: a pre-constructed collection.
          * **GeoDataFrame**: must contain geometry and an ID column (see ``geometry_id_field``).
          * **AOI**: ``dapper.domains.aoi.AOI`` instance; uses its internal GeoDataFrame.
          * **Domain**: ``dapper.domains.domain.Domain`` instance; uses ``Domain.to_geometries()``.

        - **geometry_id_field** (str, optional): ID column in provided geometries.
          Defaults to ``"gid"``. Values are copied into the ``"gid"`` property on each feature.
        - **gee_bands** (str or list[str]): Which ERA5-Land bands to export. One of:

          * ``"all"``: all available bands (from ``era5.ALL_BANDS``)
          * ``"elm"``: bands required to derive ELM variables (from ``era5.REQUIRED_RAW_BANDS``)
          * a list of band names validated against the collection

        - **gdrive_folder** (str): Google Drive folder name where CSV chunks are written.
        - **job_name** (str): Base name used to build per-batch export descriptions/filenames.
        - **gee_scale** (str or int or float): Sampling scale in meters. If ``"native"`` (or
          a value < 11132), the native ERA5-Land scale of **11132 m** is used.
        - **gee_years_per_task** (int, optional): Years per export batch (default: ``5``).

        The function sets ``params["gee_ic"] = "ECMWF/ERA5_LAND/HOURLY"`` internally.

    domain_name : str, optional
        Optional name for the returned Domain.

    skip_tasks : bool, default False
        If True, do everything except starting the GEE export tasks.

    Returns
    -------
    Domain
        Domain describing the sampling locations. The underlying GeoDataFrame contains
        at least ``"gid"``, ``"lon"``, and ``"lat"``.

    Notes
    -----
    - Call ``ee.Initialize()`` before using this function.
    - CSV selectors include ``["gid", "date"] + params["gee_bands"]``.
    - Dates are derived from ``system:time_start`` and formatted in UTC.
    - Sampling includes the image at ``end_date`` as a one-hour lookahead for
      ERA5-Land interval fields. That image is not part of the requested output
      time axis when the CSVs are converted to ELM forcing files.

    Raises
    ------
    KeyError
        If required keys are missing from ``params``.
    ValueError
        If dates are malformed or ``geometries`` is an unsupported type.
    TypeError
        If ``gee_scale`` is not ``"native"`` and not numeric.
    ee.EEException
        Propagated Earth Engine errors (e.g., authentication, export quota).

    Examples
    --------
    ::

        params = {
            "start_date": "1950-01-01",
            "end_date": "1951-12-31",
            "geometries": "users/me/my_sites_fc",
            "geometry_id_field": "gid",
            "gee_bands": "elm",
            "gee_scale": "native",
            "gee_years_per_task": 5,
            "gdrive_folder": "era5_exports",
            "job_name": "era5l_sites",
        }
        domain = sample_e5lh(params)
        domain.gdf.head()
    """
    # Populate and validate requested bands
    if params["gee_bands"] == "all":
        params["gee_bands"] = era5.ALL_BANDS
    elif params["gee_bands"] == "elm":
        params["gee_bands"] = era5.REQUIRED_RAW_BANDS
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

    # Resolve dates after asking GEE for its latest source timestamp. ``latest``
    # is the latest output boundary that still has an interval field available.
    max_timestamp = ic.aggregate_max("system:time_start").getInfo()
    start_date = _parse_era5_datetime(params["start_date"])
    end_date = _parse_era5_datetime(
        params["end_date"],
        latest_timestamp_ms=max_timestamp,
        allow_latest=True,
    )
    if end_date <= start_date:
        raise ValueError("end_date must be later than start_date.")

    # Find the exclusive source boundary. End-labeled hourly accumulation bands
    # require the image at output_end to supply the final [t, t+1) interval.
    source_end_exclusive = _era5_source_end_exclusive(end_date, max_timestamp)
    output_end_effective = source_end_exclusive - timedelta(hours=1)
    if output_end_effective <= start_date:
        raise ValueError("The requested range does not overlap available ERA5-Land data.")

    # Batch the requested output period, then extend only the final task. This
    # avoids creating a separate GEE task for the single lookahead image.
    batches = determine_gee_batches(
        start_date,
        output_end_effective,
        output_end_effective,
        years_per_task=params["gee_years_per_task"],
        verbose=not skip_tasks,
    )
    batches.loc[batches.index[-1], "task_end"] = source_end_exclusive

    # Default to 'gid' if no field provided
    if "geometry_id_field" not in params:
        params["geometry_id_field"] = "gid"

    # Convert various geometry containers (asset id, AOI, Domain, GeoDataFrame, FeatureCollection)
    geometries_fc = parse_geometry_objects(
        params["geometries"],
        geometry_id_field=params["geometry_id_field"],
    )

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
        return ee.Feature(f).set("gid", ee.String(f.get(id_field)))    
    
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

    # Build Domain from the final FeatureCollection
    domain = featurecollection_to_domain(
        geometries_fc,
        name=domain_name or params.get("job_name", "era5_gee"),
        domain_nc=None,
    )


    # Fire off the Tasks
    if skip_tasks is False:
        for batch_id, bdf in batches.iterrows():

            # Filter this Task by date range
            ic_filtered = ic.filterDate(
                bdf["task_start"].strftime("%Y-%m-%dT%H:%M:%S"),
                bdf["task_end"].strftime("%Y-%m-%dT%H:%M:%S"),
            )

            # Compute averages for each feature
            feature_collection = ic_filtered.map(image_to_features).flatten()

            # Create a unique filename for each chunk
            file_suffix = (
                f"{_gee_boundary_label(bdf['task_start'])}_"
                f"{_gee_boundary_label(bdf['task_end'])}"
            )
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

    return domain


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

def sample_image_over_polygons(
    gdf,
    image,
    geometry_id_field,
    band=None,
    reducer="mean",
    out_name=None,
    scale=None,
    ensure_pixel_centers=True,
    verbose=True,
):
    """
    Sample a single-band ee.Image over polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geometries; must have a geometry column in EPSG:4326.
    image : ee.Image
        Image to sample. Must be single-band unless 'band' is specified.
    geometry_id_field : str
        Column in gdf containing unique IDs for each geometry.
    band : str, optional
        Band name to sample. If None, image must have exactly one band.
    reducer : str or ee.Reducer, default 'mean'
        Spatial aggregator. Strings: 'mean', 'min', 'max', 'std'.
    out_name : str, optional
        Name of the output column. If None, uses '<band>_<reducer>'.
    scale : float, optional
        Pixel scale in meters. If None, uses the image's nominal scale.
    ensure_pixel_centers : bool, default True
        If True, tiny polygons with no pixel centers inside will be sampled at
        their centroid instead (see ensure_pixel_centers_within_geometries).
    verbose : bool, default True
        Print basic status.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of gdf with a new column 'out_name' added.
    """
    if geometry_id_field not in gdf.columns:
        raise KeyError(
            f"geometry_id_field '{geometry_id_field}' not found in GeoDataFrame."
        )

    # Determine scale
    if scale is None:
        scale = _nominal_scale_m(image)

    # Determine band
    if band is None:
        band_names = image.bandNames().getInfo()
        if len(band_names) != 1:
            raise ValueError(
                f"Image has {len(band_names)} bands; please specify 'band' explicitly."
            )
        band = band_names[0]

    # Determine reducer
    if isinstance(reducer, str):
        key = reducer.lower()
        if key == "mean":
            reducer_obj = ee.Reducer.mean()
        elif key == "min":
            reducer_obj = ee.Reducer.min()
        elif key == "max":
            reducer_obj = ee.Reducer.max()
        elif key in ("std", "stddev", "stdev"):
            reducer_obj = ee.Reducer.stdDev()
        else:
            raise ValueError(
                f"Unknown reducer '{reducer}'. "
                "Use: 'mean', 'min', 'max', 'std', or pass an ee.Reducer."
            )
    elif isinstance(reducer, ee.Reducer):
        reducer_obj = reducer
        key = "custom"
    else:
        raise TypeError("reducer must be a string or an ee.Reducer instance.")

    if out_name is None:
        out_name = f"{band}_{key}"

    gdf_out = gdf.copy()
    # Normalize ID field to string (parse_geometry_objects does this too)
    gdf_out[geometry_id_field] = gdf_out[geometry_id_field].astype(str).str.strip()

    # Build FeatureCollection (ID + geometry)
    fc = parse_geometry_objects(gdf_out, geometry_id_field=geometry_id_field)

    # Optionally fix tiny polygons with no pixel centers
    if ensure_pixel_centers:
        fc = ensure_pixel_centers_within_geometries(fc, image, scale)

    def _add_sample(feat):
        geom = feat.geometry()
        d = image.reduceRegion(
            reducer=reducer_obj,
            geometry=geom,
            scale=scale,
            maxPixels=1e13,
            tileScale=2,
        )
        value = ee.Dictionary(d).get(band)
        return feat.set(out_name, value)

    fc_with = fc.map(_add_sample)

    stats_gdf = try_to_download_featurecollection(fc_with, verbose=verbose)
    if stats_gdf is None:
        raise RuntimeError(
            "Failed to download sampled FeatureCollection; try coarser 'scale' or smaller AOI."
        )

    # parse_geometry_objects renames ID column to 'gid'
    if "gid" not in stats_gdf.columns:
        raise KeyError(
            "Expected 'gid' column in sampled FeatureCollection; "
            "parse_geometry_objects may have changed."
        )

    stats_gdf = stats_gdf[["gid", out_name]].drop_duplicates(subset="gid")
    stats_gdf["gid"] = stats_gdf["gid"].astype(str).str.strip()

    merged = gdf_out.merge(
        stats_gdf,
        left_on=geometry_id_field,
        right_on="gid",
        how="left",
    ).drop(columns=["gid"])

    if verbose:
        print(
            f"Attached sampled column '{out_name}' using band '{band}' "
            f"with reducer '{key}' at scale {scale} m."
        )

    return merged
