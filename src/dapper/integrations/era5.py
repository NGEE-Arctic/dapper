"""ERA5-Land sampling with automatic ECMWF ARCO/GEE backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Sequence
from urllib.request import Request, urlopen
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import transform
import xarray as xr

from dapper.config.metsources import era5
from dapper.domains.domain import Domain


ARCO_DATASET = "reanalysis-era5-land-timeseries"
ARCO_CATALOGUE_URL = (
    "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
    f"{ARCO_DATASET}"
)
ARCO_GRID_DEGREES = 0.1
ARCO_EARLIEST_TIMESTAMP = pd.Timestamp("1950-01-02 00:00:00")

# The live CDS form currently limits area requests to 1 degree in each direction.
# Keep a stricter cell/time policy for automatic selection; explicit backend="arco"
# can use the full service extent.
ARCO_MAX_EXTENT_DEGREES = 1.0
ARCO_AUTO_MAX_GRID_CELLS = 25
ARCO_AUTO_MAX_SECONDS = 300.0

# Calibrated on the ELM variable set in September 2026: about 35 seconds of CDS
# queue/setup plus 0.1 seconds and 0.54 compressed MB per grid-cell-year.
ARCO_REQUEST_OVERHEAD_SECONDS = 35.0
ARCO_CELL_YEAR_SECONDS = 0.1
ARCO_CELL_YEAR_MB = 0.54

_ARCO_AREA_PAD_DEGREES = 0.001
_SECONDS_PER_YEAR = 365.2425 * 24 * 60 * 60

_RAW_TO_CDS = {
    "temperature_2m": "2m_temperature",
    "dewpoint_temperature_2m": "2m_dewpoint_temperature",
    "surface_pressure": "surface_pressure",
    "u_component_of_wind_10m": "10m_u_component_of_wind",
    "v_component_of_wind_10m": "10m_v_component_of_wind",
    "surface_solar_radiation_downwards_hourly": (
        "surface_solar_radiation_downwards"
    ),
    "surface_thermal_radiation_downwards_hourly": (
        "surface_thermal_radiation_downwards"
    ),
    "total_precipitation_hourly": "total_precipitation",
}

_ARCO_SHORT_TO_RAW = {
    "t2m": "temperature_2m",
    "d2m": "dewpoint_temperature_2m",
    "sp": "surface_pressure",
    "u10": "u_component_of_wind_10m",
    "v10": "v_component_of_wind_10m",
    "ssrd": "surface_solar_radiation_downwards_hourly",
    "strd": "surface_thermal_radiation_downwards_hourly",
    "tp": "total_precipitation_hourly",
}


@dataclass(frozen=True)
class ERA5SamplingPlan:
    """Summary of Dapper's ERA5-Land backend decision."""

    requested_backend: str
    backend: str
    sampling_method: str
    feature_count: int
    grid_cell_count: int
    estimated_seconds: float
    estimated_download_mb: float
    reason: str


@dataclass
class _FeatureSpec:
    gid: str
    source_geometry: object
    sampling_geometry: object
    method: str
    cells: pd.DataFrame | None
    estimated_cell_count: int
    source_properties: dict
    arco_ineligible_reason: str | None = None


def _as_timestamp(value, *, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an ISO-like date or datetime.") from exc
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _planning_end(end_date) -> pd.Timestamp:
    if str(end_date).strip().lower() == "latest":
        return pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=5)
    return _as_timestamp(end_date, name="end_date")


def _resolve_variables(variables) -> tuple[list[str], list[str]]:
    if isinstance(variables, str):
        if variables.strip().lower() != "elm":
            raise ValueError("variables must be 'elm' or a sequence of supported names.")
        raw_names = list(era5.REQUIRED_RAW_BANDS)
    elif isinstance(variables, Sequence):
        raw_names = []
        cds_to_raw = {value: key for key, value in _RAW_TO_CDS.items()}
        for value in variables:
            name = str(value)
            if name in _RAW_TO_CDS:
                raw_names.append(name)
            elif name in cds_to_raw:
                raw_names.append(cds_to_raw[name])
            else:
                raise ValueError(f"ARCO does not support requested variable {name!r}.")
        raw_names = list(dict.fromkeys(raw_names))
    else:
        raise TypeError("variables must be 'elm' or a sequence of names.")

    if not raw_names:
        raise ValueError("At least one ERA5-Land variable is required.")
    return raw_names, [_RAW_TO_CDS[name] for name in raw_names]


def _domain_support(domain: Domain) -> gpd.GeoDataFrame:
    if not isinstance(domain, Domain):
        raise TypeError("domain must be a dapper.Domain instance.")
    support = domain.support_for(step="met").copy()
    if support.crs is None:
        raise ValueError("The domain meteorology support has no CRS.")
    support = support.to_crs("EPSG:4326")
    if "gid" not in support.columns:
        raise KeyError("The domain meteorology support must contain a 'gid' column.")
    support["gid"] = support["gid"].astype(str).str.strip()
    if support["gid"].duplicated().any():
        raise ValueError("The domain meteorology support contains duplicate gid values.")
    return support


def _projected_area_transformer(geometry):
    center = geometry.representative_point()
    local_equal_area = CRS.from_proj4(
        "+proj=laea "
        f"+lat_0={center.y:.10f} +lon_0={center.x:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", local_equal_area, always_xy=True).transform


def era5_land_grid_cells(geometry) -> pd.DataFrame:
    """Return intersecting 0.1-degree ERA5-Land cells and normalized area weights."""
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError("Grid-cell intersections require a Polygon or MultiPolygon.")
    if geometry.is_empty:
        raise ValueError("Cannot sample an empty geometry.")
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError("Geometry became empty while repairing it.")

    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    half = ARCO_GRID_DEGREES / 2
    scale = round(1 / ARCO_GRID_DEGREES)
    lon_min_i = math.ceil((min_lon - half) * scale - 1e-9)
    lon_max_i = math.floor((max_lon + half) * scale + 1e-9)
    lat_min_i = math.ceil((min_lat - half) * scale - 1e-9)
    lat_max_i = math.floor((max_lat + half) * scale + 1e-9)

    project = _projected_area_transformer(geometry)
    rows = []
    for lat_i in range(lat_min_i, lat_max_i + 1):
        latitude = lat_i / scale
        if not -90 <= latitude <= 90:
            continue
        for lon_i in range(lon_min_i, lon_max_i + 1):
            longitude = lon_i / scale
            if longitude < -179.9 or longitude > 180:
                continue
            cell = box(
                longitude - half,
                latitude - half,
                longitude + half,
                latitude + half,
            )
            overlap = geometry.intersection(cell)
            if overlap.is_empty:
                continue
            area_m2 = transform(project, overlap).area
            if area_m2 <= 0:
                continue
            rows.append(
                {
                    "longitude": float(longitude),
                    "latitude": float(latitude),
                    "area_m2": float(area_m2),
                }
            )

    if not rows:
        raise ValueError("The geometry does not overlap any ERA5-Land grid cells.")
    cells = pd.DataFrame(rows).sort_values(["latitude", "longitude"]).reset_index(drop=True)
    cells["weight"] = cells["area_m2"] / cells["area_m2"].sum()
    return cells


def _estimated_bbox_cell_count(geometry) -> int:
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    nx = max(1, math.ceil((max_lon - min_lon) / ARCO_GRID_DEGREES) + 1)
    ny = max(1, math.ceil((max_lat - min_lat) / ARCO_GRID_DEGREES) + 1)
    return nx * ny


def _build_feature_specs(
    domain: Domain,
    sampling_method: str,
) -> list[_FeatureSpec]:
    if sampling_method not in {"auto", "nearest", "zonal"}:
        raise ValueError("sampling_method must be 'auto', 'nearest', or 'zonal'.")

    specs = []
    support = _domain_support(domain)
    geometry_column = support.geometry.name
    for _, row in support.iterrows():
        gid = str(row["gid"])
        source_geometry = row.geometry
        properties = row.drop(labels=[geometry_column], errors="ignore").to_dict()

        if source_geometry is None or source_geometry.is_empty:
            specs.append(
                _FeatureSpec(
                    gid,
                    source_geometry,
                    source_geometry,
                    "unsupported",
                    None,
                    0,
                    properties,
                    "empty geometry",
                )
            )
            continue

        use_nearest = sampling_method == "nearest" or isinstance(source_geometry, Point)
        if use_nearest:
            point = (
                source_geometry
                if isinstance(source_geometry, Point)
                else source_geometry.representative_point()
            )
            reason = None
            if not (-180 <= point.x <= 180 and -90 <= point.y <= 90):
                reason = "coordinate falls outside the ERA5-Land grid"
            cells = pd.DataFrame(
                [{"longitude": point.x, "latitude": point.y, "weight": 1.0}]
            )
            specs.append(
                _FeatureSpec(
                    gid,
                    source_geometry,
                    point,
                    "nearest",
                    cells,
                    1,
                    properties,
                    reason,
                )
            )
            continue

        if not isinstance(source_geometry, (Polygon, MultiPolygon)):
            specs.append(
                _FeatureSpec(
                    gid,
                    source_geometry,
                    source_geometry,
                    "unsupported",
                    None,
                    0,
                    properties,
                    f"unsupported geometry type {source_geometry.geom_type}",
                )
            )
            continue

        min_lon, min_lat, max_lon, max_lat = source_geometry.bounds
        width = max_lon - min_lon
        height = max_lat - min_lat
        if width > ARCO_MAX_EXTENT_DEGREES or height > ARCO_MAX_EXTENT_DEGREES:
            specs.append(
                _FeatureSpec(
                    gid,
                    source_geometry,
                    source_geometry,
                    "zonal",
                    None,
                    _estimated_bbox_cell_count(source_geometry),
                    properties,
                    (
                        f"bounding box {width:.3f} x {height:.3f} degrees exceeds "
                        f"the {ARCO_MAX_EXTENT_DEGREES:g}-degree ARCO area limit"
                    ),
                )
            )
            continue

        cells = era5_land_grid_cells(source_geometry)
        request_width = cells["longitude"].max() - cells["longitude"].min()
        request_height = cells["latitude"].max() - cells["latitude"].min()
        request_width += 2 * _ARCO_AREA_PAD_DEGREES
        request_height += 2 * _ARCO_AREA_PAD_DEGREES
        reason = None
        if (
            request_width > ARCO_MAX_EXTENT_DEGREES
            or request_height > ARCO_MAX_EXTENT_DEGREES
        ):
            reason = (
                "intersecting grid-cell request exceeds the "
                f"{ARCO_MAX_EXTENT_DEGREES:g}-degree ARCO area limit"
            )
        specs.append(
            _FeatureSpec(
                gid,
                source_geometry,
                source_geometry,
                "zonal",
                cells,
                len(cells),
                properties,
                reason,
            )
        )
    return specs


def _build_plan(
    domain: Domain,
    start_date,
    end_date,
    *,
    backend: str,
    sampling_method: str,
    variable_count: int,
    arco_max_grid_cells: int,
    arco_max_estimated_seconds: float,
) -> tuple[ERA5SamplingPlan, list[_FeatureSpec]]:
    requested_backend = str(backend).strip().lower()
    if requested_backend not in {"auto", "arco", "gee"}:
        raise ValueError("backend must be 'auto', 'arco', or 'gee'.")

    start = _as_timestamp(start_date, name="start_date")
    end = _planning_end(end_date)
    if end <= start:
        raise ValueError("end_date must be later than start_date.")
    specs = _build_feature_specs(domain, sampling_method)
    total_cells = sum(spec.estimated_cell_count for spec in specs)
    years = max((end - start).total_seconds() / _SECONDS_PER_YEAR, 1 / 365.2425)
    variable_fraction = variable_count / len(era5.REQUIRED_RAW_BANDS)
    estimated_seconds = (
        len(specs) * ARCO_REQUEST_OVERHEAD_SECONDS
        + total_cells * years * ARCO_CELL_YEAR_SECONDS * variable_fraction
    )
    estimated_mb = total_cells * years * ARCO_CELL_YEAR_MB * variable_fraction

    hard_reasons = [
        f"{spec.gid}: {spec.arco_ineligible_reason}"
        for spec in specs
        if spec.arco_ineligible_reason
    ]
    if start < ARCO_EARLIEST_TIMESTAMP:
        hard_reasons.append(
            "ARCO begins at 1950-01-02; use GEE when 1950-01-01 is required"
        )

    if requested_backend == "arco":
        if hard_reasons:
            raise ValueError("ARCO cannot serve this request: " + "; ".join(hard_reasons))
        selected = "arco"
        reason = "ARCO was explicitly requested."
    elif requested_backend == "gee":
        selected = "gee"
        reason = "GEE was explicitly requested."
    elif hard_reasons:
        selected = "gee"
        reason = "GEE selected because " + "; ".join(hard_reasons)
    elif total_cells > arco_max_grid_cells:
        selected = "gee"
        reason = (
            f"GEE selected because {total_cells} grid cells exceed the automatic "
            f"ARCO threshold of {arco_max_grid_cells}."
        )
    elif estimated_seconds > arco_max_estimated_seconds:
        selected = "gee"
        reason = (
            f"GEE selected because the ARCO estimate ({estimated_seconds:.0f} s) "
            f"exceeds the automatic threshold ({arco_max_estimated_seconds:.0f} s)."
        )
    else:
        selected = "arco"
        reason = (
            f"ARCO selected: {total_cells} grid cells in {len(specs)} request(s), "
            f"estimated at {estimated_seconds:.0f} s."
        )

    plan = ERA5SamplingPlan(
        requested_backend=requested_backend,
        backend=selected,
        sampling_method=sampling_method,
        feature_count=len(specs),
        grid_cell_count=total_cells,
        estimated_seconds=round(estimated_seconds, 1),
        estimated_download_mb=round(estimated_mb, 1),
        reason=reason,
    )
    return plan, specs


def plan_era5_land_sampling(
    domain: Domain,
    start_date,
    end_date="latest",
    *,
    backend: str = "auto",
    sampling_method: str = "auto",
    variables="elm",
    arco_max_grid_cells: int = ARCO_AUTO_MAX_GRID_CELLS,
    arco_max_estimated_seconds: float = ARCO_AUTO_MAX_SECONDS,
) -> ERA5SamplingPlan:
    """Plan an ERA5-Land request without contacting ECMWF or GEE."""
    raw_names, _ = _resolve_variables(variables)
    plan, _ = _build_plan(
        domain,
        start_date,
        end_date,
        backend=backend,
        sampling_method=sampling_method,
        variable_count=len(raw_names),
        arco_max_grid_cells=arco_max_grid_cells,
        arco_max_estimated_seconds=arco_max_estimated_seconds,
    )
    return plan


def _read_json(url: str):
    request = Request(url, headers={"User-Agent": "dapper-era5-arco"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


@lru_cache(maxsize=16)
def _arco_available_range(cds_variables: tuple[str, ...]) -> tuple[pd.Timestamp, pd.Timestamp]:
    collection = _read_json(ARCO_CATALOGUE_URL)
    constraints_url = next(
        link["href"] for link in collection["links"] if link.get("rel") == "constraints"
    )
    constraints = _read_json(constraints_url)

    starts = []
    ends = []
    for variable in cds_variables:
        matches = [entry for entry in constraints if variable in entry.get("variable", [])]
        if not matches:
            raise ValueError(f"No ARCO availability metadata found for {variable!r}.")
        for match in matches:
            start_text, end_text = match["date"][0].split("/", 1)
            starts.append(pd.Timestamp(start_text))
            ends.append(pd.Timestamp(end_text) + pd.Timedelta(hours=23))
    return max(starts), min(ends)


def _resolve_arco_window(start_date, end_date, cds_variables):
    start = _as_timestamp(start_date, name="start_date")
    available_start, available_end = _arco_available_range(tuple(cds_variables))
    if start < available_start:
        raise ValueError(
            f"ARCO data begin at {available_start}; requested start is {start}. "
            "Use backend='gee' for the partial first day of 1950."
        )

    requested_end = (
        available_end
        if str(end_date).strip().lower() == "latest"
        else _as_timestamp(end_date, name="end_date")
    )
    source_end = min(requested_end, available_end)
    if source_end < requested_end:
        warnings.warn(
            f"Capping ARCO request at its latest common source timestamp: {source_end}.",
            stacklevel=2,
        )
    if source_end <= start:
        raise ValueError("The requested range does not overlap available ARCO data.")
    return start, source_end, available_start, available_end


def _safe_gid(gid: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(gid)).strip("._")
    if not value:
        raise ValueError(f"Cannot make an output filename from gid {gid!r}.")
    return value


def _extract_archive(archive: Path, destination: Path) -> list[Path]:
    if not zipfile.is_zipfile(archive):
        target = destination / f"{archive.stem}.nc"
        target.write_bytes(archive.read_bytes())
        return [target]

    paths = []
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents:
                raise ValueError("Unsafe path found in CDS archive.")
            if member.is_dir():
                continue
            if target.suffix.lower() != ".nc":
                continue
            bundle.extract(member, destination)
            paths.append(target)
    if not paths:
        raise ValueError("The CDS response did not contain NetCDF files.")
    return paths


def _load_arco_archive(archive: Path, output_dir: Path) -> xr.Dataset:
    with tempfile.TemporaryDirectory(prefix=".dapper_arco_", dir=output_dir) as tmp:
        paths = _extract_archive(archive, Path(tmp))
        datasets = [xr.open_dataset(path) for path in paths]
        try:
            merged = xr.merge(datasets, compat="no_conflicts", join="exact").load()
        finally:
            for dataset in datasets:
                dataset.close()
    return merged


def _select_arco_cells(dataset: xr.Dataset, spec: _FeatureSpec):
    cells = spec.cells.copy()
    if spec.method == "nearest" or len(cells) == 1:
        point = cells.iloc[0]
        if "latitude" in dataset.dims or "longitude" in dataset.dims:
            sampled = dataset.sel(
                latitude=float(point["latitude"]),
                longitude=float(point["longitude"]),
                method="nearest",
            )
        else:
            sampled = dataset
        actual_cells = pd.DataFrame(
            [
                {
                    "latitude": float(sampled["latitude"]),
                    "longitude": float(sampled["longitude"]),
                    "weight": 1.0,
                }
            ]
        )
        return sampled, actual_cells

    latitudes = xr.DataArray(cells["latitude"].to_numpy(), dims="sample_cell")
    longitudes = xr.DataArray(cells["longitude"].to_numpy(), dims="sample_cell")
    selected = dataset.sel(
        latitude=latitudes,
        longitude=longitudes,
        method="nearest",
        tolerance=ARCO_GRID_DEGREES / 2,
    )
    actual_latitudes = np.asarray(selected["latitude"]).astype(float)
    actual_longitudes = np.asarray(selected["longitude"]).astype(float)
    if (
        np.max(np.abs(actual_latitudes - cells["latitude"].to_numpy())) > 1e-6
        or np.max(np.abs(actual_longitudes - cells["longitude"].to_numpy())) > 1e-6
    ):
        raise ValueError("CDS did not return the expected ERA5-Land grid cells.")

    valid = (
        selected.to_array("variable")
        .notnull()
        .any(dim=("variable", "valid_time"))
        .to_numpy()
    )
    if not valid.any():
        raise ValueError(f"ARCO returned no valid land pixels for {spec.gid!r}.")
    selected = selected.isel(sample_cell=np.flatnonzero(valid))
    cells = cells.loc[valid].reset_index(drop=True)
    cells["weight"] = cells["weight"] / cells["weight"].sum()
    weights = xr.DataArray(cells["weight"].to_numpy(), dims="sample_cell")
    return selected.weighted(weights).mean("sample_cell"), cells


def _arco_request(spec: _FeatureSpec, cds_variables, start, source_end) -> dict:
    request = {
        "variable": list(cds_variables),
        "date": [f"{start:%Y-%m-%d}/{source_end:%Y-%m-%d}"],
        "data_format": "netcdf",
    }
    cells = spec.cells
    if spec.method == "nearest" or len(cells) == 1:
        point = cells.iloc[0]
        request["location"] = {
            "latitude": float(point["latitude"]),
            "longitude": float(point["longitude"]),
        }
    else:
        request["area"] = [
            float(cells["latitude"].max() + _ARCO_AREA_PAD_DEGREES),
            float(cells["longitude"].min() - _ARCO_AREA_PAD_DEGREES),
            float(cells["latitude"].min() - _ARCO_AREA_PAD_DEGREES),
            float(cells["longitude"].max() + _ARCO_AREA_PAD_DEGREES),
        ]
    return request


def _format_grid_values(cells: pd.DataFrame, column: str) -> str:
    if column == "weight":
        return ", ".join(f"{value:.8f}" for value in cells[column])
    return "; ".join(
        f"({lon:.1f}, {lat:.1f})"
        for lon, lat in zip(cells["longitude"], cells["latitude"])
    )


def _frame_from_arco(
    dataset: xr.Dataset,
    spec: _FeatureSpec,
    raw_names,
    start,
    source_end,
):
    missing = [
        short
        for short, raw in _ARCO_SHORT_TO_RAW.items()
        if raw in raw_names and short not in dataset
    ]
    if missing:
        raise KeyError(f"CDS response is missing expected variables: {missing}")
    if "valid_time" not in dataset.coords:
        raise KeyError("CDS response is missing the 'valid_time' coordinate.")

    dataset = dataset.sel(valid_time=slice(start, source_end))
    sampled, actual_cells = _select_arco_cells(dataset, spec)
    rename = {
        short: raw
        for short, raw in _ARCO_SHORT_TO_RAW.items()
        if raw in raw_names
    }
    frame = sampled[list(rename)].rename(rename).to_dataframe().reset_index()
    frame = frame.rename(columns={"valid_time": "date"})
    frame["gid"] = spec.gid
    frame = frame[["gid", "date", *raw_names]].sort_values("date")
    all_missing = [name for name in raw_names if frame[name].isna().all()]
    if all_missing:
        raise ValueError(f"ARCO returned no valid values for {all_missing} at {spec.gid!r}.")
    return frame, actual_cells


def _metadata_record(
    spec: _FeatureSpec,
    cells: pd.DataFrame,
    *,
    backend: str,
    output_csv: Path | None,
    start,
    source_end,
    estimated_seconds: float,
    elapsed_seconds: float | None = None,
) -> dict:
    if spec.method == "nearest":
        method = "nearest ERA5-Land grid cell via ECMWF CDS ARCO"
    else:
        method = (
            f"area-weighted mean of {len(cells)} intersecting ERA5-Land grid cells "
            "via ECMWF CDS ARCO"
        )
    record = {
        "gid": spec.gid,
        "method": method,
        "sampled_geometry": spec.sampling_geometry.wkt,
        "sampling_backend": backend,
        "sampling_dataset": ARCO_DATASET,
        "sampling_grid_cell_count": int(len(cells)),
        "sampling_grid_coordinates": _format_grid_values(cells, "coordinates"),
        "sampling_grid_weights": _format_grid_values(cells, "weight"),
        "sampling_start": str(start),
        "sampling_source_end": str(source_end),
        "sampling_output_end": str(source_end - pd.Timedelta(hours=1)),
        "sampling_estimated_seconds": round(float(estimated_seconds), 1),
    }
    if output_csv is not None:
        record["sampling_csv"] = str(output_csv)
    if elapsed_seconds is not None:
        record["sampling_elapsed_seconds"] = round(float(elapsed_seconds), 1)
    for name in ("source_file", "feature_count"):
        value = spec.source_properties.get(name)
        if value is not None and not pd.isna(value):
            record[name] = value
    return record


def _annotate_domain(domain: Domain, records: list[dict]) -> Domain:
    normalized = domain.ensure_cells_lon_lat()
    cells = normalized.cells.copy()
    cells["gid"] = cells["gid"].astype(str).str.strip()
    metadata = pd.DataFrame(records).set_index("gid")
    for column in metadata.columns:
        values = metadata[column].to_dict()
        cells[column] = cells["gid"].map(values)
    return normalized.copy(cells=cells)


def _sample_arco(
    domain: Domain,
    specs: list[_FeatureSpec],
    plan: ERA5SamplingPlan,
    raw_names,
    cds_variables,
    *,
    start_date,
    end_date,
    output_dir,
    overwrite: bool,
    keep_downloads: bool,
    cds_client,
) -> Domain:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start, source_end, available_start, available_end = _resolve_arco_window(
        start_date, end_date, cds_variables
    )

    if cds_client is None:
        try:
            import cdsapi
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "ECMWF ARCO sampling requires cdsapi>=0.7.7 and a CDS API token."
            ) from exc
        cds_client = cdsapi.Client()

    safe_names = [_safe_gid(spec.gid) for spec in specs]
    if len(safe_names) != len(set(safe_names)):
        raise ValueError("Domain gids collide after filename sanitization.")

    records = []
    per_request_estimate = plan.estimated_seconds / max(1, plan.feature_count)
    for spec, safe_name in zip(specs, safe_names):
        output_csv = output_dir / f"era5_land_arco_{safe_name}.csv"
        if output_csv.exists() and not overwrite:
            raise FileExistsError(f"{output_csv} already exists (overwrite=False).")
        archive = output_dir / f".{safe_name}_arco_download.zip"
        if archive.exists():
            archive.unlink()

        request = _arco_request(spec, cds_variables, start, source_end)
        request_started = time.perf_counter()
        print(
            f"Downloading ERA5-Land ARCO data for {spec.gid} "
            f"({spec.estimated_cell_count} grid cell(s))..."
        )
        cds_client.retrieve(ARCO_DATASET, request, str(archive))
        dataset = _load_arco_archive(archive, output_dir)
        frame, actual_cells = _frame_from_arco(
            dataset, spec, raw_names, start, source_end
        )
        elapsed = time.perf_counter() - request_started
        frame.to_csv(output_csv, index=False)
        if not keep_downloads:
            archive.unlink(missing_ok=True)
        records.append(
            _metadata_record(
                spec,
                actual_cells,
                backend="arco",
                output_csv=output_csv,
                start=start,
                source_end=source_end,
                estimated_seconds=per_request_estimate,
                elapsed_seconds=elapsed,
            )
        )
        print(f"Wrote {output_csv} in {elapsed:.1f} seconds.")

    manifest = {
        "sampling_backend": "arco",
        "sampling_dataset": ARCO_DATASET,
        "available_source_start": str(available_start),
        "available_source_end": str(available_end),
        "variables": list(raw_names),
        "plan": plan.__dict__,
        "features": records,
    }
    manifest_path = output_dir / "era5_land_sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote sampling manifest: {manifest_path}")
    return _annotate_domain(domain, records)


def _sample_gee(
    domain: Domain,
    specs: list[_FeatureSpec],
    raw_names,
    *,
    start_date,
    end_date,
    gdrive_folder: str,
    job_name: str,
    gee_scale,
    gee_years_per_task: int,
    skip_tasks: bool,
) -> Domain:
    from dapper.integrations.earthengine.gee_utils import sample_e5lh

    support = _domain_support(domain)
    if all(spec.method == "nearest" for spec in specs):
        support["geometry"] = support.geometry.apply(
            lambda geometry: geometry
            if isinstance(geometry, Point)
            else geometry.representative_point()
        )
    params = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "geometries": support,
        "geometry_id_field": "gid",
        "gee_bands": list(raw_names),
        "gee_scale": gee_scale,
        "gee_years_per_task": gee_years_per_task,
        "gdrive_folder": gdrive_folder,
        "job_name": job_name,
    }
    sampled = sample_e5lh(params, domain_name=domain.name, skip_tasks=skip_tasks)
    records = sampled.cells.drop(columns="geometry", errors="ignore").to_dict("records")
    support_by_gid = support.set_index("gid")
    for record in records:
        gid = str(record["gid"])
        record["sampling_backend"] = "gee"
        record["sampling_dataset"] = "ECMWF/ERA5_LAND/HOURLY"
        for name in ("source_file", "feature_count"):
            if name in support_by_gid.columns:
                record[name] = support_by_gid.at[gid, name]
    return _annotate_domain(domain, records)


def sample_era5_land(
    domain: Domain,
    start_date,
    end_date="latest",
    *,
    backend: str = "auto",
    sampling_method: str = "auto",
    variables="elm",
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    keep_downloads: bool = False,
    arco_max_grid_cells: int = ARCO_AUTO_MAX_GRID_CELLS,
    arco_max_estimated_seconds: float = ARCO_AUTO_MAX_SECONDS,
    cds_client=None,
    gdrive_folder: str = "dapper_era5_land",
    job_name: str | None = None,
    gee_scale="native",
    gee_years_per_task: int = 5,
    skip_tasks: bool = False,
) -> Domain:
    """Sample ERA5-Land through ECMWF ARCO or Google Earth Engine.

    Dates use an inclusive start and exclusive output end. ARCO downloads are
    synchronous and written as GEE-compatible CSV files under ``output_dir``.
    GEE exports remain asynchronous Drive tasks. ``backend='auto'`` chooses ARCO
    only when every feature fits its service limits and the estimated request is
    below the configured cell/time thresholds.

    For polygon supports, ``sampling_method='auto'`` uses a local area-weighted
    mean of intersecting ERA5-Land cells with ARCO and GEE's spatial mean with
    GEE. Points use nearest-cell sampling. Use ``sampling_method='nearest'`` to
    sample polygon representative points instead.
    """
    raw_names, cds_variables = _resolve_variables(variables)
    plan, specs = _build_plan(
        domain,
        start_date,
        end_date,
        backend=backend,
        sampling_method=sampling_method,
        variable_count=len(raw_names),
        arco_max_grid_cells=arco_max_grid_cells,
        arco_max_estimated_seconds=arco_max_estimated_seconds,
    )
    print(plan.reason)
    if skip_tasks and plan.backend == "arco":
        start = _as_timestamp(start_date, name="start_date")
        source_end = _planning_end(end_date)
        records = [
            _metadata_record(
                spec,
                spec.cells,
                backend="arco",
                output_csv=None,
                start=start,
                source_end=source_end,
                estimated_seconds=plan.estimated_seconds / max(1, plan.feature_count),
            )
            for spec in specs
        ]
        return _annotate_domain(domain, records)

    if plan.backend == "arco":
        if output_dir is None:
            raise ValueError("output_dir is required when the selected backend is ARCO.")
        return _sample_arco(
            domain,
            specs,
            plan,
            raw_names,
            cds_variables,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            overwrite=overwrite,
            keep_downloads=keep_downloads,
            cds_client=cds_client,
        )

    return _sample_gee(
        domain,
        specs,
        raw_names,
        start_date=start_date,
        end_date=end_date,
        gdrive_folder=gdrive_folder,
        job_name=job_name or f"{domain.name}_era5_land",
        gee_scale=gee_scale,
        gee_years_per_task=gee_years_per_task,
        skip_tasks=skip_tasks,
    )


__all__ = [
    "ARCO_AUTO_MAX_GRID_CELLS",
    "ARCO_AUTO_MAX_SECONDS",
    "ARCO_DATASET",
    "ERA5SamplingPlan",
    "era5_land_grid_cells",
    "plan_era5_land_sampling",
    "sample_era5_land",
]
