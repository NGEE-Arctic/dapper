# Functions specific to ERA5-Land Hourly GEE ImageCollection
import os
import ee
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from fastparquet import write
from datetime import datetime

from dapper.utils import utils
from dapper.utils import gee_utils as gu
from dapper.utils import elm_utils as eu
from dapper.met import met_io as io

def e5lh_bands():
    return pd.read_csv(utils._DATA_DIR / "e5lh_band_metadata.csv")

def sample_e5lh(params, skip_tasks=False):
    """
    Exports ERA5-Land hourly time-series data for multiple geometries (polygons or points) to Google Drive in N-year chunks.

    Input is params, a dictionary with the following keys:
        start_date (str) : YYYY-MM-DD format
        end_date (str) : YYYY-MM-DD format
        geometries (list of dict OR ee.FeatureCollection) : geopandas.GeoDataFrame with 'geometry' and 'pid' columns,
                                                            OR a pre-loaded GEE FeatureCollection.
        geometry_id_field (str) : the ID field associated with the geometries; is 'gid' by default
        gee_bands (str OR list of str) : 'all' for all bands, 'elm' for ELM-required bands, or a list of specific bands.
        gdrive_folder (str) : Google Drive folder name for export.
        file_name (str) : Base name of the exported CSV file (without extension).

        If skip_tasks is True, the tasks will not be sent to GEE. 
    """

    # Populate and validate requested bands
    if params["gee_bands"] == "all":
        params["gee_bands"] = e5lh_bands()["band_name"].tolist()
    elif params["gee_bands"] == "elm":
        params["gee_bands"] = eu.elm_data_dicts()["elm_required_bands"]
    else:
        gu.validate_bands(params["gee_bands"])

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
    batches = gu.determine_gee_batches(start_date, end_date, max_date, years_per_task=params["gee_years_per_task"], verbose=not skip_tasks)

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
    
    geometries_fc = gu.ensure_pixel_centers_within_geometries(geometries_fc, sample_img, scale)
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

    df_loc = gu.featurecollection_to_df_loc(geometries_fc)

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


def _discover_csvs_and_years(csv_directory, calendar="noleap"):
    """
    Discover CSV files in a directory and compute the overall (start_year, end_year)
    using your existing io.get_start_end_years.

    Returns
    -------
    csv_files : list of str
        Absolute paths to CSV files found (extension == '.csv').
    start_year, end_year : int, int
        Inclusive year bounds inferred from the files and calendar.
    """
    if isinstance(csv_directory, str):
        csv_directory = Path(csv_directory)

    csv_files = [
        str(csv_directory / f)
        for f in os.listdir(csv_directory)
        if os.path.splitext(f)[1].lower() == ".csv"
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .csv files found in {csv_directory}")

    start_year, end_year = io.get_start_end_years(csv_files, calendar=calendar)
    return csv_files, start_year, end_year


def _prep_df_loc(df_loc, id_col, nzones=1):
    """
    Normalize the locations dataframe for both point and gridded flows.

    - Ensures a 'gid' column (string type) using `id_col` or gu.infer_id_field(df_loc)
    - Adds 'lon_0-360'
    - Ensures a 'zone' column (1..nzones repeated) if not present
    - Sorts by (lat, lon)
    - Returns a copy
    """
    if not {"lat", "lon"}.issubset(df_loc.columns):
        missing = {"lat", "lon"} - set(df_loc.columns)
        raise KeyError(f"df_loc missing required columns: {sorted(missing)}")

    out = df_loc.copy()

    if id_col is None:
        id_col = gu.infer_id_field(out)
    if id_col not in out.columns:
        raise KeyError(f"id_col '{id_col}' not found in df_loc")

    out = out.rename(columns={id_col: "gid"})
    out["gid"] = out["gid"].astype(str)
    out["lon_0-360"] = np.mod(out["lon"].to_numpy(), 360.0)

    if "zone" not in out.columns:
        if nzones < 1:
            raise ValueError("nzones must be >= 1")
        out["zone"] = np.tile(
            np.arange(1, nzones + 1, dtype=int),
            (len(out) // nzones) + 1
        )[: len(out)]

    out = out.sort_values(by=["lat", "lon"]).reset_index(drop=True)
    return out


def _preprocess_file_to_ppdf(raw_df, start_year, end_year, calendar, dformat):
    """
    Thin wrapper around your existing preprocessing function to keep a uniform signature.
    """
    remove_leap = (calendar == "noleap")
    ppdf = _preprocess_e5lh_to_elm_file_grid(
        raw_df,
        start_year,
        end_year,
        remove_leap,
        dformat
    )
    return ppdf.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)


def _site_has_any_finite(gdf, data_cols):
    """
    Early site-level finite check used before writing parquet.
    """
    if len(data_cols) == 0:
        return False
    arr = gdf[data_cols].to_numpy()
    return bool(np.isfinite(arr).any())


def _site_level_coverage_offenders(df, exclude_cols=None, threshold=0.10):
    """
    Compute column-wise finite ratios and return offenders at or below the threshold.
    """
    exclude_cols = exclude_cols or []
    data_cols = [c for c in df.columns if c not in set(exclude_cols)]
    if len(data_cols) == 0:
        return {}

    offenders = {}
    for c in data_cols:
        vals = df[c].to_numpy()
        ratio = float(np.isfinite(vals).mean())
        if ratio <= threshold:
            offenders[c] = ratio
    return offenders


def _create_global_dtime(sample_df, calendar, dtime_units, dtime_resolution_hrs, force_half_hour_for_hourly):
    """
    Create the global DTIME axis once using an existing helper.
    """
    dtime_vals, these_dtime_units, _df_with_dtime = io.create_dtime(
        sample_df,
        calendar=calendar,
        dtime_units=dtime_units,
        dtime_resolution_hrs=dtime_resolution_hrs,
        force_half_hour_for_hourly=force_half_hour_for_hourly,
    )
    return dtime_vals, these_dtime_units


def _preprocess_e5lh_to_elm_file_grid(df, start_year, end_year, remove_leap, dformat):
    """
    Processes and resamples ERA5-Land data for ELM input.

    Parameters:
        df : pandas.DataFrame - raw data
        start_year : int - first valid year
        end_year : int - last valid year
        remove_leap : bool - remove leap days if True
        dformat : str - 'BYPASS' or 'DATM_MODE'
        dtime_resolution_hrs : int or None - desired hourly resolution (e.g. 3 for 3-hourly)
    """

    # Convert and filter time
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date", inplace=True)
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
    if remove_leap:
        df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    # Convert units
    df = e5lh_to_elm_unit_conversions(df)

    # Compute indirect variables (humidities)
    if all(col in df.columns for col in ["temperature_2m", "dewpoint_temperature_2m", "surface_pressure"]):
        df["relative_humidity"], df["specific_humidity"] = eu.compute_humidities(
            df["temperature_2m"].values,
            df["dewpoint_temperature_2m"].values,
            df["surface_pressure"].values,
        )
    else:
        print("Missing variables to compute humidities.")

    # Enforce non-negative constraint
    nonnegs = eu.elm_data_dicts()["nonneg"]
    for col in nonnegs:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Rename columns for ELM
    mdd = eu.elm_data_dicts()
    if dformat == "BYPASS":
        do_vars = [v for v in mdd["elm_req_vars"]["cbypass"] if v not in ["LONGXY", "LATIXY", "time"]]
    elif dformat == "DATM_MODE":
        do_vars = [v for v in mdd["elm_req_vars"]["datm"] if v not in ["LONGXY", "LATIXY", "time"]]
    renamer = {k: v for k, v in mdd["short_names"].items() if v in do_vars}
    renamer.update({"date": "time", "lon": "LONGXY", "lat": "LATIXY"})
    df.rename(columns=renamer, inplace=True)

    # Final column selection
    do_vars.extend(["LONGXY", "LATIXY", "time", "gid", "zone"])
    df = df[do_vars]

    return df



class e5lh_to_elm_class:
    """
    Orchestrates the ERA5-Land → ELM conversion:
      Pass 1: CSV shards -> per-site parquet
      Pass 2: parquet -> NetCDF(s) via an output-mode writer

    Output modes:
      - 'sites_file':   one multi-site file per variable    (dims ('n','DTIME'))
      - 'site_dirs':    one directory per site, per-var NCs (dims ('n','DTIME') with n=1)
      - 'gridded':      one file per variable on a lat/lon grid (dims ('DTIME','lat','lon'))

    Packing:
      - 'sites_file' and 'gridded' require pack_scope='global' (one (offset,scale) per var)
      - 'site_dirs' uses per-site packing (pack_scope is ignored)
    """

    # -------------------------- public API --------------------------

    def __init__(self,
                 csv_directory,
                 write_directory,
                 df_loc,
                 id_col=None,
                 calendar='noleap',
                 dtime_resolution_hrs=1,
                 dtime_units='days',
                 nzones=1,
                 dformat="BYPASS",
                 force_half_hour_for_hourly=True,
                 append_attrs=None,    # dict of extra global attributes for each NC file
                 chunks=None,          # optional chunk override (tuple aligned to dims)
                 include_vars=None,    # optional allowlist of ELM var names
                 exclude_vars=None     # optional blocklist of ELM var names
                 ):
        # Paths
        self.csv_directory = Path(csv_directory) if isinstance(csv_directory, str) else csv_directory
        self.write_directory = Path(write_directory) if isinstance(write_directory, str) else write_directory

        # Attributes
        self.df_loc = df_loc

        # Config
        self.id_col = id_col
        self.calendar = calendar
        self.dtime_resolution_hrs = dtime_resolution_hrs
        self.dtime_units = dtime_units
        self.nzones = nzones
        self.dformat = dformat
        self.force_half_hour_for_hourly = force_half_hour_for_hourly
        self.append_attrs = append_attrs or {}
        self.chunks = chunks
        self.include_vars = set(include_vars) if include_vars is not None else None
        self.exclude_vars = set(exclude_vars) if exclude_vars is not None else None

        # Derived state (filled in run())
        self.site_order = None
        self.gid_to_isite = None
        self.lats_all = None
        self.lons0360_all = None

        self.csv_files = None
        self.start_year = None
        self.end_year = None

        self.temp_parquet_dir = None
        self.parquet_files = None

        self.var_cols = None
        self.meta_cols = {"gid", "time", "LONGXY", "LATIXY", "zone"}

        self.dtime_vals = None
        self.dtime_units_out = None
        self.nt = None

    def run(self, output_mode, pack_scope=None):
        """
        Execute the pipeline.

        output_mode : 'sites_file' | 'site_dirs' | 'gridded'
        pack_scope  : required for 'sites_file' and 'gridded' -> must be 'global'
                      ignored for 'site_dirs'
        """
        if self.dformat not in ["DATM_MODE", "BYPASS"]:
            raise KeyError("Unsupported dformat. Only DATM_MODE and BYPASS are available.")
        if self.dformat == "DATM_MODE":
            print("DATM_MODE is not yet available. Exiting.")
            return
        if output_mode not in ("sites_file", "site_dirs", "gridded"):
            raise ValueError("output_mode must be 'sites_file', 'site_dirs', or 'gridded'.")

        if output_mode in ("sites_file", "gridded"):
            if pack_scope is None:
                raise ValueError("pack_scope must be provided ('global') for this output_mode.")
            if pack_scope != "global":
                raise ValueError(f"{output_mode} requires pack_scope='global'.")

        # 0) Prepare df_loc and discover files/years
        self._prepare_locations()
        self._discover_csvs_and_years()

        # 1) CSV -> per-site parquet
        utils.make_directory(self.write_directory, delete_all_contents=True)
        self.temp_parquet_dir = self.write_directory / "temp_parquet"
        utils.make_directory(self.temp_parquet_dir, delete_all_contents=True)
        self._pass1_csvs_to_parquet()

        self.parquet_files = list(self.temp_parquet_dir.glob("*.parquet"))
        if not self.parquet_files:
            print("No site Parquets to export; exiting.")
            return

        # 2) Discover variables and global DTIME
        self._discover_vars_and_dtime()
        if not self.var_cols:
            print("No data variables found; exiting.")
            return

        # 3) Build writer
        if output_mode == "sites_file":
            writer = self._make_sites_file_writer()
            packing = self._compute_global_packing()
        elif output_mode == "gridded":
            writer = self._make_gridded_writer()
            packing = self._compute_global_packing()
        else:  # site_dirs
            writer = self._make_site_dirs_writer()
            packing = None  # per-site packing in this mode

        # 4) Initialize files (one per var)
        writer.initialize_files(self.var_cols, packing)

        # Zone mappings
        writer.write_zone_mappings()

        # 5) Scatter sites into outputs
        for pf in self.parquet_files:
            gid = pf.stem
            site_df0 = pd.read_parquet(pf)
            if site_df0.empty:
                print(f"Skipping {gid}: parquet is empty.")
                continue

            offenders = _site_level_coverage_offenders(site_df0, exclude_cols=list(self.meta_cols), threshold=0.10)
            if offenders:
                msg = ", ".join(f"{k}={v:.1%}" for k, v in sorted(offenders.items()))
                print(f"Skipping {gid}: ≤10% finite → {msg}")
                writer.on_skip_gid(gid)
                continue

            # Align/resample DTIME per-site
            dvals_site, units_site, site_df = io.create_dtime(
                site_df0,
                calendar=self.calendar,
                dtime_units=self.dtime_units,
                dtime_resolution_hrs=self.dtime_resolution_hrs,
                force_half_hour_for_hourly=self.force_half_hour_for_hourly
            )
            if len(dvals_site) != self.nt:
                raise ValueError(f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}")

            # Append each variable
            for v in self.var_cols:
                if v not in site_df.columns:
                    continue
                vals = site_df[v].to_numpy(dtype="float64")
                if not np.isfinite(vals).any():
                    continue
                writer.append(gid, v, vals)

        # 6) Cleanup
        utils.remove_directory_contents(self.temp_parquet_dir, remove_directory=True)
        print("Export complete.")

    # -------------------------- orchestrator helpers --------------------------

    def _prepare_locations(self):
        if self.df_loc is None:
            raise ValueError("df_loc must be provided to E5LHToELMExporter(...)")

        # Normalize/validate
        self.df_loc = _prep_df_loc(self.df_loc, id_col=self.id_col, nzones=self.nzones)
        if self.df_loc["gid"].isna().any():
            raise ValueError("df_loc contains null gid values. Populate df_loc['gid'] before calling.")

        # Stable site ordering + coordinate arrays
        self.site_order = self.df_loc["gid"].tolist()
        self.gid_to_isite = {g: i for i, g in enumerate(self.site_order)}
        df_idx = self.df_loc.set_index("gid").loc[self.site_order]
        self.lats_all = df_idx["lat"].to_numpy()
        self.lons0360_all = df_idx["lon_0-360"].to_numpy()

    def _coerce_df_loc(self):
        # If you prefer, pass df_loc into __init__. This is a tiny safety valve.
        raise ValueError("df_loc must be provided to the exporter.")

    def _discover_csvs_and_years(self):
        self.csv_files, self.start_year, self.end_year = _discover_csvs_and_years(self.csv_directory, calendar=self.calendar)

    def _pass1_csvs_to_parquet(self):
        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            this_df = pd.read_csv(f)

            # normalize ID column to 'gid'
            csv_id_col = self.id_col if (self.id_col is not None and self.id_col in this_df.columns) else gu.infer_id_field(this_df)
            if csv_id_col not in this_df.columns:
                print(f"SKIP FILE: could not find an ID column in {f}. Columns: {list(this_df.columns)}")
                continue
            this_df = this_df.rename(columns={csv_id_col: "gid"})
            this_df["gid"] = this_df["gid"].astype(str).str.strip()

            # merge on gid
            merged = this_df.merge(self.df_loc[["gid", "lat", "lon", "zone"]], on="gid", how="inner")
            if merged.empty:
                # fallback: lat/lon join
                if {"lat", "lon"}.issubset(this_df.columns):
                    df_loc_ll = self.df_loc.copy()
                    df_loc_ll["lat_r"] = df_loc_ll["lat"].round(6); df_loc_ll["lon_r"] = df_loc_ll["lon"].round(6)
                    this_df_ll = this_df.copy()
                    this_df_ll["lat_r"] = this_df_ll["lat"].round(6); this_df_ll["lon_r"] = this_df_ll["lon"].round(6)
                    merged = this_df_ll.merge(
                        df_loc_ll[["gid", "lat_r", "lon_r", "zone"]],
                        on=["lat_r", "lon_r"],
                        how="inner"
                    ).drop(columns=["lat_r", "lon_r"])
                    if merged.empty:
                        print("SKIP FILE: gid-merge and lat/lon fallback both produced 0 rows.")
                        continue
                    else:
                        print(f"INFO: Used lat/lon fallback merge for {f} (matched {len(merged)} rows).")

            # preprocess shard
            ppdf = _preprocess_file_to_ppdf(
                merged,
                start_year=self.start_year,
                end_year=self.end_year,
                calendar=self.calendar,
                dformat=self.dformat
            )

            # optional var filtering
            if self.include_vars is not None or self.exclude_vars is not None:
                keep = set([c for c in ppdf.columns if c in ("gid", "time", "LONGXY", "LATIXY", "zone")])
                if self.include_vars is not None:
                    keep |= self.include_vars
                if self.exclude_vars is not None:
                    keep |= (set(ppdf.columns) - self.exclude_vars)
                ppdf = ppdf[[c for c in ppdf.columns if c in keep]]

            data_cols_ppdf = [c for c in ppdf.columns if c not in self.meta_cols]
            for gid, gdf in ppdf.groupby("gid", sort=False):
                if len(data_cols_ppdf) == 0 or not np.isfinite(gdf[data_cols_ppdf].to_numpy()).any():
                    print(f"Skipping {gid}: no finite data in any variable (pre-parquet).")
                    continue
                parquet_path = self.temp_parquet_dir / f"{gid}.parquet"
                if os.path.isfile(parquet_path):
                    write(parquet_path, gdf, append=True)
                else:
                    write(parquet_path, gdf)

    def _discover_vars_and_dtime(self):
        sample_df0 = pd.read_parquet(self.temp_parquet_dir / (self.site_order[0] + ".parquet")) \
            if (self.temp_parquet_dir / (self.site_order[0] + ".parquet")).exists() \
            else pd.read_parquet(self.parquet_files[0])

        self.var_cols = [c for c in sample_df0.columns if c not in self.meta_cols]
        self.dtime_vals, self.dtime_units_out = _create_global_dtime(
            sample_df0,
            calendar=self.calendar,
            dtime_units=self.dtime_units,
            dtime_resolution_hrs=self.dtime_resolution_hrs,
            force_half_hour_for_hourly=self.force_half_hour_for_hourly
        )
        self.nt = len(self.dtime_vals)

    def _compute_global_packing(self):
        var_min = {v: np.inf for v in self.var_cols}
        var_max = {v: -np.inf for v in self.var_cols}
        for pf in self.parquet_files:
            dfp = pd.read_parquet(pf)
            for v in self.var_cols:
                if v not in dfp.columns:
                    continue
                vals = dfp[v].to_numpy()
                if np.isfinite(vals).any():
                    var_min[v] = min(var_min[v], float(np.nanmin(vals)))
                    var_max[v] = max(var_max[v], float(np.nanmax(vals)))

        packing = {}
        for v in self.var_cols:
            if not np.isfinite(var_min[v]) or not np.isfinite(var_max[v]):
                ao, sf = eu.elm_var_packing_params(v, data=None)
            else:
                ao, sf = eu.elm_var_packing_params(v, data=np.array([var_min[v], var_max[v]]))
            if (not np.isfinite(sf)) or abs(sf) < 1e-12:
                rep = var_min[v] if np.isfinite(var_min[v]) else 0.0
                ao, sf = float(rep), 1.0
            packing[v] = (float(ao), float(sf))
        return packing

    # -------------------------- writers --------------------------

    def _make_sites_file_writer(self):
        exporter = self

        class SitesFileWriter:
            def __init__(self):
                self.var_to_nc = {}

            def initialize_files(self, var_cols, packing):
                dims = ('n', 'DTIME')
                dim_lengths = {'n': len(exporter.site_order), 'DTIME': exporter.nt}
                coord_specs = [
                    {"name": "LATIXY", "dtype": "f4", "dims": ("n",), "data": exporter.lats_all,
                     "attrs": {"units": "degrees_north", "long_name": "latitude"}},
                    {"name": "LONGXY", "dtype": "f4", "dims": ("n",), "data": exporter.lons0360_all,
                     "attrs": {"units": "degrees_east", "long_name": "longitude", "note": "0–360 convention"}}
                ]
                for v in var_cols:
                    filename = f"ERA5_{v}_{exporter.start_year}-{exporter.end_year}.nc"
                    path_nc = exporter.write_directory / filename
                    ao, sf = packing[v]
                    io.initialize_met_netcdf(
                        path_nc=path_nc,
                        var_name=v,
                        dims=dims,
                        dim_lengths=dim_lengths,
                        dtime_name='DTIME',
                        dtime_vals=exporter.dtime_vals,
                        dtime_units=exporter.dtime_units_out,
                        calendar=exporter.calendar,
                        coord_specs=coord_specs,
                        add_offset=ao,
                        scale_factor=sf,
                        dtype="i2",
                        fill_value=32767,
                        chunks=exporter.chunks,
                        write_pattern='by_site',
                        append_attrs=exporter.append_attrs,
                        nc_format="NETCDF4_CLASSIC",
                    )
                    self.var_to_nc[v] = (path_nc, ao, sf)

            def write_zone_mappings(self):
                path = exporter.write_directory / "zone_mappings.txt"
                exporter.df_loc[["gid", "zone"]].to_csv(path, index=False, header=False, sep="\t")

            def append(self, gid, v, vals):
                isite = exporter.gid_to_isite.get(gid, None)
                if isite is None:
                    return
                path_nc, ao, sf = self.var_to_nc[v]
                io.append_met_netcdf(
                    path_nc=path_nc,
                    var_name=v,
                    data=vals,
                    indexers={"n": isite, "DTIME": slice(0, exporter.nt)}
                )

            def on_skip_gid(self, gid):
                # nothing to clean up at site scope
                pass

        return SitesFileWriter()

    def _make_site_dirs_writer(self):
        exporter = self
        zms = eu.gen_zone_mappings(exporter.df_loc, site=True)

        class SiteDirsWriter:
            def __init__(self):
                self.var_cols = None

            def initialize_files(self, var_cols, packing):
                # nothing up-front; files are created per site/var
                self.var_cols = var_cols

            def write_zone_mappings(self):
                # per-site file is written on first var creation
                pass

            def append(self, gid, v, vals):
                if v not in self.var_cols:
                    return
                this_loc = exporter.df_loc[exporter.df_loc["gid"] == gid]
                if this_loc.empty:
                    return
                lat = float(this_loc["lat"].iloc[0])
                lon0360 = float(this_loc["lon_0-360"].iloc[0])
                zone_str = str(int(this_loc["zone"].iloc[0])).zfill(2)

                # per-site packing (from actual vals)
                ao, sf = eu.elm_var_packing_params(v, data=vals)
                if (not np.isfinite(sf)) or abs(sf) < 1e-12:
                    rep = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
                    ao, sf = float(rep), 1.0

                site_dir = exporter.write_directory / gid
                utils.make_directory(site_dir, delete_all_contents=False)
                # write zone mapping (once per site)
                zm_path = site_dir / "zone_mappings.txt"
                if not zm_path.exists():
                    zms[gid].to_csv(zm_path, index=False, header=False, sep="\t")

                filename = f"ERA5_{v}_{exporter.start_year}-{exporter.end_year}_z{zone_str}.nc"
                path_nc = site_dir / filename
                if not path_nc.exists():
                    dims = ('n', 'DTIME')
                    dim_lengths = {'n': 1, 'DTIME': exporter.nt}
                    coord_specs = [
                        {"name": "LATIXY", "dtype": "f4", "dims": ("n",), "data": np.array([lat], dtype="float32"),
                         "attrs": {"units": "degrees_north", "long_name": "latitude"}},
                        {"name": "LONGXY", "dtype": "f4", "dims": ("n",), "data": np.array([lon0360], dtype="float32"),
                         "attrs": {"units": "degrees_east", "long_name": "longitude", "note": "0–360 convention"}}
                    ]
                    io.initialize_met_netcdf(
                        path_nc=path_nc,
                        var_name=v,
                        dims=dims,
                        dim_lengths=dim_lengths,
                        dtime_name='DTIME',
                        dtime_vals=exporter.dtime_vals,
                        dtime_units=exporter.dtime_units_out,
                        calendar=exporter.calendar,
                        coord_specs=coord_specs,
                        add_offset=float(ao),
                        scale_factor=float(sf),
                        dtype="i2",
                        fill_value=32767,
                        chunks=exporter.chunks,
                        write_pattern='by_site',
                        append_attrs=exporter.append_attrs,
                        nc_format="NETCDF4_CLASSIC",
                    )

                io.append_met_netcdf(
                    path_nc=path_nc,
                    var_name=v,
                    data=vals,
                    indexers={"n": 0, "DTIME": slice(0, exporter.nt)}
                )

            def on_skip_gid(self, gid):
                # remove any partial directory if you want; currently no-op
                pass

        return SiteDirsWriter()

    def _make_gridded_writer(self):
        exporter = self
        # build axes & lookup
        lats = np.unique(exporter.df_loc["lat"].to_numpy()); lats.sort()
        lons0360 = np.unique(exporter.df_loc["lon_0-360"].to_numpy()); lons0360.sort()
        lat_key = {round(float(v), 6): i for i, v in enumerate(lats)}
        lon_key = {round(float(v), 6): j for j, v in enumerate(lons0360)}

        dup_check = exporter.df_loc.groupby(["lat", "lon_0-360"]).size()
        if (dup_check > 1).any():
            dups = dup_check[dup_check > 1]
            raise ValueError("Duplicate (lat,lon) entries in df_loc for gridded output: %s" % (dups.index.tolist(),))

        class GriddedWriter:
            def __init__(self):
                self.var_to_nc = {}

            def initialize_files(self, var_cols, packing):
                dims = ('DTIME', 'lat', 'lon')
                dim_lengths = {'DTIME': exporter.nt, 'lat': len(lats), 'lon': len(lons0360)}
                coord_specs = [
                    {"name": "lat", "dtype": "f4", "dims": ("lat",), "data": lats,
                     "attrs": {"units": "degrees_north", "long_name": "latitude"}},
                    {"name": "lon", "dtype": "f4", "dims": ("lon",), "data": lons0360,
                     "attrs": {"units": "degrees_east", "long_name": "longitude", "note": "0–360 convention"}}
                ]
                for v in var_cols:
                    filename = f"ERA5_{v}_{exporter.start_year}-{exporter.end_year}.nc"
                    path_nc = exporter.write_directory / filename
                    ao, sf = packing[v]
                    io.initialize_met_netcdf(
                        path_nc=path_nc,
                        var_name=v,
                        dims=dims,
                        dim_lengths=dim_lengths,
                        dtime_name='DTIME',
                        dtime_vals=exporter.dtime_vals,
                        dtime_units=exporter.dtime_units_out,
                        calendar=exporter.calendar,
                        coord_specs=coord_specs,
                        add_offset=ao,
                        scale_factor=sf,
                        dtype="i2",
                        fill_value=32767,
                        chunks=exporter.chunks,
                        write_pattern='by_cell',
                        append_attrs=exporter.append_attrs,
                        nc_format="NETCDF4_CLASSIC",
                    )
                    self.var_to_nc[v] = (path_nc, ao, sf)

            def write_zone_mappings(self):
                path = exporter.write_directory / "zone_mappings.txt"
                exporter.df_loc[["gid", "zone"]].to_csv(path, index=False, header=False, sep="\t")

            def append(self, gid, v, vals):
                row = exporter.df_loc[exporter.df_loc["gid"] == gid]
                if row.empty:
                    return
                plat = round(float(row["lat"].iloc[0]), 6)
                plon = round(float(row["lon_0-360"].iloc[0]), 6)
                iy = lat_key.get(plat, None)
                ix = lon_key.get(plon, None)
                if iy is None or ix is None:
                    return
                path_nc, ao, sf = self.var_to_nc[v]
                io.append_met_netcdf(
                    path_nc=path_nc,
                    var_name=v,
                    data=vals,
                    indexers={"DTIME": slice(0, exporter.nt), "lat": iy, "lon": ix}
                )

            def on_skip_gid(self, gid):
                # nothing to clean up for a single cell
                pass

        return GriddedWriter()
