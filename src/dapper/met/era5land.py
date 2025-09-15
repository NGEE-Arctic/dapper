# Functions specific to ERA5-Land Hourly GEE ImageCollection
import ee
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from fastparquet import write

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


def e5lh_to_elm_unit_conversions(df):
    """
    Converts ERA5-Land hourly bands to units expected by ELM.

    This is not a comprehensive function for all E5LH variables;
    only ELM variables are handled here.
    """
    # Compute wind magnitude (speed) and direction
    if (
        "u_component_of_wind_10m" in df.columns
        and "v_component_of_wind_10m" in df.columns
    ):
        df["wind_speed"] = np.sqrt(
            df["u_component_of_wind_10m"].values ** 2
            + df["v_component_of_wind_10m"].values ** 2
        )
        wind_dir = np.degrees(
            np.arctan2(
                df["u_component_of_wind_10m"].values,
                df["v_component_of_wind_10m"].values,
            )
        )
        wind_dir[np.where(wind_dir >= 180)] = wind_dir[np.where(wind_dir >= 180)] - 180
        wind_dir[np.where(wind_dir < 180)] = wind_dir[np.where(wind_dir < 180)] + 180
        df["wind_direction"] = wind_dir

    # Precipitation - convert from meters/hour to mm/second
    if "total_precipitation_hourly" in df.columns:
        df["total_precipitation_hourly"] = df["total_precipitation_hourly"].values / 3.6

    # Solar rad downwards - convert from J/hr/m2 to W/m2
    if "surface_solar_radiation_downwards_hourly" in df.columns:
        df["surface_solar_radiation_downwards_hourly"] = (
            df["surface_solar_radiation_downwards_hourly"].values / 3600
        )

    # Thermal rad downwards - convert from J/hr/m2 to W/m2
    if "surface_thermal_radiation_downwards_hourly" in df.columns:
        df["surface_thermal_radiation_downwards_hourly"] = (
            df["surface_thermal_radiation_downwards_hourly"].values / 3600
        )

    return df


def e5lh_to_elm(
    csv_directory,
    write_directory,
    df_loc,
    id_col=None,
    gridded=False,
    calendar='noleap',
    dtime_resolution_hrs=1,
    dtime_units='days',
    nzones=1,
    dformat="BYPASS",
    force_half_hour_for_hourly=True
):
    """
    Site (point) pipeline:
      1) Read GEE CSV shards, preprocess to ELM format
      2) Write per-site parquet (skip sites with no finite data)
      3) Read each per-site parquet; if ANY variable has <=5% finite data, skip the whole site
      4) Otherwise create DTIME and write NetCDFs (variables still checked for sanity)
    Notes:
      - ELM/E3SM use 0–360 longitudes.
      - Does NOT preserve 'gid' in NetCDF; export directory names are 'gid', however.
    """
    # ---- basic checks ----
    if dformat not in ["DATM_MODE", "BYPASS"]:
        raise KeyError("Unsupported dformat. Only DATM_MODE and BYPASS are available.")
    if dformat == "DATM_MODE":
        print("DATM_MODE is not yet available. Exiting.")
        return

    # ---- normalize paths ----
    if isinstance(csv_directory, str):
        csv_directory = Path(csv_directory)
    if isinstance(write_directory, str):
        write_directory = Path(write_directory)

    # ---- prep df_loc, ids, zones, longitudes ----
    df_loc = df_loc.copy()
    df_loc['lon_0-360'] = np.mod(df_loc['lon'], 360)

    if id_col is None:
        id_col = gu.infer_id_field(df_loc)

    df_loc = df_loc.rename(columns={id_col: "gid"})
    df_loc = df_loc.sort_values(by=["lat", "lon"]).reset_index(drop=True)
    df_loc["gid"] = df_loc["gid"].astype(str)

    if "zone" not in df_loc.columns:
        df_loc["zone"] = np.tile(np.arange(1, nzones + 1), (len(df_loc) // nzones) + 1)[: len(df_loc)]

    # ---- discover CSVs and overall year range ----
    csv_files = [os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if os.path.splitext(f)[1] == ".csv"]
    start_year, end_year = io.get_start_end_years(csv_files, calendar=calendar)

    # ---- output root ----
    utils.make_directory(write_directory, delete_all_contents=True)

    if not gridded:
        # ========= PASS 1: write per-site parquet =========
        path_temp_parquet = write_directory / 'temp_parquet'
        utils.make_directory(path_temp_parquet, delete_all_contents=True)

        for i, f in enumerate(csv_files):
            print(f"Processing file {i+1} of {len(csv_files)}: {f}")
            this_df = pd.read_csv(f)

            # ensure gid present & stringy to match df_loc
            this_df = this_df.rename(columns={id_col: "gid"})
            this_df["gid"] = this_df["gid"].astype(str)

            # join site meta
            this_df = this_df.merge(df_loc[["gid", "lat", "lon", "zone"]], on="gid", how="inner")

            remove_leap = (calendar == 'noleap')
            ppdf = _preprocess_e5lh_to_elm_file_grid(this_df, start_year, end_year, remove_leap, dformat)
            ppdf = ppdf.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

            # site split
            ppdfg = ppdf.groupby(by="gid")
            data_cols_ppdf = [c for c in ppdf.columns if c not in ("gid", "time", "LONGXY", "LATIXY", "zone")]

            for gid, gdf in ppdfg:
                # early site-level finite check
                if len(data_cols_ppdf) == 0 or not np.isfinite(gdf[data_cols_ppdf].to_numpy()).any():
                    print(f"Skipping {gid}: no finite data in any variable (pre-parquet).")
                    continue

                parquet_path = path_temp_parquet / f"{gid}.parquet"
                if os.path.isfile(parquet_path):
                    write(parquet_path, gdf, append=True)
                else:
                    write(parquet_path, gdf)

        # zone mappings (per site)
        zms = eu.gen_zone_mappings(df_loc, site=True)

        # ========= PASS 2: parquet → NetCDFs =========
        print(path_temp_parquet)
        parquet_files = list(path_temp_parquet.glob("*.parquet"))
        for pf in parquet_files:
            site = pf.stem  # filename is the site id (gid)
            print(f"Exporting {site}...")
            site_allvar_df0 = pd.read_parquet(pf)

            if site_allvar_df0.empty:
                print(f"Skipping {site}: parquet is empty.")
                continue

            # Required meta before DTIME
            required_before = {"time", "LONGXY", "LATIXY", "zone"}
            missing_before = sorted(required_before - set(site_allvar_df0.columns))
            if missing_before:
                raise KeyError(f"{site}: missing required columns in parquet before DTIME: {missing_before}")

            # Site-level 10% rule (skip site if ANY variable ≤10% valid)
            exclude = {"gid", "time", "LONGXY", "LATIXY", "zone"}
            data_cols0 = [c for c in site_allvar_df0.columns if c not in exclude]
            if len(data_cols0) == 0:
                print(f"Skipping {site}: no data variables found.")
                continue

            finite_ratios = {c: float(np.isfinite(site_allvar_df0[c].to_numpy()).mean()) for c in data_cols0}
            offenders = {c: r for c, r in finite_ratios.items() if r <= 0.10}
            if offenders:
                msg = ", ".join(f"{k}={v:.1%}" for k, v in sorted(offenders.items()))
                print(f"Skipping {site}: some variables have ≤5% finite data → {msg}")
                # optionally remove any partial outputs for this site
                utils.remove_directory_contents(write_directory / site, remove_directory=True)
                continue

            # Keep a one-row meta snapshot in case create_dtime drops meta cols
            meta_row = site_allvar_df0[["LONGXY", "LATIXY", "zone"]].iloc[0]

            # Create/resample DTIME
            dtime_vals, these_dtime_units, site_allvar_df = io.create_dtime(
                site_allvar_df0,
                calendar=calendar,
                dtime_units=dtime_units,
                dtime_resolution_hrs=dtime_resolution_hrs,
                force_half_hour_for_hourly=force_half_hour_for_hourly
            )

            # Restore meta if create_dtime dropped them
            for col in ["LONGXY", "LATIXY", "zone"]:
                if col not in site_allvar_df.columns:
                    site_allvar_df[col] = meta_row[col]

            # Final meta presence check
            required_meta = {"time", "LONGXY", "LATIXY", "zone"}
            missing_meta = sorted(required_meta - set(site_allvar_df.columns))
            if missing_meta:
                raise KeyError(f"{site}: missing required columns after create_dtime: {missing_meta}")

            # Make site output dir
            utils.make_directory(write_directory / site, delete_all_contents=True)

            # Pull location info for this site (for zone in filename)
            this_df_loc = df_loc[df_loc['gid'] == site]
            if this_df_loc.empty:
                print(f"Skipping {site}: not found in df_loc.")
                utils.remove_directory_contents(write_directory / site, remove_directory=True)
                continue
            site_zone_str = str(this_df_loc['zone'].iloc[0]).zfill(2)

            # Iterate variables (exclude meta)
            for elm_var in site_allvar_df.columns:
                if elm_var in ("gid", "time", "LONGXY", "LATIXY", "zone"):
                    continue

                cols_needed = ['time', 'LONGXY', 'LATIXY', 'zone', elm_var]
                missing = [c for c in cols_needed if c not in site_allvar_df.columns]
                if missing:
                    raise KeyError(f"{site}/{elm_var}: missing columns {missing}")

                site_var_df = site_allvar_df[cols_needed].sort_values('time')

                # Per-variable sanity (should pass the site-level rule already)
                vals = site_var_df[elm_var].to_numpy()
                if not np.isfinite(vals).any():
                    # defensive: should not happen given site-level rule
                    print(f"Skipping {site} / {elm_var}: all values non-finite.")
                    continue

                # Packing params from actual data
                add_offset, scale_factor = eu.elm_var_packing_params(elm_var, data=vals)

                # Initialize + write
                filename = f"ERA5_{elm_var}_{start_year}-{end_year}_z{site_zone_str}.nc"
                path_site_var = write_directory / site / filename

                io.initialize_met_netcdf(
                    this_df_loc, elm_var, dtime_vals, these_dtime_units, path_site_var,
                    add_offset=add_offset, scale_factor=scale_factor, calendar=calendar
                )
                io.append_met_netcdf(site_var_df, elm_var, path_site_var, dtime_vals, 0, dformat='BYPASS')

            # Zone mappings export
            zm_write_path = write_directory / site / "zone_mappings.txt"
            zms[site].to_csv(zm_write_path, index=False, header=False, sep="\t")

        # Cleanup temp parquet
        utils.remove_directory_contents(path_temp_parquet, remove_directory=True)

    else:
        print("Gridded exports not implemented in this branch yet.")

    return


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
