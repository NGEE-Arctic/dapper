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
    from dapper.utils import _DATA_DIR

    return pd.read_csv(_DATA_DIR / "e5lh_band_metadata.csv")


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
    geometries_fc = gu.ensure_pixel_centers_within_geometries(
        geometries_fc, sample_img, scale
    )

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
            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=export_filename,
                folder=params["gdrive_folder"],
                fileFormat="CSV",
                selectors=[params["geometry_id_field"], "date"] + params["gee_bands"],
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
    dtime_units='days',
    dtime_resolution_hrs=1,
    nzones=1,
    dformat="BYPASS",
):
    """
    Under construction:
        - needs to check for longitudinal range when exporting (0-360)
        - needs to be exapanded and re-tested for gridded exports

    Batched version for grids.

    compress_level - higher will compress more but take longer to write

    """
    compress_level = 0 # hardcoding no compression because there are issues when both packing and using compression (outputs are all fillvalue) that I couldn't solve
    if dformat not in ["DATM_MODE", "BYPASS"]:
        raise KeyError("You provided an unsupported dformat value. Currently only DATM_MODE and BYPASS are available.")
    elif dformat == "DATM_MODE":
        print("DATM_MODE is not yet available. Exiting.")
        return

    if type(csv_directory) is str:
        csv_directory = Path(csv_directory)
    if type(write_directory) is str:
        write_directory = Path(write_directory)

    # ELM/E3SM operate on a longitudinal range of 0-360, so convert from -180 to 180 if necessary
    df_loc['lon_0-360'] = np.mod(df_loc['lon'], 360)

    # Determine our date range to make sure we provide only complete years of data
    csv_files = [os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if os.path.splitext(f)[1] == ".csv"]
    start_year, end_year = io.get_start_end_years(csv_files, calendar=calendar)

    # Rename id field for consistency to 'gid'
    if id_col is None:
        id_col = gu.infer_id_field(df_loc)
    df_loc.rename(columns={id_col: "gid"}, inplace=True)

    # Prepare the netCDF grid
    df_loc = df_loc.sort_values(by=["lat", "lon"]).reset_index(drop=True)
    df_loc["gid"] = df_loc["gid"].astype(str)

    # Account for zones if not already provided in df_loc
    if "zone" not in df_loc.columns:
        df_loc["zone"] = np.tile(np.arange(1, nzones + 1), (len(df_loc) // nzones) + 1)[: len(df_loc)]
    unique_zones = list(set(df_loc["zone"]))

    # Create directory for storing results
    utils.make_directory(write_directory, delete_all_contents=True)

    # If exporting sites individually, for speed of processing and writing we use an intermediate
    # parquet file. This reduces runtime by orders of magnitude (as opposed to writing each variable, 
    # each time chunk one-at-a-time)
    if gridded is False: # site-specific exports

        # Create temporary parquet file storage directory
        path_temp_parquet = write_directory / 'temp_parquet'
        utils.make_directory(path_temp_parquet, delete_all_contents=True)

        # Read each file, format it, and compile into one parquet dataframe per site
        for i, f in enumerate(csv_files):
            print(f"Processing file {i+1} of {len(csv_files)}: {f}")
            this_df = pd.read_csv(f)

            # Ensure id_col is set properly
            this_df.rename(columns={id_col: "gid"}, inplace=True)

            # Transform data to ELM-ready
            this_df = this_df.merge(df_loc[["gid", "lat", "lon", "zone"]], on="gid", how="inner")
            remove_leap = False
            if calendar == 'noleap':
                remove_leap = True
            ppdf = _preprocess_e5lh_to_elm_file_grid(this_df, start_year, end_year, remove_leap, dformat)
            ppdf = ppdf.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

            # Split by location and save to parquet
            ppdfg = ppdf.groupby(by="gid")
            for gid, this_df in ppdfg:
                parquet_path = path_temp_parquet / f"{gid}.parquet"
                if os.path.isfile(parquet_path):
                    write(parquet_path, this_df, append=True)
                else:
                    write(parquet_path, this_df)

        zms = eu.gen_zone_mappings(df_loc, site=True)

        # Read merged parquet files and store as netCDFs
        parquet_files = path_temp_parquet.glob("*.parquet")
        for pf in parquet_files:
            site = pf.stem
            print(f"Exporting {site}...")
            site_allvar_df = pd.read_parquet(pf)
            
            # Handle time resampling 
            dtime_vals, these_dtime_units, unique_times = io.create_dtime(site_allvar_df, calendar=calendar, dtime_units=dtime_units, dtime_resolution_hrs=dtime_resolution_hrs)

            # Ensure that re-loaded data matches DTIME ordering computed earlier. DTIME (unique_times) is ordered already, so this also ensures date sorting.
            site_allvar_df = site_allvar_df.set_index('time')
            site_allvar_df = site_allvar_df.reindex(unique_times)
            site_allvar_df.index.name = 'time'
            site_allvar_df.reset_index(inplace=True)  # Restores 'time' as column

            # Make storage directory
            utils.make_directory(write_directory / site, delete_all_contents=True)


            # Pull out the location information and sampling metadata for this site            
            this_df_loc = df_loc[df_loc['gid']==site]

            # Store each variable in the file
            for elm_var in site_allvar_df.columns:
                if elm_var in ["gid", "time", "LONGXY", "LATIXY", "zone"]:
                    continue
                filename = 'ERA5_' + elm_var + '_' + str(start_year) + '-' + str(end_year) + '_z' + str(df_loc['zone'].values[df_loc['gid']==site][0]).zfill(2) + '.nc'
                path_site_var = write_directory / site / filename
                site_var_df = site_allvar_df[site_allvar_df['gid']==site]
                site_var_df = site_var_df[[elm_var, 'time', 'gid', 'LONGXY', 'LATIXY', 'zone']]
                site_var_df = site_var_df.sort_values(by='time')
                
                # Compute packing parameters using the data range, not preset range
                add_offset, scale_factor = eu.elm_var_packing_params(elm_var, data=site_var_df[elm_var].values)

                # Initialize and write netcdf for each variable
                io.initialize_met_netcdf(this_df_loc, elm_var, dtime_vals, these_dtime_units, path_site_var, add_offset=add_offset, scale_factor=scale_factor, calendar=calendar)
                io.append_met_netcdf(site_var_df, elm_var, path_site_var, dtime_vals, 0, dformat='BYPASS')

            # Zone mappings export
            zm_write_path = write_directory / site / "zone_mappings.txt"
            zms[site].to_csv(zm_write_path, index=False, header=False, sep="\t")

    # Remove temporary files
    utils.remove_directory_contents(path_temp_parquet, remove_directory=True)

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
