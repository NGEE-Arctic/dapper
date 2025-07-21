## Pangeo approach - https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html
import os
import gcsfs
import intake
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path

from dapper.elm import elm_utils as euts

# Need a smarter way to import this
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

def find_available_data(params):

    # params['variables'] = 'ps'
    if 'variables' in params and params['variables'] == 'elm':
        params['variables'] = euts.elm_data_dicts()['cmip_req_vars']
    
    param_mapping = {
        'experiment': 'experiment_id',
        'table': 'table_id',
        'variables': 'variable_id',
        'ensemble': 'member_id',
        'models': 'source_id'
    }

    # Build the search arguments
    search_args = {
        intake_key: params[key]
        for key, intake_key in param_mapping.items()
        if key in params
    }

    # Perform the search
    matches = col.search(**search_args)

    df = matches.df.copy()
    # Step 1: Find models that have both 'pr' and 'tas'
    grouped = df.groupby('source_id')
    keep = []
    for model, g in grouped:
        # print(len(g))
        # if len(g) > 7:
        #     break
        if all(x in g['variable_id'].values for x in params['variables']):
            keep.extend(g.index.tolist())
    df_export = df.iloc[keep]
    # print(df_export) # Now we have 10 models

    return df_export


def download_pangeo(df, dir_out, lat=None, lon=None, lat_bounds=None, lon_bounds=None, polygon_path=None):
    """
    Download CMIP6 data from Pangeo, with optional spatial subsetting.

    Args:
        df (pd.DataFrame): From find_available_data().
        dir_out (Path or str): Output directory.
        lat, lon (float): Point location to sample (optional).
        lat_bounds, lon_bounds (tuple): Box to sample (min, max) (optional).
        polygon_path (str): Path to GeoJSON or Shapefile with polygon (optional).

    Note: Only one of (lat/lon), (lat_bounds/lon_bounds), or (polygon_path) should be provided.
    """
    os.makedirs(dir_out, exist_ok=True)

    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    for i, row in df.iterrows():
        filename = f"{row.variable_id}_{row.source_id}_{row.experiment_id}_{row.member_id}.nc"
        try:
            ds = xr.open_zarr(fsspec.get_mapper(row.zstore, token='anon', access='read_only'), consolidated=True, decode_times=time_coder)

            # Normalize longitude to 0–360 if needed
            if 'lon' in ds.coords and ds.lon.max() > 180:
                if lon is not None and lon < 0:
                    lon = lon % 360
                if lon_bounds is not None:
                    lon_bounds = tuple(l % 360 for l in lon_bounds)

            # Point sampling
            if lat is not None and lon is not None:
                ds = ds.sel(lat=lat, lon=lon, method='nearest')

            # Box selection
            elif lat_bounds and lon_bounds:
                ds = ds.sel(
                    lat=slice(lat_bounds[0], lat_bounds[1]),
                    lon=slice(lon_bounds[0], lon_bounds[1])
                )

            # Polygon masking (optional, will still save entire box but mask values)
            elif polygon_path is not None:
                gdf = gpd.read_file(polygon_path)
                poly = gdf.unary_union

                # First select a box to speed things up
                minx, miny, maxx, maxy = poly.bounds
                ds = ds.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))

                # Mask outside polygon
                lon2d, lat2d = np.meshgrid(ds.lon, ds.lat)
                points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()))
                mask = np.array([poly.contains(pt) for pt in points]).reshape(lat2d.shape)
                ds = ds.where(mask)

            # Save output
            this_file_out = os.path.join(dir_out, filename)
            ds.to_netcdf(this_file_out)
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")


def extract_vars_from_files(files, start_date, end_date, path_out):
    """
    Robust CMIP6 NetCDF merger for multiple calendars — using CFDatetimeCoder.

    This is slow but robust.
    """
    all_dfs = []
    time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

    for file in tqdm(files, desc="Processing"):
        try:
            ds = xr.open_dataset(file, decode_times=time_coder)

            varnames = [v for v in ds.data_vars if {'time', 'lat', 'lon'}.intersection(ds[v].dims)]
            for var in varnames:
                arr = ds[var]
                time = ds['time'].values

                # If time is cftime, use safe bounds
                if isinstance(time[0], np.datetime64):
                    times = pd.to_datetime(time)
                    mask = (times >= start_date) & (times <= end_date)
                else:
                    times = time
                    mask = np.array([ (t >= cftime_date(start_date, t)) and (t <= cftime_date(end_date, t)) for t in time ])

                values = arr.values[mask]
                filtered_times = np.array(times)[mask]

                lon = ds['lon'].values.item() if ds['lon'].size == 1 else ds['lon'].values
                lat = ds['lat'].values.item() if ds['lat'].size == 1 else ds['lat'].values

                parts = Path(file).stem.split('_')
                varname = var
                model = parts[1]
                ssp = parts[2]

                df = pd.DataFrame({
                    'date': filtered_times,
                    'lon': lon,
                    'lat': lat,
                    'value': values,
                    'var': varname,
                    'model': model,
                    'ssp': ssp
                })
                all_dfs.append(df)

        except Exception as e:
            print(f"Failed: {file} — {e}")

    if all_dfs:
        out_df = pd.concat(all_dfs, ignore_index=True)
        # Convert cftime safely to ISO string for storage
        if not np.issubdtype(out_df['date'].dtype, np.datetime64):
            out_df['date'] = out_df['date'].astype(str)
        out_df.to_csv(path_out, index=False)
        print(f"Saved to {path_out}")
    else:
        print("No valid data extracted.")


def cftime_date(string_date, sample_cftime):
    """
    Convert YYYY-MM-DD to same cftime type as sample_cftime
    """
    import cftime
    y, m, d = map(int, string_date.split('-'))
    if isinstance(sample_cftime, cftime.DatetimeNoLeap):
        return cftime.DatetimeNoLeap(y, m, d)
    elif isinstance(sample_cftime, cftime.Datetime360Day):
        return cftime.Datetime360Day(y, m, min(d, 30))
    else:
        return cftime.DatetimeProlepticGregorian(y, m, d)


# # Load the Pangeo CMIP6 ESM Collection
# for experiment in params['experiment']:
#     print(experiment)
    
#     # Step 1: Find models that contain ALL required variables
#     # Group results by model and check if all variables are available
#     matching_models = query.df.groupby("source_id")["variable_id"].apply(set)
#     valid_models = matching_models[matching_models.apply(lambda x: set(params['variables']).issubset(x))]

#     # Step 2: Loop over valid models and extract point-based data
#     for model in valid_models.index:

#         print(f"Processing model: {model}")

#         # Get dataset URLs for this model
#         model_datasets = query.df[query.df["source_id"] == model]

#         datasets = {}
#         for var in variables:
#             # Find the correct dataset URL for this variable
#             zarr_url = model_datasets[model_datasets["variable_id"] == var].iloc[0].zstore
#             ds = xr.open_zarr(fsspec.get_mapper(zarr_url), consolidated=True, decode_times=time_coder)

#             # Automatically detect latitude & longitude coordinate names
#             lat_name = [name for name in ['lat', 'latitude', 'nav_lat', 'Y'] if name in ds.coords][0]
#             lon_name = [name for name in ['lon', 'longitude', 'nav_lon', 'X'] if name in ds.coords][0]

#             # Ensure consistent coordinates & select nearest grid point
#             ds = ds.sel({lon_name: lon, lat_name: lat}, method="nearest")

#             # Keep only time and variable, dropping unnecessary metadata
#             ds = ds[[var]].drop_vars(['bnds', 'lat_bnds', 'lon_bnds', 'time_bnds', 'height'], errors="ignore")

#             datasets[var] = ds

#         # Merge datasets with aligned dimensions
#         ds_combined = xr.merge(datasets.values(), compat="override")

#         # Convert to Pandas DataFrame
#         df = ds_combined.to_dataframe().reset_index()

#         # Clip to requested time range
#         # Convert cftime objects to standard datetime64
#         df['time'] = df['time'].astype(str)  # Convert to string first
#         df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Convert to datetime
#         df = df[(df["time"] >= pd.to_datetime(start_date)) & (df["time"] <= pd.to_datetime(end_date))]

#         # Save DataFrame to CSV
#         filename = f"CMIP6_{model}_{experiment}_Teller.csv"
#         df.to_csv(path_out / filename, index=False)
#         print(f"Saved: {filename}")

#     for _, row in query.df.iterrows():
#         download_pangeo_cmip_file(row['zstore'])

# import gcsfs
# import xarray as xr
# import os
# from dapper import utils 
# # Create a GCS file system object (anonymous for public buckets)
# fs = gcsfs.GCSFileSystem(token='anon')

# file_urls = query.df['zstore'].unique()  # or include duplicates if needed
# dir_out = utils._DATA_DIR / 'CMIP_Downloads'
# utils.make_directory(dir_out, delete_all_contents=True)
# # Download the files
# for i, row in query.df.iterrows():
#     filename = f"{row.variable_id}_{row.source_id}_{row.experiment_id}_{row.member_id}.nc"
#     try:
#         ds = xr.open_zarr(fsspec.get_mapper(row.zstore), consolidated=True, decode_times=time_coder)
#         this_file_out = dir_out / filename
#         ds.to_netcdf(this_file_out)
#         print(f"Saved: {filename}")
#     except Exception as e:
#         print(f"Failed to load {filename}: {e}")



# def get_pangeo_data_catalog(redownload=False):
#     """
#     Downloads the Pangeo CMIP6 catalog. Run if you want to make sure you're working
#     with the latest version.
#     """

#     CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
#     CATALOG_LOCAL = duts._DATA_DIR / "pangeo-cmip6.pkl"

#     # Download the catalog only if it's not already present
#     if not os.path.exists(CATALOG_LOCAL) or redownload is True:
#         col = intake.open_esm_datastore(CATALOG_URL)
#         df = col.df
#         df.to_pickle(CATALOG_LOCAL)

#     with open("col_object.pkl", "rb") as f:
#         col = pickle.load(f)


# pyesgf is problematic because of the different nodes you have to search, the fact that
# not all datasets are not opendap compliant (no sampling without downloading the full dataset)
# and organization is not guaranteed. For example, in the below code I was finding hourly datasets
# broken up by decade in folders labeled "Amon" (monthly). This might be a misunderstanding on
# my part about how data are organized, but the bottom line was that the complexity was too
# much to keep pursuing when a much simpler, better organized, and easier-to-use interface
# was available from pangeo/intake-esm. 
# from pyesgf.search import SearchConnection
# import xarray as xr
# import pandas as pd
# import os
# os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "1" # turns off an annoying facets warning

# # Connect to ESGF node (distributed search enabled)
# conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)

# # Define search parameters
# experiments = ['ssp126', 'ssp245', 'ssp585']  # SSP1-2.6, SSP2-4.5, SSP4-8.5
# variables = ['ps', 'pr', 'rsds', 'huss', 'tas', 'uas', 'vas']
# table_id = 'Amon'  # Monthly atmospheric data
# member_id = 'r1i1p1f1'

# # Search for datasets matching the criteria
# facets = 'project,experiment_id,source_id,variable_id,member_id'
# from_timestamp="2100-12-30T23:23:59Z", to_timestamp="2200-01-01T00:00:00Z"
# results = []
# od = []
# for exp in experiments:
#     print(exp)
#     for var in variables:
#         print(var)
#         ctx = conn.new_context(
#             project='CMIP6',
#             experiment_id=exp,
#             variable=var,
#             member_id=member_id,
#             table_id=table_id,
#             facets=facets,
#             from_timestamp="2015-12-30T23:23:59Z", 
#             to_timestamp="2100-01-01T00:00:00Z",
#         )

#         search_results = ctx.search()
#         filtered_results = [res for res in search_results if "esgf-data1.llnl.gov" not in res.dataset_id]
#         results.extend(filtered_results)
#         odurl = [fr.opendap_url for fr in filtered_results if fr.opendap_url is not None]
#         od.extend(odurl)
#         if len(od) > 0:
#             break

# print(f"Found {len(results)} matching datasets:\n")

# for result in filtered_results:
#     print(f"Dataset: {result.dataset_id}, Table: {result.metadata.get('table_id', 'Unknown')}")


# # Extract OPeNDAP URLs
# opendap_urls = []
# for i, result in enumerate(results):
#     print(i, str(len(results)))
#     file_ctx = result.file_context()
#     files = file_ctx.search()  # Get available files

#     for file in files:
#         if file.opendap_url:  # Only keep OPeNDAP-accessible datasets
#             if 'Amon' in file.opendap_url:
#                 opendap_urls.append(file.opendap_url)



# # sample the point data
# # Toolik coordinates
# lat = 68.6275
# lon = -149.5981

# def single_ds_to_df(ds, varname):
#     # Convert xarray DataArray to pandas DataFrame
#     df = ds.to_dataframe().reset_index()
#     # Rename the value column to the variable name
#     df = df.rename(columns={ds.name: varname})

#     # Keep only time and the renamed variable column
#     df = df[['time', varname]]

#     return df

# def sample_point_data(opendap_url, lon, lat):
#     try:
#         ds = xr.open_dataset(opendap_url)  # Load dataset remotely
#         variable = ds.variable_id
#         point_data = ds[variable].sel(lon=lon, lat=lat, method="nearest")
#         df = single_ds_to_df(point_data, variable)
#         return df
#     except Exception as e:
#         print(f"Error loading {opendap_url}: {e}")
#         return None

# cdata={}
# for i, url in enumerate(opendap_urls):
#     print(i)
#     ret = sample_point_data(url, lon, lat)
#     if ret is not None:
#         cdata[os.path.basename(url).split('.')[0]] = ret
#     if len(cdata) > 5:
#         break
