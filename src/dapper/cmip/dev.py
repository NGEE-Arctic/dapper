## Pangeo approach - https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html
import intake
import xarray as xr
import fsspec
import pandas as pd
from pathlib import Path

params = {
    'models' : ['BCC-CSM2-MR', 'CanESM5', 'CESM2', 'E3SM-1-0', 'E3SM-1-1', 'EC-Earth3', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-LM'],
    'variables' : ['pr', 'tas'],
    'experiment' : 'historical',
    'table' : ['Amon'],
    'ensemble' : 'r1i1p1f1',
    'start_date' : '1850-01-01',
    'end_date' : '2014-12-31'
}
path_out = Path(r'X:\Research\NGEE Arctic\CMIP output\Katrinas\Kurts_Paper')


# lat, lon = 64.7503, -165.9508 # Example: Teller



# Load the Pangeo CMIP6 ESM Collection
print('Loading Pangeo CMIP6 catalog...')
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
print('Done.')

time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)

for experiment in params['experiment']:
    print(experiment)
    
    # Step 1: Find models that contain ALL required variables
    query = col.search(
        experiment_id = experiment,
        table_id = params['table'],
        variable_id = params['variables'],
        member_id = params['ensemble'],
        source_id = params['models']
    )

    # Group results by model and check if all variables are available
    matching_models = query.df.groupby("source_id")["variable_id"].apply(set)
    valid_models = matching_models[matching_models.apply(lambda x: set(params['variables']).issubset(x))]

    # Step 2: Loop over valid models and extract point-based data
    for model in valid_models.index:

        print(f"Processing model: {model}")

        # Get dataset URLs for this model
        model_datasets = query.df[query.df["source_id"] == model]

        datasets = {}
        for var in variables:
            # Find the correct dataset URL for this variable
            zarr_url = model_datasets[model_datasets["variable_id"] == var].iloc[0].zstore
            ds = xr.open_zarr(fsspec.get_mapper(zarr_url), consolidated=True, decode_times=time_coder)

            # Automatically detect latitude & longitude coordinate names
            lat_name = [name for name in ['lat', 'latitude', 'nav_lat', 'Y'] if name in ds.coords][0]
            lon_name = [name for name in ['lon', 'longitude', 'nav_lon', 'X'] if name in ds.coords][0]

            # Ensure consistent coordinates & select nearest grid point
            ds = ds.sel({lon_name: lon, lat_name: lat}, method="nearest")

            # Keep only time and variable, dropping unnecessary metadata
            ds = ds[[var]].drop_vars(['bnds', 'lat_bnds', 'lon_bnds', 'time_bnds', 'height'], errors="ignore")

            datasets[var] = ds

        # Merge datasets with aligned dimensions
        ds_combined = xr.merge(datasets.values(), compat="override")

        # Convert to Pandas DataFrame
        df = ds_combined.to_dataframe().reset_index()

        # Clip to requested time range
        # Convert cftime objects to standard datetime64
        df['time'] = df['time'].astype(str)  # Convert to string first
        df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Convert to datetime
        df = df[(df["time"] >= pd.to_datetime(start_date)) & (df["time"] <= pd.to_datetime(end_date))]

        # Save DataFrame to CSV
        filename = f"CMIP6_{model}_{experiment}_Teller.csv"
        df.to_csv(path_out / filename, index=False)
        print(f"Saved: {filename}")

    for _, row in query.df.iterrows():
        download_pangeo_cmip_file(row['zstore'])

import gcsfs
import xarray as xr
import os
from dapper import utils 
# Create a GCS file system object (anonymous for public buckets)
fs = gcsfs.GCSFileSystem(token='anon')

file_urls = query.df['zstore'].unique()  # or include duplicates if needed
dir_out = utils._DATA_DIR / 'CMIP_Downloads'
utils.make_directory(dir_out, delete_all_contents=True)
# Download the files
for i, row in query.df.iterrows():
    filename = f"{row.variable_id}_{row.source_id}_{row.experiment_id}_{row.member_id}.nc"
    try:
        ds = xr.open_zarr(fsspec.get_mapper(row.zstore), consolidated=True, decode_times=time_coder)
        this_file_out = dir_out / filename
        ds.to_netcdf(this_file_out)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Failed to load {filename}: {e}")


ds = xr.open_dataset(r"X:\Research\NGEE Arctic\dapper\data\CMIP_Downloads\pr_BCC-CSM2-MR_historical_r1i1p1f1.nc")

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
