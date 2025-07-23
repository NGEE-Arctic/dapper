import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

path = r'X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\elm_formatted\Teller'
# Parameters
start_date = "2017-01-01"
end_date = "2020-01-01"
files = [f for f in os.listdir(path) if f.endswith(".nc")]  # adjust as needed

# Plotting setup
fig, axs = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

# Loop over files
for i, f in enumerate(files):
    if i >= 7:
        break

    ds = xr.open_dataset(os.path.join(path, f), decode_times=False)
    
    # Manually decode DTIME if needed
    if 'DTIME' in ds.variables:
        time_units = ds['DTIME'].attrs.get('units', 'hours since 1900-01-01 00:00:0.0')
        calendar = ds.attrs.get('calendar', 'standard')
        ds['DTIME'] = xr.decode_cf(ds[['DTIME']].assign_coords(DTIME=ds['DTIME']),
                                   decode_times=True)['DTIME']

    varname = [v for v in ds.data_vars if v not in ['LATIXY', 'LONGXY']][0]
    da = ds[varname].isel(n=0)

    # Time filter
    da = da.sel(DTIME=slice(start_date, end_date))

    axs[i].plot(da.DTIME.values, da.values)
    axs[i].set_title(varname)
    axs[i].grid(True)

# Turn off the last empty panel (8th one)
axs[-1].axis("off")

# Shared x-labels and formatting
fig.suptitle("Teller from dapper", fontsize=16)
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
axs[-2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.show()

## Testing Cansu's ELM run
fpath = r'X:\Research\NGEE Arctic\5. UK and IC\test_merge'
files = [f for f in os.listdir(fpath) if f.endswith(".nc")]  # adjust as needed
for i, f in enumerate(files):
    ds = xr.open_dataset(os.path.join(fpath, f), decode_times=False)

import os
import xarray as xr
import pandas as pd
import cftime

fpath = r'X:\Research\NGEE Arctic\5. UK and IC\test_merge'
files = [f for f in os.listdir(fpath) if f.endswith(".nc")]

dfs = []
for f in files:
    ds = xr.open_dataset(os.path.join(fpath, f), decode_times=True)
    
    # Extract QRUNOFF and its time dimension
    if "QRUNOFF" in ds:
        qrunoff = ds["QRUNOFF"]
        # Convert to DataFrame
        df = qrunoff.to_dataframe().reset_index()
        df["source_file"] = f  # optional: keep track of source
        dfs.append(df)
    else:
        print(f"QRUNOFF not found in {f}")

# Concatenate all into one DataFrame
result_df = pd.concat(dfs, ignore_index=True)
result_df.sort_values(by='time', inplace=True)

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(result_df["time"], result_df["QRUNOFF"])

# Format major (year) and minor (month) ticks
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

# Rotate and adjust month (minor) labels
plt.setp(ax.xaxis.get_minorticklabels(), rotation=90, fontsize=8)

# Manually offset the year (major) labels
for label in ax.get_xticklabels(minor=False):
    label.set_y(-0.1)  # move year labels lower (default is ~0)

# Titles and labels
ax.set_title("QRUNOFF")
ax.set_xlabel("Time")
ax.set_ylabel("QRUNOFF")
ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# Paths and setup
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os

# Paths and setup
pdapper = r'X:\Research\NGEE Arctic\5. UK and IC\dapper\gee_csvs\elm_formatted\Imnaviat'
pplab = r'X:\Research\NGEE Arctic\5. UK and IC\imnaviat_preplab'

dapperfiles = os.listdir(pdapper)
plabfiles = os.listdir(pplab)

vars = ['FLDS', 'FSDS', 'PRECTmms', 'PSRF', 'QBOT', 'TBOT', 'WIND']
plot_start = '1980-01-01'
plot_end = '1985-01-01'

# Setup time series figure
nvars = len(vars)
fig_ts, axs_ts = plt.subplots(nvars, 1, figsize=(12, 2.5 * nvars), sharex=True)

# Setup scatterplot figure
fig_scatter, axs_scatter = plt.subplots(4, 2, figsize=(12, 10))
axs_scatter = axs_scatter.flatten()

for i, v in enumerate(vars):
    this_dapper = [f for f in dapperfiles if v in f][0]
    this_plab = [f for f in plabfiles if v in f][0]

    ds_dap = xr.open_dataset(os.path.join(pdapper, this_dapper), decode_cf=False)
    ds_plab = xr.open_dataset(os.path.join(pplab, this_plab))

    if v == 'WIND':
        da_dap = ds_dap['WIND'].sel(n=0).sortby('DTIME').sel(DTIME=slice(plot_start, plot_end))
        da_plab = ds_plab['UWIND'].sel(n=0).sortby('DTIME').sel(DTIME=slice(plot_start, plot_end))
    else:
        da_dap = ds_dap[v].sel(n=0).sortby('DTIME').sel(DTIME=slice(plot_start, plot_end))
        da_plab = ds_plab[v].sel(n=0).sortby('DTIME').sel(DTIME=slice(plot_start, plot_end))

    # --- Time Series Plot ---
    ax_ts = axs_ts[i]
    ax_ts.plot(da_dap['DTIME'], da_dap, label='DAPPER')
    ax_ts.plot(da_plab['DTIME'], da_plab, label='PREPLAB', alpha=0.7)
    ax_ts.set_ylabel(v)
    ax_ts.set_title(f"{v} Time Series", fontsize=10)
    ax_ts.grid(True, alpha=0.3)

    # --- Scatter Plot ---
    ax_sc = axs_scatter[i]
    x = da_plab.values.flatten()
    y = da_dap.values.flatten()

    ax_sc.scatter(x, y, s=1, alpha=0.3)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax_sc.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    ax_sc.set_title(f"{v} Scatter: PREPLAB vs DAPPER", fontsize=9)
    ax_sc.set_xlabel("PREPLAB")
    ax_sc.set_ylabel("DAPPER")
    ax_sc.grid(True, alpha=0.3)


for j in range(len(vars), len(axs_scatter)):
    axs_scatter[j].axis('off')

# Final touches
axs_ts[-1].set_xlabel("Time")
axs_ts[0].legend(loc='upper right')

plt.tight_layout()
fig_ts.tight_layout()
fig_scatter.tight_layout()
plt.show()

p = r"X:\Research\NGEE Arctic\dapper\data\fengming_data\era5\Daymet_ERA5.1km_TBOT_1980-2023_z01.nc"
dsf = xr.open_dataset(p)
da_dsf = dsf[v].sel(n=0).sortby('DTIME').sel(DTIME=slice(plot_start, plot_end))

