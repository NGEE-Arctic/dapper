import xarray as xr

ds = xr.open_dataset(r"X:\Research\NGEE Arctic\3. Surface Files\workingRelevantSurffiles\surfdata_1x1pt_kougarok_baren-GRID_simyr1850_c360x720_c171002.nc")

lu = xr.open_dataset(r"X:\Research\NGEE Arctic\3. Surface Files\surfdata_map\landuse.timeseries_0.125x0.125_hist_simyr1850-2015_c191004.nc")

ds2 = xr.open_dataset(r"X:\Research\NGEE Arctic\3. Surface Files\workingRelevantSurffiles\surfdata_1x1pt_teller_baren-GRID_simyr1850_c360x720_c171002.nc")

dom = xr.open_dataset(r"X:\Research\NGEE Arctic\3. Surface Files\domain.lnd.1x1pt_Abisko-GRID.nc")
# - what are xc, xv (centroid, vertical?)
# - what is frac, and mask?
# units of area?
# how to represent a watershed? what are x and y if it's just a site?
dom['xc'].item()
dom['xv'][:]
dom['yc'].item()
dom['yv'][:]

dom['frac'].item()
dom['mask']
dom['area']