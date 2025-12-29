from dapper.utils.pathing import LANDUSE_EIGHTH_DEGREE

import xarray as xr
ds = xr.open_dataset(LANDUSE_EIGHTH_DEGREE, decode_times=False)
print(ds.dims)
print([v for v in ds.variables if v.lower() in ["lat","lon","lsmlat","lsmlon","latixy","longxy","time","year"]])
print("has LATIXY/LONGXY:", "LATIXY" in ds.variables, "LONGXY" in ds.variables)
print("example vars:", list(ds.data_vars)[:30])
