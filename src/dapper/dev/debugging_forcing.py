import netCDF4 as nc

path = r"X:\Research\NGEE Arctic\4. Using Dapper\Rerun all sites (points)\gee_csvs\elm_formatted\Abisko\ERA5_PRECTmms_1950-2024_z01.nc"

ds = nc.Dataset(path)

dtime = ds.variables['DTIME']

# Print the first two values
print(dtime[:3])
hours = (dtime[2] - dtime[1])*24
npf = 86400*(dtime[2]-dtime[1])/1800