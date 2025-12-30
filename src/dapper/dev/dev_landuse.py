from dapper.domains.aoi import AOI
from dapper.landuse.sample import sample_landuse_timeseries
from dapper.utils.pathing import LANDUSE_EIGHTH_DEGREE

aoi = AOI.from_point(lon=-149.59429, lat=68.62758, name="ToolikLake", gid="ToolikLake")
domain = aoi.to_domain_points()
domain_nc = domain.write_elm_domain("domain.lnd.1x1pt_ToolikLake-GRID.nc")

df_loc = domain.to_df_loc()  # lon/lat/weight

landuse_nc, df_cells = sample_landuse_timeseries(
    src_path=LANDUSE_EIGHTH_DEGREE,
    df_loc=df_loc.rename(columns={"lon": "lon", "lat": "lat"}),  # likely already matches
    out_path="landuse.timeseries_1x1pt_ToolikLake-GRID_simyr1850-2015.nc",
    output_lon_wrap="0_360",  # if you want output LONGXY to look like your Toolik file
)
