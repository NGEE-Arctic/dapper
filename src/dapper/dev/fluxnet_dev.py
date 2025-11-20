import pandas as pd
from dapper.met.adapters.fluxnet import FluxnetAdapter
from dapper.met.exporter import Exporter

csv_directory=r"X:\Research\NGEE Arctic\8. add flux_tower data option\AMF_US-ICh_FLUXNET_FULLSET_2007-2023_4-6\hourly"
write_directory=r"X:\Research\NGEE Arctic\8. add flux_tower data option\out"

df_loc = pd.DataFrame({
    "gid": ["US-ICh"],     # whatever name you want internally
    "lat": [90.999],
    "lon": [88.888],
    "zone": [1],           # or whatever
})

adapter = FluxnetAdapter()

exp = Exporter(
    adapter=adapter,
    csv_directory=csv_directory,
    write_directory=write_directory,
    df_loc=df_loc,
    calendar="noleap",
    dtime_resolution_hrs=1,
    dformat="BYPASS",
)

exp.run("elm-sites")   # or "elm-combined"


# df = pd.read_csv(r"X:\Research\NGEE Arctic\8. add flux_tower data option\AMF_US-ICh_FLUXNET_FULLSET_2007-2023_4-6\AMF_US-ICh_FLUXNET_FULLSET_HH_2007-2023_4-6.csv")
# print([c for c in df.columns if 'TA' in c])