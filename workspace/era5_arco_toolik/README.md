# Toolik ERA5-Land ARCO smoke test

This runner samples the Toolik watershed with Dapper's ERA5-Land dispatcher,
then converts the resulting CSV to hourly, noleap ELM forcing files. The
watershed data stay outside the repository.

```powershell
python workspace/era5_arco_toolik/sample_toolik.py `
  "X:\Research\NGEE Arctic\4. Using Dapper\watershed_polygons\Watersheds\tool_wgs84_utm_z6n_epsg32606.shp" `
  "X:\Research\NGEE Arctic\4. Using Dapper\watershed_polygons\arco_toolik"
```

The ARCO product starts on 1950-01-02. The script preserves that partial first
year and the partial latest year by explicitly setting
`clip_to_full_years=False`. Production runs that require only complete calendar
years should set it to `True`; Dapper will then drop incomplete boundary years
and raise if the input contains no complete year at all. A request beginning on
1950-01-01 warns and clamps to January 2 rather than automatically requiring
GEE.

## September 2026 benchmark

All timings used the eight variables needed by the ELM exporter:

| Request | Elapsed time |
| --- | ---: |
| One point, one year | 37.8 seconds |
| Three points, one year | 104.8 seconds |
| Toolik watershed, five cells, full record | 65.1 seconds |

The three-point request submits one CDS job per site. The full Toolik polygon is
one small-area request followed by Dapper's local intersection-area-weighted
mean, so it avoids most of that per-request queue overhead.
