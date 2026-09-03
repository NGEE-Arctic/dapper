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
year and the partial latest year rather than clipping either one.

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
