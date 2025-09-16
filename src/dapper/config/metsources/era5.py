# dapper/config/sources/era5.py
"""
ERA5(-Land hourly) → ELM canonical mapping & minimal source config.

Adapters may import this to:
- Know which raw columns to expect (REQUIRED_RAW_COLUMNS)
- Map raw names to canonical ELM names (RAW_TO_ELM)
- Recognize derived columns created during unit conversions / diagnostics
"""

from __future__ import annotations

# Minimal set of raw columns typically required from ERA5-Land hourly
# to produce the full BYPASS/DATM variable suite (some are derived later).
REQUIRED_RAW_COLUMNS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
    "total_precipitation_hourly",
    # lat/lon/time/id columns are handled outside this list
]

# Mapping from ERA5(-Land) *post-conversion* column names → canonical ELM names.
# Include derived fields your adapter may create (e.g., wind_speed, relative_humidity, specific_humidity).
RAW_TO_ELM = {
    "u_component_of_wind_10m": "UWIND",
    "v_component_of_wind_10m": "VWIND",
    "wind_speed": "WIND",  # derived in adapter from u/v
    "surface_solar_radiation_downwards_hourly": "FSDS",
    "surface_thermal_radiation_downwards_hourly": "FLDS",
    "specific_humidity": "QBOT",        # derived in adapter
    "total_precipitation_hourly": "PRECTmms",  # unit-converted in adapter
    "surface_pressure": "PSRF",
    "temperature_2m": "TBOT",
    "dewpoint_temperature_2m": "DTBOT",
    "relative_humidity": "RH",           # derived in adapter
    # coords/time are handled by the exporter/schema, not mapped here:
    # 'lat' -> LATIXY, 'lon' -> LONGXY, 'date'/'time' -> time
}

# Optional: keys that the adapter *may* synthesize during preprocessing.
DERIVED_FIELDS = [
    "wind_speed",          # from u/v
    "relative_humidity",   # from T, Td, Ps
    "specific_humidity",   # from T, Td, Ps
    # "wind_direction"   # not used by ELM; adapter may compute for QA
]

__all__ = [
    "REQUIRED_RAW_COLUMNS",
    "RAW_TO_ELM",
    "DERIVED_FIELDS",
]
