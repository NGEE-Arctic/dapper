# dapper/config/sources/era5.py
"""
ERA5(-Land hourly) → ELM canonical mapping & minimal source config.

Adapters may import this to:
- Know which raw columns to expect (REQUIRED_RAW_BANDS)
- Map raw names to canonical ELM names (RAW_TO_ELM)
- Recognize derived columns created during unit conversions / diagnostics
"""

from __future__ import annotations

ALL_BANDS = [    "dewpoint_temperature_2m",    "temperature_2m",    "skin_temperature",    "soil_temperature_level_1",    "soil_temperature_level_2",    "soil_temperature_level_3",
                 "soil_temperature_level_4",    "lake_bottom_temperature",    "lake_ice_depth",    "lake_ice_temperature",    "lake_mix_layer_depth",    "lake_mix_layer_temperature",
                 "lake_shape_factor",    "lake_total_layer_temperature",    "snow_albedo",    "snow_cover",    "snow_density",    "snow_depth",    "snow_depth_water_equivalent",
                 "snowfall",    "snowmelt",    "temperature_of_snow_layer",    "skin_reservoir_content",    "volumetric_soil_water_layer_1",    "volumetric_soil_water_layer_2",
                 "volumetric_soil_water_layer_3",    "volumetric_soil_water_layer_4",    "forecast_albedo",    "surface_latent_heat_flux",    "surface_net_solar_radiation",
                 "surface_net_thermal_radiation",    "surface_sensible_heat_flux",    "surface_solar_radiation_downwards",    "surface_thermal_radiation_downwards",    "evaporation_from_bare_soil",
                 "evaporation_from_open_water_surfaces_excluding_oceans",    "evaporation_from_the_top_of_canopy",    "evaporation_from_vegetation_transpiration",    "potential_evaporation",
                 "runoff",    "snow_evaporation",   "sub_surface_runoff",    "surface_runoff",    "total_evaporation",    "u_component_of_wind_10m",    "v_component_of_wind_10m",    "surface_pressure",
                 "total_precipitation",    "leaf_area_index_high_vegetation",    "leaf_area_index_low_vegetation",    "snowfall_hourly",    "snowmelt_hourly",    "surface_latent_heat_flux_hourly",
                 "surface_net_solar_radiation_hourly",    "surface_net_thermal_radiation_hourly",    "surface_sensible_heat_flux_hourly",    "surface_solar_radiation_downwards_hourly",
                 "surface_thermal_radiation_downwards_hourly",    "evaporation_from_bare_soil_hourly",    "evaporation_from_open_water_surfaces_excluding_oceans_hourly",    "evaporation_from_the_top_of_canopy_hourly",
                 "evaporation_from_vegetation_transpiration_hourly",    "potential_evaporation_hourly",    "runoff_hourly",    "snow_evaporation_hourly",    "sub_surface_runoff_hourly",    "surface_runoff_hourly",
                 "total_evaporation_hourly",    "total_precipitation_hourly"
]


# Minimal set of raw columns typically required from ERA5-Land hourly
# to produce the full BYPASS/DATM variable suite (some are derived later).
REQUIRED_RAW_BANDS = [
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
    "REQUIRED_RAW_BANDS",
    "RAW_TO_ELM",
    "DERIVED_FIELDS",
]
