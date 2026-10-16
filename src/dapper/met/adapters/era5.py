"""ERA5-Land adapter implementation."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dapper.config.metsources.era5 import RAW_TO_ELM
from dapper.elm import utils as eu  # for compute_humidities, packing defaults
from dapper.met import temporal as dt
from dapper.met.adapters.base import BaseAdapter
from dapper.schemas.elm import elm_required_vars, is_nonnegative


class ERA5Adapter(BaseAdapter):
    """ERA5-Land hourly → ELM adapter.

    Handles ERA5-specific file discovery, unit conversions, humidity
    diagnostics, renaming to ELM short names, and nonnegativity so the
    :class:`~dapper.met.exporter.Exporter` stays source-agnostic.

    ``preprocess_shard`` steps:

    1. keep rows from Jan 1 of ``start_year`` through Jan 1 00:00 of
       ``end_year + 1`` (a one-hour lookahead for end-labeled fluxes); Feb 29 is
       kept here and dropped later during temporal alignment
    2. unit conversions: J/m² per hour → W/m² (÷3600), m/hr → mm/s (÷3.6),
       wind speed from u/v
    3. RH and specific humidity when temperature, dewpoint, and surface
       pressure are all present
    4. rename to ELM short names via
       :data:`dapper.config.metsources.era5.RAW_TO_ELM`
    5. clip canonical nonnegative variables at 0
    6. return the format's required variables plus
       ``LONGXY, LATIXY, time, gid, zone``, sorted by time and location

    Accumulated fields (FSDS, FLDS, PRECTmms) are labeled at interval end;
    :meth:`temporal_options` tells the exporter to relabel them to interval
    start.
    """

    # NetCDF provenance metadata
    SOURCE_NAME = "ERA5-Land hourly reanalysis"
    DRIVER_TAG = "ERA5"
    INTERVAL_END_VARS = ("FSDS", "FLDS", "PRECTmms")
    SOURCE_INTERVAL_HOURS = 1.0
    GEE_COLLECTION_START = pd.Timestamp("1950-01-01 01:00:00")

    # ---------------- discovery & locations ----------------

    def discover_files(self, csv_directory, calendar, *, clip_to_full_years=None):
        """Discover ERA5 CSV shards in a directory and infer the inclusive year range."""

        csv_directory = Path(csv_directory)

        # ignore directories; only pick real files that end with .csv (case-insensitive)
        csv_files = [
            str(p)
            for p in csv_directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ]

        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {csv_directory}")

        if clip_to_full_years is None:
            clip_to_full_years = True
        start_year, end_year = dt.get_start_end_years(
            csv_files,
            calendar=calendar,
            clip_to_full_years=bool(clip_to_full_years),
        )
        return csv_files, start_year, end_year

    def id_column_for_csv(self, df_csv, id_col):
        """Return the identifier column name expected in ERA5 CSV shards ("gid").

        Deprecated: the Exporter always uses ``gid`` and never calls this.
        """
        warnings.warn(
            "ERA5Adapter.id_column_for_csv is deprecated and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )

        if "gid" not in df_csv.columns:
            raise KeyError("Expected 'gid' column in input CSV.")
        return "gid"

    # ---------------- preprocessing & requirements ----------------

    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        """Convert one merged CSV shard to canonical ELM columns (see class docstring)."""
        df = df_merged.copy()

        # --- time handling ---
        if "date" not in df.columns:
            raise KeyError("Expected 'date' column in the CSV shard.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        export_start = pd.Timestamp(year=start_year, month=1, day=1)
        lookahead_end = pd.Timestamp(year=end_year + 1, month=1, day=1)
        df = df[df["date"].between(export_start, lookahead_end, inclusive="both")]

        # Keep Feb 29 until temporal alignment. For end-labeled hourly fields,
        # Feb 29 00:00 supplies the Feb 28 23:00 interval in a noleap export.

        # --- ERA5-specific unit conversions (kept local to adapter) ---
        df = self._unit_conversions(df)

        # --- humidities if possible ---
        needed = {"temperature_2m", "dewpoint_temperature_2m", "surface_pressure"}
        if needed.issubset(df.columns):
            RH, Q = eu.compute_humidities(
                df["temperature_2m"].values,
                df["dewpoint_temperature_2m"].values,
                df["surface_pressure"].values,
            )
            df["relative_humidity"] = RH
            df["specific_humidity"] = Q

        # --- rename to canonical ELM names based on RAW_TO_ELM ---
        want_canon = set(elm_required_vars(dformat))  # includes LONGXY/LATIXY/time
        # keep only mappings that land in required canonical vars
        rename_map = {
            src: canon for src, canon in RAW_TO_ELM.items() if canon in want_canon
        }
        df = df.rename(columns=rename_map)

        # coords/time to canonical names
        df = df.rename(columns={"date": "time", "lon": "LONGXY", "lat": "LATIXY"})

        # --- enforce nonnegativity for canonical variables (post-rename) ---
        for col in list(df.columns):
            if col in df.columns and is_nonnegative(col):
                df[col] = df[col].clip(lower=0)

        # --- final selection/order ---
        # Remove coords/meta from the "required data vars" list for column ordering
        coord_meta = {"LONGXY", "LATIXY", "time", "gid", "zone"}
        required_data_vars = [
            v for v in elm_required_vars(dformat) if v not in coord_meta
        ]
        final_cols = required_data_vars + ["LONGXY", "LATIXY", "time", "gid", "zone"]

        # Keep only those that exist (some formats/inputs may not provide all)
        final_cols = [c for c in final_cols if c in df.columns]

        df = df[final_cols]
        return df.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

    def temporal_options(self, df, *, start_year, end_year, calendar):
        """Describe how GEE's ERA5-Land hourly fields are time-labeled."""
        options = {
            "interval_end_vars": self.INTERVAL_END_VARS,
            "source_interval_hrs": self.SOURCE_INTERVAL_HOURS,
        }

        times = pd.to_datetime(df["time"], errors="coerce").dropna()
        if not times.empty and times.min() == self.GEE_COLLECTION_START:
            options["target_start"] = self.GEE_COLLECTION_START - pd.Timedelta(hours=1)
        return options

    def temporal_metadata(self, options=None):
        """NetCDF provenance for ERA5-Land interval alignment."""
        options = options or {}
        attrs = {
            "source_time_convention": (
                "ERA5-Land hourly accumulation fields are labeled at interval end"
            ),
            "forcing_time_convention": (
                "FSDS, FLDS, PRECTmms represent [DTIME, DTIME + timestep)"
            ),
            "interval_start_variables": ", ".join(self.INTERVAL_END_VARS),
            "source_interval_hours": self.SOURCE_INTERVAL_HOURS,
        }
        if options.get("target_start") == self.GEE_COLLECTION_START - pd.Timedelta(
            hours=1
        ):
            attrs["initial_state_fill"] = (
                "1950-01-01 00:00 instantaneous states filled from the earliest "
                "available values because the GEE collection starts at 01:00"
            )
        return attrs

    def required_vars(self, dformat):
        """Return the canonical ELM variables required for the requested output format."""

        return elm_required_vars(dformat)

    # ---------------- packing ----------------

    def pack_params(self, elm_var, data=None):
        """Return (add_offset, scale_factor) used to pack a variable for NetCDF output."""

        ao, sf = eu.elm_var_packing_params(
            elm_var, data=(data if data is not None else [])
        )
        return float(ao), float(sf)

    # ---------------- internal: ERA5 unit conversions ----------------

    def _unit_conversions(self, df):
        """
        ERA5-Land hourly → ELM unit alignment.
        """
        out = df.copy()

        # Wind speed from u,v
        if (
            "u_component_of_wind_10m" in out.columns
            and "v_component_of_wind_10m" in out.columns
        ):
            u = out["u_component_of_wind_10m"].values
            v = out["v_component_of_wind_10m"].values
            out["wind_speed"] = np.sqrt(u**2 + v**2)

        # Precip: meters/hour → mm/s
        if "total_precipitation_hourly" in out.columns:
            out["total_precipitation_hourly"] = (
                out["total_precipitation_hourly"].values / 3.6
            )

        # SW/LW: J/hr/m2 → W/m2
        if "surface_solar_radiation_downwards_hourly" in out.columns:
            out["surface_solar_radiation_downwards_hourly"] = (
                out["surface_solar_radiation_downwards_hourly"].values / 3600.0
            )
        if "surface_thermal_radiation_downwards_hourly" in out.columns:
            out["surface_thermal_radiation_downwards_hourly"] = (
                out["surface_thermal_radiation_downwards_hourly"].values / 3600.0
            )

        return out
