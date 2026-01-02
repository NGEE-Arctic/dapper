# dapper/met/adapters/era5.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from dapper.met import temporal as dt
from dapper.met.adapters.base import BaseAdapter
from dapper.schemas.elm import elm_required_vars, is_nonnegative
from dapper.config.metsources.era5 import RAW_TO_ELM
from dapper.elm import utils as eu  # for compute_humidities, packing defaults


class ERA5Adapter(BaseAdapter):
    """ERA5-Land → ELM adapter.

    This adapter implements the `BaseAdapter` interface for ERA5-Land hourly data.
    It handles source-specific details—file discovery, unit conversions, humidity
    diagnostics, renaming to ELM short names, and nonnegativity enforcement—so the
    upstream `Exporter` can remain source-agnostic.

    Responsibilities
    ----------------
    - **discover_files**: Find CSV shards in a directory and infer the overall
      (start_year, end_year) using their date coverage.
    - **normalize_locations**: Validate and normalize the locations table
      (adds ``lon_0-360``, ensures/creates ``zone``, stable sorting).
    - **id_column_for_csv**: Declare the identifier column name in the input
      CSVs. For ERA5 we require **``"gid"``**.
    - **preprocess_shard**: Convert one merged shard (CSV rows joined to
      locations) into canonical ELM columns. Steps include:
      1) time filtering and optional “noleap” removal of Feb 29,
      2) ERA5→ELM unit conversions (e.g., J/hr/m² → W/m², m/hr → mm/s),
      3) optional humidity computation (RH/Q) if temperature, dewpoint, and
         surface pressure are available,
      4) renaming raw ERA5 fields to ELM short names via a mapping,
      5) clipping canonical nonnegative variables,
      6) returning only required columns in a deterministic order.
    - **required_vars**: Report the canonical ELM variable names required for the
      requested output format (e.g., ``"BYPASS"`` vs ``"DATM_MODE"``).
    - **pack_params**: Provide robust (add_offset, scale_factor) for a canonical
      ELM variable, given optional data to tune ranges.

    Method Contracts
    ----------------
    discover_files(csv_directory, calendar) -> (csv_files, start_year, end_year)
        Search ``csv_directory`` for ``*.csv`` and compute the inclusive year
        bounds consistent with ``calendar`` (e.g., remove Feb 29 for ``"noleap"``).

    normalize_locations(df_loc) -> pandas.DataFrame
        Input must contain at least ``{"gid","lat","lon"}``.
        Returns a new DataFrame with:
        - ``gid`` coerced to stripped strings (no nulls),
        - ``lon_0-360`` added (modulo 360),
        - ``zone`` present (created if missing; default zone=1 if missing),
        - rows sorted by (lat, lon).

    id_column_for_csv(df_csv, id_col) -> str
        Returns the name of the identifier column expected in CSVs.
        For ERA5, this is always **``"gid"``** (raises if absent).

    preprocess_shard(df_merged, start_year, end_year, calendar, dformat) -> pandas.DataFrame
        Accepts a merged shard that already includes ERA5 raw fields plus
        ``["gid","lat","lon","zone","date"]``. Produces a DataFrame containing
        a subset of canonical ELM variables (short names) plus coordinates and
        metadata: ``["LONGXY","LATIXY","time","gid","zone"]``.

    required_vars(dformat) -> list[str]
        Returns the canonical ELM short names required for the given format
        (delegates to the ELM schema).

    pack_params(elm_var, data=None) -> tuple[float, float]
        Returns ``(add_offset, scale_factor)`` for integer packing. If
        ``data`` is provided, it may be used to compute robust ranges.

    Assumptions & Inputs
    --------------------
    - Input CSVs are time-sharded (e.g., yearly) and contain at least a ``date``
      column and **``gid``**.
    - The locations table (``df_loc``) contains at least ``gid``, ``lat``, ``lon``;
      ``zone`` is optional.
    - Temperature variables are in kelvin; pressures are in pascals; ERA5
      accumulated energy fluxes are in J/m² over the step; precipitation is in
      meters over the step—this adapter converts as needed to ELM units.

    Notes
    -----
    - Humidity computation (RH, specific humidity) is performed only when
      ``temperature_2m``, ``dewpoint_temperature_2m``, and ``surface_pressure`` are present.
    - Precipitation conversion uses ``m/hr → mm/s`` via division by ``3.6``.
    - Short-name mapping and “required var” lists come from your ELM schema/
      config (e.g., ``RAW_TO_ELM``, ``elm_required_vars``); make sure those are imported.
    - Nonnegativity checks rely on an ``is_nonnegative(name)`` helper; ensure it
      matches your canonical variable set.

    Potential Pitfalls
    ------------------
    - If CSVs or ``df_loc`` do not use **``"gid"``**, preprocessing will raise;
      adapt or pre-rename columns before calling the exporter.
    - The wind vector → speed diagnostic is included; ELM typically consumes the
      magnitude (``WIND``) after short-name remapping.
    """
    # These are just for netCDF metadata
    SOURCE_NAME = "ERA5-Land hourly reanalysis"
    DRIVER_TAG  = "ERA5"

    # ---------------- discovery & locations ----------------

    def discover_files(self, csv_directory, calendar):
        csv_directory = Path(csv_directory)

        # ignore directories; only pick real files that end with .csv (case-insensitive)
        csv_files = [
            str(p)
            for p in csv_directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ]

        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {csv_directory}")

        start_year, end_year = dt.get_start_end_years(csv_files, calendar=calendar)
        return csv_files, start_year, end_year

    def id_column_for_csv(self, df_csv, id_col):
        if "gid" not in df_csv.columns:
            raise KeyError("Expected 'gid' column in input CSV.")
        return "gid"

    # ---------------- preprocessing & requirements ----------------

    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        """
        1) Filter time & handle no-leap
        2) Apply ERA5 → ELM unit conversions
        3) Compute humidities (if columns available)
        4) Rename columns to canonical ELM names using RAW_TO_ELM
        5) Clip canonical nonnegative variables
        6) Return only the canonical vars required by elm_required_vars(dformat),
           plus LONGXY/LATIXY/time/gid/zone (coords/meta).
        """
        df = df_merged.copy()

        # --- time handling ---
        if "date" not in df.columns:
            raise KeyError("Expected 'date' column in the CSV shard.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
        if str(calendar).lower() == "noleap":
            df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

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
        rename_map = {src: canon for src, canon in RAW_TO_ELM.items() if canon in want_canon}
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
        required_data_vars = [v for v in elm_required_vars(dformat) if v not in coord_meta]
        final_cols = required_data_vars + ["LONGXY", "LATIXY", "time", "gid", "zone"]

        # Keep only those that exist (some formats/inputs may not provide all)
        final_cols = [c for c in final_cols if c in df.columns]

        df = df[final_cols]
        return df.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

    def required_vars(self, dformat):
        return elm_required_vars(dformat)

    # ---------------- packing ----------------

    def pack_params(self, elm_var, data=None):
        # Delegate to your existing robust packer (range→offset/scale)
        ao, sf = eu.elm_var_packing_params(elm_var, data=(data if data is not None else []))
        return float(ao), float(sf)

    # ---------------- internal: ERA5 unit conversions ----------------

    def _unit_conversions(self, df):
        """
        ERA5-Land hourly → ELM unit alignment.
        """
        out = df.copy()

        # Wind speed from u,v
        if "u_component_of_wind_10m" in out.columns and "v_component_of_wind_10m" in out.columns:
            u = out["u_component_of_wind_10m"].values
            v = out["v_component_of_wind_10m"].values
            out["wind_speed"] = np.sqrt(u**2 + v**2)

            # Optional diagnostic (not used by ELM)
            wd = np.degrees(np.arctan2(u, v))
            wd[wd >= 180] -= 180
            wd[wd < 180] += 180
            out["wind_direction"] = wd

        # Precip: meters/hour → mm/s
        if "total_precipitation_hourly" in out.columns:
            out["total_precipitation_hourly"] = out["total_precipitation_hourly"].values / 3.6

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
