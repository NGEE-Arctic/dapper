# dapper/met/adapters/era5.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from dapper.met.adapters.base import BaseAdapter
from dapper.schemas.elm import elm_required_vars, is_nonnegative
from dapper.config.metsources.era5 import RAW_TO_ELM
from dapper.met import met_io as io
from dapper.utils import elm_utils as eu  # for compute_humidities, packing defaults


class ERA5Adapter(BaseAdapter):
    """
    ERA5-Land adapter that satisfies the BaseAdapter interface.
    Source-specific logic (unit conversions, renaming) is here.
    Canonical ELM requirements (vars/units/ranges) live in dapper.schemas.elm.
    """

    # ---------------- discovery & locations ----------------

    def discover_files(self, csv_directory, calendar):
        csv_directory = Path(csv_directory)
        csv_files = [
            str(csv_directory / f)
            for f in csv_directory.iterdir()
            if f.suffix.lower() == ".csv"
        ]
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {csv_directory}")
        start_year, end_year = io.get_start_end_years(csv_files, calendar=calendar)
        return csv_files, start_year, end_year

    def normalize_locations(self, df_loc, id_col, nzones):
        # Expect 'gid','lat','lon' already present per your latest assumption.
        required = {"gid", "lat", "lon"}
        missing = required - set(df_loc.columns)
        if missing:
            raise KeyError(f"df_loc missing required columns: {sorted(missing)}")

        out = df_loc.copy()
        if out["gid"].isna().any():
            bad = int(out["gid"].isna().sum())
            raise ValueError(f"df_loc has {bad} null gid values. Populate gid before calling.")

        out["gid"] = out["gid"].astype(str).str.strip()
        out["lon_0-360"] = np.mod(out["lon"].to_numpy(), 360.0)

        if "zone" not in out.columns:
            if nzones < 1:
                raise ValueError("nzones must be >= 1")
            out["zone"] = np.tile(
                np.arange(1, nzones + 1, dtype=int),
                (len(out) // nzones) + 1
            )[: len(out)]

        out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
        return out

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
