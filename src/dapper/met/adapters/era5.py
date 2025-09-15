# dapper/met/adapters/era5.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseAdapter                      # << avoid circulars
from dapper.met import met_io as io
from dapper.utils import elm_utils as eu


class ERA5Adapter(BaseAdapter):
    """
    Adapter for ERA5-Land hourly shards exported from GEE.
    Assumes CSVs and df_loc both contain a 'gid' column.
    """

    # ---------- discovery & locations ----------

    def discover_files(self, csv_directory, calendar):
        csv_directory = Path(csv_directory)
        csv_files = [
            str(csv_directory / f)
            for f in os.listdir(csv_directory)
            if os.path.splitext(f)[1].lower() == ".csv"
        ]
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {csv_directory}")
        start_year, end_year = io.get_start_end_years(csv_files, calendar=calendar)
        return csv_files, start_year, end_year

    def normalize_locations(self, df_loc, nzones):
        # We assume 'gid' is present; validate and normalize.
        if "gid" not in df_loc.columns:
            raise KeyError("df_loc must contain 'gid'.")

        if not {"lat", "lon"}.issubset(df_loc.columns):
            missing = {"lat", "lon"} - set(df_loc.columns)
            raise KeyError(f"df_loc missing required columns: {sorted(missing)}")

        out = df_loc.copy()
        if out["gid"].isna().any():
            bad = int(out["gid"].isna().sum())
            raise ValueError(f"df_loc has {bad} null gid values.")

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

    # ---------- preprocessing & requirements ----------

    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        df = df_merged.copy()

        # ---- time windowing ----
        if "date" not in df.columns:
            raise KeyError("Expected 'date' column in the CSV shard.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
        if str(calendar).lower() == "noleap":
            df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

        # ---- ERA5 → ELM unit conversions (your new helper) ----
        df = self._unit_conversions(df)

        # ---- derived humidities (if inputs present) ----
        if all(c in df.columns for c in ["temperature_2m", "dewpoint_temperature_2m", "surface_pressure"]):
            rh, q = eu.compute_humidities(
                df["temperature_2m"].values,
                df["dewpoint_temperature_2m"].values,
                df["surface_pressure"].values,
            )
            df["relative_humidity"] = rh
            df["specific_humidity"] = q

        # ---- enforce non-negativity on known flux/rate vars (where present) ----
        for col in eu.elm_data_dicts()["nonneg"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # ---- rename to ELM short names & select final columns ----
        mdd = eu.elm_data_dicts()
        if dformat == "BYPASS":
            want = [v for v in mdd["elm_req_vars"]["cbypass"] if v not in ["LONGXY", "LATIXY", "time"]]
        elif dformat == "DATM_MODE":
            want = [v for v in mdd["elm_req_vars"]["datm"] if v not in ["LONGXY", "LATIXY", "time"]]
        else:
            raise KeyError(f"Unsupported dformat: {dformat}")

        renamer = {src: dst for src, dst in mdd["short_names"].items() if dst in want}
        renamer.update({"date": "time", "lon": "LONGXY", "lat": "LATIXY"})
        df = df.rename(columns=renamer)

        # Optionally drop helper-only columns (e.g., 'wind_direction') if they’re not used later
        drop_maybe = [c for c in ["wind_direction"] if c in df.columns and c not in renamer.values()]
        if drop_maybe:
            df = df.drop(columns=drop_maybe)

        final_cols = list(want) + ["LONGXY", "LATIXY", "time", "gid", "zone"]
        df = df[final_cols].sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)
        return df

    def required_vars(self, dformat):
        mdd = eu.elm_data_dicts()
        if dformat == "BYPASS":
            return list(mdd["elm_req_vars"]["cbypass"])
        elif dformat == "DATM_MODE":
            return list(mdd["elm_req_vars"]["datm"])
        else:
            raise KeyError("Unsupported dformat. Only DATM_MODE and BYPASS are available.")

    # ---------- packing ----------
    def pack_params(self, elm_var, data=None):
        return eu.elm_var_packing_params(elm_var, data=data)

    
    # ---------- internal helpers ----------
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

            # wind_direction kept if you use it elsewhere; ELM uses magnitude (WIND)
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
