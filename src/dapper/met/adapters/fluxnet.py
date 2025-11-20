from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import warnings

from dapper.met.adapters.base import BaseAdapter
from dapper.schemas.elm import elm_required_vars, is_nonnegative
from dapper.utils import elm_utils as eu


class FluxnetAdapter(BaseAdapter):
    """
    AmeriFlux FLUXNET (ONEFlux) → ELM adapter.

    Assumptions
    ----------
    - User provides a *single* FLUXNET CSV (FULLSET or SUBSET) per run.
    - CSV contains TIMESTAMP_START/TIMESTAMP_END (or TIMESTAMP) columns.
    - Missing values are coded as -9999.
    - Exporter supplies df_merged with ['gid','lat','lon','zone', ...] already
      merged in from df_loc.
    """
    # These are just for netCDF metadata
    SOURCE_NAME = "FLUXNET (AmeriFlux ONEFlux) tower data"
    DRIVER_TAG  = "FLUXNET"

    def __init__(self) -> None:
        # Native FLUXNET resolution (hours, e.g. 0.5, 1, 24, 168, …)
        self.native_dt_hours: Optional[float] = None

        # Resolution code inferred from filename: HH, HR, DD, WW, MM, YY
        self.resolution: Optional[str] = None

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def discover_files(self, csv_directory, calendar: str):
        p = Path(csv_directory)

        if p.is_file() and p.suffix.lower() == ".csv":
            csv_files = [str(p)]
        else:
            if not p.is_dir():
                raise FileNotFoundError(
                    f"{csv_directory} is neither a CSV file nor a directory."
                )
            csv_files = [
                str(c)
                for c in p.iterdir()
                if c.is_file() and c.suffix.lower() == ".csv"
            ]

        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in {csv_directory}")
        if len(csv_files) > 1:
            raise ValueError(
                f"FluxnetAdapter expects a single CSV; found {len(csv_files)} in {csv_directory}"
            )

        csv_file = csv_files[0]

        # keep filename-based flag as metadata only
        self.resolution = self._infer_resolution_from_filename(csv_file)

        # --- read ONLY timestamp columns, but read the WHOLE column ---
        # (this is cheap even for a 0.5 GB file because it's a single column)
        df_ts = pd.read_csv(csv_file, usecols=lambda c: c.startswith("TIMESTAMP"))

        # infer native dt from a small sample of the timestamp table
        # (first few thousand rows are enough)
        self.native_dt_hours = infer_fluxnet_dt_hours(df_ts.iloc[:5000])

        # choose which timestamp column we'll use for year min/max
        ts_col = None
        for c in ("TIMESTAMP_END", "TIMESTAMP_START", "TIMESTAMP"):
            if c in df_ts.columns:
                ts_col = c
                break
        if ts_col is None:
            raise KeyError("No TIMESTAMP_* column found in FLUXNET CSV.")

        vals = df_ts[ts_col].astype(str)

        # parse with formats that match FLUXNET conventions
        if ts_col in ("TIMESTAMP_START", "TIMESTAMP_END"):
            # half-hourly / hourly / weekly: YYYYMMDDHHMM
            times = pd.to_datetime(vals, format="%Y%m%d%H%M", errors="coerce")
        else:
            # daily / monthly / yearly: TIMESTAMP is truncated (YYYYMMDD or shorter)
            times = pd.to_datetime(vals, format="%Y%m%d%H%M", errors="coerce")
            if times.isna().all():
                times = pd.to_datetime(vals, format="%Y%m%d", errors="coerce")
            if times.isna().all():
                times = pd.to_datetime(vals, errors="coerce")

        if times.isna().all():
            raise ValueError("Could not parse any timestamps from FLUXNET CSV.")

        start_year = int(times.dt.year.min())
        end_year = int(times.dt.year.max())

        return [csv_file], start_year, end_year

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------
    def preprocess_shard(
        self,
        df_merged: pd.DataFrame,
        start_year: int,
        end_year: int,
        calendar: str,
        dformat: str,
    ) -> pd.DataFrame:
        df = df_merged.copy()

        # Replace FLUXNET sentinel with NaN
        df = df.replace(-9999, np.nan)

        # Time handling
        time_col = None
        for c in ("TIMESTAMP_END", "TIMESTAMP_START", "TIMESTAMP"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            raise KeyError("Expected TIMESTAMP_* column in FLUXNET data.")

        vals = df[time_col].astype(str)

        if time_col in ("TIMESTAMP_START", "TIMESTAMP_END"):
            # Half-hourly / hourly / weekly: full YYYYMMDDHHMM
            date = pd.to_datetime(vals, format="%Y%m%d%H%M", errors="coerce")
        else:
            # Daily / monthly / yearly: truncated (YYYYMMDD, YYYYMM, etc.)
            date = pd.to_datetime(vals, format="%Y%m%d%H%M", errors="coerce")
            if date.isna().all():
                date = pd.to_datetime(vals, format="%Y%m%d", errors="coerce")
            if date.isna().all():
                date = pd.to_datetime(vals, errors="coerce")

        df["date"] = date
        if df["date"].isna().all():
            raise ValueError("All parsed timestamps are NaT; check TIMESTAMP_* formatting.")

        df = df.sort_values("date")

        # Year filtering / calendar
        df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
        if str(calendar).lower() == "noleap":
            df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

        # Ensure we have enough raw variables to build ELM vars
        self._check_required_raw_vars(df, dformat)

        # FLUXNET-specific unit conversions (now with smarter coalescing)
        df = self._unit_conversions(df)

        # --- specific humidity from TA/RH/PA, with some diagnostics ---
        if {"TA", "RH", "PA"}.issubset(df.columns):
            ta_na = int(df["TA"].isna().sum())
            rh_na = int(df["RH"].isna().sum())
            pa_na = int(df["PA"].isna().sum())
            if ta_na or rh_na or pa_na:
                print(
                    f"FluxnetAdapter: TA NaNs={ta_na}, RH NaNs={rh_na}, "
                    f"PA NaNs={pa_na} (these rows cannot get QBOT)."
                )

            temp_k = df["TA"].to_numpy(dtype="float64")
            temp_c = temp_k - 273.15
            q = eu.compute_specific_humidity_from_rh(
                temp_c,
                df["RH"].to_numpy(dtype="float64"),
                df["PA"].to_numpy(dtype="float64"),
            )
            df["QBOT"] = q

        # Map raw → canonical names
        want_canon = set(elm_required_vars(dformat))  # includes LONGXY/LATIXY/time
        rename_map = {
            src: canon
            for src, canon in {
                "TA": "TBOT",
                "SW_IN": "FSDS",
                "LW_IN": "FLDS",
                "PA": "PSRF",
                "WS": "WIND",
                "RH": "RH",
            }.items()
            if src in df.columns and canon in want_canon
        }
        df = df.rename(columns=rename_map)

        # Coordinates & time
        missing_coord = {"lat", "lon"} - set(df.columns)
        if missing_coord:
            raise KeyError(
                f"FLUXNET df_merged missing required coord columns: {sorted(missing_coord)}"
            )
        df = df.rename(columns={"date": "time", "lon": "LONGXY", "lat": "LATIXY"})

        # Clip nonnegative vars
        for col in list(df.columns):
            if is_nonnegative(col):
                df[col] = df[col].clip(lower=0)

        # Basic NaN diagnostics for required vars
        coord_meta = {"LONGXY", "LATIXY", "time", "gid", "zone"}
        required_data_vars = [v for v in elm_required_vars(dformat) if v not in coord_meta]
        nan_counts = {
            v: int(df[v].isna().sum())
            for v in required_data_vars
            if v in df.columns and df[v].isna().any()
        }
        if nan_counts:
            msg = ", ".join(f"{v}({n})" for v, n in nan_counts.items())
            print(f"FluxnetAdapter: variables with NaNs after conversion: {msg}")

        all_nan = [
            v for v in required_data_vars
            if v in df.columns and df[v].isna().all()
        ]
        if all_nan:
            raise ValueError(
                f"FluxnetAdapter: required ELM variables are all NaN after conversion: {all_nan}"
            )

        # Final selection / ordering
        final_cols = required_data_vars + ["LONGXY", "LATIXY", "time", "gid", "zone"]
        final_cols = [c for c in final_cols if c in df.columns]
        df = df[final_cols]
        return df.sort_values(["time", "LATIXY", "LONGXY"]).reset_index(drop=True)

    def required_vars(self, dformat: str):
        return elm_required_vars(dformat)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _infer_resolution_from_filename(self, path: str) -> Optional[str]:
        """
        Infer FLUXNET resolution flag (HH/HR/DD/WW/MM/YY) from filename:

            [NETWORK]_[SITE]_FLUXNET_[GROUPING]_[RESOLUTION]_...
        """
        name = Path(path).name
        parts = name.split("_")
        try:
            idx = parts.index("FLUXNET")
        except ValueError:
            return None

        if idx + 2 >= len(parts):
            return None
        resolution = parts[idx + 2]
        return resolution.upper()

    def _required_roots_for_dformat(self, dformat: str) -> List[str]:
        """
        FLUXNET variable *roots* needed to construct the ELM-required vars.
        """
        # These are roots; we then coalesce across *_F_MDS / *_F / *_ERA / raw.
        return ["TA", "P", "SW_IN", "LW_IN", "PA", "WS", "RH"]

    def _check_required_raw_vars(self, df: pd.DataFrame, dformat: str) -> None:
        """
        Make sure we have at least one candidate column for each required
        FLUXNET group. We don't check NaNs here; that's handled later.
        """
        root_to_candidates = {
            "TA": ["TA_F_MDS", "TA_F", "TA_ERA", "TA"],
            "P": ["P_F", "P", "P_ERA"],
            "SW_IN": ["SW_IN_F_MDS", "SW_IN_F", "SW_IN_ERA", "SW_IN"],
            "LW_IN": ["LW_IN_F_MDS", "LW_IN_F", "LW_IN_ERA", "LW_IN"],
            "PA": ["PA_F", "PA", "PA_ERA"],
            "WS": ["WS_F", "WS", "WS_ERA"],
            # We accept either direct RH or VPD for humidity
            "RH": ["RH", "VPD_F_MDS", "VPD_F", "VPD_ERA", "VPD"],
        }

        missing = []
        for root in self._required_roots_for_dformat(dformat):
            candidates = root_to_candidates.get(root, [])
            if not any(c in df.columns for c in candidates):
                missing.append(f"{root} (any of {candidates})")

        if missing:
            raise KeyError(
                "FluxnetAdapter cannot compute required ELM variables; "
                "missing FLUXNET base fields: " + ", ".join(missing)
            )

    def _unit_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FLUXNET → ELM units with prioritized coalescing:

        - TA: TA_F_MDS → TA_F → TA_ERA → TA  (deg C) → TA (K)
        - VPD: *_F_MDS → *_F → *_ERA → VPD (hPa)
        - RH: RH (%), later filled from VPD+TA where missing
        - PA: *_F → PA → *_ERA (kPa) → PA (Pa)
        - WS: *_F → WS → *_ERA (m/s)
        - SW_IN / LW_IN: *_F_MDS → *_F → *_ERA → raw (W/m²)
        - P: P_F → P → P_ERA (mm per step or per day/year) → PRECTmms (mm/s)
        """
        out = df.copy()

        # Track which raw columns actually contributed to each coalesced output
        #   coalesced_info[out_name] = {src_col: count_of_values}
        coalesced_info: dict[str, dict[str, int]] = {}

        def coalesce_priority(cols, out_name):
            """
            Take the first non-NaN among cols (in priority order) and store as out_name.

            Also record how many values came from each source; if more than one source
            actually contributed, we will emit a warning at the end of this function.
            """
            nonlocal coalesced_info

            cols = [c for c in cols if c in out.columns]
            if not cols:
                return

            # Stack and forward-fill across columns to get first non-NaN
            stacked = pd.concat([out[c] for c in cols], axis=1)
            stacked.columns = cols  # ensure col names match
            result = stacked.bfill(axis=1).iloc[:, 0]
            out[out_name] = result

            # Figure out which col was the first non-NaN at each row
            used_counts: dict[str, int] = {}
            for i, c in enumerate(cols):
                col_vals = stacked[c]
                if i == 0:
                    contrib_mask = col_vals.notna()
                else:
                    contrib_mask = col_vals.notna() & stacked[cols[:i]].isna().all(axis=1)
                if contrib_mask.any():
                    used_counts[c] = int(contrib_mask.sum())

            # Only log as "coalesced" if more than one source actually contributed
            if len(used_counts) > 1:
                coalesced_info[out_name] = used_counts

        # -------------------- coalescing raw met variables --------------------
        # Core meteorology
        coalesce_priority(["TA_F_MDS", "TA_F", "TA_ERA", "TA"], "TA_C")
        coalesce_priority(["VPD_F_MDS", "VPD_F", "VPD_ERA", "VPD"], "VPD_hPa")
        coalesce_priority(["RH"], "RH")
        coalesce_priority(["PA_F", "PA", "PA_ERA"], "PA_kPa")
        coalesce_priority(["WS_F", "WS", "WS_ERA"], "WS")
        coalesce_priority(["SW_IN_F_MDS", "SW_IN_F", "SW_IN_ERA", "SW_IN"], "SW_IN")
        coalesce_priority(["LW_IN_F_MDS", "LW_IN_F", "LW_IN_ERA", "LW_IN"], "LW_IN")
        coalesce_priority(["P_F", "P", "P_ERA"], "P_mm")

        # -------------------- unit conversions --------------------
        # Temperature: C → K
        if "TA_C" in out.columns:
            out["TA"] = out["TA_C"].astype(float) + 273.15

        # Pressure: kPa → Pa
        if "PA_kPa" in out.columns:
            out["PA"] = out["PA_kPa"].astype(float) * 1000.0

        # Cast the rest
        for col in ["RH", "VPD_hPa", "WS", "SW_IN", "LW_IN", "P_mm"]:
            if col in out.columns:
                out[col] = out[col].astype(float)

        # Precip → PRECTmms (mm/s)
        if "P_mm" in out.columns:
            p_vals = out["P_mm"].astype(float)
            res = (self.resolution or "").upper()
            if res in ("HH", "HR"):
                # mm over step → mm/s using step length
                if "TIMESTAMP_START" in out.columns and "TIMESTAMP_END" in out.columns:
                    t0 = pd.to_datetime(
                        out["TIMESTAMP_START"].astype(str).iloc[0],
                        format="%Y%m%d%H%M",
                        errors="coerce",
                    )
                    t1 = pd.to_datetime(
                        out["TIMESTAMP_END"].astype(str).iloc[0],
                        format="%Y%m%d%H%M",
                        errors="coerce",
                    )
                    dt_sec = (t1 - t0).total_seconds()
                    if not np.isfinite(dt_sec) or dt_sec <= 0:
                        dt_sec = 1800.0  # fallback for half-hourly
                else:
                    dt_sec = 1800.0
                out["PRECTmms"] = p_vals / dt_sec
            elif res in ("DD", "MM", "WW"):
                # mm d-1 → mm/s
                out["PRECTmms"] = p_vals / 86400.0
            elif res == "YY":
                # mm y-1 → mm/s (assume 365-day year)
                out["PRECTmms"] = p_vals / (365.0 * 86400.0)
            else:
                # Unknown resolution: assume half-hourly
                out["PRECTmms"] = p_vals / 1800.0

        # Fill RH from VPD if available
        if ("TA_C" in out.columns) and ("VPD_hPa" in out.columns):
            temp_c = out["TA_C"].to_numpy(dtype="float64")
            es = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))  # Pa
            vpd_pa = out["VPD_hPa"].to_numpy(dtype="float64") * 100.0  # hPa → Pa

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                rh_from_vpd = (1.0 - (vpd_pa / es)) * 100.0
            rh_from_vpd = np.clip(rh_from_vpd, 0.0, 100.0)

            if "RH" in out.columns:
                rh = out["RH"].to_numpy(dtype="float64")
                mask = ~np.isfinite(rh)
                rh[mask] = rh_from_vpd[mask]
                out["RH"] = rh
            else:
                out["RH"] = rh_from_vpd

        # -------------------- report coalescing to the user --------------------
        if coalesced_info:
            lines = []
            for out_name, counts in coalesced_info.items():
                parts = [f"{src}={n}" for src, n in counts.items()]
                lines.append(f"{out_name}: " + ", ".join(parts))
            msg = (
                "FluxnetAdapter coalesced the following variables "
                "(source_column=number_of_values_used):\n  " +
                "\n  ".join(lines)
            )
            warnings.warn(msg, UserWarning)
            # If you prefer stdout instead:
            # print(msg)

        return out


def infer_fluxnet_dt_hours(df: pd.DataFrame) -> float:
    """
    Infer native FLUXNET timestep from timestamp columns in hours.

    Handles:
      - half-hourly/hourly/weekly: TIMESTAMP_START, TIMESTAMP_END (YYYYMMDDHHMM)
      - daily/monthly/yearly: TIMESTAMP (YYYYMMDD or YYYYMM, etc.)

    Returns
    -------
    float
        Approximate timestep in hours.

    Raises
    ------
    ValueError
        If no suitable timestamp columns are found or dt cannot be inferred.
    """
    # Case 1: (half-)hourly or weekly: START/END pair
    if "TIMESTAMP_START" in df.columns and "TIMESTAMP_END" in df.columns:
        ts_start = pd.to_datetime(df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")
        ts_end   = pd.to_datetime(df["TIMESTAMP_END"].astype(str),   format="%Y%m%d%H%M")

        # Each record represents the interval [start, end); use average duration
        dt_seconds = (ts_end - ts_start).dt.total_seconds()
        # Use median to be robust to a few weird rows
        dt_sec = float(np.nanmedian(dt_seconds))
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("Could not infer timestep from TIMESTAMP_START/END.")
        return dt_sec / 3600.0

    # Case 2: daily / monthly / yearly: single TIMESTAMP
    if "TIMESTAMP" in df.columns:
        ts = pd.to_datetime(df["TIMESTAMP"].astype(str), format="%Y%m%d", errors="coerce")
        if ts.isna().all():
            # fall back to more generic parse if needed
            ts = pd.to_datetime(df["TIMESTAMP"].astype(str), errors="coerce")
        diffs = ts.diff().dt.total_seconds()
        dt_sec = float(np.nanmedian(diffs))
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("Could not infer timestep from TIMESTAMP.")
        return dt_sec / 3600.0

    raise ValueError("No FLUXNET timestamp columns found to infer timestep.")
