# dapper/met/adapters/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseAdapter(ABC):
    """
    Adapter contract for met sources. The Exporter depends only on this interface.
    Implementations may override defaults as needed.
    """

    # ---------- discovery ----------
    @abstractmethod
    def discover_files(self, csv_directory, calendar: str):
        """Return (csv_files, start_year, end_year)."""

    # ---------- locations (default provided) ----------
    def normalize_locations(self, df_loc: pd.DataFrame, id_col=None) -> pd.DataFrame:
        """
        Standardize df_loc to include ['gid','lat','lon','lon_0-360','zone'], sorted by (lat, lon).

        Zones are treated as a *per-location grouping label* (e.g., to support E3SM/ELM
        decomposition hints across a cellset). If df_loc lacks a 'zone' column, we default
        every location to zone=1.

        We intentionally do **not** auto-expand locations across multiple zones. If you want
        multiple zones, supply an explicit 'zone' column in df_loc with one row per location.
        """
        required = {"gid", "lat", "lon"}
        if not required.issubset(df_loc.columns):
            missing = required - set(df_loc.columns)
            raise KeyError(f"df_loc missing required columns: {sorted(missing)}")

        out = df_loc.copy()
        if out["gid"].isna().any():
            raise ValueError("df_loc contains null gid values.")

        out["gid"] = out["gid"].astype(str).str.strip()
        out["lon_0-360"] = np.mod(out["lon"].to_numpy(dtype=float), 360.0)

        if "zone" not in out.columns:
            out["zone"] = 1
        else:
            # Fill null zones with 1, enforce integer zone labels
            out["zone"] = out["zone"].fillna(1).astype(int)

        if (out["zone"] < 1).any():
            raise ValueError("df_loc contains zone values < 1. Zones must be positive integers.")

        return out.sort_values(["lat", "lon"]).reset_index(drop=True)

    # ---------- preprocessing (must be implemented) ----------
    @abstractmethod
    def preprocess_shard(
        self,
        df_merged: pd.DataFrame,
        start_year: int,
        end_year: int,
        calendar: str,
        dformat: str
    ) -> pd.DataFrame:
        """
        Return a DataFrame with at least:
        ['gid','time','LATIXY','LONGXY','zone', <ELM vars>]
        """

    # ---------- optional hint ----------
    def required_vars(self, dformat: str):
        """
        Optional: return a list of ELM var short names that this adapter will produce
        for the given dformat ('BYPASS' or 'DATM_MODE'). Exporter doesn’t require it.
        """
        return None

    # ---------- packing (default provided) ----------
    def pack_params(self, elm_var: str, data=None):
        """
        Default: use elm_utils.elm_var_packing_params if available; otherwise a safe fallback.
        Adapters can override for source-specific packing strategies.
        """
        try:
            from dapper.elm import utils as eu
            ao, sf = eu.elm_var_packing_params(elm_var, data=data if data is not None else [])
            return float(ao), float(sf)
        except Exception:
            # Very safe fallback: map around min(data) (or 0) with unit scale.
            arr = np.asarray(data) if data is not None else np.array([0.0])
            m = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
            return m, 1.0
