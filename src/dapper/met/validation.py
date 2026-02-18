# dapper/met/validation.py 
"""dapper module: met.validation."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, List

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

# matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------- plotting defaults & units -----------------------

DEFAULT_ELM_VARS = ["TBOT", "RH", "QBOT", "WIND", "FSDS", "FLDS", "PSRF", "PRECTmms"]
UNITS_ELM: Dict[str, str] = {
    "TBOT": "K", "DTBOT": "K", "RH": "%", "QBOT": "kg/kg", "PSRF": "Pa",
    "WIND": "m s⁻¹", "UWIND": "m s⁻¹", "VWIND": "m s⁻¹",
    "FSDS": "W m⁻²", "FLDS": "W m⁻²", "PRECTmms": "mm s⁻¹",
}

DEFAULT_RAW_VARS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
    "total_precipitation_hourly",
]
UNITS_RAW: Dict[str, str] = {
    "temperature_2m": "K",
    "dewpoint_temperature_2m": "K",
    "surface_pressure": "Pa",
    "u_component_of_wind_10m": "m s⁻¹",
    "v_component_of_wind_10m": "m s⁻¹",
    "surface_solar_radiation_downwards_hourly": "J m⁻²",
    "surface_thermal_radiation_downwards_hourly": "J m⁻²",
    "total_precipitation_hourly": "m (water eq.)",
}

def _t_from_dtime_var(vtime):
    """
    Convert numeric DTIME (+ units) to python datetimes for plotting.

    NOTE: We *must* respect the DTIME calendar attribute. In particular, when
    calendar == "noleap" (365_day), adding real (Gregorian) timedeltas to a
    pandas Timestamp will drift by one day after Feb 28 in leap years.
    """
    from datetime import datetime

    vals = np.asarray(vtime[:], dtype=float)
    units = getattr(vtime, "units", None)
    cal = getattr(vtime, "calendar", "standard")

    # Prefer CF-aware conversion (handles 'noleap' correctly).
    if units:
        try:
            dts = num2date(vals, units=units, calendar=cal, only_use_cftime_datetimes=False)
            dts = np.atleast_1d(dts)
            out = []
            for d in dts:
                if isinstance(d, datetime):
                    out.append(d)
                else:
                    # cftime -> python datetime (safe for noleap: it never produces Feb 29)
                    sec_raw = float(getattr(d, "second", 0.0))
                    sec = int(sec_raw)
                    if hasattr(d, "microsecond"):
                        usec = int(getattr(d, "microsecond", 0))
                    else:
                        usec = int(round((sec_raw - sec) * 1.0e6))
                    out.append(
                        datetime(
                            int(d.year), int(d.month), int(d.day),
                            int(getattr(d, "hour", 0)), int(getattr(d, "minute", 0)),
                            sec, usec,
                        )
                    )
            return out
        except Exception:
            pass

    # Fallback: assume days/hours since origin in units string
    units_l = (units or "").lower()
    origin = "1970-01-01 00:00:00"
    if "since" in units_l:
        origin = units_l.split("since", 1)[1].strip()
    base = pd.to_datetime(origin)
    if "day" in units_l:
        t = base + pd.to_timedelta(vals, unit="D")
    else:
        t = base + pd.to_timedelta(vals, unit="h")
    return t.to_pydatetime()

# ----------------------- public entrypoint -----------------------

def make_quicklooks(
    exporter=None,
    *,
    write_directory: Optional[Path | str] = None,
    mode: Optional[str] = None,
    vars: Optional[Iterable[str]] = None,
    gids: Optional[Iterable[str]] = None,
    out_dir: Optional[Path | str] = None,
    max_vars: int = 9,
) -> None:
    """
    Create per-site PNG quicklooks *after* an export has finished.

    Supports all modes:
      - NetCDF:  "cellset", "sites"
      - Raw:     "raw-site-parquet", "raw-site-csv"

    Parameters
    ----------
    exporter : Exporter or None
        Optionally pass the Exporter instance you used for `run(...)`.
        REQUIRED for 'cellset' (to map gids to lat/lon via the normalized
        domain geometry, i.e. ``exporter.domain_norm`` or ``exporter.df_loc_norm``).
    write_directory : path-like or None
        Where the export outputs live. If omitted and `exporter` is given,
        uses `exporter.write_directory`.
    mode : {"cellset","sites","raw-site-parquet","raw-site-csv"} or None
        Export mode. If None, auto-detected by looking under write_directory.
    vars : list[str] or None
        Variables to plot. For NetCDF modes use ELM short names;
        for raw modes use raw column names. If None, sensible defaults are used;
        if those aren’t present, first few numeric columns are chosen.
    gids : list[str] or None
        Subset of GIDs to plot. If None, plot all available.
    out_dir : path-like or None
        Destination for PNGs. Defaults to <write_directory>/quicklooks.
    max_vars : int
        When `vars` is None and no defaults match, cap the number of auto-picked
        numeric columns to avoid huge figures.

    Notes
    -----
    - NetCDF modes require `netCDF4` installed.
    - For 'cellset', pass the same `exporter` you ran with so we can use its
      `df_loc_norm` to locate each `gid` on the lat/lon axes.
    """
    if write_directory is None:
        if exporter is None:
            raise ValueError("Provide either `exporter` or `write_directory`.")

        # Backwards/forwards compatible resolution of the export output root.
        # Old Exporter used `write_directory`; refactor uses `group_dir`.
        if getattr(exporter, "write_directory", None) is not None:
            write_directory = exporter.write_directory
        elif getattr(exporter, "group_dir", None) is not None:
            write_directory = exporter.group_dir
        elif getattr(exporter, "out_dir", None) is not None:
            write_directory = exporter.out_dir
        elif getattr(getattr(exporter, "domain", None), "run_dir", None) is not None:
            write_directory = exporter.domain.run_dir
        else:
            raise AttributeError(
                "Could not infer export output directory from exporter. "
                "Pass write_directory=... explicitly."
            )
    wd = Path(write_directory)

    out_dir = Path(out_dir) if out_dir is not None else (wd / "quicklooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) detect mode if needed
    mode_eff = _detect_mode(wd, explicit=mode)

    if mode_eff in {"raw-site-parquet", "raw-site-csv"}:
        _quicklooks_raw(
            data_dir=wd / ("sites_parquet" if mode_eff == "raw-site-parquet" else "sites_csv"),
            is_parquet=(mode_eff == "raw-site-parquet"),
            out_dir=out_dir,
            vars=list(vars) if vars else None,
            gids=list(map(str, gids)) if gids else None,
            max_vars=max_vars,
        )
        return

    # from here: NetCDF modes
    if Dataset is None or num2date is None:
        raise RuntimeError("netCDF4 is required to plot NetCDF quicklooks.")

    if mode_eff == "sites":
        _quicklooks_elm_sites(
            wd=wd,
            out_dir=out_dir,
            vars=list(vars) if vars else DEFAULT_ELM_VARS,
            gids=list(map(str, gids)) if gids else None,
        )
        return

    if mode_eff == "cellset":
        if exporter is None:
            raise ValueError(
                "For 'cellset', pass the Exporter used for the run "
                "(needs domain geometry to map gids to lat/lon)."
            )

        # Prefer the normalized location table produced by the Exporter run.
        # (This includes lon_0-360, zones, etc.)
        if getattr(exporter, "df_loc_norm", None) is not None:
            df_loc_norm = exporter.df_loc_norm
        elif getattr(exporter, "domain", None) is not None and getattr(exporter, "adapter", None) is not None:
            # Best-effort fallback: reconstruct what Exporter.run(...) would have created.
            df_loc_norm = exporter.adapter.normalize_locations(exporter.domain.to_df_loc(), id_col=None)
        else:
            raise ValueError(
                "Exporter has no usable location table (expected 'df_loc_norm'). "
                "Did you run Exporter.run(...) first?"
            )

        _quicklooks_elm_combined(
            wd=wd,
            out_dir=out_dir,
            vars=list(vars) if vars else DEFAULT_ELM_VARS,
            df_loc_norm=df_loc_norm,
            gids=list(map(str, gids)) if gids else None,
        )
        return

    raise RuntimeError(f"make_quicklooks: unsupported or unknown mode '{mode_eff}'.")


# ----------------------- mode detection -----------------------

def _detect_mode(wd: Path, *, explicit: Optional[str] = None) -> str:
    if explicit in {"cellset","sites","raw-site-parquet","raw-site-csv"}:
        return explicit

    sp = wd / "sites_parquet"
    sc = wd / "sites_csv"
    if sp.exists() and any(sp.glob("*.parquet")):
        return "raw-site-parquet"
    if sc.exists() and any(sc.glob("*.csv")):
        return "raw-site-csv"

    nc_files = list(wd.rglob("*.nc"))
    if not nc_files:
        raise RuntimeError("No outputs found to plot under write_directory.")
    try:
        from netCDF4 import Dataset as _DS
        for p in nc_files:
            try:
                with _DS(p, "r") as ds:
                    m = getattr(ds, "export_mode", None)
                    if m in {"cellset","sites"}:
                        return m
                    # infer from dims of a data var
                    vname = _first_data_var_name(ds)
                    if not vname:
                        continue
                    dims = ds.variables[vname].dimensions
                    if dims == ("n","DTIME") or dims == ("DTIME","n"):
                        return "sites"
                    if dims == ("DTIME","lat","lon"):
                        return "cellset"
            except Exception:
                continue
    except Exception:
        pass
    raise RuntimeError("Could not determine export mode from outputs.")


def _first_data_var_name(ds) -> Optional[str]:
    cand = [n for n, v in ds.variables.items()
            if n not in ("DTIME","LATIXY","LONGXY","lat","lon")]
    cand = [n for n in cand if "DTIME" in ds.variables[n].dimensions] or cand
    return cand[0] if cand else None


# ----------------------- raw modes -----------------------

def _quicklooks_raw(
    *,
    data_dir: Path,
    is_parquet: bool,
    out_dir: Path,
    vars: Optional[List[str]],
    gids: Optional[List[str]],
    max_vars: int,
) -> None:
    ext = "*.parquet" if is_parquet else "*.csv"
    files = sorted(data_dir.glob(ext))
    if gids:
        sel = set(gids)
        files = [f for f in files if f.stem in sel]
    if not files:
        print(f"quicklooks: no files in {data_dir} matching selection.")
        return

    for f in files:
        gid = f.stem
        try:
            df = pd.read_parquet(f) if is_parquet else pd.read_csv(f)
        except Exception as e:
            print(f"[warn] {gid}: read failed: {e}")
            continue

        if "date" not in df.columns:
            print(f"[warn] {gid}: missing 'date' column; skipping.")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # choose vars
        meta = {"gid","date","lat","lon","zone","lon_0-360","LONGXY","LATIXY","time"}

        if vars:
            present = [c for c in vars if c in df.columns and c not in meta]
        else:
            preferred = [c for c in DEFAULT_RAW_VARS if c in df.columns]
            def _is_plottable(col: str) -> bool:
                if col in meta:
                    return False
                s = pd.to_numeric(df[col], errors="coerce")
                return np.isfinite(s).sum() > 0
            extras = sorted([c for c in df.columns if c not in preferred and _is_plottable(c)])
            present = preferred + extras

        if not present:
            print(f"[warn] {gid}: no plottable columns; skipping.")
            continue

        if max_vars is not None:
            present = present[:max_vars]
        n = len(present); ncols = 3; nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5.0, nrows*2.6), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        t = df["date"].to_numpy()
        any_all_nan = False
        for i, v in enumerate(present):
            arr = pd.to_numeric(df[v], errors="coerce").to_numpy(dtype="float64")
            ax = axes[i]
            ax.plot(t, arr, lw=0.8)
            ax.set_title(v, fontsize=10)
            ax.set_ylabel(UNITS_RAW.get(v, ""), fontsize=9)
            ax.grid(True, alpha=0.2)
            if np.all(~np.isfinite(arr)):
                any_all_nan = True

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{gid}", fontsize=12)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(out_dir / f"{gid}.png", dpi=150)
        plt.close(fig)

        if any_all_nan:
            print(f"[warn] {gid}: one or more series are entirely NaN in raw file.")

    print(f"quicklooks written to {out_dir}")


# ----------------------- NetCDF: sites -----------------------

def _quicklooks_elm_sites(
    *,
    wd: Path,
    out_dir: Path,
    vars: List[str],
    gids: Optional[List[str]],
) -> None:
    from netCDF4 import Dataset as _DS, num2date as _n2d

    subdirs = [d for d in wd.iterdir() if d.is_dir()]
    if gids:
        sel = set(gids)
        subdirs = [d for d in subdirs if d.name in sel]

    for sd in subdirs:
        gid = sd.name
        met_dir = sd / "MET"
        search_dir = met_dir if met_dir.exists() else sd
        files = list(search_dir.glob("*.nc"))
        if not files:
            continue

        var_to_path = {}
        for p in files:
            with _DS(p, "r") as ds:
                vname = _first_data_var_name(ds)
                if vname:
                    var_to_path[vname] = p

        present = [v for v in vars if v in var_to_path]
        if not present:
            continue

        with _DS(var_to_path[present[0]], "r") as ds0:
            vt = ds0.variables["DTIME"]
            t = _t_from_dtime_var(vt)

        n = len(present); ncols = 3; nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5.0, nrows*2.6), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        any_all_nan = False
        for i, v in enumerate(present):
            with _DS(var_to_path[v], "r") as ds:
                arr = np.asarray(ds.variables[v][:]).squeeze()
                if hasattr(arr, "mask"):
                    arr = np.ma.filled(arr, np.nan)
            ax = axes[i]
            ax.plot(t, arr, lw=0.8)
            ax.set_title(v, fontsize=10)
            ax.set_ylabel(UNITS_ELM.get(v, ""), fontsize=9)
            ax.grid(True, alpha=0.2)
            if np.all(~np.isfinite(arr)):
                any_all_nan = True

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{gid}", fontsize=12)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(out_dir / f"{gid}.png", dpi=150)
        plt.close(fig)

        if any_all_nan:
            print(f"[warn] {gid}: one or more variables are entirely NaN in NetCDF.")
    print(f"quicklooks written to {out_dir}")


# ----------------------- NetCDF: cellset (lat/lon) -----------------------

def _quicklooks_elm_combined(
    *,
    wd: Path,
    out_dir: Path,
    vars: List[str],
    df_loc_norm: pd.DataFrame,
    gids: Optional[List[str]],
) -> None:
    from netCDF4 import Dataset as _DS

    # --- find NetCDFs (support new layout: <run_dir>/MET/*.nc) ---
    met_dir = wd / "MET"
    search_dir = met_dir if met_dir.exists() else wd

    nc_files = list(search_dir.glob("*.nc"))
    var_to_path = {}
    axis_src = None
    for p in nc_files:
        try:
            with _DS(p, "r") as ds:
                vname = _first_data_var_name(ds)
                if vname:
                    var_to_path[vname] = p
                    axis_src = axis_src or p
        except Exception:
            pass

    present = [v for v in vars if v in var_to_path]
    if not present or axis_src is None:
        print("quicklooks: no plottable vars found in cellset outputs.")
        return

    with _DS(axis_src, "r") as ds0:
        vt = ds0.variables["DTIME"]
        t = _t_from_dtime_var(vt)
        lats = np.asarray(ds0.variables["lat"][:], dtype=float)
        lons = np.asarray(ds0.variables["lon"][:], dtype=float)

    # Tolerances: enough to bridge float32/float64 representation noise,
    # but still flag genuinely wrong coordinates.
    def _step_tol(vals: np.ndarray, fallback: float) -> float:
        u = np.unique(np.sort(vals))
        if u.size < 2:
            return fallback
        step = np.min(np.diff(u))
        return max(fallback, float(step) / 50.0)  # pretty tight

    tol_lat = _step_tol(lats, fallback=1e-4)
    tol_lon = _step_tol(lons, fallback=1e-4)

    def _nearest_index(vals: np.ndarray, target: float) -> tuple[int, float, float]:
        dif = np.abs(vals - target)
        idx = int(dif.argmin())
        return idx, float(vals[idx]), float(dif[idx])

    sel_gids = gids if gids else df_loc_norm["gid"].astype(str).tolist()

    for gid in sel_gids:
        row = df_loc_norm[df_loc_norm["gid"].astype(str) == str(gid)]
        if row.empty:
            print(f"[warn] gid not in df_loc_norm: {gid}")
            continue

        plat = float(row["lat"].iloc[0])

        # Prefer lon_0-360 if present, else fall back to lon
        if "lon_0-360" in row.columns:
            plon = float(row["lon_0-360"].iloc[0])
        else:
            plon = float(row["lon"].iloc[0])

        iy, lat_used, dlat = _nearest_index(lats, plat)

        # lon wrap safety: try lon, lon+360, lon-360; pick closest
        lon_cands = [plon, plon + 360.0, plon - 360.0]
        best = None
        for cand in lon_cands:
            ix, lon_used, dlon = _nearest_index(lons, cand)
            if best is None or dlon < best[2]:
                best = (ix, lon_used, dlon, cand)
        ix, lon_used, dlon, lon_cand_used = best

        # Warn only if we're *meaningfully* off-axis
        if dlat > tol_lat or dlon > tol_lon:
            print(
                f"[warn] {gid}: requested ({plat:.6f},{plon:.6f}) "
                f"nearest axis ({lat_used:.6f},{lon_used:.6f}) "
                f"Δ=({dlat:.3g},{dlon:.3g})"
            )

        n = len(present); ncols = 3; nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5.0, nrows*2.6), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        any_all_nan = False
        for i, v in enumerate(present):
            with _DS(var_to_path[v], "r") as ds:
                arr = np.asarray(ds.variables[v][:, iy, ix])
                if hasattr(arr, "mask"):
                    arr = np.ma.filled(arr, np.nan)

            ax = axes[i]
            ax.plot(t, arr, lw=0.8)
            ax.set_title(v, fontsize=10)
            ax.set_ylabel(UNITS_ELM.get(v, ""), fontsize=9)
            ax.grid(True, alpha=0.2)
            if np.all(~np.isfinite(arr)):
                any_all_nan = True

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{gid}  ({lat_used:.5f}, {lon_used:.5f})", fontsize=12)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(out_dir / f"{gid}.png", dpi=150)
        plt.close(fig)

        if any_all_nan:
            print(f"[warn] {gid}: one or more variables are entirely NaN in NetCDF.")

    print(f"quicklooks written to {out_dir}")
