# dapper/met/validation.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, List

import numpy as np
import pandas as pd

# matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# netCDF is only needed for NetCDF modes
try:
    from netCDF4 import Dataset, num2date
except Exception:  # pragma: no cover
    Dataset = None
    num2date = None


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
    Convert numeric DTIME (+ units) to pandas Timestamps for plotting,
    avoiding cftime objects entirely.
    """
    import pandas as pd
    import numpy as np

    vals = np.asarray(vtime[:], dtype=float)
    units = getattr(vtime, "units", "").lower()
    origin = "1970-01-01 00:00:00"
    if "since" in units:
        origin = units.split("since", 1)[1].strip()
    base = pd.to_datetime(origin)

    if "day" in units:
        t = base + pd.to_timedelta(vals, unit="D")
    else:
        # default to hours if not clearly 'day'
        t = base + pd.to_timedelta(vals, unit="h")
    # return something Matplotlib is happy with
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
      - NetCDF:  "elm-combined", "elm-sites"
      - Raw:     "raw-site-parquet", "raw-site-csv"

    Parameters
    ----------
    exporter : Exporter or None
        Optionally pass the Exporter instance you used for `run(...)`.
        REQUIRED for 'elm-combined' (to map gids to lat/lon via the normalized
        domain geometry, i.e. ``exporter.domain_norm`` or ``exporter.df_loc_norm``).
    write_directory : path-like or None
        Where the export outputs live. If omitted and `exporter` is given,
        uses `exporter.write_directory`.
    mode : {"elm-combined","elm-sites","raw-site-parquet","raw-site-csv"} or None
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
    - For 'elm-combined', pass the same `exporter` you ran with so we can use its
      `df_loc_norm` to locate each `gid` on the lat/lon axes.
    """
    if write_directory is None:
        if exporter is None:
            raise ValueError("Provide either `exporter` or `write_directory`.")
        write_directory = exporter.write_directory
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

    if mode_eff == "elm-sites":
        _quicklooks_elm_sites(
            wd=wd,
            out_dir=out_dir,
            vars=list(vars) if vars else DEFAULT_ELM_VARS,
            gids=list(map(str, gids)) if gids else None,
        )
        return

    if mode_eff == "elm-combined":
        if exporter is None:
            raise ValueError(
                "For 'elm-combined', pass the Exporter used for the run "
                "(needs domain geometry to map gids to lat/lon)."
            )

        # Prefer the Domain-based geometry; fall back to legacy df_loc_norm
        if getattr(exporter, "domain_norm", None) is not None:
            df_loc_norm = exporter.domain_norm.gdf
        elif getattr(exporter, "df_loc_norm", None) is not None:
            df_loc_norm = exporter.df_loc_norm
        else:
            raise ValueError(
                "Exporter has neither 'domain_norm' nor 'df_loc_norm'. "
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
    if explicit in {"elm-combined","elm-sites","raw-site-parquet","raw-site-csv"}:
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
                    if m in {"elm-combined","elm-sites"}:
                        return m
                    # infer from dims of a data var
                    vname = _first_data_var_name(ds)
                    if not vname:
                        continue
                    dims = ds.variables[vname].dimensions
                    if dims == ("n","DTIME") or dims == ("DTIME","n"):
                        return "elm-sites"
                    if dims == ("DTIME","lat","lon"):
                        return "elm-combined"
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


# ----------------------- NetCDF: elm-sites -----------------------

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
        files = list(sd.glob("*.nc"))
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


# ----------------------- NetCDF: elm-combined (lat/lon) -----------------------

def _quicklooks_elm_combined(
    *,
    wd: Path,
    out_dir: Path,
    vars: List[str],
    df_loc_norm: pd.DataFrame,
    gids: Optional[List[str]],
) -> None:
    from netCDF4 import Dataset as _DS, num2date as _n2d

    nc_files = list(wd.glob("*.nc"))
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
        print("quicklooks: no plottable vars found in elm-combined outputs.")
        return

    with _DS(axis_src, "r") as ds0:
        vt = ds0.variables["DTIME"]
        t = _t_from_dtime_var(vt)
        lats = np.asarray(ds0.variables["lat"][:], dtype=float)
        lons = np.asarray(ds0.variables["lon"][:], dtype=float)
    lat_key = {round(float(v), 6): i for i, v in enumerate(lats)}
    lon_key = {round(float(v), 6): j for j, v in enumerate(lons)}

    sel_gids = gids if gids else df_loc_norm["gid"].astype(str).tolist()
    for gid in sel_gids:
        row = df_loc_norm[df_loc_norm["gid"].astype(str) == str(gid)]
        if row.empty:
            print(f"[warn] gid not in df_loc_norm: {gid}")
            continue
        plat = round(float(row["lat"].iloc[0]), 6)
        plon = round(float(row["lon_0-360"].iloc[0]), 6)
        iy = lat_key.get(plat, None); ix = lon_key.get(plon, None)
        if iy is None or ix is None:
            print(f"[warn] {gid}: cell ({plat},{plon}) not on output axes.")
            continue

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

        fig.suptitle(f"{gid}  ({plat}, {plon})", fontsize=12)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(out_dir / f"{gid}.png", dpi=150)
        plt.close(fig)

        if any_all_nan:
            print(f"[warn] {gid}: one or more variables are entirely NaN in NetCDF.")

    print(f"quicklooks written to {out_dir}")
