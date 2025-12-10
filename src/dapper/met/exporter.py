# dapper/met/exporter.py
import warnings
import numpy as np
import pandas as pd
import datetime as _dt
import geopandas as gpd
from pathlib import Path
from fastparquet import write

from dapper.utils import utils
import dapper.met.temporal as dt 
from dapper.domains.domain import Domain
from dapper.met.writers import initialize_met_netcdf, append_met_netcdf

# Rounding precision for lat/lon axes and lookups.
# 1e-6 deg ~ 0.11 m at the equator, which is far below any grid we're using.
LATLON_DECIMALS = 6

class Exporter:
    """
    Source-agnostic meteorological exporter.

    This class orchestrates a two-pass pipeline that ingests time-sharded CSVs
    for many sites/cells, preprocesses them via a pluggable *adapter*, and
    writes ELM-ready NetCDF outputs in two layouts:

      1) ``"elm-combined"`` – one NetCDF per variable with dims
         ``('DTIME','lat','lon')`` (global packing; sparse lat/lon axes are OK).
      2) ``"elm-sites"`` – one directory per site; each directory contains
         one NetCDF per variable with dims ``('n','DTIME')`` where ``n=1``
         (per-site packing).

    Exporter is *source-agnostic*: all dataset-specific logic (file discovery,
    unit conversions, renaming to ELM short names, etc.) lives in an adapter
    that implements the `BaseAdapter` interface (e.g., an ``ERA5Adapter``).
    The exporter handles staging (CSV → per-site parquet), global DTIME axis
    creation, packing scans, chunking, and NetCDF I/O.

    Parameters
    ----------
    adapter : BaseAdapter
        Implements: ``discover_files``, ``normalize_locations``, ``preprocess_shard``,
        ``required_vars``, and ``pack_params``.

    csv_directory : str or pathlib.Path
        Directory containing time-sharded CSV files for all sites/cells.

    write_directory : str or pathlib.Path
        Destination directory for NetCDF outputs and temporary parquet shards.

    df_loc : pandas.DataFrame
        Locations table with at least columns ``["gid","lat","lon"]``; optional ``"zone"``.
        The adapter’s ``normalize_locations``:
        - validates columns,
        - adds ``"lon_0-360"``,
        - fills/validates ``"zone"``,
        - sorts for stable site order.

    id_col : str, optional
        Kept for backward compatibility (unused when ``"gid"`` is assumed).

    calendar : {"noleap","standard"}, default "noleap"
        Calendar for numeric DTIME coordinate; Feb 29 filtered for "noleap".

    dtime_resolution_hrs : int, default 1
        Target time resolution in hours for the DTIME axis.

    dtime_units : {"days","hours"}, default "days"
        Units of the numeric DTIME coordinate (e.g., ``"days since YYYY-MM-DD HH:MM:SS"``).

    nzones : int, default 1
        If ``df_loc`` lacks ``"zone"``, adapters may assign repeated zones ``1..nzones``.

    dformat : {"BYPASS","DATM_MODE"}, default "BYPASS"
        Target ELM format selector passed through to the adapter.

    append_attrs : dict, optional
        Extra global NetCDF attributes to include in every file. The exporter also adds:
        ``export_mode`` (``"elm-combined"`` or ``"elm-sites"``) and
        ``pack_scope`` (``"global"`` or ``"per-site"``).

    chunks : tuple[int,...], optional
        Explicit NetCDF chunk sizes.

    include_vars / exclude_vars : Iterable[str], optional
        Allow-/block-lists of ELM short names applied after preprocess. Meta
        columns ``{"gid","time","LATIXY","LONGXY","zone"}`` are always kept.

    Side Effects
    ------------
    - Creates a temporary directory of per-site parquet shards under ``write_directory``.
    - Writes NetCDF files to ``write_directory`` in the chosen layout.
    - Writes a ``zone_mappings.txt`` file either at the root (``elm-combined``)
      or inside each site directory (``elm-sites``).

    Notes
    -----
    - **Packing**: global packing for ``elm-combined``; per-site packing for ``elm-sites``.
    - **Required columns**: CSV shards and ``df_loc`` both use ``"gid"``; CSVs include the
      adapter’s date/time column (renamed to ``"time"`` during preprocess).
    - **Combined (lat/lon) layout**: does **not** enforce regular grids; axes are the unique
      sorted lat/lon from ``df_loc`` (sparse OK).
    """

    def __init__(
        self,
        adapter,
        csv_directory,
        write_directory,
        domain,
        id_col=None,
        calendar="noleap",
        dtime_resolution_hrs=1,
        dtime_units="days",
        nzones=1,
        dformat="BYPASS",
        append_attrs=None,
        chunks=None,
        include_vars=None,
        exclude_vars=None,
    ):
        """
        Parameters
        ----------
        adapter : BaseAdapter
            Implements: ``discover_files``, ``normalize_locations``,
            ``preprocess_shard``, ``required_vars``, and ``pack_params``.

        csv_directory : str or pathlib.Path
            Directory containing time-sharded CSV files for all sites/cells.

        write_directory : str or pathlib.Path
            Destination directory for NetCDF outputs and temporary parquet shards.

        domain : Domain or (Geo)DataFrame
            Canonical spatial domain. Preferred is a ``dapper.domain.Domain``
            instance. For convenience, you may also pass a df_loc-style
            (geo)DataFrame with at least ``["gid", "lon", "lat"]`` (and
            optionally ``"geometry"`` and ``"zone"``); it will be wrapped via
            ``Domain.from_gdf(...)``.

        id_col : str, optional
            Only used when ``domain`` is a DataFrame and does not yet have a
            ``"gid"`` column; in that case, ``id_col`` will be renamed to
            ``"gid"``.
        """
        self.adapter = adapter
        self.csv_directory = Path(csv_directory)
        self.write_directory = Path(write_directory)

        # ---- normalize / wrap into Domain ----
        if isinstance(domain, Domain):
            dom = domain
        else:
            # Backward-compat convenience: accept df_loc-like tables
            dom = Domain.from_gdf(domain, name="from_exporter", id_col=id_col)

        # Ensure lon/lat exist (derived from geometry if needed)
        dom = dom.ensure_lon_lat()

        self.domain = dom
        self.domain_norm = None  # will hold adapter-normalized Domain

        self.id_col = id_col
        self.calendar = calendar
        self.dtime_resolution_hrs = dtime_resolution_hrs
        self.dtime_units = dtime_units
        self.nzones = nzones
        self.dformat = dformat
        self.append_attrs = append_attrs or {}
        self.chunks = chunks
        self.include_vars = set(include_vars) if include_vars else None
        self.exclude_vars = set(exclude_vars) if exclude_vars else None

        self.temp_dir = None
        self.csv_files = None
        self.start_year = None
        self.end_year = None
        self.var_cols = None

        # Meta columns we expect after preprocess
        self.meta_cols = {
            "gid",
            "time",
            "LONGXY",
            "LATIXY",
            "zone",
            "lat",
            "lon",
            "lon_0-360",
        }
        self.dtime_vals = None
        self.dtime_units_out = None
        self.nt = None

        # Normalized location table (as DataFrame) – filled in run()
        self.df_loc_norm = None

        # derived
        self.gid_to_isite = None

    # ---------------------- public ----------------------

    def run(self, output_mode, pack_scope=None):
        """
        output_mode : 'elm-combined' | 'elm-sites' | 'raw-site-parquet' | 'raw-site-csv'
        """
        possible_output_modes = ['elm-combined', 'elm-sites', 'raw-site-parquet', 'raw-site-csv']
        if output_mode not in possible_output_modes:
            raise KeyError(f'Your requested output_mode is invalid. Choose from {possible_output_modes}.')

        # 0) prep – adapter-normalized locations
        self.df_loc_norm = self.adapter.normalize_locations(self.domain.gdf, self.nzones)
        # keep a normalized Domain alongside the raw Domain
        self.domain_norm = self.domain.with_gdf(self.df_loc_norm)

        self.csv_files, self.start_year, self.end_year = self.adapter.discover_files(
            self.csv_directory, self.calendar
        )
        self.write_directory.mkdir(parents=True, exist_ok=True)

        # 1) shards → parquet (raw or elm-prep)
        self.temp_dir = self.write_directory / "temp_parquet"
        utils._rm_and_mkdir(self.temp_dir)

        # --- raw export if requested ---
        if output_mode in ("raw-site-parquet", "raw-site-csv"):
            self._pass1_to_parquet_raw()  # writes temp_parquet/<gid>.parquet (raw columns)
            parquet_files = list(self.temp_dir.glob("*.parquet"))
            if not parquet_files:
                print("No site Parquets to export; exiting.")
                utils._rm(self.temp_dir)
                return

            # destination folder
            out_sub = "sites_parquet" if output_mode == "raw-site-parquet" else "sites_csv"
            dest_dir = self.write_directory / out_sub
            dest_dir.mkdir(parents=True, exist_ok=True)

            # zone mapping once at root of dest folder: lon, lat, zone, id
            zm_path = dest_dir / "zone_mappings.txt"
            zm = self._zone_mappings_table()
            zm[["lon", "lat", "zone_str", "id"]].to_csv(
                zm_path, index=False, header=False, sep="\t"
            )

            # write one file per gid
            for pf in parquet_files:
                gid = pf.stem
                if output_mode == "raw-site-parquet":
                    try:
                        pf.rename(dest_dir / f"{gid}.parquet")
                    except OSError:
                        df = pd.read_parquet(pf)
                        df.to_parquet(dest_dir / f"{gid}.parquet", index=False)
                else:  # raw-site-csv
                    df = pd.read_parquet(pf)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    df.to_csv(dest_dir / f"{gid}.csv", index=False)

            utils._rm(self.temp_dir)
            print(f"{output_mode} export complete → {dest_dir}")
            return

        effective_pack = self._resolve_pack_scope(output_mode, pack_scope)

        # ----- ELM path: shards → parquet (preprocessed) -----
        self._pass1_to_parquet()
        parquet_files = sorted(self.temp_dir.glob("*.parquet"))
        if not parquet_files:
            print("No site Parquets to export; exiting.")
            utils._rm(self.temp_dir)
            return

        # 2) vars + global DTIME
        sample_df = pd.read_parquet(parquet_files[0])
        self.var_cols = [c for c in sample_df.columns if c not in self.meta_cols]
        if not self.var_cols:
            print("No data variables found; exiting.")
            utils._rm(self.temp_dir)
            return

        self.dtime_vals, self.dtime_units_out, _ = dt.create_dtime(
            sample_df, self.calendar, self.dtime_units, self.dtime_resolution_hrs
        )
        self.nt = len(self.dtime_vals)

        # stable site order (used for mapping & attrs in some tools)
        site_order = self.df_loc_norm["gid"].tolist()
        self.gid_to_isite = {g: i for i, g in enumerate(site_order)}

        source_tag = getattr(self.adapter, "DRIVER_TAG", "ERA5")
        years_span = f"{self.start_year}-{self.end_year}"

        # file-level attrs with provenance
        nc_attrs = self._file_attrs(output_mode, effective_pack)

        # 3) branch by output_mode
        if output_mode == "elm-combined":
            if effective_pack != "global":
                raise ValueError("elm-combined requires pack_scope='global'.")
            self._write_elm_combined(parquet_files, years_span, nc_attrs)

        elif output_mode == "elm-sites":
            self._write_elm_sites(parquet_files, years_span, nc_attrs)

        else:
            raise ValueError("output_mode must be 'elm-combined' or 'elm-sites'.")

        # 4) cleanup
        utils._rm(self.temp_dir)

    # ---------------------- private: helpers ----------------------
    def _zone_mappings_table(self):
        """
        Build a table for zone_mappings.txt with columns:
        lon, lat, zone_str (\"01\"), id (1..N per zone).

        Keeps gid in the DataFrame for filtering, but it is not written to file.
        """
        required = {"gid", "lat", "lon", "zone"}
        missing = required - set(self.df_loc_norm.columns)
        if missing:
            raise KeyError(
                f"df_loc_norm is missing columns required for zone_mappings: {missing}"
            )

        df = self.df_loc_norm.copy()

        # Stable ordering: by zone, then gid (or you could use lat/lon)
        df = df.sort_values(["zone", "gid"]).reset_index(drop=True)

        # 1..N per zone
        df["id"] = df.groupby("zone").cumcount() + 1

        # "01", "02", ...
        df["zone_str"] = df["zone"].astype(int).astype(str).str.zfill(2)

        # Keep gid for filtering, but it won't get written to the file
        return df[["gid", "lon", "lat", "zone_str", "id"]]

    def _write_elm_combined(self, parquet_files, years_span, nc_attrs):
        # global packing scan
        packing = self._compute_global_packing(parquet_files)

        # lat/lon axes and gid→(iy,ix) from the normalized Domain
        lats_axis, lons_axis, gid_to_ij = self.domain_norm.elm_latlon_layout(
            decimals=LATLON_DECIMALS,
            use_lon_0360=True,
        )

        # initialize one lat/lon file per var
        self._grid_paths = {}
        for v in self.var_cols:
            ao, sf = packing[v]
            source_tag = getattr(self.adapter, "DRIVER_TAG", "ERA5")
            path_nc = self.write_directory / f"{source_tag}_{v}_{years_span}.nc"
            initialize_met_netcdf(
                path_nc=path_nc,
                var_name=v,
                dims=('DTIME','lat','lon'),
                dim_lengths={'DTIME': self.nt, 'lat': len(lats_axis), 'lon': len(lons_axis)},
                dtime_name='DTIME',
                dtime_vals=self.dtime_vals,
                dtime_units=self.dtime_units_out,
                calendar=self.calendar,
                coord_specs=[
                    {"name":"lat","dtype":"f4","dims":("lat",),"data":lats_axis.astype("float32"),
                     "attrs":{"units":"degrees_north","long_name":"latitude"}},
                    {"name":"lon","dtype":"f4","dims":("lon",),"data":lons_axis.astype("float32"),
                     "attrs":{"units":"degrees_east","long_name":"longitude","note":"0–360 convention"}},
                ],
                add_offset=ao, scale_factor=sf,
                dtype="i2", fill_value=32767,
                chunks=self.chunks, write_pattern="by_cell",
                append_attrs=nc_attrs, nc_format="NETCDF4_CLASSIC",
            )
            self._grid_paths[v] = path_nc

        # zone mappings at root (lon \t lat \t gid \t zone)
        zm_df = self.df_loc_norm.copy()
        zm_df["lon"] = zm_df["lon"].round(LATLON_DECIMALS)
        zm_df["lat"] = zm_df["lat"].round(LATLON_DECIMALS)
        zm_path = self.write_directory / "zone_mappings.txt"
        zm_df[["lon", "lat", "gid", "zone"]].to_csv(
            zm_path, index=False, header=False, sep="\t"
        )

        # scatter each site into lat/lon
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue

            ij = gid_to_ij.get(gid)
            if ij is None:
                # no matching geometry in Domain; skip
                continue
            iy, ix = ij

            dvals_site, _, site_df = dt.create_dtime(
                df0, self.calendar, self.dtime_units, self.dtime_resolution_hrs
            )
            if len(dvals_site) != self.nt:
                raise ValueError(f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}")

            for v in self.var_cols:
                if v not in site_df.columns:
                    continue
                vals = site_df[v].to_numpy(dtype="float64")
                if not np.isfinite(vals).any():
                    continue

                append_met_netcdf(
                    path_nc=self._grid_paths[v],
                    var_name=v,
                    data=vals,
                    indexers={"DTIME": slice(0, self.nt), "lat": iy, "lon": ix},
                )

        print("elm-combined export complete.")

    def _write_elm_sites(self, parquet_files, years_span, nc_attrs):
        # per-site packing + per-site files
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue

            # align to global dtime
            dvals_site, _, site_df = dt.create_dtime(
                df0, self.calendar, self.dtime_units, self.dtime_resolution_hrs
            )
            if len(dvals_site) != self.nt:
                raise ValueError(f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}")

            # site meta
            row = self.df_loc_norm[self.df_loc_norm["gid"] == gid]
            if row.empty:
                continue
            lat = float(row["lat"].iloc[0])
            lon = float(row["lon"].iloc[0])
            lon0360 = float(row["lon_0-360"].iloc[0])
            zone_val = row["zone"].iloc[0]
            zone_str = str(int(zone_val)).zfill(2)

            # site dir + zone mapping (lon, lat, zone, id for this gid)
            site_dir = self.write_directory / gid
            site_dir.mkdir(parents=True, exist_ok=True)
            zm_path = site_dir / "zone_mappings.txt"
            if not zm_path.exists():
                zm = self._zone_mappings_table()
                row_zm = zm[zm["gid"] == gid]
                if row_zm.empty:
                    # shouldn't happen, but don't crash if df_loc_norm is out of sync
                    continue
                row_zm[["lon", "lat", "zone_str", "id"]].to_csv(
                    zm_path, index=False, header=False, sep="\t"
                )

            # each var: per-site packing + write/append
            for v in self.var_cols:
                if v not in site_df.columns:
                    continue
                vals = site_df[v].to_numpy(dtype="float64")
                if not np.isfinite(vals).any():
                    continue

                ao, sf = self.adapter.pack_params(v, data=vals)
                if (not np.isfinite(sf)) or abs(sf) < 1e-12:
                    rep = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
                    ao, sf = float(rep), 1.0

                source_tag = getattr(self.adapter, "DRIVER_TAG", "ERA5")
                path_nc = site_dir / f"{source_tag}_{v}_{years_span}_z{zone_str}.nc"
                if not path_nc.exists():
                    initialize_met_netcdf(
                        path_nc=path_nc,
                        var_name=v,
                        dims=('n','DTIME'),
                        dim_lengths={'n': 1, 'DTIME': self.nt},
                        dtime_name='DTIME',
                        dtime_vals=self.dtime_vals,
                        dtime_units=self.dtime_units_out,
                        calendar=self.calendar,
                        coord_specs=[
                            {"name":"LATIXY","dtype":"f4","dims":("n",),"data":np.array([lat], dtype="float32"),
                             "attrs":{"units":"degrees_north","long_name":"latitude"}},
                            {"name":"LONGXY","dtype":"f4","dims":("n",),"data":np.array([lon0360], dtype="float32"),
                             "attrs":{"units":"degrees_east","long_name":"longitude","note":"0–360 convention"}},
                        ],
                        add_offset=float(ao), scale_factor=float(sf),
                        dtype="i2", fill_value=32767,
                        chunks=self.chunks, write_pattern="by_site",
                        append_attrs=nc_attrs, nc_format="NETCDF4_CLASSIC",
                    )

                append_met_netcdf(
                    path_nc=path_nc,
                    var_name=v,
                    data=vals,
                    indexers={"n": 0, "DTIME": slice(0, self.nt)},
                )

        print("elm-sites export complete.")

    # ---------------------- private: helpers ----------------------

    def _resolve_pack_scope(self, output_mode, pack_scope):
        if pack_scope is None:
            return "per-site" if output_mode == "elm-sites" else "global"
        if output_mode == "elm-combined" and pack_scope != "global":
            raise ValueError(f"{output_mode} requires pack_scope='global'.")
        if output_mode == "elm-sites" and pack_scope not in ("per-site", "per_site", "site", "local"):
            return "per-site"
        return "per-site" if output_mode == "elm-sites" else "global"

    def _file_attrs(self, output_mode: str, pack_scope: str) -> dict:
        """Merge user attrs with exporter provenance and return a new dict."""
        attrs = dict(self.append_attrs)  # copy user attrs if provided

        # Basic exporter provenance
        attrs.update({
            "export_mode": output_mode,
            "pack_scope": pack_scope,
        })

        # Adapter-driven provenance
        source_name = getattr(self.adapter, "SOURCE_NAME", None)
        driver_tag  = getattr(self.adapter, "DRIVER_TAG", None)

        if source_name and "met_source" not in attrs:
            attrs["met_source"] = source_name
        if driver_tag and "met_driver" not in attrs:
            attrs["met_driver"] = driver_tag

        # Helpful extras (won't override user-provided keys)
        attrs.setdefault("dapper_adapter", self.adapter.__class__.__name__)
        attrs.setdefault(
            "dapper_note",
            f"Created by dapper.met.exporter using {self.adapter.__class__.__name__} "
            f"with dtime_resolution_hrs={self.dtime_resolution_hrs}."
        )
        attrs.setdefault("dapper_created_utc", _dt.datetime.utcnow().isoformat() + "Z")

        return attrs

    def _pass1_to_parquet(self):
        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            df = pd.read_csv(f, dtype={"gid": "string"})

            # Handle 'gid' column
            if "gid" not in df.columns:
                # Single-site convenience: infer gid from df_loc_norm
                unique_gids = self.df_loc_norm["gid"].unique()
                if len(unique_gids) == 1:
                    single_gid = str(unique_gids[0])
                    df["gid"] = single_gid
                    warnings.warn(
                        "Source CSV has no 'gid' column; treating it as single-site and "
                        f"assigning gid='{single_gid}' to all rows.",
                        UserWarning,
                    )
                else:
                    raise KeyError(
                        "Expected a 'gid' column in CSV input, and df_loc_norm has "
                        f"{len(unique_gids)} distinct gids; cannot infer site key."
                    )

            df["gid"] = df["gid"].astype(str).str.strip()

            merged = df.merge(self.df_loc_norm[["gid","lat","lon","zone"]], on="gid", how="inner")
            if merged.empty:
                print("SKIP FILE: merge produced 0 rows.")
                continue

            ppdf = self.adapter.preprocess_shard(
                merged, self.start_year, self.end_year, self.calendar, self.dformat
            )

            # optional var filtering
            if self.include_vars is not None or self.exclude_vars is not None:
                keep_meta = {"gid","time","LONGXY","LATIXY","zone"}
                cols = set(ppdf.columns)
                if self.include_vars is not None:
                    cols = (cols & self.include_vars) | keep_meta
                if self.exclude_vars is not None:
                    cols = (cols - self.exclude_vars) | keep_meta
                ppdf = ppdf[[c for c in ppdf.columns if c in cols]]

            # data columns after preprocess (and any filtering)
            data_cols = [c for c in ppdf.columns if c not in self.meta_cols]
            if not data_cols:
                print(f"[skip] file {i+1}: no data columns after preprocess ({ppdf.columns.tolist()})")
                continue

            # write per-site parquet, skip all-NaN sites
            for gid, gdf in ppdf.groupby("gid", sort=False):
                if not np.isfinite(gdf[data_cols].to_numpy()).any():
                    continue
                out = self.temp_dir / f"{gid}.parquet"
                if out.exists():
                    write(out, gdf, append=True)
                else:
                    write(out, gdf)

    def _pass1_to_parquet_raw(self):
        """
        Read all source CSV shards, merge canonical site metadata from df_loc_norm,
        and write raw per-site Parquet files to temp_parquet/<gid>.parquet.
        """
        raw_cols = None  # capture schema from first shard to keep order consistent

        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            df = pd.read_csv(f, dtype={"gid": "string"})

            # must have 'gid' and 'date' in the source CSVs
            if "gid" not in df.columns:
                unique_gids = self.df_loc_norm["gid"].unique()
                if len(unique_gids) == 1:
                    single_gid = str(unique_gids[0])
                    df["gid"] = single_gid
                    warnings.warn(
                        "Source CSV has no 'gid' column; treating it as single-site and "
                        f"assigning gid='{single_gid}' to all rows (raw export).",
                        UserWarning,
                    )
                else:
                    raise KeyError(
                        "Expected a 'gid' column in CSV input, and df_loc_norm has "
                        f"{len(unique_gids)} distinct gids; cannot infer site key."
                    )

            if "date" not in df.columns:
                raise KeyError("Expected a 'date' column in CSV input.")

            # Prefer canonical site metadata from df_loc_norm (avoid conflicts)
            df = df.drop(columns=[c for c in ("lat", "lon", "zone") if c in df.columns])

            merged = df.merge(
                self.df_loc_norm[["gid", "lat", "lon", "zone"]],
                on="gid",
                how="inner",
            )
            if merged.empty:
                print("SKIP FILE: merge produced 0 rows.")
                continue

            # enforce consistent column order across shards
            if raw_cols is None:
                front = [c for c in ["gid", "date", "lat", "lon", "zone"] if c in merged.columns]
                rest = [c for c in merged.columns if c not in front]
                raw_cols = front + rest

            merged = merged.reindex(columns=raw_cols)

            # append rows grouped by gid
            for gid, gdf in merged.groupby("gid", sort=False):
                gdf = gdf.sort_values("date").drop_duplicates(subset="date", keep="last")
                out = self.temp_dir / f"{gid}.parquet"
                if out.exists():
                    write(str(out), gdf, append=True)
                else:
                    write(str(out), gdf)

    def _compute_global_packing(self, parquet_files):
        vmin = {v: np.inf for v in self.var_cols}
        vmax = {v: -np.inf for v in self.var_cols}
        for pf in parquet_files:
            d = pd.read_parquet(pf)
            for v in self.var_cols:
                if v not in d.columns:
                    continue
                vals = d[v].to_numpy()
                if np.isfinite(vals).any():
                    vmin[v] = min(vmin[v], float(np.nanmin(vals)))
                    vmax[v] = max(vmax[v], float(np.nanmax(vals)))
        packing = {}
        for v in self.var_cols:
            if not np.isfinite(vmin[v]) or not np.isfinite(vmax[v]):
                ao, sf = self.adapter.pack_params(v, data=None)
            else:
                ao, sf = self.adapter.pack_params(v, data=np.array([vmin[v], vmax[v]]))
            if (not np.isfinite(sf)) or abs(sf) < 1e-12:
                rep = vmin[v] if np.isfinite(vmin[v]) else 0.0
                ao, sf = float(rep), 1.0
            packing[v] = (float(ao), float(sf))
        return packing
