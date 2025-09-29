# dapper/met/exporter.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fastparquet import write

from dapper.utils import utils
import dapper.met.temporal as dt
from dapper.met.writers import initialize_met_netcdf, append_met_netcdf


class Exporter:
    """
    Source-agnostic meteorological exporter.

    This class orchestrates a two-pass pipeline that ingests time-sharded CSVs
    for many sites/cells, preprocesses them via a pluggable *adapter*, and
    writes ELM-ready NetCDF outputs in one of three layouts:

      1) ``"sites_file"``  – one NetCDF per variable with dims ``('n','DTIME')``
         containing *all* sites (global packing).
      2) ``"site_dirs"``   – one directory per site; each directory contains
         one NetCDF per variable with dims ``('n','DTIME')`` where ``n=1``
         (per-site packing).
      3) ``"gridded"``     – one NetCDF per variable with dims
         ``('DTIME','lat','lon')`` (global packing).

    Exporter is *source-agnostic*: all dataset-specific logic (file discovery,
    unit conversions, renaming to ELM short names, etc.) lives in an adapter
    that implements the `BaseAdapter` interface (e.g., an ``ERA5Adapter``).
    The exporter handles staging (CSV → per-site parquet), global DTIME axis
    creation, packing scans, chunking, and NetCDF I/O.

    Parameters
    ----------
    adapter : BaseAdapter
        An object implementing the adapter interface:
        ``discover_files``, ``normalize_locations``, ``preprocess_shard``,
        ``required_vars``, and ``pack_params``. This is the only component that
        knows how to go from your raw CSV columns to ELM variables.

    csv_directory : str or pathlib.Path
        Directory containing time-sharded CSV files for all sites/cells
        (e.g., yearly shards). Files must include a date column (named by the
        adapter; commonly ``"date"``) and a site identifier column named
        **``"gid"``** (assumed from here on).

    write_directory : str or pathlib.Path
        Destination directory for NetCDF outputs and temporary parquet shards.
        The exporter will create (and, for the temp folder, clear) subfolders
        under this path as needed.

    df_loc : pandas.DataFrame
        Locations table with at least columns:
        ``["gid", "lat", "lon"]``; ``"zone"`` is optional. The adapter’s
        ``normalize_locations`` will:
        - validate columns,
        - add ``"lon_0-360"``,
        - fill or validate ``"zone"`` values, and
        - sort to produce a stable site order.

    id_col : str, optional
        Kept for backward compatibility. Current adapters assume the input CSVs
        and ``df_loc`` already use the id column name **``"gid"``**.

    calendar : {"noleap", "standard"}, default "noleap"
        Calendar used when constructing the numeric DTIME coordinate and when
        filtering / removing Feb 29 as appropriate.

    dtime_resolution_hrs : int, default 1
        Target time resolution in hours for the output DTIME axis. If set to 1
        and ``force_half_hour_for_hourly=True``, the exporter uses a 30-minute
        axis (see Notes).

    dtime_units : {"days", "hours"}, default "days"
        Units of the numeric DTIME coordinate (e.g., ``"days since YYYY-MM-DD HH:MM:SS"``).

    nzones : int, default 1
        If ``df_loc`` lacks a ``"zone"`` column, adapters may use this to assign
        repeated zones ``1..nzones`` across rows (implementation-dependent).

    dformat : {"BYPASS", "DATM_MODE"}, default "BYPASS"
        Target ELM format selector passed through to the adapter, which decides
        which variables to produce and how to preprocess. (Most implementations
        currently support ``"BYPASS"``.)

    force_half_hour_for_hourly : bool, default True
        Workaround for an E3SM/ELM hourly index issue: when ``True`` and
        ``dtime_resolution_hrs == 1``, the exporter upsamples to 30-minute
        DTIME and fills values accordingly.

    append_attrs : dict, optional
        Extra global NetCDF attributes to include in every output file. The
        exporter will *also* add two provenance attributes:
        ``export_mode`` (one of ``"sites_file"``, ``"site_dirs"``, ``"gridded"``)
        and ``pack_scope`` (``"global"`` or ``"per-site"``), which override
        conflicting keys in ``append_attrs``.

    chunks : tuple[int, ...], optional
        Explicit NetCDF chunk sizes. If omitted, reasonable defaults are chosen
        based on layout and DTIME step (e.g., ``(1, t_chunk)`` for
        ``('n','DTIME')`` when writing by site; ``(t_chunk, 1, 1)`` for
        ``('DTIME','lat','lon')`` when writing by cell).

    include_vars : Iterable[str], optional
        Allow-list of ELM short names to retain after preprocessing. Meta
        columns ``{"gid", "time", "LATIXY", "LONGXY", "zone"}`` are always kept.

    exclude_vars : Iterable[str], optional
        Block-list of ELM short names to drop after preprocessing. Applied
        after ``include_vars`` if both are provided.

    Side Effects
    ------------
    - Creates a temporary directory of per-site parquet shards under
      ``write_directory`` during the run.
    - Writes NetCDF files to ``write_directory`` in the chosen layout.
    - Writes a ``zone_mappings.txt`` file either at the root (``sites_file``,
      ``gridded``) or inside each site directory (``site_dirs``).

    Usage
    -----
    >>> adapter = ERA5Adapter()
    >>> exp = Exporter(
    ...     adapter=adapter,
    ...     csv_directory="path/to/csvs",
    ...     write_directory="out/elm",
    ...     df_loc=df_loc,                # with gid/lat/lon/(zone)
    ...     calendar="noleap",
    ...     dtime_resolution_hrs=1,
    ...     dtime_units="days",
    ... )
    >>> exp.run(output_mode="sites_file")   # pack_scope inferred as "global"
    >>> exp.run(output_mode="site_dirs")    # pack_scope inferred as "per-site"
    >>> exp.run(output_mode="gridded")      # requires unique (lat,lon)

    Notes
    -----
    - **Packing**:  
      *Global* packing (single ``add_offset``/``scale_factor`` per variable) is
      used for ``"sites_file"`` and ``"gridded"``; *per-site* packing is used for
      ``"site_dirs"``. You may pass ``pack_scope`` to ``run(...)`` to override,
      but incompatible choices raise an error.

    - **Required columns**:  
      CSV shards and ``df_loc`` must both use ``"gid"`` as the identifier, and
      CSVs must include a date/time column as expected by the adapter
      (commonly ``"date"`` which is renamed to ``"time"`` during preprocessing).

    - **Gridded layout**:  
      Requires that ``df_loc`` have *unique* (lat, lon) pairs. Duplicates are
      rejected with a helpful error.

    - **Half-hour workaround**:  
      When ``dtime_resolution_hrs == 1`` and
      ``force_half_hour_for_hourly=True``, the exporter upsamples to 30-minute
      DTIME and fills values (linear for state variables, forward-fill for
      fluxes) to avoid a known hourly indexing issue in E3SM/ELM.

    - **Dependencies**:  
      Uses ``fastparquet`` for parquet staging and ``netCDF4`` for writing
      (via ``dapper.met.writers.initialize_met_netcdf`` and
      ``dapper.met.writers.append_met_netcdf``).

    Implementation Notes / Caveats
    ------------------------------
    - The constructor parameter ``id_col`` is currently retained for
      backward compatibility but is not used when the pipeline assumes
      identifier column name ``"gid"`` throughout.
    - In the provided code, the line ``if self.force_half_hour_for_hourly: True``
      in ``__init__`` is a no-op and can be removed.
    - The exporter stores transient state (e.g., discovered files, DTIME axis,
      packing) on the instance during a run; create a new instance or re-run
      with care if you mutate those attributes externally.
    """

    def __init__(self, adapter, csv_directory, write_directory, df_loc,
                 id_col=None, calendar='noleap', dtime_resolution_hrs=1,
                 dtime_units='days', nzones=1, dformat="BYPASS",
                 force_half_hour_for_hourly=True, append_attrs=None,
                 chunks=None, include_vars=None, exclude_vars=None):

        self.adapter = adapter
        self.csv_directory = Path(csv_directory)
        self.write_directory = Path(write_directory)
        self.df_loc = df_loc
        self.id_col = id_col
        self.calendar = calendar
        self.dtime_resolution_hrs = dtime_resolution_hrs
        self.dtime_units = dtime_units
        self.nzones = nzones
        self.dformat = dformat
        self.force_half_hour_for_hourly = force_half_hour_for_hourly
        self.append_attrs = append_attrs or {}
        self.chunks = chunks
        self.include_vars = set(include_vars) if include_vars else None
        self.exclude_vars = set(exclude_vars) if exclude_vars else None

        self.temp_dir = None
        self.csv_files = None
        self.start_year = None
        self.end_year = None
        self.var_cols = None
        self.meta_cols = {"gid","time","LONGXY","LATIXY","zone"}
        self.dtime_vals = None
        self.dtime_units_out = None
        self.nt = None
        self.df_loc_norm = None

        # derived
        self.gid_to_isite = None

        if self.force_half_hour_for_hourly:
            True

    # ---------------------- public ----------------------

    def run(self, output_mode, pack_scope=None):
        """
        output_mode : 'sites_file' | 'site_dirs' | 'gridded' | 'site_parquet' | 'site_csv'
        """
        # 0) prep
        self.df_loc_norm = self.adapter.normalize_locations(self.df_loc, self.nzones)
        self.csv_files, self.start_year, self.end_year = self.adapter.discover_files(self.csv_directory, self.calendar)
        self.write_directory.mkdir(parents=True, exist_ok=True)

        # 1) shards → parquet (raw or elm-prep)
        self.temp_dir = self.write_directory / "temp_parquet"
        utils._rm_and_mkdir(self.temp_dir)

        # --- raw export if requested ---
        if output_mode in ("site_parquet", "site_csv"):
            self._pass1_to_parquet_raw()  # writes temp_parquet/<gid>.parquet (raw columns)
            parquet_files = list(self.temp_dir.glob("*.parquet"))
            if not parquet_files:
                print("No site Parquets to export; exiting.")
                utils._rm(self.temp_dir)
                return

            # destination folder
            out_sub = "sites_parquet" if output_mode == "site_parquet" else "sites_csv"
            dest_dir = self.write_directory / out_sub
            dest_dir.mkdir(parents=True, exist_ok=True)

            # zone mapping once at root of dest folder
            zm_path = dest_dir / "zone_mappings.txt"
            self.df_loc_norm[["gid", "zone"]].to_csv(zm_path, index=False, header=False, sep="\t")

            # write one file per gid
            for pf in parquet_files:
                gid = pf.stem
                if output_mode == "site_parquet":
                    # move/rename is fast; fallback to copy if cross-device issues
                    try:
                        pf.rename(dest_dir / f"{gid}.parquet")
                    except OSError:
                        df = pd.read_parquet(pf)
                        df.to_parquet(dest_dir / f"{gid}.parquet", index=False)
                else:  # site_csv
                    df = pd.read_parquet(pf)
                    # ensure stable column order and ISO dates
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    df.to_csv(dest_dir / f"{gid}.csv", index=False)

            utils._rm(self.temp_dir)
            print(f"{output_mode} export complete → {dest_dir}")
            return
        
        effective_pack = self._resolve_pack_scope(output_mode, pack_scope)

        # 0) prep
        self.df_loc_norm = self.adapter.normalize_locations(self.df_loc, self.nzones)
        self.csv_files, self.start_year, self.end_year = self.adapter.discover_files(self.csv_directory, self.calendar)
        self.write_directory.mkdir(parents=True, exist_ok=True)

        # 1) shards → parquet
        self.temp_dir = self.write_directory / "temp_parquet"
        utils._rm_and_mkdir(self.temp_dir)
        self._pass1_to_parquet()

        parquet_files = list(self.temp_dir.glob("*.parquet"))
        if not parquet_files:
            print("No site Parquets to export; exiting.")
            return

        # 2) vars + global DTIME
        sample_df = pd.read_parquet(parquet_files[0])
        self.var_cols = [c for c in sample_df.columns if c not in self.meta_cols]
        if not self.var_cols:
            print("No data variables found; exiting.")
            utils._rm(self.temp_dir)
            return

        self.dtime_vals, self.dtime_units_out, _ = dt.create_dtime(
            sample_df, self.calendar, self.dtime_units,
            self.dtime_resolution_hrs, self.force_half_hour_for_hourly
        )
        self.nt = len(self.dtime_vals)

        # stable site order / coords (used in all modes)
        site_order = self.df_loc_norm["gid"].tolist()
        self.gid_to_isite = {g: i for i, g in enumerate(site_order)}
        idx = self.df_loc_norm.set_index("gid").loc[site_order]
        lats = idx["lat"].to_numpy()
        lons0360 = idx["lon_0-360"].to_numpy()
        years_span = f"ERA5_{self.start_year}-{self.end_year}"

        # file-level attrs with provenance
        nc_attrs = self._file_attrs(output_mode, effective_pack)

        # 3) branch by output_mode
        if output_mode == "sites_file":
            if effective_pack != "global":
                raise ValueError("sites_file requires pack_scope='global'.")
            self._write_sites_file(parquet_files, years_span, lats, lons0360, nc_attrs)

        elif output_mode == "site_dirs":
            # per-site packing; pack_scope ignored
            self._write_site_dirs(parquet_files, years_span, nc_attrs)

        elif output_mode == "gridded":
            if effective_pack != "global":
                raise ValueError("gridded requires pack_scope='global'.")
            self._write_gridded(parquet_files, years_span, nc_attrs)

        else:
            raise ValueError("output_mode must be 'sites_file', 'site_dirs', or 'gridded'.")

        # 4) cleanup
        utils._rm(self.temp_dir)

    # ---------------------- private: branches ----------------------

    def _write_sites_file(self, parquet_files, years_span, lats, lons0360, nc_attrs):
        # global packing scan
        packing = self._compute_global_packing(parquet_files)

        # initialize one multi-site file per var
        self._sites_file_paths = {}
        for v in self.var_cols:
            ao, sf = packing[v]
            path_nc = self.write_directory / f"{years_span}_{v}.nc"
            initialize_met_netcdf(
                path_nc=path_nc,
                var_name=v,
                dims=('n','DTIME'),
                dim_lengths={'n': len(self.gid_to_isite), 'DTIME': self.nt},
                dtime_name='DTIME',
                dtime_vals=self.dtime_vals,
                dtime_units=self.dtime_units_out,
                calendar=self.calendar,
                coord_specs=[
                    {"name":"LATIXY","dtype":"f4","dims":("n",),"data":lats,
                     "attrs":{"units":"degrees_north","long_name":"latitude"}},
                    {"name":"LONGXY","dtype":"f4","dims":("n",),"data":lons0360,
                     "attrs":{"units":"degrees_east","long_name":"longitude","note":"0–360 convention"}},
                ],
                add_offset=ao, scale_factor=sf,
                dtype="i2", fill_value=32767,
                chunks=self.chunks, write_pattern="by_site",
                append_attrs=nc_attrs, nc_format="NETCDF4_CLASSIC",
            )
            self._sites_file_paths[v] = path_nc

        # append each site
        for pf in parquet_files:
            gid = pf.stem
            isite = self.gid_to_isite.get(gid)
            if isite is None:
                continue
            df0 = pd.read_parquet(pf)
            dvals_site, _, site_df = dt.create_dtime(
                df0, self.calendar, self.dtime_units,
                self.dtime_resolution_hrs, self.force_half_hour_for_hourly
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
                    path_nc=self._sites_file_paths[v],
                    var_name=v,
                    data=vals,
                    indexers={"n": isite, "DTIME": slice(0, self.nt)},
                )

        # zone mappings at root
        zm_path = self.write_directory / "zone_mappings.txt"
        self.df_loc_norm[["gid", "zone"]].to_csv(zm_path, index=False, header=False, sep="\t")
        print("sites_file export complete.")

    def _write_site_dirs(self, parquet_files, years_span, nc_attrs):
        # per-site packing + per-site files
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue

            # align to global dtime
            dvals_site, _, site_df = dt.create_dtime(
                df0, self.calendar, self.dtime_units,
                self.dtime_resolution_hrs, self.force_half_hour_for_hourly
            )
            if len(dvals_site) != self.nt:
                raise ValueError(f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}")

            # site meta
            row = self.df_loc_norm[self.df_loc_norm["gid"] == gid]
            if row.empty:
                continue
            lat = float(row["lat"].iloc[0])
            lon0360 = float(row["lon_0-360"].iloc[0])
            zone_str = str(int(row["zone"].iloc[0])).zfill(2)

            # site dir + zone mapping
            site_dir = self.write_directory / gid
            site_dir.mkdir(parents=True, exist_ok=True)
            zm_path = site_dir / "zone_mappings.txt"
            if not zm_path.exists():
                pd.DataFrame({"gid":[gid], "zone":[zone_str]}).to_csv(zm_path, index=False, header=False, sep="\t")

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

                path_nc = site_dir / f"ERA5_{v}_{years_span.split('_')[1]}_z{zone_str}.nc"
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

        print("site_dirs export complete.")

    def _write_gridded(self, parquet_files, years_span, nc_attrs):
        # global packing scan
        packing = self._compute_global_packing(parquet_files)

        # build grid axes (unique sorted)
        lats_axis = np.unique(self.df_loc_norm["lat"].to_numpy()); lats_axis.sort()
        lons_axis = np.unique(self.df_loc_norm["lon_0-360"].to_numpy()); lons_axis.sort()

        # ensure no duplicate (lat, lon) cells
        dup_check = self.df_loc_norm.groupby(["lat", "lon_0-360"]).size()
        if (dup_check > 1).any():
            dups = dup_check[dup_check > 1]
            raise ValueError(f"Duplicate (lat,lon) entries in df_loc for gridded output: {dups.index.tolist()}")

        # index maps (rounded for stability)
        lat_key = {round(float(v), 6): i for i, v in enumerate(lats_axis)}
        lon_key = {round(float(v), 6): j for j, v in enumerate(lons_axis)}

        # initialize one gridded file per var
        self._grid_paths = {}
        for v in self.var_cols:
            ao, sf = packing[v]
            path_nc = self.write_directory / f"ERA5_{v}_{self.start_year}-{self.end_year}.nc"
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
                    {"name":"lat","dtype":"f4","dims":("lat",),"data":lats_axis,
                     "attrs":{"units":"degrees_north","long_name":"latitude"}},
                    {"name":"lon","dtype":"f4","dims":("lon",),"data":lons_axis,
                     "attrs":{"units":"degrees_east","long_name":"longitude","note":"0–360 convention"}},
                ],
                add_offset=ao, scale_factor=sf,
                dtype="i2", fill_value=32767,
                chunks=self.chunks, write_pattern="by_cell",
                append_attrs=nc_attrs, nc_format="NETCDF4_CLASSIC",
            )
            self._grid_paths[v] = path_nc

        # write zone mappings (root, gid \t zone)
        zm_path = self.write_directory / "zone_mappings.txt"
        self.df_loc_norm[["gid", "zone"]].to_csv(zm_path, index=False, header=False, sep="\t")

        # scatter each site into the grid
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue

            row = self.df_loc_norm[self.df_loc_norm["gid"] == gid]
            if row.empty:
                continue

            plat = round(float(row["lat"].iloc[0]), 6)
            plon = round(float(row["lon_0-360"].iloc[0]), 6)
            iy = lat_key.get(plat, None)
            ix = lon_key.get(plon, None)
            if iy is None or ix is None:
                continue

            dvals_site, _, site_df = dt.create_dtime(
                df0, self.calendar, self.dtime_units,
                self.dtime_resolution_hrs, self.force_half_hour_for_hourly
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

        print("gridded export complete.")

    # ---------------------- private: helpers ----------------------

    def _resolve_pack_scope(self, output_mode, pack_scope):
        if pack_scope is None:
            return "per-site" if output_mode == "site_dirs" else "global"
        # user supplied a preference; validate
        if output_mode in ("sites_file", "gridded") and pack_scope != "global":
            raise ValueError(f"{output_mode} requires pack_scope='global'.")
        if output_mode == "site_dirs" and pack_scope not in ("per-site", "per_site", "site", "local"):
            return "per-site"
        return "per-site" if output_mode == "site_dirs" else "global"

    def _file_attrs(self, output_mode: str, pack_scope: str) -> dict:
        """Merge user attrs with exporter provenance and return a new dict."""
        attrs = dict(self.append_attrs)  # copy
        # these two keys will override any conflicting user-supplied values
        attrs.update({
            "export_mode": output_mode,
            "pack_scope": pack_scope,
        })
        return attrs

    def _pass1_to_parquet(self):
        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            df = pd.read_csv(f)

            # Assume CSVs already use 'gid'
            if "gid" not in df.columns:
                raise KeyError("Expected a 'gid' column in CSV input.")
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

            data_cols = [c for c in ppdf.columns if c not in self.meta_cols]
            if not data_cols:
                continue

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
        Read all source CSV shards, merge in canonical site metadata from df_loc_norm,
        and write raw per-site Parquet files to temp_parquet/<gid>.parquet.
        No renaming, no conversions; preserves original column names (including 'date').
        """
        raw_cols = None  # capture schema from first shard to keep order consistent

        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            df = pd.read_csv(f)

            # must have 'gid' and 'date' in the source CSVs
            if "gid" not in df.columns:
                raise KeyError("Expected a 'gid' column in CSV input.")
            if "date" not in df.columns:
                raise KeyError("Expected a 'date' column in CSV input.")

            df["gid"] = df["gid"].astype(str).str.strip()
            df["date"] = pd.to_datetime(df["date"])

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
                # put id/date first, then the rest
                front = [c for c in ["gid", "date", "lat", "lon", "zone"] if c in merged.columns]
                rest = [c for c in merged.columns if c not in front]
                raw_cols = front + rest

            merged = merged.reindex(columns=raw_cols)

            # append rows grouped by gid
            for gid, gdf in merged.groupby("gid", sort=False):
                gdf = gdf.sort_values("date").drop_duplicates(subset="date", keep="last")
                out = self.temp_dir / f"{gid}.parquet"
                if out.exists():
                    write(out, gdf, append=True)
                else:
                    write(out, gdf)


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


# -------- small local helpers until you wire in your existing ones --------


