"""Meteorological data export pipelines."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import dapper.met.temporal as dt
from dapper.domains.domain import Domain
from dapper.elm.utils import elm_data_dicts
from dapper.geo.constants import LATLON_DECIMALS
from dapper.io import fs as utils
from dapper.io.attrs import utc_timestamp
from dapper.met.writers import append_met_netcdf, initialize_met_netcdf
from dapper.schemas.elm import ELM_UNITS


def _parquet_write(path, df, *, append: bool = False) -> None:
    """Write/append parquet using fastparquet.

    The met exporter uses Parquet as a temp on-disk cache to avoid holding the
    full time series for all points in memory.

    We import fastparquet lazily so importing dapper.met (or Domain.export_met)
    doesn't immediately fail in environments that haven't installed the optional
    Parquet dependency yet.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].astype("object")

    try:
        from fastparquet import write as _fp_write
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "dapper.met requires 'fastparquet' to write intermediate parquet caches. "
            "Install with: pip install fastparquet"
        ) from e

    _fp_write(str(path), df, append=bool(append))


class Exporter:
    """
    Source-agnostic meteorological exporter.

    Ingests time-sharded CSVs for all sites/cells, preprocesses each shard with a
    pluggable *adapter*, stages per-gid parquet files, and writes packed (int16)
    ELM MET NetCDFs. The layout follows ``domain.mode``:

    - ``"cellset"``: ``<out_dir>/MET/<file>.nc`` per variable with dims
      ``('DTIME', 'lat', 'lon')``, one global packing, and a single DTIME axis
      shared by every point (sparse lat/lon axes are allowed).
    - ``"sites"``: ``<out_dir>/<gid>/MET/<file>.nc`` per variable with dims
      ``('n', 'DTIME')`` (``n=1``), per-site packing, and a per-site DTIME axis.

    Each MET directory also gets a ``zone_mappings.txt``.

    All dataset-specific logic (file discovery, unit conversions, renaming to ELM
    short names) lives in the adapter (see
    :class:`dapper.met.adapters.base.BaseAdapter`).

    Parameters
    ----------
    adapter : BaseAdapter
        Source adapter (e.g. :class:`~dapper.met.adapters.era5.ERA5Adapter`).
    src_path : str or pathlib.Path
        Input location passed to ``adapter.discover_files``.
    domain : Domain
        Sites/cells to export; ``gid`` links CSV rows to cells.
    out_dir : str or pathlib.Path, optional
        Output root. Defaults to ``domain.run_dir`` (requires ``domain.path_out``).
    calendar : str, default "noleap"
        DTIME calendar; Feb 29 is dropped for no-leap aliases.
    dtime_resolution_hrs : float, default 1
        Target DTIME step in hours (fractions allowed).
    dtime_units : {"days", "hours"}, default "days"
        Units of the numeric DTIME coordinate.
    dformat : {"BYPASS", "DATM_MODE"}, default "BYPASS"
        ELM forcing format passed to the adapter.
    append_attrs : dict, optional
        Extra global attributes for every file. Exporter provenance
        (``domain_mode``, ``pack_scope``, ``export_start_year``, ...) is added
        without overriding user keys, except ``domain_mode``, ``pack_scope``,
        ``clip_to_full_years`` and the export years, which always win.
    chunks : tuple[int, ...], optional
        Explicit NetCDF chunk sizes; auto-chunked when omitted.
    include_vars, exclude_vars : Iterable[str], optional
        Allow/block lists of ELM short names applied after preprocessing. Meta
        columns ``gid``, ``time``, ``LATIXY``, ``LONGXY`` and ``zone`` are kept.
    clip_to_full_years : bool or None, optional
        Clip the discovered year range to complete calendar years (raises if
        none). ``None`` keeps the adapter default (True for ERA5).

    Notes
    -----
    Temporary parquet shards are written under ``<out_dir>/.dapper_tmp`` and
    removed at the end of :meth:`run`.
    """

    def __init__(
        self,
        adapter,
        src_path,
        *,
        domain: Domain,
        out_dir=None,
        calendar: str = "noleap",
        dtime_resolution_hrs: float = 1,
        dtime_units: str = "days",
        dformat: str = "BYPASS",
        append_attrs: dict | None = None,
        chunks=None,
        include_vars=None,
        exclude_vars=None,
        clip_to_full_years: bool | None = None,
    ):
        """Create a MET exporter for ``domain``; see the class docstring."""
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a dapper.domains.domain.Domain instance")

        self.adapter = adapter
        self.src_path = Path(src_path)

        # Ensure lon/lat exist (derived from geometry if needed)
        self.domain = domain.ensure_cells_lon_lat()

        # Output root for this exporter run. If not provided, uses Domain.run_dir
        # (requires Domain.path_out to be set).
        self.group_dir = Path(out_dir) if out_dir is not None else self.domain.run_dir

        self.calendar = dt.normalize_calendar(calendar)
        self.dtime_resolution_hrs = dtime_resolution_hrs
        self.dtime_units = dtime_units
        self.dformat = dformat
        self.append_attrs = append_attrs or {}
        self.chunks = chunks
        self.include_vars = set(include_vars) if include_vars else None
        self.exclude_vars = set(exclude_vars) if exclude_vars else None
        self.clip_to_full_years = clip_to_full_years

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

        self.temp_dir = None
        self.csv_files = None
        self.start_year = None
        self.end_year = None
        self.var_cols = None
        self.dtime_vals = None
        self.dtime_units_out = None
        self.nt = None
        self._sample_temporal_options = {}

        # Normalized location table (as DataFrame) – filled in run()
        self.df_loc_norm = None

        # derived
        self.gid_to_isite = None

        # Cached ELM descriptions (used for per-variable attrs)
        self._elm_desc = (elm_data_dicts() or {}).get("short_descriptions", {})

    # ---------------------- public ----------------------

    def run(
        self, *, pack_scope=None, filename: str | None = None, overwrite: bool = False
    ) -> None:
        """Run the MET export for this exporter’s Domain.

        The output layout is derived from ``Domain.mode``:
          - ``sites``: writes ``<run_dir>/<gid>/MET/{prefix_}{var}.nc`` and a per-site
            ``zone_mappings.txt`` (always zone=01, id=1).
          - ``cellset``: writes ``<run_dir>/MET/{prefix_}{var}.nc`` and a single
            ``zone_mappings.txt`` covering all locations (zones taken from df_loc, default 1).

        Parameters
        ----------
        pack_scope
            Optional packing strategy override. Defaults to ``per-site`` for sites and
            ``global`` for cellset outputs.

        filename
            Optional filename template for output NetCDF files. Supports two formats:
            - Template with {var} placeholder: 'ERA5_{var}_1950-2025_z01' generates 'ERA5_TBOT_1950-2025_z01.nc'
            - Simple prefix (legacy): 'prefix' generates 'prefix_TBOT.nc'
            If omitted, the default is '<driver>_<var>_<start>-<end>_z01.nc'.
            The '.nc' extension is added automatically.

        overwrite
            If True, clears existing MET outputs before writing.
        """
        dom_mode = getattr(self.domain, "mode", None)
        if dom_mode not in ("sites", "cellset"):
            raise ValueError(
                f"Domain.mode must be 'sites' or 'cellset' for MET export (got {dom_mode!r})."
            )

        # 0) prep – adapter-normalized locations from the Domain contract
        self.df_loc_norm = self.adapter.normalize_locations(
            self.domain.to_df_loc(), id_col=None
        )
        # Sites mode: always a single zone
        if dom_mode == "sites":
            self.df_loc_norm["zone"] = 1

        # Discover input shards
        self.csv_files, self.start_year, self.end_year = self._discover_files()
        self.group_dir.mkdir(parents=True, exist_ok=True)

        # 1) shards → parquet (ELM-prep)
        self.temp_dir = self.group_dir / ".dapper_tmp" / "met_parquet"
        if self.temp_dir.exists():
            utils.remove_directory_contents(self.temp_dir, remove_directory=False)
        else:
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Optional: clear old MET outputs to avoid accidental appends
        if overwrite:
            self._clear_existing_outputs()

        self.filename_prefix = (
            filename.strip() if isinstance(filename, str) and filename.strip() else None
        )

        effective_pack = self._resolve_pack_scope(dom_mode, pack_scope)

        self._pass1_to_parquet()
        parquet_files = sorted(self.temp_dir.glob("*.parquet"))
        if not parquet_files:
            print("No site Parquets to export; exiting.")
            utils.remove_directory_contents(self.temp_dir, remove_directory=True)
            return

        # 2) vars + canonical DTIME axis (cellset only)
        sample_df = pd.read_parquet(parquet_files[0])
        self._sample_temporal_options = self._temporal_options(sample_df)
        self.var_cols = [c for c in sample_df.columns if c not in self.meta_cols]
        if not self.var_cols:
            print("No data variables found; exiting.")
            utils.remove_directory_contents(self.temp_dir, remove_directory=True)
            return

        # For cellset (lat/lon) outputs we need a single canonical DTIME axis.
        # For sites outputs, allow each gid to have its own coverage/cadence.
        if dom_mode == "cellset":
            self.dtime_vals, self.dtime_units_out, aligned0 = self._create_dtime(
                sample_df
            )
            # Canonical datetime axis for all points in this cellset export.
            self._time_axis = pd.to_datetime(aligned0["time"]).to_numpy()
            self.nt = len(self.dtime_vals)
        else:
            self.dtime_vals = None
            self.dtime_units_out = None
            self._time_axis = None
            self.nt = None

        # stable site order
        site_order = self.df_loc_norm["gid"].tolist()
        self.gid_to_isite = {g: i for i, g in enumerate(site_order)}

        # file-level attrs with provenance
        nc_attrs = self._file_attrs(dom_mode, effective_pack)

        # 3) branch by Domain.mode
        if dom_mode == "cellset":
            if effective_pack != "global":
                raise ValueError("Domain(mode='cellset') requires pack_scope='global'.")
            self._write_elm_combined(parquet_files, nc_attrs)
        else:
            self._write_elm_sites(parquet_files, nc_attrs)

        # 4) cleanup
        utils.remove_directory_contents(self.temp_dir, remove_directory=True)

    def _run_dir_for_gid(self, gid: str) -> Path:
        """Resolve run directory for a gid under the current output root."""
        if getattr(self.domain, "mode", None) == "sites":
            return self.group_dir / str(gid)
        return self.group_dir

    def _met_dir_for_gid(self, gid: str | None = None) -> Path:
        """Resolve MET directory for a gid (sites mode) or the single run (cellset mode)."""
        if getattr(self.domain, "mode", None) == "sites":
            if gid is None:
                raise ValueError("gid is required when domain.mode == 'sites'.")
            return self._run_dir_for_gid(gid) / "MET"
        return self._run_dir_for_gid("unused") / "MET"

    def _nc_filename(self, var: str) -> str:
        """Return the output NetCDF filename for a given variable.

        Supports two formats:
        - Template with {var} placeholder: 'ERA5_{var}_1950-2025_z01' -> 'ERA5_TBOT_1950-2025_z01.nc'
        - Simple prefix (legacy): 'prefix' -> 'prefix_TBOT.nc'

        With no override, use the ELM convention with the adapter driver tag,
        actual export years, and the single-zone suffix.
        """
        if getattr(self, "filename_prefix", None):
            # Check if template contains {var} placeholder
            if "{var}" in self.filename_prefix:
                return f"{self.filename_prefix.format(var=var)}.nc"
            # Legacy prefix mode
            return f"{self.filename_prefix}_{var}.nc"
        driver = str(getattr(self.adapter, "DRIVER_TAG", "MET") or "MET").strip()
        return f"{driver}_{var}_{self.start_year}-{self.end_year}_z01.nc"

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

        # Prefer 0–360 longitude convention when available (matches LONGXY)
        if "lon_0-360" in df.columns:
            df["lon"] = df["lon_0-360"].astype(float)

        # Stable ordering: by zone, then gid (or you could use lat/lon)
        df = df.sort_values(["zone", "gid"]).reset_index(drop=True)

        # 1..N per zone
        df["id"] = df.groupby("zone").cumcount() + 1

        # "01", "02", ...
        df["zone_str"] = df["zone"].astype(int).astype(str).str.zfill(2)

        # Keep gid for filtering, but it won't get written to the file
        lon_col = "lon_0-360" if "lon_0-360" in df.columns else "lon"
        return df[["gid", lon_col, "lat", "zone_str", "id"]].rename(
            columns={lon_col: "lon"}
        )

    def _var_attrs(self, var: str) -> dict:
        """Best-effort per-variable metadata for ELM outputs."""
        attrs: dict = {}
        units = ELM_UNITS.get(var)
        if units:
            attrs["units"] = units
        long_name = self._elm_desc.get(var)
        if long_name:
            attrs["long_name"] = long_name
        if var in set(getattr(self.adapter, "INTERVAL_END_VARS", ())):
            attrs["cell_methods"] = "time: mean"
            attrs["time_representation"] = "interval_start"
        return attrs

    def _attr_value(self, value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        text = str(value).strip()
        return text if text else None

    def _wkt_geometry_type(self, value):
        text = self._attr_value(value)
        if text is None:
            return None
        return text.split("(", 1)[0].strip().lower()

    def _site_attrs(self, row: pd.Series) -> dict:
        """Per-site metadata that should travel into each site NetCDF."""
        attrs = {}
        if "method" in row.index:
            value = self._attr_value(row["method"])
            if value is not None:
                attrs["sampling_method"] = value
        if "sampled_geometry" in row.index:
            value = self._wkt_geometry_type(row["sampled_geometry"])
            if value is not None:
                attrs["sampling_geometry_type"] = value
        if "source_file" in row.index:
            value = self._attr_value(row["source_file"])
            if value is not None:
                attrs["source_geometry_file"] = value
        if "feature_count" in row.index:
            value = self._attr_value(row["feature_count"])
            if value is not None:
                attrs["source_feature_count"] = value
        for column in (
            "sampling_backend",
            "sampling_dataset",
            "sampling_reference_lon",
            "sampling_reference_lat",
            "sampling_grid_cell_count",
            "sampling_grid_coordinates",
            "sampling_grid_weights",
            "sampling_requested_start",
            "sampling_start",
            "sampling_source_end",
            "sampling_output_end",
            "sampling_estimated_seconds",
            "sampling_elapsed_seconds",
        ):
            if column in row.index:
                value = self._attr_value(row[column])
                if value is not None:
                    attrs[column] = value
        return attrs

    def _clear_existing_outputs(self) -> None:
        """Remove existing MET outputs so this run is idempotent."""
        if getattr(self.domain, "mode", None) == "sites":
            for gid in self.df_loc_norm["gid"].astype(str).tolist():
                met_dir = self._met_dir_for_gid(gid)
                if met_dir.exists():
                    utils.remove_directory_contents(met_dir, remove_directory=True)
        else:
            met_dir = self._met_dir_for_gid()
            if met_dir.exists():
                utils.remove_directory_contents(met_dir, remove_directory=False)

    def _write_elm_combined(self, parquet_files, nc_attrs):
        # global packing scan
        packing = self._compute_global_packing(parquet_files)

        # lat/lon axes and gid→(iy,ix) from the Domain
        lats_axis, lons_axis, gid_to_ij = self.domain.elm_latlon_layout(
            decimals=LATLON_DECIMALS,
            use_lon_0360=True,
        )

        met_dir = self._met_dir_for_gid()
        met_dir.mkdir(parents=True, exist_ok=True)

        # initialize one lat/lon file per var
        self._grid_paths = {}
        for v in self.var_cols:
            ao, sf = packing[v]
            path_nc = met_dir / self._nc_filename(v)
            initialize_met_netcdf(
                path_nc=path_nc,
                var_name=v,
                dims=("DTIME", "lat", "lon"),
                dim_lengths={
                    "DTIME": self.nt,
                    "lat": len(lats_axis),
                    "lon": len(lons_axis),
                },
                dtime_name="DTIME",
                dtime_vals=self.dtime_vals,
                dtime_units=self.dtime_units_out,
                calendar=self.calendar,
                start_year=self.start_year,
                end_year=self.end_year,
                coord_specs=[
                    {
                        "name": "lat",
                        "dtype": "f4",
                        "dims": ("lat",),
                        "data": lats_axis.astype("float32"),
                        "attrs": {"units": "degrees_north", "long_name": "latitude"},
                    },
                    {
                        "name": "lon",
                        "dtype": "f4",
                        "dims": ("lon",),
                        "data": lons_axis.astype("float32"),
                        "attrs": {
                            "units": "degrees_east",
                            "long_name": "longitude",
                            "note": "0–360 convention",
                        },
                    },
                ],
                add_offset=ao,
                scale_factor=sf,
                dtype="i2",
                fill_value=32767,
                chunks=self.chunks,
                write_pattern="by_cell",
                append_attrs=nc_attrs,
                var_attrs=self._var_attrs(v),
                nc_format="NETCDF4_CLASSIC",
            )
            self._grid_paths[v] = path_nc

        # zone mappings at root (lon \t lat \t zone_str \t id)
        zm_path = met_dir / "zone_mappings.txt"
        zm = self._zone_mappings_table().copy()
        zm["lon"] = zm["lon"].round(LATLON_DECIMALS)
        zm["lat"] = zm["lat"].round(LATLON_DECIMALS)
        zm[["lon", "lat", "zone_str", "id"]].to_csv(
            zm_path, index=False, header=False, sep="\t"
        )

        # scatter each site into lat/lon
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue
            if "zone" in df0.columns and df0["zone"].nunique(dropna=True) > 1:
                raise NotImplementedError(
                    f"{gid}: multi-zone MET forcing is not supported yet (found multiple zone values in the time series). "
                    "For now, export with a single zone per gid."
                )

            ij = gid_to_ij.get(gid)
            if ij is None:
                # no matching geometry in Domain; skip
                continue
            iy, ix = ij

            dvals_site, _, site_df = self._create_dtime(df0)
            if len(dvals_site) != self.nt:
                raise ValueError(
                    f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}"
                )
            site_axis = pd.to_datetime(site_df["time"]).to_numpy()
            if site_axis.shape != self._time_axis.shape or not np.array_equal(
                site_axis, self._time_axis
            ):
                raise ValueError(
                    f"{gid}: time axis does not match the canonical axis for this export. "
                    "This usually indicates the source data coverage differs across points "
                    "(start/end timestamps or cadence)."
                )

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

        print("cellset export complete.")

    def _write_elm_sites(self, parquet_files, nc_attrs):
        # per-site packing + per-site files
        for pf in parquet_files:
            gid = pf.stem
            df0 = pd.read_parquet(pf)
            if df0.empty:
                continue

            # sites mode: compute per-site DTIME independently (coverage/cadence may differ by gid)
            dvals_site, dtime_units_out, site_df = self._create_dtime(df0)
            nt_site = len(dvals_site)

            if "zone" in site_df.columns and site_df["zone"].nunique(dropna=True) > 1:
                raise NotImplementedError(
                    f"{gid}: multi-zone MET forcing is not supported yet (found multiple zone values in the time series). "
                    "For now, export with a single zone per gid."
                )

            # site meta
            row = self.df_loc_norm[self.df_loc_norm["gid"] == gid]
            if row.empty:
                continue
            site_row = row.iloc[0]
            lat = float(site_row["lat"])
            lon0360 = float(site_row["lon_0-360"])
            zone_str = "01"
            site_attrs = self._site_attrs(site_row)

            # site MET dir + zone mapping (lon, lat, zone, id for this gid)
            met_dir = self._met_dir_for_gid(gid)
            met_dir.mkdir(parents=True, exist_ok=True)
            zm_path = met_dir / "zone_mappings.txt"
            if not zm_path.exists():
                # sites output: write per-site zone_mappings with id=1
                pd.DataFrame(
                    [[lon0360, lat, zone_str, 1]],
                    columns=["lon", "lat", "zone_str", "id"],
                ).to_csv(zm_path, index=False, header=False, sep="	")

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

                path_nc = met_dir / self._nc_filename(v)
                if not path_nc.exists():
                    initialize_met_netcdf(
                        path_nc=path_nc,
                        var_name=v,
                        dims=("n", "DTIME"),
                        dim_lengths={"n": 1, "DTIME": nt_site},
                        dtime_name="DTIME",
                        dtime_vals=dvals_site,
                        dtime_units=dtime_units_out,
                        calendar=self.calendar,
                        start_year=self.start_year,
                        end_year=self.end_year,
                        coord_specs=[
                            {
                                "name": "LATIXY",
                                "dtype": "f4",
                                "dims": ("n",),
                                "data": np.array([lat], dtype="float32"),
                                "attrs": {
                                    "units": "degrees_north",
                                    "long_name": "latitude",
                                },
                            },
                            {
                                "name": "LONGXY",
                                "dtype": "f4",
                                "dims": ("n",),
                                "data": np.array([lon0360], dtype="float32"),
                                "attrs": {
                                    "units": "degrees_east",
                                    "long_name": "longitude",
                                    "note": "0–360 convention",
                                },
                            },
                        ],
                        add_offset=float(ao),
                        scale_factor=float(sf),
                        dtype="i2",
                        fill_value=32767,
                        chunks=self.chunks,
                        write_pattern="by_site",
                        append_attrs={**nc_attrs, **site_attrs},
                        var_attrs=self._var_attrs(v),
                        nc_format="NETCDF4_CLASSIC",
                    )

                append_met_netcdf(
                    path_nc=path_nc,
                    var_name=v,
                    data=vals,
                    indexers={"n": 0, "DTIME": slice(0, nt_site)},
                )

        print("sites export complete.")

    # ---------------------- private: helpers ----------------------

    def _discover_files(self):
        """Discover source files while preserving older adapter compatibility."""
        discover = self.adapter.discover_files
        try:
            params = inspect.signature(discover).parameters
        except (TypeError, ValueError):
            params = {}

        accepts_clip_kw = "clip_to_full_years" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_clip_kw:
            return discover(
                self.src_path,
                self.calendar,
                clip_to_full_years=self.clip_to_full_years,
            )

        if self.clip_to_full_years is not None:
            warnings.warn(
                f"{self.adapter.__class__.__name__}.discover_files does not accept "
                "'clip_to_full_years'; ignoring the requested value.",
                UserWarning,
            )
        return discover(self.src_path, self.calendar)

    def _resolve_pack_scope(self, dom_mode: str, pack_scope):
        """Determine packing strategy given Domain.mode and an optional override."""
        if pack_scope is None:
            return "per-site" if dom_mode == "sites" else "global"

        if dom_mode == "cellset":
            if str(pack_scope).strip().lower().replace("_", "-") != "global":
                raise ValueError("Domain(mode='cellset') requires pack_scope='global'.")
            return "global"
        # Sites mode always packs per site; any override value is accepted.
        return "per-site"

    def _temporal_options(self, df) -> dict:
        method = getattr(self.adapter, "temporal_options", None)
        if method is None:
            return {}
        return (
            method(
                df,
                start_year=self.start_year,
                end_year=self.end_year,
                calendar=self.calendar,
            )
            or {}
        )

    def _create_dtime(self, df):
        return dt.create_dtime(
            df,
            self.calendar,
            self.dtime_units,
            self.dtime_resolution_hrs,
            **self._temporal_options(df),
        )

    def _file_attrs(self, dom_mode: str, pack_scope: str) -> dict:
        """Merge user attrs with exporter provenance and return a new dict."""
        attrs = dict(self.append_attrs)  # copy user attrs if provided

        # Basic exporter provenance
        attrs.update(
            {
                "domain_mode": dom_mode,
                "pack_scope": pack_scope,
                "clip_to_full_years": (
                    "adapter_default"
                    if self.clip_to_full_years is None
                    else str(bool(self.clip_to_full_years)).lower()
                ),
                "export_start_year": int(self.start_year),
                "export_end_year": int(self.end_year),
            }
        )

        # Adapter-driven provenance
        source_name = getattr(self.adapter, "SOURCE_NAME", None)
        driver_tag = getattr(self.adapter, "DRIVER_TAG", None)

        if source_name and "met_source" not in attrs:
            attrs["met_source"] = source_name
        if driver_tag and "met_driver" not in attrs:
            attrs["met_driver"] = driver_tag

        # Helpful extras (won't override user-provided keys)
        attrs.setdefault("dapper_adapter", self.adapter.__class__.__name__)
        attrs.setdefault(
            "dapper_note",
            f"Created by dapper.met.exporter using {self.adapter.__class__.__name__} "
            f"with dtime_resolution_hrs={self.dtime_resolution_hrs}.",
        )
        attrs.setdefault("dapper_created_utc", utc_timestamp())

        temporal_metadata = getattr(self.adapter, "temporal_metadata", None)
        if temporal_metadata is not None:
            for key, value in (
                temporal_metadata(self._sample_temporal_options) or {}
            ).items():
                attrs.setdefault(key, value)

        return attrs

    def _pass1_to_parquet(self):
        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i + 1} of {len(self.csv_files)}: {f}")
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

            # Prefer canonical site metadata from df_loc_norm (avoid merge suffixes)
            # Some sources (notably FLUXNET variants) may include lat/lon columns.
            df = df.drop(
                columns=[
                    c for c in ("lat", "lon", "zone", "lon_0-360") if c in df.columns
                ]
            )

            merged = df.merge(
                self.df_loc_norm[["gid", "lat", "lon", "zone"]], on="gid", how="inner"
            )
            if merged.empty:
                print("SKIP FILE: merge produced 0 rows.")
                continue

            ppdf = self.adapter.preprocess_shard(
                merged, self.start_year, self.end_year, self.calendar, self.dformat
            )

            # optional var filtering
            if self.include_vars is not None or self.exclude_vars is not None:
                keep_meta = {"gid", "time", "LONGXY", "LATIXY", "zone"}
                cols = set(ppdf.columns)
                if self.include_vars is not None:
                    cols = (cols & self.include_vars) | keep_meta
                if self.exclude_vars is not None:
                    cols = (cols - self.exclude_vars) | keep_meta
                ppdf = ppdf[[c for c in ppdf.columns if c in cols]]

            # data columns after preprocess (and any filtering)
            data_cols = [c for c in ppdf.columns if c not in self.meta_cols]
            if not data_cols:
                print(
                    f"[skip] file {i + 1}: no data columns after preprocess ({ppdf.columns.tolist()})"
                )
                continue

            # write per-site parquet, skip all-NaN sites
            for gid, gdf in ppdf.groupby("gid", sort=False):
                if not np.isfinite(gdf[data_cols].to_numpy()).any():
                    continue
                out = self.temp_dir / f"{gid}.parquet"
                if out.exists():
                    _parquet_write(out, gdf, append=True)
                else:
                    _parquet_write(out, gdf)

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
