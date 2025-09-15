import os
import numpy as np
import pandas as pd
from pathlib import Path
from fastparquet import write

from dapper.met.writers import initialize_met_netcdf, append_met_netcdf
from dapper.met.writers import _compute_auto_chunks  # if you want it elsewhere too

class Exporter:
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

    def run(self, output_mode, pack_scope):
        # 0) prep
        self.df_loc_norm = self.adapter.normalize_locations(self.df_loc, self.id_col, self.nzones)
        self.csv_files, self.start_year, self.end_year = self.adapter.discover_files(self.csv_directory, self.calendar)
        self.write_directory.mkdir(parents=True, exist_ok=True)

        # 1) shards → parquet by gid
        self.temp_dir = self.write_directory / "temp_parquet"
        _rm_and_mkdir(self.temp_dir)
        self._pass1_to_parquet()

        parquet_files = list(self.temp_dir.glob("*.parquet"))
        if not parquet_files:
            print("No site Parquets to export; exiting.")
            return

        # 2) discover vars + global DTIME
        sample_df = pd.read_parquet(parquet_files[0])
        self.var_cols = [c for c in sample_df.columns if c not in self.meta_cols]
        self.dtime_vals, self.dtime_units_out, _ = _create_dtime(sample_df,
            self.calendar, self.dtime_units, self.dtime_resolution_hrs, self.force_half_hour_for_hourly)
        self.nt = len(self.dtime_vals)

        # 3) packing
        if pack_scope != "global":
            raise ValueError("This minimal exporter assumes global packing for now.")
        packing = self._compute_global_packing(parquet_files)

        # 4) initialize one multi-site file per var (as an example)
        site_order = self.df_loc_norm["gid"].tolist()
        idx = self.df_loc_norm.set_index("gid").loc[site_order]
        lats = idx["lat"].to_numpy()
        lons = idx["lon_0-360"].to_numpy()
        years_span = f"ERA5_{self.start_year}-{self.end_year}"

        for v in self.var_cols:
            ao, sf = packing[v]
            initialize_met_netcdf(
                path_nc=self.write_directory / f"{years_span}_{v}.nc",
                var_name=v,
                dims=('n','DTIME'),
                dim_lengths={'n': len(site_order), 'DTIME': self.nt},
                dtime_name='DTIME',
                dtime_vals=self.dtime_vals,
                dtime_units=self.dtime_units_out,
                calendar=self.calendar,
                coord_specs=[
                    {"name":"LATIXY","dtype":"f4","dims":("n",),"data":lats,
                     "attrs":{"units":"degrees_north","long_name":"latitude"}},
                    {"name":"LONGXY","dtype":"f4","dims":("n",),"data":lons,
                     "attrs":{"units":"degrees_east","long_name":"longitude","note":"0–360 convention"}},
                ],
                add_offset=ao, scale_factor=sf,
                dtype="i2", fill_value=32767,
                chunks=self.chunks, write_pattern="by_site",
                append_attrs=self.append_attrs, nc_format="NETCDF4_CLASSIC"
            )

        # 5) write data
        gid_to_isite = {g:i for i,g in enumerate(site_order)}
        for pf in parquet_files:
            gid = pf.stem
            isite = gid_to_isite.get(gid)
            if isite is None:
                continue
            df0 = pd.read_parquet(pf)
            dvals_site, units_site, site_df = _create_dtime(df0,
                self.calendar, self.dtime_units, self.dtime_resolution_hrs, self.force_half_hour_for_hourly)
            if len(dvals_site) != self.nt:
                raise ValueError(f"{gid}: per-site DTIME length {len(dvals_site)} != global {self.nt}")
            for v in self.var_cols:
                if v not in site_df.columns:
                    continue
                vals = site_df[v].to_numpy(dtype="float64")
                if not np.isfinite(vals).any():
                    continue
                append_met_netcdf(
                    path_nc=self.write_directory / f"{years_span}_{v}.nc",
                    var_name=v,
                    data=vals,
                    indexers={"n": isite, "DTIME": slice(0, self.nt)}
                )

        # 6) cleanup
        _rm(self.temp_dir)

    # ---------- helpers ----------
    def _pass1_to_parquet(self):
        for i, f in enumerate(self.csv_files):
            print(f"Processing file {i+1} of {len(self.csv_files)}: {f}")
            df = pd.read_csv(f)

            idcol = self.adapter.id_column_for_csv(df, self.id_col)
            df = df.rename(columns={idcol: "gid"})
            df["gid"] = df["gid"].astype(str).str.strip()

            merged = df.merge(self.df_loc_norm[["gid","lat","lon","zone"]], on="gid", how="inner")
            if merged.empty:
                print("SKIP FILE: merge produced 0 rows.")
                continue

            ppdf = self.adapter.preprocess_shard(merged, self.start_year, self.end_year, self.calendar, self.dformat)
            data_cols = [c for c in ppdf.columns if c not in self.meta_cols]
            for gid, gdf in ppdf.groupby("gid", sort=False):
                if not np.isfinite(gdf[data_cols].to_numpy()).any():
                    continue
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
                if v not in d.columns: continue
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

# small local helpers until you wire in your existing ones
def _create_dtime(df, calendar, dtime_units, dtime_resolution_hrs, force_half_hour_for_hourly):
    from dapper.met import met_io as io
    return io.create_dtime(df, calendar, dtime_units, dtime_resolution_hrs, force_half_hour_for_hourly)

def _rm_and_mkdir(p):
    if p.exists():
        for f in p.glob("*"): f.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)

def _rm(p):
    if p.exists():
        for f in p.glob("*"): f.unlink()
        p.rmdir()
