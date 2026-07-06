# dapper/domains/domain.py

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from dapper.geo.constants import LATLON_DECIMALS

DomainMode = Literal["sites", "cellset"]
StepName = Literal["met", "topounits"]
CellKind = Literal["site_points", "as_provided"]


def _ensure_geodf(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    *,
    id_col: str = "gid",
    crs_epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """Coerce to a GeoDataFrame with CRS and a 'gid' column."""
    if not isinstance(df, gpd.GeoDataFrame):
        if "geometry" in df.columns:
            df = gpd.GeoDataFrame(df, geometry="geometry", copy=True)
        elif {"lon", "lat"}.issubset(df.columns):
            df = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                copy=True,
                crs=f"EPSG:{crs_epsg}",
            )
        else:
            raise ValueError("Input must have 'geometry' OR both 'lon' and 'lat'.")

    gdf = df.copy()
    if gdf.crs is None:
        gdf.set_crs(epsg=crs_epsg, inplace=True)
    else:
        gdf = gdf.to_crs(f"EPSG:{crs_epsg}")

    if "gid" not in gdf.columns:
        if id_col in gdf.columns:
            gdf = gdf.rename(columns={id_col: "gid"})
        else:
            gdf["gid"] = [f"cell_{i:04d}" for i in range(len(gdf))]

    gdf["gid"] = gdf["gid"].astype(str).str.strip()
    if gdf["gid"].duplicated().any():
        dups = gdf.loc[gdf["gid"].duplicated(), "gid"].tolist()
        raise ValueError(f"Duplicate gid values detected: {dups[:10]} (and possibly more)")
    return gdf


def _rep_point(geom: BaseGeometry) -> BaseGeometry:
    """Representative point (always inside polygon); safe default over centroid."""
    if geom.geom_type == "Point":
        return geom
    return geom.representative_point()


def _ensure_lon_lat(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure lon/lat columns exist (computed from representative points)."""
    if {"lon", "lat"}.issubset(gdf.columns):
        return gdf

    pts = gdf.geometry.apply(_rep_point)
    out = gdf.copy()
    out["lon"] = np.asarray([float(p.x) for p in pts], dtype=float)
    out["lat"] = np.asarray([float(p.y) for p in pts], dtype=float)
    return out


def _make_site_points_from_support(support: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cells as representative points of support geometries (site-mode)."""
    pts = support.geometry.apply(_rep_point)
    cells = support.copy()
    cells["geometry"] = pts
    cells = cells.set_geometry("geometry")
    cells = _ensure_lon_lat(cells)
    return cells


@dataclass(frozen=True)
class Domain:
    """
    Canonical spatial object passed through the dapper pipeline.

    Geometry views:
      - provided: exactly what the user supplied (provenance/plotting)
      - support : what sampling SHOULD use (may be simplified/processed later)
      - cells   : what ELM RUNS on (site points now; per-cell geometries for cellset)

    Prepared sampling views (set during the relevant pipeline step; not on init):
      - met_support
      - topo_support

    Mode:
      - sites   : one set of outputs per row (exporters loop internally)
      - cellset : one set of outputs total, including all rows
    """

    name: str
    mode: DomainMode

    provided: gpd.GeoDataFrame
    support: gpd.GeoDataFrame
    cells: gpd.GeoDataFrame

    # --- optional topounits payload (filled after make_topounits) ---
    topounits: "gpd.GeoDataFrame | None" = None
    topounits_dim_name: str = "topounit"
    topounits_id_col: str = "topounit_id"   # the column inside topounits holding unique ids
    topounits_gid_col: str = "gid"          # column linking topounits -> parent cell/site gid

    met_support: Optional[gpd.GeoDataFrame] = None
    topo_support: Optional[gpd.GeoDataFrame] = None

    domain_nc: Optional[Path] = None
    path_out: Optional[Path] = None
    run_group: Optional[str] = None  # if None, defaults to self.name


    # ----------------------------- constructors -----------------------------

    @classmethod
    def from_provided(
        cls,
        provided: Union[gpd.GeoDataFrame, pd.DataFrame],
        *,
        name: str = "domain",
        mode: Optional[DomainMode] = None,
        id_col: str = "gid",
        support: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
        cells: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
        cell_kind: Optional[CellKind] = None,
        domain_nc: Optional[Union[str, Path]] = None,
        path_out: Optional[Union[str, Path]] = None,
        run_group: Optional[str] = None,
    ) -> "Domain":
        """Construct a Domain from a provided geometry table.
        
        The input can be a GeoDataFrame (preferred) or a DataFrame with a geometry column.
        Use *mode* to choose between a single-run cellset or a multi-run sites container.
        If *cells* is not provided, cells are derived from the provided/support geometry depending on *cell_kind*.
        """
        
        prov = _ensure_geodf(provided, id_col=id_col)

        if mode is None:
            if len(prov) > 1:
                raise ValueError(
                    "Domain.from_provided(): provided has >1 row; you must pass mode='sites' or mode='cellset'."
                )
            mode = "cellset"

        sup = _ensure_geodf(support, id_col=id_col) if support is not None else prov.copy()

        if cell_kind is None:
            # Default behavior:
            # - sites: typical is “one run per row” → site points
            # - cellset: typical is “rows ARE cells” → as_provided
            cell_kind = "site_points" if mode == "sites" else "as_provided"

        if cells is not None:
            cel = _ensure_geodf(cells, id_col=id_col)
            cel = _ensure_lon_lat(cel)
        else:
            if cell_kind == "site_points":
                cel = _make_site_points_from_support(sup)
            elif cell_kind == "as_provided":
                cel = prov.copy()
                cel = _ensure_lon_lat(cel)
            else:
                raise ValueError(f"Unknown cell_kind={cell_kind!r}")

        return cls(
            name=name,
            mode=mode,
            provided=prov,
            support=sup,
            cells=cel,
            domain_nc=Path(domain_nc) if domain_nc is not None else None,
            path_out=Path(path_out) if path_out is not None else None,
            run_group=str(run_group) if run_group is not None else None,
        )

    # Back-compat-ish alias (you said you don’t care, but it’s convenient while refactoring)
    @classmethod
    def from_gdf(cls, gdf: Union[gpd.GeoDataFrame, pd.DataFrame], **kwargs) -> "Domain":
        """Alias for :meth:`Domain.from_provided`."""
        
        return cls.from_provided(gdf, **kwargs)

    @classmethod
    def from_geometry(
        cls,
        geometry: BaseGeometry,
        *,
        gid: str = "site",
        name: str = "domain",
        mode: DomainMode = "cellset",
        cell_kind: CellKind = "site_points",
        path_out: Optional[Union[str, Path]] = None,
        run_group: Optional[str] = None,
    ) -> "Domain":
        """Construct a single-feature Domain from a shapely geometry."""
        
        gdf = gpd.GeoDataFrame({"gid": [gid], "geometry": [geometry]}, crs="EPSG:4326")
        return cls.from_provided(gdf, name=name, mode=mode, cell_kind=cell_kind,
                                 path_out=Path(path_out) if path_out is not None else None,
                                 run_group=str(run_group) if run_group is not None else None,)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        name: Optional[str] = None,
        layer: Optional[str] = None,
        id_col: str = "gid",
        mode: Optional[DomainMode] = None,
        cell_kind: Optional[CellKind] = None,
        path_out: Optional[Union[str, Path]] = None,
        run_group: Optional[str] = None,
    ) -> "Domain":
        """Load a geospatial file (e.g., GeoPackage, Shapefile) and construct a Domain."""
        
        path = Path(path)
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        return cls.from_provided(
            gdf,
            name=name or path.stem,
            id_col=id_col,
            mode=mode,
            cell_kind=cell_kind,
            path_out=Path(path_out) if path_out is not None else None,
            run_group=str(run_group) if run_group is not None else None,
        )

    @classmethod
    def from_elm_domain(
        cls,
        path_nc: Union[str, Path],
        *,
        name: Optional[str] = None,
        mask_name: str = "mask",
        frac_name: str = "frac",
        frac_threshold: float = 0.0,
        path_out: Optional[Union[str, Path]] = None,
        run_group: Optional[str] = None,
    ) -> "Domain":
        """
        Build a Domain from an ELM domain NetCDF. This naturally produces a 'cellset'.
        """
        path_nc = Path(path_nc)
        ds = xr.open_dataset(path_nc)

        xc = ds["xc"].values  # (nj, ni)
        yc = ds["yc"].values
        xv = ds["xv"].values  # (nj, ni, nv=4)
        yv = ds["yv"].values

        mask = ds[mask_name].values if mask_name in ds.variables else None
        frac = ds[frac_name].values if frac_name in ds.variables else None

        nj, ni = xc.shape
        rows = []
        gid_counter = 0
        for j in range(nj):
            for i in range(ni):
                if mask is not None and int(mask[j, i]) == 0:
                    continue
                if frac is not None and float(frac[j, i]) <= frac_threshold:
                    continue

                lon_c = float(xc[j, i])
                lat_c = float(yc[j, i])
                corners = list(zip(xv[j, i, :], yv[j, i, :]))
                poly = Polygon(corners)

                gid = f"cell_{gid_counter:05d}"
                gid_counter += 1

                rows.append(
                    {
                        "gid": gid,
                        "lon": lon_c,
                        "lat": lat_c,
                        "geometry": poly,
                        "i": i,
                        "j": j,
                        "frac": float(frac[j, i]) if frac is not None else 1.0,
                    }
                )

        if not rows:
            raise ValueError(f"No active cells found in domain file {path_nc}")

        cells = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return cls(
            name=name or path_nc.stem,
            mode="cellset",
            provided=cells.copy(),
            support=cells.copy(),
            cells=cells,
            met_support=None,
            topo_support=None,
            domain_nc=path_nc,
            path_out=Path(path_out) if path_out is not None else None,
            run_group=str(run_group) if run_group is not None else None,
        )

    # ----------------------------- core helpers -----------------------------

    def copy(self, **updates) -> "Domain":
        """Return a shallow copy of this Domain with updated fields."""
        
        return replace(self, **updates)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def gids(self) -> list[str]:
        """List of ``gid`` values (as strings) for the current cell table."""
        
        return self.cells["gid"].astype(str).tolist()

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Backwards-compat alias for older code.

        Historically dapper exposed a single GeoDataFrame on the domain object
        (often a ``df_loc``-style table with columns like gid/lon/lat/geometry).
        The closest equivalent is :attr:`cells` (the run-level geometry table).
        """
        return self.cells

    def ensure_cells_lon_lat(self) -> "Domain":
        """Ensure ``cells`` contains ``lon`` and ``lat`` columns, derived from geometry if needed."""
        
        cel = _ensure_lon_lat(self.cells)
        if cel is self.cells:
            return self
        return self.copy(cells=cel)

    # ----------------------------- geometry views -----------------------------

    def rep_points(
        self,
        *,
        source: Literal["provided", "support", "cells"] = "support",
        step: Optional[StepName] = None,
    ) -> gpd.GeoDataFrame:
        """
        Representative points for a given geometry view.
        If source='support' and step is provided, uses the prepared support for that step.
        """
        if source == "provided":
            gdf = self.provided
        elif source == "cells":
            gdf = self.cells
        else:
            gdf = self.support_for(step=step)

        out = gdf.copy()
        out["geometry"] = out.geometry.apply(_rep_point)
        out = out.set_geometry("geometry")
        out = _ensure_lon_lat(out)
        return out

    def support_for(self, *, step: Optional[StepName] = None) -> gpd.GeoDataFrame:
        """
        Return the geometry set that should be used for the given step.
        - step=None      -> support
        - step="met"     -> met_support if set else support
        - step="topounits" -> topo_support if set else support
        """
        if step is None:
            return self.support
        if step == "met":
            return self.met_support if self.met_support is not None else self.support
        if step == "topounits":
            return self.topo_support if self.topo_support is not None else self.support
        raise ValueError(f"Unknown step={step!r}")

    def with_step_support(self, step: StepName, gdf: gpd.GeoDataFrame) -> "Domain":
        """Attach a step-specific support GeoDataFrame (for "met" or "topounits")."""
        
        gdf2 = _ensure_geodf(gdf, id_col="gid")
        if step == "met":
            return self.copy(met_support=gdf2)
        if step == "topounits":
            return self.copy(topo_support=gdf2)
        raise ValueError(f"Unknown step={step!r}")

    def simplify_support(
        self,
        tolerance_m: float,
        *,
        step: StepName,
        preserve_topology: bool = True,
        equal_area_epsg: int = 6933,
    ) -> "Domain":
        """
        Simplify the support geometry for a step and store it as met_support/topo_support.
        Does NOT modify provided/support/cells.
        """
        base = self.support_for(step=step)
        g = base.to_crs(epsg=equal_area_epsg).copy()
        g["geometry"] = g.geometry.simplify(tolerance_m, preserve_topology=preserve_topology)
        g = g.to_crs("EPSG:4326")
        return self.with_step_support(step, g)

    # ----------------------------- run iteration (internal) -----------------------------

    def iter_runs(self):
        """
        Yield (run_id, run_domain) where run_domain is always a single-run 'cellset' Domain.
        - mode='cellset' -> yields exactly one run (self)
        - mode='sites'   -> yields one run per gid (single-row Domains)
        """
        if self.mode == "cellset":
            yield self.name, self
            return

        # sites mode: split into one-row domains
        prov = self.provided.set_index("gid", drop=False)
        sup = self.support.set_index("gid", drop=False)
        cel = self.cells.set_index("gid", drop=False)

        met = self.met_support.set_index("gid", drop=False) if self.met_support is not None else None
        topo = self.topo_support.set_index("gid", drop=False) if self.topo_support is not None else None

        for gid in prov.index:
            run_prov = prov.loc[[gid]].reset_index(drop=True)
            run_sup = sup.loc[[gid]].reset_index(drop=True) if gid in sup.index else run_prov.copy()
            run_cel = cel.loc[[gid]].reset_index(drop=True) if gid in cel.index else _make_site_points_from_support(run_sup)

            run_met = met.loc[[gid]].reset_index(drop=True) if met is not None and gid in met.index else None
            run_topo = topo.loc[[gid]].reset_index(drop=True) if topo is not None and gid in topo.index else None

            yield gid, Domain(
                name=str(gid),
                mode="cellset",
                provided=gpd.GeoDataFrame(run_prov, geometry="geometry", crs="EPSG:4326"),
                support=gpd.GeoDataFrame(run_sup, geometry="geometry", crs="EPSG:4326"),
                cells=gpd.GeoDataFrame(run_cel, geometry="geometry", crs="EPSG:4326"),
                met_support=gpd.GeoDataFrame(run_met, geometry="geometry", crs="EPSG:4326") if run_met is not None else None,
                topo_support=gpd.GeoDataFrame(run_topo, geometry="geometry", crs="EPSG:4326") if run_topo is not None else None,
                # propagate output layout + group name
                path_out=self.path_out,
                run_group=(self.run_group or self.name),
                # propagate topounits subset (so exporters in sites mode can see them)
                topounits=self.topounits_for_gid(str(gid)) if self.has_topounits() else None,
                topounits_dim_name=self.topounits_dim_name,
                topounits_id_col=self.topounits_id_col,
                topounits_gid_col=self.topounits_gid_col,
                domain_nc=None,
            )

    # ----------------------------- output pathing -----------------------------
    def _require_path_out(self) -> Path:
        if self.path_out is None:
            raise ValueError(
                "Domain.path_out is not set. Call domain = domain.with_path_out(...)."
            )
        return Path(self.path_out)

    @property
    def group_name(self) -> str:
        """Name of the output group directory for this Domain."""
        
        return str(self.run_group or self.name)

    @property
    def run_dir(self) -> Path:
        """
        Directory holding the main run outputs for this Domain instance.

        For top-level cellset or top-level sites container:
            ``path_out/<group_name>``

        For per-site/per-cellset run domains (created by ``iter_runs`` in sites mode):
            ``path_out/<group_name>/<domain.name>``
        """
        root = self._require_path_out()
        base = root / self.group_name

        # If this is a run-domain coming from a parent sites-domain, self.name is the gid,
        # and group_name is the parent run group. Use a leaf directory for the gid.
        if self.run_group is not None and str(self.name) != str(self.run_group):
            return base / str(self.name)

        return base

    def _run_dir_for(self, run_id: str | None = None) -> Path:
        """
        Resolve the per-run directory for this Domain.

        Notes:
          - mode='cellset': run_id is ignored; returns self.run_dir
          - mode='sites'  : run_id is required when called on the container Domain
        """
        if self.mode == "sites":
            if run_id is None:
                raise ValueError(
                    "run_id is required for path helpers on a sites-mode container Domain. "
                    "Use domain.iter_runs() (run_dom.path_*) or pass run_id explicitly."
                )
            return self.run_dir / str(run_id)
        return self.run_dir

    def path_met_dir(self, run_id: str | None = None) -> Path:
        """Path to the MET output directory for this Domain (or a specific run_id)."""
        
        return self._run_dir_for(run_id) / "MET"

    @property
    def met_dir(self) -> Path:
        """Convenience property for :meth:`Domain.path_met_dir`."""
        
        return self.path_met_dir()

    # --- canonical filenames (exporters can override, but these are the defaults) ---

    def path_domain_nc(self, filename: str = "domain.nc", run_id: str | None = None) -> Path:
        """Default output path for the domain NetCDF for this Domain (or a specific run_id)."""
        
        return self._run_dir_for(run_id) / filename

    def path_surface_nc(self, filename: str = "surfdata.nc", run_id: str | None = None) -> Path:
        """Default output path for the surface NetCDF for this Domain (or a specific run_id)."""
        
        return self._run_dir_for(run_id) / filename

    def path_landuse_nc(self, filename: str = "landuse_timeseries.nc", run_id: str | None = None) -> Path:
        """Default output path for the landuse NetCDF for this Domain (or a specific run_id)."""
        
        return self._run_dir_for(run_id) / filename

    def path_zone_mappings(self, filename: str = "zone_mappings.txt", run_id: str | None = None) -> Path:
        """Default output path for ``zone_mappings.txt`` (under the MET directory)."""
        
        return self.path_met_dir(run_id) / filename


    def ensure_output_dirs(self, *, met: bool = True) -> None:
        """
        Create output directories implied by this Domain (and its runs, if mode='sites').
        Does not write any files.
        """
        # Ensure output root is set (and directory exists)
        _ = self._require_path_out()

        for _, run_dom in self.iter_runs():
            run_dom.run_dir.mkdir(parents=True, exist_ok=True)
            if met:
                run_dom.met_dir.mkdir(parents=True, exist_ok=True)


    # ----------------------------- sampling glue -----------------------------

    def to_df_loc(
        self,
        *,
        lon_col: str = "lon",
        lat_col: str = "lat",
        weight_col: str = "weight",
        frac_col: str = "frac",
        default_weight: float = 1.0,
    ) -> pd.DataFrame:
        """
        Derived location/weight table from cells (internal glue; users shouldn't need to touch).
        """
        dom = self.ensure_cells_lon_lat()
        gdf = dom.cells

        out = pd.DataFrame(index=gdf.index)
        out["gid"] = gdf["gid"].astype(str)
        out[lon_col] = gdf["lon"].to_numpy(dtype="float64")
        out[lat_col] = gdf["lat"].to_numpy(dtype="float64")

        for col in ("method", "sampled_geometry", "source_file", "feature_count"):
            if col in gdf.columns:
                out[col] = gdf[col].to_numpy()

        # Optional per-location zone labels (used by MET cellset decomposition).
        # If missing, default to 1. For sites-mode, exporters may override to a single zone.
        if "zone" in gdf.columns:
            out["zone"] = pd.to_numeric(gdf["zone"], errors="coerce").fillna(1).astype(int)
        else:
            out["zone"] = 1

        if (out["zone"] < 1).any():
            raise ValueError("Domain cells contain zone values < 1. Zones must be positive integers.")

        if frac_col in gdf.columns:
            w = gdf[frac_col].to_numpy(dtype="float64")
        else:
            w = np.full(len(gdf), float(default_weight), dtype="float64")
        out[weight_col] = w
        return out

    def has_topounits(self) -> bool:
        """Return True if this Domain has a non-empty topounits table attached."""
        
        return self.topounits is not None and len(self.topounits) > 0


    def topounits_for_gid(self, gid: str):
        """
        Return the topounits subset for a single gid (or None if no topounits).
        """
        if not self.has_topounits():
            return None
        gdf = self.topounits
        gid_col = self.topounits_gid_col
        if gid_col not in gdf.columns:
            # If no gid column, we can only support single-cell domains safely
            return gdf
        return gdf[gdf[gid_col].astype(str) == str(gid)].copy()


    def with_topounits(
        self,
        topounits: "gpd.GeoDataFrame",
        *,
        id_col: str = "band_name",
        gid_col: str = "gid",
        dim_name: str = "topounit",
    ) -> "Domain":
        """
        Attach topounits GeoDataFrame to this Domain.
        - Ensures a stable id column name (self.topounits_id_col == 'topounit_id')
        - Ensures gid linkage column exists (self.topounits_gid_col)
        """
        import geopandas as gpd

        if not isinstance(topounits, gpd.GeoDataFrame):
            raise TypeError("topounits must be a geopandas.GeoDataFrame")
        if topounits.empty:
            raise ValueError("topounits GeoDataFrame is empty")

        gdf = topounits.copy()
        if gid_col not in gdf.columns:
            raise KeyError(f"topounits is missing required gid column '{gid_col}'")
        gdf[gid_col] = gdf[gid_col].astype(str)

        if id_col not in gdf.columns:
            raise KeyError(f"topounits is missing required id column '{id_col}'")

        # normalize id column name -> 'topounit_id'
        if id_col != "topounit_id":
            if "topounit_id" in gdf.columns and id_col != "topounit_id":
                raise ValueError("topounits already has 'topounit_id' plus the provided id_col; ambiguous.")
            gdf = gdf.rename(columns={id_col: "topounit_id"})
        gdf["topounit_id"] = gdf["topounit_id"].astype(str)

        # (optional) ensure CRS exists; leave as-is if you already enforce this elsewhere
        # if gdf.crs is None: gdf = gdf.set_crs(epsg=4326)

        return self.copy(
            topounits=gdf,
            topounits_dim_name=dim_name,
            topounits_id_col="topounit_id",
            topounits_gid_col=gid_col,
        )

    def make_topounits(
        self,
        *,
        binning: dict,
        sources: list[str] | None = None,
        combine: str = "cartesian",
        combine_order=None,
        max_topounits: int = 256,
        dem_source: str = "arcticdem",
        export_scale: str = "native",
        min_patch_pixels=None,
        target_pixels_per_topounit: int = 500,
        target_scale: float | None = None,
        verbose: bool = False,
        allow_slow_ncells: int = 25,
    ) -> "Domain":
        """
        Convenience wrapper that computes topounits for this Domain and returns a new Domain
        with `domain.topounits` attached.

        - If this Domain has multiple rows (cellset/sites), it computes topounits per-row (per gid)
          using dapper.topounit.topomake.make_topounits_for_domain.
        - If this Domain has one row, it still goes through the same path (safe + consistent).

        Users should not need to deal with ee.Geometry vs ee.Feature vs FeatureCollection here.
        """
        # Local import avoids circular imports (Domain is foundational; topounit is optional/heavy).
        from dapper.topounit.topomake import make_topounits_for_domain

        if sources is None:
            # preserve insertion order of dict keys
            sources = list(binning.keys())

        try:
            make_topounits_for_domain(
            self,
            sources=sources,
            binning=binning,
            combine=combine,
            combine_order=combine_order,
            max_topounits=max_topounits,
            dem_source=dem_source,
            export_scale=export_scale,
            min_patch_pixels=min_patch_pixels,
            target_pixels_per_topounit=target_pixels_per_topounit,
            target_scale=target_scale,
            verbose=verbose,
            allow_slow_ncells=allow_slow_ncells,
        )
        except RuntimeError as e:
            print(e)
            return None

    # ----------------------------- optional layout helper -----------------------------

    def elm_latlon_layout(
        self,
        decimals: int = LATLON_DECIMALS,
        use_lon_0360: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
        """
        Compute lat/lon axes and a gid -> (iy, ix) index map for ELM-style lat/lon layouts.
        Useful mainly when your cells actually lie on a lat/lon lattice (dense or sparse).
        """
        dom = self.ensure_cells_lon_lat()
        gdf = dom.cells.copy()

        lats = np.round(gdf["lat"].to_numpy(dtype=float), decimals=decimals)
        lons = np.round(gdf["lon"].to_numpy(dtype=float), decimals=decimals)
        if use_lon_0360:
            lons = (lons % 360.0 + 360.0) % 360.0

        lats_axis = np.unique(lats)
        lons_axis = np.unique(lons)
        lats_axis.sort()
        lons_axis.sort()

        gid_to_ij: dict[str, tuple[int, int]] = {}
        for gid, lat, lon in zip(gdf["gid"].astype(str), lats, lons):
            iy = int(np.where(lats_axis == lat)[0][0])
            ix = int(np.where(lons_axis == lon)[0][0])
            gid_to_ij[str(gid)] = (iy, ix)

        return lats_axis, lons_axis, gid_to_ij

    # ----------------------------- ELM domain writer -----------------------------

    def _to_elm_domain_dataset(
        self,
        *,
        grid_shape: tuple[int, int] | None = None,
        cell_dx_deg: float = 0.5,
        cell_dy_deg: float | None = None,
        land_frac_col: str | None = "frac",
        mask_from_frac: bool = True,
        append_attrs: dict | None = None,
    ) -> xr.Dataset:
        """Build an ELM land domain Dataset from cells (delegates to dapper.domains.elm_domain)."""
        from dapper.domains.elm_domain import to_elm_domain_dataset

        return to_elm_domain_dataset(
            self,
            grid_shape=grid_shape,
            cell_dx_deg=cell_dx_deg,
            cell_dy_deg=cell_dy_deg,
            land_frac_col=land_frac_col,
            mask_from_frac=mask_from_frac,
            append_attrs=append_attrs,
        )

    def export_domain(
        self,
        *,
        filename: str = "domain.nc",
        out_dir: str | Path | None = None,
        overwrite: bool = False,
        append_attrs: dict | None = None,
        **kwargs,
    ) -> dict[str, Path]:
        """
        Export ELM domain NetCDF(s) for this Domain.

        Output layout:
          - mode='cellset': <run_dir>/domain.nc
          - mode='sites'  : <run_dir>/<gid>/domain.nc

        Returns:
          dict[run_id, output_path]
        """
        group_dir: Path | None = None
        if out_dir is not None:
            group_dir = Path(out_dir)
            group_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, Path] = {}
        for run_id, run_dom in self.iter_runs():
            rid = str(run_id)

            if group_dir is None:
                out_path = run_dom.path_domain_nc(filename=filename)
            else:
                run_out_dir = (group_dir / rid) if self.mode == "sites" else group_dir
                out_path = run_out_dir / filename

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists (overwrite=False).")

            ds = run_dom._to_elm_domain_dataset(append_attrs=append_attrs, **kwargs)
            ds.to_netcdf(out_path)
            outputs[rid] = out_path

        return outputs

    def export_surface(
        self,
        src_path: str | Path,
        *,
        filename: str = "surfdata.nc",
        out_dir: str | Path | None = None,
        overwrite: bool = False,
        append_attrs: dict | None = None,
        **kwargs,
    ) -> dict[str, Path]:
        """Export surface NetCDF(s) for this Domain."""
        from dapper.surf.sfile import SurfaceFile

        group_dir = Path(out_dir) if out_dir is not None else self.run_dir
        return SurfaceFile.export(
            self,
            out_dir=group_dir,
            src_path=src_path,
            filename=filename,
            overwrite=overwrite,
            append_attrs=append_attrs,
            **kwargs,
        )

    def export_landuse(
        self,
        src_path: str | Path,
        *,
        filename: str = "landuse_timeseries.nc",
        out_dir: str | Path | None = None,
        overwrite: bool = False,
        append_attrs: dict | None = None,
        **kwargs,
    ) -> dict[str, Path]:
        """Export landuse timeseries NetCDF(s) for this Domain."""
        from dapper.landuse.landuse import export_landuse_timeseries

        group_dir = Path(out_dir) if out_dir is not None else self.run_dir
        return export_landuse_timeseries(
            self,
            src_path=src_path,
            out_dir=group_dir,
            filename=filename,
            overwrite=overwrite,
            append_attrs=append_attrs,
            **kwargs,
        )

    def export_met(
        self,
        src_path: str | Path,
        *,
        adapter,
        out_dir: str | Path | None = None,
        filename: str | None = None,
        overwrite: bool = False,
        append_attrs: dict | None = None,
        pack_scope=None,
        clip_to_full_years: bool | None = None,
        **kwargs,
    ) -> dict[str, Path]:
        """
        Export meteorological forcing NetCDF(s) for this Domain.

        Parameters
        ----------
        src_path : Path-like
            Directory containing input CSV(s) for the adapter.
        adapter : object
            Adapter instance implementing the met adapter protocol.
        out_dir : Path-like, optional
            Override output root. Defaults to Domain.run_dir.
        filename : str, optional
            Optional filename prefix for output NetCDFs. If provided, each var is written to '{filename}_{var}.nc'.
        overwrite : bool
            If False, raises if MET output(s) already exist.
        clip_to_full_years : bool or None
            Controls whether the export year range is clipped to complete
            calendar years. None preserves the adapter default.

        Returns
        -------
        dict[run_id, met_dir]
        """
        from dapper.met.exporter import Exporter

        group_dir = Path(out_dir) if out_dir is not None else self.run_dir
        group_dir.mkdir(parents=True, exist_ok=True)

        # Conservative overwrite guard: refuse to run if outputs already exist
        if not overwrite:
            if self.mode == "cellset":
                met_dir = group_dir / "MET"
                if met_dir.exists() and any(met_dir.glob("*.nc")):
                    raise FileExistsError(
                        f"{met_dir} already contains NetCDF files (overwrite=False)."
                    )
            else:
                if any(group_dir.glob("*/MET/*.nc")):
                    raise FileExistsError(
                        f"{group_dir} already contains MET NetCDF outputs (overwrite=False)."
                    )


        ex = Exporter(
            adapter=adapter,
            src_path=src_path,
            out_dir=group_dir,
            domain=self,
            append_attrs=append_attrs,
            clip_to_full_years=clip_to_full_years,
            **kwargs,
        )
        ex.run(
            pack_scope=pack_scope,
            filename=filename,
            overwrite=overwrite,
        )

        outputs: dict[str, Path] = {}
        for run_id, _ in self.iter_runs():
            rid = str(run_id)
            outputs[rid] = (group_dir / rid / "MET") if self.mode == "sites" else (group_dir / "MET")
        return outputs

    def __str__(self) -> str:
        return f"{self.cells}"
