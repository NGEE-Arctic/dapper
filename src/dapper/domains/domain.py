# dapper/domain.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, replace

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

LATLON_DECIMALS = 6  # keep in sync with exporter.LATLON_DECIMALS

@dataclass
class Domain:
    """
    Canonical representation of the spatial domain used for met/topounits/ELM.

    Attributes
    ----------
    name : str
        Human-readable name (e.g., "Abisko_1x1pt").
    gdf : geopandas.GeoDataFrame
        One row per site/cell with at least:
          - 'gid'      : unique string identifier
          - 'geometry' : shapely geometry (Point or Polygon)
        Usually also:
          - 'lon', 'lat': coordinates in EPSG:4326.
    domain_nc : Path or None
        Optional pointer to an ELM domain NetCDF that this Domain came from
        or is intended to match.
    """

    name: str
    gdf: gpd.GeoDataFrame
    domain_nc: Optional[Path] = None

    # ----------------------------- constructors -----------------------------

    @classmethod
    def from_gdf(
        cls,
        gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
        *,
        name: str = "domain",
        domain_nc: Optional[Union[str, Path]] = None,
        id_col: str = "gid",
    ) -> "Domain":
        """
        Build a Domain from an existing (geo)DataFrame.

        - Ensures a geometry column exists (creating Points from lon/lat if needed).
        - Ensures a 'gid' column exists, coercing from `id_col` if provided.
        - Ensures 'lon'/'lat' columns exist (derived from geometry if needed).
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            if "geometry" in gdf.columns:
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", copy=True)
            elif {"lon", "lat"}.issubset(gdf.columns):
                gdf = gpd.GeoDataFrame(
                    gdf,
                    geometry=gpd.points_from_xy(gdf["lon"], gdf["lat"]),
                    copy=True,
                    crs="EPSG:4326",
                )
            else:
                raise ValueError(
                    "Domain.from_gdf requires either a 'geometry' column or both 'lon' and 'lat'."
                )
        else:
            gdf = gdf.copy()

        if gdf.crs is None:
            # assume WGS84 if not set; caller can override
            gdf.set_crs(epsg=4326, inplace=True)

        # Ensure gid
        if "gid" not in gdf.columns:
            if id_col in gdf.columns:
                gdf = gdf.rename(columns={id_col: "gid"})
            else:
                gdf["gid"] = np.arange(len(gdf), dtype=int)
        gdf["gid"] = gdf["gid"].astype(str).str.strip()

        # Ensure lon/lat
        if not {"lon", "lat"}.issubset(gdf.columns):

            def rep_point(geom: BaseGeometry) -> tuple[float, float]:
                if geom.geom_type == "Point":
                    p = geom
                else:
                    p = geom.representative_point()
                return float(p.x), float(p.y)

            lons, lats = zip(*gdf.geometry.apply(rep_point))
            gdf["lon"] = np.asarray(lons, dtype=float)
            gdf["lat"] = np.asarray(lats, dtype=float)

        return cls(
            name=name,
            gdf=gdf,
            domain_nc=Path(domain_nc) if domain_nc is not None else None,
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
    ) -> "Domain":
        """
        Build a Domain from an E3SM/ELM domain NetCDF, e.g.:

          - domain.lnd.1x1pt_Abisko-GRID.nc
          - domain.lnd.1x1pt_ImnaviatCreek-GRID.nc

        Assumes:
          - lon/lat centers: variables 'xc', 'yc' with dims (nj, ni)
          - lon/lat corners: variables 'xv', 'yv' with dims (nj, ni, nv)
          - optional 'mask' and 'frac' to filter inactive land cells.
        """
        path_nc = Path(path_nc)
        ds = xr.open_dataset(path_nc)

        xc = ds["xc"].values  # (nj, ni)
        yc = ds["yc"].values
        xv = ds["xv"].values  # (nj, ni, nv)
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

                gid = f"col_{gid_counter:04d}"
                gid_counter += 1

                rows.append(
                    {
                        "gid": gid,
                        "lon": lon_c,
                        "lat": lat_c,
                        "geometry": poly,
                        "i": i,
                        "j": j,
                    }
                )

        if not rows:
            raise ValueError(f"No active cells found in domain file {path_nc}")

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return cls(
            name=name or path_nc.stem,
            gdf=gdf,
            domain_nc=path_nc,
        )

    # ----------------------------- helpers -----------------------------

    def copy(self, **updates) -> "Domain":
        """Shallow copy with optional field overrides."""
        return replace(self, **updates)

    @property
    def df(self) -> pd.DataFrame:
        """Plain DataFrame view (drops geometry)."""
        return pd.DataFrame(self.gdf.drop(columns=["geometry"]))

    def to_geometries(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame with at least ['gid','geometry']."""
        return self.gdf[["gid", "geometry"]].copy()

    @property
    def gids(self) -> list[str]:
        return self.gdf["gid"].astype(str).tolist()

    def with_gdf(self, gdf: gpd.GeoDataFrame) -> "Domain":
        """Return a new Domain with a different GeoDataFrame (same name/domain_nc)."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
        return Domain(name=self.name, gdf=gdf, domain_nc=self.domain_nc)

    def ensure_lon_lat(self) -> "Domain":
        """
        Ensure 'lon' and 'lat' exist; if missing, derive them from geometry.
        Returns a new Domain instance.
        """
        gdf = self.gdf.copy()
        if {"lon", "lat"}.issubset(gdf.columns):
            return self

        def rep_point(geom: BaseGeometry) -> tuple[float, float]:
            if geom.geom_type == "Point":
                p = geom
            else:
                p = geom.representative_point()
            return float(p.x), float(p.y)

        lons, lats = zip(*gdf.geometry.apply(rep_point))
        gdf["lon"] = np.asarray(lons, dtype=float)
        gdf["lat"] = np.asarray(lats, dtype=float)
        return self.with_gdf(gdf)

    def __len__(self) -> int:
        return len(self.gdf)
    
    def elm_latlon_layout(
        self,
        decimals: int = LATLON_DECIMALS,
        use_lon_0360: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
        """
        Compute lat/lon axes and a gid -> (iy, ix) index map for ELM-style grids.

        Returns
        -------
        lats_axis : np.ndarray
            Sorted unique latitudes (rounded to `decimals`).
        lons_axis : np.ndarray
            Sorted unique longitudes (0–360 if use_lon_0360=True).
        gid_to_ij : dict
            Mapping gid -> (iy, ix) indices into (lat, lon) axes.
        """
        dom = self.ensure_lon_lat()
        gdf = dom.gdf.copy()

        # choose lon column
        if use_lon_0360 and "lon_0-360" in gdf.columns:
            lon_vals = gdf["lon_0-360"].to_numpy(dtype="float64")
        else:
            lon_vals = gdf["lon"].to_numpy(dtype="float64")
            if use_lon_0360:
                lon_vals = np.mod(lon_vals, 360.0)

        lat_vals = gdf["lat"].to_numpy(dtype="float64")

        # axes
        lats_axis = np.unique(np.round(lat_vals, decimals=decimals))
        lats_axis.sort()

        lons_axis = np.unique(np.round(lon_vals, decimals=decimals))
        lons_axis.sort()

        # lookup tables for axis indices
        lat_key = {round(float(v), decimals): i for i, v in enumerate(lats_axis)}
        lon_key = {round(float(v), decimals): j for j, v in enumerate(lons_axis)}

        gid_to_ij: dict[str, tuple[int, int]] = {}
        for row in gdf.itertuples():
            gid = str(row.gid)
            lat_r = round(float(row.lat), decimals)

            # df.itertuples() turns "lon_0-360" into attribute "lon_0_360"
            if hasattr(row, "lon_0_360"):
                lon_val = float(row.lon_0_360)
            else:
                lon_val = float(row.lon)
            if use_lon_0360:
                lon_val = lon_val % 360.0
            lon_r = round(lon_val, decimals)

            iy = lat_key[lat_r]
            ix = lon_key[lon_r]
            gid_to_ij[gid] = (iy, ix)

        return lats_axis, lons_axis, gid_to_ij


    def _to_elm_domain_dataset(
        self,
        *,
        grid_shape: tuple[int, int] | None = None,
        cell_dx_deg: float = 0.5,
        cell_dy_deg: float | None = None,
        land_frac_col: str | None = None,
        mask_from_frac: bool = True,
        global_attrs: dict | None = None,
    ) -> xr.Dataset:
        """
        Build an xarray.Dataset representing an ELM land domain file
        (domain.lnd.*.nc-style) from this Domain.

        Parameters
        ----------
        grid_shape : (nj, ni) or None
            Layout of the grid in index space.
            - If None (default): use a "column" layout with (nj = n_cells, ni = 1).
            - If provided: cells are laid out row-major:
                idx -> (j = idx // ni, i = idx % ni)
            so you control 1D vs 2D by choosing (nj, ni).
        cell_dx_deg, cell_dy_deg : float
            Size of each cell in degrees when only a point geometry is available.
            For polygon geometries, the cell bounds are taken from the polygon
            bounding box instead.
        land_frac_col : str or None
            If not None and present in ``self.gdf``, use this column as the land
            fraction (0–1) for each cell. Otherwise use 1.0.
        mask_from_frac : bool
            If True, set mask=1 where frac>0, else 0. If False, mask=1 everywhere.
        global_attrs : dict or None
            Extra global attributes to merge into the Dataset.

        Notes
        -----
        - area is computed in spherical radians^2 based on the lon/lat bounds.
        - The grid index layout (ni, nj) is mostly bookkeeping; ELM really keys
        off xc/yc/xv/yv + mask/frac.
        """

        # Ensure lon/lat present and copy to avoid mutating self.gdf
        dom = self.ensure_lon_lat()
        gdf = dom.gdf.copy()
        ncell = len(gdf)
        if ncell == 0:
            raise ValueError("Domain has no cells; cannot build ELM domain.")

        # Grid shape: default 1D "column"; user can ask for 2D layout explicitly
        if grid_shape is None:
            nj, ni = ncell, 1
        else:
            nj, ni = grid_shape
            if nj * ni != ncell:
                raise ValueError(
                    f"grid_shape {grid_shape} has nj*ni={nj*ni}, "
                    f"but Domain has {ncell} cells."
                )

        if cell_dy_deg is None:
            cell_dy_deg = cell_dx_deg

        nv = 4  # 4 vertices
        area = np.zeros((nj, ni), dtype="f8")
        frac = np.zeros_like(area)
        mask = np.zeros((nj, ni), dtype="i4")
        xc = np.zeros_like(area)
        yc = np.zeros_like(area)
        xv = np.zeros((nj, ni, nv), dtype="f8")
        yv = np.zeros_like(xv)

        # Land fraction per cell
        if land_frac_col is not None and land_frac_col in gdf.columns:
            frac_vals = gdf[land_frac_col].to_numpy(dtype="f8")
        else:
            frac_vals = np.ones(ncell, dtype="f8")

        for idx, row in gdf.reset_index(drop=True).iterrows():
            j = idx // ni
            i = idx % ni

            lon_c = float(row["lon"])
            lat_c = float(row["lat"])
            geom = row.get("geometry", None)

            # Derive cell corners
            if geom is not None and hasattr(geom, "geom_type") and geom.geom_type in (
                "Polygon",
                "MultiPolygon",
            ):
                minx, miny, maxx, maxy = geom.bounds
            else:
                half_dx = cell_dx_deg / 2.0
                half_dy = cell_dy_deg / 2.0
                minx, maxx = lon_c - half_dx, lon_c + half_dx
                miny, maxy = lat_c - half_dy, lat_c + half_dy

            # Corner order: [ll, lr, ul, ur]
            x_corners = [minx, maxx, minx, maxx]
            y_corners = [miny, miny, maxy, maxy]
            xv[j, i, :] = x_corners
            yv[j, i, :] = y_corners

            # Centers
            xc[j, i] = lon_c
            yc[j, i] = lat_c

            # Spherical area in radians^2 for lat/lon rectangle
            lam1 = math.radians(minx)
            lam2 = math.radians(maxx)
            phi1 = math.radians(miny)
            phi2 = math.radians(maxy)
            area_sr = (lam2 - lam1) * (math.sin(phi2) - math.sin(phi1))
            if area_sr < 0:
                area_sr = -area_sr
            area[j, i] = area_sr

            # frac & mask
            fval = float(frac_vals[idx])
            frac[j, i] = fval
            mask[j, i] = 1 if (not mask_from_frac) or fval > 0 else 0

        # Build Dataset
        ds = xr.Dataset(
            coords={
                "nj": np.arange(nj),
                "ni": np.arange(ni),
                "nv": np.arange(nv),
            },
            data_vars={
                "area": (("nj", "ni"), area),
                "frac": (("nj", "ni"), frac),
                "mask": (("nj", "ni"), mask),
                "xc": (("nj", "ni"), xc),
                "yc": (("nj", "ni"), yc),
                "xv": (("nj", "ni", "nv"), xv),
                "yv": (("nj", "ni", "nv"), yv),
            },
        )

        # Variable attributes (same as before)
        ds["area"].attrs.update(
            {
                "long_name": "area of grid cell in radians squared",
                "coordinate": "xc yc",
                "units": "radians2",
            }
        )
        ds["frac"].attrs.update(
            {
                "long_name": "fraction of grid cell that is active",
                "coordinate": "xc yc",
                "units": "unitless",
                "filter1": "error if frac> 1.0+eps or frac < 0.0-eps; eps = 0.1000000E-11",
                "filter2": "limit frac to [fminval,fmaxval]; fminval= 0.1000000E-02 fmaxval=  1.000000",
            }
        )
        ds["mask"].attrs.update(
            {
                "long_name": "land domain mask",
                "coordinate": "xc yc",
                "note": "unitless",
                "comment": "0=ocean and 1=land, 0 indicates that cell is not active",
            }
        )
        ds["xc"].attrs.update(
            {
                "long_name": "longitude of grid cell center",
                "units": "degrees_east",
                "bounds": "xv",
            }
        )
        ds["xv"].attrs.update(
            {
                "long_name": "longitude of grid cell vertices",
                "units": "degrees_east",
            }
        )
        ds["yc"].attrs.update(
            {
                "long_name": "latitude of grid cell center",
                "units": "degrees_north",
                "bounds": "yv",
            }
        )
        ds["yv"].attrs.update(
            {
                "long_name": "latitude of grid cell vertices",
                "units": "degrees_north",
            }
        )

        attrs_default = {
            "Conventions": "NCAR-CSM:CF-1.0",
            "title": "ELM domain data: generated by dapper",
            "user_comment": f"Domain generated from dapper.Domain(name='{self.name}')",
        }
        if global_attrs:
            attrs_default.update(global_attrs)
        ds.attrs.update(attrs_default)

        return ds

    def write_elm_domain(
        self,
        path: str | Path,
        **kwargs,
    ) -> Path:
        """
        Write this Domain to an ELM land domain NetCDF file.

        Parameters
        ----------
        path : str or Path
            Output NetCDF path, e.g. "domain.lnd.1x1pt_MYSITE-GRID.nc".

        **kwargs :
            Passed through to ``_to_elm_domain_dataset``, e.g.:

            - grid_shape=(nj, ni)
            - cell_dx_deg=0.5
            - cell_dy_deg=0.5
            - land_frac_col="frac"
            - global_attrs={...}

        Returns
        -------
        Path
            The resolved output path.
        """
        out_path = Path(path)
        ds = self._to_elm_domain_dataset(**kwargs)
        ds.to_netcdf(out_path)
        return out_path
