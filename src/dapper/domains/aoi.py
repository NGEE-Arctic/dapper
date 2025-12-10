# dapper/aoi.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Union, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import base as shapely_base
from shapely.geometry import Point
from pathlib import Path

from dapper.domains.domain import Domain

GeometryLike = Union[shapely_base.BaseGeometry, Point]

@dataclass
class AOI:
    """
    AOI = Area/Location of Interest.

    This is the "geometry world": arbitrary points or polygons used for:
      - defining where to sample met data (GEE),
      - defining where to build topounits.

    It is NOT directly the ELM grid; instead, you derive a cell-based
    Domain from an AOI when you want to run ELM.
    """


    name: str
    gdf: gpd.GeoDataFrame

    @classmethod
    def from_geometries(
        cls,
        geoms: Iterable[GeometryLike],
        *,
        name: str = "aoi",
        crs: str = "EPSG:4326",
        ids: Optional[Iterable[str]] = None,
    ) -> "AOI":
        geoms = list(geoms)
        if ids is None:
            ids = [f"aoi_{i:04d}" for i in range(len(geoms))]
        df = pd.DataFrame({"gid": list(ids)})
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs)
        return cls(name=name, gdf=gdf)
    
    @classmethod
    def from_point(
        cls,
        lon: float,
        lat: float,
        *,
        name: str = "aoi_point",
        gid: str = "site_0000",
        crs: str = "EPSG:4326",
    ) -> "AOI":
        """
        Convenience constructor for a single-point AOI.
        """
        gdf = gpd.GeoDataFrame(
            {"gid": [gid]},
            geometry=[Point(lon, lat)],
            crs=crs,
        )
        return cls(name=name, gdf=gdf)

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame, *, name: str = "aoi", id_col: str = "gid") -> "AOI":
        gdf = gdf.copy()
        if id_col != "gid":
            if id_col not in gdf.columns:
                raise KeyError(f"AOI.from_gdf: id_col '{id_col}' not in GeoDataFrame.")
            gdf = gdf.rename(columns={id_col: "gid"})
        gdf["gid"] = gdf["gid"].astype(str).str.strip()
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        return cls(name=name, gdf=gdf)

    @property
    def union_geometry(self) -> GeometryLike:
        """Single combined geometry (for passing into GEE helpers)."""
        return self.gdf.unary_union

    def to_geometries_gdf(self) -> gpd.GeoDataFrame:
        """
        GeoDataFrame with ['gid','geometry']; good for:
          - sample_e5lh(..., geometries=aoi.to_geometries_gdf(), ...)
          - make_topounits(feature=aoi.union_geometry, ...)
        """
        return self.gdf[["gid", "geometry"]].copy()

    # ---- AOI → Domain (simple centroid-based sampling) ----

    def to_domain_points(
        self,
        *,
        name: Optional[str] = None,
    ) -> Domain:
        """
        Map AOI shapes to a point-based Domain by taking representative
        points (centroids) of each geometry.

        Each AOI polygon becomes a single Domain cell at its centroid.
        This is the simplest and most robust way to go from AOI (polygon)
        to Domain (cells) without committing to a full grid.
        """
        gdf = self.gdf.copy()

        # representative point for any geometry type
        def rep_point(geom):
            if geom.geom_type == "Point":
                return geom
            return geom.representative_point()

        pts = gdf.geometry.apply(rep_point)
        gdf["lon"] = pts.x
        gdf["lat"] = pts.y
        gdf["geometry"] = pts

        return Domain.from_gdf(gdf, name=name or self.name)
