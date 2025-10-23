# topounits/plotting.py
# Helpers for pretty legends + plotting topounits (GeoDataFrames) on basemaps.

from __future__ import annotations
import geopandas as gpd

# ---------- label helpers ----------

_PRETTY_SOURCE = {
    "elev": "Elev",
    "hand": "HAND",
    "aspect": "Aspect",
    "cti": "CTI",    
}

_UNITS = {
    "elev": "m",
    "hand": "m",
    "aspect": "°",
}

def _fmt_num(x, nd=0):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _numeric_label(bounds: dict, sid: str, nd=0) -> str | None:
    if not isinstance(bounds, dict):
        return None
    lo, hi = bounds.get("low"), bounds.get("high")
    if lo is None or hi is None:
        return None
    return f"{_PRETTY_SOURCE.get(sid, sid)} {_fmt_num(lo, nd)}–{_fmt_num(hi, nd)} {_UNITS.get(sid, '')}"

def _aspect_label(labels_dict: dict | None, bounds: dict | None) -> str:
    # Prefer human label from `labels` (e.g., "ASP_N") → "Aspect N"
    if isinstance(labels_dict, dict):
        lab = labels_dict.get("aspect")
        if isinstance(lab, str) and "_" in lab:
            human = lab.split("_", 1)[1]
            return f"{_PRETTY_SOURCE['aspect']} {human}"
    # Fallback to degree range
    if isinstance(bounds, dict):
        start, end = bounds.get("start"), bounds.get("end")
        if start is not None and end is not None:
            return f"{_PRETTY_SOURCE['aspect']} {_fmt_num(start)}–{_fmt_num(end)}{_UNITS['aspect']}"
    return _PRETTY_SOURCE["aspect"]

def _compose_legend_label(row, order=None, nd=0) -> str:
    """Build a combined legend label like: 'Elev 0–100 m | HAND 0–2 m | Aspect N'."""
    sids = order or (row.get("source_ids") or [])
    labels = row.get("labels") or {}
    bounds = row.get("bin_bounds") or {}

    parts = []
    for sid in sids:
        if sid in ("elev", "hand"):
            p = _numeric_label(bounds.get(sid), sid, nd=nd)
            if p: parts.append(p)
        elif sid == "aspect":
            parts.append(_aspect_label(labels, bounds.get("aspect")))
        else:
            # Unknown source: try generic numeric
            p = _numeric_label(bounds.get(sid, {}), sid, nd=nd)
            if p: parts.append(p)
    return " | ".join(parts) if parts else (row.get("band_name") or "topounit")


def _get_ctx_provider(name: str):
    """
    Map friendly basemap name -> contextily provider, lazily resolved.
    Avoids deprecated Stamen tiles.
    """
    import contextily as ctx
    # dotted provider names we know are widely available
    prov_map = {
        "positron": "CartoDB.Positron",
        "dark":     "CartoDB.DarkMatter",
        "osm":      "OpenStreetMap.Mapnik",
        "terrain":  "OpenTopoMap",           # <- Stamen retired; use OpenTopoMap
        "satellite":"Esri.WorldImagery",
    }
    key = prov_map.get(name, "CartoDB.Positron")
    prov = ctx.providers
    for part in key.split("."):
        prov = getattr(prov, part)
    return prov

# ---------- public helpers ----------

def prepare_for_plot(
    gdf: gpd.GeoDataFrame,
    area_epsg: int = 3857,
    area_col: str = "area_km2",
    ndigits_numeric: int = 0,
    order: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Returns a copy with:
      - legend_label: pretty combined label for legend/coloring
      - area_km2: polygon area (in km²) computed in a projected CRS (default EPSG:3857)
    """
    df = gdf.copy()
    # legend_label (uses source_ids order unless 'order' provided)
    df["legend_label"] = df.apply(lambda r: _compose_legend_label(r, order=order, nd=ndigits_numeric), axis=1)

    # area (project to meters for area calc)
    df_proj = df.to_crs(area_epsg)
    df[area_col] = df_proj.geometry.area / 1e6
    return df

def plot_static(
    gdf: gpd.GeoDataFrame,
    basemap: str = "positron",
    figsize=(10, 10),
    alpha: float = 0.6,
    edgecolor: str = "black",
    linewidth: float = 0.7,
    legend: bool = True,
    cmap=None,
    ax=None,                      # <-- NEW: allow injecting an axes
):
    """
    Static plot with contextily basemap.
    basemap options: 'positron', 'dark', 'osm', 'terrain', 'satellite'
    If `ax` is provided, draw into that axes (and do not create a new figure).
    """
    try:
        import matplotlib.pyplot as plt
        import contextily as ctx
    except ImportError as e:
        raise ImportError("plot_static requires `matplotlib`, `contextily`, and `xyzservices` installed.") from e

    provider = _get_ctx_provider(basemap)

    # reproject to Web Mercator for tiles
    gdf_3857 = gdf.to_crs(3857)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    gdf_3857.plot(
        ax=ax,
        column="legend_label",
        legend=legend,
        linewidth=linewidth,
        edgecolor=edgecolor,
        alpha=alpha,
        cmap=cmap,
    )
    ctx.add_basemap(ax, source=provider)
    ax.set_axis_off()
    ax.margins(0)

    return (fig, ax) if created_fig else (ax.figure, ax)

def plot_interactive(
    gdf: gpd.GeoDataFrame,
    tiles: str = "CartoDB positron",
    tooltip_cols=("legend_label", "area_km2", "band_name", "analysis_scale_m"),
    style_kwds=None,
):
    """
    Interactive Folium map. Returns a folium.Map.
    """
    try:
        import folium  # noqa: F401
    except ImportError as e:
        raise ImportError("plot_interactive requires `folium` installed.") from e

    style_kwds = style_kwds or {"fillOpacity": 0.5, "weight": 1}
    # geopandas.explore handles reprojection automatically
    return gdf.explore(
        column="legend_label",
        tooltip=list(tooltip_cols),
        style_kwds=style_kwds,
        tiles=tiles,
        name="Topounits",
    )
