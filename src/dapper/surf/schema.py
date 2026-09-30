"""ELM surface-file schema: variable registry and presence rules.

This module does not read NetCDF. It encodes what a surface file should look
like so other modules can build, write, and validate files consistently.

- ``ParDef``: schema record for one variable (dims, dtype, units, doc, attrs).
- ``REGISTRY``: ``dict[str, ParDef]`` built from
  :data:`dapper.surf.surface_var_specs.SURFACE_VAR_SPECS`.
- ``SCHEMA``: tiered presence rules. Per-variable requirement lives in
  ``ParDef.required_level``; ``choose_one_of`` groups need at least one member;
  ``conditional`` rules require dependents when a driver variable is present.

Conventions: spatial dims use ELM naming and come last (``..., lsmlat,
lsmlon``). Units of ``''`` or ``'varies'`` are not enforced by the validator.
Specs carry no dtype, so every ``ParDef.dtype`` is the ``"float32"`` default.

Used by :mod:`dapper.surf.sfile` (customization and topounit parameters) and
:mod:`dapper.surf.validate`. To add a variable, add it to
``SURFACE_VAR_SPECS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS


@dataclass(frozen=True)
class ParDef:
    """
    Schema record for one surface parameter.

    dims:
        Tuple of dimension names in model order (non-spatial first,
        then spatial, e.g. ("natpft", "lsmlat", "lsmlon")).
    dtype:
        NetCDF dtype as a string ("float32", "int16", etc.).
    units:
        Unit string; empty means “not enforced”.
    doc:
        Human-readable description, suitable for docs.
    required_level:
        Semantic requirement flag, e.g. "required", "optional",
        "recommended", "conditional". The validator treats "required"
        as hard-required.
    attrs:
        Extra NetCDF attributes (long_name, standard_name, etc.).
    """

    dims: tuple[str, ...]
    dtype: str = "float32"
    units: str = ""
    doc: str = ""
    required_level: str = ""
    attrs: dict[str, Any] | None = None
    contexts: tuple[str, ...] = ()


def pdef(
    dims,
    dtype: str = "float32",
    units: str = "",
    doc: str = "",
    required_level: str = "",
    contexts: tuple[str, ...] = (),
    **attrs,
) -> ParDef:
    """
    Convenience constructor for ParDef.

    dims can be a comma-separated string ("lsmlat,lsmlon") or an
    iterable of dim names. Any extra keyword args become NetCDF
    variable attributes (e.g., long_name="...").
    """
    if isinstance(dims, str):
        dims_tuple = tuple(d.strip() for d in dims.split(",") if d.strip())
    else:
        dims_tuple = tuple(dims)
    return ParDef(
        dims=dims_tuple,
        dtype=str(dtype),
        units=units,
        doc=doc,
        required_level=required_level,
        attrs=attrs or {},
        contexts=tuple(contexts or ()),
    )


# Common dim-sets (reusable)
DIMS_2D = "lsmlat,lsmlon"
DIMS_TIME2D = "time,lsmlat,lsmlon"
DIMS_SOIL = "nlevsoi,lsmlat,lsmlon"
DIMS_PFT = "natpft,lsmlat,lsmlon"
DIMS_SLOPE = "nlevslp,lsmlat,lsmlon"

# ---------- Variable Registry ------------------------------------------
# Built from SURFACE_VAR_SPECS, the single source of truth for surface variables.

REGISTRY: dict[str, ParDef] = {}

for name, spec in SURFACE_VAR_SPECS.items():
    attrs = spec.get("attrs", {})
    contexts = tuple(spec.get("contexts", []) or [])
    REGISTRY[name] = pdef(
        spec["dims"],
        units=spec.get("units", ""),
        doc=spec.get("doc", ""),
        required_level=spec.get("required_level", ""),
        contexts=contexts,
        **attrs,
    )


# ------------- Presence rules ------------------------------------------
# Logical tiers plus cross-variable rules. Per-variable requirement lives in
# ParDef.required_level.

SCHEMA: dict[str, dict] = {
    "TIER0_CORE_COORD_MASK": {
        # Core spatial metadata & land mask
        "vars": ["LATIXY", "LONGXY", "AREA", "LANDFRAC_PFT", "PFTDATA_MASK"],
    },
    "TIER1_LANDCOVER": {
        # choose one of these as source-of-truth for nat veg extent
        "vars": ["PCT_NATVEG", "PCT_NAT_PFT", "PCT_CROP"],
        "choose_one_of": [["PCT_NATVEG", "PCT_NAT_PFT"]],
    },
    "TIER2_SOIL": {
        "vars": ["PCT_SAND", "PCT_CLAY", "ORGANIC", "PCT_GRVL"],
    },
    "TIER3_TOPO": {
        "vars": [
            "SLOPE",
            "STDEV_ELEV",
            "STD_ELEV",
            "TOPO",
            "TERRAIN_CONFIG",
            "SKY_VIEW",
        ],
    },
    "TIER4_WATER_ICE_URBAN": {
        "vars": [
            "PCT_WETLAND",
            "PCT_LAKE",
            "PCT_GLACIER",
            "PCT_URBAN",
            "GLC_MEC",
            "PCT_GLC_MEC",
            "URBAN_REGION_ID",
        ],
        # conditional groups: evaluated by validator
        "conditional": [
            {"if_var_present": "PCT_URBAN", "then_require": ["URBAN_REGION_ID"]},
            {
                "if_var_present": "PCT_GLACIER",
                "then_require": ["GLC_MEC", "PCT_GLC_MEC"],
            },
        ],
    },
    "TIER5_CANOPY_MONTHLY": {
        "vars": [
            "MONTHLY_LAI",
            "MONTHLY_SAI",
            "MONTHLY_HEIGHT_TOP",
            "MONTHLY_HEIGHT_BOT",
        ],
    },
    "TIER6_BGC_P": {
        "vars": ["APATITE_P", "LABILE_P", "OCCLUDED_P", "SECONDARY_P"],
    },
}

# ------------- Export Policies (rule-based, dimension-aware) ------------


# ---------------------- Runtime utilities --------------------------------
