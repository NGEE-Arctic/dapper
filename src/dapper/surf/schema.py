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

from collections.abc import Iterable
from dataclasses import asdict, dataclass
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


def register_many(names: Iterable[str], v: ParDef) -> dict[str, ParDef]:
    """Register many variables with the same ParDef in one call."""
    return {name: v for name in names}


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

EXPORT_POLICIES = {
    # rules evaluated in order; first match wins
    "rules": [
        {
            "when": {"dims": ("time", "lsmlat", "lsmlon")},
            "policy": "MONTHLY_12_BANDS",
            "band_name": lambda var, sizes: [
                f"{var}_m{m:02d}" for m in range(1, sizes.get("time", 0) + 1)
            ],
            "note": "Export 12 monthly bands; optionally also annual_mean",
        },
        {
            "when": {"dims": ("nlevsoi", "lsmlat", "lsmlon")},
            "policy": "SOIL_TOP_LAYER_DEFAULT",
            "band_name": lambda var, sizes: [
                f"{var}_L{k:02d}" for k in range(sizes.get("nlevsoi", 0))
            ],
            "note": "Default L00; optionally L00/L05/L09 stack",
        },
        {
            "when": {"dims": ("natpft", "lsmlat", "lsmlon")},
            "policy": "PFT_ALL_BANDS",
            "band_name": lambda var, sizes: [
                f"{var}_pft{p:02d}" for p in range(sizes.get("natpft", 0))
            ],
            "note": "Export all PFT bands; optionally aggregated classes",
        },
        {
            "when": {"dims": ("nlevslp", "lsmlat", "lsmlon")},
            "policy": "SLOPE_REDUCE_DEFAULT",
            "band_name": lambda var, sizes: [
                f"{var}_slp{b:02d}" for b in range(sizes.get("nlevslp", 0))
            ],
            "note": "Default reduce (weighted mean); optionally keep all bins",
        },
        {
            "when": {"dims": ("lsmlat", "lsmlon")},
            "policy": "SINGLE_BAND",
            "band_name": lambda var, sizes: [var],
            "note": "Static 2D",
        },
    ],
    # fine-grained overrides if you need them (keep small)
    "overrides": {
        # "PCT_SAND": {"policy": "SOIL_MULTI_BANDS_3", "band_keep": ["L00","L05","L09"]},
        # "PCT_NAT_PFT": {"policy": "PFT_AGG_WGC", "band_keep": ["woody","grass","crop"]},
    },
}


# ---------------------- Runtime utilities --------------------------------


def expand_registry(as_json: bool = False) -> dict[str, dict]:
    """Return the full registry as plain dict (easy to dump/serialize)."""
    d = {k: asdict(v) for k, v in REGISTRY.items()}
    return d


def validate_against_schema(present_vars: Iterable[str]) -> dict[str, list[str]]:
    """
    Validate a set of variable names against SCHEMA rules.

    - Per-variable requirement is taken from ParDef.required_level
      (currently only 'required' is treated as hard-required).
    - 'choose_one_of' groups are enforced at the tier level.
    - 'conditional' rules are enforced as warnings when violated.
    """
    present = set(present_vars)
    errors: list[str] = []
    warnings: list[str] = []

    for tier, spec in SCHEMA.items():
        tier_vars = spec.get("vars", [])

        # required vars in this tier: those with required_level == "required"
        for vname in tier_vars:
            pdef_obj = REGISTRY.get(vname)
            if not pdef_obj:
                continue  # allow schema to reference vars not yet in registry
            if pdef_obj.required_level.lower() == "required" and vname not in present:
                errors.append(f"{tier}: missing required var '{vname}'")

        # choose_one_of groups
        for group in spec.get("choose_one_of", []):
            if isinstance(group, dict):
                group_vars = group.get("vars", [])
            else:
                group_vars = list(group)
            if group_vars and not (present & set(group_vars)):
                errors.append(f"{tier}: need one of {group_vars}")

        # conditional: driver presence implies dependent vars should also exist
        for cond in spec.get("conditional", []):
            driver = cond["if_var_present"]
            deps = cond["then_require"]
            if driver in present:
                for dep in deps:
                    if dep not in present:
                        warnings.append(
                            f"{tier}: '{dep}' is conditionally required because '{driver}' is present"
                        )

    return {"errors": errors, "warnings": warnings}


def propose_export_policy(
    var: str, sizes: dict[str, int], ParDef: ParDef | None = None
):
    """Return a compact policy dict for a variable, based on its dims and overrides."""
    # override first
    ov = EXPORT_POLICIES["overrides"].get(var)
    if ov:
        return {
            "var": var,
            "policy": ov["policy"],
            "note": "override",
            "bands": ov.get("band_keep", []),
        }

    dims = tuple(ParDef.dims if ParDef else ())
    for rule in EXPORT_POLICIES["rules"]:
        if dims == tuple(rule["when"]["dims"]):
            bands = (
                rule["band_name"](var, sizes) if callable(rule["band_name"]) else [var]
            )
            return {
                "var": var,
                "policy": rule["policy"],
                "note": rule.get("note", ""),
                "bands": bands,
            }

    # fallback
    return {
        "var": var,
        "policy": "UNKNOWN",
        "note": "no rule matched",
        "bands": [var],
    }
