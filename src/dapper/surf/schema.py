# elm_surface_registry.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable


"""
dapper.surf.schema
==================

Purpose
-------
Single source of truth for ELM/CLM *surface-file* structure used by Dapper.
This module **does not read any NetCDF**. Instead, it hard-codes what a
surface file *should* look like so other modules can build, write, and
validate files consistently.

What this module defines
------------------------
- VarDef: a compact schema record for one variable (dims, dtype, units, attrs).
- REGISTRY: dict[str, VarDef]
    Canonical list of surface variables with their expected dimension
    signatures and basic metadata. Think of this as the variable “spec.”
- SCHEMA: dict[str, Any]
    Tiered rules for presence/formatting:
      * "required": must exist
      * "optional": may exist
      * "choose_one_of": at least one of the group must exist
      * "conditional": if driver var is present (or nonzero in practice),
        then dependent vars must also be present

Conventions
-----------
- Spatial dims use ELM naming and appear **last** in variables: (..., lsmlat, lsmlon).
- Common non-spatial dims (typical defaults):
    time=12, nlevsoi=10, natpft=17, nlevslp=11, numurbl=3, numrad=2, nlevurb=5.
  (These are expectations for formatting/validation; datasets may omit some dims.)
- Units are simple strings ('' or 'varies' means “not enforced by validator”).
- Dtypes: use integer types for IDs/indices (e.g., URBAN_REGION_ID, GLC_MEC),
  floats for fractions/percents/continuous fields.

How other modules use this
--------------------------
- dapper.surf.sample: point/polygon sampling uses REGISTRY dims to shape outputs.
- dapper.surf.write: formats sampled arrays into an ELM-style NetCDF using REGISTRY.
- dapper.surf.validate: checks a produced NetCDF against REGISTRY/SCHEMA (presence,
  dim order/lengths, dtype/units, and conditional relationships).

Extending / editing
-------------------
- To add a new variable, add a VarDef in REGISTRY with its **full dim tuple**
  (including spatial dims if it is spatial), its dtype, and units.
- Prefer grouped/compact registration helpers (if present) over one-var-per-line.
- When changing dims or semantics for an existing variable, update REGISTRY here
  first—writers and validators derive expectations from this module.
- Add presence rules (required/optional/choose_one_of/conditional) in SCHEMA to
  influence validation behavior without touching code elsewhere.

Scope
-----
This module captures **formatting/structure** (dims, units, dtype, presence rules).
Numeric ranges, aggregation choices, and scientific provenance live in sampling
and validation layers, not here.

"""





# --------- Compact helpers so you don't write one-var-per-line ----------

@dataclass(frozen=True)
class VarDef:
    dims: Tuple[str, ...]
    dtype: str = "float32"
    units: str = ""
    attrs: Dict[str, str] = None

def vdef(dims, dtype="float32", units="", **attrs) -> VarDef:
    return VarDef(tuple(d.strip() for d in dims.split(",")), dtype, units, attrs or {})

def register_many(names: Iterable[str], v: VarDef) -> Dict[str, VarDef]:
    """Register many variables with same VarDef in one call."""
    return {name: v for name in names}

# Common dim-sets (reusable)
DIMS_2D     = "lsmlat,lsmlon"
DIMS_TIME2D = "time,lsmlat,lsmlon"
DIMS_SOIL   = "nlevsoi,lsmlat,lsmlon"
DIMS_PFT    = "natpft,lsmlat,lsmlon"
DIMS_SLOPE  = "nlevslp,lsmlat,lsmlon"

# ---------- Variable Registry (compact, grouped) ------------------------

# Core coords/mask/area
REGISTRY: Dict[str, VarDef] = {
    **register_many(["LATIXY","LONGXY"], vdef(DIMS_2D, dtype="float32", units="degrees_north/degrees_east")),
    "AREA":         vdef(DIMS_2D, units="m2"),
    "LANDFRAC_PFT": vdef(DIMS_2D, units="1"),
    "PFTDATA_MASK": vdef(DIMS_2D, dtype="int16", units="1"),
}

# Land cover
REGISTRY.update({
    "PCT_NATVEG":   vdef(DIMS_2D, units="%"),
    "PCT_CROP":     vdef(DIMS_2D, units="%"),
})
REGISTRY.update(register_many(["PCT_NAT_PFT"], vdef(DIMS_PFT, units="%")))

# Soil (layered)
REGISTRY.update(register_many(["PCT_SAND","PCT_CLAY","ORGANIC","PCT_GRVL"], vdef(DIMS_SOIL, units="%")))

# Topography / terrain
REGISTRY.update({
    "SLOPE":          vdef(DIMS_2D, units="degrees"),
    "STDEV_ELEV":     vdef(DIMS_2D, units="m"),
    "STD_ELEV":       vdef(DIMS_2D, units="m"),   # alias seen in some files
    "TOPO":           vdef(DIMS_2D, units="m"),
    "TERRAIN_CONFIG": vdef(DIMS_2D, units="1"),
    "SKY_VIEW":       vdef(DIMS_2D, units="1"),
})

# Water / ice / urban (+ conditionals)
REGISTRY.update({
    "PCT_WETLAND":     vdef(DIMS_2D, units="%"),
    "PCT_LAKE":        vdef(DIMS_2D, units="%"),
    "PCT_GLACIER":     vdef(DIMS_2D, units="%"),
    "PCT_URBAN":       vdef(DIMS_2D, units="%"),
    "URBAN_REGION_ID": vdef(DIMS_2D, dtype="int16", units="1"),
    "GLC_MEC":         vdef(DIMS_2D, dtype="int16", units="1"),
    "PCT_GLC_MEC":     vdef(DIMS_2D, units="%"),
})

# Monthly canopy structure (12)
REGISTRY.update(register_many(
    ["MONTHLY_LAI","MONTHLY_SAI","MONTHLY_HEIGHT_TOP","MONTHLY_HEIGHT_BOT"],
    vdef(DIMS_TIME2D, units="varies")
))

# Phosphorus pools (optional group)
REGISTRY.update(register_many(
    ["APATITE_P","LABILE_P","OCCLUDED_P","SECONDARY_P"],
    vdef(DIMS_2D, units="gP/m2")
))

# ------------- Minimal Schema (rules, not per-var lines) ----------------
# Expressed as tiers with small rule tokens. Validation expands at runtime.

SCHEMA = {
    "TIER0_CORE_COORD_MASK": {
        "required": ["LATIXY","LONGXY","AREA","LANDFRAC_PFT"],
        "optional": ["PFTDATA_MASK"],
    },
    "TIER1_LANDCOVER": {
        # choose one of these as source-of-truth for nat veg extent
        "choose_one_of": [["PCT_NATVEG","PCT_NAT_PFT"]],
        "optional": ["PCT_CROP"],
    },
    "TIER2_SOIL": {
        "required": ["PCT_SAND","PCT_CLAY"],
        "optional": ["ORGANIC","PCT_GRVL"],
    },
    "TIER3_TOPO": {
        "required": ["SLOPE"],
        "optional": ["STDEV_ELEV","STD_ELEV","TOPO","TERRAIN_CONFIG","SKY_VIEW"],
    },
    "TIER4_WATER_ICE_URBAN": {
        "optional": ["PCT_WETLAND","PCT_LAKE","PCT_GLACIER","PCT_URBAN","GLC_MEC","PCT_GLC_MEC","URBAN_REGION_ID"],
        # simple condition flags your pipeline can evaluate with real data later
        "conditional": [
            {"if_var_present": "PCT_URBAN",   "then_require": ["URBAN_REGION_ID"]},
            {"if_var_present": "PCT_GLACIER", "then_require": ["GLC_MEC","PCT_GLC_MEC"]},
        ],
    },
    "TIER5_CANOPY_MONTHLY": {
        "optional": ["MONTHLY_LAI","MONTHLY_SAI","MONTHLY_HEIGHT_TOP","MONTHLY_HEIGHT_BOT"],
    },
    "TIER6_BGC_P": {
        "optional": ["APATITE_P","LABILE_P","OCCLUDED_P","SECONDARY_P"],
    },
}

# ------------- Export Policies (rule-based, dimension-aware) ------------

EXPORT_POLICIES = {
    # rules evaluated in order; first match wins
    "rules": [
        {"when": {"dims": ("time","lsmlat","lsmlon")},
         "policy": "MONTHLY_12_BANDS",
         "band_name": lambda var, sizes: [f"{var}_m{m:02d}" for m in range(1, sizes.get("time",0)+1)],
         "note": "Export 12 monthly bands; optionally also annual_mean"},
        {"when": {"dims": ("nlevsoi","lsmlat","lsmlon")},
         "policy": "SOIL_TOP_LAYER_DEFAULT",
         "band_name": lambda var, sizes: [f"{var}_L{l:02d}" for l in range(sizes.get("nlevsoi",0))],
         "note": "Default L00; optionally L00/L05/L09 stack"},
        {"when": {"dims": ("natpft","lsmlat","lsmlon")},
         "policy": "PFT_ALL_BANDS",
         "band_name": lambda var, sizes: [f"{var}_pft{p:02d}" for p in range(sizes.get("natpft",0))],
         "note": "Export all PFT bands; optionally aggregated classes"},
        {"when": {"dims": ("nlevslp","lsmlat","lsmlon")},
         "policy": "SLOPE_REDUCE_DEFAULT",
         "band_name": lambda var, sizes: [f"{var}_slp{b:02d}" for b in range(sizes.get("nlevslp",0))],
         "note": "Default reduce (weighted mean); optionally keep all bins"},
        {"when": {"dims": ("lsmlat","lsmlon")},
         "policy": "SINGLE_BAND",
         "band_name": lambda var, sizes: [var],
         "note": "Static 2D"},
    ],
    # fine-grained overrides if you need them (keep small)
    "overrides": {
        # "PCT_SAND": {"policy": "SOIL_MULTI_BANDS_3", "band_keep": ["L00","L05","L09"]},
        # "PCT_NAT_PFT": {"policy": "PFT_AGG_WGC", "band_keep": ["woody","grass","crop"]},
    }
}

# ---------------------- Runtime utilities --------------------------------

def expand_registry(as_json: bool = False) -> Dict[str, Dict]:
    """Return the full registry as plain dict (easy to dump/serialize)."""
    d = {k: asdict(v) for k, v in REGISTRY.items()}
    return d

def validate_against_schema(present_vars: Iterable[str]) -> Dict[str, List[str]]:
    """Validate a set of variable names against SCHEMA rules (structure only)."""
    present = set(present_vars)
    errors, warnings = [], []

    for tier, spec in SCHEMA.items():
        # required
        for v in spec.get("required", []):
            if v not in present:
                errors.append(f"{tier}: missing required var '{v}'")

        # choose_one_of groups
        for group in spec.get("choose_one_of", []):
            group_vars = group
            if isinstance(group, dict): group_vars = group.get("vars", group.get("choose_one_of", []))
            if group_vars and not (present & set(group_vars)):
                errors.append(f"{tier}: need one of {group_vars}")

        # conditional: we only check driver presence here (values checked later by pipeline)
        for cond in spec.get("conditional", []):
            driver = cond["if_var_present"]
            deps = cond["then_require"]
            if driver in present:
                for dep in deps:
                    if dep not in present:
                        warnings.append(f"{tier}: '{dep}' conditionally required because '{driver}' present")

    return {"errors": errors, "warnings": warnings}

def propose_export_policy(var: str, sizes: Dict[str,int], vardef: VarDef | None = None):
    """Return a compact policy dict for a variable, based on its dims and overrides."""
    # override first
    ov = EXPORT_POLICIES["overrides"].get(var)
    if ov:
        return {"var": var, "policy": ov["policy"], "note": "override", "bands": ov.get("band_keep", [])}

    dims = tuple(vardef.dims if vardef else ())
    for rule in EXPORT_POLICIES["rules"]:
        if dims == rule["when"]["dims"]:
            bands = rule["band_name"](var, sizes) if callable(rule["band_name"]) else [var]
            return {"var": var, "policy": rule["policy"], "note": rule.get("note",""), "bands": bands}
    # fallback
    return {"var": var, "policy": "UNKNOWN", "note": "no rule matched", "bands": [var]}
