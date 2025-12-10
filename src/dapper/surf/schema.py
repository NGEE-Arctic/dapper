# elm_surface_registry.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional


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
- VarDef: schema record for one variable (dims, dtype, units, doc, attrs).
- REGISTRY: dict[str, VarDef]
    Canonical list of surface variables with their expected dimension
    signatures and basic metadata. Think of this as the variable “spec.”
- SCHEMA: dict[str, Any]
    Tiered rules for presence/formatting:
      * per-variable requirement lives in VarDef.required_level
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
  (including spatial dims if it is spatial), its dtype, units, and doc.
- Per-variable requirement level should be set in VarDef.required_level
  (e.g., "required", "optional", "recommended"). The validator uses this.
- Add presence rules in SCHEMA only for cross-variable relationships
  (choose-one groups, conditionals, logical tiers).
- A separate script can parse report.rst and populate REGISTRY (doc and
  required_level) without changing this module.

Scope
-----
This module captures **formatting/structure** (dims, units, dtype, presence rules).
Numeric ranges, aggregation choices, and scientific provenance live in sampling
and validation layers, not here.

"""


# --------- Compact helpers so you don't write one-var-per-line ----------

@dataclass(frozen=True)
class VarDef:
    """
    Schema record for one surface variable.

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
    dims: Tuple[str, ...]
    dtype: str = "float32"
    units: str = ""
    doc: str = ""
    required_level: str = ""
    attrs: Optional[Dict[str, str]] = None


def vdef(
    dims,
    dtype: str = "float32",
    units: str = "",
    doc: str = "",
    required_level: str = "",
    **attrs,
) -> VarDef:
    """
    Convenience constructor for VarDef.

    dims can be a comma-separated string ("lsmlat,lsmlon") or an
    iterable of dim names. Any extra keyword args become NetCDF
    variable attributes (e.g., long_name="...").
    """
    if isinstance(dims, str):
        dims_tuple = tuple(d.strip() for d in dims.split(",") if d.strip())
    else:
        dims_tuple = tuple(dims)
    return VarDef(
        dims=dims_tuple,
        dtype=str(dtype),
        units=units,
        doc=doc,
        required_level=required_level,
        attrs=attrs or {},
    )


def register_many(names: Iterable[str], v: VarDef) -> Dict[str, VarDef]:
    """Register many variables with the same VarDef in one call."""
    return {name: v for name in names}


# Common dim-sets (reusable)
DIMS_2D     = "lsmlat,lsmlon"
DIMS_TIME2D = "time,lsmlat,lsmlon"
DIMS_SOIL   = "nlevsoi,lsmlat,lsmlon"
DIMS_PFT    = "natpft,lsmlat,lsmlon"
DIMS_SLOPE  = "nlevslp,lsmlat,lsmlon"
# Future: add topounit, urban, column, etc., as they appear in report.rst.

# ---------- Variable Registry (compact, grouped) ------------------------
# This is the single source of truth. A small subset is populated now;
# additional entries from report.rst can be merged here later.

REGISTRY: Dict[str, VarDef] = {
    # Core coords/mask/area
    "LATIXY": vdef(
        DIMS_2D,
        units="degrees_north",
        doc="Latitude of land grid cell centers.",
        required_level="required",
        long_name="latitude of land gridcell centers",
    ),
    "LONGXY": vdef(
        DIMS_2D,
        units="degrees_east",
        doc="Longitude of land grid cell centers.",
        required_level="required",
        long_name="longitude of land gridcell centers",
    ),
    "AREA": vdef(
        DIMS_2D,
        units="m2",
        doc="Area of land grid cells.",
        required_level="required",
        long_name="area of land gridcell",
    ),
    "LANDFRAC_PFT": vdef(
        DIMS_2D,
        units="1",
        doc="Fraction of global gridcell that is land (PFT tile).",
        required_level="required",
        long_name="land fraction for PFT landunit",
    ),
    "PFTDATA_MASK": vdef(
        DIMS_2D,
        dtype="int16",
        units="1",
        doc="Mask for valid PFT data (1=valid land; 0=invalid).",
        required_level="optional",
        long_name="mask for valid pftdata",
    ),
}

# Land cover
REGISTRY.update({
    "PCT_NATVEG": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is natural vegetation.",
        required_level="",  # enforced via choose_one_of with PCT_NAT_PFT
        long_name="percent natural vegetation landunit",
    ),
    "PCT_CROP": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is crops.",
        required_level="optional",
        long_name="percent crop landunit",
    ),
})
REGISTRY.update({
    "PCT_NAT_PFT": vdef(
        DIMS_PFT,
        units="%",
        doc="Percent of landunit in each natural plant functional type (PFT).",
        required_level="",  # enforced via choose_one_of with PCT_NATVEG
        long_name="percent natural vegetation per PFT",
    )
})

# Soil (layered)
REGISTRY.update({
    "PCT_SAND": vdef(
        DIMS_SOIL,
        units="%",
        doc="Soil sand percentage by mass in each soil layer.",
        required_level="required",
        long_name="percent sand by mass",
    ),
    "PCT_CLAY": vdef(
        DIMS_SOIL,
        units="%",
        doc="Soil clay percentage by mass in each soil layer.",
        required_level="required",
        long_name="percent clay by mass",
    ),
    "ORGANIC": vdef(
        DIMS_SOIL,
        units="kg/m2",
        doc="Soil organic matter (or carbon) content per layer.",
        required_level="optional",
        long_name="soil organic material",
    ),
    "PCT_GRVL": vdef(
        DIMS_SOIL,
        units="%",
        doc="Percent gravel content in each soil layer.",
        required_level="optional",
        long_name="percent gravel by volume",
    ),
})

# Topography / terrain
REGISTRY.update({
    "SLOPE": vdef(
        DIMS_2D,
        units="degrees",
        doc="Mean surface slope per gridcell.",
        required_level="required",
        long_name="mean topographic slope",
    ),
    "STDEV_ELEV": vdef(
        DIMS_2D,
        units="m",
        doc="Standard deviation of elevation within each gridcell.",
        required_level="optional",
        long_name="standard deviation of elevation",
    ),
    "STD_ELEV": vdef(
        DIMS_2D,
        units="m",
        doc="Alias for STDEV_ELEV in some datasets.",
        required_level="optional",
        long_name="standard deviation of elevation (alias)",
    ),
    "TOPO": vdef(
        DIMS_2D,
        units="m",
        doc="Mean surface elevation per gridcell.",
        required_level="optional",
        long_name="mean elevation of land gridcell",
    ),
    "TERRAIN_CONFIG": vdef(
        DIMS_2D,
        units="1",
        doc="Terrain configuration index used in radiative transfer.",
        required_level="optional",
        long_name="terrain configuration factor",
    ),
    "SKY_VIEW": vdef(
        DIMS_2D,
        units="1",
        doc="Sky view factor (0–1), accounting for topographic obstruction.",
        required_level="optional",
        long_name="sky view factor",
    ),
})

# Water / ice / urban (+ conditionals)
REGISTRY.update({
    "PCT_WETLAND": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is wetland.",
        required_level="optional",
        long_name="percent wetland landunit",
    ),
    "PCT_LAKE": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is lake.",
        required_level="optional",
        long_name="percent lake landunit",
    ),
    "PCT_GLACIER": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is glacier.",
        required_level="optional",
        long_name="percent glacier landunit",
    ),
    "PCT_URBAN": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of landunit that is urban.",
        required_level="optional",
        long_name="percent urban landunit",
    ),
    "URBAN_REGION_ID": vdef(
        DIMS_2D,
        dtype="int16",
        units="1",
        doc="Urban region index for parameter lookup.",
        required_level="conditional",
        long_name="urban region identifier",
    ),
    "GLC_MEC": vdef(
        DIMS_2D,
        dtype="int16",
        units="1",
        doc="Glacier elevation class index for each gridcell.",
        required_level="conditional",
        long_name="glacier elevation class index",
    ),
    "PCT_GLC_MEC": vdef(
        DIMS_2D,
        units="%",
        doc="Percent of gridcell area for each glacier elevation class.",
        required_level="conditional",
        long_name="percent glacier area by elevation class",
    ),
})

# Monthly canopy structure (12)
REGISTRY.update(register_many(
    ["MONTHLY_LAI", "MONTHLY_SAI", "MONTHLY_HEIGHT_TOP", "MONTHLY_HEIGHT_BOT"],
    vdef(
        DIMS_TIME2D,
        units="varies",
        doc="Monthly climatology (12 months) for canopy structural properties.",
        required_level="optional",
        long_name="monthly climatology",
    ),
))

# Phosphorus pools (optional group)
REGISTRY.update(register_many(
    ["APATITE_P", "LABILE_P", "OCCLUDED_P", "SECONDARY_P"],
    vdef(
        DIMS_2D,
        units="gP/m2",
        doc="Soil phosphorus pools by geochemical form.",
        required_level="optional",
        long_name="soil phosphorus pool",
    ),
))


# ------------- Minimal Schema (rules, not per-var lines) ----------------
# SCHEMA now just organizes variables into logical tiers and handles
# cross-variable rules. Per-variable "requiredness" is stored in
# VarDef.required_level inside REGISTRY.

SCHEMA: Dict[str, Dict] = {
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
        "vars": ["SLOPE", "STDEV_ELEV", "STD_ELEV", "TOPO", "TERRAIN_CONFIG", "SKY_VIEW"],
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
            {"if_var_present": "PCT_URBAN",   "then_require": ["URBAN_REGION_ID"]},
            {"if_var_present": "PCT_GLACIER", "then_require": ["GLC_MEC", "PCT_GLC_MEC"]},
        ],
    },
    "TIER5_CANOPY_MONTHLY": {
        "vars": ["MONTHLY_LAI", "MONTHLY_SAI", "MONTHLY_HEIGHT_TOP", "MONTHLY_HEIGHT_BOT"],
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
            "band_name": lambda var, sizes: [f"{var}_m{m:02d}" for m in range(1, sizes.get("time", 0) + 1)],
            "note": "Export 12 monthly bands; optionally also annual_mean",
        },
        {
            "when": {"dims": ("nlevsoi", "lsmlat", "lsmlon")},
            "policy": "SOIL_TOP_LAYER_DEFAULT",
            "band_name": lambda var, sizes: [f"{var}_L{l:02d}" for l in range(sizes.get("nlevsoi", 0))],
            "note": "Default L00; optionally L00/L05/L09 stack",
        },
        {
            "when": {"dims": ("natpft", "lsmlat", "lsmlon")},
            "policy": "PFT_ALL_BANDS",
            "band_name": lambda var, sizes: [f"{var}_pft{p:02d}" for p in range(sizes.get("natpft", 0))],
            "note": "Export all PFT bands; optionally aggregated classes",
        },
        {
            "when": {"dims": ("nlevslp", "lsmlat", "lsmlon")},
            "policy": "SLOPE_REDUCE_DEFAULT",
            "band_name": lambda var, sizes: [f"{var}_slp{b:02d}" for b in range(sizes.get("nlevslp", 0))],
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

def expand_registry(as_json: bool = False) -> Dict[str, Dict]:
    """Return the full registry as plain dict (easy to dump/serialize)."""
    d = {k: asdict(v) for k, v in REGISTRY.items()}
    return d


def validate_against_schema(present_vars: Iterable[str]) -> Dict[str, List[str]]:
    """
    Validate a set of variable names against SCHEMA rules.

    - Per-variable requirement is taken from VarDef.required_level
      (currently only 'required' is treated as hard-required).
    - 'choose_one_of' groups are enforced at the tier level.
    - 'conditional' rules are enforced as warnings when violated.
    """
    present = set(present_vars)
    errors: List[str] = []
    warnings: List[str] = []

    for tier, spec in SCHEMA.items():
        tier_vars = spec.get("vars", [])

        # required vars in this tier: those with required_level == "required"
        for vname in tier_vars:
            vdef_obj = REGISTRY.get(vname)
            if not vdef_obj:
                continue  # allow schema to reference vars not yet in registry
            if vdef_obj.required_level.lower() == "required" and vname not in present:
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


def propose_export_policy(var: str, sizes: Dict[str, int], vardef: VarDef | None = None):
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

    dims = tuple(vardef.dims if vardef else ())
    for rule in EXPORT_POLICIES["rules"]:
        if dims == tuple(rule["when"]["dims"]):
            bands = (
                rule["band_name"](var, sizes)
                if callable(rule["band_name"])
                else [var]
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
