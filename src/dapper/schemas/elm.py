# dapper/schemas/elm.py
"""
Canonical ELM target schema & helpers.

Everything here is dataset-agnostic. Adapters should map their raw/source
variables into these canonical names *before* handing data to writers.
"""

from __future__ import annotations

# ---- Canonical units (by ELM short name) ----
ELM_UNITS = {
    "TBOT": "K",
    "DTBOT": "K",          # dewpoint temp; keep 'K' unless you store 'C'
    "RH": "%",             # relative humidity (0–100)
    "WIND": "m/s",
    "FSDS": "W/m2",        # shortwave down
    "FLDS": "W/m2",        # longwave down
    "PSRF": "Pa",
    "PRECTmms": "mm/s",    # equivalent to kg/m2/s
    "QBOT": "kg/kg",       # specific humidity
    "ZBOT": "m",
    "UWIND": "m/s",
    "VWIND": "m/s",
}

# ---- Canonical value ranges useful for packing/sanity checks ----
ELM_RANGES = {
    "PRECTmms": (-0.04, 0.04),
    "FSDS": (-20.0, 2000.0),
    "TBOT": (175.0, 350.0),
    "RH": (0.0, 100.0),
    "QBOT": (0.0, 0.04),
    "FLDS": (0.0, 1000.0),
    "PSRF": (20000.0, 120000.0),
    "WIND": (-1.0, 100.0),
    # Optional extras (not always exported)
    "UWIND": (-100.0, 100.0),
    "VWIND": (-100.0, 100.0),
    "DTBOT": (175.0, 350.0),
    "ZBOT": (0.0, 100.0),
}

# ---- Required canonical variables by output format ----
ELM_REQUIRED = {
    # Coupler bypass (most common for your workflow)
    "BYPASS": ["LONGXY", "LATIXY", "time",
               "TBOT", "PRECTmms", "QBOT", "FSDS", "FLDS", "PSRF", "WIND"],
    # DATM mode (if/when you wire it up)
    "DATM_MODE": ["LONGXY", "LATIXY", "time",
                  "ZBOT", "TBOT", "PRECTmms", "RH", "FSDS", "FLDS", "PSRF", "WIND"],
}

# ---- Canonical non-negative variables (after unit conversion) ----
NONNEG_CANONICAL = {
    "FSDS", "FLDS", "PRECTmms", "PSRF", "QBOT",
}

# ----------------- tiny helpers (no heavy deps) -----------------

def elm_required_vars(dformat: str) -> list[str]:
    """Return the list of canonical vars required by a given ELM format."""
    if dformat not in ELM_REQUIRED:
        raise KeyError(f"Unsupported dformat: {dformat}")
    return list(ELM_REQUIRED[dformat])

def elm_units(var: str) -> str:
    """Return canonical units for a variable."""
    return ELM_UNITS[var]

def elm_range(var: str) -> tuple[float, float]:
    """Return canonical recommended range for a variable."""
    return ELM_RANGES[var]

def is_nonnegative(var: str) -> bool:
    """True if var should be clipped to >= 0 after conversion."""
    return var in NONNEG_CANONICAL

__all__ = [
    "ELM_UNITS", "ELM_RANGES", "ELM_REQUIRED", "NONNEG_CANONICAL",
    "elm_required_vars", "elm_units", "elm_range", "is_nonnegative",
]
