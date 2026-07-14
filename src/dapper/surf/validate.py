# dapper/surf/validate.py
"""dapper module: surf.validate."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

# Your hard-coded schema/registry module
from dapper.surf import schema as SC  # expects REGISTRY (ParDef map), SCHEMA (tiers)

@dataclass
class CheckResult:
    """One validation result row."""
    check: str
    severity: str   # ERROR | WARN | INFO
    passed: bool
    detail: str
    var: Optional[str] = None

class SurfaceValidator:
    """
    Validator for ELM/CLM **point** surface NetCDF files (1×1 spatial cell).

    Primary (format/layout) checks
    ------------------------------
    V-001  dims: lat/lon like dims exist and both have length == 1 (ERROR)
    V-002  dims.sizes: expected sizes for common dims (WARN) e.g. time=12, natpft=17, lsmpft=17, nlevsoi=10, nlevslp=11, numurbl=3, numrad=2, nlevurb=5
    V-003  schema.required: required variables per SCHEMA are present (ERROR)
    V-004  schema.choose_one_of: at least one var present in each group (ERROR)
    V-005  schema.conditional: if driver present → dependent vars present (WARN)
    V-006  registry.known_vars: vars not in REGISTRY flagged (INFO, or ERROR if enforce_known_vars_only)
    V-007  dims.order: for spatial vars, spatial dims are the **last two** (..., lat, lon) (WARN)
    V-008  dims.match_registry: non-spatial dim order matches REGISTRY (WARN)
    V-009  dtype.match_registry: integer vs float matches REGISTRY (WARN)
    V-010  units.present: 'units' attribute exists (WARN)
    V-011  units.match_registry: exact match when REGISTRY.units not ''/'varies' (WARN)
    V-012  fillvalue.sane: floats have _FillValue (NaN ok), ints do not rely on NaN (WARN)
    V-013  coordinates.present: LATIXY/LONGXY present (WARN); optional INFO: lat/lon coord ≈ LATIXY/LONGXY

    Soft (non-blocking) checks
    --------------------------
    V-101  ranges.percent: PCT_* (and PCT_NATVEG) ∈ [0,100] (ERROR)
    V-102  ranges.unit: LANDFRAC_PFT, SKY_VIEW ∈ [0,1] (ERROR)
    V-103  ranges.nonneg: SLOPE, (ST)DEV_ELEV, AREA, TOPO ≥ 0 (ERROR)
    V-104  time.length: any var with 'time' dim → len(time)==12 (ERROR)
    V-105  consistency.pftsum: sum(PCT_NAT_PFT) ≈ PCT_NATVEG (WARN)
    V-106  conditional.urban: if max(PCT_URBAN)>0 → URBAN_REGION_ID present (WARN)
    V-107  conditional.glacier: if max(PCT_GLACIER)>0 → GLC_MEC & PCT_GLC_MEC present (WARN)

    Usage
    -----
    >>> v = SurfaceValidator()
    >>> report = v.validate(r"X:\\path\\surfdata_1x1pt.nc")
    >>> report.query("severity=='ERROR' and passed==False")
    """

    def __init__(
        self,
        *,
        expected_sizes: Dict[str, int] | None = None,
        lat_candidates: Tuple[str, ...] = ("lsmlat","lat","latitude","y"),
        lon_candidates: Tuple[str, ...] = ("lsmlon","lon","longitude","x"),
        enforce_known_vars_only: bool = False,
        require_point_dims: bool = True,
        skip_soft_checks: bool = False,
    ):
        self.expected_sizes = expected_sizes or {
            "time": 12, "natpft": 17, "lsmpft": 17, "nlevsoi": 10,
            "nlevslp": 11, "numurbl": 3, "numrad": 2, "nlevurb": 5
        }
        self.require_point_dims = require_point_dims
        self.lat_candidates = lat_candidates
        self.lon_candidates = lon_candidates
        self.enforce_known_vars_only = enforce_known_vars_only
        self.skip_soft_checks = skip_soft_checks


    # ---------- public API: path-only ----------
    def validate(self, nc_path: str) -> pd.DataFrame:
        """Run validation on a surface NetCDF and return a structured report."""
        
        if not isinstance(nc_path, (str, bytes)):
            raise TypeError("validate() expects a NetCDF file path (str).")
        ds = xr.open_dataset(nc_path)

        results: List[CheckResult] = []
        lat_dim = next((d for d in ds.dims if d in self.lat_candidates), None)
        lon_dim = next((d for d in ds.dims if d in self.lon_candidates), None)

        results += self._check_point_dims(ds, lat_dim, lon_dim)     # V-001
        results += self._check_expected_sizes(ds)                    # V-002
        results += self._check_schema_presence(ds)                   # V-003..V-005
        results += self._check_unknown_vars(ds)                      # V-006

        for name, da in ds.data_vars.items():
            results += self._check_var_layout(name, da, lat_dim, lon_dim)  # V-007..V-008
            results += self._check_dtype_units(name, da)                   # V-009..V-011
            results += self._check_fillvalue(name, da)                     # V-012

        results += self._check_latlon_coord_presence(ds)             # V-013

        if not self.skip_soft_checks:
            results += self._check_ranges(ds)                        # V-101..V-104
            results += self._check_soft_consistency(ds)              # V-105..V-107

        df = pd.DataFrame([r.__dict__ for r in results])
        order = {"ERROR":0,"WARN":1,"INFO":2}
        df["sev_o"] = df["severity"].map(order)
        df["pass_o"] = (~df["passed"]).astype(int)
        return df.sort_values(["sev_o","pass_o","check","var"]).drop(columns=["sev_o","pass_o"]).reset_index(drop=True)

    # ---------- individual checks ----------
    def _check_point_dims(self, ds: xr.Dataset, lat_dim: Optional[str], lon_dim: Optional[str]) -> List[CheckResult]:
        r = [
            CheckResult("V-001.lat_dim.present", "ERROR", lat_dim is not None, f"lat_dim={lat_dim}"),
            CheckResult("V-001.lon_dim.present", "ERROR", lon_dim is not None, f"lon_dim={lon_dim}"),
        ]
        if self.require_point_dims:
            if lat_dim:
                r.append(CheckResult("V-001.lat_dim.len1", "ERROR", int(ds.sizes[lat_dim]) == 1, f"{lat_dim}={ds.sizes[lat_dim]}"))
            if lon_dim:
                r.append(CheckResult("V-001.lon_dim.len1", "ERROR", int(ds.sizes[lon_dim]) == 1, f"{lon_dim}={ds.sizes[lon_dim]}"))
        return r

    def _check_expected_sizes(self, ds: xr.Dataset) -> List[CheckResult]:
        r: List[CheckResult] = []
        for dim, n in self.expected_sizes.items():
            if dim in ds.dims:
                r.append(CheckResult(f"V-002.size.{dim}", "WARN", int(ds.sizes[dim])==n, f"found {ds.sizes[dim]}, expected {n}"))
        return r
    
    def _check_schema_presence(self, ds: xr.Dataset) -> list[CheckResult]:
        """
        Check that required/recommended schema vars exist and schema rules are satisfied.

        Requiredness is driven by SC.REGISTRY[var].required_level, not SC.SCHEMA.
        SC.SCHEMA provides tier organization + cross-var rules (choose_one_of, conditional).
        """
        r: list[CheckResult] = []
        present = set(ds.data_vars)

        for tier, spec in SC.SCHEMA.items():
            tier_vars = list(spec.get("vars", []) or [])

            # Presence checks: required vs recommended (driven by registry)
            for v in tier_vars:
                par = SC.REGISTRY.get(v)
                if par is None:
                    # If schema references something not in registry, don't hard fail here.
                    continue

                lvl = getattr(par, "required_level", "optional") or "optional"
                if lvl == "required":
                    r.append(
                        CheckResult(
                            check=f"V-003.required.{tier}.{v}",
                            severity="ERROR",
                            passed=(v in present),
                            detail=f"Missing REQUIRED var {v} in tier {tier}" if v not in present else "present",
                            var=v,
                        )
                    )
                elif lvl == "recommended":
                    r.append(
                        CheckResult(
                            check=f"V-003.recommended.{tier}.{v}",
                            severity="WARN",
                            passed=(v in present),
                            detail=f"Missing recommended var {v} in tier {tier}" if v not in present else "present",
                            var=v,
                        )
                    )

            # choose_one_of groups (schema rule)
            for group in spec.get("choose_one_of", []) or []:
                # group is expected as list like ["PCT_NATVEG","PCT_NAT_PFT"]
                if not isinstance(group, (list, tuple)) or len(group) == 0:
                    continue
                ok = any(v in present for v in group)
                r.append(
                    CheckResult(
                        check=f"V-004.choose_one_of.{tier}",
                        severity="ERROR",
                        passed=ok,
                        detail=f"Must include at least one of {group} (tier {tier})" if not ok else "ok",
                        var="|".join(group),
                    )
                )

            # conditional rules (schema rule)
            for cond in spec.get("conditional", []) or []:
                driver = cond.get("if_var_present")
                deps = cond.get("then_require", []) or []
                if not driver or not deps:
                    continue
                if driver in present:
                    for dep in deps:
                        r.append(
                            CheckResult(
                                check=f"V-005.conditional.{tier}.{driver}->{dep}",
                                severity="WARN",
                                passed=(dep in present),
                                detail=f"{dep} should be present when {driver} is present (tier {tier})"
                                if dep not in present
                                else "ok",
                                var=dep,
                            )
                        )

        return r

    def _check_unknown_vars(self, ds: xr.Dataset) -> List[CheckResult]:
        r: List[CheckResult] = []
        known = set(SC.REGISTRY.keys())
        for v in ds.data_vars:
            if v not in known:
                sev = "ERROR" if self.enforce_known_vars_only else "INFO"
                r.append(CheckResult("V-006.unknown_var", sev, not self.enforce_known_vars_only, "not in REGISTRY", v))
        return r

    def _check_var_layout(self, name: str, da: xr.DataArray, lat_dim: Optional[str], lon_dim: Optional[str]) -> List[CheckResult]:
        r: List[CheckResult] = []
        dims = tuple(da.dims)
        # spatial dims last
        if lat_dim and lon_dim and (lat_dim in dims and lon_dim in dims):
            r.append(CheckResult("V-007.spatial_last", "WARN", dims[-2:]==(lat_dim, lon_dim), f"dims={dims}", name))
        # match registry non-spatial order
        if name in SC.REGISTRY:
            expected_dims = tuple(SC.REGISTRY[name].dims)
            if lat_dim and lon_dim and (lat_dim in dims and lon_dim in dims):
                expected_nsp = tuple(d for d in expected_dims if d not in (lat_dim, lon_dim))
                actual_nsp   = tuple(d for d in dims       if d not in (lat_dim, lon_dim))
                r.append(CheckResult("V-008.nonspatial_dim_order", "WARN", actual_nsp==expected_nsp,
                                     f"actual={actual_nsp}, expected={expected_nsp}", name))
            else:
                reg_spatial = ("lsmlat" in expected_dims and "lsmlon" in expected_dims)
                act_spatial = (lat_dim in dims and lon_dim in dims) if (lat_dim and lon_dim) else False
                r.append(CheckResult("V-008.spatialness_matches_registry", "WARN", act_spatial==reg_spatial,
                                     f"registry_spatial={reg_spatial}, actual_spatial={act_spatial}", name))
        return r

    def _check_dtype_units(self, name: str, da: xr.DataArray) -> List[CheckResult]:
        r: List[CheckResult] = []
        dt = str(da.dtype)
        units = (da.attrs or {}).get("units", "")
        # dtype int vs float
        if name in SC.REGISTRY:
            exp_dt = SC.REGISTRY[name].dtype
            ok = dt.startswith(("int","uint")) == exp_dt.startswith(("int","uint"))
            r.append(CheckResult("V-009.dtype_matches_registry", "WARN", ok, f"actual={dt}, expected={exp_dt}", name))
        # units present
        r.append(CheckResult("V-010.units_present", "WARN", isinstance(units, str) and units.strip()!="", f"units={units!r}", name))
        # units match registry
        if name in SC.REGISTRY:
            exp_units = SC.REGISTRY[name].units
            if exp_units and exp_units.lower() not in ("varies",):
                r.append(CheckResult("V-011.units_match_registry", "WARN", str(units)==exp_units,
                                     f"actual={units!r}, expected={exp_units!r}", name))
        return r

    def _check_fillvalue(self, name: str, da: xr.DataArray) -> List[CheckResult]:
        r: List[CheckResult] = []
        dt = str(da.dtype)
        fv = (da.encoding or {}).get("_FillValue", da.attrs.get("_FillValue"))
        if dt.startswith(("int","uint")):
            ok = (fv is None) or isinstance(fv, (int, np.integer))
            r.append(CheckResult("V-012.fillvalue_int", "WARN", ok, f"_FillValue={fv}", name))
        else:
            r.append(CheckResult("V-012.fillvalue_float", "WARN", fv is not None, f"_FillValue={fv}", name))
        return r

    def _check_latlon_coord_presence(self, ds: xr.Dataset) -> List[CheckResult]:
        r: List[CheckResult] = []
        r.append(CheckResult("V-013.coords.LATIXY", "WARN", "LATIXY" in ds, "present" if "LATIXY" in ds else "missing"))
        r.append(CheckResult("V-013.coords.LONGXY", "WARN", "LONGXY" in ds, "present" if "LONGXY" in ds else "missing"))
        # optional INFO comparisons
        if "LATIXY" in ds and "lat" in ds.coords:
            lat_ok = _safe_allclose(ds["LATIXY"].values, ds["lat"].values, atol=1e-3)
            r.append(CheckResult("V-013.coords.lat_vs_LATIXY", "INFO", lat_ok, "lat ≈ LATIXY"))
        if "LONGXY" in ds and "lon" in ds.coords:
            lon_ok = _safe_allclose(ds["LONGXY"].values, ds["lon"].values, atol=1e-3)
            r.append(CheckResult("V-013.coords.lon_vs_LONGXY", "INFO", lon_ok, "lon ≈ LONGXY"))
        return r

    # --------- soft checks (use reductions, never cast to float directly) ---------
    def _check_ranges(self, ds: xr.Dataset) -> List[CheckResult]:
        r: List[CheckResult] = []
        def _in_range(a, lo, hi):
            aa = np.asarray(a, dtype="float64")
            aa = aa[np.isfinite(aa)]
            if aa.size == 0: return True, "empty/all-NaN"
            ok = (aa >= lo - 1e-9) & (aa <= hi + 1e-9)
            return bool(ok.all()), f"{ok.mean()*100:.1f}% within [{lo},{hi}]"

        pct = [v for v in ds.data_vars if v.startswith("PCT_")]
        if "PCT_NATVEG" in ds: pct.append("PCT_NATVEG")
        for v in pct:
            ok, msg = _in_range(ds[v].values, 0, 100)
            r.append(CheckResult("V-101.range.percent", "ERROR", ok, f"{v}: {msg}", v))

        for v in ("LANDFRAC_PFT","SKY_VIEW"):
            if v in ds:
                ok, msg = _in_range(ds[v].values, 0, 1)
                r.append(CheckResult("V-102.range.unit", "ERROR", ok, f"{v}: {msg}", v))

        for v in ("SLOPE","STDEV_ELEV","STD_ELEV","AREA","TOPO"):
            if v in ds:
                ok, msg = _in_range(ds[v].values, 0, np.inf)
                r.append(CheckResult("V-103.range.nonneg", "ERROR", ok, f"{v}: {msg}", v))

        if "time" in ds.dims:
            tlen = int(ds.sizes["time"])
            for v, da in ds.data_vars.items():
                if "time" in da.dims:
                    r.append(CheckResult("V-104.time.len12", "ERROR", tlen==12, f"time={tlen}", v))
        return r

    def _check_soft_consistency(self, ds: xr.Dataset) -> List[CheckResult]:
        r: List[CheckResult] = []
        # sum(PCT_NAT_PFT) ≈ 100 (natural-patch weights)
        if "PCT_NAT_PFT" in ds and "natpft" in ds["PCT_NAT_PFT"].dims:
            pftsum = ds["PCT_NAT_PFT"].sum(dim="natpft", skipna=True)
            resid = np.abs(pftsum.values - 100.0)
            mean_diff = float(np.nanmean(resid))
            max_diff = float(np.nanmax(resid))
            r.append(CheckResult("V-105.consistency.pftsum", "WARN", max_diff <= 1e-6,
                                 f"mean(|sum(PCT_NAT_PFT)-100|)={mean_diff:.3e}; max={max_diff:.3e}"))

        # landunit closure ≈ 100 across available classes (incl urban aggregate)
        landunit_terms: List[xr.DataArray] = []
        for name in ("PCT_NATVEG", "PCT_CROP", "PCT_WETLAND", "PCT_LAKE", "PCT_GLACIER"):
            if name in ds:
                landunit_terms.append(ds[name].astype("float64"))
        if "PCT_URBAN" in ds:
            urb = ds["PCT_URBAN"].astype("float64")
            if "numurbl" in urb.dims:
                landunit_terms.append(urb.sum(dim="numurbl", skipna=True))
            else:
                landunit_terms.append(urb)
        if len(landunit_terms) >= 2:
            lsum = sum(landunit_terms)
            lresid = np.abs(np.asarray(lsum.values, dtype="float64") - 100.0)
            mean_diff = float(np.nanmean(lresid))
            max_diff = float(np.nanmax(lresid))
            r.append(CheckResult("V-108.consistency.landunitsum", "WARN", max_diff <= 1e-6,
                                 f"mean(|landunit_sum-100|)={mean_diff:.3e}; max={max_diff:.3e}"))

        # topounit area weights should close to 100 across topounit dim.
        for top_var in ("PCT_TOPUNIT", "TopounitFracArea"):
            if top_var in ds and "topounit" in ds[top_var].dims:
                tsum = ds[top_var].sum(dim="topounit", skipna=True)
                tresid = np.abs(np.asarray(tsum.values, dtype="float64") - 100.0)
                mean_diff = float(np.nanmean(tresid))
                max_diff = float(np.nanmax(tresid))
                r.append(CheckResult("V-109.consistency.topounitsum", "WARN", max_diff <= 1e-6,
                                     f"{top_var}: mean(|sum-100|)={mean_diff:.3e}; max={max_diff:.3e}"))
                break

        # If any cell has PCT_URBAN>0 → URBAN_REGION_ID present
        if "PCT_URBAN" in ds:
            urb_max = np.nanmax(np.asarray(ds["PCT_URBAN"].values))
            need = bool(urb_max > 0)
            have = "URBAN_REGION_ID" in ds
            r.append(CheckResult("V-106.consistency.urban_id", "WARN", (not need) or have,
                                 f"max(PCT_URBAN)={urb_max:.3f}, URBAN_REGION_ID={have}"))

        # If any cell has PCT_GLACIER>0 → GLC_MEC & PCT_GLC_MEC present
        if "PCT_GLACIER" in ds:
            gl_max = np.nanmax(np.asarray(ds["PCT_GLACIER"].values))
            need = bool(gl_max > 0)
            have = ("GLC_MEC" in ds) and ("PCT_GLC_MEC" in ds)
            r.append(CheckResult("V-107.consistency.glacier_mec", "WARN", (not need) or have,
                                 f"max(PCT_GLACIER)={gl_max:.3f}, GLC_MEC&PCT_GLC_MEC={have}"))
        return r

# --- small util ---
def _safe_allclose(a, b, atol=1e-6, rtol=0.0) -> bool:
    try:
        aa = float(np.asarray(a).squeeze())
        bb = float(np.asarray(b).squeeze())
        return np.isfinite(aa) and np.isfinite(bb) and abs(aa - bb) <= atol + rtol*abs(bb)
    except Exception:
        return False
