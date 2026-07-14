"""Shared fraction-closure logic for surface and landuse datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

# Keep closures aligned with the canonical surface variable specs.
from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS


def _scale_to_target_sum(
    da: xr.DataArray,
    *,
    dim: str,
    target: xr.DataArray,
    eps: float = 1e-12,
) -> xr.DataArray:
    """Scale values along `dim` so their sum matches `target` where possible."""
    work = da.astype(np.float64)
    summed = work.sum(dim=dim, skipna=True)
    valid = np.isfinite(summed) & (np.abs(summed) > eps) & np.isfinite(target)
    safe_den = xr.where(valid, summed, 1.0)
    factor = xr.where(valid, target.astype(np.float64) / safe_den, 1.0)
    return work * factor


def _snap_partition_sum(
    da: xr.DataArray,
    *,
    dim: str,
    target: xr.DataArray,
    eps: float = 1e-15,
) -> xr.DataArray:
    """Force exact closure by assigning residual to the last band in float64."""
    work = da.astype(np.float64)
    if dim not in work.dims or int(work.sizes.get(dim, 0)) < 1:
        return work

    head = work.isel({dim: slice(0, -1)})
    head_sum = head.sum(dim=dim, skipna=True)
    last_src = work.isel({dim: -1})
    candidate_last = target.astype(np.float64) - head_sum

    valid = np.isfinite(target) & np.isfinite(head_sum)
    snapped_last = xr.where(valid, candidate_last, last_src)
    snapped_last = xr.where(np.abs(snapped_last) < eps, 0.0, snapped_last)

    out = work.copy(deep=False)
    out[{dim: -1}] = snapped_last
    return out


def _close_partition(
    ds: xr.Dataset,
    *,
    var_name: str,
    dim: str,
    target: xr.DataArray,
) -> None:
    if var_name not in ds:
        return
    da = ds[var_name]
    if dim not in da.dims:
        return
    ds[var_name] = _scale_to_target_sum(da, dim=dim, target=target)
    ds[var_name] = _snap_partition_sum(ds[var_name], dim=dim, target=target)


def _close_unit_partition(ds: xr.Dataset, *, left: str, right: str) -> None:
    if left not in ds or right not in ds:
        return
    lhs = ds[left].astype(np.float64)
    rhs = ds[right].astype(np.float64)
    total = lhs + rhs
    valid = np.isfinite(total) & (total > 1e-12)
    factor = xr.where(valid, 1.0 / total, 1.0)
    ds[left] = lhs * factor
    ds[right] = rhs * factor


def _landunit_scalar_names(ds: xr.Dataset) -> list[str]:
    # Explicitly constrained to ELM landunit classes. We gate against canonical
    # specs so stale names are ignored automatically.
    candidates = ["PCT_NATVEG", "PCT_CROP", "PCT_WETLAND", "PCT_LAKE", "PCT_GLACIER"]
    return [v for v in candidates if (v in ds and v in SURFACE_VAR_SPECS)]


def _apply_landunit_total_closure(ds: xr.Dataset) -> None:
    """Enforce landunit totals to close to 100 (including urban aggregate)."""
    scalar_names = _landunit_scalar_names(ds)
    landunit_terms: list[xr.DataArray] = [ds[name].astype(np.float64) for name in scalar_names]

    urban_group = None
    if "PCT_URBAN" in ds:
        urban = ds["PCT_URBAN"].astype(np.float64)
        if "numurbl" in urban.dims:
            urban_group = urban
            landunit_terms.append(urban.sum(dim="numurbl", skipna=True))
        else:
            landunit_terms.append(urban)

    if len(landunit_terms) < 2:
        return

    current_total = sum(landunit_terms)
    near_100 = np.abs(current_total - 100.0) <= 1.0
    valid = np.isfinite(current_total) & (current_total > 1e-12) & near_100
    factor = xr.where(valid, 100.0 / current_total, 1.0)

    for name in scalar_names:
        ds[name] = ds[name].astype(np.float64) * factor
    if urban_group is not None:
        ds["PCT_URBAN"] = urban_group * factor
    elif "PCT_URBAN" in ds:
        ds["PCT_URBAN"] = ds["PCT_URBAN"].astype(np.float64) * factor

    total_after = 0.0
    for name in scalar_names:
        total_after = total_after + ds[name].astype(np.float64)
    if "PCT_URBAN" in ds:
        urb = ds["PCT_URBAN"].astype(np.float64)
        if "numurbl" in urb.dims:
            total_after = total_after + urb.sum(dim="numurbl", skipna=True)
        else:
            total_after = total_after + urb

    resid = 100.0 - total_after

    # Prefer scalar classes first; fallback to last urban class.
    snapped = False
    for name in ("PCT_GLACIER", "PCT_LAKE", "PCT_WETLAND", "PCT_CROP", "PCT_NATVEG"):
        if name in ds:
            ds[name] = ds[name].astype(np.float64) + resid
            snapped = True
            break

    if (not snapped) and ("PCT_URBAN" in ds):
        urb = ds["PCT_URBAN"].astype(np.float64)
        if "numurbl" in urb.dims and int(urb.sizes.get("numurbl", 0)) >= 1:
            urb[{"numurbl": -1}] = urb.isel(numurbl=-1) + resid
            ds["PCT_URBAN"] = urb
        else:
            ds["PCT_URBAN"] = urb + resid


def _full_like_from_partition(ds: xr.Dataset, *, var_name: str, dim: str, value: float) -> xr.DataArray | None:
    if var_name not in ds or dim not in ds[var_name].dims:
        return None
    return xr.full_like(ds[var_name].isel({dim: 0}, drop=True), value, dtype=np.float64)


def normalize_fraction_closure(ds: xr.Dataset) -> xr.Dataset:
    """Apply canonical fraction closure for surface/landuse datasets."""
    ds2 = ds.copy(deep=False)

    _apply_landunit_total_closure(ds2)

    # Natural-patch weights should sum to 100 for every non-natpft index tuple.
    tgt_nat = _full_like_from_partition(ds2, var_name="PCT_NAT_PFT", dim="natpft", value=100.0)
    if tgt_nat is not None:
        _close_partition(ds2, var_name="PCT_NAT_PFT", dim="natpft", target=tgt_nat)

    if "PCT_CFT" in ds2 and "cft" in ds2["PCT_CFT"].dims and "PCT_CROP" in ds2:
        _close_partition(
            ds2,
            var_name="PCT_CFT",
            dim="cft",
            target=ds2["PCT_CROP"].astype(np.float64),
        )

    if "PCT_GLC_MEC" in ds2 and "nglcec" in ds2["PCT_GLC_MEC"].dims and "PCT_GLACIER" in ds2:
        _close_partition(
            ds2,
            var_name="PCT_GLC_MEC",
            dim="nglcec",
            target=ds2["PCT_GLACIER"].astype(np.float64),
        )

    # Topounit area weights can appear under either canonical name.
    for top_var in ("PCT_TOPUNIT", "TopounitFracArea"):
        tgt_top = _full_like_from_partition(ds2, var_name=top_var, dim="topounit", value=100.0)
        if tgt_top is not None:
            _close_partition(ds2, var_name=top_var, dim="topounit", target=tgt_top)

    _close_unit_partition(ds2, left="FSURF", right="FGRD")

    return ds2


def closure_critical_variables(present_vars: Iterable[str]) -> set[str]:
    """Return vars that should be written as float64 to preserve closure."""
    present = set(present_vars)
    critical = {
        "PCT_NAT_PFT",
        "PCT_CFT",
        "PCT_GLC_MEC",
        "PCT_NATVEG",
        "PCT_CROP",
        "PCT_GLACIER",
        "PCT_WETLAND",
        "PCT_LAKE",
        "PCT_URBAN",
        "PCT_TOPUNIT",
        "TopounitFracArea",
        "FSURF",
        "FGRD",
    }
    return {v for v in critical if v in present}