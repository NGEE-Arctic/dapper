# dapper/surf/write.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import xarray as xr
from pathlib import Path
from dapper.surf import schema as SC  # expects REGISTRY (name -> VarDef), VarDef

ArrayLike = Union[np.ndarray, "xr.DataArray", float, int]


_NETCDF4_ALLOWED = {
    "blosc_shuffle","complevel","dtype","chunksizes","compression",
    "significant_digits","least_significant_digit","endian","zlib",
    "szip_pixels_per_block","contiguous","szip_coding","shuffle",
    "_FillValue","fletcher32","quantize_mode"
}
# Encodings we sometimes inherit from upstream files that netCDF4 can’t accept
_NETCDF4_STRIP = {"zstd","bzip2","blosc","szip"}  # 'szip' here is the legacy flag, not the nc4 pair

def build_surface_dataset(
    sampled: Dict[str, Any],
    *,
    include: Optional[set[str]] = None,
    drop_non_spatial_arrays: bool = False,
) -> xr.Dataset:
    """
    Turn a sampled dict (from sample_point_values) into a 1x1 ELM surface xarray.Dataset.
    Adds spatial dims back as length-1 and preserves other dims in file order.
    """
    meta = sampled["__meta__"]
    coords_src = sampled.get("__coords__", {})
    lat_dim, lon_dim = meta["lat_dim"], meta["lon_dim"]

    # Prepare coordinate arrays
    coords = {
        lat_dim: (lat_dim, np.array([0], dtype=np.int32)),  # length-1 dims; indices can be dummy
        lon_dim: (lon_dim, np.array([0], dtype=np.int32)),
    }
    # Attach known small-dim coordinates (time, natpft, nlevsoi, etc.) if present
    for dim, vals in coords_src.items():
        if isinstance(vals, np.ndarray):
            coords[dim] = (dim, vals)

    data_vars = {}

    def _is_int_dtype(dtype_str: str) -> bool:
        return dtype_str.startswith(("int","uint"))

    # Build each variable
    for name, spec in sampled.items():
        if name.startswith("__"):  # skip meta/coords
            continue
        if include and name not in include:
            continue

        dims_no_spatial = tuple(spec["dims"])
        orig_dims = tuple(spec["orig_dims"])
        data = spec["data"]
        attrs = spec.get("attrs", {})
        dtype_str = spec.get("dtype", "float32")

        # Optionally skip truly non-spatial arrays
        if drop_non_spatial_arrays and ((meta["lat_dim"] not in orig_dims) and (meta["lon_dim"] not in orig_dims)):
            continue

        # Reconstruct dims by appending spatial dims at the end (ELM convention: ... , lsmlat, lsmlon)
        # Assumption holds for standard ELM files (time/other dims precede spatial).
        target_dims = list(dims_no_spatial) + [lat_dim, lon_dim]

        # Compute target shape: non-spatial shape + (1,1)
        if data.shape == ():  # scalar -> (1,1)
            arr = np.array([[data]], dtype=np.float32)
        else:
            arr = np.asarray(data)
            arr = arr.reshape(list(arr.shape) + [1, 1])

        data_vars[name] = (target_dims, arr.astype(arr.dtype, copy=False), attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add convenience LATIXY/LONGXY variables if desired
    lat_on_grid = sampled["__meta__"]["lat_on_grid"]
    lon_on_grid = sampled["__meta__"]["lon_on_grid"]
    ds["LATIXY"] = xr.DataArray(np.array([[lat_on_grid]], dtype=np.float32), dims=(lat_dim, lon_dim))
    ds["LONGXY"] = xr.DataArray(np.array([[lon_on_grid]], dtype=np.float32), dims=(lat_dim, lon_dim))

    # Global attrs
    ds.attrs.update(sampled["__meta__"].get("global_attrs", {}))
    ds.attrs["history"] = (ds.attrs.get("history","") +
                           f" | dapper.surf.write: built from sampled point at "
                           f"({sampled['__meta__']['lat_on_grid']:.6f}, {sampled['__meta__']['lon_on_grid']:.6f})").strip()

    return ds

def write_surface_nc(ds: xr.Dataset, out_path: str) -> str:
    """
    Write xarray.Dataset to NetCDF with sane encodings:
      - float vars get float32 + _FillValue=nan
      - integer vars keep integer dtype with no FillValue
    """
    enc: Dict[str, dict] = {}
    for v in ds.data_vars:
        if str(ds[v].dtype).startswith(("int","uint")):
            enc[v] = {"_FillValue": None}
        else:
            enc[v] = {"_FillValue": np.float32(np.nan), "dtype": "float32"}
    ds.to_netcdf(out_path, encoding=enc)
    return out_path


class CustomizeError(ValueError):
    """Raised when a customization fails schema/formatting validation."""

# --------- helpers reused across update/add ---------

def _latlon_dim_names(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    lat_candidates = ("lsmlat","lat","latitude","y")
    lon_candidates = ("lsmlon","lon","longitude","x")
    lat = next((d for d in ds.dims if d in lat_candidates), None)
    lon = next((d for d in ds.dims if d in lon_candidates), None)
    return lat, lon

_DIM_ALIASES = {
    "lsmlat": ("lsmlat","lat","latitude","y"),
    "lsmlon": ("lsmlon","lon","longitude","x"),
    "natpft": ("natpft","lsmpft"),
}

def _resolve_dim_name(ds: xr.Dataset, reg_dim: str) -> Optional[str]:
    """Map a registry dim to the actual name used in ds (handles common aliases)."""
    candidates = _DIM_ALIASES.get(reg_dim, (reg_dim,))
    return next((d for d in ds.dims if d in candidates), None)

def _ensure_dataarray(value: ArrayLike, like: xr.DataArray) -> xr.DataArray:
    """
    Convert user 'value' to DataArray broadcastable to 'like' (dims kept).
      - scalar: broadcast to like.dims
      - ndarray: shape==like.shape OR 1D matching a single dim (then broadcast)
      - DataArray: .broadcast_like(like)
    """
    if isinstance(value, xr.DataArray):
        try:
            return value.broadcast_like(like)
        except Exception as e:
            raise CustomizeError(f"value for {like.name!r} not broadcastable to dims {like.dims}: {e}") from e
    if np.isscalar(value):
        return xr.full_like(like, np.asarray(value), dtype=like.dtype)
    if isinstance(value, np.ndarray):
        if value.shape == like.shape:
            return xr.DataArray(value, dims=like.dims, attrs=like.attrs)
        if value.ndim == 1:
            matches = [d for d in like.dims if like.sizes[d] == value.shape[0]]
            if len(matches) == 1:
                tmp = xr.DataArray(value, dims=[matches[0]])
                try:
                    return tmp.broadcast_like(like)
                except Exception as e:
                    raise CustomizeError(f"1D override cannot broadcast to {like.dims}: {e}") from e
        raise CustomizeError(f"ndarray shape {value.shape} not broadcastable to {like.shape}")
    raise CustomizeError(f"Unsupported customization type: {type(value).__name__}")

def _coerce_dtype(da: xr.DataArray, reg_dtype: Optional[str]) -> xr.DataArray:
    """Cast to registry dtype when provided; disallow float↔int switches."""
    if not reg_dtype:
        return da
    actual, want = str(da.dtype), reg_dtype
    if actual == want:
        return da
    a_is_int = actual.startswith(("int","uint"))
    w_is_int = want.startswith(("int","uint"))
    if a_is_int != w_is_int:
        raise CustomizeError(f"dtype mismatch for {da.name!r}: cannot cast {actual} to {want} (int/float switch).")
    try:
        return da.astype(want)
    except Exception as e:
        raise CustomizeError(f"failed dtype cast for {da.name!r}: {actual} -> {want}: {e}") from e

def _preserve_encoding(old: xr.DataArray, new: xr.DataArray) -> xr.DataArray:
    """Carry over attrs+encoding from old var (including _FillValue)."""
    new.attrs = dict(old.attrs)
    new.encoding = dict(old.encoding)
    return new

def _build_template_da_for_new_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    """
    Create a zero-valued template DataArray for a NEW variable using REGISTRY dims/dtype/units,
    aligned to the dataset's actual dim names and sizes. All required dims must already exist.
    """
    if var not in SC.REGISTRY:
        raise CustomizeError(f"unknown variable {var!r}; not found in registry (set strict_registry=False to bypass).")
    spec: SC.VarDef = SC.REGISTRY[var]  # dims, dtype, units, attrs
    reg_dims = list(spec.dims)

    # Map registry dim names -> dataset dim names (handle aliases)
    actual_dims: list[str] = []
    for d in reg_dims:
        resolved = _resolve_dim_name(ds, d)
        if resolved is None:
            raise CustomizeError(f"dataset is missing required dim for {var!r}: {d!r}")
        actual_dims.append(resolved)

    # Build coords and shape from ds
    coords = {d: ds.coords[d] if d in ds.coords else (d, np.arange(ds.sizes[d])) for d in actual_dims}
    shape = tuple(ds.sizes[d] for d in actual_dims)

    # Create template
    arr = np.zeros(shape, dtype=np.float32 if not spec.dtype else np.dtype(spec.dtype))
    da = xr.DataArray(arr, dims=tuple(actual_dims))
    # attrs/encoding from registry
    da.name = var
    da.attrs = dict(spec.attrs or {})
    if spec.units:
        da.attrs["units"] = spec.units
    # sensible default fill for new float vars
    if not str(da.dtype).startswith(("int","uint")):
        da.encoding["_FillValue"] = np.float32(np.nan)
    return da

def _sanitize_netcdf4_encoding(var_enc: dict, dtype) -> dict:
    """Remove unsupported keys/values for netCDF4 engine and normalize compression flags."""
    if not var_enc:
        return {}
    enc = {k: v for k, v in var_enc.items() if v is not None}

    # Drop known unsupported or legacy flags
    for k in list(enc):
        if k in _NETCDF4_STRIP:
            enc.pop(k, None)

    # Normalize generic 'compression' to netCDF4's 'zlib' flag when appropriate
    if "compression" in enc:
        comp = str(enc["compression"]).lower()
        if comp in ("zlib","deflate","gzip","true","1"):
            enc["zlib"] = True
        enc.pop("compression", None)

    # Keep only allowed keys
    enc = {k: v for k, v in enc.items() if k in _NETCDF4_ALLOWED}

    # Ensure _FillValue type matches dtype (avoids netCDF4 dtype errors)
    if "_FillValue" in enc:
        fv = enc["_FillValue"]
        if np.issubdtype(np.asarray([], dtype=dtype).dtype, np.floating):
            # leave NaN or castable float
            try:
                enc["_FillValue"] = np.asarray(fv, dtype=dtype).item()
            except Exception:
                enc.pop("_FillValue", None)
        else:
            # int vars: ensure integer fill (no NaN)
            try:
                enc["_FillValue"] = np.asarray(0 if fv is None or np.isnan(fv) else fv, dtype=dtype).item()
            except Exception:
                enc.pop("_FillValue", None)

    return enc

# ---------------------- PUBLIC ENTRY POINT ----------------------

def customize_surface(
    nc_in: str | Path,
    customizations: Dict[str, Any],
    nc_out: Optional[str | Path] = None,
    *,
    strict_registry: bool = True,
    allow_add: bool = True,
    run_validation: bool = False,
    validator_kwargs: Optional[Dict[str, Any]] = None,
    units_policy: str = "enforce",      # <— default enforce
    engine: str = "netcdf4",            # future-proof, we sanitize netcdf4 above
) -> Tuple[str, Optional["pd.DataFrame"]]:
    """
    Update or add parameters in an existing ELM surface NetCDF (path-only API).

    Parameters
    ----------
    nc_in : str | Path
        Path to existing surface NetCDF.
    customizations : dict
        Mapping of variable -> value OR variable -> spec dict:
          - value can be: scalar, np.ndarray, xr.DataArray (broadcasted)
          - spec dict keys:
                {"value": <required>,
                 "dims":  ["optional dim names for 1D arrays"],
                 "dtype": "optional dtype override (e.g., 'float32')",
                 "units": "optional units override (if not enforced by registry)"}
        Notes:
          * For existing variables, dims are taken from the file and 'dims' is ignored
            (value must be broadcastable to that shape).
          * For NEW variables (not in file):
              - If present in REGISTRY, dims/dtype/units come from REGISTRY.
                All of those dims must already exist in the dataset (sizes are reused).
              - If NOT in REGISTRY and strict_registry=True -> error.
              - If NOT in REGISTRY and strict_registry=False -> you must pass a spec dict
                with 'dims', 'dtype', and 'units'.
    nc_out : str | Path, optional
        Output path; default is '<stem>_custom.nc' next to input.
    strict_registry : bool
        Require variables to exist in schema.REGISTRY. True recommended.
    validate_units : bool
        Ensure file units match registry units (registry ''/'varies' are skipped).
    allow_add : bool
        Permit adding new variables; otherwise only overwrite existing ones.
    run_validation : bool
        If True, run dapper.surf.validate.SurfaceValidator on the written file and return the report.
    validator_kwargs : dict
        Passed to SurfaceValidator(...).

    Returns
    -------
    (out_path, report_df_or_None)

    Raises
    ------
    CustomizeError on shape/dtype/units/dim mismatches.
    """
    import pandas as pd  # only used in return type
    nc_in = str(nc_in)
    ds = xr.open_dataset(nc_in)
    ds_edit = ds.copy()

    lat_dim, lon_dim = _latlon_dim_names(ds)  # detected, not strictly required here

    def _parse_spec(var: str, spec: Any) -> Tuple[Any, Optional[str], Optional[str]]:
        """Return (value, dtype_override, units_override)."""
        if isinstance(spec, dict) and "value" in spec:
            return spec["value"], spec.get("dtype"), spec.get("units")
        return spec, None, None

    for var, spec in customizations.items():
        value, dtype_override, units_override = _parse_spec(var, spec)
        in_file = var in ds_edit

        # ---- Overwrite existing variable ----
        if in_file:
            targ = ds_edit[var]

            # Optional units check vs registry
            if var in SC.REGISTRY:
                reg_units = (SC.REGISTRY[var].units or "").strip()
                file_units = str((targ.attrs or {}).get("units", "")).strip()
                enforce = reg_units and reg_units.lower() not in ("varies",)
                if enforce and file_units and (file_units != reg_units):
                    if units_policy == "enforce":
                        raise CustomizeError(
                            f"units mismatch for {var}: file={file_units!r}, registry={reg_units!r}"
                        )
                    # 'warn' / 'ignore': proceed without mutating attrs

            # Value → DataArray broadcastable to targ
            new_da = _ensure_dataarray(value, like=targ)

            # Dtype coercion (registry or override)
            reg_dtype = dtype_override or (SC.REGISTRY[var].dtype if var in SC.REGISTRY else None)
            new_da = _coerce_dtype(new_da, reg_dtype)
            new_da.name = var
            new_da = _preserve_encoding(targ, new_da)

            ds_edit[var] = new_da
            continue

        # ---- Add new variable ----
        if not allow_add:
            raise CustomizeError(f"{var!r} not in file; set allow_add=True to add new variables.")

        if strict_registry:
            # Use registry dims/dtype/units and dataset dims/coords
            template = _build_template_da_for_new_var(ds_edit, var)
        else:
            # Require user-provided dims/dtype/units (minimal)
            if not (isinstance(spec, dict) and "value" in spec and "dims" in spec and "dtype" in spec and "units" in spec):
                raise CustomizeError(f"adding {var!r} without registry requires spec dict with 'value','dims','dtype','units'")
            dims = tuple(spec["dims"])
            for d in dims:
                if d not in ds_edit.dims:
                    raise CustomizeError(f"dataset missing requested dim {d!r} for new var {var!r}")
            coords = {d: ds_edit.coords[d] if d in ds_edit.coords else (d, np.arange(ds_edit.sizes[d])) for d in dims}
            arr = np.zeros(tuple(ds_edit.sizes[d] for d in dims), dtype=np.dtype(spec["dtype"]))
            template = xr.DataArray(arr, dims=dims, attrs={"units": spec["units"]})
            template.name = var
            if not str(template.dtype).startswith(("int","uint")):
                template.encoding["_FillValue"] = np.float32(np.nan)

        # Value → DataArray broadcastable to template
        new_da = _ensure_dataarray(value, like=template)

        # Dtype coercion
        reg_dtype = dtype_override or (SC.REGISTRY[var].dtype if var in SC.REGISTRY else str(template.dtype))
        new_da = _coerce_dtype(new_da, reg_dtype)
        new_da.name = var

        # Attributes: registry wins, then user overrides; keep encoding
        if var in SC.REGISTRY:
            specv: SC.VarDef = SC.REGISTRY[var]
            attrs = dict(specv.attrs or {})
            if specv.units:
                attrs["units"] = specv.units
            if units_override:
                attrs["units"] = units_override
            new_da.attrs.update(attrs)
        elif units_override:
            new_da.attrs["units"] = units_override
        new_da = _preserve_encoding(template, new_da)

        ds_edit[var] = new_da

    # Decide output path
    if nc_out is None:
        p = Path(nc_in)
        nc_out = str(p.with_name(p.stem + "_custom.nc"))

    # Write with per-var encodings preserved
    enc = {}
    for name in ds_edit.data_vars:
        enc[name] = _sanitize_netcdf4_encoding(dict(ds_edit[name].encoding), ds_edit[name].dtype)
        ds_edit.to_netcdf(nc_out, encoding=enc)

    # Optional: run validation
    report = None
    if run_validation:
        from dapper.surf.validate import SurfaceValidator  # lazy import to avoid cycles
        v = SurfaceValidator(**(validator_kwargs or {}))
        report = v.validate(str(nc_out))

    return str(nc_out), report
