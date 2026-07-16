"""Surface file construction, customization, and validation helpers."""

from typing import Dict, Any, Optional, Tuple, Union, List, Literal
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from dapper.surf import schema as SC
from dapper.surf import sample as SP  # for from_halfdegree_point 
from dapper.geo import sampling  # shared gridded sampler
from dapper.surf.surface_var_specs import SURFACE_VAR_SPECS
from dapper.surf.fraction_closure import normalize_fraction_closure, closure_critical_variables

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

def build_surface_dataset_cellset(
    sampled_list: List[Dict[str, Any]],
    *,
    include: Optional[set[str]] = None,
    drop_non_spatial_arrays: bool = False,
) -> xr.Dataset:
    """
    Build an ELM surface xarray.Dataset for a cellset laid out as (nj=N, ni=1).
    This mirrors your domain writer default of N×1, and keeps spatial dims last.

    Each entry of sampled_list is the dict returned by SurfacePointSampler.sample().
    """
    if not sampled_list:
        raise ValueError("build_surface_dataset_cellset(): sampled_list is empty.")

    meta0 = sampled_list[0]["__meta__"]
    coords_src = sampled_list[0].get("__coords__", {})
    lat_dim, lon_dim = meta0["lat_dim"], meta0["lon_dim"]

    nj = len(sampled_list)
    ni = 1

    coords = {
        lat_dim: (lat_dim, np.arange(nj, dtype=np.int32)),
        lon_dim: (lon_dim, np.arange(ni, dtype=np.int32)),
    }
    for dim, vals in coords_src.items():
        if isinstance(vals, np.ndarray):
            coords[dim] = (dim, vals)

    # Use variable names from the first sample (assume consistent filtering across samples)
    names = [n for n in sampled_list[0].keys() if not n.startswith("__")]
    if include:
        names = [n for n in names if n in include]

    data_vars: Dict[str, Any] = {}

    for name in names:
        spec0 = sampled_list[0][name]
        dims_no_spatial = tuple(spec0["dims"])
        orig_dims = tuple(spec0["orig_dims"])
        attrs = spec0.get("attrs", {})

        if drop_non_spatial_arrays and ((lat_dim not in orig_dims) and (lon_dim not in orig_dims)):
            continue

        # Collect per-cell arrays (all should share the same non-spatial shape)
        per_cell = []
        for smp in sampled_list:
            spec = smp[name]
            data = np.asarray(spec["data"])
            per_cell.append(data)

        # Stack along new lat_dim (nj), then add lon_dim (ni=1)
        stacked = np.stack(per_cell, axis=-1)   # shape: base_shape + (nj,)
        stacked = np.expand_dims(stacked, axis=-1)  # -> base_shape + (nj,1)

        target_dims = list(dims_no_spatial) + [lat_dim, lon_dim]
        data_vars[name] = (target_dims, stacked.astype(stacked.dtype, copy=False), attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # LATIXY/LONGXY: (nj,1)
    lat_on_grid = np.array([s["__meta__"]["lat_on_grid"] for s in sampled_list], dtype=np.float32).reshape(nj, 1)
    lon_on_grid = np.array([s["__meta__"]["lon_on_grid"] for s in sampled_list], dtype=np.float32).reshape(nj, 1)
    ds["LATIXY"] = xr.DataArray(lat_on_grid, dims=(lat_dim, lon_dim))
    ds["LONGXY"] = xr.DataArray(lon_on_grid, dims=(lat_dim, lon_dim))

    # Global attrs: use first sample
    ds.attrs.update(meta0.get("global_attrs", {}))
    ds.attrs["history"] = (ds.attrs.get("history", "") +
                           f" | dapper.surf.write: built from {nj} sampled points").strip()
    return ds

def write_surface_nc(
    ds: xr.Dataset,
    out_path: str,
    *,
    append_attrs: dict | None = None,
    dapper_attrs: dict | None = None,
    add_created_utc: bool = True,
) -> str:
    """Write a surface Dataset to NetCDF with ELM-friendly defaults and merged attributes."""
    
    import datetime as _dt

    # ---- global attrs ----
    ds2 = normalize_fraction_closure(ds)
    merged = dict(ds2.attrs)

    if dapper_attrs:
        for k, v in dict(dapper_attrs).items():
            merged.setdefault(k, v)

    if add_created_utc:
        merged.setdefault("dapper_created_utc", _dt.datetime.utcnow().isoformat() + "Z")

    if append_attrs:
        merged.update(dict(append_attrs))

    ds2.attrs = merged

    closure_critical = closure_critical_variables(ds2.data_vars)
    enc: Dict[str, dict] = {}
    for v in ds2.data_vars:
        # Fill values must match dtype. Most float vars are written as float32,
        # while closure-critical fractions stay float64 to preserve exact sums.
        if ds2[v].dtype.kind == "f":
            if v in closure_critical:
                enc[v] = {"dtype": "float64", "_FillValue": np.float64(-9.96921e36)}
            else:
                enc[v] = {"dtype": "float32", "_FillValue": np.float32(-9.96921e36)}

    ds2.to_netcdf(out_path, encoding=enc)
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
    spec: SC.ParDef = SC.REGISTRY[var]  # dims, dtype, units, attrs
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

def customize_surface(
    src_path: str | Path,
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
    src_path : str | Path
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
    src_path = str(src_path)
    ds = xr.open_dataset(src_path)
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
            specv: SC.ParDef = SC.REGISTRY[var]
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
        p = Path(src_path)
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

def _surface_zonal_agg_policy_from_registry(
    ds_src: xr.Dataset,
    *,
    include: Optional[set[str]],
    exclude: Optional[set[str]],
) -> tuple[dict[str, str], set[str]]:
    """
    Returns:
      - agg_policy: var -> reducer name understood by dapper.geo.zonal
      - derived_vars: vars whose registry agg == "derived"
    """
    include_set = set(include) if include else None
    drop = set(exclude or [])

    # Find registry-derived vars (we will not sample these; we compute them from Domain)
    derived_vars = {v for v, spec in SURFACE_VAR_SPECS.items() if spec.get("agg") == "derived"}

    agg_policy: dict[str, str] = {}

    # Build policy for vars we will actually sample
    for v in ds_src.data_vars:
        if include_set is not None and v not in include_set:
            continue
        if v in drop or v in derived_vars:
            continue

        spec = SURFACE_VAR_SPECS.get(v)

        # If a var isn't in the registry, fall back by dtype (keeps things robust)
        if spec is None:
            kind = ds_src[v].dtype.kind
            agg_policy[v] = "wmode" if kind in {"i", "u", "b"} else "wmean"
            continue

        agg = spec.get("agg")
        if agg is None:
            raise ValueError(f"SURFACE_VAR_SPECS[{v!r}] is missing required key 'agg'")

        if agg == "derived":
            continue
        if agg == "auto":
            kind = ds_src[v].dtype.kind
            agg = "wmode" if kind in {"i", "u", "b"} else "wmean"

        agg_policy[v] = agg

    return agg_policy, derived_vars


class SurfaceFile:
    """
    Unified interface for building and editing ELM/ELM-style surface files.

    - Wraps an in-memory xarray.Dataset (``self.ds``).
    - Knows about the surface-variable registry (``dapper.surf.schema``; ``SC.REGISTRY``).
    - Can be constructed from:

      * an existing NetCDF path (``from_netcdf``)
      * a point sampled from the global half-degree surface (``from_halfdegree_point``)
      * a Domain (``from_domain``); currently a light stub you can extend

    Parameters are added via ``add_params_from_df``. That method:

    - creates the named dimension if it does not exist yet, using the distinct
      values of ``id_col`` from the DataFrame
    - adds/overwrites 1D variables whose names come directly from DataFrame
      column names (except ``id_col`` and ``drop_cols``)
    """

    def __init__(
        self,
        ds: xr.Dataset,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> None:
        self.ds: xr.Dataset = ds
        # Fall back to global REGISTRY
        self.registry: Dict[str, SC.ParDef] = registry or SC.REGISTRY

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_netcdf(
        cls,
        path: Union[str, Path],
        registry: Optional[Dict[str, SC.ParDef]] = None,
        decode_times: bool = True,
    ) -> "SurfaceFile":
        """
        Workflow A: wrap an existing surface file for editing.
        """
        ds = xr.open_dataset(path, decode_times=decode_times)
        return cls(ds=ds, registry=registry)

    @classmethod
    def from_halfdegree_point(
        cls,
        lat: float,
        lon: float,
        *,
        src_path: str | Path,
        decode_times: bool = True,
        chunks: Optional[Dict[str, int]] = None,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> "SurfaceFile":
        """
        Sample the global half-degree surface at (lat, lon) and return a 1x1 SurfaceFile.
        Uses dapper.geo.sampling.sample_gridded_dataset_points.
        """
        ds_src = xr.open_dataset(src_path, decode_times=decode_times, chunks=chunks)

        df_loc = pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)], "weight": [1.0]})

        ds_out = sampling.sample_gridded_dataset_points(
            ds_src,
            df_loc,
            lat_col="lat",
            lon_col="lon",
            vars_include=sorted(include) if include else None,
            vars_drop=sorted(exclude) if exclude else None,
            lon_wrap="auto",
            method="nearest",
        )
        return cls(ds=ds_out, registry=registry)

    @classmethod
    def from_domain(
        cls,
        domain: Any,
        src_path: str | Path,
        *,
        decode_times: bool = True,
        chunks: Optional[Dict[str, int]] = None,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
        registry: Optional[Dict[str, SC.ParDef]] = None,
        attach_topounits: bool = True,
        sampling_method: Literal["nearest", "zonal"] = "nearest",
        lon_wrap: sampling.LonWrap = "auto",
        agg_policy: dict[str, str] | None = None,
    ) -> "SurfaceFile":
        """Sample a global surface Dataset for a single-run Domain and return a SurfaceFile."""
        
        if getattr(domain, "mode", None) == "sites":
            raise ValueError(
                "SurfaceFile.from_domain expects a single-run Domain (mode='cellset'). "
                "Use SurfaceFile.export(domain, ...) for mode='sites'."
            )

        # Use a lon/lat-guaranteed Domain view for everything in this method
        dom = domain.ensure_cells_lon_lat()
        df_loc = dom.to_df_loc()

        ds_src = xr.open_dataset(src_path, decode_times=decode_times, chunks=chunks)

        if sampling_method == "nearest":
            ds_out = sampling.sample_gridded_dataset_points(
                ds_src,
                df_loc,
                lat_col="lat",
                lon_col="lon",
                vars_include=sorted(include) if include else None,
                vars_drop=sorted(exclude) if exclude else None,
                lon_wrap=lon_wrap,
                method="nearest",
            )

        elif sampling_method == "zonal":
            from dapper.geo import zonal  # lazy import

            # Require polygon-like cells
            if dom.cells.geometry.geom_type.isin(["Point"]).all():
                raise ValueError(
                    "sampling_method='zonal' requires polygon (or at least non-point) cell geometries "
                    "in Domain.cells. Your Domain.cells are Points."
                )

            targets = dom.cells[["gid", "geometry"]].copy()
            if targets.crs is None:
                targets = targets.set_crs("EPSG:4326")
            else:
                targets = targets.to_crs("EPSG:4326")

            base_policy, derived_vars = _surface_zonal_agg_policy_from_registry(
                ds_src, include=include, exclude=exclude
            )
            if agg_policy:
                base_policy.update(dict(agg_policy))

            # Never sample derived vars
            for dv in derived_vars:
                base_policy.pop(dv, None)

            vars_drop = set(exclude or []) | set(derived_vars)
            vars_include = None if include is None else sorted(set(include) - set(derived_vars))

            zw = zonal.intersect_weights_rectilinear(
                ds_src,
                targets,
                lon_wrap=lon_wrap,
            )

            ds_out = zonal.sample_gridded_dataset_polygons(
                ds_src,
                targets,
                vars_include=vars_include,
                vars_drop=sorted(vars_drop),
                agg_policy=base_policy,
                lon_wrap=lon_wrap,
                weights=zw,
            )
            # Inject derived vars if requested
            include_set = set(include) if include else None
            want_derived = set(derived_vars) if include_set is None else (set(derived_vars) & include_set)
            want_derived -= set(exclude or [])

            if want_derived:
                spec = sampling.infer_latlon_spec(ds_src, lon_wrap=lon_wrap)
                lat_dim, lon_dim = spec.lat_dim, spec.lon_dim
                n = len(targets)

                # Ensure dims exist even if only derived requested
                if lat_dim not in ds_out.dims or lon_dim not in ds_out.dims:
                    ds_out = ds_out.expand_dims(
                        {lat_dim: np.arange(n, dtype=np.int32), lon_dim: np.arange(1, dtype=np.int32)}
                    )

                if "LATIXY" in want_derived:
                    arr = dom.cells["lat"].to_numpy(dtype=np.float64).reshape(n, 1).astype(np.float32)
                    attrs = dict(ds_src["LATIXY"].attrs) if "LATIXY" in ds_src else {"units": "degrees_north"}
                    ds_out["LATIXY"] = xr.DataArray(arr, dims=(lat_dim, lon_dim), attrs=attrs)

                if "LONGXY" in want_derived:
                    arr = dom.cells["lon"].to_numpy(dtype=np.float64).reshape(n, 1).astype(np.float32)
                    attrs = dict(ds_src["LONGXY"].attrs) if "LONGXY" in ds_src else {"units": "degrees_east"}
                    ds_out["LONGXY"] = xr.DataArray(arr, dims=(lat_dim, lon_dim), attrs=attrs)

                if "AREA" in want_derived:
                    ea = zw.equal_area_crs
                    area_m2 = targets.to_crs(ea).geometry.area.to_numpy(dtype=np.float64)

                    # Preserve upstream AREA units if present
                    units = (ds_src["AREA"].attrs.get("units") if "AREA" in ds_src else None) or "km^2"
                    if "km" in units:
                        arr = (area_m2 / 1e6).reshape(n, 1).astype(np.float32)
                        attrs = dict(ds_src["AREA"].attrs) if "AREA" in ds_src else {"units": "km^2", "long_name": "area"}
                    else:
                        arr = area_m2.reshape(n, 1).astype(np.float32)
                        attrs = dict(ds_src["AREA"].attrs) if "AREA" in ds_src else {"units": "m2", "long_name": "area"}

                    ds_out["AREA"] = xr.DataArray(arr, dims=(lat_dim, lon_dim), attrs=attrs)

            ds_out.attrs["dapper_surface_sampling_method"] = "zonal"

        else:
            raise ValueError(f"Unknown sampling_method={sampling_method!r}")

        sf = cls(ds=ds_out, registry=registry)

        if attach_topounits and hasattr(dom, "has_topounits") and dom.has_topounits():
            sf._attach_topounits_from_domain(dom)

        return sf

    def _attach_topounits_from_domain(self, domain: Any) -> None:
        """
        Attach Domain.topounits as a 1D parameter table on the surface dataset.

        This is intentionally "metadata-style" (1D along topounit dim). It does NOT
        attempt to reshape existing ELM vars to include topounits.
        """
        gdf = domain.topounits.copy()
        dim_name = getattr(domain, "topounits_dim_name", "topounit")
        id_col = getattr(domain, "topounits_id_col", "topounit_id")
        gid_col = getattr(domain, "topounits_gid_col", "gid")

        if id_col not in gdf.columns:
            raise KeyError(f"Domain.topounits missing id column '{id_col}'")

        # If topounit ids collide across gids, make them unique using gid prefix
        if gdf[id_col].astype(str).duplicated().any() and (gid_col in gdf.columns):
            gdf[id_col] = gdf[gid_col].astype(str) + "_" + gdf[id_col].astype(str)

        drop_cols = ["geometry"]
        if gid_col in gdf.columns:
            drop_cols.append(gid_col)

        self.add_params_from_df(
            dim_name=dim_name,
            df=gdf,
            id_col=id_col,
            drop_cols=drop_cols,
        )

    @classmethod
    def export(
        cls,
        domain: Any,
        *,
        out_dir: str | Path,
        src_path: str | Path,
        filename: str = "surfdata.nc",
        overwrite: bool = False,
        append_attrs=None,
        decode_times: bool = True,
        chunks: Optional[Dict[str, int]] = None,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
        registry: Optional[Dict[str, SC.ParDef]] = None,
        attach_topounits: bool = True,
        sampling_method: Literal["nearest", "zonal"] = "nearest",
        lon_wrap: sampling.LonWrap = "auto",
        agg_policy: dict[str, str] | None = None,
        validate: bool = False,
        validator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Export surface file(s) for a Domain.

        Returns: dict[run_id, path]

        - domain.mode='cellset': one file in out_dir
        - domain.mode='sites'  : one file per site in out_dir/<gid>/
        """
        from dapper.domains.domain import Domain  # local import to avoid circular deps

        if not isinstance(domain, Domain):
            raise TypeError("SurfaceFile.export() expects a dapper.domains.Domain instance.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, Path] = {}

        for run_id, run_dom in domain.iter_runs():
            run_id = str(run_id)

            run_out_dir = (out_dir / run_id) if domain.mode == "sites" else out_dir
            run_out_dir.mkdir(parents=True, exist_ok=True)

            out_path = run_out_dir / filename
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists (overwrite=False).")

            sf = cls.from_domain(
                run_dom,
                src_path=src_path,
                decode_times=decode_times,
                chunks=chunks,
                include=include,
                exclude=exclude,
                registry=registry,
                attach_topounits=False,  # avoid duplicate attachment; handled below
                sampling_method=sampling_method,
                lon_wrap=lon_wrap,
                agg_policy=agg_policy,
            )

            # Attach topounit parameters (and TopounitFracArea) exactly once
            if attach_topounits and getattr(run_dom, "topounits", None) is not None and run_dom.topounits is not None:
                sf.add_topounits_from_domain(run_dom)

            if append_attrs:
                sf.ds.attrs.update(dict(append_attrs))

            write_surface_nc(sf.ds, str(out_path))

            if validate:
                from dapper.surf.validate import SurfaceValidator
                vkw = dict(validator_kwargs or {})

                # Allow N×1 (cellset) surfaces without forcing len==1
                if int(sf.ds.sizes.get("lsmlat", 1)) > 1:
                    vkw.setdefault("require_point_dims", False)

                SurfaceValidator(**vkw).validate(str(out_path))

            outputs[run_id] = out_path

        return outputs

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add_params_from_df(
        self,
        dim_name: str,
        df,
        id_col: str,
        *,
        drop_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Attach / update 1D parameters along `dim_name` using a DataFrame.

        Parameters
        ----------
        dim_name : str
            Logical dimension name (e.g. "topounit", "pft").
        df : pandas.DataFrame or geopandas.GeoDataFrame
            Must contain `id_col` and one column per parameter.
        id_col : str
            Column containing the IDs. The distinct values (as strings),
            in order of appearance, are used as coordinates if `dim_name`
            does not already exist.
        drop_cols : list[str], optional
            Columns to ignore as parameters (e.g. "geometry").
        """
        import pandas as pd  # local to avoid a hard dependency at import time

        if id_col not in df.columns:
            raise KeyError(f"id_col '{id_col}' not found in DataFrame.")

        drop_cols = set(drop_cols or [])
        drop_cols.add(id_col)

        # Normalize ids to string for robust alignment
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        ids_series = df[id_col]
        unique_ids = pd.unique(ids_series)

        ds = self.ds

        # Create dim if missing, else align to existing coord order
        if dim_name not in ds.dims:
            ds = ds.assign_coords({dim_name: unique_ids})
        else:
            existing_ids = np.asarray(ds.coords[dim_name].values).astype(str)
            # Warn if df has ids not in existing dimension
            missing = [i for i in unique_ids if i not in existing_ids]
            if missing:
                print(
                    f"[SurfaceFile.add_params_from_df] WARNING: "
                    f"{len(missing)} ids from '{id_col}' not present in dim '{dim_name}'; ignored."
                )
        coord_ids = np.asarray(ds.coords[dim_name].values).astype(str)

        # For each parameter column, align to dim coord and write
        for col in df.columns:
            if col in drop_cols:
                continue

            # Map id -> value (one per id)
            series = (
                df[[id_col, col]]
                .dropna(subset=[col])
                .drop_duplicates(subset=[id_col])
                .set_index(id_col)[col]
            )

            # Choose dtype from series if non-empty, else float.
            # IMPORTANT: if the dtype is integer/unsigned/bool, promote to float
            # so that we can safely use NaN as a fill value without casting
            # warnings or errors.
            if len(series) > 0:
                dtype = np.asarray(series.values).dtype
            else:
                dtype = np.float64

            if getattr(dtype, "kind", "") in ("i", "u", "b"):
                dtype = np.float64

            data = np.full(coord_ids.shape, np.nan, dtype=dtype)
            for idx, id_val in enumerate(coord_ids):
                if id_val in series.index:
                    data[idx] = series.loc[id_val]

            da = xr.DataArray(data, dims=(dim_name,))

            # If registry knows this var, add units / long_name
            par_def = self.registry.get(col)
            if par_def is not None:
                attrs = dict(getattr(par_def, "attrs", {}) or {})
                if par_def.units:
                    attrs.setdefault("units", par_def.units)
                if par_def.doc:
                    attrs.setdefault("long_name", par_def.doc)
                if attrs:
                    da = da.assign_attrs(attrs)

            ds[col] = da

        self.ds = ds

    def add_topounits_from_domain(
        self,
        domain,
        *,
        gid_col: str = "gid",
        id_col: str = "topounit_id",
        pct_col: str = "TopounitPctOfCell",
        dim_name: str = "topounit",
        pct_var_name: str = "TopounitFracArea",
    ) -> None:
        """
        Attach topounits + per-cell weights to the surface dataset.

        Expects domain.topounits to exist and contain:
        - gid_col (links topounit -> cell gid)
        - id_col  (unique id per topounit across the whole run)
        - pct_col (percent of the parent cell; sums to ~100 per gid)
        """
        import pandas as pd
        import numpy as np
        import xarray as xr

        if getattr(domain, "topounits", None) is None or domain.topounits is None:
            raise ValueError("Domain has no topounits attached.")

        topos = domain.topounits.copy()
        for c in (gid_col, id_col, pct_col):
            if c not in topos.columns:
                raise KeyError(f"topounits is missing required column '{c}'")

        # Ensure strings for ids
        topos[gid_col] = topos[gid_col].astype(str)
        topos[id_col] = topos[id_col].astype(str)

        # Determine spatial dims in this surf dataset
        ds = self.ds
        lat_dim = "lsmlat" if "lsmlat" in ds.dims else None
        lon_dim = "lsmlon" if "lsmlon" in ds.dims else None
        if lat_dim is None or lon_dim is None:
            raise ValueError("Surface dataset is missing expected spatial dims (lsmlat/lsmlon).")

        nj = int(ds.sizes[lat_dim])
        ni = int(ds.sizes[lon_dim])
        if ni != 1:
            raise NotImplementedError("Topounit mapping currently assumes (nj=N, ni=1) layout for cellsets.")

        # Domain cell order must match surf layout order
        df_loc = domain.to_df_loc()
        if len(df_loc) != nj:
            raise ValueError(f"Domain has {len(df_loc)} cells but surf dataset has {nj} {lat_dim} entries.")

        gid_order = df_loc["gid"].astype(str).tolist()
        gid_to_j = {gid: j for j, gid in enumerate(gid_order)}

        # Create / align the topounit dimension
        top_ids = pd.unique(topos[id_col]).astype(str)
        top_ids = list(top_ids)

        # Add 1D topounit parameters (all non-geometry, non-(gid/id/pct) columns)
        drop = {"geometry", gid_col, id_col, pct_col}
        param_cols = [c for c in topos.columns if c not in drop]
        if param_cols:
            df_params = topos[[id_col] + param_cols].drop_duplicates(subset=[id_col]).copy()
            self.add_params_from_df(dim_name=dim_name, df=df_params, id_col=id_col, drop_cols=["geometry"] if "geometry" in df_params.columns else None)

        # Build pct mapping array: (topounit, nj, ni)
        pct = np.zeros((len(top_ids), nj, ni), dtype=np.float32)
        id_to_k = {tid: k for k, tid in enumerate(top_ids)}

        for gid, grp in topos.groupby(gid_col):
            if gid not in gid_to_j:
                raise ValueError(f"topounits contains gid={gid!r} not present in domain cells.")
            j = gid_to_j[gid]

            vals = grp[pct_col].to_numpy(dtype=np.float64)
            s = float(np.nansum(vals))
            if not np.isfinite(s) or s <= 0:
                raise ValueError(f"Topounit pct weights for gid={gid} are invalid (sum={s}).")
            # normalize to 1.0 (decimal fraction) just in case
            vals = 1.0 * (vals / s)

            for tid, v in zip(grp[id_col].astype(str).tolist(), vals):
                k = id_to_k[tid]
                pct[k, j, 0] = float(v)

        # Install the topounit coord if needed
        if dim_name not in ds.dims:
            ds = ds.assign_coords({dim_name: (dim_name, np.asarray(top_ids, dtype=object))})
        else:
            # If already exists, ensure ids match exactly
            existing = [str(x) for x in ds[dim_name].values.tolist()]
            if existing != top_ids:
                raise ValueError(f"Existing {dim_name} coord does not match topounit ids from domain.")

        ds[pct_var_name] = xr.DataArray(pct, dims=(dim_name, lat_dim, lon_dim))
        ds[pct_var_name].attrs.update({"long_name": "fraction of gridcell area in each topounit", "units": "unitless"})

        # Expand topounit-indexed variables that exist in ds but currently lack the
        # topounit dimension.  All topounits in a grid cell inherit the parent cell's
        # distribution uniformly (per-topounit differentiation is a future extension).
        top_coord = ds.coords[dim_name]
        n_top = len(top_ids)
        for _var_name, _spec in SURFACE_VAR_SPECS.items():
            if _var_name not in ds:
                continue
            _spec_dims = [d.strip() for d in _spec.get("dims", "").split(",")]
            if dim_name not in _spec_dims:
                continue
            _da = ds[_var_name]
            if dim_name in _da.dims:
                continue  # already has the topounit dim
            # Repeat identical values across all topounits
            _expanded = xr.concat([_da] * n_top,
                                   dim=xr.DataArray(top_coord.values, dims=[dim_name], name=dim_name))
            _expanded[dim_name] = top_coord
            # Reorder dims to match spec order (topounit first, spatial last)
            _existing = set(_expanded.dims)
            _ordered = [d for d in _spec_dims if d in _existing]
            _extra = [d for d in _expanded.dims if d not in _ordered]
            ds[_var_name] = _expanded.transpose(*_ordered, *_extra)

        self.ds = ds

    def drop_params(self, names: Union[str, List[str]]) -> None:
        """
        Drop one or more data variables from the surface dataset.
        """
        if isinstance(names, str):
            names = [names]
        self.ds = self.ds.drop_vars(names, errors="ignore")

    def set_global_attrs(self, **attrs: Any) -> None:
        """
        Update global attributes on the underlying Dataset.
        """
        if not attrs:
            return
        new_attrs = dict(self.ds.attrs or {})
        new_attrs.update(attrs)
        self.ds = self.ds.assign_attrs(new_attrs)

    def resize_dim(
        self,
        dim_name: str,
        new_size: int,
        *,
        fill_value: float = np.nan,
    ) -> None:
        """
        Generic "change dimensionality" helper (e.g. nlevsoi 10 → 15).

        - If new_size < old_size: truncate all vars using that dim.
        - If new_size > old_size: pad with `fill_value`.

        This is intentionally generic; you can wrap ELM-specific logic
        (e.g. updating 'nlevsoi' scalar) on top of it.
        """
        ds = self.ds
        if dim_name not in ds.dims:
            raise KeyError(f"dimension {dim_name!r} not found in dataset.")

        old_size = ds.dims[dim_name]
        if new_size == old_size:
            return

        # Coords: if missing, synthesize an index
        coord = ds.coords.get(
            dim_name,
            xr.DataArray(np.arange(old_size), dims=(dim_name,)),
        )

        if new_size < old_size:
            new_coord = coord.isel({dim_name: slice(0, new_size)})
        else:
            extra = xr.DataArray(
                np.arange(old_size, new_size),
                dims=(dim_name,),
            )
            new_coord = xr.concat([coord, extra], dim=dim_name)

        ds = ds.assign_coords({dim_name: new_coord})

        # Adjust all vars that use this dim
        for v in list(ds.data_vars):
            if dim_name not in ds[v].dims:
                continue
            da = ds[v]
            axis = da.dims.index(dim_name)
            if new_size < old_size:
                ds[v] = da.isel({dim_name: slice(0, new_size)})
            else:
                pad_shape = list(da.shape)
                pad_shape[axis] = new_size - old_size
                pad = xr.DataArray(
                    np.full(pad_shape, fill_value, dtype=da.dtype),
                    dims=da.dims,
                )
                ds[v] = xr.concat([da, pad], dim=dim_name)

        self.ds = ds

    def set_scalar(self, name: str, value: ArrayLike) -> None:
        """
        Convenience for setting scalar parameters like nlevsoi, numrad, etc.
        """
        self.ds[name] = xr.DataArray(value)

    # ------------------------------------------------------------------
    # Validation and writing
    # ------------------------------------------------------------------
    def basic_registry_check(self) -> Dict[str, set[str]]:
        """
        Quick registry sanity check.

        Returns
        -------
        {
          "known":   set of vars present in REGISTRY,
          "unknown": set of vars NOT present in REGISTRY,
        }
        """
        present = set(self.ds.data_vars.keys())
        known = set(self.registry.keys())
        return {
            "known": present & known,
            "unknown": present - known,
        }

    def validate(
        self,
        strict: bool = False,
        use_external_validator: bool = False,
        validator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Validate the surface Dataset.

        strict=False, use_external_validator=False:
            - only run basic_registry_check; print a warning for unknown vars.
        use_external_validator=True:
            - write to a temporary file and run SurfaceValidator on it;
              return the pandas.DataFrame report.
        """
        reg_info = self.basic_registry_check()
        if reg_info["unknown"]:
            msg = (
                "[SurfaceFile.validate] Variables not present in registry: "
                + ", ".join(sorted(reg_info["unknown"]))
            )
            if strict and not use_external_validator:
                raise ValueError(msg)
            else:
                print("WARNING:", msg)

        if not use_external_validator:
            return None

        # Full validation path: serialize to temp file and run SurfaceValidator
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "tmp_surface.nc"
            self.ds.to_netcdf(tmp_path)
            from dapper.surf.validate import SurfaceValidator
            v = SurfaceValidator(**(validator_kwargs or {}))
            report = v.validate(str(tmp_path))

        return report

    def to_netcdf(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
        encoding: Optional[Dict[str, Dict[str, Any]]] = None,
        append_attrs: dict | None = None,
        dapper_attrs: dict | None = None,
        add_created_utc: bool = True,
    ) -> str:
        """Write this SurfaceFile to disk as NetCDF."""
        
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")

        if encoding is None:
            # keep the surface-specific encoding behavior
            return write_surface_nc(
                self.ds,
                str(path),
                append_attrs=append_attrs,
                dapper_attrs=dapper_attrs,
                add_created_utc=add_created_utc,
            )

        # If caller supplied encoding, still merge attrs in a non-destructive way.
        import datetime as _dt
        ds2 = self.ds.copy(deep=False)
        merged = dict(ds2.attrs)

        if dapper_attrs:
            for k, v in dict(dapper_attrs).items():
                merged.setdefault(k, v)

        if add_created_utc:
            merged.setdefault("dapper_created_utc", _dt.datetime.utcnow().isoformat() + "Z")

        if append_attrs:
            merged.update(dict(append_attrs))

        ds2.attrs = merged
        ds2.to_netcdf(path, encoding=encoding)
        return str(path)
