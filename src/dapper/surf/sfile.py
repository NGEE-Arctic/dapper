from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import xarray as xr
from pathlib import Path

from dapper.surf import schema as SC  # expects REGISTRY (name -> ParDef), ParDef
from dapper.surf import sample as SP

from dapper.utils.pathing import SURFDATA_HALFDEGREE_TOP

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


class SurfaceFile:
    """
    Unified interface for building and editing ELM/ELM-style surface files.

    - Wraps an in-memory xarray.Dataset (self.ds).
    - Knows about the surface-variable registry in dapper.surf.schema (SC.REGISTRY).
    - Can be constructed from:
        * an existing NetCDF path (from_netcdf)
        * a point sampled from the global half-degree surface (from_halfdegree_point)
        * a Domain (from_domain) – currently a light stub you can extend.

    Parameters are added via `add_params_from_df`, which:
    - creates the named dimension if it does not exist yet, using the
      distinct values of `id_col` from the DataFrame
    - adds / overwrites 1D variables whose names come directly from
      DataFrame column names (except id_col + drop_cols)
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
        nc_in: Union[str, Path] = SURFDATA_HALFDEGREE_TOP,
        *,
        decode_times: bool = True,
        chunks: Optional[Dict[str, int]] = None,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> "SurfaceFile":
        """
        Workflow B (point version): sample the global half-degree surface at
        (lat, lon) and build a 1x1 surface Dataset, then wrap it.

        This is essentially:
            sampled = sample_point_values(...)
            ds = build_surface_dataset(sampled, ...)
        but returned as a SurfaceFile.
        """
        sampled = SP.sample_point_values(
            nc_in=nc_in,
            lat=lat,
            lon=lon,
            decode_times=decode_times,
            chunks=chunks,
            include=include,
            exclude=exclude,
        )
        ds = build_surface_dataset(sampled)
        return cls(ds=ds, registry=registry)

    @classmethod
    def from_domain(
        cls,
        domain: Any,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> "SurfaceFile":
        """
        Workflow B (general): create a new surface dataset from a Domain.

        Current behaviour is intentionally minimal: we just create an empty
        Dataset and copy any attrs from the Domain. You can extend this later
        to:
          - set up lsmlat/lsmlon/landunit/column dims from the Domain
          - optionally sample a global parameter file per-domain element.
        """
        base_ds = xr.Dataset()

        # Optionally propagate some global attrs from the Domain
        attrs: Dict[str, Any] = {}
        for attr_name in ("attrs", "metadata"):
            if hasattr(domain, attr_name):
                maybe = getattr(domain, attr_name)
                if isinstance(maybe, dict):
                    attrs.update(maybe)
        if attrs:
            base_ds = base_ds.assign_attrs(attrs)

        return cls(ds=base_ds, registry=registry)

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
            v = SurfaceValidator(**(validator_kwargs or {}))
            report = v.validate(str(tmp_path))

        return report

    def to_netcdf(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
        encoding: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        Write the surface Dataset to NetCDF.

        If encoding is omitted, xarray chooses defaults. If you want to
        reuse your existing write_surface_nc encodings, you can wire that
        in later.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to overwrite.")

        if encoding is None:
            self.ds.to_netcdf(path)
        else:
            self.ds.to_netcdf(path, encoding=encoding)

        return str(path)
