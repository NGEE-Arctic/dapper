from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import xarray as xr
from pathlib import Path
from dapper.surf import schema as SC  # expects REGISTRY (name -> ParDef), ParDef

class SurfaceFile:
    """
    Unified interface for building and editing ELM/ELM-style surface files.

    This class is intentionally minimal for now:

    - Wraps an in-memory xarray.Dataset (self.ds)
    - Knows about the surface-variable registry in dapper.surf.schema
    - Can be constructed either from an existing NetCDF file
      (SurfaceFile.from_netcdf) or, later, from a Domain
      (SurfaceFile.from_domain – currently a stub).

    Parameters are added via `add_params_from_df`, which:
    - creates the named dimension if it does not exist yet, using the
      distinct values of `id_col` from the DataFrame
    - adds / overwrites 1D variables whose names come directly from
      DataFrame column names (except id_col + drop_cols)

    This is deliberately conservative: we only handle parameters that are
    functions of a single "index" dimension (e.g. topounit) here. More
    complex multi-dimensional parameters can get their own helpers later.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> None:
        self.ds: xr.Dataset = ds
        # Default to the global REGISTRY from schema.py if not provided
        self.registry: Dict[str, SC.ParDef] = registry or getattr(SC, "REGISTRY", {})

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_netcdf(
        cls,
        path: Union[str, Path],
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> "SurfaceFile":
        """
        Load an existing surface NetCDF file and wrap it.

        This is the entry point for "edit an existing surface file".
        """
        ds = xr.open_dataset(path)
        return cls(ds=ds, registry=registry)

    @classmethod
    def from_domain(
        cls,
        domain: Any,
        registry: Optional[Dict[str, SC.ParDef]] = None,
    ) -> "SurfaceFile":
        """
        Create a new, *empty* surface dataset for a given Domain.

        NOTE: This is intentionally a light stub for now because we do not
        have the Domain API or full surface-layout spec wired in here.

        Current behaviour:
        - Create an xarray.Dataset with no data variables and only a copy
          of global attributes from the Domain if it exposes `.attrs` or
          `.to_xarray()` / `.ds`.
        - Dimensions and coordinates are expected to be added later via
          `add_params_from_df` (e.g. creating a 'topounit' dimension).

        You are expected to extend this method in Dapper proper to
        construct the base dims/coords you want (e.g. 1×1 lsmlat/lsmlon,
        landunit, column, etc.) from the Domain.
        """
        base_ds: xr.Dataset

        # Best-effort: allow Domain to expose a template Dataset
        if hasattr(domain, "to_xarray"):
            base_ds = domain.to_xarray()
        elif hasattr(domain, "ds") and isinstance(getattr(domain, "ds"), xr.Dataset):
            base_ds = domain.ds.copy()
        else:
            # Fallback: empty dataset with no dims; everything will be
            # created via add_params_from_df.
            base_ds = xr.Dataset()

        # Optionally propagate some global attrs from domain
        attrs: Dict[str, Any] = {}
        for attr_name in ("attrs", "metadata"):
            if hasattr(domain, attr_name):
                maybe_dict = getattr(domain, attr_name)
                if isinstance(maybe_dict, dict):
                    attrs.update(maybe_dict)
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
        drop_cols: Optional[list[str]] = None,
    ) -> None:
        """
        Attach / update 1D parameters along `dim_name` using a DataFrame.

        Parameters
        ----------
        dim_name : str
            Name of the logical dimension (e.g. "topounit", "pft").
        df : pandas.DataFrame or geopandas.GeoDataFrame
            Must contain `id_col` and one column per parameter.
        id_col : str
            Column in `df` containing the IDs. The distinct values, in the
            order they appear, are used as the coordinates for `dim_name`
            if that dimension does not already exist.
        drop_cols : list[str], optional
            Columns to ignore when creating parameters (e.g. "geometry").

        Behaviour
        ---------
        - If `dim_name` does not exist in the dataset, it is created with
          coordinates taken from the distinct `id_col` values.
        - For each parameter column (everything except id_col + drop_cols),
          a 1D variable is created/overwritten with dims=(dim_name,).
        - If an existing variable of the same name already exists, it is
          overwritten in-place.
        """
        import pandas as pd  # local import to avoid hard dependency here

        if id_col not in df.columns:
            raise KeyError(f"id_col '{id_col}' not found in DataFrame.")

        drop_cols = set(drop_cols or [])
        drop_cols.add(id_col)

        # Normalize IDs to string (robust alignment)
        ids_series = df[id_col].astype(str)
        df = df.copy()
        df[id_col] = ids_series

        # Unique IDs in appearance order
        unique_ids = pd.unique(ids_series)

        ds = self.ds

        # Create the dimension if it doesn't exist
        if dim_name not in ds.dims:
            ds = ds.assign_coords({dim_name: unique_ids})
        else:
            # If dim exists, we trust its order; warn if IDs don't align perfectly
            existing_ids = np.asarray(ds.coords[dim_name].values).astype(str)
            missing = [i for i in unique_ids if i not in existing_ids]
            if missing:
                # Minimal, non-fatal warning for now; you may want to make
                # this stricter later.
                print(
                    f"[SurfaceFile.add_params_from_df] WARNING: {len(missing)} "
                    f"IDs from id_col='{id_col}' are not present in existing "
                    f"dim '{dim_name}'. They will be ignored."
                )
            unique_ids = existing_ids  # ensure we only fill existing coords

        # Build a mapping for each parameter column and add/update variables
        for col in df.columns:
            if col in drop_cols:
                continue

            values_by_id = (
                df[[id_col, col]]
                    .dropna(subset=[col])
                    .drop_duplicates(subset=[id_col])
                    .set_index(id_col)[col]
            )

            # Align to dimension coordinates
            coord_ids = np.asarray(ds.coords[dim_name].values).astype(str)
            data = np.full(coord_ids.shape, np.nan, dtype=np.asarray(values_by_id.values).dtype)
            for idx, id_val in enumerate(coord_ids):
                if id_val in values_by_id.index:
                    data[idx] = values_by_id.loc[id_val]

            # Wrap as DataArray and assign
            da = xr.DataArray(data, dims=(dim_name,))
            # If registry knows about this variable, copy some attrs / dtype
            par_def = self.registry.get(col)
            if par_def is not None:
                # dtype
                try:
                    data = data.astype(par_def.dtype)
                    da = da.astype(par_def.dtype)
                except Exception:
                    pass
                # units / long_name from attrs
                attrs = dict(getattr(par_def, "attrs", {}) or {})
                if par_def.units:
                    attrs.setdefault("units", par_def.units)
                if par_def.doc:
                    attrs.setdefault("long_name", par_def.doc)
                if attrs:
                    da = da.assign_attrs(attrs)

            ds[col] = da

        # Store back
        self.ds = ds

    def drop_params(self, names: Union[str, list[str]]) -> None:
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
        ds = self.ds
        new_attrs = dict(ds.attrs or {})
        new_attrs.update(attrs)
        self.ds = ds.assign_attrs(new_attrs)

    # ------------------------------------------------------------------
    # Validation and writing
    # ------------------------------------------------------------------
    def basic_registry_check(self) -> Dict[str, Any]:
        """
        Very lightweight, in-memory registry check.

        Returns a dict with:
        - "unknown": set of variable names not present in the registry
        - "known":   set of variable names present in the registry

        This does *not* enforce required/optional tiers yet; it is just a
        quick sanity check that what you're about to write is at least
        mostly aligned with REGISTRY.
        """
        present = set(self.ds.data_vars.keys())
        known = set(self.registry.keys())
        unknown = present - known
        return {
            "known": present & known,
            "unknown": unknown,
        }

    def validate(
        self,
        strict: bool = False,
        use_external_validator: bool = False,
        validator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional["pd.DataFrame"]:
        """
        Validate the surface Dataset.

        Parameters
        ----------
        strict : bool, default False
            If True and use_external_validator is False, raise if any
            variables are not in the registry.
        use_external_validator : bool, default False
            If True, write the dataset to a temporary NetCDF file and run
            dapper.surf.validate.SurfaceValidator on it. This is heavier
            but leverages the existing point-surface checks.
        validator_kwargs : dict, optional
            Passed through to SurfaceValidator() if use_external_validator
            is True.

        Returns
        -------
        pandas.DataFrame or None
            If use_external_validator=True, returns the validation report
            DataFrame from SurfaceValidator.validate(). Otherwise, returns
            None (and may raise on failure if strict=True).
        """
        import tempfile
        import pandas as pd  # type: ignore

        # First: light registry-level check
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

        # Heavy-weight: delegate to SurfaceValidator on a temp file
        from dapper.surf.validate import SurfaceValidator  # lazy import

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "tmp_surface.nc"
            # We deliberately let xarray choose encodings here; customize
            # later if needed.
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

        Parameters
        ----------
        path : str or Path
            Output path.
        overwrite : bool, default False
            If False and the target exists, raise FileExistsError.
        encoding : dict, optional
            Optional variable-specific encoding dict passed to
            xarray.Dataset.to_netcdf(). If omitted, xarray chooses
            reasonable defaults.

        Returns
        -------
        str
            The string path to the written NetCDF file.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to overwrite.")

        ds = self.ds

        if encoding is None:
            ds.to_netcdf(path)
        else:
            ds.to_netcdf(path, encoding=encoding)

        return str(path)
