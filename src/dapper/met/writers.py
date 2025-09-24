# dapper/met/writers.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from netCDF4 import Dataset

# --------------------------- chunking helper ---------------------------

def _infer_dt_hours(dtime_vals, dtime_units: str) -> float:
    """Infer timestep (hours) from numeric DTIME and CF-like units string."""
    arr = np.asarray(dtime_vals, dtype=float)
    if arr.size <= 1:
        return 1.0
    dt_raw = float(np.median(np.diff(arr)))
    units = (dtime_units or "").lower()
    if "day" in units:
        return max(1e-12, dt_raw * 24.0)
    if "hour" in units:
        return max(1e-12, dt_raw)
    # Fallback: assume DTIME already in hours
    return max(1e-12, dt_raw)


def _dtype_nbytes(dtype) -> int:
    """Resolve itemsize from common dtype spellings (e.g., 'i2', 'int16', np.int16)."""
    if isinstance(dtype, str):
        try:
            return np.dtype(dtype).itemsize
        except Exception:
            # map some common aliases
            if dtype in ("i2", "int16", "short"):
                return 2
            if dtype in ("i4", "int32", "int"):
                return 4
            if dtype in ("f4", "float32"):
                return 4
            if dtype in ("f8", "float64"):
                return 8
            return 2
    return np.dtype(dtype).itemsize


def _compute_auto_chunks(
    *,
    dims: tuple[str, ...],
    dim_lengths: dict[str, int],
    dtype,
    write_pattern: str,
    dtime_name: str,
    dtime_vals,
    dtime_units: str,
    target_mb: float = 1.5,
    days_per_chunk: float = 28.0,
) -> tuple[int, ...]:
    """
    Heuristic default chunking suited to your write pattern.

    Rules of thumb:
      - by_site : keep site axis at 1; grow time until target_mb (e.g., (1, t_chunk))
      - by_cell : keep lat/lon at 1; grow time until target_mb (e.g., (t_chunk,1,1))
      - by_time : keep time at 1; grow the rest (rare for your flow)

    Always respects dimension extents. Uses DTIME cadence + `days_per_chunk` to seed t_chunk.
    """
    dims = tuple(dims)
    # Build initial chunk = 1 along all dims
    chunks = [1] * len(dims)

    # Identify axes
    if dtime_name not in dims:
        raise ValueError(f"dtime_name '{dtime_name}' not found in dims {dims}")
    t_axis = dims.index(dtime_name)
    nt = int(dim_lengths[dtime_name])

    # Timestep inference
    dt_hours = _infer_dt_hours(dtime_vals, dtime_units)
    steps_per_day = max(1.0, 24.0 / dt_hours)

    # Seed t_chunk from cadence * days_per_chunk
    t_seed = int(max(1, min(nt, round(days_per_chunk * steps_per_day))))

    # Lock certain axes to 1 depending on pattern
    pattern = (write_pattern or "").lower()
    if pattern == "by_site":
        # Keep site axis at 1 if present
        if "n" in dims:
            chunks[dims.index("n")] = 1
    elif pattern == "by_cell":
        # Keep lat/lon at 1 if present
        if "lat" in dims:
            chunks[dims.index("lat")] = 1
        if "lon" in dims:
            chunks[dims.index("lon")] = 1
    elif pattern == "by_time":
        # Keep time at 1; others can grow later (we still compute a t_chunk but won’t use it)
        pass
    else:
        # Unknown pattern: default to keeping non-time dims at 1
        pass

    # Bytes budget
    elem_bytes = _dtype_nbytes(dtype)
    target_bytes = int(max(1, target_mb * 1024 * 1024))

    # Product of non-time chunk sizes
    def prod(xs):
        p = 1
        for v in xs:
            p *= int(max(1, v))
        return p

    other_prod = prod(chunks[:t_axis] + chunks[t_axis+1:])  # should be 1 for our patterns
    other_bytes = max(elem_bytes * other_prod, elem_bytes)

    # Start with t_seed
    t_chunk = int(max(1, min(nt, t_seed)))
    cur_bytes = other_bytes * t_chunk

    if cur_bytes > target_bytes:
        # Shrink time chunk
        shrink = int(np.ceil(cur_bytes / target_bytes))
        t_chunk = max(1, t_chunk // max(1, shrink))
    else:
        # Grow time chunk up to the budget
        if other_bytes < target_bytes:
            grow_limit = int(target_bytes // other_bytes)
            if grow_limit > 0:
                t_chunk = int(max(t_chunk, min(nt, grow_limit)))

    # Apply final t_chunk (unless by_time)
    if pattern != "by_time":
        chunks[t_axis] = max(1, min(nt, t_chunk))
    else:
        chunks[t_axis] = 1  # explicit

    # Clamp each chunk to its dim length, ensure >=1
    for i, name in enumerate(dims):
        chunks[i] = int(max(1, min(int(dim_lengths[name]), int(chunks[i]))))

    return tuple(chunks)

# --------------------------- initialize / append ---------------------------

def initialize_met_netcdf(
    *,
    path_nc,
    var_name: str,
    dims: tuple[str, ...],
    dim_lengths: dict[str, int],
    dtime_name: str,
    dtime_vals,
    dtime_units: str,
    calendar: str,
    coord_specs: list[dict],
    add_offset: float,
    scale_factor: float,
    dtype="i2",
    fill_value=32767,
    chunks: tuple[int, ...] | None,
    write_pattern: str = "by_site",
    append_attrs: dict | None = None,
    nc_format: str = "NETCDF4_CLASSIC",
    zlib: bool = True,
    shuffle: bool = True,
    complevel: int = 1,
):
    """
    Create a NetCDF file with:
      - provided dims
      - numeric DTIME coord
      - site/grid coords from coord_specs
      - packed int var with add_offset/scale_factor

    If chunks is None, uses _compute_auto_chunks(...) tuned to write_pattern.
    """
    path_nc = Path(path_nc)
    path_nc.parent.mkdir(parents=True, exist_ok=True)

    # Auto-chunk if not provided
    if chunks is None:
        chunks = _compute_auto_chunks(
            dims=dims,
            dim_lengths=dim_lengths,
            dtype=dtype,
            write_pattern=write_pattern,
            dtime_name=dtime_name,
            dtime_vals=dtime_vals,
            dtime_units=dtime_units,
        )

    # Create file
    with Dataset(path_nc, "w", format=nc_format) as ds:
        # Dimensions
        for name in dims:
            ds.createDimension(name, int(dim_lengths[name]))

        # Time coordinate
        vtime = ds.createVariable(dtime_name, "f8", (dtime_name,))
        vtime[:] = np.asarray(dtime_vals, dtype="float64")
        vtime.setncattr("units", str(dtime_units))
        vtime.setncattr("calendar", str(calendar))

        # Other coords
        for spec in (coord_specs or []):
            cname = spec["name"]
            cdtype = spec.get("dtype", "f4")
            cdims = tuple(spec["dims"])
            cdata = np.asarray(spec["data"])
            cv = ds.createVariable(cname, cdtype, cdims)
            # Safety: fill coords entirely
            cv[:] = cdata.astype(np.dtype(cdtype))
            for ak, av in (spec.get("attrs") or {}).items():
                cv.setncattr(str(ak), av)

        # Data variable
        create_kwargs = dict(zlib=bool(zlib), shuffle=bool(shuffle),
                             complevel=int(complevel), fill_value=fill_value)
        if chunks is not None:
            create_kwargs["chunksizes"] = tuple(int(x) for x in chunks)

        v = ds.createVariable(var_name, dtype, tuple(dims), **create_kwargs)
        v.setncattr("add_offset", float(add_offset))
        v.setncattr("scale_factor", float(scale_factor))
        v.setncattr("missing_value", np.int16(fill_value) if _dtype_nbytes(dtype) == 2 else fill_value)

        # Global attrs
        ds.setncattr("Conventions", "CF-1.8")
        ds.setncattr("calendar", str(calendar))
        if append_attrs:
            for k, val in append_attrs.items():
                ds.setncattr(str(k), val)


def append_met_netcdf(
    *,
    path_nc,
    var_name: str,
    data,                         # 1D time series or ND slice consistent with indexers
    indexers: dict[str, int | slice],  # e.g., {"n": isite, "DTIME": slice(0, nt)} or {"DTIME": slice(0, nt), "lat": iy, "lon": ix}
):
    """
    Append `data` to variable `var_name` using `indexers` to select the region.

    Notes:
      - `data` should be float; netCDF4 will pack using var attrs (scale_factor/add_offset).
      - You can pass fewer indexers than dims; unspecified dims default to slice(None).
    """
    with Dataset(path_nc, "r+") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{var_name} not found in {path_nc}")
        v = ds.variables[var_name]
        dims = v.dimensions

        # Build selection tuple in variable-dim order
        sel = []
        for name in dims:
            if name in indexers:
                sel.append(indexers[name])
            else:
                sel.append(slice(None))
        sel = tuple(sel)

        arr = np.asarray(data, dtype="float64")
        v.set_auto_scale(True)  # defensive; True by default
        v[sel] = arr
