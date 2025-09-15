import numpy as np
from pathlib import Path
from dapper.met import met_io as io  # reuse your proven low-level writer

def _compute_auto_chunks(dims, dtime_vals, dtime_units, dtype, write_pattern):
    """
    Return chunks tuple aligned with 'dims'. Heuristic:
      - if writing by site/cell: chunk 1 along that dim, time ~ 3–4 weeks
      - keep chunk ~1–2 MB
    """
    # figure out time step in hours
    nt = len(dtime_vals)
    if nt > 1:
        dt_raw = float(np.median(np.diff(np.asarray(dtime_vals, dtype=float))))
    else:
        dt_raw = 1.0

    u = (dtime_units or "").lower()
    dt_hours = dt_raw * (24.0 if "day" in u else 1.0)
    steps_per_day = 24.0 / dt_hours if dt_hours > 0 else 48.0  # default 30-min

    # target ~28 days
    t_chunk = int(max(1, min(nt, round(28.0 * steps_per_day))))

    # map to dims
    if dims == ("n","DTIME"):
        n_chunk = 1 if write_pattern == "by_site" else 64
        chunks = (n_chunk, t_chunk)
    elif dims == ("DTIME","lat","lon"):
        # single cell writes → keep small lat/lon tile
        chunks = (t_chunk, 1, 1) if write_pattern == "by_cell" else (t_chunk, 8, 8)
    else:
        # fallback: all-ones except time
        chunks = tuple(1 if d != "DTIME" else t_chunk for d in dims)

    # shrink to ~1.5 MB
    dtype_size = np.dtype(dtype).itemsize if not isinstance(dtype, str) else np.dtype(np.int16).itemsize
    target_bytes = int(1.5 * 1024 * 1024)
    cur_bytes = np.prod(chunks) * dtype_size
    if cur_bytes > target_bytes and "DTIME" in dims:
        k = chunks[dims.index("DTIME")]
        shrink = max(1, int(np.ceil(cur_bytes / target_bytes)))
        k2 = max(1, k // shrink)
        chunks = tuple(k2 if i == dims.index("DTIME") else c for i, c in enumerate(chunks))

    return chunks

def initialize_met_netcdf(path_nc, var_name, dims, dim_lengths,
                          dtime_name, dtime_vals, dtime_units, calendar,
                          coord_specs, add_offset, scale_factor,
                          dtype="i2", fill_value=32767, chunks=None,
                          write_pattern="by_site", append_attrs=None,
                          nc_format="NETCDF4_CLASSIC"):
    """
    Thin wrapper over met_io.initialize_met_netcdf (your generic version).
    Adds default auto-chunking if chunks is None.
    """
    if chunks is None:
        chunks = _compute_auto_chunks(dims, dtime_vals, dtime_units, dtype, write_pattern)

    io.initialize_met_netcdf(
        path_nc=path_nc,
        var_name=var_name,
        dims=dims,
        dim_lengths=dim_lengths,
        dtime_name=dtime_name,
        dtime_vals=dtime_vals,
        dtime_units=dtime_units,
        calendar=calendar,
        coord_specs=coord_specs,
        add_offset=add_offset,
        scale_factor=scale_factor,
        dtype=dtype,
        fill_value=fill_value,
        chunks=chunks,
        write_pattern=write_pattern,
        append_attrs=append_attrs,
        nc_format=nc_format,
    )

def append_met_netcdf(path_nc, var_name, data, indexers):
    # delegate to your working low-level appender
    io.append_met_netcdf(
        path_nc=path_nc,
        var_name=var_name,
        data=data,
        indexers=indexers
    )
