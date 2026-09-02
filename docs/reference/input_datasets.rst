Input source datasets
=====================

Pseudo-global example files
---------------------------

The ``surf_pseudoglobal.nc`` and ``landuse_pseudoglobal.nc`` files included
with dapper are spatially cropped subsets of full, ELM-compatible global
surface and transient land-use datasets. They retain the source variables and
dimensions needed by the tutorials, but cover only the small example region.
"Pseudo-global" refers to this limited geographic coverage, not to a different
file format or sampling method.

These files are suitable for tutorials, tests, and domains contained within
their retained area. For production domains elsewhere, provide the appropriate
full global NetCDF as ``src_path`` when calling ``Domain.export_surface`` or
``Domain.export_landuse``. Check the source ``LATIXY`` and ``LONGXY`` extent
before sampling: a cropped file cannot supply cells outside its bounds or the
surrounding cells needed for a spatial average at its edge.
