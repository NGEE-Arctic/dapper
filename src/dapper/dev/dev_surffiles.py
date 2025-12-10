
from dapper.utils import pathing
from dapper.surf.sample import sample_point_values
from dapper.surf.write import build_surface_dataset, write_surface_nc

nc_in = pathing.SURFDATA_HALFDEGREE
nc_out = r'X:\Research\NGEE Arctic\3. Surface Files\out\test.nc'

# 1) Sample
sampled = sample_point_values(nc_in, lat=40.0, lon=-106.0)

# 2) Build 1x1 Dataset
ds_pt = build_surface_dataset(sampled)

# 3) Write to NetCDF
out_path = write_surface_nc(ds_pt, nc_out)
print("Wrote:", out_path)




## INFER FROM E3SM - NEEDS A BIT OF WORK, NOT SURE IF FEASIBLE
from pathlib import Path
import pandas as pd
from infer_e3sm import find_surface_vars_v3

repo_root = Path("/path/to/E3SM")  # top of the E3SM repo

rows = pd.DataFrame(
    find_surface_vars_v3(
        repo_root,
        search_root="components/elm",  # default
    )
)

# Just required surface *variables* (not dimensions)
required_vars = rows.query("object_kind == 'variable' and required_inferred == 'required'")
print(required_vars[["var_name", "file", "line", "reason"]])

# Unique required variable names:
required_var_names = sorted(required_vars["var_name"].unique())
print(required_var_names)
