
from dapper.surf.sample import sample_point_values
from dapper.utils.pathing import SURFDATA_HALFDEGREE_TOP

# Existing global surface file
nc_in = SURFDATA_HALFDEGREE_TOP / "surfdata_0.5x0.5_SOMETHING.nc"

sampled = sample_point_values(
    nc_in,
    lat=40.0,
    lon=-106.0,
    # optional filters:
    # include={"PCT_NATVEG", "PCT_NAT_PFT", "PCT_SAND", "PCT_CLAY"},
    # exclude={"APATITE_P", "LABILE_P"},
)




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
