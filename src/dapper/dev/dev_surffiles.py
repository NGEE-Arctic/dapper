
from dapper.dev import pathing
from dapper.surf.sample import sample_point_values
from dapper.surf.write import build_surface_dataset, write_surface_nc

# nc_in = pathing.SURFDATA_HALFDEGREE
# nc_out = r'X:\Research\NGEE Arctic\3. Surface Files\out\test.nc'

# # 1) Sample
# sampled = sample_point_values(nc_in, lat=40.0, lon=-106.0)

# # 2) Build 1x1 Dataset
# ds_pt = build_surface_dataset(sampled)

# # 3) Write to NetCDF
# out_path = write_surface_nc(ds_pt, nc_out)
# print("Wrote:", out_path)


# Topounit example
from dapper.surf.write import SurfaceFile
from dapper.domains.domain import Domain
import geopandas as gpd
from pathlib import Path

OUT_DIR = Path("outputs/surface_topounits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

domain_gdf = gpd.read_file("data/kup_watershed/Kuparuk_gageshed.shp")
domain = Domain.from_geodataframe(domain_gdf, id_field="gid")

topo_8  = gpd.read_file("outputs/topounits/topounits_elev8.gpkg")
topo_12 = gpd.read_file("outputs/topounits/topounits_elev12.gpkg")

def build_surface_from_topounits(topo_gdf, config_name):
    # For now this just creates an empty Dataset; you'll later extend
    # from_domain to build whatever base dims you want.
    sf = SurfaceFile.from_domain(domain)

    sf.add_params_from_df(
        dim_name="topounit",
        df=topo_gdf,
        id_col="band_name",
        drop_cols=["geometry"],
    )

    sf.set_global_attrs(
        title=f"ELM surface file with topounits ({config_name})",
        case_name=f"kup_topounits_{config_name}",
    )

    # Lightweight registry sanity check
    sf.validate(strict=False, use_external_validator=False)

    out_path = OUT_DIR / f"elm_surface_topounits_{config_name}.nc"
    sf.to_netcdf(out_path, overwrite=True)
    return out_path

surf_8  = build_surface_from_topounits(topo_8,  "elev8")
surf_12 = build_surface_from_topounits(topo_12, "elev12")



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
