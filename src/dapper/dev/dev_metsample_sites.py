# Dev script to test implementations that mess with met sampling

import ee
import json
from pathlib import Path
from shapely.geometry import Point

from dapper.domains.aoi import AOI
import dapper.utils.gee_utils as gutils
from dapper.met.exporter import Exporter
from dapper.met.adapters.era5 import ERA5Adapter

# Initialize GEE (use your own project)
ee.Initialize(project="ee-jonschwenk")

# JSON inputs
json1 = r"X:\Research\NGEE Arctic\4. Using Dapper\preplab_hack\core_sites.json"
json2 = r"X:\Research\NGEE Arctic\4. Using Dapper\preplab_hack\eval_sites.json"

# Build AOI  from site points
geoms = []
ids = []

for path in [json1, json2]:
    with open(path) as f:
        data = json.load(f)

    for key, val in data.items():
        name = val["name"]
        lat = val["point"]["lat"]
        lon = val["point"]["lon"]
        ids.append(name)                 # gid
        geoms.append(Point(lon, lat))    # geometry

aoi = AOI.from_geometries(
    geoms,
    ids=ids,
    name="p4_core_eval_sites",
)

# Sampling parameters
params = {
    "start_date": "1950-01-01",
    "end_date":   "2100-01-01",      
    "geometries": aoi,               
    "gee_bands": "elm",
    "gee_years_per_task": 10,
    "gee_scale": "native",
    "job_name": "p4_allsites",
    "gdrive_folder": "p4_allsites",
}

# Send tasks to GEE (or just regenerate locations)
dom = gutils.sample_e5lh(params, skip_tasks=True)  # returns Domain

csv_directory = Path(r'X:\Research\NGEE Arctic\dapper_data\notebook_data\era5-elm') # where I've put my downloaded .csv files
write_directory = csv_directory / 'elm_ready_elm-combined_1hr_test' # where I want my ELM-formatted netCDFs to go

exp = Exporter(
    adapter=ERA5Adapter(), # Anytime you are writing ERA5, use this Adapter. More will soon be available for custom met files and perhaps other GEE sources (Daymet, GSWP3, etc.)
    csv_directory=csv_directory, # Where dapper can find the raw csvs GEE created and you've downloaded locally
    write_directory=write_directory, # Where dapper should put your exports
    domain=dom, # This was generated when we called sample_era5() above. Note that you can re-generate it without sending more Tasks to GEE by turning on the skip_tasks parameter in that function.
    dtime_resolution_hrs=1, # Here we select our desired output time resolution. dapper can export basically any time resolution (not sure about weird stuff like 1 minute, but 0.5 hours is fine).
    append_attrs={"note":"hello world"},  # If you want to add information to the exported netCDF files, this is the place to do it. Each netCDF file (variable) will include whatever you specify here.
)

# Now we can run the export
# For output_mode, we have three choices: 'elm-sites', 'elm-combined', and 'elm-grid'. We'll look at examples of the others later; for now, we want to export
# one set of met vars per site, so we choose 'site_dir'.
exp.run(output_mode='elm-combined', pack_scope='global')