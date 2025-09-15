from dapper.met.adapters.era5 import ERA5Adapter
from dapper.met.exporter import Exporter

csv_dir = r'blah'
out_dir = csv_dir
adapter = ERA5Adapter()
exp = Exporter(adapter,
               csv_directory=CSV_DIR,
               write_directory=OUT_DIR,
               df_loc=df_loc,
               id_col=None,
               calendar='noleap',
               dtime_resolution_hrs=1,
               dtime_units='days',
               nzones=1,
               dformat='BYPASS',
               force_half_hour_for_hourly=True)

exp.run(output_mode='sites_file', pack_scope='global')
