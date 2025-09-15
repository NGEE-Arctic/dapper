import numpy as np
import pandas as pd
from pathlib import Path

from dapper.met.adapters.base import BaseAdapter
from dapper.schemas import elm as elm_schema
from dapper.met import era5land as e5   # <- wherever your current helpers live
from dapper.met import met_io as io             # for get_start_end_years
from dapper.utils import gee_utils as gu

class ERA5Adapter(BaseAdapter):
    def discover_files(self, csv_directory, calendar):
        csv_directory = Path(csv_directory)
        csvs = [str(csv_directory / f) for f in csv_directory.iterdir()
                if f.suffix.lower()==".csv"]
        if not csvs:
            raise FileNotFoundError("No .csv files found")
        start_year, end_year = io.get_start_end_years(csvs, calendar=calendar)
        return csvs, start_year, end_year

    def normalize_locations(self, df_loc, id_col, nzones):
        # reuse your existing helper
        return e5._prep_df_loc(df_loc, id_col=id_col, nzones=nzones)

    def id_column_for_csv(self, df_csv, id_col):
        if id_col is not None and id_col in df_csv.columns:
            return id_col
        guess = gu.infer_id_field(df_csv)
        return guess

    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        remove_leap = (calendar == "noleap")
        df = e5._preprocess_e5lh_to_elm_file_grid(
            df_merged, start_year, end_year, remove_leap, dformat
        )
        return df.sort_values(["time","LATIXY","LONGXY"]).reset_index(drop=True)

    def required_vars(self, dformat):
        return elm_schema.required_vars(dformat)

    def pack_params(self, elm_var, data=None):
        # delegate to your robust packer
        from dapper.utils import elm_utils as eu
        return eu.elm_var_packing_params(elm_var, data=data)
