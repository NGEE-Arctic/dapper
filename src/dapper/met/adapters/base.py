class BaseAdapter:
    """
    Minimal interface the exporter expects.
    """

    # --- discovery & locations ---
    def discover_files(self, csv_directory, calendar):
        # return (csv_files, start_year, end_year)
        raise NotImplementedError

    def normalize_locations(self, df_loc, id_col, nzones):
        # return df_loc_norm with ['gid','lat','lon','lon_0-360','zone', ...]
        raise NotImplementedError

    def id_column_for_csv(self, df_csv, id_col):
        # return the id col name present in df_csv (e.g., 'gid', 'pids', etc.)
        raise NotImplementedError

    # --- preprocessing & requirements ---
    def preprocess_shard(self, df_merged, start_year, end_year, calendar, dformat):
        # return preprocessed df with at least ['gid','time','LATIXY','LONGXY','zone', <ELM vars>]
        raise NotImplementedError

    def required_vars(self, dformat):
        # list of ELM var short names to output for this dformat
        raise NotImplementedError

    # --- packing ---
    def pack_params(self, elm_var, data=None):
        # return (add_offset, scale_factor)
        raise NotImplementedError
