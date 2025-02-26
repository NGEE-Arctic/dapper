# Functions that were only used once or have limited/no reusability.

def compute_elm_met_stats():
    """
    This function takes a set of ELM met input files and computes the mean, min, and max,
    then stores these statistics as a .csv. These statistics are used to validate
    unit conversions. 
    
    The sample files were taken from Trail Valley Creek from this repo:
    https://github.com/fmyuan/pt-e3sm-inputdata/tree/master/atm/datm7/Daymet_ERA5/cpl_bypass_TVC

    Validation should only consider order of magnitude, or other broad metrics, as TVC
    is obviously not representative of all potential sites.
    """
    import os
    import xarray as xr
    from ngeegee import utils
    path_sample_files = r'X:\Research\NGEE Arctic\NGEEGEE\data\fengming_data'
    files = os.listdir(path_sample_files)
    stats = {}
    for f in files:
        var = f.split('_')[2]
        ds = xr.open_dataset(os.path.join(path_sample_files, f))
        data = ds[var].data
        dmean = data.mean()
        dmin = np.percentile(data, 1)
        dmax = np.percentile(data, 99)
        stats[var] = {'mean' : dmean,
                    'min' : dmin,
                    'max' : dmax}
    sdf = pd.DataFrame(stats)  
    sdf.to_csv(utils._DATA_DIR / 'elm_met_var_stats.csv', index=False)  
