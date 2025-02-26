# Generic functions JPS
from pathlib import Path
import ngeegee

# Pathing for convenience
_ROOT_DIR = Path(next(iter(ngeegee.__path__))).parent
_DATA_DIR = _ROOT_DIR / "data"


def elm_data_dicts():
    """
    Defines some dictionaries for ELM-expected variables.
    """
    
    # E5LH name -> ELM name
    e5_to_elm_short_name = {'u_component_of_wind_10m' : 'UWIND',
                            'v_component_of_wind_10m' : 'VWIND',
                            'wind_speed' : 'WIND',
                            'surface_solar_radiation_downwards_hourly' : 'FSDS',
                            'surface_thermal_radiation_downwards_hourly' : 'FLDS',
                            'specific_humidity' : 'QBOT',
                            'total_precipitation_hourly' : 'PRECTmms',
                            'surface_pressure' : 'PSRF',
                            'temperature_2m' : 'TBOT',
                            'dewpoint_temperature_2m' : 'DTBOT',
                            'rel_hum' : 'RH'}

    # Output units
    units = {'TBOT' : 'K',
            'DTBOT' : 'unsure',
            'RH' : '%',
            'WIND' : 'm/s',
            'FSDS' : 'W/m2',
            'FLDS' : 'W/m2',
            'PSRF' : 'Pa',
            'PRECTmms' : 'kg/m2/s',
            'QBOT' : 'kg/kg',
            'ZBOT' : 'm',
            'UWIND' : 'm/s',
            'VWIND' : 'm/s'}

    # For scaling to make "short" netcdf. Taken from https://github.com/fmyuan/elm-pf-tools/blob/db70b67a28969154748f53e2446559ada323a136/pytools/metdata_processing/elm_metdata_write.py#L347C1-L366C1
    ranges = {'PRECTmms'  : [-0.04, 0.04],
            'FSDS'      : [-20, 2000],
            'TBOT'      : [175, 350],
            'RH'        : [0, 100],
            'QBOT'      : [0, 0.1],
            'FLDS'      : [0, 1000],
            'PSRF'      : [20000, 120000],
            'WIND'      : [-1, 100]}

    # Short descriptions
    short_descriptions = {
        'TBOT' : 'temperature at the lowest atm level (TBOT)',
        'DTBOT' : 'dewpoint temperature [era5 direct]',
        'RH' : 'relative humidity at the lowest atm level (RH)',
        'WIND' : 'wind magnitude at the lowest atm level (WIND)',
        'FSDS' : 'incident solar (FSDS)',
        'FLDS' : 'incident longwave (FLDS)',
        'PSRF' : 'pressure at the lowest atm level (PSRF)',
        'PRECTmms' : 'precipitation (PRECTmms)',
        'QBOT' : 'specific humidity at the lowest atm level (QBOT)',
        'ZBOT' : 'observational height (ZBOT)',
        'UWIND' : 'u component of wind velocity',
        'VWIND' : 'v component of wind velocity'}
    
    required_vars = {'datm' : ['LONGXY','LATIXY','time', 'ZBOT','TBOT', 'PRECTmms', 'RH', 'FSDS', 'FLDS', 'PSRF', 'WIND'],
                     'cbypass' : ['LONGXY','LATIXY','time', 'TBOT', 'PRECTmms', 'QBOT', 'FSDS', 'FLDS', 'PSRF', 'WIND']}
    
    return {'namemapper' : e5_to_elm_short_name,
            'units' : units,
            'ranges' : ranges,
            'descriptions' : short_descriptions,
            'req_vars' : required_vars}
