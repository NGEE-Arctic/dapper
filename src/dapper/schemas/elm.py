# Central place to ask for ELM-related mappings/units/etc.
# For now, forward to your existing elm_utils.elm_data_dicts().

from dapper.utils import elm_utils as eu

def data_dicts():
    return eu.elm_data_dicts()

def nonneg():
    return data_dicts()["nonneg"]

def units():
    return data_dicts()["units"]

def short_names_map(source="era5"):
    # for ERA5 adapter
    return data_dicts()["short_names"]

def required_vars(dformat):
    return data_dicts()["elm_req_vars"]["cbypass" if dformat=="BYPASS" else "datm"]
