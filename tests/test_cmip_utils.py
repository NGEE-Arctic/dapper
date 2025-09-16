import intake
from intake_esm.core import esm_datastore

def test_intake():
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    assert isinstance(col, esm_datastore)

