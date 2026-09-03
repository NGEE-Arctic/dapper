Getting Started
===============

Before You Install
------------------
``dapper`` can sample ERA5-Land through either the ECMWF Climate Data Store
(CDS) ARCO service or Google Earth Engine (GEE). To use every workflow, you
will need:

* A free GEE account and a GEE project
* The ability to run ``ee.Authenticate()`` and ``ee.Initialize(...)`` from Python
* A free CDS account and personal access token

If you do not already have a GEE account, you can register here:
`Google Earth Engine registration <https://code.earthengine.google.com/register>`_

Point and small-polygon ERA5-Land requests can use CDS without a GEE account.
Large polygon requests continue to use GEE.


Installation
~~~~~~~~~~~~
``dapper`` is in active development.

* If you want the latest features and the most up-to-date tutorials, **a live (editable) install is recommended**.
* If you want a stable snapshot that is easy to pin for reproducibility, install from **PyPI**.


PyPI install (stable snapshot)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is the fastest way to install ``dapper`` without cloning the repo.

**Step 1**

Create and activate a clean environment (conda recommended):

.. code-block:: bash

   conda create -n dapper python=3.12
   conda activate dapper

**Step 2**

Install from PyPI:

.. code-block:: bash

   pip install dapper-elm

**Step 3**

Quick import test:

.. code-block:: bash

   python -c "import dapper; from dapper.met.adapters import era5; print('dapper import OK')"


Live install (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^
A live install is the preferred setup if you expect to pull updates frequently while ``dapper`` evolves.

**Step 1**

Clone the repository to your local machine. If you're not comfortable with command-line git, `GitHub Desktop <https://desktop.github.com/download/>`_ works fine.

**Step 2**

Create the conda environment from ``environment.yml`` (recommended, since it tracks the dev dependencies used in the tutorials):

.. code-block:: bash

   cd /path/to/cloned/dapper
   conda env create -f environment.yml
   conda activate dapper

**Step 3**

Perform a live (editable) install:

.. code-block:: bash

   pip install -e .

**Step 4**

Quick import test:

.. code-block:: bash

   python -c "import dapper; from dapper.met.adapters import era5; print('dapper live install OK')"

**Step 5**

Keeping your live install up to date (typical workflow):

.. code-block:: bash

   cd /path/to/cloned/dapper
   git pull
   # no reinstall needed; your environment is using the repo checkout


Google Earth Engine authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you haven't used the GEE API before, you'll need to authenticate. The first time you run this, it should open a browser to grant access to your GEE account.

.. code-block:: bash

   conda activate dapper
   ipython
   import ee
   ee.Authenticate()
   ee.Initialize(project="ee-yourprojectname")  # replace with your actual GEE project name

You should not need to run ``ee.Authenticate()`` again (credentials are cached locally).
You will, however, need to run ``ee.Initialize(project="...")`` in each fresh Python session.


ECMWF CDS authentication
^^^^^^^^^^^^^^^^^^^^^^^^
Register or sign in at the `Climate Data Store <https://cds.climate.copernicus.eu/>`_,
accept the ERA5-Land dataset terms, and place the token shown on your profile in
``~/.cdsapirc``:

.. code-block:: text

   url: https://cds.climate.copernicus.eu/api
   key: <PERSONAL-ACCESS-TOKEN>

The official setup instructions are at `CDSAPI setup
<https://cds.climate.copernicus.eu/en/how-to-api>`_. Dapper uses the
``reanalysis-era5-land-timeseries`` ARCO product for synchronous point and
small-area downloads.


Choosing an ERA5-Land backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``backend="arco"`` or ``backend="gee"`` to force a source, or leave the
default ``backend="auto"``. The automatic planner considers the ARCO area
limit, intersecting 0.1-degree grid-cell count, requested time span, and a
measured per-request transfer estimate:

.. code-block:: python

   from dapper import plan_era5_land_sampling, sample_era5_land

   plan = plan_era5_land_sampling(
       domain,
       start_date="1950-01-02",
       end_date="latest",
   )
   print(plan)

   sampled_domain = sample_era5_land(
       domain,
       start_date="1950-01-02",
       end_date="latest",
       backend="auto",
       output_dir="era5_csvs",
   )

ARCO writes GEE-compatible CSV files plus a JSON manifest containing the exact
sampled grid cells and local area weights. Dates have an inclusive start and an
exclusive output end. ARCO radiation and precipitation use the same hourly,
interval-end accumulation convention as the GEE fields; ``ERA5Adapter`` shifts
them to ELM's interval-start timestamps and performs the existing unit
conversions. ECMWF's ARCO series begins on 1950-01-02 because the incomplete
1950-01-01 source day was omitted; automatic mode therefore chooses GEE when
that first day is required.
