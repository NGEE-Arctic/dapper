Getting Started
===============

Before You Install
------------------
``dapper`` is designed to work with Google Earth Engine (GEE). To use the full workflow (including the tutorial notebooks), you will need:

* A free GEE account and a GEE project
* The ability to run ``ee.Authenticate()`` and ``ee.Initialize(...)`` from Python

If you do not already have a GEE account, you can register here:
`Google Earth Engine registration <https://code.earthengine.google.com/register>`_

Some parts of ``dapper`` can be used without GEE, but most end-to-end tutorials assume that GEE authentication is working.


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
