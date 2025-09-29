Getting Started
===============

Before You Install
------------------
For ``dapper`` to work properly, you will need to create a free account with `Google Earth Engine <https://code.earthengine.google.com/register>`_ if you don't have one.

It is possible to make use of some ``dapper`` functionality without a GEE account, but tutorials, testing, etc. within ``dapper`` assume that the ``ee`` module is importable, and many of them assume that you can run ``ee.Authenticate()`` successfully.


Installation
~~~~~~~~~~~~~~~~~~~~~~
Due to ongoing ``dapper`` development, the only way to install the package currently is via a "live install." Instructions for this are provided below.

**Step 1**  

Clone this repository to your local machine. I recommend `Github Desktop <https://desktop.github.com/download/>`_ if you're not familiar with command-line git.

**Step 2** 

Use the ``environment.yml`` file (it will be on your local machine in the root of the cloned repo) to create a virtual Python environment. You can use whatever package manager you prefer, but instructions here are for conda. I recommend installing `mamba <https://anaconda.org/conda-forge/mamba>`_ into your base environment to make package solving faster, but it's not necessary. If you do install it, you can just replace ``conda`` with ``mamba`` in the following commands.

.. code-block:: bash

   conda env create -f environment.yml  # This will automatically name your new environment "dapper" as it's specified in the environment.yml file

**Step 3**  

Perform a "live install" of the dapper repo using ``pip``. This allows you to update your local repo (via a *Fetch origin* in Github Desktop), and any new changes will be reflected immediately in the code that you're running. No need to recompile a package and reinstall it. 
The following commands are for a Terminal/Command (or Anaconda) Prompt.

.. code-block:: bash

   cd /path/to/cloned/dapper  # Navigate to where you cloned the repo
   conda activate dapper      # Activate your dapper environment
   pip install -e .           # Live-install the repo as an importable package

**Step 4**  

Test that your install works. Again, in a Terminal/Command Prompt shell:

.. code-block:: bash

   conda activate dapper 
   ipython
   import ee # Make sure you have ee installed--should be done automatically through environment.yml
   from dapper.met.adapters import era5 # Just testing the ability to import dapper modules

If you can import from the ``dapper`` package without error, you're good to go (but don't skip the next step).

**Step 5** 

If you haven't used the GEE API before, you'll need to authenticate before you can interact with GEE via Python. For more details, or if you get stuck, check out the `official guidance <https://developers.google.com/earth-engine/guides/auth>`_. However, it's likely the following code will work for you:

.. code-block:: bash

   conda activate dapper
   ipython
   import ee
   ee.Authenticate()  # This should open a browser to allow access to your GEE account and project
   ee.Initialize(project='ee-yourprojectname')  # replace ee-yourprojectname with your actual GEE project name

You will (should) not need to run ``ee.Authenticate()`` again as it stores your credentials locally.  
You will, however, have to run ``ee.Initialize(project='my_project')`` each time you use the GEE Python API.
