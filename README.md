<p align="center">
  <img src="docs/images/dapper_logo_2.jpg" width="50%" alt="dapper">
</p>

## About
The E3SM Land Model (ELM) has become useful for a wide range of investigations across a broad array of scales--from single-site to global. Each ELM run requires a hefty set of data formatted in a particular way. Two of the required "types" of data are meteorological data (time series of "met forcings") and parameter files (customizable "surface file"). As ELM capability grows, the complexity of, for example, the surface files grows to accomodate new options and parameters. Additionally, met data is usually sampled at high temporal frequency (daily/subdaily). These data requirements place a large startup burden on (particularly new) users of ELM. 

DaPPER provides tools to alleviate these burdens to some degree. DaPPER relies on Google Earth Engine and other APIs to make sampling met data and surface file data fast, scalable to very large domains, and flexible to different "types" of grid cells (e.g. arbitrary polygons like watersheds instead of rectangular grids). DaPPER is being developed with the goals of NGEE-Arctic Crosscut 3 in mind, which focuses on scaling and resolution experiments. However, the broader vision is to develop a toolkit that is useful across the E3SM project, not just NGEE. We also hope that DaPPER can provide some documentation and clarity regarding the creation and use of these input data, which to-date seems to be lacking.

DaPPER is under development, which means that you should consider everything *beta* and subject to change daily. We try to alleviate this instability by maintaining a `dev` branch of the repo that can be very dynamic. The `main` branch is intended to be more stable. If you're not sure, just stick to the `main` repo.

DaPPER is developed by Jon Schwenk at LANL, with code contributions from Ryan Crumley (LANL), Cade Trotter (LANL), Rich Fiorella (LANL), Jemma Stachelek (LANL), and Fengming Yuan (ORNL).


## Setup and Installation
**Step 1**: Create a free account with [Google Earth Engine](https://code.earthengine.google.com/register) if you don't have one.

**Step 2**: Clone this repository to your local machine. I recommend [Github Desktop](https://desktop.github.com/download/) if you're not familiar with git commands.

**Step 3**: Use the environment.yml file (it will be on your local machine after you clone the repo) to create a virtual Python environment. You can use whatever package manager your prefer, but instructions here are for conda. I recommend installing [mamba](https://anaconda.org/conda-forge/mamba) into your base environment to make package solving faster, but it's not necessary. If you do install it, you can just replace `conda` with `mamba` in the following commands.
```
conda env create -f environment.yml # This should automatically name your new environment "dapper"
```

**Step 4**: Perform a "live install" of the dapper repo using `pip`. This is useful as it allows you to update your local `dapper` repo (via a `Fetch origin` in Github Desktop), and any new changes will be reflected immediately in the code that you're running. No need to recompile a package and reinstall it--very simple.
```
cd /path/to/cloned/dapper # Navigate to where you cloned the repo
conda activate dapper # Activate your dapper environment
pip install -e . # Live-install the repo as an importable package
```

**Step 5**: Test that your install works.
```
conda activate dapper 
ipython
from dapper import e5l_utils
```
If you can import from the `dapper` package without error, you're gravy.

**Step 5b**: If you haven't used the GEE API before, you'll need to Authenticate before you can interact with GEE via Python. For more details, or if you get stuck, check out the [official guidance](https://developers.google.com/earth-engine/guides/auth). However, it's likely the following code will work for you:
```
conda activate dapper
ipython
import ee
ee.Authenticate() # This should open a browser where you allow access to your GEE account and project
ee.Initialize(project='ee-yourprojectname') # replace ee-yourprojectname with your actual GEE project name
```
You will not need to run `ee.Authenticate()` again as it stores your credentials locally. You will, however, have to run `ee.Initialize()` each time you use the GEE Python API.

## Usage
Check out the [jupyter notebooks](https://github.com/NGEE-Arctic/dapper/tree/main/docs/notebooks) for ways to use the tools in this package.

## Contributing
Feel free to fork the repo and make improvements. Open a pull request and we'll check it out.

## Credits
Repo initialized by Jon Schwenk; maintained by Jon Schwenk and Jemma Stachelek. Some `dapper` functionality was ported from work by Ryan Crumley and Cade Trotter at Los Alamos National Laboratory. Rich Fiorella contributed OLMT-related content and assists with package maintenance. Others have contributed via the Issue Tracker and valuable feedback/testing.

## Copyright/License
O4898 (for copyright verification); see the [License](https://github.com/NGEE-Arctic/dapper/tree/main/license.md) file.
