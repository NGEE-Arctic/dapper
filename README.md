# NGEEGEE
Tools for using Google Earth Engine to fetch and format data


## Setup and Installation
**Step 1**: Create a free account with [Google Earth Engine](https://code.earthengine.google.com/register) if you don't have one.
**Step 2**: Clone this repository to your local machine. I recommend [Github Desktop](https://desktop.github.com/download/) if you're not familiar with git commands.
**Step 3**: Use the environment.yml file (it will be on your local machine after you clone the repo) to create a virtual Python environment. You can use whatever package manager your prefer, but instructions here are for conda. I recommend installing [mamba](https://anaconda.org/conda-forge/mamba) into your base environment to make package solving faster, but it's not necessary. If you do install it, you can just replace `conda` with `mamba` in the following commands.
```
conda env create -f environment.yml # This should automatically name your new environment "ngeegee"
```
**Step 4**: Perform a "live install" of the NGEEGEE repo using `pip`. This is useful as it allows you to update your local NGEEGEE repo (via a `Fetch origin` in Github Desktop), and any new changes will be reflected immediately in the code that you're running. No need to recompile a package and reinstall it--very simple.
```
cd /path/to/cloned/NGEEGEE # Navigate to where you cloned the repo
conda activate ngeegee # Activate your ngeegee environment
pip install -e . # Live-install the repo as an importable package
```

**Step 5**: Test that your install works.
```
conda activate ngeegee # if not already activated
ipython # puts you in an interactive python shell
from ngeegee import e5l_utils
```
If you can import from the `ngeegee` package without error, you're gravy.

**Step 5b**: If you haven't used the GEE API before, you'll need to Authenticate before you can interact with GEE via Python. For more details, or if you get stuck, check out the [official guidance](https://developers.google.com/earth-engine/guides/auth). However, it's likely the following code will work for you:
```
conda activate ngeegee
ipython
import ee
ee.Authenticate() # This should open a browser where you allow access to your GEE account and project
ee.Initialize(project='ee-yourprojectname') # replace ee-yourprojectname with your actual GEE project name
```
You will not need to run `ee.Authenticate()` again as it stores your credentials locally. You will, however, have to run `ee.Initialize()` each time you use the GEE Python API.

## Usage
Check out the jupyter notebooks (not yet created) for ways to use the tools in this package.

## Contributing
Feel free to fork the repo and make improvements. Open a pull request and we'll check it out.

## Contact
Repo maintained by jschwenk@lanl.gov. Tools were ported from work by Ryan Crumley and Cade Trotter at Los Alamos National Laboratory.
