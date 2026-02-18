# docker buildx build --progress=plain -f Dockerfile -t dapper . 

# ----------------------------------------------------------------------
# JupyterLab Notebook for analyzing E3SM-ELM land model output
# with Google Earth Engine and GEEmap
# ----------------------------------------------------------------------

FROM quay.io/jupyter/scipy-notebook:lab-4.4.10

# install Python packages you often use
RUN set -ex \
   && conda install --quiet --yes -c conda-forge \
   # choose the Python packages you need
   'matplotlib' \
   'plotly' \
   'folium' \
   'dask' \
   'pandas' \
   'xarray' \
   'geopandas' \
   'netCDF4' \
   'zarr' \
   'ipyleaflet' \
   'nc-time-axis' \
   'earthengine-api' \
   'geemap' \
   'pygis' \
   ilamb>=2.7.3 \
   && mamba clean --all -f -y \
   && rm -rf "/home/${NB_USER}/.cache/yarn" \
   && rm -rf "/home/${NB_USER}/.node-gyp" \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# Make local directories to use with plotting of ELM output
RUN mkdir "/home/${NB_USER}/vis_notebooks" \   
   && mkdir "/home/${NB_USER}/inputdata" \
   && mkdir "/home/${NB_USER}/output" \
   && fix-permissions "/home/${NB_USER}/vis_notebooks" \
   && fix-permissions "/home/${NB_USER}/inputdata" \
   && fix-permissions "/home/${NB_USER}/output" \
   && fix-permissions "/home/${NB_USER}"

# Setup default user, when enter docker container
USER ${NB_UID}
WORKDIR "${HOME}"

# Download, install dapper
RUN git clone https://github.com/ngee-arctic/dapper \
  && cd dapper \
  && pip install -e .

RUN \
		python -c "import ILAMB; print(ILAMB.__version__)" \
		&& python -c "import dapper; print(dapper.__version__)"

#EOF
