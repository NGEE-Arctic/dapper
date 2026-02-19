# docker buildx build --progress=plain -f Dockerfile -t dapper . 
# docker run -it dapper /bin/bash

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
   'fastparquet' \
   'pygis' \
   ilamb>=2.7.3 \
   && mamba clean --all -f -y \
   && rm -rf "/home/${NB_USER}/.cache/yarn" \
   && rm -rf "/home/${NB_USER}/.node-gyp" \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}"

# Setup default user, when enter docker container
USER ${NB_UID}
WORKDIR "${HOME}"

# Install dapper
COPY --chown=${NB_UID} . "${HOME}/dapper"

RUN \
		cd "${HOME}/dapper" \
		&& pip install -e .

RUN \
		python -c "import ILAMB; print(ILAMB.__version__)" \
		&& python -c "import dapper; print(dapper.__version__)"

#EOF
