Required variables
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - **Variable**
     - **Dimensions**
     - **Description**

   * - ``AREA``
     - ``lsmlat,lsmlon``
     - Area (m2) of each land grid cell; used to weight land-surface properties and to compare with domain/topography.

   * - ``LANDFRAC``
     - ``lsmlat,lsmlon``
     - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

   * - ``LANDFRAC_PFT``
     - ``lsmlat,lsmlon``
     - Fraction (0–1) of the global gridcell that is land in the PFT landunit; used to weight PFT tiles relative to the gridcell.

   * - ``LANDMASK``
     - ``lsmlat,lsmlon``
     - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.

   * - ``LATIXY``
     - ``lsmlat,lsmlon``
     - Latitude (degrees) of the centre of each land grid cell.

   * - ``LONGXY``
     - ``lsmlat,lsmlon``
     - Longitude (degrees) of the centre of each land grid cell; used to compare the surface file with the domain and topography files.

   * - ``PCT_CLAY``
     - ``nlevsoi,lsmlat,lsmlon``
     - Soil clay percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.

   * - ``PCT_CROP``
     - ``topounit,lsmlat,lsmlon``
     - Percent of grid cell area covered by cropland.

   * - ``PCT_LAKE``
     - ``topounit,lsmlat,lsmlon``
     - Percentage of each grid cell area that is lake within each topounit; used for lake landunit weighting.

   * - ``PCT_NATVEG``
     - ``topounit,lsmlat,lsmlon``
     - Percent of grid cell area covered by natural vegetation (non-crop vegetated landunit).

   * - ``PCT_NAT_PFT``
     - ``topounit,natpft,lsmlat,lsmlon``
     - Fraction of vegetated area allocated to each natural plant functional type; code aborts if missing.

   * - ``PCT_SAND``
     - ``nlevsoi,lsmlat,lsmlon``
     - Soil sand percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.

   * - ``PCT_WETLAND``
     - ``topounit,lsmlat,lsmlon``
     - Percentage (0–100) of each grid cell area covered by wetlands within each topounit.

   * - ``PFTDATA_MASK``
     - ``lsmlat,lsmlon``
     - Mask marking land grid cells that contain valid plant functional type (PFT) data; if missing the code stops with an error.

   * - ``SLOPE``
     - ``lsmlat,lsmlon``
     - Mean surface slope (degrees) for each land grid cell; used in topographic and hydrologic parameterizations.

   * - ``TOPO``
     - ``lsmlat,lsmlon``
     - Mean elevation (m) of each land grid cell read from the topography file; required by surfrd_get_topo.

   * - ``frac``
     - ``lsmlat,lsmlon``
     - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

   * - ``mask``
     - ``lsmlat,lsmlon``
     - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.


Conditional variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - **Variable**
     - **Dimensions**
     - **Description**

   * - ``FGRD``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of irrigation water applied via ground/sprinkler/drip; complements FSURF. (Requirement: Required when firrig_data is true; optional otherwise)

   * - ``FIRRIG``
     - ``topounit,lsmlat,lsmlon``
     - Fraction (0–1) of cropland that is irrigated; read only when irrigation data are enabled (firrig_data). (Requirement: Required when firrig_data is true; optional otherwise)

   * - ``FSURF``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of irrigation water applied via surface irrigation; required with FIRRIG when irrigation data are used. (Requirement: Required when firrig_data is true; optional otherwise)

   * - ``GLC_MEC``
     - ``lsmlat,lsmlon``
     - Integer glacier elevation-class index for each grid cell when using mechanistic glacier (MEC) landunits. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

   * - ``PCT_CFT``
     - ``topounit,cft,lsmlat,lsmlon``
     - Fraction of vegetated area allocated to each crop functional type; code aborts if missing when cft dimension exists. (Requirement: Required when the surface file includes crop functional types (cft_size > 0))

   * - ``PCT_FCP``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of natural vegetation that is flat-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

   * - ``PCT_GLACIER``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of grid cell area covered by simple glacier landunits within each topounit. (Requirement: Required unless glacier MEC landunits are used; optional when create_glacier_mec_landunit is true)

   * - ``PCT_GLC_MEC``
     - ``topounit,nglcec,lsmlat,lsmlon``
     - Percent of grid cell area assigned to mechanistic glacier classes (accumulation/ablation, etc.). (Requirement: Required when glacier MEC landunits are created; optional otherwise)

   * - ``PCT_HCP``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of natural vegetation landunit that is high-centered polygons (polygonal tundra). (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

   * - ``PCT_LCP``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of natural vegetation that is low-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

   * - ``PCT_URBAN``
     - ``topounit,numurbl,lsmlat,lsmlon``
     - Fraction of each grid cell that is urban, for each density class in the multi-density urban scheme. (Requirement: Required when the multi-density urban model is used (nlevurb > 0); optional otherwise)

   * - ``SINSL_COSAS``
     - ``lsmlat,lsmlon``
     - sin(slope) * cos(aspect) for each grid cell; used in solar-radiation calculations. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``SINSL_SINAS``
     - ``lsmlat,lsmlon``
     - sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``SKY_VIEW``
     - ``lsmlat,lsmlon``
     - Sky-view factor (0–1) describing horizon obstruction by surrounding terrain; needed by the TOP solar-radiation scheme. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``STDEV_ELEV``
     - ``lsmlat,lsmlon``
     - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; if STDEV_ELEV is missing the code tries STD_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``STD_ELEV``
     - ``lsmlat,lsmlon``
     - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; alternative name to STDEV_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``TERRAIN_CONFIG``
     - ``lsmlat,lsmlon``
     - Terrain configuration parameter used in the TOP scheme to account for terrain-induced shading. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

   * - ``TOPO_GLC_MEC``
     - ``topounit,nglcec,lsmlat,lsmlon``
     - Elevation (m) of each mechanistic glacier class. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

   * - ``URBAN_REGION_ID``
     - ``topounit,lsmlat,lsmlon``
     - Integer identifier linking urban grid cells to regional urban morphology datasets. (Requirement: Required when PCT_URBAN is used; optional otherwise)


Optional variables
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - **Variable**
     - **Dimensions**
     - **Description**

   * - ``APATITE_P``
     - ``lsmlat,lsmlon``
     - Soil phosphorus pool in apatite (primary mineral) form; used by phosphorus biogeochemistry when enabled.

   * - ``GLCMASK``
     - ``lsmlat,lsmlon``
     - Glacier mask read from a separate glacier file; ensures that glacier cells are a subset of the land mask. (Requirement: Optional (only used when a glacier mask file is provided))

   * - ``LABILE_P``
     - ``lsmlat,lsmlon``
     - Soil labile (readily available) phosphorus pool; used by P-cycle parameterizations.

   * - ``MONTHLY_HEIGHT_BOT``
     - ``time,lsmlat,lsmlon``
     - Monthly climatology of canopy bottom height (m) for vegetated landunits.

   * - ``MONTHLY_HEIGHT_TOP``
     - ``time,lsmlat,lsmlon``
     - Monthly climatology of canopy top height (m) for vegetated landunits.

   * - ``MONTHLY_LAI``
     - ``time,lsmlat,lsmlon``
     - Leaf area index (LAI; m2 leaf per m2 ground) monthly climatology; time is typically 12 months.

   * - ``MONTHLY_SAI``
     - ``time,lsmlat,lsmlon``
     - Stem area index (SAI) monthly climatology; time is typically 12 months.

   * - ``MaxTopounitElv``
     - ``lsmlat,lsmlon``
     - Maximum elevation (m) among topounits for each grid cell; read only if present. (Requirement: Optional (not required, but improves topounit characterization))

   * - ``NFERT``
     - ``topounit,cft,lsmlat,lsmlon``
     - Nitrogen fertilizer application for each crop functional type; if absent, values default to zero. (Requirement: Optional (values default to zero when absent))

   * - ``OCCLUDED_P``
     - ``lsmlat,lsmlon``
     - Soil occluded phosphorus pool (sorbed or otherwise inaccessible); part of multi-pool P parameterization.

   * - ``ORGANIC``
     - ``nlevsoi,lsmlat,lsmlon``
     - Soil organic matter or organic carbon per layer; used in biogeochemical and thermal calculations.

   * - ``PCT_GRVL``
     - ``nlevsoi,lsmlat,lsmlon``
     - Percent gravel content (0–100) in each soil layer; affects soil water storage and hydraulic conductivity.

   * - ``PFERT``
     - ``topounit,cft,lsmlat,lsmlon``
     - Phosphorus fertilizer application for each crop functional type; treated like NFERT.

   * - ``SECONDARY_P``
     - ``lsmlat,lsmlon``
     - Soil secondary mineral phosphorus pool; intermediate in reactivity between apatite and labile pools.

   * - ``TOPO2``
     - ``lsmlat,lsmlon``
     - Second topography field used in the ELM topounit framework; read only if present.

   * - ``TopounitAspect``
     - ``topounit,lsmlat,lsmlon``
     - Aspect (azimuth) of each topounit; optional and unused by default.

   * - ``TopounitAveElv``
     - ``topounit,lsmlat,lsmlon``
     - Average elevation (m) of each topounit; optional if available.

   * - ``TopounitElv``
     - ``topounit,lsmlat,lsmlon``
     - Average elevation (m) of each topounit; optional if available.

   * - ``TopounitFracArea``
     - ``topounit,lsmlat,lsmlon``
     - Fraction of the grid cell’s area represented by each topounit; read only if present.

   * - ``TopounitSlope``
     - ``topounit,lsmlat,lsmlon``
     - Slope (degrees) of each topounit; not currently used but may improve topographic representation.

   * - ``topoPerGrid``
     - ``lsmlat,lsmlon``
     - Number of topounits in each grid cell; if absent, the number of topounits is set to 1.

   * - ``xCell``
     - ``lsmlat,lsmlon``
     - Unstructured grid coordinates (e.g., MPAS x) when using an irregular mesh; the code reads them only if they exist.

   * - ``yCell``
     - ``lsmlat,lsmlon``
     - Unstructured grid coordinates (e.g., MPAS y) when using an irregular mesh; the code reads them only if they exist.

