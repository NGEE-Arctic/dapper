All surface variables
---------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``APATITE_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil phosphorus pool in apatite (primary mineral) form; used by phosphorus biogeochemistry when enabled.

      * - ``AREA``
        - ``lsmlat,lsmlon``
        - ``m2``
        - ``required``
        - ``core``, ``grid_topography``
        - Area (m2) of each land grid cell; used to weight land-surface properties and to compare with domain/topography.

      * - ``FGRD``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via ground/sprinkler/drip; complements FSURF. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FIRRIG``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction (0–1) of cropland that is irrigated; read only when irrigation data are enabled (firrig_data). (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FSURF``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via surface irrigation; required with FIRRIG when irrigation data are used. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``GLCMASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``glaciers``
        - Glacier mask read from a separate glacier file; ensures that glacier cells are a subset of the land mask. (Requirement: Optional (only used when a glacier mask file is provided))

      * - ``GLC_MEC``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``glaciers``
        - Integer glacier elevation-class index for each grid cell when using mechanistic glacier (MEC) landunits. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

      * - ``LABILE_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil labile (readily available) phosphorus pool; used by P-cycle parameterizations.

      * - ``LANDFRAC``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

      * - ``LANDFRAC_PFT``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction (0–1) of the global gridcell that is land in the PFT landunit; used to weight PFT tiles relative to the gridcell.

      * - ``LANDMASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``
        - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.

      * - ``LATIXY``
        - ``lsmlat,lsmlon``
        - ``degrees_north``
        - ``required``
        - ``core``
        - Latitude (degrees) of the centre of each land grid cell.

      * - ``LONGXY``
        - ``lsmlat,lsmlon``
        - ``degrees_east``
        - ``required``
        - ``core``
        - Longitude (degrees) of the centre of each land grid cell; used to compare the surface file with the domain and topography files.

      * - ``MONTHLY_HEIGHT_BOT``
        - ``time,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``vegetation_structure``
        - Monthly climatology of canopy bottom height (m) for vegetated landunits.

      * - ``MONTHLY_HEIGHT_TOP``
        - ``time,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``vegetation_structure``
        - Monthly climatology of canopy top height (m) for vegetated landunits.

      * - ``MONTHLY_LAI``
        - ``time,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``vegetation_structure``
        - Leaf area index (LAI; m2 leaf per m2 ground) monthly climatology; time is typically 12 months.

      * - ``MONTHLY_SAI``
        - ``time,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``vegetation_structure``
        - Stem area index (SAI) monthly climatology; time is typically 12 months.

      * - ``MaxTopounitElv``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Maximum elevation (m) among topounits for each grid cell; read only if present. (Requirement: Optional (not required, but improves topounit characterization))

      * - ``NFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``topounits``
        - Nitrogen fertilizer application for each crop functional type; if absent, values default to zero. (Requirement: Optional (values default to zero when absent))

      * - ``OCCLUDED_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil occluded phosphorus pool (sorbed or otherwise inaccessible); part of multi-pool P parameterization.

      * - ``ORGANIC``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``soil_properties``
        - Soil organic matter or organic carbon per layer; used in biogeochemical and thermal calculations.

      * - ``PCT_CFT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each crop functional type; code aborts if missing when cft dimension exists. (Requirement: Required when the surface file includes crop functional types (cft_size > 0))

      * - ``PCT_CLAY``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``soil_properties``
        - Soil clay percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.

      * - ``PCT_CROP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Percent of grid cell area covered by cropland.

      * - ``PCT_FCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is flat-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_GLACIER``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Fraction of grid cell area covered by simple glacier landunits within each topounit. (Requirement: Required unless glacier MEC landunits are used; optional when create_glacier_mec_landunit is true)

      * - ``PCT_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Percent of grid cell area assigned to mechanistic glacier classes (accumulation/ablation, etc.). (Requirement: Required when glacier MEC landunits are created; optional otherwise)

      * - ``PCT_GRVL``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``optional``
        - ``soil_properties``
        - Percent gravel content (0–100) in each soil layer; affects soil water storage and hydraulic conductivity.

      * - ``PCT_HCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation landunit that is high-centered polygons (polygonal tundra). (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_LAKE``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage of each grid cell area that is lake within each topounit; used for lake landunit weighting.

      * - ``PCT_LCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is low-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_NATVEG``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Percent of grid cell area covered by natural vegetation (non-crop vegetated landunit).

      * - ``PCT_NAT_PFT``
        - ``topounit,natpft,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each natural plant functional type; code aborts if missing.

      * - ``PCT_SAND``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``soil_properties``
        - Soil sand percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.

      * - ``PCT_URBAN``
        - ``topounit,numurbl,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``topounits``, ``urban``
        - Fraction of each grid cell that is urban, for each density class in the multi-density urban scheme. (Requirement: Required when the multi-density urban model is used (nlevurb > 0); optional otherwise)

      * - ``PCT_WETLAND``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage (0–100) of each grid cell area covered by wetlands within each topounit.

      * - ``PFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``phosphorus_cycle``, ``topounits``
        - Phosphorus fertilizer application for each crop functional type; treated like NFERT.

      * - ``PFTDATA_MASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Mask marking land grid cells that contain valid plant functional type (PFT) data; if missing the code stops with an error.

      * - ``SECONDARY_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil secondary mineral phosphorus pool; intermediate in reactivity between apatite and labile pools.

      * - ``SINSL_COSAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * cos(aspect) for each grid cell; used in solar-radiation calculations. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SINSL_SINAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SKY_VIEW``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Sky-view factor (0–1) describing horizon obstruction by surrounding terrain; needed by the TOP solar-radiation scheme. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SLOPE``
        - ``lsmlat,lsmlon``
        - ``degrees``
        - ``required``
        - ``grid_topography``
        - Mean surface slope (degrees) for each land grid cell; used in topographic and hydrologic parameterizations.

      * - ``STDEV_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; if STDEV_ELEV is missing the code tries STD_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``STD_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; alternative name to STDEV_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``TERRAIN_CONFIG``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Terrain configuration parameter used in the TOP scheme to account for terrain-induced shading. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``TOPO``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``required``
        - ``grid_topography``
        - Mean elevation (m) of each land grid cell read from the topography file; required by surfrd_get_topo.

      * - ``TOPO2``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Second topography field used in the ELM topounit framework; read only if present.

      * - ``TOPO_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``glaciers``, ``grid_topography``, ``topounits``
        - Elevation (m) of each mechanistic glacier class. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

      * - ``TopounitAspect``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Aspect (azimuth) of each topounit; optional and unused by default.

      * - ``TopounitAveElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitFracArea``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Fraction of the grid cell’s area represented by each topounit; read only if present.

      * - ``TopounitSlope``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Slope (degrees) of each topounit; not currently used but may improve topographic representation.

      * - ``URBAN_REGION_ID``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``topounits``, ``urban``
        - Integer identifier linking urban grid cells to regional urban morphology datasets. (Requirement: Required when PCT_URBAN is used; optional otherwise)

      * - ``frac``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

      * - ``mask``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``
        - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.

      * - ``topoPerGrid``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Number of topounits in each grid cell; if absent, the number of topounits is set to 1.

      * - ``xCell``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``unstructured_grid``
        - Unstructured grid coordinates (e.g., MPAS x) when using an irregular mesh; the code reads them only if they exist.

      * - ``yCell``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``unstructured_grid``
        - Unstructured grid coordinates (e.g., MPAS y) when using an irregular mesh; the code reads them only if they exist.


Variables by context
--------------------

Context: core
-------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``AREA``
        - ``lsmlat,lsmlon``
        - ``m2``
        - ``required``
        - ``core``, ``grid_topography``
        - Area (m2) of each land grid cell; used to weight land-surface properties and to compare with domain/topography.

      * - ``LANDFRAC``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

      * - ``LANDFRAC_PFT``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction (0–1) of the global gridcell that is land in the PFT landunit; used to weight PFT tiles relative to the gridcell.

      * - ``LANDMASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``
        - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.

      * - ``LATIXY``
        - ``lsmlat,lsmlon``
        - ``degrees_north``
        - ``required``
        - ``core``
        - Latitude (degrees) of the centre of each land grid cell.

      * - ``LONGXY``
        - ``lsmlat,lsmlon``
        - ``degrees_east``
        - ``required``
        - ``core``
        - Longitude (degrees) of the centre of each land grid cell; used to compare the surface file with the domain and topography files.

      * - ``PFTDATA_MASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Mask marking land grid cells that contain valid plant functional type (PFT) data; if missing the code stops with an error.

      * - ``frac``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

      * - ``mask``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``
        - Binary mask (1 = land, 0 = not-land) defining which grid cells are part of the land model; read from the surface file or domain file.


Context: crops_irrigation
-------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``FGRD``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via ground/sprinkler/drip; complements FSURF. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FIRRIG``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction (0–1) of cropland that is irrigated; read only when irrigation data are enabled (firrig_data). (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FSURF``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via surface irrigation; required with FIRRIG when irrigation data are used. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``NFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``topounits``
        - Nitrogen fertilizer application for each crop functional type; if absent, values default to zero. (Requirement: Optional (values default to zero when absent))

      * - ``PCT_CFT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each crop functional type; code aborts if missing when cft dimension exists. (Requirement: Required when the surface file includes crop functional types (cft_size > 0))

      * - ``PCT_CROP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Percent of grid cell area covered by cropland.

      * - ``PFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``phosphorus_cycle``, ``topounits``
        - Phosphorus fertilizer application for each crop functional type; treated like NFERT.


Context: glaciers
-----------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``GLCMASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``glaciers``
        - Glacier mask read from a separate glacier file; ensures that glacier cells are a subset of the land mask. (Requirement: Optional (only used when a glacier mask file is provided))

      * - ``GLC_MEC``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``glaciers``
        - Integer glacier elevation-class index for each grid cell when using mechanistic glacier (MEC) landunits. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

      * - ``PCT_GLACIER``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Fraction of grid cell area covered by simple glacier landunits within each topounit. (Requirement: Required unless glacier MEC landunits are used; optional when create_glacier_mec_landunit is true)

      * - ``PCT_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Percent of grid cell area assigned to mechanistic glacier classes (accumulation/ablation, etc.). (Requirement: Required when glacier MEC landunits are created; optional otherwise)

      * - ``TOPO_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``glaciers``, ``grid_topography``, ``topounits``
        - Elevation (m) of each mechanistic glacier class. (Requirement: Required when glacier MEC landunits are used; optional otherwise)


Context: grid_topography
------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``AREA``
        - ``lsmlat,lsmlon``
        - ``m2``
        - ``required``
        - ``core``, ``grid_topography``
        - Area (m2) of each land grid cell; used to weight land-surface properties and to compare with domain/topography.

      * - ``MaxTopounitElv``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Maximum elevation (m) among topounits for each grid cell; read only if present. (Requirement: Optional (not required, but improves topounit characterization))

      * - ``SINSL_COSAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * cos(aspect) for each grid cell; used in solar-radiation calculations. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SINSL_SINAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SKY_VIEW``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Sky-view factor (0–1) describing horizon obstruction by surrounding terrain; needed by the TOP solar-radiation scheme. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SLOPE``
        - ``lsmlat,lsmlon``
        - ``degrees``
        - ``required``
        - ``grid_topography``
        - Mean surface slope (degrees) for each land grid cell; used in topographic and hydrologic parameterizations.

      * - ``STDEV_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; if STDEV_ELEV is missing the code tries STD_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``STD_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; alternative name to STDEV_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``TERRAIN_CONFIG``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Terrain configuration parameter used in the TOP scheme to account for terrain-induced shading. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``TOPO``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``required``
        - ``grid_topography``
        - Mean elevation (m) of each land grid cell read from the topography file; required by surfrd_get_topo.

      * - ``TOPO2``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Second topography field used in the ELM topounit framework; read only if present.

      * - ``TOPO_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``glaciers``, ``grid_topography``, ``topounits``
        - Elevation (m) of each mechanistic glacier class. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

      * - ``TopounitAspect``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Aspect (azimuth) of each topounit; optional and unused by default.

      * - ``TopounitAveElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitFracArea``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Fraction of the grid cell’s area represented by each topounit; read only if present.

      * - ``TopounitSlope``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Slope (degrees) of each topounit; not currently used but may improve topographic representation.

      * - ``topoPerGrid``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Number of topounits in each grid cell; if absent, the number of topounits is set to 1.


Context: inland_water
---------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``PCT_LAKE``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage of each grid cell area that is lake within each topounit; used for lake landunit weighting.

      * - ``PCT_WETLAND``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage (0–100) of each grid cell area covered by wetlands within each topounit.


Context: land_cover
-------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``LANDFRAC``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.

      * - ``LANDFRAC_PFT``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction (0–1) of the global gridcell that is land in the PFT landunit; used to weight PFT tiles relative to the gridcell.

      * - ``PCT_CFT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each crop functional type; code aborts if missing when cft dimension exists. (Requirement: Required when the surface file includes crop functional types (cft_size > 0))

      * - ``PCT_CROP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Percent of grid cell area covered by cropland.

      * - ``PCT_FCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is flat-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_GLACIER``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Fraction of grid cell area covered by simple glacier landunits within each topounit. (Requirement: Required unless glacier MEC landunits are used; optional when create_glacier_mec_landunit is true)

      * - ``PCT_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Percent of grid cell area assigned to mechanistic glacier classes (accumulation/ablation, etc.). (Requirement: Required when glacier MEC landunits are created; optional otherwise)

      * - ``PCT_HCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation landunit that is high-centered polygons (polygonal tundra). (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_LAKE``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage of each grid cell area that is lake within each topounit; used for lake landunit weighting.

      * - ``PCT_LCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is low-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_NATVEG``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Percent of grid cell area covered by natural vegetation (non-crop vegetated landunit).

      * - ``PCT_NAT_PFT``
        - ``topounit,natpft,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each natural plant functional type; code aborts if missing.

      * - ``PCT_URBAN``
        - ``topounit,numurbl,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``topounits``, ``urban``
        - Fraction of each grid cell that is urban, for each density class in the multi-density urban scheme. (Requirement: Required when the multi-density urban model is used (nlevurb > 0); optional otherwise)

      * - ``PCT_WETLAND``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage (0–100) of each grid cell area covered by wetlands within each topounit.

      * - ``PFTDATA_MASK``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Mask marking land grid cells that contain valid plant functional type (PFT) data; if missing the code stops with an error.

      * - ``frac``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``required``
        - ``core``, ``land_cover``
        - Fraction of each grid cell that is land (0–1); used to weight surface properties; a missing variable causes a fatal error.


Context: phosphorus_cycle
-------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``APATITE_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil phosphorus pool in apatite (primary mineral) form; used by phosphorus biogeochemistry when enabled.

      * - ``LABILE_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil labile (readily available) phosphorus pool; used by P-cycle parameterizations.

      * - ``OCCLUDED_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil occluded phosphorus pool (sorbed or otherwise inaccessible); part of multi-pool P parameterization.

      * - ``PFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``phosphorus_cycle``, ``topounits``
        - Phosphorus fertilizer application for each crop functional type; treated like NFERT.

      * - ``SECONDARY_P``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``phosphorus_cycle``
        - Soil secondary mineral phosphorus pool; intermediate in reactivity between apatite and labile pools.


Context: polygonal_tundra
-------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``PCT_FCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is flat-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_HCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation landunit that is high-centered polygons (polygonal tundra). (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_LCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is low-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)


Context: soil_properties
------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``ORGANIC``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``soil_properties``
        - Soil organic matter or organic carbon per layer; used in biogeochemical and thermal calculations.

      * - ``PCT_CLAY``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``soil_properties``
        - Soil clay percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.

      * - ``PCT_GRVL``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``optional``
        - ``soil_properties``
        - Percent gravel content (0–100) in each soil layer; affects soil water storage and hydraulic conductivity.

      * - ``PCT_SAND``
        - ``nlevsoi,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``soil_properties``
        - Soil sand percentage by mass (0–100) in each soil layer; controls hydraulic and thermal properties.


Context: topographic_radiation
------------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``SINSL_COSAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * cos(aspect) for each grid cell; used in solar-radiation calculations. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SINSL_SINAS``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``SKY_VIEW``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Sky-view factor (0–1) describing horizon obstruction by surrounding terrain; needed by the TOP solar-radiation scheme. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``STDEV_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; if STDEV_ELEV is missing the code tries STD_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``STD_ELEV``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Standard deviation of elevation (m) used by the TOP solar-radiation parameterization; alternative name to STDEV_ELEV. (Requirement: Required when TOP solar-radiation is used; optional otherwise)

      * - ``TERRAIN_CONFIG``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``grid_topography``, ``topographic_radiation``
        - Terrain configuration parameter used in the TOP scheme to account for terrain-induced shading. (Requirement: Required when TOP solar-radiation is used; optional otherwise)


Context: topounits
------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``FGRD``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via ground/sprinkler/drip; complements FSURF. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FIRRIG``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction (0–1) of cropland that is irrigated; read only when irrigation data are enabled (firrig_data). (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``FSURF``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``crops_irrigation``, ``topounits``
        - Fraction of irrigation water applied via surface irrigation; required with FIRRIG when irrigation data are used. (Requirement: Required when firrig_data is true; optional otherwise)

      * - ``MaxTopounitElv``
        - ``lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Maximum elevation (m) among topounits for each grid cell; read only if present. (Requirement: Optional (not required, but improves topounit characterization))

      * - ``NFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``topounits``
        - Nitrogen fertilizer application for each crop functional type; if absent, values default to zero. (Requirement: Optional (values default to zero when absent))

      * - ``PCT_CFT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each crop functional type; code aborts if missing when cft dimension exists. (Requirement: Required when the surface file includes crop functional types (cft_size > 0))

      * - ``PCT_CROP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``crops_irrigation``, ``land_cover``, ``topounits``
        - Percent of grid cell area covered by cropland.

      * - ``PCT_FCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is flat-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_GLACIER``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Fraction of grid cell area covered by simple glacier landunits within each topounit. (Requirement: Required unless glacier MEC landunits are used; optional when create_glacier_mec_landunit is true)

      * - ``PCT_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``glaciers``, ``land_cover``, ``topounits``
        - Percent of grid cell area assigned to mechanistic glacier classes (accumulation/ablation, etc.). (Requirement: Required when glacier MEC landunits are created; optional otherwise)

      * - ``PCT_HCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation landunit that is high-centered polygons (polygonal tundra). (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_LAKE``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage of each grid cell area that is lake within each topounit; used for lake landunit weighting.

      * - ``PCT_LCP``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``polygonal_tundra``, ``topounits``
        - Fraction of natural vegetation that is low-centered polygons. (Requirement: Required when use_polygonal_tundra is true; optional otherwise)

      * - ``PCT_NATVEG``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Percent of grid cell area covered by natural vegetation (non-crop vegetated landunit).

      * - ``PCT_NAT_PFT``
        - ``topounit,natpft,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``land_cover``, ``topounits``
        - Fraction of vegetated area allocated to each natural plant functional type; code aborts if missing.

      * - ``PCT_URBAN``
        - ``topounit,numurbl,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``topounits``, ``urban``
        - Fraction of each grid cell that is urban, for each density class in the multi-density urban scheme. (Requirement: Required when the multi-density urban model is used (nlevurb > 0); optional otherwise)

      * - ``PCT_WETLAND``
        - ``topounit,lsmlat,lsmlon``
        - ``percent``
        - ``required``
        - ``inland_water``, ``land_cover``, ``topounits``
        - Percentage (0–100) of each grid cell area covered by wetlands within each topounit.

      * - ``PFERT``
        - ``topounit,cft,lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``crops_irrigation``, ``phosphorus_cycle``, ``topounits``
        - Phosphorus fertilizer application for each crop functional type; treated like NFERT.

      * - ``TOPO2``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Second topography field used in the ELM topounit framework; read only if present.

      * - ``TOPO_GLC_MEC``
        - ``topounit,nglcec,lsmlat,lsmlon``
        - ``m``
        - ``conditional``
        - ``glaciers``, ``grid_topography``, ``topounits``
        - Elevation (m) of each mechanistic glacier class. (Requirement: Required when glacier MEC landunits are used; optional otherwise)

      * - ``TopounitAspect``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Aspect (azimuth) of each topounit; optional and unused by default.

      * - ``TopounitAveElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitElv``
        - ``topounit,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Average elevation (m) of each topounit; optional if available.

      * - ``TopounitFracArea``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Fraction of the grid cell’s area represented by each topounit; read only if present.

      * - ``TopounitSlope``
        - ``topounit,lsmlat,lsmlon``
        - ``degrees``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Slope (degrees) of each topounit; not currently used but may improve topographic representation.

      * - ``URBAN_REGION_ID``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``topounits``, ``urban``
        - Integer identifier linking urban grid cells to regional urban morphology datasets. (Requirement: Required when PCT_URBAN is used; optional otherwise)

      * - ``topoPerGrid``
        - ``lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``grid_topography``, ``topounits``
        - Number of topounits in each grid cell; if absent, the number of topounits is set to 1.


Context: unstructured_grid
--------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``xCell``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``unstructured_grid``
        - Unstructured grid coordinates (e.g., MPAS x) when using an irregular mesh; the code reads them only if they exist.

      * - ``yCell``
        - ``lsmlat,lsmlon``
        - ``unknown``
        - ``optional``
        - ``unstructured_grid``
        - Unstructured grid coordinates (e.g., MPAS y) when using an irregular mesh; the code reads them only if they exist.


Context: urban
--------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``PCT_URBAN``
        - ``topounit,numurbl,lsmlat,lsmlon``
        - ``percent``
        - ``conditional``
        - ``land_cover``, ``topounits``, ``urban``
        - Fraction of each grid cell that is urban, for each density class in the multi-density urban scheme. (Requirement: Required when the multi-density urban model is used (nlevurb > 0); optional otherwise)

      * - ``URBAN_REGION_ID``
        - ``topounit,lsmlat,lsmlon``
        - ``unitless``
        - ``conditional``
        - ``topounits``, ``urban``
        - Integer identifier linking urban grid cells to regional urban morphology datasets. (Requirement: Required when PCT_URBAN is used; optional otherwise)


Context: vegetation_structure
-----------------------------

.. container:: scroll-x

   .. list-table::
      :header-rows: 1

      * - **Variable**
        - **Dimensions**
        - **Units**
        - **Required level**
        - **Contexts**
        - **Description**

      * - ``MONTHLY_HEIGHT_BOT``
        - ``time,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``vegetation_structure``
        - Monthly climatology of canopy bottom height (m) for vegetated landunits.

      * - ``MONTHLY_HEIGHT_TOP``
        - ``time,lsmlat,lsmlon``
        - ``m``
        - ``optional``
        - ``vegetation_structure``
        - Monthly climatology of canopy top height (m) for vegetated landunits.

      * - ``MONTHLY_LAI``
        - ``time,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``vegetation_structure``
        - Leaf area index (LAI; m2 leaf per m2 ground) monthly climatology; time is typically 12 months.

      * - ``MONTHLY_SAI``
        - ``time,lsmlat,lsmlon``
        - ``unitless``
        - ``optional``
        - ``vegetation_structure``
        - Stem area index (SAI) monthly climatology; time is typically 12 months.


