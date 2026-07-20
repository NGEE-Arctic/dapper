"""
Canonical ELM/ELM surface-variable spec for Dapper.

This file is the single place where new surface variables
and their basic metadata (dims, description, requiredness)
are added. Other modules (schema, validation, docs) should
import this and *not* duplicate this information elsewhere.

It was initially generated from a ChatGPT scrape of a public ELM surface file
and E3SM Fortran code, assembled by Rich Fiorella. Jon then added more
parameters based on comparing a handful of ELM surface files. Additional
parameters should be added to the SURFACE_VAR_SPECS dictionary below.
"""

from __future__ import annotations

SURFACE_VAR_SPECS = {
    "LONGXY": {
        "dims": "lsmlat,lsmlon",
        "doc": "Longitude (degrees) of the centre of each land grid cell; "
        "used to compare the surface file with the domain and "
        "topography files.",
        "required_level": "required",
        "units": "degrees_east",
        "contexts": ["core"],
    },
    "LATIXY": {
        "dims": "lsmlat,lsmlon",
        "doc": "Latitude (degrees) of the centre of each land grid cell.",
        "required_level": "required",
        "units": "degrees_north",
        "contexts": ["core"],
    },
    "LANDMASK": {
        "dims": "lsmlat,lsmlon",
        "doc": "Binary mask (1 = land, 0 = not-land) defining which grid "
        "cells are part of the land model; read from the surface "
        "file or domain file.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core"],
    },
    "mask": {
        "dims": "lsmlat,lsmlon",
        "doc": "Binary mask (1 = land, 0 = not-land) defining which grid "
        "cells are part of the land model; read from the surface file "
        "or domain file.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core"],
    },
    "LANDFRAC": {
        "dims": "lsmlat,lsmlon",
        "doc": "Fraction of each grid cell that is land (0–1); used to "
        "weight surface properties; a missing variable causes a "
        "fatal error.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core", "land_cover"],
    },
    "frac": {
        "dims": "lsmlat,lsmlon",
        "doc": "Fraction of each grid cell that is land (0–1); used to "
        "weight surface properties; a missing variable causes a fatal "
        "error.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core", "land_cover"],
    },
    "PFTDATA_MASK": {
        "dims": "lsmlat,lsmlon",
        "doc": "Mask marking land grid cells that contain valid "
        "plant functional type (PFT) data; if missing the "
        "code stops with an error.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core", "land_cover"],
    },
    "xCell": {
        "dims": "lsmlat,lsmlon",
        "doc": "Unstructured grid coordinates (e.g., MPAS x) when using an "
        "irregular mesh; the code reads them only if they exist.",
        "required_level": "optional",
        "units": "unknown",
        "contexts": ["unstructured_grid"],
    },
    "yCell": {
        "dims": "lsmlat,lsmlon",
        "doc": "Unstructured grid coordinates (e.g., MPAS y) when using an "
        "irregular mesh; the code reads them only if they exist.",
        "required_level": "optional",
        "units": "unknown",
        "contexts": ["unstructured_grid"],
    },
    "GLCMASK": {
        "dims": "lsmlat,lsmlon",
        "doc": "Glacier mask read from a separate glacier file; ensures "
        "that glacier cells are a subset of the land mask.",
        "required_level": "optional",
        "attrs": {
            "requirement": "Optional (only used when a glacier mask file is provided)"
        },
        "units": "unitless",
        "contexts": ["glaciers"],
    },
    "TOPO": {
        "dims": "lsmlat,lsmlon",
        "doc": "Mean elevation (m) of each land grid cell read from the "
        "topography file; required by surfrd_get_topo.",
        "required_level": "required",
        "units": "m",
        "contexts": ["grid_topography"],
    },
    "STDEV_ELEV": {
        "dims": "lsmlat,lsmlon",
        "doc": "Standard deviation of elevation (m) used by the TOP "
        "solar-radiation parameterization; if STDEV_ELEV is "
        "missing the code tries STD_ELEV.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP solar-radiation is "
            "used; optional otherwise"
        },
        "units": "m",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "STD_ELEV": {
        "dims": "lsmlat,lsmlon",
        "doc": "Standard deviation of elevation (m) used by the TOP "
        "solar-radiation parameterization; alternative name to "
        "STDEV_ELEV.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP solar-radiation is "
            "used; optional otherwise"
        },
        "units": "m",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "SKY_VIEW": {
        "dims": "lsmlat,lsmlon",
        "doc": "Sky-view factor (0–1) describing horizon obstruction by "
        "surrounding terrain; needed by the TOP solar-radiation "
        "scheme.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP solar-radiation is "
            "used; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "TERRAIN_CONFIG": {
        "dims": "lsmlat,lsmlon",
        "doc": "Terrain configuration parameter used in the TOP "
        "scheme to account for terrain-induced shading.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP "
            "solar-radiation is used; "
            "optional otherwise"
        },
        "units": "unitless",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "SINSL_COSAS": {
        "dims": "lsmlat,lsmlon",
        "doc": "sin(slope) * cos(aspect) for each grid cell; used in "
        "solar-radiation calculations.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP solar-radiation "
            "is used; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "SINSL_SINAS": {
        "dims": "lsmlat,lsmlon",
        "doc": "sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when TOP solar-radiation "
            "is used; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["grid_topography", "topographic_radiation"],
    },
    "PCT_WETLAND": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Percentage (0–100) of each grid cell area covered by "
        "wetlands within each topounit.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["land_cover", "inland_water", "topounits"],
    },
    "PCT_LAKE": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Percentage of each grid cell area that is lake within "
        "each topounit; used for lake landunit weighting.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["land_cover", "inland_water", "topounits"],
    },
    "PCT_GLACIER": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of grid cell area covered by simple glacier "
        "landunits within each topounit.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required unless glacier MEC "
            "landunits are used; optional when "
            "create_glacier_mec_landunit is "
            "true"
        },
        "units": "percent",
        "contexts": ["land_cover", "glaciers", "topounits"],
    },
    "PCT_URBAN": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Fraction of each grid cell that is urban, for each "
        "density class in the multi-density urban scheme.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when the multi-density urban "
            "model is used (nlevurb > 0); optional "
            "otherwise"
        },
        "units": "percent",
        "contexts": ["land_cover", "urban", "topounits"],
    },
    "URBAN_REGION_ID": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Integer identifier linking urban grid cells to "
        "regional urban morphology datasets.",
        "required_level": "conditional",
        "attrs": {"requirement": "Required when PCT_URBAN is used; optional otherwise"},
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "PCT_GLC_MEC": {
        "dims": "nglcec,topounit,lsmlat,lsmlon",
        "doc": "Percent of grid cell area assigned to mechanistic "
        "glacier classes (accumulation/ablation, etc.).",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when glacier MEC landunits "
            "are created; optional otherwise"
        },
        "units": "percent",
        "contexts": ["land_cover", "glaciers", "topounits"],
    },
    "TOPO_GLC_MEC": {
        "dims": "nglcec,topounit,lsmlat,lsmlon",
        "doc": "Elevation (m) of each mechanistic glacier class.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when glacier MEC "
            "landunits are used; optional "
            "otherwise"
        },
        "units": "m",
        "contexts": ["grid_topography", "glaciers", "topounits"],
    },
    "PCT_NATVEG": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Percent of grid cell area covered by natural "
        "vegetation (non-crop vegetated landunit).",
        "required_level": "required",
        "units": "percent",
        "contexts": ["land_cover", "topounits"],
    },
    "PCT_CROP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Percent of grid cell area covered by cropland.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["land_cover", "crops_irrigation", "topounits"],
    },
    "PCT_HCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of natural vegetation landunit that is "
        "high-centered polygons (polygonal tundra).",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when use_polygonal_tundra is "
            "true; optional otherwise"
        },
        "units": "percent",
        "contexts": ["land_cover", "polygonal_tundra", "topounits"],
    },
    "PCT_FCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of natural vegetation that is flat-centered polygons.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when use_polygonal_tundra is "
            "true; optional otherwise"
        },
        "units": "percent",
        "contexts": ["land_cover", "polygonal_tundra", "topounits"],
    },
    "PCT_LCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of natural vegetation that is low-centered polygons.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when use_polygonal_tundra is "
            "true; optional otherwise"
        },
        "units": "percent",
        "contexts": ["land_cover", "polygonal_tundra", "topounits"],
    },
    "PCT_CFT": {
        "dims": "cft,topounit,lsmlat,lsmlon",
        "doc": "Fraction of vegetated area allocated to each crop "
        "functional type; code aborts if missing when cft "
        "dimension exists.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when the surface file includes "
            "crop functional types (cft_size > 0)"
        },
        "units": "percent",
        "contexts": ["land_cover", "crops_irrigation", "topounits"],
    },
    "NFERT": {
        "dims": "cft,topounit,lsmlat,lsmlon",
        "doc": "Nitrogen fertilizer application for each crop functional "
        "type; if absent, values default to zero.",
        "required_level": "optional",
        "attrs": {"requirement": "Optional (values default to zero when absent)"},
        "units": "unknown",
        "contexts": ["crops_irrigation", "topounits"],
    },
    "PFERT": {
        "dims": "cft,topounit,lsmlat,lsmlon",
        "doc": "Phosphorus fertilizer application for each crop functional "
        "type; treated like NFERT.",
        "required_level": "optional",
        "units": "unknown",
        "contexts": ["crops_irrigation", "topounits", "phosphorus_cycle"],
    },
    "PCT_NAT_PFT": {
        "dims": "natpft,topounit,lsmlat,lsmlon",
        "doc": "Fraction of vegetated area allocated to each natural "
        "plant functional type; code aborts if missing.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["land_cover", "topounits"],
    },
    "FIRRIG": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction (0–1) of cropland that is irrigated; read only "
        "when irrigation data are enabled (firrig_data).",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["crops_irrigation", "topounits"],
    },
    "FSURF": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of irrigation water applied via surface "
        "irrigation; required with FIRRIG when irrigation data are "
        "used.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["crops_irrigation", "topounits"],
    },
    "FGRD": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of irrigation water applied via "
        "ground/sprinkler/drip; complements FSURF.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["crops_irrigation", "topounits"],
    },
    "MaxTopounitElv": {
        "dims": "lsmlat,lsmlon",
        "doc": "Maximum topounits elevation in each grid cell; a summary "
        "statistic reflecting the highest elevation among topounits.",
        "required_level": "optional",
        "attrs": {
            "requirement": "Optional (not required, but "
            "improves topounit "
            "characterization)"
        },
        "units": "m",
        "contexts": ["grid_topography", "topounits"],
    },
    "topoPerGrid": {
        "dims": "lsmlat,lsmlon",
        "doc": "Number of topounits in each grid cell; if absent, the "
        "number of topounits is set to 1.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["grid_topography", "topounits"],
    },
    "TopounitFracArea": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of the grid cell’s area represented by "
        "each topounit; read only if present.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["grid_topography", "topounits"],
    },
    "TopounitAveElv": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Average elevation (m) of each topounit; optional if available.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["grid_topography", "topounits"],
    },
    "TopounitElv": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Average elevation (m) of each topounit; optional if available.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["grid_topography", "topounits"],
    },
    "TopounitSlope": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Slope (degrees) of each topounit; not currently "
        "used but may improve topographic representation.",
        "required_level": "optional",
        "units": "degrees",
        "contexts": ["grid_topography", "topounits"],
    },
    "TopounitAspect": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Aspect (azimuth) of each topounit; optional and unused by default.",
        "required_level": "optional",
        "units": "degrees",
        "contexts": ["grid_topography", "topounits"],
    },
    "TOPO2": {
        "dims": "lsmlat,lsmlon",
        "doc": "Weighted average of topounits elevation in each grid cell; "
        "a summary statistic for topounit-based elevation characterization.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["grid_topography", "topounits"],
    },
    "AREA": {
        "dims": "lsmlat,lsmlon",
        "doc": "Area (m2) of each land grid cell; used to weight "
        "land-surface properties and to compare with "
        "domain/topography.",
        "required_level": "required",
        "units": "m2",
        "contexts": ["core", "grid_topography"],
    },
    "LANDFRAC_PFT": {
        "dims": "lsmlat,lsmlon",
        "doc": "Fraction (0–1) of the global gridcell that is land "
        "in the PFT landunit; used to weight PFT tiles "
        "relative to the gridcell.",
        "required_level": "required",
        "units": "unitless",
        "contexts": ["core", "land_cover"],
    },
    "SLOPE": {
        "dims": "lsmlat,lsmlon",
        "doc": "Mean surface slope (degrees) for each land grid cell; used "
        "in topographic and hydrologic parameterizations.",
        "required_level": "required",
        "units": "degrees",
        "contexts": ["grid_topography"],
    },
    "PCT_SAND": {
        "dims": "nlevsoi,topounit,lsmlat,lsmlon",
        "doc": "Soil sand percentage by mass (0–100) in each soil layer; "
        "controls hydraulic and thermal properties.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["soil_properties", "topounits"],
    },
    "PCT_CLAY": {
        "dims": "nlevsoi,topounit,lsmlat,lsmlon",
        "doc": "Soil clay percentage by mass (0–100) in each soil layer; "
        "controls hydraulic and thermal properties.",
        "required_level": "required",
        "units": "percent",
        "contexts": ["soil_properties", "topounits"],
    },
    "ORGANIC": {
        "dims": "nlevsoi,topounit,lsmlat,lsmlon",
        "doc": "Organic matter density per soil layer; used in "
        "biogeochemical and thermal calculations.",
        "required_level": "optional",
        "units": "kg/m3 (assumed carbon content 0.58 gC per gOM)",
        "contexts": ["soil_properties", "topounits"],
    },
    "PCT_GRVL": {
        "dims": "nlevsoi,topounit,lsmlat,lsmlon",
        "doc": "Percent gravel content (0–100) in each soil layer; "
        "affects soil water storage and hydraulic conductivity.",
        "required_level": "optional",
        "units": "percent",
        "contexts": ["soil_properties", "topounits"],
    },
    "GLC_MEC": {
        "dims": "lsmlat,lsmlon",
        "doc": "Integer glacier elevation-class index for each grid cell "
        "when using mechanistic glacier (MEC) landunits.",
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when glacier MEC landunits are "
            "used; optional otherwise"
        },
        "units": "unitless",
        "contexts": ["glaciers"],
    },
    "MONTHLY_LAI": {
        "dims": "time,lsmpft,topounit,lsmlat,lsmlon",
        "doc": "Leaf area index (LAI; m2 leaf per m2 ground) monthly "
        "climatology; time is typically 12 months.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["vegetation_structure", "topounits"],
    },
    "MONTHLY_SAI": {
        "dims": "time,lsmpft,topounit,lsmlat,lsmlon",
        "doc": "Stem area index (SAI) monthly climatology; time is "
        "typically 12 months.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["vegetation_structure", "topounits"],
    },
    "MONTHLY_HEIGHT_TOP": {
        "dims": "time,lsmpft,topounit,lsmlat,lsmlon",
        "doc": "Monthly climatology of canopy top height (m) for vegetated landunits.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["vegetation_structure", "topounits"],
    },
    "MONTHLY_HEIGHT_BOT": {
        "dims": "time,lsmpft,topounit,lsmlat,lsmlon",
        "doc": "Monthly climatology of canopy bottom height "
        "(m) for vegetated landunits.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["vegetation_structure", "topounits"],
    },
    "APATITE_P": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Apatite phosphorus; soil phosphorus pool in apatite "
        "(primary mineral) form; used by phosphorus biogeochemistry when enabled.",
        "required_level": "optional",
        "units": "gP/m2",
        "contexts": ["phosphorus_cycle", "topounits"],
    },
    "LABILE_P": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Labile inorganic phosphorus; soil labile (readily available) "
        "phosphorus pool; used by P-cycle parameterizations.",
        "required_level": "optional",
        "units": "gP/m2",
        "contexts": ["phosphorus_cycle", "topounits"],
    },
    "OCCLUDED_P": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Occluded phosphorus; soil occluded phosphorus pool (sorbed or "
        "otherwise inaccessible); part of multi-pool P parameterization.",
        "required_level": "optional",
        "units": "gP/m2",
        "contexts": ["phosphorus_cycle", "topounits"],
    },
    "SECONDARY_P": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Secondary mineral phosphorus; soil secondary mineral phosphorus "
        "pool; intermediate in reactivity between apatite and labile pools.",
        "required_level": "optional",
        "units": "gP/m2",
        "contexts": ["phosphorus_cycle", "topounits"],
    },
    "ALB_IMPROAD_DIF": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Diffuse albedo of impervious road; spectral-dependent surface "
        "reflectance for urban impervious surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_IMPROAD_DIR": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Direct albedo of impervious road; spectral-dependent surface "
        "reflectance for urban impervious surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_PERROAD_DIF": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Diffuse albedo of pervious road; spectral-dependent surface "
        "reflectance for urban pervious surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_PERROAD_DIR": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Direct albedo of pervious road; spectral-dependent surface "
        "reflectance for urban pervious surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_ROOF_DIF": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Diffuse albedo of roof; spectral-dependent surface reflectance "
        "for urban roof surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_ROOF_DIR": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Direct albedo of roof; spectral-dependent surface reflectance "
        "for urban roof surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_WALL_DIF": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Diffuse albedo of wall; spectral-dependent surface reflectance "
        "for urban wall surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "ALB_WALL_DIR": {
        "dims": "numrad,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Direct albedo of wall; spectral-dependent surface reflectance "
        "for urban wall surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "CV_IMPROAD": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Volumetric heat capacity of impervious road; thermal mass "
        "property affecting diurnal temperature variations.",
        "required_level": "optional",
        "units": "J/m^3*K",
        "contexts": ["urban", "topounits"],
    },
    "CV_ROOF": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Volumetric heat capacity of roof; thermal mass property "
        "affecting building heat dynamics.",
        "required_level": "optional",
        "units": "J/m^3*K",
        "contexts": ["urban", "topounits"],
    },
    "CV_WALL": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Volumetric heat capacity of wall; thermal mass property "
        "affecting wall temperature dynamics.",
        "required_level": "optional",
        "units": "J/m^3*K",
        "contexts": ["urban", "topounits"],
    },
    "TK_IMPROAD": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Thermal conductivity of impervious road; controls heat diffusion "
        "through urban surface layers.",
        "required_level": "optional",
        "units": "W/m*K",
        "contexts": ["urban", "topounits"],
    },
    "TK_ROOF": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Thermal conductivity of roof; controls heat transfer through "
        "building roof material.",
        "required_level": "optional",
        "units": "W/m*K",
        "contexts": ["urban", "topounits"],
    },
    "TK_WALL": {
        "dims": "nlevurb,numurbl,topounit,lsmlat,lsmlon",
        "doc": "Thermal conductivity of wall; controls heat transfer through "
        "building wall material.",
        "required_level": "optional",
        "units": "W/m*K",
        "contexts": ["urban", "topounits"],
    },
    "EM_IMPROAD": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Emissivity of impervious road; controls longwave radiation "
        "emission from urban surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "EM_PERROAD": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Emissivity of pervious road; controls longwave radiation "
        "emission from pervious urban surfaces.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "EM_ROOF": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Emissivity of roof; controls longwave radiation emission "
        "from building roof.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "EM_WALL": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Emissivity of wall; controls longwave radiation emission "
        "from building walls.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "HT_ROOF": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Height of roof; geometric property defining building structure.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["urban", "topounits"],
    },
    "THICK_ROOF": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Thickness of roof; structural property affecting heat capacity.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["urban", "topounits"],
    },
    "THICK_WALL": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Thickness of wall; structural property affecting heat capacity.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["urban", "topounits"],
    },
    "NLEV_IMPROAD": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Number of impervious road layers; structural discretization for "
        "temperature calculation.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "T_BUILDING_MAX": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Maximum interior building temperature; upper limit constraint "
        "for urban heating/cooling.",
        "required_level": "optional",
        "units": "K",
        "contexts": ["urban", "topounits"],
    },
    "T_BUILDING_MIN": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Minimum interior building temperature; lower limit constraint "
        "for urban heating/cooling.",
        "required_level": "optional",
        "units": "K",
        "contexts": ["urban", "topounits"],
    },
    "CANYON_HWR": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Canyon height to width ratio; urban geometric parameter affecting "
        "radiation and wind.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "WIND_HGT_CANYON": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Height of wind in canyon; reference height for urban wind profile.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["urban", "topounits"],
    },
    "WTLUNIT_ROOF": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Fraction of roof; weight for urban landunit distribution.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "WTROAD_PERV": {
        "dims": "numurbl,topounit,lsmlat,lsmlon",
        "doc": "Fraction of pervious road; weight for pervious surface distribution.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["urban", "topounits"],
    },
    "Ds": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "VIC Ds parameter for the ARNO curve; fractional saturated area "
        "infiltration parameter.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["hydrology", "topounits"],
    },
    "Dsmax": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "VIC Dsmax parameter for the ARNO curve; maximum infiltration rate.",
        "required_level": "optional",
        "units": "mm/day",
        "contexts": ["hydrology", "topounits"],
    },
    "F0": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Maximum gridcell fractional inundated area; controls wetland extent.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["hydrology", "inland_water", "topounits"],
    },
    "FMAX": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Maximum fractional saturated area; upper bound on inundation fraction.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["hydrology", "topounits"],
    },
    "P3": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Coefficient for qflx_surf_lag for finundated; surface runoff delay parameter.",
        "required_level": "optional",
        "units": "s/mm",
        "contexts": ["hydrology", "topounits"],
    },
    "ZWT0": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Decay factor for finundated; controls inundated area decay with water table.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["hydrology", "topounits"],
    },
    "binfl": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "VIC b parameter for the Variable Infiltration Capacity Curve; "
        "infiltration nonlinearity.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["hydrology", "topounits"],
    },
    "LAKEDEPTH": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Lake depth; average water depth for lake landunit.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["hydrology", "inland_water", "topounits"],
    },
    "SOIL_COLOR": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Soil color; categorical index affecting soil albedo parameterization.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["soil_properties", "topounits"],
    },
    "SOIL_ORDER": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Soil order; categorical soil classification for pedogenic properties.",
        "required_level": "optional",
        "units": "unitless",
        "contexts": ["soil_properties", "topounits"],
    },
    "SLP_P10": {
        "dims": "nlevslp,topounit,lsmlat,lsmlon",
        "doc": "Slope at quantiles (minimum and 10 to 100 percentile); "
        "topographic distribution parameter.",
        "required_level": "optional",
        "units": "km km^-1",
        "contexts": ["grid_topography", "topounits"],
    },
    "aveDTB": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Average depth to bedrock of the subgrid; critical for groundwater dynamics.",
        "required_level": "optional",
        "units": "m",
        "contexts": ["soil_properties", "grid_topography", "topounits"],
    },
}

# -----------------------------------------------------------------------------
# Zonal-sampling aggregation policy
# -----------------------------------------------------------------------------
#
# For polygon-based (zonal) sampling, each surface variable needs an explicit
# aggregation strategy.
#
# We keep the registry strict (every var gets an 'agg' entry) while allowing a
# lightweight default via the 'auto' strategy, which resolves at runtime based
# on dtype:
#   - integer/bool -> weighted mode (wmode)
#   - float        -> area-weighted mean (wmean)
#
# A few variables are better derived directly from the Domain cell geometry
# rather than sampled from the source dataset.

SURFACE_VAR_AGG_VALID = {
    # meta
    "auto",  # resolve by dtype at runtime
    "derived",  # derived from Domain (e.g., area/coords)
    # explicit reducers (implemented in dapper.geo.zonal)
    "wmean",
    "wmode",
    "max",
    "min",
    "wmean_threshold",  # weighted mean + threshold for boolean-ish masks
}

# Variables we prefer to derive from the Domain geometry / metadata.
SURFACE_VAR_DERIVED = {
    "LONGXY",  # domain lon
    "LATIXY",  # domain lat
    "AREA",  # domain cell area (m2)
}


def _add_default_agg_policies() -> None:
    """Ensure every SURFACE_VAR_SPECS entry has an 'agg' policy."""
    for var, spec in SURFACE_VAR_SPECS.items():
        if "agg" in spec:
            continue

        if var in SURFACE_VAR_DERIVED:
            spec["agg"] = "derived"
        elif var == "mask" or "MASK" in var:
            # Most *MASK variables in ELM surface data are boolean-ish.
            # If we discover fractional masks later, override explicitly.
            spec["agg"] = "wmean_threshold"
        else:
            spec["agg"] = "auto"

    # Validate
    missing = [v for v, s in SURFACE_VAR_SPECS.items() if "agg" not in s]
    if missing:
        raise ValueError(f"SURFACE_VAR_SPECS missing 'agg' for: {missing[:25]}")

    invalid = [
        (v, SURFACE_VAR_SPECS[v].get("agg"))
        for v in SURFACE_VAR_SPECS
        if SURFACE_VAR_SPECS[v].get("agg") not in SURFACE_VAR_AGG_VALID
    ]
    if invalid:
        bad = ", ".join([f"{v}={a}" for v, a in invalid[:25]])
        raise ValueError(f"Invalid agg policies in SURFACE_VAR_SPECS: {bad}")


_add_default_agg_policies()
