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
        "doc": (
            "Longitude (degrees) of the centre of each land grid cell; "
            "used to compare the surface file with the domain and topography files."
        ),
        "required_level": "required",
    },
    "LATIXY": {
        "dims": "lsmlat,lsmlon",
        "doc": "Latitude (degrees) of the centre of each land grid cell.",
        "required_level": "required",
    },
    "LANDMASK": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Binary mask (1 = land, 0 = not-land) defining which grid cells are part of "
            "the land model; read from the surface file or domain file."
        ),
        "required_level": "required",
    },
    "mask": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Binary mask (1 = land, 0 = not-land) defining which grid cells are part of "
            "the land model; read from the surface file or domain file."
        ),
        "required_level": "required",
    },
    "LANDFRAC": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Fraction of each grid cell that is land (0–1); used to weight surface "
            "properties; a missing variable causes a fatal error."
        ),
        "required_level": "required",
    },
    "frac": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Fraction of each grid cell that is land (0–1); used to weight surface "
            "properties; a missing variable causes a fatal error."
        ),
        "required_level": "required",
    },
    "PFTDATA_MASK": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Mask marking land grid cells that contain valid plant functional type (PFT) "
            "data; if missing the code stops with an error."
        ),
        "required_level": "required",
    },
    "xCell": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Unstructured grid coordinates (e.g., MPAS x) when using an irregular mesh; "
            "the code reads them only if they exist."
        ),
        "required_level": "optional",
    },
    "yCell": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Unstructured grid coordinates (e.g., MPAS y) when using an irregular mesh; "
            "the code reads them only if they exist."
        ),
        "required_level": "optional",
    },
    "GLCMASK": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Glacier mask read from a separate glacier file; ensures that glacier cells "
            "are a subset of the land mask."
        ),
        "required_level": "optional",
        "attrs": {
            "requirement": "Optional (only used when a glacier mask file is provided)",
        },
    },
    "TOPO": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Mean elevation (m) of each land grid cell read from the topography file; "
            "required by surfrd_get_topo."
        ),
        "required_level": "required",
    },
    "STDEV_ELEV": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Standard deviation of elevation (m) used by the TOP solar-radiation "
            "parameterization; if STDEV_ELEV is missing the code tries STD_ELEV."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "STD_ELEV": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Standard deviation of elevation (m) used by the TOP solar-radiation "
            "parameterization; alternative name to STDEV_ELEV."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "SKY_VIEW": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Sky-view factor (0–1) describing horizon obstruction by surrounding "
            "terrain; needed by the TOP solar-radiation scheme."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "TERRAIN_CONFIG": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Terrain configuration parameter used in the TOP scheme to account for "
            "terrain-induced shading."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "SINSL_COSAS": {
        "dims": "lsmlat,lsmlon",
        "doc": "sin(slope) * cos(aspect) for each grid cell; used in solar-radiation calculations.",
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "SINSL_SINAS": {
        "dims": "lsmlat,lsmlon",
        "doc": "sin(slope) * sin(aspect) for each grid cell; used with SINSL_COSAS.",
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when TOP solar-radiation is used; optional otherwise"
            ),
        },
    },
    "PCT_WETLAND": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Percentage (0–100) of each grid cell area covered by wetlands within "
            "each topounit."
        ),
        "required_level": "required",
    },
    "PCT_LAKE": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Percentage of each grid cell area that is lake within each topounit; "
            "used for lake landunit weighting."
        ),
        "required_level": "required",
    },
    "PCT_GLACIER": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction of grid cell area covered by simple glacier landunits within "
            "each topounit."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required unless glacier MEC landunits are used; optional when "
                "create_glacier_mec_landunit is true"
            ),
        },
    },
    "PCT_URBAN": {
        "dims": "topounit,numurbl,lsmlat,lsmlon",
        "doc": (
            "Fraction of each grid cell that is urban, for each density class in the "
            "multi-density urban scheme."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when the multi-density urban model is used (nlevurb > 0); "
                "optional otherwise"
            ),
        },
    },
    "URBAN_REGION_ID": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Integer identifier linking urban grid cells to regional urban morphology "
            "datasets."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when PCT_URBAN is used; optional otherwise",
        },
    },
    "PCT_GLC_MEC": {
        "dims": "topounit,nglcec,lsmlat,lsmlon",
        "doc": (
            "Percent of grid cell area assigned to mechanistic glacier classes "
            "(accumulation/ablation, etc.)."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when glacier MEC landunits are created; optional otherwise"
            ),
        },
    },
    "TOPO_GLC_MEC": {
        "dims": "topounit,nglcec,lsmlat,lsmlon",
        "doc": "Elevation (m) of each mechanistic glacier class.",
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when glacier MEC landunits are used; optional otherwise"
            ),
        },
    },
    "PCT_NATVEG": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Percent of grid cell area covered by natural vegetation (non-crop "
            "vegetated landunit)."
        ),
        "required_level": "required",
    },
    "PCT_CROP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Percent of grid cell area covered by cropland.",
        "required_level": "required",
    },
    "PCT_HCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction of natural vegetation landunit that is high-centered polygons "
            "(polygonal tundra)."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when use_polygonal_tundra is true; optional otherwise"
            ),
        },
    },
    "PCT_FCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of natural vegetation that is flat-centered polygons.",
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when use_polygonal_tundra is true; optional otherwise"
            ),
        },
    },
    "PCT_LCP": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Fraction of natural vegetation that is low-centered polygons.",
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when use_polygonal_tundra is true; optional otherwise"
            ),
        },
    },
    "PCT_CFT": {
        "dims": "topounit,cft,lsmlat,lsmlon",
        "doc": (
            "Fraction of vegetated area allocated to each crop functional type; "
            "code aborts if missing when cft dimension exists."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when the surface file includes crop functional types "
                "(cft_size > 0)"
            ),
        },
    },
    "NFERT": {
        "dims": "topounit,cft,lsmlat,lsmlon",
        "doc": (
            "Nitrogen fertilizer application for each crop functional type; if "
            "absent, values default to zero."
        ),
        "required_level": "optional",
        "attrs": {
            "requirement": "Optional (values default to zero when absent)",
        },
    },
    "PFERT": {
        "dims": "topounit,cft,lsmlat,lsmlon",
        "doc": (
            "Phosphorus fertilizer application for each crop functional type; "
            "treated like NFERT."
        ),
        "required_level": "optional",
    },
    "PCT_NAT_PFT": {
        "dims": "topounit,natpft,lsmlat,lsmlon",
        "doc": (
            "Fraction of vegetated area allocated to each natural plant functional "
            "type; code aborts if missing."
        ),
        "required_level": "required",
    },
    "FIRRIG": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction (0–1) of cropland that is irrigated; read only when irrigation "
            "data are enabled (firrig_data)."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise",
        },
    },
    "FSURF": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction of irrigation water applied via surface irrigation; required "
            "with FIRRIG when irrigation data are used."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise",
        },
    },
    "FGRD": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction of irrigation water applied via ground/sprinkler/drip; "
            "complements FSURF."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": "Required when firrig_data is true; optional otherwise",
        },
    },
    "MaxTopounitElv": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Maximum elevation (m) among topounits for each grid cell; read only if "
            "present."
        ),
        "required_level": "optional",
        "attrs": {
            "requirement": (
                "Optional (not required, but improves topounit characterization)"
            ),
        },
    },
    "topoPerGrid": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Number of topounits in each grid cell; if absent, the number of "
            "topounits is set to 1."
        ),
        "required_level": "optional",
    },
    "TopounitFracArea": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Fraction of the grid cell’s area represented by each topounit; read "
            "only if present."
        ),
        "required_level": "optional",
    },
    "TopounitAveElv": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Average elevation (m) of each topounit; optional if available.",
        "required_level": "optional",
    },
    "TopounitElv": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Average elevation (m) of each topounit; optional if available.",
        "required_level": "optional",
    },
    "TopounitSlope": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": (
            "Slope (degrees) of each topounit; not currently used but may improve "
            "topographic representation."
        ),
        "required_level": "optional",
    },
    "TopounitAspect": {
        "dims": "topounit,lsmlat,lsmlon",
        "doc": "Aspect (azimuth) of each topounit; optional and unused by default.",
        "required_level": "optional",
    },
    "TOPO2": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Second topography field used in the ELM topounit framework; read only "
            "if present."
        ),
        "required_level": "optional",
    },
    # Below here I am not sure about...
    "AREA": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Area (m2) of each land grid cell; used to weight land-surface "
            "properties and to compare with domain/topography."
        ),
        "required_level": "required",
    },
    "LANDFRAC_PFT": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Fraction (0–1) of the global gridcell that is land in the PFT landunit; "
            "used to weight PFT tiles relative to the gridcell."
        ),
        "required_level": "required",
    },
    "SLOPE": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Mean surface slope (degrees) for each land grid cell; used in "
            "topographic and hydrologic parameterizations."
        ),
        "required_level": "required",
    },
    "PCT_SAND": {
        "dims": "nlevsoi,lsmlat,lsmlon",
        "doc": (
            "Soil sand percentage by mass (0–100) in each soil layer; controls "
            "hydraulic and thermal properties."
        ),
        "required_level": "required",
    },
    "PCT_CLAY": {
        "dims": "nlevsoi,lsmlat,lsmlon",
        "doc": (
            "Soil clay percentage by mass (0–100) in each soil layer; controls "
            "hydraulic and thermal properties."
        ),
        "required_level": "required",
    },
    "ORGANIC": {
        "dims": "nlevsoi,lsmlat,lsmlon",
        "doc": (
            "Soil organic matter or organic carbon per layer; used in biogeochemical "
            "and thermal calculations."
        ),
        "required_level": "optional",
    },
    "PCT_GRVL": {
        "dims": "nlevsoi,lsmlat,lsmlon",
        "doc": (
            "Percent gravel content (0–100) in each soil layer; affects soil water "
            "storage and hydraulic conductivity."
        ),
        "required_level": "optional",
    },
    "GLC_MEC": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Integer glacier elevation-class index for each grid cell when using "
            "mechanistic glacier (MEC) landunits."
        ),
        "required_level": "conditional",
        "attrs": {
            "requirement": (
                "Required when glacier MEC landunits are used; optional otherwise"
            ),
        },
    },
    "MONTHLY_LAI": {
        "dims": "time,lsmlat,lsmlon",
        "doc": (
            "Leaf area index (LAI; m2 leaf per m2 ground) monthly climatology; "
            "time is typically 12 months."
        ),
        "required_level": "optional",
    },
    "MONTHLY_SAI": {
        "dims": "time,lsmlat,lsmlon",
        "doc": (
            "Stem area index (SAI) monthly climatology; time is typically 12 months."
        ),
        "required_level": "optional",
    },
    "MONTHLY_HEIGHT_TOP": {
        "dims": "time,lsmlat,lsmlon",
        "doc": (
            "Monthly climatology of canopy top height (m) for vegetated landunits."
        ),
        "required_level": "optional",
    },
    "MONTHLY_HEIGHT_BOT": {
        "dims": "time,lsmlat,lsmlon",
        "doc": (
            "Monthly climatology of canopy bottom height (m) for vegetated landunits."
        ),
        "required_level": "optional",
    },
    "APATITE_P": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Soil phosphorus pool in apatite (primary mineral) form; used by "
            "phosphorus biogeochemistry when enabled."
        ),
        "required_level": "optional",
    },
    "LABILE_P": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Soil labile (readily available) phosphorus pool; used by P-cycle "
            "parameterizations."
        ),
        "required_level": "optional",
    },
    "OCCLUDED_P": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Soil occluded phosphorus pool (sorbed or otherwise inaccessible); "
            "part of multi-pool P parameterization."
        ),
        "required_level": "optional",
    },
    "SECONDARY_P": {
        "dims": "lsmlat,lsmlon",
        "doc": (
            "Soil secondary mineral phosphorus pool; intermediate in reactivity "
            "between apatite and labile pools."
        ),
        "required_level": "optional",
    },
}
