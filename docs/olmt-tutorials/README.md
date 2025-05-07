# olmt-tutorials

This repo will eventually contain a set of notebooks and scripts that describe how to run ELM using OLMT on the EES servers at LANL.

If you do not have OLMT yet, please obtain it from gitlab:

`git clone -b rfiorella/era5 git@github.com:rfiorella/OLMT`

If you have a copy from a different git server, you can update the origin remote with:
`git remote set-url origin git@github.com:rfiorella/OLMT`

If you're not sure what the remote location is, try:
`git remote -v` from the OLMT folder.

### Tutorial list (to be updated):
01. Creating surface files for a site from the global gridded datasets


### Options to turn on NGEE Arctic IM capability

#### Cold initialization (closest to "IM0")
By default, if no initialization files are provided, ELM soil columns initialize at 274K and with a soil volumetric liquid water content of 0.15 mm3/mm3. Adding the `--use_arctic_init` flag to the OLMT call (or adding `use_arctic_init = .true.` to the ELM namelist) will cause this column to initialize at a temperature of 250 + 40*cos(lat) and with a saturated liquid water content (e.g., Arctic cells are allowed to freeze at saturation).

#### Polygonal tundra (IM1)
Polygonal tundra can be enabled using the `--use_polygonal_tundra` OLMT flag (or adding `use_polygonal_tundra = .true.` to the ELM namelist). It requires a custom surface file with three additional variables specified for the grid cells in the domain: PCT_HCP (% high-centered polygon), PCT_LCP (% low-centered polygon), and PCT_FCP (% flat-centered polygon). The "vegetated or bare soil" landunit type (which is `istsoil` in ELM code) is then split into four separate columns: a) non-polygonal, and then three polygon types classified as an initial condition (all trend toward high centered polygons with melt): b) low-centered polygon, c) flat-centered polygon, and d) high-centered polygon.

#### Hillslope hydrology across topounits (IM2)
A within grid cell representation of hillslope hydrology can be enabled using `--use_IM2_hillslope_hydrology` in OLMT (or setting `use_IM2_hillslope_hydrology = .true.` in ELM namelist). 

#### Vegetation parameters (IM3)
IM3 allows four vegetation parameters to be modified on the vegetation parameters file. Two of these were already vegetation parameters in ELM, but their values were hard-coded based on PFT: a) stocking, an estimate of the number of plant individuals/stems per hectare, and b) taper, a ratio of deadstem C height-to-width. The other two are new parameters: c) bendresist, a parameter on (0,1], that is meant to represent all of the physiological and physical processes that cause vegetation to have a lower canopy height with snow loading (e.g., branch bending under load, cryocampsis, etc.), and d) vegshape, a parameter that should be set to either 1 or 2, and is meant to deal with differences in dLAI/dz across different vegetation shapes.
