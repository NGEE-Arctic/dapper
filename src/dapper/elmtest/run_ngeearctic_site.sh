#!/bin/bash
# =======================================================================================
# Setup and run an ELM OLMT simulation for an NGEE Arctic site
#
#
# Phase 3 availible site names: kougarok, teller, council, beo
# Phase 4 additional sites: Toolik_Lake, Abisko, Trail_Valley_Creek, Samoylov_Island, Bayelva
# =======================================================================================
set -euo pipefail
# Print out options and help statements
Help()
{
    echo "Usage: $(basename $0) [-h|--help] [additional options]"
    echo "Options:"
    echo "  -h, --help                Display this help message"
    echo "  --site_name               Which NGEE site would you like to run? Available options:"
    echo "                            Phase 3 sites: kougarok, teller, council, beo"
    echo "                            Phase 4 sites: abisko, trail_valley_creek, toolik_lake,"
    echo "                              samoylov_island, bayelva"
    echo "  --site_group              OLMT allows different site groups to be specified - for the "
    echo "                              workshop this will always be NGEE-Arctic"
    echo "  --case_prefix             What should be appended to the beginning of the casename to distinguish"
    echo "                              between two runs at the same site?"
    echo "  -ms, --met_source         Select which meteorological forcing you would like to use. ERA5 or GSWP3 (Default: ERA5)"
    echo "  --use_arctic_init         Use modified startup condition for Arctic conditions. (ELM default is soil moisture of 0.15 m3/m3"
    echo "                            and 274K. Modified condition is at liquid saturation and at 250+40*cos(lat) K)"
    echo "  --use_IM2_hillslope_hydrology  Turn on transfer of water from high elevation topounits to low elevation."
    echo "                                 (Requires surffile with topounits > 1 to see an impact)"
    echo "  --use_polygonal_tundra    Turn on polygonal tundra parameterizations affecting runoff, depression storage, and inundation fraction"
    echo "                            (Requires surface file with polygonal tundra type area fractions specified)"
    echo "  -ady, --ad_spinup_yrs     How many years of initial spinup using accelerated decomposition rates"
    echo "                              should be used? (Default: 200)"
    echo "  -fsy, --final_spinup_yrs  How many years should the second stage of spinup run (with normal"
    echo "                              decomposition rates)? (Default: 600)"
    echo "  -try, --transient_yrs     How many years should the transient stage of the sumulation run?"
    echo "                              (Default: -1, which corresponds to 1850-2014 for GSWP3 met data and"
    echo "                              1850-2024 for ERA5)"
    echo "  -tsp, --timestep          What timestep should the model use? (Default: 1 hour, units of this number"
    echo "                               should be hours)"
    echo "  --met_source              What met source should be used? (Default: ERA5, can also use GSWP3)"
    echo "  --add_temperature         Add a constant temperature to the meteorology time series, in K"
    echo "  --startdate_add_temperature   When should the temperature perturbation be initiated?"
    echo "                                (It will continue to the end of the run). YYYYMMDD format"
    echo "  --add_co2                 Add a constant value of CO2 to the CO2 forcing data stream (in ppm)"
    echo "                            NOTE: in this mode, CO2 has no impact on temperature since temperature is provided as an input time series"
    echo "                            This switch only affects plant physiology"
    echo "  --startdate_add_co2       When should the CO2 perturbation be initiated? (Will continue to end of run), YYYYMMDD format"
    echo "  --scale_rain              Scaling factor to apply to rainfall in atmospheric forcing"
    echo "  --startdate_scale_rain    When should the rainfall scaling begin? (It will continue to the end of the run), YYYYMMDD format"
    echo "  --scale_snow              Scaling factor to apply to snowfall in atmosopheric forcing"
    echo "  --startdate_scale_snow    When should the snowfall scaling begin? (It will continue to the end of the transient run), YYYYMMDD format"
    echo "  --scale_ndep              Scaling factor to apply to nitrogen deposition in the atmospheric forcing"
    echo "  --startdate_scale_ndep    When should the nitrogen deposition scaling begin? (It will continue to the end of the transient run)"
    echo "                            YYYYMMDD format"
    echo "  --scale_pdep              Scaling factor to apply to phosphorus deposition in the atmospheric forcing"
    echo "  --startdate_scale_pdep    When should the phosphorus deposition scaling begin? (It will continue to the end of the transient run)"
    echo "                            YYYYMMDD format"
    echo "  --mod_parm_file           Use a modified PFT parameter file (Note, requires full path)"
    exit 0
}

cwd=$(pwd)
cd /home/modex_user/tools/olmt

# =======================================================================================
# Get the input options from the command line
for i in "$@"
do
case $i in
    -h|--help)
    Help
    shift
    ;;
    --site_name=*)
    site_name="${i#*=}"
    site_name="${site_name,,}"
    shift
    ;;
    --site_group=*)
    site_group="${i#*=}"
    shift 
    ;;
    --case_prefix=*)
    case_prefix="${i#*=}"
    shift
    ;;
    -ady=*|--ad_spinup_yrs=*)
    ad_spinup_years="${i#*=}"
    shift
    ;;
    -fsy=*|--final_spinup_yrs=*)
    final_spinup_years="${i#*=}"
    shift
    ;;
    -try=*|--transient_yrs=*)
    transient_years="${i#*=}"
    shift
    ;;
    -tsp=*|--timestep=*)
    timestep="${i#*=}"
    shift
    ;;
    --add_temperature=*)
    add_temperature="${i#*=}"
    shift
    ;;
    --startdate_add_temperature=*)
    startdate_add_temperature="${i#*=}"
    shift
    ;;    
    --add_co2=*)
    add_co2="${i#*=}"
    shift
    ;;
    --startdate_add_co2=*)
    startdate_add_co2="${i#*=}"
    shift
    ;;    
    -sclr=*|--scale_rain=*)
    scale_rain="${i#*=}"
    shift
    ;;
    -sdsclr=*|--startdate_scale_rain=*)
    startdate_scale_rain="${i#*=}"
    shift
    ;;    
    -scls=*|--scale_snow=*)
    scale_snow="${i#*=}"
    shift
    ;;
    -sdscls=*|--startdate_scale_snow=*)
    startdate_scale_snow="${i#*=}"
    shift
    ;;
    -scln=*|--scale_ndep=*)
    scale_ndep="${i#*=}"
    shift
    ;;
    -sdscln=*|--startdate_scale_ndep=*)
    startdate_scale_ndep="${i#*=}"
    shift
    ;;
    -sclp=*|--scale_pdep=*)
    scale_pdep="${i#*=}"
    shift
    ;;
    -sdsclp=*|--startdate_scale_pdep=*)
    startdate_scale_pdep="${i#*=}"
    shift
    ;;
    -ms=*|--met_source=*)
    met_source="${i#*=}"
    shift
    ;;
    --use_arctic_init)
    use_arctic_init=True
    shift
    ;;   
    --use_IM2_hillslope_hydrology)
    use_IM2_hillslope_hydrology=True
    shift
    ;;    
    --use_polygonal_tundra)
    use_polygonal_tundra=True
    shift
    ;;
    *)
        # unknown option
    ;;
esac
done
# =======================================================================================

# =======================================================================================

# Set defaults and print the selected options back to the screen before running
site_name="${site_name:-kougarok}"
site_group="${site_group:-NGEEArctic}"
case_prefix="${case_prefix:-NGEE-Arctic}"
ad_spinup_years="${ad_spinup_years:-200}"
final_spinup_years="${final_spinup_years:-600}"
transient_years="${transient_years:--1}"
met_source="${met_source:-era5}"
use_arctic_init="${use_arctic_init:-False}"
use_IM2_hillslope_hydrology="${use_IM2_hillslope_hydrology:-False}"
use_polygonal_tundra="${use_polygonal_tundra:-False}"
options="${options:-}"
# -1 is the default
timestep="${timestep:-1}"
# temp and co2 additions:
add_temperature="${add_temperature:-0.0}"
add_co2="${add_co2:-0.0}"
startdate_add_temperature="${startdate_add_temperature:-99991231}"
startdate_add_co2="${startdate_add_co2:-99991231}"
# precipitation scaling defaults
scale_rain="${scale_rain:-1.0}"
scale_snow="${scale_snow:-1.0}"
startdate_scale_rain="${startdate_scale_rain:-99991231}"
startdate_scale_snow="${startdate_scale_snow:-99991231}"
# N/P dep scaling
scale_ndep="${scale_ndep:-1.0}"
startdate_scale_ndep="${startdate_scale_ndep:-99991231}"
scale_pdep="${scale_pdep:-1.0}"
startdate_scale_pdep="${startdate_scale_pdep:-99991231}"

#enforce met naming in prefix
case_prefix=${case_prefix}_${met_source}

# print back selected or set options to the user
echo " "
echo " "
echo "*************************** OLMT run options ***************************"
echo "Site Name = ${site_name}"
echo "Site Group = ${site_group}"
echo "Case Prefix = ${case_prefix}"
echo "Number of AD Spinup Simulation Years = ${ad_spinup_years}"
echo "Number of Final Spinup Simulation Years = ${final_spinup_years}"
echo "Number of Transient Simulation Years = ${transient_years}"
echo "Model Timestep = ${timestep} hours"
if [ ${add_temperature} != 0.0 ]; then
  echo "Adding ${add_temperature} degreesC to forcing temperature starting on ${startdate_add_temperature}"
  options="--add_temperature ${add_temperature} --startdate_add_temperature ${startdate_add_temperature}"
fi
if [ ${add_co2} != 0.0 ]; then
  echo "Adding ${add_co2} ppm to forcing CO2 starting on ${startdate_add_co2}"
  options="$options --add_co2 ${add_co2} --startdate_add_co2 ${startdate_add_co2}"
fi
if [ ${scale_rain} != 1.0 ]; then
  echo "Forcing rainfall scaled by factor of ${scale_rain} starting on ${startdate_scale_rain}"
  options="$options --scale_rain ${scale_rain} --startdate_scale_rain ${startdate_scale_rain}"
fi
if [ ${scale_snow} != 1.0 ]; then
  echo "Forcing snowfall scaled by factor of ${scale_snow} starting on ${startdate_scale_snow}"
  options="$options  --scale_snow ${scale_snow} --startdate_scale_snow ${startdate_scale_snow}"
fi
if [ ${scale_ndep} != 1.0 ]; then
  echo "N deposition scaled by factor of ${scale_ndep} starting on ${startdate_scale_ndep}"
  options="$options --scale_ndep ${scale_ndep} --startdate_scale_ndep ${startdate_scale_ndep}"
fi 
if [ ${scale_pdep} != 1.0 ]; then
  echo "P deposition scaled by factor of ${scale_pdep} starting on ${startdate_scale_pdep}"
  options="$options --scale_pdep ${scale_pdep} --startdate_scale_pdep ${startdate_scale_pdep}"
fi
if [ ${transient_years} != -1 ]; then
  sim_years="--nyears_ad_spinup ${ad_spinup_years} --nyears_final_spinup ${final_spinup_years} \
  --nyears_transient ${transient_years}"
else
  sim_years="--nyears_ad_spinup ${ad_spinup_years} --nyears_final_spinup ${final_spinup_years}"
fi
if [ "${use_arctic_init}" = True ]; then
  echo "Using wetter, colder initialization conditions for Arctic runs"
  options="$options --use_arctic_init"
fi
if [ "${use_IM2_hillslope_hydrology}" = True ]; then
  echo "Turning on hillslope hydrology from high-elevation topounits to low-elevation topounits"
  options="$options --use_IM2_hillslope_hydrology"
fi
if [ "${use_polygonal_tundra}" = True ]; then
  echo "Turning on polygonal tundra parameterizations"
  options="$options --use_polygonal_tundra"
fi 
echo " "
# =======================================================================================

# =======================================================================================
# specify met dir prefix
# Set site codes for OLMT
# EACH OF THESE SITES ALSO NEEDS THE SURFFILE, LU FILE, DOMAIN FILE SPECIFIED>
#add hook for different source mods
# however-- 
  # - gswp3 crashed when trying to use srcmods_gswp3v1 source mode
  # - yet gswp3 completed when using the srcmods_era5cb/
  # - so use srcmods_era5cb for both  
src_mod_path="/home/modex_user/tools/olmt/srcmods_modexws/"
if [ ${met_source} = era5 ]; then
  met_root="/mnt/inputdata/atm/datm7/era5"
elif [ ${met_source} = gswp3 ]; then
  met_root="/mnt/inputdata/atm/datm7/gswp3"
fi

if [ ${site_name} = beo ]; then
  site_code="AK-BEOG"
  surf_file="surfdata_1x1pt_beo-GRID_simyr1850_c360x720_c171002.nc"
  landuse_file="landuse.timeseries_1x1pt_beo-GRID_simyr1850-2015_c180423.nc"
  domain_file="domain.lnd.1x1pt_beo-GRID_navy.nc"
  met_path="${met_root}/utq"
elif [ ${site_name} = council ]; then
  site_code="AK-CLG"
  surf_file="surfdata_1x1pt_council-GRID_simyr1850_c360x720_c171002.nc"
  landuse_file="landuse.timeseries_1x1pt_council-GRID_simyr1850-2015_c180423.nc"
  domain_file="domain.lnd.1x1pt_council-GRID_navy.nc"
  met_path="${met_root}/cnl"
elif [ ${site_name} = kougarok ]; then
  site_code="AK-K64G"
  surf_file="surfdata_1x1pt_kougarok-GRID_simyr1850_c360x720_c171002.nc"
  landuse_file="landuse.timeseries_1x1pt_kougarok-GRID_simyr1850-2015_c180423.nc"
  domain_file="domain.lnd.1x1pt_kougarok-GRID_navy.nc"  
  met_path="${met_root}/kg"
elif [ ${site_name} = teller ]; then
  site_code="AK-TLG"
  surf_file="surfdata_1x1pt_teller-GRID_simyr1850_c360x720_c171002.nc"
  landuse_file="landuse.timeseries_1x1pt_teller-GRID_simyr1850-2015_c180423.nc"
  domain_file="domain.lnd.1x1pt_teller-GRID_navy.nc"
  met_path="${met_root}/tl"
elif [ ${site_name} = toolik_lake ]; then
  site_code="AK-Tlk"
  surf_file="surfdata_1x1pt_ToolikLake-GRID_simyr1850_c360x720_c250306.nc"
  landuse_file="landuse.timeseries_1x1pt_ToolikLake-GRID_simyr1850-2015_c250306.nc"
  domain_file="domain.lnd.1x1pt_ToolikLake-GRID.nc"
  met_path="${met_root}/tfs"
elif [ ${site_name} = trail_valley_creek ]; then
  site_code="CA-TVC"
  surf_file="surfdata_1x1pt_TrailValleyCreek-GRID_simyr1850_c360x720_c250306.nc"
  landuse_file="landuse.timeseries_1x1pt_TrailValleyCreek-GRID_simyr1850-2015_c250306.nc"
  domain_file="domain.lnd.1x1pt_TrailValleyCreek-GRID.nc"
  met_path="${met_root}/tvc"
elif [ ${site_name} = abisko ]; then
  site_code="SE-Abi"
  surf_file="surfdata_1x1pt_Abisko-GRID_simyr1850_c360x720_c250306.nc"
  landuse_file="landuse.timeseries_1x1pt_Abisko-GRID_simyr1850-2015_c250306.nc"
  domain_file="domain.lnd.1x1pt_Abisko-GRID.nc"
  met_path="${met_root}/abs"
elif [ ${site_name} = bayelva ]; then
  site_code="NO-SJB"
  surf_file="surfdata_1x1pt_SJ-BlvBayelva-GRID_simyr1850_c360x720_c250306.nc"
  landuse_file="landuse.timeseries_1x1pt_SJ-BlvBayelva-GRID_simyr1850-2015_c250306.nc"
  domain_file="domain.lnd.1x1pt_SJ-BlvBayelva-GRID.nc"
  met_path="${met_root}/bs"
elif [ ${site_name} = samoylov_island ]; then
  site_code="RU-Sam"
  surf_file="surfdata_1x1pt_SamoylovIsland-GRID_simyr1850_c360x720_c250306.nc"
  landuse_file="landuse.timeseries_1x1pt_SamoylovIsland-GRID_simyr1850-2015_c250306.nc"
  domain_file="domain.lnd.1x1pt_SamoylovIsland-GRID.nc"
  met_path="${met_root}/si"
elif [ ${site_name} = upper_kuparuk ]; then
  site_code="AK-UPK"
  surf_file="surfdata_1x1pt_UpperKuparuk-GRID_simyr1850_c360x720_c250609.nc"
  landuse_file="landuse.timeseries_1x1pt_UpperKuparuk-GRID_simyr1850-2015_c250609.nc"
  domain_file="domain.lnd.1x1pt_UpperKuparuk-GRID.nc"
  if [ ${met_source} = era5 ]; then
    met_path="${met_root}/UpK_wshed"
  elif [ ${met_source} = gswp3 ]; then
    met_path="${met_root}/tfs" # use same site data as toolik
  fi   
elif [ ${site_name} = imnaviat_creek ]; then
  site_code="AK-IMC"
  surf_file="surfdata_1x1pt_ImnaviatCreek-GRID_simyr1850_c360x720_c250609.nc"
  landuse_file="landuse.timeseries_1x1pt_ImnaviatCreek-GRID_simyr1850-2015_c250609.nc"
  domain_file="domain.lnd.1x1pt_ImnaviatCreek-GRID.nc"
  if [ ${met_source} = era5 ]; then
    met_path="${met_root}/ImC_wshed"
  elif [ ${met_source} = gswp3 ]; then
    met_path="${met_root}/tfs" # use same site data as toolik
  fi  
else 
  echo " "
  echo "**** EXECUTION HALTED ****"
  echo "Please select a Site Name from:"
  echo "ALASKA: beo, council, kougarok, teller, toolik_lake, imnaviat_creek, upper_kuparuk"
  echo "CANADA: trail_valley_creek"
  echo "SWEDEN: abisko"
  echo "NORWAY/SVALBARD: bayelva" 
  echo "RUSSIA: samoylov_island"
  exit 0
fi
echo "OLMT Site Code: ${site_code}"
# =======================================================================================

# =======================================================================================
# pause to show options before continuing
sleep 10
echo " "
echo " "
# =======================================================================================

# =======================================================================================
# create the OLMT run command
echo " "
echo "**** User OLMT run command: "
runcmd="python3 ./site_fullrun.py \
      --site ${site_code} --sitegroup ${site_group} --caseidprefix ${case_prefix} \
      ${sim_years} --tstep ${timestep} --machine docker \
      --compiler gnu --mpilib openmpi \
      --cpl_bypass --${met_source} \
      --model_root /E3SM \
      --caseroot /mnt/output/cime_case_dirs \
      --ccsm_input /mnt/inputdata \
      --runroot /mnt/output/cime_run_dirs \
      --spinup_vars \
      --nopointdata \
      --metdir ${met_path} \
      --domainfile /mnt/inputdata/share/domains/domain.clm/${domain_file} \
      --surffile /mnt/inputdata/lnd/clm2/surfdata_map/${surf_file} \
      --landusefile /mnt/inputdata/lnd/clm2/surfdata_map/${landuse_file} \
      --srcmods_loc ${src_mod_path} /
      ${options} \
      & sleep 10"
echo ${runcmd}
echo " "
echo " "

echo "**** Running OLMT: "

# explicitly make the two output directories, 
# else these will be placed in /E3SM/root
mkdir -p /mnt/output/cime_case_dirs 
mkdir -p /mnt/output/cime_run_dirs

# added met source, so known
if /opt/conda/bin/python -u -X dev ./site_fullrun.py \
      --site ${site_code} --sitegroup ${site_group} --caseidprefix ${case_prefix} \
      ${sim_years} --tstep ${timestep} --machine docker \
      --compiler gnu --mpilib openmpi \
      --cpl_bypass --${met_source} \
      --model_root ~/E3SM \
      --caseroot /mnt/output/cime_case_dirs \
      --ccsm_input /mnt/inputdata \
      --runroot /mnt/output/cime_run_dirs \
      --spinup_vars \
      --nopointdata \
      --metdir ${met_path} \
      --domainfile /mnt/inputdata/share/domains/domain.clm/${domain_file} \
      --surffile /mnt/inputdata/lnd/clm2/surfdata_map/${surf_file} \
      --landusefile /mnt/inputdata/lnd/clm2/surfdata_map/${landuse_file} \
      --srcmods_loc ${src_mod_path} \
      ${options} \
      & sleep 10

then
  wait

  echo "DONE docker ELM runs !"

else
  exit 1
fi
# =======================================================================================
# =======================================================================================
#### Postprocess
### Collapse transient simulation output into a single netCDF file
echo " "
echo " "
echo " "
echo "**** Postprocessing ELM output in: "
echo "/mnt/output/cime_run_dirs/${case_prefix}_${site_code}_ICB20TRCNPRDCTCBC/run"
echo " "
echo " "
cd /mnt/output/cime_run_dirs/${case_prefix}_${site_code}_ICB20TRCNPRDCTCBC/run
echo "**** Concatenating netCDF output - Hang tight this can take awhile ****"
ncrcat --ovr *.h0.*.nc ELM_output.nc
chmod 666 ELM_output.nc
echo "**** Concatenating netCDF output: DONE ****"
sleep 2
# =======================================================================================
