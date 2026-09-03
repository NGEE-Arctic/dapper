This `workspace/` directory is a holding area for specialized or in-progress tools outside of the core functionality of the `dapper` python package. Some of these tools may eventually be implemented within dapper formally, but this provides a space for exploratory tools without the burden of full dapper integration.

# Suggeseted usage
* Tools should be related to data preparation for running ELM.
* The recommended use of workspace is a flat directory structure, with each separate tool in its own directory with a descriptive name, e.g., `elm_soil_data_preparation_for_ngee_sites`.
* Include a README.md file with a brief overview of the tool or analysis.  Longer documentation is great!
* The directory may include just a single script, or it could include a set of related scripts and tools.
* If datasets are used by the tool, provide instructions for where they are located or how to access them.
* Please, no binary data files, but small text data files are ok.
* Long-term migration to the core dapper python package is not expected for any tools in workspace, but ideas for new functionality are welcome.
