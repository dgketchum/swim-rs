Instructions to run SWIM-RS at Flux Stations

Note: These instructions are specific to the flux data set, which is dispersed. For highly clustered data, such as fields
	in a single watershed, replace the use of e.g., flux_tower_ndvi() and landsat_time_series_station() with e.g., clustered_field_ndvi()
	and landsat_time_series_multipolygon(). The varying dispersion of objects is the reason for three seperate Earth Engine data 
	extraction approaches and three different 'landsat_time_series' functions to get the data into a standard table format.

1. Get flux station data from Volk et al. (https://www.sciencedirect.com/science/article/pii/S2352340923003931).

2. Use a GIS to project the flux tower points to EPSG:5071 and add a unique identifier under a new attribute, 'FID'. 
	Use swim-rs/data_extraction and prep scripts to extract data at the flux tower shapefile points:
	
	- Run ee_props/get_cdl() and ee_props/get_irrmapper() to get CDL and IrrMapper data for each buffered flux zone.
	
	- Use etf_export/flux_tower_etf() and ndvi_export/flux_tower_ndvi() to extract SSEBop ETf and NDVI at the flux towers.
	
	- Using prep/landsat_sensing.py run landsat_time_series_station(), join_remote_sensing(), and detect_cuttings() to join the 
	  remote sensing-based information. This needs to be run for irrigated and non-irrigated plot fractions, 
	  specifying 'irr' and 'inv_irr' respectively.
	  
	- Run prep/field_properties.py to clean and join plot properties and write them for use in the model.
	
	- Run prep/field_timeseries.py: This runs gridmet_corrected(), which finds the gridmet cells closest to the plots, then
	 downloads a gridmet time series at each of those gridmet cells, and computes the corrected gridmet ETo/ETr based
	 on OpenET correction surfaces.
	 
	 Run the function join_gridmet_remote_sensing_daily(), which is used to join the landsat and gridmet timeseries into a 
	 single timeseries input for each plot. These are input data for the model.
	 
	 
3. Test the inputs by running the uncalibrated model with run/run_flux_etd.py to run the SWB model (ET-Demands or ETRM).
	This should display the accuracy of the uncalibrated model and SSEBop against the selected flux station.
	Use 'params' as the kwargs argument in the main call to run_fields() to assign parameters.
	
4. Calibrate TAW, MAD, and the NDVI-ETf coefficients using PEST++:

	- Modify flux_swim.toml configuration file to point to the data you already built, the time period to be run,
	the unique field intentifier (should be 'FID'), etc. 
	
	- Preprocess the 'observed' data to be used for calbration using examples/flux/preproc.py, the oberved data
	 is simply the corrected OpenET ETr times SSEBop ETf at this point (i.e., etr * etf = eta). Consider using 
	 eto and/or another RSET algorithm in the future. This writes obs_eta.np.
	
	- Modify custom_forward_run.py to correct paths. This is the python script PEST++ will be calling each iteration.
	
	- Clone https://github.com/dgketchum/pyemu and checkout the 'etd' branch. Install it into your python environment
	with e.g., 'pip install -e /home/projects/pyemu'. This has a small number of hacks to modify how PyEMU writes
	the .pst file that I haven't found an elegant solution to yet.
	
	- Use the calbrate/build_etd_pp.py script to build a PEST++ configuration and subdirectory. This will be a subdirectory of
	the flux project directory, i.e., swim-rs/examples/flux/pest. The function build_pest.py will erase the existing pest
	directory if there is one! It will also copy everything from examples/flux into the pest directory. This function builds
	the flux.pst file, which is the only argument needed at this time to run PEST++ on the problem. Note that during the processing
	of the ETf data, we wrote an e.g., etf_inv_irr_ct.csv table that simply marked the image capture dates. In build_pest(), the 
	observations are given weight 1.0 on these dates, and weight 0.0 on non-capture dates. The idea here is to only evaluate
	the objective function on capture dates to give the model the freedom to behave like a soil water balance model on 
	in-between dates.

    - In the * parameter data section, change 'log' to 'none' and 'factor' to 'relative' in the line for 'ndvi_alpha'.
        The use of log and negative number parameters will cause nan. There is a way to offset the number, but I haven't
        implemented an offset that is subsequently read by the model code. #TODO

	- Change 'obs_eta.np' to 'pred_eta.np' in the final line of the flux.pst control file. See note below.
	
	- Copy custom_forward_run.py into the pest directory.

	- Edit run/run_flux_etd.py to NOT use the params dict in the call to run_fields(), usage of the kwargs argument
	    will override the model's access to the parameters modified by PEST in model/etd/obs_field_cycle.py.
	
	- Install PEST++ using instructions from their github page https://github.com/usgs/pestpp.
	
	- Run 'pestpp-glm' to calibrate parameters. Read the PEST++ documentation to learn other options, configurations, etc.
	    See https://github.com/usgs/pestpp/blob/master/documentation/pestpp_users_manual.md
	
	If things are set up correctly, the model RMSE reported by SWIM should slowly go down!
	
Notes on the current PEST++ interface in SWIM-RS:
	
	PEST is designed to operate independently of the model: it only cares about accessing model output and observations,
	and comparing them with respect to its perturbations of the input parameters. SWIM tries to follow this pattern by
	interacting very little with PEST during model iterations. It does this in only two ways: 1. It reads the parameter
	proposal made by PEST and written to e.g., examples/flux/pest/mult. Each file is a parameter; and 2. It provides
	PEST with the output of every model iteration so PEST can assess how the calibration is going. 
	
	The setup SWIM has now is a little confusing because during preparation, obs_eta.np is the observation (SSEBop data),
	and is read during the flux.pst creation in the function build_pest.py. This data is written in as the observations
	in the flux.pst file, as one would expect. However, for some reason I haven't investigated,
	pyEMU writes 'obs_eta.np' into the final line of the .pst file as the file that PEST needs to read to see
	the model predictions. So this must be changed to 'pred_eta.np', which is the prediction written by
	run_flux_etd.py at the end of each model run. If this is not fixed, the model will compare model observations to
	model observations, see no difference, declare the model insensitive to the parameters, and cease execution.
