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
	
4. Calibrate TAW, MAD, and the NDVI-ETf coefficients using PEST++:

	- Modify flux_swim.toml configuration file to point to the data you already built, the time period to be run,
	the unique field intentifier (should be 'FID'), etc. 
	
	- Preprocess the 'observed' data to be used for calbration using examples/flux/preproc.py, the oberved data
	 is simply the corrected OpenET ETr times SSEBop ETf at this point (i.e., etr * etf = eta). Consider using 
	 eto and/or another RSET algorithm in the future. This writes eta.np.
	
	- Modify custom_forward_run.py to correct paths. This is the python script PEST++ will be calling each iteration.
	
	- Use the calbrate/build_etd_pp.py script to build a PEST++ configuration and subdirectory. This will be a subdirectory of
	the flux project directory, i.e., swim-rs/examples/flux/pest. The function build_pest.py will erase the existing pest
	directory if there is one! It will also copy everything from examples/flux into the pest directory. This function builds
	the flux.pst file, which is the only argument needed at this time to run PEST++ on the problem. Note that during the processing
	of the ETf data, we wrote an e.g., etf_inv_irr_ct.csv table that simply marked the image capture dates. In build_pest(), the 
	observations are given weight 1.0 on these dates, and weight 0.0 on non-capture dates. The idea here is to only evaluate
	the objective function on capture dates to give the model the freedom to behave like a soil water balance model on 
	in-between dates.
	
	- Run 'pestpp-glm' to calibrate parameters. Read the PEST++ documentation to learn other options, configurations, etc.
	See https://github.com/usgs/pestpp/blob/master/documentation/pestpp_users_manual.md
	
Notes on the current PEST++ interface in SWIM-RS:
	
	
	  
	  
	  

