To prepare the data, the order is roughly as follows:

1. field_properties.write_field_properties()

2. landsat_sensing.sparse_landsat_time_series()

3. landsat_sensing.join_remote_sensing()

4. field_timeseries.join_daily_timeseries()

5. dynamics.SamplePlotDynamics:
                spd.analyze_groundwater_subsidy()
                spd.analyze_irrigation()
                spd.analyze_k_parameters()
                spd.save_json()

6. prep_plots.prep_fields_json()

7. prep_plots.preproc()

Assume a modification to an item higher on the list requires re-running all subsequent steps.
