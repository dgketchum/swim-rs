import os

from prep import get_flux_sites, get_ensemble_parameters

project = '5_Flux_Ensemble'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)

if not os.path.isdir(root):
    root = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'swim-rs')
    data = os.path.join(root, 'tutorials', project, 'data')
    project_ws = os.path.join(root, 'tutorials', project)

landsat = os.path.join(data, 'landsat')
landsat_ee_data = os.path.join(landsat, 'extracts')
landsat_tables = os.path.join(landsat, 'tables')

# GCS - Earth Engine
fields_ = 'users/dgketchum/fields/flux'
bucket_ = 'wudr'

# Volk Benchmark static footprints
FEATURE_ID = 'site_id'
state_col = 'state'

gridmet_mapping = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
footprint_shapefile = os.path.join(data, 'gis', 'flux_static_footprints.shp')
station_metadata = os.path.join(data, 'station_metadata.csv')

# preparation-specific paths
remote_sensing_tables = os.path.join(data, 'rs_tables')
joined_timeseries = os.path.join(data, 'plot_timeseries')
station_file = os.path.join(data, 'station_metadata.csv')
irr = os.path.join(data, 'properties', 'calibration_irr.csv')
properties = os.path.join(data, 'properties', 'calibration_properties.csv')
dyanmics_data = os.path.join(landsat, 'calibration_dynamics.json')

snow_in = os.path.join(data, 'snodas', 'extracts')
snow_out = os.path.join(data, 'snodas', 'snodas.json')

prepped_input = os.path.join(data, 'prepped_input.json')

# Open-ET sites covered by overpass date image collections
sites = get_flux_sites(station_metadata, crop_only=False, western_only=True, header=1, index_col=0)


def prep_earthengine_extracts():
    from prep.remote_sensing import sparse_time_series, join_remote_sensing

    types_ = ['irr', 'inv_irr']
    sensing_params = ['etf', 'ndvi']

    models = ['openet', 'eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop', 'disalexi']
    rs_files = []

    overwrite = False

    for mask_type in types_:

        for sensing_param in sensing_params:

            yrs = [x for x in range(1987, 2025)]

            if sensing_param == 'etf':

                for model in models:
                    ee_data = os.path.join(landsat, 'extracts', f'{model}_{sensing_param}', mask_type)
                    src = os.path.join(landsat_tables, '{}_{}_{}.parquet'.format(model, sensing_param, mask_type))
                    src_ct = None
                    rs_files.extend([src])
                    if os.path.exists(src) and not overwrite:
                        continue
                    else:
                        sparse_time_series(footprint_shapefile, ee_data, yrs, src, out_pqt_ct=src_ct,
                                           instrument='landsat', algorithm=model, parameter=sensing_param,
                                           mask=mask_type, footprint_spec=3,
                                           feature_id=FEATURE_ID, select=sites, interoplate=False)

            else:
                ee_data = os.path.join(landsat, 'extracts', sensing_param, mask_type)
                src = os.path.join(landsat_tables, '{}_{}.parquet'.format(sensing_param, mask_type))
                src_ct = os.path.join(landsat_tables, '{}_{}_ct.parquet'.format(sensing_param, mask_type))
                rs_files.extend([src, src_ct])
                if os.path.exists(src) and os.path.exists(src_ct) and not overwrite:
                    continue
                else:
                    sparse_time_series(footprint_shapefile, ee_data, yrs, src, out_pqt_ct=src_ct, footprint_spec=3,
                                       instrument='landsat', algorithm='none', parameter=sensing_param, mask=mask_type,
                                       feature_id=FEATURE_ID, select=sites, interoplate=True)



    join_remote_sensing(rs_files, remote_sensing_tables, station_selection='inclusive')


def prep_field_properties():
    from prep.field_properties import write_field_properties

    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    ssurgo = os.path.join(data, 'properties', 'calibration_ssurgo.csv')
    modis_lulc = os.path.join(data, 'properties', 'calibration_lulc.csv')
    properties_json = os.path.join(data, 'properties', 'calibration_properties.json')

    flux_metadata = os.path.join(data, 'station_metadata.csv')

    write_field_properties(footprint_shapefile, irr, ssurgo, properties_json, lulc=modis_lulc, index_col=FEATURE_ID,
                           flux_meta=flux_metadata)


def prep_snow():
    from data_extraction.snodas.snodas import create_timeseries_json

    create_timeseries_json(snow_in, snow_out, feature_id=FEATURE_ID)


def prep_timeseries():
    from prep.field_timeseries import join_daily_timeseries

    met = os.path.join(data, 'met_timeseries')

    join_daily_timeseries(fields=gridmet_mapping, met_dir=met, rs_dir=remote_sensing_tables,
                          dst_dir=joined_timeseries, snow=snow_out, overwrite=True, start_date='1987-01-01',
                          end_date='2024-12-31', feature_id=FEATURE_ID, **{'met_mapping': 'GFID',
                                                                           'mapped_dir': met,
                                                                           'target_fields': sites})


def prep_dynamics():
    from prep.dynamics import SamplePlotDynamics

    dynamics = SamplePlotDynamics(joined_timeseries, irr, irr_threshold=0.3, etf_target='ssebop',
                                  out_json_file=dyanmics_data, select=sites)
    dynamics.analyze_groundwater_subsidy()
    dynamics.analyze_irrigation(lookback=5)
    dynamics.analyze_k_parameters()
    dynamics.save_json()


def prep_input_json():
    from prep.prep_plots import prep_fields_json

    params = get_ensemble_parameters()
    prep_fields_json(properties, joined_timeseries, dyanmics_data,
                     prepped_input, target_plots=sites, rs_params=params)


if __name__ == '__main__':
    # prep_earthengine_extracts()
    prep_field_properties()
    prep_timeseries()
    # prep_dynamics()
    # prep_input_json()
# ========================= EOF ====================================================================
