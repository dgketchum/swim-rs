import os

from prep import get_flux_sites, get_ensemble_parameters

project = '4_Flux_Network'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)

if not os.path.isdir(root):
    root = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'swim-rs')
    data = os.path.join(root, 'tutorials', project, 'data')
    project_ws = os.path.join(root, 'tutorials', project)

landsat = os.path.join(data, 'landsat')
extracts = os.path.join(landsat, 'extracts')
tables = os.path.join(landsat, 'tables')

# GCS - Earth Engine
fields_ = 'users/dgketchum/fields/flux'
bucket_ = 'wudr'

# Volk Benchmark static footprints
FEATURE_ID = 'site_id'
state_col = 'state'

shapefile_path = os.path.join(data, 'gis', 'flux_static_footprints.shp')

# preparation-specific paths
remote_sensing_file = os.path.join(landsat, 'remote_sensing.csv')
joined_timeseries = os.path.join(data, 'plot_timeseries')
station_file = os.path.join(data, 'station_metadata.csv')
irr = os.path.join(data, 'properties', 'calibration_irr.csv')
properties = os.path.join(data, 'properties', 'calibration_properties.csv')
dyanmics_data = os.path.join(landsat, 'calibration_dynamics.json')

snow_in = os.path.join(data, 'snodas', 'extracts')
snow_out = os.path.join(data, 'snodas', 'snodas.json')

prepped_input = os.path.join(data, 'prepped_input.json')


# Open-ET sites covered by overpass date image collections
sites = get_flux_sites(shapefile_path, crop_only=False, western_only=True, header=1, index_col=0)


def prep_remote_sensing():
    from prep.remote_sensing import sparse_time_series, join_remote_sensing

    types_ = ['irr', 'inv_irr']
    sensing_params = ['etf', 'ndvi']

    models = get_ensemble_parameters(skip='ndvi')
    rs_files = []

    ee_data, src, src_ct = None, None, None

    for mask_type in types_:

        for sensing_param in sensing_params:

            yrs = [x for x in range(1987, 2025)]

            if sensing_param == 'etf':

                for model in models:
                    ee_data = os.path.join(landsat, 'extracts', f'{model}_{sensing_param}', mask_type)
                    src = os.path.join(tables, '{}_{}_{}.csv'.format(model, sensing_param, mask_type))
                    src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(model, sensing_param, mask_type))

            else:
                ee_data = os.path.join(landsat, 'extracts', sensing_param, mask_type)
                src = os.path.join(tables, '{}_{}.csv'.format(sensing_param, mask_type))
                src_ct = os.path.join(tables, '{}_{}_ct.csv'.format(sensing_param, mask_type))

            rs_files.extend([src, src_ct])
            sparse_time_series(shapefile_path, ee_data, yrs, src, src_ct,
                               feature_id=FEATURE_ID, select=sites_)

    join_remote_sensing(rs_files, remote_sensing_file)


def prep_field_properties():
    from prep.field_properties import write_field_properties

    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    ssurgo = os.path.join(data, 'properties', 'calibration_ssurgo.csv')
    modis_lulc = os.path.join(data, 'properties', 'calibration_lulc.csv')
    properties_json = os.path.join(data, 'properties', 'calibration_properties.json')

    flux_metadata = os.path.join(data, 'station_metadata.csv')

    write_field_properties(shapefile_path, irr, ssurgo, properties_json, lulc=modis_lulc, index_col=FEATURE_ID,
                           flux_meta=flux_metadata)


def prep_snow():
    from data_extraction.snodas.snodas import create_timeseries_json

    create_timeseries_json(snow_in, snow_out, feature_id=FEATURE_ID)


def prep_timeseries():
    from prep.field_timeseries import join_daily_timeseries

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
    met = os.path.join(data, 'met_timeseries')

    # process irr/inv_irr of all rs parameters, incl. NDVI
    remote_sensing_parameters = get_ensemble_parameters()

    join_daily_timeseries(fields=fields_gridmet,
                          gridmet_dir=met,
                          landsat_table=remote_sensing_file,
                          snow=snow_out,
                          dst_dir=joined_timeseries,
                          overwrite=True,
                          start_date='1987-01-01',
                          end_date='2024-12-31',
                          feature_id=FEATURE_ID,
                          **{'params': remote_sensing_parameters,
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
    pass
# ========================= EOF ====================================================================
