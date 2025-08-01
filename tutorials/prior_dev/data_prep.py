import os

from prep import get_ensemble_parameters
from swim.config import ProjectConfig


def prep_earthengine_extracts(conf, sites=None, overwrite=False):
    from prep.remote_sensing import join_remote_sensing, clustered_time_series

    types_ = ['irr']
    sensing_params = ['ndvi']

    rs_files = []
    models = [conf.etf_target_model]
    if conf.etf_ensemble_members is not None:
        models += conf.etf_ensemble_members

    for mask_type in types_:

        for sensing_param in sensing_params:

            yrs = [x for x in range(conf.start_dt.year, conf.end_dt.year + 1)]

            if sensing_param == 'etf':

                for model in models:
                    ee_data = os.path.join(conf.landsat_dir, 'extracts', f'{model}_{sensing_param}', mask_type)
                    src = os.path.join(conf.landsat_tables_dir,
                                       '{}_{}_{}.parquet'.format(model, sensing_param, mask_type))
                    rs_files.extend([src])
                    if os.path.exists(src) and not overwrite:
                        continue
                    else:
                        clustered_time_series(conf.footprint_shapefile_shp, ee_data, yrs, src,
                                           feature_id=conf.feature_id_col,
                                           instrument='landsat', parameter=sensing_param, algorithm=model,
                                           mask=mask_type, select=sites)

            else:
                ee_data = os.path.join(conf.landsat_dir, 'extracts', sensing_param, mask_type)
                src = os.path.join(conf.landsat_tables_dir, '{}_{}.parquet'.format(sensing_param, mask_type))
                rs_files.extend([src])
                if os.path.exists(src) and not overwrite:
                    continue
                else:
                    clustered_time_series(conf.footprint_shapefile_shp, ee_data, yrs, src,
                                          feature_id=conf.feature_id_col,
                                          instrument='landsat', parameter=sensing_param, algorithm='none',
                                          mask=mask_type,
                                          select=sites)

    join_remote_sensing(rs_files, conf.remote_sensing_tables_dir, station_selection='inclusive')


def prep_field_properties(conf, sites=None):
    from prep.field_properties import write_field_properties

    write_field_properties(conf.footprint_shapefile_shp, conf.properties_json, conf.lulc_csv,
                           irr=conf.irr_csv,
                           lulc_key='modis_lc',
                           soils=conf.ssurgo_csv,
                           index_col=conf.feature_id_col,
                           flux_meta=conf.station_metadata_csv,
                           select=sites,
                           **{'extra_lulc_key': 'glc10_lc'}
                           )


def prep_snow(conf):
    from data_extraction.snodas.snodas import create_timeseries_json

    create_timeseries_json(conf.snodas_in_dir, conf.snodas_out_json, feature_id=conf.feature_id_col)


def prep_timeseries(conf, sites=None):
    from prep.field_timeseries import join_daily_timeseries

    join_daily_timeseries(fields=conf.gridmet_mapping_shp,
                          met_dir=conf.met_dir,
                          rs_dir=conf.remote_sensing_tables_dir,
                          dst_dir=conf.joined_timeseries_dir,
                          snow=conf.snodas_out_json,
                          overwrite=True,
                          start_date=conf.start_dt,
                          end_date=conf.end_dt,
                          feature_id=conf.gridmet_mapping_index_col,
                          **{'met_mapping': 'GFID'})


def prep_dynamics(conf, sites=None):
    from prep.dynamics import SamplePlotDynamics

    # sites = ['Almond_High']
    dynamics = SamplePlotDynamics(conf.joined_timeseries_dir,
                                  conf.properties_json,
                                  irr_threshold=conf.irrigation_threshold,
                                  etf_target=conf.etf_target_model,
                                  out_json_file=conf.dynamics_data_json,
                                  select=sites,
                                  use_lulc=False,
                                  use_mask=True,
                                  masks=('irr', 'inv_irr'),
                                  instruments=('landsat',))

    dynamics.analyze_irrigation(lookback=5)
    dynamics.analyze_k_parameters()
    dynamics.analyze_groundwater_subsidy()
    dynamics.save_json()


def prep_input_json(conf, sites=None):
    from prep.prep_plots import prep_fields_json

    params = get_ensemble_parameters()
    prep_fields_json(conf.properties_json,
                     conf.joined_timeseries_dir,
                     conf.dynamics_data_json,
                     conf.input_data,
                     target_plots=sites,
                     rs_params=params,
                     interp_params=('ndvi',))


if __name__ == '__main__':
    """"""
    project = 'prior_dev'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    prep_earthengine_extracts(config, overwrite=True)
    prep_field_properties(config)
    prep_snow(config)
    prep_timeseries(config)
    prep_dynamics(config)
    prep_input_json(config)

# ========================= EOF ====================================================================
