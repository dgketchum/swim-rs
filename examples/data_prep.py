import os

from swimrs.swim.config import ProjectConfig
from swimrs.prep import get_ensemble_parameters
from swimrs.utils.rs_diagnostics import summarize_observation_counts, merge_counts_dict

# Deprecated: use per-example entrypoints:
# - examples/4_Flux_Network/data_prep.py
# - examples/5_Flux_Ensemble/data_prep.py


def prep_earthengine_extracts(conf, sites, overwrite=False, add_sentinel=False):
    from swimrs.prep.remote_sensing import sparse_time_series, join_remote_sensing

    types_ = ['irr', 'inv_irr']
    sensing_params = ['etf', 'ndvi']

    rs_files = []
    counts_files = []
    models = [conf.etf_target_model]
    if conf.etf_ensemble_members is not None:
        models += conf.etf_ensemble_members

    # Process ETF first: outer loop over models, inner loop over irrigation masks
    if 'etf' in sensing_params:
        for model in models:
            for mask_type in types_:
                yrs = [x for x in range(conf.start_dt.year, conf.end_dt.year + 1)]
                sensing_param = 'etf'
                ee_data = os.path.join(conf.landsat_dir, 'extracts', f'{model}_{sensing_param}', mask_type)
                src = os.path.join(conf.landsat_tables_dir, f'{model}_{sensing_param}_{mask_type}.parquet')
                rs_files.extend([src])
                if os.path.exists(src) and not overwrite:
                    pass
                counts = sparse_time_series(conf.fields_shapefile, ee_data, yrs, src,
                                            feature_id=conf.feature_id_col,
                                            instrument='landsat', parameter=sensing_param, algorithm=model,
                                            mask=mask_type, select=sites)
                if isinstance(counts, tuple) and len(counts) == 2:
                    _, counts_json_path = counts
                    if counts_json_path and os.path.exists(counts_json_path):
                        counts_files.append(counts_json_path)

    # Then process NDVI once per mask (no model dimension)
    if 'ndvi' in sensing_params:
        for mask_type in types_:
            yrs = [x for x in range(conf.start_dt.year, conf.end_dt.year + 1)]
            sensing_param = 'ndvi'

            ee_data = os.path.join(conf.landsat_dir, 'extracts', sensing_param, mask_type)
            src = os.path.join(conf.landsat_tables_dir, f'{sensing_param}_{mask_type}.parquet')
            rs_files.extend([src])
            if os.path.exists(src) and not overwrite:
                pass
            counts = sparse_time_series(conf.fields_shapefile, ee_data, yrs, src,
                                        feature_id=conf.feature_id_col,
                                        instrument='landsat', parameter=sensing_param, algorithm='none',
                                        mask=mask_type, select=sites)
            if isinstance(counts, tuple) and len(counts) == 2:
                _, counts_json_path = counts
                if counts_json_path and os.path.exists(counts_json_path):
                    counts_files.append(counts_json_path)

            if add_sentinel:
                ee_data = os.path.join(conf.sentinel_dir, 'extracts', sensing_param, mask_type)
                src = os.path.join(conf.sentinel_tables_dir, f'{sensing_param}_{mask_type}.parquet')
                rs_files.extend([src])
                if os.path.exists(src) and not overwrite:
                    pass
                counts = sparse_time_series(conf.fields_shapefile, ee_data, yrs, src,
                                            feature_id=conf.feature_id_col, instrument='sentinel',
                                            parameter=sensing_param, algorithm='none', mask=mask_type,
                                            select=sites)

                if isinstance(counts, tuple) and len(counts) == 2:
                    _, counts_json_path = counts
                    if counts_json_path and os.path.exists(counts_json_path):
                        counts_files.append(counts_json_path)

    if counts_files:
        try:
            merged_counts = os.path.join(conf.remote_sensing_tables_dir, 'observation_counts_merged.json')
            merge_counts_dict(counts_files, merged_counts)
            print(f"Wrote merged observation counts: {merged_counts}")
        except Exception as e:
            print(f"Failed to merge observation counts: {e}")

        # Build and write a compact diagnostic summary (CSV + dense JSON)
        try:
            summary_csv = os.path.join(conf.remote_sensing_tables_dir, 'observation_summary.csv')
            summary_json = os.path.join(conf.remote_sensing_tables_dir, 'observation_summary_dense.json')
            yrs = list(range(conf.start_dt.year, conf.end_dt.year + 1))

            summary_df, instrument_summary = summarize_observation_counts(
                json_paths=[merged_counts],
                select_stations=select_sites,
                min_obs_per_year=8,
                all_years=yrs,
                model_whitelist=None,
                mask_whitelist=['irr', 'inv_irr'],
                station_whitelist=None,
                output_csv=summary_csv,
                output_json=summary_json,
            )
            for site, data in instrument_summary.items():
                for instrument, algs in data.items():
                    for alg, stats in algs.items():
                        print(
                            f"{site}: {instrument}:{alg} Mean Obs/Year: "
                            f"{stats['mean_obs']}; Missing {stats['years_w_zero_obs']}")
                print('')
            print(f"Wrote observation summaries: {summary_csv}, {summary_json}")

        except Exception as e:
            print(f"Failed to create observation summaries: {e}")

    join_remote_sensing(rs_files, conf.remote_sensing_tables_dir, station_selection='inclusive')


def prep_field_properties(conf, sites):
    from swimrs.prep.field_properties import write_field_properties

    write_field_properties(conf.fields_shapefile, conf.properties_json, conf.lulc_csv,
                           irr=conf.irr_csv,
                           lulc_key='modis_lc',
                           soils=conf.ssurgo_csv,
                           index_col=conf.feature_id_col,
                           flux_meta=conf.station_metadata_csv,
                           select=sites,
                           **{'extra_lulc_key': 'glc10_lc'}
                           )


def prep_snow(conf, index_col=None):
    from swimrs.data_extraction.snodas.snodas import create_timeseries_json

    if not index_col:
        index_col = conf.feature_id_col
    create_timeseries_json(conf.snodas_in_dir, conf.snodas_out_json, feature_id=index_col)


def prep_timeseries(conf, sites):
    from swimrs.prep.field_timeseries import join_daily_timeseries

    join_daily_timeseries(fields=conf.gridmet_mapping_shp,
                          met_dir=conf.met_dir,
                          rs_dir=conf.remote_sensing_tables_dir,
                          dst_dir=conf.joined_timeseries_dir,
                          snow=conf.snodas_out_json,
                          overwrite=True,
                          start_date=conf.start_dt,
                          end_date=conf.end_dt,
                          feature_id=conf.gridmet_mapping_index_col,
                          **{'met_mapping': 'GFID',
                             'target_fields': None})


def prep_dynamics(conf, sites, sentinel=False):
    from swimrs.prep.dynamics import process_dynamics_batch

    # sites = ['Almond_High']
    if sentinel:
        sensors = ('landsat', 'sentinel')
    else:
        sensors = ('landsat',)

    process_dynamics_batch(conf.joined_timeseries_dir,
                           conf.properties_json,
                           conf.dynamics_data_json,
                           etf_target=conf.etf_target_model,
                           irr_threshold=conf.irrigation_threshold,
                           select=sites,
                           masks=('irr', 'inv_irr'),
                           instruments=sensors,
                           use_lulc=False,
                           use_mask=True,
                           lookback=5,
                           num_workers=12)


def prep_input_json(conf, sites):
    from swimrs.prep.prep_plots import prep_fields_json

    params = get_ensemble_parameters()
    params = [p for p in params if p[0] in ['none', 'ptjpl', 'sims', 'ssebop']]
    prep_fields_json(conf.properties_json,
                     conf.joined_timeseries_dir,
                     conf.dynamics_data_json,
                     conf.input_data,
                     target_plots=sites,
                     rs_params=params,
                     interp_params=('ndvi',))


if __name__ == '__main__':
    """"""
    western_only = None
    snodas_indexer = None

    for project in ['5_Flux_Ensemble']:

        if project == '4_Flux_Network':
            sentinel = True

        if project == '5_Flux_Ensemble':
            sentinel = True

        home = os.path.expanduser('~')
        config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{project}.toml')

        config = ProjectConfig()
        config.read_config(config_file)

        # select_sites = get_flux_sites(config.station_metadata_csv, crop_only=True, western_only=western_only, header=1,
        #                               index_col=0)
        select_sites = None # ['ALARC2_Smith6']

        prep_earthengine_extracts(config, select_sites, overwrite=True, add_sentinel=True)
        prep_field_properties(config, select_sites)
        prep_snow(config, snodas_indexer)
        prep_timeseries(config, select_sites)
        prep_dynamics(config, select_sites, sentinel=True)
        prep_input_json(config, select_sites)

# ========================= EOF ====================================================================
