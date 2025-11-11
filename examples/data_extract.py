import os

from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.swim.config import ProjectConfig


def extract_snodas(conf):
    is_authorized()

    from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(feature_coll=conf.ee_fields_flux, bucket=conf.ee_bucket, debug=False, check_dir=None,
                      feature_id=conf.feature_id_col)


def extract_properties(conf):
    is_authorized()

    from swimrs.data_extraction.ee.ee_props import get_irrigation, get_ssurgo, get_cdl, get_landcover

    description = '{}_cdl'.format(project)
    get_cdl(conf.ee_fields_flux, description, selector=conf.feature_id_col)

    description = '{}_irr'.format(project)
    get_irrigation(conf.ee_fields_flux, description, debug=True, selector=conf.feature_id_col, lanid=True)

    description = '{}_ssurgo'.format(project)
    get_ssurgo(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col)

    description = '{}_landcover'.format(project)
    get_landcover(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col, out_fmt='CSV')


def extract_remote_sensing(conf, sites, get_sentinel=False, selected_models=None):
    from swimrs.data_extraction.ee.etf_export import sparse_sample_etf
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi
    is_authorized()

    models = [conf.etf_target_model] + conf.etf_ensemble_members
    for src in ['ndvi']:
        for mask in ['irr', 'inv_irr']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(conf.landsat_dir, 'extracts', src, mask)
                sparse_sample_ndvi(conf.fields_shapefile, bucket=conf.ee_bucket, debug=False,
                                   mask_type=mask, check_dir=dst, start_yr=1987, end_yr=2024,
                                   feature_id=conf.feature_id_col, satellite='landsat',
                                   state_col=conf.state_col, select=sites)

                if get_sentinel:
                    dst = os.path.join(conf.sentinel_dir, 'extracts', src, mask)
                    sparse_sample_ndvi(conf.fields_shapefile, bucket=conf.ee_bucket, debug=False,
                                       mask_type=mask, check_dir=dst, start_yr=2017, end_yr=2024,
                                       feature_id=conf.feature_id_col, satellite='sentinel',
                                       state_col=conf.state_col, select=sites)

            if src == 'etf':
                # TODO: refactor ptjpl and sims model code into this code block
                for model in models:

                    if model in ['ptjpl', 'sims']:
                        print(f'These were build with OpenET software')
                        continue

                    if selected_models is not None and model not in selected_models:
                        continue

                    dst = os.path.join(conf.landsat_dir, 'extracts', f'{model}_{src}', mask)
                    print(src, mask, model)

                    sparse_sample_etf(conf.fields_shapefile, bucket=conf.ee_bucket, debug=False,
                                      mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024,
                                      feature_id=conf.feature_id_col,
                                      state_col=conf.state_col, select=sites, model=model)


def extract_gridmet(conf, sites):
    from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_and_corrections
    from swimrs.data_extraction.gridmet.gridmet import download_gridmet

    # infer hourly NLDAS need from runoff mode only at download callsite
    nldas_needed = (conf.swb_mode == 'ier')

    assign_gridmet_and_corrections(fields=conf.gridmet_mapping_shp,
                                   gridmet_ras=conf.correction_tifs,
                                   fields_join=conf.gridmet_mapping_shp,
                                   factors_js=conf.gridmet_factors,
                                   feature_id='site_id',
                                   field_select=sites)

    download_gridmet(conf.gridmet_mapping_shp, conf.gridmet_factors, conf.met_dir, start='1987-01-01', end='2024-12-31',
                     overwrite=False, append=True, use_nldas=nldas_needed,
                     feature_id=conf.gridmet_mapping_index_col, target_fields=sites)


if __name__ == '__main__':

    for project in ['5_Flux_Ensemble']:

        if project == '5_Flux_Ensemble':
            models_select = None
        else:
            models_select = ['ssebop']

        home = os.path.expanduser('~')
        config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{project}.toml')

        config = ProjectConfig()
        config.read_config(config_file)

        # select_sites = get_flux_sites(config.station_metadata_csv, crop_only=False, western_only=False, header=1,
        #                               index_col=0)
        select_sites = None

        extract_snodas(config)
        extract_properties(config)
        extract_remote_sensing(config, select_sites, get_sentinel=True)
        extract_gridmet(config, select_sites)

# ========================= EOF ====================================================================
