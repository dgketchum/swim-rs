import os

from data_extraction.ee.ee_utils import is_authorized
from prep import get_flux_sites, get_ensemble_parameters
from swim.config import ProjectConfig


def extract_snodas(conf):
    is_authorized()

    from data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(feature_coll=conf.ee_fields_flux, bucket=conf.ee_bucket, debug=False, check_dir=None,
                      feature_id=conf.feature_id_col)


def extract_properties(conf):
    is_authorized()

    from data_extraction.ee.ee_props import get_irrigation, get_ssurgo, get_cdl, get_landcover

    description = '{}_cdl'.format(project)
    get_cdl(conf.ee_fields_flux, description, selector=conf.feature_id_col)

    description = '{}_irr'.format(project)
    get_irrigation(conf.ee_fields_flux, description, debug=True, selector=conf.feature_id_col, lanid=True)

    description = '{}_ssurgo'.format(project)
    get_ssurgo(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col)

    description = '{}_landcover'.format(project)
    get_landcover(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col, out_fmt='CSV')


def extract_remote_sensing(conf, sites, get_sentinel=False, selected_models=None):
    from data_extraction.ee.etf_export import sparse_sample_etf
    from data_extraction.ee.ndvi_export import sparse_sample_ndvi
    is_authorized()

    models = get_ensemble_parameters(skip='ndvi')
    models = list(set([m[0] for m in models]))
    for src in ['ndvi', 'etf']:
        for mask in ['irr', 'inv_irr']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(conf.landsat_dir, 'extracts', src, mask)
                sparse_sample_ndvi(conf.footprint_shapefile_shp, bucket=conf.ee_bucket, debug=False, grid_spec=3,
                                   mask_type=mask, check_dir=dst, start_yr=1987, end_yr=2024,
                                   feature_id=conf.feature_id_col, satellite='landsat',
                                   state_col=conf.state_col, select=sites)

                if get_sentinel:
                    dst = os.path.join(conf.sentinel_dir, 'extracts', src, mask)
                    sparse_sample_ndvi(conf.footprint_shapefile_shp, bucket=conf.ee_bucket, debug=False, grid_spec=3,
                                       mask_type=mask, check_dir=dst, start_yr=2018, end_yr=2024,
                                       feature_id=conf.feature_id_col, satellite='sentinel',
                                       state_col=conf.state_col, select=sites)

            if src == 'etf':
                for model in models:

                    if selected_models is not None and model not in selected_models:
                        continue

                    dst = os.path.join(conf.landsat_dir, 'extracts', f'{model}_{src}', mask)
                    print(src, mask, model)

                    sparse_sample_etf(conf.footprint_shapefile_shp, bucket=conf.ee_bucket, debug=False, grid_spec=3,
                                      mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024,
                                      feature_id=conf.feature_id_col,
                                      state_col=conf.state_col, select=sites, model=model)


def extract_gridmet(conf, sites):
    from data_extraction.gridmet.gridmet import get_gridmet_corrections
    from data_extraction.gridmet.gridmet import download_gridmet

    get_gridmet_corrections(fields=conf.gridmet_mapping_shp,
                            gridmet_ras=conf.correction_tifs,
                            fields_join=conf.gridmet_mapping_shp,
                            factors_js=conf.gridmet_factors,
                            feature_id='field_1',
                            field_select=sites)

    download_gridmet(conf.gridmet_mapping_shp, conf.gridmet_factors, conf.met_dir, start='1987-01-01', end='2024-12-31',
                     overwrite=False, append=False,
                     feature_id=conf.gridmet_mapping_index_col, target_fields=sites)


if __name__ == '__main__':

    for project in ['4_Flux_Network']:

        if project == '5_Flux_Ensemble':
            western_only = True
            models_select = None

        else:
            western_only = True
            models_select = ['ssebop']

        home = os.path.expanduser('~')
        config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

        config = ProjectConfig()
        config.read_config(config_file)

        select_sites = get_flux_sites(config.station_metadata_csv, crop_only=False, western_only=False, header=1,
                                      index_col=0)

        extract_snodas(config)
        extract_properties(config)
        extract_remote_sensing(config, select_sites)
        extract_gridmet(config, select_sites)

# ========================= EOF ====================================================================
