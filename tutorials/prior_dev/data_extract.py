import os

from data_extraction.ee.ee_utils import is_authorized
from prep import get_ensemble_parameters
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
    get_irrigation(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col, lanid=True)

    description = '{}_ssurgo'.format(project)
    get_ssurgo(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col)

    description = '{}_landcover'.format(project)
    get_landcover(conf.ee_fields_flux, description, debug=False, selector=conf.feature_id_col, out_fmt='CSV')


def extract_remote_sensing(conf, sites):
    from data_extraction.ee.etf_export import clustered_sample_etf
    from data_extraction.ee.ndvi_export import clustered_sample_ndvi
    is_authorized()

    models = get_ensemble_parameters(skip='ndvi')

    for src in ['ndvi', 'etf']:
        for mask in ['irr']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(conf.landsat_dir, 'extracts', src, mask)

                clustered_sample_ndvi(conf.ee_fields, bucket=conf.ee_bucket, debug=False,
                                   mask_type=mask, check_dir=dst, start_yr=conf.start_dt.year, end_yr=conf.end_dt.year,
                                   feature_id=conf.feature_id_col)

            if src == 'etf':
                for model in models:
                    dst = os.path.join(conf.landsat_dir, 'extracts', f'{model}_{src}', mask)

                    print(src, mask, model)

                    clustered_sample_etf(conf.ee_fields, bucket=conf.ee_bucket, debug=False,
                                      mask_type=mask, check_dir=dst, start_yr=conf.start_dt.year,
                                      end_yr=conf.end_dt.year,
                                      feature_id=conf.feature_id_col)


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

    project = 'prior_dev'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    extract_snodas(config)
    extract_properties(config)
    extract_remote_sensing(config, None)
    extract_gridmet(config, None)

# ========================= EOF ====================================================================
