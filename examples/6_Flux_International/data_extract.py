import os

from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.prep import get_flux_sites
from swimrs.swim.config import ProjectConfig


def extract_era5land(conf):
    from swimrs.data_extraction.ee.ee_era5 import sample_era5_land_variables_daily
    is_authorized()

    sample_era5_land_variables_daily(
        feature_coll_asset_id=conf.ee_fields_flux,
        bucket=conf.ee_bucket,
        debug=False,
        check_dir=None,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=conf.feature_id_col,
    )


def extract_properties(conf):
    is_authorized()
    from swimrs.data_extraction.ee.ee_props import get_landcover, get_hwsd

    get_landcover(conf.ee_fields_flux, None, debug=False, selector=conf.feature_id_col, local_file=None)

    get_hwsd(conf.ee_fields_flux, None, debug=False, selector=conf.feature_id_col, local_file=None)


def extract_remote_sensing(conf, sites):
    is_authorized()
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    src = 'ndvi'
    mask = 'no_mask'

    print('landsat', src, mask)
    dst = os.path.join(conf.landsat_ee_data_dir, src, mask)
    sparse_sample_ndvi(conf.fields_shapefile, bucket=conf.ee_bucket, debug=False, satellite='landsat',
                       mask_type=mask, check_dir=dst, start_yr=2015, end_yr=2024, feature_id=conf.feature_id_col,
                       select=sites)

    print('sentinel', src, mask)
    dst = os.path.join(conf.sentinel_ee_data_dir, src, mask)
    sparse_sample_ndvi(conf.fields_shapefile, bucket=conf.ee_bucket, debug=False, satellite='sentinel',
                       mask_type=mask, check_dir=dst, start_yr=2017, end_yr=2024, feature_id=conf.feature_id_col,
                       select=sites)


if __name__ == '__main__':


    project = '6_Flux_International'
    western_only = False

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    select_sites = get_flux_sites(config.station_metadata_csv, crop_only=False, western_only=western_only, header=1,
                                  index_col=0)

    extract_era5land(config)
    extract_properties(config)
    extract_remote_sensing(config, select_sites)

# ========================= EOF ====================================================================
