import os

import ee

from data_extraction.ee.ee_utils import is_authorized
from prep import get_flux_sites, get_ensemble_parameters
from swim.config import ProjectConfig

project = '4_Flux_Network'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)

if not os.path.isdir(root):
    root = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'swim-rs')
    data = os.path.join(root, 'tutorials', project, 'data')
    project_ws = os.path.join(root, 'tutorials', project)

config_file = os.path.join(project_ws, 'config.toml')

config = ProjectConfig()
config.read_config(config_file, project_ws)

landsat = os.path.join(config.data_folder, 'landsat')
extracts = os.path.join(landsat, 'extracts')
tables = os.path.join(landsat, 'tables')

# GCS - Earth Engine
fields_ = 'users/dgketchum/fields/flux'
bucket_ = 'wudr'

# Volk Benchmark static footprints
FEATURE_ID = 'site_id'
state_col = 'state'

shapefile_path = os.path.join(data, 'gis', 'flux_static_footprints.shp')

# Open-ET sites covered by overpass date image collections
sites = get_flux_sites(shapefile_path, crop_only=False, western_only=True)


def extract_era5land_swe():
    from data_extraction.ee.ee_era5 import sample_era5_swe_daily

    is_authorized()

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    bucket_ = 'wudr'
    fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
    chk_swe = os.path.join(d, 'examples/tutorial/era5land/extracts/swe')
    FEATURE_ID = 'sid'

    sample_era5_swe_daily(
        feature_coll_asset_id=fields,
        bucket=bucket_,
        debug=False,
        check_dir=chk_swe,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=FEATURE_ID)


def extract_properties():
    from data_extraction.ee.ee_props import get_landcover

    is_authorized()

    index_col = 'sid'

    description = '{}_landcover'.format(project)
    get_landcover(fields_, description, debug=True, selector=index_col, out_fmt='SHP')


def extract_remote_sensing():
    is_authorized()
    from data_extraction.ee.etf_export import sparse_sample_etf
    from data_extraction.ee.ndvi_export import sparse_sample_ndvi

    models = get_ensemble_parameters(skip='ndvi')

    for src in ['ndvi', 'etf']:
        for mask in ['irr', 'inv_irr']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(landsat, 'extracts', src, mask)

                sparse_sample_ndvi(shapefile_path, bucket=bucket_, debug=False, grid_spec=3,
                                   mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                   state_col=state_col, select=sites)

            if src == 'etf':
                for model in models:
                    dst = os.path.join(landsat, 'extracts', f'{model}_{src}', mask)

                    print(src, mask, model)

                    sparse_sample_etf(shapefile_path, bucket=bucket_, debug=False, grid_spec=3,
                                      mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                      state_col=state_col, select=sites, model=model)


def extract_era5land_eto():
    from data_extraction.ee.ee_era5 import sample_era5_eto_daily
    ee.Initialize()

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    bucket_ = 'wudr'
    fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
    FEATURE_ID = 'sid'

    chk_eto = os.path.join(d, 'examples/tutorial/era5land/extracts/eto')
    sample_era5_eto_daily(
        feature_coll_asset_id=fields,
        bucket=bucket_,
        debug=True,
        check_dir=chk_eto,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=FEATURE_ID
    )


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
