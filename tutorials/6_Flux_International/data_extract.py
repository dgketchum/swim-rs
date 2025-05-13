import os

from data_extraction.ee.ee_utils import is_authorized
from prep import get_flux_sites

project = '6_Flux_International'

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

era5_extracts = os.path.join(data, 'era5land', 'extracts')

# GCS - Earth Engine
fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
bucket_ = 'wudr'

# ICOS 200m station buffer shapefile index
FEATURE_ID = 'sid'

# European crop sites
shapefile_path = os.path.join(data, 'gis', '6_Flux_International_EU_crops_AEA_200mBuf.shp')
sites = get_flux_sites(shapefile_path, index_col=FEATURE_ID)


def extract_era5land_swe():
    from data_extraction.ee.ee_era5 import sample_era5_swe_daily

    is_authorized()

    fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
    chk_swe = os.path.join(era5_extracts, 'swe')

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
    get_landcover(fields, description, debug=True, selector=index_col, out_fmt='CSV')


def extract_remote_sensing():
    is_authorized()
    from data_extraction.ee.etf_export import sparse_sample_etf
    from data_extraction.ee.ndvi_export import sparse_sample_ndvi

    model = 'ptjpl'

    for src in ['ndvi', 'etf']:
        for mask in ['no_mask']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(landsat, 'extracts', src, mask)

                sparse_sample_ndvi(shapefile_path, bucket=bucket_, debug=False, grid_spec=None,
                                   mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                   state_col=None, select=sites)

            if src == 'etf':
                dst = os.path.join(landsat, 'extracts', f'{model}_{src}', mask)

                print(src, mask, model)

                sparse_sample_etf(shapefile_path, bucket=bucket_, debug=False, grid_spec=None,
                                  mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                  state_col=None, select=sites, model=model)


def extract_era5land_eto():
    from data_extraction.ee.ee_era5 import sample_era5_eto_daily
    is_authorized()

    chk_eto = os.path.join(era5_extracts, 'eto')
    sample_era5_eto_daily(
        feature_coll_asset_id=fields,
        bucket=bucket_,
        debug=True,
        check_dir=chk_eto,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=FEATURE_ID)


if __name__ == '__main__':
    extract_properties()
# ========================= EOF ====================================================================
