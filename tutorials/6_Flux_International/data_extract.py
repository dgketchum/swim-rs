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
landsat_extracts = os.path.join(landsat, 'extracts')
landsat_tables = os.path.join(landsat, 'tables')

sentinel = os.path.join(data, 'sentinel')
sentinel_extracts = os.path.join(sentinel, 'ee_extracts')
sentinel_tables = os.path.join(sentinel, 'tables')

era5_extracts = os.path.join(data, 'met_timeseries', 'extracts')

modis_lulc = os.path.join(data, 'properties', f'{project}_landcover.csv')
soils = os.path.join(data, 'properties', f'{project}_hwsd.csv')
properties_json = os.path.join(data, 'properties', 'calibration_properties.json')

# GCS - Earth Engine
fields = 'projects/ee-dgketchum/assets/swim/6_Flux_International_landcover_150mBuf_13MAY2025'
bucket_ = 'wudr'

# ICOS 200m station buffer shapefile index
FEATURE_ID = 'sid'

# European crop sites
shapefile_path = os.path.join(data, 'gis', '6_Flux_International_150mBuf.shp')
sites = get_flux_sites(shapefile_path, index_col=FEATURE_ID)


def extract_era5land():
    from data_extraction.ee.ee_era5 import sample_era5_land_variables_daily
    is_authorized()

    sample_era5_land_variables_daily(
        feature_coll_asset_id=fields,
        bucket=bucket_,
        debug=False,
        check_dir=era5_extracts,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=FEATURE_ID
    )


def extract_properties():
    is_authorized()
    from data_extraction.ee.ee_props import get_landcover, get_hwsd

    index_col = 'sid'
    get_landcover(fields, None, debug=False, selector=index_col, local_file=modis_lulc)

    get_hwsd(fields, None, debug=False, selector=index_col, local_file=soils)


def extract_remote_sensing():
    is_authorized()
    from data_extraction.ee.ndvi_export import sparse_sample_ndvi

    src = 'ndvi'
    mask = 'no_mask'

    print('landsat', src, mask)
    dst = os.path.join(landsat_extracts, src, mask)
    sparse_sample_ndvi(shapefile_path, bucket=bucket_, debug=False, satellite='landsat',
                       mask_type=mask, check_dir=dst, start_yr=2015, end_yr=2024, feature_id=FEATURE_ID,
                       select=sites)

    print('sentinel', src, mask)
    dst = os.path.join(sentinel_extracts, src, mask)
    sparse_sample_ndvi(shapefile_path, bucket=bucket_, debug=False, satellite='sentinel',
                       mask_type=mask, check_dir=dst, start_yr=2017, end_yr=2024, feature_id=FEATURE_ID,
                       select=sites)


if __name__ == '__main__':
    extract_properties()
    extract_era5land()
    extract_remote_sensing()
# ========================= EOF ====================================================================
