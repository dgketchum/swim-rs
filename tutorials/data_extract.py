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
sites = get_flux_sites(shapefile_path, crop_only=False, western_only=True, header=1, index_col=0)


def extract_snodas():
    from data_extraction.ee.snodas_export import sample_snodas_swe
    ee.Initialize()

    fields_ = 'projects/ee-dgketchum/assets/swim/mt_sid_boulder'

    sample_snodas_swe(fields_, bucket_, debug=False, check_dir=None, feature_id=FEATURE_ID)


def extract_properties():
    from data_extraction.ee.ee_props import get_irrigation, get_ssurgo, get_cdl, get_landcover

    index_col = 'sid'
    description = '{}_cdl'.format(project)
    get_cdl(fields_, description, selector=index_col)

    description = '{}_irr'.format(project)
    get_irrigation(fields_, description, debug=True, selector=index_col, lanid=True)

    description = '{}_ssurgo'.format(project)
    get_ssurgo(fields_, description, debug=False, selector=index_col)

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


def extract_gridmet():
    from data_extraction.gridmet.gridmet import get_gridmet_corrections
    from data_extraction.gridmet.gridmet import download_gridmet

    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')
    correction_tifs = os.path.join(data, 'bias_correction_tif')

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
    gridmet_factors = os.path.join(data, 'gis', 'flux_fields_gfid.json')

    get_gridmet_corrections(fields=shapefile_path,
                            gridmet_ras=correction_tifs,
                            fields_join=fields_gridmet,
                            factors_js=gridmet_factors,
                            feature_id='field_1',
                            field_select=sites)

    met = os.path.join(data, 'met_timeseries')

    download_gridmet(fields_gridmet, gridmet_factors, met, start='1987-01-01', end='2023-12-31',
                     overwrite=False, feature_id=FEATURE_ID, target_fields=None)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
