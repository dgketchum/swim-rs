import os

import ee

from data_extraction.ee.ee_utils import is_authorized
from prep import get_flux_sites, get_ensemble_parameters
from swim.config import ProjectConfig

# project = '4_Flux_Network'
project = '5_Flux_Ensemble'

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

# GCS - Earth Engine
fields_ = 'projects/ee-dgketchum/assets/swim/flux_footprints_3p'
bucket_ = 'wudr'

# Volk Benchmark static footprints
FEATURE_ID = 'site_id'
state_col = 'state'

shapefile_path = os.path.join(data, 'gis', 'flux_footprints_3p.shp')

station_metadata = os.path.join(data, 'station_metadata.csv')

# Open-ET sites covered by overpass date image collections
sites = get_flux_sites(station_metadata, crop_only=False, western_only=True, header=1, index_col=0)


def extract_snodas():
    is_authorized()

    from data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(shapefile_path, bucket_, debug=False, check_dir=None, feature_id=FEATURE_ID)


def extract_properties():
    is_authorized()

    from data_extraction.ee.ee_props import get_irrigation, get_ssurgo, get_cdl, get_landcover

    description = '{}_cdl'.format(project)
    get_cdl(fields_, description, selector=FEATURE_ID)

    description = '{}_irr'.format(project)
    get_irrigation(fields_, description, debug=True, selector=FEATURE_ID, lanid=True)

    description = '{}_ssurgo'.format(project)
    get_ssurgo(fields_, description, debug=False, selector=FEATURE_ID)

    description = '{}_landcover'.format(project)
    get_landcover(fields_, description, debug=False, selector=FEATURE_ID, out_fmt='CSV')


def extract_remote_sensing():
    from data_extraction.ee.etf_export import sparse_sample_etf
    from data_extraction.ee.ndvi_export import sparse_sample_ndvi
    is_authorized()

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
    flux_field_fid = 'field_1'

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
    gridmet_factors = os.path.join(data, 'gis', 'flux_fields_gfid.json')

    # get_gridmet_corrections(fields=shapefile_path,
    #                         gridmet_ras=correction_tifs,
    #                         fields_join=fields_gridmet,
    #                         factors_js=gridmet_factors,
    #                         feature_id='field_1',
    #                         field_select=sites)

    met = os.path.join(data, 'met_timeseries', 'gridmet')

    download_gridmet(fields_gridmet, gridmet_factors, met, start='1987-01-01', end='2024-12-31',
                     overwrite=True, append=False,
                     feature_id=flux_field_fid, target_fields=sites)


if __name__ == '__main__':
    # extract_properties()
    extract_gridmet()
    pass
# ========================= EOF ====================================================================
