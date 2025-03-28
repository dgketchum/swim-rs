import os
import sys
import time
import subprocess

import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from data_extraction.ee.ee_utils import get_lanid
from data_extraction.ee.ee_utils import is_authorized

EE = '/home/dgketchum/miniconda3/envs/swim/bin/earthengine'
GSUTIL = '/home/dgketchum/google-cloud-sdk/bin/gsutil'

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'

EC_POINTS = 'users/dgketchum/fields/flux'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'


def get_flynn():
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def export_etf_images(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr'):
    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year),
                              '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
    coll = coll.filterBounds(feature_coll)
    scenes = coll.aggregate_histogram('system:index').getInfo()

    for img_id in scenes:

        splt = img_id.split('_')
        _name = '_'.join(splt[-3:])

        img = ee.Image(os.path.join(ETF, img_id))

        if mask_type == 'no_mask':
            img = img.clip(feature_coll.geometry()).int()
        elif mask_type == 'irr':
            img = img.clip(feature_coll.geometry()).mask(irr_mask).int()
        elif mask_type == 'inv_irr':
            img = img.clip(feature_coll.geometry()).mask(irr.gt(0)).int()

        if debug:
            point = ee.Geometry.Point([-106.576, 46.26])
            data = img.sample(point, 30).getInfo()
            print(data['features'])

        task = ee.batch.Export.image.toCloudStorage(
            img,
            description='ETF_{}_{}'.format(mask_type, _name),
            bucket=bucket,
            region=feature_coll.geometry(),
            crs='EPSG:5070',
            scale=30)

        task.start()
        print(_name)


def sparse_sample_etf(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None,
                      feature_id='FID', select=None, start_yr=2000, end_yr=2024, state_col='field_3'):
    df = gpd.read_file(shapefile)
    df.index = df[feature_id]

    assert df.crs.srs == 'EPSG:5071'

    df = df.to_crs(epsg=4326)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    lanid = get_lanid()
    east = ee.FeatureCollection(EAST_STATES)

    skipped, exported = 0, 0

    for fid, row in tqdm(df.iterrows(), desc='Processing Fields', total=df.shape[0]):

        for year in range(start_yr, end_yr + 1):

            if select is not None and fid not in select:
                continue

            site = row[feature_id]
            grid_sz = row['grid_size']

            desc = 'etf_{}_p{}_{}_{}'.format(site, grid_sz, mask_type, year)
            if check_dir:
                f = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(f):
                    skipped += 1
                    continue

            if row[state_col] in STATES:
                irr = irr_coll.filterDate('{}-01-01'.format(year),
                                          '{}-12-31'.format(year)).select('classification').mosaic()
                irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            else:
                irr_mask = lanid.select(f'irr_{year}').clip(east)
                irr = ee.Image(1).subtract(irr_mask)

            polygon = ee.Geometry.Polygon([[c[0], c[1]] for c in row['geometry'].exterior.coords])
            fc = ee.FeatureCollection(ee.Feature(polygon, {feature_id: site}))

            etf_coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year),
                                                          '{}-12-31'.format(year))
            etf_coll = etf_coll.filterBounds(polygon)
            etf_scenes = etf_coll.aggregate_histogram('system:index').getInfo()

            first, bands = True, None
            selectors = [site]

            for img_id in etf_scenes:

                splt = img_id.split('_')
                _name = '_'.join(splt[-3:])

                selectors.append(_name)

                etf_img = ee.Image(os.path.join(ETF, img_id)).rename(_name)
                etf_img = etf_img.divide(10000)

                if mask_type == 'no_mask':
                    etf_img = etf_img.clip(fc.geometry())
                elif mask_type == 'irr':
                    etf_img = etf_img.clip(fc.geometry()).mask(irr_mask)
                elif mask_type == 'inv_irr':
                    etf_img = etf_img.clip(fc.geometry()).mask(irr.gt(0))

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

                if debug:
                    data = etf_img.sample(fc, 30).getInfo()
                    print(data['features'])

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.mean(),
                                       scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=desc,
                fileFormat='CSV',
                selectors=selectors)

            try:
                task.start()
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), desc, '......')
                time.sleep(600)
                task.start()
            exported += 1

    print(f'ETf: Exported {exported}, skipped {skipped} files found in {check_dir}')


def clustered_sample_etf(feature_coll, bucket=None, debug=False, mask_type='irr', check_dir=None,
                         start_yr=2000, end_yr=2024, feature_id='FID'):
    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            etf_img = ee.Image(os.path.join(ETF, img_id)).rename(_name)
            etf_img = etf_img.divide(10000)

            if mask_type == 'no_mask':
                etf_img = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        # TODO extract pixel count to filter data
        data = bands.reduceRegions(collection=feature_coll,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)

        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=desc,
            bucket=bucket,
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=selectors)

        task.start()
        print(desc)


def export_to_cloud(images_txt, bucket, pathrows=None):
    with open(images_txt, 'r') as fp:
        images = [line.strip() for line in fp.readlines()]

    prs, image, metadata = None, None, None
    if pathrows:
        df = pd.read_csv(pathrows)
        prs = [str(pr).rjust(6, '0') for pr in df['PR']]

    try:
        existing_geotiffs = subprocess.check_output(['gsutil', 'ls', f'gs://{bucket}/geotiff/']).decode('utf-8').split('\n')
        existing_geotiffs = [os.path.basename(f) for f in existing_geotiffs if f]
        existing_metadata = subprocess.check_output(['gsutil', 'ls', f'gs://{bucket}/metadata/']).decode('utf-8').split('\n')
        existing_metadata = [os.path.basename(f) for f in existing_metadata if f]
    except subprocess.CalledProcessError:
        existing_geotiffs = []
        existing_metadata = []

    for i in images:
        bname = os.path.basename(i)
        img_pr = bname.split('_')[1]

        if prs and img_pr not in prs:
            continue

        geotiff_filename = os.path.join('geotiff', f'{bname}')
        metadata_filename = os.path.join('metadata', f'{bname}')

        if f'{bname}.tif' in existing_geotiffs and f'{bname}.geojson' in existing_metadata:
            print(f"Skipping {bname} as it already exists in the bucket.")
            continue

        try:
            image = ee.Image(i)
            metadata = image.getInfo()
        except Exception as exc:
            print(bname, exc)
            continue

        if bname not in [f.split('.')[0] + '.tif' for f in existing_geotiffs]:
            task = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=bname,
                bucket=bucket,
                fileNamePrefix=geotiff_filename,
                scale=30,
                maxPixels=1e13,
                crs='EPSG:5070',
                fileFormat='GeoTIFF',
                formatOptions={
                    'noData': 0.0,
                    'cloudOptimized': True,
                }
            )
            try:
                task.start()
                print(f"Exporting GeoTIFF: {geotiff_filename}")
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), bname, '......')
                time.sleep(600)
                task.start()
        else:
            print(f"Skipping GeoTIFF export for {bname} as it already exists.")

        if bname not in [f.split('.')[0] + '.geojson' for f in existing_metadata]:
            metadata_task = ee.batch.Export.table.toCloudStorage(
                collection=ee.FeatureCollection([ee.Feature(None, metadata)]),
                description=os.path.basename(metadata_filename).split('.')[0],
                bucket=bucket,
                fileNamePrefix=metadata_filename,
                fileFormat='GeoJSON'
            )
            try:
                metadata_task.start()
                print(f"Exporting Metadata: {metadata_filename}")
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), bname, '......')
                time.sleep(600)
                metadata_task.start()
        else:
            print(f"Skipping Metadata export for {bname} as it already exists.")


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    is_authorized()

    bucket_ = 'ssebop026'
    txt = '/home/dgketchum/Downloads/ssebop_list.txt'

    prs_ = '/media/research/IrrigationGIS/swim/ssebop/wrs2_flux_volk.csv'

    export_to_cloud(txt, bucket_, prs_)
# ========================= EOF ====================================================================
