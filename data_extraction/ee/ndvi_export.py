import os
import sys
import time
from tqdm import tqdm

import ee
import geopandas as gpd

from data_extraction.ee.ee_utils import get_lanid
from data_extraction.ee.ee_utils import landsat_masked, is_authorized

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

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


def export_ndvi_images(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr'):
    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year),
                              '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    coll = landsat_masked(year, feature_coll).map(lambda x: x.normalizedDifference(['B5', 'B4']))
    scenes = coll.aggregate_histogram('system:index').getInfo()

    for img_id in scenes:

        splt = img_id.split('_')
        _name = '_'.join(splt[-3:])

        # if _name != 'LT05_035028_20080724':
        #     continue

        img = coll.filterMetadata('system:index', 'equals', img_id).first()

        if mask_type == 'no_mask':
            img = img.clip(feature_coll.geometry()).multiply(1000).int()
        elif mask_type == 'irr':
            img = img.clip(feature_coll.geometry()).mask(irr_mask).multiply(1000).int()
        elif mask_type == 'inv_irr':
            img = img.clip(feature_coll.geometry()).mask(irr.gt(0)).multiply(1000).int()

        if debug:
            point = ee.Geometry.Point([-105.793, 46.1684])
            data = img.sample(point, 30).getInfo()
            print(data['features'])

        task = ee.batch.Export.image.toCloudStorage(
            img,
            description='NDVI_{}_{}'.format(mask_type, _name),
            bucket=bucket,
            region=feature_coll.geometry(),
            crs='EPSG:5070',
            scale=30)

        task.start()
        print(_name)


def sparse_sample_ndvi(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None, grid_spec=None,
                       feature_id='FID', select=None, start_yr=2000, end_yr=2024, state_col='field_3'):
    df = gpd.read_file(shapefile)
    df.index = df[feature_id]

    assert df.crs.srs == 'EPSG:5071'

    df = df.to_crs(epsg=4326)

    s, e = '1987-01-01', '2024-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    lanid = get_lanid()
    east = ee.FeatureCollection(EAST_STATES)

    skipped, exported = 0, 0

    for fid, row in tqdm(df.iterrows(), desc='Extracting NDVI', total=df.shape[0]):

        for year in range(start_yr, end_yr + 1):

            if select is not None and fid not in select:
                continue

            site = row[feature_id]
            grid_sz = row['grid_size']

            if grid_spec is not None and grid_sz != grid_spec:
                continue

            desc = 'ndvi_{}_p{}_{}_{}'.format(site, grid_sz, mask_type, year)
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

            coll = landsat_masked(year, fc).map(lambda x: x.normalizedDifference(['B5', 'B4']))
            ndvi_scenes = coll.aggregate_histogram('system:index').getInfo()

            first, bands = True, None
            selectors = [site]
            for img_id in ndvi_scenes:

                splt = img_id.split('_')
                _name = '_'.join(splt[-3:])

                selectors.append(_name)

                nd_img = coll.filterMetadata('system:index', 'equals', img_id).first().rename(_name)

                if mask_type == 'no_mask':
                    nd_img = nd_img.clip(fc.geometry())
                elif mask_type == 'irr':
                    nd_img = nd_img.clip(fc.geometry()).mask(irr_mask)
                elif mask_type == 'inv_irr':
                    nd_img = nd_img.clip(fc.geometry()).mask(irr.gt(0))

                if first:
                    bands = nd_img
                    first = False
                else:
                    bands = bands.addBands([nd_img])

                if debug:
                    data = nd_img.sample(fc, 30).getInfo()
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

    print(f'NDVI: Exported {exported}, skipped {skipped} files found in {check_dir}')


def clustered_sample_ndvi(feature_coll, bucket=None, debug=False, mask_type='irr', check_dir=None,
                          start_yr=2004, end_yr=2023, feature_id='FID'):
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

        first, bands = True, None
        selectors = [feature_id]

        desc = 'ndvi_{}_{}'.format(mask_type, year)

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        coll = landsat_masked(year, feature_coll).map(lambda x: x.normalizedDifference(['B5', 'B4']))
        ndvi_scenes = coll.aggregate_histogram('system:index').getInfo()

        for img_id in ndvi_scenes:

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            # if splt[-1] not in ['20000514', '20000515']:
            #     continue

            nd_img = coll.filterMetadata('system:index', 'equals', img_id).first().rename(_name)

            if mask_type == 'no_mask':
                nd_img = nd_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                nd_img = nd_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                nd_img = nd_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = nd_img
                first = False
            else:
                bands = bands.addBands([nd_img])

        if debug:
            fc = ee.FeatureCollection([feature_coll.filterMetadata(feature_id, 'equals', 2).first()])
            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.mean(),
                                       scale=30).getInfo()
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


if __name__ == '__main__':

    is_authorized()

    bucket = 'wudr'

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    shapefile_path = os.path.join(root, 'footprints', 'flux_static_footprints.shp')

    data = os.path.join(root,  'tutorials', '4_Flux_Network', 'data')
    landsat_dst = os.path.join(data, 'landsat')

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')

    fdf = gpd.read_file(fields_gridmet)
    target_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    state_idx = [i for i, r in fdf.iterrows() if r['field_3'] in target_states]
    fdf = fdf.loc[state_idx]
    sites_ = list(set(fdf['field_1'].to_list()))
    sites_.sort()

    # Volk static footprints
    FEATURE_ID = 'site_id'
    state_col = 'state'

    from etf_export import sparse_sample_etf

    for src in ['ndvi', 'etf']:
        for mask in ['irr', 'inv_irr']:

            if src == 'ndvi':
                print(src, mask)
                dst = os.path.join(landsat_dst, 'extracts', src, mask)

                sparse_sample_ndvi(shapefile_path, bucket=bucket, debug=False, grid_spec=3,
                                   mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                   state_col=state_col, select=None)

            if src == 'etf':
                for model in ['openet', 'eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop', 'disalexi']:
                    dst = os.path.join(landsat_dst, 'extracts', f'{model}_{src}', mask)

                    print(src, mask, model)

                    sparse_sample_etf(shapefile_path, bucket=bucket, debug=False, grid_spec=3,
                                      mask_type=mask, check_dir=dst, start_yr=2016, end_yr=2024, feature_id=FEATURE_ID,
                                      state_col=state_col, select=None, model=model)

# ========================= EOF =======================================================================================
