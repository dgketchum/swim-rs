import os
import sys

import ee
import geopandas as gpd

from data_extraction.ee.ee_utils import is_authorized

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'

EC_POINTS = 'users/dgketchum/flux_ET_dataset/stations'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


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


def flux_tower_etf(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None):
    df = gpd.read_file(shapefile)

    assert df.crs.srs == 'EPSG:5071'

    df = df.to_crs(epsg=4326)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for fid, row in df.iterrows():

        for year in range(2000, 2015):

            state = row['field_3']
            if state not in STATES:
                continue

            site = row['field_1']

            if site not in ['US-Mj1', 'US-Mj2']:
                continue

            desc = 'etf_{}_{}_{}'.format(site, year, mask_type)
            if check_dir:
                f = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(f):
                    print(desc, 'exists, skipping')
                    continue

            irr = irr_coll.filterDate('{}-01-01'.format(year),
                                      '{}-12-31'.format(year)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            point = ee.Geometry.Point([row['field_8'], row['field_7']])
            geo = point.buffer(150.)
            fc = ee.FeatureCollection(ee.Feature(geo, {'field_1': site}))

            etf_coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year),
                                                          '{}-12-31'.format(year))
            etf_coll = etf_coll.filterBounds(geo)
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

            task.start()
            print(desc)


def clustered_field_etf(feature_coll, bucket=None, debug=False, mask_type='irr', check_dir=None):

    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for year in range(1987, 2022):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(year, mask_type)

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = ['FID']

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


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    is_authorized()
    bucket_ = 'wudr'
    fields = 'users/dgketchum/fields/tongue_annex_20OCT2023'
    for mask in ['inv_irr', 'irr']:
        chk = os.path.join(d, 'examples/tongue/landsat/extracts/etf/{}'.format(mask))
        clustered_field_etf(fields, bucket_, debug=False, mask_type=mask, check_dir=None)

# ========================= EOF ====================================================================
