import os
import sys

import ee
import geopandas as gpd

from ee_api.ee_utils import landsat_masked, is_authorized

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

EC_POINTS = 'users/dgketchum/flux_ET_dataset/stations'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def get_flynn():
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def export_ndvi(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr'):
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


def flux_tower_ndvi(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None):
    df = gpd.read_file(shapefile)

    assert df.crs.srs == 'EPSG:5071'

    df = df.to_crs(epsg=4326)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for fid, row in df.iterrows():

        for year in range(1987, 2022):

            state = row['field_3']
            if state not in STATES:
                continue

            site = row['field_1']

            desc = 'ndvi_{}_{}_{}'.format(site, year, mask_type)
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
                    data = nd_img.sample(point, 30).getInfo()
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


if __name__ == '__main__':
    is_authorized()
    bucket_ = 'wudr'

    shp = '/media/research/IrrigationGIS/et-demands/examples/flux/gis/flux_fields_sample.shp'
    for mask in ['inv_irr', 'irr']:
        hk = '/media/research/IrrigationGIS/et-demands/examples/flux/landsat/extracts/ndvi/{}'.format(mask)
        flux_tower_ndvi(shp, bucket_, debug=False, mask_type=mask)
        pass

# ========================= EOF ====================================================================
