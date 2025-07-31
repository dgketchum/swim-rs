import os
from datetime import datetime

import ee

CRS_TRANSFORM = [0.041666666666666664,
                 0, -124.78749996666667,
                 0, -0.041666666666666664,
                 49.42083333333334]

import ee


def sentinel2_sr(input_img):
    optical_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    scl_band = 'SCL'
    all_bands = optical_bands + [scl_band]

    mult = [0.0001] * len(optical_bands) + [1]
    prep_image = input_img.select(all_bands).multiply(mult)

    def _cloud_mask(i):
        scl = i.select('SCL')
        cloud_mask_values = [3, 8, 9, 10]
        mask = scl.remap(cloud_mask_values, [0] * len(cloud_mask_values), 1)
        mask = mask.rename(['cloud_mask'])
        return mask

    mask = _cloud_mask(prep_image)

    image = prep_image.select(optical_bands).updateMask(mask)
    image = image.copyProperties(input_img, ['system:time_start'])

    return image


def sentinel2_masked(yr, roi):
    start = f'{yr}-01-01'
    end_date = f'{yr + 1}-01-01'

    s2_coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start, end_date) \
        .map(sentinel2_sr)

    return s2_coll


def landsat_c2_sr(input_img):
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start'])

    return image


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l9_coll = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l9_coll).merge(l5_coll).merge(l4_coll))

    return lsSR_masked


def export_openet_correction_surfaces(local_check):
    is_authorized()

    for etref in ['etr', 'eto']:
        id_ = 'projects/openet/reference_et/gridmet/ratios/v1/monthly/{}'.format(etref)
        c = ee.ImageCollection(id_)
        scenes = c.aggregate_histogram('system:index').getInfo()
        for k in list(scenes.keys()):
            month_number = datetime.strptime(k, '%b').month
            if local_check:
                f = 'gridmet_corrected_{}'.format(etref)
                local_file = os.path.join(local_check, '{}_{}.tif'.format(f, month_number))
                if os.path.exists(local_file):
                    continue
            desc = 'gridmet_corrected_{}_{}'.format(etref, month_number)
            i = ee.Image(os.path.join(id_, k))
            task = ee.batch.Export.image.toCloudStorage(
                i,
                description=desc,
                bucket='wudr',
                dimensions='1386x585',
                fileNamePrefix=desc,
                crsTransform=CRS_TRANSFORM,
                crs='EPSG:4326')
            task.start()
            print(desc)


def get_lanid():
    first_image = ee.Image('users/xyhuwmir4/LANID_postCls/LANID_v2')
    second_image = ee.Image('users/xyhuwmir/LANID/update/LANID2018-2020')

    bands = None

    for yr in range(1987, 2018):
        if yr < 1997:
            year = 1997
        else:
            year = yr
        band_name = f'irr_{yr}'
        image = ee.Image(first_image.select([f'irMap{str(year)[-2:]}'])).rename([band_name]).int().unmask(0)
        if bands is None:
            bands = ee.Image(image)
        else:
            bands = bands.addBands([image])

    for yr in range(2018, 2025):
        if yr > 2020:
            year = 2020
        else:
            year = yr
        band_name = f'irr_{yr}'
        image = ee.Image(second_image.select([f'irMap{str(year)[-2:]}'])).rename([band_name]).int().unmask(0)
        bands = bands.addBands([image])

    return bands


def is_authorized():
    try:
        ee.Initialize(project='ee-dgketchum')
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/et-demands/gridmet/gridmet_corrected/correction_surfaces_wgs'
    export_openet_correction_surfaces(d)
# ========================= EOF ====================================================================
