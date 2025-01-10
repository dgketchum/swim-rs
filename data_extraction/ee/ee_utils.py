import os
from datetime import datetime

import ee

CRS_TRANSFORM = [0.041666666666666664,
                 0, -124.78749996666667,
                 0, -0.041666666666666664,
                 49.42083333333334]


def add_doy(image):
    """ Add day-of-year image """
    mask = ee.Date(image.get('system:time_start'))
    day = ee.Image.constant(image.date().getRelative('day', 'year')).clip(image.geometry())
    i = image.addBands(day.rename('DOY')).int().updateMask(mask)
    return i


def get_world_climate(proj, months, param='prec'):
    if months[0] > months[1]:
        months = [x for x in range(months[0], 13)] + [x for x in range(1, months[1] + 1)]
    else:
        months = [x for x in range(months[0], months[1] + 1)]
    months = [str(x).zfill(2) for x in months]
    assert param in ['tavg', 'prec']
    combinations = [(m, param) for m in months]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).
         select(param).resample('bilinear').reproject(crs=proj['crs'], scale=30) for m, p in combinations]
    i = ee.ImageCollection(l)
    if param == 'prec':
        i = i.sum()
    else:
        i = i.mean()
    return i


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


def long_term_ndvi(roi, coll=False):
    start = '1987-07-01'
    end_date = '2021-12-31'

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l9_coll = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    landsat = ee.ImageCollection(l5_coll.merge(l7_coll).merge(l8_coll).merge(l9_coll))

    if coll:
        landsat = landsat.map(lambda x: x.normalizedDifference(['B5', 'B4']))
        return landsat

    nd_max = []
    for yr in range(1987, 2023):
        s, e = '{}-01-01'.format(yr), '{}-12-31'.format(yr)
        ndvi_mx = ee.Image(landsat.filterDate(s, e).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_max'.format(yr))
        nd_max.append(ndvi_mx)

    coll = ee.ImageCollection(nd_max)
    mean_, max_, std_ = coll.mean().rename('nd_lt_mn'), coll.max().rename('nd_lt_mx'), \
        coll.reduce(ee.Reducer.stdDev()).rename('nd_lt_std')

    return mean_, max_, std_, coll


def landsat_composites(year, start, end, roi, append_name, composites_only=False):
    start_year = datetime.strptime(start, '%Y-%m-%d').year
    if start_year != year:
        year = start_year

    def evi_(x):
        return x.expression('2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))', {'NIR': x.select('B5'),
                                                                                   'RED': x.select('B4'),
                                                                                   'BLUE': x.select('B2')})

    def gi_(x):
        return x.expression('NIR / GREEN', {'NIR': x.select('B5'),
                                            'GREEN': x.select('B3')})

    bands_means = None
    lsSR_masked = landsat_masked(year, roi)
    if not composites_only:
        bands_means = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'],
                               ['B2_{}'.format(append_name),
                                'B3_{}'.format(append_name),
                                'B4_{}'.format(append_name),
                                'B5_{}'.format(append_name),
                                'B6_{}'.format(append_name),
                                'B7_{}'.format(append_name),
                                'B10_{}'.format(append_name)]
                               )).mean())

    if append_name in ['m2', 'm1', 'gs']:
        ndvi_mx = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_max_{}'.format(append_name))

        ndvi_mean = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).mean()).rename('nd_mean_{}'.format(append_name))

        ndvi = ndvi_mx.addBands([ndvi_mean])

    else:
        ndvi = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_{}'.format(append_name))

    ndwi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.normalizedDifference(['B5', 'B6'])).max()).rename('nw_{}'.format(append_name))
    evi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: evi_(x)).max()).rename('evi_{}'.format(append_name))
    gi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: gi_(x)).max()).rename('gi_{}'.format(append_name))

    if composites_only:
        bands = ndvi.addBands([ndwi, evi, gi])
    else:
        bands = bands_means.addBands([ndvi, ndwi, evi, gi])

    return bands


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
