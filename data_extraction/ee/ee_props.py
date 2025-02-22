import ee
from data_extraction.ee.ee_utils import get_lanid

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

# See https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx
# to check soil parameters

# OpenET AWC is in cm/cm
AWC = 'projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite'
# OpenET KSAT is in micrometers/sec
KSAT = 'projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite'
CLAY = 'projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite'
SAND = 'projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'


def get_cdl(fields, desc, selector='FID'):
    plots = ee.FeatureCollection(fields)
    crops, first = None, True
    cdl_years = [x for x in range(2008, 2023)]

    _selectors = [selector]

    for y in cdl_years:

        image = ee.Image('USDA/NASS/CDL/{}'.format(y))
        crop = image.select('cropland')
        _name = 'crop_{}'.format(y)
        _selectors.append(_name)
        if first:
            crops = crop.rename(_name)
            first = False
        else:
            crops = crops.addBands(crop.rename(_name))

    modes = crops.reduceRegions(collection=plots,
                                reducer=ee.Reducer.mode(),
                                scale=30)

    out_ = '{}'.format(desc)
    task = ee.batch.Export.table.toCloudStorage(
        modes,
        description=out_,
        bucket='wudr',
        fileNamePrefix=out_,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()


def get_irrigation(fields, desc, debug=False, selector='FID', lanid=False):
    """"""
    east, west = None, None
    plots = ee.FeatureCollection(fields)

    irr_coll = ee.ImageCollection(IRR)
    if lanid:
        lanid = get_lanid()
        west = ee.FeatureCollection(WEST_STATES)
        east = ee.FeatureCollection(EAST_STATES)

    _selectors = [selector, 'LAT', 'LON']
    first = True

    area, irr_img = ee.Image.pixelArea(), None

    for year in range(1987, 2025):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr = irr.lt(1).rename('irr_{}'.format(year)).int()

        if lanid:
            irr = irr.clip(west)
            lanid_yr = lanid.select(f'irr_{year}').clip(east)
            irr = ee.ImageCollection([irr, lanid_yr]).mosaic()

        _name = 'irr_{}'.format(year)
        _selectors.append(_name)

        if first:
            irr_img = irr.rename(_name)
            first = False
        else:
            irr_img = irr_img.addBands(irr.rename(_name))

    means = irr_img.reduceRegions(collection=plots,
                                  reducer=ee.Reducer.mean(),
                                  scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 'US-FPe').getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()


def get_ssurgo(fields, desc, debug=False, selector='FID'):
    plots = ee.FeatureCollection(fields)

    ksat = ee.Image(KSAT).select('b1').rename('ksat')
    awc = ee.Image(AWC).select('b1').rename('awc')
    clay = ee.Image(CLAY).select('b1').rename('clay')
    sand = ee.Image(SAND).select('b1').rename('sand')

    img = ksat.addBands([awc, clay, sand])

    _selectors = [selector, 'LAT', 'LON'] + ['awc', 'ksat', 'clay', 'sand']

    means = img.reduceRegions(collection=plots,
                              reducer=ee.Reducer.mean(),
                              scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 1789).getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()
    print(desc)


def get_landfire(fields, desc, debug=False, selector='FID'):
    plots = ee.FeatureCollection(fields)

    vtype = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1').first().rename('veg_type')

    _selectors = [selector, 'LAT', 'LON'] + ['veg_type']

    means = vtype.reduceRegions(collection=plots,
                                reducer=ee.Reducer.mode(),
                                scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 'US-CRT').getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()
    print(desc)


if __name__ == '__main__':
    ee.Initialize(project='ee-dgketchum')

    # project = 'tongue'
    # index_col = 'FID'
    # fields_ = 'users/dgketchum/fields/tongue_annex_20OCT2023'

    project = 'tutorial'
    index_col = 'field_1'
    fields_ = 'users/dgketchum/fields/flux'

    description = '{}_cdl'.format(project)
    # get_cdl(fields_, description, selector=index_col)

    description = '{}_irr'.format(project)
    # get_irrigation(fields_, description, debug=True, selector=index_col, lanid=True)

    description = '{}_ssurgo'.format(project)
    # get_ssurgo(fields_, description, debug=False, selector=index_col)

    description = '{}_landfire'.format(project)
    get_landfire(fields_, description, debug=True, selector=index_col)

# ========================= EOF ====================================================================
