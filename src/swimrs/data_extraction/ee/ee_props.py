import os
import pandas as pd

import ee
from swimrs.data_extraction.ee.ee_utils import get_lanid, as_ee_feature_collection


IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

# See https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx
# to check soil parameters



STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'


def get_cdl(fields, desc, selector='FID', dest='drive', bucket=None, drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export per-feature CDL crop class mode by year to GCS.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - selector: property to include as ID in selectors (default 'FID').

    Side Effects
    - Starts ee.batch table export of yearly modes for 2008–2022 to `wudr`.
    """
    plots = as_ee_feature_collection(fields, feature_id=selector)
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
    if dest == 'bucket':
        if not bucket:
            raise ValueError('CDL export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            modes,
            description=out_,
            bucket=bucket,
            fileNamePrefix=f'{file_prefix}/properties/{out_}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    elif dest == 'drive':
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=modes,
            description=out_,
            folder=drive_folder_name,
            fileNamePrefix=f'properties/{out_}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(desc)


def get_irrigation(fields, desc, debug=False, selector='FID', select=None, lanid=False, dest='drive', bucket=None, drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export annual irrigation fraction per feature using IrrMapper (and LANID).

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.
    - lanid: bool; if True, mosaics LANID east of WEST/EAST boundary for years.

    Side Effects
    - Starts ee.batch table export to `wudr` with mean of yearly `irr_<year>`.
    """
    east, west = None, None
    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

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

    if dest == 'bucket':
        if not bucket:
            raise ValueError('Irrigation export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            means,
            description=desc,
            bucket=bucket,
            fileNamePrefix=f'{file_prefix}/properties/{desc}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    elif dest == 'drive':
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=means,
            description=desc,
            folder=drive_folder_name,
            fileNamePrefix=f'properties/{desc}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(desc)


def get_ssurgo(fields, desc, debug=False, selector='FID', select=None, dest='drive', bucket=None, drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export SSURGO-derived soil attributes averaged per feature.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.

    Side Effects
    - Starts ee.batch table export (columns: awc, ksat, clay, sand) to `wudr`.
    """
    # OpenET AWC is in cm/cm
    awc_asset = 'projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite'
    # OpenET KSAT is in micrometers/sec
    ksat_asset = 'projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite'
    clay_asset = 'projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite'
    sand_asset = 'projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite'

    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

    ksat_ = ee.Image(ksat_asset).select('b1').rename('ksat')
    awc_ = ee.Image(awc_asset).select('b1').rename('awc')
    clay_ = ee.Image(clay_asset).select('b1').rename('clay')
    sand_ = ee.Image(sand_asset).select('b1').rename('sand')

    img = ksat_.addBands([awc_, clay_, sand_])

    _selectors = [selector] + ['awc', 'ksat', 'clay', 'sand']

    means = img.reduceRegions(collection=plots,
                              reducer=ee.Reducer.mean(),
                              scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 1789).getInfo()

    if dest == 'bucket':
        if not bucket:
            raise ValueError('SSURGO export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            means,
            description=desc,
            bucket=bucket,
            fileNamePrefix=f'{file_prefix}/properties/{desc}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    elif dest == 'drive':
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=means,
            description=desc,
            folder=drive_folder_name,
            fileNamePrefix=f'properties/{desc}',
            fileFormat='CSV',
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(desc)


def get_hwsd(fields, desc, debug=False, selector='FID', out_fmt='CSV', local_file=None, dest='drive', bucket=None, drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export or save HWSD v2 soil property (AWC) per feature.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - out_fmt: 'CSV' or other formats supported by EE table export.
    - local_file: if provided, writes a local CSV instead of GCS export.
    """
    plots = as_ee_feature_collection(fields, feature_id=selector)

    stype = ee.Image('projects/sat-io/open-datasets/FAO/HWSD_V2_SMU').select('AWC').rename('awc')

    modes = stype.reduceRegions(collection=plots,
                                reducer=ee.Reducer.mode(),
                                scale=30)

    # single value reduction results in stat name: 'mode' instead of image name
    _selectors = [selector, 'mode']

    if debug:
        debug = modes.filterMetadata('FID', 'equals', 'US-CRT').getInfo()

    if local_file:
        modes = modes.getInfo()
        df = pd.DataFrame([v['properties'] for v in modes['features']]).rename(columns={'mode': 'awc'})
        df.to_csv(local_file)

    else:

        if dest == 'bucket':
            if not bucket:
                raise ValueError('HWSD export dest="bucket" requires a bucket name/url')
            export_kwargs = dict(description=desc,
                                 bucket=bucket,
                                 fileNamePrefix=f'{file_prefix}/properties/{desc}',
                                 fileFormat=out_fmt)
            if out_fmt == 'CSV':
                export_kwargs.update({'selectors': _selectors})
            task = ee.batch.Export.table.toCloudStorage(modes, **export_kwargs)
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
            export_kwargs = dict(description=desc,
                                 folder=drive_folder_name,
                                 fileNamePrefix=f'properties/{desc}',
                                 fileFormat=out_fmt)
            if out_fmt == 'CSV':
                export_kwargs.update({'selectors': _selectors})
            task = ee.batch.Export.table.toDrive(collection=modes, **export_kwargs)
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')
        task.start()
        print(desc)


def get_landcover(fields, desc, debug=False, selector='FID', select=None, out_fmt='CSV', local_file=None, dest='drive', bucket=None, drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export dominant landcover from MODIS and FROM-GLC10 per feature.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.
    - out_fmt: 'CSV' or other formats supported by EE table export.
    - local_file: if provided, writes a local CSV instead of GCS export.
    """
    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

    vtype = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1').first().rename('modis_lc')
    vtype = vtype.addBands([ee.ImageCollection('projects/sat-io/open-datasets/FROM-GLC10')
                           .mosaic().rename('glc10_lc')])

    modes = vtype.reduceRegions(collection=plots,
                                reducer=ee.Reducer.mode(),
                                scale=30)
    _selectors = [selector, 'modis_lc', 'glc10_lc']

    if debug:
        debug = modes.filterMetadata('FID', 'equals', 'US-CRT').getInfo()

    if local_file:
        modes = modes.getInfo()
        df = pd.DataFrame([v['properties'] for v in modes['features']])[_selectors]
        df.to_csv(local_file)

    else:

        if dest == 'bucket':
            if not bucket:
                raise ValueError('Landcover export dest="bucket" requires a bucket name/url')
            export_kwargs = dict(description=desc,
                                 bucket=bucket,
                                 fileNamePrefix=f'{file_prefix}/properties/{desc}',
                                 fileFormat=out_fmt)
            if out_fmt == 'CSV':
                export_kwargs.update({'selectors': _selectors})
            task = ee.batch.Export.table.toCloudStorage(modes, **export_kwargs)
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
            export_kwargs = dict(description=desc,
                                 folder=drive_folder_name,
                                 fileNamePrefix=f'properties/{desc}',
                                 fileFormat=out_fmt)
            if out_fmt == 'CSV':
                export_kwargs.update({'selectors': _selectors})
            task = ee.batch.Export.table.toDrive(collection=modes, **export_kwargs)
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')
        task.start()
        print(desc)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
