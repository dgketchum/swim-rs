"""ETf (ET fraction) Earth Engine exporters.

Functions
- sparse_sample_etf: per-field/year export; iterates rows of a local shapefile and applies west/east masking
  with IrrMapper (west) and LANID (east). Best for widely dispersed features.
- clustered_sample_etf: per-year export for a set of clustered features; accepts a shapefile path,
  EE asset ID, or ee.FeatureCollection, supports `select` to restrict by feature_id, and `state_col` for
  consistent west/east masking (same logic as sparse). Used in the Boulder notebook; the sparse
  option is also demonstrated in examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb.
- export_etf_images: exports clipped ETf images for inspection.

Note
- All EE FeatureCollection/shapefile inputs are normalized via as_ee_feature_collection in ee_utils.
"""

import os
import subprocess
import sys
import time

import ee
import geopandas as gpd
import pandas as pd
import utm
from tqdm import tqdm

from swimrs.data_extraction.ee.ee_utils import get_lanid, as_ee_feature_collection

sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'

EC_POINTS = 'users/dgketchum/fields/flux'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'

def get_utm_epsg(latitude, longitude):
    """Return UTM EPSG code and zone string for a lat/lon.

    Parameters
    - latitude, longitude: float degrees.

    Returns
    - (epsg_code: int, zone_str: str) like (32612, '12N').
    """
    _, _, zone_number, zone_letter = utm.from_latlon(latitude, longitude)

    if zone_letter >= 'N':
        epsg_code = 32600 + zone_number
        zone_hemisphere = f"{zone_number}N"
    else:
        epsg_code = 32700 + zone_number
        zone_hemisphere = f"{zone_number}S"

    return epsg_code, zone_hemisphere


def get_flynn():
    """Return a sample polygon FeatureCollection for quick tests."""
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def export_etf_images(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr', dest='drive', drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export per-scene ET fraction images masked by irrigation to GCS.

    Parameters
    - feature_coll: ee.FeatureCollection region to clip.
    - year: int year.
    - bucket: str GCS bucket.
    - debug: bool; if True, prints sampled pixel values.
    - mask_type: {'irr','inv_irr','no_mask'}.

    Side Effects
    - Starts ee.batch image exports to `bucket` for each scene.
    """
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

        desc = 'ETF_{}_{}'.format(mask_type, _name)
        if dest == 'bucket':
            if not bucket:
                raise ValueError('ETF image export dest="bucket" requires a bucket name/url')
            task = ee.batch.Export.image.toCloudStorage(
                image=img,
                description=desc,
                bucket=bucket,
                fileNamePrefix=f'{file_prefix}/remote_sensing/landsat/extracts/etf/images/{desc}',
                region=feature_coll.geometry(),
                crs='EPSG:5070',
                scale=30,
            )
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_etf_images" if drive_categorize else drive_folder
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=desc,
                folder=drive_folder_name,
                fileNamePrefix=f'etf/images/{desc}',
                region=feature_coll.geometry(),
                crs='EPSG:5070',
                scale=30,
            )
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')

        task.start()
        print(_name)


def sparse_sample_etf(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None,
                      feature_id='FID', select=None, start_yr=2000, end_yr=2024, state_col='field_3',
                      model='ssebop', usgs_nhm=False, source=None, scale=None, dest='drive', drive_folder='swim',
                      file_prefix='swim', drive_categorize=False):
    """Export per-field ET fraction (one CSV per field-year) from OpenET/USGS sources.

    Parameters
    - shapefile: polygon features (expected 5071 or compatible CRS; reprojected to 4326).
    - bucket: str GCS bucket.
    - debug: bool; if True, prints sampled values.
    - mask_type: {'irr','inv_irr','no_mask'}.
    - check_dir: local dir to skip if CSV exists.
    - feature_id: feature identifier column.
    - select: optional subset of feature IDs.
    - start_yr, end_yr: int year range.
    - state_col: state code column for mask fallback.
    - model: one of OpenET members or 'openet'/'disalexi'/'ssebop'.
    - usgs_nhm: bool; if True and model='ssebop', use USGS NHM asset.
    - source: override image collection path.
    - scale: optional divisor for value scaling when `source` provided.

    Notes
    - This is the sparse extractor, iterating features row-by-row from a local shapefile and applying
      west/east irrigation masks (IrrMapper vs LANID) based on a state column. For clustered fields,
      prefer clustered_sample_etf, which offers `select` and direct FeatureCollection/asset inputs.
      The Boulder Step 2 notebook shows both approaches (see examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb).

    Side Effects
    - Starts ee.batch table exports to the requested destination.
    """

    df = gpd.read_file(shapefile)
    df.index = df[feature_id]

    try:
        assert df.crs.srs == 'EPSG:5071'
    except AssertionError:
        assert df.crs.name == 'Europe_Albers_Equal_Area_Conic'

    df = df.to_crs(epsg=4326)

    # Filter to selected fields before iterating
    total_fields = len(df)
    if select is not None:
        df = df[df.index.isin(select)]
    print(f'Selected {len(df)} of {total_fields} fields')

    s, e = '1987-01-01', '2024-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    lanid = get_lanid()
    east = ee.FeatureCollection(EAST_STATES)

    skipped, exported = 0, 0

    members = ['eemetric',
               'geesebal',
               'ptjpl',
               'sims',
               'ssebop']

    if source:
        pass

    elif model == 'ssebop' and usgs_nhm:
        source = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'

    elif source is None and model in members:
        source = f'projects/openet/assets/{model}/conus/gridmet/landsat/c02'

    elif model == 'disalexi':
        source = 'projects/openet/assets/disalexi/landsat/c02'

    elif model == 'openet':
        source = 'projects/openet/assets/ensemble/conus/gridmet/landsat/c02'

    else:
        raise ValueError('Invalid model name')

    for fid, row in tqdm(df.iterrows(), desc='Processing Fields', total=df.shape[0]):

        for year in range(start_yr, end_yr + 1):

            site = row[feature_id]

            desc = '{}_etf_{}_{}_{}'.format(model, site, mask_type, year)

            if check_dir:
                f = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(f):
                    skipped += 1
                    continue

            if mask_type in ['irr', 'inv_irr']:
                if row[state_col] in STATES:
                    irr = irr_coll.filterDate('{}-01-01'.format(year),
                                              '{}-12-31'.format(year)).select('classification').mosaic()
                    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

                else:
                    irr_mask = lanid.select(f'irr_{year}').clip(east)
                    irr = ee.Image(1).subtract(irr_mask)
            else:
                irr, irr_mask = None, None

            polygon = ee.Geometry.Polygon([[c[0], c[1]] for c in row['geometry'].exterior.coords])
            fc = ee.FeatureCollection(ee.Feature(polygon, {feature_id: site}))

            etf_coll = ee.ImageCollection(source).filterDate('{}-01-01'.format(year),
                                                             '{}-12-31'.format(year))
            etf_coll = etf_coll.filterBounds(polygon)
            etf_scenes = etf_coll.aggregate_histogram('system:index').getInfo()

            first, bands = True, None
            selectors = [site]

            for img_id in etf_scenes:

                splt = img_id.split('_')
                _name = '_'.join(splt[-3:])
                _dt = splt[-1]

                selectors.append(_name)

                etf_img = ee.Image(os.path.join(source, img_id))

                if source is not None and scale is not None:
                    etf_img = etf_img.select('et_fraction')
                    etf_img = etf_img.divide(scale)
                elif model == 'openet':
                    etf_img = etf_img.select('et_ensemble_mad')
                    etf_img = etf_img.divide(10000)
                elif model in ['sims', 'eemetric', 'ssebop']:
                    etf_img = etf_img.select('et_fraction')
                    etf_img = etf_img.divide(10000)
                elif model in ['geesebal', 'ptjpl', 'disalexi']:
                    et_img = etf_img.select('et')
                    et_img = et_img.divide(1000)
                    refet = ee.Image(f'projects/openet/assets/reference_et/conus/gridmet/daily/v1/{_dt}').select('eto')
                    etf_img = et_img.divide(refet)

                etf_img = etf_img.rename(_name)

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

            if dest == 'bucket':
                if not bucket:
                    raise ValueError('ETF export dest="bucket" requires a bucket name/url')
                task = ee.batch.Export.table.toCloudStorage(
                    data,
                    description=desc,
                    bucket=bucket,
                    fileNamePrefix=f'{file_prefix}/remote_sensing/landsat/extracts/{model}_etf/{mask_type}/{desc}',
                    fileFormat='CSV',
                    selectors=selectors
                )
            elif dest == 'drive':
                drive_folder_name = f"{drive_folder}_etf" if drive_categorize else drive_folder
                task = ee.batch.Export.table.toDrive(
                    collection=data,
                    description=desc,
                    folder=drive_folder_name,
                    fileNamePrefix=f'etf/{model}/{desc}',
                    fileFormat='CSV',
                    selectors=selectors
                )
            else:
                raise ValueError('dest must be one of {"drive","bucket"}')

            try:
                task.start()
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), desc, '......')
                time.sleep(600)
                task.start()
            exported += 1

    print(f'ETf: Exported {exported}, skipped {skipped} files found in {check_dir}')


def clustered_sample_etf(feature_coll,
                         bucket=None,
                         debug=False,
                         mask_type='irr',
                         check_dir=None,
                         start_yr=2000,
                         end_yr=2024,
                         feature_id='FID',
                         select=None,
                         state_col='STATE',
                         model='ssebop',
                         usgs_nhm=False,
                         source=None,
                         scale=None,
                         dest='drive',
                         drive_folder='swim',
                         file_prefix='swim',
                         drive_categorize=False):
    """Export ET fraction for all features in a collection per year.

    Parameters
    - feature_coll: fields as ee.FeatureCollection, EE asset ID, or local shapefile path (normalized via ee_utils.as_ee_feature_collection).
    - bucket: str GCS bucket.
    - debug: bool; prints samples if True.
    - mask_type: {'irr','inv_irr','no_mask'}.
    - check_dir: local check directory to skip existing outputs.
    - start_yr, end_yr: int range.
    - feature_id: property name to include in CSV selectors.
    - select: optional list[str] of feature_id values to include.
    - state_col: column/property with two-letter state code. All features must be either in WEST_STATES or all outside;
      mixed collections raise an error. West uses IrrMapper; east uses LANID (matches sparse_sample_etf).
    - model: OpenET model key ('ssebop','eemetric','sims','geesebal','ptjpl','disalexi','openet').
    - usgs_nhm: if True and model='ssebop', uses USGS NHM SSEBop asset path.
    - source: explicit override of image collection path.
    - scale: optional divisor for value scaling when `source` provided.

    Notes
    - For widely dispersed fields use sparse_sample_etf; see examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb for a comparison.

    Side Effects
    - Starts ee.batch table export per year to Drive or Cloud Storage based on `dest`.
    """
    # Accept asset id, ee.FeatureCollection, or a local shapefile path; keep state property
    feature_coll = as_ee_feature_collection(feature_coll, feature_id=feature_id, keep_props=[state_col] if state_col else [])

    # Optionally filter to a subset of features by ID
    if select is not None:
        feature_coll = feature_coll.filter(ee.Filter.inList(feature_id, select))

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    # Determine region: all-west or all-east based on state_col
    if state_col is None:
        raise ValueError('state_col must be provided for clustered ETf extraction')
    states_distinct = ee.List(feature_coll.aggregate_array(state_col)).distinct().getInfo()
    if not states_distinct:
        raise ValueError(f'No values found for state_col="{state_col}" in the feature collection')
    west_set = set(STATES)
    all_west = all(isinstance(s, str) and s in west_set for s in states_distinct)
    all_east = all(isinstance(s, str) and s not in west_set for s in states_distinct)
    if not (all_west or all_east):
        raise ValueError('clustered_sample_etf requires all features be either in WEST_STATES or all outside; mixed states detected')
    use_west = all_west

    # LANID image for east-side masks
    lanid = get_lanid()
    east_fc = ee.FeatureCollection(EAST_STATES)

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask_west = irr_min_yr_mask.updateMask(irr.lt(1))
        irr_mask_east = lanid.select(f'irr_{year}').clip(east_fc)
        irr_east = ee.Image(1).subtract(irr_mask_east)

        desc = f'etf_{mask_type}_{year}'

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        # Determine source collection like sparse_sample_etf
        members = ['eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop']
        _source = source
        if _source:
            pass
        elif model == 'ssebop' and usgs_nhm:
            _source = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'
        elif _source is None and model in members:
            _source = f'projects/openet/assets/{model}/conus/gridmet/landsat/c02'
        elif model == 'disalexi':
            _source = 'projects/openet/assets/disalexi/landsat/c02'
        elif model == 'openet':
            _source = 'projects/openet/assets/ensemble/conus/gridmet/landsat/c02'
        else:
            raise ValueError('Invalid model name')

        coll = ee.ImageCollection(_source).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])
            _dt = splt[-1]

            selectors.append(_name)

            etf_img = ee.Image(os.path.join(_source, img_id))

            # Align scaling/selection with sparse extractor
            if source is not None and scale is not None:
                etf_img = etf_img.select('et_fraction').divide(scale)
            elif model == 'openet':
                etf_img = etf_img.select('et_ensemble_mad').divide(10000)
            elif model in ['sims', 'eemetric', 'ssebop']:
                etf_img = etf_img.select('et_fraction').divide(10000)
            elif model in ['geesebal', 'ptjpl', 'disalexi']:
                et_img = etf_img.select('et').divide(1000)
                refet = ee.Image(f'projects/openet/assets/reference_et/conus/gridmet/daily/v1/{_dt}').select('eto')
                etf_img = et_img.divide(refet)
            else:
                etf_img = etf_img.select('et_fraction').divide(10000)

            etf_img = etf_img.rename(_name)

            if mask_type == 'no_mask':
                etf_masked = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_masked = etf_img.clip(feature_coll.geometry()).mask(irr_mask_west if use_west else irr_mask_east)
            elif mask_type == 'inv_irr':
                etf_masked = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0) if use_west else irr_east.gt(0))
            else:
                etf_masked = etf_img.clip(feature_coll.geometry())

            if first:
                bands = etf_masked
                first = False
            else:
                bands = bands.addBands([etf_masked])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        data = bands.reduceRegions(collection=feature_coll, reducer=ee.Reducer.mean(), scale=30)

        if dest == 'bucket':
            if not bucket:
                raise ValueError('ETF export dest="bucket" requires a bucket name/url')
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=f'{file_prefix}/remote_sensing/landsat/extracts/{model}_etf/{mask_type}/{desc}',
                fileFormat='CSV',
                selectors=selectors,
            )
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_etf" if drive_categorize else drive_folder
            task = ee.batch.Export.table.toDrive(
                collection=data,
                description=desc,
                folder=drive_folder_name,
                fileNamePrefix=f'remote_sensing/landsat/extracts/{model}_etf/{mask_type}/{desc}',
                fileFormat='CSV',
                selectors=selectors,
            )
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')

        task.start()
        print(desc)


def export_to_cloud(images_txt, bucket=None, pathrows=None, dest='drive', drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export imagery by ID list as GeoTIFF + metadata to Drive or Cloud Storage.

    Parameters
    - images_txt: path to text file with one ee.Image ID per line
    - bucket: Cloud Storage bucket (required when dest='bucket')
    - pathrows: optional CSV with `PR` column to filter Path/Row
    - dest: 'drive' (default) or 'bucket'
    - drive_folder: Drive folder when dest='drive'

    Side Effects
    - Starts ee.batch image and table exports to either Drive (folder `drive_folder`)
      or Cloud Storage (`gs://<bucket>/geotiff|metadata`).
    """
    with open(images_txt, 'r') as fp:
        images = [line.strip() for line in fp.readlines()]

    prs, image, metadata = None, None, None
    if pathrows:
        df = pd.read_csv(pathrows)
        prs = [str(pr).rjust(6, '0') for pr in df['PR']]

    existing_geotiffs = []
    existing_metadata = []
    if dest == 'bucket':
        if not bucket:
            raise ValueError('export_to_cloud with dest="bucket" requires a bucket name/url')
        try:
            existing_geotiffs = subprocess.check_output(
                ['gsutil', 'ls', f'gs://{bucket}/{file_prefix}/geotiff/']
            ).decode('utf-8').split('\n')
            existing_geotiffs = [os.path.basename(f) for f in existing_geotiffs if f]
            existing_metadata = subprocess.check_output(
                ['gsutil', 'ls', f'gs://{bucket}/{file_prefix}/metadata/']
            ).decode('utf-8').split('\n')
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

        if dest == 'bucket':
            if f'{bname}.tif' in existing_geotiffs and f'{bname}.geojson' in existing_metadata:
                print(f"Skipping {bname} as it already exists in the bucket.")
                continue

        try:
            image = ee.Image(i)
            metadata = image.getInfo()
        except Exception as exc:
            print(bname, exc)
            continue

        if dest == 'bucket':
            if bname not in [f.split('.')[0] + '.tif' for f in existing_geotiffs]:
                task = ee.batch.Export.image.toCloudStorage(
                    image=image,
                    description=bname,
                    bucket=bucket,
                    fileNamePrefix=f'{file_prefix}/{geotiff_filename}',
                    scale=30,
                    maxPixels=1e13,
                    crs='EPSG:5070',
                    fileFormat='GeoTIFF',
                    formatOptions={'noData': 0.0, 'cloudOptimized': True},
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
                    fileNamePrefix=f'{file_prefix}/{metadata_filename}',
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
        else:  # drive
            drive_folder_geotiff = f"{drive_folder}_geotiff" if drive_categorize else drive_folder
            drive_folder_metadata = f"{drive_folder}_metadata" if drive_categorize else drive_folder

            task = ee.batch.Export.image.toDrive(
                image=image,
                description=bname,
                folder=drive_folder_geotiff,
                fileNamePrefix=geotiff_filename,
                scale=30,
                maxPixels=1e13,
                crs='EPSG:5070',
                fileFormat='GeoTIFF',
            )
            try:
                task.start()
                print(f"Exporting GeoTIFF to Drive: {geotiff_filename}")
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), bname, '......')
                time.sleep(600)
                task.start()

            metadata_task = ee.batch.Export.table.toDrive(
                collection=ee.FeatureCollection([ee.Feature(None, metadata)]),
                description=os.path.basename(metadata_filename).split('.')[0],
                folder=drive_folder_metadata,
                fileNamePrefix=metadata_filename,
                fileFormat='GeoJSON'
            )
            try:
                metadata_task.start()
                print(f"Exporting Metadata to Drive: {metadata_filename}")
            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), bname, '......')
                time.sleep(600)
                metadata_task.start()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
