"""NDVI Earth Engine exporters.

Functions
- sparse_sample_ndvi: per-field/year export iterating a shapefile; applies west/east masking using
  IrrMapper (west) and LANID (east). Best for widely dispersed fields.
- clustered_sample_ndvi: per-year export for a set of clustered features; accepts shapefile path, EE asset ID,
  or ee.FeatureCollection, supports `select` to restrict by feature_id, and `state_col` for consistent
  west/east masking (same logic as sparse). The Boulder Step 2 notebook also mentions the sparse option:
  examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb.
- export_ndvi_images: exports clipped NDVI images from Landsat/Sentinel for inspection.

Note
- All FeatureCollection/shapefile inputs are normalized via as_ee_feature_collection in ee_utils.
"""

import os
import sys
import time
from tqdm import tqdm

import ee
import geopandas as gpd

from swimrs.data_extraction.ee.ee_utils import get_lanid, as_ee_feature_collection
from swimrs.data_extraction.ee.ee_utils import landsat_masked, sentinel2_masked

sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

EC_POINTS = 'users/dgketchum/fields/flux'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'


def get_flynn():
    """Return a small example polygon FeatureCollection.

    Useful for quick debugging of exports in a known area.

    Returns
    - ee.FeatureCollection with one polygon feature.
    """
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def export_ndvi_images(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr', dest='drive',
                       drive_folder='swim', file_prefix='swim', drive_categorize=False):
    """Export per-scene NDVI images to Cloud Storage for a feature collection.

    Parameters
    - feature_coll: ee.FeatureCollection defining the region to clip.
    - year: int target year.
    - bucket: str GCS bucket name.
    - debug: bool; if True, sample pixel values for inspection.
    - mask_type: {'irr','inv_irr','no_mask'} mask behavior using IrrMapper.

    Side Effects
    - Starts ee.batch export tasks to the provided `bucket`.
    """
    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year),
                              '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    # Use harmonized bands (SBAF-adjusted to OLI reference)
    coll = landsat_masked(year, feature_coll, harmonize=True).map(lambda x: x.normalizedDifference(['NIR_H', 'RED_H']))
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

        desc = 'NDVI_{}_{}'.format(mask_type, _name)
        if dest == 'bucket':
            if not bucket:
                raise ValueError('NDVI image export dest="bucket" requires a bucket name/url')
            task = ee.batch.Export.image.toCloudStorage(
                image=img,
                description=desc,
                bucket=bucket,
                fileNamePrefix=f'{file_prefix}/remote_sensing/landsat/extracts/ndvi/images/{desc}',
                region=feature_coll.geometry(),
                crs='EPSG:5070',
                scale=30,
            )
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_ndvi_images" if drive_categorize else drive_folder
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=desc,
                folder=drive_folder_name,
                fileNamePrefix=f'ndvi/images/{desc}',
                region=feature_coll.geometry(),
                crs='EPSG:5070',
                scale=30,
            )
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')

        task.start()
        print(_name)


def sparse_sample_ndvi(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None,
                       feature_id='FID', select=None, start_yr=2000, end_yr=2024, state_col='field_3',
                       satellite='landsat', dest='drive', drive_folder='swim', file_prefix='swim',
                       drive_categorize=False):
    """Export per-field NDVI timeseries (one CSV per field-year) to GCS.

    Parameters
    - shapefile: path to polygon features (CRS will be reprojected to EPSG:4326).
    - bucket: GCS bucket for output (folder structure includes satellite/mask).
    - debug: bool; if True, prints sampled values.
    - mask_type: {'irr','inv_irr','no_mask'} NDVI masking using IrrMapper or LANID (east).
    - check_dir: optional local directory; skip exports when CSV exists.
    - feature_id: attribute name for feature identifier in shapefile.
    - select: optional iterable of feature IDs to include.
    - start_yr, end_yr: export year range inclusive.
    - state_col: column indicating US state for LANID fallback east of WEST_STATES.
    - satellite: {'landsat','sentinel'} source collection.

    Notes
    - This is the sparse extractor, iterating features of a shapefile and applying west/east masks
      (IrrMapper west, LANID east) based on the state column, like the ETf sparse extractor.
      For clustered fields, prefer clustered_sample_ndvi which supports `select` and direct FeatureCollection/asset inputs.
      The Boulder Step 2 notebook shows both: examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb.

    Side Effects
    - Starts ee.batch table export tasks to the configured destination.
    Returns
    - None; prints exported/skipped counts.
    """
    df = gpd.read_file(shapefile)
    df.index = df[feature_id]

    if not df.crs.srs == 'EPSG:4326':
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

    for fid, row in tqdm(df.iterrows(), desc=f'Extracting {satellite} NDVI', total=df.shape[0]):

        for year in range(start_yr, end_yr + 1):

            site = row[feature_id]

            desc = 'ndvi_{}_{}_{}'.format(site, mask_type, year)

            if check_dir:
                if not os.path.isdir(check_dir):
                    raise ValueError(f'File checking on but directory does not exist: {check_dir}')

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

            # Use harmonized bands (SBAF-adjusted to OLI reference)
            if satellite == 'landsat':
                coll = landsat_masked(year, fc, harmonize=True).map(lambda x: x.normalizedDifference(['NIR_H', 'RED_H']))
            elif satellite == 'sentinel':
                coll = sentinel2_masked(year, fc, harmonize=True).map(lambda x: x.normalizedDifference(['NIR_H', 'RED_H']))
            else:
                raise ValueError('Must choose a satellite from landsat or sentinel')

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
                    if nd_img is not None:
                        bands = bands.addBands([nd_img])
                    else:
                        print(f'{fid} image data for {_name} is None, skipping')
                        continue

                if debug:
                    data = nd_img.sample(fc, 30).getInfo()
                    print(data['features'])

            try:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.mean(),
                                           scale=30)
            except AttributeError:
                print(f'{fid} image data for {year} is None, skipping')
                continue

            if dest == 'bucket':
                if not bucket:
                    raise ValueError('NDVI export dest="bucket" requires a bucket name/url')
                task = ee.batch.Export.table.toCloudStorage(
                    data,
                    description=f'{satellite}_{desc}',
                    bucket=bucket,
                    fileNamePrefix=f'{file_prefix}/remote_sensing/{satellite}/extracts/ndvi/{mask_type}/{desc}',
                    fileFormat='CSV',
                    selectors=selectors,
                )
            elif dest == 'drive':
                drive_folder_name = f"{drive_folder}_ndvi" if drive_categorize else drive_folder
                task = ee.batch.Export.table.toDrive(
                    collection=data,
                    description=f'{satellite}_{desc}',
                    folder=drive_folder_name,
                    fileNamePrefix=f'ndvi/{satellite}/{desc}',
                    fileFormat='CSV',
                    selectors=selectors,
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

    print(f'NDVI: Exported {exported}, skipped {skipped} files found in {check_dir}')


def clustered_sample_ndvi(feature_coll, bucket=None, debug=False, mask_type='irr', check_dir=None,
                          start_yr=2004, end_yr=2023, feature_id='FID', select=None, state_col='STATE',
                          satellite='landsat', dest='drive', drive_folder='swim', file_prefix='swim',
                          drive_categorize=False):
    """Export NDVI for all features in a collection, grouped per-year.

    Parameters
    - feature_coll: fields as ee.FeatureCollection, EE asset ID, or local shapefile path (normalized via ee_utils.as_ee_feature_collection).
    - bucket: str GCS bucket name.
    - debug: bool; if True, prints sampled values.
    - mask_type: {'irr','inv_irr','no_mask'} mask selection.
    - check_dir: optional local dir; skip if year CSV exists.
    - start_yr, end_yr: year range.
    - feature_id: property name used as the ID in outputs.
    - select: optional list[str] of feature_id values to include.
    - state_col: column/property with two-letter state code. All features must be either in WEST_STATES or all outside;
      mixed collections raise an error. West uses IrrMapper; east uses LANID (matches sparse_sample_ndvi).
    - satellite: {'landsat','sentinel'}.

    Notes
    - For widely dispersed fields use sparse_sample_ndvi; see examples/1_Boulder/step_2_earth_engine_extract/step_2a.ipynb for context.

    Side Effects
    - Starts an ee.batch table export per year to the selected destination.
    """
    # Accept asset id, ee.FeatureCollection, or a local shapefile path; keep state property
    feature_coll = as_ee_feature_collection(feature_coll, feature_id=feature_id,
                                            keep_props=[state_col] if state_col else [])

    # Optionally filter to a subset of features by ID
    if select is not None:
        feature_coll = feature_coll.filter(ee.Filter.inList(feature_id, select))

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    irr_coll = irr_coll.filterDate(s, e).select('classification')
    remap = irr_coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    # Determine region: all-west or all-east based on state_col
    if state_col is None:
        raise ValueError('state_col must be provided for clustered NDVI extraction')
    states_distinct = ee.List(feature_coll.aggregate_array(state_col)).distinct().getInfo()
    if not states_distinct:
        raise ValueError(f'No values found for state_col="{state_col}" in the feature collection')
    west_set = set(STATES)
    all_west = all(isinstance(s, str) and s in west_set for s in states_distinct)
    all_east = all(isinstance(s, str) and s not in west_set for s in states_distinct)
    if not (all_west or all_east):
        raise ValueError(
            'clustered_sample_ndvi requires all features be either in WEST_STATES or all outside; mixed states detected')
    use_west = all_west

    # LANID image for east-side masks
    lanid = get_lanid()
    east_fc = ee.FeatureCollection(EAST_STATES)

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        first, bands = True, None
        selectors = [feature_id]

        desc = 'ndvi_{}_{}'.format(year, mask_type)

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        # Use harmonized bands (SBAF-adjusted to OLI reference)
        if satellite == 'landsat':
            coll = landsat_masked(year, feature_coll, harmonize=True).map(lambda x: x.normalizedDifference(['NIR_H', 'RED_H']))
        elif satellite == 'sentinel':
            coll = sentinel2_masked(year, feature_coll, harmonize=True).map(lambda x: x.normalizedDifference(['NIR_H', 'RED_H']))
        else:
            raise ValueError('Must choose a satellite from landsat or sentinel')

        ndvi_scenes = coll.aggregate_histogram('system:index').getInfo()

        for img_id in ndvi_scenes:

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            nd_img = coll.filterMetadata('system:index', 'equals', img_id).first().rename(_name)

            if mask_type == 'no_mask':
                nd_masked = nd_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                east_mask = lanid.select(f'irr_{year}').clip(east_fc)
                nd_masked = nd_img.clip(feature_coll.geometry()).mask(irr_mask if use_west else east_mask)
            elif mask_type == 'inv_irr':
                east_inv = ee.Image(1).subtract(lanid.select(f'irr_{year}').clip(east_fc))
                nd_masked = nd_img.clip(feature_coll.geometry()).mask(irr.gt(0) if use_west else east_inv.gt(0))
            else:
                nd_masked = nd_img.clip(feature_coll.geometry())

            if first:
                bands = nd_masked
                first = False
            else:
                bands = bands.addBands([nd_masked])

        try:
            data = bands.reduceRegions(collection=feature_coll, reducer=ee.Reducer.mean(), scale=30)
        except AttributeError as exc:
            print(f'{desc} raised {exc}')
            continue

        if debug:
            print(data.first().getInfo()['features'])

        if dest == 'bucket':
            if not bucket:
                raise ValueError('NDVI export dest="bucket" requires a bucket name/url')
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=f'{file_prefix}/remote_sensing/{satellite}/extracts/ndvi/{mask_type}/{desc}',
                fileFormat='CSV',
                selectors=selectors,
            )
        elif dest == 'drive':
            drive_folder_name = f"{drive_folder}_ndvi" if drive_categorize else drive_folder
            task = ee.batch.Export.table.toDrive(
                collection=data,
                description=desc,
                folder=drive_folder_name,
                fileNamePrefix=f'ndvi/{desc}',
                fileFormat='CSV',
                selectors=selectors,
            )
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')

        task.start()
        print(desc)


if __name__ == '__main__':
    pass
# ========================= EOF =======================================================================================
