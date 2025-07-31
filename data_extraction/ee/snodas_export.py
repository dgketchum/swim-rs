import os
import ee
from tqdm import tqdm

def sample_snodas_swe(feature_coll, bucket=None, debug=False, check_dir=None, overwrite=False,
                      start_yr=2004, end_yr=2023, feature_id='FID'):
    feature_coll = ee.FeatureCollection(feature_coll)
    snodas = ee.ImageCollection('projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily')
    skipped, exported = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]
    for year, month in tqdm(dtimes, desc='Extracting SNODAS', total=len(dtimes)):

        first, bands = True, None
        selectors = [feature_id]

        desc = 'swe_{}_{}'.format(year, str(month).zfill(2))

        if check_dir and not overwrite:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                skipped += 1
                continue

        s = '{}-{}-01'.format(year, str(month).zfill(2))
        e = ee.Date(s).advance(1, 'month').format('YYYY-MM-dd').getInfo()
        coll = snodas.filterDate(s, e).select('SWE')

        days = coll.aggregate_histogram('system:index').getInfo()

        for img_id in days:
            selectors.append(img_id)

            swe_img = coll.filterMetadata(
                'system:index', 'equals', img_id).first().rename(img_id)

            # this is in meters, conversion to mm is in the snodas.py processing script
            swe_img = swe_img.clip(feature_coll.geometry())

            if first:
                bands = swe_img
                first = False
            else:
                bands = bands.addBands([swe_img])

        if debug:
            # the below 'FID_1' value ('043_000160') was taken from the tutorial dataset
            fc = ee.FeatureCollection([feature_coll.filterMetadata(feature_id, 'equals', '043_000160').first()])
            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.mean(),
                                       scale=30).getInfo()
            print(data['features'])

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
        exported += 1

    print(f'SNODAS exported {exported}, skipped {skipped} existing files')

if __name__ == '__main__':
    ee.Initialize()

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    bucket_ = 'wudr'
    # fields = 'users/dgketchum/fields/tongue_annex_20OCT2023'
    fields = 'projects/ee-dgketchum/assets/swim/mt_sid_boulder'

    # chk = os.path.join(d, 'examples/tongue/landsat/extracts/swe')
    chk = os.path.join(d, 'examples/tutorial/landsat/extracts/swe')

    FEATURE_ID = 'FID_1'
    sample_snodas_swe(fields, bucket_, debug=False, check_dir=None, feature_id=FEATURE_ID)

# ========================= EOF ====================================================================
