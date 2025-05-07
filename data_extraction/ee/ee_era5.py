import os
import ee

from openet.refetgee import Daily

from ee_utils import is_authorized

def sample_era5_swe_daily(feature_coll_asset_id, bucket=None, debug=False, check_dir=None, overwrite=False,
                          start_yr=2004, end_yr=2023, feature_id_col='FID'):
    fc = ee.FeatureCollection(feature_coll_asset_id)
    era5_land_hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')

    skipped, exported = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    scale_era5 = 11132

    for year, month in dtimes:
        first_band_in_month, monthly_bands_image = True, None
        current_month_selectors = [feature_id_col]
        desc = f'era5_swe_{year}_{str(month).zfill(2)}'

        if check_dir and not overwrite:
            output_filepath = os.path.join(check_dir, f'{desc}.csv')
            if os.path.exists(output_filepath):
                skipped += 1
                continue

        month_start_date = ee.Date.fromYMD(year, month, 1)
        month_end_date = month_start_date.advance(1, 'month')

        days_in_month_list = []
        day_iterator_date = month_start_date
        while day_iterator_date.millis().lt(month_end_date.millis()).getInfo():
            days_in_month_list.append(day_iterator_date)
            day_iterator_date = day_iterator_date.advance(1, 'day')

        if not days_in_month_list:
            continue

        for day_date_ee in days_in_month_list:
            day_str_yyyymmdd = day_date_ee.format('YYYYMMdd').getInfo()
            current_month_selectors.append(day_str_yyyymmdd)

            day_start_ee = day_date_ee
            day_end_ee = day_date_ee.advance(1, 'day')

            daily_mean_swe_img = era5_land_hourly.filterDate(day_start_ee, day_end_ee) \
                .select('snow_depth_water_equivalent') \
                .mean().multiply(1000) \
                .rename(day_str_yyyymmdd)

            daily_mean_swe_img = daily_mean_swe_img.set('system:time_start', day_start_ee.millis())

            if first_band_in_month:
                monthly_bands_image = daily_mean_swe_img
                first_band_in_month = False
            else:
                monthly_bands_image = monthly_bands_image.addBands([daily_mean_swe_img])

        if monthly_bands_image is None:
            continue

        if debug:
            debug_fc_collection = fc.filterMetadata('sid', 'equals', 'Harsleben')
            debug_data = monthly_bands_image.reduceRegions(
                collection=debug_fc_collection,
                reducer=ee.Reducer.mean(),
                scale=scale_era5,
            ).getInfo()

        output_data = monthly_bands_image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=scale_era5,
        )

        task = ee.batch.Export.table.toCloudStorage(
            collection=output_data,
            description=desc,
            bucket=bucket,
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=current_month_selectors
        )

        task.start()
        exported += 1

        print(desc)


def sample_era5_eto_daily(feature_coll_asset_id, bucket=None, debug=False, check_dir=None, overwrite=False,
                          start_yr=2004, end_yr=2023, feature_id_col='FID'):
    fc = ee.FeatureCollection(feature_coll_asset_id)
    era5_land_hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')

    skipped, exported = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    scale_era5 = 11132

    for year, month in dtimes:
        first_band_in_month, monthly_bands_image = True, None
        current_month_selectors = [feature_id_col]
        desc = f'era5_eto_{year}_{str(month).zfill(2)}'

        if check_dir and not overwrite:
            output_filepath = os.path.join(check_dir, f'{desc}.csv')
            if os.path.exists(output_filepath):
                skipped += 1
                continue

        month_start_date = ee.Date.fromYMD(year, month, 1)
        month_end_date = month_start_date.advance(1, 'month')

        days_in_month_list = []
        day_iterator_date = month_start_date
        while day_iterator_date.millis().lt(month_end_date.millis()).getInfo():
            days_in_month_list.append(day_iterator_date)
            day_iterator_date = day_iterator_date.advance(1, 'day')

        if not days_in_month_list:
            continue

        for day_date_ee in days_in_month_list:
            day_str_yyyymmdd = day_date_ee.format('YYYYMMdd').getInfo()
            current_month_selectors.append(day_str_yyyymmdd)

            day_start_ee = day_date_ee
            day_end_ee = day_date_ee.advance(1, 'day')

            era5_coll_for_day = era5_land_hourly.filterDate(day_start_ee, day_end_ee)
            daily_eto_img = Daily.era5_land(era5_coll_for_day).etr
            daily_eto_img = daily_eto_img.rename(day_str_yyyymmdd)
            daily_eto_img = daily_eto_img.set('system:time_start', day_start_ee.millis())

            if first_band_in_month:
                monthly_bands_image = daily_eto_img
                first_band_in_month = False
            else:
                monthly_bands_image = monthly_bands_image.addBands([daily_eto_img])

        if monthly_bands_image is None:
            continue

        if debug:
            debug_fc_collection = fc.filterMetadata('sid', 'equals', 'BE-Lon')
            debug_data = monthly_bands_image.reduceRegions(
                collection=debug_fc_collection,
                reducer=ee.Reducer.mean(),
                scale=scale_era5,
            ).getInfo()

        output_data = monthly_bands_image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=scale_era5,
        )

        task = ee.batch.Export.table.toCloudStorage(
            collection=output_data,
            description=desc,
            bucket=bucket,
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=current_month_selectors
        )

        task.start()
        exported += 1

        print(desc)


if __name__ == '__main__':
    is_authorized()

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    bucket_ = 'wudr'
    fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
    FEATURE_ID = 'sid'

    chk_eto = os.path.join(d, 'examples/tutorial/era5land/extracts/eto')
    sample_era5_eto_daily(
        feature_coll_asset_id=fields,
        bucket=bucket_,
        debug=False,
        check_dir=chk_eto,
        overwrite=False,
        start_yr=2015,
        end_yr=2025,
        feature_id_col=FEATURE_ID
    )

# ========================= EOF ====================================================================
