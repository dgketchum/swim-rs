import os
import ee
import time

from openet.refetgee import Daily


def sample_era5_land_variables_daily(feature_coll_asset_id, bucket=None, debug=False, check_dir=None,
                                     overwrite=False, start_yr=2004, end_yr=2023, feature_id_col='FID'):
    """Export daily ERA5-Land variables reduced over features, by month.

    For each month in the year range, builds an ee.Image with per-day bands for
    SWE (mm), ETo (alfalfa, via refetgee), Tmean/Tmin/Tmax (°C), precip (mm), and
    shortwave radiation (W/m^2), then reduces to feature means and exports to GCS.

    Parameters
    - feature_coll_asset_id: str EE asset path for the FeatureCollection.
    - bucket: str GCS bucket for outputs.
    - debug: bool; if True, prints sample reduction for a feature.
    - check_dir: local directory to skip existing month CSVs.
    - overwrite: bool; if True, re-export even if file present.
    - start_yr, end_yr: int year range inclusive.
    - feature_id_col: property name to include as ID.

    Side Effects
    - Starts ee.batch table exports, one per month, to the `bucket`.
    """
    fc = ee.FeatureCollection(feature_coll_asset_id)
    era5_land_hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')

    skipped_months, exported_months = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    scale_era5 = 11132

    for year, month in dtimes:
        first_band_in_month = True
        monthly_bands_image = None
        current_month_selectors = [feature_id_col]

        desc = f'era5_vars_{year}_{str(month).zfill(2)}'

        if check_dir and not overwrite:
            output_filepath = os.path.join(check_dir, f'{desc}.csv')
            if os.path.exists(output_filepath):
                skipped_months += 1
                continue

        month_start_date_ee = ee.Date.fromYMD(year, month, 1)
        month_end_date_ee = month_start_date_ee.advance(1, 'month')

        days_in_month_list = []
        day_iterator_date = month_start_date_ee
        while day_iterator_date.millis().lt(month_end_date_ee.millis()).getInfo():
            days_in_month_list.append(day_iterator_date)
            day_iterator_date = day_iterator_date.advance(1, 'day')

        if not days_in_month_list:
            continue

        for day_date_ee_for_selector in days_in_month_list:
            day_str_yyyymmdd_selector = day_date_ee_for_selector.format('YYYYMMdd').getInfo()
            current_month_selectors.append(f'swe_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'eto_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmean_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmin_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmax_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'precip_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'srad_{day_str_yyyymmdd_selector}')

        for day_date_ee in days_in_month_list:
            day_str_yyyymmdd = day_date_ee.format('YYYYMMdd').getInfo()

            day_start_ee = day_date_ee
            day_end_ee = day_date_ee.advance(1, 'day')

            era5_coll_for_day_hourly = era5_land_hourly.filterDate(day_start_ee, day_end_ee)

            daily_mean_swe_img = era5_coll_for_day_hourly.select('snow_depth_water_equivalent') \
                .mean().multiply(1000).rename(f'swe_{day_str_yyyymmdd}')
            daily_mean_swe_img = daily_mean_swe_img.set('system:time_start', day_start_ee.millis())

            daily_eto_img = Daily.era5_land(era5_coll_for_day_hourly).etr.rename(f'eto_{day_str_yyyymmdd}')
            daily_eto_img = daily_eto_img.set('system:time_start', day_start_ee.millis())

            temp_k_hourly_for_day = era5_coll_for_day_hourly.select('temperature_2m')
            daily_tmean_c_img = temp_k_hourly_for_day.mean().subtract(273.15).rename(f'tmean_{day_str_yyyymmdd}')
            daily_tmin_c_img = temp_k_hourly_for_day.min().subtract(273.15).rename(f'tmin_{day_str_yyyymmdd}')
            daily_tmax_c_img = temp_k_hourly_for_day.max().subtract(273.15).rename(f'tmax_{day_str_yyyymmdd}')

            daily_tmean_c_img = daily_tmean_c_img.set('system:time_start', day_start_ee.millis())
            daily_tmin_c_img = daily_tmin_c_img.set('system:time_start', day_start_ee.millis())
            daily_tmax_c_img = daily_tmax_c_img.set('system:time_start', day_start_ee.millis())

            daily_total_precip_img = era5_coll_for_day_hourly.select('total_precipitation_hourly') \
                .sum().multiply(1000).rename(f'precip_{day_str_yyyymmdd}')
            daily_total_precip_img = daily_total_precip_img.set('system:time_start', day_start_ee.millis())

            daily_mean_srad_img = era5_coll_for_day_hourly.select('surface_solar_radiation_downwards_hourly') \
                .mean().rename(f'srad_{day_str_yyyymmdd}')
            daily_mean_srad_img = daily_mean_srad_img.set('system:time_start', day_start_ee.millis())

            all_daily_bands = [
                daily_mean_swe_img, daily_eto_img,
                daily_tmean_c_img, daily_tmin_c_img, daily_tmax_c_img,
                daily_total_precip_img, daily_mean_srad_img
            ]

            if first_band_in_month:
                monthly_bands_image = ee.Image(all_daily_bands)
                first_band_in_month = False
            else:
                monthly_bands_image = monthly_bands_image.addBands(ee.Image(all_daily_bands))

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

        try:
            task.start()
        except ee.ee_exception.EEException as e:
            print('{}, waiting on '.format(e), desc, '......')
            time.sleep(600)
            task.start()
        exported_months += 1


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
