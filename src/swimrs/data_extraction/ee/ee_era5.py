import os
from datetime import date, timedelta

import ee
import time

from openet.refetgee import Daily

from swimrs.data_extraction.ee.ee_utils import as_ee_feature_collection


def _compute_utc_offset_hours(fc: ee.FeatureCollection) -> ee.Number:
    """Compute UTC offset in hours from feature collection centroid longitude.

    Uses the solar time approximation: offset_hours = longitude / 15, rounded.
    This matches the approach in openet-ptjpl for ERA5-Land reference ET.
    """
    centroid_lon = ee.Number(fc.geometry().centroid(1).coordinates().get(0))
    return centroid_lon.divide(15).round()


def _local_day_utc_bounds(day_date: date, utc_offset_hours: ee.Number) -> tuple:
    """Get UTC start/end times for a local day given UTC offset.

    Parameters
    ----------
    day_date : date
        The calendar date (interpreted as local date)
    utc_offset_hours : ee.Number
        Hours offset from UTC (e.g., -7 for Mountain Time)

    Returns
    -------
    tuple[ee.Date, ee.Date]
        (utc_start, utc_end) representing local midnight-to-midnight in UTC
    """
    local_midnight = ee.Date.fromYMD(day_date.year, day_date.month, day_date.day)
    utc_start = local_midnight.advance(utc_offset_hours.multiply(-1), 'hour')
    utc_end = utc_start.advance(1, 'day')
    return utc_start, utc_end


def _aggregate_hourly_to_daily(hourly_coll: ee.ImageCollection, day_str: str) -> ee.Image:
    """Aggregate 24 hourly images to daily values.

    Parameters
    ----------
    hourly_coll : ee.ImageCollection
        Filtered collection of ~24 hourly ERA5-Land images for one local day
    day_str : str
        Date string in YYYYMMDD format for band naming

    Returns
    -------
    ee.Image
        Multi-band image with daily aggregates
    """
    # Temperature: mean, min, max (convert K to C)
    temp = hourly_coll.select('temperature_2m')
    tmean_c = temp.mean().subtract(273.15).rename(f'tmean_{day_str}')
    tmin_c = temp.min().subtract(273.15).rename(f'tmin_{day_str}')
    tmax_c = temp.max().subtract(273.15).rename(f'tmax_{day_str}')

    # Precipitation: sum (convert m to mm)
    precip_mm = hourly_coll.select('total_precipitation_hourly').sum().multiply(1000).rename(f'precip_{day_str}')

    # Solar radiation: sum J/m² then convert to mean W/m² (divide by 86400 seconds)
    srad_wm2 = hourly_coll.select('surface_solar_radiation_downwards_hourly').sum().divide(86400).rename(f'srad_{day_str}')

    # SWE: mean of instantaneous values (convert m to mm)
    swe_mm = hourly_coll.select('snow_depth_water_equivalent').mean().multiply(1000).rename(f'swe_{day_str}')

    return ee.Image([swe_mm, tmean_c, tmin_c, tmax_c, precip_mm, srad_wm2])


def sample_era5_land_variables_daily(shapefile, bucket=None, debug=False, check_dir=None,
                                     overwrite=False, start_yr=2004, end_yr=2023, feature_id_col='FID',
                                     file_prefix='swim'):
    """Export daily ERA5-Land variables reduced over features, by month.

    Uses the ERA5-Land HOURLY collection with local-time day boundaries to match
    openet-ptjpl's reference ET calculation. The UTC offset is computed from the
    feature collection centroid longitude (offset_hours = lon / 15, rounded).

    For each month in the year range, builds an ee.Image with per-day bands for:
    - SWE (mm)
    - ETo (mm; via refetgee)
    - Tmean/Tmin/Tmax (°C)
    - precip (mm)
    - srad (W/m^2; derived from daily sum)
    then reduces to feature means and exports to GCS.

    Parameters
    - shapefile: path to local shapefile with polygon features.
    - bucket: str GCS bucket for outputs.
    - debug: bool; if True, prints sample reduction for a feature.
    - check_dir: local directory to skip existing month CSVs.
    - overwrite: bool; if True, re-export even if file present.
    - start_yr, end_yr: int year range inclusive.
    - feature_id_col: property name to include as ID.
    - file_prefix: str prefix for bucket subdirectory (e.g., project name).

    Side Effects
    - Starts ee.batch table exports, one per month, to the `bucket`.
    """
    fc = as_ee_feature_collection(shapefile, feature_id=feature_id_col)
    era5_land_hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')

    # Compute UTC offset from feature collection centroid for local-time aggregation
    utc_offset_hours = _compute_utc_offset_hours(fc)

    skipped_months, exported_months = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    # Use a scale smaller than feature sizes to ensure reduceRegions finds pixels.
    # ERA5-Land native resolution is ~11km, but small polygons (e.g., 150m buffers)
    # need a finer scale so the image is resampled before reduction.
    scale_era5 = 150

    def _days_in_month(year_: int, month_: int) -> list[date]:
        d0 = date(year_, month_, 1)
        if month_ == 12:
            d1 = date(year_ + 1, 1, 1)
        else:
            d1 = date(year_, month_ + 1, 1)
        out = []
        d = d0
        while d < d1:
            out.append(d)
            d = d + timedelta(days=1)
        return out

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

        days_in_month = _days_in_month(year, month)
        if not days_in_month:
            continue

        for d in days_in_month:
            day_str_yyyymmdd_selector = d.strftime('%Y%m%d')
            current_month_selectors.append(f'swe_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'eto_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmean_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmin_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'tmax_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'precip_{day_str_yyyymmdd_selector}')
            current_month_selectors.append(f'srad_{day_str_yyyymmdd_selector}')

        for d in days_in_month:
            day_str_yyyymmdd = d.strftime('%Y%m%d')

            # Get UTC bounds for local day
            utc_start, utc_end = _local_day_utc_bounds(d, utc_offset_hours)
            day_start_ee = ee.Date(d.isoformat())  # Keep for system:time_start

            # Filter hourly collection for local day
            hourly_for_day = era5_land_hourly.filterDate(utc_start, utc_end)

            # Aggregate hourly to daily
            daily_vars = _aggregate_hourly_to_daily(hourly_for_day, day_str_yyyymmdd)

            # ETo computed via refetgee (uses same local-time filtered collection)
            daily_eto_img = Daily.era5_land(hourly_for_day).etr.rename(f'eto_{day_str_yyyymmdd}')

            # Combine all bands and set time property
            all_daily_bands = daily_vars.addBands(daily_eto_img)
            all_daily_bands = all_daily_bands.set('system:time_start', day_start_ee.millis())

            if first_band_in_month:
                monthly_bands_image = all_daily_bands
                first_band_in_month = False
            else:
                monthly_bands_image = monthly_bands_image.addBands(all_daily_bands)

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
            fileNamePrefix=f'{file_prefix}/meteorology/era5_land/extracts/{desc}',
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
