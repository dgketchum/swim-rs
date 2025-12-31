import os
from datetime import date, timedelta

import ee
import time

from openet.refetgee import Daily

from swimrs.data_extraction.ee.ee_utils import as_ee_feature_collection


def sample_era5_land_variables_daily(shapefile, bucket=None, debug=False, check_dir=None,
                                     overwrite=False, start_yr=2004, end_yr=2023, feature_id_col='FID',
                                     file_prefix='swim'):
    """Export daily ERA5-Land variables reduced over features, by month.

    Uses the daily aggregated ERA5-Land collection when possible:
    - ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") for SWE, temperature, precipitation, and shortwave radiation.
    - ETo is still computed via openet-refet-gee from the HOURLY collection (until we add a trusted daily refET band).

    For each month in the year range, builds an ee.Image with per-day bands for:
    - SWE (mm)
    - ETo (mm; via refetgee)
    - Tmean/Tmin/Tmax (°C)
    - precip (mm)
    - srad (W/m^2; derived from daily sum if needed)
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
    era5_land_daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

    skipped_months, exported_months = 0, 0
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    # Use a scale smaller than feature sizes to ensure reduceRegions finds pixels.
    # ERA5-Land native resolution is ~11km, but small polygons (e.g., 150m buffers)
    # need a finer scale so the image is resampled before reduction.
    scale_era5 = 150

    def _select_daily_aggr(img: ee.Image) -> dict:
        """Select required DAILY_AGGR bands with known names."""
        # Based on the dataset's current band list in Earth Engine (DAILY_AGGR):
        # - temperature_2m, temperature_2m_min, temperature_2m_max
        # - snow_depth_water_equivalent
        # - total_precipitation_sum
        # - surface_solar_radiation_downwards_sum
        return {
            "swe_m": img.select("snow_depth_water_equivalent"),
            "tmean_k": img.select("temperature_2m"),
            "tmin_k": img.select("temperature_2m_min"),
            "tmax_k": img.select("temperature_2m_max"),
            "precip_m": img.select("total_precipitation_sum"),
            "srad_j": img.select("surface_solar_radiation_downwards_sum"),
        }

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

            day_start_ee = ee.Date(d.isoformat())
            day_end_ee = day_start_ee.advance(1, 'day')

            # Daily aggregate image for the day (preferred for non-refET variables)
            daily_img = era5_land_daily.filterDate(day_start_ee, day_end_ee).first()

            # DAILY_AGGR band names (current): temperature_2m (mean), temperature_2m_min/max, etc.
            # Units: temps in K, precip in m, SWE in m, srad sums in J/m^2/day.
            b = _select_daily_aggr(daily_img)

            daily_mean_swe_img = b["swe_m"].multiply(1000).rename(f'swe_{day_str_yyyymmdd}')
            daily_mean_swe_img = daily_mean_swe_img.set('system:time_start', day_start_ee.millis())

            daily_tmean_c_img = b["tmean_k"].subtract(273.15).rename(f'tmean_{day_str_yyyymmdd}')
            daily_tmin_c_img = b["tmin_k"].subtract(273.15).rename(f'tmin_{day_str_yyyymmdd}')
            daily_tmax_c_img = b["tmax_k"].subtract(273.15).rename(f'tmax_{day_str_yyyymmdd}')

            daily_tmean_c_img = daily_tmean_c_img.set('system:time_start', day_start_ee.millis())
            daily_tmin_c_img = daily_tmin_c_img.set('system:time_start', day_start_ee.millis())
            daily_tmax_c_img = daily_tmax_c_img.set('system:time_start', day_start_ee.millis())

            daily_total_precip_img = b["precip_m"].multiply(1000).rename(f'precip_{day_str_yyyymmdd}')
            daily_total_precip_img = daily_total_precip_img.set('system:time_start', day_start_ee.millis())

            # Convert daily energy sum (J/m^2/day) to mean flux density (W/m^2)
            daily_mean_srad_img = b["srad_j"].divide(86400).rename(f'srad_{day_str_yyyymmdd}')
            daily_mean_srad_img = daily_mean_srad_img.set('system:time_start', day_start_ee.millis())

            # ETo still computed from hourly (refetgee expects hourly inputs)
            era5_coll_for_day_hourly = era5_land_hourly.filterDate(day_start_ee, day_end_ee)
            daily_eto_img = Daily.era5_land(era5_coll_for_day_hourly).etr.rename(f'eto_{day_str_yyyymmdd}')
            daily_eto_img = daily_eto_img.set('system:time_start', day_start_ee.millis())

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
            fileNamePrefix=f'{file_prefix}/era5/{desc}',
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
