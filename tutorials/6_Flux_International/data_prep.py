import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from prep import get_flux_sites, get_ensemble_parameters, COLUMN_MULTIINDEX, ACCEPTED_UNITS_MAP

project = '6_Flux_International'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)

if not os.path.isdir(root):
    root = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'swim-rs')
    data = os.path.join(root, 'tutorials', project, 'data')
    project_ws = os.path.join(root, 'tutorials', project)

landsat = os.path.join(data, 'landsat')
sentinel = os.path.join(data, 'sentinel')
met_timeseries = os.path.join(data, 'met_timeseries')

ecostress = os.path.join(data, 'ecostress')
era5 = os.path.join(data, 'met_timeseries')

# GCS - Earth Engine
fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
bucket_ = 'wudr'

# ICOS 200m station buffer shapefile index
FEATURE_ID = 'sid'

# preparation-specific paths
remote_sensing_tables = os.path.join(data, 'rs_tables')

joined_timeseries = os.path.join(data, 'plot_timeseries')
station_file = os.path.join(data, 'station_metadata.csv')
static_properties = os.path.join(data, 'properties', f'{project}_properties.json')
# properties = os.path.join(data, 'properties', f'{project}_properties.csv')
dyanmics_data = os.path.join(data, f'{project}_dynamics.json')

era5_extracts = os.path.join(era5, 'ee_extracts')
era5_series = os.path.join(era5, 'era5_land')

prepped_input = os.path.join(data, 'prepped_input.json')

# Landsat processing - ETf from rasterstats code in openet-ptjpl fork, NDVI from Earth Engine
landsat_extract = os.path.join(landsat, 'extracts', 'landsat_extract.json')
landsat_tables = os.path.join(landsat, 'tables')
landsat_etf_pqt = os.path.join(landsat_tables, '{}_{}.parquet'.format('etf', 'no_mask'))

landsat_ee_data = os.path.join(landsat, 'extracts', 'ndvi', 'no_mask')
landsat_ndvi = os.path.join(landsat_tables, '{}_{}.parquet'.format('ndvi', 'no_mask'))

# Sentinel processing -- From Earth Engine
sentinel_ee_data = os.path.join(sentinel, 'extracts', 'ndvi', 'no_mask')
sentinel_tables = os.path.join(sentinel, 'tables')
sentinel_ndvi = os.path.join(sentinel_tables, '{}_{}.parquet'.format('ndvi', 'no_mask'))

# ECOSTRESS processing - from rasterstats code in openet-ptjpl fork
ecostress_extracts = os.path.join(ecostress, 'extracts', 'ecostress_extract.json')
ecostress_tables = os.path.join(ecostress, 'tables')
ecostress_etf_pqt = os.path.join(ecostress_tables, '{}_{}.parquet'.format('etf', 'no_mask'))

rs_files = [
    landsat_etf_pqt,
    landsat_ndvi,
    sentinel_ndvi,
    ecostress_etf_pqt,
]

modis_lulc = os.path.join(data, 'properties', f'{project}_landcover.csv')
soils = os.path.join(data, 'properties', f'{project}_hwsd.csv')

# flux sites
shapefile_path = os.path.join(data, 'gis', '6_Flux_International_150mBuf.shp')
sites = get_flux_sites(shapefile_path, index_col=FEATURE_ID)

ERA5LAND_PARAMS = ['swe', 'eto', 'tmean', 'tmin', 'tmax', 'precip', 'srad']
PARAMS_MAPPING = {'precip': 'prcp'}

irrigation_threshold = 0.3


def prep_era5land():
    all_sites_records = defaultdict(list)

    filelist = sorted(os.listdir(era5_extracts))
    for i, filename in enumerate(tqdm(filelist, desc="Reading monthly ERA5 LAND CSVs")):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(era5_extracts, filename)
            df_month = pd.read_csv(filepath, index_col=0)

            for sid, row_data in df_month.iterrows():
                for col_name, value in row_data.items():
                    parts = col_name.rsplit('_', 1)
                    if len(parts) != 2:
                        continue
                    param_name, date_str = parts

                    if not date_str.isdigit() or len(date_str) != 8:
                        continue
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    all_sites_records[sid].append({
                        'date': date_obj,
                        'parameter': param_name,
                        'value': value
                    })

    for i, (sid, records) in enumerate(tqdm(all_sites_records.items(), desc="Writing ERA5 LAND parquet")):
        if not records:
            continue

        df = pd.DataFrame(records)

        if df.empty:
            continue

        df = df.pivot_table(
            index='date',
            columns='parameter',
            values='value'
        )

        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)

        if 'srad' in df.columns:
            df.loc[df['srad'] == 0.0, 'srad'] = np.nan

        df = df.dropna(axis=1, how='all')

        if 'ERA5LAND_PARAMS' in globals() and ERA5LAND_PARAMS:
            missing_cols = [c for c in ERA5LAND_PARAMS if c not in df.columns]
            if any(missing_cols):
                continue

        df.sort_index(inplace=True)
        df = df.rename(columns=PARAMS_MAPPING)

        # multi-index columns
        # ['site', 'instrument', 'parameter', 'units', 'algorithm', 'flag', 'mask']
        cols = [(sid, 'none', c, ACCEPTED_UNITS_MAP.get(c, 'none'), 'era5_land', 'no_mask') for c in df.columns]
        cols = pd.MultiIndex.from_tuples(cols,  names=COLUMN_MULTIINDEX)
        df.columns = cols

        outfile = os.path.join(era5_series, f'{sid}.parquet')
        df.to_parquet(outfile)

"""
The following depends on the fork of OpenET-PTJPL at https://github.com/dgketchum/openet-ptjpl
which is modified to use ERA5-LAND data to get daily EToF for Landsat, and has scripts
I added to use NASA AppEEARS to get ECOSTRESS ET daily data in W/m^2

ECOSTRESS
1. Use ecostress_appeears_download.py to extract ECOSTRESS data from NASA AppEEARS.
2. Use ecostress_extract.py to get spatial stats for the ECOSTRESS data, in W/m^2.

Landsat
1. Use openet/ptjpl/image_export.py to export Landsat OpenET PTJPL data in chips to a bucket.
2. Use landsat_extract.py to get spatial stats for the Landsat EToF/et_fraction data.

The following functions take the results of the extract scripts and puts them in a format
like the Earth Engine-based data, so they can be combined.

"""


def prep_landsat_raster_extracts():
    with open(landsat_extract, 'r') as f:
        all_landsat_data = json.load(f)

    first_pass, adf, ctdf = True, None, None
    common_dt_index = pd.date_range('2015-01-01', '2024-12-31', freq='D')

    for site_id in sites:
        site_df = pd.DataFrame(index=common_dt_index)
        processed_data_for_site = False

        if site_id in all_landsat_data and all_landsat_data[site_id]:
            etf_data_for_site = {}
            for date_key, etf_value in all_landsat_data[site_id].items():
                parsed_date = datetime.strptime(date_key, "%Y-%m-%d")
                etf_data_for_site[parsed_date] = etf_value

            if etf_data_for_site:
                etf_series = pd.Series(etf_data_for_site)
                site_df[(site_id, 'landsat', 'etf', 'unitless', 'ptjpl', 'no_mask')] = etf_series
                processed_data_for_site = True
            else:
                print(f"No Landsat data values for site {site_id} after parsing.")
        else:
            print(f"Landsat source data not found or empty for site {site_id}.")

        if not processed_data_for_site:
            continue

        if first_pass:
            adf = site_df.copy()
            first_pass = False
        else:
            adf = pd.concat([adf, site_df], axis=1, sort=False)

    if adf is not None and not adf.empty:
        adf.columns = pd.MultiIndex.from_tuples(adf.columns, names=COLUMN_MULTIINDEX)
        adf = adf.sort_index(axis=1)
        adf.to_parquet(landsat_etf_pqt, engine='pyarrow')
        print(f'wrote {landsat_etf_pqt}')
    else:
        print("No Landsat ADF data to write.")


def prep_ecostress_raster_extracts():
    with open(ecostress_extracts, 'r') as f:
        all_ecostress_data = json.load(f)

    first_pass, adf, ctdf = True, None, None
    common_dt_index = pd.date_range('2015-01-01', '2024-12-31', freq='D')

    for site_id in sites:
        site_df = pd.DataFrame(index=common_dt_index)
        processed_data_for_site = False

        if site_id in all_ecostress_data and all_ecostress_data[site_id]:
            era5_file_path = os.path.join(era5_series, f'{site_id}.parquet')
            try:
                site_era5_df = pd.read_parquet(era5_file_path)
            except FileNotFoundError:
                print(f'ERA5-LAND data not found for site {site_id}. Skipping Ecostress for this site.')
                continue

            raw_le_data = {}
            for date_key, stats_value in all_ecostress_data[site_id].items():
                parsed_date = datetime.strptime(date_key, "%Y-%m-%d")
                raw_le_data[parsed_date] = stats_value

            etf_data_for_site = {}
            for date_obj, le_stats in raw_le_data.items():
                if le_stats['count'] < 10:
                    continue
                try:
                    idx = pd.IndexSlice
                    eto_value = site_era5_df.loc[date_obj, idx[:, :, ['eto'], :, :, 'no_mask']].item()
                except KeyError:
                    print(
                        f"Warning: ETo for {date_obj.strftime('%Y-%m-%d')} not in ERA5 for site {site_id}. "
                        f"Skipping Ecostress point.")
                    continue

                if pd.notna(eto_value) and eto_value > 0:
                    eta_value = (le_stats['mean'] * 0.03527)
                    etf_value = eta_value / eto_value
                    if etf_value > 1.3:
                        continue
                    etf_data_for_site[date_obj] = etf_value

            if etf_data_for_site:
                etf_series = pd.Series(etf_data_for_site)
                site_df[(site_id, 'ecostress', 'etf', 'unitless', 'ptjpl', 'no_mask')] = etf_series
                processed_data_for_site = True
            else:
                print(f"No Ecostress ETf values derived for site {site_id} after filtering.")
        else:
            print(f"Ecostress source data not found or empty for site {site_id}.")

        if not processed_data_for_site:
            continue

        if first_pass:
            adf = site_df.copy()
            first_pass = False
        else:
            adf = pd.concat([adf, site_df], axis=1, sort=False)

    if adf is not None and not adf.empty:
        adf.columns = pd.MultiIndex.from_tuples(adf.columns, names=COLUMN_MULTIINDEX)
        adf = adf.sort_index(axis=1)
        adf.to_parquet(ecostress_etf_pqt,
                       engine='pyarrow')
        print(f'wrote {ecostress_etf_pqt}')
    else:
        print("No Ecostress ADF data to write.")


def prep_earthengine_extracts():
    from prep.remote_sensing import sparse_time_series

    yrs = [x for x in range(2015, 2025)]

    sparse_time_series(shapefile_path, landsat_ee_data, yrs, landsat_ndvi, feature_id=FEATURE_ID,
                       instrument='landsat', parameter='ndvi', algorithm='none', mask='no_mask', select=sites)

    sparse_time_series(shapefile_path, sentinel_ee_data, yrs, sentinel_ndvi, feature_id=FEATURE_ID,
                       instrument='sentinel', parameter='ndvi', algorithm='none', mask='no_mask', select=sites)


def join_remote_sensing_data():
    from prep.remote_sensing import join_remote_sensing

    join_remote_sensing(rs_files, remote_sensing_tables, station_selection='exclusive')


def prep_field_properties():
    from prep.field_properties import write_field_properties

    write_field_properties(shapefile_path, out_js=static_properties, soils=soils, lulc=modis_lulc, index_col=FEATURE_ID,
                           flux_meta=None, lulc_key='modis_lc', **{'extra_lulc_key': 'glc10_lc'})


def prep_timeseries():
    from prep.field_timeseries import join_daily_timeseries

    join_daily_timeseries(fields=shapefile_path, met_dir=era5_series, rs_dir=remote_sensing_tables,
                          dst_dir=joined_timeseries, overwrite=True, start_date='2015-01-01', end_date='2024-12-31',
                          feature_id=FEATURE_ID, **{'target_fields': sites})


def prep_dynamics():
    from prep.dynamics import SamplePlotDynamics

    dynamics = SamplePlotDynamics(joined_timeseries, static_properties, irr_threshold=irrigation_threshold,
                                  etf_target='ptjpl', use_lulc=True, use_mask=False,
                                  out_json_file=dyanmics_data, select=sites)

    dynamics.analyze_irrigation(lookback=5)
    dynamics.analyze_groundwater_subsidy()
    dynamics.analyze_k_parameters()
    dynamics.save_json()


def prep_input_json():
    from prep.prep_plots import prep_fields_json

    params = [('none', 'ndvi', 'no_mask'), ('ptjpl', 'etf', 'no_mask')]
    prep_fields_json(static_properties, joined_timeseries, dyanmics_data,
                     prepped_input, target_plots=sites, rs_params=params,
                     interp_params=['ndvi'])


if __name__ == '__main__':
    # prep_era5land()
    # prep_landsat_raster_extracts()
    # prep_ecostress_raster_extracts()
    # prep_earthengine_extracts()
    # join_remote_sensing_data()
    # prep_field_properties()
    # prep_timeseries()
    # prep_dynamics()
    prep_input_json()

    pass
# ========================= EOF ====================================================================
