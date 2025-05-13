import json
import os
import pandas as pd
from datetime import datetime

from prep import get_flux_sites, get_ensemble_parameters

project = '6_Flux_International'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)

if not os.path.isdir(root):
    root = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'swim-rs')
    data = os.path.join(root, 'tutorials', project, 'data')
    project_ws = os.path.join(root, 'tutorials', project)

landsat = os.path.join(data, 'landsat')
met_timeseries = os.path.join(data, 'met_timeseries')

extracts = os.path.join(landsat, 'extracts')
tables = os.path.join(landsat, 'tables')
ecostress = os.path.join(data, 'ecostress')
era5 = os.path.join(data, 'era5land')

# GCS - Earth Engine
fields = 'projects/ee-dgketchum/assets/swim/eu_crop_flux_pt'
bucket_ = 'wudr'

# ICOS 200m station buffer shapefile index
FEATURE_ID = 'sid'

# preparation-specific paths
remote_sensing_file = os.path.join(landsat, 'remote_sensing.csv')
joined_timeseries = os.path.join(data, 'plot_timeseries')
station_file = os.path.join(data, 'station_metadata.csv')
irr = os.path.join(data, 'properties', 'calibration_irr.csv')
properties = os.path.join(data, 'properties', 'calibration_properties.csv')
dyanmics_data = os.path.join(landsat, 'calibration_dynamics.json')

eto_extract = os.path.join(era5, 'extracts', 'eto')
snow_extract = os.path.join(era5, 'extracts', 'swe')
snow_out = os.path.join(era5, 'era5land_swe.json')

prepped_input = os.path.join(data, 'prepped_input.json')

ecostress_extract = os.path.join(ecostress, 'ecostress_extract.json')
landsat_extract = os.path.join(landsat, 'landsat_extract.json')

# European crop sites
shapefile_path = os.path.join(data, 'gis', '6_Flux_International_landcover.shp')
sites = get_flux_sites(shapefile_path, index_col=FEATURE_ID)


def load_era5_eto_data(eto_csv_directory, swe_csv_directory, outdir):
    all_dfs = []

    for filename in os.listdir(eto_csv_directory):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(eto_csv_directory, filename)
            df = pd.read_csv(filepath, index_col=0).drop_duplicates().T
            df.index = pd.DatetimeIndex([datetime.strptime(dt, '%Y%m%d') for dt in df.index])
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_eto_df = pd.concat(all_dfs)
    combined_eto_df.sort_index(inplace=True)

    for col in combined_eto_df.columns:
        s = combined_eto_df[col]
        s.to_csv()

    return combined_eto_df


def prep_remote_sensing_input():
    """This depends on the fork of OpenET-PTJPL at https://github.com/dgketchum/openet-ptjpl
    which is modified to use ERA5-LAND data to get daily EToF for Landsat, and has scripts
    I added to use NASA AppEEARS to get ECOSTRESS ET daily data in W/m^2

    ECOSTRESS
    1. Use ecostress_appeears_download.py to extract ECOSTRESS data from NASA AppEEARS.
    2. Use ecostress_extract.py to get spatial stats for the ECOSTRESS data, in W/m^2.

    Landsat
    1. Use openet/ptjpl/image_export.py to export Landsat OpenET PTJPL data in chips to a bucket.
    2. Use landsat_extract.py to get spatial stats for the Landsat EToF/et_fraction data.

    The following function takes the results of the extract scripts and puts them in a format
    like the Earth Engine-based data, so they can be combined.

    """


    with open(ecostress_extract, 'r') as f:
        ecostress_data = json.load(f)

    with open(landsat_extract, 'r') as f:
        landsat_data = json.load(f)

    interpolated_series_all_sites = {}

    valid_sites = set(landsat_data.keys()) | set(ecostress_data.keys())

    era5_eto_df = load_era5_eto_data(eto_extract)

    for sid in sites:

        if sid not in valid_sites:
            continue

        site_et_fractions = {}

        if sid in landsat_data:
            for date_str, value in landsat_data[sid].items():
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                site_et_fractions[dt] = value

        if sid in ecostress_data and sid in era5_eto_df.columns:
            site_ecostress_data = ecostress_data[sid]
            site_eto_series = era5_eto_df[sid]

            for date_str, w_m2_value in site_ecostress_data.items():
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if dt in site_eto_series.index:
                    eto_value = site_eto_series[dt]
                    if pd.notna(eto_value) and eto_value > 0:
                        et_fraction = (w_m2_value * 0.03527) / eto_value
                        site_et_fractions[dt] = et_fraction

        if not site_et_fractions:
            interpolated_series_all_sites[sid] = pd.Series(dtype='float64')
            continue

        et_fraction_series = pd.Series(site_et_fractions).sort_index()

        if not et_fraction_series.empty:
            start_date = et_fraction_series.index.min()
            end_date = et_fraction_series.index.max()
            daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
            et_fraction_series = et_fraction_series.reindex(daily_index)

            interpolated_series = et_fraction_series.interpolate(method='time')
            interpolated_series_all_sites[sid] = interpolated_series
        else:
            interpolated_series_all_sites[sid] = pd.Series(dtype='float64')

    return interpolated_series_all_sites


def prep_remote_sensing_output():
    from prep.remote_sensing import sparse_time_series, join_remote_sensing

    sensing_params = ['etf', 'ndvi']

    model = 'ptjpl'
    rs_files = []

    ee_data, src, src_ct, mask = None, None, None, 'no_mask'

    for sensing_param in sensing_params:

        yrs = [x for x in range(1987, 2025)]

        if sensing_param == 'etf':

            ee_data = os.path.join(landsat, 'extracts', f'{model}_{sensing_param}', mask)
            src = os.path.join(tables, '{}_{}_{}.csv'.format(model, sensing_param, mask))
            src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(model, sensing_param, mask))

        else:
            ee_data = os.path.join(landsat, 'extracts', sensing_param, mask)
            src = os.path.join(tables, '{}_{}.csv'.format(sensing_param, mask))
            src_ct = os.path.join(tables, '{}_{}_ct.csv'.format(sensing_param, mask))

        rs_files.extend([src, src_ct])
        sparse_time_series(shapefile_path, ee_data, yrs, src, src_ct,
                           feature_id=FEATURE_ID, select=sites)

    join_remote_sensing(rs_files, remote_sensing_file)


def prep_field_properties():
    from prep.field_properties import write_field_properties

    modis_lulc = os.path.join(data, 'properties', 'calibration_lulc.csv')
    properties_json = os.path.join(data, 'properties', 'calibration_properties.json')

    flux_metadata = os.path.join(data, 'station_metadata.csv')

    write_field_properties(shapefile_path, js=properties_json, lulc=modis_lulc, index_col=FEATURE_ID,
                           flux_meta=flux_metadata)


def prep_snow():
    from data_extraction.snodas.snodas import create_timeseries_json

    create_timeseries_json(snow_extract, snow_out, feature_id=FEATURE_ID)


def prep_timeseries():
    from prep.field_timeseries import join_daily_timeseries

    fields_gridmet = os.path.join(data, 'gis', 'flux_fields_gfid.shp')
    met = os.path.join(data, 'met_timeseries')

    # process irr/inv_irr of all rs parameters, incl. NDVI
    remote_sensing_parameters = get_ensemble_parameters()

    join_daily_timeseries(fields=fields_gridmet,
                          gridmet_dir=met,
                          landsat_table=remote_sensing_file,
                          snow=snow_out,
                          dst_dir=joined_timeseries,
                          overwrite=True,
                          start_date='1987-01-01',
                          end_date='2024-12-31',
                          feature_id=FEATURE_ID,
                          **{'params': remote_sensing_parameters,
                             'target_fields': sites})


def prep_dynamics():
    from prep.dynamics import SamplePlotDynamics

    dynamics = SamplePlotDynamics(joined_timeseries, irr, irr_threshold=0.3, etf_target='ssebop',
                                  out_json_file=dyanmics_data, select=sites)
    dynamics.analyze_groundwater_subsidy()
    dynamics.analyze_irrigation(lookback=5)
    dynamics.analyze_k_parameters()
    dynamics.save_json()


def prep_input_json():
    from prep.prep_plots import prep_fields_json

    params = get_ensemble_parameters()
    prep_fields_json(properties, joined_timeseries, dyanmics_data,
                     prepped_input, target_plots=sites, rs_params=params)


if __name__ == '__main__':
    prep_remote_sensing_input()
    # prep_remote_sensing_output()
    # prep_field_properties()
    # prep_snow()
    # prep_timeseries()
    # prep_dynamics()
    # prep_input_json()
# ========================= EOF ====================================================================
