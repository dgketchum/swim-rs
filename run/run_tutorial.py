import os
import time
import json

import pandas as pd

from analysis.metrics import compare_etf_estimates

from model.etd import obs_field_cycle
from model.etd.tracker import TUNABLE_PARAMS

from swim.config import ProjectConfig
from swim.input import SamplePlots

from calibrate.pest_builder import PestBuilder


def run_fields(ini_path, project_ws, output_csv, forecast=False, calibrate=False):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, project_ws, forecast=forecast, calibrate=calibrate)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    fields.output = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    end_time = time.time()

    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    for fid in fields.input['order']:
        out_df = fields.output[fid].copy()

        # print(f"eta mean: {out_df['et_act'].mean()}")

        in_df = fields.input_to_dataframe(fid)

        df = pd.concat([out_df, in_df], axis=1, ignore_index=False)
        df = df.loc[config.start_dt:config.end_dt]

        out_csv = os.path.join(output_csv, f'{fid}.csv')

        df.to_csv(out_csv)
        print(f'\nWrote {fid} output file')


def compare_results(conf_path, project_ws, select=None):
    """"""

    config = ProjectConfig()
    config.read_config(conf_path, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    data_dir = os.path.join(project_ws, 'data')

    flux_meta = os.path.join(data_dir, 'station_metadata.csv')
    df = pd.read_csv(flux_meta, header=1, skip_blank_lines=True, index_col='Site ID')

    irr = fields.input['irr_data']

    for fid, row in df.iterrows():

        if row['General classification'] not in ['Croplands']:
            continue

        if select and fid not in select:
            continue

        irr_dct = irr[fid]

        flux_data = os.path.join(data_dir, 'daily_flux_files', f'{fid}_daily_data.csv')
        if not os.path.exists(flux_data):
            flux_data = os.path.join(data_dir, f'{fid}_daily_data.csv')

        out_csv = os.path.join(data_dir, 'model_output', f'{fid}.csv')

        print(f'\nReading {fid} output file')

        # TODO: instantiate the SamplePlot class to get real irrigation information
        compare_etf_estimates(out_csv, flux_data, irr=irr_dct, monthly=False, target='et')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project_ws_ = os.path.join(root, 'tutorials', 'alarc_test')

    data_ = os.path.join(project_ws_, 'data')
    out_csv_dir = os.path.join(data_, 'model_output')

    config_file = os.path.join(project_ws_, 'config.toml')
    prepped_input = os.path.join(data_, 'prepped_input.json')

    run_fields(config_file, project_ws_, out_csv_dir, forecast=True, calibrate=False)

    # open properties instead of SamplePlots object for speed
    properties_json = os.path.join(data_, 'landsat', 'calibration_dynamics.json')
    compare_results(config_file, project_ws_, select=['ALARC2_Smith6'])

# ========================= EOF ====================================================================
