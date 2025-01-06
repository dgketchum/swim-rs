import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots
from viz.swim_timeseries import plot_swim_timeseries


def run_fields(ini_path, project_ws, selected_feature, output_csv):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)
    fields.output = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    out_df = fields.output[selected_feature].copy()

    in_df = fields.input_to_dataframe(selected_feature)

    df = pd.concat([out_df, in_df], axis=1, ignore_index=False)
    df = df.loc[config.start_dt:config.end_dt]

    df.to_csv(output_csv)
    print(df.shape)


def compare_etf_estimates(combined_output_path, flux_data_path, irr=False):
    """"""
    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)['EToF']

    output = pd.read_csv(combined_output_path, index_col=0)
    output.index = pd.to_datetime(output.index)

    if irr:
        etf, ct = 'etf_irr', 'etf_irr_ct'
    else:
        etf, ct = 'etf_inv_irr', 'etf_inv_irr_ct'

    df = pd.DataFrame({'kc_act': output['kc_act'],
                       'etf': output[etf],
                       'ct': output[ct],
                       'EToF': flux_data})

    # filter for days that have a SSEBop ETf retrieval and a flux observation
    df = df.dropna()
    df = df.loc[df['ct'] == 1]

    # Calculate RMSE and R-squared
    rmse_kc_act = np.sqrt(mean_squared_error(df['EToF'], df['kc_act']))
    r2_kc_act = r2_score(df['EToF'], df['kc_act'])

    rmse_ssebop = np.sqrt(mean_squared_error(df['EToF'], df['etf']))
    r2_ssebop = r2_score(df['EToF'], df['etf'])

    print(f"SWIM Kc_act vs. Flux EToF: RMSE = {rmse_kc_act:.2f}, R-squared = {r2_kc_act:.2f}")
    print(f"SSEBop ETf vs. Flux EToF: RMSE = {rmse_ssebop:.2f}, R-squared = {r2_ssebop:.2f}")


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project_ws = os.path.join(root, 'tutorials', '2_Fort_Peck')
    # project_ws = os.path.join(root, 'tutorials', '3_Crane')

    data = os.path.join(project_ws, 'data')

    config_file = os.path.join(data, 'tutorial_config.toml')
    prepped_input = os.path.join(data, 'prepped_input.json')

    selected_feature = 'US-FPe'
    irr = False

    # selected_feature = 'S2'
    # irr = True

    out_csv = os.path.join(project_ws, 'step_2_uncalibrated_model', f'combined_output_{selected_feature}.csv')

    run_fields(config_file, project_ws, selected_feature=selected_feature, output_csv=out_csv)

    # plot_swim_timeseries(out_csv, ['et_act', 'etref', 'rain', 'melt', 'irrigation'],
    #                      start='2021-01-01', end='2021-12-31', png_file='et.png')

    flux_data = os.path.join(data, f'{selected_feature}_daily_data.csv')
    compare_etf_estimates(out_csv, flux_data, irr=False)

# ========================= EOF ====================================================================
