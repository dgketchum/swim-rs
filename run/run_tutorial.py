import os
import time

import pandas as pd

from model import obs_field_cycle
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots


def run_fields(ini_path, project_ws, output_csv, forecast=False, calibrate=False, forecast_file=None,
               input_data=None, spinup_data=None):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, project_ws, forecast=forecast,
                       calibrate=calibrate, forecast_param_csv=forecast_file)

    if input_data:
        config.input_data = input_data

    if spinup_data:
        config.spinup = spinup_data

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
        print(f'\nWrote {fid} output file to {out_csv}')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    site_ = 'ALARC2_Smith6'
    constraint_ = 'uncal'

    project_ws_ = os.path.join(root, 'tutorials', project)
    data_ = os.path.join(project_ws_, 'data')
    config_file = os.path.join(project_ws_, 'config.toml')

    project_ws_ = os.path.join(root, 'tutorials', project)
    update_dir = '/data/ssd2/swim/4_Flux_Network/results'
    if not os.path.isdir(update_dir):
        update_dir = os.path.join(project_ws_, 'results')

    output = os.path.join(update_dir, '03101708')
    prepped_input = os.path.join(output, f'prepped_input.json')
    spinup_ = os.path.join(output, f'spinup.json')
    fcst_params = os.path.join(output, '4_Flux_Network.3.par.csv')

    summary = os.path.join(output, 'results_comparison_03101708.csv')

    run_fields(config_file, project_ws_, output, forecast=True, calibrate=False, forecast_file=fcst_params,
               input_data=prepped_input, spinup_data=spinup_)

# ========================= EOF ====================================================================
