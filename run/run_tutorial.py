import os
import time

import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


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


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    project = '4_Flux_Network'

    project_ws_ = os.path.join(root, 'tutorials', project)

    data_ = os.path.join(project_ws_, 'data')
    out_csv_dir = os.path.join(data_, 'model_output')

    config_file = os.path.join(project_ws_, 'config.toml')
    prepped_input = os.path.join(data_, 'prepped_input.json')

    run_fields(config_file, project_ws_, out_csv_dir, forecast=True, calibrate=False)

# ========================= EOF ====================================================================
