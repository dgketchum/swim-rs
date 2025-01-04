import os
import time

import pandas as pd

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
    df.to_csv(output_csv)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    config_file = os.path.join(root, 'tutorials', '1_Boulder', 'data', 'tutorial_config.toml')
    project_ws_ = os.path.join(root, 'tutorials', '1_Boulder')

    selected_feature = '043_000128'
    out_csv = os.path.join(root, 'tutorials', '1_Boulder', 'step_5_model_run',
                           f'combined_output_{selected_feature}.csv')

    run_fields(config_file, project_ws_, selected_feature=selected_feature, output_csv=out_csv)

    start = '2007-01-01'
    end = '2009-12-31'
    plot_swim_timeseries(out_csv, ['snow_fall', 'rain', 'melt', 'dperc'], start=start, end=end)

    # plot_timeseries(out_csv, ['et_act', 'etref', 'irrigation'], start=start, end=end)

# ========================= EOF ====================================================================
