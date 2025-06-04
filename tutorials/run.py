import os
import time
from datetime import datetime

import pandas as pd

from tutorials.calibrate_by_station import run_pest_sequence
from model import obs_field_cycle
from prep import get_flux_sites
from swim.config import ProjectConfig
from prep.prep_plots import prep_fields_json, preproc
from prep import get_flux_sites, get_ensemble_parameters
from swim.sampleplots import SamplePlots


def run_flux_sites(fid, config, overwrite_input=False):
    start_time = time.time()

    models = [config.etf_target_model] + config.etf_ensemble_members
    rs_params_ = get_ensemble_parameters(include=models)

    target_dir = os.path.join(config.project_ws, 'testrun', fid)
    station_prepped_input = os.path.join(target_dir, f'prepped_input_{fid}.json')

    if not os.path.isfile(station_prepped_input) and not overwrite_input:

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        prep_fields_json(config.properties_json, config.plot_timeseries, config.dynamics_data_json,
                         station_prepped_input, target_plots=[fid], rs_params=rs_params_,
                         interp_params=('ndvi',))

    config.input_data = station_prepped_input

    plots = SamplePlots()
    plots.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, plots, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    df = df_dct[fid].copy()
    in_df = plots.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)

    df = df.loc[config.start_dt:config.end_dt]

    a = 1


if __name__ == '__main__':
    project = '5_Flux_Ensemble'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, 'flux_ensemble.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    run_flux_sites('S2', config)
# ========================= EOF ====================================================================
