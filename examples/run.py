import os
import time

import pandas as pd

from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.swim.config import ProjectConfig
from swimrs.prep.prep_plots import prep_fields_json
from swimrs.prep import get_ensemble_parameters
from swimrs.swim.sampleplots import SamplePlots


def run_flux_sites(fid, config, target_dir,
                   station_prepped_input, overwrite_input=False):
    start_time = time.time()

    models = [config.etf_target_model]
    if config.etf_ensemble_members is not None:
        models += config.etf_ensemble_members

    rs_params_ = get_ensemble_parameters(include=models)

    target_dir = os.path.join(config.project_ws, 'testrun', fid)
    station_prepped_input = os.path.join(target_dir, f'prepped_input_{fid}.json')

    if not os.path.isfile(station_prepped_input) or overwrite_input:

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        prep_fields_json(config.properties_json, config.plot_timeseries, config.dynamics_data_json,
                         station_prepped_input, target_plots=[fid], rs_params=rs_params_,
                         interp_params=('ndvi',))

    config.input_data = station_prepped_input
    config.spinup = os.path.join(target_dir, f'spinup_{fid}.json')

    plots = SamplePlots()
    plots.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, plots, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    df = df_dct[fid].copy()
    in_df = plots.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)

    df = df.loc[config.start_dt:config.end_dt]

    out_csv = os.path.join(target_dir, f'{fid}.csv')

    df.to_csv(out_csv)

    print(f'run complete: {fid}, wrote {out_csv}')


if __name__ == '__main__':
    """"""
    project = '4_Flux_Network'
    config_filename = 'flux_network'
    western_only = False

    # project = '5_Flux_Ensemble'
    # config_filename = 'flux_ensemble'
    # western_only = True

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{config_filename}.toml')

    config_ = ProjectConfig()
    config_.read_config(config_file)

    target_dir_ = os.path.join(config_.project_ws, 'pestrun')
    station_prepped_input_ = os.path.join(target_dir_, f'prepped_input.json')

    run_flux_sites('S2', config_, target_dir=target_dir_,
                   station_prepped_input=station_prepped_input_, overwrite_input=True)
# ========================= EOF ====================================================================
