import os
import time
from pprint import pprint

import pandas as pd

from model import obs_field_cycle
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from analysis.metrics import compare_etf_estimates
from viz.swim_timeseries import flux_pdc_timeseries


def run_flux_sites(ini_path, project_ws, flux_file, outdir, calibration_dir=None, forecast=False, calibrate=False,
                   forecast_file=None, input_data=None, spinup_data=None):

    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, project_ws, calibration_dir=calibration_dir, forecast=forecast,
                       calibrate=calibrate, forecast_param_csv=forecast_file)

    if input_data:
        config.input_data = input_data

    if spinup_data:
        config.spinup = spinup_data

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    targets = fields.input['order']

    print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')
    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    for i, fid in enumerate(targets):
        df = df_dct[fid].copy()
        in_df = fields.input_to_dataframe(fid)
        df = pd.concat([df, in_df], axis=1, ignore_index=False)
        df = df.loc[config.start_dt:config.end_dt]
        df.to_csv(os.path.join(outdir, f'{fid}.csv'))
        irr_ = fields.input['irr_data'][fid]
        daily, overpass, monthly = compare_etf_estimates(df, flux_file, irr=irr_, target='et')
        pprint(monthly)

        a = 1



if __name__ == '__main__':

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    site_ = 'ALARC2_Smith6'
    constraint_ = 'tight'

    project_ws_ = os.path.join(root, 'tutorials', project)

    data_ = os.path.join(project_ws_, 'data')
    config_file = os.path.join(project_ws_, 'config.toml')

    run_data = '/data/ssd2/swim'
    if not os.path.isdir(run_data):
        run_data = os.path.join(root, 'tutorials')

    run_const = os.path.join(run_data, '4_Flux_Network', 'results', constraint_)
    output_ = os.path.join(run_const, site_)

    prepped_input = os.path.join(output_, f'prepped_input.json')
    spinup_ = os.path.join(output_, f'spinup.json')

    flux_dir = os.path.join(project_ws_, 'data', 'daily_flux_files')
    flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')
    fcst_params = os.path.join(output_, f'{site_}.3.par.csv')

    # run_flux_sites(config_file, project_ws_, flux_data, output_, forecast=True, calibrate=False,
    #                forecast_file=fcst_params, input_data=prepped_input, spinup_data=spinup_)

    out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'png')

    flux_pdc_timeseries(run_const, flux_dir, [site_], out_fig_dir=out_fig_dir_)

# ========================= EOF ====================================================================
