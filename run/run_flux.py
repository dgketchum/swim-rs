import os
import time
from pprint import pprint

import pandas as pd

from model import obs_field_cycle
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from analysis.metrics import compare_etf_estimates
from viz.swim_timeseries import flux_pdc_timeseries


def initialize_data(ini_path, project_ws, input_data=None, spinup_data=None, calibration_dir=None,
                    forecast=False, calibrate=False, forecast_file=None):
    config = ProjectConfig()
    config.read_config(ini_path, project_ws, calibration_dir=calibration_dir, forecast=forecast,
                       calibrate=calibrate, forecast_param_csv=forecast_file)

    if input_data:
        config.input_data = input_data

    if spinup_data:
        config.spinup = spinup_data

    plots_ = SamplePlots()
    plots_.initialize_plot_data(config)

    return config, plots_


def run_flux_sites(fid, config, plot_data, outfile):
    start_time = time.time()

    df_dct = obs_field_cycle.field_day_loop(config, plot_data, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    df = df_dct[fid].copy()
    in_df = plot_data.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)

    df = df.loc[config.start_dt:config.end_dt]

    df.to_csv(outfile)


def compare_openet(fid, flux_file, model_output, openet_dir, plot_data_):
    openet_daily = os.path.join(openet_dir, 'daily_data', f'{fid}.csv')
    openet_monthly = os.path.join(openet_dir, 'monthly_data', f'{fid}.csv')
    irr_ = plot_data_.input['irr_data'][fid]
    daily, overpass, monthly = compare_etf_estimates(model_output, flux_file, openet_daily_path=openet_daily,
                                                     openet_monthly_path=openet_monthly, irr=irr_, target='et')
    print('\nMonthly\n')
    pprint(monthly)
    print('\nDaily\n')
    pprint(daily)
    print('\nOverpass\n')
    pprint(overpass)
    print('\n')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    site_ = 'ALARC2_Smith6'
    constraint_ = 'tight'

    project_ws_ = os.path.join(root, 'tutorials', project)

    data_ = os.path.join(project_ws_, 'data')
    config_file = os.path.join(project_ws_, 'config.toml')

    run_data = os.path.join(root, 'tutorials')

    bad_parameters = os.path.join(project_ws_, 'results_comparison_bad.csv')

    open_et_ = os.path.join(project_ws_, 'openet_flux')

    bad_df = pd.read_csv(bad_parameters, index_col=0)
    bad_stations = list(set(bad_df.index.unique().to_list()))
    tests = ['US-Ne3', 'BPHV', 'US-Tw3', 'Almond_High']

    overwrite_ = False

    for site_ in tests:

        print('\n', site_)

        run_const = os.path.join(run_data, '4_Flux_Network', 'results', constraint_)
        output_ = os.path.join(run_const, site_)

        prepped_input = os.path.join(output_, f'prepped_input.json')
        spinup_ = os.path.join(output_, f'spinup.json')
        if not os.path.exists(prepped_input):
            prepped_input = os.path.join(output_, f'prepped_input_{site_}.json')
            spinup_ = os.path.join(output_, f'spinup_{site_}.json')

        flux_dir = os.path.join(project_ws_, 'data', 'daily_flux_files')
        flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')
        fcst_params = os.path.join(output_, f'{site_}.3.par.csv')

        cal = os.path.join(project_ws_, f'{constraint_}_pest', 'mult')

        out_csv = os.path.join(output_, f'{site_}.csv')

        config_, fields_ = initialize_data(config_file, project_ws_, input_data=prepped_input, spinup_data=spinup_,
                                           calibration_dir=None, forecast=True, calibrate=False,
                                           forecast_file=fcst_params)

        if not os.path.exists(out_csv) or overwrite_:
            run_flux_sites(site_, config_, fields_, output_)

        compare_openet(site_, flux_data, out_csv, open_et_, fields_)

        out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'png')

        flux_pdc_timeseries(run_const, flux_dir, [site_], out_fig_dir=out_fig_dir_, spec='flux')

# ========================= EOF ====================================================================
