import collections
import os
import time
from datetime import datetime
from pprint import pprint

import pandas as pd

from analysis.metrics import compare_etf_estimates
from model import obs_field_cycle
from prep import get_flux_sites, get_ensemble_parameters
from prep.prep_plots import prep_fields_json
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots


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


def compare_openet(fid, flux_file, model_output, openet_dir, plot_data_, model='ssebop',
                   return_comparison=False, gap_tolerance=5):
    openet_daily = os.path.join(openet_dir, 'daily_data', f'{fid}.csv')
    openet_monthly = os.path.join(openet_dir, 'monthly_data', f'{fid}.csv')
    irr_ = plot_data_.input['irr_data'][fid]
    daily, overpass, monthly = compare_etf_estimates(model_output, flux_file, openet_daily_path=openet_daily,
                                                     openet_monthly_path=openet_monthly, irr=irr_, model=model,
                                                     gap_tolerance=gap_tolerance)

    # print('\nOverpass\n')
    # pprint(overpass)
    # print('Monthly')
    # pprint(monthly)

    agg_comp = monthly.copy()
    if len(agg_comp) < 3:
        return None

    rmse_values = {k.split('_')[1]: v for k, v in agg_comp.items() if k.startswith('rmse_')
                   if 'swim' in k or 'openet' in k}

    if len(rmse_values) == 0:
        return None

    lowest_rmse_model = min(rmse_values, key=rmse_values.get)
    print(f"n Samples: {agg_comp['n_samples']}")
    print('Lowest RMSE:', lowest_rmse_model)

    if not return_comparison:
        return lowest_rmse_model

    if return_comparison:

        if len(agg_comp) == 0:
            print(fid, 'empty')
            return None

        try:
            print(f"Flux Mean: {agg_comp['mean_flux']}")
            print(f"SWIM Mean: {agg_comp['mean_swim']}")
            print(f"{lowest_rmse_model} Mean: {agg_comp[f'mean_{lowest_rmse_model}']}")
            print(f"OpenET Mean: {agg_comp['mean_openet']}")
            print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
            print(f"{lowest_rmse_model} RMSE: {agg_comp[f'rmse_{lowest_rmse_model}']}")
            print(f"OpenET RMSE: {agg_comp['rmse_openet']}")
            return lowest_rmse_model

        except KeyError as exc:
            print(fid, exc)
            return None

    # print('\n')


if __name__ == '__main__':

    """"""
    project = '4_Flux_Network'
    # project = '5_Flux_Ensemble'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    if project == '5_Flux_Ensemble':
        western_only = True
        run_const = os.path.join(config.project_ws, 'results', 'tight')

    else:
        run_const = os.path.join(config.project_ws, 'results', 'tight')
        western_only = False

    open_et_ = os.path.join(config.data_dir, 'openet_flux')
    flux_dir = os.path.join(config.data_dir, 'daily_flux_files')

    sites, sdf = get_flux_sites(config.station_metadata_csv, crop_only=False,
                                return_df=True, western_only=western_only, header=1)

    print(f'{len(sites)} sites to evalutate in {project}')
    incomplete, complete, results = [], [], []

    overwrite_ = True
    use_new_params = True

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        # if lulc != 'Croplands':
        #     continue

        if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114']:
            continue

        if site_ not in ['AFS']:
            continue

        print(f'\n{ee} {site_}: {lulc}')

        output_ = os.path.join(run_const, site_)

        flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')

        target_dir = os.path.join(config.project_ws, 'ptjpl_test', site_)
        station_prepped_input = os.path.join(target_dir, f'prepped_input_{site_}.json')

        if not os.path.isfile(station_prepped_input) or overwrite_:

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            models = [config.etf_target_model]
            if config.etf_ensemble_members is not None:
                models += config.etf_ensemble_members

            rs_params_ = get_ensemble_parameters(include=models)
            prep_fields_json(config.properties_json, config.plot_timeseries, config.dynamics_data_json,
                             out_js=station_prepped_input, target_plots=[site_], rs_params=rs_params_,
                             interp_params=('ndvi',))

        config.input_data = station_prepped_input

        out_csv = os.path.join(target_dir, f'{site_}.csv')

        try:
            plots_ = SamplePlots()
            plots_.initialize_plot_data(config)
        except FileNotFoundError:
            print(f'file {config.input_data} not found')
            continue

        # bring in forecast from previous work
        config.calibrate = False
        config.forecast = True

        if use_new_params:
            config.forecast_parameters_csv = os.path.join(target_dir, f'{site_}.3.par.csv')
            config.spinup = os.path.join(target_dir, f'spinup_{site_}.json')
        else:
            config.forecast_parameters_csv = os.path.join(output_, f'{site_}.3.par.csv')
            config.spinup = os.path.join(output_, f'spinup_{site_}.json')

        if not os.path.exists(config.spinup):
            print(f'file {config.spinup} not found')
            continue
        if not os.path.exists(config.forecast_parameters_csv):
            continue

        modified_date = datetime.fromtimestamp(os.path.getmtime(config.forecast_parameters_csv))
        print(f'Calibration made {modified_date}')
        config.read_forecast_parameters()

        # try:
        if not os.path.exists(out_csv) or overwrite_:
            run_flux_sites(site_, config, plots_, out_csv)
        # except ValueError as exc:
        #     print(f'{site_} error: {exc}')
        #     continue

        result = compare_openet(site_, flux_data, out_csv, open_et_, plots_,
                                model=config.etf_target_model, return_comparison=True, gap_tolerance=5)

        if result:
            results.append((result, lulc))

        complete.append(site_)

        # out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'model_output', 'png')

        # flux_pdc_timeseries(run_const, flux_dir, [site_], out_fig_dir=out_fig_dir_, spec='flux', model=model,
        #                     members=['ssebop', 'disalexi', 'geesebal', 'eemetric', 'ptjpl', 'sims'])

    pprint({s: [t[0] for t in results].count(s) for s in set(t[0] for t in results)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results)})
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
# ========================= EOF ====================================================================
