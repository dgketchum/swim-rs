import collections
import os
from pprint import pprint

import pandas as pd

from analysis.metrics import compare_etf_estimates
from model import obs_field_cycle
from prep import get_flux_sites
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots


def compare_openet(fid, flux_file, model_output, openet_dir, plot_data_, model='ssebop',
                   return_comparison=False, gap_tolerance=5):

    openet_daily = os.path.join(openet_dir, 'daily_data', f'{fid}.csv')
    openet_monthly = os.path.join(openet_dir, 'monthly_data', f'{fid}.csv')
    irr_ = plot_data_.input['irr_data'][fid]
    daily, overpass, monthly = compare_etf_estimates(model_output, flux_file, openet_daily_path=openet_daily,
                                                     openet_monthly_path=openet_monthly, irr=irr_, target_model=model,
                                                     gap_tolerance=gap_tolerance)

    if monthly is None:
        return None

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

            if model == 'openet':
                print(f"Flux Mean: {agg_comp['mean_flux']}")
                print(f"SWIM Mean: {agg_comp['mean_swim']}")
                print(f"{lowest_rmse_model} Mean: {agg_comp[f'mean_{lowest_rmse_model}']}")
                print(f"OpenET Mean: {agg_comp['mean_openet']}")
                print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
                print(f"{lowest_rmse_model} RMSE: {agg_comp[f'rmse_{lowest_rmse_model}']}")
                print(f"OpenET RMSE: {agg_comp['rmse_openet']}")

            elif model == 'ssebop':
                print(f"Flux Mean: {agg_comp['mean_flux']}")
                print(f"SWIM Mean: {agg_comp['mean_swim']}")
                # print(f"{lowest_rmse_model} Mean: {agg_comp[f'mean_{lowest_rmse_model}']}")
                print(f"SSEBop NHM Mean: {agg_comp['mean_ssebop']}")
                print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
                print(f"{lowest_rmse_model} RMSE: {agg_comp[f'rmse_{lowest_rmse_model}']}")
                print(f"SSEBop NHM RMSE: {agg_comp['rmse_ssebop']}")

            return lowest_rmse_model

        except KeyError as exc:
            print(fid, exc)
            return None

    # print('\n')


if __name__ == '__main__':

    """"""
    # project = '4_Flux_Network'
    project = '5_Flux_Ensemble'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    if project == '5_Flux_Ensemble':
        western_only = True
        model_ = 'openet'

    else:
        western_only = True
        model_ = 'ssebop'

    # target_dir = os.path.join(config.project_ws, 'multi_test')
    target_dir = os.path.join(config.project_ws, 'diy_ensemble')

    config.forecast_parameters_csv = os.path.join(target_dir, f'{project}.3.par.csv')
    config.spinup = os.path.join(target_dir, f'spinup_{project}.json')
    station_prepped_input = os.path.join(target_dir, f'prepped_input.json')

    open_et_ = os.path.join(config.data_dir, 'openet_flux')
    flux_dir = os.path.join(config.data_dir, 'daily_flux_files')

    ec_sites, sdf = get_flux_sites(config.station_metadata_csv, crop_only=False,
                                return_df=True, western_only=True, header=1)

    incomplete, complete, results = [], [], []

    plots_ = SamplePlots()
    plots_.initialize_plot_data(config)

    config.calibrate = False
    config.forecast = True
    config.read_forecast_parameters()

    df_dct = obs_field_cycle.field_day_loop(config, plots_, debug_flag=True)

    sites = [k for k, v in df_dct.items() if k in ec_sites]

    print(f'{len(sites)} sites to evalutate in {project}')

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        if lulc != 'Croplands':
            continue

        if site_ not in ec_sites:
            continue

        # unresolved data problems
        if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114', 'MB_Pch']:
            continue

        # testing sites
        # if site_ in ['B_01', 'ALARC2_Smith6', 'S2', 'MR', 'US-FPe']:
        #     continue

        print(f'\n{ee} {site_}: {lulc}')

        flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')

        config.input_data = station_prepped_input

        out_csv = os.path.join(target_dir, f'{site_}.csv')

        df = df_dct[site_].copy()
        in_df = plots_.input_to_dataframe(site_)
        df = pd.concat([df, in_df], axis=1, ignore_index=False)

        df = df.loc[config.start_dt:config.end_dt]

        df.to_csv(out_csv)

        result = compare_openet(site_, flux_data, out_csv, open_et_, plots_,
                                model=config.etf_target_model, return_comparison=True, gap_tolerance=5)

        if result:
            results.append((result, lulc))

        complete.append(site_)

    pprint({s: [t[0] for t in results].count(s) for s in set(t[0] for t in results)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results)})
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
# ========================= EOF ====================================================================
