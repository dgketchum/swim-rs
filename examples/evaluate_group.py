import collections
import os
from pprint import pprint
from datetime import datetime

import pandas as pd

from swimrs.analysis.metrics import compare_etf_estimates
from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.prep import get_flux_sites
from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def compare_openet(fid, flux_file, model_output, openet_dir, plot_data_, model='ssebop',
                   return_comparison=False, gap_tolerance=5, ssebop_eto_source='eto_corr'):

    openet_daily = os.path.join(openet_dir, 'daily_data', f'{fid}.csv')
    openet_monthly = os.path.join(openet_dir, 'monthly_data', f'{fid}.csv')
    irr_ = plot_data_.input['irr_data'][fid]
    daily, overpass, monthly = compare_etf_estimates(model_output, flux_file, openet_daily_path=openet_daily,
                                                     openet_monthly_path=openet_monthly, irr=irr_, target_model=model,
                                                     gap_tolerance=gap_tolerance, ssebop_eto_source=ssebop_eto_source)

    if monthly is None:
        return None

    agg_comp = monthly.copy()

    rmse_all = {k.split('_', 1)[1]: v for k, v in agg_comp.items() if k.startswith('rmse_')}

    if len(rmse_all) == 0:
        return None

    best_overall_model = min(rmse_all, key=rmse_all.get)

    best_pair_model = None
    if 'rmse_swim' in agg_comp and 'rmse_openet' in agg_comp:
        best_pair_model = 'swim' if agg_comp['rmse_swim'] <= agg_comp['rmse_openet'] else 'openet'

    n_samples_ = agg_comp.get('n_samples')
    print(f"n Samples: {n_samples_}")
    print('Best overall:', best_overall_model)
    print('Best swim vs openet:', best_pair_model if best_pair_model else 'NA')

    if not return_comparison:
        return best_overall_model, best_pair_model, agg_comp

    if return_comparison:

        if len(agg_comp) == 0:
            print(fid, 'empty')
            return None

        try:

            if model == 'openet':
                print(f"Flux Mean: {agg_comp['mean_flux']}")
                print(f"SWIM Mean: {agg_comp['mean_swim']}")
                print(f"{best_overall_model} Mean: {agg_comp[f'mean_{best_overall_model}']}")
                print(f"OpenET Mean: {agg_comp['mean_openet']}")
                print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
                print(f"{best_overall_model} RMSE: {agg_comp[f'rmse_{best_overall_model}']}")
                print(f"OpenET RMSE: {agg_comp['rmse_openet']}")

            elif model == 'ssebop':
                print(f"Flux Mean: {agg_comp['mean_flux']}")
                print(f"SWIM Mean: {agg_comp['mean_swim']}")
                # print(f"{lowest_rmse_model} Mean: {agg_comp[f'mean_{lowest_rmse_model}']}")
                print(f"SSEBop NHM Mean: {agg_comp['mean_ssebop']}")
                print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
                print(f"{best_overall_model} RMSE: {agg_comp[f'rmse_{best_overall_model}']}")
                print(f"SSEBop NHM RMSE: {agg_comp['rmse_ssebop']}")

            return best_overall_model, best_pair_model, agg_comp

        except KeyError as exc:
            print(fid, exc)
            return None

    # print('\n')


if __name__ == '__main__':

    """"""
    # project = '4_Flux_Network'
    project = '5_Flux_Ensemble'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    if project == '5_Flux_Ensemble':
        western_only = False
        model_ = 'openet'

    else:
        western_only = False
        model_ = 'ssebop'

    # target_dir = os.path.join(config.project_ws, 'multi_test')
    target_dir = os.path.join(config.project_ws, 'diy_ensemble')

    config.forecast_parameters_csv = os.path.join(target_dir, f'{project}.3.par.csv')
    config.spinup = os.path.join(target_dir, f'spinup_{project}.json')
    station_prepped_input = os.path.join(target_dir, f'prepped_input.json')

    open_et_ = os.path.join(config.data_dir, 'openet_flux')
    flux_dir = os.path.join(config.data_dir, 'daily_flux_files')
    station_metadata_csv = '/data/ssd2/swim/5_Flux_Ensemble/data/station_metadata.csv'


    ec_sites, sdf = get_flux_sites(station_metadata_csv, crop_only=True,
                                   return_df=True, western_only=western_only, header=1)

    incomplete, complete, results_overall, results_pair = [], [], [], []

    plots_ = SamplePlots()
    plots_.initialize_plot_data(config)

    config.calibrate = False
    config.forecast = True
    config.read_forecast_parameters()

    modified_date = datetime.fromtimestamp(os.path.getmtime(config.forecast_parameters_csv))
    print(f'Calibration made {modified_date}')

    df_dct = field_day_loop(config, plots_, debug_flag=True)

    sites = [k for k, v in df_dct.items() if k in ec_sites]

    print(f'{len(sites)} sites to evalutate in {project}')

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        # if lulc != 'Croplands':
        #     continue

        if site_ not in ec_sites:
            continue

        # unresolved data problems
        # if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114', 'MB_Pch']:
        #     continue

        # testing sites
        # if site_ not in ['S2']:  # 'B_01', 'ALARC2_Smith6', 'S2', 'MR', 'US-FPe'
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
            best_overall, best_pair, monthly_ = result
            results_overall.append((best_overall, lulc))
            if best_pair:
                results_pair.append((best_pair, lulc))

        complete.append(site_)
        out_fig_dir_ = os.path.join(home, 'Downloads', project, 'figures', 'model_output', 'png')

        # flux_pdc_timeseries(target_dir, flux_dir, [site_], out_fig_dir=out_fig_dir_, spec='flux', model=model_,
        #                     members=['ssebop', 'ptjpl', 'sims'])

    pprint({s: [t[0] for t in results_overall].count(s) for s in set(t[0] for t in results_overall)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results_overall
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results_overall)})
    pprint({s: [t[0] for t in results_pair].count(s) for s in set(t[0] for t in results_pair)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results_pair
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results_pair)})
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
# ========================= EOF ====================================================================
