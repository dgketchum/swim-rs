import os
import time
import collections
from datetime import datetime
from pprint import pprint
import pandas as pd

from analysis.metrics import compare_etf_estimates
from initialize import initialize_data
from model import obs_field_cycle
from viz.swim_timeseries import flux_pdc_timeseries
from prep import get_flux_sites


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

    project = '5_Flux_Ensemble'
    # project = '4_Flux_Network'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    project_ws_ = os.path.join(root, project)
    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        project_ws_ = os.path.join(root, 'tutorials', project)
        data = os.path.join(project_ws_, 'data')

    config_file = os.path.join(project_ws_, 'config.toml')

    open_et_ = os.path.join(project_ws_, 'openet_flux')

    station_file = os.path.join(data, 'station_metadata.csv')
    sites, sdf = get_flux_sites(station_file, crop_only=False, return_df=True)

    incomplete, complete, results = [], [], []

    overwrite_ = False

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        # if lulc == 'Croplands':
        #     continue

        if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114']:
            continue

        # if site_ not in ['US-A74', 'US-Bkg', 'UA1_HartFarm', 'US-FR2', 'US-Rwf', 'JPL1_Smith5', 'US-xDL', 'KV_4',
        #                  'US-Ced', 'US-Me5', 'US-NR1', 'US-WCr', 'US-Rwe', 'UOVUP', 'US-xDS', 'UA2_KN20', 'US-IB2',
        #                  'US-SO3', 'US-Ne2', 'US-GMF', 'US-GLE', 'US-Me6', 'US-CMW', 'SLM001', 'US-xAE', 'US-Dk2',
        #                  'US-Ro5', 'UA3_KN15', 'US-Blo', 'UA1_JV187', 'US-OF4', 'US-FPe', 'US-SCs', 'BPLV', 'MB_Pch',
        #                  'US-SP3', 'US-A32', 'US-MOz', 'LYS_SE', 'US-Me1', 'US-Twt', 'Almond_Low', 'US-SO4',
        #                  'UA3_JV108', 'UOVMD', 'US-Oho', 'US-LS1', 'US-SRC', 'US-WBW', 'B_11', 'US-Me2', 'US-SP2',
        #                  'US-SRG', 'US-Br3', 'DVDV', 'US-MC1', 'US-MMS', 'BAR012', 'US-Fwf', 'US-Ro3', 'US-xST',
        #                  'US-Hn2', 'US-Var', 'ET_8', 'KV_2', 'US-Fuf', 'US-Ro2', 'US-xUN', 'MR', 'US-Bi2', 'US-Mj2',
        #                  'manilacotton', 'US-xSB', 'LYS_NW', 'US-Bi1', 'UMVW', 'US-SRS', 'US-CRT', 'US-Rws', 'US-SRM',
        #                  'US-Ro1', 'RIP760', 'S2', 'US-xJR', 'US-Aud', 'KV_1', 'US-xYE', 'ALARC2_Smith6', 'US-SP4',
        #                  'stonevillesoy', 'US-AR1', 'Almond_Med', 'UA2_JV330', 'US-KLS', 'US-Tw3', 'VR', 'US-NC4',
        #                  'US-KM4', 'US-Dk1', 'US-Dix', 'JPL1_JV114', 'US-Jo2', 'SV_5', 'US-OF2', 'US-Tw2', 'US-Goo',
        #                  'US-SdH', 'US-NC3', 'AFD', 'WRV_2', 'US-ARb', 'US-xNG', 'SPV_3', 'AFS', 'SV_6', 'US-Ro4',
        #                  'LYS_SW', 'SPV_1', 'US-Ro6', 'LYS_NE', 'US-xNW', 'US-xDC', 'US-Ctn', 'US-xSL', 'US-Skr',
        #                  'US-xRM', 'UOVLO', 'US-SCg', 'US-ARM', 'US-Srr', 'US-ARc', 'US-Sne', 'US-Esm', 'Almond_High',
        #                  'US-Br1', 'US-IB1', 'US-Fmf', 'ET_1', 'MOVAL', 'US-OF1', 'US-Slt', 'US-Ne3', 'UA1_KN18',
        #                  'B_01', 'US-Hn3', 'BPHV', 'US-NC2', 'US-ADR', 'US-CZ3', 'US-Bo1', 'WRV_1', 'US-Mj1',
        #                  'Ellendale', 'US-SO2', 'TAM', 'US-SCw', 'US-Ne1', 'US-Blk', 'US-OF6', 'US-Wkg', 'US-KS2',
        #                  'KV_2', 'ET_1', 'B_01', 'DVDV', 'JPL1_Smith5', 'BPLV', 'B_11', 'ET_8', 'JPL1_JV114', 'KV_1']:
        #     continue

        print(f'\n{ee} {site_}: {lulc}')

        run_const = os.path.join(project_ws_, 'results', 'verify')
        output_ = os.path.join(run_const, site_)

        prepped_input = os.path.join(output_, f'prepped_input.json')
        spinup_ = os.path.join(output_, f'spinup.json')

        if not os.path.exists(prepped_input):
            prepped_input = os.path.join(output_, f'prepped_input_{site_}.json')
            spinup_ = os.path.join(output_, f'spinup_{site_}.json')

        flux_dir = os.path.join(project_ws_, 'data', 'daily_flux_files')
        flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')

        fcst_params = os.path.join(output_, f'{site_}.3.par.csv')
        if not os.path.exists(fcst_params):
            continue

        modified_date = datetime.fromtimestamp(os.path.getmtime(fcst_params))
        print(f'Calibration made {modified_date}')
        if modified_date < pd.to_datetime('2025-07-01'):
            continue

        cal = os.path.join(project_ws_, f'tight_pest', 'mult')

        out_csv = os.path.join(output_, f'{site_}.csv')

        config_, fields_ = initialize_data(config_file, project_ws_, input_data=prepped_input, spinup_data=spinup_,
                                           forecast=True, forecast_file=fcst_params)

        try:
            if not os.path.exists(out_csv) or overwrite_:
                run_flux_sites(site_, config_, fields_, out_csv)
        except ValueError as exc:
            print(f'{site_} error: {exc}')
            continue

        result = compare_openet(site_, flux_data, out_csv, open_et_, fields_,
                                model='openet', return_comparison=True, gap_tolerance=5)

        if result:
            results.append((result, lulc))

        complete.append(site_)

        out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'model_output', 'png')

        # flux_pdc_timeseries(run_const, flux_dir, [site_], out_fig_dir=out_fig_dir_, spec='flux', model='openet',
        #                     members=['ssebop', 'disalexi', 'geesebal', 'eemetric', 'ptjpl', 'sims'])

    pprint({s: [t[0] for t in results].count(s) for s in set(t[0] for t in results)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results)})
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
# ========================= EOF ====================================================================
