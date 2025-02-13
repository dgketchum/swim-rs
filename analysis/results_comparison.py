import os

import numpy as np
import pandas as pd

from analysis.metrics import compare_etf_estimates
from swim.config import ProjectConfig
from swim.input import SamplePlots

from run.run_tutorial import run_fields


def compare_results(conf_path, project_ws, result_csv_dir, mode, select=None):
    """
    Compares results for different model modes, creating a summary table.
    Now handles daily, overpass, and monthly results separately.
    """
    config = ProjectConfig()
    config.read_config(conf_path, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    data_dir = os.path.join(project_ws, 'data')
    flux_meta = os.path.join(data_dir, 'station_metadata.csv')
    df_meta = pd.read_csv(flux_meta, header=1, skip_blank_lines=True, index_col='Site ID')

    irr = fields.input['irr_data']

    results_list = []

    for fid, row in df_meta.iterrows():
        if row['General classification'] not in ['Croplands']:
            continue
        if select and fid not in select:
            continue

        irr_dct = irr[fid]
        flux_data = os.path.join(data_dir, 'daily_flux_files', f'{fid}_daily_data.csv')
        if not os.path.exists(flux_data):
            flux_data = os.path.join(data_dir, f'{fid}_daily_data.csv')
            if not os.path.exists(flux_data):
                print(f'WARNING: file {fid} not found')
                continue
        out_csv = os.path.join(result_csv_dir, f'{fid}.csv')
        if not os.path.exists(out_csv):
            print(f"WARNING: Model output file not found for {fid}")
            continue

        print(f'\nProcessing {fid} for mode: {mode}')

        daily_results, overpass_results, monthly_results = compare_etf_estimates(
            out_csv, flux_data, irr=irr_dct, target='et'
        )

        site_results = {'fid': fid, 'mode': mode}

        # Daily Results
        if daily_results:
            site_results.update({
                'daily_rmse_swim': daily_results['rmse_swim'],
                'daily_r2_swim': daily_results['r2_swim'],
                'daily_rmse_ssebop': daily_results['rmse_ssebop'],
                'daily_r2_ssebop': daily_results['r2_ssebop'],
                'daily_n_samples': daily_results['n_samples']
            })
        else:
            site_results.update({
                'daily_rmse_swim': np.nan,
                'daily_r2_swim': np.nan,
                'daily_rmse_ssebop': np.nan,
                'daily_r2_ssebop': np.nan,
                'daily_n_samples': np.nan
            })
        # Overpass Results
        if overpass_results:
            site_results.update({
                'overpass_rmse_swim': overpass_results['rmse_swim'],
                'overpass_r2_swim': overpass_results['r2_swim'],
                'overpass_rmse_ssebop': overpass_results['rmse_ssebop'],
                'overpass_r2_ssebop': overpass_results['r2_ssebop'],
                'overpass_n_samples': overpass_results['n_samples']
            })
        else:
            site_results.update({
                'overpass_rmse_swim': np.nan,
                'overpass_r2_swim': np.nan,
                'overpass_rmse_ssebop': np.nan,
                'overpass_r2_ssebop': np.nan,
                'overpass_n_samples': np.nan
            })

        # Monthly Results
        if monthly_results:
            site_results.update({
                'monthly_rmse_swim': monthly_results['rmse_swim'],
                'monthly_r2_swim': monthly_results['r2_swim'],
                'monthly_rmse_ssebop': monthly_results['rmse_ssebop'],
                'monthly_r2_ssebop': monthly_results['r2_ssebop'],
                'monthly_n_samples': monthly_results['n_samples']
            })
        else:
            site_results.update({
                'monthly_rmse_swim': np.nan,
                'monthly_r2_swim': np.nan,
                'monthly_rmse_ssebop': np.nan,
                'monthly_r2_ssebop': np.nan,
                'monthly_n_samples': np.nan
            })


        results_list.append(site_results)

    df_results = pd.DataFrame(results_list)
    output_csv_path = os.path.join(project_ws, 'results_comparison.csv')

    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)

        df_combined = pd.merge(df_existing, df_results, on=['fid', 'mode'], how='outer', suffixes=('_old', ''))

        for col in df_combined.columns:
            if col.endswith('_old'):
                original_col = col.replace('_old', '')
                df_combined[original_col] = df_combined[original_col].fillna(df_combined[col])
                df_combined.drop(columns=[col], inplace = True)


        mode_order = ['uncal', 'loose', 'tight']
        df_combined['mode'] = pd.Categorical(df_combined['mode'], categories=mode_order, ordered=True)
        df_combined = df_combined.sort_values(by=['fid', 'mode'])
        df_combined['mode'] = df_combined['mode'].astype(str)

        cols = ['fid', 'mode']
        for prefix in ['monthly_', 'daily_', 'overpass_']:
            cols.extend([f'{prefix}rmse_swim', f'{prefix}rmse_ssebop',
                         f'{prefix}r2_swim', f'{prefix}r2_ssebop',
                         f'{prefix}n_samples'])
        cols = [col for col in cols if col in df_combined.columns]
        df_combined = df_combined[cols]

        df_combined.to_csv(output_csv_path, index=False)

    else:
        mode_order = ['uncal', 'loose', 'tight']
        df_results['mode'] = pd.Categorical(df_results['mode'], categories=mode_order, ordered=True)
        df_results = df_results.sort_values(by=['fid', 'mode'])
        df_results['mode'] = df_results['mode'].astype(str)

        cols = ['fid', 'mode']
        for prefix in ['monthly_', 'daily_', 'overpass_']:
            cols.extend([f'{prefix}rmse_swim', f'{prefix}r2_swim', f'{prefix}n_samples',
                         f'{prefix}rmse_ssebop', f'{prefix}r2_ssebop'])
        cols = [col for col in cols if col in df_results.columns]
        df_results = df_results[cols]
        df_results.to_csv(output_csv_path, index=False)

    print(f"\nResults saved to: {output_csv_path}")



    print(f"\nResults saved to: {output_csv_path}")


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    # project = '4_Flux_Network'
    project = 'alarc_test'

    for mode_ in ['loose', 'uncal']:

        project_ws_ = os.path.join(root, 'tutorials', project)

        data_ = os.path.join(project_ws_, 'data')


        config_file = os.path.join(project_ws_, 'config.toml')
        prepped_input = os.path.join(data_, 'prepped_input.json')

        if mode_ == 'uncal':
            forecast_ = False
        else:
            forecast_ = True

        out_csv_dir = os.path.join(data_, f'model_output_{mode_}')
        if not os.path.exists(out_csv_dir):
            os.mkdir(out_csv_dir)

        # run_fields(config_file, project_ws_, out_csv_dir, forecast=forecast_, calibrate=False)

        # open properties instead of SamplePlots object for speed
        properties_json = os.path.join(data_, 'landsat', 'calibration_dynamics.json')
        compare_results(config_file, project_ws_, out_csv_dir, select=['ALARC2_Smith6'], mode=mode_)

# ========================= EOF ====================================================================
