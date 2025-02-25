import os

import numpy as np
import pandas as pd

from analysis.metrics import compare_etf_estimates
from model.etd.tracker import TUNABLE_PARAMS, SampleTracker
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots

from run.run_tutorial import run_fields

def compare_results(conf_path, project_ws, result_csv_dir, mode, summary_csv, cal_dir=None, select=None,
                    updates=None):
    """
    Compares results for different model modes, creating a summary table.
    Now handles daily, overpass, and monthly results separately.
    """
    config = ProjectConfig()
    config.read_config(conf_path, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    data_dir = os.path.join(project_ws, 'data')
    flux_meta_csv = os.path.join(data_dir, 'station_metadata.csv')
    flux_meta_df = pd.read_csv(flux_meta_csv, header=1, skip_blank_lines=True, index_col='Site ID')

    irr = fields.input['irr_data']

    results_list = []
    fcst_file = None

    for fid, row in flux_meta_df.iterrows():

        if fid != 'US-Blo':
            continue

        out_csv, updated, fcst_file = None, False, None

        if select and fid not in select:
            continue

        pdc_ct = 0

        irr_dct = irr[fid]
        flux_data = os.path.join(data_dir, 'daily_flux_files', f'{fid}_daily_data.csv')
        if not os.path.exists(flux_data):
            flux_data = os.path.join(data_dir, f'{fid}_daily_data.csv')
            if not os.path.exists(flux_data):
                print(f'WARNING: file {fid} not found')
                continue

        if not mode == 'uncal':

            if updates:
                fcst_file = os.path.join(updates, mode, fid, f'{fid}.3.par.csv')
                if not os.path.exists(fcst_file):
                    fcst_file = os.path.join(updates, mode, fid, f'{fid}.2.par.csv')

                if os.path.exists(fcst_file):
                    input_file = os.path.join(updates, mode, fid, f'prepped_input_{fid}.json')
                    output_dir = os.path.join(updates, mode, fid)
                    out_csv = os.path.join(output_dir, f'{fid}.csv')
                    if not os.path.exists(out_csv):
                        print(f'\nRunning inference on updated parameters at {fid}')
                        run_fields(config_file, project_ws, output_dir, forecast=True, forecast_file=fcst_file,
                                   input_data=input_file)
                    else:
                        print(f'\nInference on updated parameters at {fid} exists, skipping new run')
                    updated = True
                else:
                    fcst_file = None

            if not fcst_file:
                fcst_file = os.path.join(project_ws, f'{mode_}_master', f'{os.path.basename(project_ws)}.3.par.csv')
                if not os.path.exists(fcst_file):
                    fcst_file = os.path.join(project_ws, f'{mode_}_master', f'{os.path.basename(project_ws)}.2.par.csv')

                pdc_file = os.path.join(cal_dir, f'{os.path.basename(project_ws)}.pdc.csv')

                if os.path.exists(pdc_file):
                    pdc = pd.read_csv(pdc_file, index_col=0)
                    pdc_ct = pdc.shape[0]

        if not out_csv:
            updated = False
            out_csv = os.path.join(result_csv_dir, f'{fid}.csv')
            if not os.path.exists(out_csv):
                print(f"WARNING: Model output file not found for {fid}")
                continue

        print(f'\nProcessing {fid} for mode: {mode}')

        daily_results, overpass_results, monthly_results = compare_etf_estimates(
            out_csv, flux_data, irr=irr_dct, target='et'
        )

        site_results = {'fid': fid, 'mode': mode, 'lulc': row['General classification'],
                        'pdc': pdc_ct, 'updated': updated}

        if daily_results:
            rmse_diff_daily = ((daily_results['rmse_ssebop'] - daily_results['rmse_swim']) /
                               daily_results['rmse_ssebop']) * 100 if daily_results['rmse_ssebop'] != 0 else np.nan

            site_results.update({
                'daily_rmse_swim': daily_results['rmse_swim'],
                'daily_rmse_ssebop': daily_results['rmse_ssebop'],
                'daily_rmse_diff_pct': rmse_diff_daily,
                'daily_r2_swim': daily_results['r2_swim'],
                'daily_r2_ssebop': daily_results['r2_ssebop'],
                'daily_n_samples': daily_results['n_samples']
            })
        else:
            site_results.update({
                'daily_rmse_swim': np.nan,
                'daily_rmse_ssebop': np.nan,
                'daily_rmse_diff_pct': np.nan,
                'daily_r2_swim': np.nan,
                'daily_r2_ssebop': np.nan,
                'daily_n_samples': np.nan
            })

        if overpass_results:
            rmse_diff_overpass = ((overpass_results['rmse_ssebop'] - overpass_results['rmse_swim']) /
                                  overpass_results['rmse_ssebop']) * 100 if overpass_results[
                                                                                'rmse_ssebop'] != 0 else np.nan

            site_results.update({
                'overpass_rmse_swim': overpass_results['rmse_swim'],
                'overpass_rmse_ssebop': overpass_results['rmse_ssebop'],
                'overpass_rmse_diff_pct': rmse_diff_overpass,
                'overpass_r2_swim': overpass_results['r2_swim'],
                'overpass_r2_ssebop': overpass_results['r2_ssebop'],
                'overpass_n_samples': overpass_results['n_samples']
            })
        else:
            site_results.update({
                'overpass_rmse_swim': np.nan,
                'overpass_rmse_ssebop': np.nan,
                'overpass_rmse_diff_pct': np.nan,
                'overpass_r2_swim': np.nan,
                'overpass_r2_ssebop': np.nan,
                'overpass_n_samples': np.nan
            })

        if monthly_results:
            rmse_diff_monthly = ((monthly_results['rmse_ssebop'] - monthly_results['rmse_swim']) /
                                 monthly_results['rmse_ssebop']) * 100 if monthly_results[
                                                                              'rmse_ssebop'] != 0 else np.nan
            site_results.update({
                'monthly_rmse_swim': monthly_results['rmse_swim'],
                'monthly_rmse_ssebop': monthly_results['rmse_ssebop'],
                'monthly_rmse_diff_pct': rmse_diff_monthly,
                'monthly_r2_swim': monthly_results['r2_swim'],
                'monthly_r2_ssebop': monthly_results['r2_ssebop'],
                'monthly_n_samples': monthly_results['n_samples']
            })
        else:
            site_results.update({
                'monthly_rmse_swim': np.nan,
                'monthly_rmse_ssebop': np.nan,
                'monthly_rmse_diff_pct': np.nan,
                'monthly_r2_swim': np.nan,
                'monthly_r2_ssebop': np.nan,
                'monthly_n_samples': np.nan
            })
        results_list.append(site_results)

    df_results = pd.DataFrame(results_list)
    df_results = df_results.set_index(['fid', 'mode'])

    if fcst_file:
        param_dist = pd.read_csv(fcst_file, index_col=0)
        param_mean = param_dist.mean(axis=0)
        param_std = param_dist.std(axis=0)
        p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) for s in list(param_mean.index)]
        param_mean.index = p_str
        param_std.index = p_str

        for p_string in param_mean.index:
            param_found = False
            while not param_found:
                for p in TUNABLE_PARAMS:
                    if p in p_string:
                        group = p
                        _fid = p_string.replace(f'{group}_', '')
                        param_found = True

            fid = [f for f in flux_meta_df.index if f.lower() == _fid][0]

            df_results.loc[(fid, mode), f'{group}_mean'] = param_mean[p_string]
            df_results.loc[(fid, mode), f'{group}_std'] = param_std[p_string]

    else:
        size = len(fields.input['order'])

        tracker = SampleTracker(size)
        tracker.load_soils(fields)
        tracker.apply_parameters(config, fields, params=None)
        for fid in fields.input['order']:
            for p in TUNABLE_PARAMS:
                df_results.loc[(fid, mode), f'{p}_mean'] = tracker.__getattribute__(p)[0, 0]
                df_results.loc[(fid, mode), f'{p}_std'] = np.nan

    if os.path.exists(summary_csv):
        df_existing = pd.read_csv(summary_csv)
        df_existing = df_existing.set_index(['fid', 'mode'])
        df_combined = pd.merge(df_existing, df_results, on=['fid', 'mode'], how='outer', suffixes=('_old', ''))

        for col in df_combined.columns:
            if col.endswith('_old'):
                original_col = col.replace('_old', '')
                df_combined[original_col] = df_combined[original_col].fillna(df_combined[col])
                df_combined.drop(columns=[col], inplace=True)

        cols = ['fid', 'mode', 'lulc', 'pdc', 'updated']
        for prefix in ['monthly_', 'daily_', 'overpass_']:
            cols.extend([f'{prefix}rmse_swim', f'{prefix}rmse_ssebop', f'{prefix}rmse_diff_pct',
                         f'{prefix}r2_swim', f'{prefix}r2_ssebop', f'{prefix}n_samples'])
        for param in TUNABLE_PARAMS:
            cols.extend([f'{param}_mean', f'{param}_std'])

        cols = [col for col in cols if col in df_combined.columns]
        df_combined = df_combined[cols]
        df_combined = df_combined.reset_index()
        df_combined.to_csv(summary_csv, index=False)

    else:
        df_results = df_results.sort_index(level=['fid', 'mode'],
                                           ascending=[True, True],
                                           sort_remaining=True)

        cols = ['fid', 'mode', 'lulc', 'pdc', 'updated']
        for prefix in ['monthly_', 'daily_', 'overpass_']:
            cols.extend([f'{prefix}rmse_swim', f'{prefix}rmse_ssebop', f'{prefix}rmse_diff_pct',
                         f'{prefix}r2_swim', f'{prefix}r2_ssebop', f'{prefix}n_samples'])
        for param in TUNABLE_PARAMS:
            cols.extend([f'{param}_mean', f'{param}_std'])

        cols = [col for col in cols if col in df_results.columns]
        df_results = df_results[cols]
        df_results = df_results.reset_index()
        df_results.to_csv(summary_csv, index=False)

    print(f"\nResults saved to: {summary_csv}")


def insert_blank_rows(csv, bad_data_csv=None):
    df = pd.read_csv(csv)
    new_df = pd.DataFrame()
    fids = df['fid'].unique()
    bad_df =  pd.DataFrame()

    for i, fid in enumerate(fids):
        subset = df[df['fid'] == fid]

        all_negative = True
        for time_scale in ['daily', 'monthly', 'overpass']:
            col_name = f'{time_scale}_rmse_diff_pct'
            if col_name in subset.columns:
                if not (subset[col_name].isna() | (subset[col_name] <= 0)).all():
                    all_negative = False
                    break
            else:
                all_negative = True
                break

        if all_negative:
            bad_df = pd.concat([bad_df, subset])

        new_df = pd.concat([new_df, subset])
        if i < len(fids) - 1:
            blank_row = pd.DataFrame({col: [None] for col in df.columns})
            new_df = pd.concat([new_df, blank_row])

    if bad_data_csv:
        bad_df.to_csv(bad_data_csv, index=False)
        print(f'Bad FIDs saved to {bad_data_csv}')

    new_df.to_csv(csv, index=False)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    # project = 'alarc_test'
    # 'ALARC2_Smith6'

    project_ws_ = os.path.join(root, 'tutorials', project)
    summary = os.path.join(project_ws_, 'results_comparison.csv')

    for mode_ in ['tight', 'loose', 'uncal']:

        data_ = os.path.join(project_ws_, 'data')

        config_file = os.path.join(project_ws_, 'config.toml')
        prepped_input = os.path.join(data_, 'prepped_input.json')

        if mode_ == 'uncal':
            forecast_ = False
            calibration_dir = None
            update_params = None
        else:
            ct = 3
            forecast_ = True
            update_params = '/data/ssd2/swim/4_Flux_Network/results'
            calibration_dir = os.path.join(project_ws_, f'{mode_}_master')


        out_csv_dir = os.path.join(data_, f'model_output_{mode_}')
        if not os.path.exists(out_csv_dir):
            os.mkdir(out_csv_dir)

        # run_fields(config_file, project_ws_, out_csv_dir, forecast=forecast_, calibrate=False,
        #            calibration_dir=calibration_dir)

        # open properties instead of SamplePlots object for speed
        properties_json = os.path.join(data_, 'landsat', 'calibration_dynamics.json')
        compare_results(config_file, project_ws_, out_csv_dir, summary_csv=summary, select=None,
                        cal_dir=calibration_dir, mode=mode_, updates=update_params)

    insert_blank_rows(summary, bad_data_csv=summary.replace('.csv', '_bad.csv'))

# ========================= EOF ====================================================================
