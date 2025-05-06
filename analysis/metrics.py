from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compare_etf_estimates(combined_output_path, flux_data_path, openet_daily_path=None, openet_monthly_path=None,
                          irr=None, model='ssebop', gap_tolerance=5):
    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)

    if isinstance(combined_output_path, pd.DataFrame):
        output = combined_output_path
    else:
        try:
            output = pd.read_csv(combined_output_path, index_col=0)
        except FileNotFoundError:
            print('Model output file not found')
            return None, None, None

    output.index = pd.to_datetime(output.index)
    irr_threshold = 0.3
    irr_years = [int(k) for k, v in irr.items() if k != 'fallow_years'
                 and v['f_irr'] >= irr_threshold]
    irr_index = [i for i in output.index if i.year in irr_years]

    try:
        output['etf'] = output['etf_inv_irr']
        output.loc[irr_index, 'etf'] = output.loc[irr_index, 'etf_irr']
        output['capture'] = output['etf_inv_irr_ct']
        output.loc[irr_index, 'capture'] = output.loc[irr_index, 'etf_irr_ct']

    except KeyError:
        output['etf'] = output[f'{model}_etf_inv_irr']
        output.loc[irr_index, 'etf'] = output.loc[irr_index, f'{model}_etf_irr']
        output['capture'] = output[f'{model}_etf_inv_irr_ct']
        output.loc[irr_index, 'capture'] = output.loc[irr_index, f'{model}_etf_irr_ct']

    df = pd.DataFrame({'kc_act': output['kc_act'], 'ET_corr': flux_data['ET_corr'],
                       'etf': output['etf'], 'capture': output['capture'], 'eto': output['eto_mm']})

    df['flux'] = flux_data['ET']
    df['flux_fill'] = flux_data['ET_fill']
    df.loc[np.isnan(df['flux']), 'flux'] = df.loc[np.isnan(df['flux']), 'flux_fill']
    df['flux_gapfill'] = flux_data['ET_gap'].astype(int)

    df['swim'] = df['eto'] * df['kc_act']

    df_monthly = df.copy()

    openet_rename = {'GEESEBAL_3x3': 'geesebal', 'PTJPL_3x3': 'ptjpl', 'SSEBOP_3x3': 'ssebop', 'SIMS_3x3': 'sims',
                     'EEMETRIC_3x3': 'eemetric', 'DISALEXI_3x3': 'disalexi', 'ensemble_mean_3x3': 'openet'}

    openet_models = [v for k, v in openet_rename.items()]

    if openet_daily_path:
        try:
            openet_daily = pd.read_csv(openet_daily_path, index_col='DATE', parse_dates=True)
            openet_daily = openet_daily.rename(columns=openet_rename)
            for model in openet_models:
                if model in openet_daily.columns:
                    df[model] = openet_daily[model]
                else:
                    df[model] = np.nan
        except FileNotFoundError:
            print(f"Warning: OpenET daily data file not found: {openet_daily_path}")
            for model in openet_models:
                df[model] = np.nan
    else:
        for model in openet_models:
            df[model] = np.nan

    if openet_monthly_path:
        try:
            openet_monthly = pd.read_csv(openet_monthly_path, index_col='DATE', parse_dates=True)
            openet_monthly = openet_monthly.rename(columns=openet_rename)
            idx = pd.to_datetime([d.replace(day=pd.Timestamp(d).days_in_month) for d in openet_monthly.index])
            openet_monthly.index = idx
        except FileNotFoundError:
            print(f"Warning: OpenET monthly data file not found: {openet_monthly_path}")
            openet_monthly = None
    else:
        openet_monthly = None

    df_daily = df.dropna(subset=['flux'])
    results_daily = {}
    all_models = ['swim']
    all_models += openet_models

    for model in all_models:
        if model not in df_daily.columns:
            continue

        df_temp = df_daily.dropna(subset=[model])
        if df_temp.empty:
            continue
        try:
            results_daily[f'rmse_{model}'] = float(np.sqrt(mean_squared_error(df_temp['flux'], df_temp[model])))
            results_daily[f'r2_{model}'] = r2_score(df_temp['flux'], df_temp[model])
            if model not in ['swim']:
                results_daily[f'{model}_mean'] = float(df_temp[model].mean())
            results_daily['n_samples'] = df_temp.shape[0]
        except ValueError:
            continue

    df_overpass = df_daily[df_daily['capture'] == 1].copy()
    results_overpass = {}
    for model in all_models:
        if model not in df_overpass.columns:
            continue
        df_temp = df_overpass.dropna(subset=[model])
        if df_temp.shape[0] >= 2:
            try:
                results_overpass[f'rmse_{model}'] = float(np.sqrt(mean_squared_error(df_temp['flux'], df_temp[model])))
                results_overpass[f'r2_{model}'] = r2_score(df_temp['flux'], df_temp[model])
                results_overpass['n_samples'] = df_temp.shape[0]

            except ValueError:
                continue
        else:
            continue

    df_monthly['days_in_month'] = 1
    ct = df_monthly[['days_in_month']].resample('ME').sum()
    df_monthly = df_monthly.rename(columns={'days_in_month': 'daily_obs'})

    start_date = ct.index.min() - pd.tseries.offsets.Day(31)
    end_date = ct.index.max() + pd.tseries.offsets.Day(31)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    ct = ct.reindex(full_date_range).fillna(0.0)
    ct = ct.resample('d').bfill()

    df_monthly.drop(columns=['etf'], inplace=True)
    df_monthly = pd.concat([df_monthly, ct], axis=1)
    df_monthly = df_monthly.dropna()

    month_check = df_monthly['flux_gapfill'].resample('ME').agg('sum')

    full_months = (month_check <= gap_tolerance).index
    full_months = [(i.year, i.month) for i in full_months]

    idx = [i for i in df_monthly.index if (i.year, i.month) in full_months]
    df_monthly = df_monthly.loc[idx].drop(columns=['daily_obs', 'days_in_month'])

    df_monthly = df_monthly.resample('ME').sum()
    if openet_monthly is not None:
        for model in openet_models:
            if model in openet_monthly.columns:
                df_monthly[model] = openet_monthly[model]

    # insertion of OpenET data builds zeros under flux/swim
    df_monthly.drop(columns=['capture'], inplace=True)
    df_monthly = df_monthly.dropna(axis=0, how='any')
    df_monthly = df_monthly.replace(0.0, np.nan)
    df_monthly = df_monthly.replace([float('inf'), float('-inf')], np.nan)
    df_monthly = df_monthly.dropna(axis=1, how='any')

    results_monthly = {}
    if df_monthly.shape[0] >= 2:
        missing = []
        results_monthly[f'mean_flux'] = df_monthly['flux'].mean().item()
        for model in all_models:
            try:
                results_monthly[f'rmse_{model}'] = float(
                    np.sqrt(mean_squared_error(df_monthly['flux'], df_monthly[model])))
                results_monthly[f'r2_{model}'] = r2_score(df_monthly['flux'], df_monthly[model])
                results_monthly[f'mean_{model}'] = df_monthly[model].mean().item()
                results_monthly['n_samples'] = df_monthly.shape[0]
            except (ValueError, KeyError) as exc:
                missing.append(model)

        print(f'missing results: {missing}')

    return results_daily, results_overpass, results_monthly


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
