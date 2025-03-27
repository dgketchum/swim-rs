import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compare_etf_estimates(combined_output_path, flux_data_path, openet_daily_path=None, openet_monthly_path=None,
                          irr=None, target='et'):
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

    output['etf'] = output['etf_inv_irr']
    output.loc[irr_index, 'etf'] = output.loc[irr_index, 'etf_irr']
    output['capture'] = output['etf_inv_irr_ct']
    output.loc[irr_index, 'capture'] = output.loc[irr_index, 'etf_irr_ct']

    df = pd.DataFrame({'kc_act': output['kc_act'], 'etf': output['etf'],
                       'EToF': flux_data['EToF'], 'ET_corr': flux_data['ET_corr'],
                       'capture': output['capture'], 'eto': output['eto_mm']})

    if target == 'et':
        df['flux'] = flux_data['ET']
    elif target == 'etf':
        df['flux'] = flux_data['EToF']
    else:
        raise NotImplementedError

    if target == 'et':
        df['ssebop'] = df['eto'] * df['etf']
        df['swim'] = df['eto'] * df['kc_act']
    elif target == 'etf':
        df['ssebop'] = df['etf']
        df['swim'] = df['kc_act']

    # set aside a monthly dataframe
    df_monthly = df.copy()

    openet_rename = {'GEESEBAL_3x3': 'GEESEBAL', 'PTJPL_3x3': 'PTJPL', 'SSEBOP_3x3': 'SSEBOP', 'SIMS_3x3': 'SIMS',
                     'EEMETRIC_3x3': 'EEMETRIC', 'DISALEXI_3x3': 'DISALEXI', 'ensemble_mean_3x3': 'openet_ensemble'}

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
    all_models = ['swim', 'ssebop']
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
            if model not in ['swim', 'ssebop']:
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
    df_monthly = pd.concat([df_monthly, ct], axis=1).dropna()

    month_check = df_monthly[['days_in_month', 'daily_obs']].resample('ME').agg({'daily_obs': 'sum',
                                                                                 'days_in_month': 'mean'})
    full_months = month_check[month_check['days_in_month'] == month_check['daily_obs']].index
    full_months = [(i.year, i.month) for i in full_months]
    idx = [i for i in df_monthly.index if (i.year, i.month) in full_months]
    df_monthly = df_monthly.loc[idx].drop(columns=['daily_obs', 'days_in_month'])

    if target == 'et':
        df_monthly = df_monthly.resample('ME').sum()
        if openet_monthly is not None:
            for model in openet_models:
                if model in openet_monthly.columns:
                    df_monthly[model] = openet_monthly[model]

    else:
        df_monthly = df_monthly.resample('ME').mean()
        if openet_monthly is not None:
            for model in openet_models:
                if model in openet_monthly.columns:
                    df_monthly[model] = openet_monthly[model]

    # insertion of OpenET data builds zeros under flux/swim
    df_monthly = df_monthly.replace(0.0, np.nan)
    df_monthly = df_monthly.replace([float('inf'), float('-inf')], np.nan).dropna(how='any', axis=0)

    results_monthly = {}
    if df_monthly.shape[0] >= 2:
        results_monthly[f'mean_flux'] = df_monthly['flux'].mean()
        for model in all_models:
            try:
                results_monthly[f'rmse_{model}'] = float(np.sqrt(mean_squared_error(df_monthly['flux'], df_monthly[model])))
                results_monthly[f'r2_{model}'] = r2_score(df_monthly['flux'], df_monthly[model])
                results_monthly[f'mean_{model}'] = df_monthly[model].mean()
                results_monthly['n_samples'] = df_monthly.shape[0]
            except (ValueError, KeyError) as exc:
                print(model, exc)
                continue

    return results_daily, results_overpass, results_monthly


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
