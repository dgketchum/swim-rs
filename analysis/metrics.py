import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def compare_etf_estimates(combined_output_path, flux_data_path, irr=None, target='et'):
    """
    Calculates RMSE and R-squared for SWIM and SSEBop, returning daily,
    overpass-filtered daily, and monthly results.
    """
    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)
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
        df['ssebop'] = df['eto'] * df['etf']
        df['swim'] = df['eto'] * df['kc_act']
        df['flux'] = flux_data['ET']
    elif target == 'etf':
        df['ssebop'] = df['etf']
        df['swim'] = df['kc_act']
        df['flux'] = df['EToF']
    else:
        raise NotImplementedError

    # Daily Results (All Days)
    df_daily = df.dropna()
    results_daily = {}
    try:
        results_daily['rmse_swim'] = np.sqrt(mean_squared_error(df_daily['flux'], df_daily['swim']))
        results_daily['r2_swim'] = r2_score(df_daily['flux'], df_daily['swim'])
        results_daily['rmse_ssebop'] = np.sqrt(mean_squared_error(df_daily['flux'], df_daily['ssebop']))
        results_daily['r2_ssebop'] = r2_score(df_daily['flux'], df_daily['ssebop'])
        results_daily['n_samples'] = df_daily.shape[0]
    except ValueError:
        results_daily = None

    # Overpass Daily Results
    df_overpass = df_daily[df_daily['capture'] == 1].copy()
    results_overpass = {}
    if df_overpass.shape[0] >= 2:
        try:
            results_overpass['rmse_swim'] = np.sqrt(mean_squared_error(df_overpass['flux'], df_overpass['swim']))
            results_overpass['r2_swim'] = r2_score(df_overpass['flux'], df_overpass['swim'])
            results_overpass['rmse_ssebop'] = np.sqrt(mean_squared_error(df_overpass['flux'], df_overpass['ssebop']))
            results_overpass['r2_ssebop'] = r2_score(df_overpass['flux'], df_overpass['ssebop'])
            results_overpass['n_samples'] = df_overpass.shape[0]
        except ValueError:
            results_overpass = None
    else:
        results_overpass = None

    # Monthly Results
    df_monthly = df.copy()
    df_monthly['day_count'] = 1
    ct = df_monthly[['day_count']].resample('ME').sum()
    first_month = ct.index.min() - pd.tseries.offsets.Day(31)
    last_month = ct.index.max() + pd.tseries.offsets.Day(31)
    full_date_range = pd.date_range(start=first_month, end=last_month, freq='ME')
    ct = ct.reindex(full_date_range).fillna(0.0)
    ct = ct.resample('d').bfill()
    ct = ct.rename(columns={'day_count': 'month_count'})
    df_monthly = pd.concat([df_monthly, ct], axis=1).dropna()

    df_monthly = df_monthly[df_monthly['month_count'] >= 20].drop(columns=['day_count', 'month_count'])

    if target == 'et':
        df_monthly = df_monthly.resample('ME').sum()
    else:
        df_monthly = df_monthly.resample('ME').mean()
    df_monthly = df_monthly.replace([float('inf'), float('-inf')], pd.NA).dropna()

    results_monthly = {}
    try:
        results_monthly['rmse_swim'] = np.sqrt(mean_squared_error(df_monthly['flux'], df_monthly['swim']))
        results_monthly['r2_swim'] = r2_score(df_monthly['flux'], df_monthly['swim'])
        results_monthly['rmse_ssebop'] = np.sqrt(mean_squared_error(df_monthly['flux'], df_monthly['ssebop']))
        results_monthly['r2_ssebop'] = r2_score(df_monthly['flux'], df_monthly['ssebop'])
        results_monthly['n_samples'] = df_monthly.shape[0]
    except ValueError:
        results_monthly = None

    return results_daily, results_overpass, results_monthly


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
