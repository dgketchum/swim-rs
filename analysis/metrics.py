import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compare_etf_estimates(combined_output_path, flux_data_path, irr=None, target='et', monthly=False):
    """
    """

    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)

    try:
        output = pd.read_csv(combined_output_path, index_col=0)
    except FileNotFoundError:
        print('Model output file not found')
        return

    # get irrigated and un-irrigated year's etf data
    output.index = pd.to_datetime(output.index)
    irr_threshold = 0.3
    irr_years = [int(k) for k, v in irr.items() if v >= irr_threshold]
    irr_index = [i for i in output.index if i.year in irr_years]

    output['etf'] = output['etf_inv_irr']
    output.loc[irr_index, 'etf'] = output.loc[irr_index, 'etf_irr']

    output['capture'] = output['etf_inv_irr_ct']
    output.loc[irr_index, 'capture'] = output.loc[irr_index, 'etf_irr_ct']

    df = pd.DataFrame({'kc_act': output['kc_act'],
                       'etf': output['etf'],
                       'EToF': flux_data['EToF'],
                       'ET_corr': flux_data['ET_corr'],
                       'capture': output['capture'],
                       'eto': output['eto_mm']})

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

    df = df.dropna()

    if monthly:
        df['count'] = 1
        ct = df[['count']].resample('ME').sum()
        first_month = ct.index.min() - pd.tseries.offsets.Day(31)
        last_month = ct.index.max() + pd.tseries.offsets.Day(31)
        full_date_range = pd.date_range(start=first_month, end=last_month, freq='ME')
        ct = ct.reindex(full_date_range)
        ct.fillna(0.0, inplace=True)
        ct = ct.resample('d').bfill()

        df.drop(columns=['count'], inplace=True)
        df = pd.concat([df, ct], axis=1, ignore_index=False)

        df = df.dropna()
        df = df[df['count'] >= 20]
        df.drop(columns=['count'])

        if target == 'et':
            df = df.resample('ME').sum()
        else:
            df = df.resample('ME').mean()

        sample_freq = 'monthly'

    else:
        sample_freq = 'daily'

    df = df.dropna()
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

    try:
        rmse_swim = np.sqrt(mean_squared_error(df['flux'], df['swim']))
        r2_swim = r2_score(df['flux'], df['swim'])

        rmse_ssebop = np.sqrt(mean_squared_error(df['flux'], df['ssebop']))
        r2_ssebop = r2_score(df['flux'], df['ssebop'])

    except ValueError as exc:
        print(f'Error: {exc}')
        return

    print(f'{df.shape[0]} {sample_freq} samples')
    print(f"SWIM vs. Flux {target}: RMSE = {rmse_swim:.2f}, R-squared = {r2_swim:.2f}")
    print(f"SSEBop vs. Flux {target}: RMSE = {rmse_ssebop:.2f}, R-squared = {r2_ssebop:.2f}\n")

    if sample_freq == 'daily':

        # filter for days that have a SSEBop ETf retrieval and a flux observation
        df = df[df['capture'] == 1]

        if df.shape[0] < 2:
            print(f'Too few overpass dates: {df.shape[0]}')
            return

        try:
            rmse_swim = np.sqrt(mean_squared_error(df['flux'], df['swim']))
            r2_swim = r2_score(df['flux'], df['swim'])

            rmse_ssebop = np.sqrt(mean_squared_error(df['flux'], df['ssebop']))
            r2_ssebop = r2_score(df['flux'], df['ssebop'])

        except ValueError as exc:
            print(f'Error: {exc}')
            return

        print(f'{df.shape[0]} overpass samples')
        print(f"SWIM vs. Flux {target}: RMSE = {rmse_swim:.2f}, R-squared = {r2_swim:.2f}")
        print(f"SSEBop vs. Flux {target}: RMSE = {rmse_ssebop:.2f}, R-squared = {r2_ssebop:.2f}\n")


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
