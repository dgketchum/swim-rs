import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compare_etf_estimates(combined_output_path, flux_data_path, irr=False):
    """"""
    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)['EToF']

    try:
        output = pd.read_csv(combined_output_path, index_col=0)
    except FileNotFoundError:
        print('Model output file not found')
        return

    output.index = pd.to_datetime(output.index)

    if irr:
        etf, ct = 'etf_irr', 'etf_irr_ct'
    else:
        etf, ct = 'etf_inv_irr', 'etf_inv_irr_ct'

    df = pd.DataFrame({'kc_act': output['kc_act'],
                       'etf': output[etf],
                       'ct': output[ct],
                       'EToF': flux_data})

    # filter for days that have a SSEBop ETf retrieval and a flux observation
    df = df.dropna()
    df = df[df['ct'] == 1]

    if df.shape[0] < 2:
        print(f'Too few overpass dates: {df.shape[0]}')
        return

    try:
        rmse_kc_act = np.sqrt(mean_squared_error(df['EToF'], df['kc_act']))
        r2_kc_act = r2_score(df['EToF'], df['kc_act'])

        rmse_ssebop = np.sqrt(mean_squared_error(df['EToF'], df['etf']))
        r2_ssebop = r2_score(df['EToF'], df['etf'])

    except ValueError as exc:
        print(f'Error: {exc}')
        return

    print(f'{df.shape[0]} overpass samples')
    print(f"SWIM Kc_act vs. Flux EToF: RMSE = {rmse_kc_act:.2f}, R-squared = {r2_kc_act:.2f}")
    print(f"SSEBop ETof vs. Flux EToF: RMSE = {rmse_ssebop:.2f}, R-squared = {r2_ssebop:.2f}")


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
