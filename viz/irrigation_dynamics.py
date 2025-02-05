import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def irrigation_timeseries(dynamics_json, remote_sensing_file, feature, out_dir=None):

    with open(dynamics_json, 'r') as f:
        input_dct = json.load(f)

    field = input_dct['irr'][f'{feature}']['2018']

    column, desc, color = f'ndvi_irr', f'Irrigated NDVI (Smoothed) - {feature}', 'green'

    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.read_csv(remote_sensing_file, index_col=0)

    df_year = df.loc['2018-01-01': '2018-07-31'].copy()
    df_year.index = pd.to_datetime(df_year.index)
    df_year['doy'] = [i.dayofyear for i in df_year.index]

    # Calculate 7-day rolling mean
    df_year[column + '_rolling'] = df_year[column].rolling(window=7, center=True).mean()
    df_year[column + '_rolling'].plot(ax=ax, label=desc + ' 7-day Mean', color=color)

    # df_year[column].plot(ax=ax, label=desc, color=color, alpha=0.5)  # Plot original data with reduced alpha

    ct_column = column + '_ct'
    scatter_data = df_year[column].copy()
    scatter_data[df_year[ct_column] == 0] = np.nan
    ax.scatter(scatter_data.index, scatter_data, marker='o', s=50, c=color, label='Capture Date Retrieval')

    irr_dates = [pd.to_datetime('2018-01-01') + pd.Timedelta(days=doy - 1) for doy in field['irr_doys']
                 if doy in df_year['doy'].tolist()]

    ax.scatter(irr_dates, df_year.loc[irr_dates, column + '_rolling'], marker='o', s=80, facecolors='none',
               edgecolors='blue',
               linewidth=2, label='Potential Irrigation Day')

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('NDVI Time Series for 2018')
    ax.legend()

    plt.tight_layout()
    if out_dir:
        fig_file = os.path.join(out_dir, 'irrigation_timeseries_endPlusFive.png')
        plt.savefig(fig_file)
        print(fig_file)
    else:
        plt.show()


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = 'alarc_test'
    feature_ = 'ALARC2_Smith6'

    data = os.path.join(root, 'tutorials', project, 'data')
    landsat = os.path.join(data, 'landsat')
    joined_timeseries = os.path.join(data, 'plot_timeseries', f'{feature_}_daily.csv')
    cuttings_json = os.path.join(landsat, 'calibration_dynamics.json')
    out_fig_dir = os.path.join(root, 'tutorials', project, 'figures', 'irrigation_dynamics')

    irrigation_timeseries(cuttings_json, remote_sensing_file=joined_timeseries, feature=feature_,
                          out_dir=out_fig_dir)
# ========================= EOF ====================================================================
