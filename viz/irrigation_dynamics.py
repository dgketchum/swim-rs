import os
import json

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def irrigation_timeseries(dynamics_json, remote_sensing_file, feature, out_dir=None):
    with open(dynamics_json, 'r') as f:
        input_dct = json.load(f)

    for year in range(2018, 2019):
        field = input_dct['irr'][f'{feature}'][str(year)]

        column, desc, color = f'ndvi_irr', f'Irrigated NDVI (Smoothed) - {feature}', 'green'

        df = pd.read_csv(remote_sensing_file, index_col=0)

        df_year = df.loc[f'{year}-01-01': f'{year}-12-31'].copy()
        df_year.index = pd.to_datetime(df_year.index)
        df_year['doy'] = [i.dayofyear for i in df_year.index]

        df_year[column + '_rolling'] = df_year[column].rolling(window=7, center=True).mean()

        ct_column = column + '_ct'
        scatter_data = df_year[column].copy()
        scatter_data[df_year[ct_column] == 0] = np.nan

        irr_dates = [pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=doy - 1) for doy in field['irr_doys']
                     if doy in df_year['doy'].tolist()]
        irr_values = df_year.loc[irr_dates, column + '_rolling']

        fig = make_subplots()

        fig.add_trace(go.Scatter(x=df_year.index, y=df_year[column + '_rolling'], mode='lines',
                                 name=desc + ' 7-day Mean', line=dict(color=color)))

        fig.add_trace(go.Scatter(x=scatter_data.index, y=scatter_data, mode='markers',
                                 name='Capture Date Retrieval', marker=dict(size=8, color=color)))

        fig.add_trace(go.Scatter(x=irr_dates, y=irr_values, mode='markers',
                                 name='Potential Irrigation Day',
                                 marker=dict(size=12, color='blue', symbol='circle-open', line=dict(width=2))))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Value',
            title=f'NDVI Time Series for {year}',
        )

        if out_dir:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            fig_file = os.path.join(out_dir, f'irrigation_timeseries_{year}.html')
            fig.write_html(fig_file)
            print(fig_file)
        else:
            fig.show()


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = 'alarc_test'
    feature_ = 'ALARC2_Smith6'

    data = os.path.join(root, 'tutorials', project, 'data')
    landsat = os.path.join(root, 'footprints', 'landsat')
    joined_timeseries = os.path.join(data, 'plot_timeseries', f'{feature_}_daily.csv')
    cuttings_json = os.path.join(landsat, 'calibration_dynamics.json')
    out_fig_dir = os.path.join(root, 'tutorials', project, 'figures', 'irrigation_dynamics')

    irrigation_timeseries(cuttings_json, remote_sensing_file=joined_timeseries, feature=feature_,
                          out_dir=None)
# ========================= EOF ====================================================================
