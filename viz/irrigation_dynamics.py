import os

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model.initialize import initialize_data


def irrigation_timeseries(field_data, feature, out_dir=None):

    for year in range(2005, 2023):
        field = field_data.input['irr_data'][f'{feature}'][str(year)]

        column, desc, color = f'ndvi_irr', f'Irrigated NDVI (Smoothed) - {feature}', 'green'

        df = field_data.input_to_dataframe(feature)

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
            title=f'NDVI Time Series for {year}')

        if out_dir:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            if 'html' in out_dir:
                fig_file = os.path.join(out_dir, f'{feature}_{year}.html')
                fig.write_html(fig_file)
            else:
                fig_file = os.path.join(out_dir, f'{feature}_{year}.png')
                fig.write_image(fig_file)

            print(fig_file)
        else:
            fig.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project = '4_Flux_Network'
    constraint_ = 'tight'

    project_ws_ = os.path.join(root, 'tutorials', project)
    run_data = os.path.join(root, 'tutorials')

    data_ = os.path.join(project_ws_, 'data')
    config_file = os.path.join(project_ws_, 'config.toml')

    station_file = os.path.join(data_, 'station_metadata.csv')

    sdf = pd.read_csv(station_file, index_col=0, header=1)
    sites = list(set(sdf.index.unique().to_list()))

    sites.sort()

    for site_ in sites:

        if site_ not in ['Almond_High']:
            continue

        run_const = os.path.join(run_data, '4_Flux_Network', 'results', constraint_)
        output_ = os.path.join(run_const, site_)

        prepped_input = os.path.join(output_, f'prepped_input.json')
        ndvi_forecast_ = os.path.join(output_, f'ndvi_forecast.json')
        if not os.path.exists(prepped_input):
            prepped_input = os.path.join(output_, f'prepped_input_{site_}.json')
            ndvi_forecast_ = os.path.join(output_, f'ndvi_forecast_{site_}.json')

        config_, fields_ = initialize_data(config_file, project_ws_, input_data=prepped_input)

        out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'irrigation', 'png')

        irrigation_timeseries(fields_, site_, out_dir=out_fig_dir_)
# ========================= EOF ====================================================================
