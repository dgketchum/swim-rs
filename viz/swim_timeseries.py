import os

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_swim_timeseries(df, parameters, start='2018-01-01', end='2018-12-31', png_file=None):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, index_col=0, parse_dates=True)

    df = df.loc[start:end]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_vars = ['rain', 'melt', 'snow_fall', 'dperc', 'irrigation']
    bar_colors = ['lightpink', 'lightblue', 'blue', 'lightsalmon', 'red']

    for i, param in enumerate(parameters):
        if param in bar_vars:
            vals = df[param]
            if param == 'dperc':
                vals *= -1
                print(max(vals))
            fig.add_trace(
                go.Bar(x=df.index, y=vals, name=param,
                       marker=dict(color=bar_colors[bar_vars.index(param)])),
                secondary_y=False,
            )
        else:
            if param in ['et_act', 'etref'] and 'et_act' in parameters and 'etref' in parameters:
                secondary_y = False
            else:
                secondary_y = True if i > 0 else False

            fig.add_trace(
                go.Scatter(x=df.index, y=df[param], name=param),
                secondary_y=secondary_y,
            )

    for param in parameters:
        if param in ['etf_irr', 'etf_inv_irr', 'ndvi_irr', 'ndvi_inv_irr']:
            ct_param = param + '_ct'
            if ct_param in df.columns:
                scatter_df = df[df[ct_param] == 1]
                fig.add_trace(
                    go.Scatter(x=scatter_df.index, y=scatter_df[param],
                               mode='markers', marker_symbol='x',
                               marker_size=5, name=f'{param} Retrieval'),
                    secondary_y=True,
                )

    kwargs = dict(title_text="SWIM Model Time Series",
                  xaxis_title="Date",
                  yaxis_title="mm",
                  height=800,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  yaxis2=dict(showgrid=False))

    if 'dperc' in parameters:
        kwargs.update(
            dict(yaxis=dict(showgrid=False, range=[-20, None]), yaxis2=dict(showgrid=False, range=[-20, None])))

    fig.update_layout(**kwargs)
    fig.update_xaxes(rangeslider_visible=True)
    if png_file:
        fig.write_image(png_file)
        return
    fig.show()


def plot_etd_timeseries(df, parameters, start='2007-05-01', end='2007-10-31', png_file=None):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, skiprows=[0], index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df[['Year', 'Month', 'Day']])

    df = df.loc[start:end]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_vars = ['PPT', 'irrigation']
    scatter_vars = ['ETpot', 'ETact', 'ETbas', 'ETcan', 'Kc', 'kc_bas', 'ETref', 'kr', 'ke', 'dperc',
                    'REW', 'P', 'RO', 'DR', 'DP', 'E', 'T', 'SW', 'SNW', 'MWC', 'Stress']
    bar_colors = ['lightsalmon', 'red']

    for i, param in enumerate(parameters):
        if param in bar_vars:
            vals = df[param]
            fig.add_trace(
                go.Bar(x=df.index, y=vals, name=param,
                       marker=dict(color=bar_colors[bar_vars.index(param)])),
                secondary_y=False,
            )
        elif param in scatter_vars:
            if param in ['et_act', 'etref'] and 'et_act' in parameters and 'etref' in parameters:
                secondary_y = False
            else:
                secondary_y = True if i > 0 else False

            fig.add_trace(
                go.Scatter(x=df.index, y=df[param], name=param),
                secondary_y=secondary_y,
            )

    kwargs = dict(title_text="ET Demands Model Time Series",
                  xaxis_title="Date",
                  yaxis_title="mm",
                  height=800,
                  width=1600,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  yaxis2=dict(showgrid=False))

    fig.update_layout(**kwargs)
    fig.update_xaxes(rangeslider_visible=True)
    if png_file:
        fig.write_image(png_file)
    fig.show()


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = 'alarc_test'
    feature_ = 'ALARC2_Smith6'

    data = os.path.join(root, 'tutorials', project, 'data')
    out_csv_dir = os.path.join(data, 'model_output')
    out_csv = os.path.join(out_csv_dir, f'{feature_}.csv')

    out_fig_dir = os.path.join(root, 'tutorials', project, 'figures')


    df = pd.read_csv(out_csv, index_col=0, parse_dates=True)

    plot_swim_timeseries(df, ['snow_fall', 'rain', 'melt', 'dperc'], start='2018-01-01', end='2019-01-01',
                         png_file=os.path.join(out_fig_dir, 'dperc_lessIrrDOY.png'))

    plot_swim_timeseries(df, ['soil_water', 'irrigation', 'rain', 'melt'], start='2018-01-01',
                         end='2018-10-01', png_file=os.path.join(out_fig_dir, 'soil_water_lessIrrDOY.png'))

    plot_swim_timeseries(df, ['et_act', 'etref', 'rain', 'melt', 'irrigation'], start='2018-01-01', end='2018-12-31',
                         png_file=os.path.join(out_fig_dir, 'irr_et_lessIrrDOY.png'))
# ========================= EOF ====================================================================
