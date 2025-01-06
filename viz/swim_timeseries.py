import os

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_swim_timeseries(df, parameters, start='2007-05-01', end='2007-10-31', png_file=None):
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
    data = '/home/dgketchum/PycharmProjects/et-demands/examples/tongue/daily_stats/18_crop_03.csv'
    png = '/home/dgketchum/Downloads/swim_figs/alfalfa_tongue_etd_Kc.png'
    # params_ = ['ETact', 'ETpot', 'Irrigation', 'PPT']
    params_ = ['Kc']
    plot_etd_timeseries(data, params_, start='2007-01-01', end='2007-12-31', png_file=png)

# ========================= EOF ====================================================================
