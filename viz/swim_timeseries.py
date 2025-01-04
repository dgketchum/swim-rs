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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
