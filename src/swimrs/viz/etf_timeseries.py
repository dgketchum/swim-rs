import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


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
    pass

# ========================= EOF ====================================================================
