import os
import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def run_fields(ini_path, project_ws, selected_feature, output_csv):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)
    fields.output = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    out_df = fields.output[selected_feature].copy()

    in_df = fields.input_to_dataframe(selected_feature)

    df = pd.concat([out_df, in_df], axis=1, ignore_index=False)
    df.to_csv(output_csv)


def plot_timeseries(df, parameters, start='2007-05-01', end='2007-10-31'):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, index_col=0, parse_dates=True)

    df = df.loc[start:end]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_vars = ['rain', 'melt', 'snow_fall', 'dperc', 'irrigation']
    bar_colors = ['lightpink', 'lightblue', 'blue', 'lightsalmon', 'lightgrey']

    if parameters[0] in bar_vars:
        if parameters[0] in bar_vars:
            vals = df[parameters[0]]
            if parameters[0] == 'dperc':
                vals *= -1
                print(min(vals))
            fig.add_trace(
                go.Bar(x=df.index, y=vals, name=parameters[0],
                       marker=dict(color=bar_colors[bar_vars.index(parameters[0])])),
                secondary_y=False,
            )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[parameters[0]], name=parameters[0]),
            secondary_y=False,
        )

    for i, param in enumerate(parameters[1:]):
        if param in bar_vars:
            vals = df[param]
            if param == 'dperc':
                vals *= -1
                print(min(vals))
            fig.add_trace(
                go.Bar(x=df.index, y=vals, name=param,
                       marker=dict(color=bar_colors[bar_vars.index(param)])),
                secondary_y=False,
            )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[param], name=param),
                secondary_y=True,
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
                  height=800,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  yaxis2=dict(showgrid=False))

    if 'dperc' in parameters:
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[-20, None])))

    fig.update_layout(**kwargs)
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    config_file = os.path.join(root, 'tutorials', '1_Boulder', 'data', 'tutorial_config.toml')
    project_ws_ = os.path.join(root, 'tutorials', '1_Boulder')
    out_csv = os.path.join(root, 'tutorials', '1_Boulder', 'step_5_model_setup', 'combined_output.csv')

    run_fields(config_file, project_ws_, selected_feature='043_000161', output_csv=out_csv)

    start = '2018-01-01'
    end = '2018-12-31'
    plot_timeseries(out_csv, ['snow_fall', 'rain', 'melt', 'dperc'], start=start, end=end)

    # plot_timeseries(out_csv, ['et_act', 'etref', 'irrigation'], start=start, end=end)

# ========================= EOF ====================================================================
