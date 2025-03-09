import os

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def plot_swim_timeseries(df, parameters, start='2018-01-01', end='2018-12-31', png_dir=None, html_dir=None, fid=None):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, index_col=0, parse_dates=True)

    df = df.loc[start:end]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_vars = ['rain', 'melt', 'snow_fall', 'dperc', 'irrigation']
    bar_colors = ['lightpink', 'lightblue', 'blue', 'lightsalmon', 'red']

    pdc_present, flux_present = 'npc', 'nfx'

    for i, param in enumerate(parameters):
        if param in bar_vars:
            vals = df[param]
            if param == 'dperc':
                vals *= -1
            fig.add_trace(
                go.Bar(x=df.index, y=vals, name=param,
                       marker=dict(color=bar_colors[bar_vars.index(param)])),
                secondary_y=False,
            )

        elif param != 'pdc':

            if param == 'flux_etf':
                flux_present = 'flx'
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[param], name=param, mode='markers'),
                    secondary_y=True if i > 0 else False)

            elif param in ['et_act', 'etref'] and 'et_act' in parameters and 'etref' in parameters:
                secondary_y = False
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[param], name=param),
                    secondary_y=secondary_y)

            else:
                secondary_y = True if i > 0 else False
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[param], name=param),
                    secondary_y=secondary_y)

    for param in parameters:
        if param in ['etf_irr', 'etf_inv_irr', 'ndvi_irr', 'ndvi_inv_irr']:
            ct_param = param + '_ct'
            if ct_param in df.columns:
                scatter_df = df[df[ct_param] == 1]
                if 'pdc' in df.columns and (param == 'etf_irr' or param == 'etf_inv_irr'):
                    scatter_df_pdc = scatter_df[scatter_df['pdc'] > 0]
                    scatter_df_no_pdc = scatter_df[scatter_df['pdc'] <= 0]

                    if scatter_df_pdc.shape[0] > 0:
                        pdc_present = 'pdc'

                    fig.add_trace(
                        go.Scatter(x=scatter_df_pdc.index, y=scatter_df_pdc[param],
                                   mode='markers', marker_symbol='circle',
                                   marker_size=15, name=f'{param} Retrieval (PDC Flagged)'),
                        secondary_y=True,
                    )
                    fig.add_trace(
                        go.Scatter(x=scatter_df_no_pdc.index, y=scatter_df_no_pdc[param],
                                   mode='markers', marker_symbol='x',
                                   marker_size=5, name=f'{param} Retrieval'),
                        secondary_y=True,
                    )
                else:
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
                  width=1600,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  yaxis2=dict(showgrid=False))

    max_bar_val = 0
    for var in bar_vars:
        if var in parameters:
            temp_max = df[var].abs().max()
            if temp_max > max_bar_val:
                max_bar_val = temp_max

    if 'dperc' in parameters:
        kwargs.update(
            dict(yaxis=dict(showgrid=False, range=[-max_bar_val * 3, max_bar_val * 3]),
                 yaxis2=dict(showgrid=False, range=[-20, None])))
    elif max_bar_val > 0:
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[0, max_bar_val * 3])))

    fig.update_layout(**kwargs)
    fig.update_xaxes(rangeslider_visible=True)

    if png_dir:
        png_file = os.path.join(png_dir, f'{fid}_{start[:4]}_{pdc_present}_{flux_present}.png')
        fig.write_image(png_file)
        print(png_file)
        return

    if html_dir:
        html_file = os.path.join(html_dir, f'{fid}_{start[:4]}_{pdc_present}_{flux_present}.html')
        fig.write_html(html_file)
        print(html_file)
        return

    else:
        fig.show()


def flux_pdc_timeseries(csv_dir, flux_file_dir, fids, out_fig_dir=None):
    """"""

    for fid in fids:

        csv = os.path.join(csv_dir, fid, f'{fid}.csv')
        pdc_file = os.path.join(csv_dir, fid, f'{fid}.pdc.csv')
        flux_file = os.path.join(flux_file_dir, f'{fid}_daily_data.csv')

        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        pdc = pd.read_csv(pdc_file, index_col=0)
        idx_file = pdc_file.replace('.pdc', '.idx')
        idx = pd.read_csv(idx_file, index_col=0, parse_dates=True)
        idx['pdc'] = [1 if obs_id in pdc.index else 0 for obs_id in idx['obs_id']]
        df['pdc'] = idx['pdc']
        pdc_yr = df[['pdc']].resample('YE').sum()

        pdc_yr = pdc_yr[pdc_yr['pdc'] > 0].index.year.to_list()

        flux_obs = pd.read_csv(flux_file, index_col=0, parse_dates=True)

        flux_yr = list(set([i.year for i, obs in flux_obs.iterrows() if np.isfinite(obs['EToF_filtered'])]))

        flux_obs = flux_obs.reindex(df.index)

        df['flux_etf'] = flux_obs['EToF_filtered']

        years = sorted(list(set(pdc_yr + flux_yr)))

        for yr in years:

            if yr in flux_yr:
                plot_swim_timeseries(df, ['irrigation', 'rain', 'etf_irr', 'kc_act', 'ndvi_irr', 'pdc', 'flux_etf'],
                                     start=f'{yr}-01-01', end=f'{yr}-12-31', png_dir=out_fig_dir, fid=fid)

            else:
                plot_swim_timeseries(df, ['irrigation', 'rain', 'etf_irr', 'kc_act', 'ndvi_irr', 'pdc'],
                                     start=f'{yr}-01-01', end=f'{yr}-12-31', png_dir=out_fig_dir, fid=fid)


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = '4_Flux_Network'
    constraint_ = 'tight'

    data = os.path.join(root, 'tutorials', project, 'data')

    results = '/data/ssd2/swim'
    if not os.path.isdir(results):
        results = os.path.join(root, 'tutorials')

    out_csv_dir = os.path.join(results, '4_Flux_Network', 'results', constraint_)

    out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'png')

    bad_params = ('/home/dgketchum/PycharmProjects/swim-rs/tutorials/4_Flux_Network/'
                  'results_comparison_05MAR2025_crops_tight.csv')

    bad_df = pd.read_csv(bad_params, index_col=0)
    bad_df = bad_df.loc[bad_df['mode'] == constraint_]
    bad_stations = bad_df.index.unique().to_list()

    flux_data = os.path.join(data, 'daily_flux_files')

    flux_pdc_timeseries(out_csv_dir, flux_data, bad_stations, out_fig_dir_)

# ========================= EOF ====================================================================
