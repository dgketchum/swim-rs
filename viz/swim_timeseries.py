import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import COLOR_MAP


def plot_swim_timeseries(df, parameters, start='2018-01-01', end='2018-12-31', fig_dir=None, fid=None):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, index_col=0, parse_dates=True)

    df = df.loc[start:end]

    bar_vars = ['rain', 'melt', 'snow_fall', 'dperc', 'irrigation', 'gw_sim']
    pdc_present, flux_present = 'npc', 'nfx'

    if 'dperc' in df.columns and (df['dperc'] > 0).any():
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                            row_heights=[0.75, 0.25], vertical_spacing=0.2)
        bar_secondary_y = False
        bar_row = 2
        main_row = 1
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        bar_secondary_y = False
        bar_row = 1
        main_row = 1

    def _add_trace(param, trace, row, secondary_y):
        fig.add_trace(trace, row=row, col=1, secondary_y=secondary_y)

    for i, param in enumerate(parameters):
        if param in bar_vars:
            vals = df[param]
            if param == 'dperc':
                vals *= -1
            trace = go.Bar(x=df.index, y=vals, name=param,
                           marker=dict(color=COLOR_MAP.get(param, 'black')),
                           width=1000 * 60 * 60 * 24 * 0.8)
            _add_trace(param, trace, bar_row, bar_secondary_y)

        elif param != 'pdc':
            if param == 'flux_etf':
                flux_present = 'flx'
                trace = go.Scatter(x=df.index, y=df[param], name=param, mode='markers',
                                   marker=dict(color=COLOR_MAP.get(param, 'black')))
                _add_trace(param, trace, main_row, True)
            else:
                secondary_y = False if (param in ['et_act', 'etref'] and 'et_act' in parameters and 'etref'
                                        in parameters) else True if i > 0 else False
                trace = go.Scatter(x=df.index, y=df[param], name=param,
                                   marker=dict(color=COLOR_MAP.get(param, 'black')))
                _add_trace(param, trace, main_row, secondary_y)

    for param in parameters:
        if param in ['etf', 'etf_irr', 'etf_inv_irr', 'ndvi', 'ndvi_irr', 'ndvi_inv_irr']:
            ct_param = param + '_ct'
            if ct_param in df.columns:
                scatter_df = df[df[ct_param] == 1]
                if 'pdc' in df.columns and (param == 'etf_irr' or param == 'etf_inv_irr'):
                    scatter_df_pdc = scatter_df[scatter_df['pdc'] > 0]
                    scatter_df_no_pdc = scatter_df[scatter_df['pdc'] <= 0]

                    if scatter_df_pdc.shape[0] > 0:
                        pdc_present = 'pdc'

                    trace = go.Scatter(x=scatter_df_pdc.index, y=scatter_df_pdc[param],
                                       mode='markers', marker_symbol='circle',
                                       marker_size=15, name=f'{param.split("_")[0]} retrieval (PDC Flagged)',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param, trace, main_row, True)

                    trace = go.Scatter(x=scatter_df_no_pdc.index, y=scatter_df_no_pdc[param],
                                       mode='markers', marker_symbol='x',
                                       marker_size=5, name=f'{param.split("_")[0]} retrieval',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param, trace, main_row, True)
                else:
                    trace = go.Scatter(x=scatter_df.index, y=scatter_df[param],
                                       mode='markers', marker_symbol='x',
                                       marker_size=5, name=f'{param.split("_")[0]} retrieval',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param, trace, main_row, True)

    kwargs = dict(title_text="SWIM Model Time Series",
                  xaxis_title="Date",
                  yaxis_title="mm",
                  height=800 if 'png' in fig_dir else 1300,
                  width=1600 if 'png' in fig_dir else 2300,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  xaxis2=dict(showgrid=False)
                  )
    if main_row == 1:
        kwargs['yaxis2'] = dict(showgrid=False)

    max_bar_val = 0

    if 'dperc' in parameters and (df['dperc'] != 0.0).any():
        kwargs.update(dict(yaxis2=dict(showgrid=False, range=[-0.2, None])))
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[-0.2, 1.4])))

    elif max_bar_val > 0:
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[0, max_bar_val * 3])))

    fig.update_layout(**kwargs)
    # fig.update_xaxes(rangeslider_visible=False, row=main_row, col=1)

    if fig_dir is not None and 'png' in fig_dir:
        png_file = os.path.join(fig_dir, f'{fid}_{start[:4]}_{pdc_present}_{flux_present}.png')
        fig.write_image(png_file)
        print(png_file)
        return

    if fig_dir is not None and 'html' in fig_dir:
        html_file = os.path.join(fig_dir, f'{fid}_{start[:4]}_{pdc_present}_{flux_present}.html')
        fig.write_html(html_file)
        print(html_file)
        return

    else:
        fig.show()


def flux_pdc_timeseries(csv_dir, flux_file_dir, fids, out_fig_dir=None, spec='flux-pdc'):
    """"""

    for fid in fids:

        csv = os.path.join(csv_dir, fid, f'{fid}.csv')
        pdc_file = os.path.join(csv_dir, fid, f'{fid}.pdc.csv')
        flux_file = os.path.join(flux_file_dir, f'{fid}_daily_data.csv')

        df = pd.read_csv(csv, index_col=0, parse_dates=True)

        df_irr = df[['irr_day']].groupby(df.index.year).agg('sum')
        irr_years = [i for i, y in df_irr.iterrows() if y['irr_day'] > 0]
        irr_index = [i for i in df.index if i.year in irr_years]

        df['ndvi'] = df['ndvi_inv_irr']
        df.loc[irr_index, 'ndvi'] = df.loc[irr_index, 'ndvi_irr']
        df['ndvi_ct'] = df['ndvi_inv_irr_ct']
        df.loc[irr_index, 'ndvi_ct'] = df.loc[irr_index, 'ndvi_irr_ct']

        df['etf'] = df['etf_inv_irr']
        df.loc[irr_index, 'etf'] = df.loc[irr_index, 'etf_irr']
        df['etf_ct'] = df['etf_inv_irr_ct']
        df.loc[irr_index, 'etf_ct'] = df.loc[irr_index, 'etf_irr_ct']

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

        if spec == 'flux':
            years = flux_yr
        elif spec == 'fluxpdc':
            years = sorted(list(set(pdc_yr + flux_yr)))
        elif spec == 'pdc':
            years = pdc_yr
        elif spec == 'all':
            years = list(set(df.index.year.to_list()))
        else:
            raise ValueError('Must choose from flux, fluxpdc, pdc, or all to select plotting years')

        for yr in years:

            if yr in flux_yr:
                plot_swim_timeseries(df,
                                     ['irrigation', 'rain', 'melt', 'dperc', 'gw_sim', 'snow_fall',
                                      'etf', 'ks', 'ke', 'kc_act', 'ndvi', 'pdc',
                                      'flux_etf'],
                                     start=f'{yr}-01-01', end=f'{yr}-12-31', fig_dir=out_fig_dir, fid=fid)

            else:
                plot_swim_timeseries(df, ['irrigation', 'rain', 'melt', 'dperc', 'gw_sim', 'snow_fall',
                                          'etf', 'ks', 'ke', 'kc_act', 'ndvi',
                                          'pdc'],
                                     start=f'{yr}-01-01', end=f'{yr}-12-31', fig_dir=out_fig_dir, fid=fid)


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = '4_Flux_Network'
    constraint_ = 'tight'

    data = os.path.join(root, 'tutorials', project, 'data')

    results = '/data/ssd2/swim'
    if not os.path.isdir(results):
        results = os.path.join(root, 'tutorials')

    out_csv_dir = os.path.join(results, '4_Flux_Network', 'results', constraint_)

    out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'html')

    bad_params = ('/home/dgketchum/PycharmProjects/swim-rs/tutorials/4_Flux_Network/'
                  'results_comparison_05MAR2025_crops_tight.csv')

    bad_df = pd.read_csv(bad_params, index_col=0)
    bad_df = bad_df.loc[bad_df['mode'] == constraint_]
    bad_stations = bad_df.index.unique().to_list()

    l = ['AFD', 'ALARC2_Smith6', 'BPHV', 'ET_1', 'JPL1_JV114', 'KV_4',
         'MOVAL', 'MR', 'S2', 'UA2_JV330', 'UA3_JV108', 'UA3_KN15',
         'UOVLO', 'US-Blo', 'US-CMW', 'US-Esm', 'US-Hn2', 'US-LS1',
         'US-OF2', 'US-Ro2', 'US-SCg']

    flux_data = os.path.join(data, 'daily_flux_files')

    flux_pdc_timeseries(out_csv_dir, flux_data, ['S2'], out_fig_dir_, spec='flux')

# ========================= EOF ====================================================================
