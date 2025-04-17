import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors

from viz import COLOR_MAP


def plot_swim_timeseries(df, parameters, model, irr_index=None, members=None, start='2018-01-01', end='2018-12-31',
                         fig_dir=None, fid=None):
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
                trace_name = f'etf ({model})' if param == 'etf' else param
                trace = go.Scatter(x=df.index, y=df[param], name=trace_name,
                                   marker=dict(color=COLOR_MAP.get(param, 'black')))

                _add_trace(param, trace, main_row, secondary_y)

    for param in parameters:
        if param in ['etf', 'etf_irr', 'etf_inv_irr', 'ndvi', 'ndvi_irr', 'ndvi_inv_irr']:
            ct_param = param + '_ct'
            if ct_param in df.columns:
                scatter_df = df[df[ct_param] == 1]
                if 'pdc' in df.columns and (param == 'etf_irr' or param == 'etf_inv_irr' or param == 'etf'):
                    scatter_df_pdc = scatter_df[scatter_df['pdc'] > 0]
                    scatter_df_no_pdc = scatter_df[scatter_df['pdc'] <= 0]

                    if scatter_df_pdc.shape[0] > 0:
                        pdc_present = 'pdc'

                    trace = go.Scatter(x=scatter_df_pdc.index, y=scatter_df_pdc[param],
                                       mode='markers', marker_symbol='circle',
                                       marker_size=15, name=f'{param.split("_")[0]} retrieval (PDC Flagged)',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param + '_pdc_markers', trace, main_row, True)

                    trace = go.Scatter(x=scatter_df_no_pdc.index, y=scatter_df_no_pdc[param],
                                       mode='markers', marker_symbol='x',
                                       marker_size=5, name=f'{param.split("_")[0]} retrieval',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param + '_markers', trace, main_row, True)
                else:
                    trace = go.Scatter(x=scatter_df.index, y=scatter_df[param],
                                       mode='markers', marker_symbol='x',
                                       marker_size=5, name=f'{param.split("_")[0]} retrieval',
                                       marker=dict(color=COLOR_MAP.get(param, 'black')))
                    _add_trace(param + '_markers', trace, main_row, True)

    if members and irr_index is not None:
        for member in members:
            member_etf_irr_col = f'{member}_etf_irr'
            member_etf_inv_irr_col = f'{member}_etf_inv_irr'

            if member_etf_irr_col in df.columns and member_etf_inv_irr_col in df.columns:
                member_etf_series = df[member_etf_inv_irr_col].copy()
                valid_irr_index = irr_index.intersection(df.index)
                member_etf_series.loc[valid_irr_index] = df.loc[valid_irr_index, member_etf_irr_col]

                rgba_tuple = matplotlib.colors.to_rgba(COLOR_MAP.get(member, 'black'), alpha=0.5)
                plotly_rgba_color = (f'rgba({int(rgba_tuple[0] * 255)}, {int(rgba_tuple[1] * 255)}, '
                                     f'{int(rgba_tuple[2] * 255)}, {rgba_tuple[3]})')

                trace = go.Scatter(x=df.index, y=member_etf_series, name=f'{member} (member)',
                                   mode='lines',
                                   line=dict(color=plotly_rgba_color, width=1),
                                   showlegend=True)

                _add_trace(f'{member}_etf', trace, main_row, True)

    kwargs = dict(title_text="SWIM Model Time Series",
                  xaxis_title="Date",
                  yaxis_title="mm",
                  height=800 if fig_dir and 'png' in fig_dir else 1300,
                  width=1600 if fig_dir and 'png' in fig_dir else 2300,
                  template='plotly_dark',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  xaxis2=dict(showgrid=False)
                  )
    if main_row == 1:
        kwargs['yaxis2'] = dict(showgrid=False)

    max_bar_val = 0
    possible_bar_vars = [v for v in bar_vars if v in df.columns]
    if possible_bar_vars:
        max_bar_val = df[possible_bar_vars].max().max()

    if 'dperc' in parameters and 'dperc' in df.columns and (df['dperc'] != 0.0).any():
        kwargs.update(dict(yaxis2=dict(showgrid=False, range=[-1, None], tick0=1, dtick=1)))
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[-0.2, 1.4])))

    elif max_bar_val > 0:
        kwargs.update(dict(yaxis=dict(showgrid=False, range=[0, max_bar_val * 3])))

    fig.update_layout(**kwargs)

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


def flux_pdc_timeseries(csv_dir, flux_file_dir, fids, out_fig_dir=None, spec='flux-pdc',
                        model='ssebop', members=None):
    for fid in fids:

        csv = os.path.join(csv_dir, fid, f'{fid}.csv')
        pdc_file = os.path.join(csv_dir, fid, f'{fid}.pdc.csv')
        flux_file = os.path.join(flux_file_dir, f'{fid}_daily_data.csv')

        if not os.path.exists(csv):
            print(f"Skipping {fid}: CSV file not found at {csv}")
            continue

        df = pd.read_csv(csv, index_col=0, parse_dates=True)

        df_irr = df[['irr_day']].groupby(df.index.year).agg('sum')
        irr_years = [i for i, y in df_irr.iterrows() if y['irr_day'] > 0]
        irr_index = df.index[df.index.year.isin(irr_years)]

        if 'ndvi_inv_irr' in df.columns and 'ndvi_irr' in df.columns:
            df['ndvi'] = df['ndvi_inv_irr']
            df.loc[irr_index, 'ndvi'] = df.loc[irr_index, 'ndvi_irr']
        if 'ndvi_inv_irr_ct' in df.columns and 'ndvi_irr_ct' in df.columns:
            df['ndvi_ct'] = df['ndvi_inv_irr_ct']
            df.loc[irr_index, 'ndvi_ct'] = df.loc[irr_index, 'ndvi_irr_ct']

        etf_col = f'{model}_etf_inv_irr'
        etf_irr_col = f'{model}_etf_irr'
        etf_ct_col = f'{model}_etf_inv_irr_ct'
        etf_irr_ct_col = f'{model}_etf_irr_ct'

        if etf_col in df.columns and etf_irr_col in df.columns:
            df['etf'] = df[etf_col]
            df.loc[irr_index, 'etf'] = df.loc[irr_index, etf_irr_col]
            if etf_ct_col in df.columns and etf_irr_ct_col in df.columns:
                df['etf_ct'] = df[etf_ct_col]
                df.loc[irr_index, 'etf_ct'] = df.loc[irr_index, etf_irr_ct_col]
        elif 'etf_inv_irr' in df.columns and 'etf_irr' in df.columns:
            df['etf'] = df['etf_inv_irr']
            df.loc[irr_index, 'etf'] = df.loc[irr_index, 'etf_irr']
            if 'etf_inv_irr_ct' in df.columns and 'etf_irr_ct' in df.columns:
                df['etf_ct'] = df['etf_inv_irr_ct']
                df.loc[irr_index, 'etf_ct'] = df.loc[irr_index, 'etf_irr_ct']
        else:
            print(
                f"Warning: Could not find primary ETF columns for model '{model}' or default 'etf_inv_irr'/'etf_irr' in {csv}")

        if os.path.exists(pdc_file):
            pdc = pd.read_csv(pdc_file, index_col=0)
            idx_file = pdc_file.replace('.pdc', '.idx')
            if os.path.exists(idx_file):
                idx = pd.read_csv(idx_file, index_col=0, parse_dates=True)
                idx['pdc'] = [1 if obs_id in pdc.index else 0 for obs_id in idx['obs_id']]
                # Align index before assignment
                idx = idx.reindex(df.index, fill_value=0)
                df['pdc'] = idx['pdc']
            else:
                df['pdc'] = 0
                print(f"Warning: Index file not found: {idx_file}")
        else:
            df['pdc'] = 0
            print(f"Warning: PDC file not found: {pdc_file}")

        pdc_yr = df[['pdc']].resample('YE').sum()
        pdc_yr = pdc_yr[pdc_yr['pdc'] > 0].index.year.to_list()

        flux_yr = []
        if os.path.exists(flux_file):
            flux_obs = pd.read_csv(flux_file, index_col=0, parse_dates=True)
            flux_yr = list(set([i.year for i, obs in flux_obs.iterrows() if pd.notna(obs.get('EToF_filtered'))]))
            flux_obs = flux_obs.reindex(df.index)
            df['flux_etf'] = flux_obs.get('EToF_filtered')
        else:
            print(f"Warning: Flux file not found: {flux_file}")
            df['flux_etf'] = np.nan

        if spec == 'flux':
            years = flux_yr
        elif spec == 'fluxpdc':
            years = sorted(list(set(pdc_yr + flux_yr)))
        elif spec == 'pdc':
            years = pdc_yr
        elif spec == 'all':
            years = sorted(list(set(df.index.year.to_list())))
        else:
            raise ValueError('Must choose from flux, fluxpdc, pdc, or all to select plotting years')

        base_params = ['irrigation', 'rain', 'melt', 'dperc', 'gw_sim', 'snow_fall',
                       'etf', 'ks', 'ke', 'kc_act', 'ndvi']
        if 'pdc' in df.columns:
            base_params.append('pdc')

        for yr in years:
            plot_params = base_params[:]
            if yr in flux_yr and 'flux_etf' in df.columns:
                plot_params.append('flux_etf')

            df_yr = df.loc[str(yr)]
            irr_index_yr = irr_index.intersection(df_yr.index)

            plot_swim_timeseries(df_yr, plot_params,  model, irr_index=irr_index_yr, members=members,
                                 start=f'{yr}-01-01', end=f'{yr}-12-31', fig_dir=out_fig_dir, fid=fid)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
