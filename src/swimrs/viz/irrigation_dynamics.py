import os

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from swimrs.swim.sampleplots import SamplePlots
from swimrs.prep import get_flux_sites, get_ensemble_parameters
from swimrs.swim.config import ProjectConfig
from swimrs.prep import prep_fields_json


def irrigation_timeseries(field_data, feature, out_dir=None):
    for year in range(1987, 2023):
        field = field_data.input['irr_data'][f'{feature}'][str(year)]

        column, desc, color = f'ndvi_irr', f'Irrigated NDVI (Smoothed) - {feature}', 'green'

        df = field_data.input_to_dataframe(feature)

        df_year = df.loc[f'{year}-01-01': f'{year}-12-31'].copy()
        df_year.index = pd.to_datetime(df_year.index)
        df_year['doy'] = [i.dayofyear for i in df_year.index]

        df_year[column + '_rolling'] = df_year[column].rolling(window=32, center=True).mean()

        irr_dates = [pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=doy - 1) for doy in field['irr_doys']
                     if doy in df_year['doy'].tolist()]
        irr_values = df_year.loc[irr_dates, column + '_rolling']

        idf = pd.DataFrame(data=irr_values, index=irr_dates)
        idf['doy'] = df_year.loc[irr_dates, 'doy']

        fig = make_subplots()

        fig.add_trace(go.Scatter(x=df_year.index, y=df_year[column + '_rolling'], mode='lines',
                                 name=desc + ' 32-day Mean', line=dict(color=color)))

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

    """"""
    # project = '4_Flux_Network'
    project = '5_Flux_Ensemble'

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'code', 'swim-rs', 'examples', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    if project == '5_Flux_Ensemble':
        western_only = True
        run_const = os.path.join(config.project_ws, 'results', 'tight')

    else:
        run_const = os.path.join(config.project_ws, 'results', 'tight')
        western_only = False

    sites, sdf = get_flux_sites(config.station_metadata_csv, crop_only=False,
                                return_df=True, western_only=western_only, header=1)

    print(f'{len(sites)} sites to evalutate in {project}')
    incomplete, complete, results = [], [], []

    overwrite_ = False
    use_new_input = True

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        # if lulc == 'Croplands':
        #     continue

        if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114']:
            continue

        if site_ not in ['ALARC2_Smith6']:
            continue

        print(f'\n{ee} {site_}: {lulc}')

        output_ = os.path.join(run_const, site_)

        target_dir = os.path.join(config.project_ws, 'ptjpl_test', site_)
        station_prepped_input = os.path.join(target_dir, f'prepped_input_{site_}.json')
        models = [config.etf_target_model]
        if config.etf_ensemble_members is not None:
            models += config.etf_ensemble_members

        dynamics = '/data/ssd2/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_dynamics.json'
        # dynamics = config.dynamics_data_json

        rs_params_ = get_ensemble_parameters(include=models)
        prep_fields_json(config.properties_json, config.plot_timeseries, dynamics,
                         station_prepped_input, target_plots=[site_], rs_params=rs_params_,
                         interp_params=('ndvi',))

        config.input_data = station_prepped_input

        out_fig_dir_ = os.path.join(os.path.expanduser('~'), 'Downloads', 'figures', 'irrigation', 'reorg')

        plots_ = SamplePlots()

        config.input_data = station_prepped_input

        plots_.initialize_plot_data(config)

        # months = [i for sl in [plots_.input['gwsub_data'][site_][str(yr)]['months'] for yr in range(1987, 2025)] for i in sl]
        # print(f'{branch}: {len(months)} months of gw subsidy')

        irrigation_timeseries(plots_, site_, out_dir=out_fig_dir_)
# ========================= EOF ====================================================================
