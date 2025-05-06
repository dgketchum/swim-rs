import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde

from model.initialize import initialize_data

pio.templates.default = "plotly_dark"


def forecast_ndvi(plots, field_id, forecast_days=28, similarity_window=7, max_years_to_consider=40, lookback=150,
                  output_json=None):
    df = plots.input_to_dataframe(field_id)
    df = df[['ndvi_irr', 'ndvi_inv_irr', 'ndvi_irr_ct', 'ndvi_inv_irr_ct']]

    df.loc['2022-07-01':, ['ndvi_irr_ct', 'ndvi_inv_irr_ct']] = np.nan

    last_obs_date = None
    for col in ['ndvi_irr_ct', 'ndvi_inv_irr_ct']:
        if col in df.columns:
            temp_last_obs = df[df[col] == 1].index[-1]
            if last_obs_date is None or temp_last_obs > last_obs_date:
                last_obs_date = temp_last_obs

    df.loc[df.index > last_obs_date, ['ndvi_irr', 'ndvi_inv_irr']] = np.nan

    last_obs_doy = last_obs_date.dayofyear
    sample_years = sorted(df.index.year.unique())
    target_years = sample_years[-1:]

    forecasts = {day: [] for day in range(1, forecast_days + 1)}
    forecast_years = {day: [] for day in range(1, forecast_days + 1)}
    exceeds_max = {day: [] for day in range(1, forecast_days + 1)}

    current_ndvi_col, last_obs_value = None, None

    for year in target_years:
        similar_years_data = {}
        for compare_year in sample_years:

            start = pd.to_datetime(f'{compare_year}-01-01') + pd.Timedelta(
                days=last_obs_doy - similarity_window // 2 - 1)
            end = pd.to_datetime(f'{compare_year}-01-01') + pd.Timedelta(days=last_obs_doy + similarity_window // 2 - 1)

            try:
                f_irr = plots.input['irr_data'][field_id][str(compare_year)]['f_irr']
            except KeyError:
                continue

            ndvi_col = 'ndvi_irr' if f_irr > 0.3 else 'ndvi_inv_irr'
            ct_col = f'{ndvi_col}_ct'
            hist_data = df.loc[start:end, [ndvi_col, ct_col]]
            hist_data = hist_data[hist_data[ct_col] == 1][ndvi_col]

            if hist_data.empty:
                continue

            similar_years_data[compare_year] = hist_data

        try:
            current_f_irr = plots.input['irr_data'][field_id][str(year)]['f_irr']
        except:
            continue

        current_ndvi_col = 'ndvi_irr' if current_f_irr > 0.3 else 'ndvi_inv_irr'
        ct_col = f'{current_ndvi_col}_ct'
        current_data = df.loc[last_obs_date - pd.Timedelta(days=similarity_window // 2):
                              last_obs_date + pd.Timedelta(days=similarity_window // 2), [current_ndvi_col, ct_col]]
        current_data = current_data[current_data[ct_col] == 1][current_ndvi_col]

        last_obs_value = df.loc[last_obs_date, current_ndvi_col]

        similarity_scores = {yr: np.sum((current_data.values - hist_data.values) ** 2)
                             for yr, hist_data in similar_years_data.items()}

        most_similar_years = sorted(similarity_scores, key=similarity_scores.get)[:max_years_to_consider]

        for compare_year in most_similar_years:
            for day in range(1, forecast_days + 1):
                f_date = pd.to_datetime(f'{compare_year}-01-01') + pd.Timedelta(days=last_obs_doy + day - 1)
                try:
                    f_irr = plots.input['irr_data'][field_id][str(compare_year)]['f_irr']
                    ndvi_col_use = 'ndvi_irr' if f_irr > 0.3 else 'ndvi_inv_irr'
                    f_value = df.loc[f_date, ndvi_col_use].item()
                    if not np.isnan(f_value):
                        forecasts[day].append(f_value)
                        forecast_years[day].append(compare_year)
                except KeyError:
                    continue

    max_ndvi_observed = df[['ndvi_irr', 'ndvi_inv_irr']].max().max()
    for day in forecasts:
        if forecasts[day]:
            first_forecast_val = forecasts[day][0]
            shift = last_obs_value - first_forecast_val
            for i, val in enumerate(forecasts[day]):
                forecasts[day][i] = val + shift
                if forecasts[day][i] > max_ndvi_observed:
                    exceeds_max[day].append(True)
                else:
                    exceeds_max[day].append(False)

    results = {}
    results['forecasts'] = forecasts
    results['forecast_years'] = forecast_years
    results['exceeds_max'] = exceeds_max
    results['last_obs_date'] = last_obs_date.strftime('%Y-%m-%d')
    year_start = pd.to_datetime(f'{last_obs_date.year}-01-01')
    lookback_date = last_obs_date - pd.Timedelta(days=lookback)
    start_date = min(year_start, lookback_date)
    results['historical_data'] = df.loc[start_date:last_obs_date, current_ndvi_col].to_dict()
    results['historical_data'] = {k.strftime('%Y-%m-%d'): v for k, v in results['historical_data'].items()}
    results['probabilities'] = {}
    results['bin_edges'] = {}

    for day in range(1, forecast_days + 1):
        if forecasts[day]:
            kde = gaussian_kde(forecasts[day])
            min_ndvi = max(min(forecasts[day]) - 0.1, 0.0)
            max_ndvi = min(max(forecasts[day]) + 0.1, 1.0)
            bins = np.linspace(min_ndvi, max_ndvi, 31)
            results['bin_edges'][day] = bins.tolist()
            probs = kde.evaluate(bins)
            probs = probs / np.sum(probs)
            results['probabilities'][day] = probs.tolist()
        else:
            results['probabilities'][day] = []
            results['bin_edges'][day] = []

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
    return results


def plot_ndvi_forecast(results, output_dir, field_id):
    with open(results, 'r') as fp:
        results = json.load(fp)
    last_obs_date = pd.to_datetime(results['last_obs_date'])
    fig = go.Figure()
    plot_start_date = pd.to_datetime(f'{last_obs_date.year}-01-01')
    plot_end_date = last_obs_date + pd.Timedelta(days=len(results['forecasts']))

    hist_data = results['historical_data']
    hist_dates = pd.to_datetime(list(hist_data.keys()))
    hist_values = list(hist_data.values())

    hist_dates_filtered = [d for d in hist_dates if d >= plot_start_date]
    hist_values_filtered = [hist_values[i] for i, d in enumerate(hist_dates) if d >= plot_start_date]

    fig.add_trace(go.Scatter(x=hist_dates_filtered, y=hist_values_filtered, mode='lines', name='Historical Data'))

    for day in range(1, len(results['forecasts']) + 1):
        x_coords = [last_obs_date + pd.Timedelta(days=int(day))] * len(results['forecasts'][str(day)])
        if results['forecasts'][str(day)]:
            fig.add_trace(go.Scatter(x=x_coords, y=results['forecasts'][str(day)],
                                     mode='markers',
                                     marker={'size': 5, 'opacity': 0.7,
                                             'color': results['probabilities'][str(day)],
                                             'colorscale': 'Viridis',
                                             'colorbar': {'title': 'Probability', 'x': 1.15}},
                                     text=[str(y) for y in results['forecast_years'][str(day)]],
                                     name=f'Day {day} Forecasts'))
        else:
            fig.add_trace(go.Scatter(x=x_coords, y=results['forecasts'][str(day)],
                                     mode='markers',
                                     marker={'size': 5},
                                     text=[str(y) for y in results['forecast_years'][str(day)]],
                                     name=f'Day {day} Forecasts'))
    fig.update_xaxes(range=[plot_start_date, plot_end_date])
    if 'html' in output_dir.lower():
        output_file = os.path.join(output_dir, f'{field_id}_ndvi_forecast.html')
        pio.write_html(fig, file=output_file)
    elif 'png' in output_dir.lower():
        output_file = os.path.join(output_dir, f'{field_id}_ndvi_forecast.png')
        pio.write_image(fig, file=output_file)
    else:
        print('Specify output directory containing "html" or "png"')
        return None

    return fig


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

        lulc = sdf.at[site_, 'General classification']

        # if lulc != 'Croplands':
        #     continue

        if site_ not in ['S2']:
            continue

        if site_ in ['US-Bi2', 'US-Dk1']:
            continue

        print(f'\n{site_}: {lulc}')

        run_const = os.path.join(run_data, '4_Flux_Network', 'results', constraint_)
        output_ = os.path.join(run_const, site_)

        prepped_input = os.path.join(output_, f'prepped_input.json')
        ndvi_forecast_ = os.path.join(output_, f'ndvi_forecast.json')
        if not os.path.exists(prepped_input):
            prepped_input = os.path.join(output_, f'prepped_input_{site_}.json')
            ndvi_forecast_ = os.path.join(output_, f'ndvi_forecast_{site_}.json')

        config_, fields_ = initialize_data(config_file, project_ws_, input_data=prepped_input)

        forecast_ndvi(fields_, site_, forecast_days=150, output_json=ndvi_forecast_,
                      similarity_window=7, max_years_to_consider=10)

        out_fig_dir_ = os.path.join(root, 'tutorials', project, 'figures', 'ndvi_forecast', 'html')

        plot_ndvi_forecast(ndvi_forecast_, output_dir=out_fig_dir_, field_id=site_)

# ========================= EOF ====================================================================
