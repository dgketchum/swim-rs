import os
import time
import collections
from datetime import datetime
from pprint import pprint
import pandas as pd

from ssebop_evaluation import evaluate_ssebop_site
from swimrs.model.initialize import initialize_data
from swimrs.model import obs_field_cycle
from swimrs.prep import get_flux_sites


def run_flux_sites(fid, config, plot_data, outfile):
    start_time = time.time()

    df_dct = obs_field_cycle.field_day_loop(config, plot_data, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    df = df_dct[fid].copy()
    in_df = plot_data.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)

    df = df.loc[config.start_dt:config.end_dt]

    df.to_csv(outfile)


def compare_ssebop(fid, flux_file, model_output, plot_data_,
                   return_comparison=False, gap_tolerance=5):
    """Compare SWIM and SSEBop against flux observations for a single site."""
    irr_ = plot_data_.input['irr_data'][fid]
    daily, overpass, monthly = evaluate_ssebop_site(
        model_output, flux_file,
        irr=irr_,
        gap_tolerance=gap_tolerance
    )

    if monthly is None:
        return None

    agg_comp = monthly.copy()
    if len(agg_comp) < 3:
        return None

    rmse_values = {k.split('_')[1]: v for k, v in agg_comp.items() if k.startswith('rmse_')
                   if 'swim' in k or 'ssebop' in k}

    if len(rmse_values) == 0:
        return None

    lowest_rmse_model = min(rmse_values, key=rmse_values.get)
    print(f"n Samples: {agg_comp['n_samples']}")
    print('Lowest RMSE:', lowest_rmse_model)

    if not return_comparison:
        return lowest_rmse_model

    try:
        print(f"Flux Mean: {agg_comp['mean_flux']}")
        print(f"SWIM Mean: {agg_comp['mean_swim']}")
        print(f"SSEBop NHM Mean: {agg_comp.get('mean_ssebop')}")
        print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
        print(f"SSEBop NHM RMSE: {agg_comp.get('rmse_ssebop')}")
        return lowest_rmse_model

    except KeyError as exc:
        print(fid, exc)
        return None


if __name__ == '__main__':

    project = '6_Flux_International'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    project_ws_ = os.path.join(root, project)
    if not os.path.isdir(root):
        root = '/home/dgketchum/code/swim-rs'
        project_ws_ = os.path.join(root, 'examples', project)
        data = os.path.join(project_ws_, 'data')

    config_file = os.path.join(project_ws_, 'config.toml')

    station_file = os.path.join(data, 'station_metadata.csv')
    sites, sdf = get_flux_sites(station_file, crop_only=False, return_df=True)

    incomplete, complete, results = [], [], []

    overwrite_ = False

    for ee, site_ in enumerate(sites):

        lulc = sdf.at[site_, 'General classification']

        # if lulc != 'Croplands':
        #     continue

        if site_ in ['US-Bi2', 'US-Dk1', 'JPL1_JV114']:
            continue

        if site_ not in ['US-Ro4']:
            continue

        print(f'\n{ee} {site_}: {lulc}')

        run_const = os.path.join(project_ws_, 'results', 'verify')
        output_ = os.path.join(run_const, site_)

        prepped_input = os.path.join(output_, f'prepped_input.json')
        spinup_ = os.path.join(output_, f'spinup.json')

        if not os.path.exists(prepped_input):
            prepped_input = os.path.join(output_, f'prepped_input_{site_}.json')
            spinup_ = os.path.join(output_, f'spinup_{site_}.json')

        flux_dir = os.path.join(project_ws_, 'data', 'daily_flux_files')
        flux_data = os.path.join(flux_dir, f'{site_}_daily_data.csv')

        fcst_params = os.path.join(output_, f'{site_}.3.par.csv')
        if not os.path.exists(fcst_params):
            continue

        modified_date = datetime.fromtimestamp(os.path.getmtime(fcst_params))
        print(f'Calibration made {modified_date}')
        if modified_date < pd.to_datetime('2025-07-01'):
            continue

        cal = os.path.join(project_ws_, f'tight_pest', 'mult')

        out_csv = os.path.join(output_, f'{site_}.csv')

        config_, fields_ = initialize_data(config_file, project_ws_, input_data=prepped_input, spinup_data=spinup_,
                                           forecast=True, forecast_file=fcst_params)

        try:
            if not os.path.exists(out_csv) or overwrite_:
                run_flux_sites(site_, config_, fields_, out_csv)
        except ValueError as exc:
            print(f'{site_} error: {exc}')
            continue

        result = compare_ssebop(site_, flux_data, out_csv, fields_,
                                return_comparison=True, gap_tolerance=5)

        if result:
            results.append((result, lulc))

        complete.append(site_)

        out_fig_dir_ = os.path.join(root, 'examples', project, 'figures', 'model_output', 'png')

        # flux_pdc_timeseries(run_const, flux_dir, [site_], out_fig_dir=out_fig_dir_, spec='flux', model=model,
        #                     members=['ssebop', 'disalexi', 'geesebal', 'eemetric', 'ptjpl', 'sims'])

    pprint({s: [t[0] for t in results].count(s) for s in set(t[0] for t in results)})
    pprint(
        {category: [item[0] for item in collections.Counter(t[0] for t in results
                                                            if t[1] == category).most_common(3)] for
         category in set(t[1] for t in results)})
    print(f'complete: {complete}')
    print(f'incomplete: {incomplete}')
# ========================= EOF ====================================================================
