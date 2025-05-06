import os
import time
from datetime import datetime

import pandas as pd

from calibrate.calibrate_by_station import run_pest_sequence
from model import obs_field_cycle
from prep import get_flux_sites


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


def run_calibration():
    project_ = '5_Flux_Ensemble'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project_, 'data')
    workers, realizations = 20, 200
    project_ws_ = os.path.join(root, project_)
    config_file = os.path.join(project_ws_, 'config.toml')

    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        data = os.path.join(root, 'tutorials', project_, 'data')
        workers, realizations = 2, 10
        project_ws_ = os.path.join(root, 'tutorials', project_)
        config_file = os.path.join(project_ws_, 'config.toml')

    station_file = os.path.join(data, 'station_metadata.csv')

    sites_ = get_flux_sites(station_file, crop_only=False, western_only=False)
    print(f'{len(sites_)} sites total')

    results = os.path.join(project_ws_, 'results', 'tight')

    incomplete = []
    for site in sites_:

        fcst_params = os.path.join(results, site, f'{site}.3.par.csv')
        if not os.path.exists(fcst_params):
            print(f'{site} has no parameters')
            continue

        modified_date = datetime.fromtimestamp(os.path.getmtime(fcst_params))

        if modified_date > pd.to_datetime('2025-04-20'):
            print(f'remove {site} calibrated {datetime.strftime(modified_date, "%Y-%m-%d")}')
        else:
            print(f'keep {site} calibrated {datetime.strftime(modified_date, "%Y-%m-%d")}')
            incomplete.append(site)

    print(f'{len(sites_)} sites not yet calibrated')
    target_ = 'openet'
    members_ = ['eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop', 'disalexi']

    run_pest_sequence(config_file, project_ws_, workers=workers, target=target_, members=members_,
                      realizations=realizations, select_stations=sites_, pdc_remove=True, overwrite=True)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
