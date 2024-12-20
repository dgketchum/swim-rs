import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots

from prep.prep_plots import FLUX_SELECT


def run_flux_sites(ini_path, flux_file, project='tongue', calibration_dir=None, parameter_distribution=None,
                   write_files=None):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, calibration_folder=calibration_dir, parameter_dist_csv=parameter_distribution)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    targets = fields.input['order']

    print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')
    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    for i, fid in enumerate(targets):
        df = df_dct[fid].copy()
        pred_et = df['et_act'].values

        print(fid)
        obs_etf = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_etf_{}.np'.format(project, fid)
        obs_etf = np.loadtxt(obs_etf)
        cols = ['etf_obs'] + list(df.columns)
        df['etf_obs'] = obs_etf

        obs_eta = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
        obs_eta = np.loadtxt(obs_eta)
        cols = ['eta_obs'] + cols
        df['eta_obs'] = obs_eta

        df = df[cols]
        df.index = [pd.to_datetime(i) for i in df.index]

        if write_files:
            file_ = write_files.format(fid)
            df.to_csv(file_)

        if not flux_file:
            continue

        comp = pd.DataFrame(data=np.vstack([obs_eta, pred_et]).T, columns=['obs', 'pred'], index=df.index)
        comp['eq'] = comp['obs'] == comp['pred']
        comp['capture'] = df['capture']

        flux_obs = pd.read_csv(flux_file.format(fid), index_col=0, parse_dates=True)
        cols = ['et_flux'] + list(df.columns)
        df['et_flux'] = flux_obs['ET']
        df = df[cols]
        sdf = df.loc['2014-01-01': '2014-12-31']

        comp = df.loc[df[df['capture'] == 1.0].index].copy()
        et_act, et_ssebop = comp['et_act'], comp['eta_obs']
        rmse = np.sqrt(((et_act - et_ssebop) ** 2).mean())

        print('{} Capture Dates; Mean SSEBop: {:.2f}, SWB Pred: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                             et_ssebop.mean(),
                                                                                             et_act.mean(), rmse))

        comp = df[~pd.isna(df['et_flux']) == 1].copy()
        comp = comp.loc[comp[comp['capture'] == 1.0].index]
        et_flux, et_ssebop = comp['et_flux'], comp['eta_obs']
        rmse = np.sqrt(((et_flux - et_ssebop) ** 2).mean())
        print('{} Flux/Capture Dates; Mean Flux: {:.2f}, Mean SSEBop: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                                   et_flux.mean(),
                                                                                                   et_ssebop.mean(),
                                                                                                   rmse))

        comp = df[~pd.isna(df['et_flux']) == 1].copy()
        et_flux, et_ssebop = comp['et_flux'], comp['eta_obs']
        rmse = np.sqrt(((et_flux - et_ssebop) ** 2).mean())
        print('{} Flux Dates; Mean Flux: {:.2f}, Mean SSEBop: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                           et_flux.mean(),
                                                                                           et_ssebop.mean(), rmse))
        comp = df[~pd.isna(df['et_flux']) == 1].copy()
        et_act, et_flux = comp['et_act'], comp['et_flux']
        comp['res'] = comp['et_act'] - comp['et_flux']
        comp = comp[['res'] + cols]
        rmse = np.sqrt(((et_act - et_flux) ** 2).mean())
        print('{} Flux Dates; Mean Flux: {:.2f}, Mean SWB: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                        et_flux.mean(),
                                                                                        et_act.mean(), rmse))
        totals = df[['et_act', 'ppt', 'dperc', 'runoff']].sum(axis=0)
        water_out = totals[['dperc', 'et_act', 'runoff']].sum()
        storage = df.loc[df.index[0], 'depl_root'] - df.loc[df.index[-1], 'depl_root']
        balance = totals['ppt'] - storage - water_out
        print('Water Balance = {:.1f}; input: {:.1f}; output: {:.1f}; storage: {:.1f}\n\n\n'.format(balance,
                                                                                                    totals['ppt'],
                                                                                                    water_out,
                                                                                                    storage))
    return None


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    project = 'flux'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    conf = os.path.join(d, '{}_swim.toml'.format(project))

    flux_obs_ = os.path.join(root, 'climate/flux_ET_dataset/daily_data_files/{}_daily_data.csv')

    calibration_folder = None

    tuned = '/media/research/IrrigationGIS/swim/examples/flux/calibrated_models/model_6JUN2024'
    pars = os.path.join(tuned, 'flux.4.par.csv')

    results_files = os.path.join(tuned, 'output_{}.csv')

    run_flux_sites(conf, flux_file=flux_obs_, project=project, calibration_dir=calibration_folder,
                   parameter_distribution=pars, write_files=results_files)
