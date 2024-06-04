import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots

from prep.prep_plots import FLUX_SELECT


def run_flux_sites(ini_path, flux_file, debug_flag=False, project='tongue', calibration_dir=None,
                   parameter_distribution=None):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, calibration_folder=calibration_dir, parameter_dist_csv=parameter_distribution)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag)

    if not debug_flag:
        etf_result, swe_result = df_dct
        for i, fid in enumerate(fields.input['order']):
            pred_eta, pred_swe = etf_result[:, i], swe_result[:, i]
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_etf_{}.np'.format(fid)), pred_eta)
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
            end_time = time.time()
        print('\n\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    if debug_flag:

        targets = fields.input['order']

        print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')
        end_time = time.time()
        print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

        for i, fid in enumerate(targets):
            df = df_dct[fid].copy()
            pred_et = df['et_act'].values

            print(fid)
            obs_et = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
            obs_et = np.loadtxt(obs_et)
            cols = ['et_obs'] + list(df.columns)
            df['et_obs'] = obs_et
            df = df[cols]
            df.index = [pd.to_datetime(i) for i in df.index]

            comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df.index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df['capture']

            flux_obs = pd.read_csv(flux_file.format(fid), index_col=0, parse_dates=True)
            cols = ['et_flux'] + list(df.columns)
            df['et_flux'] = flux_obs['ET']
            df = df[cols]
            sdf = df.loc['2014-01-01': '2014-12-31']

            comp = df.loc[df[df['capture'] == 1.0].index].copy()
            et_act, et_ssebop = comp['et_act'], comp['et_obs']
            rmse = np.sqrt(((et_act - et_ssebop) ** 2).mean())

            print('{} Capture Dates; Mean SSEBop: {:.2f}, SWB Pred: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                                 et_ssebop.mean(),
                                                                                                 et_act.mean(), rmse))

            comp = df[~pd.isna(df['et_flux']) == 1].copy()
            comp = comp.loc[comp[comp['capture'] == 1.0].index]
            et_flux, et_ssebop = comp['et_flux'], comp['et_obs']
            rmse = np.sqrt(((et_flux - et_ssebop) ** 2).mean())
            print('{} Flux/Capture Dates; Mean Flux: {:.2f}, Mean SSEBop: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                                       et_flux.mean(),
                                                                                                       et_ssebop.mean(),
                                                                                                       rmse))

            comp = df[~pd.isna(df['et_flux']) == 1].copy()
            et_flux, et_ssebop = comp['et_flux'], comp['et_obs']
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

    # calibration_folder = '/home/dgketchum/PycharmProjects/swim-rs/examples/flux/master/mult'
    calibration_folder = None

    tuned = '/media/research/IrrigationGIS/swim/examples/flux/calibrated_models/two_model_03JUN2024/flux.3.par.csv'
    # tuned_params = None

    run_flux_sites(conf, flux_file=flux_obs_, project=project, debug_flag=True,
                   calibration_dir=calibration_folder, parameter_distribution=tuned)
