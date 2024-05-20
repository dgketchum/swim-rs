import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def run_flux_sites(ini_path, flux_obs, debug_flag=False, project='tongue'):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag)

    # debug_flag=False just returns the ndarray for writing
    if not debug_flag:
        eta_result, swe_result = df_dct
        for i, fid in enumerate(fields.input['order']):
            pred_eta, pred_swe = eta_result[:, i], swe_result[:, i]
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_eta_{}.np'.format(fid)), pred_eta)
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
            end_time = time.time()
        print('\n\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    if debug_flag:

        targets = fields.input['order']
        first = True

        print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')

        for i, fid in enumerate(targets):

            df = df_dct[fid].copy()
            pred_et = df['et_act'].values

            obs_et = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
            obs_et = np.loadtxt(obs_et)
            cols = ['et_obs'] + list(df.columns)
            df['et_obs'] = obs_et
            df = df[cols]
            df.index = [pd.to_datetime(i) for i in df.index]
            sdf = df.loc['2012-01-01': '2012-12-31']

            comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df.index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df['capture']

            obs = pd.read_csv(flux_obs, index_col=0, parse_dates=True)
            cols = ['et_flux'] + ['et_ssebop'] + list(df.columns)
            df['et_flux'] = obs['ET']

            etf = [v['etf_inv_irr'][i] for k, v in fields.input['time_series'].items()]
            etr = [v['etr_mm'][i] for k, v in fields.input['time_series'].items()]
            df['et_ssebop'] = np.array(etf) * np.array(etr)
            df = df[cols]

            comp = df.loc[df[df['capture'] == 1.0].index].copy()
            et_act, et_ssebop = comp['et_act'], comp['et_ssebop']
            rmse = np.sqrt(((et_act - et_ssebop) ** 2).mean())
            end_time = time.time()
            print('Execution time: {:.2f} seconds\n'.format(end_time - start_time))
            print('{} Capture Dates; Mean SSEBop: {:.2f}, SWB Pred: {:.2f}'.format(comp.shape[0],
                                                                                   et_ssebop.mean(),
                                                                                   et_act.mean()))
            print('SWB ET/SSEBop RMSE: {:.4f}\n\n\n\n'.format(rmse))

            comp = df[~pd.isna(df['et_flux']) == 1].copy()
            comp = comp.loc[comp[comp['capture'] == 1.0].index]
            et_flux, et_ssebop = comp['et_flux'], comp['et_ssebop']
            rmse = np.sqrt(((et_flux - et_ssebop) ** 2).mean())
            print('{} Flux/Capture Dates; Mean Flux: {:.2f}, Mean SSEBop: {:.2f}'.format(comp.shape[0],
                                                                                         et_flux.mean(),
                                                                                         et_ssebop.mean()))
            print('RMSE Flux/SSEBop Capture Dates: {:.4f}\n\n\n\n'.format(rmse))

            comp = df[~pd.isna(df['et_flux']) == 1].copy()
            et_flux, et_ssebop = comp['et_flux'], comp['et_ssebop']
            rmse = np.sqrt(((et_flux - et_ssebop) ** 2).mean())
            print('{} Flux Dates; Mean Flux: {:.2f}, Mean SSEBop: {:.2f}'.format(comp.shape[0],
                                                                                 et_flux.mean(),
                                                                                 et_ssebop.mean()))
            print('RMSE Flux/All SSEBop Dates: {:.4f}\n\n\n\n'.format(rmse))

            comp = df[~pd.isna(df['et_flux']) == 1].copy()
            et_act, et_flux = comp['et_act'], comp['et_flux']
            rmse = np.sqrt(((et_act - et_flux) ** 2).mean())
            print('{} Flux Dates; Mean Flux: {:.2f}, Mean SWB: {:.2f}'.format(comp.shape[0],
                                                                              et_flux.mean(),
                                                                              et_act.mean()))
            print('RMSE Flux/SWB Dates: {:.4f}\n\n\n\n'.format(rmse))
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
    project = 'flux'
    site = 'US-Rws'

    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    conf = os.path.join(d, '{}_swim.toml'.format(project))

    flux_obs_ = os.path.join('/media/research/IrrigationGIS/climate/flux_ET_dataset/'
                             'daily_data_files/{}_daily_data.csv'.format(site))

    run_flux_sites(conf, flux_obs=flux_obs_, project=project, debug_flag=True)
