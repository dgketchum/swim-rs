import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def optimize_fields(ini_path, debug_flag=False, project='tongue'):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag)

    # debug_flag=False just returns the ndarray for writing
    if not debug_flag:
        eta_result, swe_result = df
        for i, fid in enumerate(fields.input['order']):
            pred_eta, pred_swe = eta_result[:, i], swe_result[:, i]
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_eta_{}.np'.format(fid)), pred_eta)
            np.savetxt(os.path.join(d, 'pest', 'pred', 'pred_swe_{}.np'.format(fid)), pred_swe)
            end_time = time.time()
        print('\n\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    # debug returns a dataframe
    if debug_flag:

        targets = fields.input['order']
        first = True

        print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')

        for fid in targets:

            pred_et = df[fid]['et_act'].values

            obs_et = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
            obs_et = np.loadtxt(obs_et)
            cols = ['et_obs'] + list(df[fid].columns)
            df[fid]['et_obs'] = obs_et
            df[fid] = df[fid][cols]
            a = df[fid].loc['2010-01-01': '2021-01-01']

            comp = pd.DataFrame(data=np.vstack([obs_et, pred_et]).T, columns=['obs', 'pred'], index=df[fid].index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df[fid]['capture']

            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            end_time = time.time()

            if first:
                print('Execution time: {:.2f} seconds'.format(end_time - start_time))
                first = False

            print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs_et.mean(), pred_et.mean()))
            print('{}: RMSE: {:.4f}'.format(fid, rmse))

            comp = comp.loc[a[a['capture'] == 1.0].index]
            pred_et, obs_et = comp['pred'], comp['obs']
            rmse = np.sqrt(((pred_et - obs_et) ** 2).mean())
            print('{}: RMSE Capture Dates: {:.4f}'.format(fid, rmse))

            obs_swe = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_swe_{}.np'.format(project, fid)
            obs_swe = np.loadtxt(obs_swe)
            cols = ['swe_obs'] + list(df[fid].columns)
            df[fid]['swe_obs'] = obs_swe
            df[fid] = df[fid][cols]
            swe_df = df[fid].loc['2010-01-01': '2021-01-01'][['swe_obs', 'swe']]
            swe_df.dropna(axis=0, inplace=True)
            pred_swe = swe_df['swe'].values
            obs_swe = swe_df['swe_obs'].values
            rmse = np.sqrt(((pred_swe - obs_swe) ** 2).mean())
            print('{}: RMSE SWE: {:.4f}\n\n\n\n'.format(fid, rmse))


if __name__ == '__main__':
    project_ = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project_)
    ini = os.path.join(d, '{}_swim.toml'.format(project_))
    optimize_fields(ini_path=ini, debug_flag=True, project=project_)


