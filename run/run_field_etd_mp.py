import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def optimize_fields(ini_path, debug_flag=False, field_type='irrigated', project='tongue'):
    start_time = time.time()

    config = ProjectConfig(field_type=field_type)
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df = obs_field_cycle.field_day_loop(config, fields, debug_flag=debug_flag)

    # debug_flag=False just returns the ndarray for writing
    if not debug_flag:
        for i, fid in enumerate(fields.input['order']):
            pred = df[:, 0, i]
            np.savetxt(os.path.join(d, 'pest', 'obs', 'obs_eta_{}.np'.format(fid)), pred)
            end_time = time.time()
        print('\n\nExecution time: {:.2f} seconds'.format(end_time - start_time))

    # debug returns a dataframe
    if debug_flag:

        targets = fields.input['order']
        first = True

        for fid in targets:

            pred = df[fid]['et_act'].values

            obs = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
            obs = np.loadtxt(obs)
            cols = ['et_obs'] + list(df[fid].columns)
            df[fid]['et_obs'] = obs
            df[fid] = df[fid][cols]
            a = df[fid].loc['2010-01-01': '2021-01-01']

            comp = pd.DataFrame(data=np.vstack([obs, pred]).T, columns=['obs', 'pred'], index=df[fid].index)
            comp['eq'] = comp['obs'] == comp['pred']
            comp['capture'] = df[fid]['capture']

            rmse = np.sqrt(((pred - obs) ** 2).mean())
            end_time = time.time()

            if first:
                print('Execution time: {:.2f} seconds'.format(end_time - start_time))
                first = False

            print('{}: Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(fid, obs.mean(), pred.mean()))
            print('{}: RMS Diff: {:.4f}'.format(fid, rmse))

            comp = comp.loc[a[a['capture'] == 1.0].index]
            pred, obs = comp['pred'], comp['obs']
            rmse = np.sqrt(((pred - obs) ** 2).mean())
            print('{}: RMSE Capture Dates: {:.4f}\n\n\n\n'.format(fid, rmse))


if __name__ == '__main__':
    project_ = 'tongue'
    field_type_ = 'irrigated'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project_)
    ini = os.path.join(d, '{}_swim.toml'.format(project_))
    optimize_fields(ini_path=ini, debug_flag=True, field_type=field_type_, project=project_)
