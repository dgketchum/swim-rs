import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def run_fields(ini_path, debug_flag=False, field_type='irrigated', target_fields=None, project='tongue'):
    config = ProjectConfig(field_type=field_type)
    config.read_config(ini_path)

    fields = SamplePlots()
    fields.initialize_plot_data(config, targets=target_fields)

    cell_count = 0
    for fid, field in sorted(fields.fields_dict.items()):

        cell_count += 1

        start_time = time.time()
        df = obs_field_cycle.field_day_loop(config, field, debug_flag=debug_flag)
        pred = df['et_act'].values

        np.savetxt(os.path.join(d, 'pest', 'eta.np'), pred)

        obs = '/home/dgketchum/PycharmProjects/et-demands/examples/{}/obs.np'.format(project)
        obs = np.loadtxt(obs)
        cols = ['et_obs'] + list(df.columns)
        df['et_obs'] = obs
        df = df[cols]
        a = df.loc['2019-01-01': '2021-01-01']

        comp = pd.DataFrame(data=np.vstack([obs, pred]).T, columns=['obs', 'pred'], index=df.index)
        comp['eq'] = comp['obs'] == comp['pred']

        rmse = np.sqrt(((pred - obs) ** 2).mean())
        end_time = time.time()
        print('Execution time: {:.2f} seconds'.format(end_time - start_time))
        print('Mean Obs: {:.2f}, Mean Pred: {:.2f}'.format(obs.mean(), pred.mean()))
        print('RMS Diff: {:.4f}\n\n\n\n'.format(rmse))

        comp = comp.loc[a[a['capture'] == 1.0].index]
        pred, obs = comp['pred'], comp['obs']
        rmse = np.sqrt(((pred - obs) ** 2).mean())
        print('RMSE Capture Dates: {:.4f}\n\n\n\n'.format(rmse))
        pass


if __name__ == '__main__':
    project = 'tongue'
    targets_ = [1778, 1791, 1804, 1853, 1375]
    field_type = 'unirrigated'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    ini = os.path.join(d, '{}_swim.toml'.format(project))
    run_fields(ini_path=ini, debug_flag=False, field_type=field_type,
               target_fields=targets_, project=project)
