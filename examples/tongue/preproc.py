import os
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

project = 'tongue'
field_ids = [1778, 1791, 1804, 1853, 1375]

root = '/media/research/IrrigationGIS'

d = os.path.join(root, 'swim/examples/{}/input_timeseries'.format(project))

project_dir = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)


def preproc():

    for fid in field_ids:
        obs_file = os.path.join(d, '{}_daily.csv'.format(fid))
        data = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        data.index = list(range(data.shape[0]))
        data['eta'] = data['etr_mm'] * data['etf_inv_irr']
        data = data[['eta']]
        print('preproc mean: {}'.format(np.nanmean(data.values)))
        _file = os.path.join(project_dir, 'obs', 'obs_eta_{}.np'.format(fid))
        np.savetxt(_file, data.values)
        print('Wrote obs to {}'.format(_file))


preproc()
