import os
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

project = 'flux'
field_id = 'US-FPe'

root = '/media/research/IrrigationGIS'

d = os.path.join(root, 'et-demands/examples/{}/input_timeseries'.format(project))
flux_obs = os.path.join(root, 'climate/flux_ET_dataset/daily_data_files/{}_daily_data.csv'.format(field_id))

project_dir = '/home/dgketchum/PycharmProjects/et-demands/examples/{}'.format(project)


def preproc():
    obs_file = os.path.join(d, '{}_daily.csv'.format(field_id))
    data = pd.read_csv(obs_file, index_col=0, parse_dates=True)
    data.index = list(range(data.shape[0]))
    data['eta'] = data['etr_mm'] * data['etf_inv_irr']
    data = data[['eta']]
    data.dropna(inplace=True)
    print('preproc mean: {}'.format(data.values.mean()))
    _file = os.path.join(project_dir, 'eta.np')
    np.savetxt(_file, data.values)

    obs = pd.read_csv(flux_obs, index_col=0, parse_dates=True)
    _file = os.path.join(d, 'obs.np')
    np.savetxt(_file, data.values)
    print('Writing Obs to {}'.format(_file))


preproc()
