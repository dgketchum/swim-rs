import os
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

d = '/home/dgketchum/PycharmProjects/et-demands/examples/tongue/'
field_id = '2100'


def preproc():
    obs_file = os.path.join(d, 'landsat/field_daily/{}_daily.csv'.format(field_id))
    data = pd.read_csv(obs_file, index_col=0, parse_dates=True)
    data.index = list(range(data.shape[0]))
    data['eta'] = data['etr_mm'] * data['ETF_NO_IRR']
    data = data[['eta']]
    data.dropna(inplace=True)
    print('preproc mean: {}'.format(data.values.mean()))
    _file = os.path.join(d, 'eta.np')
    np.savetxt(_file, data.values)
    _file = os.path.join(d, 'obs.np')
    np.savetxt(_file, data.values)
    print('Writing Obs to {}'.format(_file))


preproc()
