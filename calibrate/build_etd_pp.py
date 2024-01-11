import os

import pandas as pd
from pyemu.utils import PstFrom


def build_pest_etd(model_dir, pest_dir, input_data, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)

    for k, v in kwargs['pars'].items():
        _file = v.pop('file')
        pest.add_parameters(_file, 'constant', **v)

    pest.add_observations(kwargs['obs']['file'], insfile=kwargs['obs']['insfile'])

    idf = pd.read_csv(input_data, index_col=0, parse_dates=True)
    idf['dummy_idx'] = ['eta_{}'.format(str(i).rjust(6, '0')) for i in range(idf.shape[0])]
    captures = [i for i, r in idf.iterrows() if r['etf_inv_irr_ct'] and i.month in list(range(5, 11))]
    captures = idf['dummy_idx'].loc[captures]

    d = pest.obs_dfs[0].copy()
    d['weight'] = 0.0
    d['weight'].loc[captures] = 1.0
    pest.obs_dfs[0] = d

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


if __name__ == '__main__':

    fid = 'US-FPe'
    project = 'flux'
    d = '/home/dgketchum/PycharmProjects/et-demands/examples/{}'.format(project)

    data = '/media/research/IrrigationGIS/et-demands/examples/{}/input_timeseries'.format(project)
    input_csv = os.path.join(data, '{}_daily.csv'.format(fid))

    pp_dir = os.path.join(d, 'pest')

    ins = '{}.ins'.format(fid)
    p_file = os.path.join(d, 'params.csv')

    dct = {'obs': {'file': 'eta.np',
                   'insfile': ins},

           'pars': {
               'aw': {'file': p_file,
                      'initial_value': 145.0, 'lower_bound': 100.0, 'upper_bound': 1000.0,
                      'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': 0},

               'rew': {'file': p_file,
                       'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                       'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': 1},

               'tew': {'file': p_file,
                       'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                       'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': 2},

               'ndvi_alpha': {'file': p_file,
                              'initial_value': -0.2, 'lower_bound': -0.7, 'upper_bound': 1.5,
                              'pargp': 'ndvi_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': 3},

               'ndvi_beta': {'file': p_file,
                             'initial_value': 0.8, 'lower_bound': 0.5, 'upper_bound': 1.7,
                             'pargp': 'ndvi_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': 4},

               'ndvi_fc': {'file': p_file,
                           'initial_value': 1.0, 'lower_bound': 0.6, 'upper_bound': 1.5,
                           'pargp': 'ndvi_fc', 'index_cols': 0, 'use_cols': 1, 'use_rows': 5},

               'mad': {'file': p_file,
                       'initial_value': 0.6, 'lower_bound': 0.1, 'upper_bound': 0.9,
                       'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': 6},

           }
           }
    build_pest_etd(d, pp_dir, input_csv, **dct)
# ========================= EOF ====================================================================
