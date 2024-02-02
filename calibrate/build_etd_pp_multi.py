import os

import numpy as np
import pandas as pd
from pyemu.utils import PstFrom


def build_pest(model_dir, pest_dir, **kwargs):

    pest = PstFrom(model_dir, pest_dir, remove_existing=True)
    _file = None

    for k, v in kwargs['pars'].items():
        if 'file' in v.keys():
            _file = v.pop('file')
        pest.add_parameters(_file, 'constant', **v)

    for i, fid in enumerate(kwargs['targets']):

        pest.add_observations(kwargs['obs']['file'][i], insfile=kwargs['obs']['insfile'][i])
        idf = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        idf['dummy_idx'] = ['obs_eta_{}_{}'.format(fid, str(i).rjust(6, '0')) for i in range(idf.shape[0])]
        captures = [i for i, r in idf.iterrows() if r['etf_irr_ct'] and i.month in list(range(5, 11))]
        captures = idf['dummy_idx'].loc[captures]

        d = pest.obs_dfs[i].copy()
        d['weight'] = 0.0
        d['weight'].loc[captures] = 1.0
        d['weight'].loc[np.isnan(d['obsval'])] = 0.0
        d['obsval'].loc[np.isnan(d['obsval'])] = -99.0
        pest.obs_dfs[i] = d

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


if __name__ == '__main__':

    targets_ = [1778, 1791, 1804, 1853, 1375]
    project = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)

    data = '/media/research/IrrigationGIS/swim/examples/{}/input_timeseries'.format(project)

    input_csv = [os.path.join(data, '{}_daily.csv'.format(fid)) for fid in targets_]

    pp_dir = os.path.join(d, 'pest')

    ins = ['{}.ins'.format(fid) for fid in targets_]
    p_file = os.path.join(d, 'params.csv')

    pars = {
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

               'mad': {'file': p_file,
                       'initial_value': 0.6, 'lower_bound': 0.1, 'upper_bound': 0.9,
                       'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': 6},

           }

    pars = {'{}_{}'.format(k, fid): v for k, v in pars.items() for fid in targets_}

    params = [(k, v['initial_value']) for k, v in pars.items()]
    idx, vals = [x[0] for x in params], [x[1] for x in params]
    pd.Series(index=idx, data=vals, name='Name').to_csv(p_file)

    obs_files = ['obs/obs_eta_{}.np'.format(fid) for fid in targets_]

    dct = {'targets': targets_,
           'inputs': input_csv,
           'obs': {'file': obs_files,
                   'insfile': ins},
           'pars': pars
           }

    build_pest(d, pp_dir, **dct)
# ========================= EOF ====================================================================
