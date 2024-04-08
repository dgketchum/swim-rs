import os
import json
from collections import OrderedDict

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

    os.makedirs(pest_dir, 'pred')

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


if __name__ == '__main__':

    project = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)

    input_ = '/media/research/IrrigationGIS/swim/examples/{}/prepped_input/{}_input.json'.format(project, project)
    with open(input_, 'r') as f:
        input_ = json.load(f)
    targets_ = input_['order']
    aw = [input_['props'][t]['awc'] for t in targets_]

    data = '/media/research/IrrigationGIS/swim/examples/{}/input_timeseries'.format(project)
    input_csv = [os.path.join(data, '{}_daily.csv'.format(fid)) for fid in targets_]

    pp_dir = os.path.join(d, 'pest')

    ins = ['{}.ins'.format(fid) for fid in targets_]
    p_file = os.path.join(d, 'params.csv')

    pars = OrderedDict({
        'aw': {'file': p_file,
               'initial_value': None, 'lower_bound': 15.0, 'upper_bound': 700.0,
               'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'rew': {'file': p_file,
                'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'tew': {'file': p_file,
                'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'ndvi_alpha': {'file': p_file,
                       'initial_value': 0.2, 'lower_bound': -0.7, 'upper_bound': 1.5,
                       'pargp': 'ndvi_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'ndvi_beta': {'file': p_file,
                      'initial_value': 1.25, 'lower_bound': 0.5, 'upper_bound': 1.7,
                      'pargp': 'ndvi_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'mad': {'file': p_file,
                'initial_value': 0.6, 'lower_bound': 0.1, 'upper_bound': 0.9,
                'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'snow_alpha': {'file': p_file,
                       'initial_value': 0.07, 'lower_bound': -0.7, 'upper_bound': 1.5,
                       'pargp': 'snow_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'snow_beta': {'file': p_file,
                      'initial_value': 1.0, 'lower_bound': 0.5, 'upper_bound': 1.7,
                      'pargp': 'snow_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

    })

    pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets_})

    params = []
    for i, (k, v) in enumerate(pars.items()):
        if 'aw_' in k:
            params.append((k, aw[i] * 1000., 'p_inst{}_constant.csv'.format(i)))
        else:
            params.append((k, v['initial_value'], 'p_inst{}_constant.csv'.format(i)))

    idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
    vals = np.array([vals, _names]).T
    df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
    df.to_csv(p_file)

    for e, (i, r) in enumerate(df.iterrows()):
        pars[i]['use_rows'] = e
        if 'aw_' in i:
            pars[i]['initial_value'] = float(r['value'])

    obs_files = ['obs/obs_eta_{}.np'.format(fid) for fid in targets_]

    dct = {'targets': targets_,
           'inputs': input_csv,
           'obs': {'file': obs_files,
                   'insfile': ins},
           'pars': pars
           }

    build_pest(d, pp_dir, **dct)
# ========================= EOF ====================================================================
