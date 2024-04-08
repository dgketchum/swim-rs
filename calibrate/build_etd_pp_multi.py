import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyemu.utils import PstFrom
from pyemu import Pst, Matrix


def build_pest(model_dir, pest_dir, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)
    _file, count = None, None

    for k, v in kwargs['pars'].items():
        if 'file' in v.keys():
            _file = v.pop('file')
        pest.add_parameters(_file, 'constant', alt_inst_str='{}_'.format(k), **v)

    for i, fid in enumerate(kwargs['targets']):
        pest.add_observations(kwargs['et_obs']['file'][i], insfile=kwargs['et_obs']['insfile'][i])

        et_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        et_df['dummy_idx'] = ['obs_eta_{}_{}'.format(fid, str(j).rjust(6, '0')) for j in range(et_df.shape[0])]
        captures = [ix for ix, r in et_df.iterrows() if r['etf_irr_ct'] and ix.month in list(range(5, 11))]
        captures = et_df['dummy_idx'].loc[captures]

        d = pest.obs_dfs[i].copy()
        d['weight'] = 0.0
        d['weight'].loc[captures] = 1.0
        d['weight'].loc[np.isnan(d['obsval'])] = 0.0
        d['obsval'].loc[np.isnan(d['obsval'])] = -99.0
        pest.obs_dfs[i] = d

    count = i + 1

    for j, fid in enumerate(kwargs['targets']):
        pest.add_observations(kwargs['swe_obs']['file'][j], insfile=kwargs['swe_obs']['insfile'][j])

        swe_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        swe_df['dummy_idx'] = ['obs_swe_{}_{}'.format(fid, str(j).rjust(6, '0')) for j in range(swe_df.shape[0])]
        valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
        valid = swe_df['dummy_idx'].loc[valid]

        d = pest.obs_dfs[j + count].copy()
        d['weight'] = 0.0
        d['weight'].loc[valid] = 1.0
        d['weight'].loc[np.isnan(d['obsval'])] = 0.0
        d['obsval'].loc[np.isnan(d['obsval'])] = -99.0
        pest.obs_dfs[j + count] = d

    # TODO: write empty dir in pest folder: 'pred'
    # TODO: write localization matrix
    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


def localize_parameters(pst_file):

    et_params = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad']
    snow_params = ['swe_alpha', 'swe_beta']

    par_relation = {'eta': et_params, 'swe': snow_params}

    pst = Pst(pst_file)

    pdict = {}
    for i, r in pst.parameter_data.iterrows():
        if r['pargp'] not in pdict.keys():
            pdict[r['pargp']] = [r['parnme']]
        else:
            pdict[r['pargp']].append(r['parnme'])

    pnames = pst.parameter_data['parnme'].values

    df = Matrix.from_names(pst.nnz_obs_names, pnames).to_dataframe()

    localizer = df.copy()

    sites = list(set([i.split('_')[2] for i in df.index]))

    for s in sites:
        for ob_type, params in par_relation.items():
            idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
            cols = list(np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c] for p in params]).flatten())
            localizer.loc[idx, cols] = 1.0

    mat_file = os.path.join(os.path.dirname(pst_file), 'loc.mat')
    Matrix.from_dataframe(localizer).to_ascii(mat_file)

    pst.pestpp_options["ies_localizer"] = "loc.mat"
    pst.pestpp_options["ies_num_reals"] = 30

    pst.write(pst_file)


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

    et_ins = ['et_{}.ins'.format(fid) for fid in targets_]
    swe_ins = ['swe_{}.ins'.format(fid) for fid in targets_]
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

        'swe_alpha': {'file': p_file,
                      'initial_value': 0.07, 'lower_bound': -0.7, 'upper_bound': 1.5,
                      'pargp': 'snow_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'swe_beta': {'file': p_file,
                     'initial_value': 1.0, 'lower_bound': 0.5, 'upper_bound': 1.7,
                     'pargp': 'snow_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

    })

    pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets_})

    params = []
    for i, (k, v) in enumerate(pars.items()):
        if 'aw_' in k:
            params.append((k, aw[i] * 1000., 'p_{}_0_constant.csv'.format(k)))
        else:
            params.append((k, v['initial_value'], 'p_{}_0_constant.csv'.format(k)))

    idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
    vals = np.array([vals, _names]).T
    df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
    df.to_csv(p_file)

    for e, (ii, r) in enumerate(df.iterrows()):
        pars[ii]['use_rows'] = e
        if 'aw_' in ii:
            pars[ii]['initial_value'] = float(r['value'])

    et_obs_files = ['obs/obs_eta_{}.np'.format(fid) for fid in targets_]
    swe_obs_files = ['obs/obs_swe_{}.np'.format(fid) for fid in targets_]

    dct = {'targets': targets_,
           'inputs': input_csv,
           'et_obs': {'file': et_obs_files,
                      'insfile': et_ins},
           'swe_obs': {'file': swe_obs_files,
                       'insfile': swe_ins},
           'pars': pars
           }

    # build_pest(d, pp_dir, **dct)

    pst_f = '/home/dgketchum/PycharmProjects/swim-rs/examples/tongue/pest/tongue.pst'

    localize_parameters(pst_f)
# ========================= EOF ====================================================================
