import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix, ObservationEnsemble
from pyemu.utils import PstFrom

from swim.config import ProjectConfig


def build_pest(model_dir, pest_dir, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)
    _file, count = None, None

    for k, v in kwargs['pars'].items():
        if 'file' in v.keys():
            _file = v.pop('file')
        if v['lower_bound'] <= 0.0:
            transform = 'none'
        else:
            transform = 'log'
        pest.add_parameters(_file, 'constant', transform=transform, alt_inst_str='{}_'.format(k), **v)

    obsnme_str = 'oname:obs_etf_{}_otype:arr_i:{}_j:0'

    for i, fid in enumerate(kwargs['targets']):

        # only weight etf on capture dates
        et_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        if 'start' in kwargs.keys():
            et_df = et_df.loc[kwargs['start']: kwargs['end']]
            et_df.to_csv(kwargs['inputs'][i])

        pest.add_observations(kwargs['etf_obs']['file'][i], insfile=kwargs['etf_obs']['insfile'][i])

        et_df['dummy_idx'] = [obsnme_str.format(fid, j) for j in range(et_df.shape[0])]
        captures = [ix for ix, r in et_df.iterrows()
                    if r['etf_irr_ct']
                    or r['etf_inv_irr_ct']
                    and ix.month in list(range(5, 11))]

        captures = et_df['dummy_idx'].loc[captures]

        d = pest.obs_dfs[i].copy()
        d['weight'] = 0.0

        try:
            d.loc[captures, 'weight'] = 1.0
        except KeyError:
            captures = [v.lower() for v in captures.values]
            d.loc[captures, 'weight'] = 1.0

        d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
        d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0

        d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
        d = d.sort_values(by='idx')
        d.drop(columns=['idx'], inplace=True)

        pest.obs_dfs[i] = d

    count = i + 1
    obsnme_str = 'oname:obs_swe_{}_otype:arr_i:{}_j:0'

    for j, fid in enumerate(kwargs['targets']):

        # only weight swe Nov - Apr
        swe_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        if 'start' in kwargs.keys():
            swe_df = swe_df.loc[kwargs['start']: kwargs['end']]
            swe_df.to_csv(kwargs['inputs'][i])

        pest.add_observations(kwargs['swe_obs']['file'][j], insfile=kwargs['swe_obs']['insfile'][j])

        swe_df['dummy_idx'] = [obsnme_str.format(fid, j) for j in range(swe_df.shape[0])]
        valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
        valid = swe_df['dummy_idx'].loc[valid]

        d = pest.obs_dfs[j + count].copy()
        d['weight'] = 0.0

        # TODO: adjust as needed for phi visibility of etf vs. swe
        try:
            d.loc[valid, 'weight'] = 0.03
        except KeyError:
            valid = [v.lower() for v in valid.values]
            d.loc[valid, 'weight'] = 0.03

        d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
        d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0

        d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
        d = d.sort_values(by='idx')
        d.drop(columns=['idx'], inplace=True)

        pest.obs_dfs[j + count] = d

    ofiles = [str(x).replace('obs', 'pred') for x in pest.output_filenames]
    pest.output_filenames = ofiles

    os.makedirs(os.path.join(pest_dir, 'pred'))

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    # the build function doesn't appear to write standard_deviation column in obs data
    pest.build_pst(version=2)

    # the build function wrote a generic python runner that we replace with our own
    # with some work, pymeu build can do this for us
    auto_gen = os.path.join(pest_dir, 'custom_forward_run.py')
    runner = kwargs['python_script']
    shutil.copyfile(runner, auto_gen)

    # clean up the new pest directory
    for dd in ['master', 'workers', 'obs']:
        try:
            shutil.rmtree(os.path.join(pest_dir, dd))
        except FileNotFoundError:
            continue

    pst = Pst(os.path.join(pest.new_d, '{}.pst'.format(os.path.basename(model_dir))))
    obs = pst.observation_data

    obs['standard_deviation'] = 0.00
    etf_idx = [i for i in obs.index if 'etf' in i]
    obs.loc[etf_idx, 'standard_deviation'] = obs['obsval'] * 0.1

    swe_idx = [i for i, r in obs.iterrows() if 'swe' in i and r['obsval'] > 0.0]
    obs.loc[swe_idx, 'standard_deviation'] = obs['obsval'] * 0.02

    # add time information
    obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]

    pst.write(pst.filename, version=2)
    print(f'{len(swe_df)} rows in swe, {len(et_df)} rows in etf')
    print('Configured PEST++ for {} targets, '.format(len(kwargs['targets'])))


def build_localizer(pst_file, ag_json=None, irr_thresh=0.5):
    years = None
    et_params = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad']
    snow_params = ['swe_alpha', 'swe_beta']

    par_relation = {'etf': et_params, 'swe': snow_params}

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

    if ag_json:
        with open(ag_json, 'r') as f:
            input_dct = json.load(f)
        dt = [pd.to_datetime(k) for k, v in input_dct['time_series'].items()]
        dt = pd.Series(index=pd.DatetimeIndex(dt)).sort_index()
        dt.loc[dt.index] = [i for i in range(len(dt))]
        years = list(set([i.year for i in dt.index]))

    track = {k: [] for k in sites}

    irr_d = {k.lower(): v for k, v in input_dct['irr_data'].items()}

    for s in sites:

        for ob_type, params in par_relation.items():

            if ag_json and ob_type == 'etf':

                for yr in years:

                    irr = str(yr) in irr_d[s].keys()
                    if irr:
                        try:
                            f_irr = irr_d[s][str(yr)]['f_irr']
                            irr = f_irr > irr_thresh
                        except KeyError:
                            irr = False

                    t_idx = ['_i:{}_'.format(int(r)) for i, r in dt.items() if i.year == yr]

                    if irr:
                        track[s].append(yr)
                        subset_par = ['ndvi_alpha', 'ndvi_beta', 'mad']
                    else:
                        subset_par = ['aw', 'rew', 'tew']

                    idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
                    idx = [i for i in idx if '_{}_'.format(i.split('_')[4]) in t_idx]
                    cols = list(
                        np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c] for p in subset_par]).flatten())
                    localizer.loc[idx, cols] = 1.0

            else:
                idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
                cols = list(np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c] for p in params]).flatten())
                localizer.loc[idx, cols] = 1.0

    vals = localizer.values
    vals[np.isnan(vals)] = 0.0
    vals[vals < 1.0] = 0.0
    localizer.loc[localizer.index, localizer.columns] = vals.copy()
    mat_file = os.path.join(os.path.dirname(pst_file), 'loc.mat')
    Matrix.from_dataframe(localizer).to_ascii(mat_file)

    pst.write(pst_file, version=2)


def write_control_settings(pst_file, noptmax=-2, reals=250):
    pst = Pst(pst_file)
    pst.pestpp_options["ies_localizer"] = "loc.mat"
    pst.pestpp_options["ies_num_reals"] = reals
    pst.control_data.noptmax = noptmax
    oe = ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=reals)
    oe.to_csv(pst_file.replace('.pst', '.oe.csv'))
    pst.write(pst_file, version=2)


def initial_parameter_dict(param_file):
    p = OrderedDict({
        'aw': {'file': param_file,
               'initial_value': None, 'lower_bound': 15.0, 'upper_bound': 900.0,
               'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'rew': {'file': param_file,
                'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'tew': {'file': param_file,
                'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'ndvi_alpha': {'file': param_file,
                       'initial_value': 0.2, 'lower_bound': -0.7, 'upper_bound': 1.5,
                       'pargp': 'ndvi_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'ndvi_beta': {'file': param_file,
                      'initial_value': 1.25, 'lower_bound': 0.5, 'upper_bound': 1.7,
                      'pargp': 'ndvi_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'mad': {'file': param_file,
                'initial_value': 0.6, 'lower_bound': 0.1, 'upper_bound': 0.9,
                'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'swe_alpha': {'file': param_file,
                      'initial_value': 0.15, 'lower_bound': -0.5, 'upper_bound': 1.,
                      'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'swe_beta': {'file': param_file,
                     'initial_value': 1.5, 'lower_bound': 0.5, 'upper_bound': 2.5,
                     'pargp': 'snow_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

    })

    return p


def get_pest_builder_args(project_ws, input_json, data, start=None, end=None):
    with open(input_json, 'r') as f:
        input_dct = json.load(f)

    targets = input_dct['order']

    aw = [input_dct['props'][t]['awc'] for t in targets]

    input_csv = [os.path.join(data, '{}_daily.csv'.format(fid)) for fid in targets]

    et_ins = ['etf_{}.ins'.format(fid) for fid in targets]
    swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]
    p_file = os.path.join(project_ws, 'params.csv')

    pars = initial_parameter_dict(param_file=p_file)
    pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets})

    params = []
    for i, (k, v) in enumerate(pars.items()):
        if 'aw_' in k:
            aw_ = aw[i] * 1000.
            if np.isnan(aw_):
                aw_ = 150.0
            params.append((k, aw_, 'p_{}_0_constant.csv'.format(k)))
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

    etf_obs_files = ['obs/obs_etf_{}.np'.format(fid) for fid in targets]
    swe_obs_files = ['obs/obs_swe_{}.np'.format(fid) for fid in targets]

    dct = {'targets': targets,
           'inputs': input_csv,
           'etf_obs': {'file': etf_obs_files,
                       'insfile': et_ins},
           'swe_obs': {'file': swe_obs_files,
                       'insfile': swe_ins},
           'pars': pars
           }

    if start and end:
        dct.update({'start': start, 'end': end})

    return dct


if __name__ == '__main__':

    root = os.path.abspath('..')

    project_ws = os.path.join(root, 'tutorials', '3_Crane')
    data = os.path.join(project_ws, 'data')
    ini_path = os.path.join(data, 'tutorial_config.toml')

    config = ProjectConfig()
    config.read_config(ini_path, project_ws)
    start = datetime.strftime(config.start_dt, '%Y-%m-%d')
    end = datetime.strftime(config.end_dt, '%Y-%m-%d')

    # for convenience, we put all the paths we'll need in a dict
    PATHS = {'prepped_input': os.path.join(data, 'prepped_input.json'),
             'plot_timeseries': os.path.join(data, 'plot_timeseries'),
             '_pst': 'fort_peck.pst',
             'exe_': 'pestpp-ies',
             'p_dir': os.path.join(project_ws, 'pest'),
             'm_dir': os.path.join(project_ws, 'master'),
             'w_dir': os.path.join(project_ws, 'workers'),
             'obs': os.path.join(project_ws, 'obs'),
             'python_script': os.path.join(root, 'calibrate', 'custom_forward_run.py')}

    if not os.path.isdir(PATHS['obs']):
        os.makedirs(PATHS['obs'], exist_ok=True)

    dct_ = get_pest_builder_args(project_ws, PATHS['prepped_input'], PATHS['plot_timeseries'],
                                 start=start, end=end)

    # update the dict with the location of 'custom_forward_run.py'
    dct_.update({'python_script': PATHS['python_script']})
    # build_pest(project_ws, PATHS['p_dir'], **dct_)

    pst_f = os.path.join(PATHS['p_dir'], '3_Crane.pst')
    build_localizer(pst_f, ag_json=PATHS['prepped_input'])

    write_control_settings(pst_f, noptmax=3, reals=5)

# ========================= EOF ====================================================================
