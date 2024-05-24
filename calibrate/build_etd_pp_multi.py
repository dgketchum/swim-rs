import os
import json
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyemu.utils import PstFrom
from pyemu import Pst, Matrix, Cov, geostats, ObservationEnsemble


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

    # pyemu writes observations to the control file, as expected, however it also writes the observtion
    # source filename to the input/output section, e.g., 'et_1779.ins obs/obs_eta_1779.np' and during optimization
    # uses obs/obs_eta_1779.np to compare to the observations already written to the control file.
    # I think we've circumvented some functionality where we set things pest will 'observe' in the output of the model
    # see: github.com/gmdsi/GMDSI_notebooks/blob/main/tutorials/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb

    # TODO: need to weight eta and swe according to their relative contribution to error (phi)
    # i.e., weight eta higher, so the swe error doesn't dominate calibration
    # does the fact we are providing a localization matrix obviate this?
    # might be worth it anyway so we have a representative phi
    # see: github.com/gmdsi/GMDSI_notebooks/blob/main/tutorials/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb
    # or automate this with pst.adjust_weights() which will be written to the control file

    obsnme_str = 'oname:obs_eta_{}_otype:arr_i:{}_j:0'

    # TODO assign 'eta' and 'swe' to oname column in pst.observation_data df instead of 'obs'
    for i, fid in enumerate(kwargs['targets']):
        pest.add_observations(kwargs['et_obs']['file'][i], insfile=kwargs['et_obs']['insfile'][i])

        # only weight eta on capture dates
        et_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
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

        d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
        d = d.sort_values(by='idx')
        d.drop(columns=['idx'], inplace=True)

        pest.obs_dfs[i] = d

    count = i + 1
    # note the use of pest-style long names now removes our dependence on dgketchum's hacked fork of pyemu
    # TODO: implement time dimension to observations
    obsnme_str = 'oname:obs_swe_{}_otype:arr_i:{}_j:0'

    for j, fid in enumerate(kwargs['targets']):
        pest.add_observations(kwargs['swe_obs']['file'][j], insfile=kwargs['swe_obs']['insfile'][j])

        # only weight swe Nov - Apr
        swe_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
        swe_df['dummy_idx'] = [obsnme_str.format(fid, j) for j in range(swe_df.shape[0])]
        valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
        valid = swe_df['dummy_idx'].loc[valid]

        d = pest.obs_dfs[j + count].copy()
        d['weight'] = 0.0

        # TODO: adjust as needed for phi visibility of eta vs. swe
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

    # hack to write measurement std post-build, which if not included, 'weight' will be interpreted as std dev
    # this will be used to add noise to non-zero weighted obs data in e.g., tongue.obs+noise.csv
    # TODO: pre-compute observation ensembles, implement autocorrelated transient noise
    # see: github.com/gmdsi/GMDSI_notebooks/blob/main/tutorials/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb
    pst = Pst(os.path.join(pest.new_d, '{}.pst'.format(os.path.basename(model_dir))))
    obs = pst.observation_data
    obs['standard_deviation'] = 0.01
    obs.loc[[i for i in obs.index if 'eta' in i], 'standard_deviation'] = obs['obsval'] * 0.1
    obs.loc[[i for i in obs.index if 'swe' in i], 'standard_deviation'] = obs['obsval'] * 0.02

    # add time information
    obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]

    pst.write(pst.filename, version=2)


def build_localizer(pst_file):
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
                      'initial_value': 0.07, 'lower_bound': -0.7, 'upper_bound': 1.5,
                      'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        'swe_beta': {'file': param_file,
                     'initial_value': 1.0, 'lower_bound': 0.5, 'upper_bound': 1.7,
                     'pargp': 'snow_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

    })

    return p


def get_pest_builder_args(input_json, data):
    with open(input_json, 'r') as f:
        input_dct = json.load(f)

    targets = input_dct['order']

    aw = [input_dct['props'][t]['awc'] for t in targets]

    input_csv = [os.path.join(data, '{}_daily.csv'.format(fid)) for fid in targets]

    et_ins = ['et_{}.ins'.format(fid) for fid in targets]
    swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]
    p_file = os.path.join(d, 'params.csv')

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

    et_obs_files = ['obs/obs_eta_{}.np'.format(fid) for fid in targets]
    swe_obs_files = ['obs/obs_swe_{}.np'.format(fid) for fid in targets]

    dct = {'targets': targets,
           'inputs': input_csv,
           'et_obs': {'file': et_obs_files,
                      'insfile': et_ins},
           'swe_obs': {'file': swe_obs_files,
                       'insfile': swe_ins},
           'pars': pars
           }

    return dct


if __name__ == '__main__':

    data_root = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(data_root):
        data_root = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'flux'
    src = '/home/dgketchum/PycharmProjects/swim-rs'.format(project)
    d = os.path.join(src, 'examples/{}'.format(project))
    python_script = os.path.join(src, 'calibrate', 'custom_forward_run.py')

    input_ = os.path.join(data_root, 'examples/{}/prepped_input/{}_input_sample.json'.format(project, project))
    data_ = os.path.join(data_root, 'examples/{}/input_timeseries'.format(project))

    pest_dir_ = os.path.join(d, 'pest')
    pst_f = os.path.join(pest_dir_, '{}.pst'.format(project))

    dct_ = get_pest_builder_args(input_, data_)
    # noinspection PyTypedDict
    dct_.update({'python_script': python_script})
    build_pest(d, pest_dir_, **dct_)
    build_localizer(pst_f)
    write_control_settings(pst_f, 6, 100)

# ========================= EOF ====================================================================
