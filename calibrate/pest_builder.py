import os
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix, ObservationEnsemble
from pyemu.utils import PstFrom
from pyemu.utils.os_utils import run_ossystem, run_sp

from swim.config import ProjectConfig
from swim.input import SamplePlots


class PestBuilder:

    def __init__(self, config_file, project_ws, use_existing=False, python_script=None):

        self.project_ws = project_ws

        self.config = ProjectConfig()
        self.config.read_config(config_file, project_ws)

        self.plots = SamplePlots()
        self.plots.initialize_plot_data(self.config)
        self.irr = self.plots.input['irr_data']
        self.plot_order = self.plots.input['order']
        self.plot_properties = self.plots.input['props']
        self.plot_time_series = self.plots.input['time_series']

        self.params_file = None
        self.pest = None

        self.pest_dir = os.path.join(project_ws, 'pest')
        self.master_dir = os.path.join(project_ws, 'master')
        self.workers_dir = os.path.join(project_ws, 'workers')
        self.obs_dir = os.path.join(project_ws, 'obs')

        self.pst_file = os.path.join(self.pest_dir, f'{self.config.project_name}.pst')
        self.obs_idx_file = os.path.join(self.pest_dir, f'{self.config.project_name}.idx.csv')

        self.pest_args = self.get_pest_builder_args()

        if python_script is None:
            python_script = os.path.join(os.path.dirname(__file__), 'custom_forward_run.py')
            print(f'Using default Python script at: {python_script}')

        self.python_script = python_script
        self.pest_args.update({'python_script': self.python_script})

        if use_existing:
            self.overwrite_build = False
        else:
            self.overwrite_build = True

    def get_pest_builder_args(self):

        targets = self.plot_order

        aw = [self.plot_properties[t]['awc'] for t in targets]

        input_csv = [os.path.join(self.config.plot_timeseries, '{}_daily.csv'.format(fid)) for fid in targets]

        et_ins = ['etf_{}.ins'.format(fid) for fid in targets]
        swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]
        self.params_file = os.path.join(self.project_ws, 'params.csv')

        pars = self.initial_parameter_dict()
        pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets})

        params = []
        for i, (k, v) in enumerate(pars.items()):
            if 'aw_' in k:
                aw_ = aw[i] * 1000.
                if np.isnan(aw_) or aw_ < pars[k]['lower_bound']:
                    aw_ = 150.0
                params.append((k, aw_, 'p_{}_0_constant.csv'.format(k)))
            else:
                params.append((k, v['initial_value'], 'p_{}_0_constant.csv'.format(k)))

        idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
        vals = np.array([vals, _names]).T
        df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
        df.to_csv(self.params_file)

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

        return dct

    def build_pest(self):

        if self.overwrite_build is False:
            raise NotImplementedError('Use of exising Pest++ project was specified, '
                                      'running "build_pest" will overwrite it.')

        self.pest = PstFrom(self.project_ws, self.pest_dir, remove_existing=True)
        _file, count = None, None

        for k, v in self.pest_args['pars'].items():
            if 'file' in v.keys():
                _file = v.pop('file')
            if v['lower_bound'] <= 0.0:
                transform = 'none'
            else:
                transform = 'log'
            self.pest.add_parameters(_file, 'constant', transform=transform, alt_inst_str='{}_'.format(k), **v)

        obsnme_str = 'oname:obs_etf_{}_otype:arr_i:{}_j:0'

        for i, fid in enumerate(self.pest_args['targets']):

            # only weight etf on capture dates
            et_df = pd.read_csv(self.pest_args['inputs'][i], index_col=0, parse_dates=True)
            if 'start' in self.pest_args.keys():
                et_df = et_df.loc[self.config.start_dt: self.config.end_dt]
                et_df.to_csv(self.pest_args['inputs'][i])

            self.pest.add_observations(self.pest_args['etf_obs']['file'][i],
                                       insfile=self.pest_args['etf_obs']['insfile'][i])

            et_df['obs_id'] = [obsnme_str.format(fid, j).lower() for j in range(et_df.shape[0])]
            idx = et_df['obs_id']
            idx.to_csv(self.obs_idx_file)

            captures = [ix for ix, r in et_df.iterrows()
                        if r['etf_irr_ct']
                        or r['etf_inv_irr_ct']
                        and ix.month in list(range(1, 13))]

            captures = et_df['obs_id'].loc[captures]

            d = self.pest.obs_dfs[i].copy()
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

            self.pest.obs_dfs[i] = d

        count = i + 1
        obsnme_str = 'oname:obs_swe_{}_otype:arr_i:{}_j:0'

        for j, fid in enumerate(self.pest_args['targets']):

            # only weight swe Nov - Apr
            swe_df = pd.read_csv(self.pest_args['inputs'][i], index_col=0, parse_dates=True)
            if 'start' in self.pest_args.keys():
                swe_df = swe_df.loc[self.config.start_dt: self.config.end_dt]
                swe_df.to_csv(self.pest_args['inputs'][i])

            self.pest.add_observations(self.pest_args['swe_obs']['file'][j],
                                       insfile=self.pest_args['swe_obs']['insfile'][j])

            swe_df['obs_id'] = [obsnme_str.format(fid, j) for j in range(swe_df.shape[0])]
            valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
            valid = swe_df['obs_id'].loc[valid]

            d = self.pest.obs_dfs[j + count].copy()
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

            self.pest.obs_dfs[j + count] = d

        ofiles = [str(x).replace('obs', 'pred') for x in self.pest.output_filenames]
        self.pest.output_filenames = ofiles

        os.makedirs(os.path.join(self.pest_dir, 'pred'))

        self.pest.py_run_file = 'custom_forward_run.py'
        self.pest.mod_command = 'python custom_forward_run.py'

        # the build function doesn't appear to write standard_deviation column in obs data
        self.pest.build_pst(version=2)

        # the build function wrote a generic python runner that we replace with our own
        # with some work, pymeu build can do this for us
        auto_gen = os.path.join(self.pest_dir, 'custom_forward_run.py')
        runner = self.pest_args['python_script']
        shutil.copyfile(runner, auto_gen)

        # clean up the new pest directory
        for dd in ['master', 'workers', 'obs']:
            try:
                shutil.rmtree(os.path.join(self.pest_dir, dd))
            except FileNotFoundError:
                continue

        pst = Pst(self.pst_file)
        obs = pst.observation_data

        obs['standard_deviation'] = 0.00
        etf_idx = [i for i in obs.index if 'etf' in i]
        obs.loc[etf_idx, 'standard_deviation'] = obs['obsval'] * 0.3

        swe_idx = [i for i, r in obs.iterrows() if 'swe' in i and r['obsval'] > 0.0]
        obs.loc[swe_idx, 'standard_deviation'] = obs['obsval'] * 0.02

        # add time information
        obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]


        pst.write(pst.filename, version=2)
        print(f'{len(swe_df)} rows in swe, {len(et_df)} rows in etf')
        print('Configured PEST++ for {} targets, '.format(len(self.pest_args['targets'])))

    def build_localizer(self):

        et_params = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad']
        snow_params = ['swe_alpha', 'swe_beta']

        par_relation = {'etf': et_params, 'swe': snow_params}

        pst = Pst(self.pst_file)

        pdict = {}
        for i, r in pst.parameter_data.iterrows():
            if r['pargp'] not in pdict.keys():
                pdict[r['pargp']] = [r['parnme']]
            else:
                pdict[r['pargp']].append(r['parnme'])

        pnames = pst.parameter_data['parnme'].values

        df = Matrix.from_names(pst.nnz_obs_names, pnames).to_dataframe()

        localizer = df.copy()

        # TODO: fix this
        # most brittle line of code ever written
        sites = list(set(['_'.join(i.split('_')[2:-3]) for i in df.index]))

        track = {k: [] for k in sites}

        irr_d = {k.lower(): v for k, v in self.irr.items()}

        dt = [pd.to_datetime(k) for k, v in self.plot_time_series.items()]
        years = list(range(self.config.start_dt.year, self.config.end_dt.year + 1))

        for s in sites:

            for ob_type, params in par_relation.items():

                if ob_type == 'etf':

                    for yr in years:

                        irr = str(yr) in irr_d[s].keys()
                        if irr:
                            try:
                                f_irr = irr_d[s][str(yr)]['f_irr']
                                irr = f_irr > self.config.irr_threshold
                            except KeyError:
                                irr = False

                        t_idx = ['_i:{}_'.format(int(i)) for i, r in enumerate(dt) if r.year == yr]

                        if irr:
                            track[s].append(yr)
                            subset_par = ['ndvi_alpha', 'ndvi_beta', 'mad']
                        else:
                            subset_par = ['aw', 'rew', 'tew']

                        idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
                        idx = [i for i in idx if '_{}_'.format(i.split('_')[4]) in t_idx]
                        cols = list(
                            np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c]
                                      for p in subset_par]).flatten())
                        localizer.loc[idx, cols] = 1.0

                else:
                    idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
                    cols = list(np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c]
                                          for p in params]).flatten())
                    localizer.loc[idx, cols] = 1.0

        vals = localizer.values
        vals[np.isnan(vals)] = 0.0
        vals[vals < 1.0] = 0.0
        localizer.loc[localizer.index, localizer.columns] = vals.copy()
        mat_file = os.path.join(os.path.dirname(self.pst_file), 'loc.mat')
        Matrix.from_dataframe(localizer).to_ascii(mat_file)

        pst.write(self.pst_file, version=2)

    def write_control_settings(self, noptmax=-2, reals=250):
        pst = Pst(self.pst_file)
        pst.pestpp_options["ies_localizer"] = "loc.mat"
        pst.pestpp_options["ies_num_reals"] = reals
        pst.pestpp_options["ies_drop_conflicts"] = 'true'
        pst.control_data.noptmax = noptmax
        oe = ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=reals)
        oe.to_csv(self.pst_file.replace('.pst', '.oe.csv'))
        pst.write(self.pst_file, version=2)
        print(f'writing {self.pst_file} with noptmax={noptmax}, {reals} realizations')

    def initial_parameter_dict(self):
        p = OrderedDict({
            'aw': {'file': self.params_file,
                   'initial_value': None, 'lower_bound': 0.0, 'upper_bound': 1000.0,
                   'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'rew': {'file': self.params_file,
                    'initial_value': 3.0, 'lower_bound': 0.0, 'upper_bound': 6.0,
                    'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'tew': {'file': self.params_file,
                    'initial_value': 18.0, 'lower_bound': 0.0, 'upper_bound': 29.0,
                    'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_alpha': {'file': self.params_file,
                           'initial_value': 0.0, 'lower_bound': -1.5, 'upper_bound': 1.5,
                           'pargp': 'ndvi_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_beta': {'file': self.params_file,
                          'initial_value': 1.25, 'lower_bound': 0.1, 'upper_bound': 4.0,
                          'pargp': 'ndvi_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'mad': {'file': self.params_file,
                    'initial_value': 0.6, 'lower_bound': 0.01, 'upper_bound': 0.99,
                    'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_alpha': {'file': self.params_file,
                          'initial_value': 0.15, 'lower_bound': -0.1, 'upper_bound': 1.,
                          'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_beta': {'file': self.params_file,
                         'initial_value': 1.5, 'lower_bound': 0.1, 'upper_bound': 3.0,
                         'pargp': 'snow_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        })

        return p

    def dry_run(self, exe='pestpp-ies'):
        cmd = ' '.join([exe, os.path.join(self.pest_dir, self.pst_file)])
        wd = self.pest_dir
        try:
            run_ossystem(cmd, wd, verbose=True)
        except Exception:
            run_sp(cmd, wd, verbose=True)


if __name__ == '__main__':

    root_ = os.path.abspath('..')

    project = 'alarc_test'
    # project = '4_Flux_Network'

    project_ws_ = os.path.join(root_, 'tutorials', project)
    if not os.path.isdir(project_ws_):
        root_ = os.path.abspath('')
        project_ws_ = os.path.join(root_, 'tutorials', project)

    config_path_ = os.path.join(project_ws_, 'config.toml')
    py_script = os.path.join(project_ws_, 'custom_forward_run.py')

    builder = PestBuilder(project_ws=project_ws_, config_file=config_path_,
                          use_existing=False, python_script=py_script)
    builder.build_pest()
    builder.build_localizer()
    builder.dry_run('pestpp-ies')
    builder.write_control_settings(noptmax=3, reals=10)

# ========================= EOF ====================================================================
