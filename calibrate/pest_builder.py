import os
import json
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix, ObservationEnsemble
from pyemu.utils import PstFrom
from pyemu.utils.os_utils import run_ossystem, run_sp

from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from model import obs_field_cycle

from calibrate.run_pest import run_pst


class PestBuilder:

    def __init__(self, config_file, project_ws, use_existing=False, python_script=None, prior_constraint=None,
                 conflicted_obs=None):

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

        self.prior_contstraint = prior_constraint

        self.conflicted_obs = conflicted_obs

        if prior_constraint:
            self.pest_dir = os.path.join(project_ws, f'{prior_constraint}_pest')
            self.master_dir = os.path.join(project_ws, f'{prior_constraint}_master')
        else:
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
        ke_max = [self.plots.input['ke_max'][t] for t in targets]
        kc_max = [self.plots.input['kc_max'][t] for t in targets]

        input_csv = [os.path.join(self.config.plot_timeseries, '{}_daily.csv'.format(fid)) for fid in targets]

        et_ins = ['etf_{}.ins'.format(fid) for fid in targets]
        swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]
        self.params_file = os.path.join(self.project_ws, 'params.csv')

        pars = self.initial_parameter_dict()
        p_list = list(pars.keys())
        pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets})

        params = []

        for i, fid in enumerate(targets):

            for p in p_list:

                k = f'{p}_{fid}'

                if 'aw_' in k:
                    aw_ = aw[i] * 1000.
                    if np.isnan(aw_) or aw_ < pars[k]['lower_bound']:
                        aw_ = 150.0
                    params.append((k, aw_, 'p_{}_0_constant.csv'.format(k)))

                elif 'ke_max_' in k:
                    ke_max_ = ke_max[i]
                    params.append((k, ke_max_, 'p_{}_0_constant.csv'.format(k)))

                elif 'kc_max_' in k:
                    kc_max_ = kc_max[i]
                    params.append((k, kc_max_, 'p_{}_0_constant.csv'.format(k)))

                else:
                    params.append((k, pars[k]['initial_value'], 'p_{}_0_constant.csv'.format(k)))

        idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
        vals = np.array([vals, _names]).T
        df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
        df.to_csv(self.params_file)

        for e, (ii, r) in enumerate(df.iterrows()):
            pars[ii]['use_rows'] = e
            if any(prefix in ii for prefix in ['aw_', 'ke_max_', 'kc_max_']):
                val = float(r['value'])
                pars[ii]['initial_value'] = val

                if any(prefix in ii for prefix in ['ke_max_', 'kc_max_']):
                    if val < pars[ii]['lower_bound']:
                        pars[ii]['lower_bound'] = val - 0.2
                        pars[ii]['initial_value'] = val - 0.1
                        pars[ii]['upper_bound'] = val

                    if val > pars[ii]['upper_bound']:
                        pars[ii]['lower_bound'] = val - 0.3
                        pars[ii]['initial_value'] = val - 0.1
                        pars[ii]['upper_bound'] = val

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

        self._write_params()

        i = self._write_etf_obs()
        count = i + 1
        self._write_swe_obs(count, i)

        ofiles = [str(x).replace('obs', 'pred') for x in self.pest.output_filenames]
        self.pest.output_filenames = ofiles

        os.makedirs(os.path.join(self.pest_dir, 'pred'))

        self.pest.py_run_file = 'custom_forward_run.py'
        self.pest.mod_command = 'python custom_forward_run.py'

        self.pest.build_pst(version=2)

        auto_gen = os.path.join(self.pest_dir, self.pest.py_run_file)
        runner = self.pest_args['python_script']
        shutil.copyfile(runner, auto_gen)

        self._finalize_obs()

        print('Configured PEST++ for {} targets, '.format(len(self.pest_args['targets'])))

    def build_localizer(self):

        et_params = ['aw', 'rew', 'tew', 'ndvi_k', 'ndvi_0', 'mad']
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
        # most brittle lines of code ever written
        sites = list(set(['_'.join(i.split('_')[2:-3]) for i in df.index]))

        track = {k: [] for k in sites}

        dt = [pd.to_datetime(k) for k, v in self.plot_time_series.items()]
        years = list(range(self.config.start_dt.year, self.config.end_dt.year + 1))

        for s in sites:

            for ob_type, params in par_relation.items():

                if ob_type == 'etf':

                    for yr in years:
                        t_idx = ['_i:{}_'.format(int(i)) for i, r in enumerate(dt) if r.year == yr]

                        idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
                        idx = [i for i in idx if '_{}_'.format(i.split('_')[-2]) in t_idx]
                        cols = list(
                            np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c]
                                      for p in et_params]).flatten())
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

        col_sums = {col: int(localizer[col].sum()) for col in localizer.columns}
        summary = {
            "shape": localizer.shape,
            "non_zero_count": int(np.count_nonzero(localizer.values)),
            "sites": sites,
            "tracked_irrigation_years": track,
            "parameter_groups": list(pdict.keys()),
            "column_sums": col_sums,
        }
        summary_file = os.path.join(os.path.dirname(self.pst_file), 'localizer_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

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

            # 'aw' and zr are applied by Tracker.load_soils and load_root_depth

            'aw': {'file': self.params_file,
                   'initial_value': None, 'lower_bound': 100.0, 'upper_bound': 400.0,
                   'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'rew': {'file': self.params_file,
                    'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                    'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'tew': {'file': self.params_file,
                    'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                    'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'kc_max': {'file': self.params_file,
                       'initial_value': None, 'lower_bound': 0.8, 'upper_bound': 1.3,
                       'pargp': 'kc_max', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ks_alpha': {'file': self.params_file,
                         'initial_value': 0.1, 'lower_bound': 0.01, 'upper_bound': 1.0,
                         'pargp': 'ks_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'kr_alpha': {'file': self.params_file,
                         'initial_value': 0.15, 'lower_bound': 0.01, 'upper_bound': 1.0,
                         'pargp': 'kr_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_k': {'file': self.params_file,
                       'initial_value': 6.0, 'lower_bound': 1, 'upper_bound': 10,
                       'pargp': 'ndvi_k', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_0': {'file': self.params_file,
                       'initial_value': 0.25, 'lower_bound': 0.1, 'upper_bound': 0.7,
                       'pargp': 'ndvi_0', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'mad': {'file': self.params_file,
                    'initial_value': 0.6, 'lower_bound': 0.1, 'upper_bound': 0.9,
                    'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_alpha': {'file': self.params_file,
                          'initial_value': 0.15, 'lower_bound': -0.5, 'upper_bound': 1.,
                          'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_beta': {'file': self.params_file,
                         'initial_value': 1.5, 'lower_bound': 0.5, 'upper_bound': 2.5,
                         'pargp': 'swe_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        })

        if self.prior_contstraint == 'loose':
            loose_params = {
                'aw': {'lower_bound': 0.0, 'upper_bound': 1000.0},
                'ndvi_alpha': {'lower_bound': -1.5, 'upper_bound': 1.5},
                'ndvi_beta': {'lower_bound': 0.1, 'upper_bound': 4.0},
                'mad': {'lower_bound': 0.01, 'upper_bound': 0.99},
                'swe_alpha': {'lower_bound': -0.1, 'upper_bound': 1.},
                'swe_beta': {'lower_bound': 0.1, 'upper_bound': 3.0},
            }
            for key, updates in loose_params.items():
                if key in p:
                    p[key].update(updates)

        return p

    def dry_run(self, exe='pestpp-ies'):
        cmd = ' '.join([exe, os.path.join(self.pest_dir, self.pst_file)])
        wd = self.pest_dir
        try:
            run_sp(cmd, wd, verbose=False)
        except Exception:
            run_ossystem(cmd, wd, verbose=False)

    def spinup(self, overwrite=False):

        if not os.path.exists(self.config.spinup) or overwrite:
            print('RUNNING SPINUP')

            if overwrite:
                try:
                    os.remove(self.config.spinup)
                except FileNotFoundError:
                    pass

            output = obs_field_cycle.field_day_loop(self.config, self.plots, debug_flag=True)
            spn_dct = {k: v.iloc[-1].to_dict() for k, v in output.items()}
            with open(self.config.spinup, 'w') as f:
                json.dump(spn_dct, f)

        else:
            print('SPINUP exists, skipping')

    def _write_params(self):
        _file = None

        for k, v in self.pest_args['pars'].items():
            if 'file' in v.keys():
                _file = v.pop('file')
            if v['lower_bound'] <= 0.0:
                transform = 'none'
            else:
                transform = 'log'
            self.pest.add_parameters(_file, 'constant', transform=transform, alt_inst_str='{}_'.format(k), **v)

    def _write_swe_obs(self, count, i):
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
            valid = [ix for ix, r in swe_df.iterrows() if np.isfinite(r['obs_swe']) and r['obs_swe'] > 0.0]
            valid = swe_df['obs_id'].loc[valid]

            d = self.pest.obs_dfs[j + count].copy()
            d['weight'] = 0.0

            # TODO: adjust as needed for phi visibility of etf vs. swe
            try:
                d.loc[valid, 'weight'] = 0.005
            except KeyError:
                valid = [v.lower() for v in valid.values]
                d.loc[valid, 'weight'] = 0.005

            d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
            d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0

            d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
            d = d.sort_values(by='idx')
            d.drop(columns=['idx'], inplace=True)

            self.pest.obs_dfs[j + count] = d

    def _write_etf_obs(self):
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

            if self.conflicted_obs:
                self._drop_conflicts(i, fid)

        return i

    def _finalize_obs(self):
        pst = Pst(self.pst_file)
        obs = pst.observation_data

        obs['standard_deviation'] = 0.00
        etf_idx = [i for i in obs.index if 'etf' in i]
        obs.loc[etf_idx, 'standard_deviation'] = obs['obsval'] * 0.33

        swe_idx = [i for i, r in obs.iterrows() if 'swe' in i and r['obsval'] > 0.0]
        obs.loc[swe_idx, 'standard_deviation'] = obs['obsval'] * 0.33

        # add time information
        obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]

        pst.write(pst.filename, version=2)

    def _drop_conflicts(self, i, fid):

        pdc = pd.read_csv(self.conflicted_obs, index_col=0)

        d = self.pest.obs_dfs[i].copy()
        start_weight = d['weight'].sum()
        idx = [i for i in pdc.index if 'etf' in i and fid.lower() in i]
        d.loc[idx, 'weight'] = 0.0
        end_weight = d['weight'].sum()
        removed = start_weight - end_weight
        self.pest.obs_dfs[i] = d
        print(f'Removed {int(removed)} conflicted obs from etf, leaving {int(end_weight)}')

        self.pest.build_pst(version=2)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
