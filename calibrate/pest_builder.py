import json
import os
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix, ObservationEnsemble, Cov, geostats
from pyemu.utils import PstFrom
from pyemu.utils.os_utils import run_ossystem, run_sp

from model import obs_field_cycle
from swim.sampleplots import SamplePlots


class PestBuilder:

    def __init__(self, config, use_existing=False, python_script=None, prior_constraint=None,
                 conflicted_obs=None):

        self.config = config
        self.project_ws = config.project_ws
        self.pest_run_dir = config.pest_run_dir

        if not os.path.isdir(self.pest_run_dir):
            os.mkdir(self.pest_run_dir)

        self.plots = SamplePlots()
        self.plots.initialize_plot_data(self.config)
        self.irr = self.plots.input['irr_data']
        self.plot_order = self.plots.input['order']
        self.plot_properties = self.plots.input['props']
        self.plot_time_series = self.plots.input['time_series']

        self.masks = ['inv_irr', 'irr', 'no_mask']

        self.pest = None
        self.etf_std = None

        self.params_file = os.path.join(self.pest_run_dir, 'params.csv')

        self.prior_contstraint = prior_constraint

        self.conflicted_obs = conflicted_obs

        self.pest_dir = os.path.join(config.pest_run_dir, 'pest')
        self.master_dir = os.path.join(config.pest_run_dir, 'master')

        self.workers_dir = os.path.join(config.pest_run_dir, 'workers')
        self.obs_dir = os.path.join(config.pest_run_dir, 'obs')

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

        input_csv = [os.path.join(self.config.plot_timeseries, '{}.parquet'.format(fid)) for fid in targets]

        et_ins = ['etf_{}.ins'.format(fid) for fid in targets]
        swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]

        pars = self.initial_parameter_dict()
        p_list = list(pars.keys())
        pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in pars.items() for fid in targets})

        params = []

        # Prior information from pre-processing
        for i, fid in enumerate(targets):

            for p in p_list:

                k = f'{p}_{fid}'

                if 'aw_' in k:
                    aw_ = aw[i] * 1000.
                    if np.isnan(aw_) or aw_ < pars[k]['lower_bound']:
                        aw_ = 150.0

                    if aw_ > pars[k]['upper_bound']:
                        aw_ = pars[k]['upper_bound'] * 0.8

                    params.append((k, aw_, 'p_{}_0_constant.csv'.format(k)))

                elif 'ke_max_' in k:
                    ke_max_ = ke_max[i]
                    params.append((k, ke_max_, 'p_{}_0_constant.csv'.format(k)))

                elif 'kc_max_' in k:
                    kc_max_ = kc_max[i]
                    params.append((k, kc_max_, 'p_{}_0_constant.csv'.format(k)))

                elif 'mad_' in k:
                    irr = np.nanmean([self.plot_properties[fid]['irr'][str(yr)] for yr in range(1987, 2023)])
                    if irr > 0.2:
                        params.append((k, 0.01, 'p_{}_0_constant.csv'.format(k)))
                    else:
                        params.append((k, 0.6, 'p_{}_0_constant.csv'.format(k)))

                else:
                    params.append((k, pars[k]['initial_value'], 'p_{}_0_constant.csv'.format(k)))

        idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
        vals = np.array([vals, _names]).T
        df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
        df.to_csv(self.params_file)

        for e, (ii, r) in enumerate(df.iterrows()):
            pars[ii]['use_rows'] = e
            if any(prefix in ii for prefix in ['aw_', 'ke_max_', 'kc_max_', 'mad_']):
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

    def build_pest(self, target_etf='openet', members=None):

        if self.overwrite_build is False:
            raise NotImplementedError('Use of exising Pest++ project was specified, '
                                      'running "build_pest" will overwrite it.')

        self.pest = PstFrom(self.pest_run_dir, self.pest_dir, remove_existing=True)

        self._write_params()

        i = self._write_etf_obs(target_etf, members)
        count = i + 1
        self._write_swe_obs(count)

        ofiles = [str(x).replace('obs', 'pred') for x in self.pest.output_filenames]
        self.pest.output_filenames = ofiles

        os.makedirs(os.path.join(self.pest_dir, 'pred'))

        self.pest.py_run_file = 'custom_forward_run.py'
        self.pest.mod_command = 'python custom_forward_run.py'

        self.pest.build_pst(filename=self.pst_file, version=2)

        auto_gen = os.path.join(self.pest_dir, self.pest.py_run_file)
        runner = self.pest_args['python_script']
        shutil.copyfile(runner, auto_gen)

        self._finalize_obs()

        print('Configured PEST++ for {} targets, '.format(len(self.pest_args['targets'])))

    def build_localizer(self):

        et_params = ['aw', 'rew', 'tew', 'ndvi_k', 'ndvi_0', 'mad', 'kr_alpha', 'ks_alpha']
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

            # 'aw' and 'zr' are applied by Tracker.load_soils and load_root_depth

            'aw': {'file': self.params_file,
                   'initial_value': None, 'lower_bound': 100.0, 'upper_bound': 400.0,
                   'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'rew': {'file': self.params_file,
                    'initial_value': 3.0, 'lower_bound': 2.0, 'upper_bound': 6.0,
                    'pargp': 'rew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'tew': {'file': self.params_file,
                    'initial_value': 18.0, 'lower_bound': 6.0, 'upper_bound': 29.0,
                    'pargp': 'tew', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            # kc_max should only be applied for long period targets
            # 'kc_max': {'file': self.params_file,
            #            'initial_value': None, 'lower_bound': 0.8, 'upper_bound': 1.3,
            #            'pargp': 'kc_max', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ks_alpha': {'file': self.params_file,
                         'initial_value': 0.15, 'lower_bound': 0.01, 'upper_bound': 0.3,
                         'pargp': 'ks_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'kr_alpha': {'file': self.params_file,
                         'initial_value': 0.25, 'lower_bound': 0.01, 'upper_bound': 0.45,
                         'pargp': 'kr_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_k': {'file': self.params_file,
                       'initial_value': 7.0, 'lower_bound': 4.0, 'upper_bound': 10.0,
                       'pargp': 'ndvi_k', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_0': {'file': self.params_file,
                       'initial_value': 0.25, 'lower_bound': 0.1, 'upper_bound': 0.7,
                       'pargp': 'ndvi_0', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'mad': {'file': self.params_file,
                    'initial_value': None, 'lower_bound': 0.01, 'upper_bound': 0.9,
                    'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_alpha': {'file': self.params_file,
                          'initial_value': 0.15, 'lower_bound': -0.5, 'upper_bound': 1.,
                          'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_beta': {'file': self.params_file,
                         'initial_value': 1.5, 'lower_bound': 0.5, 'upper_bound': 2.5,
                         'pargp': 'swe_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        })

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

    def _write_swe_obs(self, count):
        obsnme_str = 'oname:obs_swe_{}_otype:arr_i:{}_j:0'

        for j, fid in enumerate(self.pest_args['targets']):

            # only weight swe Nov - Apr
            swe_df = pd.read_parquet(self.pest_args['inputs'][j])
            swe_df = swe_df[[c for c in swe_df.columns if 'swe' in c[2]]]
            swe_df.columns = ['swe']

            self.pest.add_observations(self.pest_args['swe_obs']['file'][j],
                                       insfile=self.pest_args['swe_obs']['insfile'][j])

            swe_df['obs_id'] = [obsnme_str.format(fid, k) for k in range(swe_df.shape[0])]
            valid = [ix for ix, r in swe_df.iterrows() if np.isfinite(r['swe']) and r['swe'] > 0.0]
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

    def _write_etf_obs(self, target, members):
        obsnme_str = 'oname:obs_etf_{}_otype:arr_i:{}_j:0'

        if members is not None:
            self.etf_std = {fid: None for fid in self.pest_args['targets']}

        all_captures = []
        for i, fid in enumerate(self.pest_args['targets']):
            etf = pd.read_parquet(self.pest_args['inputs'][i])
            etf = etf[[c for c in etf.columns if 'etf' in c[2]]]
            etf.columns = [f'{c[-2]}_etf_{c[-1]}' for c in etf.columns]

            self.pest.add_observations(self.pest_args['etf_obs']['file'][i],
                                       insfile=self.pest_args['etf_obs']['insfile'][i])

            etf['obs_id'] = [obsnme_str.format(fid, j).lower() for j in range(etf.shape[0])]
            etf['obs_id'].to_csv(self.obs_idx_file, mode='a', header=(i == 0), index=False)

            captures_for_this_target = []
            for ix, r in etf.iterrows():
                for mask in self.masks:
                    if f'{target}_etf_{mask}' in r and not np.isnan(r[f'{target}_etf_{mask}']):
                        captures_for_this_target.append(etf.loc[ix, 'obs_id'])

            all_captures.append(captures_for_this_target)

            if members is not None:
                etf_std = pd.DataFrame()
                irr = self.plots.input['irr_data'][fid]
                irr_threshold = 0.3
                irr_years = [int(k) for k, v in irr.items() if k != 'fallow_years'
                             and v['f_irr'] >= irr_threshold]
                irr_index = [i for i in etf.index if hasattr(i, 'year') and i.year in irr_years]
                members_and_target = members + [target]

                for member in members_and_target:

                    mask_cols = []

                    for mask in self.masks:

                        col = f'{member}_etf_{mask}'

                        if col in etf.columns:
                            mask_cols.append(col)

                    etf_std[member] = pd.DataFrame(etf[mask_cols].mean(axis=1))

                    if irr_index:
                        etf_std.loc[irr_index, member] = etf.loc[irr_index, f'{member}_etf_irr']

                valid_members = [m for m in members_and_target if m in etf_std.columns]

                multimodel_dt_mean = pd.Series(index=etf_std.index, dtype=float)
                multimodel_dt_std = pd.Series(index=etf_std.index, dtype=float)
                multimodel_dt_count = pd.Series(index=etf_std.index, dtype=int)

                if valid_members:
                    data_subset = etf_std[valid_members]
                    capture_mask = etf_std.replace(np.nan, 0.0).astype(bool)
                    capture_mask.columns = valid_members
                    multimodel_dt_count = capture_mask.sum(axis=1)
                    masked_data = data_subset.where(capture_mask)
                    multimodel_dt_mean = masked_data.mean(axis=1)
                    multimodel_dt_std = masked_data.std(axis=1)

                etf_std['std'] = multimodel_dt_std
                etf_std['ct'] = multimodel_dt_count
                etf_std['mean'] = multimodel_dt_mean
                self.etf_std[fid] = etf_std.copy()

            total_valid_obs = sum(len(c) for c in all_captures)

        for i, fid in enumerate(self.pest_args['targets']):
            d = self.pest.obs_dfs[i].copy()
            d.index = d.index.str.lower()
            captures_for_this_df = d.index.intersection(all_captures[i])

            d['weight'] = 0.0

            if not captures_for_this_df.empty and total_valid_obs > 0:
                d.loc[captures_for_this_df, 'weight'] = d['obsval']

            d.loc[d['obsval'].isna(), 'obsval'] = -99.0
            d.loc[d['weight'].isna(), 'weight'] = 0.0

            d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
            d = d.sort_values(by='idx').drop(columns=['idx'])

            self.pest.obs_dfs[i] = d

            if self.conflicted_obs:
                self._drop_conflicts(i, fid)

        return i

    def _finalize_obs(self):
        """
        We *should* be able to write std to the observations dataframes, in the etf and swe writers, but they are
        lost in the pest build call, so are written here.
        """
        pst = Pst(self.pst_file)
        obs = pst.observation_data

        obs['standard_deviation'] = 0.00
        etf_idx = [i for i in obs.index if 'etf' in i]

        if self.etf_std is not None:

            etf_std_vals = []

            [etf_std_vals.extend(self.etf_std[k]['std'].values) for k in self.pest_args['targets']]

            obs.loc[etf_idx, 'standard_deviation'] = np.array(etf_std_vals)

        else:
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

    def _get_spatial_data_df(self, pst):

        par_data = pst.parameter_data

        spatial_par_groups = ['aw', 'rew', 'tew', 'ks_alpha', 'kr_alpha', 'ndvi_k', 'ndvi_0']

        records = []
        for par_name in par_data.parnme:
            pargp = par_data.loc[par_name, "pargp"]
            if pargp in spatial_par_groups:
                site_id = '_'.join(par_name.split('_')[1:])

                if site_id in self.plot_properties:
                    x = self.plot_properties[site_id]['x_coord']
                    y = self.plot_properties[site_id]['y_coord']
                    records.append({"parnme": par_name, "x_coord": x, "y_coord": y, "pargp": pargp})

        if not records:
            raise ValueError("Could not create spatial data. Check site_ids and plot_properties.")

        return pd.DataFrame(records)

    def add_regularization(self, aws_prior_std=30.0, spatial_reg_groups=None):
        """
        Adds prior information equations to the Pst object for regularization.
        """
        pst = Pst(self.pst_file)

        # 1. Add Tikhonov Regularization for AWS
        # ----------------------------------------
        print("Adding Tikhonov regularization for AWS parameters...")
        aws_pars = pst.parameter_data.loc[pst.parameter_data.pargp == "aw", "parnme"]

        # Get the prior values you stored during setup
        # You might need to adjust this logic to match your data structures
        prior_info = []
        for par_name in aws_pars:
            fid = '_'.join(par_name.split('_')[1:])  # Assumes format 'aw_FIELD_ID'
            try:
                prior_val = self.plot_properties[fid]['awc'] * 1000.0  # convert m to mm
                if np.isnan(prior_val):
                    continue
                # Create a prior information equation for this parameter
                # The weight is 1 / (std_dev**2)
                pi = (par_name, prior_val, 1.0 / (aws_prior_std ** 2))
                prior_info.append(pi)
            except KeyError:
                print(f"Warning: Could not find prior AWS for field {fid}. Skipping regularization.")

        if prior_info:
            df = pd.DataFrame(prior_info, columns=["parnme", "prior_val", "weight"])
            # Use pyEMU's helper to add these as PI equations
            pst.add_prior_information_from_df(df, obs_group_name="pi_aws")
            print(f"Added {len(prior_info)} prior information equations for AWS.")

        # 2. Add Geostatistical Regularization (see next section)
        # --------------------------------------------------------
        if spatial_reg_groups:
            self._add_geostatistical_regularization(pst, spatial_reg_groups)

        pst.write(self.pst_file, version=2)

    def apply_geostatistical_regularization(self, correlation_range=3000.0, cov_filename="prior.cov"):

        print("Applying geostatistical regularization...")

        pst = Pst(self.pst_file)

        try:
            spatial_df = self._get_spatial_data_df(pst)
        except Exception as e:
            print(f"Could not generate spatial dataframe for regularization: {e}")
            print("Skipping geostatistical regularization.")
            return

        v = geostats.ExpVario(contribution=1.0, a=correlation_range)
        gs = geostats.GeoStruct(nugget=0.0, variograms=[v])

        cov = Cov(x=pst.par_names, isdiagonal=True)  # Start with a diagonal matrix

        for pargp in spatial_df.pargp.unique():
            print(f"  Building covariance for parameter group: {pargp}")

            df_grp = spatial_df[spatial_df.pargp == pargp]

            par_info = pst.parameter_data.loc[df_grp.parnme.iloc]
            variance = ((par_info.parubnd - par_info.parlbnd) / 4.0) ** 2

            gs.variograms.contribution = variance
            cov_grp = gs.covariance_matrix(x=df_grp.x_coord, y=df_grp.y_coord, names=df_grp.parnme)

            cov.add(cov_grp)

        cov_filepath = os.path.join(self.pest_dir, cov_filename)
        cov.to_binary(cov_filepath)
        print(f"Prior covariance matrix saved to {cov_filepath}")

        pst.pestpp_options["ies_prior_cov"] = cov_filename

        pst.write(self.pst_file, version=2)
        print("PEST++ configured to use prior covariance matrix for IES.")


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
