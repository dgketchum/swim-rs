import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix, ObservationEnsemble
from pyemu.utils import PstFrom
from pyemu.utils.os_utils import run_ossystem, run_sp

from swimrs.process.input import build_swim_input


class PestBuilder:
    """Builder for PEST++ IES calibration control files.

    Constructs PEST++ control files, observation files, and parameter templates
    for calibrating SWIM-RS model parameters against ET fraction and SWE observations.

    The builder handles:
    - Parameter setup with prior information from soil and vegetation data
    - Observation file generation from remote sensing ET and SNODAS SWE
    - Localization matrix construction for ensemble methods
    - Forward run script generation

    Attributes:
        config: ProjectConfig instance with calibration settings.
        pest_run_dir: Root directory for PEST++ files.
        pest_dir: Directory containing the .pst control file.
        master_dir: Directory for PEST++ master process.
        pst_file: Path to the generated .pst control file.

    Example:
        >>> from swimrs.swim import ProjectConfig
        >>> from swimrs.calibrate import PestBuilder
        >>>
        >>> config = ProjectConfig()
        >>> config.read_config("project.toml", calibrate=True)
        >>>
        >>> with PestBuilder(config) as builder:
        ...     builder.spinup()
        ...     builder.build_pest(target_etf='ssebop')
        ...     builder.build_localizer()
        ...     builder.write_control_settings(noptmax=4, reals=250)
    """

    def __init__(
        self,
        config,
        container,
        use_existing: bool = False,
        python_script: str | None = None,
        prior_constraint: dict | None = None,
        conflicted_obs: str | None = None,
    ) -> None:
        """Initialize PestBuilder for PEST++ calibration.

        Args:
            config: ProjectConfig instance
            container: SwimContainer instance or path to .swim directory.
                       Required - all data is sourced from the container.
            use_existing: If True, use existing PEST++ setup
            python_script: Path to custom forward run script
            prior_constraint: Prior constraint settings
            conflicted_obs: Path to conflicted observations file
        """
        self.config = config
        self.project_ws = config.project_ws
        self.pest_run_dir = config.pest_run_dir

        # Initialize container (required)
        self._container = None
        self._container_path = None
        self._owns_container = False
        self._init_container(container)

        if not os.path.isdir(self.pest_run_dir):
            os.mkdir(self.pest_run_dir)

        # Extract data from container (replaces SamplePlots)
        self._load_data_from_container()

        self.observation_index = {}

        self.masks = ['inv_irr', 'irr', 'no_mask']

        self.pest = None
        self.etf_std = None
        self.etf_capture_indexes = []

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

    def _init_container(self, container) -> None:
        """Initialize container from instance or path."""
        from swimrs.container import SwimContainer

        if isinstance(container, (str, Path)):
            self._container_path = Path(container)
            self._container = SwimContainer.open(self._container_path, mode='r')
            self._owns_container = True
        else:
            self._container = container
            self._owns_container = False

    def _load_data_from_container(self) -> None:
        """Load all data from container (replaces SamplePlots).

        Populates:
        - self.plot_order: field UIDs
        - self.plot_properties: field properties dict
        - self.irr: irrigation data dict
        - self.ke_max: bare soil evaporation coefficient dict
        - self.kc_max: max crop coefficient dict
        - self.date_range: (start_date, end_date) tuple
        """
        if self._container is None:
            raise ValueError("Container not initialized")

        # Field order
        self.plot_order = self._container.field_uids

        # Get properties and dynamics from container's export infrastructure
        exporter = self._container.export
        self.plot_properties = exporter._get_properties_dict(self.plot_order)

        # Get dynamics (irr_data, gwsub_data, ke_max, kc_max)
        dynamics = exporter._get_dynamics_dict(self.plot_order)
        self.irr = dynamics.get("irr", {})
        self.ke_max = dynamics.get("ke_max", {})
        self.kc_max = dynamics.get("kc_max", {})
        self.gwsub_data = dynamics.get("gwsub", {})

        # Date range from container
        self.date_range = (self._container.start_date, self._container.end_date)

    def _get_etf_data(self, fid: str, model: str = 'ssebop') -> pd.DataFrame:
        """
        Get ETf data for a field from container.

        Returns DataFrame with columns like '{model}_etf_{mask}' for each mask.
        """
        if self._container is None:
            raise ValueError("No container available. Pass container to PestBuilder.__init__")

        result = pd.DataFrame(index=pd.date_range(
            self.config.start_dt, self.config.end_dt, freq='D'
        ))

        for mask in ['irr', 'inv_irr']:
            path = f"remote_sensing/etf/landsat/{model}/{mask}"
            if path in self._container.state.root:
                df = self._container.query.dataframe(path, fields=[fid])
                if fid in df.columns:
                    result[f'{model}_etf_{mask}'] = df[fid]

        return result

    def _get_swe_data(self, fid: str) -> pd.DataFrame:
        """Get SWE data for a field from container."""
        if self._container is None:
            raise ValueError("No container available. Pass container to PestBuilder.__init__")

        path = "snow/snodas/swe"
        if path not in self._container.state.root:
            raise ValueError(f"SWE data not found in container at {path}")

        df = self._container.query.dataframe(path, fields=[fid])
        result = pd.DataFrame(index=df.index)
        result['swe'] = df[fid] if fid in df.columns else np.nan
        return result

    def close(self) -> None:
        """Close container if we own it."""
        if self._owns_container and self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self) -> "PestBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_pest_builder_args(self) -> dict:

        targets = self.plot_order

        # Some projects (international) may not have SSURGO; allow missing AWC
        aw = [self.plot_properties.get(t, {}).get('awc', np.nan) for t in targets]
        ke_max = [self.ke_max.get(t, 1.0) for t in targets]
        kc_max = [self.kc_max.get(t, 1.0) for t in targets]

        input_csv = [os.path.join(self.config.plot_timeseries, '{}.parquet'.format(fid)) for fid in targets]

        et_ins = ['etf_{}.ins'.format(fid) for fid in targets]
        swe_ins = ['swe_{}.ins'.format(fid) for fid in targets]

        init_pars = self.initial_parameter_dict()
        p_list = list(init_pars.keys())
        pars = OrderedDict({'{}_{}'.format(k, fid): v.copy() for k, v in init_pars.items() for fid in targets})

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
                    # Prefer properties-based irrigation fraction when present, otherwise use inferred dynamics.
                    try:
                        irr = np.nanmean([self.plot_properties[fid]['irr'][str(yr)] for yr in range(1987, 2023)])
                    except Exception:
                        irr_data = self.irr.get(fid, {})
                        irr_vals = []
                        for yy, vv in irr_data.items():
                            if yy == 'fallow_years':
                                continue
                            try:
                                irr_vals.append(float(vv.get('f_irr', np.nan)))
                            except Exception:
                                continue
                        irr = float(np.nanmean(irr_vals)) if irr_vals else 0.0
                    if irr > 0.2:
                        params.append((k, 0.02, 'p_{}_0_constant.csv'.format(k)))
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

    def build_pest(self, target_etf: str = 'openet', members: list[str] | None = None) -> None:
        """Build the PEST++ control file and supporting files.

        Creates the .pst control file, observation files, parameter templates,
        and forward run script in the pest directory.

        Uses the process package with portable swim_input.h5 file. Workers are
        fully self-contained and can run without shared storage.

        Args:
            target_etf: ET model to use as calibration target ('ssebop', 'ptjpl', etc.).
            members: Optional list of ensemble member models for uncertainty weighting.
                If provided, observation weights are computed from inter-model spread.

        Raises:
            NotImplementedError: If use_existing=True was set in constructor.
        """
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

        # Build portable input file and generate forward run script
        self._build_swim_input()
        self._write_forward_run_script()

        self._finalize_obs()

        print('Configured PEST++ for {} targets, '.format(len(self.pest_args['targets'])))

    def _write_forward_run_script(self) -> None:
        """Generate custom_forward_run.py with portable relative paths.

        Uses the process package with swim_input.h5 for fully portable workers.
        All paths are relative to the worker directory - no shared storage needed.
        """
        script_path = os.path.join(self.pest_dir, 'custom_forward_run.py')

        script_content = '''"""Auto-generated forward run script for PEST++ calibration.

Uses the swimrs.process package with portable swim_input.h5 file.
All paths are relative - workers are fully self-contained.
"""
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """Forward runner for PEST++ workers."""
    start_time = time.time()

    from swimrs.process.input import SwimInput
    from swimrs.process.loop import run_daily_loop
    from swimrs.process.state import (
        CalibrationParameters,
        load_pest_mult_properties,
    )

    cwd = os.getcwd()

    # All paths relative to worker directory
    h5_path = os.path.join(cwd, "swim_input.h5")
    mult_dir = os.path.join(cwd, "mult")
    pred_dir = os.path.join(cwd, "pred")

    os.makedirs(pred_dir, exist_ok=True)

    # Load portable input data
    swim_input = SwimInput(h5_path=h5_path)

    # Update parameters and properties from PEST++ multiplier files
    params = CalibrationParameters.from_pest_mult(
        mult_dir=mult_dir,
        fids=swim_input.fids,
        base=swim_input.parameters,
    )
    props = load_pest_mult_properties(
        mult_dir=mult_dir,
        fids=swim_input.fids,
        base_props=swim_input.properties,
    )

    # Run the model
    output, _ = run_daily_loop(
        swim_input=swim_input,
        properties=props,
        parameters=params,
    )

    # Write predictions (ETf and SWE)
    for i, fid in enumerate(swim_input.fids):
        etf_path = os.path.join(pred_dir, f"pred_etf_{fid}.np")
        swe_path = os.path.join(pred_dir, f"pred_swe_{fid}.np")
        np.savetxt(etf_path, output.etf[:, i])
        np.savetxt(swe_path, output.swe[:, i])

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    run()
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

    def build_localizer(self) -> None:
        """Build the localization matrix for ensemble Kalman methods.

        Creates a sparse matrix that restricts parameter-observation correlations
        to physically meaningful relationships. ET observations only update
        ET-related parameters, SWE observations only update snow parameters.

        Writes loc.mat and localizer_summary.json to the pest directory.
        """
        et_params = ['aw', 'ndvi_k', 'ndvi_0', 'mad', 'kr_alpha', 'ks_alpha']
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

        # Date range from container
        dt = list(pd.date_range(self._container.start_date, self._container.end_date, freq='D'))
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

    def write_control_settings(self, noptmax: int = -2, reals: int = 250) -> None:
        """Write PEST++ IES control settings to the .pst file.

        Args:
            noptmax: Maximum optimization iterations. Use -2 for parameter
                estimation mode, positive values for optimization.
            reals: Number of realizations in the ensemble.
        """
        pst = Pst(self.pst_file)
        pst.pestpp_options["ies_localizer"] = "loc.mat"
        pst.pestpp_options["ies_num_reals"] = reals
        pst.pestpp_options["ies_drop_conflicts"] = 'true'
        # pst.pestpp_options["ies_reg_factor"] = 0.25
        # pst.pestpp_options["ies_use_approx"] = 'true'

        pst.control_data.noptmax = noptmax
        oe = ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=reals)
        oe.to_csv(self.pst_file.replace('.pst', '.oe.csv'))
        pst.write(self.pst_file, version=2)
        print(f'writing {self.pst_file} with noptmax={noptmax}, {reals} realizations')

    def initial_parameter_dict(self) -> OrderedDict:

        p = OrderedDict({

            # 'aw' and 'zr' are applied by Tracker.load_soils and load_root_depth

            'aw': {'file': self.params_file, 'std': 50.0,
                   'initial_value': None, 'lower_bound': 100.0, 'upper_bound': 400.0,
                   'pargp': 'aw', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            # Stress coefficients - centered in bounds
            'ks_alpha': {'file': self.params_file, 'std': 0.15,
                         'initial_value': 0.5, 'lower_bound': 0.01, 'upper_bound': 1.0,
                         'pargp': 'ks_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'kr_alpha': {'file': self.params_file, 'std': 0.15,
                         'initial_value': 0.5, 'lower_bound': 0.01, 'upper_bound': 1.0,
                         'pargp': 'kr_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            # NDVI-Kcb relationship
            'ndvi_k': {'file': self.params_file, 'std': 0.75,
                       'initial_value': 7.0, 'lower_bound': 4.0, 'upper_bound': 10.0,
                       'pargp': 'ndvi_k', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_0': {'file': self.params_file, 'std': 0.25,
                       'initial_value': 0.4, 'lower_bound': 0.1, 'upper_bound': 0.7,
                       'pargp': 'ndvi_0', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            # Management allowed depletion
            'mad': {'file': self.params_file, 'std': 0.15,
                    'initial_value': None, 'lower_bound': 0.01, 'upper_bound': 0.9,
                    'pargp': 'mad', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            # Snow melt parameters
            'swe_alpha': {'file': self.params_file, 'std': 0.2,
                          'initial_value': 0.3, 'lower_bound': -0.5, 'upper_bound': 1.,
                          'pargp': 'swe_alpha', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'swe_beta': {'file': self.params_file, 'std': 0.3,
                         'initial_value': 1.5, 'lower_bound': 0.5, 'upper_bound': 2.5,
                         'pargp': 'swe_beta', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

        })

        return p

    def dry_run(self, exe: str = 'pestpp-ies') -> None:
        cmd = ' '.join([exe, os.path.join(self.pest_dir, self.pst_file)])
        wd = self.pest_dir
        try:
            run_sp(cmd, wd, verbose=False)
        except Exception:
            run_ossystem(cmd, wd, verbose=False)

    def spinup(self, overwrite: bool = False) -> None:
        """Run model spinup to initialize state variables.

        Runs the model with initial parameters and saves the final state
        to the spinup JSON file for warm-starting calibration runs.

        Args:
            overwrite: If True, regenerate spinup even if file exists.
        """
        from swimrs.process.loop import run_daily_loop

        if not os.path.exists(self.config.spinup) or overwrite:
            print('RUNNING SPINUP')

            if overwrite:
                try:
                    os.remove(self.config.spinup)
                except FileNotFoundError:
                    pass

            # Build temporary HDF5 for spinup (no existing spinup state)
            spinup_h5 = os.path.join(self.pest_dir, 'spinup_temp.h5')
            swim_input = build_swim_input(
                container=self._container,
                output_h5=spinup_h5,
                start_date=self.config.start_dt,
                end_date=self.config.end_dt,
                runoff_process=getattr(self.config, 'runoff_process', 'cn'),
            )

            # Run simulation to generate spinup state
            output, final_state = run_daily_loop(swim_input)
            swim_input.close()

            # Save final state as spinup JSON
            spn_dct = {}
            for i, fid in enumerate(swim_input.fids):
                spn_dct[fid] = {
                    'depl_root': float(final_state.depl_root[i]),
                    'swe': float(final_state.swe[i]),
                    'kr': float(final_state.kr[i]),
                    'ks': float(final_state.ks[i]),
                    'zr': float(final_state.zr[i]),
                }
                # Add optional state if available
                if final_state.depl_ze is not None:
                    spn_dct[fid]['depl_ze'] = float(final_state.depl_ze[i])
                if final_state.s is not None:
                    spn_dct[fid]['s'] = float(final_state.s[i])
                    spn_dct[fid]['s1'] = float(final_state.s1[i])
                    spn_dct[fid]['s2'] = float(final_state.s2[i])
                    spn_dct[fid]['s3'] = float(final_state.s3[i])
                    spn_dct[fid]['s4'] = float(final_state.s4[i])

            with open(self.config.spinup, 'w') as f:
                json.dump(spn_dct, f, indent=2)

            # Clean up temp file
            os.remove(spinup_h5)
            print(f'Spinup saved to {self.config.spinup}')

        else:
            print('SPINUP exists, skipping')

    def _build_swim_input(self, overwrite: bool = False) -> str:
        """Build portable swim_input.h5 file for workers.

        Creates a self-contained HDF5 file with all input data needed
        for model execution. This file is copied to each PEST++ worker,
        enabling fully isolated execution without shared storage.

        Args:
            overwrite: If True, regenerate even if file exists.

        Returns:
            str: Path to the created swim_input.h5 file.
        """
        h5_path = os.path.join(self.pest_dir, 'swim_input.h5')

        if os.path.exists(h5_path) and not overwrite:
            print(f'swim_input.h5 exists at {h5_path}, skipping')
            return h5_path

        print('Building portable swim_input.h5...')

        # Get spinup path if available
        spinup_path = None
        if hasattr(self.config, 'spinup') and self.config.spinup:
            if os.path.exists(self.config.spinup):
                spinup_path = self.config.spinup

        # Build the HDF5 file from container
        build_swim_input(
            container=self._container,
            output_h5=h5_path,
            spinup_json_path=spinup_path,
            start_date=self.config.start_dt,
            end_date=self.config.end_dt,
            runoff_process=getattr(self.config, 'runoff_process', 'cn'),
        )

        print(f'Created swim_input.h5 at {h5_path}')
        return h5_path

    def _write_params(self) -> None:
        _file = None

        for k, v in self.pest_args['pars'].items():
            # pop out unneeded 'std' keyword
            _ = v.pop('std')
            if 'file' in v.keys():
                _file = v.pop('file')
            if v['lower_bound'] <= 0.0:
                transform = 'none'
            else:
                transform = 'log'
            self.pest.add_parameters(_file, 'constant', transform=transform, alt_inst_str='{}_'.format(k), **v)

    def _write_swe_obs(self, count: int) -> None:
        obsnme_str = 'oname:obs_swe_{}_otype:arr_i:{}_j:0'

        for j, fid in enumerate(self.pest_args['targets']):
            # Get SWE data from container
            swe_df = self._get_swe_data(fid)

            self.pest.add_observations(self.pest_args['swe_obs']['file'][j],
                                       insfile=self.pest_args['swe_obs']['insfile'][j])

            swe_df['obs_id'] = [obsnme_str.format(fid, k) for k in range(swe_df.shape[0])]
            valid = [ix for ix, r in swe_df.iterrows() if np.isfinite(r['swe']) and r['swe'] > 0.0]
            valid = swe_df['obs_id'].loc[valid]

            d = self.pest.obs_dfs[j + count].copy()
            d['weight'] = 0.0

            # TODO: adjust as needed for phi visibility of etf vs. swe
            try:
                d.loc[valid, 'weight'] = 1 / 1000.0
            except KeyError:
                valid = [v.lower() for v in valid.values]
                d.loc[valid, 'weight'] = 1 / 1000.0

            d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
            d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0

            d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
            d = d.sort_values(by='idx')
            d.drop(columns=['idx'], inplace=True)

            self.pest.obs_dfs[j + count] = d

    def _write_etf_obs(self, target: str, members: list[str] | None) -> int:
        obsnme_str = 'oname:obs_etf_{}_otype:arr_i:{}_j:0'

        if members is not None:
            self.etf_std = {fid: None for fid in self.pest_args['targets']}

        total_valid_obs = 0
        for i, fid in enumerate(self.pest_args['targets']):
            # Get ETf data from container
            etf = self._get_etf_data(fid, model=target)

            self.pest.add_observations(self.pest_args['etf_obs']['file'][i],
                                       insfile=self.pest_args['etf_obs']['insfile'][i])

            etf['obs_id'] = [obsnme_str.format(fid, j).lower() for j in range(etf.shape[0])]
            etf['obs_id'].to_csv(self.obs_idx_file, mode='a', header=(i == 0), index=False)

            self.observation_index[fid] = pd.DataFrame(data=etf['obs_id'].index, index=etf['obs_id'],
                                                       columns=['obs_idx']).copy()

            captures_for_this_target = []
            for ix, r in etf.iterrows():
                for mask in self.masks:
                    if f'{target}_etf_{mask}' in r and not np.isnan(r[f'{target}_etf_{mask}']):
                        captures_for_this_target.append(etf.loc[ix, 'obs_id'])

            self.etf_capture_indexes.append(captures_for_this_target)

            if members is not None:
                etf_std = pd.DataFrame()
                irr = self.irr.get(fid, {})
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

            total_valid_obs = sum(len(c) for c in self.etf_capture_indexes)

        for i, fid in enumerate(self.pest_args['targets']):

            d = self.pest.obs_dfs[i].copy()
            d.index = d.index.str.lower()
            captures_for_this_df = d.index.intersection(self.etf_capture_indexes[i])
            capture_dates = self.observation_index[fid].loc[captures_for_this_df,  'obs_idx'].to_list()

            d['weight'] = 0.0

            if not captures_for_this_df.empty and total_valid_obs > 0:
                if self.etf_std:
                    d.loc[captures_for_this_df, 'weight'] = 1 / (self.etf_std[fid].loc[capture_dates, 'std'].values + 0.1)
                else:
                    d.loc[captures_for_this_df, 'weight'] = 1 / 0.33

            d.loc[d['obsval'].isna(), 'obsval'] = -99.0
            d.loc[d['weight'].isna(), 'weight'] = 0.0

            d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
            d = d.sort_values(by='idx').drop(columns=['idx'])

            self.pest.obs_dfs[i] = d

            if self.conflicted_obs:
                self._drop_conflicts(i, fid)

        return i

    def _finalize_obs(self) -> None:
        """Write std to observations dataframes.

        We *should* be able to write std to the observations dataframes in the etf
        and swe writers, but they are lost in the pest build call, so are written here.
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
        obs.loc[swe_idx, 'standard_deviation'] = 5.0

        # add time information
        obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]

        pst.write(pst.filename, version=2)

    def add_regularization(self) -> None:
        pst = Pst(self.pst_file)

        for pargp, values in self.initial_parameter_dict().items():

            target_params = pst.parameter_data.loc[pst.parameter_data.pargp == pargp].copy()
            is_log = target_params["partrans"].iloc[0] == "log"
            prior_std = values['std']

            for par_name, row in target_params.iterrows():

                fid = par_name.split(':')[1].replace(f'{row["pname"]}_{row["pargp"]}_', '')
                fid = fid[:-1]
                prior_val = row['parval1']
                rhs = np.log10(prior_val) if is_log else prior_val
                weight = 1.0 / (prior_std ** 2)

                pst.add_pi_equation(
                    par_names=[par_name],
                    pilbl=f"pi_{pargp}_{fid}",
                    rhs=rhs,
                    weight=weight,
                    obs_group=f"pi_{pargp}"
                )

        pst.reg_data.phimlim = sum(len(c) for c in self.etf_capture_indexes)
        pst.reg_data.phimaccept = 1.1 * pst.reg_data.phimlim

        pst.write(self.pst_file, version=2)

    def _drop_conflicts(self, i: int, fid: str) -> None:

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
