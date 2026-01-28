import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress pyemu's flopy warning - flopy is optional and not needed for SWIM-RS
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to import legacy module")
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

        If model='ensemble', computes the mean across all available ETf models.
        """
        if self._container is None:
            raise ValueError("No container available. Pass container to PestBuilder.__init__")

        result = pd.DataFrame(index=pd.date_range(
            self.config.start_dt, self.config.end_dt, freq='D'
        ))

        if model == 'ensemble':
            # Find all available ETf models in the container
            available_models = self._discover_etf_models()
            if not available_models:
                return result

            for mask in ['irr', 'inv_irr']:
                mask_data = []
                for m in available_models:
                    path = f"remote_sensing/etf/landsat/{m}/{mask}"
                    if path in self._container.state.root:
                        df = self._container.query.dataframe(path, fields=[fid])
                        if fid in df.columns:
                            mask_data.append(df[fid])

                if mask_data:
                    # Compute mean across all models
                    combined = pd.concat(mask_data, axis=1)
                    result[f'ensemble_etf_{mask}'] = combined.mean(axis=1)
        else:
            for mask in ['irr', 'inv_irr']:
                path = f"remote_sensing/etf/landsat/{model}/{mask}"
                if path in self._container.state.root:
                    df = self._container.query.dataframe(path, fields=[fid])
                    if fid in df.columns:
                        result[f'{model}_etf_{mask}'] = df[fid]

        return result

    def _discover_etf_models(self) -> list[str]:
        """Discover available ETf models in the container."""
        known_models = ['ssebop', 'ptjpl', 'sims', 'geesebal', 'eemetric', 'disalexi']
        available = []

        for model in known_models:
            # Check if at least one mask exists for this model
            for mask in ['irr', 'inv_irr']:
                path = f"remote_sensing/etf/landsat/{model}/{mask}"
                if path in self._container.state.root:
                    available.append(model)
                    break

        return available

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

                elif 'ndvi_0_' in k:
                    # Informed prior based on irrigation status (see PARAMETER_SEARCH.md):
                    # With fc in kc_act equation (kc_act = fc*ks*kcb + ke):
                    #   - Grassland/non-irrigated: ndvi_0 ~ 0.20 (transpiration begins at low NDVI)
                    #   - Irrigated crops: ndvi_0 ~ 0.55 (optimal region 0.50-0.60)
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
                        params.append((k, 0.55, 'p_{}_0_constant.csv'.format(k)))
                    else:
                        params.append((k, 0.20, 'p_{}_0_constant.csv'.format(k)))

                else:
                    params.append((k, pars[k]['initial_value'], 'p_{}_0_constant.csv'.format(k)))

        idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
        vals = np.array([vals, _names]).T
        df = pd.DataFrame(index=idx, data=vals, columns=['value', 'mult_name'])
        df.to_csv(self.params_file)

        for e, (ii, r) in enumerate(df.iterrows()):
            pars[ii]['use_rows'] = e
            if any(prefix in ii for prefix in ['aw_', 'ke_max_', 'kc_max_', 'mad_', 'ndvi_0_']):
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

        # Create minimal template directory for PstFrom
        # (Avoids copying workers/master/pest dirs which causes recursive copying)
        import shutil
        template_dir = os.path.join(self.pest_run_dir, '_template')
        if os.path.exists(template_dir):
            shutil.rmtree(template_dir)
        os.makedirs(template_dir)

        # Copy only the files PstFrom needs (params.csv, obs files)
        # Update self.params_file to point to template location for PstFrom
        if os.path.exists(self.params_file):
            shutil.copy2(self.params_file, template_dir)
            self.params_file = os.path.join(template_dir, 'params.csv')
            # Update parameter dicts to use the new path
            for k, v in self.pest_args['pars'].items():
                if 'file' in v:
                    v['file'] = self.params_file

        if os.path.exists(self.obs_dir):
            shutil.copytree(self.obs_dir, os.path.join(template_dir, 'obs'))

        self.pest = PstFrom(template_dir, self.pest_dir, remove_existing=True)

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
        self.print_build_diagnostics()

        # Clean up template directory
        if os.path.exists(template_dir):
            shutil.rmtree(template_dir)

        print('Configured PEST++ for {} targets, '.format(len(self.pest_args['targets'])))

    def print_build_diagnostics(self, max_groups: int = 25) -> pd.DataFrame:
        """Print a compact diagnostics table after building the PEST++ project.

        This is meant to make it obvious whether calibration is actually using
        the intended observations/weights (e.g., ETf weights not all zero).

        Returns
        -------
        pd.DataFrame
            Per-observation-group summary table (also printed).
        """
        try:
            pst = Pst(self.pst_file)
            obs = pst.observation_data.copy()
        except Exception as e:
            print(f"[PEST++ diagnostics] Failed to load pst/observation data: {e}")
            return pd.DataFrame()

        if obs is None or obs.empty:
            print("[PEST++ diagnostics] No observations found in pst.")
            return pd.DataFrame()

        w = obs["weight"].fillna(0.0).astype(float) if "weight" in obs.columns else pd.Series(0.0, index=obs.index)
        y = obs["obsval"].astype(float) if "obsval" in obs.columns else pd.Series(np.nan, index=obs.index)
        valid_obs = np.isfinite(y.values) & (y.values != -99.0)
        nonzero_w = w.values > 0.0

        # Overall header
        print("\n=== PEST++ Build Diagnostics ===")
        print(f"pst: {self.pst_file}")
        print(
            "observations: "
            f"total={len(obs)}, valid={int(valid_obs.sum())}, "
            f"nonzero_weight={int(nonzero_w.sum())}"
        )

        # Type-specific quick checks
        etf_mask = obs.index.to_series().str.contains("etf", case=False, regex=False)
        swe_mask = obs.index.to_series().str.contains("swe", case=False, regex=False)
        if etf_mask.any():
            etf_nonzero = int((nonzero_w & etf_mask.values).sum())
            etf_valid = int((valid_obs & etf_mask.values).sum())
            print(f"ETf: valid={etf_valid}, nonzero_weight={etf_nonzero}")
        if swe_mask.any():
            swe_nonzero = int((nonzero_w & swe_mask.values).sum())
            swe_valid = int((valid_obs & swe_mask.values).sum())
            print(f"SWE: valid={swe_valid}, nonzero_weight={swe_nonzero}")

        table = self._build_obs_diagnostics_table(obs)
        # Limit printed rows for readability
        show = table.head(max_groups).copy()
        if len(table) > max_groups:
            more = len(table) - max_groups
            print(f"\nTop {max_groups} observation groups (of {len(table)}). ({more} more not shown)")
        else:
            print(f"\nObservation groups: {len(table)}")

        # Make output stable/compact
        pd.set_option("display.max_colwidth", 90)
        print(show.to_string(index=False))

        # Parameter quick summary (helps confirm tuned params exist)
        try:
            par = pst.parameter_data.copy()
            if par is not None and not par.empty:
                at_lower = (par["parval1"].astype(float) <= par["parlbnd"].astype(float) + 1e-12).sum()
                at_upper = (par["parval1"].astype(float) >= par["parubnd"].astype(float) - 1e-12).sum()
                print(
                    "\nparameters: "
                    f"n={len(par)}, at_lower={int(at_lower)}, at_upper={int(at_upper)}, "
                    f"groups={par['pargp'].nunique() if 'pargp' in par.columns else 'n/a'}"
                )
        except Exception:
            pass

        return table

    @staticmethod
    def _build_obs_diagnostics_table(obs: pd.DataFrame) -> pd.DataFrame:
        """Build per-observation-group diagnostics for a PEST++ observation table."""
        if obs is None or obs.empty:
            return pd.DataFrame()

        grp = obs["obgnme"] if "obgnme" in obs.columns else pd.Series("obs", index=obs.index)

        rows: list[dict] = []
        for gname, gdf in obs.groupby(grp, dropna=False):
            gw = (
                gdf["weight"].fillna(0.0).astype(float)
                if "weight" in gdf.columns
                else pd.Series(0.0, index=gdf.index)
            )
            gy = gdf["obsval"].astype(float) if "obsval" in gdf.columns else pd.Series(np.nan, index=gdf.index)
            gsd = (
                gdf["standard_deviation"].astype(float)
                if "standard_deviation" in gdf.columns
                else pd.Series(np.nan, index=gdf.index)
            )
            gvalid = np.isfinite(gy.values) & (gy.values != -99.0)
            gnonzero = gw.values > 0.0
            rows.append(
                {
                    "group": str(gname),
                    "n": int(len(gdf)),
                    "valid": int(gvalid.sum()),
                    "w>0": int(gnonzero.sum()),
                    "w_sum": float(gw.sum()),
                    "w_max": float(gw.max()) if len(gw) else 0.0,
                    "obs_min": float(np.nanmin(gy.values[gvalid])) if gvalid.any() else np.nan,
                    "obs_max": float(np.nanmax(gy.values[gvalid])) if gvalid.any() else np.nan,
                    "sd_nan%": float(np.mean(~np.isfinite(gsd.values)) * 100.0) if len(gsd) else 0.0,
                }
            )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(by=["w_sum", "valid", "n"], ascending=False)

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
    from swimrs.process.loop_fast import run_daily_loop_fast
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

    # Run the model (uses fast JIT-compiled loop)
    output, _ = run_daily_loop_fast(
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

            # NDVI-Kcb relationship: sigmoid function
            # kcb = kc_max / (1 + exp(-k * (NDVI - ndvi_0)))
            # kc_act = fc * ks * kcb + ke (FAO-56 dual crop coefficient)
            'ndvi_k': {'file': self.params_file, 'std': 1.0,
                       'initial_value': 10.0, 'lower_bound': 3.0, 'upper_bound': 20.0,
                       'pargp': 'ndvi_k', 'index_cols': 0, 'use_cols': 1, 'use_rows': None},

            'ndvi_0': {'file': self.params_file, 'std': 0.15,
                       'initial_value': 0.55, 'lower_bound': 0.1, 'upper_bound': 0.85,
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

        This method also creates the initial swim_input.h5 file (without spinup
        state). After spinup completes, _build_swim_input() rebuilds the h5
        with the spinup state baked in.

        Args:
            overwrite: If True, regenerate spinup even if file exists.
        """
        from swimrs.process.loop_fast import run_daily_loop_fast

        if not os.path.exists(self.config.spinup) or overwrite:
            print('RUNNING SPINUP')

            if overwrite:
                try:
                    os.remove(self.config.spinup)
                except FileNotFoundError:
                    pass

            # Build swim_input.h5 for spinup (no existing spinup state)
            # This file will be rebuilt with spinup state by _build_swim_input()
            h5_path = os.path.join(self.pest_dir, 'swim_input.h5')
            swim_input = build_swim_input(
                container=self._container,
                output_h5=h5_path,
                start_date=self.config.start_dt,
                end_date=self.config.end_dt,
                runoff_process=getattr(self.config, 'runoff_process', 'cn'),
                refet_type=getattr(self.config, "refet_type", "eto") or "eto",
            )

            # Run simulation to generate spinup state (uses fast JIT loop)
            output, final_state = run_daily_loop_fast(swim_input)
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

            print(f'Spinup saved to {self.config.spinup}')

        else:
            print('SPINUP exists, skipping')

    def _build_swim_input(self) -> str:
        """Build portable swim_input.h5 file for workers with spinup state.

        Creates a self-contained HDF5 file with all input data needed
        for model execution, including spinup state if available. This file
        is copied to each PEST++ worker for isolated execution.

        If spinup() was called first, this rebuilds the h5 with spinup state
        baked in. The rebuild is necessary because spinup creates the h5
        without spinup state (since it's generating it).

        Returns:
            str: Path to the created swim_input.h5 file.
        """
        h5_path = os.path.join(self.pest_dir, 'swim_input.h5')

        # Get spinup path if available
        spinup_path = None
        if hasattr(self.config, 'spinup') and self.config.spinup:
            if os.path.exists(self.config.spinup):
                spinup_path = self.config.spinup

        # If h5 exists but no spinup, keep as-is (no spinup available)
        # If h5 exists and spinup exists, rebuild to bake in spinup state
        if os.path.exists(h5_path) and spinup_path is None:
            print(f'swim_input.h5 exists at {h5_path} (no spinup), skipping')
            return h5_path

        print('Building portable swim_input.h5 with spinup state...')

        # Build the HDF5 file from container
        build_swim_input(
            container=self._container,
            output_h5=h5_path,
            spinup_json_path=spinup_path,
            start_date=self.config.start_dt,
            end_date=self.config.end_dt,
            runoff_process=getattr(self.config, 'runoff_process', 'cn'),
            refet_type=getattr(self.config, "refet_type", "eto") or "eto",
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
                    # Get this member's ETf data directly
                    member_etf = self._get_etf_data(fid, model=member)

                    mask_cols = []
                    for mask in self.masks:
                        col = f'{member}_etf_{mask}'
                        if col in member_etf.columns:
                            mask_cols.append(col)

                    if mask_cols:
                        etf_std[member] = member_etf[mask_cols].mean(axis=1)
                        if irr_index:
                            irr_col = f'{member}_etf_irr'
                            if irr_col in member_etf.columns:
                                etf_std.loc[irr_index, member] = member_etf.loc[irr_index, irr_col]
                    else:
                        # No data for this member, fill with NaN
                        etf_std[member] = pd.Series(np.nan, index=etf.index)

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
                if self.etf_std is not None and self.etf_std.get(fid) is not None:
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

        # Diagnostic warning: detect if all ETf observations got zero weights
        total_nonzero_etf = sum(
            (df['weight'] > 0).sum()
            for df in self.pest.obs_dfs[:len(self.pest_args['targets'])]
        )
        if total_valid_obs > 0 and total_nonzero_etf == 0:
            warnings.warn(
                f"All {total_valid_obs} ETf observations have zero weight. "
                "Check etf_std index alignment with capture_dates.",
                UserWarning,
                stacklevel=2
            )

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
