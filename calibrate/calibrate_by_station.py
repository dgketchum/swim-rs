import os
import shutil
import tempfile
from datetime import datetime

import pandas as pd

from calibrate.pest_builder import PestBuilder
from calibrate.run_pest import run_pst
from prep.prep_plots import prep_fields_json, preproc
from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from prep import get_openet_sites, get_ensemble_parameters


def run_pest_sequence(conf_path, project_ws, workers, realizations, target, members,
                      select_stations=None, pdc_remove=False, overwrite=False):
    """"""
    config = ProjectConfig()
    config.read_config(conf_path, project_ws)

    project = os.path.basename(project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    data_dir = os.path.join(project_ws, 'data')
    properties_json = os.path.join(data_dir, 'properties', 'calibration_properties.json')
    landsat = os.path.join(data_dir, 'landsat')
    dynamics_data = os.path.join(landsat, 'calibration_dynamics.json')
    joined_timeseries = os.path.join(data_dir, 'plot_timeseries')

    print(f'\n{properties_json}')
    print(f'{landsat}')
    print(f'{dynamics_data}')
    print(f'{joined_timeseries}\n')

    flux_meta_csv = os.path.join(data_dir, 'station_metadata.csv')
    flux_meta_df = pd.read_csv(flux_meta_csv, header=1, skip_blank_lines=True, index_col='Site ID')
    sites = sorted(flux_meta_df.index.to_list())

    for fid in sites:

        prepped_data, prepped_input = False, None

        if fid in ['US-Bi2', 'US-Dk1', 'MB_Pch']:
            continue

        if select_stations and fid not in select_stations:
            continue

        for prior_constraint in ['tight']:

            target_dir = os.path.join(project_ws, 'results', prior_constraint, fid)

            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            elif overwrite:
                pass
            else:
                print(f'{fid} {prior_constraint} exists, skipping')
                continue

            os.chdir(os.path.dirname(__file__))

            if not prepped_data:
                prepped_input = os.path.join(data_dir, 'prepped_input.json')

                rs_params_ = get_ensemble_parameters(include=['ssebop'])
                prep_fields_json(properties_json, joined_timeseries, dynamics_data,
                                 prepped_input, target_plots=[fid], rs_params=rs_params_)

                obs_dir = os.path.join(project_ws, 'obs')
                if not os.path.isdir(obs_dir):
                    os.makedirs(obs_dir, exist_ok=True)

                preproc(conf_path, project_ws, etf_target_model=target)

                prepped_data = True

            py_script = os.path.join(project_ws, 'custom_forward_run.py')

            station_prepped_input = os.path.join(target_dir, f'prepped_input_{fid}.json')
            shutil.copyfile(prepped_input, station_prepped_input)

            p_dir = os.path.join(project_ws, f'{prior_constraint}_pest')
            if os.path.isdir(p_dir):
                shutil.rmtree(p_dir)

            m_dir = os.path.join(project_ws, f'{prior_constraint}_master')
            if os.path.isdir(m_dir):
                shutil.rmtree(m_dir)

            w_dir = os.path.join(project_ws, 'workers')
            if os.path.isdir(w_dir):
                shutil.rmtree(w_dir)

            r_dir = os.path.join(project_ws, 'results')
            if not os.path.isdir(r_dir):
                os.mkdir(r_dir)

            station_results = os.path.join(r_dir, prior_constraint, fid)
            if not os.path.exists(station_results):
                os.mkdir(station_results)

            # cwd must be reset to avoid FileNotFound on PstFrom.log
            os.chdir(os.path.dirname(__file__))

            builder = PestBuilder(project_ws=project_ws, config_file=conf_path,
                                  use_existing=False, python_script=py_script,
                                  prior_constraint=prior_constraint, conflicted_obs=None)
            builder.build_pest(target_etf=target, members=members)
            builder.build_localizer()

            # short run sets up base realization and checks for prior-data conflict
            if pdc_remove:
                builder.write_control_settings(noptmax=-1, reals=5)
            else:
                builder.write_control_settings(noptmax=0)

            builder.spinup(overwrite=True)
            # copy spinup to station results location

            spinup_out = os.path.join(station_results, f'spinup_{fid}.json')
            shutil.copyfile(builder.config.spinup, spinup_out)

            exe_ = 'pestpp-ies'

            builder.dry_run(exe_)

            # Check for prior-data conflict, remove and rebuild if necessary
            pdc_file = os.path.join(p_dir, f'{project}.pdc.csv')

            if os.path.exists(pdc_file) and pdc_remove:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_pdc = os.path.join(temp_dir, f'{project}.pdc.csv')
                    shutil.copyfile(pdc_file, temp_pdc)

                    builder = PestBuilder(project_ws=project_ws, config_file=conf_path,
                                          use_existing=False, python_script=py_script,
                                          prior_constraint=prior_constraint, conflicted_obs=temp_pdc)
                    builder.build_pest(target_etf=target, members=members)
                    builder.build_localizer()
                    builder.write_control_settings(noptmax=0)
                    builder.dry_run(exe_)

            builder.write_control_settings(noptmax=3, reals=realizations)

            _pst = f'{project}.pst'

            run_pst(p_dir, exe_, _pst, num_workers=workers, worker_root=w_dir,
                    master_dir=m_dir, verbose=False, cleanup=False)

            fcst_file = os.path.join(m_dir, f'{project}.3.par.csv')
            fcst_out = os.path.join(station_results, f'{fid}.3.par.csv')
            if not os.path.exists(fcst_file):
                fcst_file = os.path.join(m_dir, f'{project}.2.par.csv')
                fcst_out = os.path.join(station_results, f'{fid}.2.par.csv')

            shutil.copyfile(fcst_file, fcst_out)
            print(f'Wrote {fcst_out}')

            phi_csv = os.path.join(m_dir, f'{project}.phi.meas.csv')
            phi_csv_out = os.path.join(station_results, f'{fid}.phi.meas.csv')

            shutil.copyfile(phi_csv, phi_csv_out)
            print(f'Wrote {phi_csv_out}')

            pdc_file = os.path.join(m_dir, f'{project}.pdc.csv')
            pdc_out = os.path.join(station_results, f'{fid}.pdc.csv')

            obs_idx_file = os.path.join(m_dir, f'{project}.idx.csv')
            obs_idx_out = os.path.join(station_results, f'{fid}.idx.csv')

            if os.path.exists(pdc_file):
                shutil.copyfile(pdc_file, pdc_out)
                shutil.copyfile(obs_idx_file, obs_idx_out)
                print(f'Wrote {pdc_out}')
                print(f'Wrote {obs_idx_out}')
                print('')

            shutil.rmtree(p_dir)
            shutil.rmtree(m_dir)
            shutil.rmtree(w_dir)


if __name__ == '__main__':
    pass
# ========================= EOF ============================================================================
