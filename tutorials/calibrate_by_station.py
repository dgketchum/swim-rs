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
from prep import get_flux_sites, get_ensemble_parameters
from calibrate.run_mp import optimize_fields


def run_pest_sequence(conf, results, select_stations=None, pdc_remove=False, overwrite=False):
    """"""

    for i, fid in enumerate(select_stations, start=1):

        print(f'{fid}: {i} of {len(select_stations)} stations')

        # rebuild pest calibration directory from scratch
        if os.path.isdir(conf.pest_run_dir):
            shutil.rmtree(conf.pest_run_dir)

        os.makedirs(conf.pest_run_dir, exist_ok=False)

        target_dir = os.path.join(results_dir, fid)
        station_prepped_input = os.path.join(target_dir, f'prepped_input_{fid}.json')

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        if not os.path.isfile(station_prepped_input) or overwrite:

            models = [conf.etf_target_model]
            if conf.etf_ensemble_members is not None:
                models += conf.etf_ensemble_members

            rs_params_ = get_ensemble_parameters(include=models)
            prep_fields_json(config.properties_json, conf.plot_timeseries, conf.dynamics_data_json,
                             conf.input_data, target_plots=[fid], rs_params=rs_params_,
                             interp_params=('ndvi',))
            preproc(conf)

        # move station-specific input data to the results dir and to the pest project
        shutil.copyfile(conf.input_data, station_prepped_input)
        shutil.copyfile(conf.input_data, os.path.join(conf.pest_run_dir, os.path.basename(conf.input_data)))

        p_dir = os.path.join(conf.pest_run_dir, 'pest')
        m_dir = os.path.join(conf.pest_run_dir, 'master')
        w_dir = os.path.join(conf.pest_run_dir, 'workers')

        station_results = os.path.join(results, fid)
        if not os.path.exists(station_results):
            os.mkdir(station_results)

        # cwd must be reset to avoid FileNotFound on PstFrom.log
        os.chdir(os.path.dirname(__file__))

        builder = PestBuilder(conf, use_existing=False, python_script=conf.source_python_script,
                              conflicted_obs=None)
        builder.build_pest(target_etf=conf.etf_target_model, members=conf.etf_ensemble_members)
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

                builder = PestBuilder(conf, use_existing=False, python_script=conf.source_python_script,
                                      conflicted_obs=temp_pdc)
                builder.build_pest(target_etf=conf.etf_target_model, members=conf.etf_ensemble_members)
                builder.build_localizer()
                builder.write_control_settings(noptmax=0)
                builder.dry_run(exe_)

        builder.write_control_settings(noptmax=3, reals=conf.realizations)

        _pst = f'{project}.pst'

        run_pst(p_dir, exe_, _pst, num_workers=conf.workers, worker_root=w_dir,
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
    """"""
    # project = '5_Flux_Ensemble'
    project = '4_Flux_Network'

    if project == '5_Flux_Ensemble':
        western_only = True
    else:
        western_only = False

    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials', project, f'{project}.toml')

    config = ProjectConfig()
    config.read_config(config_file)

    results_dir = os.path.join(config.project_ws, 'ptjpl_test')

    sites, sdf = get_flux_sites(config.station_metadata_csv, crop_only=False,
                                return_df=True, western_only=western_only, header=1)
    sites = ['AFD']
    run_pest_sequence(config, results_dir, select_stations=sites, overwrite=True)
# ========================= EOF ============================================================================
