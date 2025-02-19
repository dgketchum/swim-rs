import json
import os
import shutil

import pandas as pd

from calibrate.pest_builder import PestBuilder
from calibrate.run_pest import run_pst
from swim.config import ProjectConfig
from swim.input import SamplePlots
from prep.prep_plots import prep_fields_json, preproc


def run_pest_sequence(conf_path, project_ws, workers, realizations):
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

    flux_meta_csv = os.path.join(data_dir, 'station_metadata.csv')
    flux_meta_df = pd.read_csv(flux_meta_csv, header=1, skip_blank_lines=True, index_col='Site ID')

    for fid, row in flux_meta_df.iterrows():

        for prior_constraint in ['loose', 'tight']:

            prepped_input = os.path.join(data_dir, 'prepped_input.json')

            prep_fields_json(properties_json, joined_timeseries, dynamics_data,
                             prepped_input, target_plots=[fid])

            obs_dir = os.path.join(project_ws_, 'obs')
            if not os.path.isdir(obs_dir):
                os.makedirs(obs_dir, exist_ok=True)

            preproc(conf_path, project_ws_)

            config_path_ = os.path.join(project_ws_, 'config.toml')
            py_script = os.path.join(project_ws_, 'custom_forward_run.py')

            builder = PestBuilder(project_ws=project_ws_, config_file=config_path_,
                                  use_existing=False, python_script=py_script, prior_constraint=prior_constraint)
            builder.build_pest()
            builder.build_localizer()
            builder.dry_run('pestpp-ies')
            builder.write_control_settings(noptmax=3, reals=realizations)

            p_dir = os.path.join(project_ws_, f'{prior_constraint}_pest')
            m_dir = os.path.join(project_ws_, f'{prior_constraint}_master')
            w_dir = os.path.join(project_ws_, 'workers')

            r_dir = os.path.join(project_ws_, 'results')
            if not os.path.isdir(r_dir):
                os.mkdir(r_dir)

            exe_ = 'pestpp-ies'

            _pst = f'{project}.pst'

            run_pst(p_dir, exe_, _pst, num_workers=workers, worker_root=w_dir,
                    master_dir=m_dir, verbose=False, cleanup=True)

            fcst_file = os.path.join(project_ws_, m_dir, f'{project}.3.par.csv')
            fcst_out = os.path.join(project_ws_, r_dir, f'{fid}.3.par.csv')
            if not os.path.exists(fcst_file):
                fcst_file = os.path.join(project_ws_, m_dir, f'{project}.2.par.csv')
                fcst_out = os.path.join(project_ws_, r_dir, f'{fid}.2.par.csv')

            shutil.copyfile(fcst_file, fcst_out)
            print(f'Wrote {os.path.basename(fcst_out)} to {r_dir}.')

            pdc_file = os.path.join(m_dir, f'{project}.pdc.csv')
            pdc_out = os.path.join(r_dir, f'{fid}.pdc.csv')

            obs_idx_file = os.path.join(m_dir, f'{project}.idx.csv')
            obs_idx_out = os.path.join(r_dir, f'{fid}.idx.csv')

            if os.path.exists(pdc_file):
                shutil.copyfile(pdc_file, pdc_out)
                shutil.copyfile(obs_idx_file, obs_idx_out)
                print(f'Wrote {os.path.basename(pdc_out)} to {r_dir}.')
                print(f'Wrote {os.path.basename(obs_idx_out)} to {r_dir}.')
                print('')



if __name__ == '__main__':
    d = '/data/ssd2/swim'

    project_ = '4_Flux_Network'
    project_ws_ = os.path.join(d, project_)

    config_file = os.path.join(project_ws_, 'config.toml')

    run_pest_sequence(config_file, project_ws_, workers=50, realizations=300)

# ========================= EOF ====================================================================
