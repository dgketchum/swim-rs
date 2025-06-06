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


def run_pest_sequence(conf_path, project_ws, workers, realizations, bad_params=None, pdc_remove=False):
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

    if bad_params:
        bad_df = pd.read_csv(bad_params, index_col=0)
        bad_df.dropna(inplace=True, how='all', axis=0)
        bad_df = bad_df[(bad_df['lulc'] == 'Evergreen Forests') & (bad_df['mode'] == 'tight')]
        bad_df = bad_df[bad_df['monthly_rmse_diff_pct'] < 0.0]
        fids = bad_df.index.unique().to_list() + ['ET_8', 'S2']

    else:
        fids = [f for f, r in flux_meta_df.iterrows() if r['General classification'] == 'Evergreen']

    os.chdir(os.path.dirname(__file__))

    prepped_input = os.path.join(data_dir, 'prepped_input.json')

    prep_fields_json(properties_json, joined_timeseries, dynamics_data,
                     prepped_input, target_plots=fids)

    obs_dir = os.path.join(project_ws, 'obs')
    if not os.path.isdir(obs_dir):
        os.makedirs(obs_dir, exist_ok=True)

    preproc(conf_path, project_ws)

    py_script = os.path.join(project_ws, 'custom_forward_run.py')

    now = datetime.now().strftime('%m%d%H%M')
    target_dir = os.path.join(project_ws, 'results', now)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    station_prepped_input = os.path.join(target_dir, f'prepped_input.json')
    shutil.copyfile(prepped_input, station_prepped_input)

    p_dir = os.path.join(project_ws, f'tight_pest')
    if os.path.isdir(p_dir):
        shutil.rmtree(p_dir)

    m_dir = os.path.join(project_ws, f'tight_master')
    if os.path.isdir(m_dir):
        shutil.rmtree(m_dir)

    w_dir = os.path.join(project_ws, 'workers')
    if os.path.isdir(w_dir):
        shutil.rmtree(w_dir)

    r_dir = os.path.join(project_ws, 'results')
    if not os.path.isdir(r_dir):
        os.mkdir(r_dir)

    # cwd must be reset to avoid FileNotFound on PstFrom.log
    os.chdir(os.path.dirname(__file__))

    builder = PestBuilder(project_ws=project_ws, config_file=conf_path,
                          use_existing=False, python_script=py_script,
                          prior_constraint='tight', conflicted_obs=None)
    builder.build_pest()
    builder.build_localizer()

    # short run sets up base realization and checks for prior-data conflict
    if pdc_remove:
        builder.write_control_settings(noptmax=-1, reals=5)
    else:
        builder.write_control_settings(noptmax=0)

    builder.spinup(overwrite=True)
    # copy spinup to station results location

    spinup_out = os.path.join(target_dir, f'spinup.json')
    shutil.copyfile(builder.config.spinup, spinup_out)

    builder.dry_run('pestpp-ies')

    # Check for prior-data conflict, remove and rebuild if necessary
    pdc_file = os.path.join(p_dir, f'{project}.pdc.csv')

    if os.path.exists(pdc_file) and pdc_remove:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdc = os.path.join(temp_dir, f'{project}.pdc.csv')
            shutil.copyfile(pdc_file, temp_pdc)

            builder = PestBuilder(project_ws=project_ws, config_file=conf_path,
                                  use_existing=False, python_script=py_script,
                                  prior_constraint='tight', conflicted_obs=temp_pdc)
            builder.build_pest()
            builder.build_localizer()
            builder.write_control_settings(noptmax=0)
            builder.dry_run('pestpp-ies')

    builder.write_control_settings(noptmax=3, reals=realizations)

    exe_ = 'pestpp-ies'

    _pst = f'{project}.pst'

    run_pst(p_dir, exe_, _pst, num_workers=workers, worker_root=w_dir,
            master_dir=m_dir, verbose=False, cleanup=False)

    fcst_file = os.path.join(m_dir, f'{project}.3.par.csv')
    fcst_out = os.path.join(target_dir, f'{project}.3.par.csv')
    if not os.path.exists(fcst_file):
        fcst_file = os.path.join(m_dir, f'{project}.2.par.csv')
        fcst_out = os.path.join(target_dir, f'{project}.2.par.csv')

    shutil.copyfile(fcst_file, fcst_out)
    print(f'Wrote {fcst_out}')

    phi_csv = os.path.join(m_dir, f'{project}.phi.meas.csv')
    phi_csv_out = os.path.join(target_dir, f'{project}.phi.meas.csv')

    shutil.copyfile(phi_csv, phi_csv_out)
    print(f'Wrote {phi_csv_out}')

    pdc_file = os.path.join(m_dir, f'{project}.pdc.csv')
    pdc_out = os.path.join(target_dir, f'{project}.pdc.csv')

    obs_idx_file = os.path.join(m_dir, f'{project}.idx.csv')
    obs_idx_out = os.path.join(target_dir, f'{project}.idx.csv')

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
    d = '/data/ssd2/swim'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'PycharmProjects', 'swim-rs', 'tutorials')

    project_ = '4_Flux_Network'
    project_ws_ = os.path.join(d, project_)

    config_file = os.path.join(project_ws_, 'config.toml')

    bad_parameters = ('/home/dgketchum/PycharmProjects/swim-rs/tutorials/'
                      '4_Flux_Network/results_comparison_bad.csv')

    run_pest_sequence(config_file, project_ws_, workers=22, realizations=200, bad_params=bad_parameters,
                      pdc_remove=True)

# ========================= EOF ============================================================================
