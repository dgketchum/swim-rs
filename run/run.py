import os
import time

import numpy as np
import pandas as pd

from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from swim.input import SamplePlots


def run_fields(ini_path, project='tongue', calibration_dir=None, parameter_distribution=None,
               parameter_set=None, write_files=None):
    start_time = time.time()

    config = ProjectConfig()
    config.read_config(ini_path, calibration_folder=calibration_dir, parameter_dist_csv=parameter_distribution)

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

    targets = fields.input['order']

    print('Warning: model runner is set to debug=True, it will not write results accessible to PEST++')
    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    for i, fid in enumerate(targets):
        df = df_dct[fid].copy()
        pred_et = df['et_act'].values

        irr_data = fields.input['irr_data'][fid]
        f_irr = np.nanmean([irr_data[str(yr)]['f_irr'] for yr in range(1989, 2023) if str(yr) in irr_data.keys()])
        print('\n{} Mean Irrigated Fraction: {:.2f}'.format(fid, f_irr))
        obs_etf = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_etf_{}.np'.format(project, fid)
        obs_etf = np.loadtxt(obs_etf)
        cols = ['etf_obs'] + list(df.columns)
        df['etf_obs'] = obs_etf

        obs_eta = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}/obs/obs_eta_{}.np'.format(project, fid)
        obs_eta = np.loadtxt(obs_eta)
        cols = ['eta_obs'] + cols
        df['eta_obs'] = obs_eta

        df = df[cols]
        df.index = [pd.to_datetime(i) for i in df.index]

        if write_files:
            file_ = write_files.format(fid)
            df.to_csv(file_)

        comp = pd.DataFrame(data=np.vstack([obs_eta, pred_et]).T, columns=['obs', 'pred'], index=df.index)
        comp['eq'] = comp['obs'] == comp['pred']
        comp['capture'] = df['capture']

        df = df[cols]
        sdf = df.loc['2020-01-01': '2020-12-31']

        comp = df.loc[df[df['capture'] == 1.0].index].copy()
        et_act, et_ssebop = comp['et_act'], comp['eta_obs']
        rmse = np.sqrt(((et_act - et_ssebop) ** 2).mean())

        print('{} Capture Dates; Mean SSEBop: {:.2f}, SWB Pred: {:.2f}, RMSE: {:.4f}'.format(comp.shape[0],
                                                                                             et_ssebop.mean(),
                                                                                             et_act.mean(), rmse))

        totals = df[['et_act', 'ppt', 'dperc', 'runoff', 'irrigation']].sum(axis=0)
        water_out = totals[['dperc', 'et_act', 'runoff']].sum()
        storage = df.loc[df.index[0], 'depl_root'] - df.loc[df.index[-1], 'depl_root']
        balance = totals['irrigation'] + totals['ppt'] - storage - water_out
        print('Water Balance = {:.1f}; input: {:.1f}; output: {:.1f}; storage: {:.1f}; irrigation: {:.1f}\n\n\n'.format(
            balance,
            totals['ppt'],
            water_out,
            storage,
            totals['irrigation']))

    return None


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    project = 'tongue'
    d = '/home/dgketchum/PycharmProjects/swim-rs/examples/{}'.format(project)
    conf = os.path.join(d, '{}_swim.toml'.format(project))

    tuned = os.path.join(root, 'swim/examples/{}/calibrated_models/model_tongue_7JUN2024/'.format(project, project))
    pars = os.path.join(tuned, '{}.4.par.csv'.format(project))

    # TODO: kwargs to override the forecast and calibration setting on the .toml
    run_fields(conf, project=project, write_files=None, parameter_distribution=pars)

# ========================= EOF ====================================================================
