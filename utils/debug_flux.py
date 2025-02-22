import os

import pandas as pd
from pyemu import Pst

from calibrate.pest_builder import PestBuilder


def debug_calibration(conf_path, project_ws, fid, params=None, pdc_file=None):
    """"""

    builder = PestBuilder(project_ws=project_ws_, config_file=conf_path,
                          use_existing=True)

    irr = builder.plots.input['irr_data'][fid]

    data_dir = os.path.join(project_ws, 'data')

    flux_data = os.path.join(data_dir, 'daily_flux_files', f'{fid}_daily_data.csv')
    if not os.path.exists(flux_data):
        flux_data = os.path.join(data_dir, f'{fid}_daily_data.csv')
    flux_data = pd.read_csv(flux_data, index_col='date', parse_dates=True)

    out_csv = os.path.join(data_dir, 'model_output', f'{fid}.csv')
    output = pd.read_csv(out_csv, index_col=0)

    output.index = pd.to_datetime(output.index)

    irr_threshold = 0.3
    irr_years = [int(k) for k, v in irr.items() if k != 'fallow_years'
                 and v['f_irr'] >= irr_threshold]
    irr_index = [i for i in output.index if i.year in irr_years]

    output['etf'] = output['etf_inv_irr']
    output.loc[irr_index, 'etf'] = output.loc[irr_index, 'etf_irr']

    output['capture'] = output['etf_inv_irr_ct']
    output.loc[irr_index, 'capture'] = output.loc[irr_index, 'etf_irr_ct']

    output['ndvi'] = output['ndvi_inv_irr']
    output.loc[irr_index, 'ndvi'] = output.loc[irr_index, 'ndvi_irr']

    output['pdc'] = 0

    df = pd.DataFrame({'kc_act': output['kc_act'],
                       'ndvi': output['ndvi'],
                       'etf': output['etf'],
                       'EToF': flux_data['EToF'],
                       'ET_corr': flux_data['ET_corr'],
                       'capture': output['capture'],
                       'eto': output['eto_mm']})
    if params:
        df[params] = output[params]

    if pdc_file is not None and os.path.exists(pdc_file):
        pdc = pd.read_csv(pdc_file, index_col=0)
        idx_file = pdc_file.replace('.pdc', '.idx')
        idx = pd.read_csv(idx_file, index_col=0, parse_dates=True)
        idx['pdc'] = [1 if obs_id in pdc.index else 0 for obs_id in idx['obs_id']]

    df['pdc'] = idx['pdc']
    pdc_yr = df[['pdc']].resample('A').sum()
    dfpdc = df[df['pdc'] == 1]

    a = 1


if __name__ == '__main__':
    home = os.path.expanduser('~')

    project = '4_Flux_Network'

    root = os.path.join(home, 'PycharmProjects', 'swim-rs')
    project_ws_ = os.path.join(root, 'tutorials', project)
    config_file = os.path.join(project_ws_, 'config.toml')
    pdc = os.path.join(project_ws_, 'master', f'{project}.pdc.csv')

    add_params = ['irr_day', 'irrigation', 'depl_root', 'ks', 'depl_ze']

    pdc_ = '/data/ssd2/swim/4_Flux_Network/results/loose/US-Blo/US-Blo.pdc.csv'
    debug_calibration(config_file, project_ws_, 'US-Blo', add_params, pdc_file=pdc_)

# ========================= EOF ====================================================================
