"""
Model input preparation for SWIM-RS.

DEPRECATED: This module is deprecated in favor of swimrs.container components.

Migration guide:
    # Old way (deprecated)
    from swimrs.prep.prep_plots import prep_fields_json
    prep_fields_json(properties, time_series, dynamics, out_js, rs_params)

    # New way (recommended)
    from swimrs.container import SwimContainer
    container = SwimContainer.open("project.swim")
    container.export.prepped_input_json("output.json")
"""

try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

from swimrs.swim.sampleplots import SamplePlots
from swimrs.prep.ndvi_regression import sentinel_adjust_quantile_mapping
from swimrs.prep import _emit_deprecation_warning

REQUIRED = ['tmin', 'tmax', 'srad', 'swe', 'prcp', 'nld_ppt_d',
            'prcp_hr_00', 'prcp_hr_01', 'prcp_hr_02', 'prcp_hr_03', 'prcp_hr_04',
            'prcp_hr_05', 'prcp_hr_06', 'prcp_hr_07', 'prcp_hr_08', 'prcp_hr_09', 'prcp_hr_10',
            'prcp_hr_11', 'prcp_hr_12', 'prcp_hr_13', 'prcp_hr_14', 'prcp_hr_15', 'prcp_hr_16',
            'prcp_hr_17', 'prcp_hr_18', 'prcp_hr_19', 'prcp_hr_20', 'prcp_hr_21', 'prcp_hr_22',
            'prcp_hr_23']

REQ_UNIRR = ['eto']

REQ_IRR = ['eto_corr']

ACCEPT_NAN = REQ_IRR + REQ_UNIRR + ['swe']


def prep_fields_json(properties, time_series, dynamics, out_js, rs_params, target_plots=None,
                     interp_params=None):
    """
    DEPRECATED: Use container.export.prepped_input_json() instead.

    Prepare model input JSON from time series and properties data.
    """
    _emit_deprecation_warning(stacklevel=2)

    with open(properties, 'r') as fp:
        properties = json.load(fp)

    if target_plots is None:
        target_plots = list(properties.keys())

    dct = {'props': {i: r for i, r in properties.items() if i in target_plots}}

    missing = [x for x in target_plots if x not in dct['props'].keys()]
    missing += [x for x in target_plots if not os.path.exists(os.path.join(time_series, '{}.parquet'.format(x)))]
    missing = list(set(missing))

    if missing:
        print('Target sample(s) missing from time series data: {}'.format(missing))
        [target_plots.remove(f) for f in missing]
        dct['props'] = {k: v for k, v in dct['props'].items() if k not in missing}
        if not target_plots:
            return target_plots, missing

    with open(dynamics, 'r') as fp:
        dynamics = json.load(fp)

    required_met_params = REQUIRED + REQ_IRR + REQ_UNIRR
    dct['irr_data'] = {fid: v for fid, v in dynamics['irr'].items() if fid in target_plots}
    dct['gwsub_data'] = {fid: v for fid, v in dynamics['gwsub'].items() if fid in target_plots}
    dct['ke_max'] = {fid: v for fid, v in dynamics['ke_max'].items() if fid in target_plots}
    dct['kc_max'] = {fid: v for fid, v in dynamics['kc_max'].items() if fid in target_plots}

    dts, order = None, []
    first, shape, data = True, None, None
    arrays = {r: [] for r in required_met_params}

    rs_params_strs = ['_'.join(r) if 'etf' in r else '_'.join(r[1:])  for r in rs_params]
    [arrays.update({r: [] for r in rs_params_strs})]

    for fid, v in tqdm(dct['props'].items(), total=len(dct['props']), desc='Writing model input json'):

        if fid in missing:
            continue

        _file = os.path.join(time_series, '{}.parquet'.format(fid))
        df = pd.read_parquet(_file)
        if first:
            shape = df.shape[0]
            doys = [int(dt.strftime('%j')) for dt in df.index]
            dts = [f'{i.year}-{i.month:02d}-{i.day:02d}' for i in df.index]
            data = {dt: {'doy': doy} for dt, doy in zip(dts, doys)}
            order = [fid]
            first = False
        else:
            order.append(fid)
            if not df.shape[0] == shape:
                print('{} does not have shape {}'.format(fid, df.shape[0]))
                continue

        idx = pd.IndexSlice
        p_cols = [c[2] for c in df.columns]
        for p in required_met_params:

            # TODO: get hourly ERA5-Land precip data
            if p == 'nld_ppt_d' and p not in  p_cols:
                a = df.loc[:, idx[:, :, ['prcp'], :, :, :]].values

            elif p in REQUIRED[6:] and not any([c[2] == p for c in df.columns]):
                a = df.loc[:, idx[:, :, ['prcp'], :, :, :]].values / 24.0

            elif p in ['etr_corr', 'eto_corr'] and p not in p_cols:
                a = np.ones_like(df.loc[:, idx[:, :, ['prcp'], :, :, :]].values) * np.nan

            else:
                a = df.loc[:, idx[:, :, [p], :, :, :]].values

            arrays[p].append(a)

        for p in rs_params:
            alg, param, mask = p

            try:
                if interp_params is not None and param in interp_params:
                    int_arr = df.loc[:, idx[:, :, [param], :, [alg], [mask]]]
                    if int_arr.shape[1] > 1:
                        lndvi = df.loc[:, idx[:, ['landsat'], ['ndvi'], :, :, mask]]
                        sndvi_unadj = df.loc[:, idx[:, ['sentinel'], ['ndvi'], :, :, mask]]
                        sent_adj_ndvi = sentinel_adjust_quantile_mapping(sentinel_ndvi_df=sndvi_unadj,
                                                                         landsat_ndvi_df=lndvi,
                                                                         min_pairs=20, window_days=1)
                        int_arr = pd.concat([lndvi, pd.DataFrame(sent_adj_ndvi)],
                                            ignore_index=False, axis=1).mean(axis=1)

                    int_arr = int_arr.interpolate(limit=100)
                    a = int_arr.bfill().ffill().values.reshape((df.shape[0], -1))
                    arrays['_'.join(p[1:])].append(a)
                else:
                    a = df.loc[:, idx[:, :, [param], :, [alg], [mask]]].values
                    if df.shape[1] > 1:
                        a = np.nanmean(a, axis=1).reshape((df.shape[0], 1))
                    arrays['_'.join(p)].append(a)

            except KeyError:
                a = np.ones((df.shape[0], 1)) * np.nan
                if param == 'ndvi':
                    arrays['_'.join(p[1:])].append(a)
                else:
                    arrays['_'.join(p)].append(a)

    all_params = required_met_params + rs_params_strs
    for p in all_params:
        a = np.array(arrays[p]).T
        arrays[p] = a

    for i, dt in enumerate(dts):
        for p in all_params:
            data[dt][p] = arrays[p][0, i, :].tolist()

    dct.update({'order': order, 'time_series': data})
    dct.update({'missing': missing})

    # write large json line-by-line
    with open(out_js, 'w') as f:
        for key, value in dct.items():
            json.dump({key: value}, f)
            f.write('\n')

    print(f'wrote {out_js}')

    return target_plots, missing


def preproc(config):
    ct = 0

    print('Writing observations to file...')

    start = datetime.strftime(config.start_dt, '%Y-%m-%d')
    end = datetime.strftime(config.end_dt, '%Y-%m-%d')

    fields = SamplePlots()
    fields.initialize_plot_data(config)

    # somehow this needs chdir to write savetxt
    if not os.path.isdir(config.obs_folder):
        os.makedirs(config.obs_folder, exist_ok=True)
        os.chdir(config.obs_folder)

    for fid in fields.input['order']:

        data = fields.input_to_dataframe(fid)
        irr_threshold = config.irr_threshold

        data = data.loc[start: end]

        # Prefer irrigated/unirrigated masks when present; otherwise fall back to no_mask (international workflow).
        target = config.etf_target_model
        col_inv = f'{target}_etf_inv_irr'
        col_irr = f'{target}_etf_irr'
        col_nomask = f'{target}_etf_no_mask'

        if col_inv in data.columns:
            data['etf'] = data[col_inv]
            try:
                irr_years = [
                    int(k) for k, v in fields.input['irr_data'][fid].items()
                    if k != 'fallow_years' and v.get('f_irr', 0.0) >= irr_threshold
                ]
            except Exception:
                irr_years = []
            if irr_years and col_irr in data.columns:
                irr_index = [i for i in data.index if i.year in irr_years]
                idx_match = [i for i in irr_index if i in data.index]
                data.loc[idx_match, 'etf'] = data.loc[idx_match, col_irr]
        elif col_nomask in data.columns:
            data['etf'] = data[col_nomask]
        elif col_irr in data.columns:
            data['etf'] = data[col_irr]
        else:
            print(f'{fid}: missing ETf columns for target={target}')
            continue

        print('\n{}\npreproc ETf mean: {:.2f}'.format(fid, np.nanmean(data['etf'].values)))
        _file = os.path.join(config.obs_folder, 'obs_etf_{}.np'.format(fid))
        np.savetxt(_file, data['etf'].values)

        print('preproc SWE mean: {:.2f}\n'.format(np.nanmean(data['swe'].values)))
        _file = os.path.join(config.obs_folder, 'obs_swe_{}.np'.format(fid))
        np.savetxt(_file, data['swe'].values)

        ct += 1

    print('Prepped {} fields input'.format(ct))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
