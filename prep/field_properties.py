import json

import geopandas as gpd
import numpy as np
import pandas as pd

from prep import MAX_EFFECTIVE_ROOTING_DEPTH as RZ


def write_field_properties(shp, js, lulc, irr=None, ssurgo=None, cdl=None, flux_meta=None, index_col='FID'):
    """"""
    if lulc is None:
        raise ValueError("The 'lulc' CSV file path must be provided.")

    lulc_df = pd.read_csv(lulc, index_col=index_col)
    rz = lulc_df[['mode']].copy()
    rz = rz.T.to_dict()
    dct = {}
    for k, v in rz.items():
        dct[k] = {
            'root_depth': RZ.get(str(v['mode']), {}).get('rooting_depth', np.nan),
            'zr_mult': RZ.get(str(v['mode']), {}).get('zr_multiplier', np.nan),
            'lulc_code': v['mode']
        }

    if irr is not None:
        irr_df = pd.read_csv(irr, index_col=index_col)
        if 'LAT' in irr_df.columns and 'LON' in irr_df.columns:
            irr_df.drop(columns=['LAT', 'LON'], inplace=True)
        irr_dct = irr_df.T.to_dict()
        for k, v in irr_dct.items():
            if k in dct:
                dct[k]['irr'] = {int(kk.split('_')[1]): vv for kk, vv in v.items()}

    if ssurgo is not None:
        soils = pd.read_csv(ssurgo, index_col=index_col)
        awc = soils[['awc']].copy().T.to_dict()
        ksat = soils[['ksat']].copy().T.to_dict()
        clay = soils[['clay']].copy().T.to_dict()
        sand = soils[['sand']].copy().T.to_dict()
        for k in dct:
            if k in awc:
                dct[k]['awc'] = awc[k]['awc']
            if k in ksat:
                dct[k]['ksat'] = ksat[k]['ksat']
            if k in clay:
                dct[k]['clay'] = clay[k]['clay']
            if k in sand:
                dct[k]['sand'] = sand[k]['sand']

    fields = gpd.read_file(shp)
    fields.index = fields[index_col]
    fields['area_sq_m'] = [g.area for g in fields['geometry']]
    area_sq_m = fields[['area_sq_m']].copy().T.to_dict()
    for k in dct:
        if k in area_sq_m:
            dct[k]['area_sq_m'] = area_sq_m[k]['area_sq_m']

    if cdl is not None:
        cdl_df = pd.read_csv(cdl, index_col=index_col)
        try:
            cdl_df.drop(columns=['LAT', 'LON'], inplace=True)
        except KeyError:
            pass
        cdl_df.fillna(0.0, axis=0, inplace=True)
        cdl_dct = cdl_df.T.to_dict()
        for k, v in cdl_dct.items():
            if k in dct:
                dct[k]['cdl'] = {int(kk.split('_')[1]): int(vv) for kk, vv in v.items()}

    if flux_meta is not None:
        flux = pd.read_csv(flux_meta, header=1, skip_blank_lines=True, index_col='Site ID')
        flux_lulc = flux[['General classification']].copy().T.to_dict()
        for k, v in flux_lulc.items():
            if k in dct:
                dct[k]['flux_lulc'] = v['General classification']

    d = dct.copy()
    for k, v in dct.items():
        has_nan = False
        if 'awc' in v and np.isnan(v['awc']):
            has_nan = True
        if 'ksat' in v and np.isnan(v['ksat']):
            has_nan = True
        if 'area_sq_m' in v and np.isnan(v['area_sq_m']):
            has_nan = True

        if has_nan:
            _ = d.pop(k)
            print(f'skipping {k}: has nan')
        elif 'area_sq_m' in v and v['area_sq_m'] < 900.:
            _ = d.pop(k)
            print(f'skipping {k}: has small area')

    with open(js, 'w') as fp:
        json.dump(d, fp, indent=4)
    print(f'wrote {len(d)} fields\n {js}')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
