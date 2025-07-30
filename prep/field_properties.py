import json

import geopandas as gpd
import numpy as np
import pandas as pd

from prep import MAX_EFFECTIVE_ROOTING_DEPTH as RZ


def write_field_properties(shp, out_js, lulc, irr=None, soils=None, cdl=None, flux_meta=None, index_col='FID',
                           select=None,
                           lulc_key='mode', **kwargs):
    """"""
    if lulc is None:
        raise ValueError("The 'lulc' CSV file path must be provided.")

    dct = {}

    lulc_df = pd.read_csv(lulc, index_col=index_col)
    # lulc can be for global landcover or modis; this is modis
    lc = lulc_df[[lulc_key]].copy().replace(np.nan, '0').astype(int)
    lc = lc.T.to_dict()
    for k, v in lc.items():
        if select is not None and k not in select:
            continue
        dct[k] = {
            'root_depth': RZ.get(str(v[lulc_key]), {}).get('rooting_depth', np.nan),
            'zr_mult': RZ.get(str(v[lulc_key]), {}).get('zr_multiplier', np.nan),
            'lulc_code': v[lulc_key]
        }

    if 'extra_lulc_key' in kwargs:
        elc_key = kwargs['extra_lulc_key']
        elc = lulc_df[[elc_key]].copy().replace(np.nan, '0').astype(int)
        elc = elc.T.to_dict()
        for k, v in elc.items():
            if k in dct:
                val = v[elc_key]
                if val == 10 and dct[k]['lulc_code'] != 12:
                    print(f'{k}: Override MODIS non-crop code {dct[k]["lulc_code"]} with crop code from {elc_key}')
                    dct[k] = {
                        'root_depth': RZ.get(str(12), {}).get('rooting_depth', np.nan),
                        'zr_mult': RZ.get(str(12), {}).get('zr_multiplier', np.nan),
                        'lulc_code': 12
                    }
                dct[k][elc_key] = val

    if irr is not None:
        irr_df = pd.read_csv(irr, index_col=index_col)
        if 'LAT' in irr_df.columns and 'LON' in irr_df.columns:
            irr_df.drop(columns=['LAT', 'LON'], inplace=True)
        irr_dct = irr_df.T.to_dict()
        for k, v in irr_dct.items():
            if k in dct:
                vals = {int(kk.split('_')[1]): vv for kk, vv in v.items()}
                irr_val = np.array([v for v in vals.values()]).mean()
                if irr_val > 0.3 and dct[k]['lulc_code'] != 12:
                    print(f'{k}: Override MODIS non-crop code {dct[k]["lulc_code"]} with crop code from irrigation mask')
                    dct[k] = {
                        'root_depth': RZ.get(str(12), {}).get('rooting_depth', np.nan),
                        'zr_mult': RZ.get(str(12), {}).get('zr_multiplier', np.nan),
                        'lulc_code': 12
                    }
                dct[k]['irr'] = vals

    if soils is not None:
        soil_df = pd.read_csv(soils, index_col=index_col)
        props = ['awc', 'ksat', 'clay', 'sand']

        for prop in props:
            if prop not in soil_df.columns:
                continue
            d = soil_df[[prop]].copy().T.to_dict()
            for k in dct:
                if k in d:
                    try:
                        dct[k].update({prop: d[k][prop]})
                    except KeyError:
                        continue

    fields = gpd.read_file(shp)
    fields.index = fields[index_col]

    if fields.crs and fields.crs.is_geographic:
        dummy = fields.copy().to_crs(epsg=6933)
        fields['area_sq_m'] = dummy['geometry'].area
    else:
        fields['area_sq_m'] = fields.geometry.area

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

    with open(out_js, 'w') as fp:
        json.dump(d, fp, indent=4)
    print(f'wrote {len(d)} fields properties\n {out_js}')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
