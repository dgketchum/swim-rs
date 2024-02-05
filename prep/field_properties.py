import json
import os

import geopandas as gpd
import pandas as pd


def write_field_properties(shp, irr, cdl, ssurgo, js):
    irr = pd.read_csv(irr, index_col='FID')
    irr.drop(columns=['LAT', 'LON'], inplace=True)
    dct = irr.T.to_dict()
    dct = {k: {'irr': {int(kk.split('_')[1]): vv for kk, vv in v.items()}} for k, v in dct.items()}
    cdl = pd.read_csv(cdl, index_col='FID')
    try:
        cdl.drop(columns=['LAT', 'LON'], inplace=True)
    except KeyError:
        pass
    cdl.fillna(0.0, axis=0, inplace=True)
    cdl = cdl.T.to_dict()
    [dct[k].update({'cdl': {int(kk.split('_')[1]): int(vv) for kk, vv in cdl[k].items()}}) for k in dct.keys()]

    soils = pd.read_csv(ssurgo, index_col='FID')

    awc = soils[['awc']].copy()
    awc = awc.T.to_dict()
    [dct[k].update({'awc': awc[k]['awc']}) for k in dct.keys()]

    ksat = soils[['ksat']].copy()
    ksat = ksat.T.to_dict()
    [dct[k].update({'ksat': ksat[k]['ksat']}) for k in dct.keys()]

    fields = gpd.read_file(shp)
    fields.index = fields['FID']
    fields['area_sq_m'] = [g.area for g in fields['geometry']]
    area_sq_m = fields[['area_sq_m']].copy()
    area_sq_m = area_sq_m.T.to_dict()
    [dct[k].update({'area_sq_m': area_sq_m[k]['area_sq_m']}) for k in dct.keys()]

    with open(js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    fields_shp = os.path.join(project_ws, 'gis', '{}_fields.shp'.format(project))

    irr_ = os.path.join(project_ws, 'properties', '{}_irr.csv'.format(project))
    cdl_ = os.path.join(project_ws, 'properties', '{}_cdl.csv'.format(project))
    _ssurgo = os.path.join(project_ws, 'properties', '{}_ssurgo.csv'.format(project))
    jsn = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))

    write_field_properties(fields_shp, irr_, cdl_, _ssurgo, jsn)

# ========================= EOF ====================================================================
