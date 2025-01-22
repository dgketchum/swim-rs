import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

from prep.prep_plots import TONGUE_SELECT


def write_field_properties(shp, irr, ssurgo, js, cdl=None, landfire=None, index_col='FID',
                           shp_add=False, targets=None):
    """"""
    irr = pd.read_csv(irr, index_col=index_col)
    irr.drop(columns=['LAT', 'LON'], inplace=True)

    dct = irr.T.to_dict()
    dct = {k: {'irr': {int(kk.split('_')[1]): vv for kk, vv in v.items()}} for k, v in dct.items()}

    soils = pd.read_csv(ssurgo, index_col=index_col)

    awc = soils[['awc']].copy()
    awc = awc.T.to_dict()
    [dct[k].update({'awc': awc[k]['awc']}) for k in dct.keys()]

    ksat = soils[['ksat']].copy()
    ksat = ksat.T.to_dict()
    [dct[k].update({'ksat': ksat[k]['ksat']}) for k in dct.keys()]

    clay = soils[['clay']].copy()
    clay = clay.T.to_dict()
    [dct[k].update({'clay': clay[k]['clay']}) for k in dct.keys()]

    sand = soils[['sand']].copy()
    sand = sand.T.to_dict()
    [dct[k].update({'sand': sand[k]['sand']}) for k in dct.keys()]

    fields = gpd.read_file(shp)
    fields.index = fields[index_col]
    fields['area_sq_m'] = [g.area for g in fields['geometry']]
    area_sq_m = fields[['area_sq_m']].copy()
    area_sq_m = area_sq_m.T.to_dict()
    [dct[k].update({'area_sq_m': area_sq_m[k]['area_sq_m']}) for k in dct.keys()]

    if cdl is not None:
        cdl = pd.read_csv(cdl, index_col=index_col)
        try:
            cdl.drop(columns=['LAT', 'LON'], inplace=True)
        except KeyError:
            pass
        cdl.fillna(0.0, axis=0, inplace=True)
        cdl = cdl.T.to_dict()
        [dct[k].update({'cdl': {int(kk.split('_')[1]): int(vv) for kk, vv in cdl[k].items()}}) for k in dct.keys()]

    if landfire is not None:
        landfire = pd.read_csv(landfire, index_col=index_col)
        plant_height = landfire[['height']].copy()
        plant_height = plant_height.T.to_dict()
        [dct[k].update({'plant_height': plant_height[k]['height']}) for k in dct.keys()]

    d = dct.copy()
    for k, v in dct.items():
        if np.any(np.isnan([v['awc'], v['ksat'], v['area_sq_m']])):
            _ = d.pop(k)
            print('skipping {}: has nan'.format(k))
        elif v['area_sq_m'] < 900.:
            _ = d.pop(k)
            print('skipping {}: has small area'.format(k))

    if shp_add:
        gdf = gpd.read_file(shp_add)
        gdf.index = gdf[index_col]
        irr['irr_mean'] = irr.mean(axis=1)
        irr['irr_std'] = irr.std(axis=1)

        idx = [i for i in irr.index if i in gdf.index]
        gdf.loc[idx, 'irr_mean'] = irr.loc[idx, 'irr_mean']
        gdf.loc[idx, 'irr_std'] = irr.loc[idx, 'irr_std']
        areas = pd.Series(data=[dct[k]['area_sq_m'] for k in idx], index=idx)
        gdf.loc[idx, 'area'] = areas.loc[idx]

        gdf.drop(columns=[index_col], inplace=True)
        if targets:
            gdf = gdf.iloc[targets]
        gdf.to_file(shp_add.replace('.shp', '_sample_19JUNE2024.shp'))

    with open(js, 'w') as fp:
        json.dump(d, fp, indent=4)


if __name__ == '__main__':

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    data = os.path.join(root, 'tutorials', '4_Flux_Network', 'data')
    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')

    FEATURE_ID = 'field_1'

    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    ssurgo = os.path.join(data, 'properties', 'calibration_ssurgo.csv')
    properties_json = os.path.join(data, 'properties', 'calibration_properties.json')

    write_field_properties(shapefile_path, irr, ssurgo, properties_json, index_col=FEATURE_ID, shp_add=None,
                           targets=None)

# ========================= EOF ====================================================================
