import json
import os

import geopandas as gpd
import pandas as pd


def write_field_properties(shp, irr, cdl, soils, js):
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

    fields = gpd.read_file(shp)
    fields.index = fields['FID']
    awc = gpd.read_file(os.path.join(soils, 'AWC_WTA_0to152cm_statsgo.shp'), mask=fields)

    intersect = gpd.overlay(awc, fields, how='intersection')
    intersect['area'] = [g.area for g in intersect['geometry']]

    for fid, row in fields.iterrows():
        inter = intersect[intersect['FID'] == fid]
        tot_area = inter['area'].sum()
        awc = (inter['AWC'] * inter['area'] / tot_area).sum()

        dct[fid]['stn_whc'] = awc * 12

    with open(js, 'w') as fp:
        json.dump(dct, fp, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'

    soils_ = os.path.join(d, 'soils_aea')

    project = 'tongue'
    project_ws = os.path.join(d, 'examples', project)

    fields_shp = os.path.join(project_ws, 'gis', '{}_fields.shp'.format(project))

    irr_ = os.path.join(project_ws, 'properties', '{}_irr.csv'.format(project))
    cdl_ = os.path.join(project_ws, 'properties', '{}_cdl.csv'.format(project))
    jsn = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))

    write_field_properties(fields_shp, irr_, cdl_, soils_, jsn)

# ========================= EOF ====================================================================
