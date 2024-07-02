import os
import json

import pandas as pd
import geopandas as gpd


def join_site_timeseries(irr_dir, unirr_dir, joined_dir, summary_json, missing_json):
    try:
        with open(summary_json, 'r') as f:
            p_dct = json.load(f)
    except FileNotFoundError:
        p_dct = {}

    missing = {'irr_missing': [],
               'uirr_missing': [],
               'irr_partial': [],
               'uirr_partial': []}

    irr_files = [f for f in os.listdir(irr_dir) if f.endswith('.csv')]
    unirr_files = [f for f in os.listdir(unirr_dir) if f.endswith('.csv')]

    irr_ids = [os.path.basename(f).split('_')[0] for f in irr_files]
    unirr_ids = [os.path.basename(f).split('_')[0] for f in unirr_files]

    ids = list(set(irr_ids + unirr_ids))

    for i, sid in enumerate(ids):

        irr_file = os.path.join(irr_dir, f'{sid}_output.csv')
        unirr_file = os.path.join(unirr_dir, f'{sid}_output.csv')

        try:
            idf = pd.read_csv(irr_file, parse_dates=True, index_col=0)
            idf.columns = ['i_{}'.format(c) for c in idf.columns]
            if idf.index[0] > pd.to_datetime('1989-01-02'):
                missing['irr_partial'].append(sid)

        except FileNotFoundError:
            missing['irr_missing'].append(sid)
            idf = pd.DataFrame()

        try:
            udf = pd.read_csv(unirr_file, parse_dates=True, index_col=0)
            udf.columns = ['u_{}'.format(c) for c in udf.columns]
            if udf.index[0] > pd.to_datetime('1989-01-02'):
                missing['uirr_partial'].append(sid)

        except FileNotFoundError:
            missing['uirr_missing'].append(sid)
            udf = pd.DataFrame()

        joined_df = pd.concat([idf, udf], axis=1, ignore_index=False)

        output_file = os.path.join(joined_dir, f'{sid}_joined.csv')

        try:
            sum_ = joined_df.loc['2012-01-01': '2023-12-31', ['i_irrigation',
                                                              'i_et_act',
                                                              'u_et_act']].sum(axis=0) / 10
            irr = int(sum_['i_irrigation'])
            i_eta = int(sum_['i_et_act'])
            dryland_eta = int(sum_['u_et_act'])
            print(f"{str(i).rjust(4, '.')}: {sid.rjust(6, '.')}; irr: {irr:5d}, i_eta: {i_eta:5d},"
                  f" dryland eta: {dryland_eta:5d}")
            p_dct[sid] = (irr, i_eta, dryland_eta)

        except KeyError:
            print(f'{sid} is missing a dataframe')
            continue

        joined_df.to_csv(output_file)

    with open(summary_json, 'w') as fp:
        json.dump(p_dct, fp, indent=4)

    with open(missing_json, 'w') as fp:
        json.dump(missing, fp, indent=4)


def write_summaries_to_shapefile(in_shp, meta_js, out_shp):
    with open(meta_js, 'r') as f:
        p_dct = json.load(f)

    gdf = gpd.read_file(in_shp)
    gdf.index = [str(f) for f in gdf['FID']]

    for fid, data in p_dct.items():
        gdf.loc[fid, ['irr', 'ieta', 'ueta']] = data

    gdf.to_file(out_shp)


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/swim'

    project = 'tongue'
    data = os.path.join(root, 'examples', project)

    irr_ = os.path.join(data, 'output', 'irr_output')
    unirr_ = os.path.join(data, 'output', 'unirr_output')
    joined_ = os.path.join(data, 'output', 'joined_output')
    js_summary = os.path.join(data, 'output', 'summary.json')
    js_missing = os.path.join(data, 'output', 'missing.json')
    # join_site_timeseries(irr_, unirr_, joined_, js_summary, js_missing)

    ishp = os.path.join(data, 'gis', 'tongue_fields_gfid.shp')
    oshp = os.path.join(data, 'gis', 'tongue_fields_irr.shp')
    write_summaries_to_shapefile(ishp, js_summary, oshp)

# ========================= EOF ====================================================================
