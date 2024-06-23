import os
import shutil
import os
import json

import geopandas as gpd

from prep.field_timeseries import join_daily_timeseries
from prep.prep_plots import prep_fields_json, preproc

from calibrate.build_etd_pp_multi import get_pest_builder_args, build_pest, build_localizer, write_control_settings
from calibrate.run_pest import run_pst

d = '/media/research/IrrigationGIS/swim'
if not os.path.exists(d):
    d = '/home/dgketchum/data/IrrigationGIS/swim'

project = 'tongue'

data = os.path.join(d, 'examples', project)
src = '/home/dgketchum/PycharmProjects/swim-rs'
project_ws = os.path.join(src, 'examples', project)

DATA_DIRS = {'fields_gridmet': os.path.join(data, 'gis', '{}_fields_gfid.shp'.format(project)),
             'met_data': os.path.join(data, 'met_timeseries'),
             'landsat': os.path.join(data, 'landsat', '{}_sensing.csv'.format(project)),
             'snow_data': os.path.join(data, 'snow_timeseries', 'snodas_{}.json'.format(project)),
             'input_ts_out': os.path.join(data, 'input_timeseries'),
             'props': os.path.join(data, 'properties', '{}_props.json'.format(project)),
             'prepped_input': os.path.join(data, 'prepped_input', '{}_input_sample.json'.format(project)),
             'cuttings': os.path.join(d, 'examples/{}/landsat/{}_cuttings.json'.format(project, project)),
             }

PEST_DATA = {'_pst': '{}.pst'.format(project),
             '_workers': 6,
             'exe_': 'pestpp-ies',
             'm_dir': os.path.join(project_ws, 'master'),
             'p_dir': os.path.join(project_ws, 'pest'),
             'w_dir': os.path.join(project_ws, 'workers'),
             'python_script': os.path.join(src, 'calibrate', 'custom_forward_run.py')}

LST_PARAMS = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']


def run_pest_sequence(project_tracker, index_col='FID', chunk_sz=10, realizations=100,
                      start_date='2018-01-01', end_date='2021-12-31'):
    if not os.path.exists(project_tracker):
        p_dct = {'project': project,
                 'start_date': start_date,
                 'end_data': end_date,
                 'fields': {}}
    else:
        with open(project_tracker, 'r') as f:
            p_dct = json.load(f)

    gmt, met, lst = DATA_DIRS['fields_gridmet'], DATA_DIRS['met_data'], DATA_DIRS['landsat']
    snow, ts = DATA_DIRS['snow_data'], DATA_DIRS['input_ts_out']

    gdf = gpd.read_file(gmt)
    gdf.index = gdf[index_col]

    while len(gdf.index) > len(p_dct['fields']):

        targets = []
        for i in gdf.index:
            try:
                if len(targets) < chunk_sz:
                    targets.append(str(i))
                else:
                    break
            except IndexError:
                break

        # join_daily_timeseries(gmt, met, lst, snow, ts, overwrite=True,
        #                       start_date=start_date, end_date=end_date, **{'target_fields': targets,
        #                                                                    'params': LST_PARAMS})
        #
        # prep_fields_json(DATA_DIRS['props'], targets, ts, DATA_DIRS['prepped_input'], irr_data=DATA_DIRS['cuttings'])
        #
        # preproc(targets, ts, project_ws)

        # noinspection PyTypedDict
        dct_ = get_pest_builder_args(project_ws, DATA_DIRS['prepped_input'], ts)
        dct_.update({'python_script': PEST_DATA['python_script']})
        build_pest(project_ws, PEST_DATA['p_dir'], **dct_)

        pst_f = os.path.join(PEST_DATA['p_dir'], '{}.pst'.format(project))

        build_localizer(pst_f, ag_json=DATA_DIRS['prepped_input'])
        write_control_settings(pst_f, 3, realizations)

        run_pst(PEST_DATA['p_dir'], PEST_DATA['exe_'], PEST_DATA['_pst'],
                num_workers=PEST_DATA['_workers'], worker_root=PEST_DATA['w_dir'],
                master_dir=PEST_DATA['m_dir'], verbose=False)


if __name__ == '__main__':
    p_tracker = os.path.join(data, '{}_params.json'.format(project))

    run_pest_sequence(p_tracker, index_col='FID', chunk_sz=2, realizations=5,
                      start_date='2018-01-01', end_date='2021-12-31')
# ========================= EOF ====================================================================
