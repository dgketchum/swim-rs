import os
import json
import time
import shutil

import pandas as pd
import geopandas as gpd

from prep.field_timeseries import join_daily_timeseries
from prep.prep_plots import prep_fields_json, preproc

from calibrate.build_pp_files import get_pest_builder_args, initial_parameter_dict
from calibrate.build_pp_files import build_pest, build_localizer, write_control_settings
from calibrate.run_pest import run_pst

d = '/media/research/IrrigationGIS/swim'
if not os.path.exists(d):
    d = '/home/dgketchum/data/IrrigationGIS/swim'

project = 'flux'
data = os.path.join(d, 'examples', project)

# annex_project = 'tongue_annex'
# annex_data = os.path.join(d, 'examples', annex_project)

src = '/home/dgketchum/PycharmProjects/swim-rs'
project_ws = os.path.join(src, 'examples', project)

DATA_DIRS = {'fields_gridmet': os.path.join(data, 'gis', '{}_fields_gfid.shp'.format(project)),
             'met_data': os.path.join(data, 'met_timeseries'),
             'landsat': os.path.join(data, 'landsat', '{}_sensing.csv'.format(project)),
             'snow_data': os.path.join(data, 'snow_timeseries', 'snodas_{}.json'.format(project)),
             'plot_timeseries': os.path.join(data, 'plot_timeseries'),
             'props': os.path.join(data, 'properties', '{}_props.json'.format(project)),
             'prepped_input': os.path.join(data, 'prepped_input', '{}_input_sample.json'.format(project)),
             'cuttings': os.path.join(d, 'examples/{}/landsat/{}_cuttings.json'.format(project, project)),
             }

PEST_DATA = {'_pst': '{}.pst'.format(project),
             'exe_': 'pestpp-ies',
             'm_dir': os.path.join(project_ws, 'master'),
             'p_dir': os.path.join(project_ws, 'pest'),
             'w_dir': os.path.join(project_ws, 'workers'),
             'python_script': os.path.join(src, 'calibrate', 'custom_forward_run.py')}


def run_pest_sequence(project_tracker, n_workers, index_col='FID', chunk_sz=10, realizations=100, iterations=3,
                      start_date='2018-01-01', end_date='2021-12-31'):
    pars = [k for k, v in initial_parameter_dict('').items()]

    if not os.path.exists(project_tracker):
        p_dct = {'project': project,
                 'start_date': start_date,
                 'end_data': end_date,
                 'fields': {}}
    else:
        with open(project_tracker, 'r') as f:
            p_dct = json.load(f)

    gmt, met, lst = DATA_DIRS['fields_gridmet'], DATA_DIRS['met_data'], DATA_DIRS['landsat']
    snow, ts = DATA_DIRS['snow_data'], DATA_DIRS['plot_timeseries']

    gdf = gpd.read_file(gmt)
    gdf.index = gdf[index_col]

    covered, excluded = list(p_dct['fields'].keys()), []

    unprocessed = [i for i in gdf.index if i not in list(p_dct['fields'].keys())]

    while unprocessed:

        targets = []
        for i in gdf.index:

            if str(i) in covered or str(i) in excluded:
                continue

            try:
                if len(targets) < chunk_sz:
                    targets.append(str(i))
                else:
                    break

            except IndexError:
                break

        join_daily_timeseries(gmt, met, lst, snow, ts, overwrite=True,
                              start_date=start_date, end_date=end_date, **{'target_fields': targets})

        prepped_targets, excluded_targets = prep_fields_json(DATA_DIRS['props'], ts, DATA_DIRS['prepped_input'],
                                                             targets, irr_data=DATA_DIRS['cuttings'])

        excluded += excluded_targets

        remove = [PEST_DATA['p_dir'], os.path.join(project_ws, 'obs'), PEST_DATA['m_dir'], PEST_DATA['w_dir']]

        try:
            os.chdir(project_ws)
            [print('rmtree: {}'.format(rmdir)) for rmdir in remove]
            [shutil.rmtree(rmdir) for rmdir in remove]
        except FileNotFoundError:
            pass

        [os.mkdir(mkdir) for mkdir in remove[1:]]

        preproc(prepped_targets, ts, project_ws)

        # noinspection PyTypedDict
        dct_ = get_pest_builder_args(project_ws, DATA_DIRS['prepped_input'], ts)
        dct_.update({'python_script': PEST_DATA['python_script']})
        build_pest(project_ws, PEST_DATA['p_dir'], **dct_)

        pst_f = os.path.join(PEST_DATA['p_dir'], '{}.pst'.format(project))

        build_localizer(pst_f, ag_json=DATA_DIRS['prepped_input'])
        write_control_settings(pst_f, 3, realizations)

        try:
            run_pst(PEST_DATA['p_dir'], PEST_DATA['exe_'], PEST_DATA['_pst'],
                    num_workers=n_workers, worker_root=PEST_DATA['w_dir'],
                    master_dir=PEST_DATA['m_dir'], verbose=False, cleanup=True)
        except Exception:
            time.sleep(3)
            run_pst(PEST_DATA['p_dir'], PEST_DATA['exe_'], PEST_DATA['_pst'],
                    num_workers=n_workers, worker_root=PEST_DATA['w_dir'],
                    master_dir=PEST_DATA['m_dir'], verbose=False, cleanup=False)

        params = os.path.join(PEST_DATA['m_dir'], '{}.{}.par.csv'.format(project, iterations))
        pdf = pd.read_csv(params, index_col=0).mean(axis=0)
        p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) for s in list(pdf.index)]
        pdf.index = p_str

        for t in prepped_targets:
            p_dct['fields'][t] = e = {}
            for p in pars:
                key = '{}_{}'.format(p, t)
                e[p] = pdf.at[key]

        with open(project_tracker, 'w') as fp:
            json.dump(p_dct, fp, indent=4)

        print('Write {} to \n{}'.format(targets, os.path.basename(project_tracker)))

        covered += prepped_targets

        unprocessed = [i for i in gdf.index if i not in list(p_dct['fields'].keys())]


if __name__ == '__main__':

    p_tracker = os.path.join(data, '{}_params.json'.format(project))

    run_pest_sequence(p_tracker, 6, index_col='FID', chunk_sz=50, realizations=100, iterations=3,
                      start_date='2012-01-01', end_date='2021-12-31')
# ========================= EOF ====================================================================
