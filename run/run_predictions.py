import os
import json
import time

from swim.input import SamplePlots
from model.etd import obs_field_cycle
from swim.config import ProjectConfig
from prep.prep_plots import prep_fields_json
from prep.field_timeseries import join_daily_timeseries

d = '/media/research/IrrigationGIS/swim'
if not os.path.exists(d):
    d = '/home/dgketchum/data/IrrigationGIS/swim'

project = 'tongue'
data = os.path.join(d, 'examples', project)

# project_annex = 'tongue_annex'
# data_annex = os.path.join(d, 'examples', project_annex)

DATA_DIRS = {'fields_gridmet': os.path.join(data, 'gis', '{}_fields_gfid.shp'.format(project)),
             'met_data': os.path.join(data, 'met_timeseries'),
             'landsat': os.path.join(data, 'landsat', '{}_sensing.csv'.format(project)),
             'snow_data': os.path.join(data, 'snow_timeseries', 'snodas_{}.json'.format(project)),
             'input_ts': os.path.join(data, 'input_timeseries'),
             'props': os.path.join(data, 'properties', '{}_props.json'.format(project)),
             'prepped_input': os.path.join(data, 'prepped_input', '{}_input_sample.json'.format(project)),
             'cuttings': os.path.join(d, 'examples/{}/landsat/{}_cuttings.json'.format(project, project)),
             'conf': os.path.join(data, 'calibrated_models', '{}_swim.toml'.format(project)),
             'params': os.path.join(data, 'calibrated_models', '{}_params.json'.format(project)),
             'output': os.path.join(data, 'output', 'irr_output'),
             'unirr_ts': os.path.join(data, 'ts_cluster', 'median_ts.json')
             }

REQUIRED_COLUMNS = ['et_act',
                    'ppt',
                    'melt',
                    'rain',
                    'soil_water',
                    'delta_soil_water',
                    'irrigation',
                    'niwr',
                    ]


def run_fields(ini_path, parameter_set, write_files, chunk_sz=100, subset=None, force_nonirr=None,
               overwrite=False, overwrite_inputs=False):
    with open(parameter_set, 'r') as f:
        dct = json.load(f)

    all_sites = list(dct['fields'].keys())

    if subset:
        sub_ = [str(i) for i in list(range(subset[0], subset[1] + 1))]
        all_sites = [s for s in all_sites if s in sub_]

    if overwrite:
        covered, excluded = [], []
    else:
        covered, excluded = [x.split('_')[0] for x in os.listdir(write_files)], []

    unprocessed = [i for i in all_sites if i not in covered]

    while unprocessed:

        targets = []
        to_download = []
        for i in unprocessed:

            if str(i) in covered or str(i) in excluded:
                continue

            try:
                if chunk_sz:
                    if len(targets) < chunk_sz:
                        targets.append(str(i))
                    else:
                        break
                else:
                    targets.append(str(i))

                if not os.path.exists(os.path.join(DATA_DIRS['input_ts'], '{}_daily.csv'.format(i))):
                    to_download.append(str(i))

            except IndexError:
                break

        try:
            start_time = time.time()

            gmt, met, lst = DATA_DIRS['fields_gridmet'], DATA_DIRS['met_data'], DATA_DIRS['landsat']
            snow, ts = DATA_DIRS['snow_data'], DATA_DIRS['input_ts']

            if to_download:
                join_daily_timeseries(gmt, met, lst, snow, ts, overwrite=True,
                                      start_date='1989-01-01', end_date='2021-12-31', **{'target_fields': to_download})

            if overwrite_inputs:
                join_daily_timeseries(gmt, met, lst, snow, ts, overwrite=True,
                                      start_date='1989-01-01', end_date='2021-12-31', **{'target_fields': targets})

            if force_nonirr:
                unirr_loc = DATA_DIRS['unirr_ts']
            else:
                unirr_loc = None
            targets, excluded_targets = prep_fields_json(DATA_DIRS['props'], targets, ts,
                                                         DATA_DIRS['prepped_input'], irr_data=DATA_DIRS['cuttings'],
                                                         force_unirrigated=unirr_loc)

            excluded += excluded_targets

            if len(targets) == 0:
                unprocessed = [i for i in all_sites if i not in covered and i not in excluded]
                continue

            config = ProjectConfig()
            config.read_config(ini_path, parameter_set_json=parameter_set)

            fields = SamplePlots()
            fields.initialize_plot_data(config)

            df_dct = obs_field_cycle.field_day_loop(config, fields, debug_flag=True)

            end_time = time.time()
            print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

            for i, fid in enumerate(targets):
                df = df_dct[fid].copy()
                df = df[REQUIRED_COLUMNS]
                _fname = os.path.join(write_files, '{}_output.csv'.format(fid))
                df.to_csv(_fname)
                covered.append(fid)
                print(fid)

            unprocessed = [i for i in all_sites if i not in covered and i not in excluded_targets]

        except Exception as e:
            print('error', e)
            covered += targets


if __name__ == '__main__':
    run_fields(DATA_DIRS['conf'], parameter_set=DATA_DIRS['params'], write_files=DATA_DIRS['output'],
               subset=None, chunk_sz=20, overwrite=False, force_nonirr=False, overwrite_inputs=True)

# ========================= EOF ====================================================================
