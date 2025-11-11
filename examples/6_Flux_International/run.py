import os
import time

import pandas as pd

from examples.calibrate_by_station import run_pest_sequence
from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.prep import get_flux_sites



project = '6_Flux_International'

root = '/data/ssd2/swim'
data = os.path.join(root, project, 'data')
project_ws = os.path.join(root, project)
config_file = os.path.join(project_ws, 'config.toml')

prepped_input = os.path.join(data, 'prepped_input.json')

# ICOS 200m station buffer shapefile index
FEATURE_ID = 'sid'

# flux sites
shapefile_path = os.path.join(data, 'gis', '6_Flux_International_150mBuf.shp')
sites = get_flux_sites(shapefile_path, index_col=FEATURE_ID)

results = os.path.join(project_ws, 'results', 'tight')


def run_flux_sites(fid, config, plot_data, outfile):
    start_time = time.time()

    df_dct = obs_field_cycle.field_day_loop(config, plot_data, debug_flag=True)

    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    df = df_dct[fid].copy()
    in_df = plot_data.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)

    df = df.loc[config.start_dt:config.end_dt]

    df.to_csv(outfile)


def run_calibration():

    run_pest_sequence(config_file, project_ws, workers=10, target='ptjpl', members=None,
                      realizations=20, select_stations=sites, pdc_remove=True, overwrite=True)

if __name__ == '__main__':
    run_calibration()
    pass
# ========================= EOF ====================================================================
