import os
import json

import pandas as pd
import toml
from pprint import pprint


class ProjectConfig:
    """Data container

    Attributes
    ----------

    """

    def __init__(self):
        super().__init__()

        self.parameter_list = None
        self.spinup = None
        self.forecast_parameter_groups = None
        self.forecast_parameters = None
        self.forecast = None

        self.project_name = None
        self.data_folder = None
        self.obs_folder = None
        self.input_json = None
        self.plot_timeseries = None
        self.shapefile = None

        self.kc_proxy = None
        self.irr_threshold = None
        self.cover_proxy = None
        self.project_ws = None
        self.field_index = None
        self.ts_quantity = None
        self.start_dt = None
        self.end_dt = None
        self.elev_units = None
        self.refet_type = None
        self.input_data = None
        self.calibrate = None
        self.calibration_dir = None
        self.calibrated_parameters = None
        self.calibration_files = None
        self.calibration_groups = None

        self.static_keys = None
        self.initial_keys = None
        self.date_range = None
        self.use_individual_kcb = None
        self.ro_reinf_frac = None
        self.swb_mode = None
        self.rew_ceff = None
        self.evap_ceff = None
        self.winter_evap_limiter = None
        self.winter_end_day = None
        self.winter_start_day = None

    def read_config(self, conf, project_root, calibrate=False, forecast=False, calibration_dir=None,
                    parameter_set_json=None, forecast_param_csv=None):

        print(f'Config File: {conf}')
        with open(conf, 'r') as f:
            config = toml.load(f)

        for section in config:
            for key in config[section]:
                if isinstance(config[section][key], str) and '{project_root}' in config[section][key]:
                    config[section][key] = config[section][key].format(project_root=project_root)

        crop_et_sec = 'FIELDS'
        calib_sec = 'CALIBRATION'
        forecast_sec = 'FORECAST'

        self.calibrate = calibrate
        self.calibration_dir = calibration_dir
        self.forecast = forecast

        self.project_name = config[crop_et_sec]['project_name']
        self.shapefile = config[crop_et_sec]['shapefile']

        self.kc_proxy = config[crop_et_sec]['kc_proxy']
        self.cover_proxy = config[crop_et_sec]['cover_proxy']

        self.project_ws = config[crop_et_sec]['project_folder']
        if not os.path.isdir(self.project_ws):
            raise NotADirectoryError(f'{self.project_ws} is not a directory')

        self.data_folder = config[crop_et_sec]['data_folder']

        if not os.path.exists(self.data_folder):
            raise NotADirectoryError(f'{self.data_folder} is not a directory')

        # this folder will be built if not exists
        self.obs_folder = config[crop_et_sec]['obs_folder']

        self.plot_timeseries = config[crop_et_sec]['plot_timeseries']

        self.field_index = config[crop_et_sec]['field_index']

        self.ts_quantity = int(1)

        sdt = config[crop_et_sec]['start_date']
        self.start_dt = pd.to_datetime(sdt)
        edt = config[crop_et_sec]['end_date']
        self.end_dt = pd.to_datetime(edt)

        self.irr_threshold = config[crop_et_sec]['irr_threshold']

        # elevation units
        self.elev_units = config[crop_et_sec]['elev_units']
        assert self.elev_units == 'm'

        self.refet_type = config[crop_et_sec]['refet_type']

        # et cells properties
        # TODO: get ksat for runoff generation
        # self.soils = config.get(crop_et_sec, 'soils')

        self.input_data = os.path.join(self.data_folder, config[crop_et_sec]['input_data'])

        self.spinup = config[crop_et_sec]['spinup_data']

        if self.calibrate is True or self.calibration_dir is not None:

            if not self.calibrate:
                self.calibrate = True

            if not self.calibration_dir:
                self.calibration_dir = config[calib_sec]['calibration_dir']

            if not os.path.isdir(self.calibration_dir):
                raise ValueError(f'Calibration is on but calibration folder {self.calibration_dir} does not exist')

            initial_values_csv = config[calib_sec]['initial_values_csv']

            param_init = pd.read_csv(initial_values_csv, index_col=0)
            self.calibrated_parameters = param_init.index
            _files = list(param_init['mult_name'])
            cal_files, mult_files = set(os.listdir(self.calibration_dir)), set(_files)
            if cal_files != mult_files:
                print(f"File mismatch between directory '{self.calibration_dir}' and CSV '{initial_values_csv}':")
                only_in_dir = sorted(list(cal_files - mult_files))
                if only_in_dir:
                    print("\nOnly in directory:\n- " + '\n- '.join(only_in_dir))
                only_in_csv = sorted(list(mult_files - cal_files))
                if only_in_csv:
                    print("\nOnly in CSV:\n- " + '\n- '.join(only_in_csv))
                raise KeyError("Directory file list does not match CSV file list.")

            _files = [os.path.join(self.calibration_dir, f) for f in _files]
            self.calibration_files = {k: v for k, v in zip(self.calibrated_parameters, _files)}

        elif self.forecast:

            self.calibrate = 0
            self.calibration_dir = None

            if forecast_param_csv:
                parameter_dist_csv = forecast_param_csv
            else:
                parameter_dist_csv = config[forecast_sec]['forecast_parameters']

            if not os.path.isfile(parameter_dist_csv):
                raise ValueError(f'Forecast is on but forecast csv {parameter_dist_csv} does not exist')

            if parameter_dist_csv:
                fcst = parameter_dist_csv
                param_dist = pd.read_csv(fcst, index_col=0)
                param_mean = param_dist.mean(axis=0)

                p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) for s in list(param_mean.index)]
                param_mean.index = p_str
                self.forecast_parameters = param_mean.copy()
                self.parameter_list = param_mean.index.to_list()

            elif parameter_set_json:
                with open(parameter_set_json, 'r') as f:
                    param_arr = json.load(f)
                d = param_arr['fields']
                self.forecast_parameter_groups = [list(v.keys()) for k, v in d.items()][0]
                k = [['{}_{}'.format(s, key) for s, p in vals.items()] for key, vals in d.items()]
                k = [item for sublist in k for item in sublist]
                tup_ = [('_'.join(i.split('_')[:-1]), i.split('_')[-1]) for i in k]
                v = [d[t[1]][t[0]] for t in tup_]
                self.forecast_parameters = pd.Series(index=k, data=v)
                self.parameter_list = self.forecast_parameters.index.to_list()


if __name__ == '__main__':
    pass
