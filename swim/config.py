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

        self.forecast_parameter_groups = None
        self.forecast_parameters = None
        self.forecast = None
        self.data_folder = None

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
        self.calibration = None
        self.calibration_folder = None
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

    def read_config(self, conf, project_root, calibration_folder=None, parameter_dist_csv=None,
                    parameter_set_json=None):

        with open(conf, 'r') as f:
            config = toml.load(f)

        for section in config:
            for key in config[section]:
                if isinstance(config[section][key], str) and '{project_root}' in config[section][key]:
                    config[section][key] = config[section][key].format(project_root=project_root)

        crop_et_sec = 'FIELDS'
        calib_sec = 'CALIBRATION'
        forecast_sec = 'FORECAST'
        runspec_sec = 'RUNSPEC'

        self.kc_proxy = config[crop_et_sec]['kc_proxy']
        self.cover_proxy = config[crop_et_sec]['cover_proxy']

        self.project_ws = config[crop_et_sec]['project_folder']
        if not os.path.isdir(self.project_ws):
            raise NotADirectoryError(f'{self.project_ws} is not a directory')

        self.data_folder = config[crop_et_sec]['data_folder']

        if not os.path.exists(self.data_folder):
            raise NotADirectoryError(f'{self.data_folder} is not a directory')

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

        self.calibration = bool(config[calib_sec]['calibrate_flag'])

        if self.calibration:
            stdout = 'ON'
        else:
            stdout = 'OFF'

        print('\n')
        print('Config: {}'.format(conf))
        print('CALIBRATION {}'.format(stdout))

        if self.calibration:

            if calibration_folder:
                cf = calibration_folder
            else:
                cf = config[calib_sec]['calibration_folder']

            self.calibration_folder = cf
            initial_values_csv = config[calib_sec]['initial_values_csv']
            pdf = pd.read_csv(initial_values_csv, index_col=0)
            self.calibrated_parameters = pdf.index
            _files = list(pdf['mult_name'])
            cal_files, mult_files = set(os.listdir(self.calibration_folder)), set(_files)
            assert cal_files == mult_files

            _files = [os.path.join(self.calibration_folder, f) for f in _files]
            self.calibration_files = {k: v for k, v in zip(self.calibrated_parameters, _files)}
            self.calibration_groups = list(set(['_'.join(p.split('_')[:-1]) for p in pdf.index]))

        self.forecast = bool(config[forecast_sec]['forecast_flag'])
        if self.forecast:
            stdout = 'ON'
        else:
            stdout = 'OFF'
        print('FORECAST {}'.format(stdout))

        if self.forecast:

            if os.path.isfile(config[forecast_sec]['forecast_parameters']) and parameter_dist_csv is None:
                parameter_dist_csv = config[forecast_sec]['forecast_parameters']

            if parameter_dist_csv:
                fcst = parameter_dist_csv
                pdf = pd.read_csv(fcst, index_col=0).mean(axis=0)
                p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) for s in list(pdf.index)]
                pdf.index = p_str
                self.forecast_parameters = pdf.copy()
                self.forecast_parameter_groups = list(set(['_'.join(p.split('_')[:-1]) for p in pdf.index]))

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

            else:
                fcst = config[forecast_sec]['forecast_parameters']


if __name__ == '__main__':
    pass
