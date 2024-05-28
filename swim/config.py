import os

import pandas as pd
import toml


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

    def read_config(self, conf, calibration_folder=None, parameter_dist_csv=None):

        with open(conf, 'r') as f:
            config = toml.load(f)

        crop_et_sec = 'FIELDS'
        calib_sec = 'CALIBRATION'
        forecast_sec = 'FORECAST'
        runspec_sec = 'RUNSPEC'

        self.kc_proxy = config[crop_et_sec]['kc_proxy']
        self.cover_proxy = config[crop_et_sec]['cover_proxy']

        self.project_ws = config[crop_et_sec]['project_folder']
        assert os.path.isdir(self.project_ws)

        self.data_folder = config[crop_et_sec]['data_folder']
        if not os.path.exists(self.data_folder):
            self.data_folder = config[crop_et_sec]['alt_data_folder']
        assert os.path.exists(self.data_folder)

        self.field_index = config[crop_et_sec]['field_index']

        self.ts_quantity = int(1)

        sdt = config[crop_et_sec]['start_date']
        self.start_dt = pd.to_datetime(sdt)
        edt = config[crop_et_sec]['end_date']
        self.end_dt = pd.to_datetime(edt)

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

            if calibration_folder:
                cf = calibration_folder
            else:
                cf = config[calib_sec]['calibration_folder']

            self.calibration_folder = cf
            initial_values_csv = config[calib_sec]['initial_values_csv']
            pdf = pd.read_csv(initial_values_csv, index_col=0)
            self.calibrated_parameters = pdf.index
            _files = list(pdf['mult_name'])
            assert set(os.listdir(self.calibration_folder)) == set(_files)
            _files = [os.path.join(self.calibration_folder, f) for f in _files]
            self.calibration_files = {k: v for k, v in zip(self.calibrated_parameters, _files)}
            self.calibration_groups = list(set(['_'.join(p.split('_')[:-1]) for p in pdf.index]))

        self.forecast = bool(config[forecast_sec]['forecast_flag'])

        if self.forecast:

            if parameter_dist_csv:
                fcst = parameter_dist_csv
            else:
                fcst = config[forecast_sec]['forecast_parameters']

            pdf = pd.read_csv(fcst, index_col=0).mean(axis=0)
            p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) for s in list(pdf.index)]
            pdf.index = p_str
            self.forecast_parameters = pdf.copy()
            self.forecast_parameter_groups = list(set(['_'.join(p.split('_')[:-1]) for p in pdf.index]))

        # TODO: remove these ETRM-specific config attributes

        self.static_keys = ('plant_height', 'rew', 'root_z', 'soil_ksat', 'taw', 'tew')

        self.initial_keys = 'de', 'dr', 'drew'

        self.date_range = (self.start_dt, self.end_dt)
        self.swb_mode = config.get(runspec_sec, 'swb_mode')
        self.rew_ceff = config.get(runspec_sec, 'rew_ceff')
        self.evap_ceff = config.get(runspec_sec, 'evap_ceff')
        self.winter_evap_limiter = config.get(runspec_sec, 'winter_evap_limiter')
        self.winter_end_day = config.get(runspec_sec, 'winter_end_day')
        self.winter_start_day = config.get(runspec_sec, 'winter_start_day')
