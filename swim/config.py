import os
import logging
import toml
from datetime import datetime

import pandas as pd

from model.etrm import STATIC_KEYS, INITIAL_KEYS


class ProjectConfig:
    """Data container

    Attributes
    ----------

    """

    def __init__(self, field_type='irrigated'):
        super().__init__()

        self.field_type = field_type

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

    def read_config(self, conf):

        with open(conf, 'r') as f:
            config = toml.load(f)

        crop_et_sec = 'FIELDS'
        calib_sec = 'CALIBRATION'

        self.kc_proxy = config[crop_et_sec]['kc_proxy']
        self.cover_proxy = config[crop_et_sec]['cover_proxy']

        self.project_ws = config[crop_et_sec]['project_folder']
        self.field_index = config[crop_et_sec]['field_index']

        assert os.path.isdir(self.project_ws)

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

        self.input_data = config[crop_et_sec]['input_data']

        self.calibration = bool(config[calib_sec]['calibrate_flag'])

        if self.calibration:
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


        # TODO: remove these ETRM-specific config attributes

        # self.static_keys = ('plant_height', 'rew', 'root_z', 'soil_ksat', 'taw', 'tew')
        #
        # self.initial_keys = 'de', 'dr', 'drew'
        #
        # dtr = config.get(runspec_sec, 'date_range').split(',')
        # fmt = '%Y-%m-%d'
        # s, e = datetime.strptime(dtr[0], fmt), datetime.strptime(dtr[1], fmt)
        # self.date_range = (s, e)
        # self.use_individual_kcb = bool(config.get(runspec_sec, 'use_individual_kcb'))
        # self.ro_reinf_frac = float(config.get(runspec_sec, 'ro_reinf_frac'))
        # self.swb_mode = config.get(runspec_sec, 'swb_mode')
        # self.rew_ceff = config.get(runspec_sec, 'rew_ceff')
        # self.evap_ceff = config.get(runspec_sec, 'evap_ceff')
        # self.winter_evap_limiter = config.get(runspec_sec, 'winter_evap_limiter')
        # self.winter_end_day = config.get(runspec_sec, 'winter_end_day')
        # self.winter_start_day = config.get(runspec_sec, 'winter_start_day')

    @property
    def initial_pairs(self):
        try:
            return tuple((k, getattr(self, k)) for k in INITIAL_KEYS)
        except AttributeError:
            pass

    @property
    def static_pairs(self):
        try:
            return tuple((k, getattr(self, k)) for k in STATIC_KEYS)
        except AttributeError:
            pass
