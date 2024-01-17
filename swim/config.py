import os
import logging
import configparser
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
        self.cover_proxy = None
        self.field_cuttings = None
        self.refet_type = None
        self.soils = None
        self.field_index = None
        self.calibration = None
        self.calibration_folder = None
        self.calibrated_parameters = None
        self.calibration_files = None
        self.parameter_values = None
        self.field_type = field_type
        self.kc_proxy = None
        self.project_ws = None
        self.ts_quantity = None
        self.start_dt = None
        self.end_dt = None
        self.sensing_folder = None
        self.cell_crops_path = None
        self.elev_units = None
        self.field_properties = None
        self.input_timeseries = None
        self.fields_path = None
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

    def read_config(self, ini_path, debug_flag=False):
        logging.info('  INI: {}'.format(os.path.basename(ini_path)))

        # Check that INI file can be read
        config = configparser.RawConfigParser()
        config.read_file(open(ini_path))

        crop_et_sec = 'CROP_ET'
        calib_sec = 'CALIBRATION'
        # runspec_sec = 'RUNSPEC'

        self.kc_proxy = config.get(crop_et_sec, 'kc_proxy')
        self.cover_proxy = config.get(crop_et_sec, 'cover_proxy')

        self.project_ws = config.get(crop_et_sec, 'project_folder')
        self.field_index = 'FID'

        assert os.path.isdir(self.project_ws)

        self.ts_quantity = int(1)

        sdt = config.get(crop_et_sec, 'start_date')
        self.start_dt = pd.to_datetime(sdt)
        edt = config.get(crop_et_sec, 'end_date')
        self.end_dt = pd.to_datetime(edt)

        # elevation units
        self.elev_units = config.get(crop_et_sec, 'elev_units')
        assert self.elev_units == 'm'

        self.refet_type = config.get(crop_et_sec, 'refet_type')

        # et cells properties
        self.soils = config.get(crop_et_sec, 'soils')

        self.fields_path = config.get(crop_et_sec, 'fields_path')
        self.field_properties = config.get(crop_et_sec, 'field_properties')
        self.input_timeseries = config.get(crop_et_sec, 'input_timeseries')
        self.irrigation_data = config.get(crop_et_sec, 'irrigation_data')

        self.calibration = bool(config.get(calib_sec, 'calibrate_flag'))

        if self.calibration:
            cf = config.get(calib_sec, 'calibration_folder')
            self.calibration_folder = cf
            self.calibrated_parameters = config.get(calib_sec, 'calibrated_parameters').split(',')
            _files = sorted([os.path.join(cf, f) for f in os.listdir(cf)])
            self.calibration_files = {k: v for k, v in zip(self.calibrated_parameters, _files)}

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
