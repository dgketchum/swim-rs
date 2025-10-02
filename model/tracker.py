import json
import os.path

import numpy as np
import pandas as pd

from model import TRACKER_PARAMS

de_initial = 10.0

TUNABLE_PARAMS = ['aw', 'ndvi_k', 'ndvi_0', 'mad', 'swe_alpha', 'swe_beta', 'ks_alpha', 'kr_alpha']

# params not included here (e.g., 'tew') are taken from soils data
TUNABLE_DEFAULTS = {'aw': 177.56,
                    # 'kc_max': 1.00,
                    'kr_alpha': 0.25,
                    'ks_alpha': 0.15,
                    'mad': 0.59,
                    'ndvi_0': 0.41,
                    'ndvi_k': 4.99,
                    'rew': 2.93,
                    'swe_alpha': 0.48,
                    'swe_beta': 1.31,
                    'tew': 15.24,
                    # 'zr_adj': 1.0,
                    }


class SampleTracker:
    """Holds field-scale state and parameters for the SWB simulation.

    All numerical attributes are shaped to 1xN numpy arrays for vectorized
    updates over fields. Methods support loading spinup states, applying
    calibrated/initial parameters, and exporting debug dataframes.
    """

    def __init__(self, config, plots, size):

        self.plots = plots
        self.conf = config

        self.perennial = None
        self.zr = None
        self.size = size

        self.cover_proxy = None
        self.crop_df = None

        self.aw = 0.
        self.taw = 0.
        self.taw3 = 0.
        self.aw3 = 0.
        self.aw = 0
        self.cum_evap = 0.
        self.cum_evap_prev = 0.
        self.depl_ze = 0.
        self.daw3 = 0.0
        self.daw3_prev = 0.
        self.depl_root = 0.
        self.depl_root_prev = 0.
        self.soil_water = 0.
        self.soil_water_prev = 0.
        self.dperc = 0.
        self.depl_surface = 0.
        self.etc_act = 0.
        self.fc = 0.
        self.gw_sub = 0
        self.irr_sim = 0.
        self.kc_act = 0.
        self.kc_max = 1.25
        self.kc_min = 0.
        self.kc_bas = 0.
        self.kc_bas_mid = 0.
        self.ke = 0.
        self.kr = 0.
        self.ks = 0.
        self.ks_prev = 0.0
        self.kr_prev = 1.0
        self.ke_max = 1.0
        self.kc_max = 1.0
        self.ksat = 0.

        self.ksat_hourly = None

        self.irr_continue = False
        self.gw_sim = 0
        self.next_day_irr = 0.

        self.min_albedo = 0.45
        self.albedo = self.min_albedo

        self.melt = 0.
        self.rain = 0.
        self.snow_fall = 0.

        self.mad = 0.

        self.p_rz = 0.
        self.p_eft = 0.
        self.ppt_inf = 0.
        self.ppt_inf_prev = 0.
        self.rew = 3.0
        self.tew = 18.0
        self.sro = 0.
        self.swe = 0.
        self.s = 0.
        self.s1 = 0.
        self.s2 = 0.
        self.s3 = 0.
        self.s4 = 0.

        self.isnan = []

        self.zr = 0.1
        self.zr_mult = 1.0
        self.zr_adj = 1.0
        self.zr_min = 0.1
        self.zr_max = 1.7

        self.irr_flag = False
        self.totwatin_ze = 0.

        self.wt_irr = 0.
        self.niwr = 0.
        # TODO: apply this according to irrigation type
        self.max_irr_rate = 25.4
        # TODO use irr_min in application routine
        self.irr_min = 10.

        # convert all numerical attributes to numpy arrays, shape (1, -1)
        for p in TRACKER_PARAMS:
            try:
                v = self.__getattribute__(p)
                self.__setattr__(p, np.ones((1, size)) * v)

            except AttributeError as e:
                print(p, e)

    def load_root_depth(self):
        """Initialize root depth parameters from per-field properties.

        Uses landcover codes to distinguish perennial vs crop behavior, sets
        `zr_max`, `zr_min`, current `zr`, and `perennial` flags. Root depths are
        adjusted by `zr_adj` and per-field `zr_mult` from properties.
        """

        fields = self.plots.input['order']

        codes = [self.plots.input['props'][f]['lulc_code'] for f in fields]
        crops = [12, 14]
        perennials = [c for c in range(1, 18) if c not in crops]

        # depends on both the root depth and code from modis, see prep.__init__
        rz_depth = [self.plots.input['props'][f]['root_depth'] for f in fields]

        zr_mult = [self.plots.input['props'][f]['zr_mult'] for f in fields]
        zr_mult = np.array(zr_mult)

        rz_depth = self.zr_adj * zr_mult * np.array(rz_depth)

        self.zr = np.array([rz if cd in perennials else self.zr_min[0, 0]
                            for rz, cd in zip(rz_depth, codes)]).reshape(1, -1)

        self.zr_max = np.array([rz for rz in rz_depth]).reshape(1, -1)
        self.zr_min = np.array([rz if cd in perennials else self.zr_min[0, 0]
                                for rz, cd in zip(rz_depth, codes)]).reshape(1, -1)

        self.perennial = np.array([1 if cd in perennials else 0 for cd in codes]).reshape(1, -1)

    def load_soils(self):
        """Initialize soil parameters and derived rates from properties.

        Sets available water (`aw`) when not in calibrate/forecast modes, the
        saturated hydraulic conductivity (`ksat`) in mm/day and hourly form,
        and initializes `daw3` and initial root-zone depletion.
        """

        fields = self.plots.input['order']

        if self.conf.calibrate or self.conf.forecast:
            pass
        else:
            self.aw = np.array([self.plots.input['props'][f]['awc'] for f in fields]).reshape(1, -1) * 1000.

        self.ksat = np.array([self.plots.input['props'][f]['ksat'] for f in fields]).reshape(1, -1)
        self.ksat = self.ksat * 0.001 * 86400.
        self.ksat_hourly = np.ones((24, self.ksat.shape[1])) * self.ksat / 24.

        self.daw3 = np.zeros_like(self.aw)
        self.depl_root = self.aw * self.zr * 0.2

    def setup_dataframe(self, targets):
        """Prepare the debug data container keyed by field ID.

        Parameters
        - targets: list[str] of field IDs in simulation order.
        """

        self.crop_df = {target: {} for target in targets}

    def apply_parameters(self, params=None):
        """Apply tunable parameters from calibration, forecast, or defaults.

        In calibration mode, reads multipliers from `config.calibration_files`.
        In forecast mode, uses `config.forecast_parameters` (means or sets).
        Otherwise, assigns `TUNABLE_DEFAULTS` uniformly across fields.

        Parameters
        - params: optional dict[str, float] to override values by name during
          runtime (e.g., forward runs in PEST workers).
        """
        size = len(self.plots.input['order'])

        if self.conf.calibrate:
            print('CALIBRATION')

            cal_arr = {k: np.zeros((1, size)) for k in TUNABLE_PARAMS}

            ct = 0
            for k, f in self.conf.calibration_files.items():

                param_found = False

                while not param_found:
                    ct += 1
                    for p in TUNABLE_PARAMS:
                        if p in k:
                            group = p
                            fid = k.replace(f'{group}_', '')
                            param_found = True
                    if ct > 1000:
                        raise ValueError(f'Calibration Parameter Match Not Found\n Parameter {k} not in {f}')

                idx = self.plots.input['order'].index(fid)

                if params:
                    value = params[k]
                else:
                    v = pd.read_csv(f, index_col=None, header=0)
                    value = v.loc[0, '1']

                cal_arr[group][0, idx] = value

            for k, v in cal_arr.items():
                self.__setattr__(k, v)

        elif self.conf.forecast:
            print('FORECAST')

            param_arr = {k: np.zeros((1, size)) for k in TUNABLE_PARAMS}

            ct = 0
            for k, v in self.conf.forecast_parameters.items():

                param_found = False

                while not param_found:
                    ct += 1
                    for p in TUNABLE_PARAMS:
                        if p in k:
                            group = p
                            fid = k.replace(f'{group}_', '')
                            param_found = True
                    if ct > 1000:
                        raise ValueError(f'Forecast Parameter Match Not Found\n Parameter {k} not in {v}')

                # PEST++ has lower-cased the FIDs
                l = [x.lower() for x in self.plots.input['order']]

                if fid not in l:
                    continue

                idx = l.index(fid)

                if params:
                    value = params[k]
                else:
                    value = v

                param_arr[group][0, idx] = value

            for k, v in param_arr.items():
                self.__setattr__(k, v)

        else:
            print('USING PARAMETER DEFAULTS')

            for k, v in TUNABLE_DEFAULTS.items():
                arr = np.ones((1, size)) * v
                self.__setattr__(k, arr)

    def apply_initial_conditions(self):
        """Load spinup water balance state if configured and map to fields.

        Reads the spinup JSON keyed by field IDs and sets matching tracker
        state variables, preserving array shapes (1xN). Prints a message when
        spinup is applied.
        """
        order = self.plots.input['order']
        size = len(order)
        tracker_array = None

        if os.path.exists(self.conf.spinup):
            with open(self.conf.spinup, 'r') as fp:
                sdct = json.load(fp)

            first = True

            for fid, var_dct in sdct.items():

                if first:
                    tracker_array = {p: np.zeros((1, size)) for p in var_dct.keys()}
                    first = False

                idx = self.plots.input['order'].index(fid)

                for k, v in var_dct.items():

                    if k in TRACKER_PARAMS:
                        tracker_array[k][0, idx] = v

            print('USING SPINUP WATER BALANCE INFORMATION')

            for k, v in tracker_array.items():
                self.__setattr__(k, v)

    def update_dataframe(self, targets, day_data, step_dt):
        """Append per-field daily diagnostics into `crop_df` (debug mode).

        Records ET components, water balance terms, storage changes, and key
        meteorological inputs for later analysis/plotting.
        """
        for i, fid in enumerate(targets):
            self.crop_df[fid][step_dt] = {}
            sample_idx = 0, i
            self.crop_df[fid][step_dt]['etref'] = day_data.refet[sample_idx]

            eta_act = self.etc_act[sample_idx]
            self.crop_df[fid][step_dt]['capture'] = day_data.capture[sample_idx]
            self.crop_df[fid][step_dt]['t'] = self.t[sample_idx]
            self.crop_df[fid][step_dt]['e'] = self.e[sample_idx]
            self.crop_df[fid][step_dt]['kc_act'] = self.kc_act[sample_idx]
            self.crop_df[fid][step_dt]['ks'] = self.ks[sample_idx]
            self.crop_df[fid][step_dt]['ke'] = self.ke[sample_idx]

            # water balance components
            self.crop_df[fid][step_dt]['et_act'] = eta_act

            ppt = day_data.precip[sample_idx]
            self.crop_df[fid][step_dt]['ppt'] = ppt

            melt = self.melt[sample_idx]
            self.crop_df[fid][step_dt]['melt'] = melt
            rain = self.rain[sample_idx]
            self.crop_df[fid][step_dt]['rain'] = rain

            runoff = self.sro[sample_idx]
            self.crop_df[fid][step_dt]['runoff'] = runoff
            dperc = self.dperc[sample_idx]
            self.crop_df[fid][step_dt]['dperc'] = dperc

            depl_root = self.depl_root[sample_idx]
            self.crop_df[fid][step_dt]['depl_root'] = depl_root
            depl_root_prev = self.depl_root_prev[sample_idx]
            self.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

            daw3 = self.daw3[sample_idx]
            self.crop_df[fid][step_dt]['daw3'] = daw3
            daw3_prev = self.daw3_prev[sample_idx]
            self.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
            delta_daw3 = daw3 - daw3_prev
            self.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

            soil_water = self.soil_water[sample_idx]
            self.crop_df[fid][step_dt]['soil_water'] = soil_water
            soil_water_prev = self.soil_water_prev[sample_idx]
            self.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
            delta_soil_water = self.delta_soil_water[sample_idx]
            self.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

            depl_ze = self.depl_ze[sample_idx]
            self.crop_df[fid][step_dt]['depl_ze'] = depl_ze
            self.crop_df[fid][step_dt]['p_rz'] = self.p_rz[sample_idx]
            self.crop_df[fid][step_dt]['p_eft'] = self.p_eft[sample_idx]
            self.crop_df[fid][step_dt]['fc'] = self.fc[sample_idx]
            self.crop_df[fid][step_dt]['few'] = self.few[sample_idx]
            self.crop_df[fid][step_dt]['aw'] = self.aw[sample_idx]
            self.crop_df[fid][step_dt]['aw3'] = self.aw3[sample_idx]
            self.crop_df[fid][step_dt]['taw'] = self.taw[sample_idx]
            self.crop_df[fid][step_dt]['taw3'] = self.taw3[sample_idx]

            self.crop_df[fid][step_dt]['irrigation'] = self.irr_sim[sample_idx]
            self.crop_df[fid][step_dt]['irr_day'] = day_data.irr_day[sample_idx]

            self.crop_df[fid][step_dt]['gw_sim'] = self.gw_sim[sample_idx]

            self.crop_df[fid][step_dt]['swe'] = self.swe[sample_idx]
            self.crop_df[fid][step_dt]['snow_fall'] = self.snow_fall[sample_idx]
            self.crop_df[fid][step_dt]['tavg'] = day_data.temp_avg[sample_idx]
            self.crop_df[fid][step_dt]['tmax'] = day_data.max_temp[sample_idx]
            self.crop_df[fid][step_dt]['zr'] = self.zr[sample_idx]
            self.crop_df[fid][step_dt]['kc_bas'] = self.kc_bas[sample_idx]
            self.crop_df[fid][step_dt]['niwr'] = self.niwr[sample_idx]

            water_out = eta_act + dperc + runoff
            water_stored = soil_water - soil_water_prev
            water_in = melt + rain
            balance = water_in - water_stored - water_out

            self.crop_df[fid][step_dt]['wbal'] = balance

            if abs(balance) > 0.1 and day_data.year > 2000:
                pass
                # raise WaterBalanceError('Check November water balance')

    def set_kc_max(self):
        """Set per-field Kc,max from `plots.input['kc_max']` mapping."""
        fields = self.plots.input['order']
        self.kc_max = np.array([self.plots.input['kc_max'][f] for f in fields]).reshape(1, -1)

    def set_ke_max(self):
        """Set per-field Ke,max from `plots.input['ke_max']` mapping."""
        fields = self.plots.input['order']
        self.ke_max = np.array([self.plots.input['ke_max'][f] for f in fields]).reshape(1, -1)
