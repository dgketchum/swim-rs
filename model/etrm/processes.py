import os
import shutil
import time

import numpy as np

from model.etrm import dict_setup, INITIAL_KEYS, STATIC_KEYS, initialize_tracker
from model.etrm.tools import time_it, day_generator, millimeter_to_acreft


class NotConfiguredError(BaseException):
    def __str__(self):
        return 'The model has not been configured. Processes.configure_run must be called before Processes.run'


class Processes(object):
    """
    The purpose of this class is update the etrm master dict daily.

    See function explanations.

    """

    # Config values. Default values should be specified in RunSpec not here.
    _date_range = None
    _use_individual_kcb = None
    _ro_reinf_frac = None
    _swb_mode = None
    _rew_ceff = None
    _evap_ceff = None
    _winter_evap_limiter = None
    _winter_end_day = None
    _winter_start_day = None

    _is_configured = False

    def __init__(self, cfg, **kwargs):
        # JIR
        # self.tracker = None
        self.point_tracker = None
        self._initial_depletions = None

        print('##############################')
        # Define user-controlled constants, these are constants to start with day one, replace
        # with spin-up data when multiple years are covered

        self._info('Constructing/Initializing Processes')

        self._constants = dict_setup.set_constants()

        # Initialize point and raster dicts for static values (e.g. TAW) and initial conditions (e.g. de)
        # from spin up. Define shape of domain. Create a month and annual dict for output raster variables
        # as defined in self._outputs. Don't initialize point_tracker until a time step has passed

        for k, v in kwargs.items():
            setattr(cfg, k, v)

        self._cfg = cfg

        self._static = dict_setup.initialize_static_dict(cfg)

        shape = self._static['taw'].shape
        self._master = dict_setup.initialize_master_dict(shape)

        self.initialize()

    def configure_run(self, runspec, field):
        """
        configure the model run with a RunSpec object

        :param runspec: RunSpec
        :return:
        """

        self._info('Configuring Processes')

        self._date_range = runspec.date_range
        self._use_individual_kcb = runspec.use_individual_kcb
        self._ro_reinf_frac = runspec.ro_reinf_frac
        self._swb_mode = runspec.swb_mode
        self._rew_ceff = runspec.rew_ceff
        self._evap_ceff = runspec.evap_ceff
        self._winter_evap_limiter = runspec.winter_evap_limiter
        self._winter_end_day = runspec.winter_end_day
        self._winter_start_day = runspec.winter_start_day
        print('---------- CONFIGURATION ---------------')
        for attr in ('date_range', 'use_individual_kcb',
                     'winter_evap_limiter', 'winter_end_day', 'winter_start_day',
                     'ro_reinf_frac', 'swb_mode', 'rew_ceff', 'evap_ceff'):
            print('{:<20s}{}'.format(attr, getattr(self, '_{}'.format(attr))))
        print('----------------------------------------')
        self._is_configured = True

        self.field = field

    def run(self):
        """
        run the ETRM model
        :return:
        """
        if not self._is_configured:
            raise NotConfiguredError()

        self._info('Run started. Simulation period: start={}, end={}'.format(*self._date_range))

        c = self._constants
        m = self._master
        s = self._static

        start_monsoon, end_monsoon = c['s_mon'].timetuple().tm_yday, c['e_mon'].timetuple().tm_yday
        self._info('Monsoon: {} to {}'.format(start_monsoon, end_monsoon))
        # big_counter = 0
        st = time.time()

        for counter, day in enumerate(day_generator(*self._date_range)):
            tm_yday = day.timetuple().tm_yday
            self._info('DAY:     {}({})'.format(day, tm_yday))

            time_it(self._do_daily_data_load, day)
            # modify the PRISM precipitation
            if start_monsoon <= tm_yday <= end_monsoon:
                m['precip'] = np.maximum((m['precip'] - 1.612722) / 0.676904, 0)
            else:
                m['precip'] = np.maximum((m['precip'] - 0.488870) / 0.993831, 0)

            m['inten'] = m['precip'] * 0

            # generate random number
            random_number = np.random.randn()
            # percentile = norm.cdf(random_number)

            log_precip = np.log(m['precip'][m['precip'] > 0])
            log_inten = np.zeros_like(log_precip)

            # precipitation intensity here

            m['inten'][m['precip'] > 0] = np.exp(log_inten)  # mm/min

            # Assume 2-hour storms in the monsoon season, and 6 hour storms otherwise
            # If melt is occurring (calculated in _do_snow), infiltration will be set to 24 hours
            # [mm/day] #
            m['soil_ksat'] = s['soil_ksat']
            # ksat-runoff version
            # if start_monsoon <= tm_yday <= end_monsoon:
            #     m['soil_ksat'] = s['soil_ksat'] * 2 / 24.
            # else:
            #     m['soil_ksat'] = s['soil_ksat'] * 6 / 24.

            time_it(self._do_snow, m, c)
            # time_it(self._do_soil_ksat_adjustment, m, s) # forest litter adjustment is hard to justify
            time_it(self._do_dual_crop_transpiration, tm_yday, m, s, c)
            time_it(self._do_fraction_covered, m, s, c)

            # if self._swb_mode == 'fao':
            #     time_it(self._do_fao_soil_water_balance, m, s, c)
            # elif self._swb_mode == 'vertical':
            #     time_it(self._do_vert_soil_water_balance, m, s, c)

            func = self._do_fao_soil_water_balance if self._swb_mode == 'fao' else self._do_vert_soil_water_balance
            time_it(func, m, s, c, tm_yday)

            time_it(self._do_mass_balance, day, swb=self._swb_mode)

            time_it(self._do_accumulations)

            is_first = counter == 0
            time_it(self._update_master_tracker, m, day, is_first)
            self._update_point_tracker(m, day, is_first)

        self._info('saving tabulated data')

        # self.save_tracker() #TODO check gabe merge
        self._info('Execution time: {}'.format(time.time() - st))

    def set_save_dates(self, dates):
        """
        set the individual days to write

        :param dates: list of datetimes
        :return:
        """
        self._raster_manager.set_save_dates(dates)

    def modify_master(self, alpha=1, beta=1, gamma=1, zeta=1, theta=1):
        """
        modify the master dictionary

        :param alpha: temp scalar
        :param beta: precip scalar
        :param gamma: etrs scalar
        :param zeta: kcb scalar
        :param theta: soil_ksat scalar
        :return:
        """
        m = self._master
        m['temp'] += alpha
        m['precip'] *= beta
        m['etrs'] *= gamma
        m['kcb'] *= zeta
        m['soil_ksat'] *= theta

    def modify_taw(self, taw_modification):
        """
        Gets the taw array, modifies it by a constant scalar value
        (taw_modification) and returns the resulting array

        :param taw_modification: object
        :return: taw array

        """

        s = self._static
        taw = s['taw']
        taw = taw * taw_modification
        s['taw'] = taw

        return taw

    def uniform_taw(self, taw_value):
        """Replaces the taw array with a single value

        :param taw_value: object
        :return taw_uniform array

        """
        print('===========================\nrunning uniform_taw\n===========================')
        m = self._master  # testing 6/2/17
        s = self._static
        taw = s['taw']
        taw_shape = taw.shape
        s['taw'] = np.full(taw_shape, taw_value)
        taw = s['taw']
        m['pdr'] = m['dr'] = taw

        return taw

    def get_taw(self):
        """
        Gets the TAW array and returns it

        :return: TAW array
        """

        s = self._static
        taw = s['taw']

        return taw

    def initialize(self):
        """
        initialize the models initial state

        :return:
        """
        # JIR
        initial = dict_setup.initialize_initial_conditions_dict(self._cfg)

        self._info('Initialize initial model state')
        m = self._master
        # JIR
        print('initial dr {}'.format(initial['dr']))
        # m['pdr'] = m['dr'] = self._initial['dr'] # TODO - major change here 6/2/2017
        m['pdr'] = m['dr'] = self._static['taw']  # This makes the whole state start totally dry

        # JIR
        m['pde'] = m['de'] = initial['de']
        # JIR
        m['pdrew'] = m['drew'] = initial['drew']

        s = self._static
        for key in ('rew', 'tew', 'taw', 'soil_ksat'):
            v = s[key]
            msg = '{} median: {}, mean: {}, max: {}, min: {}'.format(key, np.median(v), v.mean(), v.max(), v.min())
            self._debug(msg)

        self._initial_depletions = m['dr']  # + m['de'] + m['drew']

    def _do_snow(self, m, c):
        """ Calibrated snow model that runs using PRISM temperature and precipitation.

        :return: None
        """

        temp = m['temp']
        palb = m['albedo']

        precip = m['precip']

        a_min = c['a_min']
        a_max = c['a_max']

        # The threshold values here were 0.0 and were changed to 4.0 in revision 84238ff
        # If the threshold values are going to be manipulated then the should change to Config values
        # and be set in the configuration file
        sf = np.where(temp < 4.0, precip, 0)
        rain = np.where(temp >= 4.0, precip, 0)

        alb = np.where(sf > 3.0, a_max, palb)
        alb = np.where(sf <= 3.0, a_min + (palb - a_min) * np.exp(-0.12), alb)
        alb = np.where(sf == 0.0, a_min + (palb - a_min) * np.exp(-0.05), alb)
        alb = np.where(alb < a_min, a_min, alb)

        m['swe'] += sf

        melt = np.maximum(((1 - alb) * m['rg'] * c['snow_alpha']) + (temp - 1.8) * c['snow_beta'], 0)

        m['melt'] = melt = np.minimum(m['swe'], melt)
        m['swe'] -= melt

        m['rain'] = rain
        m['snow_fall'] = sf
        m['albedo'] = alb

    def _do_soil_ksat_adjustment(self, m, s):
        """ Adjust soil hydraulic conductivity according to land surface cover type.

        :return: None
        """

        water = m['rain'] + m['melt']
        land_cover = s['land_cover']
        soil_ksat = m['soil_ksat']

        # TO DO: Fix limits to match mm/day units
        for lc, wthres, ksat_scalar in ((41, 50.0, 2.0),
                                        (41, 12.0, 1.2),

                                        (42, 50.0, 2.0),
                                        (42, 12.0, 1.2),

                                        (43, 50.0, 2.0),
                                        (43, 12.0, 1.2)):
            soil_ksat = np.where((land_cover == lc) & (water < wthres), soil_ksat * ksat_scalar, soil_ksat)

        m['soil_ksat'] = soil_ksat

    def _do_dual_crop_transpiration(self, tm_yday, m, s, c):
        """ Calculate dual crop coefficients for transpiration only.

        """

        # basal crop coefficient - GELP
        kcb = m['kcb']
        # Ref ET -GELP
        etrs = m['etrs']
        # Root zone depletion -GELP
        pdr = m['dr']

        ####
        # transpiration:
        # ks- stress coeff- ASCE pg 226, Eq 10.6
        # TAW could be zero at lakes.
        taw = np.maximum(s['taw'], 0.001)
        ks = ((taw - pdr) / ((1 - c['p']) * taw))
        ks = np.minimum(1 + 0.001, ks)  # this +0.001 may be unneeded
        ks = np.maximum(0, ks)
        m['ks'] = ks

        # Transpiration from dual crop coefficient
        transp = np.maximum(ks * kcb * etrs, 0.0)

        # enforce winter dormancy of vegetation
        m['transp_adj'] = False
        if self._winter_end_day > tm_yday or tm_yday > self._winter_start_day:
            # super-important winter evap limiter. Jan suggested 0.03 (aka 3%) but that doesn't match ameriflux.
            # Using 30% DDC 2/20/17
            transp *= self._winter_evap_limiter
            m['transp_adj'] = True

        # limit transpiration so it doesn't exceed the amount of water available in the root zone
        transp = np.minimum(transp, (taw - pdr))
        m['transp'] = transp

    def _do_fraction_covered(self, m, s, c):
        """ Calculate covered fraction and fraction evaporating-wetted.

        """
        kcb = m['kcb']
        kc_max = np.maximum(c['kc_max'], kcb + 0.05)
        kc_min = c['kc_min']

        # Cover Fraction- ASCE pg 199, Eq 9.27
        plant_exponent = s['plant_height'] * 0.5 + 1  # h varaible, derived from ??
        numerator_term = np.maximum(kcb - kc_min, 0.01)
        denominator_term = np.maximum(kc_max - kc_min, 0.01)

        cover_fraction_unbound = (numerator_term / denominator_term) ** plant_exponent

        # ASCE pg 198, Eq 9.26
        m['fcov'] = fcov = np.maximum(np.minimum(cover_fraction_unbound, 1), 0.001)  # covered fraction of ground
        m['few'] = np.maximum(1 - fcov, 0.001)  # uncovered fraction of ground

    def _do_fao_soil_water_balance(self, m, s, c, tm_yday, ro_local_reinfilt_frac=None, rew_ceff=None, evap_ceff=None):
        """ Calculate evap and all soil water balance at each time step.

        """
        if ro_local_reinfilt_frac is None:
            ro_local_reinfilt_frac = self._ro_reinf_frac

        if rew_ceff is None:
            rew_ceff = self._rew_ceff

        if evap_ceff is None:
            evap_ceff = self._evap_ceff

        m['pdr'] = pdr = m['dr']
        m['pde'] = pde = m['de']
        m['pdrew'] = pdrew = m['drew']

        taw = np.maximum(s['taw'], 0.001)
        m['taw'] = taw  # add taw to master dict - Jul 9 2017, GELP
        tew = np.maximum(s['tew'], 0.001)  # TEW is zero at lakes in our data set
        rew = s['rew']

        kcb = m['kcb']
        kc_max = np.maximum(c['kc_max'], kcb + 0.05)
        ks = m['ks']
        etrs = m['etrs']  # comes from get_penman function

        # Start Evaporation Energy Balancing
        st_1_dur = (s['rew'] - m['pdrew']) / (c['ke_max'] * etrs)  # ASCE 194 Eq 9.22; called Fstage1
        st_1_dur = np.minimum(st_1_dur, 1.0)
        m['st_1_dur'] = st_1_dur = np.maximum(st_1_dur, 0)
        m['st_2_dur'] = st_2_dur = (1.0 - st_1_dur)

        # kr- evaporation reduction coefficient Allen 2011 Eq
        # Slightly different from 7ASCE pg 193, eq 9.21, but the Fstage coefficients come into the ke calc.
        tew_rew_diff = np.maximum(tew - rew, 0.001)
        kr = np.maximum(np.minimum((tew - m['pde']) / (tew_rew_diff), 1), 0)
        # EXPERIMENTAL: stage two evap has been too high, force slowdown with decay
        # kr *= (1 / m['dry_days'] **2)

        m['kr'] = kr

        # Simple version for 3-bucket model
        ke_init = (kc_max - (ks * kcb))
        m['ke_init'] = ke_init

        # ke evaporation efficency; Allen 2011, Eq 13a
        few = m['few']
        ke = np.minimum((st_1_dur + st_2_dur * kr) * (kc_max - (ks * kcb)), few * kc_max)
        ke = np.maximum(0.0, np.minimum(ke, 1))
        m['ke'] = ke

        # Ketchum Thesis eq 36, 37
        e1 = st_1_dur * ke_init * etrs
        m['evap_1'] = np.minimum(e1, rew - pdrew)
        e2 = st_2_dur * kr * ke_init * etrs
        m['evap_2'] = np.minimum(e2, (tew - pde) - e1)

        # Allen 2011
        evap = ke * etrs

        # limit evap so it doesn't exceed the amount of soil moisture available in the TEW
        evap = np.minimum(evap, (tew - pde))

        # limit evap so it doesn't exceed the amount of soil moisture available after transp occurs
        transp = m['transp']
        m['evap'] = evap = np.minimum(evap, (taw - pdr) - transp)

        m['eta'] = et_actual = evap + transp

        # Start Water Balance
        water = m['rain'] + m['melt']

        # if snow is melting, set ksat to 24 hour value
        m['soil_ksat'] = soil_ksat = np.where(m['melt'] > 0.0, s['soil_ksat'], m['soil_ksat'])

        # Dry days are only used if Experimental stage 2 reduction is used
        dd = m['dry_days']
        dd = np.where(water < 0.1, dd + 1, 1)
        m['dry_days'] = dd

        # Surface runoff (Hortonian- using storm duration modified ksat values)
        # ro = np.where(water > soil_ksat, water - soil_ksat, 0)
        # ro *= (1 - ro_local_reinfilt_frac)
        '''Esther's Stats'''
        num = np.random.uniform(0, 1)
        start_monsoon, end_monsoon = c['s_mon'].timetuple().tm_yday, c['e_mon'].timetuple().tm_yday
        if start_monsoon <= tm_yday <= end_monsoon:
            ro = 0.001160957 * (m['rain'] ** 2) + 0.199019984 * m['rain'] * m['inten']
            if num > 0.01392405:
                ro = np.where(m['rain'] <= 2, 0, ro)
            if num > 0.05977011:
                ro = np.where((m['rain'] <= 5) & (m['rain'] > 2), 0, ro)
            if num > 0.06521739:
                ro = np.where((m['rain'] <= 8) & (m['rain'] > 5), 0, ro)
            if num > 0.2393617:
                ro = np.where((m['rain'] <= 12) & (m['rain'] > 8), 0, ro)
            if num > 0.4554455:
                ro = np.where((m['rain'] <= 22) & (m['rain'] > 12), 0, ro)
            if num > 0.8:
                ro = np.where((m['rain'] <= 40) & (m['rain'] > 22), 0, ro)
            if num > 0.9:
                ro = np.where(m['rain'] > 40, 0, ro)
        else:
            ro = 0.0003765849 * (m['rain'] ** 2) + 0.0964337598 * m['rain'] * m['inten']
            if num > 0.001658375:
                ro = np.where(m['rain'] <= 10, 0, ro)
            if num > 0.05504587:
                ro = np.where((m['rain'] <= 20) & (m['rain'] > 10), 0, ro)
            if num > 0.1111111:
                ro = np.where((m['rain'] <= 30) & (m['rain'] > 20), 0, ro)
            if num > 0.5454545:
                ro = np.where(m['rain'] > 30, 0, ro)
        m['ro'] = ro

        # Calculate Deep Percolation (etrm or infiltration)
        m['infil'] = dp = np.maximum(water - ro - et_actual - pdr, 0)

        # Calculate depletion in TAW, full root zone
        m['dr'] = dr = np.minimum(np.maximum(pdr - (water - ro) + et_actual + dp, 0), taw)

        # Calculate depletion in TEW, full evaporative layer
        # ceff, capture efficiency, reduces depletion recovery as representation of bypass flow through macropores
        m['de'] = de = np.minimum(np.maximum(pde - ((water - ro) * evap_ceff) + evap / few, 0), tew)

        # Calculate depletion in REW, skin layer; ceff, capture efficiency, reduces depletion recovery
        m['drew'] = np.minimum(np.maximum(pdrew - ((water - ro) * rew_ceff) + evap / few, 0), rew)

        m['soil_storage'] = (pdr - dr)

        m['rzsm'] = 1 - (dr / taw)  # add root zone soil moisture (RZSM) to master dict - Jul 9, 2017 GELP

    def _fill_depletions(self, arr_1, arr_2, t, evap):

        # fill depletion in REW if possible
        drew = np.where(arr_1 >= evap, 0, evap - arr_1)
        drew = np.minimum(drew, t)

        # add excess water to the water available to TEW (Is this coding ok?)
        arr_2 = np.where(arr_1 >= evap, arr_2 + arr_1 - evap, arr_2)

        return arr_2, drew

    def _do_accumulations(self):
        """ This function simply accumulates all terms.

        :return: None
        """
        m = self._master

        for k in ('infil', 'etrs', 'eta', 'precip', 'rain', 'melt', 'ro', 'swe'):
            kk = 'tot_{}'.format(k)
            m[kk] = m[k] + m[kk]

        m['soil_storage_all'] = self._initial_depletions - (m['pdr'])  # removed m['pde'] + m['pdrew'] 6/2/17

        func = self._output_function
        ms = [func(m[k]) for k in ('infil', 'etrs', 'eta', 'precip', 'ro', 'swe', 'soil_storage')]
        print('today infil: {}, etrs: {}, eta: {}, precip: {}, ro: {}, swe: {}, stor {}'.format(*ms))

        ms = [func(m[k]) for k in ('tot_infil', 'tot_etrs', 'tot_eta', 'tot_precip', 'tot_ro', 'tot_swe')]
        print('total infil: {}, etrs: {}, eta: {}, precip: {}, ro: {}, swe: {}'.format(*ms))

    def _do_mass_balance(self, date, swb):
        """ Checks mass balance.
        :return:
        """

        m = self._master

        melt = m['melt']
        rain = m['rain']
        # ro = m['ro']
        # transp = m['transp']
        # evap = m['evap']
        # infil = m['infil']

        ddr = m['pdr'] - m['dr']
        dde = m['pde'] - m['de']
        ddrew = m['pdrew'] - m['drew']

        b = 0
        if swb == 'vertical':
            b = dde + ddrew

        a = m['ro'] + m['transp'] + m['evap'] + m['infil'] + ddr + b
        mass = rain + melt - a
        # mass = rain + melt - ro + transp + evap + infil - ddr

        # if swb == 'fao':
        #     m['mass'] = m['rain'] + m['melt'] - (m['ro'] + m['transp'] + m['evap'] + m['infil'] +
        #                              (m['pdr'] - m['dr']))
        # elif swb == 'vertical':
        #     m['mass'] = m['rain'] + m['melt'] - (m['ro'] + m['transp'] + m['evap'] + m['infil'] +
        #                                      ((m['pdr'] - m['dr']) + (m['pde'] - m['de']) +
        #                                       (m['pdrew'] - m['drew'])))
        # if swb == 'vertical':
        # mass -= (m['pde'] - m['de']) + (m['pdrew'] - m['drew'])

        m['mass'] = mass
        # print 'mass from _do_mass_balance: {}'.format(mm_af(m['mass']))
        # if date == self._date_range[0]:
        #     # print 'zero mass balance first day'
        #     m['mass'] = zeros(m['mass'].shape)
        m['tot_mass'] = tot_mass = abs(mass) + m['tot_mass']
        self._debug('total mass balance error: {}'.format(self._output_function(tot_mass)))

    def _do_daily_data_load(self, date):
        """ Find daily raster image for each ETRM input.

        param date: datetime.day object
        :return: None
        """
        m = self._master
        ds = date.strftime('%Y-%m-%d')
        d = self.field.input[ds]
        m['min_temp'] = d['tmin_c']
        m['max_temp'] = d['tmax_c']
        m['temp'] = (d['tmin_c'] + d['tmax_c']) / 2
        m['precip'] = d['prcp_mm']
        m['etrs'] = d['eto_mm']
        m['rs'] = d['rs']
        m['pkcb'] = m['kcb']
        m['kcb'] = d['ndvi_inv_irr'] * self._cfg.ndvi_beta + self._cfg.ndvi_alpha

    def _update_master_tracker(self, m, date, is_first):  # TODO check gabe merge
        def factory(k):

            v = m[k]

            if k in ('dry_days', 'kcb', 'kr', 'ks', 'ke', 'fcov', 'few', 'albedo',
                     'max_temp', 'min_temp', 'rg', 'st_1_dur', 'st_2_dur',):
                v = v.mean()
            elif k == 'transp_adj':
                v = np.median(v)
            else:
                v = self._output_function(v)
            return v

        keys = sorted(m.keys())
        values = [factory(key) for key in keys]
        path = self._get_tracker_path()
        self._write_csv('Master tracker', path, date, keys, values, is_first)

    def _write_csv(self, msg, path, date, keys, values, is_first):
        values = [str(v) for v in values]
        lines = ['{},{}'.format(date, ','.join(values))]
        if is_first:
            header = 'date,{}'.format(','.join(keys))
            print('writinsad heasder', msg, header)
            lines.insert(0, header)

        print('{} - path {}'.format(msg, path))
        with open(path, 'a') as wfile:
            for line in lines:
                wfile.write('{}\n'.format(line))

    def _output_function(self, param):
        '''determine if units are acre-feet (volume, summed over area of interest) or mm (average depth)'''
        if self._cfg.output_units == 'mm':
            param = param.mean()
        if not self._cfg.output_units == 'mm':
            param = millimeter_to_acreft(param)
        return param

    def _info(self, msg):
        print('-------------------------------------------------------')
        print(msg)
        print('-------------------------------------------------------')

    def _debug(self, msg):
        print('%%%%%%%%%%%%%%%% {}'.format(msg))


if __name__ == '__main__':
    pass

# ============= EOF =============================================
