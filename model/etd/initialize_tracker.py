"""initialize_crop_cycle.py
Defines InitializeCropCycle class
Called by crop_cycle.py
"""

import numpy as np

de_initial = 10.0

FOO_FMT = ['et_act',
           'etref',
           'kc_act',
           'kc_bas',
           'ks',
           'ke',
           'ppt',
           'depl_root',
           'depl_surface',
           'irrigation',
           'dperc',
           'fc',
           'few',
           'zr',
           'aw3',
           'p_rz',
           'p_eft',
           'niwr',
           'runoff',
           'season',
           'cutting',
           'et_bas',
           ]

TRACKER_PARAMS = ['taw',
                  'taw3',
                  'albedo',
                  'min_albedo',
                  'melt',
                  'rain',
                  'snow_fall',
                  'daw3',
                  'daw3_prev',
                  'aw3',
                  'cn2',
                  'cgdd',
                  'aw',
                  'cgdd_penalty',
                  'cum_evap',
                  'ad',
                  'cum_evap_prev',
                  'depl_ze',
                  'dperc',
                  'dperc_ze',
                  'density',
                  'depl_root',
                  'depl_root_prev',
                  'soil_water',
                  'soil_water_prev',
                  'etc_act',
                  'etc_pot',
                  'etc_bas',
                  'etref_30',
                  'fc',
                  'grow_root',
                  'height_min',
                  'height_max',
                  'height',
                  'irr_sim',
                  'kc_act',
                  'kc_pot',
                  'kc_max',
                  'kc_min',
                  'kc_bas',
                  'kc_bas_mid',
                  'kc_bas_prev',
                  'ke',
                  'kr',
                  'ks',
                  'mad',
                  'max_irr_rate',
                  'next_day_irr',
                  'niwr',
                  'p_rz',
                  'p_eft',
                  'ppt_inf',
                  'ppt_inf_prev',
                  'rew',
                  'tew',
                  's',
                  's1',
                  's2',
                  's3',
                  's4',
                  'sro',
                  'zr_min',
                  'zr_max',
                  'z',
                  'invoke_stress',
                  'doy_start_cycle',
                  'cutting',
                  'cycle',
                  'height_initial',
                  'height_max',
                  'height_min',
                  'height',
                  'totwatin_ze',
                  'cgdd_at_planting',
                  'wt_irr',
                  'kc_bas_prev',
                  'irr_min']


class PlotTracker:
    def __init__(self, size):
        """Initialize for crops cycle"""

        self.size = size

        self.cover_proxy = None
        self.crop_df = None

        self.aw = 0.
        self.taw = 0.
        self.taw3 = 0.
        self.aw3 = 0.
        self.cn2 = 0.
        self.cgdd = 0.
        self.aw = 0
        self.cgdd_penalty = 0.
        self.cum_evap = 0.
        self.ad = 0.
        self.cum_evap_prev = 0.
        self.depl_ze = 0.
        self.daw3 = 0.0
        self.daw3_prev = 0.
        self.depl_root = 0.
        self.depl_root_prev = 0.
        self.soil_water = 0.
        self.soil_water_prev = 0.
        self.dperc = 0.
        self.dperc_ze = 0.
        self.density = 0.
        self.depl_surface = 0.
        self.etc_act = 0.
        self.etc_pot = 0.
        self.etc_bas = 0.
        self.etref_30 = 0.  # thirty day mean ETref  ' added 12/2007
        self.fc = 0.
        self.grow_root = 0.
        self.height = 0
        self.height_min = 0.1
        self.height_max = 2.0
        self.irr_sim = 0.
        self.kc_act = 0.
        self.kc_pot = 0
        self.kc_max = 0.
        self.kc_min = 0.
        self.kc_bas = 0.
        self.kc_bas_mid = 0.
        self.kc_bas_prev = 0.
        self.ke = 0.
        self.kr = 0.
        self.ks = 0.
        self.ksat = 0.

        self.ksat_hourly = None

        self.irr_continue = False
        self.next_day_irr = 0.

        self.min_albedo = 0.45
        self.albedo = self.min_albedo

        self.melt = 0.
        self.rain = 0.
        self.snow_fall = 0.

        self.mad = 0.
        self.max_irr_rate = 25.4

        self.niwr = 0.
        self.p_rz = 0.
        self.p_eft = 0.
        self.ppt_inf = 0.
        self.ppt_inf_prev = 0.
        self.rew = 0.
        self.tew = 0.
        self.s = 0.
        self.s1 = 0.
        self.s2 = 0.
        self.s3 = 0.
        self.s4 = 0.
        self.sro = 0.
        self.swe = 0.
        self.z = 0.
        self.isnan = []

        self.zr_min = 0.1
        self.zr_max = 1.7

        # TODO add invoke stess as a tunable parameter
        self.invoke_stress = 0.9

        # CGM - I don't remember why these are grouped separately
        # Maybe because they are "flags"

        self.doy_start_cycle = 0
        self.cutting = 0
        self.cycle = 1
        self.real_start = False
        self.obs_flag = True
        self.irr_flag = False
        self.in_season = False  # false if outside season, true if inside
        self.dormant_setup_flag = False
        self.crop_setup_flag = True  # flag to setup crop parameter information

        # dgketchum hacks to remove crop-type dependence
        self.height_initial = 0.1
        self.height_max = 2.0
        self.height_min = 0.1
        self.height = self.height_initial

        # TP - Looks like its value comes from compute_crop_et(),
        # but needed for setup_dormant() below...

        self.totwatin_ze = 0.

        # CGM - These are not pre-initialized in the 32-bit VB code

        self.cgdd_at_planting = 0.
        self.wt_irr = 0.

        # CGM - Initialized to 0 in latest VB code

        self.kc_bas_prev = 0.

        # TP - Added

        self.max_lines_in_crop_curve_table = 34

        # TP - Minimum net depth of application for germination irrigation, etc.

        self.irr_min = 10.

        # convert all numerical attributes to numpy arrays, shape (1, -1)
        for p in TRACKER_PARAMS:
            try:
                v = self.__getattribute__(p)
                self.__setattr__(p, np.ones((1, size)) * v)
            except AttributeError as e:
                print(p, e)

    def load_soils(self, plots):
        """Assign characteristics for crop from crop Arrays
        Parameters
        ---------
        plots : dict
            configuration data from INI file

        et_cell :

        crop :


        Returns
        -------
        None

        Notes
        -----
        Called by crop_cycle.crop_cycle() just before time loop

        """

        self.zr = self.zr_min  # initialize rooting depth at beginning of time
        self.height = self.height_min

        # Available water in soil
        # This will be optimized
        # TODO: get this from props
        # self.aw = et_cell.props['stn_whc'] / 12 * 1000.  # in/ft to mm/m

        fields = plots.input['order']

        self.aw = np.array([plots.input['props'][f]['awc'] for f in fields]).reshape(1, -1) * 1000.

        self.ksat = np.array([plots.input['props'][f]['ksat'] for f in fields]).reshape(1, -1)
        # micrometer/sec to mm/day
        self.ksat = self.ksat * 0.001 * 86400.
        self.ksat_hourly = np.ones((24, self.ksat.shape[1])) * self.ksat / 24.

        # Estimate readily evaporable water and total evaporable water from WHC
        # REW is from regression of REW vs. AW from FAO-56 soils table
        # R.Allen, August 2006, R2=0.92, n = 9

        self.rew = 0.8 + 54.4 * self.aw / 1000  # REW is in mm and AW is in mm/m

        self.tew = -3.7 + 166 * self.aw / 1000  # TEW is in mm and AW is in mm/m

        condition = self.rew > 0.8 * self.tew
        self.rew = np.where(condition, 0.8 * self.tew, self.rew)  # limit REW based on TEW

        self.daw3 = np.zeros_like(self.aw)
        self.depl_root = self.aw * self.zr

    def setup_dataframe(self, targets):
        """Initialize output dataframe

        Attributes
        ----------
        et_cell :


        Returns
        -------

        Notes
        -----

        """

        self.crop_df = {target: {} for target in targets}

    def set_kc_max(self):
        self.kc_max = 1.05
