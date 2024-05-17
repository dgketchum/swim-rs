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
                  'albedo',
                  'min_albedo',
                  'melt',
                  'rain',
                  'snow_fall',
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
        self.aw3 = 0.
        self.cn2 = 0.
        self.cgdd = 0.
        self.aw = 0
        self.cgdd_penalty = 0.
        self.cum_evap = 0.
        self.ad = 0.
        self.cum_evap_prev = 0.
        self.depl_ze = 0.
        self.dperc = 0.
        self.dperc_ze = 0.
        self.density = 0.
        self.depl_surface = 0.
        self.depl_root = 0.
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

        # Reinitialize zr, but actCount for additions of DP into reserve (zrmax - zr) for rainfed

        # Convert current moisture content below Zr at end of season to AW for new crop
        # (into starting moisture content of layer 3).  This is required if zr_min != zr_dormant
        # Calc total water currently in layer 3

        # AW3 is mm/m and daw3 is mm in layer 3 (in case Zr<zr_max)

        zr_dormant = 0.1

    def setup_crop(self):
        """Initialize some variables for beginning of crop seasons

        Attributes
        ----------
        crop :


        Returns
        -------
        None

        Notes
        -----
        Called in crop_cycle if not in season and crop setup flag is true
        Called in kcb_daily for startup/greenup type 1, 2, and 3 when startup conditions are met

        """

        zr_dormant = 0.0

        # setup_crop is called from crop_cycle if is_season is false and crop_setup_flag is true
        # thus only setup 1st time for crop (not each year)
        # also called from kcb_daily each time GU/Plant date is reached, thus at growing season start

        self.height = self.height_min
        self.irr_auto = 0
        self.irr_sim = 0

        # Reinitialize zr, but actCount for additions of DP into reserve (zrmax - zr) for rainfed

        # Convert current moisture content below Zr at end of season to AW for new crop
        # (into starting moisture content of layer 3). This is required if zr_min != zr_dormant
        # Calc total water currently in layer 3

        # AW3 is mm/m and daw3 is mm in layer 3 (in case Zr<zr_max)

        self.daw3 = self.aw3 * (self.zr_max - zr_dormant)

        # Layer 3 is soil depth between current rootzone (or dormant rootdepth) and max root for crop
        # AW3 is set to 0 first time through for crop.

        # Potential water in root zone below zr_dormant

        taw3 = self.aw * (self.zr_max - zr_dormant)

        # Make sure that AW3 has been collecting DP from zr_dormant layer during winter

        # if daw3 < 0.0: daw3 = 0.
        self.daw3 = np.maximum(0, self.daw3)
        # if taw3 < 0.0: taw3 = 0.
        taw3 = np.maximum(0, taw3)

        # Create boolean masks
        mask_zr_max_greater_zr_min = self.zr_max > self.zr_min

        # Adjust depletion for extra starting root zone at plant or GU
        self.depl_root = np.where(
            self.zr_min > zr_dormant,
            self.depl_root + (taw3 - self.daw3) * (self.zr_min - zr_dormant) / (self.zr_max - zr_dormant),
            self.depl_root)

        # Enlarge depth of water for the case where zr_max > zr_min
        self.daw3 = np.where(
            self.zr_max > self.zr_min,
            self.daw3 + (zr_dormant - self.zr_min) / zr_dormant * (self.aw * zr_dormant - self.depl_root),
            self.daw3)

        # Adjust depl_root in proportion to zr_min / zr_dormant
        self.depl_root = np.where(
            self.zr_max > self.zr_min,
            self.depl_root * self.zr_min / zr_dormant,
            self.depl_root)

        # denom is layer 3 depth at start of season
        self.aw3 = np.where(
            self.zr_max > self.zr_min,
            self.daw3 / (self.zr_max - self.zr_min),
            self.aw3)

        # Ensure self.aw3 is within bounds
        self.aw3 = np.maximum(0.0, self.aw3)
        self.aw3 = np.where(self.aw3 > self.aw, self.aw, self.aw3)
        self.aw3 = np.minimum(self.aw, self.aw3)

        # Ensure depl_root is non-negative
        self.depl_root = np.maximum(0.0, self.depl_root)

        # Initialize rooting depth at beginning of time
        self.zr = self.zr_min
        self.crop_setup_flag = False

    def setup_dormant(self):
        """Start of dormant season
            a) Set up for soil water reservoir during non-growing season
                to collect soil moisture for next growing season
            b) Set for type of surface during non-growing season

        Parameters
        ---------
        et_cell :

        crop :

        Returns
        -------
        None

        Notes
        -----
        Called at termination of crop from crop_cycle()
        If in_season is false and dormant_flag is true,
        dormant_flag set at GU each year.
        It is called each year as soon as season is 0.

        """

        # Assume that 'rooting depth' for dormant surfaces is 0.1 or 0.15 m
        # This is depth that will be applied with a stress function to reduce kc_bas

        zr_dormant = 0.1  # was 0.15

        # Convert current moisture content of Zr layer
        #   (which should be at zr_max at end of season)
        #   into starting moisture content of layer 3
        # This is done at end of season

        # Calc total water currently in layer 3 (the dynamic layer below zr)
        # AW is mm/m and daw3 is mm in layer 3 (in case zr < zr_max)

        self.daw3 = self.aw3 * (self.zr_max - self.zr)

        # Add TAW - depl_root that is in root zone below zr_dormant.
        # Assume fully mixed root zone including zr_dormant part

        # Potential water in root zone

        taw_root = self.aw * (self.zr)

        # Actual water in root zone based on depl_root at end of season

        daw_root = np.maximum(taw_root - self.depl_root, 0)

        # Depth of evaporation layer (This only works when ze < zr_dormant)

        ze = 0.1

        # assuming values for daw_root, zr_dormant, self.zr, self.totwatin_ze, ze, self.fc are defined

        # Reduce daw_root by water in evap layer and rest of zr_dormant and then proportion
        aw_root = np.where(zr_dormant < self.zr, daw_root / self.zr, 0)

        # determine water in zr_dormant layer
        # combine water in ze layer (1-fc fraction) to that in balance of zr_dormant depth
        # need to mix ze and zr_dormant zones.  Assume current Zr zone of crop just ended is fully mixed.
        # totwatin_ze is water in fc fraction of Ze.
        totwatinzr_dormant = np.where(zr_dormant > ze,
                                      ((self.totwatin_ze + aw_root * (zr_dormant - ze)) * (1 - self.fc) +
                                       aw_root * zr_dormant * self.fc),
                                      ((self.totwatin_ze * (1 - (ze - zr_dormant) / ze)) * (1 - self.fc) +
                                       aw_root * zr_dormant * self.fc))

        # This requires that zr_dormant > ze.

        # Proportionate water between zr_dormant and zr
        daw_below = np.where(daw_root > totwatinzr_dormant, daw_root - totwatinzr_dormant, 0)

        # Actual water in mm/m below zr_dormant
        self.aw3 = np.where(zr_dormant < self.zr, (daw_below + self.daw3) / (self.zr_max - zr_dormant), self.aw3)

        # initialize depl_root for dormant season
        # Depletion below evaporation layer:

        # depl_root_below_Ze = (depl_root - de)  # / (zr - ze) #'mm/m
        # If depl_root_below_ze < 0 Then depl_root_below_ze = 0
        # assume fully mixed profile below Ze
        # depl_root = depl_root_below_ze * (zr_dormant - ze) / (zr - ze) + de

        self.depl_root = self.aw * zr_dormant - totwatinzr_dormant

        # set Zr for dormant season

        self.zr = zr_dormant

        # This value for zr will hold constant all dormant season.  dp from zr will be
        # used to recharge zr_max - zr zone.
        # Make sure that grow_root is not called during dormant season.

        self.irr_sim = 0
        self.dormant_setup_flag = False
        self.cutting = 0

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
