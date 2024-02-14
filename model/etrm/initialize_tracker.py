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

TRACKER_PARAMS = [
                  'aw',
                  'depl_ze',
                  'depl_zep',
                  'dperc',
                  'dperc_ze',
                  'depl_surface',
                  'depl_root',
                  'etc_act',
                  'etc_pot',
                  'etc_bas',
                  'fc',
                  'fw',
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
                  'ke_irr',
                  'ke_ppt',
                  'kr',
                  'ks',
                  'mad',
                  'niwr',
                  'p_rz',
                  'p_eft',
                  'ppt_inf',
                  'ppt_inf_prev',
                  'rew',
                  'tew',
                  'runoff',
                  'zr_min',
                  'zr_max',
                  'zr',
                  'height_initial',
                  'height_max',
                  'height_min',
                  'height',
                  'totwatin_ze',
                  'kc_bas_prev',
                  'irr_min']


class PlotTracker:
    def __init__(self, size):
        """Initialize for crops cycle"""

        self.size = size

        self.cover_proxy = None
        self.crop_df = None
        self.ksat_hourly = None

        self.aw = 0.0
        self.depl_ze = 0.0
        self.depl_zep = 0.0
        self.dperc = 0.0
        self.dperc_ze = 0.0
        self.depl_surface = 0.0
        self.depl_root = 0.0
        self.etc_act = 0.0
        self.etc_pot = 0.0
        self.etc_bas = 0.0
        self.fc = 0.0
        self.fw = 0.0
        self.grow_root = 0.0
        self.height_min = 0.0
        self.height_max = 0.0
        self.height = 0.0
        self.irr_sim = 0.0
        self.kc_act = 0.0
        self.kc_pot = 0.0
        self.kc_max = 1.05
        self.kc_min = 0.0
        self.kc_bas = 0.0
        self.kc_bas_mid = 0.0
        self.kc_bas_prev = 0.0
        self.ke = 0.0
        self.ke_irr = 0.0
        self.ke_ppt = 0.0
        self.kr = 0.0
        self.ks = 0.0
        self.ksat = 0.0
        self.mad = 0.0
        self.niwr = 0.0
        self.p_rz = 0.0
        self.p_eft = 0.0
        self.ppt_inf = 0.0
        self.ppt_inf_prev = 0.0
        self.rew = 0.0
        self.tew = 0.0
        self.runoff = 0.0
        self.zr_min = 0.0
        self.zr_max = 0.0
        self.zr = 0.0
        self.height_initial = 0.0
        self.height_max = 0.0
        self.height_min = 0.0
        self.height = 0.0
        self.totwatin_ze = 0.0
        self.kc_bas_prev = 0.0
        self.irr_mi = 0.0

        # TP - Minimum net depth of application for germination irrigation, etc.

        self.irr_min = 10.

        # convert all numerical attributes to numpy arrays, shape (1, -1)
        for p in TRACKER_PARAMS:
            v = self.__getattribute__(p)
            self.__setattr__(p, np.ones((1, size)) * v)

    def load_soils(self, plots):

        self.zr = self.zr_min  # initialize rooting depth at beginning of time
        self.height = self.height_min

        fields = plots.input['order']

        self.aw = np.array([plots.input['props'][f]['awc'] for f in fields]).reshape(1, -1)

        self.ksat = np.array([plots.input['props'][f]['ksat'] for f in fields]).reshape(1, -1)
        # micrometer/sec to mm/day
        self.ksat = self.ksat * 0.001 * 86400.
        self.ksat_hourly = np.ones((24, self.ksat.shape[1])) * self.ksat / 24.

        self.fw_irr = self.fw_std  # fw changed to fw_irr 8/10/06

        # Reinitialize zr, but actCount for additions of DP into reserve (zrmax - zr) for rainfed

        # Convert current moisture content below Zr at end of season to AW for new crop
        # (into starting moisture content of layer 3).  This is required if zr_min != zr_dormant
        # Calc total water currently in layer 3

        # AW3 is mm/m and daw3 is mm in layer 3 (in case Zr<zr_max)

        zr_dormant = 0.1

        daw3 = self.aw3 * (self.zr_max - zr_dormant)

        # Layer 3 is soil depth between current rootzone (or dormant rootdepth) and max root for crop
        # AW3 is set to 0 first time throught for crop.

        # Potential water in root zone below zr_dormant

        taw3 = self.aw * (self.zr_max - zr_dormant)

        # Make sure that AW3 has been collecting DP from zr_dormant layer during winter

        # if daw3 < 0.0: daw3 = 0.
        daw3 = np.where(daw3 < 0.0,
                        np.zeros_like(daw3),
                        daw3)

        # if taw3 < 0.0: taw3 = 0.
        taw3 = np.where(taw3 < 0.0,
                        np.zeros_like(taw3),
                        taw3)

        # Was, until 5/9/07:
        # Assume moisture right above zr_dormant is same as below
        # depl_root = depl_root - (taw3 - daw3) * (zr_dormant - zr_min) / (zr_max - zr_min)
        # Following added 5/9/07
        # Enlarge depth of water

        self.depl_root = np.where(self.zr_min > zr_dormant,
                                  self.depl_root + (taw3 - daw3) * (self.zr_min - zr_dormant) / (
                                      self.zr_max - zr_dormant),
                                  self.depl_root)

        daw3 = np.where(self.zr_max > self.zr_min,
                        daw3 + (zr_dormant - self.zr_min) / zr_dormant * (self.aw * zr_dormant - self.depl_root),
                        daw3)

        self.depl_root = np.where(self.zr_max > self.zr_min,
                                  self.depl_root * self.zr_min / zr_dormant,
                                  self.depl_root)

        self.aw3 = np.where(self.zr_max > self.zr_min,
                            daw3 / (self.zr_max - self.zr_min),
                            self.aw3)

        self.aw3 = np.where(self.aw3 < 0.0,
                            np.zeros_like(self.aw3),
                            self.aw3)

        self.aw3 = np.where(self.aw3 > self.aw,
                            self.aw,
                            self.aw3)

        self.depl_root = np.where(self.depl_root < 0.,
                                  np.zeros_like(self.depl_root),
                                  self.depl_root)

    def setup_dataframe(self):

        self.crop_df = {}
