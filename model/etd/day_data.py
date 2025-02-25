import os

import numpy as np
import pandas as pd

hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]


class DayData:

    def __init__(self):

        self.sdays = 0
        self.doy_prev = 0

        self.doy = None
        self.dt_string = None
        self.year = None
        self.month = None
        self.day = None

        self.irr_status = None
        self.irr_day = None
        self.irr_doys = None

        self.gwsub_status = None

        self.capture = None
        self.ndvi = None
        self.refet = None
        self.min_temp = None
        self.max_temp = None
        self.temp_avg = None
        self.srad = None
        self.precip = None
        self.hr_precip = None

    def update_day(self, step_dt, size, doy):
        self.sdays += 1
        self.doy = doy
        self.dt_string = step_dt
        dt = pd.to_datetime(step_dt)

        self.year = dt.year
        self.month = dt.month
        self.day = dt.day

        self.ndvi = np.zeros((1, size))
        self.capture = np.zeros((1, size))
        self.refet = np.zeros((1, size))
        self.irr_day = np.zeros((1, size), dtype=int)

    def update_annual_irrigation(self, plots):
        self.irr_status = np.zeros((1, len(plots.input['order'])))
        self.irr_doys = []

        for i, fid in enumerate(plots.input['order']):
            try:
                irrigated = plots.input['irr_data'][fid][str(self.year)]['irrigated']
                self.irr_doys.append(plots.input['irr_data'][fid][str(self.year)]['irr_doys'])
                self.irr_status[0, i] = irrigated
            except KeyError:
                self.irr_status[0, i] = 0
                self.irr_doys.append([])

    def update_annual_groundwater_subsidy(self, plots):
        self.gwsub_status = np.zeros((1, len(plots.input['order'])))

        for i, fid in enumerate(plots.input['order']):
            try:
                gw_sub = plots.input['gwsub_data'][fid][str(self.year)]['f_sub']
                if gw_sub > 0.2:
                    self.gwsub_status[0, i] = 1
            except KeyError:
                self.gwsub_status[0, i] = 0

    def update_daily_irrigation(self, plots, vals, config):
        for i, fid in enumerate(plots.input['order']):
            irrigated = self.irr_status[0, i]
            if irrigated:
                self.ndvi[0, i] = vals['ndvi_irr'][i]
                self.capture[0, i] = vals['etf_irr_ct'][i]
                self.refet[0, i] = vals['{}_mm'.format(config.refet_type)][i]
                self.irr_day[0, i] = int(self.doy in self.irr_doys[i])

            else:
                self.ndvi[0, i] = vals['ndvi_inv_irr'][i]
                self.capture[0, i] = vals['etf_inv_irr_ct'][i]
                self.refet[0, i] = vals['{}_mm_uncorr'.format(config.refet_type)][i]
                self.irr_day[0, i] = 0

    def update_daily_inputs(self, vals, size):
        self.ndvi = self.ndvi.reshape(1, -1)
        self.capture = self.capture.reshape(1, -1)
        self.refet = self.refet.reshape(1, -1)

        self.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        self.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        self.temp_avg = (self.min_temp + self.max_temp) / 2.
        self.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        self.precip = np.array(vals['prcp_mm'])

        hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
        if np.any(self.precip > 0.):
            self.hr_precip = hr_ppt

        else:
            self.hr_precip = np.zeros_like(hr_ppt)

        self.precip = self.precip.reshape(1, -1)

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
