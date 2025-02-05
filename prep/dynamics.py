import os

import pandas as pd
import numpy as np
import json
from tqdm import tqdm


class SamplePlotDynamics:
    def __init__(self, plot_timeseries, irr_csv_file, out_json_file, irr_threshold=0.1, select=None):
        self.time_series = plot_timeseries
        self.irr_csv_file = irr_csv_file
        self.out_json_file = out_json_file
        self.irr_threshold = irr_threshold
        self.select = select

        self.irr = None
        self.years = None
        self.fields = {'irr': {}, 'gwsub': {}}

        self._load_data()

    def analyze_irrigation(self):

        for fid in tqdm(self.irr.index, desc='Analyzing Irrigation', total=len(self.irr.index)):
            if self.select and fid not in self.select:
                continue

            _file = os.path.join(self.time_series, f'{fid}_daily.csv')
            try:
                field_time_series = pd.read_csv(_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue

            field_data = self._analyze_field_irrigation(fid, field_time_series)
            if field_data is not None:
                self.fields['irr'][fid] = field_data

    def analyze_groundwater_subsidy(self):

        for fid in tqdm(self.irr.index, desc='Analyzing Groundwater Subsidy', total=len(self.irr.index)):
            if self.select and fid not in self.select:
                continue

            _file = os.path.join(self.time_series, f'{fid}_daily.csv')
            try:
                field_time_series = pd.read_csv(_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue

            field_data = self._analyze_field_groundwater_subsidy(fid, field_time_series)
            if field_data is not None:
                self.fields['gwsub'][fid] = field_data

    def save_json(self):
        with open(self.out_json_file, 'w') as fp:
            print(f'wrote {self.out_json_file}')
            json.dump(self.fields, fp, indent=4)

    def _load_data(self):
        self.irr = pd.read_csv(self.irr_csv_file, index_col=0)
        self.irr.drop(columns=['LAT', 'LON'], inplace=True)
        self.years = list(sorted([int(c.split('_')[-1]) for c in self.irr.columns]))

        try:
            _ = float(self.irr.index[0])
            self.irr.index = [str(i) for i in self.irr.index]
        except (TypeError, ValueError):
            pass

    def _analyze_field_groundwater_subsidy(self, field, field_time_series):
        if field not in self.irr.index:
            print(f'{field} not in index')
            return None

        if np.all(np.isnan(self.irr.loc[field])):
            print(f'{field} is all nan in {self.irr_csv_file}')
            return None

        field_data = {}
        selectors = ['etf_inv_irr', 'etf_irr', 'prcp_mm', 'eto_mm_uncorr', 'eto_mm']

        for yr in self.years:
            if yr > 2022:
                continue

            try:
                f_irr = self.irr.at[field, f'irr_{yr}']
            except (ValueError, KeyError):
                f_irr = np.nan

            irrigated = f_irr > self.irr_threshold

            df = field_time_series.loc[f'{yr}-01-01': f'{yr}-12-31', selectors]
            df['doy'] = [i.dayofyear for i in df.index]

            if irrigated:
                df['eta'] = df['eto_mm'] * df['etf_irr']
            else:
                df['eta'] = df['eto_mm_uncorr'] * df['etf_inv_irr']

            eta, ppt = df['eta'].sum(), df['prcp_mm'].sum()
            ratio = eta / ppt

            if irrigated:
                field_data[yr] = {'subsidized': 0,
                                  'f_sub': 0,
                                  'f_irr': f_irr,
                                  'ratio': ratio,
                                  'months': [],
                                  'ppt': ppt,
                                  'eta': eta}
                continue

            mdf = df.resample('ME').sum()
            months = mdf[mdf['eta'] > mdf['prcp_mm']].index.month.to_list()

            if ratio > 1:
                subsidized = 1
                f_sub = (ratio - 1) / ratio
            else:
                subsidized = 0
                f_sub = 0

            field_data[yr] = {'subsidized': subsidized,
                              'f_sub': f_sub,
                              'f_irr': f_irr,
                              'ratio': eta / ppt,
                              'months': months,
                              'ppt': ppt,
                              'eta': eta}

        return field_data

    def _analyze_field_irrigation(self, field, field_time_series):
        if field not in self.irr.index:
            print(f'{field} not in index')
            return None

        if np.all(np.isnan(self.irr.loc[field])):
            print(f'{field} is all nan in {self.irr_csv_file}')
            return None

        field_data = {}
        fallow = []

        selector = 'ndvi_irr'

        for yr in self.years:
            if yr > 2022:
                continue

            irr_doys, periods = [], 0

            try:
                f_irr = self.irr.at[field, f'irr_{yr}']
            except (ValueError, KeyError):
                f_irr = np.nan

            irrigated = f_irr > self.irr_threshold

            if not irrigated:
                fallow.append(yr)
                field_data[yr] = {'irr_doys': irr_doys,
                                  'irrigated': int(irrigated),
                                  'f_irr': f_irr}
                continue

            df = field_time_series.loc[f'{yr}-01-01': f'{yr}-12-31', [selector]]
            df['doy'] = [i.dayofyear for i in df.index]
            df[selector] = df[selector].rolling(window=10, center=True).mean()

            df['diff'] = df[selector].diff()

            nan_ct = np.count_nonzero(np.isnan(df[selector].values))
            if nan_ct > 200:
                fallow.append(yr)
                continue

            local_min_indices = df[(df['diff'] > 0) & (df['diff'].shift(1) < 0)].index

            positive_slope = (df['diff'] > 0)
            groups = (positive_slope != positive_slope.shift()).cumsum()
            df['groups'] = groups
            group_counts = positive_slope.groupby(groups).sum()
            long_positive_slope_groups = group_counts[group_counts >= 10].index

            for group in long_positive_slope_groups:
                group_indices = positive_slope[groups == group].index
                start_index = group_indices[0]
                end_index = group_indices[-1]

                if start_index in local_min_indices:
                    start_doy = (start_index - pd.Timedelta(days=5)).dayofyear
                    end_doy = (end_index + pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1
                else:
                    start_doy = start_index.dayofyear
                    end_doy = (end_index + pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1

            irr_doys = sorted(list(set(irr_doys)))

            field_data[yr] = {'irr_doys': irr_doys,
                              'irrigated': int(irrigated),
                              'f_irr': f_irr}

        field_data['fallow_years'] = fallow
        return field_data


if __name__ == '__main__':
    root = '/home/dgketchum/PycharmProjects/swim-rs'

    project = 'alarc_test'

    data = os.path.join(root, 'tutorials', project, 'data')
    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')

    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    landsat = os.path.join(data, 'landsat')
    joined_timeseries = os.path.join(data, 'plot_timeseries')

    FEATURE_ID = 'field_1'
    cuttings_json = os.path.join(landsat, 'calibration_dynamics.json')

    dynamics = SamplePlotDynamics(joined_timeseries, irr, irr_threshold=0.3,
                                  out_json_file=cuttings_json, select=['ALARC2_Smith6'])
    dynamics.analyze_groundwater_subsidy()
    dynamics.analyze_irrigation()
    dynamics.save_json()

# ========================= EOF ====================================================================
