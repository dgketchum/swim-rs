import os

import pandas as pd
import geopandas as gpd
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
        self.years = None

        self.irr = None
        self.fields = {'irr': {}, 'gwsub': {}, 'ke_max': {}, 'kc_max': {}}

        self._load_data()

    def analyze_irrigation(self, lookback=10):

        for fid in tqdm(self.irr.index, desc='Analyzing Irrigation', total=len(self.irr.index)):
            if self.select and fid not in self.select:
                continue

            _file = os.path.join(self.time_series, f'{fid}_daily.csv')
            try:
                field_time_series = pd.read_csv(_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue

            self.years = [int(y) for y in field_time_series['year'].unique()]
            field_data = self._analyze_field_irrigation(fid, field_time_series, lookback)
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

            self.years = [int(y) for y in field_time_series['year'].unique()]
            field_data = self._analyze_field_groundwater_subsidy(fid, field_time_series)
            if field_data is not None:
                self.fields['gwsub'][fid] = field_data

    def analyze_k_parameters(self):
        for fid in tqdm(self.irr.index, desc='Calculating K Parameters', total=len(self.irr.index)):
            if self.select and fid not in self.select:
                continue

            _file = os.path.join(self.time_series, f'{fid}_daily.csv')
            try:
                field_time_series = pd.read_csv(_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue
            self._find_field_k_parameters(fid, field_time_series)

    def save_json(self):
        with open(self.out_json_file, 'w') as fp:
            json.dump(self.fields, fp, indent=4)
            print(f'wrote {self.out_json_file}')

    def _load_data(self):
        self.irr = pd.read_csv(self.irr_csv_file, index_col=0)
        self.irr.drop(columns=['LAT', 'LON'], inplace=True)

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

        irr_overall = np.mean([self.irr.at[field, f'irr_{yr}'] > self.irr_threshold for yr in range(2016, 2025)]).item()

        if irr_overall > 0.6:
            generally_irrigated = True
        else:
            generally_irrigated = False

        for yr in self.years:
            if yr > 2024:
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
            ratio = eta / (ppt + 1.0)

            if irrigated or generally_irrigated:
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

    def _analyze_field_irrigation(self, field, field_time_series, lookback, backfill_irr=True):
        if field not in self.irr.index:
            print(f'{field} not in index')
            return None

        if np.all(np.isnan(self.irr.loc[field])):
            print(f'{field} is all nan in {self.irr_csv_file}')
            return None

        field_data = {}
        fallow = []
        if backfill_irr:
            irr_fill = []
        else:
            irr_fill = None

        selector = 'ndvi_irr'

        for yr in self.years:
            if yr > 2024:
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

            if df.empty:
                print(f'{field} in {yr} is empty')
                return None

            df['doy'] = [i.dayofyear for i in df.index]
            df[selector] = df[selector].rolling(window=32, center=True).mean()

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
                    start_doy = (start_index - pd.Timedelta(days=lookback)).dayofyear
                else:
                    start_doy = start_index.dayofyear

                end_doy = (end_index + pd.Timedelta(days=2))
                if df.loc[end_doy - pd.Timedelta(days=1)][selector] > 0.3:
                    ndvi_doy = df.loc[end_doy - pd.Timedelta(days=1)][selector]
                    while ndvi_doy > 0.3:
                        end_doy += pd.Timedelta(days=1)
                        ndvi_doy = df.loc[end_doy - pd.Timedelta(days=1)][selector]

                elif df.loc[end_doy - pd.Timedelta(days=1)][selector] < 0.5:
                    continue

                end_doy = (end_doy + pd.Timedelta(days=1)).dayofyear
                irr_doys.extend(range(start_doy, end_doy))
                periods += 1

            irr_doys = sorted(list(set(irr_doys)))

            if len(irr_doys) == 0:
                if backfill_irr:
                    irr_fill.append(yr)
                print(f'Warning {field} is irrigated in {yr} but has no irrigated days')

            field_data[yr] = {'irr_doys': irr_doys,
                              'irrigated': int(irrigated),
                              'f_irr': f_irr}

        for yr in irr_fill:
            candidates = [y for y in field_data if 'f_irr' in field_data[y] and field_data[y]['f_irr'] > 0]
            if not candidates:
                continue

            diffs = [abs(yr - y) for y in candidates]
            min_idx = diffs.index(min(diffs))
            best_match = candidates[min_idx]
            field_data[yr]['irr_doys'] = field_data[best_match]['irr_doys']

        field_data['fallow_years'] = fallow
        return field_data

    def _find_field_k_parameters(self, fid, field_time_series):

        etf_columns = [col for col in ['etf_inv_irr', 'etf_irr'] if col in field_time_series.columns]
        ndvi_columns = [col for col in ['ndvi_inv_irr', 'ndvi_irr'] if col in field_time_series.columns]
        if not etf_columns or not ndvi_columns:
            return

        all_etf = field_time_series[etf_columns].values.flatten()
        all_ndvi = field_time_series[ndvi_columns].values.flatten()

        nan_mask = np.isnan(all_etf) | np.isnan(all_ndvi)
        all_etf = all_etf[~nan_mask]
        all_ndvi = all_ndvi[~nan_mask]

        ke_max_mask = all_ndvi < 0.3
        if np.any(ke_max_mask):
            ke_max = np.percentile(all_etf[ke_max_mask], 90)
        else:
            ke_max = 1.0
            print(f'Warning: No NDVI values below 0.3 for {fid}. Setting ke_max=1.0')

        kc_max = np.percentile(all_etf, 90)

        self.fields['ke_max'][fid] = float(ke_max)
        self.fields['kc_max'][fid] = float(kc_max)


if __name__ == '__main__':
    project = '4_Flux_Network'

    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        data = os.path.join(root, 'tutorials', project, 'data')

    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')

    irr = os.path.join(data, 'properties', 'calibration_irr.csv')

    landsat = os.path.join(data, 'landsat')

    joined_timeseries = os.path.join(data, 'plot_timeseries')

    FEATURE_ID = 'field_1'

    cuttings_json = os.path.join(landsat, 'calibration_dynamics.json')

    fdf = gpd.read_file(shapefile_path)
    target_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    state_idx = [i for i, r in fdf.iterrows() if r['field_3'] in target_states]
    fdf = fdf.loc[state_idx]
    sites_ = list(set(fdf['field_1'].to_list()))
    sites_.sort()

    dynamics = SamplePlotDynamics(joined_timeseries, irr, irr_threshold=0.3,
                                  out_json_file=cuttings_json, select=sites_)
    dynamics.analyze_irrigation(lookback=5)
    dynamics.analyze_groundwater_subsidy()
    dynamics.analyze_k_parameters()
    dynamics.save_json()

# ========================= EOF ====================================================================
