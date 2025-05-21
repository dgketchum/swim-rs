import os

import pandas as pd
import geopandas as gpd
import numpy as np
import json
from tqdm import tqdm

from prep import get_flux_sites


class SamplePlotDynamics:
    def __init__(self, plot_timeseries, properties_json, out_json_file, etf_target='ssebop',
                 irr_threshold=0.1, select=None):

        self.time_series = plot_timeseries

        self.model = etf_target
        self.properties_json = properties_json
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

            _file = os.path.join(self.time_series, f'{fid}.parquet')
            try:
                field_time_series = pd.read_parquet(_file)
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

            _file = os.path.join(self.time_series, f'{fid}.parquet')
            try:
                field_time_series = pd.read_parquet(_file)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue

            self.years = [int(y) for y in field_time_series.index.year.unique()]
            field_data = self._analyze_field_groundwater_subsidy(fid, field_time_series)
            if field_data is not None:
                self.fields['gwsub'][fid] = field_data

    def analyze_k_parameters(self):
        for fid in tqdm(self.irr.index, desc='Calculating K Parameters', total=len(self.irr.index)):
            if self.select and fid not in self.select:
                continue

            _file = os.path.join(self.time_series, f'{fid}.parquet')
            try:
                field_time_series = pd.read_parquet(_file)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue
            self._find_field_k_parameters(fid, field_time_series)

    def save_json(self):
        with open(self.out_json_file, 'w') as fp:
            json.dump(self.fields, fp, indent=4)
            print(f'wrote {self.out_json_file}')

    def _load_data(self):
        with open(self.properties_json, 'r') as fp:
            js_data = json.load(fp)
        self.irr = pd.DataFrame.from_dict(js_data).T
        try:
            self.irr.drop(columns=['LAT', 'LON'], inplace=True)
        except KeyError:
            pass

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
            print(f'{field} is all nan in {self.properties_json}')
            return None

        field_data = {}
        selectors = [f'{self.model}_etf_inv_irr', f'{self.model}_etf_irr', 'prcp_mm', 'eto_mm_uncorr', 'eto_mm']

        check = field_time_series[[f'{self.model}_etf_inv_irr', f'{self.model}_etf_irr']]
        check = check.fillna(0).sum(axis=1)
        years = check[check > 0.0].index.year.unique().to_list()

        irr_overall = np.mean([self.irr.at[field, f'irr_{yr}'] > self.irr_threshold for yr in range(2016, 2025)]).item()

        if irr_overall > 0.6:
            generally_irrigated = True
        else:
            generally_irrigated = False

        gw_ct = 0
        missing_years = [y for y in self.years if y not in years]

        for yr in years:
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
                df['eta'] = df['eto_mm_uncorr'] * df[f'{self.model}_etf_irr']
            else:
                df['eta'] = df['eto_mm'] * df[f'{self.model}_etf_inv_irr']

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
            gw_ct += 1

        if len(years) > 0 and gw_ct / len(years) > 0.5 and len(missing_years) > 0:
            mean_sub = np.array([field_data[yr]['f_sub'] for yr in years]).mean()
            months = []
            months = list(set([months.extend(field_data[yr]['months']) for yr in years]))
            mean_ppt = np.array([field_data[yr]['ppt'] for yr in years]).mean()
            mean_eta = np.array([field_data[yr]['eta'] for yr in years]).mean()
            for y in missing_years:
                field_data[y] = {'subsidized': 1,
                                 'f_sub': mean_sub,
                                 'f_irr': 0.0,
                                 'ratio': mean_ppt / mean_eta,
                                 'months': months,
                                 'ppt': mean_ppt,
                                 'eta': mean_eta}

        return field_data

    def _analyze_field_irrigation(self, field, field_time_series, lookback, backfill_irr=True):
        if field not in self.irr.index:
            print(f'{field} not in index')
            return None

        if np.all(np.isnan(self.irr.loc[field])):
            print(f'{field} is all nan in {self.properties_json}')
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

        etf_columns = [col for col in [f'{self.model}_etf_inv_irr', f'{self.model}_etf_irr']
                       if col in field_time_series.columns]
        ndvi_columns = [col for col in ['ndvi_inv_irr', 'ndvi_irr'] if col in field_time_series.columns]
        if not etf_columns or not ndvi_columns:
            raise ValueError('Remote sensing parameters not found')

        all_etf = field_time_series[etf_columns].values.flatten()
        all_ndvi = field_time_series[ndvi_columns].values.flatten()

        nan_mask = np.isnan(all_etf)
        all_etf = all_etf[~nan_mask]
        sub_ndvi = all_ndvi[~nan_mask]

        ke_max_mask = sub_ndvi < 0.3
        if np.any(ke_max_mask):
            ke_max = np.nanpercentile(all_etf[ke_max_mask], 90)
        else:
            ke_max = 1.0
            print(f'Warning: No NDVI values below 0.3 for {fid}. Setting ke_max=1.0')

        try:
            kc_max = np.percentile(all_etf, 90)
        except IndexError:
            kc_max = 1.25

        self.fields['ke_max'][fid] = float(ke_max)
        self.fields['kc_max'][fid] = float(kc_max)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
