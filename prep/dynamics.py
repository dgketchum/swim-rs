import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from prep.ndvi_regression import sentinel_adjust_quantile_mapping


class SamplePlotDynamics:
    def __init__(self, plot_timeseries, properties_json, out_json_file, etf_target='ssebop',
                 irr_threshold=0.1, select=None, masks=('no_mask',), instruments=('landsat',),
                 use_mask=False, use_lulc=False):

        self.time_series = plot_timeseries

        self.model = etf_target
        self.properties_json = properties_json
        self.out_json_file = out_json_file
        self.irr_threshold = irr_threshold
        self.select = select
        self.years = None
        self.masks = masks
        self.target_fid = None
        self.instruments = list(instruments)

        self.use_mask = use_mask
        self.use_lulc = use_lulc

        if not (self.use_mask or self.use_lulc):
            raise ValueError('Must use either an irrigation mask or land cover product for this module')

        self.properties = None
        self.fields = {'irr': {}, 'gwsub': {}, 'ke_max': {}, 'kc_max': {}}

        self._load_data()

    def analyze_irrigation(self, lookback=10):

        for fid, data in tqdm(self.properties.items(), desc='Analyzing Irrigation', total=len(self.properties)):

            if self.select and fid not in self.select:
                continue

            self.target_fid = fid

            _file = os.path.join(self.time_series, f'{fid}.parquet')
            try:
                field_time_series = pd.read_parquet(_file)
            except FileNotFoundError:
                print(f'{_file} not found, skipping')
                continue

            self.years = [int(y) for y in field_time_series.index.year.unique()]
            field_data = self._analyze_field_irrigation(fid, field_time_series, lookback)
            if field_data is not None:
                self.fields['irr'][fid] = field_data

    def analyze_groundwater_subsidy(self):

        for fid, data in tqdm(self.properties.items(), desc='Analyzing Groundwater Subsidy',
                              total=len(self.properties)):
            if self.select and fid not in self.select:
                continue

            self.target_fid = fid

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
        for fid, data in tqdm(self.properties.items(), desc='Calculating K Parameters', total=len(self.properties)):

            if self.select and fid not in self.select:
                continue

            self.target_fid = fid

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
            self.properties = json.load(fp)

    def _analyze_field_groundwater_subsidy(self, field, df):
        """"""
        field_data = {}
        idx = pd.IndexSlice

        check = df.loc[:, idx[:, :, ['etf'], :, self.model, :]]
        check = check.resample('YE').sum()
        if check.shape[0] > 1:
            check = check.mean(axis=1)
        check = check[check > 0.0].dropna(axis=0)
        etf_years = check.index.year.unique().to_list()

        if 'irr' in self.properties[field]:
            irr_overall = np.mean([self.properties[field]['irr'][str(yr)] > self.irr_threshold
                                   for yr in etf_years]).item()
            use_props_irr = True


        elif field in self.fields['irr']:
            irr_overall = np.mean([self.fields['irr'][field][yr]['f_irr'] > self.irr_threshold
                                   for yr in etf_years]).item()
            use_props_irr = False

        else:
            raise ValueError('There is no irrrigation information in either the properties or '
                             'previously analyzed by this class')

        if irr_overall > 0.6:
            generally_irrigated = True
        else:
            generally_irrigated = False

        gw_ct = 0
        missing_years = [y for y in self.years if y not in etf_years]

        for yr in etf_years:

            try:
                if use_props_irr:
                    f_irr = self.properties[field]['irr'][str(yr)]
                else:
                    f_irr = self.fields['irr'][field][yr]['f_irr']

            except (ValueError, KeyError):
                f_irr = np.nan

            irrigated = f_irr > self.irr_threshold

            if irrigated and 'irr' in self.masks:
                mask = 'irr'
            elif 'irr' in self.masks:
                mask = 'inv_irr'
            else:
                mask = 'no_mask'

            t_index = [i for i in df.index if i.year == yr]

            etf_ = df.loc[t_index, idx[:, :, ['etf'], :, [self.model], mask]]
            if etf_.shape[0] > 1:
                etf_ = pd.DataFrame(df.loc[t_index, idx[:, :, ['etf'], :, [self.model], mask]].mean(axis=1))

            ppt_ = df.loc[t_index, idx[:, :, ['prcp'], :, :, ['no_mask']]]
            eto_ = df.loc[t_index, idx[:, :, ['eto'], :, :, ['no_mask']]]
            ydf = pd.DataFrame(data=np.array([etf_, ppt_, eto_]).T[0], index=t_index, columns=['etf', 'ppt', 'eto'])
            ydf['etf'] = ydf['etf'].interpolate()
            ydf['etf'] = ydf['etf'].bfill().ffill()

            ydf['eta'] = ydf['etf'] * ydf['eto']

            df['doy'] = [i.dayofyear for i in df.index]

            ratio = ydf['eta'].sum() / (ydf['ppt'].sum() + 1.0)

            ppt_yr = ydf['ppt'].sum().item()
            if ppt_yr == 0.0:
                raise ValueError('Check your precip data')

            eta_yr = ydf['eta'].sum().item()

            if irrigated or generally_irrigated:
                field_data[yr] = {'subsidized': 0,
                                  'f_sub': 0,
                                  'f_irr': f_irr,
                                  'ratio': ratio,
                                  'months': [],
                                  'ppt': ppt_yr,
                                  'eta': eta_yr}
                continue

            mdf = ydf.resample('ME').sum()
            months = mdf[mdf['eta'] > mdf['ppt']].index.month.to_list()

            if ratio > 1:
                subsidized = 1
                f_sub = (ratio - 1) / ratio
            else:
                subsidized = 0
                f_sub = 0

            field_data[yr] = {'subsidized': subsidized,
                              'f_sub': f_sub,
                              'f_irr': f_irr,
                              'ratio': eta_yr / ppt_yr,
                              'months': months,
                              'ppt': ppt_yr,
                              'eta': eta_yr}
            if f_sub > 0.1:
                gw_ct += 1

        if len(etf_years) > 0 and gw_ct / len(etf_years) > 0.5 and len(missing_years) > 0:
            mean_sub = np.array([field_data[yr]['f_sub'] for yr in etf_years]).mean()
            months = [field_data[yr]['months'] for yr in etf_years]
            months = list(set([y for sl in months for y in sl]))
            mean_ppt = np.array([field_data[yr]['ppt'] for yr in etf_years]).mean()
            mean_eta = np.array([field_data[yr]['eta'] for yr in etf_years]).mean()
            for y in missing_years:
                field_data[y] = {'subsidized': 1,
                                 'f_sub': mean_sub,
                                 'f_irr': 0.0,
                                 'ratio': mean_ppt / mean_eta,
                                 'months': months,
                                 'ppt': mean_ppt,
                                 'eta': mean_eta}

        return field_data

    def _analyze_field_irrigation(self, field, df, lookback, backfill_irr=True):
        idx = pd.IndexSlice

        field_data = {}
        fallow = []

        if backfill_irr:
            irr_fill = []
        else:
            irr_fill = None

        check = df.loc[:, idx[:, :, ['etf'], :, self.model, :]]
        check = check.resample('YE').sum()
        if check.shape[1] > 1:
            # multiple instruments
            if len(check.columns.levels[1]) > 1:
                check = check.loc[:, idx[:, ['landsat'], ['etf'], :, [self.model], :]]

        check = check[check > 0.0].dropna(axis=0)
        etf_years = check.index.year.unique().to_list()

        lulc_code = self.properties[field]['lulc_code']
        glc10_code = self.properties[field].get('glc10_lc', None)

        if glc10_code is not None:
            cropped = (lulc_code in [12, 13, 14]) | (glc10_code == 10)
        else:
            cropped = lulc_code in [12, 13, 14]

        for yr in self.years:

            if yr == self.years[0]:
                extended_years = [yr, yr + 1]
            elif yr == self.years[-1]:
                extended_years = [yr - 1, yr]
            else:
                extended_years = [yr - 1, yr, yr + 1]

            if yr not in etf_years and self.use_lulc and backfill_irr:
                irr_fill.append(yr)

            t_index = [i for i in df.index if i.year == yr]
            ext_index = [i for i in df.index if i.year in extended_years]

            ppt_ = df.loc[ext_index, idx[:, :, ['prcp'], :, :, ['no_mask']]]
            eto_ = df.loc[ext_index, idx[:, :, ['eto'], :, :, ['no_mask']]]

            etf_ = df.loc[ext_index, idx[:, :, ['etf'], :, [self.model], :]]
            if etf_.shape[1] > 1:
                # mutliple mask options
                if len(etf_.columns.levels[5]) > 1 and etf_.shape[1] > 1:
                    etf_ = df.loc[ext_index, idx[:, :, ['etf'], :, [self.model], 'irr']]

                # multiple instruments
                if len(etf_.columns.levels[1]) > 1 and etf_.shape[1] > 1:
                    etf_ = etf_.loc[ext_index, idx[:, self.instruments,
                                               ['etf'], :, [self.model], :]].mean(axis=1).values.reshape((-1, 1))

            ydf = pd.DataFrame(data=np.array([etf_, ppt_, eto_]).T[0], index=ext_index,
                               columns=['etf', 'ppt', 'eto'])

            ydf['etf'] = ydf['etf'].interpolate()
            ydf['etf'] = ydf['etf'].bfill().ffill()
            ydf['eta'] = ydf['etf'] * ydf['eto']

            ydf['doy'] = [i.dayofyear for i in ydf.index]

            irr_doys, periods = [], 0

            if self.use_mask:
                try:
                    f_irr = self.properties[field]['irr'][str(yr)]
                except (ValueError, KeyError):
                    f_irr = np.nan

                irrigated = f_irr > self.irr_threshold

            elif self.use_lulc:
                subsidy = ydf[['eto', 'ppt', 'etf', 'eta']].resample('ME').sum()
                subsidy_ct = (subsidy['eta'] / (subsidy['ppt'] + 1.0) > 1.3).loc[f'{yr}-01-01': f'{yr}-12-31'].sum()

                if subsidy_ct >= 3 and cropped:
                    irrigated = True
                    f_irr = 1.0

                else:
                    irrigated = False
                    f_irr = 0.0

            else:
                raise ValueError('Must choose between "use_lulc" and "use_mask" for irrigation analysis')

            if not irrigated:
                fallow.append(yr)
                field_data[yr] = {'irr_doys': irr_doys,
                                  'irrigated': int(irrigated),
                                  'f_irr': f_irr}
                continue

            if ydf.empty:
                print(f'{field} in {yr} is empty')
                return None

            ydf['doy'] = [i.dayofyear for i in ydf.index]

            if self.use_mask:
                mask = 'irr'
            else:
                mask = 'no_mask'

            ndvi_ = df.loc[ext_index, idx[:, :, ['ndvi'], :, :, mask]]

            # too-complex check for pre- and post- year-of-interest unirrigated
            if mask == 'irr':
                for y_ in extended_years:
                    if y_ == yr:
                        continue
                    for instrument in self.instruments:
                        idx_ = [i for i in df.index if i.year == y_]
                        nd_check = df.loc[idx_, idx[:, [instrument], ['ndvi'], :, :, mask]]
                        if np.all(np.isnan(nd_check.values)):
                            inv_irr = df.loc[idx_, idx[:, [instrument], ['ndvi'], :, :, 'inv_irr']].copy()
                            ndvi_.loc[idx_, idx[:, [instrument], ['ndvi'], :, :, 'irr']] = inv_irr.values

            if ndvi_.shape[1] > 1:

                lndvi = df.loc[ext_index, idx[:, ['landsat'], ['ndvi'], :, :, mask]]
                sndvi = df.loc[ext_index, idx[:, ['sentinel'], ['ndvi'], :, :, mask]]

                if np.isnan(sndvi.loc[t_index].values).sum() == sndvi.loc[t_index].shape[0]:
                    ndvi_ = lndvi.values.flatten()

                else:
                    ndvi_alg_cols = [c[4] for c in df.columns if 'ndvi' in c[2]]
                    adj_col_name = 'quantile_adj_to_landsat'

                    if adj_col_name not in ndvi_alg_cols:
                        lndvi = df.loc[:, idx[:, ['landsat'], ['ndvi'], :, :, mask]]
                        sndvi = df.loc[:, idx[:, ['sentinel'], ['ndvi'], :, :, mask]]
                        sent_adj_ndvi = sentinel_adjust_quantile_mapping(sentinel_ndvi_df=sndvi, landsat_ndvi_df=lndvi,
                                                                         min_pairs=20, window_days=1)
                        df[field, 'sentinel', 'ndvi', 'none', adj_col_name, mask] = sent_adj_ndvi

                    sndvi = df.loc[ext_index, idx[:, ['sentinel'], ['ndvi'], :, adj_col_name, mask]].copy()
                    lndvi = df.loc[ext_index, idx[:, ['landsat'], ['ndvi'], :, :, mask]].copy()

                    ndvi_ = pd.concat([lndvi, sndvi], ignore_index=False, axis=1).mean(axis=1)

            ydf['ndvi'] = ndvi_

            ydf['ndvi'] = ydf['ndvi'].interpolate()
            ydf['ndvi'] = ydf['ndvi'].bfill().ffill()

            ydf['ndvi'] = ydf['ndvi'].rolling(window=32, center=True).mean()

            ydf['ndvi'] = ydf['ndvi'].bfill().ffill()

            ydf['diff'] = ydf['ndvi'].diff()

            nan_ct = np.count_nonzero(np.isnan(ydf['ndvi'].values))
            if nan_ct > 200:
                fallow.append(yr)
                continue

            local_min_indices = ydf[(ydf['diff'] > 0) & (ydf['diff'].shift(1) < 0)].index

            positive_slope = (ydf['diff'] > 0)
            groups = (positive_slope != positive_slope.shift()).cumsum()
            ydf['groups'] = groups
            group_counts = positive_slope.groupby(groups).sum()
            long_positive_slope_groups = group_counts[group_counts >= 10].index

            for group in long_positive_slope_groups:
                group_indices = positive_slope[groups == group].index
                start_index = group_indices[0]
                end_index = group_indices[-1]

                if start_index in local_min_indices:
                    start_day = (start_index - pd.Timedelta(days=lookback))
                else:
                    start_day = start_index

                end_day = (end_index + pd.Timedelta(days=2))

                prev_day_ndvi = ydf.loc[end_day - pd.Timedelta(days=1)]['ndvi']

                if prev_day_ndvi > 0.3:

                    ndvi_doy = ydf.loc[end_day - pd.Timedelta(days=1)]['ndvi']

                    while ndvi_doy > 0.3 and end_day in ydf.index:
                        end_day += pd.Timedelta(days=1)
                        ndvi_doy = ydf.loc[end_day - pd.Timedelta(days=1)]['ndvi']

                elif ydf.loc[end_day - pd.Timedelta(days=1)]['ndvi'] < 0.5:
                    continue

                end_day = (end_day + pd.Timedelta(days=1))
                doys = [i.dayofyear for i in pd.date_range(start_day, end_day) if i.year == yr]
                irr_doys.extend(doys)
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

    def _find_field_k_parameters(self, fid, df):

        idx = pd.IndexSlice

        check = df.loc[:, idx[:, :, ['etf'], :, self.model, :]]
        check = check.resample('YE').sum()
        check = check[check > 0.0].dropna(axis=0)
        years = check.index.year.unique().to_list()

        t_index = [i for i in df.index if i.year in years]

        etf_ = df.loc[t_index, idx[:, :, ['etf'], :, [self.model], :, :]]
        ndvi_ = df.loc[t_index, idx[:, :, ['ndvi'], :, :, :]]

        try:
            ydf = pd.DataFrame(data=np.array([etf_, ndvi_]).T[0], index=t_index,
                               columns=['etf', 'ndvi'])
        except ValueError:
            ndvi_ = df.loc[t_index, idx[:, :, ['ndvi'], :, :, 'irr']]
            ydf = pd.DataFrame(data=np.array([etf_, ndvi_]).T[0], index=t_index,
                               columns=['etf', 'ndvi'])

        ydf['etf'] = ydf['etf'].interpolate()
        ydf['etf'] = ydf['etf'].bfill().ffill()
        ydf['doy'] = [i.dayofyear for i in ydf.index]

        all_etf = ydf['etf'].values.flatten()
        all_ndvi = ydf['ndvi'].values.flatten()

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
