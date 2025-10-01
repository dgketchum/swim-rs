import os
import json
import pytz
import time
from datetime import timedelta, date, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import geopandas as gpd

import pyproj
import pynldas2 as nld
from rasterstats import zonal_stats

from data_extraction.gridmet.thredds import GridMet
from prep import COLUMN_MULTIINDEX, ACCEPTED_UNITS_MAP
from swim.config import ProjectConfig

CLIMATE_COLS = {
    'etr': {
        'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_alfalfa',
        'col': 'etr'},
    'pet': {
        'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_grass',
        'col': 'eto'},
    'pr': {
        'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
        'var': 'precipitation_amount',
        'col': 'prcp'},
    'srad': {
        'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
        'var': 'daily_mean_shortwave_radiation_at_surface',
        'col': 'srad'},
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin'},
    'vs': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'u2'},
    'sph': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'q'},
}

GRIDMET_GET = ['elev',
               'tmin',
               'tmax',
               'etr',
               'etr_corr',
               'eto',
               'eto_corr',
               'prcp',
               'srad',
               'u2',
               'ea',
               ]

BASIC_REQ = ['date', 'year', 'month', 'day', 'centroid_lat', 'centroid_lon']

COLUMN_ORDER = BASIC_REQ + GRIDMET_GET


def _build_raster_list(gridmet_ras):
    rasters = []
    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, f'gridmet_corrected_{v}_{m}.tif')) for m in range(1, 13)]
    return rasters


def _compute_lat_lon_from_centroids(gdf_5071):
    centroids = gdf_5071.geometry.centroid
    wgs84 = centroids.to_crs('EPSG:4326')
    return wgs84.y.values, wgs84.x.values


def assign_gridmet_and_corrections(fields,
                                   gridmet_ras,
                                   fields_join,
                                   factors_js,
                                   gridmet_points=None,
                                   field_select=None,
                                   feature_id='FID',
                                   gridmet_id_col='GFID'):
    """Map fields to GridMET and write correction factors.

    - If `gridmet_points` provided, assigns the nearest GridMET centroid via spatial join.
    - Otherwise, assigns a unique GFID per field and samples at field centroids.

    Outputs:
    - Writes `fields_join` with GFID, LAT, LON, ELEV.
    - Writes `factors_js` JSON keyed by GFID with monthly 'etr'/'eto' factors and lat/lon.
    """
    print('Find field-gridmet joins')

    fields = gpd.read_file(fields)
    if fields.crs is None:
        fields.set_crs('EPSG:5071', inplace=True)

    rasters = _build_raster_list(gridmet_ras)

    fields_cent = fields.copy()
    fields_cent['geometry'] = fields_cent.geometry.centroid
    lat_vals, lon_vals = _compute_lat_lon_from_centroids(fields_cent)
    fields['LAT'] = lat_vals
    fields['LON'] = lon_vals

    if field_select is not None:
        mask = fields[feature_id].astype(str).isin(set(field_select))
        fields = fields.loc[mask].copy()
        fields_cent = fields_cent.loc[mask].copy()

    gridmet_targets = {}

    if gridmet_points is not None:
        pts = gpd.read_file(gridmet_points)
        if pts.crs != fields_cent.crs:
            pts = pts.to_crs(fields_cent.crs)

        keep_cols = [c for c in [gridmet_id_col, 'lat', 'lon', 'geometry'] if c in pts.columns]
        pts = pts[keep_cols]

        joined = gpd.sjoin_nearest(fields_cent[[feature_id, 'geometry']],
                                   pts,
                                   how='left',
                                   distance_col='dist')

        fields[gridmet_id_col] = joined[gridmet_id_col].values
        fields['STATION_ID'] = fields[gridmet_id_col]

        unique_ids = pd.unique(fields[gridmet_id_col].values)
        pts_indexed = pts.set_index(gridmet_id_col)

        for gfid in unique_ids:
            if pd.isna(gfid):
                continue
            gfid_int = int(gfid)
            geom = pts_indexed.loc[gfid_int, 'geometry']
            gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=fields_cent.crs)
            gridmet_targets[gfid_int] = {str(m): {} for m in range(1, 13)}
            if 'lat' in pts_indexed.columns and 'lon' in pts_indexed.columns:
                plat = float(pts_indexed.loc[gfid_int, 'lat'])
                plon = float(pts_indexed.loc[gfid_int, 'lon'])
            else:
                wgs_pt = gpd.GeoSeries([geom], crs=fields_cent.crs).to_crs('EPSG:4326')
                plat, plon = wgs_pt.iloc[0].y, wgs_pt.iloc[0].x

            gridmet_targets[gfid_int]['lat'] = plat
            gridmet_targets[gfid_int]['lon'] = plon

            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'], nodata=np.nan)[0]['mean']
                gridmet_targets[gfid_int][month].update({_var: stats})
    else:
        fields[gridmet_id_col] = range(len(fields))
        for i, field in tqdm(fields.iterrows(), desc='Assigning GridMET IDs', total=fields.shape[0]):
            gfid_int = int(fields.at[i, gridmet_id_col])
            geom = fields_cent.at[i, 'geometry']
            gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=fields_cent.crs)
            plat, plon = fields.at[i, 'LAT'], fields.at[i, 'LON']

            gridmet_targets[gfid_int] = {str(m): {} for m in range(1, 13)}
            gridmet_targets[gfid_int]['lat'] = plat
            gridmet_targets[gfid_int]['lon'] = plon
            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'], nodata=np.nan)[0]['mean']
                gridmet_targets[gfid_int][month].update({_var: stats})

    for i, field in tqdm(fields.iterrows(), desc='Fetching elevations', total=fields.shape[0]):
        g = GridMet('elev', lat=fields.at[i, 'LAT'], lon=fields.at[i, 'LON'])
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV'] = elev

    oshape = fields.shape[0]
    fields = fields[~pd.isna(fields[gridmet_id_col])]
    print(f'Writing {fields.shape[0]} of {oshape} input features')
    fields[gridmet_id_col] = fields[gridmet_id_col].fillna(-1).astype(int)
    fields.to_file(fields_join, crs=fields.crs or 'EPSG:5071', engine='fiona')

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)
    print(f'wrote {factors_js}')


def download_gridmet(fields, gridmet_factors, gridmet_csv_dir, start=None, end=None, overwrite=False,
                     append=False, target_fields=None, feature_id='FID', return_df=False,
                     use_nldas=False):
    """Download GridMET time series and optionally NLDAS-2 hourly precipitation.

    Set ``use_nldas=True`` to append hourly precipitation fields derived from
    NLDAS-2 via pynldas2. When False, hourly precip fields are omitted from
    outputs and downstream code will derive them from daily precip if needed.
    """
    if not start:
        start = '1987-01-01'
    if not end:
        end = '2021-12-31'

    fields = gpd.read_file(fields)
    fields.index = fields[feature_id]
    fields = fields.sample(frac=1)

    with open(gridmet_factors, 'r') as f:
        gridmet_factors = json.load(f)

    hr_cols = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    downloaded, skipped_exists = {}, []
    _file = None

    for k, v in tqdm(fields.iterrows(), desc='Downloading GridMET', total=len(fields)):

        try:
            elev, existing = None, None
            out_cols = COLUMN_ORDER.copy()
            if use_nldas:
                out_cols += ['nld_ppt_d'] + hr_cols
            df, first = pd.DataFrame(), True

            if target_fields and str(k) not in target_fields:
                continue

            g_fid = str(int(v['GFID']))

            if g_fid in downloaded.keys():
                downloaded[g_fid].append(k)

            _file = os.path.join(gridmet_csv_dir, '{}.parquet'.format(g_fid))
            if os.path.exists(_file) and not overwrite and not append:
                skipped_exists.append(_file)
                continue

            if os.path.exists(_file) and append:
                existing = pd.read_parquet(_file)
                target_dates = pd.date_range(start, end, freq='D')
                missing_dates = [i for i in target_dates if i not in existing.index]

                if len(missing_dates) == 0 and not return_df:
                    continue
                elif len(missing_dates) == 0 and return_df:
                    return df
                else:
                    start, end = missing_dates[0].strftime('%Y-%m-%d'), missing_dates[-1].strftime('%Y-%m-%d')

            r = gridmet_factors[g_fid]
            lat, lon = r['lat'], r['lon']

            for thredds_var, cols in CLIMATE_COLS.items():
                variable = cols['col']

                if not thredds_var:
                    continue

                try:
                    g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                    s = g.get_point_timeseries()
                except OSError as e:
                    print('Error on {}, {}'.format(k, e))

                df[variable] = s[thredds_var]

                if first:
                    df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
                    df['year'] = [i.year for i in df.index]
                    df['month'] = [i.month for i in df.index]
                    df['day'] = [i.day for i in df.index]
                    df['centroid_lat'] = [lat for _ in range(df.shape[0])]
                    df['centroid_lon'] = [lon for _ in range(df.shape[0])]
                    g = GridMet('elev', lat=lat, lon=lon)
                    elev = g.get_point_elevation()
                    df['elev'] = [elev for _ in range(df.shape[0])]
                    first = False

                if thredds_var == 'pr' and use_nldas:
                    # gridmet is utc-6, US/Central, NLDAS is UTC-0
                    # shifting NLDAS to UTC-6 is the most straightforward alignment
                    s = pd.to_datetime(start) - timedelta(days=1)
                    s = s.strftime('%Y-%m-%d')
                    e = pd.to_datetime(end) + timedelta(days=2)
                    e = e.strftime('%Y-%m-%d')
                    nldas = nld.get_bycoords((lon, lat), start_date=s, end_date=e, variables=['prcp'])
                    if nldas.size == 0:
                        raise ValueError(f'Failed to download NLDAS-2 for {k} on GFID {g_fid}')

                    central = pytz.timezone('US/Central')
                    nldas = nldas.tz_convert(central)
                    hourly_ppt = nldas.pivot_table(columns=nldas.index.hour, index=nldas.index.date, values='prcp')
                    df[hr_cols] = hourly_ppt.loc[df.index]

                    nan_ct = np.sum(np.isnan(df[hr_cols].values), axis=0)
                    if sum(nan_ct) > 100:
                        raise ValueError('Too many NaN in NLDAS data')
                    if np.any(nan_ct):
                        df[hr_cols] = df[hr_cols].fillna(0.)

                    df['nld_ppt_d'] = df[hr_cols].sum(axis=1)

            p_air = air_pressure(df['elev'])
            ea_kpa = actual_vapor_pressure(df['q'], p_air)
            df['ea'] = ea_kpa.copy()

            for variable in ['etr', 'eto']:
                for month in range(1, 13):
                    corr_factor = gridmet_factors[g_fid][str(month)][variable]
                    idx = [i for i in df.index if i.month == month]
                    df.loc[idx, variable] = df.loc[idx, variable]
                    df.loc[idx, '{}_corr'.format(variable)] = df.loc[idx, variable] * corr_factor

            df['tmax'] = df.tmax - 273.15
            df['tmin'] = df.tmin - 273.15

            df = df[out_cols]
            # ['site', 'instrument', 'parameter', 'units', 'algorithm', 'mask']
            target_cols = []
            for c in df.columns:
                vals = [k, 'none', c, ACCEPTED_UNITS_MAP.get(c, 'none'), None, 'no_mask']
                if 'prcp_hr' in c or 'nld_ppt' in c:
                    vals[4] = 'nldas2'
                else:
                    vals[4] = 'gridmet'

                target_cols.append(tuple(vals))

            target_cols = pd.MultiIndex.from_tuples(target_cols, names=COLUMN_MULTIINDEX)
            df.columns = target_cols

            if existing is not None and not overwrite and append:
                df = pd.concat([df, existing], axis=0, ignore_index=False)
                df = df.sort_index()

            df.to_parquet(_file)
            print(f'wrote {_file}')
            downloaded[g_fid] = [k]

            if return_df:
                return df

        except Exception as exc:
            print(f'Error on {_file}: {exc}')
            continue

    print(f'downloaded {len(downloaded)} files')
    print(f'skipped {len(skipped_exists)} existing files')


# from CGMorton's RefET (github.com/WSWUP/RefET)
def air_pressure(elev, method='asce'):
    """Mean atmospheric pressure at station elevation (Eqs. 3 & 34)

    Parameters
    ----------
    elev : scalar or array_like of shape(M, )
        Elevation [m].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.

    Returns
    -------
    ndarray
        Air pressure [kPa].

    Notes
    -----
    The current calculation in Ref-ET:
        101.3 * (((293 - 0.0065 * elev) / 293) ** (9.8 / (0.0065 * 286.9)))
    Equation 3 in ASCE-EWRI 2005:
        101.3 * (((293 - 0.0065 * elev) / 293) ** 5.26)
    Per Dr. Allen, the calculation with full precision:
        101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** (9.80665 / (0.0065 * 286.9)))

    """
    pair = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    if method == 'asce':
        pair += 293
        pair /= 293
        np.power(pair, 5.26, out=pair)
    elif method == 'refet':
        pair += 293
        pair /= 293
        np.power(pair, 9.8 / (0.0065 * 286.9), out=pair)
    # np.power(pair, 5.26, out=pair)
    pair *= 101.3

    return pair


# from CGMorton's RefET (github.com/WSWUP/RefET)
def actual_vapor_pressure(q, pair):
    """"Actual vapor pressure from specific humidity

    Parameters
    ----------
    q : scalar or array_like of shape(M, )
        Specific humidity [kg/kg].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].

    Returns
    -------
    ndarray
        Actual vapor pressure [kPa].

    Notes
    -----
    ea = q * pair / (0.622 + 0.378 * q)

    """
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q

    return ea


# from CGMorton's RefET (github.com/WSWUP/RefET)
def wind_height_adjust(uz, zw):
    """Wind speed at 2 m height based on full logarithmic profile (Eq. 33)

    Parameters
    ----------
    uz : scalar or array_like of shape(M, )
        Wind speed at measurement height [m s-1].
    zw : scalar or array_like of shape(M, )
        Wind measurement height [m].

    Returns
    -------
    ndarray
        Wind speed at 2 m height [m s-1].

    """
    return uz * 4.87 / np.log(67.8 * zw - 5.42)


def gridmet_elevation(shp_in, shp_out):
    df = gpd.read_file(shp_in)
    l = []
    for i, r in df.iterrows():
        lat, lon = r['lat'], r['lon']
        g = GridMet('elev', lat=lat, lon=lon)
        elev = g.get_point_elevation()
        l.append((i, elev))

    df['ELEV_M'] = [i[1] for i in l]
    df.to_file(shp_out)


if __name__ == '__main__':
        pass
# ========================= EOF ====================================================================
