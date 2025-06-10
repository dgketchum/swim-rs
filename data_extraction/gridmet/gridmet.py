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


def find_gridmet_points(fields, gridmet_points, gridmet_ras, fields_join,
                        factors_js, field_select=None, feature_id='FID'):
    """This depends on running 'Raster Pixels to Points' on a WGS Gridmet raster,
     attributing GFID, lat, and lon in the attribute table, and saving to project crs: 5071.
     GFID is an arbitrary identifier e.g., @row_number. It further depends on projecting the
     rasters to EPSG:5071, usng the project.sh bash script

     The reason we're not just doing a zonal stat on correction surface for every object is that
     there may be many fields that only need data from one gridmet cell. This prevents us from downloading
     many redundant data sets.
    """

    print('Find field-gridmet joins')

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    fields = gpd.read_file(fields)
    gridmet_pts = gpd.read_file(gridmet_points)
    gridmet_pts.index = gridmet_pts['GFID']

    rasters = []

    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

    gridmet_targets = {}
    first = True
    for i, field in tqdm(fields.iterrows(), desc='Finding Nearest GridMET Neighbors', total=fields.shape[0]):

        if field_select:
            if str(field[feature_id]) not in field_select:
                continue

        min_distance = 1e13
        closest_fid = None

        xx, yy = field['geometry'].centroid.x, field['geometry'].centroid.y
        lat, lon = convert_to_wgs84(xx, yy)
        fields.at[i, 'LAT'] = lat
        fields.at[i, 'LON'] = lon

        for j, g_point in gridmet_pts.iterrows():
            distance = field['geometry'].centroid.distance(g_point['geometry'])

            if distance < min_distance:
                min_distance = distance
                closest_fid = j
                closest_geo = g_point['geometry']

        fields.at[i, 'GFID'] = closest_fid
        fields.at[i, 'STATION_ID'] = closest_fid

        if first:
            print('Matched {} to {}'.format(field[feature_id], closest_fid))
            first = False

        if closest_fid not in gridmet_targets.keys():
            gridmet_targets[closest_fid] = {str(m): {} for m in range(1, 13)}
            gdf = gpd.GeoDataFrame({'geometry': [closest_geo]})
            gridmet_targets[closest_fid]['lat'] = gridmet_pts.loc[closest_fid]['lat']
            gridmet_targets[closest_fid]['lon'] = gridmet_pts.loc[closest_fid]['lon']
            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'])[0]['mean']
                gridmet_targets[closest_fid][month].update({_var: stats})

        g = GridMet('elev', lat=fields.at[i, 'LAT'], lon=fields.at[i, 'LON'])
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV'] = elev

    fields.to_file(fields_join, crs='EPSG:5071', engine='fiona')

    len_ = len(gridmet_targets.keys())
    print('Get gridmet for {} target points'.format(len_))

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)


def get_gridmet_corrections(fields, gridmet_ras, fields_join,
                            factors_js, field_select=None, feature_id='FID'):
    print('Find field-gridmet joins')

    convert_to_wgs84 = lambda x, y: pyproj.Transformer.from_crs('EPSG:5071', 'EPSG:4326').transform(x, y)

    fields = gpd.read_file(fields)

    oshape = fields.shape[0]

    rasters = []
    for v in ['eto', 'etr']:
        [rasters.append(os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(v, m))) for m in range(1, 13)]

    gridmet_targets = {}

    for j, (i, field) in enumerate(tqdm(fields.iterrows(), desc='Assigning GridMET IDs', total=fields.shape[0])):

        if field_select:
            if str(field[feature_id]) not in field_select:
                continue

        xx, yy = field['geometry'].centroid.x, field['geometry'].centroid.y
        lat, lon = convert_to_wgs84(xx, yy)
        fields.at[i, 'LAT'] = lat
        fields.at[i, 'LON'] = lon

        closest_fid = j

        fields.at[i, 'GFID'] = closest_fid

        if closest_fid not in gridmet_targets.keys():
            gridmet_targets[closest_fid] = {str(m): {} for m in range(1, 13)}
            gdf = gpd.GeoDataFrame({'geometry': [field['geometry'].centroid]})
            gridmet_targets[closest_fid]['lat'] = lat
            gridmet_targets[closest_fid]['lon'] = lon
            for r in rasters:
                splt = r.split('_')
                _var, month = splt[-2], splt[-1].replace('.tif', '')
                stats = zonal_stats(gdf, r, stats=['mean'], nodata=np.nan)[0]['mean']
                # TODO: raise so tif/shp mismatch doesn't pass silent
                gridmet_targets[closest_fid][month].update({_var: stats})

        g = GridMet('elev', lat=fields.at[i, 'LAT'], lon=fields.at[i, 'LON'])
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV'] = elev

    fields = fields[~np.isnan(fields['GFID'])]
    print(f'Writing {fields.shape[0]} of {oshape} input features')
    fields['GFID'] = fields['GFID'].fillna(-1).astype(int)

    fields.to_file(fields_join, crs='EPSG:5071', engine='fiona')

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)
    print(f'wrote {factors_js}')


def download_gridmet(fields, gridmet_factors, gridmet_csv_dir, start=None, end=None, overwrite=False,
                     append=False, target_fields=None, feature_id='FID', return_df=False):
    if not start:
        start = '1987-01-01'
    if not end:
        end = '2021-12-31'

    fields = gpd.read_file(fields)
    fields.index = fields[feature_id]

    with open(gridmet_factors, 'r') as f:
        gridmet_factors = json.load(f)

    hr_cols = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    downloaded, skipped_exists = {}, []

    for k, v in tqdm(fields.iterrows(), desc='Downloading GridMET', total=len(fields)):

        elev, existing = None, None
        out_cols = COLUMN_ORDER.copy() + ['nld_ppt_d'] + hr_cols
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

            if thredds_var == 'pr':
                # gridmet is utc-6, US/Central, NLDAS is UTC-0
                # shifting NLDAS to UTC-6 is the most straightforward alignment
                s = pd.to_datetime(start) - timedelta(days=1)
                e = pd.to_datetime(end) + timedelta(days=2)

                nldas = nld.get_bycoords((lon, lat), start_date=s, end_date=e, variables=['prcp'], source='netcdf')
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
        downloaded[g_fid] = [k]

        if return_df:
            return df

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
