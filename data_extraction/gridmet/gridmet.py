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

CLIMATE_COLS = {
    'etr': {
        'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_alfalfa',
        'col': 'etr_mm'},
    'pet': {
        'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_grass',
        'col': 'eto_mm'},
    'pr': {
        'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
        'var': 'precipitation_amount',
        'col': 'prcp_mm'},
    'srad': {
        'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
        'var': 'daily_mean_shortwave_radiation_at_surface',
        'col': 'srad_wm2'},
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax_k'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin_k'},
    'vs': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'u2_ms'},
    'sph': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'q'},
}

GRIDMET_GET = ['elev_m',
               'tmin_c',
               'tmax_c',
               'etr_mm',
               'eto_mm',
               'prcp_mm',
               'srad_wm2',
               'u2_ms',
               'ea_kpa',
               ]

BASIC_REQ = ['date', 'year', 'month', 'day', 'centroid_lat', 'centroid_lon']

COLUMN_ORDER = BASIC_REQ + GRIDMET_GET


def find_gridmet_points(fields, gridmet_points, gridmet_ras, fields_join,
                        factors_js, field_select=None):
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
    for i, field in tqdm(fields.iterrows(), total=fields.shape[0]):

        if field_select:
            if str(field['FID']) not in field_select:
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
            print('Matched {} to {}'.format(field['FID'], closest_fid))
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

    fields.to_file(fields_join, crs='EPSG:5071')

    len_ = len(gridmet_targets.keys())
    print('Get gridmet for {} target points'.format(len_))

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)


def download_gridmet(fields, gridmet_factors, gridmet_csv_dir, start=None, end=None, overwrite=False,
                     target_fields=None):
    if not start:
        start = '1987-01-01'
    if not end:
        end = '2021-12-31'

    fields = gpd.read_file(fields)
    fields.index = fields['FID']

    with open(gridmet_factors, 'r') as f:
        gridmet_factors = json.load(f)

    hr_cols = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    downloaded = {}

    print('Downloading GridMET')
    for k, v in fields.iterrows():
        elev = None
        out_cols = COLUMN_ORDER.copy() + ['nld_ppt_d'] + hr_cols
        df, first = pd.DataFrame(), True

        if target_fields and str(k) not in target_fields:
            continue

        g_fid = str(int(v['GFID']))

        if g_fid in downloaded.keys():
            downloaded[g_fid].append(k)
            print('Gridmet Cell {} downloaded for {}'.format(g_fid, downloaded[g_fid]))

        _file = os.path.join(gridmet_csv_dir, 'gridmet_historical_{}.csv'.format(g_fid))
        if os.path.exists(_file) and not overwrite:
            print('{} exists, skipping'.format(_file))
            continue

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
                df['elev_m'] = [elev for _ in range(df.shape[0])]
                first = False

            if thredds_var == 'pr':
                # gridmet is utc-6, US/Central, NLDAS is UTC-0
                # shifting NLDAS to UTC-6 is the most straightforward alignment
                s = pd.to_datetime(start) - timedelta(days=1)
                e = pd.to_datetime(end) + timedelta(days=2)
                nldas = nld.get_bycoords((lon, lat), start_date=s, end_date=e, variables=['prcp'])
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

        p_air = air_pressure(df['elev_m'])
        ea_kpa = actual_vapor_pressure(df['q'], p_air)
        df['ea_kpa'] = ea_kpa.copy()

        for _var in ['etr', 'eto']:
            variable = '{}_mm'.format(_var)
            out_cols.append('{}_uncorr'.format(variable))
            for month in range(1, 13):
                corr_factor = gridmet_factors[g_fid][str(month)][_var]
                idx = [i for i in df.index if i.month == month]
                df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

        df['tmax_c'] = df.tmax_k - 273.15
        df['tmin_c'] = df.tmin_k - 273.15

        df = df[out_cols]
        df.to_csv(_file, index=False)
        print(os.path.basename(_file))
        downloaded[g_fid] = [k]


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


def gridmet_elevation(lat, lon):
    g = GridMet('elev', lat=lat, lon=lon)
    elev = g.get_point_elevation()
    return elev


if __name__ == '__main__':
    in_ = '/media/research/IrrigationGIS/swim/gridmet/gridmet_centroids_tongue.shp'
    out_ = '/media/research/IrrigationGIS/swim/gridmet/gridmet_centroids_tongue_elev.shp'
    df = gpd.read_file(in_)
    l = []
    for i, r in df.iterrows():
        elev = gridmet_elevation(r['lat'], r['lon'])
        l.append((i, elev))
    pass
    df['ELEV_M'] = [i[1] for i in l]
    df.to_file(out_)
# ========================= EOF ====================================================================
