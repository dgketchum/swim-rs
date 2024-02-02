import os
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
import pyproj
from tqdm import tqdm

from prep.gridmet_corrected.thredds import GridMet

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
    'sph': {
        'nc': 'agg_met_sph_1979_CurrentYear_CONUS',
        'var': 'daily_mean_specific_humidity',
        'col': 'q_kgkg'},
    'srad': {
        'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
        'var': 'daily_mean_shortwave_radiation_at_surface',
        'col': 'srad_wm2'},
    'vs': {
        'nc': 'agg_met_vs_1979_CurrentYear_CONUS',
        'var': 'daily_mean_wind_speed',
        'col': 'u10_ms'},
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax_k'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin_k'},
    'th': {
        'nc': 'agg_met_th_1979_CurrentYear_CONUS',
        'var': 'daily_mean_wind_direction',
        'col': 'wdir_deg'},
    'vpd': {
        'nc': 'agg_met_vpd_1979_CurrentYear_CONUS',
        'var': 'daily_mean_vapor_pressure_deficit',
        'col': 'vpd_kpa'}
}

COLUMN_ORDER = ['date',
                'year',
                'month',
                'day',
                'centroid_lat',
                'centroid_lon',
                'elev_m',
                'u2_ms',
                'tmin_c',
                'tmax_c',
                'srad_wm2',
                'ea_kpa',
                'pair_kpa',
                'prcp_mm',
                'etr_mm',
                'eto_mm']


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
            if field['FID'] not in field_select:
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

        g = GridMet('elev', lat=field['LAT'], lon=field['LON'])
        elev = g.get_point_elevation()
        fields.at[i, 'ELEV'] = elev

    fields.to_file(fields_join, crs='EPSG:5071')

    len_ = len(gridmet_targets.keys())
    print('Get gridmet for {} target points'.format(len_))

    with open(factors_js, 'w') as fp:
        json.dump(gridmet_targets, fp, indent=4)


def download_gridmet(fields, gridmet_factors, gridmet_csv_dir, start=None, end=None):
    if not start:
        start = '1987-01-01'
    if not end:
        end = '2021-12-31'

    fields = gpd.read_file(fields)
    fields.index = fields['FID']

    with open(gridmet_factors, 'r') as f:
        gridmet_factors = json.load(f)

    print('Downloading GridMET')
    for k, v in tqdm(fields.iterrows(), total=fields.shape[0]):
        out_cols = COLUMN_ORDER.copy()
        df, first = pd.DataFrame(), True
        for thredds_var, cols in CLIMATE_COLS.items():
            variable = cols['col']

            if not thredds_var:
                continue

            g_fid = str(int(v['GFID']))
            r = gridmet_factors[g_fid]
            lat, lon = r['lat'], r['lon']
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            s = g.get_point_timeseries()
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

        for _var in ['etr', 'eto']:
            variable = '{}_mm'.format(_var)
            out_cols.append('{}_uncorr'.format(variable))
            for month in range(1, 13):
                corr_factor = gridmet_factors[g_fid][str(month)][_var]
                idx = [i for i in df.index if i.month == month]
                df.loc[idx, '{}_uncorr'.format(variable)] = df.loc[idx, variable]
                df.loc[idx, variable] = df.loc[idx, '{}_uncorr'.format(variable)] * corr_factor

        zw = 10
        df['u2_ms'] = wind_height_adjust(
            df.u10_ms, zw)
        df['pair_kpa'] = air_pressure(
            df.elev_m, method='asce')
        df['ea_kpa'] = actual_vapor_pressure(
            df.q_kgkg, df.pair_kpa)

        df['tmax_c'] = df.tmax_k - 273.15
        df['tmin_c'] = df.tmin_k - 273.15

        df = df[out_cols]
        _file = os.path.join(gridmet_csv_dir, 'gridmet_historical_{}.csv'.format(g_fid))
        df.to_csv(_file, index=False)


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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
