import os
from datetime import datetime

import ee
import geopandas as gpd
from shapely.geometry import mapping

CRS_TRANSFORM = [0.041666666666666664,
                 0, -124.78749996666667,
                 0, -0.041666666666666664,
                 49.42083333333334]

# Spectral Bandpass Adjustment Factors (SBAF) for harmonization to Landsat 8 OLI
# References:
#   Roy, D.P., et al. (2016). Characterization of Landsat-7 to Landsat-8
#   reflectance and NDVI differences. Remote Sensing of Environment, 176, 163-180.
#   https://www.sciencedirect.com/science/article/pii/S0034425716300220
#
#   Claverie, M., et al. (2018). The Harmonized Landsat and Sentinel-2 (HLS)
#   product. Remote Sensing of Environment, 219, 145-161.
#   https://www.sciencedirect.com/science/article/pii/S0034425718304139

# Landsat factors from https://github.com/google/earthengine-community/blob/master/tutorials/landsat-etm-to-oli-harmonization/index.md

# Roy et al. (2016) OLS coefficients for ETM+ to OLI transformation:
#   Band order: [Blue, Green, Red, NIR, SWIR1, SWIR2]
#   Slopes:     [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
#   Intercepts: [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]

SBAF_COEFFICIENTS = {
    # Landsat 4/5 TM and Landsat 7 ETM+ -> OLI (Roy et al., 2016, Table 2 OLS Surface Reflectance)
    'TM': {'red_slope': 0.9047, 'red_intercept': 0.0061,
           'nir_slope': 0.8462, 'nir_intercept': 0.0412},
    'ETM': {'red_slope': 0.9047, 'red_intercept': 0.0061,
            'nir_slope': 0.8462, 'nir_intercept': 0.0412},

    # Sentinel-2 MSI -> OLI (HLS, average of S2A and S2B, uses B8A for NIR)
    # https://hls.gsfc.nasa.gov/algorithms/bandpass-adjustment/
    # S2A Red (B4):  slope=0.9765, offset=0.0009  |  S2B: slope=0.9761, offset=0.0010
    # S2A NIR (B8A): slope=0.9983, offset=-0.0001 |  S2B: slope=0.9966, offset=0.0000

    'MSI': {'red_slope': 0.9763, 'red_intercept': 0.00095,
            'nir_slope': 0.99745, 'nir_intercept': -0.00005},

    # OLI is reference - no adjustment needed
    'OLI': {'red_slope': 1.0, 'red_intercept': 0.0,
            'nir_slope': 1.0, 'nir_intercept': 0.0},
}


def harmonize_landsat_to_oli(image):
    """Apply SBAF coefficients to harmonize Landsat TM/ETM+ to OLI reference.

    Adds RED_H and NIR_H bands with harmonized reflectance values. Landsat 8/9 (OLI)
    images pass through unchanged (coefficients are 1.0/0.0).

    After landsat_c2_sr processing, all Landsat sensors have:
    - B4 = Red band
    - B5 = NIR band

    Args:
        image: ee.Image from landsat_c2_sr with SPACECRAFT_ID property

    Returns:
        ee.Image with added RED_H and NIR_H bands
    """
    spacecraft_id = ee.String(image.get('SPACECRAFT_ID'))

    # OLI sensors (Landsat 8/9) are the reference - no adjustment needed
    # TM (Landsat 4/5) and ETM+ (Landsat 7) use the same SBAF coefficients
    is_oli = ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(spacecraft_id)

    # Select coefficients based on sensor
    coef = ee.Dictionary(ee.Algorithms.If(
        is_oli,
        SBAF_COEFFICIENTS['OLI'],
        SBAF_COEFFICIENTS['TM']
    ))

    red_slope = ee.Number(coef.get('red_slope'))
    red_intercept = ee.Number(coef.get('red_intercept'))
    nir_slope = ee.Number(coef.get('nir_slope'))
    nir_intercept = ee.Number(coef.get('nir_intercept'))

    # Apply linear transformation: harmonized = slope * original + intercept
    red_h = image.select('B4').multiply(red_slope).add(red_intercept).rename('RED_H')
    nir_h = image.select('B5').multiply(nir_slope).add(nir_intercept).rename('NIR_H')

    return image.addBands(red_h).addBands(nir_h)


def harmonize_sentinel_to_oli(image):
    """Apply SBAF coefficients to harmonize Sentinel-2 MSI to OLI reference.

    Adds RED_H and NIR_H bands with harmonized reflectance values.

    Sentinel-2 band mapping:
    - B4 = Red band (665nm)
    - B8A = Narrow NIR band (865nm) - used for HLS harmonization

    Args:
        image: ee.Image from sentinel2_sr

    Returns:
        ee.Image with added RED_H and NIR_H bands
    """
    coef = SBAF_COEFFICIENTS['MSI']

    # Apply linear transformation: harmonized = slope * original + intercept
    red_h = image.select('B4').multiply(coef['red_slope']).add(coef['red_intercept']).rename('RED_H')
    nir_h = image.select('B8A').multiply(coef['nir_slope']).add(coef['nir_intercept']).rename('NIR_H')

    return image.addBands(red_h).addBands(nir_h)


def sentinel2_sr(input_img):
    """Prepare Sentinel-2 SR image with basic scaling and cloud mask.

    Parameters
    - input_img: ee.Image, raw Sentinel-2 SR image (HARMONIZED).

    Returns
    - ee.Image with optical bands scaled to reflectance and cloudy pixels masked.
    """
    optical_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    scl_band = 'SCL'
    all_bands = optical_bands + [scl_band]

    mult = [0.0001] * len(optical_bands) + [1]
    prep_image = input_img.select(all_bands).multiply(mult)

    def _cloud_mask(i):
        scl = i.select('SCL')
        cloud_mask_values = [3, 8, 9, 10]
        mask = scl.remap(cloud_mask_values, [0] * len(cloud_mask_values), 1)
        mask = mask.rename(['cloud_mask'])
        return mask

    mask = _cloud_mask(prep_image)

    image = prep_image.select(optical_bands).updateMask(mask)
    image = image.copyProperties(input_img, ['system:time_start'])

    return image


def sentinel2_masked(yr, roi, harmonize=True):
    """Return a masked Sentinel-2 SR ImageCollection for a year and ROI.

    Parameters
    - yr: int, year of interest.
    - roi: ee.Geometry or ee.FeatureCollection bounds.
    - harmonize: bool, apply SBAF harmonization to OLI reference (default True).

    Returns
    - ee.ImageCollection of cloud-masked, reflectance-scaled optical bands.
      If harmonize=True, includes RED_H and NIR_H bands.
    """
    start = f'{yr}-01-01'
    end_date = f'{yr + 1}-01-01'

    s2_coll = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start, end_date) \
        .map(sentinel2_sr)

    if harmonize:
        s2_coll = s2_coll.map(harmonize_sentinel_to_oli)

    return s2_coll


def landsat_c2_sr(input_img):
    """Prepare Landsat Collection 2 SR image with scaling and cloud/saturation mask.

    Parameters
    - input_img: ee.Image with SPACECRAFT_ID and Collection 2 SR bands.

    Returns
    - ee.Image with renamed bands, scaled surface reflectance, temperature band,
      and cloud/saturation mask applied; preserves time property.
    """
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start', 'SPACECRAFT_ID'])

    return image


def landsat_masked(yr, roi, harmonize=True):
    """Return cloud-masked Landsat C2 SR ImageCollection merged across sensors.

    Parameters
    - yr: int, year of interest.
    - roi: ee.Geometry or ee.FeatureCollection bounds.
    - harmonize: bool, apply SBAF harmonization to OLI reference (default True).

    Returns
    - ee.ImageCollection with scaled/renamed bands and cloud/saturation mask.
      If harmonize=True, includes RED_H and NIR_H bands for consistent NDVI.
    """
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l9_coll = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l9_coll).merge(l5_coll).merge(l4_coll))

    if harmonize:
        lsSR_masked = lsSR_masked.map(harmonize_landsat_to_oli)

    return lsSR_masked


def export_openet_correction_surfaces(local_check):
    """Export monthly OpenET GridMET correction images to GCS.

    Exports both ETr and ETo ratios for each month to the `wudr` bucket, skipping
    any month where a local GeoTIFF already exists in `local_check`.

    Parameters
    - local_check: str or None; directory to check for existing local files.
    """
    is_authorized()

    for etref in ['etr', 'eto']:
        id_ = 'projects/openet/reference_et/gridmet/ratios/v1/monthly/{}'.format(etref)
        c = ee.ImageCollection(id_)
        scenes = c.aggregate_histogram('system:index').getInfo()
        for k in list(scenes.keys()):
            month_number = datetime.strptime(k, '%b').month
            if local_check:
                f = 'gridmet_corrected_{}'.format(etref)
                local_file = os.path.join(local_check, '{}_{}.tif'.format(f, month_number))
                if os.path.exists(local_file):
                    continue
            desc = 'gridmet_corrected_{}_{}'.format(etref, month_number)
            i = ee.Image(os.path.join(id_, k))
            task = ee.batch.Export.image.toCloudStorage(
                i,
                description=desc,
                bucket='wudr',
                dimensions='1386x585',
                fileNamePrefix=desc,
                crsTransform=CRS_TRANSFORM,
                crs='EPSG:4326')
            task.start()
            print(desc)


def get_lanid():
    """Build a multi-band LANID irrigation mask image for 1987–2024.

    Returns
    - ee.Image with bands named `irr_<year>` where 1 indicates irrigated.
    """
    first_image = ee.Image('users/xyhuwmir4/LANID_postCls/LANID_v2')
    second_image = ee.Image('users/xyhuwmir/LANID/update/LANID2018-2020')

    bands = None

    for yr in range(1987, 2018):
        if yr < 1997:
            year = 1997
        else:
            year = yr
        band_name = f'irr_{yr}'
        image = ee.Image(first_image.select([f'irMap{str(year)[-2:]}'])).rename([band_name]).int().unmask(0)
        if bands is None:
            bands = ee.Image(image)
        else:
            bands = bands.addBands([image])

    for yr in range(2018, 2025):
        if yr > 2020:
            year = 2020
        else:
            year = yr
        band_name = f'irr_{yr}'
        image = ee.Image(second_image.select([f'irMap{str(year)[-2:]}'])).rename([band_name]).int().unmask(0)
        bands = bands.addBands([image])

    return bands


def as_ee_feature_collection(fields, feature_id='FID', keep_props=None):
    """Return an ee.FeatureCollection from an asset ID, ee object, or shapefile path.

    Parameters
    - fields: one of
      - str: EE asset path ('projects/...', 'users/...') or local shapefile path
      - ee.FeatureCollection
      - any object accepted by ee.FeatureCollection constructor
    - feature_id: name of the ID property to preserve when building from shapefile
    - keep_props: optional list of additional property names to preserve from shapefile rows

    Returns
    - ee.FeatureCollection with properties limited to those requested.
    """
    if keep_props is None:
        keep_props = []
    # Always include the feature_id if provided
    if feature_id and feature_id not in keep_props:
        keep_props = [feature_id] + keep_props

    # Already a FeatureCollection instance
    if isinstance(fields, ee.featurecollection.FeatureCollection):  # type: ignore[attr-defined]
        return fields

    # String inputs: asset id or path
    if isinstance(fields, str):
        # EE asset path
        if fields.startswith('projects/') or fields.startswith('users/'):
            return ee.FeatureCollection(fields)
        # Local shapefile path
        if os.path.exists(fields):
            gdf = gpd.read_file(fields)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            feats = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                props = {}
                for k in keep_props:
                    if k in row:
                        props[k] = row[k]
                geo = mapping(geom)
                if geo['type'] == 'Polygon':
                    ee_geom = ee.Geometry.Polygon(geo['coordinates'])
                elif geo['type'] == 'MultiPolygon':
                    ee_geom = ee.Geometry.MultiPolygon(geo['coordinates'])
                else:
                    ee_geom = ee.Geometry(geo)
                feats.append(ee.Feature(ee_geom, props))
            return ee.FeatureCollection(feats)

    # Fallback: let EE attempt construction
    return ee.FeatureCollection(fields)


def is_authorized():
    """Initialize the Earth Engine client using the configured project.

    Raises a RuntimeError if initialization fails.
    Returns True on success.
    """
    try:
        ee.Initialize(project='ee-dgketchum')
        return True
    except Exception as e:
        raise RuntimeError(f'Earth Engine authorization failed: {e}')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/et-demands/gridmet/gridmet_corrected/correction_surfaces_wgs'
    export_openet_correction_surfaces(d)
# ========================= EOF ====================================================================
