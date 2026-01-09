"""
geeSEBAL ET fraction zonal statistics export module.

Export per-scene geeSEBAL ET fraction zonal means for polygons to Google Cloud Storage as CSVs.
"""
import os

import ee
import geopandas as gpd
from tqdm import tqdm

import openet.geesebal as geesebal

from .common import (
    LANDSAT_COLLECTIONS,
    GRIDMET_SOURCE,
    GRIDMET_BAND,
    GRIDMET_FACTOR,
    WEST_STATES,
    load_shapefile,
    setup_irrigation_masks,
    get_irrigation_mask,
    build_feature_collection,
    export_table_to_gcs,
    parse_scene_name,
)


def export_geesebal_zonal_stats(
    shapefile,
    bucket,
    feature_id='FID',
    select=None,
    start_yr=2000,
    end_yr=2024,
    mask_type='no_mask',
    check_dir=None,
    state_col='state',
    buffer=None,
    batch_size=15,
    file_prefix='swim',
):
    """
    Export per-scene geeSEBAL ET fraction zonal means for polygons to GCS CSVs.

    Parameters
    ----------
    shapefile : str
        Path to polygon shapefile with feature IDs.
    bucket : str
        GCS bucket name (no scheme).
    feature_id : str, optional
        Field name for feature identifier.
    select : list, optional
        Optional list of feature IDs to process.
    start_yr : int, optional
        Inclusive start year (default: 2000).
    end_yr : int, optional
        Inclusive end year (default: 2024).
    mask_type : {'no_mask', 'irr', 'inv_irr'}, optional
        Irrigation masking strategy (default: 'no_mask').
    check_dir : str, optional
        If set, skip exports when CSV already exists at check_dir/<desc>.csv.
    state_col : str, optional
        Column with state abbreviation for mask source selection.
    buffer : float, optional
        Buffer distance in meters to apply to geometries.
    batch_size : int, optional
        Number of scenes to process per export batch (default: 15).
        Smaller batches reduce server-side memory usage.
    file_prefix : str, optional
        Bucket path prefix, typically project name (default: 'swim').
    """
    df = load_shapefile(shapefile, feature_id, buffer=buffer)

    # Setup irrigation mask resources
    irr_coll, irr_min_yr_mask, lanid, east_fc = setup_irrigation_masks()

    for fid, row in tqdm(df.iterrows(), desc='Export geeSEBAL zonal stats', total=df.shape[0]):
        if row['geometry'].geom_type == 'Point':
            continue
        elif row['geometry'].geom_type == 'Polygon':
            polygon = ee.Geometry(row.geometry.__geo_interface__)
        else:
            continue

        if select is not None and fid not in select:
            continue

        for year in range(start_yr, end_yr + 1):
            # Get irrigation mask if needed
            if mask_type in ['irr', 'inv_irr']:
                state = row.get(state_col, None) if state_col in row else None
                irr, irr_mask = get_irrigation_mask(
                    year, state, irr_coll, irr_min_yr_mask, lanid, east_fc
                )
            else:
                irr, irr_mask = None, None

            # Get scene IDs for this year and geometry
            coll = geesebal.Collection(
                LANDSAT_COLLECTIONS,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
                geometry=polygon,
                cloud_cover_max=70,
            )
            scenes = coll.get_image_ids()
            scenes = list(set(scenes))
            scenes = sorted(scenes, key=lambda item: item.split('_')[-1])

            if not scenes:
                continue

            # Process scenes in batches to avoid server-side memory issues
            n_batches = (len(scenes) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(scenes))
                batch_scenes = scenes[batch_start:batch_end]

                # Include batch suffix only if multiple batches
                if n_batches > 1:
                    desc = f'geesebal_etf_{fid}_{mask_type}_{year}_b{batch_idx:02d}'
                else:
                    desc = f'geesebal_etf_{fid}_{mask_type}_{year}'
                fn_prefix = f'{file_prefix}/remote_sensing/landsat/extracts/geesebal_etf/{mask_type}/{desc}'

                if check_dir:
                    f = os.path.join(check_dir, f'{desc}.csv')
                    if os.path.exists(f):
                        print(f'{f} exists, skipping')
                        continue

                first, bands = True, None
                selectors = [feature_id]

                for img_id in batch_scenes:
                    _name = parse_scene_name(img_id)
                    selectors.append(_name)

                    try:
                        # Create geeSEBAL image with reference ET parameters
                        # geeSEBAL requires et_reference params to compute et_fraction
                        geesebal_img = geesebal.Image.from_landsat_c2_sr(
                            img_id,
                            et_reference_source=GRIDMET_SOURCE,
                            et_reference_band=GRIDMET_BAND,
                            et_reference_factor=GRIDMET_FACTOR,
                        )

                        # Get ET fraction
                        etf_img = geesebal_img.et_fraction.rename(_name)

                    except ee.ee_exception.EEException as e:
                        print(f'{_name} returned error {e}')
                        continue

                    # Apply masking
                    if mask_type == 'no_mask':
                        etf_img = etf_img.clip(polygon)
                    elif mask_type == 'irr':
                        etf_img = etf_img.clip(polygon).mask(irr_mask)
                    elif mask_type == 'inv_irr':
                        etf_img = etf_img.clip(polygon).mask(irr.gt(0))

                    if first:
                        bands = etf_img
                        first = False
                    else:
                        bands = bands.addBands([etf_img])

                if bands is None:
                    continue

                # Compute zonal statistics
                fc = build_feature_collection(polygon, fid, feature_id)
                data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

                # Export to GCS
                export_table_to_gcs(data, desc, bucket, fn_prefix, selectors)


if __name__ == '__main__':
    ee.Initialize()

    project = '5_Flux_Ensemble'
    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    project_ws = os.path.join(root, project)

    if not os.path.isdir(root):
        root = '/home/dgketchum/code/swim-rs'
        project_ws = os.path.join(root, 'examples', project)
        data = os.path.join(project_ws, 'data')

    shapefile_ = os.path.join(data, 'gis', 'flux_footprints_3p.shp')
    chk_dir = os.path.join(data, 'landsat', 'extracts', 'geesebal_etf')

    FEATURE_ID = 'site_id'

    export_geesebal_zonal_stats(
        shapefile=shapefile_,
        bucket='wudr',
        feature_id=FEATURE_ID,
        start_yr=1987,
        end_yr=2024,
        select=None,
        mask_type='no_mask',
        check_dir=chk_dir,
        state_col='state',
        buffer=None,
    )
