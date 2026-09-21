"""
PT-JPL ET fraction zonal statistics export module (International / ERA5LAND).

Export per-scene PT-JPL ET fraction zonal means for polygons to Google Cloud Storage as CSVs.
This version uses ERA5LAND for meteorology and reference ET, suitable for international sites.

Requires openet-ptjpl >= 0.5.0 with ERA5LAND support (era5land-updates branch):
    uv pip install -e /home/dgketchum/code/openet-ptjpl
"""

import os

import ee
import openet.ptjpl as ptjpl  # requires era5land-updates branch (>= 0.5.0)
from tqdm import tqdm

from .common import (
    ERA5LAND_BAND,
    ERA5LAND_FACTOR,
    ERA5LAND_RESAMPLE,
    ERA5LAND_SOURCE,
    LANDSAT_COLLECTIONS,
    build_feature_collection,
    export_table_to_gcs,
    load_shapefile,
    parse_scene_name,
)

# PT-JPL uses ERA5LAND for all meteorology sources
PTJPL_KWARGS = {
    "ta_source": "ERA5LAND",
    "ea_source": "ERA5LAND",
    "windspeed_source": "ERA5LAND",
    "rs_source": "ERA5LAND",
    "lwin_source": "ERA5LAND",
}


def export_ptjpl_zonal_stats(
    shapefile,
    bucket,
    feature_id="FID",
    select=None,
    start_yr=2000,
    end_yr=2024,
    check_dir=None,
    buffer=None,
    batch_size=15,
    file_prefix="swim",
):
    """
    Export per-scene PT-JPL ET fraction zonal means for polygons to GCS CSVs.

    This version uses ERA5LAND for meteorology and is suitable for international
    flux sites where US-specific datasets (GRIDMET, IrrMapper) are not available.

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
    check_dir : str, optional
        If set, skip exports when CSV already exists at check_dir/<desc>.csv.
    buffer : float, optional
        Buffer distance in meters to apply to geometries.
    batch_size : int, optional
        Number of scenes to process per export batch (default: 15).
        Smaller batches reduce server-side memory usage.
    file_prefix : str, optional
        Bucket path prefix, typically project name (default: 'swim').
    """
    df = load_shapefile(shapefile, feature_id, buffer=buffer)

    for fid, row in tqdm(df.iterrows(), desc="Export PT-JPL zonal stats", total=df.shape[0]):
        if row["geometry"].geom_type == "Point":
            continue
        elif row["geometry"].geom_type == "Polygon":
            polygon = ee.Geometry(row.geometry.__geo_interface__)
        else:
            continue

        if select is not None and fid not in select:
            continue

        for year in range(start_yr, end_yr + 1):
            # Get scene IDs for this year and geometry
            coll = ptjpl.Collection(
                LANDSAT_COLLECTIONS,
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                geometry=polygon,
                cloud_cover_max=70,
            )
            scenes = coll.get_image_ids()
            scenes = list(set(scenes))
            scenes = sorted(scenes, key=lambda item: item.split("_")[-1])

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
                    desc = f"ptjpl_etf_{fid}_no_mask_{year}_b{batch_idx:02d}"
                else:
                    desc = f"ptjpl_etf_{fid}_no_mask_{year}"
                fn_prefix = (
                    f"{file_prefix}/remote_sensing/landsat/extracts/ptjpl_etf/no_mask/{desc}"
                )

                if check_dir:
                    f = os.path.join(check_dir, f"{desc}.csv")
                    if os.path.exists(f):
                        print(f"{f} exists, skipping")
                        continue

                first, bands = True, None
                selectors = [feature_id]

                for img_id in batch_scenes:
                    _name = parse_scene_name(img_id)
                    selectors.append(_name)

                    try:
                        # Create PT-JPL image with ERA5LAND meteorology
                        ptjpl_img = ptjpl.Image.from_landsat_c2_sr(
                            img_id,
                            et_reference_source=ERA5LAND_SOURCE,
                            et_reference_band=ERA5LAND_BAND,
                            et_reference_factor=ERA5LAND_FACTOR,
                            et_reference_resample=ERA5LAND_RESAMPLE,
                            **PTJPL_KWARGS,
                        )
                        etf_img = ptjpl_img.et_fraction.rename(_name)

                    except ee.ee_exception.EEException as e:
                        print(f"{_name} returned error {e}")
                        continue

                    # Clip to polygon (no irrigation masking for international sites)
                    etf_img = etf_img.clip(polygon)

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
