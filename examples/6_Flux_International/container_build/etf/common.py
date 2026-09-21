"""
Shared utilities for OpenET ETf zonal statistics export modules (International).

This module provides common constants, helper functions, and Earth Engine
utilities used by the export modules. This version is for international
flux sites and does not include US-specific irrigation masking.
"""

import time

import ee
import geopandas as gpd

# Landsat Collection 2 Level 2 Surface Reflectance collections
LANDSAT_COLLECTIONS = [
    "LANDSAT/LT04/C02/T1_L2",
    "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]

# ERA5-Land reference ET configuration (global coverage)
ERA5LAND_SOURCE = "ERA5LAND"
ERA5LAND_BAND = "eto"
ERA5LAND_FACTOR = 1.0
ERA5LAND_RESAMPLE = "bilinear"


def load_shapefile(shapefile, feature_id, buffer=None):
    """
    Load and prepare a shapefile for processing.

    Parameters
    ----------
    shapefile : str
        Path to the shapefile.
    feature_id : str
        Field name for feature identifier.
    buffer : float, optional
        Buffer distance in CRS units. Applied before CRS transformation.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame indexed by feature_id, in EPSG:4326.
    """
    df = gpd.read_file(shapefile, engine="fiona")
    df = df.set_index(feature_id, drop=False)

    if buffer:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and original_crs.srs != "EPSG:4326":
        df = df.to_crs(4326)

    return df


def build_feature_collection(polygon, fid, feature_id):
    """
    Build an ee.FeatureCollection from a polygon geometry.

    Parameters
    ----------
    polygon : ee.Geometry
        The polygon geometry.
    fid : str
        Feature ID value.
    feature_id : str
        Feature ID field name.

    Returns
    -------
    ee.FeatureCollection
    """
    return ee.FeatureCollection(ee.Feature(polygon, {feature_id: fid}))


def export_table_to_gcs(data, desc, bucket, fn_prefix, selectors):
    """
    Export a FeatureCollection to Google Cloud Storage as CSV.

    Parameters
    ----------
    data : ee.FeatureCollection
        The data to export.
    desc : str
        Export task description.
    bucket : str
        GCS bucket name (without gs:// prefix).
    fn_prefix : str
        File name prefix (path within bucket).
    selectors : list
        Column names to include in export.

    Returns
    -------
    bool
        True if export started successfully, False otherwise.
    """
    task = ee.batch.Export.table.toCloudStorage(
        data,
        description=desc,
        bucket=bucket,
        fileNamePrefix=fn_prefix,
        fileFormat="CSV",
        selectors=selectors,
    )

    try:
        task.start()
        print(desc, flush=True)
        return True
    except ee.ee_exception.EEException as e:
        error_message = str(e)

        if "payload size exceeds the limit" in error_message:
            print(f"Payload size limit exceeded for {desc}. Skipping task.")
            return False

        elif "already started with the given request_id" in error_message:
            # Task was already submitted (e.g. by a parallel process) — treat as success
            print(f"{desc} already submitted, skipping.")
            return True

        elif "many tasks already in the queue" in error_message:
            print(f"Task queue full. Waiting 10 minutes to retry {desc}...")
            time.sleep(600)
            # Recreate the task — reusing the same object fails with
            # "A different Operation was already started with the given request_id"
            retry_task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=fn_prefix,
                fileFormat="CSV",
                selectors=selectors,
            )
            retry_task.start()
            print(desc, flush=True)
            return True

        else:
            raise


def parse_scene_name(img_id):
    """
    Parse a Landsat scene ID to get the short name.

    Parameters
    ----------
    img_id : str
        Full Landsat scene ID (e.g., 'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716').

    Returns
    -------
    str
        Short scene name (e.g., 'LC08_044033_20170716').
    """
    splt = img_id.split("/")[-1].split("_")
    return "_".join(splt[-3:])
