"""
Shared utilities for OpenET ETf zonal statistics export modules.

This module provides common constants, helper functions, and Earth Engine
utilities used by the SSEBop, DisALEXI, and geeSEBAL export modules.
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

# Reference ET source configuration
GRIDMET_SOURCE = "IDAHO_EPSCOR/GRIDMET"
GRIDMET_BAND = "eto"
GRIDMET_FACTOR = 1.0

# Irrigation mask sources and regions
IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
WEST_STATES = ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"]
EAST_STATES_FC = "users/dgketchum/boundaries/eastern_38_dissolved"

# LANID irrigation dataset for eastern states
LANID_ASSET = "projects/ee-dgketchum/assets/lanid/LANID_V1"


def get_lanid() -> ee.Image:
    """Get the LANID irrigation dataset as an ee.Image."""
    return ee.Image(LANID_ASSET)


def load_shapefile(
    shapefile: str,
    feature_id: str,
    buffer: float | None = None,
) -> gpd.GeoDataFrame:
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
    df = gpd.read_file(shapefile)
    df = df.set_index(feature_id, drop=False)

    if buffer:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and original_crs.srs != "EPSG:4326":
        df = df.to_crs(4326)

    return df


def get_irrigation_mask(
    year: int,
    state: str | None,
    irr_coll: ee.ImageCollection,
    irr_min_yr_mask: ee.Image,
    lanid: ee.Image,
    east_fc: ee.FeatureCollection,
) -> tuple[ee.Image, ee.Image]:
    """
    Get irrigation mask for a given year and state.

    Parameters
    ----------
    year : int
        Year for the irrigation mask.
    state : str
        State abbreviation.
    irr_coll : ee.ImageCollection
        IrrMapper image collection.
    irr_min_yr_mask : ee.Image
        Minimum year mask from IrrMapper.
    lanid : ee.Image
        LANID irrigation dataset.
    east_fc : ee.FeatureCollection
        Eastern states feature collection for clipping.

    Returns
    -------
    tuple
        (irr, irr_mask) where irr is the irrigation image and irr_mask is
        the irrigation mask for the year.
    """
    if state in WEST_STATES:
        irr = (
            irr_coll.filterDate(f"{year}-01-01", f"{year}-12-31").select("classification").mosaic()
        )
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
    else:
        irr_mask = lanid.select(f"irr_{year}").clip(east_fc)
        irr = ee.Image(1).subtract(irr_mask)

    return irr, irr_mask


def setup_irrigation_masks() -> tuple[ee.ImageCollection, ee.Image, ee.Image, ee.FeatureCollection]:
    """
    Initialize Earth Engine irrigation mask resources.

    Returns
    -------
    tuple
        (irr_coll, irr_min_yr_mask, lanid, east_fc) - EE objects for irrigation masking.
    """
    irr_coll = ee.ImageCollection(IRR)
    s, e = "1987-01-01", "2025-12-31"
    remap = irr_coll.filterDate(s, e).select("classification").map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    east_fc = ee.FeatureCollection(EAST_STATES_FC)
    lanid = get_lanid()

    return irr_coll, irr_min_yr_mask, lanid, east_fc


def build_feature_collection(
    polygon: ee.Geometry,
    fid: str,
    feature_id: str,
) -> ee.FeatureCollection:
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


def export_table(
    data: ee.FeatureCollection,
    desc: str,
    selectors: list[str],
    dest: str = "drive",
    bucket: str | None = None,
    fn_prefix: str = "swim",
    drive_folder: str = "swim",
) -> bool:
    """
    Export a FeatureCollection to Google Drive or Cloud Storage as CSV.

    Parameters
    ----------
    data : ee.FeatureCollection
        The data to export.
    desc : str
        Export task description.
    selectors : list
        Column names to include in export.
    dest : str
        Export destination: 'drive' or 'bucket'.
    bucket : str, optional
        GCS bucket name (required if dest='bucket').
    fn_prefix : str
        File name prefix (path within bucket or Drive folder).
    drive_folder : str
        Google Drive folder name (used if dest='drive').

    Returns
    -------
    bool
        True if export started successfully, False otherwise.
    """
    if dest == "bucket":
        if not bucket:
            raise ValueError("bucket is required when dest='bucket'")
        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=desc,
            bucket=bucket,
            fileNamePrefix=fn_prefix,
            fileFormat="CSV",
            selectors=selectors,
        )
    elif dest == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=data,
            description=desc,
            folder=drive_folder,
            fileNamePrefix=fn_prefix,
            fileFormat="CSV",
            selectors=selectors,
        )
    else:
        raise ValueError(f"dest must be 'drive' or 'bucket', got '{dest}'")

    try:
        task.start()
        print(desc, flush=True)
        return True
    except ee.ee_exception.EEException as e:
        error_message = str(e)

        if "payload size exceeds the limit" in error_message:
            print(f"Payload size limit exceeded for {desc}. Skipping task.")
            return False

        elif "many tasks already in the queue" in error_message:
            print(f"Task queue full. Waiting 10 minutes to retry {desc}...")
            time.sleep(600)
            task.start()
            print(desc, flush=True)
            return True

        else:
            raise


def export_table_to_gcs(
    data: ee.FeatureCollection,
    desc: str,
    bucket: str,
    fn_prefix: str,
    selectors: list[str],
) -> bool:
    """
    Export a FeatureCollection to Google Cloud Storage as CSV.

    Deprecated: Use export_table() with dest='bucket' instead.
    """
    return export_table(
        data=data,
        desc=desc,
        selectors=selectors,
        dest="bucket",
        bucket=bucket,
        fn_prefix=fn_prefix,
    )


def get_scene_ids(
    model_collection: type,
    start_date: str,
    end_date: str,
    polygon: ee.Geometry,
) -> list[str]:
    """
    Get sorted list of Landsat scene IDs for a geometry and date range.

    Parameters
    ----------
    model_collection : type
        OpenET model collection class (e.g., ssebop.Collection).
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    polygon : ee.Geometry
        Geometry to filter scenes.

    Returns
    -------
    list
        Sorted list of scene IDs.
    """
    coll = model_collection(
        LANDSAT_COLLECTIONS,
        start_date=start_date,
        end_date=end_date,
        geometry=polygon,
        cloud_cover_max=70,
    )
    scenes = coll.get_image_ids()
    scenes = list(set(scenes))
    scenes = sorted(scenes, key=lambda item: item.split("_")[-1])
    return scenes


def parse_scene_name(img_id: str) -> str:
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


def get_pathrows_from_scenes(scenes: list[str]) -> set[str]:
    """
    Extract unique Landsat path/row combinations from scene IDs.

    Parameters
    ----------
    scenes : list
        List of Landsat scene IDs.

    Returns
    -------
    set
        Unique path/row strings (e.g., {'044033', '044034'}).
    """
    pathrows = set()
    for scene in scenes:
        # Scene format: 'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716' or 'LC08_044033_20170716'
        parts = scene.split("/")[-1].split("_")
        if len(parts) >= 2:
            pathrows.add(parts[1])  # e.g., '044033'
    return pathrows


def shapefile_to_feature_collection(
    shapefile: str,
    feature_id: str,
    select: list[str] | None = None,
    keep_props: list[str] | None = None,
) -> ee.FeatureCollection:
    """
    Convert a local shapefile to an ee.FeatureCollection.

    Parameters
    ----------
    shapefile : str
        Path to the shapefile.
    feature_id : str
        Field name for feature identifier.
    select : list, optional
        List of feature IDs to include (filters the collection).
    keep_props : list, optional
        Additional properties to preserve (beyond feature_id).

    Returns
    -------
    ee.FeatureCollection
    """
    df = load_shapefile(shapefile, feature_id)

    if select is not None:
        df = df[df.index.isin(select)]

    props_to_keep = [feature_id] + (keep_props or [])
    props_to_keep = [p for p in props_to_keep if p in df.columns]

    features = []
    for fid, row in df.iterrows():
        geom = ee.Geometry(row.geometry.__geo_interface__)
        props = {p: row[p] for p in props_to_keep if p in row}
        features.append(ee.Feature(geom, props))

    return ee.FeatureCollection(features)
