"""Copy OpenET images to GCS as deduplicated GeoTIFF chips.

Discovers scenes overlapping each site, deduplicates by scene ID (so scenes
covering multiple co-located sites are exported only once), clips to the union
of overlapping sites' 4km buffers, and exports to
``gs://{bucket}/openet/etf/{model}/{scene_id}.tif``.

Usage:
    python copy_openet_assets.py --shapefile <path> --models ssebop \\
        [--project ee-dgketchum] [--start-date 2016-01-01] [--end-date 2024-12-31] \\
        [--sites SITE1,SITE2] [--buffer 4000] [--bucket wudr] [--single-test]
"""

import subprocess
import time

import ee
import geopandas as gpd
import shapely.ops
from pyproj import CRS, Transformer
from tqdm import tqdm

# OpenET v2.1 source collections
OPENET_SOURCES = {
    "ssebop": "projects/openet/assets/ssebop/conus/gridmet/landsat/v2_1",
    "sims": "projects/openet/assets/sims/conus/gridmet/landsat/v2_1",
    "geesebal": "projects/openet/assets/geesebal/conus/gridmet/landsat/v2_1",
    "eemetric": "projects/openet/assets/eemetric/conus/gridmet/landsat/v2_1",
    "ensemble": "projects/openet/assets/ensemble/conus/gridmet/landsat/v2_1",
    "ptjpl": "projects/openet/assets/ptjpl/conus/nldas2/landsat/v2_1",
    "disalexi": "projects/openet/assets/disalexi/conus/cfsr/landsat/v2_1",
}

DEFAULT_BUCKET = "wudr"
GCS_PREFIX = "openet/etf"


def _build_site_geometries(shapefile, feature_id, buffer_m=4000, select=None):
    """Load shapefile, buffer each site, return shapely and EE geometry dicts.

    Returns
    -------
    shapely_geoms : dict[str, shapely.geometry.base.BaseGeometry]
        Mapping of site_id to buffered shapely geometry in EPSG:4326.
    ee_geoms : dict[str, ee.Geometry]
        Mapping of site_id to the same buffered geometry as ee.Geometry.
    """
    gdf = gpd.read_file(shapefile, engine="fiona")
    gdf = gdf.set_index(feature_id)

    if select is not None:
        gdf = gdf[gdf.index.isin(select)]

    # Project to a meter-based CRS for buffering
    src_crs = gdf.crs
    if src_crs is None or src_crs.is_geographic:
        proj_crs = CRS.from_epsg(5070)
        gdf_proj = gdf.to_crs(proj_crs)
    else:
        gdf_proj = gdf
        proj_crs = gdf_proj.crs

    transformer = Transformer.from_crs(proj_crs, CRS.from_epsg(4326), always_xy=True)

    shapely_geoms = {}
    ee_geoms = {}
    for site_id, row in gdf_proj.iterrows():
        buffered = row.geometry.buffer(buffer_m)
        geom_4326 = shapely.ops.transform(transformer.transform, buffered)
        shapely_geoms[site_id] = geom_4326
        ee_geoms[site_id] = ee.Geometry(geom_4326.__geo_interface__)

    return shapely_geoms, ee_geoms


def _discover_scenes(src_path, start_date, end_date, ee_geoms):
    """Query EE for scene IDs overlapping each site, return deduplicated mapping.

    Returns
    -------
    scene_to_sites : dict[str, set[str]]
        Mapping of scene_id to set of overlapping site_ids.
    """
    scene_to_sites = {}
    for site_id, site_geom in tqdm(ee_geoms.items(), desc="Discovering scenes"):
        coll = ee.ImageCollection(src_path).filterDate(start_date, end_date).filterBounds(site_geom)
        scene_hist = coll.aggregate_histogram("system:index").getInfo()
        if not scene_hist:
            continue
        for scene_id in scene_hist:
            scene_to_sites.setdefault(scene_id, set()).add(site_id)

    return scene_to_sites


def _list_existing_tifs(bucket, prefix, existing_list=None):
    """Return set of filenames (without .tif) already in the GCS prefix.

    If *existing_list* is provided it should be a path to a text file with one
    ``gs://`` URI per line (the output of ``gsutil ls -r``).  Only lines whose
    path component matches *prefix* are considered.  This avoids needing
    ``gsutil ls`` access on the running machine.
    """
    if existing_list is not None:
        names = set()
        target = f"{prefix}/"
        with open(existing_list) as fh:
            for line in fh:
                line = line.strip()
                if not line or not line.endswith(".tif"):
                    continue
                # line looks like gs://bucket/openet/etf/model/scene.tif
                path = line.split(f"{bucket}/", 1)[-1]
                if not path.startswith(target):
                    continue
                blob = path.rsplit("/", 1)[-1]
                names.add(blob[:-4])
        return names

    uri = f"gs://{bucket}/{prefix}/"
    try:
        result = subprocess.run(
            ["gsutil", "ls", uri],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return set()
        names = set()
        for line in result.stdout.strip().splitlines():
            blob = line.rsplit("/", 1)[-1]
            if blob.endswith(".tif"):
                names.add(blob[:-4])
        return names
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def _get_pending_task_descriptions():
    """Return set of descriptions for READY and RUNNING EE export tasks."""
    ops = ee.data.listOperations()
    pending = set()
    for op in ops:
        meta = op.get("metadata", {})
        state = meta.get("state", "")
        if state in ("PENDING", "RUNNING", "READY"):
            desc = meta.get("description", "")
            if desc:
                pending.add(desc)
    return pending


def _start_export(task, desc, max_retries=6, wait_minutes=10):
    """Start an EE export task, retrying on queue-full errors."""
    for attempt in range(max_retries + 1):
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg = str(e)
            if "many tasks already in the queue" in msg:
                if attempt < max_retries:
                    print(f"Queue full. Waiting {wait_minutes}min to retry {desc}...")
                    time.sleep(wait_minutes * 60)
                else:
                    print(f"Queue still full after {max_retries} retries, giving up on {desc}")
                    return False
            else:
                print(f"Export error for {desc}: {msg}")
                return False
    return False


def copy_openet_to_bucket(
    shapefile,
    model,
    bucket=DEFAULT_BUCKET,
    source_collection=None,
    feature_id="site_id",
    buffer_m=4000,
    start_date="2016-01-01",
    end_date="2024-12-31",
    select=None,
    single_test=False,
    existing_list=None,
):
    """Export clipped OpenET images to GCS as deduplicated GeoTIFFs.

    Discovers scenes overlapping each site, deduplicates by scene ID so
    that a scene covering N co-located sites produces one export (not N),
    clips to the union of overlapping sites' buffers, and exports via
    ``Export.image.toCloudStorage()``.

    Parameters
    ----------
    shapefile : str
        Path to polygon shapefile with site geometries.
    model : str
        Model name (key into OPENET_SOURCES).
    bucket : str
        GCS bucket name.
    source_collection : str, optional
        Override source EE ImageCollection path.
    feature_id : str
        Column name for feature identifier.
    buffer_m : float
        Buffer distance in meters around each site.
    start_date, end_date : str
        Date range (YYYY-MM-DD).
    select : list[str], optional
        Subset of feature IDs to process.
    single_test : bool
        If True, export one image and wait for completion to verify.
    """
    if model not in OPENET_SOURCES and source_collection is None:
        raise ValueError(f"Unknown model: {model}. Provide --source or use: {list(OPENET_SOURCES)}")

    src_path = source_collection or OPENET_SOURCES[model]
    gcs_prefix = f"{GCS_PREFIX}/{model}"

    print(f"Source: {src_path}")
    print(f"Destination: gs://{bucket}/{gcs_prefix}/")

    # Build per-site buffered geometries (shapely for unions, EE for queries)
    shapely_geoms, ee_geoms = _build_site_geometries(shapefile, feature_id, buffer_m, select)
    print(f"Built {len(ee_geoms)} site geometries (buffer={buffer_m}m)")

    # Phase 1: Discover all scenes and build scene → sites dedup mapping
    scene_to_sites = _discover_scenes(src_path, start_date, end_date, ee_geoms)

    per_site_total = sum(len(sites) for sites in scene_to_sites.values())
    print(
        f"Found {len(scene_to_sites)} unique scenes "
        f"(was {per_site_total} per-site exports, "
        f"{per_site_total / max(len(scene_to_sites), 1):.1f}x dedup ratio)"
    )

    # Check what's already in the bucket
    existing = _list_existing_tifs(bucket, gcs_prefix, existing_list=existing_list)
    if existing:
        print(f"Found {len(existing)} existing TIFs in destination, will skip")

    # Check in-flight EE tasks so we don't duplicate ongoing exports
    pending_tasks = _get_pending_task_descriptions()
    if pending_tasks:
        print(f"Found {len(pending_tasks)} pending/running EE tasks, will skip duplicates")

    submitted, skipped = 0, 0

    # Phase 2: Export each unique scene once
    sorted_scenes = sorted(scene_to_sites.keys(), key=lambda s: s.split("_")[-1])

    for scene_id in tqdm(sorted_scenes, desc=f"Export {model}"):
        site_ids = scene_to_sites[scene_id]

        if scene_id in existing:
            skipped += 1
            continue

        desc = f"cp_{model}_{scene_id}".replace("/", "_")[:100]

        if desc in pending_tasks:
            skipped += 1
            continue

        # Build export region: union of overlapping sites' buffers
        if len(site_ids) == 1:
            sid = next(iter(site_ids))
            export_region = ee_geoms[sid]
        else:
            union = shapely.ops.unary_union([shapely_geoms[sid] for sid in site_ids])
            export_region = ee.Geometry(union.__geo_interface__)

        src_image = ee.Image(f"{src_path}/{scene_id}")
        clipped = src_image.clip(export_region)

        task = ee.batch.Export.image.toCloudStorage(
            image=clipped,
            description=desc,
            bucket=bucket,
            fileNamePrefix=f"{gcs_prefix}/{scene_id}",
            region=export_region,
            scale=30,
            crs="EPSG:5070",
            maxPixels=1e9,
            fileFormat="GeoTIFF",
        )

        if single_test:
            sites_str = ", ".join(sorted(site_ids))
            print(f"\n--- SINGLE TEST: {scene_id} (covers: {sites_str}) ---")
            print(f"  File: gs://{bucket}/{gcs_prefix}/{scene_id}.tif")
            print(f"  Description: {desc}")
            task.start()
            print("  Export started. Waiting for completion...")
            while task.status()["state"] in ("READY", "RUNNING"):
                time.sleep(5)
            status = task.status()
            print(f"  Final state: {status['state']}")
            if status["state"] != "COMPLETED":
                err = status.get("error_message", "unknown")
                print(f"  Error: {err}")
            return

        if _start_export(task, desc):
            submitted += 1

    print(f"Submitted {submitted} export tasks for {model} (skipped {skipped})")


if __name__ == "__main__":
    import argparse
    import traceback

    from swimrs.data_extraction.ee.ee_utils import is_authorized

    parser = argparse.ArgumentParser(description="Copy OpenET images to GCS as GeoTIFF chips")
    parser.add_argument("--shapefile", required=True, help="Path to polygon shapefile")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(OPENET_SOURCES),
        required=True,
        help="One or more OpenET model names",
    )
    parser.add_argument("--source", type=str, default=None, help="Override source collection path")
    parser.add_argument("--project", type=str, default="ee-dgketchum", help="EE project ID")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site IDs")
    parser.add_argument("--start-date", type=str, default="2016-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--feature-id", type=str, default="site_id")
    parser.add_argument("--buffer", type=int, default=4000, help="Buffer distance in meters")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="GCS bucket name")
    parser.add_argument(
        "--single-test", action="store_true", help="Export one image and wait to verify"
    )
    parser.add_argument(
        "--existing-list",
        type=str,
        default=None,
        help="Path to text file listing existing gs:// TIF URIs (replaces gsutil ls)",
    )
    args = parser.parse_args()

    is_authorized(project=args.project)

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None

    for model in args.models:
        try:
            print(f"\n{'=' * 60}")
            print(f"  Starting model: {model}")
            print(f"{'=' * 60}\n")
            copy_openet_to_bucket(
                shapefile=args.shapefile,
                model=model,
                bucket=args.bucket,
                source_collection=args.source,
                feature_id=args.feature_id,
                buffer_m=args.buffer,
                start_date=args.start_date,
                end_date=args.end_date,
                select=sites,
                single_test=args.single_test,
                existing_list=args.existing_list,
            )
        except Exception:
            traceback.print_exc()
            print(f"\n*** {model} failed, skipping to next model ***\n")
