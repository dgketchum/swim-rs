# NDVI extraction for Nevada DRI field boundaries using swimrs EE modules.
# Companion to the Feb 2026 ETf pull in gs://wudr/nv/ — same partitioning
# (FIPS + a/b/c... chunks of <=900 fields), same feature ID (OPENET_ID),
# writes gs://wudr/nv/{label}/ndvi/{mask}/ndvi_{mask}_{year}.csv

import math
import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd

from swimrs.data_extraction.ee.common import export_table, shapefile_to_feature_collection
from swimrs.data_extraction.ee.ee_utils import is_authorized, landsat_masked

WAIT_MINUTES = 10
MAX_RETRIES = 6

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
IRR_MIN_YR_ASSET = "projects/ee-dgketchum/assets/swim/nv_irr_min_yr_mask"
# IrrMapper ends at 2023 — cap year for 2024-2025
IRR_MAX_YEAR = 2023

FEATURE_ID = "OPENET_ID"
SHAPEFILE = (
    "/nas/Nevada/dri_field_pts/fields_gis/Nevada_Agricultural_Field_Boundaries_20250214/"
    "Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.shp"
)

# Max fields per export chunk; ceil(n/900) reproduces the ETf partition
# layout already in gs://wudr/nv/ (32001a..c, 32007a..d, etc.)
CHUNK_SIZE = 900
CHUNK_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"


def export_irr_min_yr_mask(shapefile):
    """Export the multi-year IrrMapper mask over Nevada as an EE asset (one-time).

    The mask identifies pixels irrigated in >= 5 of 37 years (1987-2023).
    Pre-computing it as an asset eliminates the 37-image computation graph
    that otherwise causes 'too many bands' errors in toBands() exports.

    Region comes from the shapefile's total bounds computed client-side —
    embedding the 24k-polygon FeatureCollection in the export request
    exceeds EE's 10 MB payload limit.
    """
    irr_coll = ee.ImageCollection(IRR)
    remap = irr_coll.filterDate("1987-01-01", "2024-12-31").select("classification")
    irr_min_yr_mask = remap.map(lambda img: img.lt(1)).sum().gte(5).toByte()

    bounds = gpd.read_file(shapefile, engine="fiona").to_crs(4326).total_bounds
    region = ee.Geometry.Rectangle(list(bounds))
    task = ee.batch.Export.image.toAsset(
        image=irr_min_yr_mask,
        description="nv_irr_min_yr_mask",
        assetId=IRR_MIN_YR_ASSET,
        region=region,
        scale=30,
        maxPixels=1e13,
    )
    task.start()
    print(f"Exporting irr_min_yr_mask asset — task ID: {task.id}")
    print(f"  Asset path: {IRR_MIN_YR_ASSET}")
    print(f"  Monitor: earthengine task info {task.id}")
    return task


def extract_ndvi(
    feature_coll,
    mask_type="irr",
    start_yr=1995,
    end_yr=2025,
    years=None,
    half=None,
    feature_id=FEATURE_ID,
    dest="bucket",
    bucket="wudr",
    file_prefix="nv",
):
    """Extract mean NDVI per field using harmonized Landsat.

    Uses SBAF-adjusted NIR_H/RED_H for consistent cross-sensor NDVI.

    When dest="local", uses ee.data.computeFeatures for synchronous return
    and returns a concatenated DataFrame.

    When dest="bucket", starts one ee.batch export task per year to GCS
    and returns None.

    Parameters
    ----------
    years : list[int] or None
        Explicit list of years to process.  Overrides start_yr/end_yr.
    half : str or None
        "h1" for Jan-Jun, "h2" for Jul-Dec.  Reduces band count per export.
    """
    irr_coll = ee.ImageCollection(IRR)

    # Load pre-computed multi-year mask asset to avoid graph explosion in
    # toBands(). Falls back to live computation if asset doesn't exist.
    try:
        irr_min_yr_mask = ee.Image(IRR_MIN_YR_ASSET)
        irr_min_yr_mask.getInfo()  # verify asset exists
        print("  Using pre-computed irr_min_yr_mask asset")
    except ee.ee_exception.EEException:
        print("  WARNING: irr_min_yr_mask asset not found, computing live")
        remap = irr_coll.filterDate("1987-01-01", "2024-12-31").select("classification")
        irr_min_yr_mask = remap.map(lambda img: img.lt(1)).sum().gte(5)

    if years is None:
        years = list(range(start_yr, end_yr + 1))

    dfs = []

    for year in years:
        # IrrMapper year capped at 2023 for years beyond coverage
        irr_year = min(year, IRR_MAX_YEAR)
        irr = (
            irr_coll.filterDate(f"{irr_year}-01-01", f"{irr_year}-12-31")
            .select("classification")
            .mosaic()
        )
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        coll = landsat_masked(year, feature_coll, harmonize=True).select(["NIR_H", "RED_H"])

        if half == "h1":
            coll = coll.filterDate(f"{year}-01-01", f"{year}-07-01")
        elif half == "h2":
            coll = coll.filterDate(f"{year}-07-01", f"{year + 1}-01-01")

        # Apply mask per-image BEFORE toBands to avoid EE graph-expansion
        # bug where mask(irr_mask) + reduceRegions on a toBands image
        # causes EE to count >5000 bands from the IrrMapper .sum() graph.
        if mask_type == "irr":
            coll = coll.map(
                lambda x, _m=irr_mask: x.normalizedDifference(["NIR_H", "RED_H"]).updateMask(_m)
            )
        elif mask_type == "inv_irr":
            coll = coll.map(
                lambda x, _i=irr: x.normalizedDifference(["NIR_H", "RED_H"]).updateMask(_i.gt(0))
            )
        else:
            coll = coll.map(lambda x: x.normalizedDifference(["NIR_H", "RED_H"]))

        for attempt in range(MAX_RETRIES):
            try:
                scenes = coll.aggregate_histogram("system:index").getInfo()
                break
            except ee.ee_exception.EEException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"  getInfo failed ({exc}), retrying in {WAIT_MINUTES} min...")
                time.sleep(WAIT_MINUTES * 60)

        band_names = sorted(scenes.keys())
        half_tag = f" ({half})" if half else ""
        print(f"  {year}{half_tag}: {len(band_names)} scenes")
        if not band_names:
            print(f"  {year}: no scenes, skipping")
            continue
        bands = coll.toBands().rename(band_names)

        data = bands.reduceRegions(
            collection=feature_coll,
            reducer=ee.Reducer.mean(),
            scale=30,
            tileScale=8,
        )

        if dest == "local":
            data_df = ee.data.computeFeatures(
                {"expression": data, "fileFormat": "PANDAS_DATAFRAME"}
            )
            data_df.index = data_df[feature_id]
            data_df.drop(columns=["geo"], inplace=True, errors="ignore")
            dfs.append(data_df)
        elif dest == "bucket":
            half_suffix = f"_{half}" if half else ""
            desc = f"ndvi_{mask_type}_{year}{half_suffix}"
            selectors = [feature_id] + band_names
            for attempt in range(MAX_RETRIES):
                try:
                    export_table(
                        data,
                        desc=desc,
                        selectors=selectors,
                        dest="bucket",
                        bucket=bucket,
                        fn_prefix=f"{file_prefix}/ndvi/{mask_type}/{desc}",
                    )
                    break
                except ee.ee_exception.EEException as exc:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    print(f"  export failed ({exc}), retrying in {WAIT_MINUTES} min...")
                    time.sleep(WAIT_MINUTES * 60)

    if dest == "local":
        return pd.concat(dfs, axis=1)
    return None


def _chunk_list(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def partition_fields(gdf):
    """Group fields by FIPS and chunk to match the gs://wudr/nv/ ETf layout.

    Returns list of (label, fids) like ('32001a', [...]), ('32009', [...]).
    """
    partitions = []
    for fips, grp in gdf.groupby("FIPS"):
        fids = grp[FEATURE_ID].tolist()
        n_chunks = math.ceil(len(fids) / CHUNK_SIZE)
        if n_chunks > 1:
            for ci, chunk_fids in enumerate(_chunk_list(fids, n_chunks)):
                partitions.append((f"{fips}{CHUNK_SUFFIXES[ci]}", chunk_fids))
        else:
            partitions.append((str(fips), fids))
    return partitions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nevada NDVI extraction")
    parser.add_argument("--fips", type=str, default=None, help="Comma-separated FIPS codes")
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Comma-separated partition labels (e.g. 32001a,32007c); overrides --fips",
    )
    parser.add_argument("--start-yr", type=int, default=1995)
    parser.add_argument("--end-yr", type=int, default=2025)
    parser.add_argument(
        "--years", type=str, default=None, help="Comma-separated years (overrides start/end)"
    )
    parser.add_argument(
        "--mask-types", type=str, default="irr,inv_irr", help="Comma-separated mask types"
    )
    parser.add_argument(
        "--half",
        choices=["h1", "h2"],
        default=None,
        help="Export half-year: h1=Jan-Jun, h2=Jul-Dec",
    )
    parser.add_argument("--dest", choices=["bucket", "local"], default="bucket")
    parser.add_argument("--bucket", type=str, default="wudr")
    parser.add_argument("--project", type=str, default="ee-dgketchum", help="EE project ID")
    parser.add_argument(
        "--export-mask",
        action="store_true",
        help="Export irr_min_yr_mask as EE asset and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the partition plan and exit without submitting tasks",
    )
    args = parser.parse_args()

    year_list = [int(y) for y in args.years.split(",")] if args.years else None
    mask_types = [m.strip() for m in args.mask_types.split(",")]

    root = "/data/ssd2/swim/nv"
    os.makedirs(root, exist_ok=True)
    sys.setrecursionlimit(5000)

    if not args.dry_run:
        is_authorized(args.project)

    if args.export_mask:
        export_irr_min_yr_mask(SHAPEFILE)
        sys.exit(0)

    gdf = gpd.read_file(SHAPEFILE, engine="fiona")

    dupes = gdf.loc[gdf[FEATURE_ID].duplicated(keep=False), FEATURE_ID].tolist()
    if dupes:
        print(f"WARNING: duplicated {FEATURE_ID} values (kept as-is, matching ETf pull): {dupes}")

    partitions = partition_fields(gdf)

    if args.partitions:
        selected = {p.strip() for p in args.partitions.split(",")}
        partitions = [(lbl, fids) for lbl, fids in partitions if lbl in selected]
    elif args.fips:
        selected = {f.strip() for f in args.fips.split(",")}
        partitions = [
            (lbl, fids) for lbl, fids in partitions if lbl.rstrip(CHUNK_SUFFIXES) in selected
        ]

    n_years = len(year_list) if year_list else args.end_yr - args.start_yr + 1
    n_tasks = len(partitions) * len(mask_types) * n_years
    print(
        f"{len(partitions)} partitions x {len(mask_types)} masks x {n_years} years = {n_tasks} export tasks"
    )

    if args.dry_run:
        for lbl, fids in partitions:
            print(f"  {lbl}: {len(fids)} fields")
        sys.exit(0)

    for label, fids in partitions:
        for mask_type in mask_types:
            print(f"\n=== {label} ({len(fids)} fields) mask={mask_type} ===")

            fc = shapefile_to_feature_collection(SHAPEFILE, FEATURE_ID, select=fids)

            start_time = time.time()
            result = extract_ndvi(
                fc,
                mask_type=mask_type,
                start_yr=args.start_yr,
                end_yr=args.end_yr,
                years=year_list,
                half=args.half,
                feature_id=FEATURE_ID,
                dest=args.dest,
                bucket=args.bucket,
                file_prefix=f"nv/{label}",
            )
            elapsed = time.time() - start_time

            if result is not None:
                out_csv = os.path.join(root, f"{label}_ndvi_{mask_type}.csv")
                result.to_csv(out_csv)
                print(
                    f"  {result.shape[0]} fields x {result.shape[1]} scenes "
                    f"in {elapsed:.1f}s -> {out_csv}"
                )
            else:
                print(f"  Export tasks submitted in {elapsed:.1f}s")

# ========================= EOF ====================================================================
