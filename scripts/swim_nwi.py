# One-stop Earth Engine extraction for SWIM Nevada (NWI) runs.
#
# From a fields shapefile, exports the full SWIM container input set to GCS,
# partitioned by FIPS county + a/b/c... chunks of <=900 fields (the layout
# established by the Feb 2026 gs://wudr/nv pull):
#
#   {prefix}/{label}/etf/{mask}/{model}_etf_{mask}_{year}.csv   OpenET v2.1 ETf
#   {prefix}/{label}/ndvi/{mask}/ndvi_{mask}_{year}.csv         harmonized Landsat NDVI
#   {prefix}/{label}/ndvi/sentinel/{mask}/ndvi_sentinel_{mask}_{year}.csv  Sentinel-2 NDVI (2017+)
#   {prefix}/{label}/met/eto/eto_{year}.csv                     OpenET bias-corrected GridMET ETo (mm)
#   {prefix}/{label}/snow/snodas/extracts/swe_{year}.csv        SNODAS SWE (meters)
#   {prefix}/{label}/properties/ssurgo_{label}.csv              SSURGO awc/ksat/clay/sand
#   {prefix}/{label}/properties/landcover_{label}.csv           MODIS + FROM-GLC10 mode landcover
#   {prefix}/{label}/properties/irr_{label}.csv                 IrrMapper irrigation fraction per year
#
# Irrigation masks come from IrrMapper in EE (irr = irrigated this year AND
# >=5 years over the full IrrMapper record; inv_irr = not irrigated this
# year). Years past the collection's latest classification reuse that year;
# the latest year is detected from the collection at runtime.
#
# Run from a swim-rs checkout (uv sync first), e.g.:
#   uv run python scripts/swim_nwi.py --shapefile fields.shp --project my-ee-project \
#       --bucket my-bucket --dry-run
#   uv run python scripts/swim_nwi.py --shapefile fields.shp --project my-ee-project \
#       --bucket my-bucket --targets all
#
# If the default IrrMapper min-yr mask asset is not readable from your EE
# project, or your fields extend beyond Nevada, export your own first:
#   ... --export-mask --min-yr-asset projects/<your-project>/assets/irr_min_yr_mask

import argparse
import math
import re
import sys
import time
from pathlib import Path

import ee
import geopandas as gpd

try:
    import swimrs  # noqa: F401
except ImportError:  # running from a checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swimrs.data_extraction.ee.common import export_table, load_shapefile
from swimrs.data_extraction.ee.ee_props import get_irrigation, get_landcover, get_ssurgo
from swimrs.data_extraction.ee.ee_utils import landsat_masked, sentinel2_masked

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
IRR_MIN_YR_ASSET = "projects/ee-dgketchum/assets/swim/nv_irr_min_yr_mask"
IRR_MAX_YEAR = None  # latest IrrMapper year, detected at runtime; later years reuse it

# OpenET v2.1 source collections (6 members + ensemble), 1999+ (disalexi 2001+)
OPENET_SOURCES = {
    "ssebop": "projects/openet/assets/ssebop/conus/gridmet/landsat/v2_1",
    "sims": "projects/openet/assets/sims/conus/gridmet/landsat/v2_1",
    "geesebal": "projects/openet/assets/geesebal/conus/gridmet/landsat/v2_1",
    "eemetric": "projects/openet/assets/eemetric/conus/gridmet/landsat/v2_1",
    "ensemble": "projects/openet/assets/ensemble/conus/gridmet/landsat/v2_1",
    "ptjpl": "projects/openet/assets/ptjpl/conus/nldas2/landsat/v2_1",
    "disalexi": "projects/openet/assets/disalexi/conus/cfsr/landsat/v2_1",
}
DIRECT_ETF_MODELS = {"ssebop", "sims", "eemetric"}  # band et_fraction / 10000
ET_BAND_MODELS = {"geesebal", "disalexi", "ptjpl"}  # band et / 1000, divide by eto
ENSEMBLE_MODELS = {"ensemble"}  # band et_ensemble_mad / 10000

# OpenET bias-corrected GridMET daily reference ET
REFET = "projects/openet/assets/reference_et/conus/gridmet/daily/v1"
SNODAS = "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"

ETF_START_YR = 1999  # OpenET v2.1 coverage (disalexi 2001+; empty years skip)
SWE_START_YR = 2004  # SNODAS coverage
SENTINEL_START_YR = 2017  # S2 SR archive (Ex5 convention)
IRR_START_YR = 1985  # IrrMapper record start (irrigation-fraction export)

TARGETS = ["etf", "ndvi", "eto", "swe", "soils", "props"]
CHUNK_SIZE = 900  # fields per export partition (EE payload limit)
CHUNK_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"
WAIT_MINUTES = 10
MAX_RETRIES = 6


def _retry(fn, desc):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except ee.ee_exception.EEException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  {desc} failed ({exc}), retrying in {WAIT_MINUTES} min...", flush=True)
            time.sleep(WAIT_MINUTES * 60)


def partition_fields(gdf, feature_id, partition_col):
    """Group fields by partition_col, chunked to <=CHUNK_SIZE per label.

    Returns list of (label, fids) like ('32001a', [...]), ('32009', [...]).
    Falls back to sequential numeric labels if partition_col is absent.
    """

    def chunk(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    if partition_col not in gdf.columns:
        print(f"WARNING: no '{partition_col}' column; using sequential partitions")
        fids = gdf[feature_id].tolist()
        chunks = chunk(fids, math.ceil(len(fids) / CHUNK_SIZE))
        return [(f"grp{i:02d}", c) for i, c in enumerate(chunks)]

    partitions = []
    for key, grp in gdf.groupby(partition_col):
        fids = grp[feature_id].tolist()
        n_chunks = math.ceil(len(fids) / CHUNK_SIZE)
        if n_chunks > 1:
            for ci, chunk_fids in enumerate(chunk(fids, n_chunks)):
                partitions.append((f"{key}{CHUNK_SUFFIXES[ci]}", chunk_fids))
        else:
            partitions.append((str(key), fids))
    return partitions


def to_fc(df, feature_id):
    """Build an ee.FeatureCollection from an EPSG:4326 GeoDataFrame subset."""
    feats = [
        ee.Feature(ee.Geometry(geom.__geo_interface__), {feature_id: fid})
        for fid, geom in zip(df[feature_id], df.geometry)
    ]
    return ee.FeatureCollection(feats)


def load_min_yr_mask(asset):
    """IrrMapper multi-year mask (irrigated >=5 years over the full record).

    Prefers the pre-computed asset — the multi-image live graph can trigger
    'too many bands' errors in toBands() exports.
    """
    try:
        img = ee.Image(asset)
        img.getInfo()
        print(f"Using min-yr mask asset {asset}")
        return img
    except ee.ee_exception.EEException:
        print(f"WARNING: cannot read {asset}; computing min-yr mask live")
        coll = ee.ImageCollection(IRR)
        return coll.select("classification").map(lambda i: i.lt(1)).sum().gte(5)


def detect_irr_max_year():
    """Set IRR_MAX_YEAR to the latest year present in the IrrMapper collection."""
    global IRR_MAX_YEAR
    idx = ee.ImageCollection(IRR).aggregate_array("system:index").getInfo()
    IRR_MAX_YEAR = max(int(i[-4:]) for i in idx if i[-4:].isdigit())
    print(f"IrrMapper classifications available through {IRR_MAX_YEAR}")


def irr_images(year):
    """Single-year IrrMapper classification mosaic (capped at the latest year)."""
    irr_year = min(year, IRR_MAX_YEAR)
    return (
        ee.ImageCollection(IRR)
        .filterDate(f"{irr_year}-01-01", f"{irr_year}-12-31")
        .select("classification")
        .mosaic()
    )


def apply_mask(coll, mask_type, year, min_yr_mask):
    """Mask a single-band collection per-image (before toBands)."""
    if mask_type == "no_mask":
        return coll
    irr = irr_images(year)
    if mask_type == "irr":
        mask = min_yr_mask.updateMask(irr.lt(1))
    elif mask_type == "inv_irr":
        mask = irr.gt(0)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    return coll.map(lambda img, _m=mask: img.updateMask(_m))


def export_wide(coll, fc, feature_id, desc, fn_prefix, bucket, date_columns=False):
    """Export one wide CSV: rows=fields, columns=sorted scene/date ids.

    Returns 1 if an export task started, 0 if the collection was empty.
    """
    scenes = _retry(lambda: coll.aggregate_histogram("system:index").getInfo(), desc)
    band_names = sorted(scenes.keys())
    if not band_names:
        print(f"  {desc}: no images, skipping")
        return 0

    cols = band_names
    if date_columns:  # strict YYYYMMDD columns for daily collections
        dates = [re.search(r"\d{8}", b) for b in band_names]
        if all(dates) and len({d.group() for d in dates}) == len(dates):
            cols = [d.group() for d in dates]

    bands = coll.toBands().rename(cols)
    data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30, tileScale=8)
    _retry(
        lambda: export_table(
            data,
            desc=desc,
            selectors=[feature_id] + cols,
            dest="bucket",
            bucket=bucket,
            fn_prefix=fn_prefix,
        ),
        desc,
    )
    return 1


def _normalize_etf(model, image):
    """Per-model band selection/scaling to a uniform 'etf' band (clamped 0-2)."""
    if model in DIRECT_ETF_MODELS:
        etf = image.select("et_fraction").divide(10000).clamp(0, 2).rename("etf")
    elif model in ENSEMBLE_MODELS:
        etf = image.select("et_ensemble_mad").divide(10000).clamp(0, 2).rename("etf")
    elif model in ET_BAND_MODELS:
        etf = image.select("et").divide(1000).divide(image.select("eto")).clamp(0, 2).rename("etf")
    else:
        raise ValueError(f"Unknown model: {model}")
    return ee.Image(etf.copyProperties(image, ["system:time_start", "system:index"]))


def etf_collection(model, year, fc):
    """OpenET v2.1 scene collection normalized to ETf for one model-year."""
    coll = (
        ee.ImageCollection(OPENET_SOURCES[model])
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(fc.geometry())
    )
    if model in ET_BAND_MODELS:
        # Join daily refET by date; model time_start is overpass time, refET midnight
        refet = ee.ImageCollection(REFET).filterDate(f"{year}-01-01", f"{year}-12-31").select("eto")
        filt = ee.Filter.maxDifference(
            difference=86400000, leftField="system:time_start", rightField="system:time_start"
        )
        joined = ee.ImageCollection(ee.Join.saveFirst("refet_match").apply(coll, refet, filt))
        coll = joined.map(lambda img: img.addBands(ee.Image(img.get("refet_match")).select("eto")))
    return coll.map(lambda img, _m=model: _normalize_etf(_m, img))


def run_etf(fc, label, args, min_yr_mask, years):
    n = 0
    for model in args.model_list:
        for mask_type in args.mask_list:
            for year in years:
                coll = apply_mask(etf_collection(model, year, fc), mask_type, year, min_yr_mask)
                stem = f"{model}_etf_{mask_type}_{year}"
                n += export_wide(
                    coll,
                    fc,
                    args.feature_id,
                    desc=f"{label}_{stem}",
                    fn_prefix=f"{args.file_prefix}/{label}/etf/{mask_type}/{stem}",
                    bucket=args.bucket,
                )
    return n


def run_ndvi(fc, label, args, min_yr_mask, years):
    n = 0
    for instrument in args.instrument_list:
        if instrument == "landsat":
            source, subdir, tag, inst_years = landsat_masked, "ndvi", "ndvi", years
        else:
            source = sentinel2_masked
            subdir, tag = "ndvi/sentinel", "ndvi_sentinel"
            inst_years = [y for y in years if y >= SENTINEL_START_YR]
        for mask_type in args.mask_list:
            for year in inst_years:
                coll = (
                    source(year, fc)
                    .select(["NIR_H", "RED_H"])
                    .map(lambda img: img.normalizedDifference(["NIR_H", "RED_H"]))
                )
                coll = apply_mask(coll, mask_type, year, min_yr_mask)
                stem = f"{tag}_{mask_type}_{year}"
                n += export_wide(
                    coll,
                    fc,
                    args.feature_id,
                    desc=f"{label}_{stem}",
                    fn_prefix=f"{args.file_prefix}/{label}/{subdir}/{mask_type}/{stem}",
                    bucket=args.bucket,
                )
    return n


def run_eto(fc, label, args, years):
    n = 0
    for year in years:
        coll = ee.ImageCollection(REFET).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        n += export_wide(
            coll.select("eto"),
            fc,
            args.feature_id,
            desc=f"{label}_eto_{year}",
            fn_prefix=f"{args.file_prefix}/{label}/met/eto/eto_{year}",
            bucket=args.bucket,
            date_columns=True,
        )
    return n


def run_swe(fc, label, args, years):
    n = 0
    for year in years:
        coll = ee.ImageCollection(SNODAS).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        n += export_wide(
            coll.select("SWE"),
            fc,
            args.feature_id,
            desc=f"{label}_swe_{year}",
            fn_prefix=f"{args.file_prefix}/{label}/snow/snodas/extracts/swe_{year}",
            bucket=args.bucket,
            date_columns=True,
        )
    return n


def run_soils(fc, label, args):
    get_ssurgo(
        fc,
        desc=f"ssurgo_{label}",
        selector=args.feature_id,
        dest="bucket",
        bucket=args.bucket,
        file_prefix=f"{args.file_prefix}/{label}",
    )
    return 1


def run_props(fc, label, args):
    """Landcover (MODIS/GLC10 mode) and per-year IrrMapper irrigation fraction."""
    get_landcover(
        fc,
        desc=f"landcover_{label}",
        selector=args.feature_id,
        dest="bucket",
        bucket=args.bucket,
        file_prefix=f"{args.file_prefix}/{label}",
    )
    get_irrigation(
        fc,
        desc=f"irr_{label}",
        selector=args.feature_id,
        lanid=False,
        dest="bucket",
        bucket=args.bucket,
        file_prefix=f"{args.file_prefix}/{label}",
        start_year=IRR_START_YR,
        end_year=IRR_MAX_YEAR,
    )
    return 2


def export_min_yr_mask(shapefile, asset):
    """One-time export of the IrrMapper min-yr mask over the shapefile bounds."""
    coll = ee.ImageCollection(IRR)
    mask = coll.select("classification").map(lambda i: i.lt(1)).sum().gte(5).toByte()
    bounds = gpd.read_file(shapefile, engine="fiona").to_crs(4326).total_bounds
    task = ee.batch.Export.image.toAsset(
        image=mask,
        description="irr_min_yr_mask",
        assetId=asset,
        region=ee.Geometry.Rectangle(list(bounds)),
        scale=30,
        maxPixels=1e13,
    )
    task.start()
    print(f"Exporting min-yr mask to {asset} — task ID: {task.id}")


def main():
    parser = argparse.ArgumentParser(
        description="SWIM NWI extraction: OpenET ETf, Landsat NDVI, bias-corrected "
        "GridMET ETo, SNODAS SWE, and SSURGO soils via Earth Engine"
    )
    parser.add_argument("--shapefile", required=True, help="Fields shapefile (any CRS)")
    parser.add_argument("--feature-id", default="NWI_ID", help="Unique field ID column")
    parser.add_argument("--partition-col", default="FIPS", help="Column to partition exports by")
    parser.add_argument("--project", required=True, help="EE cloud project ID")
    parser.add_argument("--key-file", default=None, help="Optional EE service-account JSON key")
    parser.add_argument("--bucket", required=True, help="GCS bucket for exports")
    parser.add_argument("--file-prefix", default="nv", help="Path prefix within the bucket")
    parser.add_argument(
        "--targets", default="all", help=f"Comma-separated from {TARGETS}, or 'all'"
    )
    parser.add_argument(
        "--models",
        default=",".join(OPENET_SOURCES),
        help="Comma-separated OpenET models for the etf target",
    )
    parser.add_argument("--mask-types", default="irr,inv_irr", help="irr, inv_irr, and/or no_mask")
    parser.add_argument(
        "--ndvi-instruments",
        default="landsat,sentinel",
        help="Comma-separated from landsat,sentinel (sentinel starts 2017)",
    )
    parser.add_argument(
        "--start-yr",
        type=int,
        default=1985,
        help="Landsat-NDVI/ETo start year (NWI POR; ETf/SWE/Sentinel clamp to "
        "their own coverage starts, so 1985 spawns no empty tasks)",
    )
    parser.add_argument("--end-yr", type=int, default=2025)
    parser.add_argument("--years", default=None, help="Comma-separated years (overrides start/end)")
    parser.add_argument("--fips", default=None, help="Run only these comma-separated FIPS codes")
    parser.add_argument(
        "--partitions", default=None, help="Run only these partition labels; overrides --fips"
    )
    parser.add_argument("--min-yr-asset", default=IRR_MIN_YR_ASSET)
    parser.add_argument(
        "--export-mask",
        action="store_true",
        help="Export the min-yr mask to --min-yr-asset and exit",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the export plan and exit")
    args = parser.parse_args()

    args.mask_list = [m.strip() for m in args.mask_types.split(",")]
    args.model_list = [m.strip() for m in args.models.split(",")]
    args.instrument_list = [i.strip() for i in args.ndvi_instruments.split(",")]
    targets = TARGETS if args.targets == "all" else [t.strip() for t in args.targets.split(",")]
    unknown = (
        set(targets) - set(TARGETS)
        | set(args.model_list) - set(OPENET_SOURCES)
        | set(args.instrument_list) - {"landsat", "sentinel"}
    )
    if unknown:
        sys.exit(f"Unknown targets/models: {sorted(unknown)}")

    years = (
        [int(y) for y in args.years.split(",")]
        if args.years
        else list(range(args.start_yr, args.end_yr + 1))
    )
    etf_years = [y for y in years if y >= ETF_START_YR]
    swe_years = [y for y in years if y >= SWE_START_YR]

    sys.setrecursionlimit(5000)
    if not args.dry_run or args.export_mask:
        if args.key_file:
            creds = ee.ServiceAccountCredentials(None, args.key_file)
            ee.Initialize(creds, project=args.project)
        else:
            ee.Initialize(project=args.project)

    if args.export_mask:
        export_min_yr_mask(args.shapefile, args.min_yr_asset)
        sys.exit(0)

    gdf = load_shapefile(args.shapefile, args.feature_id)  # EPSG:4326, indexed by ID
    dupes = gdf.loc[gdf[args.feature_id].duplicated(keep=False), args.feature_id].tolist()
    if dupes:
        print(f"WARNING: duplicated {args.feature_id} values (kept as-is): {sorted(set(dupes))}")

    partitions = partition_fields(gdf, args.feature_id, args.partition_col)
    if args.partitions:
        keep = {p.strip() for p in args.partitions.split(",")}
        partitions = [(lbl, fids) for lbl, fids in partitions if lbl in keep]
    elif args.fips:
        keep = {f.strip() for f in args.fips.split(",")}
        partitions = [(lbl, fids) for lbl, fids in partitions if lbl.rstrip(CHUNK_SUFFIXES) in keep]

    sentinel_years = [y for y in years if y >= SENTINEL_START_YR]
    per_partition = {
        "etf": len(args.model_list) * len(args.mask_list) * len(etf_years),
        "ndvi": len(args.mask_list)
        * sum(len(years) if i == "landsat" else len(sentinel_years) for i in args.instrument_list),
        "eto": len(years),
        "swe": len(swe_years),
        "soils": 1,
        "props": 2,
    }
    plan = {t: len(partitions) * per_partition[t] for t in targets}
    print(f"{len(partitions)} partitions, {len(gdf)} fields -> {sum(plan.values())} export tasks")
    for t in targets:
        print(f"  {t}: {plan[t]}")
    if args.dry_run:
        for lbl, fids in partitions:
            print(f"  {lbl}: {len(fids)} fields")
        sys.exit(0)

    min_yr_mask = None
    if {"etf", "ndvi", "props"} & set(targets):
        detect_irr_max_year()
    if {"etf", "ndvi"} & set(targets):
        min_yr_mask = load_min_yr_mask(args.min_yr_asset)

    started = 0
    for label, fids in partitions:
        print(f"\n=== {label} ({len(fids)} fields) ===", flush=True)
        fc = to_fc(gdf[gdf[args.feature_id].isin(fids)], args.feature_id)
        if "etf" in targets:
            started += run_etf(fc, label, args, min_yr_mask, etf_years)
        if "ndvi" in targets:
            started += run_ndvi(fc, label, args, min_yr_mask, years)
        if "eto" in targets:
            started += run_eto(fc, label, args, years)
        if "swe" in targets:
            started += run_swe(fc, label, args, swe_years)
        if "soils" in targets:
            started += run_soils(fc, label, args)
        if "props" in targets:
            started += run_props(fc, label, args)

    print(f"\nStarted {started} export tasks to gs://{args.bucket}/{args.file_prefix}/")


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
