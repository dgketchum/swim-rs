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
#   {prefix}/{label}/properties/cdl_{label}.csv                 USDA CDL crop class per year (2008+)
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
import subprocess
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
from swimrs.data_extraction.ee.ee_props import (
    get_cdl,
    get_irrigation,
    get_landcover,
    get_ssurgo,
)
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

TARGETS = ["etf", "ndvi", "eto", "swe", "soils", "props", "cdl"]
CHUNK_SIZE = 900  # fields per export partition (EE payload limit)
CHUNK_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"
WAIT_MINUTES = 10
MAX_RETRIES = 6


def list_existing(bucket, prefix):
    """Object names already under the destination prefix, for --resume.

    Shells out to gsutil rather than google-cloud-storage: gsutil picks up the
    gcloud user credentials already configured here, while the storage client
    wants application-default credentials that are not set up on this host.
    """
    uri = f"gs://{bucket}/{prefix}/**"
    try:
        out = subprocess.run(
            ["gsutil", "ls", "-r", uri], capture_output=True, text=True, timeout=3600
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"WARNING: could not list {uri} ({exc}); resume disabled")
        return set()
    if out.returncode != 0:
        tail = out.stderr.strip().splitlines()[-1:] or ["unknown error"]
        print(f"WARNING: gsutil ls failed ({tail[0]}); resume disabled")
        return set()
    head = f"gs://{bucket}/"
    return {
        ln.strip()[len(head) :]
        for ln in out.stdout.splitlines()
        if ln.strip().endswith(".csv") and ln.strip().startswith(head)
    }


def pending_tasks():
    """EE operations not yet finished. Returns None if the count is unavailable."""
    try:
        ops = ee.data.listOperations()
    except Exception as exc:  # noqa: BLE001 - listOperations raises broadly
        print(f"  WARNING: cannot read task list ({exc}); throttle skipped this round")
        return None
    return sum(1 for o in ops if o.get("metadata", {}).get("state") in ("PENDING", "RUNNING"))


class Gate:
    """Skip-existing and pending-task throttle for a many-thousand-task run.

    Two failure modes this exists for. First, EE caps how many tasks may be
    queued at once; a statewide run submits far more than that, and without
    backpressure the overflow surfaces as errors that _retry can only stall
    on (6 attempts x 10 min) before killing the run. Second, a run that dies
    at task 20,000 has no way to pick up, because nothing records what was
    already submitted -- so the destination prefix is used as the record.

    An existing object means a *completed* export. Tasks still queued or
    running have written nothing yet, so a resume started while the previous
    run's tasks are still draining will resubmit them; wait for the queue to
    empty before resuming.
    """

    def __init__(
        self, bucket, prefix, resume=False, max_pending=2500, poll_every=50, poll_seconds=120
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.max_pending = max_pending
        self.poll_every = poll_every
        self.poll_seconds = poll_seconds
        self.submitted = 0
        self.skipped = 0
        self.empty = 0
        self._since_poll = poll_every  # force a check before the first submission
        self.existing = list_existing(bucket, prefix) if resume else set()
        if resume:
            print(f"resume: {len(self.existing)} CSVs already under gs://{bucket}/{prefix}/")

    def done(self, name):
        """True if this export's output is already in the bucket."""
        if name in self.existing:
            self.skipped += 1
            return True
        return False

    def before_submit(self):
        """Block until the pending-task count is under the cap."""
        if not self.max_pending:
            return
        if self._since_poll < self.poll_every:
            self._since_poll += 1
            return
        self._since_poll = 1
        while True:
            n = pending_tasks()
            if n is None or n < self.max_pending:
                return
            print(
                f"  {n} tasks pending >= cap {self.max_pending}; waiting {self.poll_seconds}s",
                flush=True,
            )
            time.sleep(self.poll_seconds)


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


COVERAGE_SAMPLE = 40  # representative points probed per partition


def check_mask_coverage(gdf, partitions, feature_id, asset, sample=COVERAGE_SAMPLE):
    """Split partitions by whether the min-yr mask actually has data over them.

    A bounding-box test is not enough. The deployed asset spans a rectangle
    around Nevada, but that rectangle's eastern-California corner reads back
    null -- three CA counties sit inside the bbox with no mask data at all.
    Where the mask is defined it is 0/1 and never null, so probing a spread of
    fields per partition separates the two cases cleanly in one round trip.

    Fields outside the mask do not fail loudly: the collection is still
    non-empty, so the irr export writes a CSV of nulls, which is
    indistinguishable downstream from a genuinely dry field.
    """
    img = ee.Image(asset).rename("cov")
    feats, probed = [], {}
    for label, fids in partitions:
        sub = gdf[gdf[feature_id].isin(fids)]
        step = max(1, len(sub) // sample)
        pts = list(sub.geometry.representative_point().iloc[::step])[:sample]
        probed[label] = len(pts)
        feats += [ee.Feature(ee.Geometry.Point([p.x, p.y]), {"part": label}) for p in pts]

    try:
        res = _retry(
            lambda: img.reduceRegions(
                collection=ee.FeatureCollection(feats),
                # setOutputs, not the band name: a single-band reduceRegions
                # writes its result to "first" regardless of what the band is
                # called, and reading the wrong key looks exactly like no data.
                reducer=ee.Reducer.first().setOutputs(["cov"]),
                scale=30,
            ).getInfo(),
            "mask coverage probe",
        )
    except ee.ee_exception.EEException as exc:
        print(f"WARNING: mask coverage probe failed ({exc}); coverage check skipped")
        return partitions, []

    hits = dict.fromkeys(probed, 0)
    for f in res["features"]:
        if f["properties"].get("cov") is not None:
            hits[f["properties"]["part"]] += 1

    inside, outside = [], []
    for label, fids in partitions:
        frac = hits[label] / probed[label] if probed[label] else 0.0
        if frac == 1.0:
            inside.append((label, fids))
        else:
            outside.append((label, fids, frac))
    return inside, outside


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


def export_wide(coll, fc, feature_id, desc, fn_prefix, bucket, date_columns=False, gate=None):
    """Export one wide CSV: rows=fields, columns=sorted scene/date ids.

    Returns 1 if an export task started, 0 if it was skipped or the collection
    was empty.
    """
    if gate is not None and gate.done(f"{fn_prefix}.csv"):
        return 0
    # The scene list costs a synchronous EE round trip, so the resume check
    # goes first: a skipped export should cost nothing at all.
    scenes = _retry(lambda: coll.aggregate_histogram("system:index").getInfo(), desc)
    band_names = sorted(scenes.keys())
    if not band_names:
        print(f"  {desc}: no images, skipping")
        if gate is not None:
            gate.empty += 1
        return 0

    cols = band_names
    if date_columns:  # strict YYYYMMDD columns for daily collections
        dates = [re.search(r"\d{8}", b) for b in band_names]
        if all(dates) and len({d.group() for d in dates}) == len(dates):
            cols = [d.group() for d in dates]

    bands = coll.toBands().rename(cols)
    data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30, tileScale=8)
    if gate is not None:
        gate.before_submit()
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
    if gate is not None:
        gate.submitted += 1
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


def run_etf(fc, label, args, min_yr_mask, years, gate=None):
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
                    gate=gate,
                )
    return n


def run_ndvi(fc, label, args, min_yr_mask, years, gate=None):
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
                    gate=gate,
                )
    return n


def run_eto(fc, label, args, years, gate=None):
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
            gate=gate,
        )
    return n


def run_swe(fc, label, args, years, gate=None):
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
            gate=gate,
        )
    return n


def run_soils(fc, label, args, gate=None):
    if gate is not None and gate.done(f"{args.file_prefix}/{label}/properties/ssurgo_{label}.csv"):
        return 0
    if gate is not None:
        gate.before_submit()
        gate.submitted += 1
    get_ssurgo(
        fc,
        desc=f"ssurgo_{label}",
        selector=args.feature_id,
        dest="bucket",
        bucket=args.bucket,
        file_prefix=f"{args.file_prefix}/{label}",
    )
    return 1


def run_props(fc, label, args, gate=None):
    """Landcover (MODIS/GLC10 mode) and per-year IrrMapper irrigation fraction."""
    n = 0
    base = f"{args.file_prefix}/{label}/properties"
    if gate is not None and gate.done(f"{base}/landcover_{label}.csv"):
        pass
    else:
        if gate is not None:
            gate.before_submit()
            gate.submitted += 1
        n += 1
        get_landcover(
            fc,
            desc=f"landcover_{label}",
            selector=args.feature_id,
            dest="bucket",
            bucket=args.bucket,
            file_prefix=f"{args.file_prefix}/{label}",
        )
    if gate is not None and gate.done(f"{base}/irr_{label}.csv"):
        return n
    if gate is not None:
        gate.before_submit()
        gate.submitted += 1
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
    return n + 1


def run_cdl(fc, label, args, gate=None):
    """USDA CDL crop class per year, 2008 through the latest published year.

    Feeds the cdl_cultivated override in the container exporter, which can
    rescue a unit from perennial mechanics but never push one into them. Note
    get_cdl resolves the year list with its own getInfo, so this costs one
    extra round trip per partition.
    """
    if gate is not None and gate.done(f"{args.file_prefix}/{label}/properties/cdl_{label}.csv"):
        return 0
    if gate is not None:
        gate.before_submit()
        gate.submitted += 1
    get_cdl(
        fc,
        desc=f"cdl_{label}",
        selector=args.feature_id,
        dest="bucket",
        bucket=args.bucket,
        file_prefix=f"{args.file_prefix}/{label}",
    )
    return 1


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
    parser.add_argument(
        "--file-prefix",
        default="nwi/data",
        help="Path prefix within the bucket. The default is the canonical NWI "
        "location that nwi_build_container.py reads; the older 'nv' prefix holds "
        "a partial Feb 2026 pull whose chunk labels no longer match this shapefile",
    )
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
    parser.add_argument(
        "--exclude-fips",
        default=None,
        help="Drop these comma-separated FIPS codes, applied after --fips/--partitions",
    )
    parser.add_argument("--min-yr-asset", default=IRR_MIN_YR_ASSET)
    parser.add_argument(
        "--uncovered",
        choices=["skip", "fail", "include"],
        default="skip",
        help="What to do with partitions reaching outside the min-yr mask asset: "
        "skip them (default), abort, or export them anyway and accept null irr rows",
    )
    parser.add_argument(
        "--export-mask",
        action="store_true",
        help="Export the min-yr mask to --min-yr-asset and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip exports whose CSV is already in the bucket (lists the prefix first). "
        "Only completed exports leave objects, so let a previous run's queue drain first.",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=2500,
        help="Wait rather than submit when this many EE tasks are pending/running (0 disables)",
    )
    parser.add_argument(
        "--poll-every",
        type=int,
        default=50,
        help="Submissions between pending-task checks",
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
    # Initialize even for --dry-run: the plan is only truthful if the mask
    # coverage check has run, and that needs one read of the asset bounds.
    # A dry run still submits nothing.
    if True:
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

    # Applied last so it composes with the include filters above, and so the
    # remaining scope stays whatever the mask covers rather than a hand-kept
    # list that goes stale when the mask is re-exported.
    if args.exclude_fips:
        drop = {f.strip() for f in args.exclude_fips.split(",")}
        partitions = [
            (lbl, fids) for lbl, fids in partitions if lbl.rstrip(CHUNK_SUFFIXES) not in drop
        ]

    if {"etf", "ndvi"} & set(targets) and args.uncovered != "include":
        partitions, outside = check_mask_coverage(
            gdf, partitions, args.feature_id, args.min_yr_asset
        )
        if outside:
            n_out = sum(len(f) for _, f, _ in outside)
            print(
                f"{len(outside)} partitions ({n_out} fields) are not covered by the "
                f"min-yr mask {args.min_yr_asset}:"
            )
            for label, fids, frac in sorted(outside, key=lambda r: r[2]):
                print(f"  {label}: {len(fids)} fields, mask present at {frac:.0%} of probes")
            if args.uncovered == "fail":
                sys.exit(
                    "Aborting: re-export the mask over the wider bounds "
                    "(--export-mask --min-yr-asset ...), or pass --uncovered skip"
                )
            print("  skipped; their irr exports would be null-filled, not empty\n")

    sentinel_years = [y for y in years if y >= SENTINEL_START_YR]
    per_partition = {
        "etf": len(args.model_list) * len(args.mask_list) * len(etf_years),
        "ndvi": len(args.mask_list)
        * sum(len(years) if i == "landsat" else len(sentinel_years) for i in args.instrument_list),
        "eto": len(years),
        "swe": len(swe_years),
        "soils": 1,
        "props": 2,
        "cdl": 1,
    }
    plan = {t: len(partitions) * per_partition[t] for t in targets}
    # Count fields in the *selected* partitions: with --fips/--partitions the
    # shapefile total is not what this run covers, and that header line is what
    # a reader checks the request against.
    n_fields = sum(len(f) for _, f in partitions)
    print(f"{len(partitions)} partitions, {n_fields} fields -> {sum(plan.values())} export tasks")
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

    gate = Gate(
        args.bucket,
        args.file_prefix,
        resume=args.resume,
        max_pending=args.max_pending,
        poll_every=args.poll_every,
    )

    started = 0
    for label, fids in partitions:
        print(f"\n=== {label} ({len(fids)} fields) ===", flush=True)
        fc = to_fc(gdf[gdf[args.feature_id].isin(fids)], args.feature_id)
        if "etf" in targets:
            started += run_etf(fc, label, args, min_yr_mask, etf_years, gate=gate)
        if "ndvi" in targets:
            started += run_ndvi(fc, label, args, min_yr_mask, years, gate=gate)
        if "eto" in targets:
            started += run_eto(fc, label, args, years, gate=gate)
        if "swe" in targets:
            started += run_swe(fc, label, args, swe_years, gate=gate)
        if "soils" in targets:
            started += run_soils(fc, label, args, gate=gate)
        if "props" in targets:
            started += run_props(fc, label, args, gate=gate)
        if "cdl" in targets:
            started += run_cdl(fc, label, args, gate=gate)

    print(f"\nStarted {started} export tasks to gs://{args.bucket}/{args.file_prefix}/")
    print(f"  skipped (already in bucket): {gate.skipped}")
    print(f"  skipped (collection empty):  {gate.empty}")


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
