# DEPRECATED — this script is scheduled for removal.
# The ETf extraction logic has moved to swim-tools/utils/openet_etf.py.
# Montana SID-specific extraction should use swim-mtdnrc/src/swim_mtdnrc/extraction/sid_etf.py.
#
# Original description:
# ETf extraction for Montana SID fields 2016-2025 using OpenET v2.1 source collections.
# Outputs per-county CSVs to GCS (same chunking/CLI pattern as sid_ndvi.py).

import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd

from swimrs.data_extraction.ee.common import export_table, shapefile_to_feature_collection
from swimrs.data_extraction.ee.ee_utils import is_authorized

WAIT_MINUTES = 10
MAX_RETRIES = 6

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
FEATURE_ID = "FID"
SHAPEFILE = (
    "/nas/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024_aea.shp"
)

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

# Reference ET for converting ET -> ETf (geesebal, disalexi, ptjpl)
REFET_COLLECTION = "projects/openet/assets/reference_et/conus/gridmet/daily/v1"

# Models where the band is already ETf (et_fraction / 10000)
DIRECT_ETF_MODELS = {"ssebop", "sims", "eemetric"}

# Models where the band is ET (et / 1000) and must be divided by ETo (mm, not scaled)
ET_BAND_MODELS = {"geesebal", "disalexi", "ptjpl"}

# Ensemble model: et_ensemble_mad / 10000
ENSEMBLE_MODELS = {"ensemble"}

# IrrMapper ends at 2023 — cap year for 2024-2025
IRR_MAX_YEAR = 2023


def _stamp_calendar_day(img):
    """Add a ``cal_day`` property (UTC yyyy-MM-dd) for date-keyed joins."""
    return img.set("cal_day", ee.Date(img.get("system:time_start")).format("yyyy-MM-dd"))


def _normalize_etf(model, image):
    """Apply per-model band selection and scaling to produce uniform ETf.

    Returns a single-band 'etf' image (float, clamped 0-2) with system:time_start
    and system:index copied from the source image.

    For ET_BAND_MODELS (disalexi, geesebal, ptjpl), the image must already have
    an 'eto' band attached via join before calling this function.
    """
    if model in DIRECT_ETF_MODELS:
        etf = image.select("et_fraction").divide(10000).clamp(0, 2).rename("etf")
    elif model in ENSEMBLE_MODELS:
        etf = image.select("et_ensemble_mad").divide(10000).clamp(0, 2).rename("etf")
    elif model in ET_BAND_MODELS:
        et_img = image.select("et").divide(1000)
        eto_img = image.select("eto")
        etf = et_img.divide(eto_img).clamp(0, 2).rename("etf")
    else:
        raise ValueError(f"Unknown model: {model}")

    return ee.Image(etf.copyProperties(image, ["system:time_start", "system:index"]))


def extract_etf(
    feature_coll,
    irr_coll,
    irr_min_yr_mask,
    model="ensemble",
    mask_type="irr",
    start_yr=2016,
    end_yr=2025,
    years=None,
    feature_id="FID",
    dest="bucket",
    bucket="wudr",
    file_prefix="sid",
):
    """Extract mean ETf per field from OpenET v2.1 source collections.

    When dest="bucket", starts one ee.batch export task per year to GCS.
    When dest="local", uses ee.data.computeFeatures for synchronous return.

    Parameters
    ----------
    model : str
        OpenET model name (e.g. 'ensemble', 'ssebop').
    years : list[int] or None
        Explicit list of years to process. Overrides start_yr/end_yr.
    """
    src_path = OPENET_SOURCES[model]

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

        coll = (
            ee.ImageCollection(src_path)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(feature_coll.geometry())
        )

        # For ET-band models, join reference ET (eto) by calendar day before normalizing.
        # DisALEXI/geeSEBAL/ptJPL time_start is Landsat overpass time; refET is midnight.
        # A timestamp-difference join matches both the same-day and next-day refET
        # images, so the join key is the formatted date.
        if model in ET_BAND_MODELS:
            refet = (
                ee.ImageCollection(REFET_COLLECTION)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("eto")
                .map(_stamp_calendar_day)
            )
            coll = coll.map(_stamp_calendar_day)
            filt = ee.Filter.equals(leftField="cal_day", rightField="cal_day")
            joined = ee.ImageCollection(ee.Join.saveFirst("refet_match").apply(coll, refet, filt))
            coll = joined.map(
                lambda img: img.addBands(ee.Image(img.get("refet_match")).select("eto"))
            )

        # Normalize ETf per image and apply mask
        if mask_type == "irr":
            coll = coll.map(
                lambda x, _m=irr_mask, _mdl=model: _normalize_etf(_mdl, x).updateMask(_m)
            )
        elif mask_type == "inv_irr":
            coll = coll.map(
                lambda x, _i=irr, _mdl=model: _normalize_etf(_mdl, x).updateMask(_i.gt(0))
            )
        else:
            coll = coll.map(lambda x, _mdl=model: _normalize_etf(_mdl, x))

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
        print(f"  {year}: {len(band_names)} scenes ({model})")
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
            desc = f"{model}_etf_{mask_type}_{year}"
            selectors = [feature_id] + band_names
            for attempt in range(MAX_RETRIES):
                try:
                    export_table(
                        data,
                        desc=desc,
                        selectors=selectors,
                        dest="bucket",
                        bucket=bucket,
                        fn_prefix=f"{file_prefix}/etf/{mask_type}/{desc}",
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


CHUNK_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SID ETf extraction (OpenET v2.1)")
    parser.add_argument("--counties", type=str, default=None, help="Comma-separated county numbers")
    parser.add_argument("--chunks", type=int, default=1, help="Split each county into N groups")
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Run only this chunk (0-indexed, e.g. 2 for 'c')",
    )
    parser.add_argument(
        "--models", type=str, default="ensemble", help="Comma-separated model names"
    )
    parser.add_argument(
        "--mask-types", type=str, default="irr,inv_irr", help="Comma-separated mask types"
    )
    parser.add_argument("--start-yr", type=int, default=2016)
    parser.add_argument("--end-yr", type=int, default=2025)
    parser.add_argument(
        "--years", type=str, default=None, help="Comma-separated years (overrides start/end)"
    )
    parser.add_argument("--dest", choices=["bucket", "local"], default="bucket")
    parser.add_argument("--bucket", type=str, default="wudr")
    parser.add_argument("--project", type=str, default="ee-hoylman", help="EE project ID")
    args = parser.parse_args()

    year_list = [int(y) for y in args.years.split(",")] if args.years else None
    mask_types = [m.strip() for m in args.mask_types.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    root = "/data/ssd2/swim/sid"
    os.makedirs(root, exist_ok=True)
    sys.setrecursionlimit(5000)

    is_authorized(args.project)

    irr_coll = ee.ImageCollection(IRR)
    remap = irr_coll.filterDate("1987-01-01", "2024-12-31").select("classification")
    irr_min_yr_mask = remap.map(lambda img: img.lt(1)).sum().gte(5)
    print("Computed irr_min_yr_mask (live)")

    gdf = gpd.read_file(SHAPEFILE)
    county_fids = gdf.groupby("COUNTY_NO")[FEATURE_ID].apply(list).to_dict()

    if args.counties:
        selected = {int(c.strip()) for c in args.counties.split(",")}
        county_fids = {k: v for k, v in county_fids.items() if k in selected}

    for county_no, fids in county_fids.items():
        county = f"{county_no:03d}"
        name = gdf.loc[gdf["COUNTY_NO"] == county_no, "COUNTYNAME"].iloc[0]

        if args.chunks > 1:
            chunks = _chunk_list(fids, args.chunks)
        else:
            chunks = [fids]

        for ci, chunk_fids in enumerate(chunks):
            if args.chunk_index is not None and ci != args.chunk_index:
                continue

            suffix = CHUNK_SUFFIXES[ci] if len(chunks) > 1 else ""
            label = f"{county}{suffix}"

            for model in models:
                for mask_type in mask_types:
                    print(
                        f"\n=== {label} ({name}, {len(chunk_fids)} fields) "
                        f"model={model} mask={mask_type} ==="
                    )

                    fc = shapefile_to_feature_collection(SHAPEFILE, FEATURE_ID, select=chunk_fids)

                    start_time = time.time()
                    result = extract_etf(
                        fc,
                        irr_coll,
                        irr_min_yr_mask,
                        model=model,
                        mask_type=mask_type,
                        start_yr=args.start_yr,
                        end_yr=args.end_yr,
                        years=year_list,
                        feature_id=FEATURE_ID,
                        dest=args.dest,
                        bucket=args.bucket,
                        file_prefix=f"sid/{label}",
                    )
                    elapsed = time.time() - start_time

                    if result is not None:
                        out_csv = os.path.join(root, f"{label}_{model}_etf_{mask_type}.csv")
                        result.to_csv(out_csv)
                        print(
                            f"  {result.shape[0]} fields x {result.shape[1]} scenes "
                            f"in {elapsed:.1f}s -> {out_csv}"
                        )
                    else:
                        print(f"  Export tasks submitted in {elapsed:.1f}s")

# ========================= EOF ====================================================================
