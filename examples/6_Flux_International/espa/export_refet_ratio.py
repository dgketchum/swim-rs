"""Export ERA5-Land grass ETo and alfalfa ETr for the E2 reference-basis sidecar.

The ESPA SSEBop ETF product is alfalfa-referenced; E2 multiplies it by ERA5-Land grass
ETo. Converting the product to a grass basis needs ``ETr/ETo`` on every site-date, from
the same ERA5-Land hourly collection and the same local-day convention that produced the
E2 forcing (``swimrs.data_extraction.ee.ee_era5``). This script re-uses that module's
local-day helpers and asks one ``refetgee`` ``Daily`` object per local day for both
``.eto`` and ``.etr``, so the two series can never come from different windows.

One export task per (year, UTC-offset group): a table of per-day ``eto_YYYYMMDD`` /
``etr_YYYYMMDD`` feature means reduced at the same 150 m scale as the forcing export.
Stored E2 ETo is reproduced from the ``eto_`` bands as the sidecar's own validation.

Nothing here rewrites SWIM meteorology. Every task start is recorded in a ledger under
``--out-dir`` together with the code SHA, collection, scale, and shapefile hash.

Usage (dry run writes the request manifest only; no Earth Engine task is started):

    uv run python examples/6_Flux_International/espa/export_refet_ratio.py \
        --shapefile /path/to/cohort.fgb --years 2013 2025 --out-dir /path/qa --dry-run

    # one offset-year pilot
    ... --pilot 2015 utc_m08

    # everything not already present in --check-dir
    ... --check-dir /path/refet_ratio_era5land
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time

import ee
import geopandas as gpd
import pandas as pd

try:
    from openet.refetgee import Daily
except ImportError:  # pragma: no cover
    Daily = None

from swimrs.data_extraction.ee.ee_era5 import (
    _format_offset_suffix,
    _get_unique_offsets,
    _local_day_utc_bounds,
    _tag_features_with_utc_offset,
)
from swimrs.data_extraction.ee.ee_utils import as_ee_feature_collection
from swimrs.units import GEE_ERA5_LAND_HOURLY_DATASET

# Same reduction scale as sample_era5_land_variables_daily.
SCALE_M = 150
DEFAULT_BUCKET = "wudr"
DEFAULT_PREFIX = "6_Flux_International/remote_sensing/espa/refet_ratio_era5land"
FILE_STEM = "refet_ratio"


def local_utc_offset(lon: float) -> int:
    """Rounded solar-time offset, matching ``_tag_features_with_utc_offset`` (round(lon/15))."""
    return int(math.floor(lon / 15.0 + 0.5))


def days_in_year(year: int) -> list[dt.date]:
    d0, d1 = dt.date(year, 1, 1), dt.date(year + 1, 1, 1)
    return [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days)]


def year_selectors(feature_id: str, days: list[dt.date]) -> list[str]:
    """Deterministic export column order: id, then eto/etr pairs in date order."""
    cols = [feature_id]
    for d in days:
        ds = d.strftime("%Y%m%d")
        cols.extend([f"eto_{ds}", f"etr_{ds}"])
    return cols


def daily_refet_bands(hourly_for_day, day_str: str, daily_cls=Daily) -> tuple:
    """Return ``(eto_band, etr_band)`` from a single ``Daily.era5_land`` object.

    Both bands must come from one object so they share the hourly window and every
    derived intermediate (Rs, Rn, u2, ea, ...).
    """
    daily = daily_cls.era5_land(hourly_for_day)
    return daily.eto.rename(f"eto_{day_str}"), daily.etr.rename(f"etr_{day_str}")


def build_year_image(era5_hourly, year: int, utc_offset: int, daily_cls=Daily):
    """Multi-band image of eto_/etr_ for every local day of ``year`` at one UTC offset."""
    offset = ee.Number(utc_offset)
    bands = []
    for d in days_in_year(year):
        utc_start, utc_end = _local_day_utc_bounds(d, offset)
        hourly_for_day = era5_hourly.filterDate(utc_start, utc_end)
        bands.extend(daily_refet_bands(hourly_for_day, d.strftime("%Y%m%d"), daily_cls))
    return ee.Image(bands)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_request_manifest(
    shapefile: str, years: list[int], feature_id: str = "sid"
) -> pd.DataFrame:
    """One row per (year, UTC-offset group) with the sites it covers.

    Offsets are computed locally from centroid longitude with the same rounding as the
    EE tagging; the caller re-checks them against EE at run time. Duplicate feature IDs
    are an error, never collapsed.
    """
    gdf = gpd.read_file(shapefile, engine="fiona")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    if feature_id not in gdf.columns:
        raise ValueError(f"{shapefile} has no '{feature_id}' field")
    dup = gdf[feature_id][gdf[feature_id].duplicated()].tolist()
    if dup:
        raise ValueError(f"duplicate {feature_id} values in {shapefile}: {sorted(set(dup))}")
    gdf["utc_offset"] = [local_utc_offset(x) for x in gdf.geometry.centroid.x]

    rows = []
    for offset, grp in gdf.groupby("utc_offset"):
        sites = sorted(grp[feature_id].astype(str))
        suffix = _format_offset_suffix(int(offset))
        for year in years:
            rows.append(
                {
                    "year": int(year),
                    "utc_offset": int(offset),
                    "suffix": suffix,
                    "description": f"{FILE_STEM}_{year}_{suffix}",
                    "n_sites": len(sites),
                    "n_days": len(days_in_year(int(year))),
                    "sites": ";".join(sites),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "utc_offset"]).reset_index(drop=True)


def select_requests(
    manifest: pd.DataFrame,
    pilot: tuple[int, str] | None = None,
    check_dir: str | None = None,
) -> pd.DataFrame:
    """Rows to export: the single pilot row, or every row without a CSV in ``check_dir``."""
    out = manifest
    if pilot is not None:
        year, suffix = pilot
        out = out[(out["year"] == int(year)) & (out["suffix"] == suffix)]
        if out.empty:
            raise ValueError(f"pilot ({year}, {suffix}) is not in the request manifest")
    if check_dir:
        present = out["description"].map(
            lambda d: os.path.exists(os.path.join(check_dir, f"{d}.csv"))
        )
        out = out[~present]
    return out.reset_index(drop=True)


def start_exports(
    requests: pd.DataFrame,
    fc,
    feature_id: str,
    bucket: str,
    prefix: str,
    dry_run: bool,
    daily_cls=Daily,
) -> list[dict]:
    """Start one table export per request row; return the task ledger.

    With ``dry_run`` the ledger is built but no task is created or started.
    """
    era5_hourly = None if dry_run else ee.ImageCollection(GEE_ERA5_LAND_HOURLY_DATASET)
    ledger = []
    for row in requests.itertuples(index=False):
        entry = {
            "description": row.description,
            "year": int(row.year),
            "utc_offset": int(row.utc_offset),
            "n_sites": int(row.n_sites),
            "gcs_uri": f"gs://{bucket}/{prefix}/{row.description}.csv",
            "dry_run": bool(dry_run),
        }
        if dry_run:
            entry["task_id"] = None
            ledger.append(entry)
            continue

        fc_group = fc.filter(ee.Filter.eq("utc_offset_hours", int(row.utc_offset)))
        image = build_year_image(era5_hourly, int(row.year), int(row.utc_offset), daily_cls)
        reduced = image.reduceRegions(collection=fc_group, reducer=ee.Reducer.mean(), scale=SCALE_M)
        task = ee.batch.Export.table.toCloudStorage(
            collection=reduced,
            description=row.description,
            bucket=bucket,
            fileNamePrefix=f"{prefix}/{row.description}",
            fileFormat="CSV",
            selectors=year_selectors(feature_id, days_in_year(int(row.year))),
        )
        try:
            task.start()
        except ee.ee_exception.EEException as exc:
            print(f"{exc}; waiting 600 s before retrying {row.description}")
            time.sleep(600)
            task.start()
        entry["task_id"] = task.id
        entry["started_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        ledger.append(entry)
        print(f"started {row.description} -> {entry['gcs_uri']} (task {task.id})")
    return ledger


def _code_sha() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--years", nargs=2, type=int, metavar=("START", "END"), required=True)
    ap.add_argument("--out-dir", required=True, help="QA root for manifest/ledger/provenance")
    ap.add_argument("--feature-id", default="sid")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--check-dir", default=None, help="skip requests whose CSV already exists here")
    ap.add_argument("--pilot", nargs=2, metavar=("YEAR", "SUFFIX"), default=None)
    ap.add_argument("--dry-run", action="store_true", help="write manifest/ledger; start nothing")
    args = ap.parse_args(argv)

    if Daily is None:  # pragma: no cover
        raise ImportError("openet-refet-gee is required (uv sync --all-extras)")

    os.makedirs(args.out_dir, exist_ok=True)
    years = list(range(args.years[0], args.years[1] + 1))
    manifest = build_request_manifest(args.shapefile, years, args.feature_id)
    manifest.to_csv(os.path.join(args.out_dir, "refet_request_manifest.csv"), index=False)

    pilot = (int(args.pilot[0]), args.pilot[1]) if args.pilot else None
    requests = select_requests(manifest, pilot=pilot, check_dir=args.check_dir)
    print(
        f"{len(requests)} of {len(manifest)} request rows selected"
        f"{' (dry run)' if args.dry_run else ''}"
    )

    fc = None
    ee_offsets = None
    if not args.dry_run:
        ee.Initialize()
        fc = _tag_features_with_utc_offset(
            as_ee_feature_collection(args.shapefile, feature_id=args.feature_id)
        )
        ee_offsets = _get_unique_offsets(fc)
        local_offsets = sorted(manifest["utc_offset"].unique().tolist())
        if ee_offsets != local_offsets:
            raise RuntimeError(
                f"UTC offset groups differ: local {local_offsets} vs Earth Engine {ee_offsets}"
            )

    ledger = start_exports(requests, fc, args.feature_id, args.bucket, args.prefix, args.dry_run)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    provenance = {
        "generated_utc": stamp,
        "code_sha": _code_sha(),
        "script": os.path.abspath(__file__),
        "collection": GEE_ERA5_LAND_HOURLY_DATASET,
        "reducer": "mean",
        "scale_m": SCALE_M,
        "local_day_convention": "ee_era5._local_day_utc_bounds with round(lon/15) offset",
        "refet": "openet.refetgee.Daily.era5_land(hourly_for_day).eto / .etr, one object per day",
        "shapefile": os.path.abspath(args.shapefile),
        "shapefile_sha256": _sha256_file(args.shapefile),
        "feature_id": args.feature_id,
        "n_features": int(manifest.drop_duplicates("utc_offset")["n_sites"].sum()),
        "years": years,
        "utc_offsets_local": sorted(manifest["utc_offset"].unique().tolist()),
        "utc_offsets_ee": ee_offsets,
        "bucket": args.bucket,
        "prefix": args.prefix,
        "pilot": pilot,
        "dry_run": bool(args.dry_run),
        "tasks": ledger,
    }
    ledger_path = os.path.join(args.out_dir, f"refet_task_ledger_{stamp}.json")
    with open(ledger_path, "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"ledger -> {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
