"""Phase 6, step 3: verify the delivered ESPA repair products and freeze the raw tree (Gate G6).

Runs after ``espa_download_orders.py`` has marked every row of the repair manifest
``download_status = complete``. Nothing here reads ESPA; everything is checked against what is on
disk and what was approved:

* **checksums** — every tarball is re-hashed (md5 against the ``.md5`` sidecar ESPA shipped,
  independent of the downloader's own check) and fingerprinted (sha256) for the freeze;
* **identity** — for every product ID in the approved payload, the ``*_ETF.tif`` must exist and its
  filename, the ``LANDSAT_PRODUCT_ID`` / ``SPACECRAFT_ID`` / ``DATE_ACQUIRED`` in the delivered
  ``*_MTL.txt``, the sensor recorded in the scene-level manifest, the UTM zone requested in the
  payload, the nodata value and the chip bounds must all agree;
* **coverage** — the cohort polygon (``flux_crop_pub_66_150m.shp``, the geometry the extents were
  built from) is overlaid on each chip with the same ``rasterstats`` zonal call the extractor
  uses, so every unit gets a terminal category: ``delivered-valid-file``, ``delivered-nodata``,
  ``delivered-implausible``, ``checksum-fail``, ``missing-file`` or ``manifest-error`` (a product on
  disk that was never requested);
* **coverage projection** (user question 2026-09-07) — the container's PT-JPL capture dates for
  the 66-site cohort are compared with the SSEBop dates it holds now and with the dates the
  delivery adds, by year and by site;
* ``--freeze`` — after a passing gate, hashes every raw and extracted file into
  ``le07_repair_frozen_hashes.sha256`` and removes write permission from them.

Outputs (QA root): ``le07_delivery_ledger.csv`` (one row per requested or delivered product),
``le07_delivery_summary.json`` (counts and G6 gate booleans), ``le07_coverage_by_year.csv``,
``le07_coverage_by_site.csv``.

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase6_verify_delivery.py [--freeze]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import sys

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import mapping, shape

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_daily_basis_gate import container_table  # noqa: E402

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
REPAIR_TREE = os.path.join(DATA, "remote_sensing", "espa_le07_repair")
SCENE_MANIFEST = os.path.join(QA_ROOT, "le07_missing_only_manifest.csv")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
YEARS = (2013, 2025)

PRODUCT_RE = re.compile(
    r"^(?P<sensor>LT04|LT05|LE07|LC08|LC09)_L2\w{2}_(?P<pathrow>\d{6})_(?P<date>\d{8})_\d{8}_\d{2}_\w+$"
)
SPACECRAFT = {"LE07": "LANDSAT_7", "LC08": "LANDSAT_8", "LC09": "LANDSAT_9", "LT05": "LANDSAT_5"}
# same conventions as espa/espa_extract_etf.py and the container ingestor
ETF_SCALE_FACTOR = 0.0001
PLAUSIBLE_ETF_RANGE = (-0.1, 2.5)
EXPECTED_NODATA = -9999.0
INGEST_MIN_ETF = 0.05
BOUNDS_TOL_M = 30.0  # ESPA snaps the requested extent to its 30 m grid

# outcome ranking for a PT-JPL-only site-date with several candidate scenes (best first)
OUTCOME_RANK = [
    "new_valid",
    "recoverable_delivered_not_ingested",
    "new_valid_below_min_etf",
    "new_nodata",
    "new_implausible",
    "delivered_nodata",
    "native_below_min_etf",
    "not_delivered",
]


# --------------------------------------------------------------------------- hashing / parsing
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_product_id(pid: str) -> dict:
    m = PRODUCT_RE.match(pid)
    if not m:
        raise ValueError(f"not a Collection 2 product ID: {pid}")
    return m.groupdict()


def expected_epsg(payload: dict) -> int:
    utm = payload["projection"]["utm"]
    base = 32600 if utm["zone_ns"].lower().startswith("n") else 32700
    return base + int(utm["zone"])


def read_mtl(path: str) -> dict:
    """First occurrence of each key: the L2SP block precedes the L1 block in ESPA's MTL."""
    keys = ("LANDSAT_PRODUCT_ID", "SPACECRAFT_ID", "SENSOR_ID", "DATE_ACQUIRED", "UTM_ZONE")
    out: dict = {}
    with open(path) as fh:
        for line in fh:
            if "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            if k in keys and k not in out:
                out[k] = v.strip('"')
    return out


def bounds_ok(bounds, extents: dict, tol_m: float = BOUNDS_TOL_M) -> bool:
    """Delivered chip lies within the requested extent snapped to the 30 m grid."""
    return (
        bounds.left >= extents["west"] - tol_m
        and bounds.right <= extents["east"] + tol_m
        and bounds.bottom >= extents["south"] - tol_m
        and bounds.top <= extents["north"] + tol_m
        and (bounds.right - bounds.left) >= (extents["east"] - extents["west"]) - tol_m
        and (bounds.top - bounds.bottom) >= (extents["north"] - extents["south"]) - tol_m
    )


def inspect_etf(tif: str, geom_4326, cache: dict) -> dict:
    with rasterio.open(tif) as src:
        crs = src.crs
        arr = src.read(1)
        nodata = src.nodata
        info = {
            "epsg": crs.to_epsg(),
            "nodata": nodata,
            "dtype": src.dtypes[0],
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "chip_valid_frac": float((arr != nodata).mean()) if nodata is not None else np.nan,
        }
    key = str(crs)
    if key not in cache:
        cache[key] = mapping(gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(crs).iloc[0])
    stats = zonal_stats(cache[key], tif, stats=["count", "mean"], geojson_out=False)
    st = stats[0] if stats and stats[0] else {}
    count = int(st.get("count") or 0)
    mean = st.get("mean")
    info["site_pixel_count"] = count
    info["site_mean_etf"] = (
        float(mean) * ETF_SCALE_FACTOR if (count and mean is not None) else np.nan
    )
    return info


def categorize(rec: dict) -> str:
    if not rec["requested"]:
        return "manifest-error"
    if not rec["tarball_exists"] or not rec["etf_exists"]:
        return "missing-file"
    if not rec["md5_ok"]:
        return "checksum-fail"
    if rec["site_pixel_count"] == 0:
        return "delivered-nodata"
    m = rec["site_mean_etf"]
    if not (PLAUSIBLE_ETF_RANGE[0] <= m <= PLAUSIBLE_ETF_RANGE[1]):
        return "delivered-implausible"
    return "delivered-valid-file"


def identity_ok(rec: dict, expected_sensor_from_manifest: str | None) -> tuple[bool, str]:
    problems = []
    if rec["mtl_product_id"] and rec["mtl_product_id"] != rec["product_id"]:
        problems.append("mtl_product_id")
    if rec["mtl_spacecraft"] and rec["mtl_spacecraft"] != SPACECRAFT.get(rec["sensor"]):
        problems.append("spacecraft")
    if rec["mtl_date_acquired"] and rec["mtl_date_acquired"].replace("-", "") != rec["date"]:
        problems.append("date")
    if expected_sensor_from_manifest and expected_sensor_from_manifest != rec["sensor"]:
        problems.append("manifest_sensor")
    if rec["epsg"] is not None and rec["epsg"] != rec["epsg_expected"]:
        problems.append("epsg")
    if rec["nodata"] is not None and rec["nodata"] != EXPECTED_NODATA:
        problems.append("nodata")
    if rec["etf_exists"] and not rec["bounds_ok"]:
        problems.append("bounds")
    return (not problems, ";".join(problems))


# --------------------------------------------------------------------------- per site-year
def verify_site_year(row: pd.Series, geom_4326, scene_sensor: dict[str, str]) -> list[dict]:
    with open(row["payload_json"]) as fh:
        payload = json.load(fh)
    requested = []
    for block, spec in payload.items():
        if isinstance(spec, dict) and "inputs" in spec:
            requested.extend(spec["inputs"])
    extents = payload["image_extents"]
    epsg_exp = expected_epsg(payload)
    out_dir = row["output_dir"]
    raw_dir, ext_dir = os.path.join(out_dir, "raw"), os.path.join(out_dir, "extract")

    on_disk = set()
    if os.path.isdir(ext_dir):
        for name in os.listdir(ext_dir):
            if name.endswith("_ETF.tif"):
                on_disk.add(name[: -len("_ETF.tif")])
    cache: dict = {}
    records = []
    for pid in sorted(set(requested) | on_disk):
        ident = parse_product_id(pid)
        tar = os.path.join(raw_dir, f"{pid}.tar.gz")
        md5_side = os.path.join(raw_dir, f"{pid}.md5")
        etf = os.path.join(ext_dir, f"{pid}_ETF.tif")
        mtl = os.path.join(ext_dir, f"{pid}_MTL.txt")
        rec = {
            "site": row["site"],
            "year": int(row["year"]),
            "order_id": row["order_id"],
            "payload_id": f"{row['site']}_{row['year']}",
            "product_id": pid,
            "sensor": ident["sensor"],
            "pathrow": ident["pathrow"],
            "date": ident["date"],
            "requested": pid in requested,
            "in_scene_manifest": pid in scene_sensor,
            "tarball": tar if os.path.exists(tar) else "",
            "tarball_exists": os.path.exists(tar),
            "tarball_sha256": sha256_file(tar) if os.path.exists(tar) else "",
            "md5_expected": "",
            "md5_ok": False,
            "etf_tif": etf if os.path.exists(etf) else "",
            "etf_exists": os.path.exists(etf),
            "mtl_product_id": "",
            "mtl_spacecraft": "",
            "mtl_sensor_id": "",
            "mtl_date_acquired": "",
            "epsg": None,
            "epsg_expected": epsg_exp,
            "nodata": None,
            "dtype": "",
            "width": None,
            "height": None,
            "bounds_ok": False,
            "chip_valid_frac": np.nan,
            "site_pixel_count": 0,
            "site_mean_etf": np.nan,
        }
        if rec["tarball_exists"] and os.path.exists(md5_side):
            with open(md5_side) as fh:
                rec["md5_expected"] = fh.read().split()[0].strip().lower()
            rec["md5_ok"] = md5_file(tar) == rec["md5_expected"]
        if os.path.exists(mtl):
            m = read_mtl(mtl)
            rec["mtl_product_id"] = m.get("LANDSAT_PRODUCT_ID", "")
            rec["mtl_spacecraft"] = m.get("SPACECRAFT_ID", "")
            rec["mtl_sensor_id"] = m.get("SENSOR_ID", "")
            rec["mtl_date_acquired"] = m.get("DATE_ACQUIRED", "")
        if rec["etf_exists"]:
            info = inspect_etf(etf, geom_4326, cache)
            rec.update(
                {
                    "epsg": info["epsg"],
                    "nodata": info["nodata"],
                    "dtype": info["dtype"],
                    "width": info["width"],
                    "height": info["height"],
                    "bounds_ok": bounds_ok(info["bounds"], extents),
                    "chip_valid_frac": info["chip_valid_frac"],
                    "site_pixel_count": info["site_pixel_count"],
                    "site_mean_etf": info["site_mean_etf"],
                }
            )
        rec["category"] = categorize(rec)
        rec["identity_ok"], rec["identity_problems"] = identity_ok(rec, scene_sensor.get(pid))
        records.append(rec)
    return records


# --------------------------------------------------------------------------- coverage
def site_date_outcomes(ledger: pd.DataFrame, scenes: pd.DataFrame) -> pd.DataFrame:
    """One row per PT-JPL-only cohort site-date with its best outcome after the delivery."""
    delivered = ledger[ledger["requested"]].copy()
    cat_to_outcome = {
        "delivered-valid-file": "new_valid",
        "delivered-nodata": "new_nodata",
        "delivered-implausible": "new_implausible",
        "checksum-fail": "not_delivered",
        "missing-file": "not_delivered",
    }
    delivered["outcome"] = delivered["category"].map(cat_to_outcome)
    below = (delivered["outcome"] == "new_valid") & (delivered["site_mean_etf"] < INGEST_MIN_ETF)
    delivered.loc[below, "outcome"] = "new_valid_below_min_etf"
    # a scene can serve several neighbouring sites, so the unit key is (site, product_id)
    by_unit = delivered.set_index(["site", "product_id"])["outcome"]
    if not by_unit.index.is_unique:
        raise ValueError("ledger has duplicate (site, product_id) rows")

    sc = scenes.copy()
    unit_key = pd.MultiIndex.from_frame(sc[["site", "product_id"]])
    sc["outcome"] = np.where(
        sc["request"].astype(str) == "True",
        pd.Series(unit_key.map(by_unit), index=sc.index).fillna("not_delivered"),
        sc["status"].map(
            {
                "delivered_not_ingested": "recoverable_delivered_not_ingested",
                "delivered_nodata": "delivered_nodata",
                "native_below_min_etf": "native_below_min_etf",
            }
        ),
    )
    rank = {o: i for i, o in enumerate(OUTCOME_RANK)}
    sc["rank"] = sc["outcome"].map(rank)
    if sc["rank"].isna().any():
        bad = sc.loc[sc["rank"].isna(), ["status", "request", "outcome"]].drop_duplicates()
        raise ValueError(f"unranked outcomes:\n{bad}")
    best = sc.sort_values("rank").drop_duplicates(["site", "date"], keep="first")
    return best[["site", "date", "year", "sensor", "outcome"]].reset_index(drop=True)


def coverage_tables(
    table: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PT-JPL capture dates vs SSEBop dates before and after the delivery, by year and by site.

    ``table`` is the container view (site, date, ssebop_container, ptjpl, eto, year).
    """
    t = table.copy()
    t["has_ptjpl"] = np.isfinite(t["ptjpl"])
    t["has_ssebop"] = np.isfinite(t["ssebop_container"])
    o = outcomes.set_index(["site", "date"])["outcome"]
    t["outcome"] = pd.MultiIndex.from_frame(t[["site", "date"]]).map(o)
    t["adds_after_delivery"] = t["has_ptjpl"] & ~t["has_ssebop"] & (t["outcome"] == "new_valid")
    t["adds_after_recovery"] = (
        t["has_ptjpl"] & ~t["has_ssebop"] & (t["outcome"] == "recoverable_delivered_not_ingested")
    )

    def agg(g: pd.DataFrame) -> pd.Series:
        n_pt = int(g["has_ptjpl"].sum())
        paired_before = int((g["has_ptjpl"] & g["has_ssebop"]).sum())
        paired_delivery = paired_before + int(g["adds_after_delivery"].sum())
        paired_full = paired_delivery + int(g["adds_after_recovery"].sum())
        res = g.loc[g["has_ptjpl"] & ~g["has_ssebop"], "outcome"].value_counts()
        out = {
            "n_ptjpl_dates": n_pt,
            "n_ssebop_dates_now": int(g["has_ssebop"].sum()),
            "n_ssebop_only_now": int((g["has_ssebop"] & ~g["has_ptjpl"]).sum()),
            "paired_now": paired_before,
            "paired_after_delivery": paired_delivery,
            "paired_after_recovery": paired_full,
            "frac_paired_now": paired_before / n_pt if n_pt else np.nan,
            "frac_paired_after_delivery": paired_delivery / n_pt if n_pt else np.nan,
            "frac_paired_after_recovery": paired_full / n_pt if n_pt else np.nan,
        }
        for oc in OUTCOME_RANK:
            out[f"residual_{oc}"] = int(res.get(oc, 0))
        return pd.Series(out)

    by_year = t.groupby("year").apply(agg, include_groups=False).reset_index()
    by_site = t.groupby("site").apply(agg, include_groups=False).reset_index()
    return by_year, by_site


# --------------------------------------------------------------------------- freeze
def freeze_tree(orders_root: str, hashes_path: str) -> dict:
    """sha256 every raw and extracted file, write the list, remove write permission."""
    n_files, n_bytes = 0, 0
    with open(hashes_path, "w") as out:
        for site in sorted(os.listdir(orders_root)):
            for year in sorted(os.listdir(os.path.join(orders_root, site))):
                for sub in ("raw", "extract"):
                    d = os.path.join(orders_root, site, year, sub)
                    if not os.path.isdir(d):
                        continue
                    for name in sorted(os.listdir(d)):
                        p = os.path.join(d, name)
                        if not os.path.isfile(p):
                            continue
                        out.write(f"{sha256_file(p)}  {p}\n")
                        n_files += 1
                        n_bytes += os.path.getsize(p)
                        mode = os.stat(p).st_mode
                        os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return {"n_files": n_files, "n_bytes": n_bytes, "hashes": hashes_path}


# --------------------------------------------------------------------------- main
def cohort_geometries(path: str) -> dict:
    with fiona.open(path) as src:
        return {f["properties"]["sid"]: shape(f["geometry"]) for f in src}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repair-tree", default=REPAIR_TREE)
    ap.add_argument("--scene-manifest", default=SCENE_MANIFEST)
    ap.add_argument("--cohort", default=COHORT_66)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--out-dir", default=QA_ROOT)
    ap.add_argument(
        "--freeze", action="store_true", help="hash and write-protect raw+extract after a pass"
    )
    args = ap.parse_args(argv)

    manifest_path = os.path.join(args.repair_tree, "espa_manifest.csv")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    scenes = pd.read_csv(args.scene_manifest, dtype=str, keep_default_na=False)
    scene_sensor = scenes.set_index("product_id")["sensor"].to_dict()
    geoms = cohort_geometries(args.cohort)
    missing_geom = sorted(set(manifest["site"]) - set(geoms))
    if missing_geom:
        raise SystemExit(f"sites without cohort geometry: {missing_geom}")

    print(f"verifying {len(manifest)} site-years, {manifest['n_scenes'].astype(int).sum()} units")
    records = []
    for i, (_, row) in enumerate(manifest.iterrows(), 1):
        records.extend(verify_site_year(row, geoms[row["site"]], scene_sensor))
        if i % 25 == 0:
            print(f"  {i}/{len(manifest)} site-years, {len(records)} products")
    ledger = pd.DataFrame(records)
    ledger_path = os.path.join(args.out_dir, "le07_delivery_ledger.csv")
    ledger.to_csv(ledger_path, index=False)

    req = ledger[ledger["requested"]]
    cats = ledger["category"].value_counts().to_dict()
    n_units_manifest = int(manifest["n_scenes"].astype(int).sum())
    n_requested_scenes = int((scenes["request"].astype(str) == "True").sum())
    row_terminal = (manifest["download_status"] == "complete").all() and (
        manifest["order_status"].isin(["ready_for_download", "complete"]).all()
    )
    gate = {
        "every_row_terminal": bool(row_terminal),
        "checksums_ok": bool(req["md5_ok"].all()),
        "identity_ok": bool(ledger.loc[ledger["etf_exists"], "identity_ok"].all()),
        "no_missing_files": int(cats.get("missing-file", 0)) == 0,
        "no_unrequested_products": int(cats.get("manifest-error", 0)) == 0,
        "le07_not_mislabeled": bool(
            (req.loc[req["sensor"] == "LE07", "mtl_spacecraft"] == "LANDSAT_7").all()
            and (req.loc[req["sensor"] != "LE07", "mtl_spacecraft"] != "LANDSAT_7").all()
        ),
        "counts_reconcile": len(req) == n_units_manifest == n_requested_scenes
        and sum(cats.values()) == len(ledger),
        "all_requested_in_scene_manifest": bool(req["in_scene_manifest"].all()),
    }
    gate["pass"] = all(gate.values())

    # coverage projection: PT-JPL vs SSEBop for the 66-site cohort, 2013-2025
    table = container_table(args.container, set(geoms), YEARS)
    outcomes = site_date_outcomes(ledger, scenes)
    by_year, by_site = coverage_tables(table, outcomes)
    by_year.to_csv(os.path.join(args.out_dir, "le07_coverage_by_year.csv"), index=False)
    by_site.to_csv(os.path.join(args.out_dir, "le07_coverage_by_site.csv"), index=False)
    pooled = by_year.drop(columns="year").sum(numeric_only=True)
    n_pt = int(pooled["n_ptjpl_dates"])
    coverage = {
        "n_ptjpl_dates": n_pt,
        "n_ssebop_dates_now": int(pooled["n_ssebop_dates_now"]),
        "n_ssebop_only_now": int(pooled["n_ssebop_only_now"]),
        "paired_now": int(pooled["paired_now"]),
        "paired_after_delivery": int(pooled["paired_after_delivery"]),
        "paired_after_recovery": int(pooled["paired_after_recovery"]),
        "frac_paired_now": pooled["paired_now"] / n_pt,
        "frac_paired_after_delivery": pooled["paired_after_delivery"] / n_pt,
        "frac_paired_after_recovery": pooled["paired_after_recovery"] / n_pt,
        "residual_by_outcome": {
            oc: int(pooled[f"residual_{oc}"]) for oc in OUTCOME_RANK if oc not in ("new_valid",)
        },
        "new_valid_site_dates_by_sensor": outcomes.loc[outcomes["outcome"] == "new_valid", "sensor"]
        .value_counts()
        .to_dict(),
        "delivered_valid_units_by_sensor": req.loc[
            req["category"] == "delivered-valid-file", "sensor"
        ]
        .value_counts()
        .to_dict(),
        "site_mean_etf_quantiles_valid": req.loc[
            req["category"] == "delivered-valid-file", "site_mean_etf"
        ]
        .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        .round(4)
        .to_dict(),
    }

    summary = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "repair_tree": args.repair_tree,
        "manifest_sha256": sha256_file(manifest_path),
        "scene_manifest_sha256": sha256_file(args.scene_manifest),
        "cohort": args.cohort,
        "n_site_years": int(len(manifest)),
        "n_units_manifest": n_units_manifest,
        "n_requested_scenes_scene_manifest": n_requested_scenes,
        "n_ledger_rows": int(len(ledger)),
        "categories": cats,
        "categories_by_sensor": req.groupby(["sensor", "category"]).size().to_dict(),
        "identity_problems": ledger.loc[~ledger["identity_ok"], "identity_problems"]
        .value_counts()
        .to_dict(),
        "chip_valid_frac_quantiles": req["chip_valid_frac"]
        .quantile([0.05, 0.5, 0.95])
        .round(3)
        .to_dict(),
        "gate": gate,
        "coverage_66_cohort_2013_2025": coverage,
        "ledger": ledger_path,
        "ledger_sha256": sha256_file(ledger_path),
    }
    summary["categories_by_sensor"] = {
        f"{k[0]}:{k[1]}": int(v) for k, v in summary["categories_by_sensor"].items()
    }

    if args.freeze:
        if not gate["pass"]:
            print("gate failed; not freezing")
        else:
            summary["freeze"] = freeze_tree(
                os.path.join(args.repair_tree, "orders"),
                os.path.join(args.out_dir, "le07_repair_frozen_hashes.sha256"),
            )
            summary["freeze"]["frozen_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    summary_path = os.path.join(args.out_dir, "le07_delivery_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("ledger",)}, indent=2, default=str
        )
    )
    print(f"\nledger: {ledger_path}\nsummary: {summary_path}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
